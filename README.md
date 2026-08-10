# RLM — a minimal Recursive Language Model

A small, readable implementation of one idea: **instead of pushing a whole document into a model's context, let the model navigate the document.**

The model never sees the document. It sees a *table of contents* — one line per section with an id, a heading path, a size and a short preview — and decides which sections are worth reading. A section small enough gets read in full. A section too big gets **descended into**, which repeats the same decision one level down. At the end, only the findings are combined into an answer.

The cost of that is proportional to **what the model chose to read**, not to how large the document is.

```
$ python -m app --question "What does SAP charge for indirect access?"

[RLM] chunk: 26 chunks, 7769 tokens, tree depth 3, leaf target 600
[RLM] context limit 1500 tok < document 7769 tok -> recursive mode
...
[RLM] read 1,639 of 7,769 document tokens (21.1%)
```

> **This is an educational implementation**, built to make the idea legible in a few hundred lines of Python. It is not a reproduction of any particular RLM paper, and it does not attempt the full space of recursive-inference techniques. It is meant to be read, run, and extended.

---

## Contents

- [Why this problem is interesting](#why-this-problem-is-interesting)
- [An honest note about the demo](#an-honest-note-about-the-demo)
- [Architecture](#architecture)
- [How the recursion works](#how-the-recursion-works)
- [Example run](#example-run)
- [Install](#install)
- [Configuration](#configuration)
- [Running it](#running-it)
- [Tests](#tests)
- [Design decisions](#design-decisions)
- [Limitations](#limitations)
- [Where to take it next](#where-to-take-it-next)

---

## Why this problem is interesting

The obvious way to answer a question about a long document is to put the whole document in the prompt. That has three problems that get worse with length:

1. **It costs.** You pay for every token on every question, whether or not it was relevant.
2. **Attention dilutes.** Accuracy on facts buried in the middle of a very long context is measurably worse than on the same facts in a short one.
3. **It has a ceiling.** Eventually the corpus is bigger than any window, and "just use a bigger model" stops being an answer.

Classical RAG addresses this by embedding chunks and retrieving the top-k. That works well, but retrieval is a *single flat step*: you get one shot at picking the right chunks, from a similarity score, with no ability to look at something and then decide where to look next.

An RLM makes the lookup itself a reasoning step, and lets it repeat:

| | Stuff the whole context | Flat RAG | RLM (this repo) |
|---|---|---|---|
| What the model sees first | everything | top-k chunks | a table of contents |
| Selection made by | — | vector similarity | the model, with a stated reason |
| Can it look again? | — | no | yes, bounded |
| Can it zoom in? | — | no | yes — that's the recursion |
| Cost scales with | document size | k | what it chose to read |

The trade is real and worth stating plainly: an RLM spends **more model calls** to read **fewer tokens**, and it can explain which sections it consulted and why.

---

## An honest note about the demo

The sample document, [`test_files/erp-ai-capabilities.md`](test_files/erp-ai-capabilities.md), is about **7,800 tokens**. That fits comfortably in any current model's context window. Pretending otherwise would make this demo a lie.

So the context limit here is an **explicit, configurable knob**, and it defaults to a deliberately small `max_context_tokens = 1500` — small enough that recursion genuinely triggers on a document you can read yourself in ten minutes.

That is a *simulation of the long-context regime*, and it is the honest way to demonstrate the mechanism at a size you can inspect. The same code with `max_context_tokens=100_000` against a multi-million-token corpus does exactly what it does here.

And the knob works in both directions — raise it above the document size and the engine takes the base case:

```console
$ python -m app --mock --max-context-tokens 20000 --question "Summarise the key findings."

[RLM] load: erp-ai-capabilities.md (14 sections, 7519 tokens)
[RLM] chunk: 26 chunks, 7769 tokens, tree depth 3, leaf target 600
[RLM] document fits the 20000 tok context -> answering directly, no recursion needed
[RLM] done in 0.0s | 1 LLM calls | 8,588 in / 122 out | max depth 0
[RLM] read 7,769 of 7,769 document tokens (100.0%)
```

One call, 100% of the document read. **That is the system being correct, not broken.** Recursion is a response to a constraint; with no constraint, the right amount of recursion is none.

---

## Architecture

```mermaid
flowchart TD
    A[Document<br/>.md / .docx / .pdf] --> B[Loader<br/>heading-aware sections]
    B --> C[Chunker<br/>tree: collapse what fits,<br/>split what does not]
    C --> D[Index<br/>id + heading + size + preview]

    Q[Question] --> E
    D --> E{{"ROUTE<br/>which sections?"}}

    E -->|small enough| F[INSPECT<br/>read the section]
    E -->|too big| G[RECURSE<br/>index its children]
    G -.->|one level down| E

    F --> H[Findings]
    G --> H
    H -->|readers came back short| E
    H --> I{{"SYNTHESISE<br/>findings only"}}
    I --> J[Answer + citations]

    style E fill:#2d5f8a,color:#fff
    style G fill:#2d5f8a,color:#fff
    style I fill:#3d7a4a,color:#fff
```

**Only `INSPECT` ever receives document text.** Routing, compression and synthesis all operate on the index or on findings. That is what keeps the reasoning context bounded no matter how large the input is.

```
app/
├── cli.py              argparse, REPL, output rendering
├── config.py           Settings dataclass; env + CLI overrides; key never logged
├── models.py           every dataclass: Chunk, Finding, TraceEvent, Stats, RLMResult
├── trace.py            Tracer -- emits [RLM] lines AND records the structured trace
├── llm/
│   ├── base.py         LLMClient protocol, JSON extraction ladder, repair retry
│   ├── openai_client.py  the only file that imports the OpenAI SDK
│   └── mock.py         test doubles + the offline client behind --mock
├── loaders/
│   ├── _structure.py   shared heading-stack logic + level normalisation
│   ├── markdown_loader.py   stdlib
│   ├── docx_loader.py       stdlib zipfile + ElementTree (no python-docx, no lxml)
│   └── pdf_loader.py        pypdf, best-effort structure
└── rlm/
    ├── tokens.py       tiktoken with an automatic chars/4 fallback
    ├── chunker.py      sections -> chunk tree
    ├── context.py      build_index() -- the compact TOC the router sees
    ├── retrieval.py    BM25-lite lexical pre-filter (no vector store)
    ├── prompts.py      all four prompts, in one place
    └── engine.py       route -> inspect -> recurse -> synthesise, and the guards
```

---

## How the recursion works

### 1. The document becomes a tree

Two rules, applied to the heading hierarchy:

- **Collapse** — a subtree that already fits `chunk_target_tokens` becomes one leaf. Without this, the index fills with seven-token wrapper headings that tell the router nothing.
- **Split** — a leaf still too big is cut along paragraph boundaries, then sentence boundaries, then whitespace. Never mid-word. Consecutive parts carry a small overlap so an argument cut in half is still readable.

For the sample document (`python -m app --show-tree`):

```
c1             68 tok  leaf         Pre-Sales Dossier: AI Capabilities & Agent Integration Surfaces...
c2            343 tok  leaf         TL;DR
c3            466 tok  leaf         Key Findings
c4           5289 tok  4 children   Details by Platform
  c4.1         1502 tok  4 children   Details by Platform > 1) SAP (S/4HANA / SAP Business AI Platform)
    c4.1.1        397 tok  leaf         ... > part 1/4
    c4.1.2        503 tok  leaf         ... > part 2/4
    c4.1.3        515 tok  leaf         ... > part 3/4
    c4.1.4         87 tok  leaf         ... > part 4/4
  c4.2         1399 tok  3 children   Details by Platform > 2) Microsoft Dynamics 365 (...)
  c4.3         1246 tok  3 children   Details by Platform > 3) Oracle Fusion Cloud Applications (...)
  c4.4         1142 tok  3 children   Details by Platform > 4) Odoo (Odoo 19 AI app)
c5            458 tok  leaf         Cross-Platform Comparison
c6            822 tok  2 children   Recommendations (staged, with thresholds)
c7            323 tok  leaf         Caveats
```

Note `c5`: it has three subsections in the source, but the whole thing fits in 600 tokens, so it collapsed to one leaf.

### 2. The router sees an index, not the text

```
[c2] TL;DR | 343 tok | leaf | "- **Feasibility ranking for external, third-party AI-agent tool-calling: Oracle Fusion and Odoo are the most open; Microsoft Dynamics 365 is..."
[c4] Details by Platform | 5289 tok | 4 subsections | "**Native AI (mid-2026).** SAP's copilot is **Joule**, now positioned inside the unified **SAP Business AI Platform** announced at Sapphire 2..."
[c5] Cross-Platform Comparison | 458 tok | leaf | "Feasibility ranking for third-party agent tool-calling (best → hardest) 1. **Oracle Fusion** — clean `/invokeAsync` REST + MCP tool + A2A, m..."
```

**That whole index is 396 tokens — 5.1% of the document.** And it is capped: previews shrink from 140 characters down to zero, and entries are dropped last, so the router prompt stays under `max_index_tokens` for a 10 KB document and a 10 MB one alike. Index cost is O(entries shown), not O(document).

The router replies with JSON — ids and a sub-question for each:

```json
{
  "reasoning": "SAP licensing details live under the per-platform breakdown",
  "selections": [
    {"chunk_id": "c4", "sub_question": "What does SAP charge for external system access?", "why": "per-platform detail"}
  ]
}
```

Ids it invents are dropped in code, not trusted. Ids already inspected are dropped. The list is truncated to `max_selections_per_round`.

### 3. Read, or descend

- Chunk has no children → **inspect** it. This is the only call carrying document text, and it is bounded by construction.
- Chunk has children → **recurse**: build an index over *those* children and run the same loop at `depth + 1`.

Sub-findings are **compressed** into a single finding before travelling back up, so a parent level's context does not grow with how much was read beneath it.

### 4. Look again if needed

If readers report `needs_more` or `found: false`, the level routes again over what is left — this time told what is already known, with any sections the readers pointed at pinned to the front. Bounded by `max_iterations`.

### 5. Synthesise

One final call over **the findings only**. Its context is bounded regardless of document size. If it fails to parse twice, the answer is assembled deterministically in Python from the findings rather than crashing.

### The guards

Three independent ways for a run to stop:

| Guard | Default | Stops |
|---|---|---|
| `max_depth` | 3 | how far down |
| `max_iterations` | 2 | how many routing rounds per level |
| `max_llm_calls` | 25 | total spend |

The call budget is enforced in **exactly one place** — every model call in the project passes through a single wrapper that counts it. There is no second code path that could forget.

---

## Example run

Reproducible with no API key. `--mock` swaps in an offline client: the *reasoning* is fake, but the chunking, indexing, routing, descent, budgets and token accounting are all real.

```console
$ python -m app --mock --question "What does SAP charge for indirect access, and how does it gate external agents?"

[RLM] load: erp-ai-capabilities.md (14 sections, 7519 tokens)
[RLM] chunk: 26 chunks, 7769 tokens, tree depth 3, leaf target 600
[RLM] context limit 1500 tok < document 7769 tok -> recursive mode
[RLM] depth 0 | iteration 1
[RLM]   prefilter: skipped (7 candidates <= 12)
[RLM]   index: 7 chunks -> 396 tokens (5.1% of document)
[RLM]   route -> c1 "...", c4 "..."  [call 1]
[RLM]   inspect c1 (68 tok) [call 2] -> found=true conf=0.60
[RLM]   recurse: c4 'Details by Platform' is 5289 tok > 600 -> descending
[RLM]   depth 1 | iteration 1
[RLM]     index: 4 chunks -> 294 tokens (3.8% of document)
[RLM]     route -> c4.1 "...", c4.2 "..."  [call 3]
[RLM]     recurse: c4.1 'Details by Platform > 1) SAP (...)' is 1502 tok > 600 -> descending
[RLM]     depth 2 | iteration 1
[RLM]       index: 4 chunks -> 290 tokens (3.7% of document)
[RLM]       route -> c4.1.4 "...", c4.1.3 "..."  [call 4]
[RLM]       inspect c4.1.4 (87 tok) [call 5] -> found=true conf=0.60
[RLM]       inspect c4.1.3 (515 tok) [call 6] -> found=true conf=0.60
[RLM]     compress: 2 findings from c4.1 -> 1  [call 7]
[RLM]     recurse: c4.2 'Details by Platform > 2) Microsoft Dynamics 365 (...)' is 1399 tok > 600 -> descending
[RLM]     depth 2 | iteration 1
[RLM]       route -> c4.2.2 "...", c4.2.1 "..."  [call 8]
[RLM]       inspect c4.2.2 (496 tok) [call 9] -> found=true conf=0.60
[RLM]       inspect c4.2.1 (473 tok) [call 10] -> found=true conf=0.60
[RLM]     compress: 2 findings from c4.2 -> 1  [call 11]
[RLM]   compress: 2 findings from c4 -> 1  [call 12]
[RLM] synthesize: 2 findings -> answer  [call 13]
[RLM] done in 0.0s | 13 LLM calls | 6,891 in / 1,683 out | max depth 2
[RLM] read 1,639 of 7,769 document tokens (21.1%)
```

The indentation is the recursion. The last line is the point: **the question was answered after reading 21% of the document.**

Drop `--mock` and set a key for real reasoning. The trace shape is identical; only the routing choices and the answer change.

---

## Install

Python 3.11 or newer.

```bash
git clone https://github.com/mallahim-ai/RLM_proj.git
cd RLM_proj

python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

Five dependencies: `openai`, `python-dotenv`, `tiktoken`, `pypdf`, and `pytest` for the tests.

```bash
cp .env.example .env               # Windows: Copy-Item .env.example .env
# then put your key in .env
```

You can skip the key entirely and run everything with `--mock`.

---

## Configuration

Every setting has a default, an `RLM_*` environment variable and a CLI flag. Precedence is **CLI flag > environment > default**. See [`.env.example`](.env.example) for the full list.

| Setting | Default | What it does |
|---|---|---|
| `RLM_MODEL` | `gpt-4o-mini` | any chat model supporting JSON response format |
| `RLM_MAX_CONTEXT_TOKENS` | `1500` | the simulated window — the headline knob |
| `RLM_CHUNK_TARGET_TOKENS` | `600` | target leaf size; drives how deep the tree gets |
| `RLM_CHUNK_OVERLAP` | `60` | carried between hard-split parts only |
| `RLM_MAX_DEPTH` | `3` | recursion guard |
| `RLM_MAX_ITERATIONS` | `2` | routing rounds per level |
| `RLM_MAX_LLM_CALLS` | `25` | total call budget per question |
| `RLM_PREFILTER_THRESHOLD` | `12` | above this many candidates, lexical scoring narrows first |
| `RLM_TOKENIZER` | `auto` | `heuristic` forces chars/4 and never touches the network |

`OPENAI_API_KEY` is unprefixed, by SDK convention. It is **never logged**: `Settings.__repr__` masks it, so neither a stray `print` nor an exception dump can leak it, and `.env` is gitignored.

Contradictory settings fail at startup rather than mid-run:

```console
$ python -m app --max-context-tokens 500 --chunk-target-tokens 5000
configuration error: chunk_target_tokens=5000 exceeds the inspect budget of 200 implied by
max_context_tokens=500. Lower the chunk size or raise the context window.
```

---

## Running it

```bash
# offline: the full recursive trace, no key, two seconds
python -m app --mock

# one question against the real model
python -m app --question "Which platform is most open to third-party agents?"

# interactive  (:help, :tree, :stats, :trace, :quit)
python -m app

# see the chunk tree and stop
python -m app --show-tree

# machine-readable; stdout stays pure JSON, the trace goes to stderr
python -m app --json --question "..." | jq .stats

# the other sample documents
python -m app --doc "test_files/WhatsApp Architecture and Technology Deep Dive.pdf" --question "..."
python -m app --doc test_files/knowledge-product-pakistan.docx --question "..."
```

Two scripts in [`examples/`](examples/):

```bash
python examples/run_mock_demo.py      # no key needed; also runs in CI as a smoke test
python examples/run_erp_question.py   # six question shapes against the real API
```

`examples/run_erp_question.py` walks a set of deliberately different retrieval problems — breadth, filtering, summarisation, comparison, one fact buried three levels down, and one needing two distant sections combined.

---

## Tests

```bash
pytest                  # 116 tests, fully offline, no API calls, no network
pytest -m integration   # the paid tests; needs OPENAI_API_KEY
```

The default run is free by construction: the `integration` marker is excluded in `pyproject.toml`, and `conftest.py` pins the heuristic tokenizer so counts are deterministic and nothing fetches a BPE table.

What is covered:

| Area | Examples |
|---|---|
| Loading | heading paths on the real file; UTF-8 survival; `#` inside a code fence; docx level normalisation; PDF heading heuristic |
| Chunking | deterministic ids; collapse; hard-split; overlap bounds; **no chunk cut mid-word**; unsplittable input still terminates |
| Index | stays under budget for 200 chunks; previews shrink before entries drop |
| Retrieval | heading terms outweigh body; prefilter only fires above threshold; pinned chunks never dropped |
| JSON | fenced, prose-wrapped and brace-in-string replies; one repair retry, then give up |
| Engine | base case is exactly one call; descent reaches depth 2; only findings travel upward |
| Guards | depth, iteration and call budgets each enforced independently |
| Resilience | malformed replies, **hallucinated chunk ids**, wrong JSON types, failed synthesis |
| OpenAI client | request construction and usage parsing, against a stubbed SDK |
| CLI | `--mock`, `--json` purity, exit codes, config errors, and `examples/` actually executing |

The engine tests drive the recursion with scripted clients, so they assert on *control flow* — how many calls, at what depth, in what order — rather than on model output.

---

## Design decisions

A few choices worth explaining, since the alternative was often the more obvious one:

**Lexical retrieval, not embeddings.** BM25 over the candidates at the current level is forty lines, has no index to keep in sync and no embedding calls. Heading paths are dense summaries, so weighting their terms 3× is remarkably effective. Embeddings belong here only once this demonstrably stops working.

**`tiktoken` with a fallback, not just chars/4.** This project is *about* enforcing a token budget, so mis-measuring its own budget would undercut the point. But tiktoken fetches its table over the network on first use, so failure latches a heuristic permanently and logs once — a fresh clone with no connectivity still works.

**No `python-docx`.** A `.docx` is a zip containing XML with explicit heading styles; reading it with `zipfile` and `ElementTree` is about thirty-five lines and zero dependencies. `python-docx` would have pulled in `lxml`, a multi-megabyte C extension.

**No `tenacity`.** The OpenAI SDK already retries connection errors, 429s and 5xx with backoff. The only retry this project adds is *semantic* — asking a model to repair malformed JSON — which a retry library cannot express anyway.

**Malformed model output is expected, not exceptional.** Four layers handle it: JSON response mode, an extraction ladder (raw → strip fences → balanced-brace scan), required-key validation, and one repair retry. After that it is **non-fatal**: a failed route ends that level and synthesis proceeds with whatever was found.

---

## Limitations

Stated plainly, because a demo that oversells is worse than one that undersells:

- **Structure-dependent.** The index is only as good as the headings. Markdown is ideal; a `.docx` with real heading styles is fine; a PDF is best-effort — it has no heading markup, so the loader guesses from short title-cased lines and falls back to page boundaries when that guess fires too often. A wall of unstructured text degrades this toward flat chunking.
- **More calls than RAG.** Reading 21% of a document took 13 model calls. For small documents that is strictly worse than one big prompt. The crossover is a function of document size, and this demo is well below it — see [the honest note](#an-honest-note-about-the-demo).
- **The router can be wrong.** It picks from headings and 140-character previews. If a fact sits under a misleading heading with an unrevealing opening, the router may not go there. Hallucinated ids are dropped safely, but a *plausible wrong* choice is just a wrong answer.
- **No caching or persistence.** Every question re-chunks and re-routes from scratch. Two identical questions cost twice. There is no database, by design.
- **Single document.** The engine navigates one document's tree. A corpus of thousands of files would want another level above this one.
- **Findings are not cross-checked.** If two sections disagree, synthesis sees both and does its best. There is no contradiction detection.
- **Sequential.** Sibling sections at the same level are inspected one after another; they are independent and could run concurrently.

---

## Where to take it next

Roughly in order of value per unit of added complexity:

1. **Parallel inspection** of siblings at a level — pure latency win, no accuracy change, no new dependency.
2. **A response cache** keyed on `(chunk_id, sub_question)`. SQLite, one table. Makes iterating on prompts far cheaper.
3. **Embeddings as a second pre-filter stage**, after BM25 rather than instead of it — worth doing once a corpus is large enough that lexical scoring visibly misses.
4. **Multi-document routing**: one more level of index, over files instead of sections. The engine is already recursive; this is mostly a loader change.
5. **Confidence-driven re-reading** — let a low-confidence finding trigger a targeted second look rather than the current binary `needs_more`.
6. **Answer verification**: a final pass checking each claim against the evidence quotes actually returned.
7. **Streaming the trace to a UI**, since `RLMResult.trace` is already structured for exactly that.

---

## License

MIT.
