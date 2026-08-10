# Pre-Sales Dossier: AI Capabilities & Agent Integration Surfaces of SAP, Microsoft Dynamics 365, Oracle Fusion Cloud, and Odoo

*Prepared for an AI development firm (Pakistan-based, serving Gulf/Saudi enterprise clients) building RAG knowledge bases and LangGraph/Milvus/OpenAI multi-agent products, plus a forward-deployed engineering (FDE) offer. Research current as of late July 2026. Preview/beta vs GA flagged throughout.*

## TL;DR

- **Feasibility ranking for external, third-party AI-agent tool-calling: Oracle Fusion and Odoo are the most open; Microsoft Dynamics 365 is open but expensive to meter; SAP is the most locked-down and now actively gatekeeps agent traffic through its own Joule/Agent Gateway layer.** All four vendors ship native copilots and agent builders (SAP Joule/Joule Studio, Microsoft Copilot Studio, Oracle AI Agent Studio, Odoo 19 AI app), so the pitch must position a custom RAG/agent layer as cross-system orchestration and specialized reasoning that native single-vendor copilots do not deliver.
- **The commercial gotchas differ sharply**: SAP charges *digital/indirect access* per document created by external systems and (since June 2026) routes agent traffic through governed gateways; Microsoft meters everything in *Copilot Credits* where a self-triggered autonomous agent action costs 25 credits with no exceptions; Oracle bundles AI Agent Studio at *no additional cost* with 20,000 free monthly AI Units and does not count tokens on OCI-hosted models; Odoo requires *Enterprise* edition plus prepaid IAP credits (Community has no native AI).
- **Recommended go-to-market**: lead with **read-only RAG integrations** (which sit outside SAP's document count and minimize credit/token burn), sequence transactional writes behind human-approval gates, set up the free Oracle and Odoo sandboxes first (lowest friction), pursue Oracle OCI GenAI + Microsoft Copilot Studio/agent certifications for fastest credibility, and target Gulf/Saudi clients where data-residency-compliant cloud regions and Arabic support are now materializing.

## Key Findings

1. **Every vendor now has a native agent-builder, and every vendor has embraced MCP and A2A** — but their *posture* toward external (non-native) agents ranges from "welcome" (Oracle, Odoo) to "allowed but governed" (Microsoft) to "must route through us" (SAP).
2. **SAP is the strategic outlier.** As of a security patch enforced June 9, 2026, SAP technically blocks non-compliant ODP-over-RFC calls and its API Use Policy names four endorsed pathways for agent access (Joule Agents, Integration Suite MCP Gateway, Business Data Cloud, and A2A via the SAP Agent Gateway). External agents are expected to route through these governed layers, not call raw APIs directly.
3. **Microsoft has the deepest, most mature MCP surface**: the Dataverse MCP server (GA in Copilot Studio, Azure AI Foundry, and MCP-compatible clients including Claude, Cursor, GitHub Copilot), a GA Dynamics 365 ERP (Finance & Supply Chain) MCP server (Feb 2026), a new Commerce MCP server (NRF 2026), and Business Central agents — but cost is metered aggressively in Copilot Credits.
4. **Oracle is the most integration-friendly for a firm like this**: the Fusion AI Agent Studio exposes an `/invokeAsync` REST endpoint for external apps, supports MCP tools and A2A, is model-agnostic (OpenAI, Meta, Cohere hosted; Anthropic/Google/xAI selectable), and is included at no extra cost with human-approval nodes built in.
5. **Odoo is the cheapest and most permissive at the API layer** — the new External JSON-2 API (bearer API key) plus a rich community MCP-server ecosystem — but its native AI requires Enterprise licensing plus IAP credits, and full external API access requires a Custom pricing plan.
6. **Gulf/Saudi readiness is arriving fast**: Oracle has live Jeddah (2020) and Riyadh (Oct 2024) regions; Microsoft Azure Saudi Arabia East launches Q4 2026; sovereign-cloud arrangements are being explored across the region. Arabic support in the *native AI* features is still limited/preview for SAP Joule and requires configuration in Oracle.

## Details by Platform

---

### 1) SAP (S/4HANA / SAP Business AI Platform)

**Native AI (mid-2026).** SAP's copilot is **Joule**, now positioned inside the unified **SAP Business AI Platform** announced at Sapphire 2026 (May 11–13, Orlando), which consolidated SAP BTP, Business Data Cloud, and AI Foundation, anchored by the **SAP Knowledge Graph** (SAP's semantic encoding of decades of ERP engineering logic). **Joule Studio** (professional developer agent builder on BTP) reached **GA in the first half of 2026**. **Joule Studio 2.0** — a fully SAP-managed, zero-infrastructure, intent-based agent builder with no-code and pro-code paths — was announced at Sapphire, began first-customer rollout in **June 2026**, with broader GA expected **H2 2026**. SAP is shipping a large library of domain-specific agents (finance, supply chain, HR) and offers **free design-time access through end of 2026** under fair-use limits, plus a €100M partner fund. *Flag: specific counts such as "200 agents / 50 assistants" and the €100M fund figure come from partner/analyst blogs, not SAP primary docs — treat as directional.*

**Models.** SAP is deliberately **model-agnostic** via the **Generative AI Hub in SAP AI Core**: governed access to Azure OpenAI (GPT-4o and GPT-5-class), Anthropic Claude (Sonnet/Opus 4.x observed in AI Core), Google Gemini, Meta Llama, plus SAP-owned models (the ABAP LLM / SAP-ABAP-1, and tabular foundation models from its ~€1B Prior Labs acquisition). SAP deepened an Anthropic/Claude partnership for Joule's reasoning/agentic layer in 2026 (Claude delivered via Amazon Bedrock through the Generative AI Hub for some scenarios).

**External agent integration surfaces.** SAP exposes REST/OData APIs via the **SAP Business Accelerator Hub** (api.sap.com), with a **free sandbox** (`sandbox.api.sap.com`) for read (GET) trial calls against test data using an auto-provided API key; write operations require your own BTP tenant. Authentication is via **SAP Cloud Identity Services (IAS)** — OAuth 2.0 and, for the Agent Gateway, IAS App2App tokens with named-user context. For agents specifically, SAP's Architecture Center defines the endorsed model:
- **MCP for external exposure**: the **MCP Gateway in SAP Integration Suite (API Management)** lets customers expose SAP and non-SAP APIs as governed, MCP-compliant tools consumable by any AI agent (e.g., a LangGraph agent). This is SAP's recommended control plane for external agent tool-calls.
- **A2A via the SAP Agent Gateway**: exposes Joule agents over the Agent2Agent protocol to third-party agents/platforms (Google Vertex, Microsoft Copilot Studio, AWS Bedrock AgentCore). *Flag: the Agent Gateway is **not yet GA** as of mid-2026.*
- **MCP inside Joule Studio**: agents built in Joule Studio can connect to any external MCP server.

**Licensing / commercial gotchas.** The critical exposure is **SAP Digital Access (indirect/"digital" access)**: when an external system (including an AI agent) **creates** one of nine chargeable document types in SAP, it triggers a per-document charge counted at line-item level on initial creation only — **Read, Update, and Delete are not counted**. Per SAP's own Digital Access model, the nine types are **Sales, Invoice, Purchase, Service & Maintenance, Manufacturing, Quality Management, Time Management, Financial (weighted 0.2), and Material documents**. This "read-only is free" rule is a strong reason to sequence read-only RAG first. The Digital Access Adoption Program (DAAP) offers conversion credits (commonly cited 90%-discount and 15%-growth options). Separately, SAP's June 2026 API-policy enforcement means agent traffic is expected to route through the four endorsed pathways; SAP reserves the right to throttle/suspend non-compliant patterns.

**Partner program.** **SAP PartnerEdge** (tiers commonly framed as Silver/Gold/Platinum; some tracks use a bronze/silver/gold Value Points system with thresholds). Entry requires due diligence, a partner agreement, a business plan, a program fee, and track-specific requirements; requirement checks run twice yearly (Jan/Jul). Certifications now carry 12-month validity with an annual "Stay Certified" reassessment.

**Certifications (AI-engineer credibility).** SAP exams are paid (learning journeys on learning.sap.com are free): **SAP Certified Associate – SAP BTP Extension Developer**; **Backend Developer – SAP Cloud Application Programming Model (CAP)** (C_CPE_16); **Integration Developer – SAP Integration Suite** (C_CPI_2506); **Generative AI Developer** (C_AIG); **SAP Business AI** (C_BCBAI). Fees are bundle-based (a two-attempt bundle is around US$276; a single US attempt is around US$560; a six-attempt Certification Hub subscription covers all exams). *Flag: SAP migrated to date-stamped exam codes in 2026 — verify live on training.sap.com. Many "SAP Joule certifications" sold online are third-party prep, not official SAP credentials.*

**Sandbox/dev access.** **SAP BTP Trial** (90-day, free, no credit card, includes Cloud Foundry, HANA Cloud, Business Application Studio) — but trial work cannot be moved to production and is deleted at expiry. **Free-tier** service plans on a Pay-As-You-Go/CPEA enterprise account persist longer and can upgrade to paid. The Business Accelerator Hub sandbox is free for API discovery/read testing.

**Gulf/Saudi & Arabic.** SAP is the enterprise standard among the largest Gulf entities (Aramco, SABIC, STC, Maaden, Almarai, major banks, giga-projects), and Saudi Arabia is the largest SAP market in the Middle East by license value. IDC has historically credited SAP with regional enterprise-application leadership. **Joule officially supports ~12 languages; AI-powered features in additional languages (including Arabic) are in preview / not officially supported** — Joule can *understand* unsupported languages via its LLM but with quality caveats. This is a genuine gap for Arabic-first Gulf deployments.

**Security/governance.** SAP explicitly frames external agent access around governance: the Architecture Center's "Third-Party MCP Access to SAP Solutions" guidance references the **OWASP MCP Top 10** risks, semantic enrichment via the Knowledge Graph, and recommends its managed MCP (Integration Suite) / A2A (Agent Gateway) paths over raw third-party MCP servers.

---

### 2) Microsoft Dynamics 365 (Dataverse / Power Platform / Copilot)

**Native AI (mid-2026).** Microsoft's copilot fabric spans **Microsoft 365 Copilot**, **Copilot Studio** (low-code agent builder on Power Platform/Dataverse), and role/prebuilt agents across Dynamics 365. The 2026 Release Wave 1 (Apr–Sep 2026) reframes Copilot as an **agentic operating layer**: prebuilt agents (Sales Order Agent, Payables Agent, Supplier Communication Agent), **Business Central custom AI agent design** (low-code, natural-language; GA May 2026), and multi-agent orchestration. Autonomous agents can act within finance/operations workflows with approval workflows. Microsoft also previewed forward-looking projects (ClawPilot / OpenClaw) at DynamicsMinds 2026 — not yet GA.

**Models.** Primarily Azure OpenAI (GPT-4o and GPT-5-class) via Azure AI Foundry; Copilot Studio and Foundry allow model choice and bring-your-own models; MCP enables connecting external tools/models.

**External agent integration surfaces (strongest MCP story).**
- **Dataverse Web API** (OData) + the **Dataverse MCP server** — GA and natively supported in Copilot Studio, Azure AI Foundry, GitHub Copilot, Claude Desktop/Code, Cursor, and other MCP-compatible clients. Tools cover query, schema/metadata inspection, search over structured+unstructured data, and create/update records. RBAC is per-user: the agent inherits the caller's existing Dataverse security role, not an elevated identity.
- **Dynamics 365 ERP MCP server** (Finance & Supply Chain) — **GA February 2026** (dynamic server; the older "static" 13-tool server retires **Oct 1, 2026**). Requires Tier 2+ or a Unified Developer Environment; **not supported on Cloud Hosted Environments (CHE)**; external agent platforms must be allow-listed in "Allowed MCP Clients."
- **Dynamics 365 Commerce MCP server** — introduced NRF 2026 (product discovery, inventory, pricing, checkout, store ops).
- **Business Central MCP server** — conversational/agentic access to live BC data.
- Auth via **Microsoft Entra ID** (OAuth 2.0).

**Licensing / commercial gotchas (the big one is credits).** Copilot Studio is metered in **Copilot Credits**, pooled per tenant, **not per seat**. Per Microsoft's official Copilot Studio pricing page, **capacity packs are $200 per tenant per month for 25,000 Copilot Credits** (~$0.008/credit on annual commit) or **PAYG ~$0.01/credit** via an Azure meter. Consumption varies wildly by feature: a classic answer ~1 credit, a generative answer ~2, tenant-graph grounding ~10, and — per Microsoft's published 2026 rates — **a self-triggered autonomous agent action always costs 25 credits with no exceptions, even for M365 Copilot users**. A single tenant-graph-grounded *reasoning* response stacks to **112+ credits** (≈10 grounding + 2 generative + ~100 reasoning). Prepaid Copilot Credit Commit Units (CCCUs; $1 = 100 credits) offer 5–20% volume discounts. For M365 Copilot–licensed users ($30/user/mo Enterprise; $18 Business promo ≤300 users through June 30, 2026), *internal, employee-facing* agent use is largely zero-rated within fair-use; **external/autonomous agents draw metered credits**. Licensing note: an agent's *identity* doesn't need its own license, but users interacting with it need the relevant D365 app license; agents built in Copilot Studio or reaching F&O via the ERP MCP servers follow this rule. Azure OpenAI token costs and Azure compute (Functions/Logic Apps) land as separate line items.

**Partner program.** **Microsoft AI Cloud Partner Program (MAICPP)** — free to join, no revenue minimum, no cert requirement at enrollment. **Solutions Partner for Business Applications** (and Data & AI) designations require a Partner Capability Score of 70+ (Performance/Skilling/Customer Success/Growth categories) and cost **US$4,875/year** (some sources cite $4,730). Certified-software / Industry-AI designations add third-party audit fees (~$2,400–$3,600). Advanced specializations available.

**Certifications.** Directly relevant, all paid (~$99 Fundamentals / ~$165 Associate–Expert): **PL-200** (Power Platform Functional Consultant), **PL-400** (Developer), **PL-600** (Solution Architect — retires June 30, 2026), **AI-102** (Azure AI Engineer — retires June 30, 2026) succeeded by **AI-103** (Azure AI Apps and Agents Developer — RAG/agents/Foundry). New agent-focused credentials: **MB-820** (Copilot Studio Functional Consultant), **AB-620** (AI Agent Builder Associate — in beta as of April 2026; covers MCP/A2A/RAG/multi-agent), **AB-100** (Agentic AI Business Solutions Architect Expert).

**Sandbox/dev access.** **Microsoft 365 Developer Program** free renewable developer tenants; free Azure account ($200 credit); Power Platform Developer Plan; Copilot Studio has a free build/test trial and the in-context Agent Builder is included with M365 Copilot. Strong, low-friction practice paths.

**Gulf/Saudi & Arabic.** Microsoft's integrated Azure + D365 + Power Platform stack is popular in Saudi mid-market and enterprise. Per the Microsoft EMEA newsroom (Feb 10, 2026), the **Azure Saudi Arabia East region will run customer cloud workloads from Q4 2026** (Eastern Province, three availability zones each with independent power, cooling, and networking; first announced 2023). Sovereign-cloud exploration with PIF/SITE is ongoing, and existing UAE regions serve KSA today. Arabic is well supported across the Microsoft productivity stack; Copilot supports Arabic broadly.

**Security/governance.** Microsoft frames agent governance across **design time, promotion time, and runtime**, with an **Agent Runtime Controller** returning allow / allow-with-redaction / require-human-review / deny at execution time. Governance is enforced through Entra identity, Dataverse RBAC inheritance, allow-listed MCP clients, and Purview.

---

### 3) Oracle Fusion Cloud Applications (AI Agent Studio)

**Native AI (mid-2026).** Oracle embeds hundreds of generative-AI use cases and **1,000+ embedded agent templates** across Fusion ERP/HCM/SCM/CX, plus **Oracle AI Agent Studio** (GA since Release 25C; a Fusion-native no-code/low-code design-time environment to create, configure, extend, test, validate, deploy, and monitor agents and agent teams). Release 26B added **22 prebuilt agentic applications** and **Workflow Agents**; an **AI Agent Marketplace** offers Oracle-tested partner templates. An integrated **AI help agent** assists builders.

**Models.** Model-agnostic: Oracle-hosted **OpenAI GPT-OSS ("Basic LLM"), Meta Llama, and Cohere**, with **Anthropic, Google, xAI** and others selectable. Crucially, **tokens are not counted or charged when using OCI-hosted models** (GPT-OSS/Llama/Cohere); token limits (a 200M-tokens/month base allocation per subscription) apply only when using OpenAI's hosted models such as GPT-5 mini / GPT-4.1 mini. General Actions on the Basic LLM carry **$0.00 AI Unit cost**.

**External agent integration surfaces (very open).**
- **Fusion REST APIs** across all modules.
- **`/invokeAsync` REST endpoint** (`/api/fusion-ai/orchestrator/agent/v2/<AgentTeamCode>/invokeAsync`) lets any external app invoke a Fusion **Agent Team** asynchronously (invoke-then-poll with a job ID); SSE/callback patterns supported. This is a clean path for a LangGraph orchestrator to call Oracle agents.
- **MCP Tool** support inside AI Agent Studio (agents call external MCP servers) and **A2A** with agent-card configuration.
- **Auth**: OAuth 2.0 bearer tokens via **OCI IAM / IDCS Identity Domains**; grant types include client credentials and — recommended for production — **JWT user assertion**. A confidential application represents the external caller with scoped access (e.g., the Fusion AI "Spectra" scope).

**Licensing / commercial gotchas (most generous).** AI Agent Studio and embedded agents are **included at no additional cost** with Fusion subscriptions. Under the 26C **AI Units** model, **1 AI Unit = US$0.01**, and **every Fusion Cloud customer receives 20,000 AI Units per month at no additional charge** (reset monthly, no rollover). Crossing into **Custom Agents** (extending beyond a template's original design, adding your own RAG docs/data sources, connecting new external systems) requires a **paid Custom AI subscription** (seat/employee-based; per-authorized-user SKUs include ~2M tokens/user/month). A separate **Agentic Applications production SKU (B112535)** adds ~30M AI Units/year and the right to publish agentic *applications* to production. *Practical implication: sticking to OCI-hosted LLMs eliminates token-cost concerns; the licensing trigger is extension, not complexity.*

**Partner program.** **Oracle PartnerNetwork (OPN)** — track-based (Cloud Build/Sell/Service/License & Hardware). Fusion demo/test environments are typically accessed through OPN membership or customer subscriptions.

**Certifications.** **OCI 2026 Generative AI Professional** (1Z0-1127-26; LLMs, prompt design, RAG chatbots on OCI Generative AI), **OCI AI Foundations Associate** (1Z0-1122), **Oracle AI Vector Search**, and Fusion-specific **AI Agent Studio Foundations Associate** / **AI Agent Studio Developer Professional**. Standard exams are paid (~$245), but **Oracle frequently offers free training + free exam vouchers** through Oracle University "Race to Certification" promotions (self-paced training on learn.oracle.com is generally free year-round). *Flag: the OCI GenAI Professional 2025 path is scheduled to archive Sep 30, 2026, replaced by the 2026 edition.*

**Sandbox/dev access.** OCI Free Tier / Always Free plus trial credits for the AI/GenAI services and vector search practice; Fusion Apps trials via partner/customer channels. Practicing the *AI Agent Studio* itself typically needs a Fusion tenant (partner or customer).

**Gulf/Saudi & Arabic.** Oracle is the most established hyperscaler in KSA with **live Jeddah (2020) and Riyadh regions** — the Riyadh region became available to clients in **October 2024, hosted by Center3 (stc Group) as part of a $1.5 billion investment** in Saudi digital infrastructure — plus UAE presence (Du's National Hypercloud reportedly running 150+ Oracle services). This gives strong data-residency/SAMA/NCA compliance options. Oracle Fusion has the region's most active cloud-ERP pipeline in oil & gas, aviation, banking, and government. Arabic is a supported Fusion language, but **AI-agent response language isn't auto-synced to the user's locale** — it must be set at design time or specified in the prompt.

**Security/governance.** Oracle builds governance into AI Agent Studio from the outset: **human-approval checkpoints (Human Approval Node)** for high-value transactions, audit trails, debugging/inspection tools, and OCI-developed **guardrails** (prompt-injection detection; harmful-content and PII detection — PERSON/EMAIL/TELEPHONE_NUMBER etc. with confidence scoring; subject-area restriction). Oracle explicitly advocates **runtime governance** ("governed execution"), treating policy enforcement as an execution-time decision (allow/deny/redact/escalate), aligned to the OWASP Agentic Top 10 (e.g., ASI09 human-agent trust exploitation).

---

### 4) Odoo (Odoo 19 AI app)

**Native AI (mid-2026).** **Odoo 19** (live since Sep 2025) introduced a dedicated **AI app** as a central hub, plus configurable **AI Agents**, an **"Ask AI"** natural-language assistant (Ctrl+K), AI-assisted fields, AI document capture, and prompt-based **AI server actions** — reaching across CRM, Accounting, Helpdesk, and HR. Agents have a defined purpose, instructions, information sources, and tools; they can be trained on your own documents/Knowledge-app content (a built-in RAG-style capability), connected to Live Chat, and used internally or via the website. **Odoo 20** (to be unveiled at Odoo Experience, Sep 24–26 2026, Brussels; GA ~Oct–Nov 2026) is slated to move from assistive to agentic multi-step workflows.

**Models.** Odoo connects to **OpenAI (ChatGPT) and Google Gemini** natively, with additional providers (Claude, etc.) connectable via modules or bring-your-own API key.

**External agent integration surfaces (cheapest, most permissive).**
- **External JSON-2 API** — the modern REST-style API (`POST /json/2/<model>/<method>` with `Authorization: Bearer <api_key>`). Introduced in Odoo 19; **XML-RPC is deprecated and slated for removal in Odoo 20** (legacy XML-RPC/JSON-RPC still work through 18/19). Auth uses **bearer API keys** tied to a bot user with least-privilege permissions (password can be disabled).
- **MCP**: **no official Odoo MCP server exists** as of mid-2026, but a healthy **community ecosystem** does — e.g., `twtrubiks/odoo19-mcp-server` and `AlanOgic/odoo-mcp-19` (both JSON-2-based, stdio + HTTP transports, with read-only safety modes and write-enable flags), the PyPI `odoo-mcp` package, and commercial offerings (e.g., Pantalytics). These expose search/read/create/update tools; models must be explicitly enabled for MCP access.
- *Flag: External API access is only available on **Custom** Odoo pricing plans (not the entry "Standard"/one-app free tiers).*

**Licensing / commercial gotchas.** Native AI **requires Odoo Enterprise** (every Standard AI feature is Enterprise-exclusive / OEEL-1 licensed); **Community edition has no built-in AI**. AI calls consume **Odoo in-app-purchase (IAP) prepaid credits** (billed by consumption) *or* your own LLM provider API key; Enterprise subscribers get some free IAP credits to test. Because a firm can bring its own OpenAI key and call the JSON-2 API directly, **Odoo is the lowest-cost-of-entry platform for an external agent layer** — but external API access requires the Custom plan.

**Partner program.** **Odoo Partner Program** — tiers **Ready / Silver / Gold**. Published thresholds: **Ready** = 10 new Enterprise users/year + 1 certified resource (10% commission); **Silver** = 75 users + 3 certified + 70% retention (15%); **Gold** = 300 users + 6 certified + 80% retention (20%). Odoo's become-a-partner pricing references up to ~$3,950 and up to ~$180/month ($1,728/year) enablement figures. Certification headcount is by named individuals and feeds tier status.

**Certifications.** **Odoo Functional Certification** (currently the Odoo 18 exam; ~**US$250**, discounted to ~$220 on odoo.com; ~120–125 questions, ~90 min, 70% pass, wrong answers penalized). The latest exam variant adds an **AI subject**. Anyone can sit it (no partner requirement); free sample test + free e-learning at odoo.com/slides. Certification is nominative (per individual, non-transferable).

**Sandbox/dev access.** **Odoo Online free 15-day trial**; **Odoo Community (open source)** self-hosted is free forever for practicing the JSON-2 API and community MCP servers; **Odoo.sh** developer hosting. This is the easiest environment for the team to stand up locally and practice agent tool-calls immediately.

**Gulf/Saudi & Arabic.** Odoo is the **most popular ERP among Saudi SMEs** (pay-as-you-grow, per-app pricing), with **180+ officially listed Saudi implementation partners** — one of the largest Odoo markets in MENA. **ZATCA-compliant e-invoicing** via the Saudi localization module; **Arabic is a first-class RTL interface language**. Odoo is a mid-market/SME play, not typically the ERP of the largest Gulf enterprises (which run SAP/Oracle).

**Security/governance.** Least-privilege bot API keys; per-model MCP enablement; community MCP servers ship read-only defaults and explicit write-enable flags. Governance is largely the integrator's responsibility (Odoo provides the primitives, not a managed agent-governance plane).

---

## Cross-Platform Comparison

### Feasibility ranking for third-party agent tool-calling (best → hardest)
1. **Oracle Fusion** — clean `/invokeAsync` REST + MCP tool + A2A, model-agnostic, generous free tier, built-in approval gates. Best balance of openness and enterprise governance.
2. **Odoo** — simplest and cheapest API (JSON-2 bearer key), thriving community MCP, self-hostable Community for practice; but Enterprise+Custom-plan gating and no official MCP server.
3. **Microsoft Dynamics 365** — richest, most mature MCP surface and identity model, but credit metering makes high-volume autonomous agents expensive and several environment restrictions apply (no CHE; allow-listing; Tier 2+).
4. **SAP** — technically capable (MCP Gateway, A2A) but **strategically gated**: SAP wants agent traffic through Joule/Agent Gateway, enforces API policy (June 2026), and layers Digital Access document charges on writes. Highest integration friction and commercial risk.

### Integration effort (rough)
- **Odoo**: low — days to a working read/write agent tool against JSON-2 on a self-hosted instance.
- **Oracle**: low-moderate — OAuth/IDCS setup + confidential-app registration, then `/invokeAsync`.
- **Microsoft**: moderate — Entra app registration, Dataverse/ERP MCP allow-listing, environment tiering; then well-documented.
- **SAP**: high — BTP/Integration Suite provisioning, IAS App2App, MCP Gateway configuration, and policy/licensing navigation; Agent Gateway not yet GA.

### Cost exposure to the *client* from external agent activity
- **SAP**: highest and least predictable — per-document Digital Access on writes (nine document types) + gateway metering.
- **Microsoft**: high for autonomous/write-heavy agents — 25 credits per self-triggered action, 112+ per grounded reasoning response, plus Azure token costs.
- **Oracle**: lowest for read/embedded — 20,000 free AI Units/month, no token counting on OCI-hosted models; Custom-Agent subscription only when extending.
- **Odoo**: low — IAP credits or bring-your-own API key; Custom plan for API access.

## Recommendations (staged, with thresholds)

**Stage 0 — Team enablement (weeks 1–6).**
- **Stand up the free sandboxes in this order**: (1) **Odoo Community** self-hosted + JSON-2 API + a community MCP server (fastest end-to-end agent-tool loop, zero cost); (2) **Microsoft 365 Developer tenant + Power Platform Developer Plan + free Azure** to exercise the Dataverse and ERP MCP servers; (3) **Oracle OCI Free Tier / trial** for GenAI + vector search, and pursue a Fusion trial via a partner/customer; (4) **SAP BTP Trial + Business Accelerator Hub sandbox** last, given friction.
- **Certifications to pursue first (credibility-per-dollar)**: Oracle **OCI 2026 GenAI Professional** (often free via Oracle promos) and **Fusion AI Agent Studio Developer Professional**; Microsoft **AB-620 (AI Agent Builder)** + **MB-820 (Copilot Studio)** + **AI-103**; Odoo **Functional Certification** (cheap, opens partner path); SAP **Generative AI Developer (C_AIG)** + **Integration Developer (C_CPI_2506)** when an SAP deal is live.

**Stage 1 — Read-only RAG first (the safe wedge).** Lead every engagement with a **read-only knowledge-base / RAG layer** (Milvus + OpenAI, LangGraph orchestration) that queries ERP data via the vendor's read APIs/MCP. This:
- **Avoids SAP Digital Access charges** (reads are explicitly outside the document count).
- **Minimizes Copilot Credits / Oracle AI Units / Odoo IAP burn.**
- Delivers immediate value (cross-system search, policy Q&A, exec summaries) that native single-vendor copilots don't do well across heterogeneous estates.
*Threshold to advance*: client sign-off on data-access scope + a logged, evaluated read-agent running against a non-production tenant.

**Stage 2 — Transactional writes behind human-approval gates.** Introduce write actions (create/update records, post documents) only with **explicit human-approval checkpoints** — natively supported by Oracle's Human Approval Node and Microsoft's runtime controller, and implementable in LangGraph for Odoo/SAP. For SAP specifically, **route writes through the endorsed pathway (Integration Suite MCP Gateway or A2A/Agent Gateway once GA)** and **model Digital Access document exposure before go-live**.
*Threshold to advance*: approval-trigger precision/recall measured; audit trail complete; licensing/credit cost per transaction modeled and accepted by the client's procurement.

**Stage 3 — Multi-agent orchestration + FDE.** Position the **FDE (embedded engineer)** offer around what native copilots structurally can't provide: **cross-ERP orchestration** (e.g., an agent reasoning across SAP + a non-SAP CRM), **custom RAG over the client's private corpus**, **model choice/cost optimization**, and **governed runtime enforcement** the client owns rather than rents from a single vendor.

**Positioning against native AI (the pitch spine).**
- *"We don't replace Joule/Copilot/AI Agent Studio — we orchestrate across and beyond them."* Native agents are strongest inside their own vendor's data; your differentiation is **heterogeneous, cross-system, model-agnostic reasoning with a RAG layer the client controls.**
- *Cost narrative*: native metering (SAP documents, Copilot Credits) punishes autonomous writes; a **read-first + approval-gated** architecture on your stack controls spend.
- *Sovereignty narrative for Gulf/Saudi*: emphasize data-residency-aligned deployment (Milvus/agents in-region), Arabic handling (a real gap in SAP Joule's official support), and ZATCA/SAMA/NCA-aware design.

## Caveats

- **Fast-moving, preview vs GA**: SAP **Agent Gateway (A2A) is not yet GA**; Joule Studio 2.0 is in first-customer rollout with GA expected H2 2026; Microsoft **AB-620** cert is in beta; the Dynamics 365 **static ERP MCP server retires Oct 1, 2026**; Oracle's OCI GenAI 2025 cert path archives Sep 30, 2026; Odoo 20 is not yet GA. Re-verify before any client commitment.
- **Third-party vs primary sourcing**: several SAP figures (agent counts, €100M fund, "200 agents") and some pricing specifics come from partner/analyst blogs, not vendor primary docs — flagged inline. SAP exam pricing is bundle-based and hard to pin to a single official number.
- **SAP API lockdown severity**: reporting (including Forrester-cited enforcement dates) indicates SAP is actively restricting non-endorsed agent/API access; the practical latitude for "raw API" third-party agents against SAP is narrowing — validate the specific client's contract and RISE terms.
- **Arabic in native AI**: materially limited for SAP Joule (preview/unsupported) and requires explicit configuration in Oracle; do not assume production-grade Arabic in vendors' native agents.
- **Licensing is client-specific**: Digital Access exposure, Copilot Credit consumption, Oracle Custom-Agent triggers, and Odoo plan gating all depend on the client's exact contract, volumes, and architecture — model each per deal rather than quoting these figures as fixed.