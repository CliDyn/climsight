# ClimSight - Codebase Review

_Abdulhakim Gafai - March 2026_


## a) What Does This System Do?

ClimSight is a location-based climate analysis assistant. A user picks a place on a map, asks a question such as "What climate risks should I expect here?" or "How suitable will this area be for agriculture?", and the system assembles a readable answer from climate model outputs, geographic context, retrieved climate literature, and AI-generated synthesis.

For a non-technical audience, the simplest description is:

ClimSight turns complex climate data into an explanation a person can act on.

Instead of asking the user to interpret raw NetCDF files, climate model grids, or IPCC(including data from the Intergovernmental Panel on Climate Change) report excerpts, the system does the heavy lifting for them. It looks up the selected location, gathers climate projections and environmental context, optionally retrieves relevant literature, and produces a structured response with narrative explanation, charts, and practical takeaways.

It is strongest as an analytics and interpretation engine. It helps answer questions such as:

- What is happening at this location?
- What do climate projections suggest?
- What risks or tradeoffs should a decision-maker pay attention to?

What it is not yet, at least in its current form, is a full collaborative planning platform with workflow, governance, shared decision records, role-based participation, or institutional operating processes built around the analysis.

However, it's a translator between massive, complex climate datasets and the people who need to make decisions based on them: urban planners, policymakers, farmers, and researchers who don't have time to wrangle raw data themselves.

The project is backed by two peer-reviewed publications (Kuznetsov et al. 2025 in npj Climate Action, and Koldunov & Jung 2024 in Communications Earth & Environment) and is developed at the Alfred Wegener Institute, one of Germany's leading polar and marine research centers.

---

## b) Architecture - Key Components, Data Flow, External Dependencies

### High-Level Architecture

ClimSight currently supports two interaction models:

- A React frontend backed by FastAPI and WebSockets. This appears to be the primary product direction.
- A legacy Streamlit interface that still exists and remains useful for research-oriented local usage.

At a high level, the system looks like this:

```text
┌──────────────┐     WebSocket      ┌──────────────────┐
│  React UI    │<------------------>│  FastAPI Backend  │
│  (Vite/TS)   │                    │  (Uvicorn)        │
└──────────────┘                    └────────┬─────────┘
                                             │
                                    ┌────────▼─────────┐
                                    │  ClimSight Engine │
                                    │  (LangGraph)      │
                                    └────────┬─────────┘
                                             │
                    ┌────────────────────────┬┴───────────────────────┐
                    │                        │                        │
           ┌────────▼───────┐    ┌──────────▼─────────┐   ┌────────▼────────┐
           │ Climate Data   │    │ RAG / Literature    │   │ Data Analysis   │
           │ Providers      │    │ Retrieval           │   │ Agent + Python  │
           └────────┬───────┘    └──────────┬─────────┘   └────────┬────────┘
                    │                        │                        │
           ┌────────▼───────┐    ┌──────────▼─────────┐   ┌────────▼────────┐
           │ NetCDF / Zarr  │    │ ChromaDB +         │   │ Jupyter Kernel  │
           │ Climate Files  │    │ embeddings         │   │ + sandbox files │
           └────────────────┘    └────────────────────┘   └─────────────────┘
```

### Main Request Flow

The main user flow starts in the frontend and passes through a WebSocket endpoint in `api/routes/analysis.py`. From there:

1. The API validates that a session exists and receives a request with `lat`, `lon`, and `query`.
2. The backend loads config, prepares sandbox directories, and calls `location_request()` to resolve the selected location.
3. The request enters `llm_request()` / `agent_clim_request()` in `src/climsight/climsight_engine.py`.
4. A LangGraph workflow runs a set of agents:
   - `intro_agent`
   - parallel retrieval agents such as `data_agent`, `zero_rag_agent`, `ipcc_rag_agent`, `general_rag_agent`, and optionally `smart_agent`
   - `prepare_predefined_data`
   - optionally `data_analysis_agent`
   - `combine_agent`
5. The final answer, plots, and references are streamed back to the frontend.

### Key Components

#### 1. API / Session Layer

- `api/main.py`
- `api/routes/analysis.py`
- `api/routes/sessions.py`
- `src/climsight/session_manager.py`

This layer handles session creation, WebSocket streaming, request routing, sandbox preparation, and artifact download endpoints.

The implementation is straightforward and readable, but session state is in-memory only, with no TTL, no persistence, and no bounded resource policy.

#### 2. Engine / Orchestration Layer

- `src/climsight/climsight_engine.py`
- `src/climsight/climsight_classes.py`

This is the core orchestrator. It wires together geographic context lookup, climate data extraction, literature retrieval, predefined plots, optional sandboxed analysis, and final LLM synthesis.

Architecturally, this is the most important file in the repository and also the least modular one. The orchestration logic is powerful, but too much of it is concentrated in one place.

#### 3. Climate Data Provider Layer

- `src/climsight/climate_data_providers.py`

This is the strongest abstraction in the codebase. It defines a `ClimateDataProvider` interface and concrete implementations for:

- `NextGEMSProvider`
- `ICCPProvider`
- `AWICMProvider`
- `DestinEProvider`

The value of this design is that heterogeneous climate sources are normalized behind one contract. That makes it much easier to add or swap providers without rewriting the rest of the system.

#### 4. RAG / Literature Retrieval

- `src/climsight/rag.py`
- `src/climsight/embedding_utils.py`
- `rag_db/`

The system maintains separate vector stores for IPCC and general climate literature and supports different embedding backends. This separation is a good design choice because it keeps citation-oriented scientific retrieval distinct from broader context retrieval.

#### 5. Sandbox Analysis Layer

- `src/climsight/data_analysis_agent.py`
- `src/climsight/tools/python_repl.py`
- `src/climsight/tools/visualization_tools.py`
- `tmp/sandbox/...`

This is the most ambitious part of the system. The analysis agent can generate and execute Python code in a sandboxed Jupyter environment, produce plots, inspect data, and refine its own output with reflection tools.

This gives ClimSight unusually strong exploratory power, but it also creates the biggest operational risks: high memory cost, long-running jobs, complicated failure modes, and increased burden on observability and safety.

### External Dependencies

The system depends on a fairly large stack of external services and datasets:

#### AI and language services

- OpenAI models for synthesis, smart agent behavior, and embeddings
- optional AITTA and Mistral paths

#### Climate and observational data

- nextGEMS
- ICCP
- AWI-CM
- DestinE
- ERA5 climatology / ERA5 retrieval

#### Data and ML infrastructure

- xarray
- netCDF4
- zarr
- ChromaDB
- Jupyter kernel tooling
- scipy / cKDTree
- pyproj
- LangChain / LangGraph

### Storage and State

ClimSight is not built around a traditional transactional database. Instead:

- climate data is stored as files on disk
- vector stores are persisted under `rag_db/`
- generated artifacts live in per-session sandbox folders
- sessions are kept in an in-memory Python dictionary

This is acceptable for a research tool or local deployment, but it becomes fragile quickly once multiple real users or long-running processes enter the picture.

---

## c) What Works Well / What Concerns Me

### What Works Well

**The climate data provider abstraction is the standout design decision.** `climate_data_providers.py` creates a clean interface between the rest of the application and the specifics of each dataset. That is exactly the kind of abstraction I would want in a system that may need to integrate multiple data sources over time.

**The spatial data handling is serious engineering, not superficial glue code.** The NextGEMS path in particular shows real domain awareness: HEALPix grids, `cKDTree`, interpolation, and projection-aware distance handling. This is where the repository feels strongest and most defensible.

**The analysis modes are pragmatic.** The `fast`, `smart`, and `deep` modes in `data_analysis_agent.py` reflect an important product truth: not every query deserves the same amount of compute, tooling, and latency. This is a useful operating pattern.

**The system separates data gathering from synthesis in a sensible way.** The overall shape - gather location context, gather climate data, gather literature, optionally analyze, then synthesize - is conceptually right. The pipeline has a reasonable mental model.

**There is clear evidence of careful local tool-building.** The plotting tools, sandbox layout, artifact generation, and references handling show that the team built for real investigative work, not just a demo.

### What Concerns Me

**The engine is too monolithic.** `climsight_engine.py` concentrates too much orchestration logic in one place. It defines multiple agents as nested closures, shares state implicitly through outer scope, and mixes workflow construction with operational behavior. This will slow maintenance, testing, and onboarding.

**Failure handling is inconsistent, and degraded modes are not communicated clearly enough.** Some errors are swallowed, some are logged, some are re-raised, and some result in partial output without clear user-facing warnings. In a climate decision-support context, silent degradation is dangerous because the answer can still look polished.

**Critical components fail silently with no user indication.** I experienced this firsthand: I ran ClimSight without the RAG databases populated (the `embedded_chunks_db_openai.zip` was corrupted during my clone setup, so `rag_db/` was empty). The system produced what looked like a complete climate analysis — but with zero IPCC citations and zero grounding in scientific literature. The failure chain is: `is_valid_rag_db()` returns `False` → `load_rag()` logs a warning to `climsight.log` and returns `(False, None)` → `query_rag()` returns `(None, [])` → `combine_agent` silently skips adding RAG context → the LLM fills in from its training data instead. At no point does the user see a message like "Warning: IPCC reports database unavailable, response may lack scientific citations." The system's most important quality — grounding answers in peer-reviewed climate science — was completely absent, and the output looked indistinguishable from a fully functional response. This is the most dangerous kind of failure: it looks correct.

**The codebase is much stronger as an analysis engine than as a planning product.** ClimSight can produce analysis, charts, and narrative synthesis. What it does not yet provide are the structures that turn analysis into institutional decision-making: saved scenarios, role-based workflows, review/approval paths, auditability, collaborative state, or explicit planning cycles. That is not necessarily a flaw for this repository's current scope, but it is an important boundary to understand.

**Operational state is fragile.** In-memory session management, sandbox accumulation, Jupyter kernels, and multiple external services create a lot of operational surface area without much built-in control.

**The open source governance surface is thin.** I did not find `CONTRIBUTING`, `CODE_OF_CONDUCT`, `GOVERNANCE`, or similar maintainer-facing documents. For a research project that wants community traction, this matters. Good architecture alone does not create a healthy contributor path.

**There is significant dependency weight for self-hosting.** Large climate files, vector stores, API keys, and multiple optional data integrations make the platform powerful, but they also raise the barrier to entry for public institutions, NGOs, or researchers with limited infrastructure support.

---

## d) What Breaks First Under Real Users / What I Would Do About It

If ClimSight needed to serve real users beyond a small research or demo setting, I would expect failures to appear in this order:

### 1. Concurrency and resource exhaustion

Each analysis can trigger multiple retrieval steps, LLM calls, file reads, and in deeper modes a sandboxed Python analysis path. The Jupyter-based analysis layer is especially expensive.

What breaks first:

- too many concurrent requests
- long-running analyses blocking worker capacity
- kernel or memory pressure
- session state accumulating indefinitely

What I would do:

- add bounded concurrency and backpressure
- separate interactive requests from heavy background analyses
- introduce queueing for deep analyses
- implement TTL-based cleanup for sessions and sandbox artifacts

### 2. Hidden dependency failures

This system depends on local data presence, vector databases, model APIs, optional observational baselines, and external climate retrieval services. Right now, some of these fail quietly or only emit logs.

What breaks first:

- partial answers that look complete
- scientific grounding disappearing when RAG is unavailable
- optional enrichments failing without clear user notice

What I would do:

- make degraded modes explicit in the UI and final response
- track dependency health centrally
- distinguish between "no data available" and "data source failed"

### 3. Self-hosting friction

For a system like this to be useful outside a research lab, especially in public-sector or institution-run settings, it needs to be understandable and operable by small technical teams.

What breaks first:

- installation complexity
- dataset packaging and updates
- secrets management
- unclear infrastructure requirements

What I would do:

- provide a supported self-hosted deployment profile
- validate data presence at startup
- document hardware and storage expectations explicitly
- ship a smaller "demo / pilot" data bundle

### 4. Lack of observability

A complex AI-and-data workflow without structured metrics becomes very difficult to operate once more than a few people depend on it.

What breaks first:

- inability to explain latency
- inability to trace why a result degraded
- no easy way to measure queue depth, failure rate, or kernel health

What I would do:

- add structured logs with request IDs
- add metrics for latency, error class, active sessions, and queue depth
- expand health checks beyond "process is alive"

---

## e) Testing, Deployment, and Operational Resilience

### Testing

The existing repository has useful unit coverage for parts of the data-processing layer, but the main gap is system-level confidence.

I would use four layers of testing:

#### 1. Data and math unit tests

Keep expanding unit tests around:

- provider interpolation logic
- coordinate normalization
- climate extraction edge cases
- geographic helper failures

This is where deterministic tests add the most value.

#### 2. Pipeline integration tests

This is the biggest missing layer.

Examples:

- A request in `fast` mode should skip the analysis agent.
- A request with RAG disabled should still return a valid but explicitly downgraded response.
- A request with missing optional datasets should not crash the pipeline.

These tests should use mocked LLMs and mocked external services so that orchestration can be tested deterministically.

#### 3. Contract tests for external dependencies

The application relies on outside systems and file formats. I would add tests that periodically verify:

- expected response shapes from external APIs
- availability and structure of required local datasets
- vector store health and schema assumptions

#### 4. Load and failure-mode tests

I would add targeted tests for:

- concurrent session creation
- repeated sandbox creation / cleanup
- timeouts in long-running analysis paths
- partial dependency failure

This matters more here than pixel-perfect frontend tests.

### Deployment

I would think about deployment in three explicit modes:

#### 1. Demo / evaluation mode

- small bundled sample data
- easy startup
- enough realism to show value quickly

#### 2. Research / local power-user mode

- full data support
- optional heavy analysis paths
- local experimentation

#### 3. Institutional self-hosted mode

- repeatable deployment
- clear data packaging
- secret management
- supportable upgrade path

Right now the project is closer to mode 2 than mode 3.

Concretely, I would add:

- a production-oriented Compose or equivalent deployment for the React + FastAPI stack
- startup validation of required files and vector stores
- documented storage and compute expectations
- environment profiles for "demo", "full local", and "institutional"

### Operational Resilience

If this system were expected to support real decision-making, I would focus on:

#### Graceful degradation

The system should explicitly say when:

- literature grounding is unavailable
- observational baseline data is unavailable
- optional geographic enrichment failed

#### Timeout and cancellation behavior

Heavy agentic systems need clear limits. I would set hard ceilings for:

- LLM call duration
- sandbox analysis duration
- external retrieval attempts

#### Artifact and data lifecycle

Generated plots, temporary files, sandbox directories, and downloaded data need retention policy and cleanup policy.

#### Offline and intermittent-connectivity awareness

If this were heading toward institutional use in constrained settings, I would avoid making the entire workflow depend on live external services. At minimum, the product should make clear which features are local, which are cached, and which require active network access.

---

## f) One Component I Would Redesign: The Engine / Orchestration Layer

### Why

If I had to choose one redesign target, it would be the orchestration layer centered on `climsight_engine.py`.

This is the right place to intervene because:

- it sits on the critical path for every important user interaction
- it currently carries too much implicit shared state
- it is the main bottleneck for maintainability and testability
- improving it would make other changes easier, not harder

The current design is understandable as a fast-moving research implementation. It keeps many things close together and makes it easy to share variables across nested agent functions. But that convenience becomes a liability once multiple people need to extend, test, or operate the system.

### What I Would Change

I would separate orchestration into a small package with explicit responsibilities:

```text
src/climsight/engine/
├── __init__.py
├── context.py
├── pipeline.py
├── state.py
└── agents/
    ├── intro.py
    ├── climate_data.py
    ├── environment.py
    ├── rag.py
    ├── predefined.py
    ├── synthesis.py
    └── analysis.py
```

Instead of nested closures, each agent would take explicit dependencies, for example:

```python
@dataclass
class AgentContext:
    config: dict
    references: dict
    stream_handler: object
    llm_clients: dict
    rag_databases: dict
    runtime_paths: dict
```

That redesign would improve:

- testability
- discoverability
- contributor onboarding
- separation of concerns
- ability to document and reason about degraded modes

I would keep the overall LangGraph model. I would not replace the workflow idea itself. The problem is not that the system is agentic; the problem is that its agentic behavior is concentrated in a file structure that makes change harder than it should be.

---

## g) How I Would Build Community and a Sustainable Model Around It

To me, this is not just a funding question. It is also a maintainership and governance question.

### 1. Define the project boundary clearly

The first step is to be explicit about what ClimSight is and is not:

- core analysis engine
- supported climate providers
- supported deployment modes
- extension points for tools and datasets

Open source projects become hard to maintain when the boundary is fuzzy and every feature request feels like it might belong.

### 2. Establish maintainer discipline early

If this project wants to grow sustainably, maintainers need a visible process for deciding:

- what gets accepted
- what stays experimental
- what belongs in a fork or downstream implementation

I would add:

- `CONTRIBUTING.md`
- a lightweight RFC template for larger changes
- issue labels for "good first issue", "design discussion", "provider addition", and "research idea"
- a published roadmap

One theme that came through strongly in my conversation with Satish was that maintainers own the long-term cost of code, not just the short-term excitement of a feature. I agree with that framing. A healthy project says "no" clearly when a feature increases maintenance cost more than it increases long-term value.

### 3. Build community around real implementers, not just developers

For a project like this, the community should include:

- climate researchers
- public-sector technical teams
- NGOs and policy implementers
- data providers and scientific collaborators
- developers extending integrations

That means community material should not be limited to code contribution. It should also include:

- deployment documentation
- reference data packages
- known limitations
- example use cases
- comparison of deployment profiles

### 4. Create a path from research code to trusted public infrastructure

If the long-term ambition is broader public adoption, the project needs:

- self-hostable core
- transparent governance
- documented release process
- stable interfaces for extension
- clear provenance for data and citations

That is the difference between "interesting open source repo" and "software institutions will actually rely on".

### 5. Sustainability model

I see three realistic funding paths that can coexist:

#### Institutional partnerships

Organizations fund new integrations, provider support, or region-specific capabilities while the core stays open.

#### Implementation and support services

Offer setup, deployment, adaptation, training, and maintenance support for institutions that want to run the system but do not want to build internal expertise from scratch.

#### Hosted demo / reference environment

Not as the only model, but as a way to reduce evaluation friction and help institutions understand the product before committing to self-hosting.

The key principle is:

Keep the core open, keep governance credible, and fund long-term work through services, partnerships, and targeted expansion rather than by closing the platform.

---

## h) How I Approached This Assignment

### Process

I approached the assignment as a systems review, not just a code skim.

My process was:

1. Read the project framing in `README.md` and `CLAUDE.md`.
2. Trace the main request path from frontend to backend to engine.
3. Read the orchestration layer in detail.
4. Read the provider abstraction, RAG path, and analysis agent separately.
5. Check the deployment and test surface.
6. Run parts of the system locally and inspect real failure paths.

The most important step was tracing the actual request flow:

- frontend WebSocket client
- `api/routes/analysis.py`
- `location_request()`
- `llm_request()` / `agent_clim_request()`
- the LangGraph agents
- artifact and response handling back to the client

That gave me a concrete mental model of how the system behaves, not just how it is described.

### Tools I Used

I used:

- terminal tools such as `rg`, `sed`, `git`, and `pytest`
- direct code reading of the main Python and frontend files
- local execution of targeted tests
- an AI assistant(codex and claude-code) to speed up repository navigation and synthesis

Where I used AI, I treated it as an accelerator for exploration, not as a source of truth. I manually checked the code paths behind the claims in this review.

### What Surprised Me

**The climate-science side is stronger than the infrastructure side.** The spatial handling and provider abstraction are much more sophisticated than the operational scaffolding around them.

**The system is more ambitious than a typical "chat with your data" app.** The sandboxed Python analysis path, plotting, and reflection tools push it into a much more capable class of system.

**The biggest risks are not in the climate math.** They are in orchestration, failure visibility, operational discipline, and maintainability.

**The codebase has a credible core.** I do not see this as a toy. I see it as a serious analysis engine that now needs the kind of engineering work that helps research systems survive real users.
