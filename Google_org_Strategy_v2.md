# Google.org AI for Science — Revised Proposal Strategy

## The Strategic Reframe

**Old framing:** ClimAssist is a climate-health-agriculture decision platform for Katsina State.

**New framing:** We're building an **open-source climate-health decision intelligence framework** — a reusable stack that any state, district, or LGA in the Global South can deploy to connect climate projections with local health and agriculture outcomes. Katsina State is the first deployment, not the product.

The Google.org grant funds the *infrastructure and open science*, not just the Katsina pilot.

---

## Why This Reframe Works

Google.org's evaluation criteria map perfectly:

| Criterion | How we hit it |
|-----------|---------------|
| **Scientific Ambition & Impact** | Novel: no existing tool connects sub-national climate projections (ERA5, nextGEMS, DestinE) with LGA-level health surveillance and crop suitability in a single decision loop. We're building the missing middle between global climate models and local health/agriculture action. |
| **Innovative & Responsible Use of AI** | Multi-agent RAG architecture grounded in IPCC literature, not hallucinating. Open weights regional fine-tuning. Transparent reasoning chains. |
| **Feasibility** | ClimSight core already works (we forked it, fixed bugs, deployed). Katsina has institutional buy-in (SA on Climate Change). We have a working prototype, not a slide deck. |
| **Scalability & Sustainability** | OSS framework means any state can spin up their own instance. Government cost-sharing model. Framework, not a product. |

**Critical requirement they enforce:** *"AI should be a core component of the solution, developed in alignment with Google's Responsible AI Principles and **shared via open-source licensing** to benefit the public."*

This isn't optional — open source is mandatory. That's why the framing has to be about building public infrastructure, not a proprietary platform.

---

## What We're Actually Building (OSS Deliverables)

### Layer 1: Climate-Health Data Pipeline (the plumbing)
- **Open-source connectors** for ERA5 reanalysis, nextGEMS projections, and DestinE Climate DT data → standardized local climate indicators at district/LGA resolution
- **Health data ingestion toolkit**: scripts and schemas to normalize facility-level disease surveillance data (malaria cases, malnutrition admissions, respiratory illness) from DHIS2 or paper-based systems into the pipeline
- **Crop suitability engine**: ECOCROP database + climate projections → planting calendars and yield risk by location

This is the stuff that currently doesn't exist in one place. Climate scientists have ERA5 tools. Health informaticians have DHIS2. Agricultural extension has ECOCROP. Nobody has wired them together at the local level.

### Layer 2: Multi-Agent RAG Engine (the brain)
- Fork and extend ClimSight's LangGraph architecture into a **configurable, deployable framework**
- RAG over IPCC AR6, regional climate assessments, national health strategies, and agricultural guidance — all pluggable
- Regional embedding models and retrieval pipelines that work on low-bandwidth infrastructure
- All agents, prompts, and retrieval logic open-sourced under Apache 2.0

### Layer 3: Decision Interfaces (the last mile)
- **Policy dashboard**: Web-based, LGA-level climate risk maps overlaid with health and agriculture indicators. For state-level decision-makers.
- **Farmer advisory**: SMS/WhatsApp-based planting guidance driven by climate projections for the specific location
- **Health early warning**: 8-week forward malaria/heat risk forecasts for primary health centers, correlated with rainfall and temperature data

### Layer 4: Evaluation & Reproducibility
- Open datasets: curated climate-health correlation datasets for Katsina (first), extensible to other Nigerian states and Sahelian countries
- Benchmarks: evaluation framework for climate-health prediction accuracy against ground truth
- Documentation: deployment playbooks so other states can replicate

---

## What the Funding Actually Pays For

| Budget Area | Purpose | Est. Range |
|------------|---------|------------|
| **Climate data infrastructure** | Cloud compute for ERA5/DestinE processing, storage for high-res projections | $150-250K |
| **Health data digitization** | Partner with Katsina primary health centers to digitize malaria/nutrition records across priority LGAs | $200-300K |
| **Engineering** | 2-3 engineers for 18 months building the OSS framework, connectors, and interfaces | $300-500K |
| **Field deployment** | Extension worker training, SMS gateway, local server infrastructure | $100-150K |
| **Evaluation & research** | Ground truth validation, publication of climate-health correlation findings | $100-150K |
| **Community & OSS** | Documentation, contributor onboarding, workshops for other states/countries to adopt | $50-100K |

**Total ask: $1M–$1.5M** (sweet spot for Google.org — ambitious but not overreaching)

---

## The Scope Health Angle

### What they've built (validated insights from their work)

Scope Impact (Helsinki, est. 2008, ~20 core team) is the most relevant organization in the climate-health space right now:

- **Hala Health**: AI self-care companion, 7 modules (mental health, immunization, reproductive health, nutrition, etc.), Flutter app, tested in Manipur/South Africa/Bihar, beta May 2026, targeting 250M young adults by 2030
- **CHIP**: methodology for retrofitting health facilities against extreme heat/flooding
- **CHART**: "Climate x Health Adaptation and Resilience Tool" — decision support for local health actors. First focus: maternal/newborn/child health + heat stress, launching India and Kenya
- **Partnerships**: part of PATH-led #ClimateXHealth Challenge alongside Rockefeller Foundation, Gates Foundation India, USAID, Selco Foundation
- **WHO partnership**: validating Hala Health against WHO's digital health framework for government recommendation
- **Tech approach**: building own LLM (MedGemma → OLMO for transparency), Flutter mobile, 3-week sprints, small distributed team

### How Scope validates our thesis

1. **CHART is the exact thing we're building** — but they haven't built it yet. They described it as an "AI driven climate and health adaptation and resilience tool" for local health system actors. Their website says it "provides identification of contextual climate hazards and linked health risks, assessments of local vulnerability and resilience gaps, and provision of evidence-based interventions." That's ClimAssist's architecture almost word for word.

2. **They have community-level data we don't.** Hala Health generates behavioral health data from individuals. ClimAssist operates at the climate projection → policy level. Together, you'd have the full stack: individual behavior → facility data → climate projections → government action.

3. **Their partnership network is exactly the funders we'd want.** PATH, Rockefeller, Gates Foundation are all in the climate-health funding ecosystem. Being aligned with (or partnered with) an organization in that network strengthens any application.

4. **PATH's Digital Square initiative** is specifically advancing Digital Public Goods (DPGs) for climate-health data systems with Rockefeller + Wellcome funding ($1.5M). Our OSS framework aligns perfectly with the DPG framing.

### Partnership options (in order of strategic value)

**Option A — Letter of support / advisory relationship.** Scope endorses the ClimAssist proposal. They're referenced as a complementary deployment partner. Low commitment, adds credibility. Easiest to execute before April 17.

**Option B — Named implementation partner for health data layer.** Scope provides the community health data expertise and Hala Health integration. Scolo provides the climate intelligence and policy dashboard. The proposal explicitly builds the connector between the two. Stronger application, requires negotiation.

**Option C — Joint application.** Both organizations co-apply. Scope brings the health track credibility + WHO/PATH/Rockefeller relationships. Scolo brings the climate track + Katsina government relationship + working ClimSight fork. This is the strongest possible application but hardest to coordinate in 3 weeks.

---

## The Narrative Arc for the Application

### Opening (the gap)

Climate data exists at global scale (ERA5, CMIP6, DestinE). Health surveillance data exists at facility level. Agricultural guidance exists in databases like ECOCROP. But in Katsina State, Nigeria — where 77% of severe childhood malaria correlates with seasonal rainfall, where 60% of farmlands face drought, where 75% of the population depends on agriculture — no local decision-maker can access any of it in a form they can act on.

This isn't a data problem. It's an integration and last-mile intelligence problem.

### Middle (what we're building)

An open-source climate-health decision intelligence framework that:
- Ingests climate projections (ERA5, nextGEMS, DestinE) and downscales to LGA resolution
- Correlates with local health surveillance and agricultural suitability data
- Uses multi-agent AI with RAG over scientific literature to provide grounded, citation-backed analysis
- Delivers actionable outputs: farmer planting guidance, health early warnings, policy dashboards
- Is fully open source (Apache 2.0) and designed for replication across the Sahel and beyond

### Closing (why now, why us)

- Working prototype: ClimSight fork with bug fixes, RAG pipeline, ERA5/DestinE integration
- Institutional buy-in: Katsina State SA on Climate Change is our implementation partner
- Katsina is uniquely ready: #2 in Nigeria's climate governance ranking, 30% budget pledged to climate resilience, 30 bankable climate projects, Africa's first subnational green procurement EO
- The framework we build for Katsina becomes a template for 774 LGAs across Nigeria and every climate-vulnerable district in the Sahel

---

## What Changes From v1 Vision Doc

| Aspect | v1 (ClimAssist for Katsina) | v2 (OSS Framework + Katsina pilot) |
|--------|---------------------------|--------------------------------------|
| **Framing** | Product for one state | Open infrastructure for any state |
| **Primary output** | Dashboard + advisory system | Open-source framework + datasets + deployment playbook |
| **Health data** | "Digitize basic health indicators" | Structured ingestion toolkit compatible with DHIS2 and community health apps (e.g., Hala Health) |
| **Funding justification** | Build a platform | Fund open science + public infrastructure |
| **Scalability story** | "Other states can adopt later" | "The OSS framework is the deliverable; Katsina proves it works" |
| **AI emphasis** | Implicit (uses LLMs) | Core: multi-agent RAG, regional fine-tuning, responsible AI, transparent reasoning |
| **OSS** | Not mentioned | Apache 2.0 licensing, contributor docs, replication playbooks |

---

## Immediate Action Items (Before April 17)

1. **Restructure the application narrative** around the OSS framework, not the Katsina product
2. **Reach out to Scope's Mari** — frame it as: "We're building the climate intelligence layer that CHART needs. Can we reference your work / get a letter of support for our Google.org application?"
3. **Get a letter from Professor Al-Amin** confirming Katsina State's commitment as implementation partner
4. **Prepare a technical architecture diagram** showing the OSS stack layers and how they connect to existing tools (DHIS2, Hala Health, ECOCROP, ERA5)
5. **Identify 2-3 academic advisors** who can lend credibility (climate science, health informatics, AI ethics)
6. **Write the application** — the Submittable form is at googlenewaccount2.submittable.com
7. **Prepare a 3-minute demo video** of ClimSight running with Katsina data (if the challenge accepts supplementary materials)

---

## Risk Assessment

| Risk | Mitigation |
|------|-----------|
| Scolo is an LLC, not an NGO | Google.org explicitly allows "for-profit social enterprise company with a clear and explicit social impact purpose." Frame Scolo's mission statement accordingly. |
| New entity, no track record | Emphasize: working prototype (ClimSight fork), personal track record (biomedical AI, policy work), institutional partner (Katsina State government) |
| 3 weeks to deadline | The core work is done — ClimSight exists, Katsina relationship exists. The application is writing + framing, not building from scratch. |
| Scope partnership may not materialize in time | Option A (letter of support) is low-friction. Even without Scope, the application stands on its own merits. Reference their work as validating the space. |
| Health data digitization is hard | Scope this as Phase 2 contingent. Phase 1 uses publicly available disease burden data + ERA5 climate data to demonstrate the correlation framework. |
