# ClimSight

ClimSight is an AI-powered climate decision-support system that combines Large Language Models with multi-source climate data to deliver localized, actionable insights. It serves researchers, policymakers, agricultural planners, urban developers, and anyone who needs to understand how climate change affects real-world decisions.

![Image](https://github.com/user-attachments/assets/f9f89735-ef08-4c91-bc03-112c8e4c0896)

### What makes it different

- **Agentic AI pipeline** — specialized agents (smart agent, data analysis agent, search agent) orchestrate data retrieval, code execution, and synthesis autonomously
- **Multi-source climate data** — nextGEMS, ICCP, AWI-CM CMIP6, DestinE, and ERA5 reanalysis via Arraylake
- **RAG-augmented responses** — retrieves context from IPCC reports and scientific literature
- **Interactive map-based interface** — click anywhere on the globe and ask a climate question
- **Python REPL execution** — the agent writes and runs analysis code in a sandboxed Jupyter kernel

---

## Architecture

ClimSight has two interface modes:

| | **React UI** (new) | **Streamlit UI** (legacy) |
|---|---|---|
| Frontend | React + TypeScript + Tailwind CSS + Vite | Streamlit |
| Backend  | FastAPI + WebSocket (real-time streaming) | Streamlit server |
| Command  | `uvicorn` + `npm run dev` | `streamlit run` |

The **React UI** is the actively developed interface with real-time streaming, a dark/light theme toggle, and a modern component architecture. The Streamlit UI is kept for backwards compatibility.

---

## Quick Start (React UI)

### Prerequisites

- **Python 3.11+**
- **Node.js 18+** and **npm**
- An **OpenAI API key** (or other supported LLM provider)

### 1. Clone & set up the Python environment

```bash
git clone https://github.com/CliDyn/climsight.git
cd climsight

# Option A: conda/mamba (recommended)
mamba env create -f environment.yml
conda activate climsight

# Option B: pip + venv
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-api.txt
```

### 2. Download the climate data

```bash
python download_data.py

# Optional: DestinE data (~12 GB)
python download_data.py DestinE
```

### 3. Configure API keys

Create a `.env` file in the repo root (or export the variables):

```bash
OPENAI_API_KEY="sk-..."

# Optional — enables ERA5 time series retrieval
ARRAYLAKE_API_KEY="your-arraylake-key"
```

### 4. Start the FastAPI backend

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

The API runs on `http://localhost:8000`. Health check: `GET /health`.

### 5. Start the React frontend

```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173** in your browser. The Vite dev server automatically proxies `/api` and `/artifacts` requests to the backend.

---

## Running the Legacy Streamlit UI

If you prefer the original interface:

```bash
streamlit run src/climsight/climsight.py
```

Opens on `http://localhost:8501`.

---

## Docker (Stable Release v1.0.0)

> [!NOTE]
> The Docker image ships the Streamlit UI (v1.0.0). The React UI is not yet containerized.

```bash
export OPENAI_API_KEY="sk-..."
docker pull koldunovn/climsight:stable
docker run -p 8501:8501 -e OPENAI_API_KEY=$OPENAI_API_KEY koldunovn/climsight:stable
```

---

## Configuration

ClimSight reads `config.yml` from the working directory. Key settings:

```yaml
llm_smart:
  model_type: "openai"
  model_name: "gpt-5.2"         # main reasoning model

use_smart_agent: true            # enable the agentic pipeline
use_era5_data: true              # ERA5 time series via Arraylake
use_powerful_data_analysis: true # agent writes & executes Python code

climate_data_source: "nextGEMS"  # nextGEMS | ICCP | AWI_CM | DestinE
```

See `config.yml` for the full reference (climate data sources, RAG settings, system prompts, etc.).

---

## Project Structure

```
climsight/
├── api/                    # FastAPI backend
│   ├── main.py             #   app factory, CORS, static files
│   └── routes/             #   REST + WebSocket endpoints
├── frontend/               # React UI (Vite + Tailwind + TypeScript)
│   ├── src/
│   │   ├── App.tsx          #   main app shell
│   │   ├── components/      #   MapPanel, QueryForm, ReportView, SettingsPanel, StatusBar
│   │   └── api/             #   API client
│   ├── package.json
│   └── vite.config.ts
├── src/climsight/          # Core Python engine
│   ├── smart_agent.py      #   agentic orchestration (LangGraph)
│   ├── agent_helpers.py    #   tool definitions & agent utilities
│   ├── session_manager.py  #   per-session state & memory
│   └── tools/              #   ERA5 retrieval, reflection, REPL, search
├── data/                   # Climate datasets (downloaded via download_data.py)
├── rag_db/                 # ChromaDB vector stores for RAG
├── config.yml              # Main configuration
├── environment.yml         # Conda environment spec
├── requirements.txt        # Pip dependencies (core)
├── requirements-api.txt    # Pip dependencies (FastAPI backend)
└── pyproject.toml          # Package metadata
```

---

## Batch Processing

The `sequential/` directory contains tools for generating, validating, and processing climate questions in bulk. See [sequential/README.md](sequential/README.md).

---

## Citation

If you use or refer to ClimSight in your work, please cite:

> Kuznetsov, I., Jost, A.A., Pantiukhin, D. et al. Transforming climate services with LLMs and multi-source data integration. _npj Clim. Action_ **4**, 97 (2025). https://doi.org/10.1038/s44168-025-00300-y

> Koldunov, N., Jung, T. Local climate services for all, courtesy of large language models. _Commun Earth Environ_ **5**, 13 (2024). https://doi.org/10.1038/s43247-023-01199-1
