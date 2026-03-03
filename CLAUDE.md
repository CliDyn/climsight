# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ClimSight is a climate decision support system that integrates Large Language Models (LLMs) with climate data to provide localized climate insights. It uses a multi-agent architecture built on LangGraph to combine climate model data, geographic information, RAG (Retrieval Augmented Generation) from scientific reports, and LLM reasoning.

**Key Technologies:**
- **LangChain/LangGraph**: Multi-agent orchestration framework
- **Streamlit**: Web UI framework
- **xarray/NetCDF4**: Climate data processing
- **Chroma**: Vector database for RAG
- **GeoPandas/OSMnx**: Geospatial analysis
- **OpenAI API**: LLM backend (supports custom models via AITTA platform)
- **earthkit.data**: DestinE data retrieval via polytope
- **Arraylake**: ERA5 reanalysis data access

## Common Commands

### Environment Setup

```bash
# Using conda/mamba (recommended)
mamba env create -f environment.yml
conda activate climsight
python download_data.py  # Downloads ~8GB of climate data

# Using pip
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e .
python download_data.py
```

### Running the Application

```bash
# Development mode (from repo root)
streamlit run src/climsight/climsight.py

# If installed via pip
climsight

# Testing mode (no OpenAI API calls)
streamlit run src/climsight/climsight.py skipLLMCall
```

### Testing

```bash
# Run all tests (DestinE tests excluded by default)
cd test
pytest

# Run specific test categories (see test/pytest.ini for all markers)
pytest -m geo           # Geographic functions
pytest -m climate       # Climate data functions
pytest -m env           # Environmental functions
pytest -m "not request" # Skip tests requiring HTTP requests

# DestinE tool tests (require ~/.polytopeapirc token + OPENAI_API_KEY)
pytest -m destine -v                    # All DestinE tests
pytest -m destine -v -k search          # RAG search only (fast)
pytest -m destine -v -k retrieve        # Data retrieval only

# Run single test file
pytest test_geofunctions.py

# Run with verbose output
pytest -v
```

### Linting

```bash
# Syntax errors and undefined names only (CI uses this)
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
```

### Batch Processing (sequential/)

```bash
cd sequential

# Generate climate questions
python question_generator.py

# Visualize questions on map
streamlit run question_map.py

# Process questions through ClimSight
python question_runner.py --questions_file Q_1.json --llm_model gpt-4.1-nano
```

## Architecture

### Multi-Agent Workflow (LangGraph)

ClimSight uses a state machine with specialized agents that process user questions:

1. **intro_agent** ([climsight_engine.py:1079](src/climsight/climsight_engine.py#L1079))
   - Entry point that filters invalid requests
   - Uses exclusion-based logic to determine if query is climate-related
   - Routes to either FINISH (invalid) or CONTINUE (parallel agents)

2. **Parallel Information Gathering Agents:**
   - **ipcc_rag_agent** ([climsight_engine.py:1031](src/climsight/climsight_engine.py#L1031)): Searches IPCC reports via RAG
   - **general_rag_agent** ([climsight_engine.py:1052](src/climsight/climsight_engine.py#L1052)): Searches general climate literature via RAG
   - **data_agent** ([climsight_engine.py:854](src/climsight/climsight_engine.py#L854)): Extracts climate model data for location
   - **zero_rag_agent** ([climsight_engine.py:723](src/climsight/climsight_engine.py#L723)): Gathers geographic/environmental context
   - **smart_agent** ([smart_agent.py:71](src/climsight/smart_agent.py#L71)) — OPTIONAL:
     - Information gathering only (Wikipedia, RAG, ECOCROP)
     - Controlled by `use_smart_agent` in config
     - No Python REPL — that is now in data_analysis_agent

3. **prepare_predefined_data** ([climsight_engine.py:917](src/climsight/climsight_engine.py#L917))
   - Runs after all parallel agents complete
   - Extracts ERA5 climatology and generates predefined plots (climate comparison with ERA5 overlay, disaster summary, population projection)
   - Routes via `route_after_prepare`: if `use_powerful_data_analysis` → data_analysis_agent, otherwise → combine_agent

4. **data_analysis_agent** ([data_analysis_agent.py:476](src/climsight/data_analysis_agent.py#L476)) — OPTIONAL:
   - Receives all outputs from parallel agents + predefined plots
   - Performs data extraction, post-processing, and visualization using tool-calling
   - Tools: Python REPL (Jupyter kernel), ERA5 retrieval, DestinE search + retrieval, image viewer, reflection, visualization strategy
   - Controlled by `use_powerful_data_analysis` in config

5. **combine_agent** ([climsight_engine.py:1181](src/climsight/climsight_engine.py#L1181))
   - Synthesizes all agent outputs into final answer
   - Generates references and formatted response

**Routing Logic:**
- `route_fromintro` launches parallel agents (including smart_agent if enabled)
- All parallel agents converge to `prepare_predefined_data`
- `route_after_prepare` conditionally invokes `data_analysis_agent` or skips to `combine_agent`
- `data_analysis_agent` flows to `combine_agent`

### Key Modules

**Core Engine:**
- [climsight_engine.py](src/climsight/climsight_engine.py): Main orchestration, agent definitions, workflow setup
- [climsight_classes.py](src/climsight/climsight_classes.py): `AgentState` Pydantic model for state sharing between agents
- [data_container.py](src/climsight/data_container.py): Container for DataFrames, xarray datasets, and matplotlib figures

**Data Processing:**
- [climate_functions.py](src/climsight/climate_functions.py): Load and extract climate model data (temperature, precipitation, wind)
- [extract_climatedata_functions.py](src/climsight/extract_climatedata_functions.py): Request climate data with model-specific handling
- [geo_functions.py](src/climsight/geo_functions.py): Geocoding, land/water detection, elevation, soil, land use
- [environmental_functions.py](src/climsight/environmental_functions.py): Biodiversity, natural hazards
- [economic_functions.py](src/climsight/economic_functions.py): Population data and projections

**RAG System:**
- [rag.py](src/climsight/rag.py): RAG query interface supporting multiple embedding backends (OpenAI, AITTA, Mistral)
- [embedding_utils.py](src/climsight/embedding_utils.py): Embedding model initialization for different backends

**Information Gathering Agent:**
- [smart_agent.py](src/climsight/smart_agent.py): LangChain agent with tools for Wikipedia, RAG, and ECOCROP database queries (information gathering only — no Python REPL)

**Data Analysis Agent:**
- [data_analysis_agent.py](src/climsight/data_analysis_agent.py): Full tool-calling agent for data extraction, analysis, and visualization
  - `_create_tool_prompt()` (line 120): Dynamically builds system prompt based on config
  - `data_analysis_agent()` (line 476): Main entry point
  - Registers tools: Python REPL, ERA5 retrieval, DestinE search/retrieval, image viewer, reflect, wise_agent

**Tools (src/climsight/tools/):**
- [python_repl.py](src/climsight/tools/python_repl.py): Sandboxed Python execution with persistent Jupyter kernel (used by data_analysis_agent)
- [image_viewer.py](src/climsight/tools/image_viewer.py): View and analyze generated plots
- [reflection_tools.py](src/climsight/tools/reflection_tools.py): Quality feedback on generated plots (7/10 threshold)
- [visualization_tools.py](src/climsight/tools/visualization_tools.py): List sandbox files and visualization strategy advice
- [predefined_plots.py](src/climsight/tools/predefined_plots.py): Standard climate visualizations (climate comparison with ERA5 overlay, disaster summary, population projection)
- [era5_climatology_tool.py](src/climsight/tools/era5_climatology_tool.py): Extract ERA5 ground truth (10-year climatology)
- [era5_retrieval_tool.py](src/climsight/tools/era5_retrieval_tool.py): Download ERA5 time series via Arraylake
- [destine_retrieval_tool.py](src/climsight/tools/destine_retrieval_tool.py): DestinE parameter search (RAG over 82 params) + data retrieval via earthkit.data/polytope

**Interfaces:**
- [streamlit_interface.py](src/climsight/streamlit_interface.py): Web UI with map selection
- [terminal_interface.py](src/climsight/terminal_interface.py): CLI interface
- [stream_handler.py](src/climsight/stream_handler.py): Progress updates for UI

## Configuration

**Primary Config:** [config.yml](config.yml)

Key configuration sections:
- `model_type`: "openai" | "local" | "aitta"
- `model_name*`: Different models for RAG, tools, agents, combine step
- `use_smart_agent`: Enable/disable information gathering agent
- `use_powerful_data_analysis`: Enable/disable data_analysis_agent (Python REPL + tools)
- `use_high_resolution_climate_model`: Use high-res nextGEMS data
- `climate_model_input_files`: Map of NetCDF files with metadata
- `rag_settings`: Embedding model type, Chroma DB paths per backend
- `rag_template`: System prompt for RAG queries
- `era5_climatology`: ERA5 ground truth settings
- `use_destine_data`: Enable/disable DestinE Climate DT data retrieval
- `destine_settings`: Chroma DB path and collection name for parameter search

**Data Sources:** [data_sources.yml](data_sources.yml)
- Defines remote data URLs and local extraction paths
- Used by `download_data.py` to fetch climate data, natural hazards, population, geographic boundaries

**Reference Data:** [references.yml](references.yml)
- Citation information for datasets and reports
- Automatically added to outputs

## Important Patterns

### LangChain Import Strategy

The codebase handles LangChain 1.0+ migration with try/except blocks:

```python
try:
    from langchain.chains import LLMChain
except ImportError:
    from langchain_classic.chains import LLMChain
```

This pattern appears in [climsight_engine.py](src/climsight/climsight_engine.py#L24-27) and [smart_agent.py](src/climsight/smart_agent.py#L11-14). Maintain this pattern when adding LangChain imports.

### Climate Data Structure

Climate data uses xarray Datasets loaded via [climate_functions.py](src/climsight/climate_functions.py):
- Historical data: `data['hist']` (typically 1995-2014)
- Future projections: `data['future']` (2020-2049 in decadal chunks)
- Variables mapped via `config['variable_mappings']`: Temperature→tas, Precipitation→pr, etc.
- Coordinates mapped via `config['dimension_mappings']`: lat, lon, month

High-resolution nextGEMS data uses HEALPix coordinate system (see `coordinate_system: 'healpix'` in config).

### Agent State Management

All agents receive `AgentState` ([climsight_classes.py](src/climsight/climsight_classes.py)) and return dictionaries that update state:

```python
def my_agent(state: AgentState):
    # Access shared state
    user_query = state.user
    lat = float(state.input_params['lat'])

    # Process...

    # Return updates (merged into state)
    return {'my_agent_response': result}
```

Key AgentState fields:
- **Agent outputs:** `data_agent_response`, `zero_agent_response`, `ipcc_rag_agent_response`, `general_rag_agent_response`, `smart_agent_response`, `data_analysis_response`
- **Sandbox paths:** `thread_id`, `uuid_main_dir`, `results_dir`, `climate_data_dir`, `era5_data_dir`, `destine_data_dir`
- **Artifacts:** `df_list` (climate DataFrames), `predefined_plots`, `data_analysis_images`, `references`
- **ERA5/DestinE:** `era5_climatology_response`, `era5_tool_response`, `destine_tool_response`

The workflow automatically merges return values into state for downstream agents.

### Predefined Plots with ERA5 Overlay

[predefined_plots.py](src/climsight/tools/predefined_plots.py) generates standard climate visualizations with ERA5 observational data overlay:
- Supports multiple climate model variable naming conventions (nextGEMS: mean2t/tp, AWI-CM: tas/pr, DestinE: avg_2t/avg_tprate)
- Maps model variables to ERA5 equivalents via `era5_var_map` and `descriptive_era5_map`
- Handles cross-dataframe column matching for models with different historical/future column names (e.g., AWI-CM)
- Computes ERA5 wind speed from u10/v10 components

### DestinE Data Retrieval

The DestinE tool uses a two-step workflow:
1. **Parameter search** — RAG semantic search over 82 DestinE Climate DT parameters via Chroma vector store (`data/destine/chroma_db/`)
2. **Data retrieval** — Download via `earthkit.data.from_source("polytope", "destination-earth", ...)` using token from `~/.polytopeapirc`

Authentication: Run `desp-authentication.py` to obtain a token (written to `~/.polytopeapirc`). No username/password passed at runtime.

### RAG Database Initialization

RAG databases are initialized in [climsight_engine.py](src/climsight/climsight_engine.py) before workflow creation:
- Checks if Chroma DB exists with `is_valid_rag_db()`
- Falls back to creating new DB if invalid
- Supports multiple backends (OpenAI, AITTA) with separate DB paths

Embedding backend is selected via `config['rag_settings']['embedding_model_type']`.

### Location Validation

Before processing any query, [location_request()](src/climsight/climsight_engine.py#L113) validates the point is on land:
- Returns `(None, None)` if point is in ocean (line 139)
- Distinguishes between ocean and inland water bodies
- Fetches address, elevation, soil, land use data

This critical check prevents wasted processing on invalid locations.

## Development Notes

### Adding New Climate Variables

1. Add NetCDF file to `config['climate_model_input_files']`
2. Add variable mapping to `config['climate_model_variable_mapping']` or `config['variable_mappings']`
3. Update [extract_climatedata_functions.py](src/climsight/extract_climatedata_functions.py) to handle new variable
4. Update `data_agent` in [climsight_engine.py](src/climsight/climsight_engine.py#L854) to include in prompt
5. Update ERA5 variable maps in [predefined_plots.py](src/climsight/tools/predefined_plots.py) for overlay support

### Adding New Tools to Data Analysis Agent

1. Create tool function in [tools/](src/climsight/tools/) directory
2. Import in [data_analysis_agent.py](src/climsight/data_analysis_agent.py)
3. Register tool in the tools list (around line 595)
4. Add tool description to `_create_tool_prompt()` (line 120) so the agent knows when to use it

### Adding New Tools to Smart Agent

1. Create tool function in [tools/](src/climsight/tools/) directory
2. Import and register in [smart_agent.py](src/climsight/smart_agent.py)
3. Update system prompt to describe when to use the tool

### Testing with Mock Data

Use `skipLLMCall` mode to bypass OpenAI API:
- Test files in [test/](test/) use mock configs
- Expected outputs stored as CSV files (e.g., `expected_df_climate_data.csv`)
- Use pytest markers to skip network-dependent tests
- DestinE tests require `-m destine` flag and `~/.polytopeapirc` token

### Logging

All modules log to `climsight.log` in the working directory. Check this file for detailed execution traces when debugging agent behavior.

### Model Compatibility

The config supports multiple LLM model types:
- Standard OpenAI models (gpt-4o, gpt-4.1-nano, etc.)
- o1 models (automatically sets temperature=1, see [smart_agent.py:77](src/climsight/smart_agent.py#L77))
- AITTA platform models via `get_aitta_chat_model()` function

When adding new models, check temperature requirements and tool-calling compatibility.

### Prompt Template Safety

When adding code examples to agent prompts (e.g., in `_create_tool_prompt()`), escape curly braces as `{{}}` — otherwise `ChatPromptTemplate` interprets `{}` as a template variable placeholder.
