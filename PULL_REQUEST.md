# FIRST ACCEPT PREVIOUS PR ;)

**This PR is based on PR #197 (analysis modes) and must be merged after it.**

---

## Data Tab: Downloadable Datasets

Adds a new **Data** tab to the UI where users can download all datasets generated during a session. Also renames "Additional information" → "Figures".

### What's new

- **`downloadable_datasets` tracking** — a new field on `AgentState` that accumulates dataset entries (`{label, path, source}`) as they're created throughout the pipeline
- **Climate model CSVs** — tracked after `write_climate_data_manifest()` in `data_agent`
- **ERA5 climatology JSON** — tracked in `prepare_predefined_data()` after extraction
- **ERA5 time series Zarr** — tracked in `data_analysis_agent` after `retrieve_era5_data` tool execution
- **DestinE time series Zarr** — tracked in `data_analysis_agent` after `retrieve_destine_data` tool execution
- **Data tab in UI** — lists all tracked datasets with download buttons; Zarr directories are zipped on the fly, JSON/CSV files download directly
- **Tab rename** — "Additional information" → "Figures"
- **Data tab always visible** — shown regardless of whether figures are available

### Pipeline fix

Each agent node now **returns** `downloadable_datasets` in its return dict so LangGraph properly merges state across stages (in-place mutation alone is not enough).

### Files changed

| File | Change |
|------|--------|
| `climsight_classes.py` | Add `downloadable_datasets: list = []` to `AgentState` |
| `climsight_engine.py` | Track datasets in `data_agent`, `prepare_predefined_data`, pass through `combine_agent` |
| `data_analysis_agent.py` | Track ERA5/DestinE Zarr outputs from tool intermediate steps |
| `streamlit_interface.py` | Rename tab, add Data tab with download buttons |

### Works in all modes

- **fast** — climate model CSVs + ERA5 climatology JSON
- **smart** — above + ERA5 time series Zarr
- **deep** — above + DestinE time series Zarr
