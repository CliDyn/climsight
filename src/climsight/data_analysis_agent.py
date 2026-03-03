"""
Data analysis agent for Climsight.

This agent mirrors PangaeaGPT's oceanographer/visualization style while operating
on local climatology. It filters context, then uses tools to extract or analyze
climate data, saving outputs into the sandbox.
"""

import json
import logging
import os
from typing import Any, Dict, List

from climsight_classes import AgentState

try:
    from utils import make_json_serializable
except ImportError:
    from .utils import make_json_serializable
from sandbox_utils import ensure_thread_id, ensure_sandbox_dirs, get_sandbox_paths
from agent_helpers import create_standard_agent_executor
from tools.era5_retrieval_tool import create_era5_retrieval_tool
from tools.destine_retrieval_tool import create_destine_search_tool, create_destine_retrieval_tool
from tools.python_repl import CustomPythonREPLTool
from tools.image_viewer import create_image_viewer_tool
from tools.reflection_tools import reflect_tool
from tools.visualization_tools import (
    list_plotting_data_files_tool,
    wise_agent_tool,
)

from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)


def _build_climate_data_summary(df_list: List[Dict[str, Any]]) -> str:
    """Summarize available climatology without exposing raw values."""
    if not df_list:
        return "No climatology data available."

    lines = []
    for entry in df_list:
        vars_summary = []
        for var_name, var_info in entry.get("extracted_vars", {}).items():
            full_name = var_info.get("full_name", var_name)
            units = var_info.get("units", "")
            vars_summary.append(f"{full_name} ({units})")
        vars_text = ", ".join(vars_summary) if vars_summary else "Unknown variables"
        lines.append(
            f"- {entry.get('years_of_averaging', '')}: {entry.get('description', '')} | {vars_text}"
        )

    return "\n".join(lines)


def _build_datasets_text(state) -> str:
    """Build simple dataset paths text for prompt injection.

    IMPORTANT: The Python REPL kernel CWD is already set to the sandbox root,
    so we tell the agent to use RELATIVE paths (not full tmp/sandbox/... paths).
    """
    lines = [
        "## Sandbox Paths (Python REPL is ALREADY inside the sandbox directory)",
        "**CRITICAL: Use RELATIVE paths in your Python code, NOT full paths starting with 'tmp/sandbox/...'**",
        f"- Current Working Directory: '.' (which is {state.uuid_main_dir})",
        f"- Results directory: 'results' (save all plots here)",
        f"- Climate data: 'climate_data'",
    ]

    if state.era5_data_dir:
        lines.append(f"- ERA5 data: 'era5_data'")
    if state.destine_data_dir:
        lines.append(f"- DestinE data: 'destine_data/'")

    # List available climate data files
    if state.climate_data_dir and os.path.exists(state.climate_data_dir):
        try:
            files = os.listdir(state.climate_data_dir)
            if files:
                lines.append(f"\n## Climate Data Files Available (in 'climate_data/' folder)")
                lines.append(f"Files: {', '.join(files)}")
                # Highlight the main data.csv file
                if "data.csv" in files:
                    lines.append("Note: Load with `pd.read_csv('climate_data/data.csv')`")
        except Exception as e:
            logger.warning(f"Could not list climate data files: {e}")

    return "\n".join(lines)


def _build_filter_prompt() -> str:
    """Prompt for the analysis brief filter LLM."""
    return (
        "You are a context filter for ClimSight's data analysis agent.\n"
        "Your output will be consumed by an agent that has Python REPL, ERA5 data access,\n"
        "and climate model data. Focus on what it should COMPUTE and PLOT.\n\n"
        "Extract ONLY actionable analysis requirements as concise bullets:\n"
        "- Target variables with units (e.g., 'Temperature (°C)', 'Precipitation (mm/month)')\n"
        "- Quantitative thresholds or criteria (e.g., 'days above 35°C', 'monthly rainfall < 50mm')\n"
        "- Time ranges or scenario labels (e.g., '2020-2029 vs 2040-2049', 'SSP5-8.5')\n"
        "- Spatial specifics (location name, coordinates, search radius)\n"
        "- Requested analyses (trend detection, seasonal comparison, anomaly identification, custom plots)\n"
        "- Mentioned crops, infrastructure, or decision topics (e.g., 'wheat cultivation', 'solar panel siting')\n\n"
        "Rules:\n"
        "- Do NOT include raw climate data values or lengthy text passages.\n"
        "- Do NOT include RAG or Wikipedia excerpts — only summarize their KEY requirements.\n"
        "- Omit vague statements that cannot be translated into a computation or plot.\n"
        "- If no specific analysis is requested, default to: temperature trends, precipitation comparison,\n"
        "  wind assessment, and a climate change signal summary.\n\n"
        "Example output:\n"
        "- Temperature (°C): compare ERA5 baseline vs model projections for all months\n"
        "- Precipitation (mm/month): identify driest and wettest months, flag months below 30mm\n"
        "- Wind speed (m/s): assess seasonal wind patterns\n"
        "- Decision context: wheat cultivation — flag months with heat stress risk (>30°C)\n"
        "- Requested plots: temperature trend with ERA5 overlay, precipitation seasonal bar chart\n"
    )


def _create_tool_prompt(datasets_text: str, config: dict, lat: float = None, lon: float = None,
                        has_climate_data: bool = False, has_hazard_data: bool = False,
                        has_population_data: bool = False, has_era5_data: bool = False) -> str:
    """System prompt for tool-driven analysis - dynamically built based on config.

    NOTE: This agent only runs when python_REPL is enabled (use_powerful_data_analysis=True).
    ERA5 climatology and predefined plots are already generated by prepare_predefined_data.
    """
    has_era5_download = config.get("use_era5_data", False)
    has_destine = config.get("use_destine_data", False)

    # --- Build prompt without f-strings for code blocks to avoid brace escaping ---
    sections = []

    # ── ROLE ──────────────────────────────────────────────────────────────
    sections.append(
        "You are ClimSight's data analysis agent.\n"
        "Your job: provide ADDITIONAL quantitative climate analysis beyond the standard plots.\n"
        "You have a persistent Python REPL, pre-extracted data files, and optional ERA5 download access.\n\n"
        "CRITICAL EFFICIENCY RULES:\n"
        "- HARD LIMIT: 30 tool calls total for the entire session.\n"
        "- MAX 3-4 tool calls per response. Never fire 5+ tools in a single response.\n"
        "- Write focused Python scripts — each one should accomplish a meaningful chunk of work.\n"
        "- SEQUENTIAL ordering: first Python_REPL to generate plots, THEN reflect_on_image in a LATER response.\n"
        "  Never call reflect_on_image in the same response as Python_REPL.\n"
        "- Ideal session: 3-4 REPL calls → 1-2 reflect calls → final answer. That is 6-7 tool calls total.\n"
    )

    # ── 1. DATA ALREADY AVAILABLE ─────────────────────────────────────────
    sections.append("## 1. DATA ALREADY IN THE SANDBOX (do not re-extract)\n")

    if has_era5_data:
        sections.append(
            "### ERA5 Climatology (observational ground truth)\n"
            "- File: `era5_climatology.json` in sandbox root\n"
            "- Content: monthly averages of t2m (°C), cp+lsp (precipitation, m), u10/v10 (m/s) — period 2015-2025\n"
            "- Load: `era5 = json.load(open('era5_climatology.json'))`\n"
            "- Role: treat as GROUND TRUTH for validating model data.\n"
        )

    if has_climate_data:
        sections.append(
            "### Climate Model Data\n"
            "- Manifest: `climate_data/climate_data_manifest.json` — READ FIRST to discover all simulations\n"
            "- Data files: `climate_data/simulation_N.csv` + `simulation_N_meta.json`\n"
            "- Shortcut: `climate_data/data.csv` = baseline simulation only\n"
            "- Columns: Month, mean2t (°C), cp (convective precip, m), lsp (large-scale precip, m), wind_u, wind_v, wind_speed, wind_direction\n"
        )

    # ── 2. PRE-GENERATED PLOTS ────────────────────────────────────────────
    predefined_plots = []
    if has_climate_data:
        predefined_plots.append("- `results/climate_*.png` — temperature, precipitation, wind comparison with ERA5 overlay")
    if has_hazard_data:
        predefined_plots.append("- `results/disaster_counts.png` — historical disaster events by type")
    if has_population_data:
        predefined_plots.append("- `results/population_projection.png` — population trends")

    if predefined_plots:
        sections.append(
            "## 2. PRE-GENERATED PLOTS (already created — DO NOT recreate)\n\n"
            + "\n".join(predefined_plots) + "\n\n"
            "Analyze the underlying DATA directly for your own insights.\n"
            "Only use `image_viewer` on plots YOU create, not on these predefined ones.\n"
        )
    else:
        sections.append("## 2. PRE-GENERATED PLOTS\n\nNone yet — you may create any visualizations needed.\n")

    # ── 3. ERA5 TIME SERIES (optional download) ──────────────────────────
    if has_era5_download:
        sections.append(
            "## 3. ERA5 TIME SERIES DOWNLOAD (year-by-year data)\n\n"
            "Use `retrieve_era5_data` to download full annual time series (2015-2024) from Earthmover.\n"
            "This gives you YEAR-BY-YEAR values — far richer than the 10-year climatology average above.\n\n"
            "When to download:\n"
            "- You need to detect warming/drying TRENDS → download t2, cp, and lsp (sum cp+lsp for total precip)\n"
            "- You need interannual variability or extreme-year identification → download the relevant variables\n"
            "- You only need monthly climatology for comparison → skip, use era5_climatology.json\n\n"
            "**PARALLEL DOWNLOADS**: Call `retrieve_era5_data` for ALL needed variables in a SINGLE response.\n"
            "Example: if you need t2, cp, and lsp — call all three retrieve_era5_data in ONE response, not sequentially.\n\n"
            "Tool parameters:\n"
            "- Variable codes: `t2` (temperature), `cp` (convective precip), `lsp` (large-scale precip), `u10`/`v10` (wind), `mslp` (pressure)\n"
            "- NOTE: `tp` (total precipitation) is NOT available. Use `cp` + `lsp` and sum them for total precipitation.\n"
            "- Always pass `work_dir='.'`\n"
            "- Output: Zarr store saved to `era5_data/` folder\n\n"
            "Loading ERA5 Zarr in Python_REPL:\n"
        )
        # Code example WITHOUT f-string to avoid brace escaping
        sections.append(
            "```python\n"
            "import xarray as xr, glob\n"
            "era5_files = glob.glob('era5_data/*.zarr')\n"
            "print(era5_files)\n"
            "ds = xr.open_dataset(era5_files[0], engine='zarr', chunks={{}})\n"
            "data = ds['t2'].to_series()\n"
            "```\n"
        )

    # ── 3b. DESTINE CLIMATE PROJECTIONS ─────────────────────────────────
    if has_destine:
        sections.append(
            "## 3b. DESTINE CLIMATE PROJECTIONS (SSP3-7.0, 82 parameters)\n\n"
            "You have access to the DestinE Climate DT — high-resolution projections (IFS-NEMO, 2020-2039).\n"
            "Use a TWO-STEP workflow:\n\n"
            "**Step 1: Search for ALL needed parameters FIRST**\n"
            "Before downloading anything, decide which variables you need (temperature, precipitation, wind, etc.).\n"
            "Call `search_destine_parameters` for each query to collect all param_ids and levtypes.\n"
            "Example: search_destine_parameters('temperature at 2 meters') → param_id='167', levtype='sfc'\n\n"
            "**Step 2: Download ALL variables IN PARALLEL**\n"
            "Once you have all param_ids, call `retrieve_destine_data` for ALL of them in a SINGLE response.\n"
            "The tools support parallel execution — multiple retrieve calls in one response run concurrently.\n"
            "This is MUCH faster than downloading one variable at a time sequentially.\n\n"
            "Example: if you need temperature (167) and precipitation (228), call BOTH retrieve_destine_data\n"
            "in the SAME response, not one after the other in separate responses.\n\n"
            "- Dates: YYYYMMDD format, range 20200101-20391231\n"
            "- **By default request the FULL period**: start_date=20200101, end_date=20391231 (20 years of projections)\n"
            "- Only use a shorter range if the user explicitly asks for a specific period\n"
            "- Output: Zarr store saved to `destine_data/` folder\n\n"
            "Loading DestinE data in Python_REPL:\n"
        )
        sections.append(
            "```python\n"
            "import xarray as xr, glob\n"
            "destine_files = glob.glob('destine_data/*.zarr')\n"
            "print(destine_files)\n"
            "ds = xr.open_dataset(destine_files[0], engine='zarr', chunks={{}})\n"
            "print(ds)\n"
            "```\n"
        )

    # ── 4. TOOLS ──────────────────────────────────────────────────────────
    sections.append("## 4. AVAILABLE TOOLS\n")
    tools_list = []
    if has_era5_download:
        tools_list.append("- **retrieve_era5_data** — download ERA5 year-by-year time series (see section 3)")
    if has_destine:
        tools_list.append("- **search_destine_parameters** — find DestinE parameters via RAG search (see section 3b)")
        tools_list.append("- **retrieve_destine_data** — download DestinE time series (see section 3b)")
    tools_list.append(
        "- **Python_REPL** — execute Python code in a sandboxed environment.\n"
        "  All files are relative to the sandbox root.\n"
        "  The `results/` directory is pre-created for saving plots.\n"
        "  Datasets are pre-loaded into the sandbox (see paths below).\n"
        "  STRATEGY: DIVIDE AND CONQUER. Split your work into a few focused scripts,\n"
        "  each tackling ONE logical task (e.g., load+explore, then analyze+plot-set-1,\n"
        "  then analyze+plot-set-2). This avoids cascading errors from monolithic scripts.\n"
        "  But don't go overboard with tiny one-liner calls either — find a reasonable balance.\n"
        "  Each script should be self-contained: import what it needs, do meaningful work, print results."
    )
    tools_list.append("- **list_plotting_data_files** — discover files in sandbox directories")
    tools_list.append("- **image_viewer** — view and analyze plots in `results/` (use relative paths)")
    tools_list.append(
        "- **reflect_on_image** — get quality feedback on a plot you created.\n"
        "  Call once per plot — reflect on ALL generated plots, not just one.\n"
        "  MUST be called in a SEPARATE response AFTER the Python_REPL that created the plots.\n"
        "  Always verify the file exists (via os.path.exists in REPL) BEFORE calling this tool.\n"
        "  MINIMUM ACCEPTABLE SCORE: 7/10. If score < 7, you MUST re-plot with fixes applied.\n"
        "  Read the fix suggestions from the reviewer and apply them in your next REPL call."
    )
    tools_list.append("- **wise_agent** — ask for visualization strategy advice before coding")
    sections.append("\n".join(tools_list) + "\n")

    # ── 5. WORKFLOW ───────────────────────────────────────────────────────
    step = 1
    sections.append("## 5. REQUIRED WORKFLOW\n")

    sections.append(
        f"**Step {step} — Explore and load data:**\n"
        "MANDATORY FIRST STEP: Load data files AND print their structure before any analysis.\n"
        "This prevents cascading errors from wrong column names or data formats.\n\n"
        "CRITICAL DATA FORMAT WARNINGS:\n"
        "- Month column may contain STRING NAMES ('January', 'February') — convert before using as int\n"
        "- CSV paths in manifest are FILENAMES ONLY — always prepend 'climate_data/'\n"
        "- Precipitation column may be 'tp' (total precip in mm) OR separate 'cp'/'lsp' (in meters)\n"
        "- Always print df.columns.tolist() and df.head(2) BEFORE writing analysis code\n\n"
    )
    # Code example without f-string braces
    sections.append(
        "```python\n"
        "import os, json\n"
        "import pandas as pd\n"
        "\n"
        "# 1. Load manifest and print structure\n"
        "manifest = json.load(open('climate_data/climate_data_manifest.json'))\n"
        "for e in manifest['entries']:\n"
        "    csv_path = os.path.join('climate_data', os.path.basename(e['csv']))\n"
        "    print(e['years_of_averaging'], csv_path, '(baseline)' if e.get('main') else '')\n"
        "\n"
        "# 2. Load one CSV and inspect columns/dtypes\n"
        "csv_path = os.path.join('climate_data', os.path.basename(manifest['entries'][0]['csv']))\n"
        "df = pd.read_csv(csv_path)\n"
        "print('Columns:', df.columns.tolist())\n"
        "print('Dtypes:', df.dtypes.to_dict())\n"
        "print(df.head(2))\n"
        "\n"
        "# 3. Convert Month column (handles both 'January' strings and integers)\n"
        "month_map = {{name: i+1 for i, name in enumerate(\n"
        "    ['January','February','March','April','May','June',\n"
        "     'July','August','September','October','November','December'])}}\n"
        "if not pd.api.types.is_numeric_dtype(df['Month']):\n"
        "    df['Month'] = df['Month'].map(month_map)\n"
        "df['Month'] = df['Month'].astype(int)\n"
        "```\n\n"
    )
    if has_era5_data:
        sections.append(
            "```python\n"
            "# ERA5 observations (ground truth)\n"
            "era5 = json.load(open('era5_climatology.json'))\n"
            "era5_temp = era5['variables']['t2m']['monthly_values']  # dict: month_name → value\n"
            "```\n\n"
        )
    step += 1

    if has_era5_download:
        sections.append(
            f"**Step {step} — (Optional) Download ERA5/DestinE data — ALL IN PARALLEL:**\n"
            "Decide which variables you need, then call ALL download tools in a SINGLE response.\n"
            "ERA5: call `retrieve_era5_data` for t2, cp, lsp simultaneously.\n"
            "DestinE: call `search_destine_parameters` first, then call `retrieve_destine_data` for all param_ids in one response.\n"
            "Load the resulting Zarr files in Python_REPL (see sections 3/3b for loading patterns).\n"
        )
        step += 1

    sections.append(
        f"**Step {step} — Climatology analysis + comparison plots:**\n"
        "Load ALL climate model CSVs from Step 1. Use the EXACT column names you printed in Step 1.\n"
        "REMINDER: prepend 'climate_data/' to CSV filenames from manifest.\n"
        "REMINDER: Convert Month strings ('January'→1) if needed (see Step 1 code).\n"
        "REMINDER: Precipitation may be column 'tp' (already in mm) or 'cp'/'lsp' (in meters, multiply by 1000).\n"
        "Compute monthly means, deltas between decades.\n"
        "Create 2-3 comparison plots (temperature, precipitation, wind) saved to `results/`.\n"
        "Print a concise summary of baseline values and projected changes.\n"
    )
    step += 1

    sections.append(
        f"**Step {step} — Threshold & risk analysis + additional plots:**\n"
        "If ERA5 time series were downloaded: compute threshold exceedances (heat days, frost days,\n"
        "dry spells, wind extremes). Create 2-3 threshold/risk plots saved to `results/`.\n"
        "Print quantitative risk metrics. If no ERA5 time series, skip this step.\n"
    )
    step += 1

    sections.append(
        f"**Step {step} — Verify ALL plots (SEPARATE response, after plots exist):**\n"
        "In a NEW response (never in the same response as Python_REPL), call `reflect_on_image`\n"
        "once per generated plot — QA ALL of them, not just one.\n"
        "MINIMUM SCORE: 7/10. If any plot scores below 7:\n"
        "- Read the reviewer's fix suggestions carefully\n"
        "- Write a NEW Python_REPL script applying those exact fixes\n"
        "- Do NOT give up or skip re-plotting — the fixes are usually simple (font sizes, legend position)\n"
    )

    # ── 6. PLOTTING CONVENTIONS ───────────────────────────────────────────
    sections.append(
        "\n## 6. PLOTTING CONVENTIONS\n\n"
        "Every plot you create MUST follow these rules:\n"
        "- `plt.figure(figsize=(12, 6))` — wide-format for readability\n"
        "- `plt.savefig('results/filename.png', dpi=150, bbox_inches='tight')`\n"
        "- `plt.close()` — ALWAYS close to prevent memory leaks\n"
        "- Font sizes: title 14pt, axis labels 12pt, tick labels 10pt, legend 10pt\n"
        "- Color palette: use scientific defaults — blue=#2196F3 cold, red=#F44336 hot,\n"
        "  green=#4CAF50 precipitation; use 'tab10' for multi-series\n"
        "- ERA5 observations: always plot as BLACK solid line with circle markers ('k-o')\n"
        "- Model projections: colored dashed lines, labeled by decade\n"
        "- Include units on EVERY axis (°C, mm/month, m/s)\n"
        "- Use `plt.tight_layout()` or `bbox_inches='tight'` to prevent label clipping\n"
    )

    # ── 7. ERROR RECOVERY ─────────────────────────────────────────────────
    sections.append(
        "\n## 7. ERROR RECOVERY\n\n"
        "MOST COMMON ERRORS (fix these FIRST):\n"
        "- `ValueError: invalid literal for int()` on Month → Month column has string names like 'January'.\n"
        "  FIX: Use month_map dict to convert (see Step 1 code example).\n"
        "- `FileNotFoundError: simulation_1.csv` → Manifest paths are filenames only.\n"
        "  FIX: Prepend 'climate_data/': `os.path.join('climate_data', os.path.basename(e['csv']))`\n"
        "- `KeyError: 'cp'` or `'lsp'` → CSV column is 'tp' (total precip in mm), not cp/lsp.\n"
        "  FIX: Check df.columns.tolist() first, use whatever precipitation column exists.\n\n"
        "Other errors:\n"
        "- File not found? → Run `list_plotting_data_files` to see available files and adapt paths.\n"
        "- Zarr load fails? → Check `era5_data/` contents with `glob.glob('era5_data/*')`.\n"
        "- Plot save fails? → Ensure `results/` dir exists: `os.makedirs('results', exist_ok=True)`.\n"
        "- JSON parse error? → Print the file contents first, then fix the loading code.\n"
        "- Empty DataFrame? → Print `df.head()` and `df.columns.tolist()` to inspect structure.\n"
    )

    # ── 8. SANDBOX PATHS ──────────────────────────────────────────────────
    sections.append(
        f"\n## 8. SANDBOX PATHS AND DATA\n\n{datasets_text}\n"
    )

    # ── 9. PROACTIVE ANALYSIS ─────────────────────────────────────────────
    sections.append(
        "\n## 9. PROACTIVE ANALYSIS\n\n"
        "Even if the user's query is vague, you SHOULD proactively:\n"
        "- Create a temperature trend visualization (all decades + ERA5 baseline)\n"
        "- Create a precipitation comparison chart\n"
        "- Highlight the 3 months with the largest projected changes\n"
        "- Identify potential climate risks relevant to the query\n"
    )

    # ── 10. OUTPUT FORMAT ─────────────────────────────────────────────────
    sections.append(
        "\n## 10. OUTPUT FORMAT\n\n"
        "Your final response MUST include:\n"
    )
    if has_era5_data:
        sections.append(
            "1. **Observed Climate** — current conditions from ERA5 (2015-2025 baseline)\n"
            "2. **Model Performance** — how well projections match ERA5 observations\n"
            "3. **Projected Changes** — future vs baseline, with magnitude and timing\n"
            "4. **Critical Months** — months with largest changes or highest risk\n"
            "5. **Visualizations** — list of created plot files in `results/`\n"
            "6. **Implications** — interpretation relevant to the user's query\n"
        )
    else:
        sections.append(
            "1. **Key Climate Values** — temperature, precipitation, wind summary\n"
            "2. **Climate Change Signal** — future vs historical differences\n"
            "3. **Critical Months** — months with largest changes\n"
            "4. **Visualizations** — list of created plot files\n"
            "5. **Implications** — interpretation relevant to the user's query\n"
        )

    # ── TOOL BUDGET ───────────────────────────────────────────────────────
    sections.append(
        "\n## TOOL BUDGET (HARD LIMIT: 30 tool calls total, max 3-4 per response)\n\n"
        "Plan your session carefully — you have at most 30 tool calls:\n"
        "- Python_REPL: a few calls, each focused on ONE logical task\n"
        "- retrieve_era5_data: 0-3 calls (one per variable: t2, cp, lsp)\n"
        "- search_destine_parameters: 1-2 calls (find param_ids before downloading)\n"
        "- retrieve_destine_data: 0-3 calls (use full 2020-2039 range by default)\n"
        "- reflect_on_image: one call per plot — QA ALL generated plots, not just one\n"
        "- list_plotting_data_files / image_viewer: 0-2 calls\n"
        "- wise_agent: 0-1 calls\n\n"
        "DIVIDE AND CONQUER — Python_REPL strategy:\n"
        "- Script 1: Load ALL data, explore structure, print column names and shapes\n"
        "- Script 2: Climatology analysis + comparison plots (temp, precip, wind)\n"
        "- Script 3: Threshold/risk analysis + additional plots (if ERA5 time series available)\n"
        "- Script 4 (if needed): Fix any errors from previous scripts, create missing plots\n\n"
        "WHY: One massive all-in-one script causes cascading errors — one bug kills everything.\n"
        "Splitting into reasonable chunks lets you catch and fix errors between steps.\n\n"
        "ANTI-SPAM RULES:\n"
        "- Never call more than 3-4 tools in a single response.\n"
        "- Never call reflect_on_image in the same response as Python_REPL.\n"
        "- Never call reflect_on_image more than twice total.\n"
        "- Don't spam tiny one-liner REPL calls — each script should do meaningful work.\n"
    )

    return "\n".join(sections)


def _normalize_tool_observation(observation: Any) -> Any:
    """Normalize tool output into a plain Python object."""
    try:
        from langchain_core.messages import AIMessage
    except Exception:
        AIMessage = None

    if AIMessage is not None and isinstance(observation, AIMessage):
        return observation.content
    return observation


def data_analysis_agent(
    state: AgentState,
    config: dict,
    api_key: str,
    api_key_local: str,
    stream_handler,
    llm_dataanalysis_agent=None,
):
    """Run filtered analysis + tool-based climatology extraction."""
    stream_handler.update_progress("Data analysis: preparing sandbox...")

    # Ensure sandbox paths are available.
    thread_id = ensure_thread_id(existing_thread_id=state.thread_id)
    sandbox_paths = get_sandbox_paths(thread_id)
    ensure_sandbox_dirs(sandbox_paths)

    state.thread_id = thread_id
    state.uuid_main_dir = sandbox_paths["uuid_main_dir"]
    state.results_dir = sandbox_paths["results_dir"]
    state.climate_data_dir = sandbox_paths["climate_data_dir"]
    state.era5_data_dir = sandbox_paths["era5_data_dir"]
    state.destine_data_dir = sandbox_paths.get("destine_data_dir", "")

    # Build analysis context for filtering.
    climate_summary = _build_climate_data_summary(state.df_list)
    context_sections = [
        f"User query: {state.user}",
        f"Location: {state.input_params.get('location_str', '')}",
        f"Coordinates: {state.input_params.get('lat', '')}, {state.input_params.get('lon', '')}",
        f"Climatology summary:\n{climate_summary}",
    ]

    if state.ipcc_rag_agent_response:
        context_sections.append(f"IPCC RAG: {state.ipcc_rag_agent_response}")
    if state.general_rag_agent_response:
        context_sections.append(f"General RAG: {state.general_rag_agent_response}")
    if state.smart_agent_response:
        context_sections.append(f"Smart agent: {state.smart_agent_response.get('output', '')}")
    if state.ecocrop_search_response:
        context_sections.append(f"ECOCROP: {state.ecocrop_search_response}")
    if state.zero_agent_response:
        safe_zero_context = make_json_serializable(state.zero_agent_response)
        context_sections.append(f"Local context: {json.dumps(safe_zero_context, indent=2)}")

    analysis_context = "\n\n".join(context_sections)

    # Check if filter step is enabled (configurable)
    use_filter_step = config.get("llm_dataanalysis", {}).get("use_filter_step", True)

    if use_filter_step and llm_dataanalysis_agent is not None:
        stream_handler.update_progress("Data analysis: filtering context...")
        filter_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", _build_filter_prompt()),
                ("user", "{context}"),
            ]
        )
        result = llm_dataanalysis_agent.invoke(filter_prompt.format_messages(context=analysis_context))
        filtered_context = result.content if hasattr(result, "content") else str(result)

        # CRITICAL: Always preserve the user's original question
        location_str = state.input_params.get('location_str', 'Unknown location')
        analysis_brief = f"""USER QUESTION: {state.user}

Location: {location_str}
Coordinates: {state.input_params.get('lat', '')}, {state.input_params.get('lon', '')}

ANALYSIS REQUIREMENTS:
{filtered_context}
"""
    else:
        # Skip filter step - pass essential context directly
        stream_handler.update_progress("Data analysis: preparing context (no filter)...")
        location_str = state.input_params.get('location_str', 'Unknown location')
        analysis_brief = f"""USER QUESTION: {state.user}

Location: {location_str}
Coordinates: {state.input_params.get('lat', '')}, {state.input_params.get('lon', '')}

Available climatology:
{climate_summary}

Required analysis:
- Extract Temperature and Precipitation data
- Compare historical vs future projections
- Create visualizations if Python_REPL is available
"""

    state.data_analysis_prompt_text = analysis_brief

    brief_path = os.path.join(state.uuid_main_dir, "analysis_brief.txt")
    with open(brief_path, "w", encoding="utf-8") as f:
        f.write(analysis_brief)

    # Build simplified datasets_text for prompt
    datasets_text = _build_datasets_text(state)

    # Build datasets dict for Python REPL
    datasets = {
        "uuid_main_dir": state.uuid_main_dir,
        "results_dir": state.results_dir,
    }
    if state.climate_data_dir:
        datasets["climate_data_dir"] = state.climate_data_dir
    if state.era5_data_dir:
        datasets["era5_data_dir"] = state.era5_data_dir

    # Get coordinates for prompt
    lat = state.input_params.get('lat')
    lon = state.input_params.get('lon')
    try:
        lat = float(lat) if lat is not None else None
        lon = float(lon) if lon is not None else None
    except (ValueError, TypeError):
        lat, lon = None, None

    # Tool setup for data_analysis_agent
    # NOTE: This agent only runs when python_REPL is enabled (use_powerful_data_analysis=True)
    # ERA5 climatology and predefined plots are already generated by prepare_predefined_data
    tools = []

    has_era5_data = config.get("era5_climatology", {}).get("enabled", True)
    # This agent only runs when use_powerful_data_analysis=True, so Python REPL is always available
    has_python_repl = True

    # NOTE: ERA5 climatology and predefined plots are ALREADY generated by prepare_predefined_data
    # The agent should use the pre-extracted data, not re-extract it
    logger.info(f"data_analysis_agent starting - predefined plots: {state.predefined_plots}")
    logger.info(f"ERA5 climatology available: {bool(state.era5_climatology_response)}")

    # 3. ERA5 time series retrieval (if enabled - for detailed year-by-year analysis)
    if config.get("use_era5_data", False):
        arraylake_api_key = config.get("arraylake_api_key", "")
        if arraylake_api_key:
            tools.append(create_era5_retrieval_tool(arraylake_api_key))
        else:
            logger.warning("ERA5 data enabled but no arraylake_api_key in config. ERA5 retrieval tool not added.")

    # 3b. DestinE parameter search + data retrieval (if enabled)
    if config.get("use_destine_data", False):
        tools.append(create_destine_search_tool(config))
        tools.append(create_destine_retrieval_tool())

    # 4. Python REPL for analysis/visualization (if enabled)
    if has_python_repl:
        repl_tool = CustomPythonREPLTool(
            datasets=datasets,
            results_dir=state.results_dir,
            session_key=thread_id,
        )
        tools.append(repl_tool)

    # 5. Helper tools
    tools.append(list_plotting_data_files_tool)

    # 6. Image viewer - ALWAYS available (plots are saved to results/ folder)
    #    Works with predefined plots even when Python REPL is disabled
    vision_model = config.get("llm_combine", {}).get("model_name", "gpt-4o")
    image_viewer_tool = create_image_viewer_tool(
        openai_api_key=api_key,
        model_name=vision_model,
        sandbox_path=state.results_dir  # Point to results folder where plots are saved
    )
    tools.append(image_viewer_tool)

    # 7. Image reflection and wise_agent - ONLY when Python REPL is enabled
    #    (these tools are for evaluating/creating visualizations)
    if has_python_repl:
        tools.append(reflect_tool)
        tools.append(wise_agent_tool)

    stream_handler.update_progress("Data analysis: running tools...")
    tool_prompt = _create_tool_prompt(
        datasets_text, config, lat=lat, lon=lon,
        has_climate_data=bool(state.df_list),
        has_hazard_data=state.hazard_data is not None,
        has_population_data=bool(state.population_config),
        has_era5_data=bool(state.era5_climatology_response)
    )

    if llm_dataanalysis_agent is None:
        from langchain_openai import ChatOpenAI

        llm_dataanalysis_agent = ChatOpenAI(
            openai_api_key=api_key,
            model_name=config.get("llm_combine", {}).get("model_name", "gpt-4.1-nano"),
        )

    agent_executor = create_standard_agent_executor(
        llm_dataanalysis_agent,
        tools,
        tool_prompt,
        max_iterations=20,
    )

    agent_input = {
        "input": analysis_brief or state.user,
        "messages": state.messages,
    }

    result = agent_executor(agent_input)

    data_components_outputs = []
    plot_images: List[str] = []
    era5_climatology_output = None
    agent_references: List[str] = []  # Collect references from agent tools

    for action, observation in result.get("intermediate_steps", []):
        if action.tool == "get_era5_climatology":
            obs = _normalize_tool_observation(observation)
            if isinstance(obs, dict) and "error" not in obs:
                era5_climatology_output = obs
                state.era5_climatology_response = obs
                # Collect reference from ERA5 climatology
                if "reference" in obs:
                    agent_references.append(obs["reference"])
        if action.tool == "get_data_components":
            obs = _normalize_tool_observation(observation)
            data_components_outputs.append(obs)
            # Collect references from get_data_components
            if isinstance(obs, dict):
                if "reference" in obs:
                    agent_references.append(obs["reference"])
                if "references" in obs:
                    agent_references.extend(obs["references"])
        if action.tool in ("Python_REPL", "python_repl"):
            obs = _normalize_tool_observation(observation)
            if isinstance(obs, dict):
                plot_images.extend(obs.get("plot_images", []))
        if action.tool == "retrieve_era5_data":
            # Handle ERA5 retrieval tool output
            obs = _normalize_tool_observation(observation)
            if isinstance(obs, dict):
                era5_output = str(obs)
                # Collect reference from ERA5 retrieval
                if "reference" in obs:
                    agent_references.append(obs["reference"])
            elif hasattr(obs, 'content'):
                era5_output = obs.content
            else:
                era5_output = str(obs)
            # Store in state
            state.era5_tool_response = era5_output
            state.input_params.setdefault("era5_results", []).append(obs)
        if action.tool == "retrieve_destine_data":
            obs = _normalize_tool_observation(observation)
            if isinstance(obs, dict):
                if "reference" in obs:
                    agent_references.append(obs["reference"])
            state.destine_tool_response = str(obs)
            state.input_params.setdefault("destine_results", []).append(obs)

    # Add agent-collected references to state.references (deduplicate)
    for ref in agent_references:
        if ref and ref not in state.references:
            state.references.append(ref)

    analysis_text = result.get("output", "")

    # Append ERA5 climatology summary if available
    if era5_climatology_output:
        analysis_text += "\n\n### ERA5 Observational Baseline (2015-2025)\n"
        analysis_text += f"Location: {era5_climatology_output.get('extracted_location', {})}\n"
        if "variables" in era5_climatology_output:
            for var_name, var_data in era5_climatology_output["variables"].items():
                analysis_text += f"\n**{var_data.get('full_name', var_name)}** ({var_data.get('units', '')}):\n"
                monthly = var_data.get("monthly_values", {})
                # Show a few key months
                for month in ["January", "April", "July", "October"]:
                    if month in monthly:
                        analysis_text += f"  {month}: {monthly[month]}\n"

    if data_components_outputs:
        analysis_text += "\n\n### Climate Model Extracts:\n"
        for item in data_components_outputs:
            analysis_text += json.dumps(item, indent=2) + "\n"

    # Combine all plot images (pre-generated + agent-generated)
    all_plot_images = state.predefined_plots + plot_images

    state.data_analysis_response = analysis_text
    state.data_analysis_images = all_plot_images

    stream_handler.update_progress("Data analysis complete.")

    return {
        "data_analysis_response": analysis_text,
        "data_analysis_images": all_plot_images,
        "predefined_plots": state.predefined_plots,
        "data_analysis_prompt_text": analysis_brief,
        "era5_climatology_response": state.era5_climatology_response,
        "era5_tool_response": getattr(state, 'era5_tool_response', None),
        "destine_tool_response": getattr(state, 'destine_tool_response', None),
        "references": state.references,  # Propagate collected references
    }
