"""
Data analysis agent for Climsight.

Restructured as a LangGraph sub-graph with three stages:
1. filter_node  — LLM filters context into an analysis brief
2. planner_node — LLM decides which ERA5/DestinE variables to download
3. download_node — ThreadPoolExecutor downloads ALL variables in parallel
4. analysis_node — AgentExecutor with Python REPL for analysis/visualization
"""

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from climsight_classes import AgentState

try:
    from utils import make_json_serializable
except ImportError:
    from .utils import make_json_serializable
from sandbox_utils import ensure_thread_id, ensure_sandbox_dirs, get_sandbox_paths
from agent_helpers import create_standard_agent_executor
from tools.era5_retrieval_tool import create_era5_retrieval_tool, retrieve_era5_data
from tools.destine_retrieval_tool import (
    create_destine_search_tool,
    create_destine_retrieval_tool,
    retrieve_destine_data,
    _search_destine_parameters,
)
from tools.python_repl import CustomPythonREPLTool
from tools.image_viewer import create_image_viewer_tool
from tools.reflection_tools import reflect_tool
from tools.visualization_tools import (
    list_plotting_data_files_tool,
    wise_agent_tool,
)

from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable

logger = logging.getLogger(__name__)

# ── Analysis mode presets ────────────────────────────────────────────────
ANALYSIS_MODES = {
    "fast": {
        "use_powerful_data_analysis": False,
        "use_era5_data": False,
        "use_destine_data": False,
        "use_smart_agent": False,
        "hard_tool_limit": 10,
        "ideal_tool_calls": "3-5",
        "max_per_response": 3,
        "max_reflect": 0,
        "max_iterations": 8,
    },
    "smart": {
        "use_powerful_data_analysis": True,
        "use_era5_data": True,
        "use_destine_data": False,
        "use_smart_agent": True,
        "hard_tool_limit": 50,
        "ideal_tool_calls": "15-20",
        "max_per_response": 5,
        "max_reflect": 8,
        "max_iterations": 30,
    },
    "deep": {
        "use_powerful_data_analysis": True,
        "use_era5_data": True,
        "use_destine_data": True,
        "use_smart_agent": True,
        "hard_tool_limit": 150,
        "ideal_tool_calls": "40-50",
        "max_per_response": 10,
        "max_reflect": 15,
        "max_iterations": 80,
    },
}


def resolve_analysis_config(config: dict) -> dict:
    """Merge analysis mode defaults with explicit config overrides.

    The mode (fast/smart/deep) sets default values for toggles and budgets.
    If the user explicitly sets a toggle in config or UI, that override wins.
    """
    mode = config.get("analysis_mode", "smart")
    defaults = ANALYSIS_MODES.get(mode, ANALYSIS_MODES["smart"]).copy()
    # Explicit toggles in config override mode defaults
    for key in ["use_powerful_data_analysis", "use_era5_data",
                 "use_destine_data", "use_smart_agent"]:
        if key in config:
            defaults[key] = config[key]
    return defaults


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


# ==========================================================================
# Planner prompt — tells the LLM what downloads are available and asks it
# to output a structured JSON plan
# ==========================================================================

def _create_planner_prompt(
    has_era5_download: bool,
    has_destine: bool,
    lat: float = None,
    lon: float = None,
) -> str:
    """Build the system prompt for the download planner LLM."""
    sections = [
        "You are a data download planner for ClimSight's climate analysis agent.\n"
        "Given an analysis brief, decide which ERA5 and/or DestinE variables to download.\n"
        "You MUST output ONLY valid JSON — no commentary, no markdown fences.\n\n"
    ]

    if has_era5_download:
        sections.append(
            "## ERA5 Reanalysis Time Series (1979-2024)\n"
            "Available variables:\n"
            "- `t2` — 2m air temperature (K → °C)\n"
            "- `cp` — convective precipitation (m)\n"
            "- `lsp` — large-scale precipitation (m)\n"
            "  NOTE: `tp` does NOT exist. For total precipitation, download BOTH `cp` and `lsp`.\n"
            "- `u10` — 10m U-wind component (m/s)\n"
            "- `v10` — 10m V-wind component (m/s)\n"
            "- `mslp` — mean sea level pressure (Pa)\n"
            "- `sst` — sea surface temperature (only for coastal/ocean points)\n"
            "- `sp` — surface pressure (Pa)\n"
            "- `tcc` — total cloud cover (0-1)\n"
            "- `sd` — snow depth (m)\n"
            "- `skt` — skin temperature (K)\n"
            "- `d2` — 2m dewpoint temperature (K)\n\n"
            "Default date range: 1975-01-01 to 2024-12-31 (50 years of observations).\n"
            "ALWAYS use the full default range unless the user explicitly asks for a shorter period.\n"
            "ERA5 downloads are fast, so prefer downloading the full range.\n"
            "Only download variables relevant to the analysis — don't download everything.\n"
            "Common patterns:\n"
            "- Temperature analysis → t2\n"
            "- Precipitation analysis → cp + lsp (always both)\n"
            "- Wind analysis → u10 + v10\n"
            "- General climate overview → t2, cp, lsp\n\n"
        )

    if has_destine:
        sections.append(
            "## DestinE Climate Projections (SSP3-7.0, 2020-2039)\n"
            "High-resolution IFS-NEMO model. 82 parameters available.\n"
            "You specify search queries (natural language), and the system will find the right param_ids.\n"
            "Default date range: 20200101-20391231 (full 20 years).\n"
            "Common searches:\n"
            "- 'temperature at 2 meters'\n"
            "- 'total precipitation'\n"
            "- 'wind speed at 10 meters' (returns u10 and v10 separately)\n"
            "- 'sea surface temperature'\n\n"
        )

    coord_text = ""
    if lat is not None and lon is not None:
        coord_text = f"Coordinates: lat={lat}, lon={lon}\n"

    sections.append(
        "## Output Format\n"
        f"{coord_text}"
        "Return a JSON object with this structure:\n"
        "{{\n"
        '  "era5_downloads": [\n'
        '    {{"variable_id": "t2", "start_date": "1975-01-01", "end_date": "2024-12-31"}}\n'
        "  ],\n"
        '  "destine_searches": ["temperature at 2 meters", "total precipitation"],\n'
        '  "destine_date_range": {{"start": "20200101", "end": "20391231"}},\n'
        '  "reasoning": "Brief explanation of why these variables are needed"\n'
        "}}\n\n"
        "Rules:\n"
        "- If ERA5 download is not available, set era5_downloads to []\n"
        "- If DestinE is not available, set destine_searches to []\n"
        "- For precipitation, ALWAYS include BOTH cp and lsp (never just one)\n"
        "- For wind, include BOTH u10 and v10\n"
        "- If the analysis only needs climatology comparison (no trends), set all to [] — "
        "existing data is sufficient\n"
    )

    return "".join(sections)


# ==========================================================================
# Analysis prompt — same as the old _create_tool_prompt but WITHOUT download
# tool sections. Download results are injected as pre-available data.
# ==========================================================================

def _create_analysis_prompt(
    datasets_text: str,
    config: dict,
    lat: float = None,
    lon: float = None,
    has_climate_data: bool = False,
    has_hazard_data: bool = False,
    has_population_data: bool = False,
    has_era5_data: bool = False,
    mode_config: dict = None,
    downloaded_data_text: str = "",
) -> str:
    """System prompt for the analysis agent — no download tools, all data pre-fetched."""
    if mode_config is None:
        mode_config = resolve_analysis_config(config)

    has_python_repl = mode_config.get("use_powerful_data_analysis", True)

    hard_limit = mode_config.get("hard_tool_limit", 30)
    ideal_calls = mode_config.get("ideal_tool_calls", "6-7")
    max_per_resp = mode_config.get("max_per_response", 4)
    max_reflect = mode_config.get("max_reflect", 2)

    # Reduce budget since downloads are already done
    # Estimate: downloads would have used ~3-6 tool calls
    adjusted_limit = max(hard_limit - 6, 10)

    sections = []

    # ── ROLE ──────────────────────────────────────────────────────────────
    role_lines = [
        "You are ClimSight's data analysis agent.\n"
        "Your job: provide ADDITIONAL quantitative climate analysis beyond the standard plots.\n"
    ]
    if has_python_repl:
        role_lines.append(
            "You have a persistent Python REPL, pre-extracted data files, and pre-downloaded "
            "ERA5/DestinE time series (if any).\n"
            "ALL data downloads have already been completed — you do NOT have download tools.\n"
            "Focus entirely on analysis and visualization.\n\n"
            "CRITICAL EFFICIENCY RULES:\n"
            f"- HARD LIMIT: {adjusted_limit} tool calls total for the entire session.\n"
            f"- MAX {max_per_resp} tool calls per response. Never fire {max_per_resp + 1}+ tools in a single response.\n"
            "- Write focused Python scripts — each one should accomplish a meaningful chunk of work.\n"
            "- SEQUENTIAL ordering: first Python_REPL to generate plots, THEN reflect_on_image in a LATER response.\n"
            "  Never call reflect_on_image in the same response as Python_REPL.\n"
            f"- Ideal session: {ideal_calls} tool calls total.\n"
        )
    else:
        role_lines.append(
            "You have pre-extracted data files and predefined plots.\n\n"
            "CRITICAL EFFICIENCY RULES:\n"
            f"- HARD LIMIT: {adjusted_limit} tool calls total for the entire session.\n"
            f"- MAX {max_per_resp} tool calls per response.\n"
            f"- Ideal session: {ideal_calls} tool calls total.\n"
        )
    sections.append("".join(role_lines))

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

    # ── 1b. DOWNLOADED TIME SERIES (from planner/download stage) ──────────
    if downloaded_data_text:
        sections.append(
            "### Downloaded Time Series (pre-fetched by download planner)\n"
            + downloaded_data_text + "\n"
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
            "These plots already cover basic monthly climatology comparisons (model decades vs ERA5).\n"
            "DO NOT recreate similar plots — they are already done.\n"
            "Instead, use the same underlying data for DEEPER analysis: trends, anomalies, extremes,\n"
            "distributions, seasonal decomposition — things the predefined plots do NOT cover.\n"
            "Only use `image_viewer` on plots YOU create, not on these predefined ones.\n"
        )
    else:
        sections.append("## 2. PRE-GENERATED PLOTS\n\nNone yet — you may create any visualizations needed.\n")

    # ── 3. AVAILABLE TOOLS ────────────────────────────────────────────────
    sections.append("## 3. AVAILABLE TOOLS\n")
    tools_list = []
    if has_python_repl:
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
    if has_python_repl and max_reflect > 0:
        tools_list.append(
            "- **reflect_on_image** — get quality feedback on a plot you created.\n"
            "  Call once per plot — reflect on ALL generated plots, not just one.\n"
            "  MUST be called in a SEPARATE response AFTER the Python_REPL that created the plots.\n"
            "  Always verify the file exists (via os.path.exists in REPL) BEFORE calling this tool.\n"
            "  MINIMUM ACCEPTABLE SCORE: 7/10. If score < 7, you MUST re-plot with fixes applied.\n"
            "  Read the fix suggestions from the reviewer and apply them in your next REPL call."
        )
    if has_python_repl:
        tools_list.append("- **wise_agent** — ask for visualization strategy advice before coding")
    sections.append("\n".join(tools_list) + "\n")

    # ── 4. WORKFLOW ───────────────────────────────────────────────────────
    step = 1
    sections.append("## 4. REQUIRED WORKFLOW\n")

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

    if downloaded_data_text:
        sections.append(
            f"**Step {step} — Load downloaded time series:**\n"
            "ERA5/DestinE data was pre-downloaded by the planner. Load the Zarr files:\n\n"
            "CRITICAL — analyze the FULL time range of each dataset:\n"
            "- ERA5 covers ~1975–2024 (up to 50 years of observations). This is your primary record.\n"
            "  Compute long-term trends, decadal averages, anomalies from the 30-year climatological mean.\n"
            "- DestinE covers 2020–2039 (20 years of high-resolution projections under SSP3-7.0).\n"
            "  Analyze the full projected range — trends, seasonal shifts, comparison to ERA5 baseline.\n"
            "- Do NOT limit analysis to the overlap period (2020–2024). Each dataset tells its own story:\n"
            "  ERA5 = what happened over decades, DestinE = what is projected for the next two decades.\n"
            "- Use the overlap (2020–2024) ONLY for validating DestinE against ERA5 observations.\n\n"
        )
        sections.append(
            "```python\n"
            "import xarray as xr, glob\n"
            "# ERA5 time series (full historical record)\n"
            "era5_files = glob.glob('era5_data/*.zarr')\n"
            "for f in era5_files:\n"
            "    ds = xr.open_dataset(f, engine='zarr', chunks={{}})\n"
            "    print(f, list(ds.data_vars), 'time range:', str(ds.time.values[0])[:10], 'to', str(ds.time.values[-1])[:10])\n"
            "# DestinE projections (if available)\n"
            "destine_files = glob.glob('destine_data/*.zarr')\n"
            "for f in destine_files:\n"
            "    ds = xr.open_dataset(f, engine='zarr', chunks={{}})\n"
            "    print(f, list(ds.data_vars), 'time range:', str(ds.time.values[0])[:10], 'to', str(ds.time.values[-1])[:10])\n"
            "```\n\n"
        )
        step += 1

    sections.append(
        f"**Step {step} — Deeper climatology analysis (DO NOT re-plot predefined figures):**\n"
        "IMPORTANT: Monthly climatology comparison plots (temperature, precipitation, wind vs ERA5)\n"
        "are ALREADY generated as predefined plots in `results/climate_*.png`.\n"
        "Do NOT recreate them. Instead, use the same underlying data for DEEPER analysis:\n\n"
        "Load ALL climate model CSVs from Step 1. Use the EXACT column names you printed in Step 1.\n"
        "REMINDER: prepend 'climate_data/' to CSV filenames from manifest.\n"
        "REMINDER: Convert Month strings ('January'→1) if needed (see Step 1 code).\n"
        "REMINDER: Precipitation may be column 'tp' (already in mm) or 'cp'/'lsp' (in meters, multiply by 1000).\n\n"
        "Focus on analysis that goes BEYOND the predefined plots:\n"
        "- Decadal change analysis: compute and plot differences between earliest and latest decades\n"
        "- Seasonal cycle shifts: how does the seasonal pattern change across decades?\n"
        "- If ERA5 time series downloaded: annual mean time series over the full record (1975-2024)\n"
        "  with linear trend line, annotate slope (°C/decade or mm/decade)\n"
        "- Multi-variable correlation: do temperature and precipitation changes correlate?\n"
        "- Anomaly plots: departure from long-term mean per year or month\n"
        "Print a concise summary of baseline values, projected changes, and their magnitudes.\n"
    )
    step += 1

    sections.append(
        f"**Step {step} — Statistical analysis, trends, and extremes:**\n"
        "Perform deep quantitative analysis on downloaded time series (4-6 plots minimum):\n\n"
        "Trend analysis:\n"
        "- Linear regression on annual means for each variable (scipy.stats.linregress)\n"
        "- Report slope (per decade), R², and p-value for statistical significance\n"
        "- Plot time series with trend line and confidence interval\n\n"
        "Extreme value analysis:\n"
        "- Threshold exceedances: heat days (>30°C), frost days (<0°C), heavy precip events, dry spells\n"
        "- Compute annual counts of threshold exceedances and plot their trend over time\n"
        "- Percentile analysis: 90th, 95th, 99th percentiles per decade — are extremes intensifying?\n\n"
        "Distribution analysis:\n"
        "- Compare distributions across decades (histograms or KDE plots)\n"
        "- Seasonal decomposition: how do trends differ by season (DJF, MAM, JJA, SON)?\n\n"
        "If DestinE projections available:\n"
        "- Compare projected extremes (2020-2039) against ERA5 historical baseline (1975-2024)\n"
        "- Quantify projected changes in threshold exceedances\n\n"
        "Print ALL computed metrics — trends, p-values, exceedance counts, percentile shifts.\n"
        "If no ERA5 time series, skip this step.\n"
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

    # ── 5. PLOTTING CONVENTIONS ───────────────────────────────────────────
    sections.append(
        "\n## 5. PLOTTING CONVENTIONS\n\n"
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

    # ── 6. ERROR RECOVERY ─────────────────────────────────────────────────
    sections.append(
        "\n## 6. ERROR RECOVERY\n\n"
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

    # ── 7. SANDBOX PATHS ──────────────────────────────────────────────────
    sections.append(
        f"\n## 7. SANDBOX PATHS AND DATA\n\n{datasets_text}\n"
    )

    # ── 8. PROACTIVE ANALYSIS ─────────────────────────────────────────────
    sections.append(
        "\n## 8. PROACTIVE ANALYSIS\n\n"
        "You have a generous tool budget — USE IT. More analysis is always better than less.\n"
        "Think like a climate scientist writing a technical report. The user needs quantitative evidence.\n\n"
        "Even if the user's query is vague, you MUST proactively:\n"
        "- For EVERY downloaded variable: produce (a) full time series plot, (b) trend analysis with slope,\n"
        "  (c) anomaly plot relative to long-term mean, (d) seasonal breakdown\n"
        "- Compute and report: trend significance (p-value), standard deviations, correlation between variables\n"
        "- Highlight the 3 months with the largest projected changes\n"
        "- Identify potential climate risks relevant to the query\n"
        "- If both ERA5 and DestinE are available: create a combined timeline plot showing\n"
        "  historical observations transitioning into projections\n\n"
        "Even if a finding won't make the final report, the analysis informs a better answer.\n"
        "Do NOT stop after 2-3 plots — exploit your full tool budget for thorough analysis.\n"
    )

    # ── 9. OUTPUT FORMAT ─────────────────────────────────────────────────
    sections.append(
        "\n## 9. OUTPUT FORMAT (CRITICAL)\n\n"
        "Your final text response is the ONLY information passed to the downstream combine agent.\n"
        "Raw Python REPL output (print statements, statistics) is NOT forwarded — only YOUR written response.\n"
        "Therefore you MUST include ALL quantitative results in your final response:\n"
        "- Every computed number: trend slopes (°C/decade), p-values, R² values, percentiles\n"
        "- Threshold exceedance counts, mean values, standard deviations\n"
        "- For EVERY plot: describe what it shows and the key finding (do NOT include file paths in your text)\n\n"
        "Your final response MUST include:\n"
    )
    if has_era5_data:
        sections.append(
            "1. **Observed Climate** — current conditions from ERA5 with specific values\n"
            "   (e.g., 'Mean annual temperature: 10.3°C, warmest month: July at 21.5°C')\n"
            "2. **Long-term Trends** — trend slope, significance, time period analyzed\n"
            "   (e.g., 'Temperature trend: +0.35°C/decade (p<0.01) over 1975-2024')\n"
            "3. **Model Performance** — how well projections match ERA5 observations, with bias values\n"
            "4. **Projected Changes** — future vs baseline with magnitude and timing\n"
            "   (e.g., 'Temperature +2.1°C by 2040s relative to 1995-2014 baseline')\n"
            "5. **Extremes** — threshold exceedances, percentile shifts, with counts\n"
            "   (e.g., 'Heat days (>30°C): increased from 5/year (1975-1984) to 18/year (2015-2024)')\n"
            "6. **Critical Months** — months with largest changes or highest risk, with values\n"
            "7. **Visualizations** — for each plot: one-sentence description of what it shows and key finding\n"
            "8. **Implications** — interpretation relevant to the user's query\n"
        )
    else:
        sections.append(
            "1. **Key Climate Values** — temperature, precipitation, wind with specific numbers\n"
            "2. **Climate Change Signal** — future vs historical differences with magnitudes\n"
            "3. **Critical Months** — months with largest changes, with values\n"
            "4. **Visualizations** — for each plot: one-sentence description of what it shows and key finding\n"
            "5. **Implications** — interpretation relevant to the user's query\n"
        )

    # ── TOOL BUDGET ───────────────────────────────────────────────────────
    budget_lines = [
        f"\n## TOOL BUDGET (HARD LIMIT: {adjusted_limit} tool calls total, max {max_per_resp} per response)\n\n"
        f"Plan your session carefully — you have at most {adjusted_limit} tool calls:\n"
    ]
    if has_python_repl:
        budget_lines.append("- Python_REPL: a few calls, each focused on ONE logical task\n")
    if max_reflect > 0:
        budget_lines.append(f"- reflect_on_image: one call per plot, max {max_reflect} total — QA ALL generated plots\n")
    budget_lines.append("- list_plotting_data_files / image_viewer: 0-2 calls\n")
    budget_lines.append("- wise_agent: 0-1 calls\n\n")

    if has_python_repl:
        budget_lines.append(
            "DIVIDE AND CONQUER — Python_REPL strategy:\n"
            "- Script 1: Load ALL data, explore structure, print column names, shapes, and time ranges\n"
            "- Script 2: ERA5 long-term time series analysis + trend plots (full historical record)\n"
            "- Script 3: Climatology comparison — model decades vs ERA5 baseline (monthly plots)\n"
            "- Script 4: Extreme/threshold analysis + distribution plots (if time series available)\n"
            "- Script 5: DestinE projection analysis — trends, comparison to ERA5 (if DestinE available)\n"
            "- Script 6: Statistical summary — print all computed metrics for the final report\n"
            "- Additional scripts: fix errors, re-plot after reflection feedback\n\n"
            "WHY: One massive all-in-one script causes cascading errors — one bug kills everything.\n"
            "Splitting into reasonable chunks lets you catch and fix errors between steps.\n"
            "You have a large budget — use more scripts for deeper analysis, not fewer.\n\n"
        )

    budget_lines.append(
        "ANTI-SPAM RULES:\n"
        f"- Never call more than {max_per_resp} tools in a single response.\n"
    )
    if has_python_repl and max_reflect > 0:
        budget_lines.append("- Never call reflect_on_image in the same response as Python_REPL.\n")
        budget_lines.append(f"- Never call reflect_on_image more than {max_reflect} times total.\n")
    if has_python_repl:
        budget_lines.append("- Don't spam tiny one-liner REPL calls — each script should do meaningful work.\n")

    sections.append("".join(budget_lines))

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


# ==========================================================================
# Helper: process download results
# ==========================================================================

def _process_download_results(download_results: list) -> dict:
    """Process download results into references, downloadable_datasets, tool responses, and prompt text."""
    references = []
    downloadable_datasets = []
    era5_tool_response = ""
    destine_tool_response = ""
    era5_results = []
    destine_results = []
    downloaded_data_lines = []

    for entry in download_results:
        source = entry.get("source", "")
        result = entry.get("result", {})
        if not isinstance(result, dict):
            continue

        if result.get("reference"):
            if result["reference"] not in references:
                references.append(result["reference"])

        zarr_path = result.get("output_path_zarr", "")
        variable = result.get("variable", entry.get("params", {}).get("variable_id", "unknown"))

        if source == "era5":
            era5_tool_response = str(result)
            era5_results.append(result)
            if zarr_path:
                downloadable_datasets.append({
                    "label": f"ERA5 Time Series: {variable}",
                    "path": zarr_path,
                    "source": "ERA5",
                })
                downloaded_data_lines.append(
                    f"- ERA5 `{variable}`: `{zarr_path}` — load with "
                    f"`xr.open_dataset('{zarr_path}', engine='zarr', chunks={{{{}}}})`"
                )

        elif source == "destine":
            destine_tool_response = str(result)
            destine_results.append(result)
            param_id = result.get("param_id", variable)
            if zarr_path:
                downloadable_datasets.append({
                    "label": f"DestinE Time Series: {param_id}",
                    "path": zarr_path,
                    "source": "DestinE",
                })
                downloaded_data_lines.append(
                    f"- DestinE param `{param_id}`: `{zarr_path}` — load with "
                    f"`xr.open_dataset('{zarr_path}', engine='zarr', chunks={{{{}}}})`"
                )

    downloaded_data_text = "\n".join(downloaded_data_lines) if downloaded_data_lines else ""

    return {
        "references": references,
        "downloadable_datasets": downloadable_datasets,
        "era5_tool_response": era5_tool_response,
        "destine_tool_response": destine_tool_response,
        "era5_results": era5_results,
        "destine_results": destine_results,
        "downloaded_data_text": downloaded_data_text,
    }


# ==========================================================================
# Main entry point — builds and runs the sub-graph
# ==========================================================================

def data_analysis_agent(
    state: AgentState,
    config: dict,
    api_key: str,
    api_key_local: str,
    stream_handler,
    llm_dataanalysis_agent=None,
):
    """Run the data analysis sub-graph: filter → plan → download → analyze.

    Node functions are defined as closures to capture infrastructure objects
    (LLM, stream_handler, config, etc.) without passing them through LangGraph state.
    """
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

    # Get coordinates
    lat = state.input_params.get('lat')
    lon = state.input_params.get('lon')
    try:
        lat = float(lat) if lat is not None else None
        lon = float(lon) if lon is not None else None
    except (ValueError, TypeError):
        lat, lon = None, None

    # Resolve effective config
    effective = resolve_analysis_config(config)
    analysis_mode = config.get("analysis_mode", "smart")
    logger.info(f"data_analysis_agent: mode={analysis_mode}, effective={effective}")

    # Check if filter step is enabled
    use_filter_step = config.get("llm_dataanalysis", {}).get("use_filter_step", True)

    # Build datasets text
    datasets_text = _build_datasets_text(state)

    # Initialize LLM
    if llm_dataanalysis_agent is None:
        from langchain_openai import ChatOpenAI
        llm_dataanalysis_agent = ChatOpenAI(
            openai_api_key=api_key,
            model_name=config.get("llm_combine", {}).get("model_name", "gpt-4.1-nano"),
        )

    logger.info(f"data_analysis_agent starting - predefined plots: {state.predefined_plots}")
    logger.info(f"ERA5 climatology available: {bool(state.era5_climatology_response)}")

    # ==================================================================
    # Node functions as closures — capture llm, stream_handler, config,
    # effective, api_key so they don't need to be in graph state.
    # ==================================================================

    @traceable(name="filter_node", hide_inputs=True)
    def filter_node(gstate: dict) -> dict:
        """Node 1: Filter context into an analysis brief via LLM."""
        if use_filter_step and llm_dataanalysis_agent is not None:
            stream_handler.update_progress("Data analysis: filtering context...")
            filter_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", _build_filter_prompt()),
                    ("user", "{context}"),
                ]
            )
            result = llm_dataanalysis_agent.invoke(
                filter_prompt.format_messages(context=gstate["analysis_context"])
            )
            filtered_context = result.content if hasattr(result, "content") else str(result)

            analysis_brief = (
                f"USER QUESTION: {gstate['user_query']}\n\n"
                f"Location: {gstate['location_str']}\n"
                f"Coordinates: {gstate['lat']}, {gstate['lon']}\n\n"
                f"ANALYSIS REQUIREMENTS:\n{filtered_context}\n"
            )
        else:
            stream_handler.update_progress("Data analysis: preparing context (no filter)...")
            analysis_brief = (
                f"USER QUESTION: {gstate['user_query']}\n\n"
                f"Location: {gstate['location_str']}\n"
                f"Coordinates: {gstate['lat']}, {gstate['lon']}\n\n"
                f"Available climatology:\n{gstate.get('climate_summary', '')}\n\n"
                "Required analysis:\n"
                "- Extract Temperature and Precipitation data\n"
                "- Compare historical vs future projections\n"
                "- Create visualizations if Python_REPL is available\n"
            )

        return {"analysis_brief": analysis_brief}

    @traceable(name="planner_node", hide_inputs=True)
    def planner_node(gstate: dict) -> dict:
        """Node 2: LLM decides which ERA5/DestinE variables to download."""
        has_era5_download = effective.get("use_era5_data", False)
        has_destine = effective.get("use_destine_data", False)

        # If no download sources are available, skip planning
        if not has_era5_download and not has_destine:
            logger.info("planner_node: no download sources available, skipping")
            return {"download_plan": {"era5": [], "destine": []}}

        g_lat = gstate["lat"]
        g_lon = gstate["lon"]
        analysis_brief = gstate["analysis_brief"]

        stream_handler.update_progress("Data analysis: planning downloads...")

        planner_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", _create_planner_prompt(has_era5_download, has_destine, g_lat, g_lon)),
                ("user", "{brief}"),
            ]
        )

        result = llm_dataanalysis_agent.invoke(
            planner_prompt.format_messages(brief=analysis_brief)
        )
        raw_text = result.content if hasattr(result, "content") else str(result)

        # Parse the JSON output from the planner
        try:
            cleaned = raw_text.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                cleaned = "\n".join(lines[1:-1])
            plan = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning(f"planner_node: failed to parse LLM output as JSON: {raw_text[:200]}")
            plan = {"era5_downloads": [], "destine_searches": [],
                    "reasoning": "Parse error — skipping downloads"}

        logger.info(f"planner_node plan: {plan}")

        # Build the download plan
        download_plan = {"era5": [], "destine": []}

        # ERA5 downloads
        if has_era5_download:
            for item in plan.get("era5_downloads", []):
                download_plan["era5"].append({
                    "variable_id": item["variable_id"],
                    "start_date": item.get("start_date", "1975-01-01"),
                    "end_date": item.get("end_date", "2024-12-31"),
                    "min_latitude": g_lat if g_lat is not None else -90.0,
                    "max_latitude": g_lat if g_lat is not None else 90.0,
                    "min_longitude": g_lon if g_lon is not None else 0.0,
                    "max_longitude": g_lon if g_lon is not None else 359.75,
                    "work_dir": ".",
                })

        # DestinE: resolve search queries to param_ids, then build download tasks
        if has_destine:
            destine_settings = config.get("destine_settings", {})
            chroma_db_path = destine_settings.get("chroma_db_path", "data/destine/chroma_db")
            collection_name = destine_settings.get(
                "collection_name", "climate_parameters_with_usage_notes"
            )
            dest_api_key = (api_key
                           or config.get("openai_api_key", "")
                           or os.environ.get("OPENAI_API_KEY", ""))

            destine_date_range = plan.get(
                "destine_date_range", {"start": "20200101", "end": "20391231"}
            )

            for query in plan.get("destine_searches", []):
                stream_handler.update_progress(
                    f"Data analysis: searching DestinE for '{query}'..."
                )
                search_result = _search_destine_parameters(
                    query=query,
                    k=3,
                    chroma_db_path=chroma_db_path,
                    collection_name=collection_name,
                    openai_api_key=dest_api_key,
                )
                if search_result.get("success") and search_result.get("candidates"):
                    top = search_result["candidates"][0]
                    download_plan["destine"].append({
                        "param_id": top["param_id"],
                        "levtype": top["levtype"],
                        "start_date": destine_date_range.get("start", "20200101"),
                        "end_date": destine_date_range.get("end", "20391231"),
                        "latitude": g_lat,
                        "longitude": g_lon,
                        "work_dir": ".",
                    })
                    logger.info(
                        f"DestinE search '{query}' → param_id={top['param_id']} "
                        f"({top.get('name', '')})"
                    )
                else:
                    logger.warning(f"DestinE search failed for '{query}': {search_result}")

        total_downloads = len(download_plan["era5"]) + len(download_plan["destine"])
        logger.info(f"planner_node: {total_downloads} downloads planned "
                    f"(ERA5: {len(download_plan['era5'])}, DestinE: {len(download_plan['destine'])})")

        return {"download_plan": download_plan}

    @traceable(name="download_node", hide_inputs=True)
    def download_node(gstate: dict) -> dict:
        """Node 3: Execute ALL downloads in parallel using ThreadPoolExecutor."""
        plan = gstate["download_plan"]

        tasks = []
        for item in plan.get("era5", []):
            tasks.append(("era5", item))
        for item in plan.get("destine", []):
            tasks.append(("destine", item))

        if not tasks:
            return {"download_results": []}

        total = len(tasks)
        stream_handler.update_progress(
            f"Data analysis: downloading {total} datasets in parallel..."
        )
        logger.info(f"download_node: starting {total} parallel downloads")

        import time as _time

        results = []
        completed = 0
        max_workers = min(total, 6)
        # Per-source elapsed times for summary
        timings = {"era5": [], "destine": []}

        # Build human-readable labels for progress messages
        def _label(source, params):
            if source == "era5":
                return f"ERA5 {params.get('variable_id', '?')}"
            return f"DestinE {params.get('param_id', '?')}"

        start_times = {}
        t_wall_start = _time.time()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for source, params in tasks:
                lbl = _label(source, params)
                stream_handler.update_progress(f"  ↳ starting download: {lbl}")
                if source == "era5":
                    fut = executor.submit(retrieve_era5_data, **params)
                else:
                    fut = executor.submit(retrieve_destine_data, **params)
                futures[fut] = (source, params)
                start_times[fut] = _time.time()

            for fut in as_completed(futures):
                source, params = futures[fut]
                lbl = _label(source, params)
                elapsed = _time.time() - start_times[fut]
                timings[source].append(elapsed)
                try:
                    dl_result = fut.result()
                    results.append({"source": source, "params": params, "result": dl_result})
                    if dl_result.get("success"):
                        completed += 1
                        stream_handler.update_progress(
                            f"  ✓ {lbl} done ({completed}/{total}) — {elapsed:.1f}s"
                        )
                        logger.info(f"download_node: {lbl} succeeded in {elapsed:.1f}s")
                    else:
                        completed += 1
                        err = dl_result.get('error', '')
                        stream_handler.update_progress(
                            f"  ✗ {lbl} failed ({completed}/{total}) — {elapsed:.1f}s: {err[:80]}"
                        )
                        logger.warning(f"download_node: {lbl} error in {elapsed:.1f}s: {err}")
                except Exception as e:
                    completed += 1
                    stream_handler.update_progress(
                        f"  ✗ {lbl} error ({completed}/{total}) — {elapsed:.1f}s: {str(e)[:80]}"
                    )
                    logger.error(f"download_node: {lbl} failed in {elapsed:.1f}s: {e}")
                    results.append({"source": source, "params": params, "error": str(e)})

        t_wall = _time.time() - t_wall_start

        # Per-source timing summary
        parts = []
        if timings["era5"]:
            t_max = max(timings["era5"])
            parts.append(f"ERA5 {len(timings['era5'])} vars in {t_max:.1f}s")
        if timings["destine"]:
            t_max = max(timings["destine"])
            parts.append(f"DestinE {len(timings['destine'])} vars in {t_max:.1f}s")
        summary = f"all {total} downloads finished — wall time {t_wall:.1f}s"
        if parts:
            summary += f" ({', '.join(parts)})"
        stream_handler.update_progress(f"Data analysis: {summary}")
        return {"download_results": results}

    @traceable(name="analysis_node", hide_inputs=True)
    def analysis_node(gstate: dict) -> dict:
        """Node 4: Run the analysis AgentExecutor (Python REPL, image_viewer, reflect)."""
        analysis_brief = gstate["analysis_brief"]
        downloaded_data_text = gstate.get("downloaded_data_text", "")

        has_python_repl = effective.get("use_powerful_data_analysis", True)
        max_reflect = effective.get("max_reflect", 2)

        # Build tool list (NO download tools)
        tools = []

        if has_python_repl:
            datasets = {
                "uuid_main_dir": gstate["uuid_main_dir"],
                "results_dir": gstate["results_dir"],
            }
            if gstate.get("climate_data_dir"):
                datasets["climate_data_dir"] = gstate["climate_data_dir"]
            if gstate.get("era5_data_dir"):
                datasets["era5_data_dir"] = gstate["era5_data_dir"]

            repl_tool = CustomPythonREPLTool(
                datasets=datasets,
                results_dir=gstate["results_dir"],
                session_key=gstate["thread_id"],
            )
            tools.append(repl_tool)

        tools.append(list_plotting_data_files_tool)

        vision_model = config.get("llm_combine", {}).get("model_name", "gpt-4o")
        image_viewer_tool = create_image_viewer_tool(
            openai_api_key=api_key,
            model_name=vision_model,
            sandbox_path=gstate["results_dir"],
        )
        tools.append(image_viewer_tool)

        if has_python_repl:
            if max_reflect > 0:
                tools.append(reflect_tool)
            tools.append(wise_agent_tool)

        stream_handler.update_progress("Data analysis: running analysis agent...")

        tool_prompt = _create_analysis_prompt(
            gstate["datasets_text"],
            config,
            lat=gstate["lat"],
            lon=gstate["lon"],
            has_climate_data=gstate.get("has_climate_data", False),
            has_hazard_data=gstate.get("has_hazard_data", False),
            has_population_data=gstate.get("has_population_data", False),
            has_era5_data=gstate.get("has_era5_data", False),
            mode_config=effective,
            downloaded_data_text=downloaded_data_text,
        )

        max_iterations = effective.get("max_iterations", 20)
        agent_executor = create_standard_agent_executor(
            llm_dataanalysis_agent,
            tools,
            tool_prompt,
            max_iterations=max_iterations,
        )

        agent_input = {
            "input": analysis_brief or gstate["user_query"],
            "messages": gstate.get("messages", []),
        }

        result = agent_executor.invoke(agent_input)

        # Process intermediate steps for plot images
        plot_images: List[str] = []
        for action, observation in result.get("intermediate_steps", []):
            if action.tool in ("Python_REPL", "python_repl"):
                obs = _normalize_tool_observation(observation)
                if isinstance(obs, dict):
                    plot_images.extend(obs.get("plot_images", []))

        analysis_text = result.get("output", "")

        return {
            "analysis_text": analysis_text,
            "plot_images": plot_images,
        }

    # ==================================================================
    # Run the pipeline: filter → plan → download (parallel) → analyze
    # Sequential calls — each step is a closure with access to outer scope.
    # Parallelism happens INSIDE download_node via ThreadPoolExecutor.
    # ==================================================================

    # Shared mutable state dict for the pipeline
    gstate = {
        # Context for filter_node
        "analysis_context": analysis_context,
        "climate_summary": climate_summary,
        "user_query": state.user,
        "location_str": state.input_params.get("location_str", ""),
        "lat": lat,
        "lon": lon,

        # Sandbox paths
        "uuid_main_dir": state.uuid_main_dir,
        "results_dir": state.results_dir,
        "climate_data_dir": state.climate_data_dir,
        "era5_data_dir": state.era5_data_dir,
        "destine_data_dir": state.destine_data_dir,
        "thread_id": thread_id,

        # Data availability flags for analysis prompt
        "has_climate_data": bool(state.df_list),
        "has_hazard_data": state.hazard_data is not None,
        "has_population_data": bool(state.population_config),
        "has_era5_data": bool(state.era5_climatology_response),

        # For analysis_node tools
        "datasets_text": datasets_text,
        "messages": state.messages,

        # Will be populated by nodes
        "analysis_brief": "",
        "download_plan": {"era5": [], "destine": []},
        "download_results": [],
        "downloaded_data_text": "",
        "analysis_text": "",
        "plot_images": [],
    }

    # Step 1: Filter context
    gstate.update(filter_node(gstate))

    # Step 2: Plan downloads
    gstate.update(planner_node(gstate))

    # Step 3: Download (parallel) — only if there's something to download
    if gstate["download_plan"].get("era5") or gstate["download_plan"].get("destine"):
        gstate.update(download_node(gstate))

    # Step 3b: Process download results into prompt text
    download_processed = _process_download_results(gstate.get("download_results", []))
    gstate["downloaded_data_text"] = download_processed["downloaded_data_text"]

    # Step 4: Run analysis agent
    gstate.update(analysis_node(gstate))

    # ── Save analysis brief ──────────────────────────────────────────────
    analysis_brief = gstate.get("analysis_brief", "")
    brief_path = os.path.join(state.uuid_main_dir, "analysis_brief.txt")
    try:
        with open(brief_path, "w", encoding="utf-8") as f:
            f.write(analysis_brief)
    except Exception as e:
        logger.warning(f"Could not save analysis brief: {e}")

    state.data_analysis_prompt_text = analysis_brief

    # Update state with download results
    state.downloadable_datasets.extend(download_processed["downloadable_datasets"])
    for ref in download_processed["references"]:
        if ref and ref not in state.references:
            state.references.append(ref)

    if download_processed["era5_tool_response"]:
        state.era5_tool_response = download_processed["era5_tool_response"]
    if download_processed["destine_tool_response"]:
        state.destine_tool_response = download_processed["destine_tool_response"]
    for r in download_processed.get("era5_results", []):
        state.input_params.setdefault("era5_results", []).append(r)
    for r in download_processed.get("destine_results", []):
        state.input_params.setdefault("destine_results", []).append(r)

    # Inject downloaded data text into analysis results
    if download_processed["downloaded_data_text"]:
        # This was already used in analysis_node prompt; also store for reference
        state.input_params["downloaded_data_text"] = download_processed["downloaded_data_text"]

    # Get analysis results
    analysis_text = gstate.get("analysis_text", "")
    plot_images = gstate.get("plot_images", [])

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
        "references": state.references,
        "downloadable_datasets": state.downloadable_datasets,
    }
