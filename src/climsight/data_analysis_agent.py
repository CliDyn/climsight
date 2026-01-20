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
from tools.get_data_components import create_get_data_components_tool
from tools.era5_retrieval_tool import era5_retrieval_tool
from tools.python_repl import CustomPythonREPLTool
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
    """Build simple dataset paths text for prompt injection."""
    lines = [
        "## Sandbox Paths (use these exact paths in your code)",
        f"- Main directory: {state.uuid_main_dir}",
        f"- Results directory (save plots here): {state.results_dir}",
        f"- Climate data: {state.climate_data_dir}",
    ]

    if state.era5_data_dir:
        lines.append(f"- ERA5 data: {state.era5_data_dir}")

    # List available climate data files
    if state.climate_data_dir and os.path.exists(state.climate_data_dir):
        try:
            files = os.listdir(state.climate_data_dir)
            if files:
                lines.append(f"\n## Climate Data Files Available")
                lines.append(f"Files in climate_data/: {', '.join(files)}")
                # Highlight the main data.csv file
                if "data.csv" in files:
                    lines.append("Note: 'data.csv' contains the main climatology dataset")
        except Exception as e:
            logger.warning(f"Could not list climate data files: {e}")

    return "\n".join(lines)


def _build_filter_prompt() -> str:
    """Prompt for the analysis brief filter LLM."""
    return (
        "You are a data analysis filter for a climate assistant.\n"
        "Extract only actionable analysis requirements.\n\n"
        "Output format (bullets only):\n"
        "- Target variables (with units if specified)\n"
        "- Thresholds or criteria\n"
        "- Time ranges or scenarios\n"
        "- Spatial specifics (location, buffers)\n"
        "- Analysis tasks (comparisons, trends, plots)\n\n"
        "Rules:\n"
        "- Do NOT include raw climate data values.\n"
        "- Do NOT include long RAG or Wikipedia text.\n"
        "- Omit vague statements that are not actionable.\n"
    )


def _create_tool_prompt(datasets_text: str, config: dict) -> str:
    """System prompt for tool-driven analysis - dynamically built based on config."""
    has_era5 = config.get("use_era5_data", False)
    has_repl = config.get("use_powerful_data_analysis", False)

    prompt = """You are the data analysis agent for ClimSight.
Your job is to provide quantitative climate analysis with visualizations.

## AVAILABLE TOOLS
"""

    tool_num = 1

    # Only include get_data_components if Python_REPL is NOT available
    if not has_repl:
        prompt += f"""
{tool_num}. **get_data_components** - Extract climate variables from stored climatology
   - Variables: Temperature, Precipitation, u_wind, v_wind
   - Returns monthly values for historical AND future climate projections
   - Example: get_data_components(environmental_data="Temperature", months=["Jan", "Feb", "Mar"])
   - ALWAYS use this first to get the baseline climatology data
"""
        tool_num += 1

    if has_era5:
        prompt += f"""
{tool_num}. **retrieve_era5_data** - Download ERA5 reanalysis data (OBSERVED REALITY)
   - Variables: 2m_temperature, total_precipitation, 10m_u_component_of_wind, 10m_v_component_of_wind
   - Specify start_date, end_date (YYYY-MM-DD), and lat/lon bounds
   - Data saved to sandbox as Zarr files for Python analysis

   **CRITICAL**: ERA5 is OBSERVED climate data (ground truth), NOT model projections!
   - Use ERA5 to validate climate models by comparing with model historical period
   - ERA5 shows the ACTUAL climate at this location (recent 10-20 years)
   - Climate models may have bias - ERA5 reveals this bias

   **Standard workflow**: Download ~10 years of recent ERA5 data (e.g., 2014-2024)
   - Example: retrieve_era5_data(variable_id="2m_temperature", start_date="2014-01-01", end_date="2023-12-31", min_latitude=48.0, max_latitude=49.0, min_longitude=11.0, max_longitude=12.0)
   - Then use Python_REPL to compare ERA5 vs climate model historical period
"""
        tool_num += 1

    if has_repl:
        era5_note = ""
        if has_era5:
            era5_note = """
   **ERA5 DATA PROCESSING** (when available):
   - ERA5 data is saved as Zarr format in era5_data/ directory
   - Use xarray to load: `ds = xr.open_zarr('era5_data/<variable>.zarr')`
   - ERA5 = OBSERVED reality, climate_data/data.csv = MODEL projections
   - ALWAYS compare ERA5 with model historical period to show model bias
"""

        prompt += f"""
{tool_num}. **Python_REPL** - Execute Python code for data analysis and visualizations
   - Pre-loaded libraries: pandas (pd), numpy (np), matplotlib.pyplot (plt), xarray (xr)
   - Climate data is AUTO-LOADED in the namespace (see data files below)
   - Working directory is the sandbox root
   - ALWAYS save plots to results/ directory

   **Climate Model Data** (climate_data/data.csv):
   - Monthly climatology from climate models
   - Columns: Month, mean2t (temperature), tp (precipitation), wind_u, wind_v
   - Contains BOTH historical period (1995-2014) AND future projections (2020-2049)
   - These are MODEL outputs, not observations!
{era5_note}
   Example workflow (climate models only):
   ```python
   import pandas as pd
   import matplotlib.pyplot as plt

   # Read climate model data
   df = pd.read_csv('climate_data/data.csv')
   print(df.head())

   # Create temperature plot
   plt.figure(figsize=(10, 6))
   plt.plot(df['Month'], df['mean2t'], marker='o', label='Temperature')
   plt.xlabel('Month')
   plt.ylabel('Temperature (°C)')
   plt.title('Monthly Temperature')
   plt.legend()
   plt.savefig('results/temperature_plot.png', dpi=150, bbox_inches='tight')
   plt.close()
   print('Plot saved to results/temperature_plot.png')
   ```
"""
        tool_num += 1

    prompt += f"""
{tool_num}. **list_plotting_data_files** - List files in sandbox directories
{tool_num + 1}. **reflect** - Think through analysis approach before executing
{tool_num + 2}. **wise_agent** - Get guidance on complex analysis decisions

## REQUIRED WORKFLOW
"""

    if has_era5 and has_repl:
        # ERA5 + Python_REPL: Most comprehensive workflow
        prompt += """
**CRITICAL**: ERA5 is enabled - you MUST download and use it as ground truth!

1. **DOWNLOAD ERA5 OBSERVATIONS** - This is MANDATORY, not optional:
   - Call retrieve_era5_data for the last 10 years (e.g., 2014-01-01 to 2023-12-31)
   - Download at least temperature and precipitation
   - Use exact coordinates from the user query
   - ERA5 = observed reality at this location

2. **LOAD CLIMATE MODEL DATA** - Use Python_REPL:
   - Read climate_data/data.csv (contains model historical + future projections)
   - These are MODEL outputs, not observations

3. **COMPARE ERA5 vs MODEL** - This reveals model bias:
   - Load ERA5 data with xarray: `ds = xr.open_zarr('era5_data/<variable>.zarr')`
   - Calculate ERA5 monthly climatology (mean by month across years)
   - Compare with model historical period (1995-2014)
   - Show the difference (bias) between observations and model

4. **ANALYZE FUTURE PROJECTIONS** - With ERA5 context:
   - Show model future projections
   - Explain how model bias affects confidence in projections
   - Use ERA5 as the baseline ("current climate")

5. **CREATE VISUALIZATIONS** - MANDATORY:
   - Plot 1: ERA5 observations vs model historical vs model future (3-way comparison)
   - Plot 2: Model bias (ERA5 minus model historical)
   - Plot 3: Future change (model future minus ERA5 baseline)
   - Save ALL plots to results/ directory
"""
    elif has_repl and not has_era5:
        # Python_REPL without ERA5: Standard workflow
        prompt += """
1. **READ THE DATA** - Use Python_REPL to inspect climate_data/data.csv
2. **ANALYZE** - Compare historical vs future projections in the data
3. **CREATE VISUALIZATIONS** - This is MANDATORY:
   - Plot monthly temperature comparison (historical vs future)
   - Plot precipitation patterns if relevant
   - Plot wind data if relevant to the query
   - Save ALL plots to results/ directory
"""
    elif not has_repl:
        # Without Python_REPL, use get_data_components
        prompt += """
1. **ALWAYS START** by calling get_data_components for Temperature (all months)
2. **THEN** call get_data_components for Precipitation (all months)
3. **ANALYZE** the data - compare historical vs future projections
"""

        if has_era5:
            prompt += """

**NOTE**: ERA5 is enabled but Python_REPL is not. You can download ERA5 data, but
you won't be able to process it effectively. Consider using get_data_components only.
"""

    prompt += f"""
## SANDBOX PATHS AND DATA

{datasets_text}

## PROACTIVE ANALYSIS

Even if the user doesn't explicitly ask for plots, you SHOULD:
- Create temperature trend visualizations
- Show precipitation comparisons between historical and future
- Highlight months with the largest projected changes
- Identify potential climate risks (e.g., heat stress, drought, flooding)
"""

    if has_era5:
        prompt += """
**With ERA5 data, ALWAYS include:**
- Model bias analysis (ERA5 vs model historical)
- Current climate baseline (from ERA5 observations)
- Confidence assessment (how well do models match observations?)
"""

    prompt += """
## OUTPUT FORMAT

Your final response should include:
"""

    if has_era5 and has_repl:
        prompt += """1. **Current Climate (ERA5 Observations)**: Recent observed temperature, precipitation
2. **Model Performance**: How well climate models match ERA5 observations (bias analysis)
3. **Future Projections**: Model predictions with context of model bias
4. **Climate Change Signal**: Differences between ERA5 baseline and future projections
5. **Critical Months**: Which months show largest changes
6. **Visualizations**: List of plot files created (MANDATORY: 3-way comparison plots)
7. **Implications**: Interpretation with confidence levels based on model-observation agreement
"""
    else:
        prompt += """1. **Key Climate Values**: Extracted temperature, precipitation data
2. **Climate Change Signal**: Differences between historical and future projections
3. **Critical Months**: Which months show largest changes
4. **Visualizations**: List of plot files created (if Python_REPL available)
5. **Implications**: Brief interpretation relevant to the user's query
"""

    prompt += """
Limit total tool calls to 20.
"""
    return prompt


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

    # Tool setup - ORDER MATTERS (matches prompt workflow)
    tools = []

    has_python_repl = config.get("use_powerful_data_analysis", False)

    # 1. get_data_components - ONLY if Python_REPL is NOT available
    #    (When Python_REPL is enabled, it's redundant - agent reads CSV directly)
    if not has_python_repl:
        tools.append(create_get_data_components_tool(state, config, stream_handler))

    # 2. ERA5 retrieval (if enabled)
    if config.get("use_era5_data", False):
        tools.append(era5_retrieval_tool)

    # 3. Python REPL for analysis/visualization (if enabled)
    if has_python_repl:
        repl_tool = CustomPythonREPLTool(
            datasets=datasets,
            results_dir=state.results_dir,
            session_key=thread_id,
        )
        tools.append(repl_tool)

    # 4. Helper tools
    tools.extend([
        list_plotting_data_files_tool,
        reflect_tool,
        wise_agent_tool,
    ])

    stream_handler.update_progress("Data analysis: running tools...")
    tool_prompt = _create_tool_prompt(datasets_text, config)

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

    for action, observation in result.get("intermediate_steps", []):
        if action.tool == "get_data_components":
            data_components_outputs.append(_normalize_tool_observation(observation))
        if action.tool in ("Python_REPL", "python_repl"):
            obs = _normalize_tool_observation(observation)
            if isinstance(obs, dict):
                plot_images.extend(obs.get("plot_images", []))
        if action.tool == "retrieve_era5_data":
            # Keep ERA5 outputs in state only (no raw text in summary).
            state.input_params.setdefault("era5_results", []).append(
                _normalize_tool_observation(observation)
            )

    analysis_text = result.get("output", "")
    if data_components_outputs:
        analysis_text += "\n\nClimatology extracts:\n"
        for item in data_components_outputs:
            analysis_text += json.dumps(item, indent=2) + "\n"

    state.data_analysis_response = analysis_text
    state.data_analysis_images = plot_images

    stream_handler.update_progress("Data analysis complete.")

    return {
        "data_analysis_response": analysis_text,
        "data_analysis_images": plot_images,
        "data_analysis_prompt_text": analysis_brief,
    }
