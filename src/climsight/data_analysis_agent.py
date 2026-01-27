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
from tools.era5_climatology_tool import create_era5_climatology_tool
from tools.era5_retrieval_tool import era5_retrieval_tool
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


def _create_tool_prompt(datasets_text: str, config: dict, lat: float = None, lon: float = None) -> str:
    """System prompt for tool-driven analysis - dynamically built based on config."""
    has_era5_climatology = config.get("era5_climatology", {}).get("enabled", True)
    has_era5_download = config.get("use_era5_data", False)
    has_repl = config.get("use_powerful_data_analysis", False)

    prompt = """You are the data analysis agent for ClimSight.
Your job is to provide quantitative climate analysis with visualizations.

## AVAILABLE TOOLS
"""

    tool_num = 1

    # TOOL #1: ERA5 Climatology - ALWAYS FIRST (observational ground truth)
    if has_era5_climatology:
        coord_example = ""
        if lat is not None and lon is not None:
            coord_example = f"   - For this query: get_era5_climatology(latitude={lat}, longitude={lon}, variables=[\"t2m\", \"tp\", \"u10\", \"v10\"])\n"

        prompt += f"""
{tool_num}. **get_era5_climatology** - Extract OBSERVED climate data (CALL THIS FIRST!)
   - Source: ERA5 reanalysis 2015-2025 monthly climatology
   - This is GROUND TRUTH - actual observations, not model output
   - Variables: temperature (t2m), precipitation (tp), wind_u (u10), wind_v (v10), dewpoint (d2m), pressure (msl)
   - Returns monthly averages for the nearest grid point (~28km resolution)
{coord_example}
   **CRITICAL**: This tool provides what the climate ACTUALLY IS at this location.
   Use this as the BASELINE to compare against climate model projections.
"""
        tool_num += 1

    # TOOL #2: get_data_components - only if Python_REPL is NOT available
    if not has_repl:
        prompt += f"""
{tool_num}. **get_data_components** - Extract climate variables from climate MODEL projections
   - Variables: Temperature, Precipitation, u_wind, v_wind
   - Returns monthly values for historical AND future climate projections
   - Example: get_data_components(environmental_data="Temperature", months=["Jan", "Feb", "Mar"])
   - These are MODEL outputs - compare with ERA5 observations to assess model quality
"""
        tool_num += 1

    # TOOL #3: ERA5 time series download (optional, for detailed analysis)
    if has_era5_download:
        work_dir_str = sandbox_paths.get("uuid_main_dir", "") if "sandbox_paths" in dir() else ""
        prompt += f"""
{tool_num}. **retrieve_era5_data** - Retrieve ERA5 Surface climate data from Earthmover (Arraylake)
   <Important>
   Use this tool to retrieve **historical weather/climate context (Time Series)**.

   **DATA SOURCE:** Earthmover (Arraylake), hardcoded to "temporal" mode.
   **VARIABLE CODES:** Use short codes: 't2' (Temp), 'u10'/'v10' (Wind), 'mslp' (Pressure), 'tp' (Precip).
   **PATH RULE:** You MUST always pass `work_dir` to this tool.

   **OUTPUT USAGE:**
   The tool returns a path to a Zarr store.
   **CRITICAL:** If you requested a specific point, the data is likely already reduced to that point (nearest neighbor).

   How to load result in Python_REPL:
   ```python
   import xarray as xr
   ds = xr.open_dataset(path_from_tool_response, engine='zarr', chunks={{{{}}}})
   # If lat/lon are scalar, access directly:
   data = ds['t2'].to_series()
   ```
   </Important>
"""
        tool_num += 1

    # TOOL #4: Python REPL
    if has_repl:
        prompt += f"""
{tool_num}. **Python_REPL** - Execute Python code for data analysis and visualizations
   - Pre-loaded: pandas (pd), numpy (np), matplotlib.pyplot (plt), xarray (xr)
   - Working directory is the sandbox root
   - ALWAYS save plots to results/ directory

   **Climate Model Data** (climate_data/data.csv):
   - Monthly climatology from climate models (MODEL projections, not observations)
   - Columns: Month, mean2t (temperature °C), tp (precipitation), wind_u, wind_v
   - Contains historical period AND future projections

   **ERA5 Climatology** (era5_climatology.json):
   - After calling get_era5_climatology, results are saved here
   - Load with: `import json; era5 = json.load(open('era5_climatology.json'))`
   - Use ERA5 as GROUND TRUTH baseline for comparisons

   Example workflow:
   ```python
   import pandas as pd
   import json
   import matplotlib.pyplot as plt

   # Load ERA5 observations (ground truth)
   era5 = json.load(open('era5_climatology.json'))
   era5_temp = era5['variables']['t2m']['monthly_values']

   # Load climate model projections
   df = pd.read_csv('climate_data/data.csv')

   # Compare: ERA5 observations vs climate model
   months = list(era5_temp.keys())
   era5_values = list(era5_temp.values())
   model_values = df['mean2t'].tolist()[:12]  # historical period

   plt.figure(figsize=(10, 6))
   plt.plot(months, era5_values, 'b-o', label='ERA5 Observations (2015-2025)')
   plt.plot(months, model_values, 'r--s', label='Climate Model')
   plt.xlabel('Month')
   plt.ylabel('Temperature (°C)')
   plt.title('Observed vs Modeled Temperature')
   plt.legend()
   plt.xticks(rotation=45)
   plt.tight_layout()
   plt.savefig('results/era5_vs_model.png', dpi=150)
   plt.close()
   print('Plot saved to results/era5_vs_model.png')
   ```
"""
        tool_num += 1

    prompt += f"""
{tool_num}. **list_plotting_data_files** - List files in sandbox directories
"""
    tool_num += 1

    # image_viewer only available with Python REPL
    if has_repl:
        prompt += f"""
{tool_num}. **image_viewer** - View generated plots to verify quality
   <Important>
   Pass the EXACT path printed by Python_REPL after saving a plot.
   Use this to verify your visualizations before finalizing.
   </Important>
"""
        tool_num += 1

    # reflect_on_image and wise_agent only available with Python REPL
    if has_repl:
        prompt += f"""{tool_num}. **reflect_on_image** - Analyze a generated plot and get feedback for improvements
{tool_num + 1}. **wise_agent** - Get guidance on complex visualization decisions
"""
        tool_num += 2

    prompt += """
## REQUIRED WORKFLOW
"""

    if has_era5_climatology:
        prompt += """
**STEP 1 - GET OBSERVATIONS (MANDATORY):**
Call get_era5_climatology FIRST to get the observed climate baseline.
- Extract at minimum: temperature (t2m), precipitation (tp)
- This is what the climate ACTUALLY IS at this location (2015-2025 average)
"""

        if has_repl:
            prompt += """
**STEP 2 - LOAD MODEL DATA:**
Use Python_REPL to read climate_data/data.csv (model projections)

**STEP 3 - COMPARE OBSERVATIONS vs MODEL:**
- ERA5 = ground truth (what we observe NOW)
- Model historical = what the model simulates for recent past
- Difference = MODEL BIAS (critical for interpreting future projections)

**STEP 4 - ANALYZE FUTURE WITH CONTEXT:**
- Show future projections from climate model
- Explain how model bias affects confidence
- Future change = (Model future) - (ERA5 baseline)

**STEP 5 - CREATE VISUALIZATIONS (MANDATORY):**
- Plot 1: ERA5 observations vs model (shows model bias)
- Plot 2: Future projections with ERA5 baseline
- Save ALL plots to results/ directory
"""
        else:
            prompt += """
**STEP 2 - GET MODEL DATA:**
Call get_data_components for Temperature and Precipitation

**STEP 3 - COMPARE:**
- ERA5 climatology = observations (ground truth)
- Model data = projections
- Note any differences between observed and modeled values
"""
    elif has_repl:
        prompt += """
1. **READ THE DATA** - Use Python_REPL to inspect climate_data/data.csv
2. **ANALYZE** - Compare historical vs future projections
3. **CREATE VISUALIZATIONS** - Save plots to results/ directory
"""
    else:
        prompt += """
1. **ALWAYS START** by calling get_data_components for Temperature (all months)
2. **THEN** call get_data_components for Precipitation (all months)
3. **ANALYZE** the data - compare historical vs future projections
"""

    prompt += f"""
## SANDBOX PATHS AND DATA

{datasets_text}

## PROACTIVE ANALYSIS

Even if the user doesn't explicitly ask for plots, you SHOULD:
- Create temperature trend visualizations
- Show precipitation comparisons
- Highlight months with largest projected changes
- Identify potential climate risks (heat stress, drought, flooding)
"""

    if has_era5_climatology:
        prompt += """
**With ERA5 observations, ALWAYS include:**
- Current observed climate (from ERA5 - this is REALITY)
- Model performance assessment (how well does the model match observations?)
- Future projections interpreted in context of model quality
"""

    prompt += """
## OUTPUT FORMAT

Your final response should include:
"""

    if has_era5_climatology:
        prompt += """1. **Current Climate (ERA5 Observations)**: What the climate ACTUALLY IS (2015-2025 average)
2. **Model Assessment**: How well climate models match ERA5 observations
3. **Future Projections**: Model predictions with confidence based on model-observation agreement
4. **Climate Change Signal**: Projected changes from current observed baseline
5. **Critical Months**: Which months show largest changes
6. **Visualizations**: List of plot files created
7. **Implications**: Interpretation relevant to the user's query
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

    # Get coordinates for prompt
    lat = state.input_params.get('lat')
    lon = state.input_params.get('lon')
    try:
        lat = float(lat) if lat is not None else None
        lon = float(lon) if lon is not None else None
    except (ValueError, TypeError):
        lat, lon = None, None

    # Tool setup - ORDER MATTERS (matches prompt workflow)
    tools = []

    has_python_repl = config.get("use_powerful_data_analysis", False)
    has_era5_climatology = config.get("era5_climatology", {}).get("enabled", True)

    # 1. ERA5 Climatology - ALWAYS FIRST (observational ground truth)
    if has_era5_climatology:
        tools.append(create_era5_climatology_tool(state, config, stream_handler))

    # 2. get_data_components - ONLY if Python_REPL is NOT available
    #    (When Python_REPL is enabled, it's redundant - agent reads CSV directly)
    if not has_python_repl:
        tools.append(create_get_data_components_tool(state, config, stream_handler))

    # 3. ERA5 time series retrieval (if enabled - for detailed year-by-year analysis)
    if config.get("use_era5_data", False):
        tools.append(era5_retrieval_tool)

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

    # 6. Image viewer - ONLY when Python REPL is enabled
    if has_python_repl:
        # Extract model name from config or default to gpt-4o
        vision_model = config.get("llm_combine", {}).get("model_name", "gpt-4o")
        
        # FIX: Pass the required api_key and model_name
        image_viewer_tool = create_image_viewer_tool(api_key, vision_model)
        tools.append(image_viewer_tool)

    # 7. Image reflection and wise_agent - ONLY when Python REPL is enabled
    #    (these tools are for evaluating/creating visualizations)
    if has_python_repl:
        tools.append(reflect_tool)
        tools.append(wise_agent_tool)

    stream_handler.update_progress("Data analysis: running tools...")
    tool_prompt = _create_tool_prompt(datasets_text, config, lat=lat, lon=lon)

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

    for action, observation in result.get("intermediate_steps", []):
        if action.tool == "get_era5_climatology":
            obs = _normalize_tool_observation(observation)
            if isinstance(obs, dict) and "error" not in obs:
                era5_climatology_output = obs
                state.era5_climatology_response = obs
        if action.tool == "get_data_components":
            data_components_outputs.append(_normalize_tool_observation(observation))
        if action.tool in ("Python_REPL", "python_repl"):
            obs = _normalize_tool_observation(observation)
            if isinstance(obs, dict):
                plot_images.extend(obs.get("plot_images", []))
        if action.tool == "retrieve_era5_data":
            # Handle ERA5 retrieval tool output
            obs = _normalize_tool_observation(observation)
            if isinstance(obs, dict):
                era5_output = str(obs)
            elif hasattr(obs, 'content'):
                era5_output = obs.content
            else:
                era5_output = str(obs)
            # Store in state
            state.era5_tool_response = era5_output
            state.input_params.setdefault("era5_results", []).append(obs)

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

    state.data_analysis_response = analysis_text
    state.data_analysis_images = plot_images

    stream_handler.update_progress("Data analysis complete.")

    return {
        "data_analysis_response": analysis_text,
        "data_analysis_images": plot_images,
        "data_analysis_prompt_text": analysis_brief,
        "era5_climatology_response": state.era5_climatology_response,
        "era5_tool_response": getattr(state, 'era5_tool_response', None),
    }
