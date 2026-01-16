"""
Data Analysis Agent

This agent receives all state from previous agents and performs:
1. Climate data extraction from datasets
2. Data post-processing and calculations
3. Visualization and analysis using Python REPL
4. Scientific interpretation of results

Currently a stub implementation - will be fully implemented in next phase.

This agent will eventually include:
- get_data_components tool (from smart_agent_backup.py)
- python_repl tool for analysis and visualization
- image_viewer tool for scientific interpretation of plots
- Statistical calculations and trend analysis
"""

from climsight_classes import AgentState
import logging

logger = logging.getLogger(__name__)


def data_analysis_agent(state: AgentState, config, api_key, api_key_local, stream_handler):
    """
    Data analysis and visualization agent (stub implementation).

    This agent will be responsible for:
    - Extracting specific climate data components based on user needs
    - Performing statistical analysis and calculations
    - Creating visualizations and plots
    - Interpreting results using image_viewer
    - Providing data-driven insights

    Args:
        state: AgentState containing all previous agent outputs including:
            - user: Original user query
            - input_params: Location and other parameters
            - df_list: Climate data from data_agent
            - smart_agent_response: Background information (if smart_agent was enabled)
            - ipcc_rag_agent_response: IPCC report insights
            - general_rag_agent_response: General climate literature insights
            - zero_rag_agent_response: Geographic/environmental context
        config: Configuration dictionary
        api_key: OpenAI API key
        api_key_local: Local model API key
        stream_handler: Stream handler for progress updates

    Returns:
        dict: Updated state with analysis results
    """
    stream_handler.update_progress("Data analysis agent (stub - passing through)...")

    logger.info("Data analysis agent stub called")
    logger.info(f"User query: {state.user}")
    logger.info(f"Location: {state.input_params.get('location_str', 'unknown')}")

    # TODO: Implement full data analysis functionality
    # Phase 1 implementation will include:
    # 1. Move get_data_components tool from smart_agent_backup.py
    # 2. Move python_repl tool initialization with climate data context
    # 3. Move image_viewer tool for plot interpretation
    # 4. Create agent executor with data analysis prompts
    # 5. Integrate with combine_agent for final synthesis

    # For now, just pass through with a stub message
    stub_response = "Data analysis agent not yet implemented. Analysis and visualization features will be added in the next implementation phase."

    return {
        'data_analysis_response': stub_response
    }
