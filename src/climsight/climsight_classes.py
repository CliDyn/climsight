# climsight_classes.py

from pydantic import BaseModel
from typing import Sequence
from typing import Annotated
from langchain_core.messages import BaseMessage
import operator

class AgentState(BaseModel):
    messages: Annotated[Sequence[BaseMessage], operator.add]  # Not in use up to now
    user: str = ""  # User question
    next: str = ""  # List of next actions
    rag_agent_response: str = ""
    data_agent_response: dict = {}
    zero_agent_response: dict = {}
    final_answer: str = ""
    content_message: str = ""
    input_params: dict = {}
    smart_agent_response: dict = {}
    wikipedia_tool_response: list = []
    ecocrop_search_response: str = ""
    rag_search_response: list = []
    ipcc_rag_agent_response: str = ""
    general_rag_agent_response: str = ""
    data_analysis_response: str = ""  # Response from data analysis agent
    data_analysis_prompt_text: str = ""  # Filtered analysis brief for tools
    data_analysis_images: list = []  # Paths to generated analysis images
    df_list: list = [] # List of dataframes with climate data
    references: list = [] # List of references
    combine_agent_prompt_text: str = ""
    thread_id: str = ""  # Session ID for sandbox storage
    uuid_main_dir: str = ""  # Root sandbox path
    results_dir: str = ""  # Plot output directory
    climate_data_dir: str = ""  # Saved climatology directory
    era5_data_dir: str = ""  # ERA5 output directory
    era5_climatology_response: dict = {}  # ERA5 observed climatology (ground truth)
    # stream_handler: StreamHandler  # Uncomment if needed
