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
    wikipedia_tool_response: str = ""
    ecocrop_search_response: str = ""
    rag_search_response: str = ""
    ipcc_rag_agent_response: str = ""
    general_rag_agent_response: str = ""    
    df_list: list = [] # List of dataframes with climate data
    references: list = [] # List of references
    combine_agent_prompt_text: str = ""
    # stream_handler: StreamHandler  # Uncomment if needed
