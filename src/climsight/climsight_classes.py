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
    final_answser: str = ""
    content_message: str = ""
    input_params: dict = {}
    smart_agent_response: dict = {}
    wikipedia_tool_response: dict = {}
    # stream_handler: StreamHandler  # Uncomment if needed
