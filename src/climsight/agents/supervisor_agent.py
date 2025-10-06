# src/climsight/agents/supervisor_agent.py

from typing import Literal
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import MessagesState
from pydantic import BaseModel
import os

# This defines the structure of the supervisor's decision
class SupervisorDecision(BaseModel):
    next_agent: Literal["researcher", "data_agent", "__end__"]
    reasoning: str

def get_aitta_chat_model(model_name, **kwargs):
    try:
        from aitta_client import Model, Client
        from aitta_client.authentication import APIKeyAccessTokenSource
        aitta_url = 'https://api-climatedt-aitta.2.rahtiapp.fi'
        aitta_api_key = os.environ['AITTA_API_KEY']
        client = Client(aitta_url, APIKeyAccessTokenSource(aitta_api_key, aitta_url))
        model = Model.load(model_name, client)
        access_token = client.access_token_source.get_access_token()
        return ChatOpenAI(
            openai_api_key=access_token,
            openai_api_base=model.openai_api_url,
            model_name=model.id,
            **kwargs
        )
    except Exception:
        return None

# This is your actual supervisor node
def supervisor(state: MessagesState, config: dict, api_key: str, api_key_local: str) -> dict:
    """Supervisor node that routes to appropriate agents based on user query."""
    
    # Initialize LLM based on config
    temperature = 0
    if config['model_type'] == "local":
        model = ChatOpenAI(
            openai_api_base="http://localhost:8000/v1",
            model_name=config['model_name_agents'],
            openai_api_key=api_key_local,
            temperature=temperature
        )
    elif config['model_type'] == "openai":
        model = ChatOpenAI(
            openai_api_key=api_key,
            model_name=config['model_name_agents'],
            temperature=temperature
        )
    elif config['model_type'] == "aitta":
        model = get_aitta_chat_model(config['model_name_tools'], temperature=temperature)
    
    # Get the latest user message
    user_query = state["messages"][-1].content if state["messages"] else ""
    
    # Check for previous agent responses
    previous_responses = [
        f"{msg.name}: {msg.content}"
        for msg in state["messages"]
        if hasattr(msg, "name") and msg.name in ["researcher", "data_agent"]
    ]
    
    prompt = ChatPromptTemplate.from_template("""
    You are the supervisor for ClimSight's smart agent. Based on the user query and any previous agent responses, decide the next agent to call.
    
    User query: {query}
    
    Previous agent responses:
    {previous_responses}
    
    Available agents:
    - "researcher": For Wikipedia searches, RAG, or ECOCROP queries (e.g., crop info, general knowledge).
    - "data_agent": For climate data extraction, analysis, or visualization (e.g., using python_repl).
    - "__end__": If the query has been fully addressed.
    
    Guidelines:
    1. If the query needs background info, route to "researcher".
    2. If the query needs climate data analysis, route to "data_agent".
    3. For complex queries, start with "researcher" to get context, then use "data_agent".
    4. Return "__end__" only when all aspects of the query are resolved.
    
    Provide your decision and reasoning.
    """)
    
    structured_model = model.with_structured_output(SupervisorDecision)
    
    response = structured_model.invoke(
        prompt.format(
            query=user_query,
            previous_responses="\n".join(previous_responses) if previous_responses else "None"
        )
    )
    
    # The output determines the next step in the graph
    return {"next_agent": response.next_agent}