# smart_agent.py

from pydantic import BaseModel, Field
from typing import Optional, Literal, List
import netCDF4 as nc
import numpy as np
import os
import ast
from typing import Union

from langchain.agents import AgentExecutor, create_openai_tools_agent, Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.tools import StructuredTool
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.schema import AIMessage
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate

#Import tools
from tools.python_repl import create_python_repl_tool
from tools.image_viewer import create_image_viewer_tool

#import requests
#from bs4 import BeautifulSoup
#from urllib.parse import quote_plus
#from langchain.schema import Document
from langchain_community.document_loaders import WikipediaLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

#Import for working Path
import uuid
import streamlit as st
from pathlib import Path
try:
    from aitta_client import Model, Client
    from aitta_client.authentication import APIKeyAccessTokenSource
except:
    pass

# Import AgentState from climsight_classes
from climsight_classes import AgentState
import calendar
import pandas as pd

# Import LangGraph components
from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, AIMessage
from typing import TypedDict, Annotated

# Import agent nodes
from agents.supervisor_agent import supervisor
from agents.researcher_agent import researcher
from agents.data_agent import data_agent

def get_aitta_chat_model(model_name, **kwargs):
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

def smart_agent(state: AgentState, config, api_key, api_key_local, stream_handler):
    """Refactored smart agent using LangGraph supervisor architecture."""
    stream_handler.update_progress("Running advanced analysis with smart agent...")
    
    # Define a simple state class for the graph
    class GraphState(TypedDict):
        messages: list
        next_agent: str
    
    # Build the LangGraph
    builder = StateGraph(GraphState)
    
    # Add nodes
    builder.add_node("supervisor", lambda s: supervisor(s, config, api_key, api_key_local))
    builder.add_node("researcher", lambda s: researcher(s, config, api_key, api_key_local))
    builder.add_node("data_agent", lambda s: data_agent(s, config, api_key, api_key_local, stream_handler, state))
    
    # Define edges
    builder.add_edge(START, "supervisor")
    builder.add_edge("researcher", "supervisor")  # After researcher finishes, go back to supervisor
    builder.add_edge("data_agent", "supervisor")   # After data_agent finishes, go back to supervisor
    
    # The supervisor's decision routes to the correct agent or ends the process
    builder.add_conditional_edges(
        "supervisor",
        lambda x: x["next_agent"],
        {
            "researcher": "researcher",
            "data_agent": "data_agent",
            "__end__": END,
        },
    )
    
    # Compile the graph
    graph = builder.compile()
    
    # Convert state to GraphState format
    initial_messages = []
    if state.user:
        initial_messages.append(HumanMessage(content=state.user))
    
    initial_state = {
        "messages": initial_messages,
        "next_agent": "supervisor"  # Start with supervisor
    }
    
    # Run the graph
    result = graph.invoke(initial_state)
    
    # Extract the final response and update state
    final_messages = result.get("messages", [])
    
    # Find all agent responses
    agent_responses = {
        "researcher": [],
        "data_agent": [],
        "final": ""
    }
    
    for msg in final_messages:
        if hasattr(msg, "name"):
            if msg.name == "researcher":
                agent_responses["researcher"].append(msg.content)
            elif msg.name == "data_agent":
                agent_responses["data_agent"].append(msg.content)
        elif isinstance(msg, AIMessage):
            agent_responses["final"] = msg.content
    
    # Compile final response
    smart_agent_response = agent_responses["final"] or "Analysis complete."
    
    # Extract tool outputs from agent responses (parse from response content)
    state.smart_agent_response = {'output': smart_agent_response}
    
    # Parse references from researcher responses
    for resp in agent_responses["researcher"]:
        if "References:" in resp:
            refs_section = resp.split("References:")[1].strip()
            refs = refs_section.split("\n")
            state.references.extend([ref.strip() for ref in refs if ref.strip()])
    
    # Return the results
    return {
        'smart_agent_response': state.smart_agent_response,
        'wikipedia_tool_response': state.wikipedia_tool_response,
        'ecocrop_search_response': state.ecocrop_search_response,
        'rag_search_response': state.rag_search_response,
        'references': state.references
    }
