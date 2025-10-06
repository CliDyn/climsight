# src/climsight/agents/data_agent.py

import os
import ast
import uuid
import calendar
import pandas as pd
import numpy as np
import netCDF4 as nc
from pathlib import Path
from typing import Optional, Union, List, Literal

from langchain.agents import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor

from tools.python_repl import create_python_repl_tool
from tools.image_viewer import create_image_viewer_tool

import streamlit as st

# --- Helper Function for AITTA Models (if needed by tools) ---
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

# --- Data Agent Node Function ---
def data_agent(state: dict, config: dict, api_key: str, api_key_local: str, stream_handler, agent_state) -> dict:
    """Data agent node that handles climate data extraction and analysis."""
    
    lat = float(agent_state.input_params['lat'])
    lon = float(agent_state.input_params['lon'])

    # --- Tool Functions ---
    def get_data_components(**kwargs) -> dict:
        """
        Retrieves environmental data components (Temperature, Precipitation, u_wind, v_wind) from climate datasets
        for specific months at the given location. Returns both historical and future climate projections.
        """
        stream_handler.update_progress("Retrieving data for advanced analysis with a smart agent...")

        if isinstance(kwargs.get("months"), str):
            try:
                kwargs["months"] = ast.literal_eval(kwargs["months"])
            except (ValueError, SyntaxError):
                kwargs["months"] = None
                
        args = GetDataComponentsArgs(**kwargs)
        environmental_data = args.environmental_data
        months = args.months
        
        if environmental_data is None:
            return {"error": "No environmental data type specified."}
        if environmental_data not in ["Temperature", "Precipitation", "u_wind", "v_wind"]:
            return {"error": f"Invalid environmental data type: {environmental_data}"}
       
        if config['use_high_resolution_climate_model']:
            df_list = agent_state.df_list
            response = {}                 

            environmental_mapping = {
                "Temperature": "mean2t",
                "Precipitation": "tp",
                "u_wind": "wind_u",
                "v_wind": "wind_v"
            }

            if environmental_data not in environmental_mapping:
                return {"error": f"Invalid environmental data type: {environmental_data}"}
            
            var_name = environmental_mapping[environmental_data]
            
            if not months:
                months = [calendar.month_abbr[m] for m in range(1, 13)]
                
            month_mapping = {calendar.month_abbr[m]: calendar.month_name[m] for m in range(1, 13)}
            selected_months = [month_mapping[abbr] for abbr in months]

            for entry in df_list:
                df = entry.get('dataframe')
                var_meta = entry.get('extracted_vars').get(var_name)
                if df is None:
                    raise ValueError(f"Entry does not contain a 'dataframe' key.")
                    
                data_values = df[df['Month'].isin(selected_months)][var_name].tolist()
                ext_data = {month: np.round(value,2) for month, value in zip(selected_months, data_values)}
                ext_exp = f"Monthly mean values of {environmental_data}, {var_meta['units']} for years: " + entry['years_of_averaging']
                response.update({ext_exp: ext_data})
            return response    
        else:
            lat = float(agent_state.input_params['lat'])
            lon = float(agent_state.input_params['lon'])
            data_path = config['data_settings']['data_path']

            data_files_historical = {
                "Temperature": ("AWI_CM_mm_historical.nc", "tas"),
                "Precipitation": ("AWI_CM_mm_historical_pr.nc", "pr"),
                "u_wind": ("AWI_CM_mm_historical_uas.nc", "uas"),
                "v_wind": ("AWI_CM_mm_historical_vas.nc", "vas")
            }

            data_files_ssp585 = {
                "Temperature": ("AWI_CM_mm_ssp585.nc", "tas"),
                "Precipitation": ("AWI_CM_mm_ssp585_pr.nc", "pr"),
                "u_wind": ("AWI_CM_mm_ssp585_uas.nc", "uas"),
                "v_wind": ("AWI_CM_mm_ssp585_vas.nc", "vas")
            }

            if environmental_data not in data_files_historical:
                return {"error": f"Invalid environmental data type: {environmental_data}"}

            file_name_hist, var_name_hist = data_files_historical[environmental_data]
            file_name_ssp585, var_name_ssp585 = data_files_ssp585[environmental_data]

            file_path_hist = os.path.join(data_path, file_name_hist)
            file_path_ssp585 = os.path.join(data_path, file_name_ssp585)

            if not os.path.exists(file_path_hist):
                return {"error": f"Data file {file_name_hist} not found in {data_path}"}
            if not os.path.exists(file_path_ssp585):
                return {"error": f"Data file {file_name_ssp585} not found in {data_path}"}

            dataset_hist = nc.Dataset(file_path_hist)
            dataset_ssp585 = nc.Dataset(file_path_ssp585)

            lats_hist = dataset_hist.variables['lat'][:]
            lons_hist = dataset_hist.variables['lon'][:]
            lats_ssp585 = dataset_ssp585.variables['lat'][:]
            lons_ssp585 = dataset_ssp585.variables['lon'][:]

            lat_idx_hist = (np.abs(lats_hist - lat)).argmin()
            lon_idx_hist = (np.abs(lons_hist - lon)).argmin()
            lat_idx_ssp585 = (np.abs(lats_ssp585 - lat)).argmin()
            lon_idx_ssp585 = (np.abs(lons_ssp585 - lon)).argmin()

            data_hist = dataset_hist.variables[var_name_hist][:, :, :, lat_idx_hist, lon_idx_hist]
            data_ssp585 = dataset_ssp585.variables[var_name_ssp585][:, :, :, lat_idx_ssp585, lon_idx_ssp585]

            data_hist = np.squeeze(data_hist)
            data_ssp585 = np.squeeze(data_ssp585)

            if environmental_data == "Temperature":
                data_hist = data_hist - 273.15
                data_ssp585 = data_ssp585 - 273.15
                units = "°C"
            elif environmental_data == "Precipitation":
                days_in_month = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
                seconds_in_month = days_in_month * 24 * 3600
                data_hist = data_hist * seconds_in_month
                data_ssp585 = data_ssp585 * seconds_in_month
                units = "mm/month"
            elif environmental_data in ["u_wind", "v_wind"]:
                units = "m/s"
            else:
                units = "unknown"

            dataset_hist.close()
            dataset_ssp585.close()

            all_months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            month_indices = {month: idx for idx, month in enumerate(all_months)}

            if months:
                valid_months = [month for month in months if month in month_indices]
                if not valid_months:
                    return {"error": "Invalid months provided."}
                selected_indices = [month_indices[month] for month in valid_months]
                selected_months = valid_months
            else:
                selected_indices = list(range(12))
                selected_months = all_months

            data_hist = data_hist[selected_indices]
            data_ssp585 = data_ssp585[selected_indices]

            hist_data_dict = {month: f"{value:.2f} {units}" for month, value in zip(selected_months, data_hist)}
            ssp585_data_dict = {month: f"{value:.2f} {units}" for month, value in zip(selected_months, data_ssp585)}

            return {
                f"{environmental_data}_historical": hist_data_dict,
                f"{environmental_data}_ssp585": ssp585_data_dict
            }
    
    class GetDataComponentsArgs(BaseModel):
        environmental_data: Optional[Union[str, Literal["Temperature", "Precipitation", "u_wind", "v_wind"]]] = Field(default=None)
        months: Optional[Union[str, List[str]]] = Field(default=None)

    data_extraction_tool = StructuredTool.from_function(func=get_data_components, name="get_data_components", args_schema=GetDataComponentsArgs)

    python_repl_tool = create_python_repl_tool()

    # Create working directory
    if 'session_uuid' not in st.session_state:
        st.session_state.session_uuid = str(uuid.uuid4())
    work_dir = Path("tmp/sandbox") / st.session_state.session_uuid
    work_dir.mkdir(parents=True, exist_ok=True)

    # Inject context into Python REPL
    if hasattr(python_repl_tool.func, '__self__'):
        repl_instance = python_repl_tool.func.__self__
        
        context = {
            'lat': lat,
            'lon': lon,
            'location_str': agent_state.input_params.get('location_str', ''),
            'work_dir': str(work_dir)
        }
        
        if agent_state.df_list:
            # Add all dataframes from df_list
            for i, entry in enumerate(agent_state.df_list):
                df = entry.get('dataframe')
                if df is not None:
                    context[f'climate_df_{i}'] = df
                    context[f'climate_info_{i}'] = {
                        'years': entry.get('years_of_averaging', ''),
                        'description': entry.get('description', ''),
                        'variables': entry.get('extracted_vars', {})
                    }
            
            # Build DATA_CATALOG dynamically
            catalog = "Available climate datasets:\n"
            for i, entry in enumerate(agent_state.df_list):
                years = entry.get('years_of_averaging', '')
                desc = entry.get('description', '')
                is_main = " (historical reference)" if entry.get('main', False) else ""
                catalog += f"- climate_df_{i}: {years}{is_main} - {desc}\n"
            
            catalog += "\nEach dataset contains monthly values for:\n"
            if agent_state.df_list and agent_state.df_list[0].get('extracted_vars'):
                for var_name, var_info in agent_state.df_list[0]['extracted_vars'].items():
                    catalog += f"- {var_info['full_name']} ({var_info['units']})\n"
            
            context['DATA_CATALOG'] = catalog
                        
        repl_instance.locals.update(context)
        
    data_tools = [data_extraction_tool, python_repl_tool]
    
    if config['model_type'] == "openai":
        try:
            image_viewer_tool = create_image_viewer_tool(api_key, config['model_name_agents'])
            data_tools.append(image_viewer_tool)
        except Exception:
            pass
            
    # --- Agent Definition ---
    temperature = 1 if "o1" in config['model_name_tools'] else 0
    
    if config['model_type'] == "local":
        llm = ChatOpenAI(openai_api_base="http://localhost:8000/v1", model_name=config['model_name_agents'], openai_api_key=api_key_local, temperature=temperature)
    elif config['model_type'] == "openai":
        llm = ChatOpenAI(openai_api_key=api_key, model_name=config['model_name_agents'], temperature=temperature)
    elif config['model_type'] == "aitta":
        llm = get_aitta_chat_model(config['model_name_tools'], temperature=temperature)

    # --- Agent Definition ---
    from langchain.prompts import PromptTemplate

    # This is a standard ReAct prompt template that includes all required variables.
    REACT_PROMPT_TEMPLATE = """Answer the following questions as best you can. You are the data agent of ClimSight.
Your task is to retrieve necessary components of the climatic datasets and perform analysis based on the user's request.
Location of interest: latitude: {lat}, longitude: {lon}, location name: {location_str}

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

    # We need to partially format the prompt with the location context first.
    # The agent executor will then fill in the rest of the variables.
    agent_prompt = PromptTemplate.from_template(REACT_PROMPT_TEMPLATE).partial(
        lat=lat,
        lon=lon,
        location_str=agent_state.input_params.get('location_str', '')
    )

    # Create agent and executor
    agent = create_react_agent(llm, data_tools, agent_prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=data_tools,
        verbose=True,
        return_intermediate_steps=True,
        handle_parsing_errors=True  # Helps prevent crashes
    )
    
    # Get the latest user message
    user_query = state["messages"][-1].content if state["messages"] else ""
    
    # Run the agent
    result = agent_executor.invoke({"input": user_query})
    
    # Extract tool outputs and compile response
    tool_outputs = []
    
    for action, observation in result['intermediate_steps']:
        if action.tool == 'get_data_components':
            tool_outputs.append(f"Data Components Retrieved:\n{observation}")
        elif action.tool == 'python_repl':
            tool_outputs.append(f"Python Analysis:\n{observation}")
        elif action.tool == 'image_viewer':
            tool_outputs.append(f"Image Analysis:\n{observation}")
    
    # Compile response
    response_content = result['output']
    if tool_outputs:
        response_content += "\n\nDetailed outputs:\n" + "\n\n".join(tool_outputs)
    
    # Return updated state
    new_message = {"role": "assistant", "content": response_content, "name": "data_agent"}
    return {
        "messages": state["messages"] + [new_message]
    }