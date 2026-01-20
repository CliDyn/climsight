"""
    Terminal Wrapper Module 
"""
#general 
import logging
import yaml
import os
import matplotlib.pyplot as plt
import time

# rag
from rag import load_rag 

# climsight modules
from stream_handler import StreamHandler
from climsight_engine import llm_request, forming_request, location_request
from data_container import DataContainer

from extract_climatedata_functions import plot_climate_data
from sandbox_utils import ensure_thread_id, ensure_sandbox_dirs, get_sandbox_paths

logger = logging.getLogger(__name__)

data_pocket = DataContainer()

def input_with_default(prompt, default_value):
    user_input = input(prompt)
    if user_input == "":
        return default_value
    return user_input

def print_verbose(verbose, message):
    if verbose:
        print(message)  

def run_terminal(config, api_key='', skip_llm_call=False, lon=None, lat=None, user_message='', show_add_info='',verbose=True, rag_activated=None, references=None):
    '''
        Inputs:
        - config (dict): Configuration, default is an empty dictionary.   
        - api_key (string): API Key, default is an empty string. default ''
        - skip_llm_call (bool): If True - skipp final call to LLM. default False    
        - lat (float): Latitude of the location to analyze. default None
        - lon (float): Longitude of the location to analyze. default None
        - user_message (string): Question for the LLM. default empty ''
        - show_add_info (string): If 'y' - show additional information, if 'n' - do not show additional information. default ''
        - verbose (bool): If True - print additional information, if False - do not print additional information (used for loop request). default True
        - rag_activated (bool): whether or not to include the text based rag
        Output:
        - output (string): Output from the LLM.
    '''      
    # Config
    try:
        climatemodel_name = config['climatemodel_name']
        lat_default = config['lat_default']
        lon_default = config['lon_default']
        rag_default = config['rag_settings']['rag_activated']
    except KeyError as e:
        logging.error(f"Missing configuration key: {e}")
        raise RuntimeError(f"Missing configuration key: {e}")   

    # Ensure sandbox exists for this CLI session.
    thread_id = ensure_thread_id()
    sandbox_paths = get_sandbox_paths(thread_id)
    ensure_sandbox_dirs(sandbox_paths)

    if not isinstance(skip_llm_call, bool):
        logging.error(f"skip_llm_call must be bool")
        raise TypeError("skip_llm_call must be  bool")    

############################# input
    print_verbose(verbose, f"\n \n \n")
    print_verbose(verbose, f"Welcome to Climsight!")
    print_verbose(verbose, f"\n")    
    if lon is None:
        lon = input_with_default(f"Please provide longitude of the location ({lon_default}): ", lon_default)
        try:
            lon = float(lon)
        except Exception as e:
            logging.error(f"lat and lon must be floats: {e}")
            raise RuntimeError(f"lat and lon must be floats: {e}")            
    print_verbose(verbose, f"Longitude: {lon}")
    print_verbose(verbose, f"\n")        
    if lat is None:
        lat = input_with_default(f"Please provide latitude of the location ({lat_default}): ", lat_default)
        try:
            lat = float(lat)
        except Exception as e:
            logging.error(f"lat and lon must be floats: {e}")
            raise RuntimeError(f"lat and lon must be floats: {e}")
    print_verbose(verbose, f"Latitude: {lat}")
    print_verbose(verbose, f"\n")    

    if not isinstance(user_message, str):
        logging.error(f"user_message must be a string ")
        raise TypeError("user_message must be a string")
    
    if not user_message: 
        user_message = input(f"Describe the activity that you would like to evaluate:\n")
        print_verbose(verbose, f"\n")    

    if not isinstance(api_key, str):
        logging.error(f"api_key must be a string ")
        raise TypeError("api_key must be a string")

    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY") # check if OPENAI_API_KEY is set in the environment
    if (not api_key) and (not skip_llm_call):
        api_key = input("Please provide openAI API key: ")
        print_verbose(verbose, f"\n")    
    else:
        print_verbose(verbose, "openAI API key accepted.")
        print_verbose(verbose, f"\n")

    api_key_local = os.environ.get("OPENAI_API_KEY_LOCAL")
    if not api_key_local:
        api_key_local = ""

    if rag_activated is None:
        rag_activated = input_with_default(f"Do you want to run ClimSight with (y) or without (n) additional text source RAG? (Default depends on your config settings): ", rag_default)
        if isinstance(rag_activated, str):
            if rag_activated == 'y':
                rag_activated = True
            elif rag_activated == 'n':
                rag_activated =  False
            else:
                logging.error("rag_activated must either be 'y', 'n', or empty, but nothing else")
                raise TypeError("Please enter either 'y', 'n', or leave it empty to use the default value.")
        if not isinstance(rag_activated, bool):
            logging.error('rag_activated must be a bool')
    print_verbose(verbose, f"RAG activated: {rag_activated}")
    print_verbose(verbose, f"\n")

    if not isinstance(show_add_info, str):
        logging.error(f"show_add_info must be a string ")
        raise TypeError("show_add_info must be a string")
    if not show_add_info:
        show_add_info = input_with_default("Do you want to see and save additional information? (y/n, default y): ","y")
    if show_add_info=="n":
        show_add_info=False
        print_verbose(verbose, f"Additional inforamtion will be not shown.")        
    else:
        show_add_info=True
        print_verbose(verbose, f"Additional inforamtion will be shown and saved in files.")                
    config['show_add_info'] = show_add_info
    
    print_verbose(verbose, "")
    print_verbose(verbose, "Getting info on a point...")
    
    # Record the start time
    start_time = time.time()

    # RAG
    if not skip_llm_call and rag_activated:
        try:
            logger.info("RAG is activated and skipllmcall is False. Loading IPCC RAG database...")
            ipcc_rag_ready, ipcc_rag_db = load_rag(config, openai_api_key=api_key, db_type='ipcc')
        except Exception as e:
            logger.warning(f"IPCC RAG database initialization skipped or failed: {e}")
            rag_ready = False
            rag_db = None
        try:
            logger.info("RAG is activated and skipllmcall is False. Loading general RAG database...")
            general_rag_ready, general_rag_db = load_rag(config, openai_api_key=api_key, db_type='general')
        except Exception as e:
            logger.warning(f"(General) RAG database initialization skipped or failed: {e}")
            general_rag_ready = False
            general_rag_db = None
            
    is_on_land = True

    if config['llmModeKey'] == "direct_llm":
        generator = forming_request(config, lat, lon, user_message)
        while True:
            try:
                # Get the next intermediate result from the generator
                result = next(generator)
                print_verbose(verbose, f"{result}")
            except StopIteration as e:
                # The generator is exhausted, and e.value contains the final result
                gen_output = e.value
                # check if Error ocure:
                if isinstance(gen_output,str):
                    if "Error" in gen_output:
                        if "point_is_in_ocean" in gen_output:
                            is_on_land = False
                            print_verbose(verbose, f"The selected point is in the ocean. Please choose a location on land.")
                else:    
                    content_message, input_params, df_data, figs, data = e.value
                    data_pocket.df['df_data'] = df_data
                    data_pocket.figs = figs
                    data_pocket.data = data                
                break     
    else:
        # Agent LLM mode (load only location info)
        # get first location information only, input_params and content_message are only partly filled
        content_message, input_params = location_request(config, lat, lon)
        if not input_params:
            is_on_land = False
            print_verbose(verbose, f"The selected point is in the ocean. Please choose a location on land.")
        else:
            # Pass sandbox paths into the agent state.
            input_params['thread_id'] = thread_id
            input_params.update(sandbox_paths)
            # extend input_params with user_message
            input_params['user_message'] = user_message
            content_message = "Human request: {user_message} \n " + content_message
            print_verbose(verbose, f"{input_params['location_str_for_print']}")
            if input_params['is_inland_water']:
                print_verbose(verbose, f"""{input_params['water_body_status']}: Our analyses are currently only meant for land areas. Please select another location for a better result.""")
    
    # Record the start time
    forming_request_time = time.time() - start_time
        
    if is_on_land:        
        start_time = time.time()
        stream_handler = StreamHandler()
        output = ''
        def print_progress(message):
            print(f"[PROGRESS] {message}")     
        stream_handler.update_progress = print_progress
               
        if not skip_llm_call:
            output, input_params, content_message, combine_agent_prompt_text = llm_request(content_message, 
                                                                input_params, 
                                                                config, 
                                                                api_key, api_key_local, 
                                                                stream_handler, 
                                                                ipcc_rag_ready, ipcc_rag_db, general_rag_ready, general_rag_db, 
                                                                data_pocket,
                                                                references=references)   
            figs = data_pocket.figs
            data = data_pocket.data
                
            print_verbose(verbose, "|=============================================================================")    
            print_verbose(verbose, "")    
            print_verbose(verbose, output)            
            print_verbose(verbose, "")    
            print_verbose(verbose, "|=============================================================================")    
        else:
            output = content_message.format(**input_params)
            print_verbose(verbose, "|============================ Prompt after formatting:  ======================")    
            print_verbose(verbose, "")            
            print_verbose(verbose, config['system_role'])    
            print_verbose(verbose, "")            
            print_verbose(verbose, output)            
            print_verbose(verbose, )    
            print_verbose(verbose, "|=============================================================================")    
                
        # Record the time
        llm_request_time = time.time() - start_time

        # PLOTTING ADDITIONAL INFORMATION
        if show_add_info: 
            print_verbose(verbose, "Additional information")
            if 'lat' and 'lon' in input_params:
                print_verbose(verbose, f"**Coordinates:** {input_params['lat']}, {input_params['lon']}")
            if 'elevation' in input_params:
                print_verbose(verbose, f"**Elevation:** {input_params['elevation']} m")
            if 'current_land_use' in input_params:
                print_verbose(verbose, f"**Current land use:** {input_params['current_land_use']}")
            if 'soil' in input_params:
                print_verbose(verbose, f"**Soil type:** {input_params['soil']}")
            if 'biodiv' in input_params:
                print_verbose(verbose, f"**Occuring species:** {input_params['biodiv']}")
            if 'distance_to_coastline' in input_params:
                print_verbose(verbose, f"**Distance to the shore:** {round(float(input_params['distance_to_coastline']), 2)} m")
            # figures need to move to engine
            # Climate Data
            # print("**Climate data:**")
            # print(
            #     "Near surface temperature (in °C)",
            # )
            # st.line_chart(
            #     df_data,
            #     x="Month",
            #     y=["Present Day Temperature", "Future Temperature"],
            #     color=["#d62728", "#0000ff"],
            # )
            # print(
            #     "Precipitation (in mm)",
            # )
            # st.line_chart(
            #     df_data,
            #     x="Month",
            #     y=["Present Day Precipitation", "Future Precipitation"],
            #     color=["#d62728", "#0000ff"],
            # )
            # print(
            #     "Wind speed (in m*s-1)",
            # )
            # st.line_chart(
            #     df_data,
            #     x="Month",
            #     y=["Present Day Wind Speed", "Future Wind Speed"],
            #     color=["#d62728", "#0000ff"],
            # )
            # Determine the model information string based on climatemodel_name
            # if climatemodel_name == 'AWI_CM':
            #     model_info = 'AWI-CM-1-1-MR, scenarios: historical and SSP5-8.5'
            # elif climatemodel_name == 'tco1279':
            #     model_info = 'AWI-CM-3 TCo1279_DART, scenarios: historical (2000-2009) and SSP5-8.5 (2090-2099)'
            # elif climatemodel_name == 'tco319':
            #     model_info = 'AWI-CM-3 TCo319_DART, scenarios: historical (2000-2009), and SSP5-8.5 (2090-2099)'
            # else:
            #     model_info = 'unknown climate model'

            # print("Climate model: ")
            # print("   ", model_info)
            if config['use_high_resolution_climate_model']:
                try:
                    df_list = data_pocket.data['high_res_climate']['df_list']
                    figs_climate = plot_climate_data(df_list)
                    for fig_dict in figs_climate:
                        fname=f"{fig_dict['full_name']}.png"
                        print_verbose(verbose, f"Figure with {fig_dict['full_name']} was saved in {fname}.")
                        fig_dict['fig'].savefig(fname)
                    print_verbose(verbose, "Source for this figures: ")
                    print_verbose(verbose, figs_climate[0]['source'])
                    print_verbose(verbose, "\n")    
                except KeyError as e:
                    logger.warning(f"Error by ploting climate data: {e}")
                                
            # Natural Hazards
            if 'haz_fig' in figs:
                fname = "natural_hazards.png"
                print_verbose(verbose, f"Figure with natural hazards was saved in {fname}.")
                figs['haz_fig']['fig'].savefig(fname)
                print_verbose(verbose, "Source for this figure: ")
                print_verbose(verbose, figs['haz_fig']['source'])
                print_verbose(verbose, "\n")    
            # Population Data
            if 'population_plot' in figs:
                fname = "population_data.png"            
                print_verbose(verbose, f"Figure with population data was saved in {fname}.")
                figs['population_plot']['fig'].savefig(fname)
                print_verbose(verbose, "Source for this figure: ")
                print_verbose(verbose, figs['population_plot']['source'])
                
    
        #print(f"Time for forming request: {forming_request_time}")
        #print(f"Time for LLM request: {llm_request_time}")
        
    return output, input_params, content_message, combine_agent_prompt_text
