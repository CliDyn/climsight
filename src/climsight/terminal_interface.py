"""
    Terminal Wrapper Module 
"""
#general 
import logging
import yaml
import os
import matplotlib.pyplot as plt
import time

# climsight modules
from stream_handler import StreamHandler
from climsight_engine import llm_request, forming_request

logger = logging.getLogger(__name__)

def input_with_default(prompt, default_value):
    user_input = input(prompt)
    if user_input == "":
        return default_value
    return user_input

def print_verbose(verbose, message):
    if verbose:
        print(message)  

def run_terminal(config, api_key='', skip_llm_call=False, lon=None, lat=None, user_message='', show_add_info='',verbose=True):
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
        Output:
        - output (string): Output from the LLM.
    '''      
    # Config
    try:
        climatemodel_name = config['climatemodel_name']
        lat_default = config['lat_default']
        lon_default = config['lon_default']
    except KeyError as e:
        logging.error(f"Missing configuration key: {e}")
        raise RuntimeError(f"Missing configuration key: {e}")   

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

    if not isinstance(api_key, str):
        logging.error(f"api_key must be a string ")
        raise TypeError("api_key must be a string")
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY") # check if OPENAI_API_KEY is set in the environment
    if (not api_key) and (not skip_llm_call):
        api_key = input("Please provide openAI API key: ")
    else:
        print_verbose(verbose, "openAI API key accepted.")

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

    print_verbose(verbose, "")
    print_verbose(verbose, "Getting info on a point...")
    
    # Record the start time
    start_time = time.time()
    
    is_on_land = True
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
            break     
    # Record the start time
    forming_request_time = time.time() - start_time
        
    if is_on_land:        
        start_time = time.time()
        stream_handler = StreamHandler()
        output = ''
        if not skip_llm_call:
            output = llm_request(content_message, input_params, config, api_key, stream_handler)   
                
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
            print_verbose(verbose, f"**Coordinates:** {input_params['lat']}, {input_params['lon']}")
            print_verbose(verbose, f"**Elevation:** {input_params['elevation']} m")
            print_verbose(verbose, f"**Current land use:** {input_params['current_land_use']}")
            print_verbose(verbose, f"**Soil type:** {input_params['soil']}")
            print_verbose(verbose, f"**Occuring species:** {input_params['biodiv']}")
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

            # Natural Hazards
            if 'haz_fig' in figs:
                fname = "natural_hazards.png"
                print_verbose(verbose, "Figure with natural hazards was saved in {fname}.")
                figs['haz_fig']['fig'].savefig(fname)
                print_verbose(verbose, "Source for this figure: ")
                print_verbose(verbose, figs['haz_fig']['source'])
                print_verbose(verbose, "\n")    
            # Population Data
            if 'population_plot' in figs:
                fname = "population_data.png"            
                print_verbose(verbose, "Figure with population data was saved in {fname}.")
                figs['population_plot']['fig'].savefig(fname)
                print_verbose(verbose, "Source for this figure: ")
                print_verbose(verbose, figs['population_plot']['source'])
                
    
        #print(f"Time for forming request: {forming_request_time}")
        #print(f"Time for LLM request: {llm_request_time}")
        
    return output