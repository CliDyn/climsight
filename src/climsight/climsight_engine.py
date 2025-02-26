"""
Engine for Climsight: This module contains functions that combine environmental and 
climate data based on latitude and longitude. It constructs prompts for language 
learning model (LLM) queries and handles the interaction with the LLM to generate 
responses based on the input data.

The main inputs include latitude, longitude, and a user_message. Data such as historical 
and future data (from climate models) can be provided with pre_data. By default, 
pre_data is an empty dictionary, and data will be loaded anew each time.
The config parameter is a configuration dictionary; if not provided, it will be 
loaded again from a YAML file.

"""
import yaml
import os
import logging
# import classes for climsight
from stream_handler import StreamHandler
import pandas as pd
from data_container import DataContainer

# import langchain functions
# from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
# import components for used by agent
from pydantic import BaseModel
#from typing import Annotated
from typing import Sequence
#import operator
#from langchain_core.messages import BaseMessage
from langgraph.graph import END, StateGraph, START

# import RAG components
from langchain_chroma import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

from rag import query_rag
from typing import Optional, Literal, Union, List

# import climsight classes
from climsight_classes import AgentState

# import smart_agent
from smart_agent import smart_agent

# import climsight functions
from geo_functions import (
   get_location,
   where_is_point,
   get_adress_string,
   get_location_details,
   closest_shore_distance,
   get_elevation_from_api,
   fetch_land_use,
   get_soil_from_api,
   is_point_onland,
   is_point_in_inlandwater
 )
from environmental_functions import (
   fetch_biodiversity,
   load_nat_haz_data,
   filter_events_within_square,
   plot_disaster_counts
)

from climate_functions import (
   load_data,
   extract_climate_data
)
from economic_functions import (
   get_population,
   plot_population,
   x_year_mean_population   
)
from extract_climatedata_functions import (
    request_climate_data,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
   filename='climsight.log',
   level=logging.INFO,
   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
   datefmt='%Y-%m-%d %H:%M:%S'
)

def normalize_longitude(lon):
    if lon < -180 or lon > 180:
        lon = (lon + 180) % 360 - 180
    return lon
 
def location_request(config, lat, lon):
    content_message = None
    input_params = None 

    # ----- Check input types ------------
    if not isinstance(lat, float) or not isinstance(lon, float):
        logging.error(f"lat and lon must be floats in clim_request(...) ")
        raise TypeError("lat and lon must be floats")
    # Config
    try:
        natural_e_path = config['natural_e_path']
    except KeyError as e:
        logging.error(f"Missing configuration key: {e}")
        raise RuntimeError(f"Missing configuration key: {e}")
       
    ##  =================== prepare data ==================
    logger.debug(f"is_point_onland : {lat}, {lon}")        
    try:
        is_on_land, water_body_status = is_point_onland(lat, lon, natural_e_path)
    except Exception as e:
        logging.error(f"Unexpected error in is_point_onland: {e}")
        raise RuntimeError(f"Unexpected error in is_point_onland: {e}")

    ######################## Here is a critical point ######################
    
    if not is_on_land:
        return content_message, input_params
    ######################## Here is a critical point ######################
    ##  ===== location
    logger.debug(f"Retrieving location from: {lat}, {lon}")        
    try:
        location = get_location(lat, lon)
    except Exception as e:
        logging.error(f"Unexpected error in get_location: {e}")
        raise RuntimeError(f"Unexpected error in get_location: {e}")
    ##  == adress 
    logger.debug(f"get_adress_string from: {lat}, {lon}")        
    try:
        location_str, location_str_for_print, country = get_adress_string(location)
    except Exception as e:
        logging.error(f"Unexpected error in get_adress_string: {e}")
        raise RuntimeError(f"Unexpected error in get_adress_string: {e}")

    logger.debug(f"is_point_onland from: {lat}, {lon}")            
    try:
        is_inland_water, water_body_status = is_point_in_inlandwater(lat, lon)
    except Exception as e:
        logging.error(f"Unexpected error in where_is_point: {e}")
        raise RuntimeError(f"Unexpected error in where_is_point: {e}")

    ##  == location details
    logger.debug(f"get_location_details")            
    try:
        add_properties = get_location_details(location)
    except Exception as e:
        logging.error(f"Unexpected error in get_location_details: {e}")
        raise RuntimeError(f"Unexpected error in get_location_details: {e}")
    
    content_message = """
        Location: latitude = {lat}, longitude = {lon} \n
        Adress: {location_str} \n
        Where is this point?: {water_body_status} \n
        Additional location information: {add_properties} \n
        """        
    input_params = {
        "lat": str(lat),
        "lon": str(lon),
        "location_str": location_str,
        "water_body_status": water_body_status,
        "add_properties": add_properties,
        "location_str_for_print": location_str_for_print,
        "is_inland_water": is_inland_water,
        "country": country
    }         
    return content_message, input_params


def forming_request(config, lat, lon, user_message, data={}, show_add_info=True):
    '''
    Inputs:
    - config (dict): Configuration 
    - lat (float): Latitude of the location to analyze.
    - lon (float): Longitude of the location to analyze.
    - user_message (string): Question for the LLM.
    - data (dict): Preloaded data, default is an empty dictionary.
    - show_add_info (bool): add additional info, here plot fiugures
    be aware that data could be modified  by this function
    
    Outputs:
    - several yields 
    - final return: content_message, input_params, df_data, figs, data
    
    How to call it in wrapers (strealit, terminal, ... )
        logger = logging.getLogger(__name__)
        logging.basicConfig( ...
        lat, lon, user_message = ...
        stream_handler = StreamHandler(...)

        generator = clim_request(lat, lon, user_message, stream_handler)

        while True:
        try:
            # Get the next intermediate result from the generator
            result = next(generator)
            print(f"Intermediate result: {result}")
        except StopIteration as e:
            # The generator is exhausted, and e.value contains the final result
            final_result = e.value
            print(f"Final result: {final_result}")
            break
    '''
    
    # ----- Check input types ------------
    if not isinstance(lat, float) or not isinstance(lon, float):
        logging.error(f"lat and lon must be floats in clim_request(...) ")
        raise TypeError("lat and lon must be floats")
   
    if not isinstance(user_message, str):
        logging.error(f"user_message must be a string in clim_request(...) ")
        raise TypeError("user_message must be a string")

    # Config
    try:
        data_path = config['data_settings']['data_path']
        coastline_shapefile = config['coastline_shapefile']
        haz_path = config['haz_path']
        pop_path = config['pop_path']
        distance_from_event = config['distance_from_event']
        year_step = config['year_step']
        start_year = config['start_year']
        end_year = config['end_year']
        natural_e_path = config['natural_e_path']
    except KeyError as e:
        logging.error(f"Missing configuration key: {e}")
        raise RuntimeError(f"Missing configuration key: {e}")
        
    #content_message defined below now

    ##  =================== prepare data ==================
    logger.debug(f"is_point_onland : {lat}, {lon}")        
    try:
        is_on_land, water_body_status = is_point_onland(lat, lon, natural_e_path)
    except Exception as e:
        logging.error(f"Unexpected error in is_point_onland: {e}")
        raise RuntimeError(f"Unexpected error in is_point_onland: {e}")

    ######################## Here is a critical point ######################
    if not is_on_land:
        return "Error: point_is_in_ocean"
    ######################## Here is a critical point ######################
        
    ##  ===== location
    logger.debug(f"Retrieving location from: {lat}, {lon}")        
    try:
        location = get_location(lat, lon)
    except Exception as e:
        logging.error(f"Unexpected error in get_location: {e}")
        raise RuntimeError(f"Unexpected error in get_location: {e}")
    ##  == adress 
    logger.debug(f"get_adress_string from: {lat}, {lon}")        
    try:
        location_str, location_str_for_print, country = get_adress_string(location)
    except Exception as e:
        logging.error(f"Unexpected error in get_adress_string: {e}")
        raise RuntimeError(f"Unexpected error in get_adress_string: {e}")

    yield f"**Coordinates:** {round(lat, 4)}, {round(lon, 4)}"
    ##  == wet / dry
    # logger.debug(f"where_is_point from: {lat}, {lon}")            
    # try:
    #     is_on_land, in_lake, lake_name, near_river, river_name, water_body_status = where_is_point(lat, lon)
    # except Exception as e:
    #     logging.error(f"Unexpected error in where_is_point: {e}")
    #     raise RuntimeError(f"Unexpected error in where_is_point: {e}")
    logger.debug(f"is_point_onland from: {lat}, {lon}")            
    try:
        is_inland_water, water_body_status = is_point_in_inlandwater(lat, lon)
    except Exception as e:
        logging.error(f"Unexpected error in where_is_point: {e}")
        raise RuntimeError(f"Unexpected error in where_is_point: {e}")

    
    
    ##  == location details
    logger.debug(f"get_location_details")            
    try:
        add_properties = get_location_details(location)
    except Exception as e:
        logging.error(f"Unexpected error in get_location_details: {e}")
        raise RuntimeError(f"Unexpected error in get_location_details: {e}")
    
    #if is_on_land:  We already have return if is_on_land
    if not is_inland_water:
        yield f"{location_str_for_print}"            
        pass
    if is_inland_water:
        yield f"{water_body_status} Our analyses are currently only meant for land areas. Please select another location for a better result."
        #yield f"You have choose {'lake ' + lake_name if lake_name else 'a lake'}. Our analyses are currently only meant for land areas. Please select another location for a better result."
        logging.info(f"location in inland water: water_body_status= {water_body_status}")
            
        #if near_river:
        #    yield f"You have choose on a place that might be in {'the river ' + river_name if river_name else 'a river'}. Our analyses are currently only meant for land areas. Please select another location for a better result."              
        #    logging.info(f"location in {'the river ' + river_name if river_name else 'a river'}")            
    # else:
    #     yield f"You have selected a place somewhere in the ocean. Our analyses are currently only meant for land areas. Please select another location for a better result."
    #     logging.info(f"place somewhere in the ocean")
    #     country = None
    #     location_str = None
    #     add_properties = None
    ##  == elevation
    logger.debug(f"get_elevation_from_api from: {lat}, {lon}")        
    try:
        elevation = get_elevation_from_api(lat, lon)
    except Exception as e:
        elevation = "Not known"
        logging.exception(f"elevation = Not known: {e}")
    ##  == land use        
    logger.debug(f"fetch_land_use from: {lat}, {lon}")        
    try:
        land_use_data = fetch_land_use(lon, lat)
    except Exception as e:
        land_use_data = "Not known"
        logging.exception(f"land_use_data = Not known: {e}")

    logger.debug(f"get current_land_use from land_use_data")              
    try:
        current_land_use = land_use_data["elements"][0]["tags"]["landuse"]
    except Exception as e:
        current_land_use = "Not known"
        logging.exception(f"{e}. Continue with: current_land_use = Not known")
    ##  == soil
    logger.debug(f"get current_land_use from land_use_data")              
    try:
        soil = get_soil_from_api(lat, lon)
    except Exception as e:
        soil = "Not known"
        logging.exception(f"soil = Not known: {e}")
    ##  == biodiversity
    logger.debug(f"fetch_biodiversity from: {round(lat), round(lon)}")              
    try:
        biodiv = fetch_biodiversity(round(lat), round(lon))
    except Exception as e:
        biodiv = "Not known"
        logging.error(f"Unexpected error in fetch_biodiversity: {e}")
    ##  == coast distance
    logger.debug(f"closest_shore_distance from: {lat, lon}")              
    try:
        distance_to_coastline = closest_shore_distance(lat, lon, coastline_shapefile)
    except Exception as e:
        distance_to_coastline = "Not known"
        logging.error(f"Unexpected error in closest_shore_distance: {e}")

    ## == hazards
    logger.debug(f"filter_events_within_square for: {lat, lon}")              
    try:
        filtered_events_square, promt_hazard_data = filter_events_within_square(lat, lon, haz_path, distance_from_event)
    except Exception as e:
        logging.error(f"Unexpected error in filter_events_within_square: {e}")
        raise RuntimeError(f"Unexpected error in filter_events_within_square: {e}")
    ## == population
    logger.debug(f"x_year_mean_population for: {pop_path, country}")              
    try:
        population = x_year_mean_population(pop_path, country, year_step=year_step, start_year=start_year, end_year=end_year)
    except Exception as e:
        logging.error(f"Unexpected error in filter_events_within_square: {e}")
        raise RuntimeError(f"Unexpected error in filter_events_within_square: {e}")

    ##  ===================  plotting      =========================   
    if show_add_info:
        figs = {}
        
        logger.debug(f"plot_disaster_counts for filtered_events_square")              
        try:
            haz_fig = plot_disaster_counts(filtered_events_square)
            source = '''
                        *The GDIS data descriptor*  
                        Rosvold, E.L., Buhaug, H. GDIS, a global dataset of geocoded disaster locations. Sci Data 8,
                        61 (2021). https://doi.org/10.1038/s41597-021-00846-6  
                        *The GDIS dataset*  
                        Rosvold, E. and H. Buhaug. 2021. Geocoded disaster (GDIS) dataset. Palisades, NY: NASA
                        Socioeconomic Data and Applications Center (SEDAC). https://doi.org/10.7927/zz3b-8y61.
                        Accessed DAY MONTH YEAR.  
                        *The EM-DAT dataset*  
                        Guha-Sapir, Debarati, Below, Regina, & Hoyois, Philippe (2014). EM-DAT: International
                        disaster database. Centre for Research on the Epidemiology of Disasters (CRED).
                    '''
            if not (haz_fig is None):
                figs['haz_fig'] = {'fig':haz_fig,'source':source}
        except Exception as e:
            logging.error(f"Unexpected error in plot_disaster_counts: {e}")
            raise RuntimeError(f"Unexpected error in plot_disaster_counts: {e}")

        logger.debug(f"plot_population for: {pop_path, country}")              
        try:
            population_plot = plot_population(pop_path, country)
            source = '''
                    United Nations, Department of Economic and Social Affairs, Population Division (2022). World Population Prospects 2022, Online Edition. 
                    Accessible at: https://population.un.org/wpp/Download/Standard/CSV/.
                    '''
            if not (population_plot is None):
                figs['population_plot'] = {'fig':population_plot,'source':source}        
        except Exception as e:
            logging.error(f"Unexpected error in population_plot: {e}")
            raise RuntimeError(f"Unexpected error in population_plot: {e}")       
        
    ##  =================== climate data 
    # data
    df_data = {}
    if config['use_high_resolution_climate_model']:
        try:
            data_agent_response, df_list = request_climate_data(config, lon, lat)
        except Exception as e:
            logging.error(f"Unexpected error in request_climate_data: {e}")
            raise RuntimeError(f"Unexpected error in request_climate_data: {e}")
        data['high_res_climate'] = {
        'df_list': df_list,
        'data_agent_response': data_agent_response,
        'lon': lon,
        'lat': lat
        }

    else:
        datakeys = list(data)
        if 'hist' and 'future' not in datakeys:
            logger.info(f"reading data from: {data_path}")        
            yield f"reading data"
            hist, future = load_data(config)
            data['hist'] = hist
            data['future'] = future
        else:
            logger.info(f"Data are preloaded in data dict")                
            hist, future = data['hist'], data['future']
                
        ## == create pandas dataframe
        logger.debug(f"extract_climate_data for: {lat, lon}")              
        try:
            df_data, data_dict = extract_climate_data(lat, lon, hist, future, config)
        except Exception as e:
            logging.error(f"Unexpected error in extract_climate_data: {e}")
            raise RuntimeError(f"Unexpected error in extract_climate_data: {e}")
        data_agent_response = {}
        data_agent_response['content_message'] = """ Current mean monthly temperature for each month: {hist_temp_str} \n
         Future monthly temperatures for each month at the location: {future_temp_str}\n
         Current precipitation flux (mm/month): {hist_pr_str} \n
         Future precipitation flux (mm/month): {future_pr_str} \n
         Current u wind component (in m/s): {hist_uas_str} \n
         Future u wind component (in m/s): {future_uas_str} \n
         Current v wind component (in m/s): {hist_vas_str} \n
         Future v wind component (in m/s): {future_vas_str} \n 
         """
        data_agent_response['input_params'] = {
        "hist_temp_str": data_dict["hist_Temperature"],
        "future_temp_str": data_dict["future_Temperature"],
        "hist_pr_str": data_dict["hist_Precipitation"],
        "future_pr_str": data_dict["future_Precipitation"],
        "hist_uas_str": data_dict["hist_u_wind"],
        "future_uas_str": data_dict["future_u_wind"],
        "hist_vas_str": data_dict["hist_v_wind"],
        "future_vas_str": data_dict["future_v_wind"]
        }

    
    ## == policy IS NOT IN USE
    policy = ""
    
    content_message = """Question from user: {user_message} \n \
        \n\n Additional information: \n \
        Location: latitude = {lat}, longitude = {lon} \n
        Adress: {location_str} \n
        Where is this point?: {water_body_status} \n
        Policy: {policy} \n
        Additional location information: {add_properties} \n
        Distance to the closest coastline: {distance_to_coastline} \n
        Elevation above sea level: {elevation} \n
        Current landuse: {current_land_use} \n
        Current soil type: {soil} \n
        Occuring species: {biodiv} \n
        Natural hazards: {nat_hazards} \n
        """  
    content_message += f"Population in {country} data: {{population}} \n"
    content_message += data_agent_response['content_message']          

    input_params = {
        "user_message": user_message,
        "lat": str(lat),
        "lon": str(lon),
        "location_str": location_str,
        "water_body_status": water_body_status,
        "add_properties": add_properties,
        "policy": policy,
        "distance_to_coastline": str(distance_to_coastline),
        "elevation": str(elevation),
        "current_land_use": current_land_use,
        "soil": soil,
        "biodiv": biodiv,
        "nat_hazards": promt_hazard_data,
        "population": population,
    }               
    input_params.update(data_agent_response['input_params'])
    
    return content_message, input_params, df_data, figs, data

def llm_request(content_message, input_params, config, api_key, stream_handler, ipcc_rag_ready, ipcc_rag_db, general_rag_ready, general_rag_db, data_pocket):
    """
    Handles LLM requests based on the mode specified in the configuration.

    Parameters:
    - content_message (str): The message content for the LLM.
    - input_params (dict): Input parameters for the LLM request.
    - config (dict): Configuration settings, including 'llmModeKey'.
    - api_key (str): API key for the LLM service.
    - stream_handler (StreamHandler): An instance of the StreamHandler class, used for streaming responses from the LLM.
    - rag_ready (bool): A flag indicating whether the RAG database is ready and available for queries.
    - rag_db (Chroma or None): The loaded RAG database object, used to retrieve relevant documents for the LLM prompt.    

    Returns:
    output (any): The output from the LLM request.

    Raises:
    TypeError: If 'llmModeKey' in the config is not recognized.
    """
    if config['llmModeKey'] == "direct_llm":
        output = direct_llm_request(content_message, input_params, config, api_key, stream_handler, ipcc_rag_ready, ipcc_rag_db, general_rag_ready, general_rag_db)
    elif config['llmModeKey'] == "agent_llm":
        output, input_params, content_message = agent_llm_request(content_message, input_params, config, api_key, stream_handler, ipcc_rag_ready, ipcc_rag_db, general_rag_ready, general_rag_db, data_pocket)
    else:
        logging.error(f"Wrong llmModeKey in config file: {config['llmModeKey']}")
        raise TypeError(f"Wrong llmModeKey in config file: {config['llmModeKey']}")
    return output, input_params, content_message

def direct_llm_request(content_message, input_params, config, api_key, stream_handler, ipcc_rag_ready, ipcc_rag_db, general_rag_ready, general_rag_db):
    """
    Sends a request to the LLM with optional RAG integration and returns the generated response.

    Args:
    - content_message (str): The message or prompt to be sent to the LLM.
    - input_params (dict): A dictionary of input parameters for the LLM, including information such as latitude, longitude, and other context.
    - config (dict): Configuration settings for the LLM and RAG.
    - api_key (str): The OpenAI API key used to authenticate the LLM request.
    - stream_handler (StreamHandler): An instance of the StreamHandler class, used for streaming responses from the LLM.
    - rag_ready (bool): A flag indicating whether the RAG database is ready and available for queries.
    - rag_db (Chroma or None): The loaded RAG database object, used to retrieve relevant documents for the LLM prompt.

    Returns:
    - str: The response generated by the LLM based on the input message and parameters.
    """

    if not isinstance(stream_handler, StreamHandler):
        logging.error(f"stream_handler must be an instance of StreamHandler")
        raise TypeError("stream_handler must be an instance of StreamHandler")    
    
    ##  ===================  start with LLM =========================
    logging.info(f"Generating...")    

    ## === RAG integration === ##
    ipcc_rag_response = query_rag(input_params, config, api_key, ipcc_rag_ready, ipcc_rag_db)
    general_rag_response = query_rag(input_params, config, api_key, general_rag_ready, general_rag_db)
    logger.debug(f"IPCC RAG is:", ipcc_rag_response)
    logger.debug(f"General RAG is:", general_rag_response)

    # Combine RAG responses with source labels
    rag_response_parts = []
    if ipcc_rag_response and ipcc_rag_response != "None":
        rag_response_parts.append(f"Source: IPCC RAG\n{ipcc_rag_response}")
    if general_rag_response and general_rag_response != "None":
        rag_response_parts.append(f"Source: General RAG\n{general_rag_response}")

    if rag_response_parts:
        rag_response = "\n\n".join(rag_response_parts)
    else:
        rag_response = None

    # Check if rag_response is valid and not the string "None"
    if rag_response and rag_response != "None":
        content_message += f"\n        RAG(text) response:\n {rag_response}"
        input_params['rag_response'] = rag_response
    else:
        # Log the absence of a valid RAG response
        logger.info("RAG response is None. Proceeding without RAG context.")

    logger.debug(f"start ChatOpenAI, LLMChain ")                 
    llm = ChatOpenAI(
        openai_api_key=api_key,
        model_name=config['model_name'],
        streaming=True,
        callbacks=[stream_handler],
    )
    if "o1" in config['model_name']:
        system_message_prompt = HumanMessagePromptTemplate.from_template(config['system_role'])
    else:
        system_message_prompt = SystemMessagePromptTemplate.from_template(config['system_role'])
    human_message_prompt = HumanMessagePromptTemplate.from_template(content_message)
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )
    chain = LLMChain(
        llm=llm,
        prompt=chat_prompt,
        output_key="review",
        verbose=True,
    )

    logger.info("Calling LLM with configured chain.")
    logger.debug(f"call  LLM, chain.run ")                 
    # Pass the input_params dictionary to chain.run() using the ** operator
    output = chain.run(**input_params, verbose=True)
    logger.info("LLM request completed successfully.")

    return output

def agent_llm_request(content_message, input_params, config, api_key, stream_handler, ipcc_rag_ready, ipcc_rag_db, general_rag_ready, general_rag_db, data_pocket):
    # function similar to llm_request but with agent structure
    # agent is consist of supervisor and nod that is responsible to call RAG
    # supervisor need to decide if call RAG or not

    if not isinstance(stream_handler, StreamHandler):
        logging.error(f"stream_handler must be an instance of StreamHandler")
        raise TypeError("stream_handler must be an instance of StreamHandler")
    
    lat = float(input_params['lat']) # should be already present in input_params
    lon = float(input_params['lon']) # should be already present in input_params
    
    logger.info(f"start agent_request")
    llm_intro = ChatOpenAI(
        openai_api_key=api_key,
        model_name=config['model_name_agents'],
    )    
    llm_combine_agent = ChatOpenAI(
        openai_api_key=api_key,
        model_name=config['model_name_combine_agent'],
    )    
        # streaming=True,
        # callbacks=[stream_handler],    
    '''
    class AgentState(BaseModel):
        messages: Annotated[Sequence[BaseMessage], operator.add]  #not in use up to now
        user: str = "" #user question
        next: str = "" #list of next actions
        ipcc_rag_agent_response: str = ""
        general_rag_agent_response: str = ""
        data_agent_response: dict = {}
        zero_agent_response: dict = {}
        final_answser: str = ""
        content_message: str = ""
        input_params: dict = {}
        smart_agent_response: dict = {}
        # stream_handler: StreamHandler
    '''
               
    def zero_rag_agent(state: AgentState, figs = {}):
      
        logger.debug(f"get_elevation_from_api from: {lat}, {lon}")        
        try:
            elevation = get_elevation_from_api(lat, lon)
        except Exception as e:
            elevation = None
            logging.exception(f"elevation = Not known: {e}")
        ##  == land use        
        logger.debug(f"fetch_land_use from: {lat}, {lon}")        
        try:
            land_use_data = fetch_land_use(lon, lat)
        except Exception as e:
            land_use_data = None
            logging.exception(f"land_use_data = None: {e}")

        logger.debug(f"get current_land_use from land_use_data")              
        try:
            current_land_use = land_use_data["elements"][0]["tags"]["landuse"]
        except Exception as e:
            current_land_use = None
            logging.exception(f"{e}. Continue with: current_land_use = None")
        ##  == soil
        logger.debug(f"get current_land_use from land_use_data")              
        try:
            soil = get_soil_from_api(lat, lon)
        except Exception as e:
            soil = None
            logging.exception(f"soil = None: {e}")
        ##  == biodiversity
        logger.debug(f"fetch_biodiversity from: {round(lat), round(lon)}")              
        try:
            biodiv = fetch_biodiversity(round(lat), round(lon))
        except Exception as e:
            biodiv = None
            logging.error(f"Unexpected error in fetch_biodiversity: {e}")
        ##  == coast distance
        logger.debug(f"closest_shore_distance from: {lat, lon}")              
        try:
            distance_to_coastline = closest_shore_distance(lat, lon, config['coastline_shapefile'])
        except Exception as e:
            distance_to_coastline = None
            logging.error(f"Unexpected error in closest_shore_distance: {e}")

        ## == hazards
        logger.debug(f"filter_events_within_square for: {lat, lon}")              
        try:
            filtered_events_square, promt_hazard_data = filter_events_within_square(lat, lon, config['haz_path'], config['distance_from_event'])
        except Exception as e:
            promt_hazard_data = None
            filtered_events_square = None
            logging.error(f"Unexpected error in filter_events_within_square: {e}")
            raise RuntimeError(f"Unexpected error in filter_events_within_square: {e}")
              ## !!!!!!!!! raise should be changed to logging (add exceptions for ploting below)  
        ## == population
        logger.debug(f"x_year_mean_population for: {config['pop_path'], state.input_params['country']}")              
        try:
            population = x_year_mean_population(config['pop_path'], state.input_params['country'], 
                                                year_step=config['year_step'], start_year=config['start_year'], end_year=config['end_year'])
        except Exception as e:
            population = None
            logging.error(f"Unexpected error in filter_events_within_square: {e}")
            raise RuntimeError(f"Unexpected error in filter_events_within_square: {e}")
            ## !!!!!!!!! raise should be changed to logging (add exceptions for ploting below)  

        
        zero_agent_response = {}
        zero_agent_response['input_params'] = {}
        zero_agent_response['content_message'] = ""
        if elevation:
            zero_agent_response['input_params']['elevation'] = str(elevation)
            zero_agent_response['content_message'] += "Elevation above sea level: {elevation} \n"
        if current_land_use:
            zero_agent_response['input_params']['current_land_use'] = current_land_use
            zero_agent_response['content_message'] += "Current landuse: {current_land_use} \n"  
        if soil:
            zero_agent_response['input_params']['soil'] = soil
            zero_agent_response['content_message'] += "Current soil type: {soil} \n"        
        if biodiv:
            zero_agent_response['input_params']['biodiv'] = biodiv
            zero_agent_response['content_message'] += "Occuring species: {biodiv} \n"                    
        if distance_to_coastline:
            zero_agent_response['input_params']['distance_to_coastline'] = str(distance_to_coastline)
            zero_agent_response['content_message'] += "Distance to the closest coastline: {distance_to_coastline} \n"     
        if promt_hazard_data is not None:
            zero_agent_response['input_params']['nat_hazards'] = promt_hazard_data
            zero_agent_response['content_message'] += "Natural hazards: {nat_hazards} \n"    
        if population is not None:
            zero_agent_response['input_params']['population'] = population
            zero_agent_response['content_message'] += f"Population in {state.input_params['country']} data: {{population}} \n"

        ##  ===================  plotting      =========================   
        if config['show_add_info']:
            logger.debug(f"plot_disaster_counts for filtered_events_square")              
            try:
                haz_fig = plot_disaster_counts(filtered_events_square)
                source = '''
                            *The GDIS data descriptor*  
                            Rosvold, E.L., Buhaug, H. GDIS, a global dataset of geocoded disaster locations. Sci Data 8,
                            61 (2021). https://doi.org/10.1038/s41597-021-00846-6  
                            *The GDIS dataset*  
                            Rosvold, E. and H. Buhaug. 2021. Geocoded disaster (GDIS) dataset. Palisades, NY: NASA
                            Socioeconomic Data and Applications Center (SEDAC). https://doi.org/10.7927/zz3b-8y61.
                            Accessed DAY MONTH YEAR.  
                            *The EM-DAT dataset*  
                            Guha-Sapir, Debarati, Below, Regina, & Hoyois, Philippe (2014). EM-DAT: International
                            disaster database. Centre for Research on the Epidemiology of Disasters (CRED).
                        '''
                if not (haz_fig is None):
                    figs['haz_fig'] = {'fig':haz_fig,'source':source}
            except Exception as e:
                logging.error(f"Unexpected error in plot_disaster_counts: {e}")
                raise RuntimeError(f"Unexpected error in plot_disaster_counts: {e}")

            logger.debug(f"plot_population for: {config['pop_path'], state.input_params['country'] }")              
            try:
                population_plot = plot_population(config['pop_path'], state.input_params['country'] )
                source = '''
                        United Nations, Department of Economic and Social Affairs, Population Division (2022). World Population Prospects 2022, Online Edition. 
                        Accessible at: https://population.un.org/wpp/Download/Standard/CSV/.
                        '''
                if not (population_plot is None):
                    figs['population_plot'] = {'fig':population_plot,'source':source}        
            except Exception as e:
                logging.error(f"Unexpected error in population_plot: {e}")
                raise RuntimeError(f"Unexpected error in population_plot: {e}")
  
        logger.info(f"zero_agent_response: {zero_agent_response}")
        return {'zero_agent_response': zero_agent_response}
                    
    def data_agent(state: AgentState, data={}, df={}):
        # data
        # config, lat, lon  -  from the outer function (agent_clim_request(config,...))
        df_list = []
        if config['use_high_resolution_climate_model']:
            try:
                data_agent_response, df_list = request_climate_data(config, lon, lat)
            except Exception as e:
                logging.error(f"Unexpected error in request_climate_data: {e}")
                raise RuntimeError(f"Unexpected error in request_climate_data: {e}")
            data['high_res_climate'] = {
            'df_list': df_list,
            'data_agent_response': data_agent_response,
            'lon': lon,
            'lat': lat
            }    
            state.df_list = df_list
        else:    
            datakeys = list(data)
            if 'hist' and 'future' not in datakeys:
                logger.info(f"reading data from: {config['data_settings']['data_path']}")        
                data['hist'], data['future'] = load_data(config)
            else:
                logger.info(f"Data are preloaded in data dict")                

            ## == create pandas dataframe
            logger.debug(f"extract_climate_data for: {lat, lon}")              
            try:
                df_data_local, data_dict = extract_climate_data(lat, lon, data['hist'], data['future'], config)
                df['df_data'] = df_data_local
            except Exception as e:
                logging.error(f"Unexpected error in extract_climate_data: {e}")
                raise RuntimeError(f"Unexpected error in extract_climate_data: {e}")        
            
            data_agent_response = {}
            data_agent_response['input_params'] = {}
            data_agent_response['content_message'] = ""

            data_agent_response['input_params']["hist_temp_str"]   = data_dict["hist_Temperature"],
            data_agent_response['input_params']["future_temp_str"] = data_dict["future_Temperature"],
            data_agent_response['input_params']["hist_pr_str"]     = data_dict["hist_Precipitation"],
            data_agent_response['input_params']["future_pr_str"]   = data_dict["future_Precipitation"],
            data_agent_response['input_params']["hist_uas_str"]    = data_dict["hist_u_wind"],
            data_agent_response['input_params']["future_uas_str"]  = data_dict["future_u_wind"],
            data_agent_response['input_params']["hist_vas_str"]    = data_dict["hist_v_wind"],
            data_agent_response['input_params']["future_vas_str"]  = data_dict["future_v_wind"],       
            data_agent_response['content_message'] += """\n
            Current mean monthly temperature for each month: {hist_temp_str} \n
            Future monthly temperatures for each month at the location: {future_temp_str}\n
            Current precipitation flux (mm/month): {hist_pr_str} \n
            Future precipitation flux (mm/month): {future_pr_str} \n
            Current u wind component (in m/s): {hist_uas_str} \n
            Future u wind component (in m/s): {future_uas_str} \n
            Current v wind component (in m/s): {hist_vas_str} \n
            Future v wind component (in m/s): {future_vas_str} \n """
        
        logger.info(f"Data agent in work.")
        
        respond = {'data_agent_response': data_agent_response, 'df_list': df_list}
    
        logger.info(f"data_agent_response: {data_agent_response}")
        return respond

      
    def ipcc_rag_agent(state: AgentState):
        ## === RAG integration === ##
        logger.info(f"IPCC RAG agent in work.")
        ipcc_rag_response = query_rag(input_params, config, api_key, ipcc_rag_ready, ipcc_rag_db)
        # logger.info(f"IPCC RAG says: {ipcc_rag_response}")
        logger.info(f"ipcc_rag_agent_response: {ipcc_rag_response}")
        return {'ipcc_rag_agent_response': ipcc_rag_response}

    def general_rag_agent(state: AgentState):
        ## === RAG integration === ##
        logger.info(f"General RAG agent in work.")
        general_rag_response = query_rag(input_params, config, api_key, general_rag_ready, general_rag_db)
        # logger.info(f"General RAG says: {general_rag_response}")
        logger.info(f"general_rag_agent_response: {general_rag_response}")
        return {'general_rag_agent_response': general_rag_response}
    
################# start of intro_agent #############################
    def intro_agent(state: AgentState):
        intro_message = """ 
        You are the introductory interface for a system named ClimSight, designed to help individuals evaluate the impact of climate change
        on current decision-making (e.g., installing wind turbines, solar panels, constructing buildings, creating parking lots, 
        opening a shop, or purchasing cropland). ClimSight operates on a local scale, providing data-driven insights specific to particular
        locations and aiding in climate-informed decision-making.

        ClimSight answers questions regarding the impacts of climate change on planned activities,
        using high-resolution climate data combined with an LLM to deliver actionable, location-specific information.
        This approach supports local decisions effectively, removing scalability and expertise limitations.

        Your task is to assess the potential climate-related risks and/or benefits associated with the user's planned activities.
        Additionally, use information about the user's country to retrieve relevant policies and regulations regarding climate change,
        environmental usage, and the specific activity the user has requested.

        **What you should do now:**

        At this stage, perform a quick pre-analysis of the user's question and decide on one of the following actions:

        1. **FINISH:** If the question is unrelated to ClimSight's purpose or is a simple inquiry outside your primary objectives,
        you can choose to finish the conversation by selecting FINISH and providing a concise answer. Examples of unrelated or simple questions:
        - “Hi”
        - “How are you?”
        - “Who are you?”
        - “Write an essay on the history of trains.”
        - “Translate some text for me.”

        2. **CONTINUE:** For all other cases, if the question relates to climate or location, select CONTINUE to proceed,
        which will prompt other agents to address the user's question. Note that the specific location may not be mentioned in the user's initial question, 
        but it will be clarified by subsequent agents.

        Based on the conversation, decide on one of the following responses:
        - **FINISH**: Provide a final answer to end the conversation.
        - **CONTINUE**: Indicate that the process should proceed without a final answer at this stage.

        Given this guidance, respond with either "FINISH" and the final answer, or "CONTINUE."
        """        
        intro_options = ["FINISH", "CONTINUE"]
        intro_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", intro_message),
            ("user", "{user_text}"),
        ])            
        class routeResponse(BaseModel):
            next: Literal["FINISH", "CONTINUE"]  # Accepts single value only
            final_answer: str = ""  
              
        chain = (
             intro_prompt
             | llm_intro.with_structured_output(routeResponse, method="function_calling")
         )
        # Pass the dictionary to invoke
        input = {"user_text": state.user}
        response = chain.invoke(input)
        state.final_answser = response.final_answer
        state.next = response.next
        return state
    
    
################# end of intro_agent #############################
    def combine_agent(state: AgentState): 
        logger.info('combine_agent in work')
        
        #add IPCC RAG response to content_message and input_params
        if state.ipcc_rag_agent_response != "None" and state.ipcc_rag_agent_response != "":
            state.content_message += "\n        RAG(text) response: {ipcc_rag_response} "
            state.input_params['ipcc_rag_response'] = state.ipcc_rag_agent_response

        #add general RAG response to content_message and input_params
        if state.general_rag_agent_response != "None" and state.general_rag_agent_response != "":
            state.content_message += "\n        RAG(text) response: {general_rag_response} "
            state.input_params['general_rag_response'] = state.general_rag_agent_response

        #add zero_agent response to content_message and input_params                    
        if state.zero_agent_response != {}:
            state.content_message += state.zero_agent_response['content_message']
            state.input_params.update(state.zero_agent_response['input_params'])    
        
        #add data_agent response to content_message and input_params                    
        if state.data_agent_response != {}:
            state.content_message += state.data_agent_response['content_message']
            state.input_params.update(state.data_agent_response['input_params']) 

        if state.smart_agent_response != {}:
            smart_analysis = state.smart_agent_response.get('output', '')
            state.input_params['smart_agent_analysis'] = smart_analysis
            state.content_message += "\n Smart Data Extractor Agent Analysis: {smart_agent_analysis} "
            logger.info(f"smart_agent_response: {state.smart_agent_response}")

            # Add Wikipedia tool response
        if state.wikipedia_tool_response != {}:
            wiki_response = state.wikipedia_tool_response
            state.input_params['wikipedia_tool_response'] = wiki_response
            state.content_message += "\n Wikipedia Search Response: {wikipedia_tool_response} "
            logger.info(f"Wikipedia_tool_reponse: {state.wikipedia_tool_response}")
        if state.ecocrop_search_response != {}:
            ecocrop_response = state.ecocrop_search_response
            state.input_params['ecocrop_search_response'] = ecocrop_response
            state.content_message += "\n ECOCROP Search Response: {ecocrop_search_response} "
            logger.info(f"Ecocrop_search_response: {state.ecocrop_search_response}")
      
                   
        if "o1" in config['model_name_combine_agent']:
            system_message_prompt = HumanMessagePromptTemplate.from_template(config['system_role'])
        else:
            system_message_prompt = SystemMessagePromptTemplate.from_template(config['system_role'])
        human_message_prompt = HumanMessagePromptTemplate.from_template(state.content_message)
        chat_prompt = ChatPromptTemplate.from_messages(
            [system_message_prompt, human_message_prompt]
        )
        chain = (
            chat_prompt
            | llm_combine_agent
        )
        output = chain.invoke(state.input_params)
        logger.info(f"Final_answer: {output.content}")
        return {'final_answser': output.content, 'input_params': state.input_params, 'content_message': state.content_message}
    
    def route_fromintro(state: AgentState) -> Sequence[str]:
        output = []
        if "FINISH" in state.next:
            return "FINISH"
        else:
            output.append("ipcc_rag_agent")
            output.append("general_rag_agent")
            output.append("data_agent")
            output.append("zero_rag_agent")
            #output.append("smart_agent")
        return output
    def route_fromdata(state: AgentState) -> Sequence[str]:
        output = []
        if config['use_smart_agent']:
            output.append("smart_agent")
        else:
            output.append("combine_agent")
        return output
        
    workflow = StateGraph(AgentState)

    figs = data_pocket.figs
    data = data_pocket.data
    df = data_pocket.df
    
     # Add nodes to the graph
    workflow.add_node("intro_agent", intro_agent)
    workflow.add_node("ipcc_rag_agent", ipcc_rag_agent)
    workflow.add_node("general_rag_agent", general_rag_agent)
    workflow.add_node("data_agent", lambda s: data_agent(s, data, df))  # Pass `data` as argument
    workflow.add_node("zero_rag_agent", lambda s: zero_rag_agent(s, figs))  # Pass `figs` as argument    
    workflow.add_node("smart_agent", lambda s: smart_agent(s, config, api_key))
    workflow.add_node("combine_agent", combine_agent)   

    path_map = {'ipcc_rag_agent':'ipcc_rag_agent', 'general_rag_agent':'general_rag_agent', 'data_agent':'data_agent','zero_rag_agent':'zero_rag_agent','FINISH':END}
    path_map_data = {'combine_agent':'combine_agent', 'smart_agent':'smart_agent'}    

    workflow.set_entry_point("intro_agent") # Set the entry point of the graph
    
    workflow.add_conditional_edges("intro_agent", route_fromintro, path_map=path_map)
    workflow.add_conditional_edges("data_agent", route_fromdata, path_map=path_map_data)    

    #if config['use_smart_agent']:
    #    workflow.add_edge(["ipcc_rag_agent","general_rag_agent","data_agent","zero_rag_agent"], "combine_agent")
    #else:
    workflow.add_edge(["ipcc_rag_agent","general_rag_agent","smart_agent","zero_rag_agent"], "combine_agent")
        
    #workflow.add_edge("ipcc_rag_agent", "combine_agent")
    #workflow.add_edge("general_rag_agent", "combine_agent")
    #workflow.add_edge("data_agent", "combine_agent")
    #workflow.add_edge("zero_rag_agent", "combine_agent")
    #workflow.add_edge("smart_agent", "combine_agent")
    workflow.add_edge("combine_agent", END)
    # Compile the graph
    app = workflow.compile()
    
    # from IPython.display import Image, display
    # graph_image_path = 'graph_image.png'  # Specify the desired path for the image
    # graph_img= app.get_graph().draw_mermaid_png()
    # with open(graph_image_path, 'wb') as f:
    #     f.write(graph_img)  # Write the image bytes to the file
    
    state = AgentState(messages=[], input_params=input_params, user=input_params['user_message'], content_message=content_message)
    
    output = app.invoke(state)

    input_params = output['input_params']
    content_message = output['content_message']
    
    stream_handler.send_text(output['final_answser'])
   
    return output['final_answser'], input_params, content_message
