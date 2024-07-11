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

# import langchaion functions
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
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

logger = logging.getLogger(__name__)



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
        
    # data
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

    ##  =================== climate data 
    ## == create pandas dataframe
    logger.debug(f"extract_climate_data for: {lat, lon}")              
    try:
        df_data, data_dict = extract_climate_data(lat, lon, hist, future, config)
    except Exception as e:
        logging.error(f"Unexpected error in extract_climate_data: {e}")
        raise RuntimeError(f"Unexpected error in extract_climate_data: {e}")
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
        
        

    ## == policy IS NOT IN USE
    policy = ""
    
    content_message = "{user_message} \n \
        Location: latitude = {lat}, longitude = {lon} \
        Adress: {location_str} \
        Where is this point?: {water_body_status} \
        Policy: {policy} \
        Additional location information: {add_properties} \
        Distance to the closest coastline: {distance_to_coastline} \
        Elevation above sea level: {elevation} \
        Current landuse: {current_land_use} \
        Current soil type: {soil} \
        Occuring species: {biodiv} \
        Current mean monthly temperature for each month: {hist_temp_str} \
        Future monthly temperatures for each month at the location: {future_temp_str}\
        Current precipitation flux (mm/month): {hist_pr_str} \
        Future precipitation flux (mm/month): {future_pr_str} \
        Current u wind component (in m/s): {hist_uas_str} \
        Future u wind component (in m/s): {future_uas_str} \
        Current v wind component (in m/s): {hist_vas_str} \
        Future v wind component (in m/s): {future_vas_str} \
        Natural hazards: {nat_hazards} \
        Population data: {population} \
        "        
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
        "hist_temp_str": data_dict["hist_Temperature"],
        "future_temp_str": data_dict["future_Temperature"],
        "hist_pr_str": data_dict["hist_Precipitation"],
        "future_pr_str": data_dict["future_Precipitation"],
        "hist_uas_str": data_dict["hist_u_wind"],
        "future_uas_str": data_dict["future_u_wind"],
        "hist_vas_str": data_dict["hist_v_wind"],
        "future_vas_str": data_dict["future_v_wind"],
        "nat_hazards": promt_hazard_data,
        "population": population,
    }               
    return content_message, input_params, df_data, figs, data

def llm_request(content_message, input_params, config, api_key, stream_handler):

    if not isinstance(stream_handler, StreamHandler):
        logging.error(f"stream_handler must be an instance of StreamHandler")
        raise TypeError("stream_handler must be an instance of StreamHandler")    
    
    ##  ===================  start with LLM =========================
    #yield f"Generating..."
    logging.info(f"Generating...")    

    logger.debug(f"start ChatOpenAI, LLMChain ")                 
    llm = ChatOpenAI(
        openai_api_key=api_key,
        model_name=config['model_name'],
        streaming=True,
        callbacks=[stream_handler],
    )
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

    logger.debug(f"call  LLM, chain.run ")                 
    # Pass the input_params dictionary to chain.run() using the ** operator
    output = chain.run(**input_params, verbose=True)

    return output