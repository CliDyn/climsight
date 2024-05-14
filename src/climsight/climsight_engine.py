"""
Engine for Climsight: This module contains functions that combine environmental and 
climate data based on latitude and longitude. It constructs prompts for language 
learning model (LLM) queries and handles the interaction with the LLM to generate 
responses based on the input data.

The main inputs include latitude, longitude, and a question. Data such as historical 
and future data (from climate models) can be provided with pre_data. By default, 
pre_data is an empty dictionary, and data will be loaded anew each time.
The config parameter is a configuration dictionary; if not provided, it will be 
loaded again from a YAML file.

"""
import yaml
import os
import logging

from geo_functions import (
   get_location,
   where_is_point,
   get_adress_string,
   get_location_details,
   closest_shore_distance,
   get_elevation_from_api,
   fetch_land_use,
   get_soil_from_api
 )
# from economic_functions import (
#    get_population,
#    plot_population,
#    x_year_mean_population   
# )

from climate_functions import (
   load_data,
   extract_climate_data
)
# from environmental_functions import (
#    fetch_biodiversity,
#    load_nat_haz_data,
#    filter_events_within_square,
#    plot_disaster_counts
# )

logger = logging.getLogger(__name__)



def clim_request(lat, lon, question, data={}, config={}, api_key='', skip_llm_call=False):
    '''
    Inputs:
    - lat (float): Latitude of the location to analyze.
    - lon (float): Longitude of the location to analyze.
    - question (string): Question for the LLM.
    - pre_data (dict): Preloaded data, default is an empty dictionary.
    - config (dict): Configuration, default is an empty dictionary.
    - api_key (string): API Key, default is an empty string.
    - skip_llm_call (bool): If True - skipp final call to LLM
    Outputs:
    - The LLM's response.
    '''
    # Config
    if not config:
        config_path = os.getenv('CONFIG_PATH', 'config.yml')
        logger.info(f"reading config from: {config_path}")
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
        except Exception as e:
            logging.error(f"An error occurred while reading the file: {config_path}")
            raise RuntimeError(f"An error occurred while reading the file: {config_path}") from e
    try:
        model_name = config['model_name']
        data_path = config['data_path']
        coastline_shapefile = config['coastline_shapefile']
        haz_path = config['haz_path']
        pop_path = config['pop_path']
        distance_from_event = config['distance_from_event']
        lat_default = config['lat_default']
        lon_default = config['lon_default']
        year_step = config['year_step']
        start_year = config['start_year']
        end_year = config['end_year']
        system_role = config['system_role']
    except KeyError as e:
        logging.error(f"Missing configuration key: {e}")
        raise RuntimeError(f"Missing configuration key: {e}")
        
    # data
    datakeys = list(data)
    if 'hist' and 'future' not in datakeys:
        logger.info(f"reading data from: {data_path}")        
        hist, future = load_data(data_path)
        data['hist'] = hist
        data['future'] = future
    else:
        logger.info(f"Data are preloaded in data dict")                
        hist, future = data['hist'], data['future']

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


    logger.info(f"Retrieving location from: {lat}, {lon}")        
    try:
        location = get_location(lat, lon)
    except Exception as e:
        logging.error(f"Unexpected error in get_location: {e}")
        raise RuntimeError(f"Unexpected error in get_location: {e}")

    logger.info(f"get_adress_string from: {lat}, {lon}")        
    try:
        location_str, location_str_for_print, country = get_adress_string(location)
    except Exception as e:
        logging.error(f"Unexpected error in get_adress_string: {e}")
        raise RuntimeError(f"Unexpected error in get_adress_string: {e}")

    yield f"**Coordinates:** {round(lat, 4)}, {round(lon, 4)}"

    logger.info(f"where_is_point from: {lat}, {lon}")            
    try:
        is_on_land, in_lake, lake_name, near_river, river_name, water_body_status = where_is_point(lat, lon)
    except:
        logging.error(f"Unexpected error in where_is_point: {e}")
        raise RuntimeError(f"Unexpected error in where_is_point: {e}")

    logger.info(f"get_location_details")            
    try:
        add_properties = get_location_details(location)
    except:
        logging.error(f"Unexpected error in get_location_details: {e}")
        raise RuntimeError(f"Unexpected error in get_location_details: {e}")
    
    if is_on_land:
        if not in_lake or not near_river:
            yield f"{location_str_for_print}"            
            pass
        if in_lake:
            #yield f"You have clicked on {'lake ' + lake_name if lake_name else 'a lake'}. Our analyses are currently only meant for land areas. Please select another location for a better result."
            yield f"You have choose {'lake ' + lake_name if lake_name else 'a lake'}. Our analyses are currently only meant for land areas. Please select another location for a better result."
        if near_river:
            yield f"You have choose on a place that might be in {'the river ' + river_name if river_name else 'a river'}. Our analyses are currently only meant for land areas. Please select another location for a better result."              
    else:
        yield f"You have selected a place somewhere in the ocean. Our analyses are currently only meant for land areas. Please select another location for a better result."
        country = None
        location_str = None
        add_properties = None

    return

## ADD to MAin
# Initialize logging at the beginning of your main application
#logging.basicConfig(
#    level=logging.INFO,
#    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#    datefmt='%Y-%m-%d %H:%M:%S'
#)

#   for result in func2():
#        print(result)
#        # Additional processing can be done here if needed

# def func1():
#     # Create a generator object by calling func2
#     generator = func2()
    
#     while True:
#         try:
#             # Get the next intermediate result from the generator
#             result = next(generator)
#             print(f"Intermediate result: {result}")
#         except StopIteration as e:
#             # The generator is exhausted, and e.value contains the final result
#             final_result = e.value
#             print(f"Final result: {final_result}")
#             break

# # Run func1
# func1()
