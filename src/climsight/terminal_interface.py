"""
    Terminal Wrapper Module 
"""
#general 
import logging
import yaml
import os

# climsight modules
from stream_handler import StreamHandler
from climsight_engine import llm_request, forming_request

logger = logging.getLogger(__name__)

def run_terminal(config, api_key='', skip_llm_call=False, lon=None, lat=None, user_message=''):
    '''
        Inputs:
        - config (dict): Configuration, default is an empty dictionary.   
        - api_key (string): API Key, default is an empty string. default ''
        - skip_llm_call (bool): If True - skipp final call to LLM. default False    
        - lat (float): Latitude of the location to analyze. default None
        - lon (float): Longitude of the location to analyze. default None
        - user_message (string): Question for the LLM. default empty ''
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
    
    if not isinstance(api_key, str):
        logging.error(f"api_key must be a string")
        raise TypeError("api_key must be a string")
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY") # check if OPENAI_API_KEY is set in the environment

#############################3 conbtiune HERE
    print(f"Welcome to Climsight!")
    if lon is None:
        user_message = input(f"Describe the activity that you would like to evaluate:\n")    
    user_message = input(f"Describe the activity that you would like to evaluate:\n")

    print(f"Welcome to CLimsight!")




    if not isinstance(lat, float) or not isinstance(lon, float):
        logging.error(f"lat and lon must be floats in clim_request(...) ")
        raise TypeError("lat and lon must be floats")
               
    if not lon:
        lon = lon_default
    if not lat:
        lat = lat_default

    return