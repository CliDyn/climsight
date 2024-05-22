from streamlit_interface import run_streamlit
import logging
import yaml
import os

#Initialize logging at the beginning of your main application
logger = logging.getLogger(__name__)
logging.basicConfig(
   filename='climsight.log',
   level=logging.INFO,
   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
   datefmt='%Y-%m-%d %H:%M:%S'
)

skip_llm_call = False

config = {}
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
   climatemodel_name = config['climatemodel_name']
   data_path = config['data_settings']['data_path']
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



run_streamlit(config, skip_llm_call=skip_llm_call)