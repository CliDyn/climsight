import logging
import yaml
import os
import sys

from streamlit_interface import run_streamlit
from terminal_interface import run_terminal

#Initialize logging at the beginning of your main application
logger = logging.getLogger(__name__)
logging.basicConfig(
   filename='climsight.log',
   level=logging.INFO,
   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
   datefmt='%Y-%m-%d %H:%M:%S'
)
# Check arguments
skip_llm_call = 'skipLLMCall' in sys.argv
if skip_llm_call:
   logger.info(f"skipLLMCall is in arguments, no call to LLM would be performed")
terminal_call = 'terminal' in sys.argv
if terminal_call:
   logger.info(f"terminal is in arguments, run without Streamlit")   

config = {}
# reading configuration file
if not config:
   config_path = os.getenv('CONFIG_PATH', 'config.yml')
   logger.info(f"reading config from: {config_path}")
   try:
      with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
   except Exception as e:
      logging.error(f"An error occurred while reading the file: {config_path}")
      raise RuntimeError(f"An error occurred while reading the file: {config_path}") from e
# preliminary check config file   
try:
   model_name = config['model_name']
   climatemodel_name = config['climatemodel_name']
   llmModeKey = config['llmModeKey'] 
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
   rag_settings = config['rag_settings']
   embedding_model = rag_settings['embedding_model']
   chroma_path_ipcc = rag_settings['chroma_path_ipcc']
   chroma_path_general = rag_settings['chroma_path_general']
   document_path = rag_settings['document_path']
   chunk_size = rag_settings['chunk_size']
   chunk_overlap = rag_settings['chunk_overlap']
   separators = rag_settings['separators']
   rag_activated = rag_settings['rag_activated']
   rag_template = config['rag_template']
except KeyError as e:
   logging.error(f"Missing configuration key: {e}")
   raise RuntimeError(f"Missing configuration key: {e}")

chroma_path = [chroma_path_ipcc, chroma_path_general]

references = {}
# reading references file
if not references:
   references_path = os.getenv('CONFIG_PATH', 'references.yml')
   logger.info(f"reading references from: {config_path}")
   try:
      with open(references_path, 'r') as file:
            references = yaml.safe_load(file)
            references['used'] = []
   except Exception as e:
      logging.error(f"An error occurred while reading the file: {references_path}")
      raise RuntimeError(f"An error occurred while reading the file: {references_path}") from e
if not terminal_call:
   run_streamlit(config, skip_llm_call=skip_llm_call, rag_activated=rag_activated, embedding_model=embedding_model, chroma_path=chroma_path, references=references)
else:   
   output = run_terminal(config, skip_llm_call=skip_llm_call, embedding_model=embedding_model, chroma_path=chroma_path, references=references)
