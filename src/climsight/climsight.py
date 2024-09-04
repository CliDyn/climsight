import logging
import yaml
import os
import sys

from streamlit_interface import run_streamlit
from terminal_interface import run_terminal
from rag import initialize_rag, load_rag

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
   openai_api_key = os.getenv('OPENAI_API_KEY') 
   embedding_model = rag_settings['embedding_model']
   chroma_path = rag_settings['chroma_path']
   document_path = rag_settings['document_path']
   timestamp_file = rag_settings['timestamp_file']
   chunk_size = rag_settings['chunk_size']
   chunk_overlap = rag_settings['chunk_overlap']
   separators = rag_settings['separators']
   rag_activated = rag_settings['rag_activated']
   rag_template = config['rag_template']
except KeyError as e:
   logging.error(f"Missing configuration key: {e}")
   raise RuntimeError(f"Missing configuration key: {e}")

rag_ready = False
if not skip_llm_call and rag_activated:
    try:
         logger.info("RAG is activated and skipllmcall is False. Initializing RAG...")
         initialize_rag(config, False)  # initialize chunking and embedding
         load_rag(embedding_model, chroma_path, openai_api_key) # load the RAG database 
         rag_ready = True
    except Exception as e:
         logger.warning(f"RAG database initialization skipped or failed: {e}")

if not terminal_call:
   run_streamlit(config, skip_llm_call=skip_llm_call)
else:   
   output = run_terminal(config, skip_llm_call=skip_llm_call)