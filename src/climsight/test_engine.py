from climsight_engine import llm_request, forming_request
import logging
from stream_handler import StreamHandler
import os
#import streamlit as st

## ADD to MAin
#Initialize logging at the beginning of your main application
logger = logging.getLogger(__name__)
logging.basicConfig(
   filename='climsight.log',
   level=logging.INFO,
   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
   datefmt='%Y-%m-%d %H:%M:%S'
)

lat = 52.5240
lon = 13.3700
user_message = "Where Am I ?"
skip_llm_call = True
api_key=''   

# Create a generator object by calling func2
generator = forming_request(lat, lon, user_message)

while True:
   try:
      # Get the next intermediate result from the generator
      result = next(generator)
      print(f"Intermediate result: {result}")
   except StopIteration as e:
      # The generator is exhausted, and e.value contains the final result
      content_message, input_params, config, df_data, figs, data = e.value
      break


stream_handler = StreamHandler()

if not isinstance(skip_llm_call, bool):
    logging.error(f"skip_llm_call must be bool in clim_request(...) ")
    raise TypeError("skip_llm_call must be  bool")    

if not isinstance(api_key, str):
    logging.error(f"api_key must be a string in clim_request(...) ")
    raise TypeError("api_key must be a string")
if not api_key:
    api_key = os.environ.get("OPENAI_API_KEY") # check if OPENAI_API_KEY is set in the environment
    if not api_key:        
        skip_llm_call=True
        api_key='Dummy' #for longchain api_key should be non empty str


if not skip_llm_call:
   output = llm_request(content_message, input_params, config, api_key, stream_handler)   
   with open('output.txt', 'w') as file:
      # Write the content to the file
      print(output)
      file.write(output) 
'''
print("second time ")
# Create a generator object by calling func2
generator = clim_request(lat, lon, question, stream_handler)

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
