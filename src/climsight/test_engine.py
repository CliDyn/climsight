from climsight_engine import clim_request
import logging

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
question = "Where Am I ?"

#clim_request(lat, lon, question, data=[], config=[], api_key='', skip_llm_call=False):

# Create a generator object by calling func2
generator = clim_request(lat, lon, question)

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

print("second time ")
# Create a generator object by calling func2
generator = clim_request(lat, lon, question)

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

