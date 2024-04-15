import subprocess
import os
import sys

def launch_streamlit():
    # Get the directory where launch.py is located
    launch_dir = os.path.dirname(os.path.realpath(__file__))
   
    # Construct the absolute path to climsight.py
    climsight_path = os.path.join(launch_dir, "climsight.py")
    
    # Check if 'skipLLMCall' argument is provided
    skip_llm_call = 'skipLLMCall' in sys.argv

    # Prepare the command to run
    command = ["streamlit", "run", climsight_path]    
    
    if skip_llm_call:
        climsight_path = climsight_path + " skipLLMCall"
    if skip_llm_call:
        command.append("skipLLMCall") 
        
    # Run the command with the absolute path
    subprocess.run(command)
