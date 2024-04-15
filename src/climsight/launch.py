import subprocess
import os

def launch_streamlit():
    # Get the directory where launch.py is located
    launch_dir = os.path.dirname(os.path.realpath(__file__))
    # Construct the absolute path to climsight.py
    climsight_path = os.path.join(launch_dir, "climsight.py")

    # Run the command with the absolute path
    subprocess.run(["streamlit", "run", climsight_path])
