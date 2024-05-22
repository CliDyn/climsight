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

def run_terminal(config={}, api_key='', skip_llm_call=False, lon=None, lat=None, user_message=''):
