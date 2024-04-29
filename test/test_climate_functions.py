'''
This script performs unit testing for the climsight 
   climate
            functions,
ensuring configuration files are correctly loaded ...
It uses pytest to test 
The tests validate both the 
presence of essential config files and the accuracy of data fetched from 
external geocoding services. 
'''

import pytest
import yaml
import warnings
import sys
import os
import requests
import pandas as pd
import matplotlib.pyplot as plt

# Append the directory containing the module to sys.path
module_dir = os.path.abspath('../src/climsight/')
if module_dir not in sys.path:
    sys.path.append(module_dir)


from climate_functions import (
   load_data,
   extract_climate_data
)

config_path = '../config.yml'
test_config_path = 'config_test_climate_functions.yml'

def test_config_files_exist():
    assert os.path.exists(config_path), f"Configuration file does not exist: {config_path}"
    assert os.path.exists(test_config_path), f"Test configuration file does not exist: {test_config_path}"


@pytest.fixture
def config_main():
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

@pytest.fixture
def config_test():
    with open(test_config_path, 'r') as file:
        config_data = yaml.safe_load(file)
    return config_data

def are_dataframes_equal(df1, df2, tol=1e-6):
    """
    Compares two DataFrames for equality within a specified tolerance level,
    ignoring NaN values in comparison.
    Returns True if they are equal within the tolerance, otherwise False.
    """
    try:
        pd.testing.assert_frame_equal(df1, df2, check_dtype=False, atol=tol, 
                                      check_exact=False, check_names=False)
        return True
    except AssertionError:
        return False
      
@pytest.mark.climate    
@pytest.mark.local_data  
def test_climate_data(config_main, config_test):
    lat, lon   = config_test['test_location']['lat'], config_test['test_location']['lon']
    expected_data_dict = config_test['test_location']['data_dict']
    data_path  = config_main['data_path']
    expected_df = pd.read_csv('expected_df_climate_data.csv', index_col=0)    
    
    #I do not like it (we need to eliminate hardcode paths)
    os.chdir('..') 
    try:
        hist, future = load_data(data_path)
    except Exception as e:
        warnings.warn(f"Failed to load_data(data_path): {str(e)}", RuntimeWarning)
        # Explicitly fail the test with a message
        pytest.fail(f"Test failed due to an error in load_data: {str(e)}, no file probably ?")        

    try:
        df, data_dict = extract_climate_data(lat, lon, hist, future)
    except Exception as e:
        warnings.warn(f"Failed to extract_climate_data(lat, lon, hist, future): {str(e)}", RuntimeWarning)
        # Explicitly fail the test with a message
        pytest.fail(f"Test failed due to an error in extract_climate_data: {str(e)}, no file probably ?")        
            
    assert are_dataframes_equal(df,expected_df,tol=1e-6), f'error in dataframes after extract_climate_data'
    assert expected_data_dict == data_dict,f'error in dictionary after extract_climate_data'


   
    #I do not like it (we need to eliminate hardcode paths)
    os.chdir('test')
        
        
        
