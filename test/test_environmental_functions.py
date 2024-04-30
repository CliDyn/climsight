'''
This script performs unit testing for the climsight 
   environmenatl
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


from environmental_functions import (
   fetch_biodiversity,
   load_nat_haz_data,
   filter_events_within_square,
   plot_disaster_counts
)

config_path = '../config.yml'
test_config_path = 'config_test_environmental_functions.yml'

def test_config_files_exist():
    assert os.path.exists(config_path), f"Configuration file does not exist: {config_path}"
    assert os.path.exists(test_config_path), f"Test configuration file does not exist: {test_config_path}"


@pytest.fixture
def config_main():
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

#config = config_main


@pytest.fixture
def config_test():
    with open(test_config_path, 'r') as file:
        config_data = yaml.safe_load(file)
    return config_data

#config_env = config_data

@pytest.fixture
def latlon(config_test):
    lat = float(config_test['test_location']['lat'])
    lon = float(config_test['test_location']['lon'])
    return lat, lon  
 
 #----------------------------- Test biodiversity
@pytest.fixture
def expected_biodiv(config_test):
    return config_test['test_location']['biodiv']

@pytest.mark.request 
@pytest.mark.env
def test_fetch_biodiversity(config_test, expected_biodiv):
    lat, lon = config_test['test_location']['lat'], config_test['test_location']['lon']
    biodiv = fetch_biodiversity(round(lat), round(lon))
    assert biodiv == expected_biodiv, f"error in fetch_biodiversity, point({lat},{lon})"

 #----------------------------- Test filter_events_within_square
def are_dataframes_equal(df1, df2, tol=1e-6):
    """
    Compares two DataFrames for equality within a specified tolerance level.
    Returns True if they are equal within the tolerance, otherwise False.
    """
    try:
        pd.testing.assert_frame_equal(df1, df2, check_dtype=False, atol=tol)
        return True
    except AssertionError:
        return False

@pytest.mark.hazard
@pytest.mark.env
@pytest.mark.local_data    
def test_filter_events_within_square(config_test, config_main, expected_biodiv):
    #I do not like it (we need to eliminate hardcode paths)
    os.chdir('..')   
    
    haz_path = config_main['haz_path']
    distance_from_event = config_main['distance_from_event']
    lat, lon = config_test['test_location']['lat'], config_test['test_location']['lon']
    try:
        filtered_events_square, promt_hazard_data = filter_events_within_square(lat, lon, haz_path, distance_from_event)
    except Exception as e:
        warnings.warn(f"Failed to filter_events_within_square: {str(e)}", RuntimeWarning)
        # Explicitly fail the test with a message
        pytest.fail(f"Test failed due to an error in filter_events_within_square: {str(e)}, no files in data ?")        
            
    expected_promt_hazard_data = pd.DataFrame(config_test['test_location']['promt_hazard_data'])    
    expected_filtered_events_square = pd.DataFrame(config_test['test_location']['filtered_events_square'])    
    expected_filtered_events_square['latitude'] = pd.to_numeric(expected_filtered_events_square['latitude'], errors='coerce')
    expected_filtered_events_square['longitude'] = pd.to_numeric(expected_filtered_events_square['longitude'], errors='coerce')

    promt_hazard_data = promt_hazard_data.reset_index(drop=True)
    filtered_events_square = filtered_events_square.reset_index(drop=True)
    
    assert promt_hazard_data.equals(expected_promt_hazard_data)
    assert are_dataframes_equal(expected_filtered_events_square, filtered_events_square),f'dataframe filtered_events_square not the same'    

    #I do not like it (we need to eliminate hardcode paths)
    os.chdir('test')   
    
    #try to plot
    haz_fig = plot_disaster_counts(filtered_events_square)
    assert isinstance(haz_fig, plt.Figure), f'error, It is not a Figure'
    plt.close(haz_fig)
