'''
This script performs unit testing for the climsight 
   economic
            functions,
ensuring configuration files are correctly loaded ...
It uses pytest to test 
The tests validate both the 
presence of essential config files and the accuracy of data fetched from 
external geocoding services. 
here wi do all the test in : test_population
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


from economic_functions import (
    get_population,
    plot_population,
    x_year_mean_population   
)

config_path = '../config.yml'
test_config_path = 'config_test_economic_functions.yml'

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
    
@pytest.mark.economic    
@pytest.mark.local_data  
def test_population(config_main, config_test):
    lat, lon   = config_test['test_location']['lat'], config_test['test_location']['lon']
    country    = config_test['test_location']['country']
    year_step  = config_test['test_location']['year_step']
    start_year = config_test['test_location']['start_year']
    end_year   = config_test['test_location']['end_year']
    
    pop_path   = config_main['pop_path']
    expected_mean_population = pd.read_csv('expected_x_year_mean_population.csv')
    expected_reduced_pop_data = pd.read_csv('expected_population.csv', index_col=0)    
    
    
    #I do not like it (we need to eliminate hardcode paths)
    os.chdir('..')  

    # --- test x_year_mean_population 
    mean_population = x_year_mean_population(pop_path, country, year_step=year_step, start_year=start_year, end_year=end_year)
    reduced_pop_data = get_population(pop_path, country)
    population_plot = plot_population(pop_path, country)
    
    #I do not like it (we need to eliminate hardcode paths)
    os.chdir('test')
    assert are_dataframes_equal(mean_population,expected_mean_population,tol=1e-6), f'dataframe x_year_mean_population not the same' 
    assert are_dataframes_equal(reduced_pop_data,expected_reduced_pop_data,tol=1e-6), f'dataframe _population not the same'     
    #try to plot
    assert isinstance(population_plot, plt.Figure), f'error, It is not a Figure'
    plt.close(population_plot)    
        
        
        
