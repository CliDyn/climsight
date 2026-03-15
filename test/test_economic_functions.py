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
from pathlib import Path

# Append the directory containing the module to sys.path
TEST_DIR = Path(__file__).resolve().parent
REPO_ROOT = TEST_DIR.parent
module_dir = REPO_ROOT / 'src' / 'climsight'
if module_dir not in sys.path:
    sys.path.append(str(module_dir))


from economic_functions import (
    calc_mean,
    get_population,
    plot_population,
    x_year_mean_population   
)

config_path = REPO_ROOT / 'config.yml'
test_config_path = TEST_DIR / 'config_test_economic_functions.yml'

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
    expected_mean_population = pd.read_csv(TEST_DIR / 'expected_x_year_mean_population.csv')
    expected_reduced_pop_data = pd.read_csv(TEST_DIR / 'expected_population.csv', index_col=0)    

    # --- test x_year_mean_population 
    mean_population = x_year_mean_population(pop_path, country, year_step=year_step, start_year=start_year, end_year=end_year)
    reduced_pop_data = get_population(pop_path, country)
    population_plot = plot_population(pop_path, country)

    assert are_dataframes_equal(mean_population,expected_mean_population,tol=1e-6), f'dataframe x_year_mean_population not the same' 
    assert not mean_population.drop(columns=['Time']).isna().all(axis=1).any(), 'x_year_mean_population returned an all-NaN row'
    assert are_dataframes_equal(reduced_pop_data,expected_reduced_pop_data,tol=1e-6), f'dataframe _population not the same'     
    #try to plot
    assert isinstance(population_plot, plt.Figure), f'error, It is not a Figure'
    plt.close(population_plot)    


def test_calc_mean_groups_partial_windows_without_nan_tail():
    dataset = pd.DataFrame({
        'Time': pd.to_datetime(['2000', '2001', '2002', '2003', '2004']),
        'a': [1.0, 3.0, 5.0, 7.0, None],
        'b': [2.0, 4.0, 6.0, 8.0, None],
    })

    expected = pd.DataFrame({
        'Time': [2000, 2002],
        'a': [2.0, 6.0],
        'b': [3.0, 7.0],
    })

    result = calc_mean(2, dataset)

    assert are_dataframes_equal(result, expected, tol=1e-6), 'calc_mean did not aggregate fixed year windows as expected'


def test_calc_mean_rejects_non_positive_window():
    dataset = pd.DataFrame({
        'Time': pd.to_datetime(['2000']),
        'a': [1.0],
    })

    with pytest.raises(ValueError):
        calc_mean(0, dataset)
