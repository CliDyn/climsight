"""
Functions for climat data extracting, processing, analysis.
climate patterns, historical and future climate projections
"""
import streamlit as st
import xarray as xr
import numpy as np
import pandas as pd
import glob
import os
import warnings

#from climate_functions import (
#    load_data,
#    extract_climate_data
#)

@st.cache_data
def load_data(config):
    """
    load climate model data from specified directory patterns

    Args:
        config (dict): configuration dictionary containing data path and file name patterns

    Returns:
        xarray.core.dataset.Dataset, xarray.core.dataset.Dataset : data from historical (hindacst) runs and climate projection
    """   
    try:
        data_path = config['data_settings']['data_path']
        historical_pattern = config['data_settings']['historical']
        projection_pattern = config['data_settings']['projection']

        hist_files = glob.glob(os.path.join(data_path, f"*{historical_pattern}*.nc"))
        future_files = glob.glob(os.path.join(data_path, f"*{projection_pattern}*.nc"))

        if hist_files:
            hist = xr.open_mfdataset(hist_files, combine='by_coords', compat='override')
        else:
            raise FileNotFoundError(f"No historical files found matching pattern {historical_pattern} in {data_path}")

        if future_files:
            future = xr.open_mfdataset(future_files, combine='by_coords', compat='override')
        else:
            raise FileNotFoundError(f"No projection files found matching pattern {projection_pattern} in {data_path}")

        return hist, future
    
    except Exception as e:
        st.error(f"Failed to load data: {str(e)}")
        return None, None

def convert_to_mm_per_month(monthly_precip_kg_m2_s1):
    days_in_months = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    return monthly_precip_kg_m2_s1 * 60 * 60 * 24 * days_in_months

def extract_climate_data(lat, lon, hist, future, config):
    """
    Extracts climate data for a given latitude and longitude from historical and future datasets.
    This function then handles different climatic variables and specifically calculates wind speed from u and v wind components.
    
    Args:
        - lat (float): Latitude of the location to extract data for.
        - lon (float): Longitude of the location to extract data for.
        - hist (xarray.Dataset): Historical climate dataset.
        - future (xarray.Dataset): Future climate dataset.
        - config (dict): Configuration dictionary containing variable mappings and dimension settings.

    Returns:
        - df (pandas.DataFrame): DataFrame containing present day and future data for temperature, precipitation, and wind speed for each month of the year, with 'Month' as a column.
        - data_dict (dict): Dictionary containing string representations of the extracted climate data for all variables including historical and future datasets.
    
    In more detail:
    The function processes each variable defined in the configuration, checks unit consistency, applies necessary transformations (e.g., temperature from Kelvin to Celsius, precipitation conversion), and calculates wind speed using the u and v wind components. Data for each variable, except wind components, is added directly to the DataFrame and a data dictionary. Wind speed is calculated from the wind components and added to the DataFrame.
    """
    variables = config['variable_mappings']
    dimensions = config['dimension_mappings']
    df = pd.DataFrame({'Month': range(1, 13)})

    data_dict = {}
    # Initialize wind variables
    hist_wind_u = hist_wind_v = future_wind_u = future_wind_v = None

    for key, nc_var in variables.items():
        
        hist_data = hist[nc_var].sel(**{
            dimensions['latitude']: lat,
            dimensions['longitude']: lon
        }, method="nearest")
        future_data = future[nc_var].sel(**{
            dimensions['latitude']: lat,
            dimensions['longitude']: lon
        }, method="nearest")

        # THIS IS ONLY A BROAD IDEA - NOT TESTED YET! 
        # Ensure data covers all 12 months, potentially using groupby if data spans multiple years
        # if 'time' in hist_data.dims:
        #     hist_data = hist_data.groupby('time.month').mean('time')  # Averaging over 'time' assuming data includes multiple years
        #     future_data = future_data.groupby('time.month').mean('time')

        # Check if data is in unusual format and reshape if necessary (only for AWI_CM files currently neccessary)
        if hist_data.values.shape == (1, 1, 12):
            hist_data = hist_data.squeeze()  # this removes dimensions of size 1
            future_data = future_data.squeeze()

        # Apply unit-specific transformations and check if units are identical for all projections (they should be)
        hist_units = hist_data.attrs.get('units', '')
        future_units = future_data.attrs.get('units', '')
        if hist_units != future_units:
            warnings.warn(f"Unit mismatch for {key}: historical units '{hist_units}' vs future units '{future_units}'. Please verify consistency.")
            print(f"Units mismatch found in variable {key}: Historical '{hist_units}', Future '{future_units}'.")

        # unit-specific transformations 
        if key == 'Temperature': 
            if 'K' in hist_units: # transformation to Celsius if data in Kelvin (using only hist_units form here on as hist_units and future_units are identical if no error was thrown before)
                hist_data -= 273.15
                future_data -= 273.15
                print(f"Converted temperature from Kelvin to Celsius for variable {key}.")
            elif 'C' not in hist_units:  # Check if not already in Celsius
                warnings.warn(f"Unexpected temperature units for {key}: {hist_units}. Please check the unit manually.")
                print(f"Units found: {hist_units}")

        if key == 'Precipitation': 
            if 'kg m-2 s-1' in hist_units:
                hist_data = convert_to_mm_per_month(hist_data)
                future_data = convert_to_mm_per_month(future_data)
                print(f"Converted precipitation from kg/m^2/s to mm/month for variable {key}.")
            elif 'm' in hist_units:
                hist_data /= 100
                future_data /= 100
                print(f"Converted precipitation from m/month to mm/month for variable {key}.")
            elif 'mm' not in hist_units:  # Check if not already in Celsius
                warnings.warn(f"Unexpected precipitation units for {key}: {hist_units}. Please check the unit manually.")
                print(f"Units found: {hist_units}")

        if key == 'u_wind':
            hist_wind_u = hist_data
            future_wind_u = future_data
        elif key == 'v_wind':
            hist_wind_v = hist_data
            future_wind_v = future_data
        else: 
            df[f"Present Day {key.capitalize()}"] = hist_data.values
            df[f"Future {key.capitalize()}"] = future_data.values
            
        # Calculate wind speed if both components are present and add to DataFrame
        if hist_wind_u is not None and hist_wind_v is not None:
            hist_wind_speed = np.hypot(hist_wind_u, hist_wind_v)
            future_wind_speed = np.hypot(future_wind_u, future_wind_v)
            df['Present Day Wind Speed'] = hist_wind_speed.values
            df['Future Wind Speed'] = future_wind_speed.values


        data_dict[f"hist_{key}"] = np.array2string(hist_data.values.ravel(), precision=3)
        data_dict[f"future_{key}"] = np.array2string(future_data.values.ravel(), precision=3)

    return df, data_dict
