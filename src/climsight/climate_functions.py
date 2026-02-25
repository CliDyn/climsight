"""
Functions for climat data extracting, processing, analysis.
climate patterns, historical and future climate projections
"""
import xarray as xr
import numpy as np
import pandas as pd
import glob
import os
import warnings

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
        climatemodel_name = config['climatemodel_name']
        historical_pattern = config['data_settings']['historical']
        projection_pattern = config['data_settings']['projection']

        if climatemodel_name == 'tco1279' or climatemodel_name == 'tco319':
            all_files = glob.glob(os.path.join(data_path, '*.nc'))
            hist_files = [f for f in all_files if climatemodel_name in f and historical_pattern in f]
            future_files = [f for f in all_files if climatemodel_name in f and projection_pattern in f]
        elif climatemodel_name == 'AWI_CM':
            hist_files = glob.glob(os.path.join(data_path, f"*{historical_pattern}*.nc"))
            future_files = glob.glob(os.path.join(data_path, f"*{projection_pattern}*.nc"))
        else: 
            warnings.warn("Unknown model type. Cannot evaluate given climate data.")

        if hist_files and climatemodel_name != "AWI_CM":
            hist = xr.open_mfdataset(hist_files, concat_dim='time_counter', combine='nested')
        elif hist_files and climatemodel_name == 'AWI_CM':

            hist = xr.open_mfdataset(hist_files, combine='by_coords', compat='override')
        else:
            raise FileNotFoundError(f"No historical files found matching pattern {historical_pattern} in {data_path}")

        if future_files and climatemodel_name != 'AWI_CM':
            future = xr.open_mfdataset(future_files, concat_dim='time_counter', combine='nested')
        elif future_files and climatemodel_name == 'AWI_CM':

            future = xr.open_mfdataset(future_files, combine='by_coords', compat='override')
        else:
            raise FileNotFoundError(f"No projection files found matching pattern {projection_pattern} in {data_path}")

        return hist, future
    
    except Exception as e:
        raise RuntimeError(f"Failed to load data: {str(e)}")
        #st.error(f"Failed to load data: {str(e)}")
        return None, None

# functions used in extract_climate_data
def select_data(dataset, variable, dimensions, lat, lon):
    """
    Selects data for a given variable at specified latitude and longitude.
    Handles longitude normalization for datasets using 0-360 range.
    """
    # Normalize longitude if dataset uses 0-360 range and input is negative
    lon_dim = dimensions['longitude']
    if lon < 0 and dataset[lon_dim].min() >= 0:
        lon = lon + 360

    return dataset[variable].sel(**{
        dimensions['latitude']: lat,
        lon_dim: lon},
        method="nearest") 

def verify_shape(hist_units, future_units, variable):
    """
    Sanity check if dimensions of historical and future data set are fine (mainly relevant for using AWI_CM data)
    """
    if hist_units != future_units:
        warning_msg = f"Shape / Dimension mismatch for {variable}: historical units '{hist_units}' vs future units '{future_units}'."
        warnings.warn(warning_msg)
        print(warning_msg)

def convert_temperature(hist_units, hist_data, future_data):
    """"
    Converts temperature from Kelvin into C and checks for any units beyond K and C (not compatible).
    """
    if 'K' in hist_units: # transformation to Celsius if data in Kelvin (using only hist_units form here on as hist_units and future_units are identical if no error was thrown before)
        hist_data -= 273.15
        future_data -= 273.15
        #print(f"Converted temperature from Kelvin to Celsius.")
    elif 'C' not in hist_units:  # Check if not already in Celsius
        warnings.warn(f"Unexpected temperature units: {hist_units}. Please check the unit manually.")
        #print(f"Units found: {hist_units}")
    return hist_data, future_data

def convert_to_mm_per_month(monthly_precip_kg_m2_s1):
    """"
    Converts kg_m2_s1 to mm per month
    """
    days_in_months = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    return monthly_precip_kg_m2_s1 * 60 * 60 * 24 * days_in_months # input in seconds 

def convert_precipitation(hist_units, hist_data, future_data):
    """
    Converts precipitation data from any unit to mm or throws error if unknown unit.
    """
    if 'kg m-2 s-1' in hist_units:
        hist_data = convert_to_mm_per_month(hist_data)
        future_data = convert_to_mm_per_month(future_data)
        #print(f"Converted precipitation from kg/m^2/s to mm/month.")
    elif 'm' in hist_units: # this is not perfectly handled yet, but the awi-cm-3 tco... data comes in unit m but is actually m^2/s
        hist_data = convert_to_mm_per_month(hist_data) * 1000
        future_data = convert_to_mm_per_month(future_data) * 1000
        #print(f"Converted precipitation from kg/m^2/s to mm/month.")
        # hist_data /= 100
        # future_data /= 100
        # print(f"Converted precipitation from m/month to mm/month.") 

    elif 'mm' not in hist_units:  # Check if not already in Celsius
        warnings.warn(f"Unexpected precipitation units: {hist_units}. Please check the unit manually.")
        print(f"Units found: {hist_units}")
    return hist_data, future_data

def process_data(data):
    """
    Processes the given data by squeezing out singleton dimensions, ensuring the 
    remaining dimensions include the month dimension (size 12).
    
    Args:
        data (xarray.DataArray or xarray.Dataset): The climate data to be processed.
    
    Returns:
        xarray.DataArray or xarray.Dataset: The processed climate data with singleton
        dimensions removed.
    """
    # Squeeze out all singleton dimensions from the data
    processed_data = data.squeeze()

    # Check if the squeezed data still contains the month dimension (expected to be size 12)
    if 12 not in processed_data.shape:
        raise ValueError("The month dimension (size 12) is missing from the processed data.")

    return processed_data

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
    """
    variables = config['variable_mappings']
    dimensions = config['dimension_mappings']
    df = pd.DataFrame({'Month': range(1, 13)})
    data_dict = {}

    # Initialize wind variables
    hist_wind_u = hist_wind_v = future_wind_u = future_wind_v = None

    for key, nc_var in variables.items():
        
        hist_data = select_data(hist, nc_var, dimensions, lat, lon)
        future_data = select_data(future, nc_var, dimensions, lat, lon)

        # THIS IS ONLY A BROAD IDEA - NOT TESTED YET! 
        # Ensure data covers all 12 months, potentially using groupby if data spans multiple years
        # if 'time' in hist_data.dims:
        #     hist_data = hist_data.groupby('time.month').mean('time')  # Averaging over 'time' assuming data includes multiple years
        #     future_data = future_data.groupby('time.month').mean('time')

        # Check if data is in unusual format and reshape if necessary 
        hist_data = process_data(hist_data)
        future_data = process_data(future_data)


        # Get units for both projections
        hist_units = hist_data.attrs.get('units', '')
        future_units = future_data.attrs.get('units', '')

        # check their shape for potential mismatch
        verify_shape(hist_units, future_units, key)

        # perform unit-specific transformations 
        if key == 'Temperature': 
            hist_data, future_data = convert_temperature(hist_units, hist_data, future_data)

        if key == 'Precipitation': 
            hist_data, future_data = convert_precipitation(hist_units, hist_data, future_data)

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

        # Convert arrays to lists and format to strings with specified precision
        hist_data_list = np.round(hist_data.values, 3).tolist()
        future_data_list = np.round(future_data.values, 3).tolist()

        hist_data_str = ', '.join(f'{x:.3f}' for x in hist_data_list)
        future_data_str = ', '.join(f'{x:.3f}' for x in future_data_list)

        data_dict[f"hist_{key}"] = hist_data_str
        data_dict[f"future_{key}"] = future_data_str

    #print(data_dict)
    return df, data_dict
