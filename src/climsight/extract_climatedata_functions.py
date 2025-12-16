"""
Climate Data Extraction Functions

This module provides backwards-compatible functions for extracting climate data.
It now uses the climate_data_providers module for the underlying implementation.

For new code, prefer using climate_data_providers directly:
    from climate_data_providers import get_climate_data_provider, ClimateDataResult
"""

import streamlit as st
import xarray as xr
import numpy as np
import pandas as pd
import os
import warnings
from scipy.spatial import cKDTree
import pyproj
import re
import calendar
import logging
import matplotlib.pyplot as plt

# Import provider system for new unified interface
from climate_data_providers import (
    get_climate_data_provider,
    get_available_providers,
    migrate_legacy_config,
    ClimateDataResult,
    ClimateDataProvider,
    NextGEMSProvider,
    ICCPProvider,
    AWICMProvider
)


logger = logging.getLogger(__name__)
logging.basicConfig(
   filename='climsight.log',
   level=logging.INFO,
   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
   datefmt='%Y-%m-%d %H:%M:%S'
)


def build_spatial_index_healpix(nc_file):
    """
    Builds a spatial index (cKDTree) using longitude and latitude from a Healpix NetCDF file.
    
    Parameters:
    - nc_file: Path to the NetCDF file.
    
    Returns:
    - tree: cKDTree built from longitude and latitude.
    - lons: Array of longitudes.
    - lats: Array of latitudes.
    """
    # Open the dataset to get longitude and latitude
    dataset = xr.open_dataset(nc_file)
    lons = dataset['lon'].values
    lats = dataset['lat'].values
    dataset.close()
    
    # Build cKDTree directly on longitude and latitude
    points = np.column_stack((lons, lats))
    tree = cKDTree(points)
    
    return tree, lons, lats

def extract_data_healpix(nc_file, desired_lon, desired_lat, tree, lons, lats, months=None, variable_mapping=None):
    """
    Extracts and interpolates data for a specific location and months from a Healpix NetCDF file.
    
    Parameters:
    - nc_file: Path to the NetCDF file.
    - desired_lon: Longitude of the desired location.
    - desired_lat: Latitude of the desired location.
    - tree: cKDTree built from longitude and latitude.
    - lons: Array of longitudes.
    - lats: Array of latitudes.
    - months: List of months to extract data for (1-12). Defaults to all months.
    - variable_mapping: Dictionary mapping full variable names to file variable names.
    
    Returns:
    - df: DataFrame containing the interpolated data for each month.
    - extracted_vars: List of file variable names that were successfully extracted.
    """
    # Open the dataset
    dataset = xr.open_dataset(nc_file)
    
    # Query 4 nearest neighbors using the cKDTree
    distances, indices = tree.query([desired_lon, desired_lat], k=4)
    
    # Get neighbor longitudes and latitudes
    neighbors_lons = lons[indices]
    neighbors_lats = lats[indices]
    
    # Define stereographic projection centered at the desired location
    pste = pyproj.Proj(proj="stere", errcheck=True, ellps='WGS84',
                      lat_0=desired_lat, lon_0=desired_lon)
    
    # Project neighbors and desired point
    neighbors_x, neighbors_y = pste(neighbors_lons, neighbors_lats)
    desired_x, desired_y = pste(desired_lon, desired_lat)
    
    # Compute distances in projected coordinates
    dx = neighbors_x - desired_x
    dy = neighbors_y - desired_y
    distances_proj = np.hypot(dx, dy)
    
    # Handle months
    if months is None:
        months = list(range(1, 13))  # 1 to 12
    else:
        months = [int(m) for m in months]
    month_indices = [m - 1 for m in months]  # Convert to 0-based indices
    
    # Initialize dataframe with month names
    df = pd.DataFrame({'Month': [calendar.month_name[m] for m in months]})
    
    df_vars = {}
    
    if np.any(distances_proj == 0):
        # Use value at the exact location
        zero_index = np.where(distances_proj == 0)[0][0]
    else:
        inv_distances = 1.0 / distances_proj
        weights = inv_distances / inv_distances.sum()
   
    for full_var, file_var in variable_mapping.items():
        if file_var not in dataset.variables:
            warnings.warn(f"Variable '{file_var}' not found in '{nc_file}'. Skipping.")
            continue
        interpolated_values = []
        for month_index in month_indices:
            data_values = dataset[file_var][month_index, indices].values  # shape (4,)
            # Interpolate
            if np.any(distances_proj == 0):
                # Use value at the exact location
                interpolated_value = data_values[zero_index]
            else:
                interpolated_value = np.dot(weights, data_values)
            interpolated_values.append(interpolated_value)
        interpolated_values = np.array(interpolated_values)
        df[file_var] = interpolated_values
        var_param = {}
        var_param['name'] = file_var
        var_param['units'] = dataset[file_var].attrs.get('units', '')
        var_param['full_name'] = full_var
        var_param['long_name'] = dataset[file_var].attrs.get('long_name', '')
        df_vars[file_var] = var_param
    
    dataset.close()
    
    return df, df_vars

def post_process_data(df, df_vars):
    """
    Applies unit conversions and calculates wind speed and direction.
    
    Parameters:
    - df: DataFrame containing raw extracted data.
    - df_vars: Dict of dictionaries containing variable metadata like:  {'name': '', 'unit': '', 'full_name': '', 'long_name': ''}.
    
    Returns:
    - df_processed: DataFrame with processed data, including wind speed and direction.
    """
    df_processed = df.copy()
    df_vars_processed = df_vars.copy()
    # Convert units if necessary
    if 'mean2t' in df_processed.columns:
        df_processed['mean2t'] = df_processed['mean2t'] - 273.15  # K to °C
        df_vars_processed['mean2t']['units'] = '°C'
        
    if 'tp' in df_processed.columns:
        df_processed['tp'] = df_processed['tp'] * 1000.0  # m to mm/month
        df_vars_processed['tp']['units'] = 'mm/month'
    
    # Calculate wind speed and direction if wind components are present
    if 'wind_u' in df_processed.columns and 'wind_v' in df_processed.columns:
        wind_speed = np.sqrt(df_processed['wind_u']**2 + df_processed['wind_v']**2)
        wind_direction = (180.0 + np.degrees(np.arctan2(df_processed['wind_u'], df_processed['wind_v']) )) % 360
        df_vars_processed['wind_speed'] = {'name': 'wind_speed', 'units': 'm/s', 'full_name': 'Wind Speed', 'long_name': 'Wind Speed'}
        df_vars_processed['wind_direction'] = {'name': 'wind_direction', 'units': '°', 'full_name': 'Wind Direction', 'long_name': 'Wind Direction'}
        df_processed['wind_speed'] = wind_speed.round(2)
        df_processed['wind_direction'] = wind_direction.round(2)
    
    # Round other variables to 2 decimal places (skip non-numeric columns like 'Month')
    for var in df_processed.columns:
        if var != 'Month' and pd.api.types.is_numeric_dtype(df_processed[var]):
            df_processed[var] = df_processed[var].round(2)
    
    return df_processed, df_vars_processed

def extract_data(nc_file, desired_lon, desired_lat, tree, coords, months=None, coordinate_system='healpix', variable_mapping=None):
    """
    Wrapper function to extract and process data based on coordinate system.
    
    Parameters:
    - nc_file: Path to the NetCDF file.
    - desired_lon: Longitude of the desired location.
    - desired_lat: Latitude of the desired location.
    - tree: cKDTree built from the coordinates.
    - coords: Tuple of coordinate arrays (e.g., lons and lats).
    - months: List of months to extract data for (1-12). Defaults to all months.
    - coordinate_system: Coordinate system of the file.
    - variable_mapping: Dictionary mapping full variable names to file variable names.
    
    Returns:
    - df_processed: DataFrame containing the processed data for each month.
    - extracted_vars: List of file variable names that were successfully extracted.
    """
    if coordinate_system.lower() == 'healpix':
        lons, lats = coords
        df_raw, df_vars = extract_data_healpix(
            nc_file, desired_lon, desired_lat, tree, lons, lats, months, variable_mapping
        )
        df_processed, df_vars_processed  = post_process_data(df_raw, df_vars)
        return df_processed, df_vars_processed
    else:
        warnings.warn(f"Coordinate system '{coordinate_system}' is not supported yet.")
        return None, []


def extract_all_data(input_files, desired_lon, desired_lat, variable_mapping, months=None):
    """
    Extracts data from multiple NetCDF files and returns a list of DataFrames.
    
    Parameters:
    - input_files: Dictionary of NetCDF file paths with metadata.
    - desired_lon: Longitude of the desired location.
    - desired_lat: Latitude of the desired location.
    - months: List of months to extract data for (1-12). Defaults to all months.
    - variable_mapping: Dictionary mapping full variable names to file variable names.
    
    Returns:
    - df_list: List of dictionaries containing metadata and DataFrames for each file.
    """
    df_list = []
    
    # Build spatial indices for each coordinate system
    spatial_indices = {}
    for nc_file, meta in input_files.items():
        file_name = meta.get('file_name', nc_file)
        coordinate_system = meta.get('coordinate_system', 'healpix').lower()
        if coordinate_system == 'healpix' and coordinate_system not in spatial_indices:
            if not os.path.exists(file_name):
                warnings.warn(f"File '{file_name}' does not exist. Skipping.")
                continue
            tree, lons, lats = build_spatial_index_healpix(file_name)
            spatial_indices[coordinate_system] = (tree, (lons, lats))
        elif coordinate_system != 'healpix':
            warnings.warn(f"Coordinate system '{coordinate_system}' for file '{file_name}' is not supported yet.")
    
    # Extract data from each file
    for nc_file, meta in input_files.items():
        file_name = meta.get('file_name', file_name)
        coordinate_system = meta.get('coordinate_system', 'healpix').lower()
        if coordinate_system != 'healpix':
            continue  # Skip unsupported coordinate systems for now
        
        if coordinate_system in spatial_indices:
            tree, coords = spatial_indices[coordinate_system]
        else:
            warnings.warn(f"No spatial index available for coordinate system '{coordinate_system}' in file '{file_name}'. Skipping.")
            continue
        
        df, df_vars = extract_data(
            nc_file=file_name,
            desired_lon=desired_lon,
            desired_lat=desired_lat,
            tree=tree,
            coords=coords,
            months=months,
            coordinate_system=coordinate_system,
            variable_mapping=variable_mapping
        )
        
        if df is not None and not df.empty:
            df_list.append({
                'filename': file_name,
                'years_of_averaging': meta.get('years_of_averaging', ''),
                'description': meta.get('description', ''),
                'dataframe': df,
                'extracted_vars': df_vars,
                'main': meta.get('is_main', False),  # Flag the main simulation
                'source': meta.get('source', '')
            })
    
    return df_list

def prepare_data_agent_response(df_list):
    """
    Prepares the data_agent_response dictionary from the extracted DataFrames.

    Parameters:
    - df_list: List of dictionaries containing metadata and DataFrames for each file.

    Returns:
    - data_agent_response: Dictionary with 'input_params' and 'content_message'.
    """

    response2llm = {
        'input_params': {},
        'content_message': ""
    }

    total_simulations = len(df_list)
    # Add dataframes to  input parameters from the all files
    for i, entry in enumerate(df_list):
        # rename columns and add units
        df = rename_columns(entry['dataframe'], entry['extracted_vars'])
        # Convert all columns except 'Month' to numeric, forcing errors to NaN
        for col in df.columns:
            if col != 'Month' and not col.lower().startswith('month'):
                df[col] = pd.to_numeric(df[col], errors='coerce')
        # Convert the renamed DataFrame to JSON
        df_json = df.to_json(orient='records', indent=2)
        sim_name = 'simulation'+str(i+1)
        response2llm['input_params'][sim_name] = df_json

    # Add explanation to content_message
    for i, entry in enumerate(df_list):
        sim_name = 'simulation'+str(i+1)
        ifmain = ""
        if entry['main']:
            ifmain = "This simulation serves as a historical reference to extract its values from other simulations."            
        response2llm['content_message'] += "\n\n Climate parameters are obtained from atmospheric model simulations. The climatology is based on the average years: " +entry['years_of_averaging']+ ". "+entry['description']+ifmain+" :\n {"+sim_name+"}"

    # find index of main simulation
    for i, entry in enumerate(df_list):
        if entry['main']:
            main_index = i
            break
    # Add dataframes to  input parameters from the difference between files
    for i, entry in enumerate(df_list):
        if entry['main']:
            continue
        df_main = df_list[main_index]['dataframe'].copy()
        df = entry['dataframe'].copy()
        # Convert all columns except 'Month' to numeric first
        for col in df.columns:
            if col != 'Month' and not col.lower().startswith('month'):
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df_main[col] = pd.to_numeric(df_main[col], errors='coerce')
        numeric_cols = df.select_dtypes(include=['number']).columns
        df = df[numeric_cols] - df_main[numeric_cols]
        # rename columns and add units
        df = rename_columns(df, entry['extracted_vars'])
        # Convert the renamed DataFrame to JSON
        df_json = df.to_json(orient='records', indent=2)
        sim_name = 'simulation'+str(total_simulations+i+1)
        response2llm['input_params'][sim_name] = df_json

    # Add explanation to content_message
    for i, entry in enumerate(df_list):
        if entry['main']:
            continue        
        sim_name = 'simulation'+str(total_simulations +i+1)
        response2llm['content_message'] += "\n\n Difference between simulations (changes in climatological parameters (Δ)) for the years " +entry['years_of_averaging']+ " compared to the simulation for " +df_list[main_index]['years_of_averaging']+ ". :\n {"+sim_name+"}"
    
    # logger.info(f"Response to LLM from data agent is:, {response2llm}")
    return response2llm

# Function to rename columns
def rename_columns(df, extracted_vars):
    rename_mapping = {}
    for col in df.columns:
        if col in extracted_vars:
            full_name = extracted_vars[col]['full_name']
            units = extracted_vars[col]['units']
            # Append units to the full_name
            new_name = f"{full_name} ({units})"
            rename_mapping[col] = new_name
        else:
            # Keep the original name if not in extracted_vars
            rename_mapping[col] = col
    # Rename the DataFrame columns
    return df.rename(columns=rename_mapping)

def request_climate_data(config, desired_lon, desired_lat, months=[i for i in range(1,13)], source_override=None):
    """
    Request climate data for a given location.

    This function now supports the new provider system while maintaining
    backwards compatibility with the legacy config format.

    Args:
        config: Configuration dictionary
        desired_lon: Longitude of the location
        desired_lat: Latitude of the location
        months: List of months (1-12) to extract data for
        source_override: Optional source name to override config ('nextGEMS', 'ICCP', 'AWI_CM')

    Returns:
        Tuple of (data_agent_response, df_list)
    """
    # Migrate config if in legacy format
    config = migrate_legacy_config(config)

    # Check if using new provider system
    if 'climate_data_source' in config or source_override:
        # Use new provider system
        try:
            provider = get_climate_data_provider(config, source_override)
            result = provider.extract_data(desired_lon, desired_lat, months)
            return result.data_agent_response, result.df_list
        except NotImplementedError as e:
            logger.warning(f"Provider not implemented: {e}")
            raise
        except Exception as e:
            logger.error(f"Error using provider: {e}")
            # Fall back to legacy implementation
            logger.info("Falling back to legacy implementation")

    # Legacy implementation for backwards compatibility
    # Process all files to extract data
    df_list = extract_all_data(
        input_files=config['climate_model_input_files'],
        desired_lon=desired_lon,
        desired_lat=desired_lat,
        months=months,
        variable_mapping=config['climate_model_variable_mapping']
    )

    # Prepare data_agent_response from extracted data
    data_agent_response = prepare_data_agent_response(
        df_list=df_list,
    )
    return data_agent_response, df_list


def request_climate_data_with_provider(config, desired_lon, desired_lat, months=None, source_override=None):
    """
    Request climate data using the new provider system.

    This is the preferred function for new code. It returns a ClimateDataResult
    object with unified output format regardless of the data source.

    Args:
        config: Configuration dictionary
        desired_lon: Longitude of the location
        desired_lat: Latitude of the location
        months: List of months (1-12) to extract data for. None means all months.
        source_override: Optional source name to override config ('nextGEMS', 'ICCP', 'AWI_CM')

    Returns:
        ClimateDataResult with unified output format
    """
    config = migrate_legacy_config(config)
    provider = get_climate_data_provider(config, source_override)
    return provider.extract_data(desired_lon, desired_lat, months)

def plot_climate_data(df_list):

    figs = []
    # Get the list of parameters to plot (excluding 'Month', 'wind_u', 'wind_v')
    parameters = df_list[0]['dataframe'].columns.tolist()
    parameters_to_plot = [param for param in parameters if param not in ['Month', 'wind_u', 'wind_v']]

    fs = 18
    for param in parameters_to_plot:
        fig, axes = plt.subplots(nrows=1, ncols=1,figsize=(10,6))

        # Get parameter name and units from extracted_vars
        var_info = df_list[0]['extracted_vars'][param]
        source = df_list[0]['source']
        param_full_name = var_info['full_name']
        units = var_info['units']

        for data in df_list:
            df = data['dataframe']
            plt.plot(df['Month'], df[param], marker='o', label=data['years_of_averaging'])

        plt.title(f"{param_full_name} ({units})", fontsize=fs)
        plt.xlabel('Month', fontsize=fs)
        plt.ylabel(f"{param_full_name} ({units})", fontsize=fs)
        plt.xticks(rotation=45)
        axes.tick_params(labelsize=fs)
        axes.grid(color='k', alpha=0.5, linestyle='--')
        axes.legend(fontsize=fs)            
        fig.tight_layout()
        #fig.savefig(f"{param}_plot.png")    
        figl = {'fig': fig, 'param': param, 'source': source, 'full_name': param_full_name, 'units': units}
        figs.append(figl)
    
    return figs


def main():
    # Example Input Dictionary
    input_files = {
        'climatology_IFS_9-FESOM_5-production_2020x_compressed.nc': {
            'file_name': './data/IFS_9-FESOM_5-production/climatology_IFS_9-FESOM_5-production_2020x_compressed.nc',
            'years_of_averaging': '2020-2029',
            'description': 'The nextGEMS pre-final simulations for years 2030x..',
            'coordinate_system': 'healpix',
            'is_main': True
        },
        'climatology_IFS_9-FESOM_5-production_2030x_compressed.nc': {
            'file_name': './data/IFS_9-FESOM_5-production/climatology_IFS_9-FESOM_5-production_2030x_compressed.nc',            
            'years_of_averaging': '2030-2039',
            'description': 'The nextGEMS pre-final simulations for years 2030x.',
            'coordinate_system': 'healpix'
        },
        'climatology_IFS_9-FESOM_5-production_2040x_compressed.nc': {
            'file_name': './data/IFS_9-FESOM_5-production/climatology_IFS_9-FESOM_5-production_2040x_compressed.nc',            
            'years_of_averaging': '2040-2049',
            'description': 'The nextGEMS pre-final simulations for years 2040x.',
            'coordinate_system': 'healpix'
        }
        # Add more files as needed
    }
    
    # Desired location
    desired_lon = 13.37  # Example longitude
    desired_lat = 52.524  # Example latitude

    # Define variable mapping: Full names to file variable names
    variable_mapping = {
        'Temperature': 'mean2t',
        'Total Precipitation': 'tp',
        'Wind U': 'wind_u',
        'Wind V': 'wind_v',
        # Add more mappings as needed
    }
    
    # Process all files to extract data
    df_list = extract_all_data(
        input_files=input_files,
        desired_lon=desired_lon,
        desired_lat=desired_lat,
        months=[i for i in range(1,13)],  # Example: Extract data for January, February, March
        variable_mapping=variable_mapping
    )
    
    # Prepare data_agent_response from extracted data
    data_agent_response = prepare_data_agent_response(
        df_list=df_list,
    )

if __name__ == "__main__":
    main()