"""
Climate Data Providers Module

This module provides an abstraction layer for different climate data sources.
Each provider implements a common interface to extract climate data for a given location.

Supported providers:
- NextGEMSProvider: High-resolution nextGEMS data on HEALPix grid
- ICCPProvider: ICCP TCo319-DART climate data on regular lat/lon grid
- AWICMProvider: AWI-CM CMIP6 model data on regular lat/lon grid
"""

import os
import logging
import warnings
import calendar
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any

import numpy as np
import pandas as pd
import xarray as xr
import pyproj
from scipy.spatial import cKDTree

# Import existing climate functions for AWI_CM provider
from climate_functions import load_data, extract_climate_data

logger = logging.getLogger(__name__)


@dataclass
class ClimateDataResult:
    """Unified output structure for all climate data providers.

    Attributes:
        df_list: List of dictionaries containing:
            - filename: str
            - years_of_averaging: str
            - description: str
            - dataframe: pd.DataFrame with columns: Month, var1, var2, ...
            - extracted_vars: dict mapping var_name to {name, units, full_name, long_name}
            - main: bool - True if this is the reference/baseline period
            - source: str - Source description for citations
        data_agent_response: Dictionary with:
            - input_params: dict - Parameters for LLM prompt
            - content_message: str - Template message for LLM
        source_name: Name of the data source ("nextGEMS", "ICCP", or "AWI_CM")
        source_description: Human-readable description of the data source
    """
    df_list: List[Dict[str, Any]]
    data_agent_response: Dict[str, Any]
    source_name: str
    source_description: str


class ClimateDataProvider(ABC):
    """Abstract base class for climate data providers.

    All climate data providers must implement this interface to ensure
    consistent behavior across different data sources.
    """

    def __init__(self, source_config: dict, global_config: dict = None):
        """Initialize provider with configuration.

        Args:
            source_config: Configuration specific to this data source
            global_config: Global configuration (for shared settings)
        """
        self.source_config = source_config
        self.global_config = global_config or {}

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this data source."""
        pass

    @property
    @abstractmethod
    def coordinate_system(self) -> str:
        """Return the coordinate system used (healpix, regular, etc.)."""
        pass

    @property
    def description(self) -> str:
        """Return human-readable description of the data source."""
        return self.source_config.get('description', f'{self.name} climate data')

    @abstractmethod
    def extract_data(
        self,
        lon: float,
        lat: float,
        months: Optional[List[int]] = None
    ) -> ClimateDataResult:
        """Extract climate data for a location.

        Args:
            lon: Longitude of the location
            lat: Latitude of the location
            months: List of months (1-12) to extract. None means all months.

        Returns:
            ClimateDataResult with unified output format
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if data files exist and are accessible.

        Returns:
            True if the provider can be used, False otherwise
        """
        pass

    def _prepare_data_agent_response(self, df_list: List[Dict]) -> Dict:
        """Prepare the data_agent_response dictionary from extracted DataFrames.

        This is a shared utility method that can be used by providers that follow
        the standard df_list format (NextGEMS, ICCP, etc.).

        Args:
            df_list: List of dictionaries with climate data

        Returns:
            Dictionary with formatted data for LLM consumption
        """
        response = {
            'input_params': {},
            'content_message': ""
        }

        if not df_list:
            return response

        total_simulations = len(df_list)

        # Add dataframes as JSON to input parameters
        for i, entry in enumerate(df_list):
            df = self._rename_columns(entry['dataframe'], entry['extracted_vars'])
            for col in df.columns:
                if col != 'Month' and not col.lower().startswith('month'):
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            df_json = df.to_json(orient='records', indent=2)
            sim_name = f'simulation{i+1}'
            response['input_params'][sim_name] = df_json

        # Add explanations to content_message
        for i, entry in enumerate(df_list):
            sim_name = f'simulation{i+1}'
            ifmain = ""
            if entry['main']:
                ifmain = " This simulation serves as a historical reference to extract its values from other simulations."
            response['content_message'] += (
                f"\n\n Climate parameters are obtained from atmospheric model simulations. "
                f"The climatology is based on the average years: {entry['years_of_averaging']}. "
                f"{entry['description']}{ifmain} :\n {{{sim_name}}}"
            )

        # Find main simulation index
        main_index = 0
        for i, entry in enumerate(df_list):
            if entry['main']:
                main_index = i
                break

        # Add difference dataframes
        for i, entry in enumerate(df_list):
            if entry['main']:
                continue

            df_main = df_list[main_index]['dataframe'].copy()
            df = entry['dataframe'].copy()

            for col in df.columns:
                if col != 'Month' and not col.lower().startswith('month'):
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    df_main[col] = pd.to_numeric(df_main[col], errors='coerce')

            numeric_cols = df.select_dtypes(include=['number']).columns
            df_diff = df[numeric_cols] - df_main[numeric_cols]
            df_diff = self._rename_columns(df_diff, entry['extracted_vars'])

            df_json = df_diff.to_json(orient='records', indent=2)
            sim_name = f'simulation{total_simulations + i + 1}'
            response['input_params'][sim_name] = df_json

        # Add difference explanations
        for i, entry in enumerate(df_list):
            if entry['main']:
                continue
            sim_name = f'simulation{total_simulations + i + 1}'
            response['content_message'] += (
                f"\n\n Difference between simulations (changes in climatological parameters (Δ)) "
                f"for the years {entry['years_of_averaging']} compared to the simulation for "
                f"{df_list[main_index]['years_of_averaging']}. :\n {{{sim_name}}}"
            )

        return response

    def _rename_columns(self, df: pd.DataFrame, extracted_vars: Dict) -> pd.DataFrame:
        """Rename columns to include full names and units.

        Args:
            df: DataFrame with raw column names
            extracted_vars: Dictionary mapping variable names to metadata

        Returns:
            DataFrame with renamed columns
        """
        rename_mapping = {}
        for col in df.columns:
            if col in extracted_vars:
                full_name = extracted_vars[col]['full_name']
                units = extracted_vars[col]['units']
                rename_mapping[col] = f"{full_name} ({units})"
            else:
                rename_mapping[col] = col
        return df.rename(columns=rename_mapping)


class NextGEMSProvider(ClimateDataProvider):
    """Provider for nextGEMS high-resolution climate data on HEALPix grid.

    This provider handles unstructured HEALPix grid data using cKDTree
    for efficient spatial lookups and inverse distance weighting for interpolation.
    """

    def __init__(self, source_config: dict, global_config: dict = None):
        super().__init__(source_config, global_config)
        self._spatial_indices = {}  # Cache for spatial indices

    @property
    def name(self) -> str:
        return "nextGEMS"

    @property
    def coordinate_system(self) -> str:
        return "healpix"

    def is_available(self) -> bool:
        """Check if nextGEMS data files exist."""
        input_files = self.source_config.get('input_files', {})
        if not input_files:
            return False

        # Check if at least one file exists
        for file_key, meta in input_files.items():
            file_name = meta.get('file_name', file_key)
            if os.path.exists(file_name):
                return True
        return False

    def _build_spatial_index(self, nc_file: str) -> Tuple[cKDTree, np.ndarray, np.ndarray]:
        """Build a spatial index (cKDTree) using longitude and latitude from a Healpix NetCDF file.

        Args:
            nc_file: Path to the NetCDF file

        Returns:
            Tuple of (tree, lons, lats)
        """
        if nc_file in self._spatial_indices:
            return self._spatial_indices[nc_file]

        dataset = xr.open_dataset(nc_file)
        lons = dataset['lon'].values
        lats = dataset['lat'].values
        dataset.close()

        points = np.column_stack((lons, lats))
        tree = cKDTree(points)

        self._spatial_indices[nc_file] = (tree, lons, lats)
        return tree, lons, lats

    def _extract_data_healpix(
        self,
        nc_file: str,
        desired_lon: float,
        desired_lat: float,
        tree: cKDTree,
        lons: np.ndarray,
        lats: np.ndarray,
        months: Optional[List[int]] = None,
        variable_mapping: Optional[Dict[str, str]] = None
    ) -> Tuple[pd.DataFrame, Dict]:
        """Extract and interpolate data for a specific location from a Healpix NetCDF file.

        Uses inverse distance weighting from 4 nearest neighbors.
        """
        dataset = xr.open_dataset(nc_file)

        # Normalize longitude to 0-360 range (HEALPix data uses 0-360)
        query_lon = desired_lon
        if desired_lon < 0:
            query_lon = desired_lon + 360

        # Query 4 nearest neighbors
        distances, indices = tree.query([query_lon, desired_lat], k=4)

        neighbors_lons = lons[indices]
        neighbors_lats = lats[indices]

        # Define stereographic projection centered at desired location
        pste = pyproj.Proj(
            proj="stere", errcheck=True, ellps='WGS84',
            lat_0=desired_lat, lon_0=desired_lon
        )

        # Project neighbors and compute distances
        neighbors_x, neighbors_y = pste(neighbors_lons, neighbors_lats)
        desired_x, desired_y = pste(desired_lon, desired_lat)

        dx = neighbors_x - desired_x
        dy = neighbors_y - desired_y
        distances_proj = np.hypot(dx, dy)

        # Handle months
        if months is None:
            months = list(range(1, 13))
        else:
            months = [int(m) for m in months]
        month_indices = [m - 1 for m in months]

        df = pd.DataFrame({'Month': [calendar.month_name[m] for m in months]})
        df_vars = {}

        # Calculate weights
        if np.any(distances_proj == 0):
            zero_index = np.where(distances_proj == 0)[0][0]
            use_exact = True
        else:
            inv_distances = 1.0 / distances_proj
            weights = inv_distances / inv_distances.sum()
            use_exact = False

        for full_var, file_var in (variable_mapping or {}).items():
            if file_var not in dataset.variables:
                warnings.warn(f"Variable '{file_var}' not found in '{nc_file}'. Skipping.")
                continue

            interpolated_values = []
            for month_index in month_indices:
                data_values = dataset[file_var][month_index, indices].values

                if use_exact:
                    interpolated_value = data_values[zero_index]
                else:
                    interpolated_value = np.dot(weights, data_values)
                interpolated_values.append(interpolated_value)

            df[file_var] = np.array(interpolated_values)
            df_vars[file_var] = {
                'name': file_var,
                'units': dataset[file_var].attrs.get('units', ''),
                'full_name': full_var,
                'long_name': dataset[file_var].attrs.get('long_name', '')
            }

        dataset.close()
        return df, df_vars

    def _post_process_data(
        self,
        df: pd.DataFrame,
        df_vars: Dict
    ) -> Tuple[pd.DataFrame, Dict]:
        """Apply unit conversions and calculate wind speed/direction."""
        df_processed = df.copy()
        df_vars_processed = df_vars.copy()

        # Temperature: K to °C
        if 'mean2t' in df_processed.columns:
            df_processed['mean2t'] = df_processed['mean2t'] - 273.15
            df_vars_processed['mean2t']['units'] = '°C'

        # Precipitation: m to mm/month
        if 'tp' in df_processed.columns:
            df_processed['tp'] = df_processed['tp'] * 1000.0
            df_vars_processed['tp']['units'] = 'mm/month'

        # Calculate wind speed and direction
        if 'wind_u' in df_processed.columns and 'wind_v' in df_processed.columns:
            wind_speed = np.sqrt(df_processed['wind_u']**2 + df_processed['wind_v']**2)
            wind_direction = (180.0 + np.degrees(
                np.arctan2(df_processed['wind_u'], df_processed['wind_v'])
            )) % 360

            df_vars_processed['wind_speed'] = {
                'name': 'wind_speed', 'units': 'm/s',
                'full_name': 'Wind Speed', 'long_name': 'Wind Speed'
            }
            df_vars_processed['wind_direction'] = {
                'name': 'wind_direction', 'units': '°',
                'full_name': 'Wind Direction', 'long_name': 'Wind Direction'
            }
            df_processed['wind_speed'] = wind_speed.round(2)
            df_processed['wind_direction'] = wind_direction.round(2)

        # Round numeric columns
        for var in df_processed.columns:
            if var != 'Month' and pd.api.types.is_numeric_dtype(df_processed[var]):
                df_processed[var] = df_processed[var].round(2)

        return df_processed, df_vars_processed

    def extract_data(
        self,
        lon: float,
        lat: float,
        months: Optional[List[int]] = None
    ) -> ClimateDataResult:
        """Extract climate data for a location from nextGEMS data."""
        input_files = self.source_config.get('input_files', {})
        variable_mapping = self.source_config.get('variable_mapping', {})

        if months is None:
            months = list(range(1, 13))

        df_list = []

        # Build spatial indices and extract data
        for file_key, meta in input_files.items():
            file_name = meta.get('file_name', file_key)
            coord_system = meta.get('coordinate_system', 'healpix').lower()

            if coord_system != 'healpix':
                warnings.warn(f"Coordinate system '{coord_system}' not supported by NextGEMSProvider")
                continue

            if not os.path.exists(file_name):
                warnings.warn(f"File '{file_name}' does not exist. Skipping.")
                continue

            tree, lons, lats = self._build_spatial_index(file_name)

            df_raw, df_vars = self._extract_data_healpix(
                file_name, lon, lat, tree, lons, lats, months, variable_mapping
            )
            df_processed, df_vars_processed = self._post_process_data(df_raw, df_vars)

            if df_processed is not None and not df_processed.empty:
                df_list.append({
                    'filename': file_name,
                    'years_of_averaging': meta.get('years_of_averaging', ''),
                    'description': meta.get('description', ''),
                    'dataframe': df_processed,
                    'extracted_vars': df_vars_processed,
                    'main': meta.get('is_main', False),
                    'source': meta.get('source', '')
                })

        # Prepare data_agent_response
        data_agent_response = self._prepare_data_agent_response(df_list)

        return ClimateDataResult(
            df_list=df_list,
            data_agent_response=data_agent_response,
            source_name=self.name,
            source_description=self.description
        )


class ICCPProvider(ClimateDataProvider):
    """Provider for ICCP climate data on regular lat/lon grid.

    Uses xarray bilinear interpolation on the regular 640x1312 lat/lon grid.
    Supports multiple time periods (2020-2099) from TCo319-DART SSP5-8.5 scenario.

    Data structure:
    - Coordinates: lat (640), lon (1312), time (12 months)
    - Variables: mean2t (temperature K), wind_u, wind_v (m/s), tp (precipitation m)
    - Longitude range: -180 to 180
    - Post-processing: K→°C, m→mm/month, wind speed/direction calculation
    """

    @property
    def name(self) -> str:
        return "ICCP"

    @property
    def coordinate_system(self) -> str:
        return "regular"

    def is_available(self) -> bool:
        """Check if ICCP data files exist.

        Currently returns False as data files are not yet available.
        """
        input_files = self.source_config.get('input_files', {})
        if not input_files:
            return False

        for file_key, meta in input_files.items():
            file_name = meta.get('file_name', file_key)
            if os.path.exists(file_name):
                return True
        return False

    def _extract_data_regular_grid(
        self,
        file_path: str,
        lon: float,
        lat: float,
        months: List[int],
        variable_mapping: Dict[str, str]
    ) -> Tuple[pd.DataFrame, Dict]:
        """Extract data from a single NetCDF file using bilinear interpolation.

        Args:
            file_path: Path to NetCDF file
            lon: Longitude (-180 to 180)
            lat: Latitude (-90 to 90)
            months: List of month indices (1-12)
            variable_mapping: Dict mapping display names to NetCDF variable names

        Returns:
            Tuple of (DataFrame with raw data, metadata dict)
        """
        import xarray as xr
        import calendar

        try:
            # Open NetCDF file
            ds = xr.open_dataset(file_path)

            # Normalize longitude to 0-360 range if dataset uses that convention
            query_lon = lon
            if lon < 0 and ds.lon.min() >= 0:
                query_lon = lon + 360

            # Use bilinear interpolation to get data at the exact point
            ds_interpolated = ds.interp(lat=lat, lon=query_lon, method='linear')

            # Extract variables for all 12 months
            month_names = [calendar.month_name[i] for i in range(1, 13)]

            # Build dataframe with Month column
            df_data = {'Month': month_names}
            df_vars = {}

            # Get variable mapping (reverse: NetCDF var -> display name)
            var_map = variable_mapping or {}

            # Extract each variable
            for display_name, nc_var in var_map.items():
                if nc_var in ds_interpolated.variables:
                    # Extract all 12 months
                    values = ds_interpolated[nc_var].values
                    if len(values) == 12:
                        df_data[nc_var] = values

                        # Get metadata from NetCDF
                        var_attrs = ds[nc_var].attrs
                        df_vars[nc_var] = {
                            'name': nc_var,
                            'units': var_attrs.get('units', ''),
                            'full_name': display_name,
                            'long_name': var_attrs.get('long_name', display_name)
                        }
                else:
                    warnings.warn(f"Variable '{nc_var}' not found in {file_path}")

            ds.close()

            df = pd.DataFrame(df_data)
            return df, df_vars

        except Exception as e:
            logger.error(f"Error extracting data from {file_path}: {e}")
            raise

    def _post_process_data(
        self,
        df: pd.DataFrame,
        df_vars: Dict
    ) -> Tuple[pd.DataFrame, Dict]:
        """Apply unit conversions and calculate wind speed/direction.

        Args:
            df: DataFrame with raw data
            df_vars: Metadata dict for variables

        Returns:
            Tuple of (processed DataFrame, updated metadata dict)
        """
        df_processed = df.copy()
        df_vars_processed = df_vars.copy()

        # Temperature: K to °C
        if 'mean2t' in df_processed.columns:
            df_processed['mean2t'] = df_processed['mean2t'] - 273.15
            df_vars_processed['mean2t']['units'] = '°C'

        # Precipitation: m to mm/month
        if 'tp' in df_processed.columns:
            df_processed['tp'] = df_processed['tp'] * 1000.0
            df_vars_processed['tp']['units'] = 'mm/month'

        # Calculate wind speed and direction
        if 'wind_u' in df_processed.columns and 'wind_v' in df_processed.columns:
            wind_speed = np.sqrt(df_processed['wind_u']**2 + df_processed['wind_v']**2)
            wind_direction = (180.0 + np.degrees(
                np.arctan2(df_processed['wind_u'], df_processed['wind_v'])
            )) % 360

            df_vars_processed['wind_speed'] = {
                'name': 'wind_speed', 'units': 'm/s',
                'full_name': 'Wind Speed', 'long_name': 'Wind Speed'
            }
            df_vars_processed['wind_direction'] = {
                'name': 'wind_direction', 'units': '°',
                'full_name': 'Wind Direction', 'long_name': 'Wind Direction'
            }
            df_processed['wind_speed'] = wind_speed.round(2)
            df_processed['wind_direction'] = wind_direction.round(2)

        # Round numeric columns
        for var in df_processed.columns:
            if var != 'Month' and pd.api.types.is_numeric_dtype(df_processed[var]):
                df_processed[var] = df_processed[var].round(2)

        return df_processed, df_vars_processed

    def extract_data(
        self,
        lon: float,
        lat: float,
        months: Optional[List[int]] = None
    ) -> ClimateDataResult:
        """Extract climate data for a location from ICCP data files.

        Args:
            lon: Longitude (-180 to 180)
            lat: Latitude (-90 to 90)
            months: Optional list of month indices (1-12), defaults to all months

        Returns:
            ClimateDataResult with extracted data from all ICCP files
        """
        input_files = self.source_config.get('input_files', {})
        variable_mapping = self.source_config.get('variable_mapping', {})

        if months is None:
            months = list(range(1, 13))

        df_list = []

        # Extract data from each file
        for file_key, meta in input_files.items():
            file_name = meta.get('file_name', file_key)

            if not os.path.exists(file_name):
                warnings.warn(f"File '{file_name}' does not exist. Skipping.")
                continue

            try:
                # Extract raw data using bilinear interpolation
                df_raw, df_vars = self._extract_data_regular_grid(
                    file_name, lon, lat, months, variable_mapping
                )

                # Apply post-processing (unit conversions, wind calculations)
                df_processed, df_vars_processed = self._post_process_data(df_raw, df_vars)

                if df_processed is not None and not df_processed.empty:
                    df_list.append({
                        'filename': file_key,
                        'years_of_averaging': meta.get('years_of_averaging', ''),
                        'description': meta.get('description', ''),
                        'dataframe': df_processed,
                        'extracted_vars': df_vars_processed,
                        'main': meta.get('is_main', False),
                        'source': meta.get('source', 'ICCP model')
                    })

            except Exception as e:
                logger.error(f"Error processing {file_name}: {e}")
                warnings.warn(f"Skipping {file_name} due to error: {e}")
                continue

        # Prepare data_agent_response using inherited method
        data_agent_response = self._prepare_data_agent_response(df_list)

        return ClimateDataResult(
            df_list=df_list,
            data_agent_response=data_agent_response,
            source_name=self.name,
            source_description=self.description
        )


class AWICMProvider(ClimateDataProvider):
    """Provider for AWI-CM CMIP6 climate model data.

    This provider wraps the existing climate_functions.py implementation
    for backwards compatibility with the AWI-CM data format.
    """

    def __init__(self, source_config: dict, global_config: dict = None):
        super().__init__(source_config, global_config)
        self._data_cache = {}  # Cache for loaded datasets

    @property
    def name(self) -> str:
        return "AWI_CM"

    @property
    def coordinate_system(self) -> str:
        return "regular"

    def is_available(self) -> bool:
        """Check if AWI-CM data files exist."""
        data_path = self.source_config.get('data_path', './data/')
        historical_pattern = self.source_config.get('historical_pattern', 'historical')

        # Check for at least one historical file
        import glob
        hist_files = glob.glob(os.path.join(data_path, f"*{historical_pattern}*.nc"))
        return len(hist_files) > 0

    def _get_legacy_config(self) -> dict:
        """Create a config dict compatible with legacy climate_functions.py."""
        return {
            'data_settings': {
                'data_path': self.source_config.get('data_path', './data/'),
                'historical': self.source_config.get('historical_pattern', 'historical'),
                'projection': self.source_config.get('projection_pattern', 'ssp585'),
            },
            'climatemodel_name': self.source_config.get('climatemodel_name', 'AWI_CM'),
            'variable_mappings': self.source_config.get('variable_mapping', {
                'Temperature': 'tas',
                'Precipitation': 'pr',
                'u_wind': 'uas',
                'v_wind': 'vas'
            }),
            'dimension_mappings': self.source_config.get('dimension_mappings', {
                'latitude': 'lat',
                'longitude': 'lon',
                'time': 'month'
            })
        }

    def extract_data(
        self,
        lon: float,
        lat: float,
        months: Optional[List[int]] = None
    ) -> ClimateDataResult:
        """Extract climate data from AWI-CM data using legacy functions."""
        legacy_config = self._get_legacy_config()

        # Load data (cached)
        cache_key = 'awi_cm_data'
        if cache_key not in self._data_cache:
            hist, future = load_data(legacy_config)
            self._data_cache[cache_key] = (hist, future)
        else:
            hist, future = self._data_cache[cache_key]

        # Extract climate data
        df_data, data_dict = extract_climate_data(lat, lon, hist, future, legacy_config)

        # Convert to ClimateDataResult format
        # Create df_list with historical and future as separate entries
        df_list = [
            {
                'filename': 'AWI-CM historical',
                'years_of_averaging': '1995-2014',
                'description': 'AWI-CM-1-1-MR historical simulation',
                'dataframe': df_data[['Month', 'Present Day Temperature', 'Present Day Precipitation', 'Present Day Wind Speed']].copy() if 'Present Day Wind Speed' in df_data.columns else df_data[['Month', 'Present Day Temperature', 'Present Day Precipitation']].copy(),
                'extracted_vars': {
                    'Present Day Temperature': {'name': 'tas', 'units': '°C', 'full_name': 'Temperature', 'long_name': 'Near-Surface Air Temperature'},
                    'Present Day Precipitation': {'name': 'pr', 'units': 'mm/month', 'full_name': 'Precipitation', 'long_name': 'Precipitation'},
                    'Present Day Wind Speed': {'name': 'wind', 'units': 'm/s', 'full_name': 'Wind Speed', 'long_name': 'Wind Speed'}
                },
                'main': True,
                'source': 'AWI-CM-1-1-MR, scenarios: historical and SSP5-8.5'
            },
            {
                'filename': 'AWI-CM SSP5-8.5',
                'years_of_averaging': '2081-2100',
                'description': 'AWI-CM-1-1-MR SSP5-8.5 projection',
                'dataframe': df_data[['Month', 'Future Temperature', 'Future Precipitation', 'Future Wind Speed']].copy() if 'Future Wind Speed' in df_data.columns else df_data[['Month', 'Future Temperature', 'Future Precipitation']].copy(),
                'extracted_vars': {
                    'Future Temperature': {'name': 'tas', 'units': '°C', 'full_name': 'Temperature', 'long_name': 'Near-Surface Air Temperature'},
                    'Future Precipitation': {'name': 'pr', 'units': 'mm/month', 'full_name': 'Precipitation', 'long_name': 'Precipitation'},
                    'Future Wind Speed': {'name': 'wind', 'units': 'm/s', 'full_name': 'Wind Speed', 'long_name': 'Wind Speed'}
                },
                'main': False,
                'source': 'AWI-CM-1-1-MR, scenarios: historical and SSP5-8.5'
            }
        ]

        # Create data_agent_response in legacy format
        data_agent_response = {
            'content_message': """
            Current mean monthly temperature for each month: {hist_temp_str}
            Future monthly temperatures for each month at the location: {future_temp_str}
            Current precipitation flux (mm/month): {hist_pr_str}
            Future precipitation flux (mm/month): {future_pr_str}
            Current u wind component (in m/s): {hist_uas_str}
            Future u wind component (in m/s): {future_uas_str}
            Current v wind component (in m/s): {hist_vas_str}
            Future v wind component (in m/s): {future_vas_str}
            """,
            'input_params': {
                'hist_temp_str': data_dict.get('hist_Temperature', ''),
                'future_temp_str': data_dict.get('future_Temperature', ''),
                'hist_pr_str': data_dict.get('hist_Precipitation', ''),
                'future_pr_str': data_dict.get('future_Precipitation', ''),
                'hist_uas_str': data_dict.get('hist_u_wind', ''),
                'future_uas_str': data_dict.get('future_u_wind', ''),
                'hist_vas_str': data_dict.get('hist_v_wind', ''),
                'future_vas_str': data_dict.get('future_v_wind', ''),
            }
        }

        return ClimateDataResult(
            df_list=df_list,
            data_agent_response=data_agent_response,
            source_name=self.name,
            source_description=self.description
        )


# Factory functions

def get_climate_data_provider(
    config: dict,
    source_override: Optional[str] = None
) -> ClimateDataProvider:
    """Factory function to get the appropriate climate data provider.

    Args:
        config: Global configuration dictionary
        source_override: Optional source name to override config setting

    Returns:
        ClimateDataProvider instance for the requested source

    Raises:
        ValueError: If the requested source is unknown
    """
    source = source_override or config.get('climate_data_source', 'nextGEMS')
    sources_config = config.get('climate_data_sources', {})

    if source == 'nextGEMS':
        return NextGEMSProvider(sources_config.get('nextGEMS', {}), config)
    elif source == 'ICCP':
        return ICCPProvider(sources_config.get('ICCP', {}), config)
    elif source == 'AWI_CM':
        return AWICMProvider(sources_config.get('AWI_CM', {}), config)
    else:
        raise ValueError(f"Unknown climate data source: {source}")


def get_available_providers(config: dict) -> List[str]:
    """Return list of available (enabled and data present) providers.

    This function checks each provider to see if it has data available
    and returns the names of providers that can be used.

    Args:
        config: Global configuration dictionary

    Returns:
        List of provider names that are available
    """
    available = []
    for source in ['nextGEMS', 'ICCP', 'AWI_CM']:
        try:
            provider = get_climate_data_provider(config, source)
            if provider.is_available():
                available.append(source)
        except Exception as e:
            logger.debug(f"Provider {source} not available: {e}")
    return available


def migrate_legacy_config(config: dict) -> dict:
    """Legacy compatibility function - now just returns config as-is.

    This function is kept for backwards compatibility with existing code
    that calls it, but no longer performs any migration since legacy
    config format is no longer supported.

    Args:
        config: Configuration dictionary

    Returns:
        The same configuration dictionary (unchanged)
    """
    return config
