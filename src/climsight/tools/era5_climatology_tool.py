"""Tool for extracting ERA5 climatology data from pre-computed Zarr file.

This tool provides OBSERVED climate data (ground truth) from ERA5 reanalysis.
It should be called FIRST before analyzing climate model projections.
"""

import json
import logging
import os
from typing import List, Optional, Union

import numpy as np
import xarray as xr
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Default path to ERA5 climatology file
DEFAULT_ERA5_CLIMATOLOGY_PATH = "data/era5/era5_climatology_2015_2025.zarr"

# Variable name mapping: agent-friendly names -> ERA5 variable names
VARIABLE_ALIASES = {
    # Temperature
    "temperature": "t2m",
    "temp": "t2m",
    "2m_temperature": "t2m",
    "t2m": "t2m",
    # Dewpoint
    "dewpoint": "d2m",
    "dewpoint_temperature": "d2m",
    "2m_dewpoint": "d2m",
    "d2m": "d2m",
    # Precipitation
    "precipitation": "tp",
    "precip": "tp",
    "total_precipitation": "tp",
    "tp": "tp",
    # Wind U component
    "wind_u": "u10",
    "u_wind": "u10",
    "10m_u_wind": "u10",
    "u10": "u10",
    # Wind V component
    "wind_v": "v10",
    "v_wind": "v10",
    "10m_v_wind": "v10",
    "v10": "v10",
    # Pressure
    "pressure": "msl",
    "mean_sea_level_pressure": "msl",
    "msl": "msl",
    # Surface pressure
    "surface_pressure": "sp",
    "sp": "sp",
    # Sea surface temperature
    "sea_surface_temperature": "sst",
    "sst": "sst",
}

# Variable metadata
# Note: tp in ERA5 Zarr is stored as m/day (daily average rate), converted to mm/month
VARIABLE_INFO = {
    "t2m": {"full_name": "2 metre temperature", "units": "K", "convert_to_celsius": True},
    "d2m": {"full_name": "2 metre dewpoint temperature", "units": "K", "convert_to_celsius": True},
    "tp": {"full_name": "Total precipitation", "units": "m/day", "convert_to_mm": True},
    "u10": {"full_name": "10 metre U wind component", "units": "m/s", "convert_to_celsius": False},
    "v10": {"full_name": "10 metre V wind component", "units": "m/s", "convert_to_celsius": False},
    "msl": {"full_name": "Mean sea level pressure", "units": "Pa", "convert_to_celsius": False},
    "sp": {"full_name": "Surface pressure", "units": "Pa", "convert_to_celsius": False},
    "sst": {"full_name": "Sea surface temperature", "units": "K", "convert_to_celsius": True},
}

MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]

# Days per month (climatological average, Feb=28.25 for leap year average)
DAYS_IN_MONTH = [31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


def extract_era5_climatology_direct(
    latitude: float,
    longitude: float,
    era5_path: str = None,
    variables: List[str] = None,
    save_to_dir: str = None
) -> dict:
    """
    Extract ERA5 climatology directly without using the tool wrapper.

    This function can be called before agent runs to get ERA5 data for predefined plots.

    Args:
        latitude: Latitude of the location
        longitude: Longitude of the location
        era5_path: Path to ERA5 climatology Zarr file (uses default if not provided)
        variables: List of variables to extract (defaults to t2m, tp, u10, v10)
        save_to_dir: Optional directory to save the result JSON

    Returns:
        Dictionary with ERA5 climatology data or error dict
    """
    if era5_path is None:
        era5_path = DEFAULT_ERA5_CLIMATOLOGY_PATH

    if variables is None:
        variables = ["t2m", "tp", "u10", "v10"]  # Default: temp, precip, wind

    # Normalize variable names
    normalized_vars = []
    for var in variables:
        var_lower = var.lower().strip()
        if var_lower in VARIABLE_ALIASES:
            normalized_vars.append(VARIABLE_ALIASES[var_lower])
        else:
            logger.warning(f"Unknown variable: {var}. Skipping.")

    if not normalized_vars:
        return {
            "error": "No valid variables specified.",
            "available_variables": list(VARIABLE_INFO.keys())
        }

    # Remove duplicates while preserving order
    normalized_vars = list(dict.fromkeys(normalized_vars))

    # Check if ERA5 file exists
    if not os.path.exists(era5_path):
        return {
            "error": f"ERA5 climatology file not found at: {era5_path}",
            "suggestion": "Check config.yml era5_climatology.path setting"
        }

    try:
        # Open the Zarr dataset
        ds = xr.open_zarr(era5_path)

        # Normalize longitude to 0-360 range (ERA5 convention)
        lon_normalized = _normalize_longitude(longitude)

        # Find nearest point
        nearest = ds.sel(
            latitude=latitude,
            longitude=lon_normalized,
            method="nearest"
        )

        actual_lat = float(nearest.latitude.values)
        actual_lon = float(nearest.longitude.values)

        # Convert actual longitude back to -180 to 180 if needed
        actual_lon_display = actual_lon if actual_lon <= 180 else actual_lon - 360

        # Calculate distance to nearest point
        distance_km = _haversine_distance(latitude, longitude, actual_lat, actual_lon_display)

        # Extract data for each variable
        variables_data = {}
        for var_name in normalized_vars:
            if var_name not in ds.data_vars:
                logger.warning(f"Variable {var_name} not in dataset. Available: {list(ds.data_vars)}")
                continue

            var_info = VARIABLE_INFO.get(var_name, {})
            # Force compute if dask array and convert to numpy
            raw_data = nearest[var_name]
            if hasattr(raw_data, 'compute'):
                raw_data = raw_data.compute()
            values = np.array(raw_data.values, dtype=np.float64)

            # Convert units if needed
            if var_info.get("convert_to_celsius", False):
                values = values - 273.15  # K to °C
                units = "°C"
            elif var_info.get("convert_to_mm", False):
                # ERA5 tp is in m/day (daily average rate), convert to mm/month
                values_monthly = np.array([
                    values[i] * 1000.0 * DAYS_IN_MONTH[i]
                    for i in range(len(values))
                ])
                values = values_monthly
                units = "mm/month"
            else:
                units = var_info.get("units", "")

            logger.debug(f"Variable {var_name}: raw range [{np.min(values):.4f}, {np.max(values):.4f}] {units}")

            # Build monthly values dict
            monthly_values = {}
            for i, month_name in enumerate(MONTH_NAMES):
                monthly_values[month_name] = round(float(values[i]), 2)

            variables_data[var_name] = {
                "full_name": var_info.get("full_name", var_name),
                "units": units,
                "monthly_values": monthly_values
            }

        ds.close()

        # Build result
        result = {
            "source": "ERA5 Reanalysis (ECMWF)",
            "data_type": "OBSERVATIONS (ground truth)",
            "period": "2015-2025 monthly climatology (10-year average)",
            "resolution": "0.25° grid (~28 km)",
            "description": (
                "This is OBSERVED climate data from ERA5 reanalysis. "
                "Use these values as the GROUND TRUTH baseline. "
                "Climate model projections should be compared against this observational data."
            ),
            "requested_location": {
                "latitude": latitude,
                "longitude": longitude
            },
            "extracted_location": {
                "latitude": actual_lat,
                "longitude": actual_lon_display
            },
            "distance_from_requested_km": round(distance_km, 1),
            "note": (
                f"ERA5 grid resolution is 0.25° (~28km). "
                f"Data extracted from nearest grid point, {round(distance_km, 1)} km from requested location."
            ),
            "variables": variables_data,
            "usage_guidance": {
                "comparison": "Compare ERA5 values with climate model historical period to assess model bias",
                "baseline": "Use ERA5 as the 'current climate' baseline (what we observe NOW)",
                "interpretation": "ERA5 represents actual observed conditions, climate models are projections"
            },
            "reference": "Hersbach, H., Bell, B., Berrisford, P., et al. (2020). The ERA5 global reanalysis. Q.J.R. Meteorol. Soc., 146, 1999-2049. https://doi.org/10.1002/qj.3803"
        }

        # Save to directory if provided
        if save_to_dir:
            output_path = os.path.join(save_to_dir, "era5_climatology.json")
            try:
                import json
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2)
                logger.info(f"ERA5 climatology saved to: {output_path}")
                result["saved_to"] = output_path
            except Exception as e:
                logger.warning(f"Could not save ERA5 climatology: {e}")

        return result

    except Exception as e:
        logger.error(f"Error extracting ERA5 climatology: {e}")
        return {
            "error": f"Failed to extract ERA5 climatology: {str(e)}",
            "latitude": latitude,
            "longitude": longitude,
            "variables_requested": normalized_vars
        }


def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance in km between two points using Haversine formula."""
    R = 6371  # Earth's radius in km

    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)

    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

    return R * c


def _normalize_longitude(lon: float) -> float:
    """Normalize longitude to 0-360 range (ERA5 uses 0-360)."""
    if lon < 0:
        return lon + 360
    return lon


class GetERA5ClimatologyArgs(BaseModel):
    latitude: float = Field(
        description="Latitude of the location to extract data for (decimal degrees, -90 to 90)"
    )
    longitude: float = Field(
        description="Longitude of the location to extract data for (decimal degrees, -180 to 180)"
    )
    variables: Optional[Union[str, List[str]]] = Field(
        default=None,
        description=(
            "List of variables to extract. Options: temperature (t2m), precipitation (tp), "
            "wind_u (u10), wind_v (v10), dewpoint (d2m), pressure (msl), surface_pressure (sp), sst. "
            "If not specified, extracts temperature and precipitation by default."
        )
    )


def create_era5_climatology_tool(state, config, stream_handler=None):
    """Create a StructuredTool for extracting ERA5 climatology data.

    Args:
        state: AgentState with sandbox paths
        config: Configuration dict
        stream_handler: Optional progress handler

    Returns:
        StructuredTool instance
    """
    try:
        from langchain.tools import StructuredTool
    except ImportError:
        from langchain_core.tools import StructuredTool

    # Get ERA5 climatology path from config or use default
    era5_config = config.get("era5_climatology", {})
    era5_path = era5_config.get("path", DEFAULT_ERA5_CLIMATOLOGY_PATH)

    def get_era5_climatology(
        latitude: float,
        longitude: float,
        variables: Optional[Union[str, List[str]]] = None
    ) -> dict:
        """Extract ERA5 climatology for a specific location.

        This provides OBSERVED climate data (2015-2025 average) as ground truth.
        Use this data as the baseline to compare against climate model projections.
        """
        if stream_handler is not None:
            stream_handler.update_progress("Extracting ERA5 observational climatology...")

        # Parse variables input
        if variables is None:
            variables = ["t2m", "tp"]  # Default: temperature and precipitation
        elif isinstance(variables, str):
            # Handle string input (might be JSON list or comma-separated)
            try:
                import ast
                variables = ast.literal_eval(variables)
            except (ValueError, SyntaxError):
                variables = [v.strip() for v in variables.split(",")]

        # Normalize variable names
        normalized_vars = []
        for var in variables:
            var_lower = var.lower().strip()
            if var_lower in VARIABLE_ALIASES:
                normalized_vars.append(VARIABLE_ALIASES[var_lower])
            else:
                logger.warning(f"Unknown variable: {var}. Skipping.")

        if not normalized_vars:
            return {
                "error": "No valid variables specified.",
                "available_variables": list(VARIABLE_INFO.keys())
            }

        # Remove duplicates while preserving order
        normalized_vars = list(dict.fromkeys(normalized_vars))

        # Check if ERA5 file exists
        if not os.path.exists(era5_path):
            return {
                "error": f"ERA5 climatology file not found at: {era5_path}",
                "suggestion": "Check config.yml era5_climatology.path setting"
            }

        try:
            # Open the Zarr dataset
            ds = xr.open_zarr(era5_path)

            # Normalize longitude to 0-360 range (ERA5 convention)
            lon_normalized = _normalize_longitude(longitude)

            # Find nearest point
            nearest = ds.sel(
                latitude=latitude,
                longitude=lon_normalized,
                method="nearest"
            )

            actual_lat = float(nearest.latitude.values)
            actual_lon = float(nearest.longitude.values)

            # Convert actual longitude back to -180 to 180 if needed
            actual_lon_display = actual_lon if actual_lon <= 180 else actual_lon - 360

            # Calculate distance to nearest point
            distance_km = _haversine_distance(latitude, longitude, actual_lat, actual_lon_display)

            # Extract data for each variable
            variables_data = {}
            for var_name in normalized_vars:
                if var_name not in ds.data_vars:
                    logger.warning(f"Variable {var_name} not in dataset. Available: {list(ds.data_vars)}")
                    continue

                var_info = VARIABLE_INFO.get(var_name, {})
                # Force compute if dask array and convert to numpy
                raw_data = nearest[var_name]
                if hasattr(raw_data, 'compute'):
                    raw_data = raw_data.compute()
                values = np.array(raw_data.values, dtype=np.float64)

                # Convert units if needed
                if var_info.get("convert_to_celsius", False):
                    values = values - 273.15  # K to °C
                    units = "°C"
                elif var_info.get("convert_to_mm", False):
                    # ERA5 tp is in m/day (daily average rate), convert to mm/month
                    # Multiply by 1000 (m->mm) and by days in each month
                    values_monthly = np.array([
                        values[i] * 1000.0 * DAYS_IN_MONTH[i]
                        for i in range(len(values))
                    ])
                    values = values_monthly
                    units = "mm/month"
                else:
                    units = var_info.get("units", "")

                logger.debug(f"Variable {var_name}: raw range [{np.min(values):.4f}, {np.max(values):.4f}] {units}")

                # Build monthly values dict
                monthly_values = {}
                for i, month_name in enumerate(MONTH_NAMES):
                    monthly_values[month_name] = round(float(values[i]), 2)

                variables_data[var_name] = {
                    "full_name": var_info.get("full_name", var_name),
                    "units": units,
                    "monthly_values": monthly_values
                }

            ds.close()

            # Build result
            result = {
                "source": "ERA5 Reanalysis (ECMWF)",
                "data_type": "OBSERVATIONS (ground truth)",
                "period": "2015-2025 monthly climatology (10-year average)",
                "resolution": "0.25° grid (~28 km)",
                "description": (
                    "This is OBSERVED climate data from ERA5 reanalysis. "
                    "Use these values as the GROUND TRUTH baseline. "
                    "Climate model projections should be compared against this observational data."
                ),
                "requested_location": {
                    "latitude": latitude,
                    "longitude": longitude
                },
                "extracted_location": {
                    "latitude": actual_lat,
                    "longitude": actual_lon_display
                },
                "distance_from_requested_km": round(distance_km, 1),
                "note": (
                    f"ERA5 grid resolution is 0.25° (~28km). "
                    f"Data extracted from nearest grid point, {round(distance_km, 1)} km from requested location."
                ),
                "variables": variables_data,
                "usage_guidance": {
                    "comparison": "Compare ERA5 values with climate model historical period to assess model bias",
                    "baseline": "Use ERA5 as the 'current climate' baseline (what we observe NOW)",
                    "interpretation": "ERA5 represents actual observed conditions, climate models are projections"
                },
                "reference": "Hersbach, H., Bell, B., Berrisford, P., et al. (2020). The ERA5 global reanalysis. Q.J.R. Meteorol. Soc., 146, 1999-2049. https://doi.org/10.1002/qj.3803"
            }

            # Save to sandbox if available
            if hasattr(state, 'uuid_main_dir') and state.uuid_main_dir:
                output_path = os.path.join(state.uuid_main_dir, "era5_climatology.json")
                try:
                    with open(output_path, "w", encoding="utf-8") as f:
                        json.dump(result, f, indent=2)
                    logger.info(f"ERA5 climatology saved to: {output_path}")
                    result["saved_to"] = output_path
                except Exception as e:
                    logger.warning(f"Could not save ERA5 climatology: {e}")

            # Also store in state
            if hasattr(state, 'era5_climatology_response'):
                state.era5_climatology_response = result

            return result

        except Exception as e:
            logger.error(f"Error extracting ERA5 climatology: {e}")
            return {
                "error": f"Failed to extract ERA5 climatology: {str(e)}",
                "latitude": latitude,
                "longitude": longitude,
                "variables_requested": normalized_vars
            }

    return StructuredTool.from_function(
        func=get_era5_climatology,
        name="get_era5_climatology",
        description=(
            "Extract OBSERVED climate data from ERA5 reanalysis (2015-2025 climatology). "
            "This provides GROUND TRUTH observations - call this FIRST before analyzing climate model data. "
            "Returns monthly averages for temperature, precipitation, wind, etc. at the specified location."
        ),
        args_schema=GetERA5ClimatologyArgs,
    )
