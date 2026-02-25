# src/climsight/tools/era5_retrieval_tool.py
"""
ERA5 data retrieval tool for use in the visualization agent.
Retrieves ERA5 Surface climate data from Earthmover (Arraylake).
Saves the retrieved data locally in Zarr format.
Hardcoded to 'temporal' query mode for efficient time-series retrieval.
Uses 'nearest' neighbor selection for point coordinates to prevent empty spatial slices.
"""

import os
import sys
import logging
import shutil
import xarray as xr
import pandas as pd
from pydantic import BaseModel, Field
from typing import Optional, Literal
from langchain_core.tools import StructuredTool

# --- IMPORTS & CONFIGURATION ---
try:
    import zarr
    import arraylake
    from arraylake import Client
except ImportError as e:
    install_command = "pip install --upgrade xarray zarr arraylake pandas numpy pydantic langchain-core"
    raise ImportError(
        f"Required libraries missing. Please ensure arraylake is installed.\n"
        f"Try running: {install_command}"
    ) from e

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# =============================================================================
# EARTHMOVER (ARRAYLAKE) IMPLEMENTATION
# =============================================================================

# Variable Mapping: Maps friendly names to Earthmover short codes
# IMPORTANT: These are the ONLY variables available in the Arraylake ERA5 surface dataset.
# 'tp' (total_precipitation) does NOT exist in this dataset.
# For precipitation, use 'cp' (convective) or 'lsp' (large-scale) instead.
VARIABLE_MAPPING = {
    # Temperature
    "sea_surface_temperature": "sst",
    "2m_temperature": "t2",
    "temperature": "t2",
    "skin_temperature": "skt",
    "dewpoint_temperature": "d2",
    "2m_dewpoint_temperature": "d2",

    # Wind
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "u_component_of_wind": "u10",
    "v_component_of_wind": "v10",

    # Pressure
    "surface_pressure": "sp",
    "mean_sea_level_pressure": "mslp",

    # Clouds/Precip/Snow
    "total_cloud_cover": "tcc",
    "convective_precipitation": "cp",
    "large_scale_precipitation": "lsp",
    "snow_depth": "sd",

    # Identity mappings (so short codes also work directly)
    "t2": "t2", "sst": "sst", "mslp": "mslp", "u10": "u10", "v10": "v10",
    "sp": "sp", "tcc": "tcc", "cp": "cp", "lsp": "lsp", "sd": "sd",
    "skt": "skt", "d2": "d2"
}

class ERA5RetrievalArgs(BaseModel):
    # query_type is removed from the agent's view (hardcoded internally)
    variable_id: Literal[
        "t2", "sst", "mslp", "u10", "v10", "sp", "tcc", "cp", "lsp", "sd", "skt", "d2",
        "sea_surface_temperature", "surface_pressure", "total_cloud_cover",
        "convective_precipitation", "large_scale_precipitation", "snow_depth",
        "10m_u_component_of_wind", "10m_v_component_of_wind", "2m_temperature", "2m_dewpoint_temperature",
        "temperature", "u_component_of_wind", "v_component_of_wind", "skin_temperature", "dewpoint_temperature",
        "mean_sea_level_pressure"
    ] = Field(description=(
        "ERA5 variable to retrieve from Arraylake. Use short codes. "
        "AVAILABLE VARIABLES (short code - description): "
        "'t2' = 2m air temperature (K); "
        "'d2' = 2m dewpoint temperature (K); "
        "'sst' = sea surface temperature (K, ocean only); "
        "'skt' = skin temperature (K, surface radiative temp); "
        "'u10' = 10m U-component of wind (m/s, eastward); "
        "'v10' = 10m V-component of wind (m/s, northward); "
        "'mslp' = mean sea level pressure (Pa); "
        "'sp' = surface pressure (Pa); "
        "'tcc' = total cloud cover (fraction 0-1); "
        "'cp' = convective precipitation (m, use for convective rain/showers); "
        "'lsp' = large-scale precipitation (m, use for stratiform/frontal rain); "
        "'sd' = snow depth (m water equivalent). "
        "WARNING: 'tp' (total_precipitation) does NOT exist in this dataset. "
        "For total precipitation, retrieve BOTH 'cp' and 'lsp' and sum them."
    ))

    start_date: str = Field(description="Start date (YYYY-MM-DD). Data available 1979-2024.")
    end_date: str = Field(description="End date (YYYY-MM-DD).")

    # Coordinates
    min_latitude: float = Field(-90.0, description="Minimum latitude. For a specific point, use the same value as max_latitude.")
    max_latitude: float = Field(90.0, description="Maximum latitude. For a specific point, use the same value as min_latitude.")
    min_longitude: float = Field(0.0, description="Minimum longitude. For a specific point, use the same value as max_longitude.")
    max_longitude: float = Field(359.75, description="Maximum longitude. For a specific point, use the same value as min_longitude.")

    work_dir: Optional[str] = Field(None, description="The absolute path to the working directory where data should be saved.")

def _generate_descriptive_filename(variable_id: str, query_type: str, start_date: str, end_date: str) -> str:
    """Generate a descriptive directory name for the Zarr store."""
    clean_var = variable_id.replace('_', '')
    clean_start = start_date.split()[0].replace('-', '')
    clean_end = end_date.split()[0].replace('-', '')
    # We use .zarr extension, but it is a directory
    return f"era5_{clean_var}_{query_type}_{clean_start}_{clean_end}.zarr"

def retrieve_era5_data(
    variable_id: str,
    start_date: str,
    end_date: str,
    min_latitude: float = -90.0,
    max_latitude: float = 90.0,
    min_longitude: float = 0.0,
    max_longitude: float = 359.75,
    work_dir: Optional[str] = None,
    arraylake_api_key: Optional[str] = None,
    **kwargs # Catch-all for unused args
) -> dict:
    """
    Retrieves ERA5 Surface data from Earthmover (Arraylake).
    Hardcoded to use 'temporal' (Time-series) queries.
    Uses Nearest Neighbor selection for points to avoid empty slices.

    Args:
        arraylake_api_key: Arraylake API key. If not provided, falls back to
                          ARRAYLAKE_API_KEY environment variable.
    """
    ds = None
    local_zarr_path = None
    query_type = "temporal" # Hardcoded for Climsight point-data focus

    # Get API Key - prefer passed parameter, fall back to environment
    ARRAYLAKE_API_KEY = arraylake_api_key or os.environ.get("ARRAYLAKE_API_KEY")

    if not ARRAYLAKE_API_KEY:
        return {"success": False, "error": "Missing Arraylake API Key", "message": "Please provide arraylake_api_key parameter or set ARRAYLAKE_API_KEY environment variable."}

    try:
        # Map Variable Name
        short_var = VARIABLE_MAPPING.get(variable_id.lower(), variable_id)
        logging.info(f"🌍 Earthmover ERA5 Retrieval ({query_type}): {short_var} | {start_date} to {end_date}")

        # --- 1. Sandbox / Path Logic ---
        # Priority: 1) CLIMSIGHT_THREAD_ID env var, 2) work_dir, 3) default
        main_dir = None
        thread_id = os.environ.get("CLIMSIGHT_THREAD_ID")

        if thread_id:
            # Use sandbox path based on thread_id (set by sandbox_utils)
            main_dir = os.path.join("tmp", "sandbox", thread_id)

        # Fallback to work_dir if no sandbox available
        if not main_dir and work_dir:
            main_dir = work_dir

        if not main_dir:
            main_dir = os.path.join("tmp", "sandbox", "era5_default")

        os.makedirs(main_dir, exist_ok=True)

        # FIX: Prevent double nesting if work_dir already ends with 'era5_data'
        if os.path.basename(main_dir.rstrip(os.sep)) == "era5_data":
            era5_dir = main_dir
        else:
            era5_dir = os.path.join(main_dir, "era5_data")
        os.makedirs(era5_dir, exist_ok=True)

        # Check Cache
        zarr_dirname = _generate_descriptive_filename(short_var, query_type, start_date, end_date)
        local_zarr_path = os.path.join(era5_dir, zarr_dirname)
        absolute_zarr_path = os.path.abspath(local_zarr_path)

        if os.path.exists(local_zarr_path):
            logging.info(f"⚡ Cache hit: {local_zarr_path}")
            return {
                "success": True,
                "output_path_zarr": absolute_zarr_path,
                "full_path": absolute_zarr_path,
                "variable": short_var,
                "query_type": query_type,
                "message": f"Cached ERA5 data found at {absolute_zarr_path}",
                "reference": "Copernicus Climate Change Service (C3S): ERA5 hourly data on single levels from 1940 to present. https://doi.org/10.24381/cds.adbb2d47"
            }

        # --- 2. Connect to Earthmover ---
        logging.info("Connecting to Arraylake...")
        client = Client(token=ARRAYLAKE_API_KEY)
        repo_name = "earthmover-public/era5-surface-aws"
        repo = client.get_repo(repo_name)
        session = repo.readonly_session("main")

        # Open Dataset
        ds = xr.open_dataset(
            session.store, 
            engine="zarr", 
            consolidated=False, 
            zarr_format=3, 
            chunks=None, 
            group=query_type
        )

        if short_var not in ds: 
            return {"success": False, "error": f"Variable '{short_var}' not found. Available: {list(ds.data_vars)}"}

        # --- 3. Slicing & Selection ---
        start_datetime_obj = pd.to_datetime(start_date)
        end_datetime_obj = pd.to_datetime(end_date)
        time_slice = slice(start_datetime_obj, end_datetime_obj)

        # Check if it's a point query (min approx equal to max)
        is_point_query = (abs(max_latitude - min_latitude) < 0.01) and (abs(max_longitude - min_longitude) < 0.01)

        if is_point_query:
            # CRITICAL FIX: Cannot use method="nearest" with slice objects!
            # Must do spatial selection FIRST (with method="nearest"), THEN time slicing.
            center_lat = (min_latitude + max_latitude) / 2.0
            center_lon = (min_longitude + max_longitude) / 2.0

            logging.info(f"Point query detected: selecting nearest neighbor to {center_lat}, {center_lon}")

            # Step 1: Select nearest spatial point FIRST (no slices!)
            subset = ds[short_var].sel(
                latitude=center_lat,
                longitude=center_lon,
                method="nearest"
            )
            # Step 2: THEN slice by time
            subset = subset.sel(time=time_slice)
        else:
            # Box query - no method="nearest" needed, standard slicing
            # Earthmover lat is typically sorted Max -> Min
            req_min_lon = min_longitude % 360
            req_max_lon = max_longitude % 360

            if req_min_lon > req_max_lon:
                # Simple clamp for now
                lon_slice = slice(req_min_lon, 359.75)
            else:
                lon_slice = slice(req_min_lon, req_max_lon)

            subset = ds[short_var].sel(
                time=time_slice,
                latitude=slice(max_latitude, min_latitude),
                longitude=lon_slice
            )

        # Validate Data Existence
        if subset.sizes.get('time', 0) == 0:
            return {"success": False, "error": "Empty time slice.", "message": "No data found for the requested dates."}

        # --- 4. Save to Zarr (Atomic Write) ---
        logging.info(f"Downloading data to {local_zarr_path}...")

        ds_out = subset.to_dataset(name=short_var)
        
        # Clear encoding
        for var in ds_out.variables:
            ds_out[var].encoding = {}

        # Atomic write: write to temp dir first, then rename on success
        temp_zarr_path = local_zarr_path + ".tmp"
        
        # Clean up any previous failed temp directory
        if os.path.exists(temp_zarr_path):
            shutil.rmtree(temp_zarr_path)

        try:
            ds_out.to_zarr(temp_zarr_path, mode="w", consolidated=True, compute=True)
            
            # Atomic replace: remove old cache (if exists) and rename temp to final
            if os.path.exists(local_zarr_path):
                shutil.rmtree(local_zarr_path)
            os.rename(temp_zarr_path, local_zarr_path)
            logging.info("✅ Download complete.")
        except Exception as write_error:
            # Clean up failed temp directory
            if os.path.exists(temp_zarr_path):
                shutil.rmtree(temp_zarr_path, ignore_errors=True)
            raise write_error

        return {
            "success": True,
            "output_path_zarr": absolute_zarr_path,
            "full_path": absolute_zarr_path,
            "variable": short_var,
            "query_type": query_type,
            "message": f"ERA5 data ({query_type} optimized) retrieved and saved to {absolute_zarr_path}",
            "reference": "Copernicus Climate Change Service (C3S): ERA5 hourly data on single levels from 1940 to present. https://doi.org/10.24381/cds.adbb2d47"
        }

    except Exception as e:
        logging.error(f"Error in ERA5 Earthmover retrieval: {e}", exc_info=True)
        if local_zarr_path and os.path.exists(local_zarr_path):
            shutil.rmtree(local_zarr_path, ignore_errors=True)
        return {"success": False, "error": str(e), "message": f"Failed: {str(e)}"}
    finally:
        if ds is not None:
            ds.close()

def create_era5_retrieval_tool(arraylake_api_key: str):
    """
    Create the ERA5 retrieval tool with the API key bound.

    Args:
        arraylake_api_key: Arraylake API key for accessing Earthmover data

    Returns:
        StructuredTool configured for ERA5 data retrieval
    """
    def retrieve_era5_wrapper(
        variable_id: str,
        start_date: str,
        end_date: str,
        min_latitude: float = -90.0,
        max_latitude: float = 90.0,
        min_longitude: float = 0.0,
        max_longitude: float = 359.75,
        work_dir: Optional[str] = None,
    ) -> dict:
        return retrieve_era5_data(
            variable_id=variable_id,
            start_date=start_date,
            end_date=end_date,
            min_latitude=min_latitude,
            max_latitude=max_latitude,
            min_longitude=min_longitude,
            max_longitude=max_longitude,
            work_dir=work_dir,
            arraylake_api_key=arraylake_api_key,
        )

    return StructuredTool.from_function(
        func=retrieve_era5_wrapper,
        name="retrieve_era5_data",
        description=(
            "Retrieves ERA5 Surface climate data from Earthmover (Arraylake). "
            "Optimized for TEMPORAL time-series extraction at specific locations. "
            "Automatically snaps to nearest grid point to ensure data is returned. "
            "Returns a Zarr directory path. "
            "Available variables: t2 (2m air temperature), d2 (2m dewpoint temperature), "
            "sst (sea surface temperature), skt (skin temperature), "
            "u10 (10m U-wind), v10 (10m V-wind), "
            "mslp (mean sea level pressure), sp (surface pressure), "
            "tcc (total cloud cover), cp (convective precipitation), "
            "lsp (large-scale precipitation), sd (snow depth). "
            "WARNING: 'tp' does NOT exist. For total precipitation, retrieve 'cp' + 'lsp' and sum them."
        ),
        args_schema=ERA5RetrievalArgs
    )


# Keep backward-compatible module-level tool that uses environment variable
era5_retrieval_tool = StructuredTool.from_function(
    func=retrieve_era5_data,
    name="retrieve_era5_data",
    description=(
        "Retrieves ERA5 Surface climate data from Earthmover (Arraylake). "
        "Optimized for TEMPORAL time-series extraction at specific locations. "
        "Automatically snaps to nearest grid point to ensure data is returned. "
        "Returns a Zarr directory path. "
        "Available variables: t2 (2m air temperature), d2 (2m dewpoint temperature), "
        "sst (sea surface temperature), skt (skin temperature), "
        "u10 (10m U-wind), v10 (10m V-wind), "
        "mslp (mean sea level pressure), sp (surface pressure), "
        "tcc (total cloud cover), cp (convective precipitation), "
        "lsp (large-scale precipitation), sd (snow depth). "
        "WARNING: 'tp' does NOT exist. For total precipitation, retrieve 'cp' + 'lsp' and sum them."
    ),
    args_schema=ERA5RetrievalArgs
)