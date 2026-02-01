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
    # Optional: Check for Streamlit to support session state if available
    try:
        import streamlit as st
    except ImportError:
        st = None
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
VARIABLE_MAPPING = {
    # Temperature
    "sea_surface_temperature": "sst",
    "2m_temperature": "t2",
    "temperature": "t2",
    "skin_temperature": "skt",
    "dewpoint_temperature": "d2",

    # Wind
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "u_component_of_wind": "u10", 
    "v_component_of_wind": "v10",

    # Pressure
    "surface_pressure": "sp",
    "mean_sea_level_pressure": "mslp",

    # Clouds/Precip
    "total_cloud_cover": "tcc",
    "convective_precipitation": "cp",
    "large_scale_precipitation": "lsp",
    "total_precipitation": "tp",

    # Identity mappings (so short codes work)
    "t2": "t2", "sst": "sst", "mslp": "mslp", "u10": "u10", "v10": "v10",
    "sp": "sp", "tcc": "tcc", "cp": "cp", "lsp": "lsp", "sd": "sd", "tp": "tp"
}

class ERA5RetrievalArgs(BaseModel):
    # query_type is removed from the agent's view (hardcoded internally)
    variable_id: Literal[
        "t2", "sst", "mslp", "u10", "v10", "sp", "tcc", "cp", "lsp", "sd", "skt", "d2", "tp",
        "sea_surface_temperature", "surface_pressure", "total_cloud_cover", "total_precipitation",
        "10m_u_component_of_wind", "10m_v_component_of_wind", "2m_temperature", "2m_dewpoint_temperature",
        "temperature", "u_component_of_wind", "v_component_of_wind"
    ] = Field(description="ERA5 variable to retrieve. Preferred short codes: 't2' (Air Temp), 'sst' (Sea Surface Temp), 'u10'/'v10' (Wind), 'mslp' (Pressure), 'tp' (Total Precip).")

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
    **kwargs # Catch-all for unused args
) -> dict:
    """
    Retrieves ERA5 Surface data from Earthmover (Arraylake).
    Hardcoded to use 'temporal' (Time-series) queries.
    Uses Nearest Neighbor selection for points to avoid empty slices.
    """
    ds = None
    local_zarr_path = None
    query_type = "temporal" # Hardcoded for Climsight point-data focus

    # Get API Key from environment
    ARRAYLAKE_API_KEY = os.environ.get("ARRAYLAKE_API_KEY")

    if not ARRAYLAKE_API_KEY: 
        return {"success": False, "error": "Missing Arraylake API Key", "message": "Please set ARRAYLAKE_API_KEY environment variable."}

    try:
        # Map Variable Name
        short_var = VARIABLE_MAPPING.get(variable_id.lower(), variable_id)
        logging.info(f"🌍 Earthmover ERA5 Retrieval ({query_type}): {short_var} | {start_date} to {end_date}")

        # --- 1. Sandbox / Path Logic ---
        # Priority: 1) CLIMSIGHT_THREAD_ID env var, 2) Streamlit session, 3) work_dir, 4) default
        main_dir = None
        thread_id = os.environ.get("CLIMSIGHT_THREAD_ID")

        if thread_id:
            # Use sandbox path based on thread_id (set by sandbox_utils)
            main_dir = os.path.join("tmp", "sandbox", thread_id)
        elif "streamlit" in sys.modules and st is not None and hasattr(st, 'session_state'):
            try:
                session_uuid = getattr(st.session_state, "session_uuid", None)
                if session_uuid:
                    main_dir = os.path.join("tmp", "sandbox", session_uuid)
                else:
                    st_thread_id = st.session_state.get("thread_id")
                    if st_thread_id:
                        main_dir = os.path.join("tmp", "sandbox", st_thread_id)
            except:
                pass

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
                "message": f"Cached ERA5 data found at {absolute_zarr_path}"
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

        # --- 4. Save to Zarr ---
        logging.info(f"Downloading data to {local_zarr_path}...")

        ds_out = subset.to_dataset(name=short_var)
        
        # Clear encoding
        for var in ds_out.variables:
            ds_out[var].encoding = {}

        if os.path.exists(local_zarr_path):
            shutil.rmtree(local_zarr_path)

        ds_out.to_zarr(local_zarr_path, mode="w", consolidated=True, compute=True)
        logging.info("✅ Download complete.")

        return {
            "success": True,
            "output_path_zarr": absolute_zarr_path,
            "full_path": absolute_zarr_path,
            "variable": short_var,
            "query_type": query_type,
            "message": f"ERA5 data ({query_type} optimized) retrieved and saved to {absolute_zarr_path}"
        }

    except Exception as e:
        logging.error(f"Error in ERA5 Earthmover retrieval: {e}", exc_info=True)
        if local_zarr_path and os.path.exists(local_zarr_path):
            shutil.rmtree(local_zarr_path, ignore_errors=True)
        return {"success": False, "error": str(e), "message": f"Failed: {str(e)}"}
    finally:
        if ds is not None:
            ds.close()

era5_retrieval_tool = StructuredTool.from_function(
    func=retrieve_era5_data,
    name="retrieve_era5_data",
    description=(
        "Retrieves ERA5 Surface climate data from Earthmover (Arraylake). "
        "Optimized for TEMPORAL time-series extraction at specific locations. "
        "Automatically snaps to nearest grid point to ensure data is returned. "
        "Returns a Zarr directory path. "
        "Available vars: t2 (temp), sst, u10/v10 (wind), mslp (pressure), tp (precip)."
    ),
    args_schema=ERA5RetrievalArgs
)