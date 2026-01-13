# src/climsight/tools/era5_retrieval_tool.py
"""
ERA5 data retrieval tool for use in the visualization agent.
Retrieves climate data from the Google Cloud ARCO-ERA5 dataset.
Saves the retrieved data ONLY in Zarr format.
The Zarr filename is based on a hash of the request parameters.
Optimized for memory efficiency using lazy loading (Dask) and streaming.
"""

import os
import sys
import logging
import hashlib
import shutil
import xarray as xr
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field
from typing import Optional, Literal
from langchain_core.tools import StructuredTool

# --- IMPORTS & CONFIGURATION ---
try:
    import zarr
    import gcsfs
    # Optional: Check for Streamlit to support session state if available
    try:
        import streamlit as st
    except ImportError:
        st = None
except ImportError as e:
    install_command = "pip install --upgrade xarray zarr numcodecs gcsfs blosc pandas numpy pydantic langchain-core"
    raise ImportError(
        f"Required libraries missing. Please ensure zarr, gcsfs, and others are installed.\n"
        f"Try running: {install_command}"
    ) from e

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Target the Analysis-Ready (AR) Zarr store (Chunk-1 optimized)
ARCO_ERA5_MAIN_ZARR_STORE = 'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3'

class ERA5RetrievalArgs(BaseModel):
    variable_id: Literal[
        "sea_surface_temperature", "surface_pressure", "total_cloud_cover", "total_precipitation",
        "10m_u_component_of_wind", "10m_v_component_of_wind", "2m_temperature", "2m_dewpoint_temperature",
        "geopotential", "specific_humidity", "temperature", "u_component_of_wind",
        "v_component_of_wind", "vertical_velocity"
    ] = Field(description="ERA5 variable to retrieve (must match Zarr store names).")
    start_date: str = Field(description="Start date (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS).")
    end_date: str = Field(description="End date (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS).")
    min_latitude: float = Field(-90.0, description="Minimum latitude (–90 to 90).")
    max_latitude: float = Field(90.0, description="Maximum latitude (–90 to 90).")
    min_longitude: float = Field(0.0, description="Minimum longitude (0–360 or –180 to 360).")
    max_longitude: float = Field(359.75, description="Maximum longitude (0–360 or –180 to 360).")
    pressure_level: Optional[int] = Field(None, description="Pressure level in hPa for 3D variables.")
    work_dir: Optional[str] = Field(None, description="The absolute path to the working directory where data should be saved.")

def _generate_descriptive_filename(
    variable_id: str, start_date: str, end_date: str,
    pressure_level: Optional[int] = None
) -> str:
    """
    Generate a descriptive filename based on request parameters.
    """
    clean_variable = variable_id.replace('-', '_').replace('/', '_').replace(' ', '_')
    clean_start = start_date.split()[0].replace('-', '_')
    clean_end = end_date.split()[0].replace('-', '_')
    
    filename_parts = [clean_variable, clean_start, "to", clean_end]
    if pressure_level is not None:
        filename_parts.extend(["level", str(pressure_level)])
    
    filename = "_".join(filename_parts) + ".zarr"
    return filename

def _generate_request_hash(
    variable_id: str, start_date: str, end_date: str,
    min_latitude: float, max_latitude: float,
    min_longitude: float, max_longitude: float,
    pressure_level: Optional[int]
) -> str:
    params_string = (
        f"{variable_id}-{start_date}-{end_date}-"
        f"{min_latitude:.2f}-{max_latitude:.2f}-"
        f"{min_longitude:.2f}-{max_longitude:.2f}-"
        f"{pressure_level if pressure_level is not None else 'None'}"
    )
    return hashlib.md5(params_string.encode('utf-8')).hexdigest()

def retrieve_era5_data(
    variable_id: str, start_date: str, end_date: str,
    min_latitude: float = -90.0, max_latitude: float = 90.0,
    min_longitude: float = 0.0, max_longitude: float = 359.75,
    pressure_level: Optional[int] = None,
    work_dir: Optional[str] = None
) -> dict:
    ds_arco = None
    zarr_path = None

    try:
        logging.info(f"ERA5 retrieval: var={variable_id}, time={start_date}→{end_date}, "
                     f"lat=[{min_latitude},{max_latitude}], lon=[{min_longitude},{max_longitude}], "
                     f"level={pressure_level}")

        # --- 1. Sandbox / Path Logic ---
        main_dir = None
        
        # Priority 1: Explicit work_dir passed from agent (Fixes the path issue)
        if work_dir and os.path.exists(work_dir):
            main_dir = work_dir
            logging.info(f"Using explicitly provided work_dir: {main_dir}")
        
        # Priority 2: Streamlit Session (Fallback logic)
        if not main_dir and "streamlit" in sys.modules and st is not None and hasattr(st, 'session_state'):
            try:
                # Check session_uuid first (climsight uses this)
                session_uuid = getattr(st.session_state, "session_uuid", None)
                if session_uuid:
                    main_dir = os.path.join("tmp", "sandbox", session_uuid)
                    logging.info(f"Found session UUID. Using persistent sandbox: {main_dir}")
                else:
                    # Legacy fallback
                    thread_id = st.session_state.get("thread_id")
                    if thread_id:
                        main_dir = os.path.join("tmp", "sandbox", thread_id)
                        logging.info(f"Found session thread_id. Using persistent sandbox: {main_dir}")
            except Exception:
                pass
        
        # Priority 3: Generic Fallback
        if not main_dir:
            main_dir = os.path.join("tmp", "sandbox", "era5_data")
            logging.info(f"Using general sandbox: {main_dir}")

        os.makedirs(main_dir, exist_ok=True)
        
        # Store ERA5 data in a subfolder of the session directory to keep it clean
        era5_specific_dir = os.path.join(main_dir, "era5_data")
        os.makedirs(era5_specific_dir, exist_ok=True)

        # Check cache
        zarr_filename = _generate_descriptive_filename(variable_id, start_date, end_date, pressure_level)
        zarr_path = os.path.join(era5_specific_dir, zarr_filename)
        
        # CRITICAL: Return absolute path so python_repl can find it easily regardless of CWD
        absolute_zarr_path = os.path.abspath(zarr_path)

        if os.path.exists(zarr_path):
             logging.info(f"Data already cached at {zarr_path}")
             return {
                "success": True, 
                "output_path_zarr": absolute_zarr_path, 
                "full_path": absolute_zarr_path,
                "variable": variable_id, 
                "message": f"Cached ERA5 data found at {absolute_zarr_path}"
            }

        # --- 2. Open Dataset (Lazy / Dask) ---
        logging.info(f"Opening ERA5 dataset: {ARCO_ERA5_MAIN_ZARR_STORE}")
        
        # CRITICAL OPTIMIZATION: chunks={'time': 24}
        # This prevents loading the whole dataset into RAM and creates an efficient download graph.
        ds_arco = xr.open_zarr(
            ARCO_ERA5_MAIN_ZARR_STORE, 
            chunks={'time': 24}, 
            consolidated=True, 
            storage_options={'token': 'anon'}
        )

        if variable_id not in ds_arco:
            raise ValueError(f"Variable '{variable_id}' not found. Available: {list(ds_arco.data_vars)}")

        var_data_from_arco = ds_arco[variable_id]

        # --- 3. Lazy Subsetting ---
        start_datetime_obj = pd.to_datetime(start_date)
        end_datetime_obj = pd.to_datetime(end_date)
        time_filtered_data = var_data_from_arco.sel(time=slice(start_datetime_obj, end_datetime_obj))

        lat_slice_coords = slice(max_latitude, min_latitude)
        lon_min_request_360 = min_longitude % 360
        lon_max_request_360 = max_longitude % 360

        # Anti-Meridian / Date Line Logic
        if lon_min_request_360 > lon_max_request_360:
            part1 = time_filtered_data.sel(latitude=lat_slice_coords, longitude=slice(lon_min_request_360, 359.75))
            part2 = time_filtered_data.sel(latitude=lat_slice_coords, longitude=slice(0, lon_max_request_360))
            
            if part1.sizes.get('longitude', 0) > 0 and part2.sizes.get('longitude', 0) > 0:
                space_filtered_data = xr.concat([part1, part2], dim='longitude')
            elif part1.sizes.get('longitude', 0) > 0: 
                space_filtered_data = part1
            elif part2.sizes.get('longitude', 0) > 0: 
                space_filtered_data = part2
            else: 
                space_filtered_data = time_filtered_data.sel(latitude=lat_slice_coords, longitude=slice(0,0))
        else:
            space_filtered_data = time_filtered_data.sel(
                latitude=lat_slice_coords,
                longitude=slice(lon_min_request_360, lon_max_request_360)
            )

        if pressure_level is not None and "level" in space_filtered_data.coords:
            space_filtered_data = space_filtered_data.sel(level=pressure_level)

        if not all(dim_size > 0 for dim_size in space_filtered_data.sizes.values()):
            msg = "Selected region/time period has zero size."
            logging.warning(msg)
            return {"success": False, "error": msg, "message": f"Failed: {msg}"}

        # --- 4. Prepare Output Dataset ---
        final_subset_ds = space_filtered_data.to_dataset(name=variable_id)

        # Clear encoding (fixes Zarr v3 codec conflicts)
        for var in final_subset_ds.variables:
            final_subset_ds[var].encoding = {}

        # RE-CHUNK for safe streaming (24h blocks are manageable for Mac RAM)
        final_subset_ds = final_subset_ds.chunk({'time': 24, 'latitude': -1, 'longitude': -1})

        # Metadata
        request_hash_str = _generate_request_hash(
            variable_id, start_date, end_date, min_latitude, max_latitude,
            min_longitude, max_longitude, pressure_level
        )
        final_subset_ds.attrs.update({
            'title': f"ERA5 {variable_id} data subset",
            'source_arco_era5': ARCO_ERA5_MAIN_ZARR_STORE,
            'retrieval_parameters_hash': request_hash_str
        })

        if os.path.exists(zarr_path):
            logging.warning(f"Zarr path {zarr_path} exists. Removing for freshness.")
            shutil.rmtree(zarr_path)

        # --- 5. Streaming Write (NO .load()) ---
        logging.info(f"Streaming data to Zarr store: {zarr_path}")
        
        # compute=True triggers the download/write graph chunk-by-chunk
        final_subset_ds.to_zarr(
            store=zarr_path,
            mode='w',
            consolidated=True,
            compute=True
        )

        logging.info(f"Successfully saved: {zarr_path}")

        return {
            "success": True, 
            "output_path_zarr": absolute_zarr_path, 
            "full_path": absolute_zarr_path,
            "variable": variable_id, 
            "message": f"ERA5 data saved to {absolute_zarr_path}"
        }
    
    except Exception as e:
        logging.error(f"Error in ERA5 retrieval: {e}", exc_info=True)
        error_msg = str(e)
        
        if "Resource Exhausted" in error_msg or "Too many open files" in error_msg:
             error_msg += " (GCS access issue or system limits. Try again.)"
        elif "Unsupported type for store_like" in error_msg:
            error_msg += " (Issue with Zarr store path. Ensure gs:// URI is used.)"
        
        # Cleanup partial data on failure
        if zarr_path and os.path.exists(zarr_path):
             shutil.rmtree(zarr_path, ignore_errors=True)

        return {"success": False, "error": error_msg, "message": f"Failed: {error_msg}"}
    
    finally:
        if ds_arco is not None:
            ds_arco.close()

era5_retrieval_tool = StructuredTool.from_function(
    func=retrieve_era5_data,
    name="retrieve_era5_data",
    description=(
        "Retrieves a subset of the ARCO-ERA5 Zarr climate reanalysis dataset. "
        "Saves the data locally as a Zarr store. "
        "Uses efficient streaming to prevent memory overflow on large files."
    ),
    args_schema=ERA5RetrievalArgs
)