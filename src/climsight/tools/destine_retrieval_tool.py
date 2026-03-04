"""
DestinE Climate Adaptation data retrieval tool for the data analysis agent.

Two tools:
1. search_destine_parameters — RAG semantic search over 82 DestinE parameters
2. retrieve_destine_data — download point time series via earthkit.data + polytope

Authentication:
  Assumes ~/.polytopeapirc already exists with a valid token
  (created by desp-authentication.py). The tool checks for the file
  and fails with a clear message if missing.
"""

import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

try:
    import earthkit.data
except ImportError:
    earthkit = None

try:
    import streamlit as st
except ImportError:
    st = None

logger = logging.getLogger(__name__)

# ============================================================================
# Constants
# ============================================================================

POLYTOPE_ADDRESS = "polytope.lumi.apps.dte.destination-earth.eu"
POLYTOPEAPIRC_PATH = Path.home() / ".polytopeapirc"


# ============================================================================
# Tool A: search_destine_parameters (RAG)
# ============================================================================

class DestinESearchArgs(BaseModel):
    query: str = Field(description=(
        "Natural language query describing the climate variable you need. "
        "Examples: 'temperature at 2 meters', 'precipitation', 'wind speed at 10m', "
        "'sea surface temperature', 'cloud cover', 'soil moisture'"
    ))
    k: int = Field(default=5, description="Number of candidate parameters to return (default: 5)")


def _search_destine_parameters(
    query: str,
    k: int = 5,
    chroma_db_path: str = "data/destine/chroma_db",
    collection_name: str = "climate_parameters_with_usage_notes",
    openai_api_key: str = None,
) -> dict:
    """
    Search DestinE parameter vector store for relevant climate parameters.
    """
    try:
        from langchain_openai import OpenAIEmbeddings
        from langchain_chroma import Chroma
    except ImportError as e:
        return {
            "success": False,
            "error": f"Missing dependency for DestinE parameter search: {e}",
            "candidates": []
        }

    api_key = openai_api_key or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return {
            "success": False,
            "error": "No OpenAI API key available for embedding search",
            "candidates": []
        }

    if not os.path.exists(chroma_db_path):
        return {
            "success": False,
            "error": f"DestinE parameter vector store not found at: {chroma_db_path}",
            "candidates": []
        }

    try:
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=api_key
        )

        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=chroma_db_path
        )

        results = vector_store.similarity_search(query=query, k=k)

        candidates = []
        for doc in results:
            meta = doc.metadata
            candidates.append({
                "param_id": meta.get("paramId", ""),
                "levtype": meta.get("levtype", ""),
                "name": meta.get("name", ""),
                "shortName": meta.get("shortName", ""),
                "units": meta.get("units", ""),
                "description": meta.get("param_description", ""),
                "usage_notes": meta.get("usage_notes", ""),
            })

        logger.info(f"DestinE parameter search for '{query}': found {len(candidates)} candidates")

        return {
            "success": True,
            "query": query,
            "candidates": candidates,
            "message": (
                f"Found {len(candidates)} candidate parameters. "
                "Use param_id and levtype with retrieve_destine_data to download data."
            )
        }

    except Exception as e:
        logger.error(f"DestinE parameter search failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "candidates": []
        }


def create_destine_search_tool(config: dict) -> StructuredTool:
    """Create the DestinE parameter search tool with config bound."""
    destine_settings = config.get("destine_settings", {})
    chroma_db_path = destine_settings.get("chroma_db_path", "data/destine/chroma_db")
    collection_name = destine_settings.get("collection_name", "climate_parameters_with_usage_notes")
    api_key = config.get("openai_api_key", "") or os.environ.get("OPENAI_API_KEY", "")

    def search_wrapper(query: str, k: int = 5) -> dict:
        return _search_destine_parameters(
            query=query,
            k=k,
            chroma_db_path=chroma_db_path,
            collection_name=collection_name,
            openai_api_key=api_key,
        )

    return StructuredTool.from_function(
        func=search_wrapper,
        name="search_destine_parameters",
        description=(
            "Search the DestinE Climate DT parameter database (82 parameters) using semantic search. "
            "Input a natural language description of the climate variable you need. "
            "Returns candidate parameters with param_id, levtype, name, units, and description. "
            "Use the returned param_id and levtype with retrieve_destine_data to download the actual data. "
            "Examples: 'temperature at 2 meters', 'total precipitation', 'wind speed', 'sea surface temperature'"
        ),
        args_schema=DestinESearchArgs,
    )


# ============================================================================
# Tool B: retrieve_destine_data (earthkit.data + polytope)
# ============================================================================

class DestinERetrievalArgs(BaseModel):
    param_id: str = Field(description=(
        "DestinE parameter ID from search_destine_parameters results. "
        "Examples: '167' (2m temperature), '228' (total precipitation), "
        "'165' (10m U-wind), '166' (10m V-wind)"
    ))
    levtype: str = Field(description=(
        "Level type from search_destine_parameters results. "
        "Examples: 'sfc' (surface), 'pl' (pressure levels), 'o2d' (ocean 2D)"
    ))
    start_date: str = Field(description="Start date in YYYYMMDD format. Data available 20200101-20391231.")
    end_date: str = Field(description="End date in YYYYMMDD format. Data available 20200101-20391231. Default to full range (20200101-20391231) unless user requests shorter.")
    latitude: float = Field(description="Latitude of the point (-90 to 90)")
    longitude: float = Field(description="Longitude of the point (-180 to 180)")
    work_dir: Optional[str] = Field(None, description="Working directory. Pass '.' for sandbox root.")


def retrieve_destine_data(
    param_id: str,
    levtype: str,
    start_date: str,
    end_date: str,
    latitude: float,
    longitude: float,
    work_dir: Optional[str] = None,
    **kwargs,
) -> dict:
    """
    Download DestinE Climate Adaptation data via earthkit.data + polytope.

    Point time series extraction for SSP3-7.0 scenario (IFS-NEMO model).
    Requires ~/.polytopeapirc with a valid token (run desp-authentication.py first).
    """
    # --- 1. Check earthkit.data is available ---
    if earthkit is None:
        return {
            "success": False,
            "error": "earthkit-data package not installed",
            "message": "Install with: pip install earthkit-data"
        }

    # --- 2. Check token file exists ---
    if not POLYTOPEAPIRC_PATH.exists():
        return {
            "success": False,
            "error": f"Polytope token file not found at {POLYTOPEAPIRC_PATH}",
            "message": (
                "Run desp-authentication.py first to obtain a token. "
                "The token is written to ~/.polytopeapirc automatically."
            )
        }

    # Validate token file has content
    try:
        with open(POLYTOPEAPIRC_PATH, 'r') as f:
            token_data = json.load(f)
        if not token_data.get("user_key"):
            return {
                "success": False,
                "error": "Polytope token file exists but contains no user_key",
                "message": "Re-run desp-authentication.py to refresh the token."
            }
    except (json.JSONDecodeError, OSError) as e:
        return {
            "success": False,
            "error": f"Cannot read polytope token file: {e}",
            "message": "Re-run desp-authentication.py to refresh the token."
        }

    logger.info(f"DestinE retrieval: param_id={param_id}, levtype={levtype}, "
                f"dates={start_date}-{end_date}, location=({latitude}, {longitude})")

    try:
        # --- 3. Resolve sandbox path (same pattern as ERA5 tool) ---
        main_dir = None
        thread_id = os.environ.get("CLIMSIGHT_THREAD_ID")

        if thread_id:
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
            except Exception:
                pass

        if not main_dir and work_dir:
            main_dir = work_dir

        if not main_dir:
            main_dir = os.path.join("tmp", "sandbox", "destine_default")

        os.makedirs(main_dir, exist_ok=True)

        if os.path.basename(main_dir.rstrip(os.sep)) == "destine_data":
            destine_dir = main_dir
        else:
            destine_dir = os.path.join(main_dir, "destine_data")
        os.makedirs(destine_dir, exist_ok=True)

        # --- 4. Cache check ---
        zarr_dirname = f"destine_{param_id}_{levtype}_{start_date}_{end_date}.zarr"
        local_zarr_path = os.path.join(destine_dir, zarr_dirname)
        absolute_zarr_path = os.path.abspath(local_zarr_path)

        if os.path.exists(local_zarr_path):
            logger.info(f"Cache hit: {local_zarr_path}")
            return {
                "success": True,
                "output_path_zarr": absolute_zarr_path,
                "full_path": absolute_zarr_path,
                "variable": param_id,
                "param_id": param_id,
                "levtype": levtype,
                "message": f"Cached DestinE data found at {absolute_zarr_path}",
                "reference": "Destination Earth Climate Adaptation Digital Twin (Climate DT). https://destine.ecmwf.int/"
            }

        # --- 5. Build request (earthkit/polytope format — no ecmwf: prefix) ---
        request = {
            'class': 'd1',
            'dataset': 'climate-dt',
            'type': 'fc',
            'expver': '0001',
            'generation': '1',
            'realization': '1',
            'activity': 'ScenarioMIP',
            'experiment': 'SSP3-7.0',
            'model': 'IFS-NEMO',
            'param': param_id,
            'levtype': levtype,
            'resolution': 'high',
            'stream': 'clte',
            'date': f'{start_date}/to/{end_date}',
            'time': ['0000', '0100', '0200', '0300', '0400', '0500',
                     '0600', '0700', '0800', '0900', '1000', '1100',
                     '1200', '1300', '1400', '1500', '1600', '1700',
                     '1800', '1900', '2000', '2100', '2200', '2300'],
            'feature': {
                "type": "timeseries",
                "points": [[latitude, longitude]],  # NOTE: lat, lon order!
                "time_axis": "date",
            }
        }

        logger.info(f"Sending earthkit.data request to polytope at {POLYTOPE_ADDRESS}...")

        # --- 6. Download via earthkit.data ---
        data = earthkit.data.from_source(
            "polytope", "destination-earth",
            request,
            address=POLYTOPE_ADDRESS,
            stream=False,
        )
        ds = data.to_xarray()
        logger.info(f"Received xarray dataset: {ds}")

        # --- 7. Save as Zarr ---
        temp_zarr_path = local_zarr_path + ".tmp"
        if os.path.exists(temp_zarr_path):
            shutil.rmtree(temp_zarr_path)

        # Clear encoding for portability
        for var in ds.data_vars:
            ds[var].encoding.clear()
        for coord in ds.coords:
            ds[coord].encoding.clear()

        ds.to_zarr(temp_zarr_path, mode='w')
        os.rename(temp_zarr_path, local_zarr_path)
        logger.info(f"Saved Zarr: {local_zarr_path}")

        # --- 8. Return result ---
        return {
            "success": True,
            "output_path_zarr": absolute_zarr_path,
            "full_path": absolute_zarr_path,
            "variable": param_id,
            "param_id": param_id,
            "levtype": levtype,
            "message": f"DestinE data (param={param_id}, levtype={levtype}) saved to {absolute_zarr_path}",
            "data_summary": str(ds),
            "reference": "Destination Earth Climate Adaptation Digital Twin (Climate DT). https://destine.ecmwf.int/"
        }

    except Exception as e:
        logger.error(f"DestinE retrieval failed: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "message": "DestinE data retrieval failed. Try a shorter date range or check credentials/token."
        }


def create_destine_retrieval_tool() -> StructuredTool:
    """Create the DestinE retrieval tool. No credentials needed — uses ~/.polytopeapirc token."""

    def retrieve_wrapper(
        param_id: str,
        levtype: str,
        start_date: str,
        end_date: str,
        latitude: float,
        longitude: float,
        work_dir: Optional[str] = None,
    ) -> dict:
        return retrieve_destine_data(
            param_id=param_id,
            levtype=levtype,
            start_date=start_date,
            end_date=end_date,
            latitude=latitude,
            longitude=longitude,
            work_dir=work_dir,
        )

    return StructuredTool.from_function(
        func=retrieve_wrapper,
        name="retrieve_destine_data",
        description=(
            "Download DestinE Climate Adaptation projection data via earthkit.data + polytope. "
            "Point time series for SSP3-7.0 scenario (IFS-NEMO model, 2020-2039). "
            "FIRST use search_destine_parameters to find the right param_id and levtype, "
            "then call this tool to download. "
            "By default request the FULL period 20200101-20391231 for maximum coverage. "
            "Returns a Zarr directory path for loading in Python REPL."
        ),
        args_schema=DestinERetrievalArgs,
    )
