# api/routes/sessions.py
"""
REST endpoints for session lifecycle management.
"""

import logging
import os
import uuid
import yaml
from typing import Optional, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from session_manager import SessionManager
from sandbox_utils import ensure_thread_id, get_sandbox_paths, ensure_sandbox_dirs
from climate_data_providers import get_available_providers

router = APIRouter()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
AVAILABLE_MODELS = [
    "gpt-5.2", "gpt-5", "gpt-5-mini", "gpt-5-nano",
    "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano",
    "gpt-4o", "o3-mini", "o3",
]


class SessionResponse(BaseModel):
    session_id: str
    thread_id: str


class SessionInfo(BaseModel):
    session_id: str
    thread_id: str
    model_name: Optional[str] = None
    climate_source: Optional[str] = None
    use_smart_agent: bool = False
    use_era5_data: bool = False
    use_powerful_data_analysis: bool = False


class ModelSetRequest(BaseModel):
    model_name: str


class ConfigUpdateRequest(BaseModel):
    use_smart_agent: Optional[bool] = None
    use_era5_data: Optional[bool] = None
    use_powerful_data_analysis: Optional[bool] = None
    climate_data_source: Optional[str] = None
    model_name: Optional[str] = None


# ---------------------------------------------------------------------------
# Config helper
# ---------------------------------------------------------------------------
_config_cache = None

def _load_config() -> dict:
    global _config_cache
    if _config_cache is not None:
        return _config_cache
    config_path = os.path.join(os.getcwd(), "config.yml")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            _config_cache = yaml.safe_load(f)
    else:
        _config_cache = {}
    return _config_cache


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@router.post("/sessions", response_model=SessionResponse)
async def create_session():
    """Create a new session with a unique ID."""
    session_id = str(uuid.uuid4())
    session_data = SessionManager.get_session(session_id)

    # Create thread and sandbox
    thread_id = ensure_thread_id(existing_thread_id="")
    sandbox_paths = get_sandbox_paths(thread_id)
    ensure_sandbox_dirs(sandbox_paths)

    session_data["thread_id"] = thread_id
    session_data.update(sandbox_paths)

    # Load defaults from config
    config = _load_config()
    session_data["model_name"] = config.get("llm_combine", {}).get("model_name", "gpt-5.2")
    session_data["climate_data_source"] = config.get("climate_data_source", "nextGEMS")
    session_data["use_smart_agent"] = config.get("use_smart_agent", False)
    session_data["use_era5_data"] = config.get("use_era5_data", False)
    session_data["use_powerful_data_analysis"] = config.get("use_powerful_data_analysis", False)

    logging.info(f"Created session {session_id} (thread_id: {thread_id})")
    return SessionResponse(session_id=session_id, thread_id=thread_id)


@router.get("/sessions/{session_id}", response_model=SessionInfo)
async def get_session(session_id: str):
    """Get metadata about an existing session."""
    session_data = SessionManager.get_session(session_id)
    if "thread_id" not in session_data:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")

    return SessionInfo(
        session_id=session_id,
        thread_id=session_data["thread_id"],
        model_name=session_data.get("model_name"),
        climate_source=session_data.get("climate_data_source"),
        use_smart_agent=session_data.get("use_smart_agent", False),
        use_era5_data=session_data.get("use_era5_data", False),
        use_powerful_data_analysis=session_data.get("use_powerful_data_analysis", False),
    )


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Cleanup a session."""
    if not SessionManager.delete_session(session_id):
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")
    logging.info(f"Deleted session: {session_id}")
    return {"status": "deleted", "session_id": session_id}


@router.put("/sessions/{session_id}/config")
async def update_config(session_id: str, body: ConfigUpdateRequest):
    """Update session-level toggles and config."""
    session_data = SessionManager.get_session(session_id)
    if "thread_id" not in session_data:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")

    if body.model_name is not None:
        session_data["model_name"] = body.model_name
    if body.use_smart_agent is not None:
        session_data["use_smart_agent"] = body.use_smart_agent
    if body.use_era5_data is not None:
        session_data["use_era5_data"] = body.use_era5_data
    if body.use_powerful_data_analysis is not None:
        session_data["use_powerful_data_analysis"] = body.use_powerful_data_analysis
    if body.climate_data_source is not None:
        session_data["climate_data_source"] = body.climate_data_source

    return {"status": "updated"}


# ---------------------------------------------------------------------------
# Model & Climate Source Lists
# ---------------------------------------------------------------------------
@router.get("/models")
async def list_models():
    """Return the list of available LLM models."""
    config = _load_config()
    default = config.get("llm_combine", {}).get("model_name", "gpt-5.2")
    return {"models": AVAILABLE_MODELS, "default": default}


@router.get("/climate-sources")
async def list_climate_sources():
    """Return available climate data sources."""
    config = _load_config()
    sources = get_available_providers(config)
    source_descriptions = {
        "nextGEMS": "nextGEMS (High resolution)",
        "ICCP": "ICCP (AWI-CM3, medium resolution)",
        "AWI_CM": "AWI-CM (CMIP6, low resolution)",
        "DestinE": "DestinE IFS-FESOM (High resolution, SSP3-7.0)",
    }
    default = config.get("climate_data_source", "nextGEMS")
    return {
        "sources": sources,
        "descriptions": source_descriptions,
        "default": default,
    }
