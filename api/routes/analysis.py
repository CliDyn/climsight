# api/routes/analysis.py
"""
WebSocket endpoint for real-time climate analysis.
Streams status updates, agent progress, and final responses to the client.
"""

import asyncio
import json
import logging
import os
import traceback
import yaml
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from session_manager import SessionManager
from sandbox_utils import ensure_thread_id, get_sandbox_paths, ensure_sandbox_dirs

router = APIRouter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
async def send_json(ws: WebSocket, msg_type: str, data: Any):
    """Send a typed JSON message over the WebSocket.
    Uses a custom encoder to handle numpy types that standard json can't serialize.
    """
    await ws.send_text(json.dumps({"type": msg_type, "data": data}, cls=_SafeEncoder))


class _SafeEncoder(json.JSONEncoder):
    """Handle numpy scalars and other non-standard types."""
    def default(self, o):
        try:
            import numpy as np
            if isinstance(o, (np.bool_,)):
                return bool(o)
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
        except ImportError:
            pass
        return super().default(o)


def _make_artifact_url(plot_path: str) -> str:
    """Convert an absolute plot path to a relative /artifacts/ URL."""
    marker = os.path.join("tmp", "sandbox")
    idx = plot_path.find(marker)
    if idx != -1:
        relative = plot_path[idx + len(marker):]
        relative = relative.replace(os.sep, "/")
        if not relative.startswith("/"):
            relative = "/" + relative
        return f"/artifacts{relative}"
    return f"/artifacts/{os.path.basename(plot_path)}"


def _load_config() -> dict:
    config_path = os.path.join(os.getcwd(), "config.yml")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}


def _load_references() -> dict:
    ref_path = os.path.join(os.getcwd(), "references.yml")
    if os.path.exists(ref_path):
        with open(ref_path, "r") as f:
            refs = yaml.safe_load(f)
            return {"references": refs or {}, "used": []}
    return {"references": {}, "used": []}


# ---------------------------------------------------------------------------
# WebSocket Endpoint
# ---------------------------------------------------------------------------
@router.websocket("/sessions/{session_id}/agent/ws")
async def analysis_websocket(ws: WebSocket, session_id: str):
    """
    Bidirectional WebSocket for climate analysis.

    Client sends:
        {"lat": 52.52, "lon": 13.37, "query": "...", "config": {...}}

    Server streams:
        {"type": "status",   "data": {"message": "..."}}
        {"type": "location", "data": {"lat": ..., "lon": ..., "address": ..., ...}}
        {"type": "response", "data": {"content": "...", "plot_urls": [...], ...}}
        {"type": "error",    "data": {"message": "..."}}
    """
    await ws.accept()
    logging.info(f"WebSocket connected: session {session_id}")

    session_data = SessionManager.get_session(session_id)
    if "thread_id" not in session_data:
        await send_json(ws, "error", {
            "message": "Session not initialized. Call POST /api/sessions first."
        })
        await ws.close()
        return

    try:
        while True:
            raw = await ws.receive_text()
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                await send_json(ws, "error", {"message": "Invalid JSON payload."})
                continue

            lat = payload.get("lat")
            lon = payload.get("lon")
            query = payload.get("query", "").strip()

            if lat is None or lon is None:
                await send_json(ws, "error", {"message": "lat and lon are required."})
                continue
            if not query:
                await send_json(ws, "error", {"message": "Empty query."})
                continue

            lat = float(lat)
            lon = float(lon)

            # Apply any per-request config overrides
            per_request_config = payload.get("config", {})

            await send_json(ws, "status", {
                "message": f"Processing query for ({lat:.4f}, {lon:.4f})...",
                "lat": lat, "lon": lon,
            })

            # --- Run the analysis in a thread pool (blocking calls) ---
            try:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: _run_analysis(
                        session_id, session_data, lat, lon, query, per_request_config
                    ),
                )
            except Exception as e:
                logging.error(f"Analysis error: {e}", exc_info=True)
                await send_json(ws, "error", {
                    "message": f"Analysis error: {str(e)}",
                    "traceback": traceback.format_exc(),
                })
                continue

            if result is None:
                await send_json(ws, "error", {"message": "Analysis returned no result."})
                continue

            # --- Send results ---
            if result.get("error"):
                await send_json(ws, "error", {"message": result["error"]})
                continue

            # Location info
            if result.get("location_info"):
                await send_json(ws, "location", result["location_info"])

            # Plot URLs
            plot_urls = [
                _make_artifact_url(p) for p in result.get("plot_images", []) if p
            ]

            # Final response
            await send_json(ws, "response", {
                "content": result.get("output", ""),
                "plot_urls": plot_urls,
                "input_params": result.get("input_params", {}),
                "references": result.get("references", {}),
            })

    except WebSocketDisconnect:
        logging.info(f"WebSocket disconnected: session {session_id}")
    except Exception as e:
        logging.error(f"WebSocket fatal error for session {session_id}: {e}", exc_info=True)
        try:
            await send_json(ws, "error", {"message": f"Fatal error: {str(e)}"})
        except Exception:
            pass
        await ws.close()


# ---------------------------------------------------------------------------
# Blocking analysis runner (runs in thread pool)
# ---------------------------------------------------------------------------
def _run_analysis(
    session_id: str,
    session_data: dict,
    lat: float,
    lon: float,
    query: str,
    per_request_config: dict,
) -> dict:
    """
    Execute the full ClimSight analysis pipeline.
    Returns a dict with output, plot_images, input_params, references, error.
    """
    from climsight_engine import normalize_longitude, location_request, llm_request
    from stream_handler import StreamHandler
    from data_container import DataContainer
    from rag import load_rag

    # Normalize longitude
    lon = normalize_longitude(lon)

    # Load base config and merge session/per-request overrides
    config = _load_config()
    config["use_smart_agent"] = per_request_config.get(
        "use_smart_agent", session_data.get("use_smart_agent", config.get("use_smart_agent", False))
    )
    config["use_era5_data"] = per_request_config.get(
        "use_era5_data", session_data.get("use_era5_data", config.get("use_era5_data", False))
    )
    config["use_powerful_data_analysis"] = per_request_config.get(
        "use_powerful_data_analysis",
        session_data.get("use_powerful_data_analysis", config.get("use_powerful_data_analysis", False)),
    )
    config["climate_data_source"] = per_request_config.get(
        "climate_data_source",
        session_data.get("climate_data_source", config.get("climate_data_source", "nextGEMS")),
    )
    config["show_add_info"] = True
    config["llmModeKey"] = "agent_llm"

    # Model override
    model_name = per_request_config.get(
        "model_name", session_data.get("model_name", config.get("llm_combine", {}).get("model_name"))
    )
    if model_name:
        if "llm_combine" not in config:
            config["llm_combine"] = {}
        config["llm_combine"]["model_name"] = model_name
        if "gpt" in model_name or "o1" in model_name or "o3" in model_name:
            config["llm_combine"]["model_type"] = "openai"
        else:
            config["llm_combine"]["model_type"] = "local"

    # API keys
    api_key = os.environ.get("OPENAI_API_KEY", "")
    api_key_local = os.environ.get("OPENAI_API_KEY_LOCAL", "")

    if not api_key and config.get("llm_combine", {}).get("model_type") == "openai":
        return {"error": "No OPENAI_API_KEY set in environment."}

    # Arraylake key for ERA5
    if config.get("use_era5_data"):
        arraylake_key = os.environ.get("ARRAYLAKE_API_KEY", "")
        if not arraylake_key:
            return {"error": "ARRAYLAKE_API_KEY required for ERA5 data."}
        config["arraylake_api_key"] = arraylake_key

    # Sandbox setup
    thread_id = session_data.get("thread_id", "")
    if not thread_id:
        thread_id = ensure_thread_id()
        session_data["thread_id"] = thread_id

    sandbox_paths = get_sandbox_paths(thread_id)
    ensure_sandbox_dirs(sandbox_paths)
    os.environ["CLIMSIGHT_THREAD_ID"] = thread_id

    # Location request
    content_message, input_params = location_request(config, lat, lon)

    if not input_params:
        return {"error": "The selected point is in the ocean. Please choose a location on land."}

    location_info = {
        "lat": lat,
        "lon": lon,
        "address": input_params.get("location_str", ""),
        "address_display": input_params.get("location_str_for_print", ""),
        "is_inland_water": input_params.get("is_inland_water", False),
        "water_body_status": input_params.get("water_body_status", ""),
    }

    # Sandbox paths into input_params
    input_params["thread_id"] = thread_id
    input_params.update(sandbox_paths)
    input_params["user_message"] = query
    content_message = "Human request: {user_message} \n " + content_message

    # Load RAG databases
    references = _load_references()
    ipcc_rag_ready, ipcc_rag_db = False, None
    general_rag_ready, general_rag_db = False, None

    try:
        ipcc_rag_ready, ipcc_rag_db = load_rag(config, openai_api_key=api_key, db_type="ipcc")
    except Exception as e:
        logging.warning(f"IPCC RAG load failed: {e}")

    try:
        general_rag_ready, general_rag_db = load_rag(config, openai_api_key=api_key, db_type="general")
    except Exception as e:
        logging.warning(f"General RAG load failed: {e}")

    # Create a no-op stream handler (no Streamlit containers)
    stream_handler = StreamHandler()
    data_pocket = DataContainer()

    # Run LLM request
    output, input_params, content_message, _ = llm_request(
        content_message, input_params, config, api_key, api_key_local,
        stream_handler, ipcc_rag_ready, ipcc_rag_db,
        general_rag_ready, general_rag_db, data_pocket,
        references=references,
    )

    # Collect plot images from sandbox results dir
    plot_images = input_params.get("data_analysis_images", [])
    results_dir = sandbox_paths.get("results_dir", "")
    if results_dir and os.path.isdir(results_dir):
        for fname in sorted(os.listdir(results_dir)):
            if fname.lower().endswith((".png", ".jpg", ".svg", ".pdf")):
                full_path = os.path.join(results_dir, fname)
                if full_path not in plot_images:
                    plot_images.append(full_path)

    return {
        "output": output,
        "plot_images": plot_images,
        "input_params": {
            k: v for k, v in input_params.items()
            if isinstance(v, (str, int, float, bool, list))
        },
        "references": references,
        "location_info": location_info,
    }
