"""
Sandbox utilities for per-session data storage.

This mirrors the PangaeaGPT layout while keeping the API minimal for Climsight.
"""

import json
import logging
import os
import uuid
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


def ensure_thread_id(existing_thread_id: str = "") -> str:
    """Ensure a stable session thread_id."""
    thread_id = existing_thread_id or ""

    if not thread_id:
        thread_id = os.environ.get("CLIMSIGHT_THREAD_ID", "")

    if not thread_id:
        thread_id = uuid.uuid4().hex

    # Expose to all tools via environment variable.
    os.environ["CLIMSIGHT_THREAD_ID"] = thread_id

    return thread_id


def get_sandbox_paths(thread_id: str) -> Dict[str, str]:
    """Return sandbox paths for a given session."""
    base_dir = Path("tmp") / "sandbox" / thread_id
    return {
        "uuid_main_dir": str(base_dir),
        "results_dir": str(base_dir / "results"),
        "climate_data_dir": str(base_dir / "climate_data"),
        "era5_data_dir": str(base_dir / "era5_data"),
    }


def ensure_sandbox_dirs(paths: Dict[str, str]) -> None:
    """Create sandbox directories if they do not exist."""
    for key, path in paths.items():
        if not path:
            continue
        os.makedirs(path, exist_ok=True)
        logger.debug("Ensured sandbox dir %s: %s", key, path)


def write_climate_data_manifest(
    df_list: List[Dict],
    climate_data_dir: str,
    source: str,
) -> Tuple[str, List[Dict]]:
    """Persist climate dataframes and metadata into the sandbox.

    Returns:
        (manifest_path, entries)
    """
    os.makedirs(climate_data_dir, exist_ok=True)

    entries: List[Dict] = []
    main_index = 0

    for i, entry in enumerate(df_list):
        if entry.get("main"):
            main_index = i
            break

    for i, entry in enumerate(df_list):
        df = entry.get("dataframe")
        if df is None:
            continue

        csv_name = f"simulation_{i + 1}.csv"
        meta_name = f"simulation_{i + 1}_meta.json"
        csv_path = os.path.join(climate_data_dir, csv_name)
        meta_path = os.path.join(climate_data_dir, meta_name)

        df.to_csv(csv_path, index=False)

        meta = {
            "years_of_averaging": entry.get("years_of_averaging", ""),
            "description": entry.get("description", ""),
            "extracted_vars": entry.get("extracted_vars", {}),
            "main": bool(entry.get("main", False)),
            "source": entry.get("source", source),
            "filename": entry.get("filename", ""),
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        entries.append({
            "csv": csv_name,
            "meta": meta_name,
            "years_of_averaging": meta["years_of_averaging"],
            "description": meta["description"],
            "main": meta["main"],
        })

    # Provide a simple auto-load CSV for Python_REPL parity.
    if df_list:
        main_df = df_list[main_index].get("dataframe")
        if main_df is not None:
            main_df.to_csv(os.path.join(climate_data_dir, "data.csv"), index=False)

    manifest = {
        "source": source,
        "entries": entries,
    }
    manifest_path = os.path.join(climate_data_dir, "climate_data_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest_path, entries


def load_climate_data_manifest(manifest_path: str) -> Dict:
    """Load a climate data manifest from disk."""
    if not manifest_path or not os.path.exists(manifest_path):
        return {}

    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)
