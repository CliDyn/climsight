"""Utility helpers reused by tool modules."""

import json
import logging
import os
import re
import time
import uuid
from datetime import date, datetime
from typing import Any

import pandas as pd


def generate_unique_image_path(sandbox_path: str = None) -> str:
    """Generate a unique image path for saving plots."""
    unique_filename = f"fig_{uuid.uuid4()}.png"
    if sandbox_path and os.path.exists(sandbox_path):
        results_dir = os.path.join(sandbox_path, "results")
        os.makedirs(results_dir, exist_ok=True)
        return os.path.join(results_dir, unique_filename)

    figs_dir = os.path.join("tmp", "figs")
    os.makedirs(figs_dir, exist_ok=True)
    return os.path.join(figs_dir, unique_filename)


def sanitize_input(query: str) -> str:
    return query.strip()


def make_json_serializable(obj: Any) -> Any:
    """Convert objects into JSON-serializable structures."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "item"):
        return obj.item()
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    if isinstance(obj, set):
        return list(obj)
    if hasattr(obj, "__dict__"):
        return make_json_serializable(obj.__dict__)
    return str(obj)


def log_history_event(session_data: dict, event_type: str, details: dict) -> None:
    """Append a structured event to session history."""
    if "execution_history" not in session_data:
        session_data["execution_history"] = []

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    event = {
        "type": event_type,
        "timestamp": timestamp,
    }

    try:
        serializable_details = make_json_serializable(details)
        event.update(serializable_details)
    except Exception as exc:
        logging.error("Failed to serialize event details: %s", exc)
        event.update({
            "serialization_error": str(exc),
            "original_keys": list(details.keys()) if isinstance(details, dict) else "not_dict",
        })

    session_data["execution_history"].append(event)


def list_directory_contents(path: str) -> str:
    """Return a formatted tree of directory contents."""
    result = []
    for root, _, files in os.walk(path):
        level = root.replace(path, "").count(os.sep)
        indent = " " * 4 * level
        result.append(f"{indent}{os.path.basename(root)}/")
        sub_indent = " " * 4 * (level + 1)
        for file in files:
            result.append(f"{sub_indent}{file}")
    return "\n".join(result)


def escape_curly_braces(text: str) -> str:
    if isinstance(text, str):
        return text.replace("{", "{{").replace("}", "}}")
    return str(text)


def get_last_python_repl_command(session_state: dict) -> str:
    """Extract last Python_REPL tool call from a LangChain intermediate steps list."""
    intermediate_steps = session_state.get("intermediate_steps")
    if not intermediate_steps:
        return ""

    python_repl_commands = []
    for action, _ in intermediate_steps:
        if action.get("tool") == "Python_REPL":
            python_repl_commands.append(action)

    if python_repl_commands:
        last_command_action = python_repl_commands[-1]
        return last_command_action.get("tool_input", "")

    return ""
