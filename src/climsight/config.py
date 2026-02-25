"""Minimal config helpers for tool modules."""

import os


def _get_openai_api_key() -> str:
    return os.environ.get("OPENAI_API_KEY", "")


API_KEY = _get_openai_api_key()
