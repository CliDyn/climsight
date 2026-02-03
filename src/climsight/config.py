"""Minimal config helpers for tool modules."""

import os

try:
    import streamlit as st
except ImportError:
    st = None


def _get_openai_api_key() -> str:
    if st is not None and hasattr(st, "secrets"):
        try:
            return st.secrets["general"]["openai_api_key"]
        except Exception:
            pass
    return os.environ.get("OPENAI_API_KEY", "")


API_KEY = _get_openai_api_key()
