"""
Thread-safe in-memory session manager for ClimSight.

Replaces st.session_state with a dictionary-based store that works
with FastAPI, CLI, and any non-Streamlit context.
"""

import logging
import threading
from typing import Any, Dict

logger = logging.getLogger(__name__)


class SessionManager:
    """Singleton-style, thread-safe session store."""

    _sessions: Dict[str, Dict[str, Any]] = {}
    _lock = threading.Lock()

    @classmethod
    def get_session(cls, session_id: str) -> Dict[str, Any]:
        """Get or create a session dict for the given ID."""
        with cls._lock:
            if session_id not in cls._sessions:
                cls._sessions[session_id] = {}
                logger.info("Created new session: %s", session_id)
            return cls._sessions[session_id]

    @classmethod
    def delete_session(cls, session_id: str) -> bool:
        """Remove a session. Returns True if it existed."""
        with cls._lock:
            return cls._sessions.pop(session_id, None) is not None

    @classmethod
    def list_sessions(cls):
        """Return list of active session IDs."""
        with cls._lock:
            return list(cls._sessions.keys())
