# src/climsight/tools/__init__.py
"""
Tools for Climsight smart agents.
"""

from .image_viewer import create_image_viewer_tool
from .python_repl import CustomPythonREPLTool

__all__ = ['CustomPythonREPLTool', 'create_image_viewer_tool']

# Backwards-compatible import if older code expects create_python_repl_tool.
try:
    from .python_repl import create_python_repl_tool  # type: ignore
    __all__.append('create_python_repl_tool')
except Exception:
    pass
