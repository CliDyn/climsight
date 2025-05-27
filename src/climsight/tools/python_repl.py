# src/climsight/tools/python_repl.py
"""
Simple Python REPL Tool for Climsight with persistent state.
"""

import sys
import logging
from io import StringIO
from typing import Dict, Any

logger = logging.getLogger(__name__)


class PersistentPythonREPL:
    """
    A persistent Python REPL that maintains state between executions.
    Variables created in one execution persist to the next, like a Jupyter notebook.
    """
    
    def __init__(self):
        """Initialize with an empty locals dictionary and pre-imported modules."""
        # Persistent locals dictionary - this is THE KEY FEATURE
        self.locals = {}
        
        # Pre-import common modules into the persistent namespace
        self.locals.update({
            'pd': __import__('pandas'),
            'np': __import__('numpy'),
            'plt': __import__('matplotlib.pyplot', fromlist=['pyplot']),
            'xr': __import__('xarray'),
        })
        
    def execute(self, code: str) -> str:
        """
        Execute Python code in the persistent environment.
        
        Args:
            code: Python code to execute
            
        Returns:
            String containing output or error message
        """
        # Capture stdout
        old_stdout = sys.stdout
        stdout_capture = StringIO()
        sys.stdout = stdout_capture
        
        try:
            # Try to execute as an expression first (to show return values)
            try:
                result = eval(code, self.locals, self.locals)
                if result is not None:
                    print(repr(result))
            except SyntaxError:
                # If it's not an expression, execute as statement(s)
                exec(code, self.locals, self.locals)
            
            # Get the output
            output = stdout_capture.getvalue()
            
            if output:
                return f"Output:\n{output}"
            else:
                return "Code executed successfully (no output)."
                
        except Exception as e:
            return f"Error: {type(e).__name__}: {str(e)}"
            
        finally:
            # Restore stdout
            sys.stdout = old_stdout


# Create a global instance for persistence across tool calls
_repl_instance = PersistentPythonREPL()


def create_python_repl_tool():
    """
    Create a Python REPL tool for LangChain agents with persistent state.
    
    Returns:
        StructuredTool configured for Python code execution
    """
    from langchain.tools import StructuredTool
    from pydantic import BaseModel, Field
    
    class PythonREPLInput(BaseModel):
        code: str = Field(
            description="Python code to execute. Has access to pandas (pd), numpy (np), matplotlib.pyplot (plt), and xarray (xr). Variables persist between executions."
        )
    
    tool = StructuredTool.from_function(
        func=_repl_instance.execute,  # Use the instance method
        name="python_repl",
        description=(
            "Execute Python code for data analysis and calculations. "
            "Available: pandas as pd, numpy as np, matplotlib.pyplot as plt, xarray as xr. "
            "Variables and imports persist between executions like a Jupyter notebook."
        ),
        args_schema=PythonREPLInput
    )
    
    return tool