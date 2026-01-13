# src/climsight/tools/python_repl.py
"""
Simple Python REPL Tool for Climsight with persistent state.
Enhanced with AST parsing to ensure expression results are printed automatically.
"""

import sys
import logging
import ast
from io import StringIO
from typing import Dict, Any
from pydantic import BaseModel, Field

try:
    from langchain.tools import StructuredTool
except ImportError:
    from langchain_core.tools import StructuredTool

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
            'os': __import__('os'),
            'Path': __import__('pathlib').Path,
        })
        
    def execute(self, code: str) -> str:
        """
        Execute Python code in the persistent environment.
        Attempts to print the result of the last expression if it exists.
        
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
            # Parse the code into an abstract syntax tree
            tree = ast.parse(code)
            
            # Check if there is code to execute
            if not tree.body:
                return "No code to execute."
                
            # Get the last node
            last_node = tree.body[-1]
            
            # If the last node is an expression (not assignment, loop, etc.), we want to print it
            if isinstance(last_node, ast.Expr):
                # Compile and exec everything BEFORE the last expression
                if len(tree.body) > 1:
                    exec_code = compile(ast.Module(body=tree.body[:-1], type_ignores=[]), "<string>", "exec")
                    exec(exec_code, self.locals, self.locals)
                
                # Compile and eval the LAST expression
                eval_code = compile(ast.Expression(body=last_node.value), "<string>", "eval")
                result = eval(eval_code, self.locals, self.locals)
                
                # Explicitly print the result so it is captured in stdout
                if result is not None:
                    print(repr(result))
            else:
                # If the last node is NOT an expression (e.g. a = 10), just exec everything normally
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
    
    class PythonREPLInput(BaseModel):
        code: str = Field(
            description="Python code to execute. Has access to pandas (pd), numpy (np), matplotlib.pyplot (plt), and xarray (xr). Variables persist between executions."
        )
    # Get available variables info
    available_vars = []
    if hasattr(_repl_instance, 'locals'):
        for key, value in _repl_instance.locals.items():
            if not key.startswith('__'):
                available_vars.append(f"{key} ({type(value).__name__})")
    
    vars_description = ""
    if available_vars:
        vars_description = f"\nPre-loaded variables: {', '.join(available_vars[:10])}..."
    
    tool = StructuredTool.from_function(
        func=_repl_instance.execute,
        name="python_repl",
        description=(
            "Execute Python code for data analysis and calculations. "
            "Available: pandas as pd, numpy as np, matplotlib.pyplot as plt, xarray as xr. "
            "Variables and imports persist between executions like a Jupyter notebook. "
            "The last expression in the code block will be automatically printed."
            + vars_description
        ),
        args_schema=PythonREPLInput
    )
    return tool