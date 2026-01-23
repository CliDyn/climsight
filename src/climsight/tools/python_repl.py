# src/climsight/tools/python_repl.py
"""
Simple Python REPL Tool for Climsight.
Thread-safe implementation that maintains state within a specific agent execution context.
Includes Matplotlib backend fixes and Markdown sanitization.
"""

import sys
import logging
import ast
import re
import traceback
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
    A Python REPL that maintains state between executions within a single session.
    """
    
    def __init__(self):
        """Initialize with an empty locals dictionary and pre-imported modules."""
        self.locals = {}
        
        # Pre-import common modules into the namespace
        self.locals.update({
            'pd': __import__('pandas'),
            'np': __import__('numpy'),
            'plt': plt,
            'xr': __import__('xarray'),
            'os': __import__('os'),
            'Path': __import__('pathlib').Path,
        })
        
    def execute(self, code: str) -> str:
        """
        Execute Python code and capture stdout/stderr.
        Automatically prints the result of the last expression.
        """
        # 1. Sanitize input (remove Markdown code blocks)
        # This fixes your specific SyntaxError
        code = re.sub(r"^(\s|`)*(?i:python)?\s*", "", code)
        code = re.sub(r"(\s|`)*$", "", code)

        # 2. Capture stdout/stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        stdout_capture = StringIO()
        stderr_capture = StringIO()
        sys.stdout = stdout_capture
        sys.stderr = stderr_capture

        try:
            # 3. Parse the code
            tree = ast.parse(code)
            
            if not tree.body:
                return "No code to execute."

            # 4. Execution Logic: Handle last expression printing
            last_node = tree.body[-1]
            if isinstance(last_node, ast.Expr):
                # Execute everything before the last expression
                if len(tree.body) > 1:
                    module = ast.Module(body=tree.body[:-1], type_ignores=[])
                    exec(compile(module, "<string>", "exec"), self.locals, self.locals)
                
                # Evaluate the last expression and print result
                expr = ast.Expression(body=last_node.value)
                result = eval(compile(expr, "<string>", "eval"), self.locals, self.locals)
                if result is not None:
                    print(repr(result))
            else:
                # Execute the whole block if it doesn't end in an expression
                exec(compile(tree, "<string>", "exec"), self.locals, self.locals)

            output = stdout_capture.getvalue()
            errors = stderr_capture.getvalue()
            
            # Combine output
            final_output = output
            if errors:
                final_output += f"\n[Stderr]: {errors}"
                
            if not final_output.strip():
                return "Code executed successfully (no output)."
                
            return final_output.strip()

        except Exception:
            # Capture the full traceback for debugging
            return traceback.format_exc()
            
        finally:
            # Restore stdout/stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def create_python_repl_tool() -> StructuredTool:
    """
    Create a NEW Python REPL tool instance.
    
    CRITICAL CHANGE: This creates a NEW PersistentPythonREPL() every time it is called.
    This ensures that User A's variables do not leak into User B's session in Streamlit.
    """
    
    # Instantiate a fresh REPL for this specific tool instance
    repl_instance = PersistentPythonREPL()

    class PythonREPLInput(BaseModel):
        code: str = Field(
            description="Python code to execute. Pre-loaded: pandas(pd), numpy(np), matplotlib.pyplot(plt), xarray(xr)."
        )

    # Bind the execute method of THIS specific instance
    tool = StructuredTool.from_function(
        func=repl_instance.execute,
        name="python_repl",
        description=(
            "Execute Python code for data analysis and visualization. "
            "State persists between calls within this session. "
            "The last line expression is automatically printed. "
            "Plots must be saved to 'work_dir' using plt.savefig()."
        ),
        args_schema=PythonREPLInput
    )
    
    return tool