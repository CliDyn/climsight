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
import os
import matplotlib
# Force non-interactive backend before importing pyplot
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import StringIO
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, PrivateAttr

# Try importing StructuredTool from different locations based on version
try:
    from langchain.tools import StructuredTool
except ImportError:
    from langchain_core.tools import StructuredTool
from langchain_experimental.tools import PythonREPLTool

logger = logging.getLogger(__name__)

class PersistentPythonREPL:
    """
    A persistent Python REPL that maintains state between executions within a single session.
    Replaces the Jupyter Kernel approach with a lighter in-process exec approach.
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
            'json': __import__('json'),
        })
        # Add matplotlib.pyplot explicitly as plt (redundant but safe)
        self.locals['plt'] = plt

    def execute(self, code: str) -> str:
        """
        Execute Python code and capture stdout/stderr.
        Automatically prints the result of the last expression.
        """
        # 1. Sanitize input (remove Markdown code blocks)
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

class CustomPythonREPLTool(PythonREPLTool):
    """
    A wrapper class to maintain compatibility with Agent logic that expects 
    CustomPythonREPLTool. It uses PersistentPythonREPL internally.
    """
    _datasets: dict = PrivateAttr()
    _results_dir: Optional[str] = PrivateAttr()
    _session_key: Optional[str] = PrivateAttr()
    _repl_instance: PersistentPythonREPL = PrivateAttr()

    def __init__(self, datasets, results_dir=None, session_key=None, **kwargs):
        super().__init__(**kwargs)
        self._datasets = datasets
        self._results_dir = results_dir
        self._session_key = session_key
        self._repl_instance = PersistentPythonREPL()
        self._initialize_session()

    def _initialize_session(self) -> None:
        """Pre-load variables into the REPL locals."""
        logging.info("Initializing PersistentPythonREPL session...")
        
        # Inject dataset variables
        if self._results_dir:
            self._repl_instance.locals['results_dir'] = self._results_dir
            # Ensure it exists
            os.makedirs(self._results_dir, exist_ok=True)

        if self._datasets.get("uuid_main_dir"):
             self._repl_instance.locals['uuid_main_dir'] = self._datasets["uuid_main_dir"]
        
        # Auto-load data.csv if available
        climate_data_dir = self._datasets.get("climate_data_dir")
        if climate_data_dir:
            self._repl_instance.locals['climate_data_dir'] = climate_data_dir
            data_csv = os.path.join(climate_data_dir, 'data.csv')
            if os.path.exists(data_csv):
                try:
                    pd = self._repl_instance.locals['pd']
                    self._repl_instance.locals['df'] = pd.read_csv(data_csv)
                    logging.info(f"Auto-loaded dataframe 'df' from {data_csv}")
                except Exception as e:
                    logging.error(f"Failed to auto-load data.csv: {e}")

    def _run(self, query: str, **kwargs) -> Any:
        """
        Execute code using the persistent REPL.
        """
        logging.info(f"Executing code via PersistentPythonREPL:\n{query}")
        
        # Execute
        output = self._repl_instance.execute(query)
        
        # Detect plots in results_dir
        generated_plots = []
        if self._results_dir and os.path.exists(self._results_dir):
             # Logic to find recently created images
             # This simplistic check just looks for files mentioned in the output or assumes recent creation
             matches = re.findall(r'([a-zA-Z0-9_\-/\\.]+\.(png|jpg|jpeg|svg|pdf))', output, re.IGNORECASE)
             for match in matches:
                 path = match[0]
                 # Handle relative paths if they start with results_dir name
                 if os.path.basename(self._results_dir) in path:
                     full_path = os.path.join(os.path.dirname(self._results_dir), path)
                     if os.path.exists(full_path):
                         generated_plots.append(full_path)
                 # Check direct existence
                 elif os.path.exists(path):
                     generated_plots.append(path)

        return {
            "result": output,
            "output": output,
            "plot_images": list(set(generated_plots)) # Deduplicate
        }

# Factory function for direct tool creation (used by Smart Agent)
def create_python_repl_tool() -> StructuredTool:
    """
    Create a NEW Python REPL tool instance.
    CRITICAL CHANGE: This creates a NEW PersistentPythonREPL() every time it is called.
    """
    from pydantic import BaseModel, Field
    
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
            "Plots must be saved to 'results_dir' using plt.savefig()."
        ),
        args_schema=PythonREPLInput
    )
    return tool