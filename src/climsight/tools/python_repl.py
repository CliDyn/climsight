# src/climsight/tools/python_repl.py
"""
Python REPL Tool for Climsight using Jupyter Kernel.
Executes code in a persistent, isolated Jupyter Kernel process.
"""

import os
import sys
import logging
import re
import threading
import queue
import time
import atexit
import textwrap
from typing import Any, Dict, Optional, List, Set
from pydantic import BaseModel, Field, PrivateAttr

# Import LangChain components
try:
    from langchain.tools import StructuredTool
except ImportError:
    from langchain_core.tools import StructuredTool
from langchain_experimental.tools import PythonREPLTool

# Import Jupyter Client
try:
    from jupyter_client import KernelManager
    from jupyter_client.client import KernelClient
except ImportError:
    raise ImportError("Missing dependencies for persistent REPL. Install jupyter_client and ipykernel.")

logger = logging.getLogger(__name__)

# --- Helper: Unified Session ID ---

def get_global_session_id() -> str:
    """
    Returns a unified session ID from environment variable or default fallback.
    """
    return os.environ.get("CLIMSIGHT_THREAD_ID", "default_cli_session")

# --- Jupyter Kernel Executor (ISOLATED PROCESS) ---

class JupyterKernelExecutor:
    """
    Manages a persistent Jupyter kernel for code execution.
    Running in a separate process ensures isolation.
    """
    def __init__(self, working_dir=None):
        self._working_dir = working_dir
        # Use default python3 kernel
        self.km = KernelManager(kernel_name="python3")
        self.kc: Optional[KernelClient] = None
        self.is_initialized = False
        self._start_kernel()

    def _start_kernel(self):
        cwd = self._working_dir if (self._working_dir and os.path.exists(self._working_dir)) else os.getcwd()
        logging.info(f"Starting Jupyter kernel in {cwd}...")
        
        try:
            # Robust check for kernel spec
            if not self.km.kernel_spec:
                raise RuntimeError("No kernel spec found")
        except Exception as e:
            logging.warning(f"Kernel spec warning: {e}. Forcing sys.executable for ipykernel.")
            # Fallback: explicitly call the current python executable to avoid path issues
            self.km.kernel_cmd = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]

        try:
            self.km.start_kernel(cwd=cwd)
            self.kc = self.km.client()
            self.kc.start_channels()
            self.kc.wait_for_ready(timeout=60)
            logging.info(f"Jupyter kernel started successfully.")
        except Exception as e:
            logging.error(f"Kernel failed to start: {e}")
            self.close()
            raise

    def restart_kernel(self):
        """Hard restart in case of stuck process or timeout."""
        logging.warning("Restarting Jupyter Kernel...")
        self.close()
        self._start_kernel()
        self.is_initialized = False

    def _drain_channels(self, timeout=1.0):
        """Drain messages from channels to prevent cross-contamination after interrupt/timeout."""
        if not self.kc: return
        start = time.time()
        while time.time() - start < timeout:
            try:
                self.kc.get_iopub_msg(timeout=0.1)
            except queue.Empty:
                break

    def _execute_code(self, code: str, timeout: float = 300.0) -> Dict[str, Any]:
        if not self.kc:
            return {"status": "error", "error": "Kernel client not available."}

        # Flush previous messages
        self._drain_channels(timeout=0.1)

        msg_id = self.kc.execute(code)
        result = {"status": "success", "stdout": "", "stderr": "", "display_data": []}
        start_time = time.time()

        while True:
            # 1. Check Timeout
            if time.time() - start_time > timeout:
                logging.warning(f"Code execution timed out ({timeout}s). Interrupting kernel.")
                self.km.interrupt_kernel()
                # Drain leftover messages from the interrupted execution
                self._drain_channels(timeout=2.0)
                result["status"] = "error"
                result["error"] = f"Timeout after {timeout}s. Execution interrupted."
                break

            # 2. Check Kernel Vitality
            if not self.km.is_alive():
                result["status"] = "error"
                result["error"] = "Kernel died unexpectedly. Session was restarted - please retry your command."
                self.restart_kernel()
                self.is_initialized = False  # Force re-initialization on next run
                break

            # 3. Get Message
            try:
                msg = self.kc.get_iopub_msg(timeout=0.1)
            except queue.Empty:
                continue

            # 4. Filter by Parent Message ID
            if msg['parent_header'].get('msg_id') != msg_id:
                continue

            msg_type = msg['msg_type']
            content = msg['content']

            if msg_type == 'status' and content['execution_state'] == 'idle':
                break
            elif msg_type == 'stream':
                if content['name'] == 'stdout': result["stdout"] += content['text']
                elif content['name'] == 'stderr': result["stderr"] += content['text']
            elif msg_type in ('display_data', 'execute_result'):
                result["display_data"].append(content['data'])
                if 'text/plain' in content['data']:
                    result["stdout"] += content['data']['text/plain'] + "\n"
            elif msg_type == 'error':
                result["status"] = "error"
                # Remove ANSI colors for readable error logs
                error_trace = "\n".join(content['traceback'])
                clean_error = re.sub(r'\x1b\[[0-9;]*m', '', error_trace)
                result["error"] = f"{content['ename']}: {content['evalue']}\n{clean_error}"
                break

        return result

    def run(self, code: str) -> Dict[str, Any]:
        """
        Executes code and returns full result dict. 
        Sanitizes markdown fences but PRESERVES indentation.
        """
        # 1. Dedent to fix common indentation errors (e.g. code inside list items)
        code = textwrap.dedent(code)
        
        # 2. Remove markdown fences
        code = code.strip()
        if code.startswith("```"):
             # Remove leading ```python or ``` (case insensitive)
             code = re.sub(r"^\s*```(?:python)?\s*\n", "", code, flags=re.IGNORECASE)
             # Remove trailing ```
             code = re.sub(r"\n\s*```\s*$", "", code)
        
        return self._execute_code(code)

    def close(self):
        if self.kc: 
            try: self.kc.stop_channels()
            except: pass
        if self.km and self.km.is_alive(): 
            try: self.km.shutdown_kernel(now=True)
            except: pass

# --- Session Manager ---

class REPLManager:
    """Singleton to manage Kernels per session (thread_id)."""
    _instances: dict[str, JupyterKernelExecutor] = {}
    _lock = threading.Lock()

    @classmethod
    def get_repl(cls, session_id: str, sandbox_path: str = None) -> JupyterKernelExecutor:
        with cls._lock:
            if session_id not in cls._instances:
                logging.info(f"Creating new Kernel for session: {session_id}")
                try:
                    cls._instances[session_id] = JupyterKernelExecutor(working_dir=sandbox_path)
                except Exception as e:
                    logging.error(f"Failed to create kernel: {e}")
                    raise
            
            # Restart dead kernels if necessary (Zombie check)
            if not cls._instances[session_id].km.is_alive():
                 logging.warning(f"Kernel for {session_id} died. Restarting.")
                 cls._instances[session_id].close()
                 cls._instances[session_id] = JupyterKernelExecutor(working_dir=sandbox_path)

            return cls._instances[session_id]

    @classmethod
    def cleanup_all(cls):
        for sid in list(cls._instances.keys()):
            try:
                cls._instances[sid].close()
            except Exception:
                pass
            del cls._instances[sid]

atexit.register(REPLManager.cleanup_all)

# --- LangChain Tool Wrappers ---

class CustomPythonREPLTool(PythonREPLTool):
    """Used by data_analysis_agent. Imports dataset paths automatically."""
    _datasets: dict = PrivateAttr()
    _results_dir: Optional[str] = PrivateAttr()
    _session_key: Optional[str] = PrivateAttr()

    def __init__(self, datasets, results_dir=None, session_key=None, **kwargs):
        super().__init__(**kwargs)
        self._datasets = datasets
        self._results_dir = results_dir
        self._session_key = session_key
    
    def _run(self, query: str, **kwargs) -> Any:
        # 1. Get Unified Session ID
        sid = self._session_key or get_global_session_id()
        
        # 2. Get/Create Kernel
        sandbox_path = self._datasets.get("uuid_main_dir")
        try:
            repl = REPLManager.get_repl(sid, sandbox_path=sandbox_path)
        except Exception as e:
            return f"System Error: Could not start Python environment. {str(e)}"
        
        # 3. Auto-init if fresh
        if not repl.is_initialized:
            # Initialize matplotlib backend inside the kernel!
            init_code = [
                "import pandas as pd",
                "import numpy as np",
                "import matplotlib",
                "matplotlib.use('Agg')", # CRITICAL: Non-interactive backend
                "import matplotlib.pyplot as plt",
                "import xarray as xr",
                "import os",
                "import json"
            ]
            
            # Safe path injection using repr()
            if self._results_dir:
                # FIX: Since the kernel CWD is already the sandbox root, 
                init_code.append(f"results_dir = 'results'") 
                init_code.append("os.makedirs(results_dir, exist_ok=True)")
            
            if "climate_data_dir" in self._datasets:
                # FIX: Set simple relative path because Kernel CWD is already the sandbox root
                init_code.append(f"climate_data_dir = 'climate_data'")
                # Auto-load main data if exists
                init_code.append(f"try:\n    if os.path.exists(f'{{climate_data_dir}}/data.csv'):\n        df = pd.read_csv(f'{{climate_data_dir}}/data.csv')\n        print('Loaded df from data.csv')\nexcept Exception as e: print(f'Auto-load failed: {{e}}')")

            if "era5_data_dir" in self._datasets:
                # FIX: Set simple relative path because Kernel CWD is already the sandbox root
                init_code.append(f"era5_data_dir = 'era5_data'")

            # Execute and check status
            init_res = repl.run("\n".join(init_code))
            if init_res["status"] == "success":
                repl.is_initialized = True
            else:
                return f"Kernel Initialization Error: {init_res.get('error')}"

        # 4. Detect Plots (Snapshot Method - Robust)
        existing_files: Set[str] = set()
        if self._results_dir and os.path.exists(self._results_dir):
            try:
                existing_files = set(os.listdir(self._results_dir))
            except Exception:
                pass

        # 5. Execute User Code
        res = repl.run(query)
        
        output_str = res["stdout"]
        if res["stderr"]:
            output_str += f"\n[STDERR]\n{res['stderr']}"
        
        if res["status"] == "error":
            return f"Error:\n{res.get('error')}"

        # 6. Identify New Plots (Diff)
        plots = []
        if self._results_dir and os.path.exists(self._results_dir):
            try:
                current_files = set(os.listdir(self._results_dir))
                new_files = current_files - existing_files
                
                for f in new_files:
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf', '.svg')):
                        full_path = os.path.join(self._results_dir, f)
                        plots.append(full_path)
            except Exception:
                pass
        
        return {
            "result": output_str.strip() or "Code executed successfully (no output).", 
            "output": output_str.strip(), 
            "plot_images": plots
        }

def create_python_repl_tool() -> StructuredTool:
    """Factory for simple use cases (Smart Agent)."""
    class Input(BaseModel):
        code: str = Field(description="Python code to execute.")

    def run_repl(code: str):
        sid = get_global_session_id()
        
        # Determine sandbox path
        sandbox = None
        if sid != "default_cli_session":
             sandbox = os.path.join("tmp", "sandbox", sid)
             if not os.path.exists(sandbox):
                 os.makedirs(sandbox, exist_ok=True)
            
        try:
            repl = REPLManager.get_repl(sid, sandbox_path=sandbox)
            # Basic init
            if not repl.is_initialized:
                 repl.run("import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt; import pandas as pd; import numpy as np; import os")
                 repl.is_initialized = True

            res = repl.run(code)
            if res["status"] == "success":
                return res["stdout"] or "Executed."
            return f"Error: {res.get('error')}"
        except Exception as e:
            return f"Kernel Error: {e}"

    return StructuredTool.from_function(
        func=run_repl,
        name="python_repl",
        description="Execute Python code in a persistent Jupyter Kernel. State is preserved. Use this for calculations and plotting.",
        args_schema=Input
    )