# src/tools/python_repl.py

import os
import sys
import logging
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import streamlit as st
from io import StringIO
from pydantic import BaseModel, Field, PrivateAttr
from langchain_experimental.tools import PythonREPLTool
from typing import Any, Dict, Optional, List
import re
import threading
import json
import queue
import time
import atexit

# Import Jupyter Client
try:
    # We use KernelManager for lifecycle management and KernelClient for communication
    from jupyter_client import KernelManager
    from jupyter_client.client import KernelClient
except ImportError:
    logging.error("Jupyter client not installed. Please run 'pip install jupyter_client ipykernel'")
    # Raise error to ensure environment is correctly set up
    raise ImportError("Missing dependencies for persistent REPL. Install jupyter_client and ipykernel.")

try:
    from ..utils import log_history_event
except ImportError:
    from utils import log_history_event

# --- New Jupyter Kernel Executor ---

class JupyterKernelExecutor:
    """
    Manages a persistent Jupyter kernel for code execution.
    Replaces the previous subprocess-based PersistentREPL.
    """
    def __init__(self, working_dir=None):
        self._working_dir = working_dir
        # Initialize the KernelManager to use the Climsight kernel by default.
        kernel_name = os.environ.get("CLIMSIGHT_KERNEL_NAME", "climsight")
        self.km = KernelManager(kernel_name=kernel_name)
        self.kc: Optional[KernelClient] = None
        self.is_initialized = False
        self._start_kernel()

    def _start_kernel(self):
        """Starts the Jupyter kernel."""
        logging.info("Starting Jupyter kernel...")
        
        # Ensure the ipykernel is available.
        try:
            # Accessing kernel_spec validates that the kernel is installed and found
            self.km.kernel_spec
        except Exception as e:
            logging.warning(
                "Could not find python3 kernel spec. Falling back to current interpreter. Error: %s",
                e,
            )
            self.km.kernel_cmd = [
                sys.executable,
                "-m",
                "ipykernel_launcher",
                "-f",
                "{connection_file}",
            ]

        # If kernel spec points to a missing interpreter, fall back to current Python.
        try:
            kernel_cmd = list(self.km.kernel_cmd or [])
            if kernel_cmd:
                exec_path = kernel_cmd[0]
                if os.path.isabs(exec_path) and not os.path.exists(exec_path):
                    logging.warning(
                        "Kernel spec uses missing interpreter: %s. Using current Python.",
                        exec_path,
                    )
                    self.km.kernel_cmd = [
                        sys.executable,
                        "-m",
                        "ipykernel_launcher",
                        "-f",
                        "{connection_file}",
                    ]
        except Exception as e:
            logging.warning("Failed to validate kernel_cmd: %s", e)

        # Start the kernel in the specified working directory (CWD)
        if self._working_dir and os.path.exists(self._working_dir):
            self.km.start_kernel(cwd=self._working_dir)
        else:
            self.km.start_kernel()

        # Create the synchronous client
        self.kc = self.km.client()
        self.kc.start_channels()
        
        # Wait for the kernel to be ready
        try:
            self.kc.wait_for_ready(timeout=60)
            logging.info("Jupyter kernel started and ready.")
        except RuntimeError as e:
            logging.error(f"Kernel failed to start: {e}")
            self.km.shutdown_kernel()
            raise

    def _execute_code(self, code: str, timeout: float = 300.0) -> Dict[str, Any]:
        """Executes code synchronously in the kernel and captures outputs."""
        if not self.kc:
            return {"status": "error", "error": "Kernel client not available."}

        # Send the execution request
        msg_id = self.kc.execute(code)
        result = {
            "status": "success",
            "stdout": "",
            "stderr": "",
            "display_data": []
        }
        start_time = time.time()

        # Process messages until the kernel is idle or timeout occurs
        while True:
            if time.time() - start_time > timeout:
                logging.warning("Kernel execution timed out. Interrupting kernel.")
                self.km.interrupt_kernel()
                result["status"] = "error"
                result["error"] = f"Execution timed out after {timeout} seconds."
                break

            try:
                # Get messages from the IOPub channel (where outputs are published)
                # Use a small timeout to remain responsive
                msg = self.kc.get_iopub_msg(timeout=0.1)
            except queue.Empty:
                # Check if the kernel is still alive if the queue is empty
                if not self.km.is_alive():
                    result["status"] = "error"
                    result["error"] = "Kernel died unexpectedly."
                    break
                continue
            except Exception as e:
                logging.error(f"Error getting IOPub message: {e}")
                continue

            # Process the message if it relates to our execution request
            if msg['parent_header'].get('msg_id') == msg_id:
                msg_type = msg['msg_type']

                if msg_type == 'status':
                    if msg['content']['execution_state'] == 'idle':
                        break  # Execution finished
                elif msg_type == 'stream':
                    if msg['content']['name'] == 'stdout':
                        result["stdout"] += msg['content']['text']
                    elif msg['content']['name'] == 'stderr':
                        result["stderr"] += msg['content']['text']
                elif msg_type == 'display_data' or msg_type == 'execute_result':
                    # Capture rich display data (e.g., for interactive plots in the future)
                    result["display_data"].append(msg['content']['data'])
                    # Also capture text representation for the current agent output
                    if 'text/plain' in msg['content']['data']:
                        result["stdout"] += msg['content']['data']['text/plain'] + "\n"
                elif msg_type == 'error':
                    result["status"] = "error"
                    ename = msg['content']['ename']
                    evalue = msg['content']['evalue']
                    # Clean ANSI escape codes from traceback
                    traceback = "\n".join(msg['content']['traceback'])
                    traceback = re.sub(r'\x1b\[[0-9;]*m', '', traceback)
                    result["error"] = f"{ename}: {evalue}\n{traceback}"
                    break

        return result

    @staticmethod
    def sanitize_input(query: str) -> str:
        """Sanitize input (from LangChain)."""
        # Remove surrounding backticks, whitespace, and the 'python' keyword
        query = re.sub(r"^(\s|`)*(?i:python)?\s*", "", query)
        query = re.sub(r"(\s|`)*$", "", query)
        return query

    def _sanitize_code(self, code: str) -> str:
        """Clean specific syntax."""
        # Jupyter kernels handle magic commands natively, so we just need LangChain sanitization
        return self.sanitize_input(code)

    def run(self, code: str, timeout: int = 300) -> str:
        """Execute code in the Jupyter kernel and return formatted output."""
        cleaned_code = self._sanitize_code(code)

        result = self._execute_code(cleaned_code, timeout)

        if result["status"] == "success":
            output = result["stdout"]
            if result.get("stderr"):
                # Include stderr output (often contains warnings)
                if output:
                    output += "\n[STDERR]\n" + result["stderr"]
                else:
                    output = "[STDERR]\n" + result["stderr"]
            return output.strip()
        elif result["status"] == "error":
            return result["error"]
        else:
            return f"Fatal error: {result.get('error', 'Unknown error')}"

    def close(self):
        """Cleanup and terminate the kernel."""
        if self.kc:
            self.kc.stop_channels()
        if hasattr(self, 'km') and self.km.is_alive():
            logging.info("Shutting down Jupyter kernel...")
            self.km.shutdown_kernel(now=True)
            logging.info("Jupyter kernel shut down.")

# REPLManager updated to manage JupyterKernelExecutor instances
class REPLManager:
    """
    Singleton for managing JupyterKernelExecutor instances for each session (thread_id).
    """
    _instances: dict[str, JupyterKernelExecutor] = {}
    _lock = threading.Lock()

    @classmethod
    def get_repl(cls, session_id: str, sandbox_path: str = None) -> JupyterKernelExecutor:
        with cls._lock:
            if session_id not in cls._instances:
                logging.info(f"Creating new Jupyter Kernel for session: {session_id}")
                if sandbox_path:
                    logging.info(f"Starting Kernel in sandbox directory: {sandbox_path}")
                try:
                    cls._instances[session_id] = JupyterKernelExecutor(working_dir=sandbox_path)
                except Exception as e:
                    logging.error(f"Failed to create Jupyter Kernel for session {session_id}: {e}")
                    # Raise the exception so the calling agent can handle the failure
                    raise RuntimeError(f"Failed to initialize execution environment: {e}")
            return cls._instances[session_id]

    @classmethod
    def cleanup_repl(cls, session_id: str):
        with cls._lock:
            if session_id in cls._instances:
                logging.info(f"Closing and removing Kernel for session: {session_id}")
                cls._instances[session_id].close()
                del cls._instances[session_id]

    @classmethod
    def cleanup_all(cls):
        """Cleanup all running kernels."""
        logging.info("Cleaning up all kernels on exit...")
        sessions = list(cls._instances.keys())
        for session_id in sessions:
            cls.cleanup_repl(session_id)

# Ensure kernels are cleaned up when the application exits
atexit.register(REPLManager.cleanup_all)

# --- End of new code ---


class CustomPythonREPLTool(PythonREPLTool):
    _datasets: dict = PrivateAttr()
    _results_dir: Optional[str] = PrivateAttr()
    _session_key: Optional[str] = PrivateAttr()

    def __init__(self, datasets, results_dir=None, session_key=None, **kwargs):
        super().__init__(**kwargs)
        self._datasets = datasets
        self._results_dir = results_dir
        self._session_key = session_key

    def _initialize_session(self, repl: JupyterKernelExecutor) -> None:
        """Prepare the Kernel session by importing libraries and auto-loading data."""
        logging.info("Initializing persistent Kernel session with auto-loading...")

        initialization_code = [
            "import os",
            "import sys",
            "import pandas as pd",
            "import numpy as np",
            "import matplotlib.pyplot as plt",
            "import xarray as xr",
            "import logging",
            "from io import StringIO",
            # Configure matplotlib for non-interactive backend (Crucial for kernels)
            "import matplotlib",
            "matplotlib.use('Agg')"
        ]

        # Add results_dir if available
        if self._results_dir:
            # Ensure paths are correctly formatted for the kernel environment (use forward slashes)
            results_dir_path = os.path.abspath(self._results_dir).replace('\\', '/')
            initialization_code.append(f"results_dir = r'{results_dir_path}'")
            # Kernel CWD should handle the existence, but we ensure the 'results' subdirectory exists
            initialization_code.append("os.makedirs(results_dir, exist_ok=True)")

        # Inject dataset variables by auto-loading from files
        for key, value in self._datasets.items():
            if key.startswith('dataset_') and isinstance(value, str) and os.path.isdir(value):
                # 1. Create the path variable (e.g., dataset_1_path)
                path_var_name = f"{key}_path"
                abs_path = os.path.abspath(value).replace('\\', '/')
                initialization_code.append(f"{path_var_name} = r'{abs_path}'")

                # 2. Attempt to find and auto-load data.csv
                csv_path = os.path.join(value, 'data.csv')
                if os.path.exists(csv_path):
                    initialization_code.append(f"# Auto-loading {key} from data.csv")
                    initialization_code.append(f"try:")
                    initialization_code.append(f"    {key} = pd.read_csv(os.path.join({path_var_name}, 'data.csv'))")
                    initialization_code.append(f"    print(f'Successfully auto-loaded data.csv into `{key}` variable.')")
                    initialization_code.append(f"except Exception as e:")
                    initialization_code.append(f"    print(f'Could not auto-load {key}: {{e}}')")

            elif key == "uuid_main_dir" and isinstance(value, str):
                # Add the main UUID directory
                abs_path = os.path.abspath(value).replace('\\', '/')
                initialization_code.append(f"uuid_main_dir = r'{abs_path}'")

        # Execute all initialization code in one go
        full_init_code = "\n".join(initialization_code)
        result = repl.run(full_init_code)
        logging.info(f"Kernel session initialization result: {result.strip()}")
        repl.is_initialized = True

    def _run(self, query: str, **kwargs) -> Any:
        """
        Execute code using the persistent Jupyter Kernel tied to the session.
        """
        # Resolve session_id for Streamlit or CLI execution.
        session_id = ""
        if hasattr(self, "_session_key") and self._session_key:
            session_id = self._session_key
        else:
            try:
                session_id = st.session_state.thread_id
            except Exception:
                session_id = os.environ.get("CLIMSIGHT_THREAD_ID", "")

        if not session_id:
            logging.error("CRITICAL ERROR: Session ID (thread_id) is missing.")
            return {
                "result": "ERROR: Session ID is missing. Cannot continue execution.",
                "output": "ERROR: Session ID is missing. Cannot continue execution.",
                "plot_images": []
            }

        # Get sandbox path from datasets (this is the intended working directory)
        sandbox_path = self._datasets.get("uuid_main_dir")
        
        try:
            # Retrieve or create the kernel executor
            repl = REPLManager.get_repl(session_id, sandbox_path=sandbox_path)
        except RuntimeError as e:
            # Handle kernel initialization failure
            return {
                "result": f"ERROR: Failed to initialize the execution environment (Jupyter Kernel). Details: {e}",
                "output": f"ERROR: Failed to initialize the execution environment (Jupyter Kernel). Details: {e}",
                "plot_images": []
            }

        # "Warm up" the REPL on first call in the session
        if not repl.is_initialized:
            self._initialize_session(repl)

        logging.info(f"Executing code in persistent Kernel for session {session_id}:\n{query}")

        # Execute user code
        output = repl.run(query)

        # Logic for detecting generated plots
        generated_plots = []
        image_extensions = ('.png', '.jpg', '.jpeg', '.svg', '.pdf')

        # --- Plot Detection Logic ---
        # The kernel executes with the sandbox_path as its CWD.
        # We rely on the agent saving plots to relative paths (e.g., 'results/plot.png')
        
        # 1. Check for plot files mentioned in the output (stdout/stderr)
        # Use regex to find potential paths (e.g., "Plot saved to: results/my_plot.png")
        matches = re.findall(r'([a-zA-Z0-9_\-/\\.]+\.(png|jpg|jpeg|svg|pdf))', output, re.IGNORECASE)
        for match in matches:
            potential_path = match[0].strip()
            
            # Resolve the path relative to the sandbox directory (where the kernel is running)
            if sandbox_path:
                # Normalize slashes
                potential_path = potential_path.replace('\\', '/')
                full_potential_path = os.path.join(sandbox_path, potential_path)
            else:
                # If no sandbox path (e.g. CLI fallback), check relative to current working dir
                full_potential_path = os.path.abspath(potential_path)

            if os.path.exists(full_potential_path):
                if full_potential_path not in generated_plots:
                    generated_plots.append(full_potential_path)
                    logging.info(f"Found plot path in output: {full_potential_path}")

        # 2. Check results directory for recently created files (fallback mechanism)
        if self._results_dir and os.path.exists(self._results_dir):
            for file in os.listdir(self._results_dir):
                if file.lower().endswith(image_extensions):
                    full_path = os.path.join(self._results_dir, file)
                    if full_path not in generated_plots and os.path.exists(full_path):
                        # Check if file was created recently (within the last 10 seconds)
                        # This prevents picking up old plots from the same session
                        if time.time() - os.path.getmtime(full_path) < 10:
                             generated_plots.append(full_path)
                             logging.info(f"Found recent plot in results directory: {full_path}")

        if generated_plots:
            # Update UI flag if using Streamlit (safe check)
            if 'streamlit' in sys.modules and hasattr(st, 'session_state'):
                st.session_state.new_plot_generated = True
            
            log_history_event(
                st.session_state, "plot_generated",
                {"plot_paths": generated_plots, "agent": "PythonREPL", "content": query}
            )

        return {
            "result": output,
            "output": output,  # For backward compatibility
            "plot_images": generated_plots
        }
