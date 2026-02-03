"""Helper utilities for tool-based agents (PangaeaGPT parity)."""

import logging
import os
from typing import Any, Dict, List, Tuple

try:
    import streamlit as st
except ImportError:
    st = None

try:
    from langchain.agents import AgentExecutor, create_openai_tools_agent
except ImportError:
    from langchain_classic.agents import AgentExecutor, create_openai_tools_agent

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

logger = logging.getLogger(__name__)


def prepare_visualization_environment(datasets_info: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], str, List[str]]:
    """Prepare dataset variables and prompt text for tool agents."""
    datasets_text = ""
    dataset_variables: List[str] = []
    datasets: Dict[str, Any] = {}

    uuid_main_dir = None
    for info in datasets_info:
        sandbox_path = info.get("sandbox_path")
        if sandbox_path and isinstance(sandbox_path, str) and os.path.isdir(sandbox_path):
            uuid_main_dir = os.path.dirname(os.path.abspath(sandbox_path))
            logger.info("Found main UUID directory from sandbox_path: %s", uuid_main_dir)
            break

    datasets["uuid_main_dir"] = uuid_main_dir

    results_dir = None
    uuid_dir_files: List[str] = []
    if uuid_main_dir and os.path.exists(uuid_main_dir):
        results_dir = os.path.join(uuid_main_dir, "results")
        os.makedirs(results_dir, exist_ok=True)
        datasets["results_dir"] = results_dir
        try:
            uuid_dir_files = os.listdir(uuid_main_dir)
        except Exception as exc:
            logger.error("Error listing UUID directory files: %s", exc)

    # Path instructions for the prompt
    uuid_paths = "WARNING: EXACT DATASET PATHS - USE THESE EXACTLY AS SHOWN\n"
    uuid_paths += "The following paths contain unique IDs that MUST be used with os.path.join().\n\n"

    if uuid_main_dir:
        uuid_paths += "# MAIN OUTPUT DIRECTORY\n"
        uuid_paths += f"uuid_main_dir = r'{uuid_main_dir}'\n"
        uuid_paths += f"results_dir = r'{results_dir}'  # Save all plots here\n\n"
        uuid_paths += f"# Files in main directory: {', '.join(uuid_dir_files) if uuid_dir_files else 'None'}\n\n"

    for i, info in enumerate(datasets_info):
        var_name = f"dataset_{i + 1}"
        datasets[var_name] = info.get("dataset")
        dataset_variables.append(var_name)

        sandbox_path = info.get("sandbox_path")
        if sandbox_path and isinstance(sandbox_path, str) and os.path.isdir(sandbox_path):
            full_uuid_path = os.path.abspath(sandbox_path).replace("\\", "/")
            uuid_paths += f"# Dataset {i + 1}: {info.get('name', 'unknown')}\n"
            uuid_paths += f"{var_name}_path = r'{full_uuid_path}'\n\n"
            if os.path.exists(full_uuid_path):
                try:
                    files = os.listdir(full_uuid_path)
                    uuid_paths += f"# Files available in {var_name}_path: {', '.join(files)}\n\n"
                except Exception as exc:
                    uuid_paths += f"# Error listing files: {exc}\n\n"

    uuid_paths += "# WARNINGS\n"
    uuid_paths += "# 1. Never use placeholder paths.\n"
    uuid_paths += "# 2. Always use the dataset_X_path variables shown above.\n"
    uuid_paths += "# 3. Check which files exist before reading.\n\n"

    datasets_summary = ""
    for i, info in enumerate(datasets_info):
        datasets_summary += (
            f"Dataset {i + 1}:\n"
            f"Name: {info.get('name', 'Unknown')}\n"
            f"Description: {info.get('description', 'No description available')}\n"
            f"Type: {info.get('data_type', 'Unknown type')}\n"
            f"Sample Data: {info.get('df_head', 'No sample available')}\n\n"
        )

    datasets_text = uuid_paths + datasets_summary

    if st is not None and hasattr(st, "session_state"):
        st.session_state["viz_datasets_text"] = datasets_text

    return datasets, datasets_text, dataset_variables


def create_standard_agent_executor(llm, tools, prompt_template, max_iterations: int = 25) -> AgentExecutor:
    """Create an OpenAI tools agent executor with standard wiring."""
    agent = create_openai_tools_agent(
        llm,
        tools=tools,
        prompt=ChatPromptTemplate.from_messages(
            [
                ("system", prompt_template),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="messages"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        ),
    )

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=max_iterations,
        return_intermediate_steps=True,
    )
