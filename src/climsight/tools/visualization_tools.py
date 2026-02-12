# src/tools/visualization_tools.py
import os
import logging
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import streamlit as st

try:
    from ..config import API_KEY as _API_KEY
except ImportError:
    from config import API_KEY as _API_KEY

class ExampleVisualizationArgs(BaseModel):
    query: str = Field(description="The user's query about plotting.")

def get_example_of_visualizations(query: str) -> str:
    """
    Retrieves example visualizations related to the query.

    Parameters:
    - query (str): The user's query about plotting.

    Returns:
    - str: The content of the most relevant example file.
    """
    # Initialize embeddings from session state or config
    embeddings = OpenAIEmbeddings(api_key=_API_KEY)

    # Load the existing vector store
    vector_store = Chroma(
        collection_name="example_collection",
        embedding_function=embeddings,
        persist_directory=os.path.join('data', 'examples_database', 'chroma_langchain_notebooks')
    )

    # Perform a similarity search
    results = vector_store.similarity_search_with_score(query, k=1)

    # Extract the most relevant document
    doc, score = results[0]

    # Construct the full path to the txt file
    file_name = doc.metadata['source'].lstrip('./')
    full_path = os.path.join('data', 'examples_database', file_name)

    # Read and return the content of the txt file
    try:
        with open(full_path, 'r', encoding='utf-8') as file:
            content = file.read()
            return content
    except Exception as e:
        logging.error(f"An error occurred while reading the file: {str(e)}")
        return ""  # Return empty string if error occurs

# Create the example visualization tool
example_visualization_tool = StructuredTool.from_function(
    func=get_example_of_visualizations,
    name="get_example_of_visualizations",
    description="Retrieves example visualization code related to the user's query.",
    args_schema=ExampleVisualizationArgs
)

# File listing tool definition
class ListPlottingDataFilesArgs(BaseModel):
    dummy_arg: str = Field(default="", description="(No arguments needed)")  

def list_plotting_data_files(dummy_arg: str = "") -> str:
    """
    Lists ALL files recursively from two sources:
    1. The data/plotting_data directory (static resources)
    2. All files in the current UUID sandbox directories (active datasets)
    
    Returns a flat list of all available file paths using relative paths.
    """
    import os
    import streamlit as st
    
    all_files = []
    cwd = os.getcwd()
    
    # Part 1: List files from data/plotting_data
    plotting_data_dir = os.path.join("data", "plotting_data")
    if os.path.exists(plotting_data_dir):
        for root, dirs, files in os.walk(plotting_data_dir):
            for filename in files:
                full_path = os.path.join(root, filename)
                # Keep this as a relative path
                all_files.append(f"STATIC: {full_path}")
    
    # Part 2: List all files from the current sandbox directory
    thread_id = st.session_state.get("thread_id") if hasattr(st, "session_state") else None
    if not thread_id:
        thread_id = os.environ.get("CLIMSIGHT_THREAD_ID")

    if thread_id:
        sandbox_dir = os.path.join("tmp", "sandbox", thread_id)
        if os.path.exists(sandbox_dir):
            for root, dirs, files in os.walk(sandbox_dir):
                for filename in files:
                    full_path = os.path.join(root, filename)
                    if full_path.startswith(cwd):
                        rel_path = full_path[len(cwd) + 1:]
                    else:
                        rel_path = full_path

                    rel_path = rel_path.replace('\\', '/')

                    if "era5_data" in rel_path:
                        all_files.append(f"ERA5: {rel_path}")
                    else:
                        all_files.append(f"DATA: {rel_path}")
    
    # Return a simple list of all available files
    if all_files:
        return "Available files:\n" + "\n".join(all_files)
    else:
        return "No files found in plotting_data or active datasets."

# Create the list plotting data files tool
list_plotting_data_files_tool = StructuredTool.from_function(
    func=list_plotting_data_files,
    name="list_plotting_data_files",
    description="Lists ALL available files recursively, including plotting resources, dataset files, and ERA5 data. Use this to see exactly what files you can work with.",
    args_schema=ListPlottingDataFilesArgs
)

class WiseAgentToolArgs(BaseModel):
    query: str = Field(description="The query about visualization to send to Claude for advice. Include details about your dataset structure, variables, and visualization goals.")

def wise_agent(query: str) -> str:
    """
    A tool that provides visualization advice using either OpenAI or Anthropic models.
    
    Args:
        query: The query about visualization to send to the AI model
        
    Returns:
        str: AI's advice on visualization
    """
    import streamlit as st
    import logging
    import yaml
    import os
    
    # Load configuration (Climsight uses config.yml by default)
    config_path = os.path.join(os.getcwd(), "config.yml")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            app_config = yaml.safe_load(f)
    else:
        app_config = {}
    
    # Get wise agent configuration
    wise_agent_config = app_config.get("wise_agent", {})
    provider = wise_agent_config.get("provider", "openai")  # Default to OpenAI
    
    # Get dataset information from session state
    datasets_text = st.session_state.get("viz_datasets_text", "")
    
    if not datasets_text:
        datasets_text = "No dataset information available"
    
    # Get the list of available plotting data files
    try:
        available_files = list_plotting_data_files("")
        logging.info("Successfully retrieved available plotting data files")
    except Exception as e:
        logging.error(f"Error retrieving available files: {str(e)}")
        available_files = f"Error retrieving available files: {str(e)}"
    
    # Create the system prompt
    system_prompt = (
        "You are WISE_AGENT, a scientific visualization expert for climate and environmental research data.\n\n"
        "Your goal: provide specific, actionable advice that produces publication-quality figures\n"
        "suitable for peer-reviewed journals.\n\n"
        "When advising:\n"
        "1. ANALYZE THE DATA STRUCTURE first — recommend plot types based on actual dimensions and variables\n"
        "2. Apply SCIENTIFIC DOMAIN conventions:\n"
        "   - Climate: blue=cold, red=warm; diverging palettes (RdBu_r) for anomalies; sequential for absolutes\n"
        "   - Depth/elevation: Y-axis inverted (0 at top) for depth, normal for height\n"
        "   - Precipitation: green/blue sequential; use mm/month or mm/day consistently\n"
        "   - Wind: quiver plots or wind roses for direction; speed in m/s\n"
        "3. Recommend specific matplotlib/seaborn code strategies tailored to the data\n"
        "4. For spatial/geographic data, recommend cartopy projections and coastline overlays\n"
        "5. For time series, recommend appropriate temporal aggregation and trend visualization\n"
        "6. Always prioritize: clarity > density > aesthetics\n\n"
        "Respond with this structure:\n"
        "1. **Recommended plot type** — with brief justification\n"
        "2. **Code strategy** — 3-5 lines of key matplotlib/seaborn calls\n"
        "3. **Color scheme** — specific colormap name and rationale\n"
        "4. **Layout tips** — figsize, subplot arrangement, axis formatting\n"
        "5. **Common pitfalls** — what to avoid for this data type\n"
    )
    
    # Enhance the query with dataset information and available files
    enhanced_query = f"""
DATASET INFORMATION:
{datasets_text}

AVAILABLE PLOTTING DATA FILES:
{available_files}

USER QUERY:
{query}

Please provide visualization advice based on this information.
"""
    
    try:
        if provider.lower() == "anthropic":
            # Use Anthropic's Claude
            try:
                anthropic_api_key = st.secrets["general"]["anthropic_api_key"]
                logging.info("Using Anthropic Claude for wise_agent")
            except KeyError:
                logging.error("Anthropic API key not found in .streamlit/secrets.toml")
                return "Error: Anthropic API key not found in .streamlit/secrets.toml. Please add it to use WISE_AGENT with Claude."
            
            anthropic_model = wise_agent_config.get("anthropic_model", "claude-3-7-sonnet-20250219")
            
            from langchain_anthropic import ChatAnthropic
            llm = ChatAnthropic(
                model=anthropic_model,
                anthropic_api_key=anthropic_api_key,
                #temperature=0.2,
            )
            
            logging.info(f"Making request to Claude model: {anthropic_model}")
            
        else:  # Default to OpenAI
            from langchain_openai import ChatOpenAI

            logging.info("Using OpenAI for wise_agent")

            openai_model = wise_agent_config.get("openai_model", "gpt-5")
            llm = ChatOpenAI(
                api_key=_API_KEY,
                model_name=openai_model,
            )

            logging.info(f"Making request to OpenAI model: {openai_model}")
        
        # Generate the response
        response = llm.invoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": enhanced_query}
            ]
        )
        
        logging.info("Successfully received response from AI model")
        return response.content
        
    except Exception as e:
        logging.error(f"Error using WISE_AGENT: {str(e)}")
        return f"Error using WISE_AGENT: {str(e)}"

# Create the wise agent tool
wise_agent_tool = StructuredTool.from_function(
    func=wise_agent,
    name="wise_agent",
    description="A tool that provides expert visualization advice using advanced AI models. Use this tool FIRST when planning complex visualizations or when you need guidance on best visualization practices for scientific data. Provide a detailed description of the data structure and visualization goals.",
    args_schema=WiseAgentToolArgs
)
