# ClimSight

ClimSight is an advanced tool that integrates Large Language Models (LLMs) with climate data to provide localized climate insights for decision-making. ClimSight transforms complex climate data into actionable insights for agriculture, urban planning, disaster management, and policy development.

The target audience includes researchers, providers of climate services, policymakers, agricultural planners, urban developers, and other stakeholders who require detailed climate information to support decision-making. ClimSight is designed to democratize access to climate 
data, empowering users with insights relevant to their specific contexts.

![Image](https://github.com/user-attachments/assets/f9f89735-ef08-4c91-bc03-112c8e4c0896)

ClimSight distinguishes itself through several key advancements:
- **Integration of LLMs**: ClimSight leverages state-of-the-art LLMs to interpret complex climate-related queries, synthesizing information from diverse data sources.
- **Multi-Source Data Integration**: Unlike conventional systems that rely solely on structured climate data, ClimSight integrates information from multiple sources.
- **Evidence-Based Approach**: ClimSight ensures contextually accurate answers by retrieving relevant knowledge from scientific reports, IPCC documents, and geographical databases.
- **Modular Architecture**: Specialized components handle distinct tasks, such as data retrieval, contextual understanding, and result synthesis, leading to more accurate outputs.
- **Real-World Applications**: ClimSight is validated through practical examples, such as assessing climate risks for specific agricultural activities and urban planning scenarios.


## Installation Options

You can use ClimSight in three ways:
1. Run a pre-built Docker container (simplest approach)
2. Build and run a Docker container from source
3. Install the Python package (via pip or conda/mamba)

Using ClimSight requires an OpenAI API key unless using the `skipLLMCall` mode for testing. The API key is only needed when running the application, not during installation.

## 1. Running with Docker (Pre-built Container)

The simplest way to get started is with our pre-built Docker container:

```bash
# Make sure your OpenAI API key is set as an environment variable
export OPENAI_API_KEY="your-api-key-here"

# Pull and run the container
docker pull koldunovn/climsight:stable
docker run -p 8501:8501 -e OPENAI_API_KEY=$OPENAI_API_KEY koldunovn/climsight:stable
```

Then open `http://localhost:8501/` in your browser.

## 2. Building and Running from Source with Docker

If you prefer to build from the latest source:

```bash
# Clone the repository
git clone https://github.com/CliDyn/climsight.git
cd climsight

# Download required data
python download_data.py

# Build and run the container
docker build -t climsight .
docker run -p 8501:8501 -e OPENAI_API_KEY=$OPENAI_API_KEY climsight
```

Visit `http://localhost:8501/` in your browser once the container is running.

For testing without OpenAI API calls:
```bash
docker run -p 8501:8501 -e STREAMLIT_ARGS="skipLLMCall" climsight
```

## 3. Python Package Installation

### Option A: Building from source with conda/mamba

```bash
# Clone the repository
git clone https://github.com/CliDyn/climsight.git
cd climsight

# Create and activate the environment
mamba env create -f environment.yml
conda activate climsight

# Download required data
python download_data.py
```

### Option B: Using pip

It's recommended to create a virtual environment to avoid dependency conflicts:
```bash
# Option 1: Install from source
git clone https://github.com/CliDyn/climsight.git
cd climsight

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install ClimSight
pip install -e .
python download_data.py
```

Or if you prefer to set up without cloning the repository:

```bash
# Option 2: Install from PyPI
# Create and activate a virtual environment
python -m venv climsight_env
source climsight_env/bin/activate  # On Windows: climsight_env\Scripts\activate

# Install the package
pip install climsight

# Create a directory for data
mkdir -p climsight
cd climsight

# Download necessary configuration files
wget https://raw.githubusercontent.com/CliDyn/climsight/main/data_sources.yml
wget https://raw.githubusercontent.com/CliDyn/climsight/main/download_data.py
wget https://raw.githubusercontent.com/CliDyn/climsight/main/config.yml

# Download the required data (about 8 GB)
python download_data.py
```

## Configuration

ClimSight will automatically use a `config.yml` file from the current directory. You can modify this file to customize settings:

```yaml
# Key settings you can modify in config.yml:
# - LLM model (gpt-4, ...)
# - Climate data sources
# - RAG database configuration
# - Agent parameters
```
## Running ClimSight

### If installed with conda/mamba from source:

```bash
# Run from the repository root
streamlit run src/climsight/climsight.py
```

### If installed with pip:

```bash
# Make sure you're in the directory with your data and config
climsight
```

You can optionally set your OpenAI API key as an environment variable:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

Otherwise, you can enter your API key directly in the browser interface when prompted.

### Testing without an OpenAI API key:

```bash
# From source:
streamlit run src/climsight/climsight.py skipLLMCall

# Or if installed with pip:
climsight skipLLMCall
```

The application will open in your browser automatically. Just type your climate-related questions and press "Generate" to get insights.

<img width="800" alt="ClimSight Interface" src="https://github.com/koldunovn/climsight/assets/3407313/569a4c38-a601-4014-b10d-bd34c59b91bb">

## Citation

If you use or refer to ClimSight in your work, please cite:

Koldunov, N., Jung, T. Local climate services for all, courtesy of large language models. _Commun Earth Environ_ **5**, 13 (2024). https://doi.org/10.1038/s43247-023-01199-1
