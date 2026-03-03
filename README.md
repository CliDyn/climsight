# ClimSight

ClimSight is an advanced tool that integrates Large Language Models (LLMs) with climate data to provide localized climate insights for decision-making. ClimSight transforms complex climate data into actionable insights for agriculture, urban planning, disaster management, and policy development.

The target audience includes researchers, providers of climate services, policymakers, agricultural planners, urban developers, and other stakeholders who require detailed climate information to support decision-making. ClimSight is designed to democratize access to climate data, empowering users with insights relevant to their specific contexts.

![Image](https://github.com/user-attachments/assets/f9f89735-ef08-4c91-bc03-112c8e4c0896)

ClimSight distinguishes itself through several key advancements:
- **Integration of LLMs**: ClimSight leverages state-of-the-art LLMs to interpret complex climate-related queries, synthesizing information from diverse data sources.
- **Multi-Source Data Integration**: Unlike conventional systems that rely solely on structured climate data, ClimSight integrates information from multiple sources.
- **Evidence-Based Approach**: ClimSight ensures contextually accurate answers by retrieving relevant knowledge from scientific reports, IPCC documents, and geographical databases.
- **Modular Architecture**: Specialized components handle distinct tasks, such as data retrieval, contextual understanding, and result synthesis, leading to more accurate outputs.
- **Real-World Applications**: ClimSight is validated through practical examples, such as assessing climate risks for specific agricultural activities and urban planning scenarios.


## Installation

### Recommended: Building from source with conda/mamba

This is the recommended installation method to get the latest features and updates.

```bash
# Clone the repository
git clone https://github.com/CliDyn/climsight.git
cd climsight

# Create and activate the environment
mamba env create -f environment.yml
conda activate climsight

# Download required data
python download_data.py

# Optional: download DestinE data (large ~12 GB, not downloaded by default)
python download_data.py DestinE
```

### Alternative: Using pip from source

```bash
# Clone the repository
git clone https://github.com/CliDyn/climsight.git
cd climsight

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download required data
python download_data.py

# Optional: download DestinE data (large ~12 GB, not downloaded by default)
python download_data.py DestinE
```

### Running with Docker (Stable Release v1.0.0)

The Docker container provides a stable release (v1.0.0) of ClimSight. For the latest features, please install from source as described above.

```bash
# Make sure your OpenAI API key is set as an environment variable
export OPENAI_API_KEY="your-api-key-here"

# Pull and run the container
docker pull koldunovn/climsight:stable
docker run -p 8501:8501 -e OPENAI_API_KEY=$OPENAI_API_KEY koldunovn/climsight:stable
```

Then open `http://localhost:8501/` in your browser.

### Using pip from PyPI (Stable Release v1.0.0)

The PyPI package provides a stable release (v1.0.0) of ClimSight. For the latest features, please install from source as described above.

```bash
pip install climsight
```

## Configuration

ClimSight will automatically use a `config.yml` file from the current directory. You can modify this file to customize settings:

```yaml
# Key settings you can modify in config.yml:
# - LLM model (gpt-4, gpt-5, ...)
# - Climate data sources
# - RAG database configuration
# - Agent parameters
# - ERA5 data retrieval settings
```

## API Keys

### OpenAI API Key

ClimSight requires an OpenAI API key for LLM functionality. You can set it as an environment variable:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

Alternatively, you can enter your API key directly in the browser interface when prompted.

### Arraylake API Key (Optional - for ERA5 Data)

If you want to use ERA5 time series data retrieval (enabled via the "Enable ERA5 data" toggle in the UI), you need an Arraylake API key from [Earthmover](https://earthmover.io/). This allows downloading ERA5 reanalysis data for detailed historical climate analysis.

```bash
export ARRAYLAKE_API_KEY="your-arraylake-api-key-here"
```

You can also enter the Arraylake API key in the browser interface when the ERA5 data option is enabled.

### DestinE Data Retrieval (Optional - for Destination Earth Climate DT)

To download DestinE Climate Adaptation Digital Twin data (enabled via the "Enable DestinE data" toggle in the UI), you need a [Destination Earth](https://destination-earth.eu/) (DESP) account. Authentication is handled via a token stored in `~/.polytopeapirc`.

To set up authentication, run the provided script:

```bash
python desp-authentication.py -u YOUR_DESP_USERNAME -p YOUR_DESP_PASSWORD
```

This writes a token to `~/.polytopeapirc` which is then used automatically by the DestinE retrieval tool. For more details on authentication and the polytope API, see the [polytope-examples](https://github.com/destination-earth-digital-twins/polytope-examples) repository.

## Running ClimSight

```bash
# Run from the repository root
streamlit run src/climsight/climsight.py
```

The application will open in your browser automatically. Just type your climate-related questions and press "Generate" to get insights.

<img width="800" alt="ClimSight Interface" src="https://github.com/koldunovn/climsight/assets/3407313/569a4c38-a601-4014-b10d-bd34c59b91bb">

## Batch Processing

For batch processing of climate questions, the `sequential` directory contains specialized tools for generating, validating, and processing questions in bulk. These tools are particularly useful for research and analysis requiring multiple climate queries. See the [sequential/README.md](sequential/README.md) for detailed usage instructions.

## Citation

If you use or refer to ClimSight in your work, please cite:

Kuznetsov, I., Jost, A.A., Pantiukhin, D. et al. Transforming climate services with LLMs and multi-source data integration. _npj Clim. Action_ **4**, 97 (2025). https://doi.org/10.1038/s44168-025-00300-y

Koldunov, N., Jung, T. Local climate services for all, courtesy of large language models. _Commun Earth Environ_ **5**, 13 (2024). https://doi.org/10.1038/s43247-023-01199-1
