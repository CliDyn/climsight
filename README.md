# Climate Foresight

Prototype of a system that answers questions about climate change impacts on planned human activities.
![screencast](https://github.com/koldunovn/climsight/assets/3407313/bf7cd327-c8a9-4a09-bfb5-778269fcd15c)


## Running with docker

### simplest: running prebuild container

You should have [Docker](https://docs.docker.com/engine/install/) installed. Then execute:

```bash
docker pull koldunovn/climsight:stable
docker run -p 8501:8501 -e OPENAI_API_KEY=$OPENAI_API_KEY climsight
```

Then open `http://localhost:8501/` in your browser.

### Build and run container with the latest code

You should have the following packages installed:

- git
- wget
- docker

As long as you have them, do:

```bash
git clone https://github.com/koldunovn/climsight.git
cd climsight
./download_data.sh
docker build -t climsight .
docker run -p 8501:8501 climsight
```
Then open `http://localhost:8501/` in your browser. If you don't want to add OpenAI key every time, you can expose it through:

```bash
docker run -p 8501:8501 -e OPENAI_API_KEY=$OPENAI_API_KEY climsight
```
where `$OPENAI_API_KEY` not necessarily should be environment variable, you can insert the key directly.

If you do not have an OpenAI key but want to test Climsight without sending requests to OpenAI, you can run Climsight with the `skipLLMCall` argument:
```bash
docker run -p 8501:8501 -e STREAMLIT_ARGS="skipLLMCall" climsight
```

## Installation

The easiest way is to install it through conda or mamba. We recommend mamba, as it's faster. 

[Install mamba](https://mamba.readthedocs.io/en/latest/mamba-installation.html#mamba-install) if you don't have it.

```bash
git clone https://github.com/koldunovn/climsight.git
cd climsight
```

Create environment and install necessary packages:

```bash

mamba env create -f environment.yml
```

Activate the environment:

```bash
conda activate climsight
```
## Climsight package installation 
```bash
pip install climsight
```

For installation, use either pip alone for all packages and dependencies in a pure Python setup, or use mamba for dependencies followed by pip for Climsight in a Conda environment. Mixing package sources can lead to conflicts and is generally not recommended.

## Before you run

You have to download example climate data and NaturalEarth coastlines. To do it simply run:

```bash
./download_data.sh
```
You also need to download the [natural hazard data](https://sedac.ciesin.columbia.edu/data/set/pend-gdis-1960-2018/data-download) (for which you have to create a free account). Please download the **CSV - Disaster Location Centroids [zip file]** and unpack it into the 'data/natural_hazards' folder. Your file should automatically be called 'pend-gdis-1960-2018-disasterlocations.csv'. If not, please change the file name accordingly. 

You would also need an [OpenAI API key](https://platform.openai.com/docs/api-reference) to run the prototype. You can provide it as environment variable:

```bash
export OPENAI_API_KEY="???????"
```

There is a possibility to also provide it in the running app. The cost of each request (status September 2023) is about 6 cents with `gpt-4` and about 0.3 cents with `gpt-3.5-turbo` (you can change it in the beggining of `climsight.py` script).

### Running 

Change to the `climsight` folder:

```bash
cd climsight
streamlit run src/climsight/climsight.py
```

If you install climsight via pip, make sure to run it in the directory where the data folder has been downloaded:
```bash
climsight
```

The browser window should pop up, with the app running. Ask the questions and don't forget to press "Generate".

<img width="800" alt="Screenshot 2023-09-26 at 15 26 51" src="https://github.com/koldunovn/climsight/assets/3407313/569a4c38-a601-4014-b10d-bd34c59b91bb">

If you do not have an OpenAI key but want to test Climsight without sending requests to OpenAI, you can run Climsight with the `skipLLMCall` argument:
```bash
streamlit run src/climsight/climsight.py skipLLMCall
```


## Citation

If you use or refer to ClimSight in your work, please cite the following publication:

Koldunov, N., Jung, T. Local climate services for all, courtesy of large language models. _Commun Earth Environ_ **5**, 13 (2024). https://doi.org/10.1038/s43247-023-01199-1 
