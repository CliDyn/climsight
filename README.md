# Climate Forsight

Prototype of a system that answers questions about climate change impacts on planned human activities.

## Installation

The easiest way is to install it through conda/mamba. We recomend mamba, as it's faster. 

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

## Before you run

You have to download example climate data and NaturalEarth coastlines. Change to the To do it simply run:

```bash
./download_data.sh
```

You would also need an [OpenAI API key](https://platform.openai.com/docs/api-reference) to run the prototype. You can provide it as environment varaible:

```bash
export OPENAI_API_KEY="???????"
```

There is a possibility to also provide it in the running app. The cost of each request (status September 2023) is about 6 cents with `gpt-4` and about 0.3 cents with `gpt-3.5-turbo` (you can change it in the beggining of `climsight.py` script).

### Running 

Change to the `climsight` folder:

```bash
cd climsight
streamlit run climsight.py
```

THe browser window should pop up, with the app running. Ask the questions and dont' forget to press "Generate".

