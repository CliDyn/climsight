# Climate Forsight

Prototype of a system that answers questions about climate change impacts on planned human activities.
![screencast](https://github.com/koldunovn/climsight/assets/3407313/1d755b66-fa04-460f-a3ff-8a28569e7b52)

## Running with docker

### simplest: running prebuild container

You have to ahve [Docker](https://docs.docker.com/engine/install/) installed. Then execute:

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
where `$OPENAI_API_KEY` not necesarelly should be environment variable, you can insert the key directly.

## Installation

The easiest way is to install it through conda or mamba. We recomend mamba, as it's faster. 

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

The browser window should pop up, with the app running. Ask the questions and dont' forget to press "Generate".

<img width="1025" alt="Screenshot 2023-09-26 at 15 26 51" src="https://github.com/koldunovn/climsight/assets/3407313/41ed9802-8b63-473b-ba13-8c4f3639ee97">

