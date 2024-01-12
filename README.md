# Climate Forsight

Prototype of a system that answers questions about climate change impacts on planned human activities.
![screencast](https://github.com/koldunovn/climsight/assets/3407313/1d755b66-fa04-460f-a3ff-8a28569e7b52)

## Running with docker

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
Moreover, you need to download the current [population data](https://population.un.org/wpp/Download/Standard/CSV/) from the UN. You need the **'Demographic Indicators' 1950-2100, medium CSV file**. Please download it into a folder named 'population_data' within this folder and extract the file. Don't change the file's name.

You also need to download the [natural hazard data](https://public.emdat.be/data) (for which you have to create a free account). Please select **Natural** for classification and leave all the other options the way they are. Download it into a folder named 'natural_hazard_data' and **rename the file to 'public_emdat.xlsx'**.

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

