Running ClimSight with Docker
================================

**Running with Docker**
Simplest way to run Climsight is to use Docker.
There are several ways to run it using Docker:

1. **Using a Pre-built Container**

   .. code-block:: bash

       docker pull koldunovn/climsight:stable
       docker run -p 8501:8501 -e OPENAI_API_KEY=$OPENAI_API_KEY climsight

   Access the system by navigating to `http://localhost:8501/` in your web browser.

2. **Building and Running from Source**

   Ensure you have `git`, `wget`, and `docker` installed, then execute:

   .. code-block:: bash

       git clone https://github.com/koldunovn/climsight.git
       cd climsight
       ./download_data.sh
       docker build -t climsight .
       docker run -p 8501:8501 climsight

   If you don't want to add OpenAI key every time, you can expose it through:

   .. code-block:: bash

       docker run -p 8501:8501 -e OPENAI_API_KEY=$OPENAI_API_KEY climsight
      
   where $OPENAI_API_KEY not necessarily should be environment variable, you can insert the key directly.

   If you do not have an OpenAI key but want to test Climsight without sending requests to OpenAI, you can run Climsight with the skipLLMCall argument:

   If you don't want to add OpenAI key every time, you can expose it through:

   .. code-block:: bash

       docker run -p 8501:8501 -e STREAMLIT_ARGS="skipLLMCall" climsight

