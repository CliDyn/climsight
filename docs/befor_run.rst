Before you run
--------------

You have to download example climate data and NaturalEarth coastlines. To do it simply run:

.. code-block:: bash

    ./download_data.sh

You also need to download the `natural hazard data <https://sedac.ciesin.columbia.edu/data/set/pend-gdis-1960-2018/data-download>`_ (for which you have to create a free account). Please download the **CSV - Disaster Location Centroids [zip file]** and unpack it into the 'data/natural_hazards' folder. Your file should automatically be called 'pend-gdis-1960-2018-disasterlocations.csv'. If not, please change the file name accordingly.

You would also need an `OpenAI API key <https://platform.openai.com/docs/api-reference>`_ to run the prototype. You can provide it as environment variable:

.. code-block:: bash

    export OPENAI_API_KEY="???????"

There is a possibility to also provide it in the running app. The cost of each request (status September 2023) is about 6 cents with `gpt-4` and about 0.3 cents with `gpt-3.5-turbo` (you can change it in the beginning of `climsight.py` script).
