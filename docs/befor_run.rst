Before you run
--------------

You have to download example climate data and NaturalEarth coastlines. To do it simply run:

.. code-block:: bash

    python download_data.py

You would also need an `OpenAI API key <https://platform.openai.com/docs/api-reference>`_ to run the prototype. You can provide it as environment variable:

.. code-block:: bash

    export OPENAI_API_KEY="???????"

There is a possibility to also provide it in the running app. The cost of each request (status September 2023) is about 6 cents with `gpt-4` and about 0.3 cents with `gpt-3.5-turbo` (you can change it in the beginning of `climsight.py` script).
