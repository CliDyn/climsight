Running Climsight
==================

Change to the ``climsight`` folder:

.. code-block:: bash

    cd climsight
    streamlit run src/climsight/climsight.py

If you install climsight via pip, make sure to run it in the directory where the data folder has been downloaded:

.. code-block:: bash

    climsight

The browser window should pop up, with the app running. Ask the questions and don't forget to press "Generate".

.. image:: https://github.com/koldunovn/climsight/assets/3407313/569a4c38-a601-4014-b10d-bd34c59b91bb
   :width: 800
   :alt: Screenshot 2023-09-26 at 15 26 51

If you do not have an OpenAI key but want to test Climsight without sending requests to OpenAI, you can run Climsight with the ``skipLLMCall`` argument:

.. code-block:: bash

    streamlit run src/climsight/climsight.py skipLLMCall
