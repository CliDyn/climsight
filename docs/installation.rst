Installation
============

The easiest way is to install it through conda or mamba. We recommend mamba, as it's faster.

`Install mamba <https://mamba.readthedocs.io/en/latest/mamba-installation.html#mamba-install>`_ if you don't have it.

.. code-block:: bash

    git clone https://github.com/koldunovn/climsight.git
    cd climsight

Create environment and install necessary packages:

.. code-block:: bash

    mamba env create -f environment.yml

Activate the environment:

.. code-block:: bash

    conda activate climsight

Climsight package installation

.. code-block:: bash

    pip install climsight

For installation, use either pip alone for all packages and dependencies in a pure Python setup, or use mamba for dependencies followed by pip for Climsight in a Conda environment. Mixing package sources can lead to conflicts and is generally not recommended.