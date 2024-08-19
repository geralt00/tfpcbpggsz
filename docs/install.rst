Install
=======

Prerequisites
``````````````````


- TensorFlow: Ensure TensorFlow is installed on your system.
- Cudatoolkit >= 11.5: Required for running the tool.

Known Issues
``````````````````


- There may be compatibility issues with certain versions of TensorFlow. See the `issue report on GitHub <https://github.com/tensorflow/tensorflow/issues/63362#issuecomment-2016019354>`_ for more details.

Setup
``````````````````


Clone the repository and set up the required environment using Conda:

.. code-block:: bash

    git clone https://github.com/shenghui/tf-pcgamma.git
    cd tf-pcgamma
    conda env create -f requirements.yml

Install the package
```````````````````````````



.. code-block:: bash

    pip install -e . --no-deps