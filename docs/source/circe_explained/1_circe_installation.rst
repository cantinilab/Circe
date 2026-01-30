Installation
------------

CIRCE requires Python 3.8+ and can be installed via pip or from source. It is recommended to use a virtual environment or a conda environment to manage dependencies.


The package can be installed using pip:

.. code-block:: python
    
    pip install circe-py

and from github

.. code-block:: python

    pip install "git+https://github.com/cantinilab/circe.git"


Essential dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~
Since CIRCE depends on `lapack <https://www.netlib.org/lapack/>`__ and `blas <https://www.netlib.org/blas>`__ to compute the graphical lasso, you may need to install these libraries first.

It means that you might not be able to use CIRCE in a standard Windows environment without WSL or similar.
For that, you can use a singularity or docker container, such as provided here (`singularity <https://zenodo.org/records/17450969/files/circe.sif?download=1>`__).

If you are working in a conda environment and are encountering any issue with the installation of circe-py, you can try installing first these dependencies via:

.. code-block:: bash

    conda install conda-forge::lapack
    conda install libfortran=3
    conda install -c conda-forge cmake llvmlite=0.46
    conda install gxx_linux-64

If you are not working with conda,
on Ubuntu, you can do this via:

.. code-block:: bash

   sudo apt-get install libblas-dev liblapack-dev

On macOS, you can use Homebrew:

.. code-block:: bash    

   brew install openblas
