.. _install:

Install
=======

Prerequisites
-------------

| **Python:** >=3.9, <3.13
| **Operating System:** Linux, Windows, MacOS

All :code:`gnss_lib_py` classes and methods are tested in Python 3.9,
and 3.12 in the latest Ubuntu, MacOS and Windows versions.
:code:`gnss_lib_py` was originally developed in Python 3.8.9 in
Ubuntu 20/22 and Ubuntu 20 for WSL2.

Use on Google Colab
-------------------

|colab|

.. |colab| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/drive/1DYfuiM5ipz0B-lgjKYcL1Si-V4jNBEac?usp=sharing

We provide support to run :code:`gnss_lib_py` on Google Colab. To run on
Google Colab, use the following command to install :code:`gnss_lib_py`:

.. code-block:: bash

   %pip install gnss-lib-py --quiet --progress-bar off

To include :code:`gnss_lib_py` in the Google Colab environment, use the following
command

.. code-block:: python

   import gnss_lib_py as glp

To use :code:`gnss_lib_py` from a specific branch, use the following code block
to clone :code:`gnss_lib_py`, checkout the desired branch, and install:

.. code-block:: bash

   import os
   os.makedirs("/content/lib", exist_ok=True)
   %cd /content/lib
   !pip install --upgrade pip --quiet --progress-bar off
   !git clone https://github.com/Stanford-NavLab/gnss_lib_py.git --quiet
   %cd gnss_lib_py
   !git checkout desired-branch-name-here CHANGE TO DESIRED BRANCH
   !git pull
   !pip install -e . --quiet --progress-bar off

Standard Installation
---------------------

1. :code:`gnss_lib_py` is available through :code:`pip` installation
   with:

   .. code-block:: bash

      pip install gnss-lib-py

Editable Installation
---------------------

1. Clone the GitHub repository:

   .. code-block:: bash

      git clone https://github.com/Stanford-NavLab/gnss_lib_py.git

2. Install dependencies with pip:

   .. code-block:: bash

       pip3 install -r requirements.txt

3. Update pip version.

   a. For Linux and MacOS:

      .. code-block:: bash

         pip install -U pip

   b. For Windows:

      .. code-block:: bash

          python -m pip install -U pip

4. Install :code:`gnss_lib_py` locally from directory containing :code:`pyproject.toml`

   .. code-block:: bash

      pip install -e .

5. Verify installation by running :code:`pytest`.
   A successful installation will be indicated by all tests passing.

   .. code-block:: bash

      pytest

.. _developer install:

Developer Installation
----------------------

This project is being developed using :code:`pyenv` and :code:`poetry`
for python version and environment control respectively.

Linux/WSL2 and MacOS
++++++++++++++++++++

1. Install :code:`pyenv` using the installation instructions
   `here <https://github.com/pyenv/pyenv#installation>`__. The steps are
   briefly summarized below:

   a. Install the `Python build dependencies <https://github.com/pyenv/pyenv/wiki#suggested-build-environment>`__.

   b. Either use the `automatic installer <https://github.com/pyenv/pyenv-installer>`__
      or the `Basic GitHub Checkout <https://github.com/pyenv/pyenv#basic-github-checkout>`__.

   c. In either case, you will need to configure your shell's
      environment variables for :code:`pyenv` as indicated in the install
      instructions. For example, for :code:`bash`, you can add the
      following lines to the end of your :code:`.bashrc`

      .. code-block:: bash

         export PATH="$HOME/.pyenv/bin:$PATH"
         eval "$(pyenv init --path)"
         eval "$(pyenv virtualenv-init -)"

2. Install Python 3.9.0 or above with :code:`pyenv`. For example,
   :code:`pyenv install 3.9.19`.

3. Clone the :code:`gnss_lib_py` repository.

4. Inside the :code:`gnss_lib_py` run :code:`pyenv local 3.9.19` (switching
   out with the version of Python you installed in the previous step
   if different than 3.9.19) to set the Python version that code in the
   repository will run.

5. Install :code:`poetry>=1.2` using the instructions
   `here <https://python-poetry.org/docs/master/#installation>`__.

6. Install Python dependencies using :code:`poetry install`.

.. _install_pandoc:

7. Install pandoc to be able to build documentation. See details
   `here <https://pandoc.org/installing.html>`__.

   a. For Linux :code:`sudo apt install pandoc`

   b. For Windows :code:`choco install pandoc`

   c. For MacOS :code:`brew install pandoc`


8. Verify that the code is working by running tests on the code using

   .. code-block:: bash

      poetry run pytest

   Check the :ref:`Testing<testing>` section in the Contribution guide
   for more details

9. Verify that the documentation is building locally using

   .. code-block:: bash

      ./build_docs.sh

Windows
+++++++

1. Currently, full support is not offered for Windows, but :code:`pyenv`
   can be installed following instructions
   `here <https://pypi.org/project/pyenv-win/>`__.

2. The workflow for installing :code:`poetry` and :code:`gnss_lib_py` is
   similar once :code:`pyenv` has been set up.


Refer to the :ref:`Documentation<documentation>` section once you add
code/documentation and want to build and view the documentation locally.
