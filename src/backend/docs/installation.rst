============
Installation
============

Requirements
===========

Python Environment
-----------------

The MFE Toolbox requires Python 3.12 or later. The package is built upon the Python scientific stack and requires the following libraries:

* NumPy (1.26.3 or later): For array operations and numerical computations
* SciPy (1.11.4 or later): For optimization and statistical functions
* Pandas (2.1.4 or later): For time series handling
* Statsmodels (0.14.1 or later): For econometric modeling
* Numba (0.59.0 or later): For performance optimization
* PyQt6 (6.6.1 or later): For GUI components (optional, only needed for graphical interface)

Platform Support
---------------

The MFE Toolbox is designed to be platform-agnostic and should run on any system that supports Python 3.12 and the required dependencies:

* Windows (x86_64)
* Linux (x86_64)
* macOS (x86_64 and arm64)

Installation
===========

Standard Installation
--------------------

The easiest way to install the MFE Toolbox is using pip from PyPI::

    pip install mfe

This will automatically download and install the package along with its dependencies.

Development Installation
-----------------------

For development purposes, you can install the package in editable mode from the source directory::

    git clone https://github.com/username/mfe-toolbox.git
    cd mfe-toolbox
    pip install -e .

This allows you to modify the source code and have the changes immediately reflected without reinstallation.

Virtual Environment
------------------

It's recommended to install the MFE Toolbox in a virtual environment to avoid conflicts with other packages. You can create a virtual environment using ``venv``::

    python -m venv mfe-env
    
On Windows, activate the environment::

    mfe-env\Scripts\activate

On macOS and Linux::

    source mfe-env/bin/activate

Then proceed with the installation::

    pip install mfe

Verifying Installation
=====================

To verify that the MFE Toolbox has been installed correctly, start Python and try to import the package::

    python
    >>> import mfe
    >>> print(mfe.__version__)

This should display the version number of the installed package without any errors.

Dependencies Installation
========================

If you prefer to install dependencies manually or need specific versions, you can install them separately::

    pip install numpy>=1.26.3
    pip install scipy>=1.11.4
    pip install pandas>=2.1.4
    pip install statsmodels>=0.14.1
    pip install numba>=0.59.0
    pip install PyQt6>=6.6.1  # For GUI components

Installing from Source
=====================

If you wish to install from source, you can download the source distribution:

1. Download the source tarball (mfe-toolbox-x.y.z.tar.gz) from PyPI or GitHub releases
2. Extract the archive
3. Navigate to the extracted directory
4. Run::

    pip install .

Building from Source
===================

If you need to build the package from source, the MFE Toolbox uses modern Python packaging tools:

1. Ensure you have the latest setuptools, wheel, and build::

    pip install --upgrade setuptools wheel build

2. Clone the repository or download the source
3. Navigate to the source directory
4. Build the package::

    python -m build

This will create both source distribution (.tar.gz) and wheel (.whl) files in the dist/ directory.

Troubleshooting
==============

Common Issues
------------

Package Conflicts
~~~~~~~~~~~~~~~~

If you encounter package conflicts during installation, try installing in a fresh virtual environment::

    python -m venv fresh-env
    source fresh-env/bin/activate  # or fresh-env\Scripts\activate on Windows
    pip install mfe

Missing Dependencies
~~~~~~~~~~~~~~~~~~~

If you receive an error about missing dependencies, ensure that you're using Python 3.12 or later and that all required packages are installed::

    pip install numpy scipy pandas statsmodels numba PyQt6

Numba JIT Compilation Issues
~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you encounter issues with Numba JIT compilation:

1. Ensure you have a compatible C compiler installed (gcc, MSVC, or clang)
2. Check that you have the latest version of Numba::

    pip install --upgrade numba

3. If problems persist, try running with Numba debug::

    NUMBA_DEBUG=1 python your_script.py

PyQt6 Installation Problems
~~~~~~~~~~~~~~~~~~~~~~~~~

On some systems, installing PyQt6 might require additional system libraries:

- On Debian/Ubuntu::

    sudo apt-get install python3-pyqt6

- On macOS (with Homebrew)::

    brew install pyqt@6

- On Windows, the pip installation should work directly, but you may need to install Visual C++ Redistributable.

Path Configuration Issues
~~~~~~~~~~~~~~~~~~~~~~~

If you encounter import errors, check that the package is correctly installed in your Python path::

    python -c "import sys; print(sys.path)"

Upgrading
=========

To upgrade to the latest version::

    pip install --upgrade mfe

Uninstallation
=============

To remove the MFE Toolbox::

    pip uninstall mfe

This will remove the package but leave dependencies intact.

Additional Resources
==================

- Official Documentation: [URL to documentation]
- GitHub Repository: [URL to GitHub]
- Issue Tracker: [URL to issues page]
- PyPI Page: [URL to PyPI package page]