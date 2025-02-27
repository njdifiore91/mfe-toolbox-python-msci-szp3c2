.. _getting_started:

Getting Started with the MFE Toolbox
=====================================

Introduction
------------

The MFE Toolbox is a comprehensive suite of Python modules designed for modeling financial time series and conducting advanced econometric analyses. The toolbox has been completely re-implemented in Python 3.12, featuring modern programming constructs such as async/await patterns and strict type hints.

Installation
------------

Step-by-step instructions for installing the MFE Toolbox, including prerequisites, installation methods (pip, source), and troubleshooting common installation issues.

Prerequisites
^^^^^^^^^^^^^

The MFE Toolbox requires Python 3.12 or later. The package is built upon the Python scientific stack and requires the following libraries:

* NumPy (1.26.3 or later): For array operations and numerical computations
* SciPy (1.11.4 or later): For optimization and statistical functions
* Pandas (2.1.4 or later): For time series handling
* Statsmodels (0.14.1 or later): For econometric modeling
* Numba (0.59.0 or later): For performance optimization
* PyQt6 (6.6.1 or later): For GUI components (optional, only needed for graphical interface)

.. note::
   The MFE Toolbox requires Python 3.12 or later due to its use of modern language features and typing enhancements. Make sure to use a compatible Python version.

Installing via pip
^^^^^^^^^^^^^^^^^^

The easiest way to install the MFE Toolbox is using pip from PyPI::

    .. code-block:: python

       # Install the MFE Toolbox
       pip install mfe

This will automatically download and install the package along with its dependencies.

Installing from Source
^^^^^^^^^^^^^^^^^^^^^^

For development purposes, you can install the package in editable mode from the source directory::

    git clone https://github.com/username/mfe-toolbox.git
    cd mfe-toolbox
    pip install -e .

This allows you to modify the source code and have the changes immediately reflected without reinstallation.

Package Structure
-----------------

The MFE Toolbox is organized into four main namespaces:

*   ``mfe.core``: Contains fundamental statistical and computational components
*   ``mfe.models``: Houses time series and volatility modeling implementations
*   ``mfe.utils``: Provides utility functions and helper routines
*   ``mfe.ui``: Manages user interface components

Basic Usage
-----------

Introduction to using the MFE Toolbox, including importing modules, initializing the environment, and working with basic functionality.

Importing Modules
^^^^^^^^^^^^^^^^^

To use the MFE Toolbox, import the necessary modules from the appropriate namespaces::

    .. code-block:: python

       # Import main modules
       import mfe
       from mfe.models import arma, garch
       from mfe.core import bootstrap, distributions
       from mfe.utils import data_handling

Environment Initialization
^^^^^^^^^^^^^^^^^^^^^^^^^^

Before using the toolbox, initialize the environment to configure paths and dependencies::

    .. code-block:: python

       # Initialize the environment
       mfe.initialize()

.. warning::
   When importing modules for the first time, make sure to call ``mfe.initialize()`` to properly set up the environment. This ensures all paths and dependencies are correctly configured.

Data Handling
-------------

Overview of working with time series data in the MFE Toolbox, including importing, transforming, and preparing financial data for analysis.

Loading Data
^^^^^^^^^^^^

The ``mfe.utils.data_handling`` module provides utilities for loading financial time series data from various sources.

Data Transformation
^^^^^^^^^^^^^^^^^^^

Common data transformations for financial time series, including returns calculation, standardization, and handling missing values.

Core Features Overview
----------------------

Summary of key features available in the MFE Toolbox, with brief explanations and pointers to more detailed documentation.

Time Series Modeling
^^^^^^^^^^^^^^^^^^^^

The MFE Toolbox provides a comprehensive ARMA/ARMAX modeling and forecasting framework.

    .. code-block:: python
    
        import numpy as np
        from mfe.models.arma import ARMAModel
        
        # Generate sample data
        np.random.seed(42)
        data = np.random.normal(0, 1, 500) + 0.7 * np.roll(np.random.normal(0, 1, 500), 1)
        data[0] = np.random.normal(0, 1)
        
        # Create and fit ARMA model
        model = ARMAModel(p=1, q=1)
        result = model.fit(data)
        
        # Display results
        print(result.summary())
        
        # Generate forecast
        forecast = result.forecast(steps=10)
        print(f"Forecast: {forecast}")

Volatility Modeling
^^^^^^^^^^^^^^^^^^^

Unified framework for univariate and multivariate volatility models supporting GARCH variants (AGARCH, EGARCH, FIGARCH) and multivariate specifications (BEKK, DCC).

Bootstrap Analysis
^^^^^^^^^^^^^^^^^^^^

Robust resampling for dependent time series using block and stationary bootstrap methods.

Statistical Distributions
^^^^^^^^^^^^^^^^^^^^^^^^^

Advanced statistical distribution and testing framework supporting advanced distributions (GED, Hansen's skewed T) and diagnostic tests.

High-Frequency Analysis
^^^^^^^^^^^^^^^^^^^^^^^

Advanced realized volatility estimation and noise filtering system for intraday data analysis.

Example: Time Series Analysis
-----------------------------

A complete walkthrough example of time series analysis using the ARMAModel class, demonstrating the workflow from data preparation to model estimation and forecasting.

Example: Bootstrap Analysis
---------------------------

A comprehensive example of bootstrap analysis for statistical inference, showing various bootstrap methods and their applications.

Asynchronous Operations
-----------------------

The MFE Toolbox leverages Python's ``async/await`` patterns for non-blocking operations, particularly useful for long-running computations and responsive applications.

    .. code-block:: python

        import asyncio
        from mfe.models.arma import ARMAModel
        
        async def fit_and_forecast():
            # Create model
            model = ARMAModel(p=1, q=1)
            
            # Define progress callback
            def progress_callback(progress):
                print(f"Estimation progress: {progress:.2%}")
            
            # Fit model asynchronously
            result = await model.fit_async(data, progress_callback=progress_callback)
            
            # Generate forecast asynchronously
            forecast = await result.forecast_async(steps=10)
            return result, forecast
        
        # Run the async function
        result, forecast = asyncio.run(fit_and_forecast())

Performance Optimization with Numba
-----------------------------------

The MFE Toolbox leverages Numba for performance optimization through JIT compilation, with guidelines for efficient usage and potential performance gains.

Advanced Time Series Modeling
-----------------------------

Overview of advanced time series modeling techniques available in the MFE Toolbox, including multivariate models, seasonal components, and handling non-stationary data.

    .. code-block:: python

        # Advanced ARMA modeling with seasonal components
        from mfe.models.arma import ARMAModel
        import numpy as np
        
        # Create seasonal data with AR(1) pattern
        np.random.seed(42)
        t = np.arange(0, 500)
        seasonal = 0.5 * np.sin(2 * np.pi * t / 12)  # Monthly seasonality
        ar_component = np.zeros(500)
        
        # Create AR(1) component
        phi = 0.7
        for i in range(1, 500):
            ar_component[i] = phi * ar_component[i-1] + np.random.normal(0, 0.1)
        
        # Combine components
        data = ar_component + seasonal
        
        # Create and fit ARMA model with seasonal dummies
        seasonal_dummies = np.zeros((500, 11))
        for i in range(11):
            seasonal_dummies[:, i] = (t % 12 == i+1).astype(float)
        
        # Fit ARMA model with exogenous seasonal variables
        model = ARMAModel(p=1, q=0, include_constant=True)
        result = model.fit(data, exog=seasonal_dummies)
        print(result.summary())
        
        # Forecast future periods
        future_steps = 24
        future_t = np.arange(500, 500 + future_steps)
        future_dummies = np.zeros((future_steps, 11))
        for i in range(11):
            future_dummies[:, i] = (future_t % 12 == i+1).astype(float)
        
        forecasts = model.forecast(data, steps=future_steps, exog_forecast=future_dummies)
        print("Seasonal forecasts:", forecasts)

Next Steps
----------

Suggestions for further learning and exploration of the MFE Toolbox, with links to related API documentation, advanced topics, and reference documentation.

.. seealso::

   * :doc:`installation.rst` - Detailed installation instructions
   * :doc:`../../api/models` - API reference for time series and volatility models
   * :doc:`../../api/index` - Complete API reference documentation