.. _models_api:

Models API Reference
====================

.. toctree::
   :maxdepth: 3
   :caption: Model Categories:
   :hidden:

   time_series
   volatility
   multivariate
   high_frequency

Introduction
------------

This page documents the time series and volatility models provided in the MFE Toolbox.
These models are implemented using Python 3.12 with modern programming features including async/await patterns and strict type hints.
Performance-critical calculations are optimized using Numba's just-in-time compilation through @jit decorators.

Time Series Models
------------------

The time series models module provides comprehensive ARMA and ARMAX modeling capabilities, supporting robust parameter estimation, forecasting, and diagnostic analysis.

.. automodule:: mfe.models.arma
   :members:
   :undoc-members:
   :show-inheritance:

ARMA Models
^^^^^^^^^^^

.. include:: ../examples/arma_example.py
   :code:

ARMAX Models
^^^^^^^^^^^

The ARMAX models extend the ARMA framework to include exogenous variables, enabling more complex time series analysis.

.. automodule:: mfe.models.armax
   :members:
   :undoc-members:
   :show-inheritance:

Volatility Models
-----------------

The volatility models module includes various GARCH-type models for modeling conditional heteroskedasticity in financial time series.
These range from standard GARCH to specialized variants that capture asymmetry, long memory, and threshold effects.

Base Volatility Framework
^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: mfe.models.volatility
   :members:
   :undoc-members:
   :show-inheritance:

GARCH Models
^^^^^^^^^^^^

.. automodule:: mfe.models.garch
   :members:
   :undoc-members:
   :show-inheritance:

.. include:: ../examples/garch_example.py
   :code:

Asymmetric GARCH Models
^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: mfe.models.agarch
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: mfe.models.egarch
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: mfe.models.figarch
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: mfe.models.igarch
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: mfe.models.tarch
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: mfe.models.aparch
   :members:
   :undoc-members:
   :show-inheritance:

Multivariate Volatility Models
------------------------------

The multivariate volatility models enable joint modeling of multiple time series, capturing dynamic covariance structures and conditional correlations between assets.

.. automodule:: mfe.models.multivariate
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: mfe.models.bekk
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: mfe.models.ccc
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: mfe.models.dcc
   :members:
   :undoc-members:
   :show-inheritance:

.. include:: ../examples/multivariate_example.py
   :code:

High-Frequency Models
---------------------

The high-frequency analytics and realized volatility modules provide tools for analyzing intraday data, estimating volatility from high-frequency observations, and handling microstructure noise.

Realized Volatility Measures
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: mfe.models.realized
   :members:
   :undoc-members:
   :show-inheritance:

.. include:: ../examples/realized_volatility_example.py
   :code:

.. automodule:: mfe.models.high_frequency
   :members:
   :undoc-members:
   :show-inheritance: