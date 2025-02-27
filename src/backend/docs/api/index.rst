.. _mfe_api_reference:

MFE Toolbox API Reference
=========================

Introduction
------------

The MFE Toolbox is a comprehensive Python library for financial econometrics, implemented in Python 3.12 with modern programming constructs including async/await patterns and strict type hints. The library leverages NumPy, SciPy, Pandas, Statsmodels, and Numba for high-performance statistical computing.

Module Organization
-------------------

The library is organized into three main namespaces: core for statistical modules, models for time series and volatility implementations, and utils for support functions and utilities.

### Core Modules

The core namespace (mfe.core) contains fundamental statistical and computational components including bootstrap analysis, statistical distributions, optimization routines, cross-sectional tools, and testing modules.

### Models Modules

The models namespace (mfe.models) houses time series and volatility modeling implementations including ARMA/ARMAX, GARCH variants, multivariate models (BEKK, CCC, DCC), and high-frequency analytics.

### Utility Modules

The utils namespace (mfe.utils) provides support functions including asynchronous helpers, data handling utilities, Numba optimization tools, and validation functions.

Python and Numba Integration
-----------------------------

Performance-critical functions are optimized using Numba's just-in-time compilation through @jit decorators, providing near-native execution speed while maintaining the flexibility of Python's scientific computing ecosystem.

Asynchronous Operations
-----------------------

The library extensively uses Python's async/await patterns for non-blocking operations, making it well-suited for interactive environments and responsive user interfaces.

.. toctree::
   :maxdepth: 2
   :caption: API Modules:

   utils
   core
   models