###############
Utility Modules
###############

Overview
--------

The utils module provides support functions and utilities for the MFE Toolbox. It includes asynchronous operation helpers, data handling functions, Numba optimization utilities, NumPy and Pandas integration helpers, printing utilities, Statsmodels integration, and comprehensive validation functions.

Module Structure
---------------

The utils module is organized into several submodules each providing specific functionality to support the core econometric analysis capabilities of the toolbox.

Asynchronous Operation Helpers
-----------------------------

The async_helpers module provides utilities for managing asynchronous operations using Python's async/await pattern. It includes task management, concurrency control, progress reporting, and error handling for long-running computations.

.. automodule:: mfe.utils.async_helpers
   :members:
   :undoc-members: False
   :show-inheritance:
   :member-order: bysource

Data Handling
------------

The data_handling module provides functions for manipulating, transforming, and preprocessing financial time series data. It includes utilities for converting between different formats, handling missing values, detecting outliers, calculating returns, and managing high-frequency data.

.. automodule:: mfe.utils.data_handling
   :members:
   :undoc-members: False
   :show-inheritance:
   :member-order: bysource

Numba Optimization
----------------

The numba_helpers module provides decorators and utilities for applying Numba JIT compilation to performance-critical functions. It includes options for fallback to Python implementations, parallel execution, and vectorized operations with graceful degradation when Numba is unavailable.

.. automodule:: mfe.utils.numba_helpers
   :members:
   :undoc-members: False
   :show-inheritance:
   :member-order: bysource

NumPy Helpers
------------

The numpy_helpers module provides utilities for working with NumPy arrays, including array manipulation, validation, and specialized numerical operations optimized for financial time series analysis.

.. automodule:: mfe.utils.numpy_helpers
   :members:
   :undoc-members: False
   :show-inheritance:
   :member-order: bysource

Pandas Helpers
-------------

The pandas_helpers module provides utilities for working with Pandas DataFrames and Series, with a focus on time series operations, index manipulation, and data alignment for financial time series.

.. automodule:: mfe.utils.pandas_helpers
   :members:
   :undoc-members: False
   :show-inheritance:
   :member-order: bysource

Printing Utilities
----------------

The printing module provides functions for formatted output of econometric results, including model parameter tables, statistical test results, and diagnostic information with support for various output formats.

.. automodule:: mfe.utils.printing
   :members:
   :undoc-members: False
   :show-inheritance:
   :member-order: bysource

Statsmodels Integration
---------------------

The statsmodels_helpers module provides utilities for integrating with the Statsmodels package, enhancing its functionality for financial econometrics and providing conversion utilities between MFE Toolbox and Statsmodels objects.

.. automodule:: mfe.utils.statsmodels_helpers
   :members:
   :undoc-members: False
   :show-inheritance:
   :member-order: bysource

Validation
---------

The validation module provides comprehensive input validation functions for ensuring data integrity and proper parameter constraints across all econometric models and numerical operations with strict type safety.

.. automodule:: mfe.utils.validation
   :members:
   :undoc-members: False
   :show-inheritance:
   :member-order: bysource

Asynchronous Support
------------------

Many utilities in the module support asynchronous operations using Python's async/await pattern, enabling non-blocking execution for long-running computations and responsive user interfaces.

Numba Integration
---------------

Performance-critical utility functions are optimized using Numba's JIT compilation capabilities, providing near-native execution speed for computationally intensive operations while maintaining Python's flexibility.