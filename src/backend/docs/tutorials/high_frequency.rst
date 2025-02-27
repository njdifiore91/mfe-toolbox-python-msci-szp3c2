=============================
High-Frequency Data Analysis
=============================

Introduction
===========

Overview
--------
High-frequency financial data analysis focuses on examining intraday price movements 
at fine time scales (seconds or milliseconds) to estimate volatility and understand 
market microstructure effects. The MFE Toolbox provides comprehensive tools for 
analyzing such data, implementing modern methodologies for realized volatility 
estimation that address the unique challenges of high-frequency data.

This tutorial covers the fundamentals of high-frequency analysis, realized volatility 
estimation methods, sampling schemes, noise filtering techniques, and asynchronous 
computation patterns for efficient processing.

Prerequisites
------------
To follow this tutorial, you'll need:

* Python 3.12 or higher
* The MFE Toolbox installed with its dependencies:

  * NumPy (1.26.3+)
  * pandas (2.1.4+)
  * matplotlib (3.8.2+) for visualization
  * Numba (0.59.0+) for performance optimization

Basic knowledge of time series analysis and financial markets is also helpful.

Basic Concepts
=============

Market Microstructure
--------------------
Market microstructure refers to the study of the process and outcomes of exchanging 
assets under specific trading rules. In high-frequency data, microstructure effects 
cause price observations to contain both information about the fundamental asset value 
and noise components. These noise components can significantly distort volatility estimates 
if not properly handled.

Common market microstructure effects include:

* Bid-ask bounce: Trades alternating between bid and ask prices
* Discreteness: Prices occurring at discrete ticks
* Asynchronous trading: Irregular timing of trades

The MFE Toolbox provides methods to address these issues through noise filtering and 
robust estimation techniques.

Realized Volatility
------------------
Realized volatility is a non-parametric measure of return variation based on 
intraday price observations. Unlike parametric models like GARCH that estimate volatility 
indirectly, realized volatility measures provide direct observations of volatility 
from high-frequency data.

In theory, as sampling frequency increases, realized volatility converges to the 
integrated volatility (the true, unobservable volatility process). However, market 
microstructure effects prevent us from using the highest possible frequency in practice.

The basic realized variance estimator is:

.. math::

   RV_t = \sum_{i=1}^{n} r_{t,i}^2

where :math:`r_{t,i}` represents intraday returns.

Sampling Schemes
---------------
Sampling schemes determine how high-frequency data is selected for realized volatility 
calculations. The MFE Toolbox supports several sampling approaches:

1. **Calendar Time Sampling**: Samples data at fixed time intervals (e.g., every 5 minutes)
2. **Business Time Sampling**: Samples based on market activity, using a fixed or variable 
   number of transactions
3. **Calendar Uniform Sampling**: Creates evenly spaced samples across the trading day
4. **Business Uniform Sampling**: Creates evenly spaced samples across the number of observations
5. **Fixed Sampling**: Uses a fixed number of evenly spaced observations

The choice of sampling scheme affects estimates, with different schemes suited to 
different analysis goals.

Basic Usage
==========

Data Preparation
---------------
Before computing realized measures, it's important to prepare high-frequency data 
properly. The MFE Toolbox provides utilities for preprocessing:

.. code-block:: python

   import numpy as np
   import pandas as pd
   from mfe.models.realized import preprocess_price_data

   # Example high-frequency data
   prices = np.array([100.0, 100.5, 101.2, 100.8, 100.9, 101.3, 101.1, 101.6])
   times = np.array([9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0])  # hours

   # Preprocess data (e.g., detect outliers)
   clean_prices, clean_times = preprocess_price_data(
       prices, times, time_type='hours', 
       detect_outliers=True, threshold=3.0
   )

The preprocessing step removes outliers that could distort volatility estimates and 
ensures data quality for subsequent analysis.

Computing Realized Variance
--------------------------
The basic approach to compute realized variance uses the `realized_variance` function:

.. code-block:: python

   from mfe.models.realized import realized_variance

   # Compute realized variance with calendar time sampling
   rv, rv_ss = realized_variance(
       prices=clean_prices,
       times=clean_times,
       time_type='hours',
       sampling_type='CalendarTime',
       sampling_interval=1.0  # 1-hour intervals
   )

   print(f"Realized Variance: {rv:.6f}")
   print(f"Subsampled Realized Variance: {rv_ss:.6f}")

The function returns both the standard realized variance and a subsampled version, 
which helps assess the robustness of the estimate.

Computing Realized Volatility
----------------------------
Realized volatility (the square root of realized variance) can be computed directly:

.. code-block:: python

   from mfe.models.realized import realized_volatility

   # Compute realized volatility with business time sampling
   vol, vol_ss = realized_volatility(
       prices=clean_prices,
       times=clean_times,
       time_type='hours',
       sampling_type='BusinessTime',
       sampling_interval=(1, 2),  # Sample every 1-2 observations
       annualize=True,  # Annualize the volatility
       scale=252        # Annualization factor (252 trading days)
   )

   print(f"Annualized Realized Volatility: {vol:.4f}")

The `annualize` parameter converts the volatility to an annual scale, making it 
comparable with other volatility measures like implied volatility from options.

Advanced Techniques
==================

Kernel-Based Estimation
----------------------
Kernel-based estimators provide more robust realized volatility estimates by accounting for 
autocorrelation in returns and addressing market microstructure noise. The MFE Toolbox 
implements these through the `realized_kernel` function:

.. code-block:: python

   from mfe.models.realized import realized_kernel

   # Compute realized kernel with Bartlett kernel
   rk = realized_kernel(
       prices=clean_prices,
       times=clean_times,
       time_type='hours',
       kernel_type='bartlett',  # Options: 'bartlett', 'parzen', 'tukey-hanning', 'qs', 'truncated'
       bandwidth=None           # Automatically determine the bandwidth
   )

   # Convert to volatility
   vol_kernel = np.sqrt(rk)
   print(f"Realized Kernel Volatility: {vol_kernel:.6f}")

Different kernel types provide varying degrees of noise robustness:

* **Bartlett**: A simple triangular kernel, good general-purpose choice
* **Parzen**: Smoother than Bartlett, with better finite-sample properties
* **Tukey-Hanning**: Provides good balance between bias and variance
* **Quadratic Spectral (QS)**: Optimal for many noise processes
* **Truncated**: Basic flat kernel with a cutoff

Noise Filtering
--------------
Market microstructure noise can significantly distort volatility estimates. The MFE Toolbox 
provides explicit noise filtering through the `noise_adjust` parameter in realized 
volatility functions:

.. code-block:: python

   # Generate noisy high-frequency data
   np.random.seed(42)
   prices = 100 + np.cumsum(np.random.normal(0, 0.01, 1000))
   times = np.linspace(0, 86400, 1000)  # seconds in a day

   # Compute realized volatility with noise filtering
   vol_filtered, _ = realized_volatility(
       prices=prices,
       times=times,
       time_type='seconds',
       sampling_type='CalendarTime',
       sampling_interval=300,  # 5-minute sampling
       noise_adjust=True,      # Enable noise filtering
       annualize=True
   )

   # Compare with non-filtered version
   vol_raw, _ = realized_volatility(
       prices=prices,
       times=times,
       time_type='seconds',
       sampling_type='CalendarTime',
       sampling_interval=300,
       noise_adjust=False,
       annualize=True
   )

   print(f"Raw Volatility: {vol_raw:.4f}")
   print(f"Filtered Volatility: {vol_filtered:.4f}")
   print(f"Difference: {((vol_filtered - vol_raw) / vol_raw * 100):.2f}%")

The difference between filtered and unfiltered estimates can be substantial, especially 
for very high-frequency data.

Sampling Effects
--------------
The choice of sampling frequency has a significant impact on realized volatility estimates. 
A "signature plot" helps visualize this effect:

.. code-block:: python

   # Compute realized volatility at different sampling frequencies
   frequencies = [60, 300, 600, 1800, 3600]  # seconds
   results = {}

   for freq in frequencies:
       vol, _ = realized_volatility(
           prices=prices,
           times=times,
           time_type='seconds',
           sampling_type='CalendarTime',
           sampling_interval=freq,
           annualize=True
       )
       results[freq] = vol

   # Create a signature plot
   plt.figure(figsize=(10, 6))
   plt.plot(frequencies, list(results.values()), 'o-')
   plt.title('Signature Plot: Volatility vs. Sampling Frequency')
   plt.xlabel('Sampling Interval (seconds)')
   plt.ylabel('Annualized Volatility')
   plt.grid(True)
   plt.show()

Typically, volatility estimates increase at very high frequencies (small sampling intervals) 
due to market microstructure noise, then stabilize at moderate frequencies, before potentially 
rising again at very low frequencies due to estimation error.

The 5-minute (300-second) sampling frequency is commonly used in the literature as a 
compromise between information loss and noise impact.

Covariance Estimation
-------------------
For multiple assets, you can compute realized covariance:

.. code-block:: python

   from mfe.models.realized import realized_covariance

   # Example with two price series
   prices_1 = 100 + np.cumsum(np.random.normal(0, 0.01, 1000))
   prices_2 = 100 + np.cumsum(np.random.normal(0, 0.015, 1000) + 0.002)  # With correlation
   times = np.linspace(0, 86400, 1000)  # seconds in a day

   # Compute realized covariance
   rcov = realized_covariance(
       prices_1=prices_1,
       prices_2=prices_2,
       times=times,
       time_type='seconds',
       sampling_type='CalendarTime',
       sampling_interval=300  # 5-minute sampling
   )

   print(f"Realized Covariance: {rcov:.6f}")

   # Compute individual volatilities for correlation
   vol_1, _ = realized_volatility(prices_1, times, 'seconds', 'CalendarTime', 300)
   vol_2, _ = realized_volatility(prices_2, times, 'seconds', 'CalendarTime', 300)

   # Calculate realized correlation
   rcorr = rcov / (vol_1 * vol_2)
   print(f"Realized Correlation: {rcorr:.4f}")

Using the RealizedVolatility Class
=================================

Class Initialization
------------------
The `RealizedVolatility` class provides a comprehensive interface for realized volatility analysis:

.. code-block:: python

   from mfe.models.realized import RealizedVolatility

   # Initialize the class
   rv_analyzer = RealizedVolatility()

Data and Parameter Setting
------------------------
After initialization, set data and parameters:

.. code-block:: python

   # Set data
   rv_analyzer.set_data(
       prices=prices,
       times=times,
       time_type='seconds'
   )

   # Set analysis parameters
   rv_analyzer.set_params({
       'sampling_type': 'CalendarTime',
       'sampling_interval': 300,  # 5-minute sampling
       'noise_adjust': True,
       'kernel_type': 'bartlett',
       'detect_outliers': True,
       'outlier_threshold': 3.0,
       'annualize': True,
       'scale': 252  # Annualization factor
   })

Computing Multiple Measures
-------------------------
The class allows computing multiple realized measures at once:

.. code-block:: python

   # Compute multiple measures
   results = rv_analyzer.compute(
       measures=['variance', 'volatility', 'kernel']
   )

   # Access results
   print(f"Realized Variance: {results['variance']:.6f}")
   print(f"Realized Volatility: {results['volatility']:.6f}")
   print(f"Realized Kernel: {results['kernel']:.6f}")

You can easily adjust parameters and recompute:

.. code-block:: python

   # Change parameters
   rv_analyzer.set_params({
       'sampling_interval': 600,  # 10-minute sampling
       'noise_adjust': False      # Disable noise filtering
   })

   # Recompute with new parameters
   new_results = rv_analyzer.compute(['volatility'])
   print(f"New Realized Volatility: {new_results['volatility']:.6f}")

Retrieving Results
----------------
All computed results are stored and can be retrieved:

.. code-block:: python

   # Get all computed results
   all_results = rv_analyzer.get_results()
   print(all_results)

   # Clear all data and results
   rv_analyzer.clear()

Asynchronous Computation
======================

Async Fundamentals
----------------
The MFE Toolbox supports asynchronous computation using Python's async/await pattern, 
which allows non-blocking execution of long-running tasks:

.. code-block:: python

   import asyncio
   from mfe.models.realized import RealizedVolatility

   # Define an async function
   async def compute_measures(prices, times):
       # Initialize analyzer
       rv_analyzer = RealizedVolatility()
       rv_analyzer.set_data(prices, times, 'seconds')
       
       # Set parameters
       rv_analyzer.set_params({
           'sampling_type': 'CalendarTime',
           'sampling_interval': 300,
           'noise_adjust': True
       })
       
       # Compute measures asynchronously
       results = await rv_analyzer.compute_async(['variance', 'volatility', 'kernel'])
       return results

Async Realized Variance
---------------------
For individual functions, async versions are available:

.. code-block:: python

   from mfe.models.realized import async_realized_variance

   async def compute_variance():
       # Compute realized variance asynchronously
       rv, rv_ss = await async_realized_variance(
           prices=prices,
           times=times,
           time_type='seconds',
           sampling_type='CalendarTime',
           sampling_interval=300,
           noise_adjust=True
       )
       return rv, rv_ss

Async RealizedVolatility Class
----------------------------
The `RealizedVolatility` class provides the `compute_async` method for concurrent analysis:

.. code-block:: python

   async def analyze_multiple_assets(price_list, time_list):
       tasks = []
       
       # Create tasks for each asset
       for i, (prices, times) in enumerate(zip(price_list, time_list)):
           rv_analyzer = RealizedVolatility()
           rv_analyzer.set_data(prices, times, 'seconds')
           rv_analyzer.set_params({'sampling_type': 'CalendarTime', 'sampling_interval': 300})
           
           # Add task to list
           task = rv_analyzer.compute_async(['variance', 'volatility'])
           tasks.append(task)
       
       # Run all tasks concurrently
       results = await asyncio.gather(*tasks)
       return results

   # Run the async function
   asset_results = asyncio.run(analyze_multiple_assets(price_list, time_list))

Performance Benefits
------------------
Asynchronous computation can significantly improve performance when:

1. Processing multiple assets simultaneously
2. Computing various measures with different parameters
3. Handling very large datasets
4. Integrating with other async processes (e.g., data retrieval)

Here's a comparison of synchronous vs. asynchronous execution:

.. code-block:: python

   import time
   import asyncio
   from mfe.models.realized import RealizedVolatility

   # Generate multiple price series
   n_assets = 5
   price_list = [100 + np.cumsum(np.random.normal(0, 0.01, 1000)) for _ in range(n_assets)]
   time_list = [np.linspace(0, 86400, 1000) for _ in range(n_assets)]

   # Synchronous execution
   start_time = time.time()
   sync_results = []
   
   for prices, times in zip(price_list, time_list):
       rv_analyzer = RealizedVolatility()
       rv_analyzer.set_data(prices, times, 'seconds')
       rv_analyzer.set_params({'sampling_interval': 300})
       result = rv_analyzer.compute(['variance', 'volatility'])
       sync_results.append(result)
       
   sync_time = time.time() - start_time
   print(f"Synchronous execution time: {sync_time:.4f} seconds")

   # Asynchronous execution
   async def process_all():
       tasks = []
       for prices, times in zip(price_list, time_list):
           rv_analyzer = RealizedVolatility()
           rv_analyzer.set_data(prices, times, 'seconds')
           rv_analyzer.set_params({'sampling_interval': 300})
           task = rv_analyzer.compute_async(['variance', 'volatility'])
           tasks.append(task)
       return await asyncio.gather(*tasks)
   
   start_time = time.time()
   async_results = asyncio.run(process_all())
   async_time = time.time() - start_time
   
   print(f"Asynchronous execution time: {async_time:.4f} seconds")
   print(f"Speedup: {sync_time/async_time:.2f}x")

Performance Optimization
======================

Numba Integration
---------------
The MFE Toolbox leverages Numba for performance optimization. Numba accelerates 
Python code by JIT compiling it to optimized machine code:

.. code-block:: python

   # Example of a Numba-optimized function in the MFE Toolbox
   from numba import jit

   @jit(nopython=True)
   def fast_computation(data):
       result = 0.0
       for i in range(len(data)):
           result += data[i] * data[i]
       return result

Key realized volatility functions are optimized with Numba to deliver near-C performance 
while maintaining Python's ease of use.

JIT Compilation
-------------
Just-In-Time (JIT) compilation is used throughout the MFE Toolbox for performance-critical 
operations in realized volatility calculation. For example, the noise filtering function 
is optimized with Numba's `@jit` decorator:

.. code-block:: python

   @jit(nopython=True)
   def noise_filter(prices, returns, filter_type, filter_params):
       # Optimized implementation...
       return filtered_returns

When these functions are called the first time, Numba compiles them to machine code, 
which is then reused for subsequent calls. This provides significant performance benefits 
for computationally intensive operations like realized volatility estimation.

Optimization Strategies
---------------------
The MFE Toolbox employs several optimization strategies:

1. **Vectorization**: Using NumPy's vectorized operations when possible
2. **JIT compilation**: Accelerating loop-heavy code with Numba
3. **Asynchronous computation**: Executing independent tasks concurrently
4. **Efficient memory usage**: Minimizing memory allocations in critical paths

To maximize performance:

* Prefer the `RealizedVolatility` class for analyzing multiple measures
* Use asynchronous methods for processing multiple assets
* Apply appropriate sampling frequencies to balance precision and performance
* Enable noise filtering only when necessary
* Use the most efficient sampling scheme for your data characteristics

Complete Examples
===============

Basic Analysis
------------
This example demonstrates a complete basic realized volatility analysis:

.. code-block:: python

   import numpy as np
   import pandas as pd
   import matplotlib.pyplot as plt
   from mfe.models.realized import realized_volatility, realized_variance

   # Generate or load high-frequency data
   np.random.seed(42)
   prices = 100 + np.cumsum(np.random.normal(0, 0.01, 1000))
   times = np.linspace(0, 86400, 1000)  # seconds in a day

   # Compute realized volatility
   vol, vol_ss = realized_volatility(
       prices=prices,
       times=times,
       time_type='seconds',
       sampling_type='CalendarTime',
       sampling_interval=300,  # 5-minute sampling
       annualize=True
   )

   # Print results
   print(f"Annualized Realized Volatility: {vol:.4f}")
   
   # Create a simple plot of returns
   returns = np.diff(np.log(prices)) * 100  # percentage returns
   
   plt.figure(figsize=(12, 6))
   plt.plot(returns)
   plt.title(f'Intraday Returns (Realized Vol: {vol:.4f}%)')
   plt.ylabel('Returns (%)')
   plt.grid(True)
   plt.show()

Advanced Analysis
---------------
This example demonstrates a more comprehensive analysis exploring different techniques:

.. code-block:: python

   import numpy as np
   import pandas as pd
   import matplotlib.pyplot as plt
   from mfe.models.realized import RealizedVolatility, preprocess_price_data

   # Generate or load high-frequency data
   np.random.seed(42)
   prices = 100 + np.cumsum(np.random.normal(0, 0.01, 1000))
   times = np.linspace(0, 86400, 1000)  # seconds in a day

   # Preprocess data
   clean_prices, clean_times = preprocess_price_data(
       prices, times, 'seconds', detect_outliers=True
   )

   # Initialize analyzer
   rv_analyzer = RealizedVolatility()
   rv_analyzer.set_data(clean_prices, clean_times, 'seconds')

   # Compare different sampling intervals
   intervals = [60, 300, 600, 1800]
   results = {}

   for interval in intervals:
       rv_analyzer.set_params({
           'sampling_type': 'CalendarTime',
           'sampling_interval': interval,
           'noise_adjust': True,
           'annualize': True
       })
       
       result = rv_analyzer.compute(['volatility'])
       results[f'{interval}s'] = result['volatility']

   # Compare different kernel types
   kernels = ['bartlett', 'parzen', 'qs', 'tukey-hanning']
   rv_analyzer.set_params({'sampling_interval': 300})  # Reset to 5-minute sampling

   for kernel in kernels:
       rv_analyzer.set_params({'kernel_type': kernel})
       result = rv_analyzer.compute(['kernel'])
       results[f'kernel_{kernel}'] = np.sqrt(result['kernel'])

   # Visualize results
   plt.figure(figsize=(12, 6))
   plt.bar(results.keys(), results.values())
   plt.title('Comparison of Realized Volatility Measures')
   plt.ylabel('Annualized Volatility')
   plt.xticks(rotation=45)
   plt.grid(axis='y')
   plt.tight_layout()
   plt.show()

Asynchronous Workflow
-------------------
This example demonstrates a complete asynchronous workflow for analyzing multiple assets:

.. code-block:: python

   import numpy as np
   import asyncio
   import matplotlib.pyplot as plt
   from mfe.models.realized import RealizedVolatility

   async def analyze_asset(prices, times, asset_name):
       """Analyze a single asset asynchronously"""
       rv_analyzer = RealizedVolatility()
       rv_analyzer.set_data(prices, times, 'seconds')
       
       # Standard analysis
       rv_analyzer.set_params({
           'sampling_type': 'CalendarTime',
           'sampling_interval': 300,
           'noise_adjust': True,
           'annualize': True
       })
       
       # Compute multiple measures
       results = await rv_analyzer.compute_async(['variance', 'volatility', 'kernel'])
       
       # Add asset name to results
       results['asset'] = asset_name
       return results

   async def analyze_portfolio():
       """Analyze a portfolio of assets concurrently"""
       # Generate data for multiple assets
       assets = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'FB']
       price_data = {}
       time_data = {}
       
       for asset in assets:
           # Generate synthetic data with different volatility for each asset
           vol = np.random.uniform(0.008, 0.02)
           price_data[asset] = 100 + np.cumsum(np.random.normal(0, vol, 1000))
           time_data[asset] = np.linspace(0, 86400, 1000)
       
       # Create and run tasks concurrently
       tasks = [
           analyze_asset(price_data[asset], time_data[asset], asset)
           for asset in assets
       ]
       
       # Wait for all tasks to complete
       results = await asyncio.gather(*tasks)
       return results

   # Run the async analysis
   portfolio_results = asyncio.run(analyze_portfolio())

   # Process and display results
   assets = [r['asset'] for r in portfolio_results]
   volatilities = [r['volatility'] for r in portfolio_results]

   plt.figure(figsize=(10, 6))
   plt.bar(assets, volatilities)
   plt.title('Portfolio Realized Volatility Analysis')
   plt.ylabel('Annualized Volatility')
   plt.grid(axis='y')
   plt.tight_layout()
   plt.show()

API Reference
===========

Core Functions
------------

realized_variance
^^^^^^^^^^^^^^^^

.. code-block:: python

   def realized_variance(
       prices: np.ndarray,
       times: np.ndarray or pd.Series,
       time_type: str,
       sampling_type: str,
       sampling_interval=DEFAULT_SAMPLING_INTERVAL,
       noise_adjust: bool = False
   ) -> tuple[float, float]

Computes the realized variance of a high-frequency price series with optional noise filtering.

**Parameters:**

* **prices** (np.ndarray): High-frequency price series
* **times** (np.ndarray or pd.Series): Timestamps corresponding to each price observation
* **time_type** (str): Type of time data: 'datetime', 'seconds', or 'businesstime'
* **sampling_type** (str): Method for sampling: 'CalendarTime', 'BusinessTime', 'CalendarUniform', 'BusinessUniform', or 'Fixed'
* **sampling_interval** (int or tuple, default=300): Sampling interval specification, interpretation depends on sampling_type
* **noise_adjust** (bool, default=False): If True, applies noise filtering to the returns

**Returns:**

* **tuple[float, float]**: Realized variance and subsampled realized variance

realized_volatility
^^^^^^^^^^^^^^^^^

.. code-block:: python

   def realized_volatility(
       prices: np.ndarray,
       times: np.ndarray or pd.Series,
       time_type: str,
       sampling_type: str,
       sampling_interval=DEFAULT_SAMPLING_INTERVAL,
       noise_adjust: bool = False,
       annualize: bool = False,
       scale: float = 252
   ) -> tuple[float, float]

Computes the realized volatility (square root of realized variance) of a price series.

**Parameters:**

* **prices** (np.ndarray): High-frequency price series
* **times** (np.ndarray or pd.Series): Timestamps corresponding to each price observation
* **time_type** (str): Type of time data: 'datetime', 'seconds', or 'businesstime'
* **sampling_type** (str): Method for sampling: 'CalendarTime', 'BusinessTime', 'CalendarUniform', 'BusinessUniform', or 'Fixed'
* **sampling_interval** (int or tuple, default=300): Sampling interval specification, interpretation depends on sampling_type
* **noise_adjust** (bool, default=False): If True, applies noise filtering to the returns
* **annualize** (bool, default=False): If True, annualizes the volatility
* **scale** (float, default=252): Annualization factor (252 for daily data -> annual)

**Returns:**

* **tuple[float, float]**: Realized volatility and its subsampled version

Advanced Functions
----------------

realized_kernel
^^^^^^^^^^^^^

.. code-block:: python

   def realized_kernel(
       prices: np.ndarray,
       times: np.ndarray or pd.Series,
       time_type: str,
       kernel_type: str = DEFAULT_KERNEL_TYPE,
       bandwidth: float = None
   ) -> float

Implements kernel-based estimation of realized volatility using various kernel functions.

**Parameters:**

* **prices** (np.ndarray): High-frequency price series
* **times** (np.ndarray or pd.Series): Timestamps corresponding to each price observation
* **time_type** (str): Type of time data: 'datetime', 'seconds', or 'businesstime'
* **kernel_type** (str, default='bartlett'): Type of kernel function: 'bartlett', 'parzen', 'tukey-hanning', 'qs', 'truncated'
* **bandwidth** (float, default=None): Bandwidth parameter for the kernel. If None, determined automatically.

**Returns:**

* **float**: Realized kernel estimate of volatility

sampling_scheme
^^^^^^^^^^^^^

.. code-block:: python

   def sampling_scheme(
       prices: np.ndarray,
       times: np.ndarray or pd.Series,
       time_type: str,
       sampling_type: str,
       sampling_interval: int or tuple
   ) -> tuple[np.ndarray, np.ndarray]

Implements different sampling schemes for intraday data including calendar time and business time sampling.

**Parameters:**

* **prices** (np.ndarray): High-frequency price series
* **times** (np.ndarray or pd.Series): Timestamps corresponding to each price observation
* **time_type** (str): Type of time data: 'datetime', 'seconds', or 'businesstime'
* **sampling_type** (str): Method for sampling: 'CalendarTime', 'BusinessTime', 'CalendarUniform', 'BusinessUniform', or 'Fixed'
* **sampling_interval** (int or tuple): Sampling interval specification, interpretation depends on sampling_type

**Returns:**

* **tuple[np.ndarray, np.ndarray]**: Sampled prices and times

noise_filter
^^^^^^^^^^

.. code-block:: python

   @jit(nopython=True)
   def noise_filter(
       prices: np.ndarray,
       returns: np.ndarray,
       filter_type: str,
       filter_params: dict = None
   ) -> np.ndarray

Filters microstructure noise from high-frequency price data.

**Parameters:**

* **prices** (np.ndarray): Price series corresponding to the returns
* **returns** (np.ndarray): Returns series to be filtered
* **filter_type** (str): Type of filter to apply: 'MA', 'Kernel', 'HodrickPrescott', 'WaveletThresholding'
* **filter_params** (dict, default=None): Parameters specific to the chosen filter

**Returns:**

* **np.ndarray**: Noise-filtered returns

RealizedVolatility Class
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   class RealizedVolatility:
       def __init__(self)
       def set_data(self, prices, times, time_type)
       def set_params(self, params)
       def compute(self, measures)
       async def compute_async(self, measures)
       def get_results(self)
       def clear()

A comprehensive class for computing and analyzing realized volatility measures from high-frequency financial data.

**Methods:**

* **__init__()**: Initializes the RealizedVolatility class with default parameters
* **set_data(prices, times, time_type)**: Sets the price and time data for analysis
* **set_params(params)**: Sets parameters for realized volatility computation
* **compute(measures)**: Computes specified realized measures
* **compute_async(measures)**: Asynchronously computes specified realized measures
* **get_results()**: Returns the computed realized measures
* **clear()**: Clears all data and results

Conclusion
=========

Summary
------
This tutorial covered comprehensive techniques for high-frequency financial data analysis 
using the MFE Toolbox, including:

1. Basic concepts of realized volatility and market microstructure
2. Various sampling schemes and their impact on estimates
3. Advanced techniques like kernel-based estimation and noise filtering
4. Using the RealizedVolatility class for efficient analysis
5. Asynchronous computation for improved performance
6. Numba-based optimization for computational efficiency

The MFE Toolbox provides a powerful Python-based framework for researchers and practitioners 
to analyze high-frequency financial data with robust, efficient, and well-documented tools.

Further Reading
-------------
For more advanced topics in high-frequency analysis, consider exploring:

1. Jump detection in high-frequency data
2. Multi-asset portfolio realized covariance estimation
3. Forecasting realized volatility
4. Comparing realized volatility with implied volatility
5. High-frequency trading strategies

Related documentation:

* :doc:`../garch_models` - GARCH volatility modeling
* :doc:`../arma_models` - Time series modeling
* :doc:`../multivariate_models` - Multivariate analysis
* :doc:`../../api/models.rst#realized-volatility` - Detailed API documentation