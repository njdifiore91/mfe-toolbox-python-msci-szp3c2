.. _garch_models:

Introduction to GARCH Models
=============================

This tutorial provides a comprehensive guide to using GARCH (Generalized Autoregressive Conditional Heteroskedasticity) models within the MFE Toolbox. GARCH models are essential for modeling volatility in financial time series, capturing the time-varying nature of variance.

Theoretical Background
-----------------------

GARCH models are a class of statistical models used to estimate volatility in time series data. They assume that the variance of the current error term is a function of the variances of the past error terms, making them suitable for financial data where volatility clustering is observed.

The basic GARCH(p, q) model is defined as:

.. math::
   \sigma_t^2 = \omega + \alpha_1 \epsilon_{t-1}^2 + \dots + \alpha_q \epsilon_{t-q}^2 + \beta_1 \sigma_{t-1}^2 + \dots + \beta_p \sigma_{t-p}^2

Where:

-  :math:`\sigma_t^2` is the conditional variance at time t
-  :math:`\omega` is a constant term
-  :math:`\alpha_i` are the coefficients for the lagged squared error terms
-  :math:`\epsilon_{t-i}` are the lagged error terms
-  :math:`\beta_j` are the coefficients for the lagged conditional variance terms

Python Implementation
---------------------

The MFE Toolbox provides a robust and efficient implementation of GARCH models in Python, leveraging Numba for optimized performance. The models are designed to seamlessly integrate with other components of the toolbox, offering a consistent and user-friendly API.

Basic GARCH Model
=================

This section details the implementation and usage of the standard GARCH model within the MFE Toolbox.

Model Definition
----------------

The GARCH(p, q) model is defined by two parameters: p, the order of the GARCH terms, and q, the order of the ARCH terms. The model estimates the conditional variance based on past squared errors and past variances.

Model Estimation
----------------

To estimate the parameters of a GARCH model, you can use the ``fit`` or ``fit_async`` methods. The ``fit`` method performs synchronous estimation, while ``fit_async`` allows for non-blocking estimation using Python's ``async/await`` patterns.

Code Example
------------

.. code-block:: python
   :title: Basic GARCH Model Example

   import numpy as np
   from mfe.models.garch import GARCH

   # Create a GARCH(1,1) model with normal errors
   model = GARCH(p=1, q=1, distribution='normal')

   # Example returns data
   returns = np.random.randn(1000) * 0.01

   # Fit the model
   result = model.fit(returns)

   # Print summary
   print(model.summary())

   # Forecast volatility for the next 10 periods
   forecasts = model.forecast(horizon=10)
   print(f"Volatility forecast: {np.sqrt(forecasts)}")

.. code-block:: python
   :title: Async GARCH Fitting

   import asyncio
   import numpy as np
   from mfe.models.garch import GARCH

   async def fit_garch_async():
       # Create a GARCH(1,1) model
       model = GARCH(p=1, q=1)
       
       # Example returns data
       returns = np.random.randn(1000) * 0.01
       
       # Fit the model asynchronously
       result = await model.fit_async(returns)
       print("Estimation completed")
       print(model.summary())
       
   # Run the async function
   await fit_garch_async()

GARCH Model Variants
====================

The MFE Toolbox includes several specialized GARCH model variants to capture different aspects of volatility dynamics.

EGARCH Model
------------

The EGARCH (Exponential GARCH) model captures asymmetric responses to positive and negative shocks.

.. code-block:: python
   :title: EGARCH News Impact Curve

   import numpy as np
   import matplotlib.pyplot as plt
   from mfe.models.egarch import EGARCH

   # Create and fit an EGARCH(1,1,1) model
   model = EGARCH(p=1, o=1, q=1)
   returns = np.random.randn(1000) * 0.01
   model.fit(returns)

   # Calculate and plot the news impact curve
   z_range = np.linspace(-5, 5, 100)
   impact = model.get_news_impact_curve(z_range)

   plt.figure(figsize=(10, 6))
   plt.plot(z_range, impact)
   plt.title('EGARCH News Impact Curve')
   plt.xlabel('Standardized Shock (z)')
   plt.ylabel('Impact on Log-Variance')
   plt.axvline(x=0, color='red', linestyle='--')
   plt.grid(True)
   plt.show()

AGARCH Model
------------

The AGARCH (Asymmetric GARCH) model also captures leverage effects, where negative returns have a different impact on volatility than positive returns.

IGARCH Model
------------

The IGARCH (Integrated GARCH) model imposes a unit root in the variance equation, implying that shocks to volatility are persistent.

FIGARCH Model
-------------

The FIGARCH (Fractionally Integrated GARCH) model captures long-memory effects in volatility, where past shocks have a long-lasting impact on current volatility.

Other Variants
--------------

The MFE Toolbox also supports other specialized GARCH models, including the TARCH (Threshold ARCH) model and the PGARCH (Power GARCH) model.

Working with GARCH Models
=========================

This section provides practical guidance on using GARCH models for financial analysis.

Forecasting Volatility
----------------------

GARCH models can be used to forecast future volatility by iterating the conditional variance equation forward. The ``forecast`` method provides a convenient way to generate volatility forecasts for a specified horizon.

Monte Carlo Simulation
----------------------

GARCH models can be used to simulate future returns by generating random shocks from a specified distribution and iterating the conditional variance equation forward. The ``simulate`` method provides a way to generate simulated return paths.

.. code-block:: python
   :title: Monte Carlo Simulation with GARCH

   import numpy as np
   import matplotlib.pyplot as plt
   from mfe.models.garch import GARCH

   # Create and fit a GARCH(1,1) model
   model = GARCH(p=1, q=1)
   returns = np.random.randn(1000) * 0.01
   model.fit(returns)

   # Simulate 500 future periods
   sim_returns, sim_volatility = model.simulate(n_periods=500)

   # Plot the simulated returns and volatility
   plt.figure(figsize=(12, 8))

   plt.subplot(2, 1, 1)
   plt.plot(sim_returns)
   plt.title('Simulated Returns')
   plt.grid(True)

   plt.subplot(2, 1, 2)
   plt.plot(np.sqrt(sim_volatility))
   plt.title('Simulated Volatility')
   plt.grid(True)

   plt.tight_layout()
   plt.show()

Model Diagnostics
-----------------

Evaluating the fit of a GARCH model is crucial for ensuring its reliability. Diagnostic tools include residual analysis, such as examining the autocorrelation and distribution of standardized residuals.

Asynchronous Operations
-----------------------

The MFE Toolbox leverages Python's ``async/await`` patterns to enable non-blocking GARCH estimation. This is particularly useful for long-running computations, allowing you to perform other tasks while the model is being estimated.

Multivariate GARCH Models
=========================

This section introduces multivariate GARCH models, which extend the GARCH framework to handle multiple assets and their covariances.

.. code-block:: python
   :title: Multivariate GARCH Example

   import numpy as np
   from mfe.models.multivariate import create_multivariate_model

   # Generate example multivariate returns (2 assets)
   n_assets = 2
   returns = np.random.randn(500, n_assets) * 0.01

   # Create a DCC-GARCH model
   model = create_multivariate_model('DCC', n_assets=n_assets, params={
       'garch_orders': {'p': 1, 'q': 1}
   })

   # Fit the model
   result = model.fit(returns)

   # Forecast covariance matrices
   forecasts = model.forecast(horizon=10)

   # Print the forecasted correlation for the next period
   corr_forecast = forecasts.get_forecast(1)['correlation']
   print(f"Forecasted correlation matrix:\n{corr_forecast}")

BEKK Model
----------

The BEKK (Baba-Engle-Kraft-Kroner) model is a multivariate GARCH specification that ensures positive definite covariance matrices.

CCC Model
---------

The CCC (Constant Conditional Correlation) model assumes that the conditional correlations between assets are constant over time.

DCC Model
---------

The DCC (Dynamic Conditional Correlation) model allows the conditional correlations between assets to vary over time.

Practical Applications
----------------------

Multivariate GARCH models are used in various financial applications, including portfolio optimization, risk management, and asset allocation.

Performance Optimization
========================

This section details the performance optimizations applied to GARCH models within the MFE Toolbox.

Numba JIT Compilation
---------------------

Numba is used to accelerate the core computations of GARCH models, providing near-C performance. The ``@jit`` decorator is applied to performance-critical functions, enabling just-in-time compilation.

Optimized Recursion
-------------------

The GARCH variance recursion is implemented using Numba-optimized code, ensuring efficient calculation of conditional variances.

Likelihood Optimization
-----------------------

Parameter estimation is performed using optimized likelihood functions, leveraging SciPy's numerical optimization routines.

Advanced Topics
===============

This section covers advanced usage patterns and customizations for GARCH models.

Custom Distributions
--------------------

The MFE Toolbox supports various error distributions for GARCH models, including the Normal distribution, Student's t-distribution, and Generalized Error Distribution (GED).

Risk Metrics
------------

GARCH models can be used to calculate risk metrics such as Value-at-Risk (VaR) and Expected Shortfall (ES).

Integration with Other Models
-----------------------------

GARCH models can be combined with other time series models, such as ARIMA, to provide a comprehensive framework for financial time series analysis.

References
==========

-   Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. *Journal of Econometrics*, *31*(3), 307-327.
-   Engle, R. F. (1982). Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation. *Econometrica*, *50*(4), 987-1007.
-   Hansen, B. E. (1994). Autoregressive conditional density estimation. *International Economic Review*, *35*(3), 705-730.