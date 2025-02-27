.. _multivariate_models:

Multivariate Volatility Models
=================================
Working with BEKK, CCC, and DCC Models in the MFE Toolbox

Introduction
------------

This tutorial introduces multivariate volatility models available in the MFE Toolbox.
Multivariate volatility models are used to analyze and forecast the volatility of multiple assets simultaneously,
capturing the dynamic relationships between them. The MFE Toolbox provides implementations of three popular multivariate GARCH models:

- BEKK (Baba-Engle-Kraft-Kroner)
- CCC (Constant Conditional Correlation)
- DCC (Dynamic Conditional Correlation)

This tutorial provides comprehensive examples of model estimation, forecasting, and visualization.

BEKK Model
----------

The BEKK model is a multivariate GARCH model that ensures positive definite covariance matrices.
It is defined as:

.. math::
   H_t = C'C + A'r_{t-1}r'_{t-1}A + B'H_{t-1}B

where :math:`H_t` is the conditional covariance matrix, :math:`C` is a lower triangular matrix,
:math:`A` and :math:`B` are parameter matrices, and :math:`r_t` is the vector of asset returns.

To demonstrate the BEKK model, we will use the ``BEKKModel`` class from the ``mfe.models.bekk`` module.

.. code-block:: python

    from mfe.models.bekk import BEKKModel

The ``BEKKModel`` class has the following key methods:

- ``fit(returns)``: Fits the BEKK model to the return data.
- ``forecast(horizon)``: Generates volatility forecasts for the specified horizon.
- ``simulate(n_obs)``: Simulates multivariate returns from the fitted BEKK model.

Here's an example of how to use the ``BEKKModel`` class:

.. code-block:: python

    from src.backend.docs.examples.multivariate_example import demonstrate_bekk_model
    import pandas as pd
    import numpy as np

    # Generate synthetic multivariate return data
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=1000, freq='B')
    returns_data = np.random.multivariate_normal([0, 0, 0], [[0.01, 0.005, 0.002], [0.005, 0.01, 0.003], [0.002, 0.003, 0.01]], 1000)
    returns = pd.DataFrame(returns_data, index=dates)

    # Demonstrate the BEKK model
    demonstrate_bekk_model(returns)

CCC Model
---------

The CCC model is a simpler multivariate GARCH model that assumes constant conditional correlations between assets.
It models individual asset volatilities using univariate GARCH processes and combines them with a constant correlation matrix.

To demonstrate the CCC model, we will use the ``CCCModel`` class from the ``mfe.models.ccc`` module.

.. code-block:: python

    from mfe.models.ccc import CCCModel

The ``CCCModel`` class has the following key methods:

- ``fit(returns)``: Fits the CCC model to the return data.
- ``forecast(horizon)``: Generates volatility forecasts for the specified horizon.
- ``simulate(n_obs)``: Simulates multivariate returns from the fitted CCC model.

Here's an example of how to use the ``CCCModel`` class:

.. code-block:: python

    from src.backend.docs.examples.multivariate_example import demonstrate_ccc_model
    import pandas as pd
    import numpy as np

    # Generate synthetic multivariate return data
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=1000, freq='B')
    returns_data = np.random.multivariate_normal([0, 0, 0], [[0.01, 0.005, 0.002], [0.005, 0.01, 0.003], [0.002, 0.003, 0.01]], 1000)
    returns = pd.DataFrame(returns_data, index=dates)

    # Demonstrate the CCC model
    demonstrate_ccc_model(returns)

DCC Model
---------

The DCC model is a more flexible multivariate GARCH model that allows for dynamic conditional correlations between assets.
It models individual asset volatilities using univariate GARCH processes and then models the time-varying correlations using a GARCH-like process.

To demonstrate the DCC model, we will use the ``DCCModel`` class from the ``mfe.models.dcc`` module.

.. code-block:: python

    from mfe.models.dcc import DCCModel

The ``DCCModel`` class has the following key methods:

- ``fit(returns)``: Fits the DCC model to the return data.
- ``forecast(horizon)``: Generates volatility forecasts for the specified horizon.
- ``simulate(n_obs)``: Simulates multivariate returns from the fitted DCC model.

Here's an example of how to use the ``DCCModel`` class:

.. code-block:: python

    from src.backend.docs.examples.multivariate_example import demonstrate_dcc_model
    import pandas as pd
    import numpy as np

    # Generate synthetic multivariate return data
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=1000, freq='B')
    returns_data = np.random.multivariate_normal([0, 0, 0], [[0.01, 0.005, 0.002], [0.005, 0.01, 0.003], [0.002, 0.003, 0.01]], 1000)
    returns = pd.DataFrame(returns_data, index=dates)

    # Demonstrate the DCC model
    demonstrate_dcc_model(returns)

Asynchronous Operations
-----------------------

The MFE Toolbox supports asynchronous operations for long-running computations, such as model estimation.
This allows you to perform other tasks while the model is being estimated in the background.

Here's an example of how to use the asynchronous estimation feature:

.. code-block:: python

    from src.backend.docs.examples.multivariate_example import demonstrate_async_estimation
    import pandas as pd
    import numpy as np
    import asyncio

    # Generate synthetic multivariate return data
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=1000, freq='B')
    returns_data = np.random.multivariate_normal([0, 0, 0], [[0.01, 0.005, 0.002], [0.005, 0.01, 0.003], [0.002, 0.003, 0.01]], 1000)
    returns = pd.DataFrame(returns_data, index=dates)

    # Demonstrate asynchronous estimation
    asyncio.run(demonstrate_async_estimation(returns))

Advanced Topics
---------------

- Model comparison: Use information criteria (AIC, BIC) to compare different models.
- Diagnostic testing: Perform residual analysis to check model assumptions.
- Parameter interpretation: Understand the meaning of the estimated parameters.

.. toctree::
   :maxdepth: 1
   :caption: API References

   ../api/models