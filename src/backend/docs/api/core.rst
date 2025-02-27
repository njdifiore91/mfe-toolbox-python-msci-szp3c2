.. _core_api_reference:

Core API Reference
==================

.. toctree::
   :hidden:

This section provides detailed API documentation for the core modules of the MFE Toolbox.

Bootstrap Module
----------------

.. automodule:: mfe.core.bootstrap
   :members: block_bootstrap, stationary_bootstrap

   .. code-block:: python
      :caption: Bootstrap Example
      :name: bootstrap-example

      import numpy as np  # numpy 1.26.3
      from mfe.core.bootstrap import Bootstrap  # src/backend/mfe/core/bootstrap.py

      def mean_statistic(data: np.ndarray) -> float:
          return np.mean(data)

      # Generate sample data
      np.random.seed(42)
      data = np.random.randn(100)

      # Initialize Bootstrap object
      bootstrap = Bootstrap(method='block', params={'block_size': 20}, num_bootstrap=500)

      # Run bootstrap analysis
      result = bootstrap.run(data, mean_statistic)

      # Print results
      print(f"Original Mean: {result.original_statistic:.4f}")
      print(f"Bootstrap Mean: {result.summary()['bootstrap_mean']:.4f}")

Distribution Module
-------------------

.. automodule:: mfe.core.distributions
   :members: ged_distribution, skewed_t_distribution, jarque_bera

   .. code-block:: python
      :caption: Custom Distribution Example
      :name: custom-distribution-example

      import numpy as np  # numpy 1.26.3
      from mfe.core.distributions import GeneralizedErrorDistribution  # src/backend/mfe/core/distributions.py

      # Generate sample data
      np.random.seed(42)
      data = np.random.randn(100)

      # Initialize GED distribution
      ged = GeneralizedErrorDistribution(mu=0, sigma=1, nu=2)

      # Calculate log-likelihood
      log_likelihood = ged.loglikelihood(data)
      print(f"Log-likelihood: {log_likelihood:.4f}")

Cross-Section Module
--------------------

.. automodule:: mfe.core.cross_section
   :members: CrossSectionAnalysis, pca

   .. code-block:: python
      :caption: Cross-Section Analysis Example
      :name: cross-section-example

      import numpy as np  # numpy 1.26.3
      from mfe.core.cross_section import CrossSectionAnalysis  # src/backend/mfe/core/cross_section.py

      # Generate sample data
      np.random.seed(42)
      y = np.random.randn(100)
      X = np.random.randn(100, 5)

      # Initialize CrossSectionAnalysis object
      analysis = CrossSectionAnalysis()

      # Fit the model
      results = analysis.fit(y, X)

      # Print results
      print(f"R-squared: {results.r_squared_:.4f}")
      print(f"Coefficients: {results.coef_}")

Statistical Testing Module
--------------------------

.. automodule:: mfe.core.testing
   :members: unit_root_test, normality_test

   .. code-block:: python
      :caption: Statistical Testing Example
      :name: statistical-testing-example

      import numpy as np  # numpy 1.26.3
      from mfe.core.testing import jarque_bera  # src/backend/mfe/core/testing.py

      # Generate sample data
      np.random.seed(42)
      data = np.random.randn(100)

      # Perform Jarque-Bera test
      jb_stat, p_value = jarque_bera(data)

      # Print results
      print(f"Jarque-Bera Statistic: {jb_stat:.4f}")
      print(f"P-value: {p_value:.4f}")

Optimization Module
-------------------

.. automodule:: mfe.core.optimization
   :members: Optimizer

   .. code-block:: python
      :caption: Optimization Example
      :name: optimization-example

      import numpy as np  # numpy 1.26.3
      from mfe.core.optimization import Optimizer  # src/backend/mfe/core/optimization.py

      # Define objective function and gradient
      def objective(x):
          return x[0]**2 + x[1]**2

      def gradient(x):
          return np.array([2*x[0], 2*x[1]])

      # Initialize Optimizer object
      optimizer = Optimizer()

      # Set initial parameters
      initial_params = np.array([1.0, 1.0])

      # Run optimization
      result = optimizer.minimize(objective, initial_params, gradient=gradient)

      # Print results
      print(f"Converged: {result.converged}")
      print(f"Parameters: {result.parameters}")