"""
MFE Toolbox - Core Statistical Module

Initialization file for the MFE Toolbox's core statistical module,
exposing functionality for bootstrap analysis, distribution computing,
optimization, cross-sectional analysis, and statistical testing.
This module provides the foundation for statistical operations
throughout the toolbox.
"""

import logging  # Python 3.12: Support for flexible event logging in the core module
from typing import List, Dict, Tuple, Optional, Union, Any, Callable  # Python 3.12: Type hint support for enhanced code safety and documentation

# Internal imports
from .bootstrap import *  # Import bootstrap analysis functionality for time series data
from .distributions import *  # Import statistical distribution functions and models
from .optimization import *  # Import optimization algorithms for model estimation
from .cross_section import *  # Import cross-sectional analysis tools
from .testing import *  # Import statistical testing functionality

# Set up module logger
logger = logging.getLogger(__name__)

# Core module version information
__version__ = "1.0.0"

# List of all items to be exported from the module
__all__ = [
    # Bootstrap exports
    'block_bootstrap', 'stationary_bootstrap', 'moving_block_bootstrap',
    'block_bootstrap_async', 'stationary_bootstrap_async', 'moving_block_bootstrap_async',
    'BootstrapResult', 'Bootstrap', 'calculate_bootstrap_ci',

    # Distribution exports
    'ged_pdf', 'ged_cdf', 'ged_ppf', 'ged_random',
    'skewt_pdf', 'skewt_cdf', 'skewt_ppf', 'skewt_random',
    'jarque_bera', 'lilliefors', 'shapiro_wilk', 'ks_test',
    'GeneralizedErrorDistribution', 'SkewedTDistribution', 'DistributionTest',
    'distribution_fit', 'distribution_forecast',

    # Optimization exports
    'minimize', 'root_find', 'gradient_descent', 'newton_raphson',
    'bfgs', 'nelder_mead', 'numerical_gradient', 'numerical_hessian',
    'async_minimize', 'async_root_find', 'Optimizer', 'OptimizationParameters',
    'OptimizationResult',

    # Cross-section exports
    'cross_sectional_regression', 'principal_component_analysis', 'cross_sectional_stats',
    'async_regression', 'async_pca', 'compute_cross_correlation',
    'compute_heteroscedasticity_test', 'CrossSectionalRegression', 'PrincipalComponentAnalysis',

    # Testing exports
    'diagnostic_tests', 'hypothesis_tests', 'unit_root_test',
    'serial_correlation_test', 'arch_test', 'heteroskedasticity_test',
    'normality_test', 'box_test', 'kpss_test', 'adf_test',
    'TestResult', 'TestBattery', 'DiagnosticTestBattery', 'HypothesisTestBattery',
    'run_async_tests'
]