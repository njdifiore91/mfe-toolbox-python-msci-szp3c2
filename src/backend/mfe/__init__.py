"""
Main initialization file for the MFE (MATLAB Financial Econometrics) Toolbox,
implementing the Python package structure and providing the entry point for the toolbox.
This file establishes the package namespace, imports and exposes key functionality from submodules,
sets up version information, and initializes the environment for the package.
"""

import os  # Python 3.12: Access operating system functionality for path operations
import sys  # Python 3.12: System-specific parameters and functions for path management
import logging  # Python 3.12: Flexible event logging system for package initialization
import warnings  # Python 3.12: Issue warning messages for environment compatibility issues
from typing import List, Dict, Optional, Union, Any  # Python 3.12: Support for type hints throughout the package

# Internal imports
from .initialize import initialize, initialize_environment, check_python_version, check_required_packages, check_numba_availability, setup_path_configuration  # Import initialization functions for environment setup
from .core import *  # Import core statistical modules for bootstrap analysis, distributions, optimization, and testing
from .models import *  # Import modeling modules for time series, volatility, and high-frequency analysis
from .utils import validate_params, check_numba_availability  # Import utility functions for validation and Numba compatibility checks

# Define package-level logger
_logger = logging.getLogger(__name__)

# Package metadata
__version__ = "4.0.0"
__author__ = "Kevin Sheppard"
__email__ = "kevin.sheppard@economics.ox.ac.uk"

# Explicitly define the public API of the package
__all__ = [
    # Core statistical functionality
    'block_bootstrap', 'stationary_bootstrap', 'ged_pdf', 'ged_cdf', 'skewt_pdf', 'skewt_cdf', 'minimize', 'root_find', 'gradient_descent', 'cross_sectional_regression', 'principal_component_analysis', 'diagnostic_tests', 'hypothesis_tests',
    # Time series models
    'ARMAModel', 'ARMAXModel', 'ARMAResults', 'ARMAXResults',
    # Volatility models
    'VolatilityModel', 'VolatilityResult', 'VolatilityType', 'GARCH', 'estimate_volatility', 'forecast_volatility', 'simulate_volatility',
    # Realized volatility
    'realized_variance', 'realized_kernel', 'realized_volatility', 'realized_covariance', 'RealizedVolatility',
    # Initialization functions
    'initialize', 'initialize_environment', 'check_python_version', 'check_required_packages', 'check_numba_availability', 'setup_path_configuration',
    # Utility exports
    'validate_params'
]


def init_logging(level: Optional[int] = None) -> logging.Logger:
    """
    Initializes logging for the MFE package

    Parameters
    ----------
    level : Optional[int]
        Logging level to set. If provided, overrides quiet parameter.

    Returns
    -------\n    logging.Logger
        Configured logger for the package
    """
    # Get or create a logger using logging.getLogger(__name__)
    logger = logging.getLogger('mfe')
    # Set logger level if provided (defaults to INFO)
    if level is not None:
        logger.setLevel(level)
    # Configure a stream handler if no handlers exist
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    # Return the configured logger
    return logger


def get_version() -> str:
    """
    Returns the current version of the MFE Toolbox

    Returns
    -------\n    str
        Version string in semantic versioning format
    """
    # Return the __version__ global variable
    return __version__


def auto_initialize() -> bool:
    """
    Performs automatic initialization of the MFE Toolbox on import

    Returns
    -------\n    bool
        True if initialization was successful, False otherwise
    """
    try:
        # Call initialize_environment with default parameters
        is_initialized = initialize_environment()
        # Log initialization status
        if is_initialized:
            _logger.info("MFE Toolbox automatically initialized.")
        else:
            _logger.warning("MFE Toolbox automatic initialization failed.")
        # Return initialization success status
        return is_initialized
    except Exception as e:
        # Issue warning if initialization fails
        warnings.warn(f"MFE Toolbox automatic initialization failed: {e}")
        return False