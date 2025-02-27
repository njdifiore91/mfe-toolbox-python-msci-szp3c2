"""
MFE Toolbox - Utilities Module

This module provides utility functions and tools for the MFE Toolbox's
financial econometric modeling capabilities. It includes support for data 
handling, validation, NumPy/Pandas operations, asynchronous processing,
Numba optimizations, and statistical operations.

The utilities serve as a foundation for the MFE Toolbox, providing robust,
type-safe implementations that leverage Python 3.12 features including 
async/await patterns and strict type hints for financial time series
modeling and econometric analysis.
"""

import sys
import logging
from typing import List, Dict, Any, Optional, Union, Callable

# Import utility submodules with their exports
from .async_helpers import *
from .data_handling import *
from .numba_helpers import *
from .numpy_helpers import *
from .pandas_helpers import *
from .printing import *
from .statsmodels_helpers import *
from .validation import *

# Version tracking
__version__ = '0.1.0'

# Set up module logger
_logger = logging.getLogger('mfe.utils')

def get_version() -> str:
    """
    Returns the current version of the utils module.
    
    Returns
    -------
    str
        Version string in semantic versioning format
    """
    return __version__

def init_logging(logger: Optional[logging.Logger] = None, 
                level: Optional[int] = None) -> logging.Logger:
    """
    Initializes logging for the utils module.
    
    Parameters
    ----------
    logger : Optional[logging.Logger], default=None
        Logger to use. If None, uses the module logger.
    level : Optional[int], default=None
        Logging level to set.
        
    Returns
    -------
    logging.Logger
        Configured logger for the module
    """
    global _logger
    
    # Use module logger if none provided
    if logger is None:
        logger = _logger
    
    # Set log level if provided
    if level is not None:
        logger.setLevel(level)
    
    # Add stream handler if no handlers exist
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

# Define aliases for functions that have different names in the modules
# For data_handling.py
load_data = load_financial_data
save_data = save_financial_data
prepare_time_series = convert_time_series

# For numba_helpers.py
jit_if_available = optimized_jit
parallel_if_available = parallel_jit
check_numba_availability = check_numba_compatibility

# Explicitly list all exports based on the JSON specification
__all__ = [
    # From async_helpers
    'run_async', 'async_to_sync', 'AsyncTask',
    
    # From data_handling (with aliases)
    'load_data', 'save_data', 'prepare_time_series',
    
    # From numba_helpers (with aliases)
    'jit_if_available', 'parallel_if_available', 'check_numba_availability',
    
    # From numpy_helpers
    'ensure_array', 'lag_matrix', 'window_array',
    
    # From pandas_helpers
    'to_returns', 'lag_dataframe', 'rolling_window',
    
    # From printing
    'format_results', 'print_model_summary', 'table_formatter',
    
    # From statsmodels_helpers
    'acf', 'pacf', 'ljung_box',
    
    # From validation
    'validate_params', 'validate_data', 'check_dimensions',
    
    # Module-level exports
    'get_version'
]