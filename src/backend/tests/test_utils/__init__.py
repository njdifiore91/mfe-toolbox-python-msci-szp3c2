"""
MFE Toolbox - Utility Tests Package

This module provides common fixtures, testing utilities, and helper functions specific to
testing the utility modules of the MFE Toolbox. It configures test markers and specialized 
test parametrization for async, numba, and data handling components.
"""

import pytest  # pytest 7.4.3
import numpy as np  # numpy 1.26.3
import pandas as pd  # pandas 2.1.4
import logging  # Python 3.12
from typing import Any, Callable, Dict, List, Optional, Union
from functools import wraps
import asyncio  # Python 3.12
import numba  # numba 0.59.0

# Import common test utilities from parent package
from .. import check_test_environment, requires_numba, get_test_data_path

# Configure logger
logger = logging.getLogger('mfe.tests.utils')

# Define exports
__all__ = [
    'parametrize_async_tests',
    'parametrize_numba_tests',
    'validate_test_data',
    'check_utils_imports',
    'create_test_samples'
]

# Apply utils marker to all tests in this package
pytestmark = [pytest.mark.utils]


def parametrize_async_tests(test_cases: list, skip_timeouts: bool = False) -> Callable:
    """
    Helper function for parametrizing asynchronous test cases with common test data.
    
    Parameters
    ----------
    test_cases : list
        List of test case parameters to pass to the test function
    skip_timeouts : bool, default=False
        If True, skips test cases marked as potentially timing out
    
    Returns
    -------
    Callable
        Decorator function that can be applied to test functions
    """
    # Filter out cases that might timeout if requested
    if skip_timeouts:
        filtered_cases = [
            case for case in test_cases 
            if not getattr(case, 'might_timeout', False)
        ]
    else:
        filtered_cases = test_cases
    
    # Create decorator that combines asyncio marker with parametrization
    decorator = pytest.mark.parametrize('test_case', filtered_cases)
    
    # Return the combined decorator
    return lambda func: pytest.mark.asyncio(decorator(func))


def parametrize_numba_tests(functions: list, test_inputs: dict) -> Callable:
    """
    Helper function for parametrizing Numba optimization test cases.
    
    Parameters
    ----------
    functions : list
        List of functions to test with Numba optimization
    test_inputs : dict
        Dictionary of test inputs to use for each function
    
    Returns
    -------
    Callable
        Decorator function that can be applied to test functions
    """
    # Verify functions are callable
    for func in functions:
        if not callable(func):
            raise ValueError(f"Expected callable function, got {type(func)}")
    
    # Create combinations of functions and inputs
    test_cases = []
    for func in functions:
        func_name = func.__name__
        if func_name in test_inputs:
            inputs = test_inputs[func_name]
            for input_case in inputs:
                test_cases.append((func, input_case))
        else:
            test_cases.append((func, None))
    
    # Return parametrize decorator with numba marker
    return pytest.mark.numba(
        pytest.mark.parametrize(('function', 'inputs'), test_cases)
    )


def validate_test_data(data: Union[np.ndarray, pd.DataFrame, pd.Series], 
                     expected_type: str) -> bool:
    """
    Validate test data for consistency and correct format.
    
    Parameters
    ----------
    data : Union[np.ndarray, pd.DataFrame, pd.Series]
        Data to validate
    expected_type : str
        Expected type of data ('array', 'dataframe', 'series')
    
    Returns
    -------
    bool
        True if data is valid, otherwise raises appropriate exception
    """
    # Check data is not None
    if data is None:
        raise ValueError("Test data cannot be None")
    
    # Check data has the correct type
    if expected_type.lower() == 'array':
        if not isinstance(data, np.ndarray):
            raise TypeError(f"Expected numpy array, got {type(data)}")
        
        # Additional checks for arrays
        if not np.isfinite(data).all():
            raise ValueError("Array contains non-finite values (NaN or Inf)")
        
        # Check array shape
        if data.ndim == 0:
            raise ValueError("Array must have at least 1 dimension")
        
        if data.size == 0:
            raise ValueError("Array is empty")
    
    elif expected_type.lower() == 'dataframe':
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(data)}")
        
        # Additional checks for DataFrames
        if data.empty:
            raise ValueError("DataFrame is empty")
        
        # Check for NaN or Inf values if specified
        if not data.isna().any().any() and np.isfinite(data.values).all():
            pass  # Data is valid
        else:
            # Count NaN and Inf values
            nan_count = data.isna().sum().sum()
            inf_mask = ~np.isfinite(data.values) & ~np.isnan(data.values)
            inf_count = np.sum(inf_mask)
            
            if nan_count > 0 or inf_count > 0:
                logger.warning(f"DataFrame contains {nan_count} NaN and {inf_count} Inf values")
    
    elif expected_type.lower() == 'series':
        if not isinstance(data, pd.Series):
            raise TypeError(f"Expected pandas Series, got {type(data)}")
        
        # Additional checks for Series
        if data.empty:
            raise ValueError("Series is empty")
        
        # Check for NaN or Inf values if specified
        if not data.isna().any() and np.isfinite(data.values).all():
            pass  # Data is valid
        else:
            # Count NaN and Inf values
            nan_count = data.isna().sum()
            inf_mask = ~np.isfinite(data.values) & ~np.isnan(data.values)
            inf_count = np.sum(inf_mask)
            
            if nan_count > 0 or inf_count > 0:
                logger.warning(f"Series contains {nan_count} NaN and {inf_count} Inf values")
    
    else:
        raise ValueError(f"Unknown expected_type: {expected_type}")
    
    return True


def check_utils_imports() -> bool:
    """
    Verify that utility modules can be correctly imported.
    
    Returns
    -------
    bool
        True if all imports are successful, False otherwise
    """
    try:
        # Try importing key utility modules
        from mfe.utils import validation
        from mfe.utils import numba_helpers
        from mfe.utils import data_handling
        from mfe.utils import numpy_helpers
        from mfe.utils import pandas_helpers
        
        # Check for critical functions in each module
        assert hasattr(validation, 'validate_array')
        assert hasattr(numba_helpers, 'optimized_jit')
        assert hasattr(data_handling, 'convert_time_series')
        assert hasattr(numpy_helpers, 'ensure_array')
        assert hasattr(pandas_helpers, 'convert_to_dataframe')
        
        logger.info("All utility modules imported successfully")
        return True
        
    except ImportError as e:
        logger.error(f"Failed to import utility module: {str(e)}")
        return False
    
    except AssertionError as e:
        logger.error(f"Missing critical function in utility module: {str(e)}")
        return False


def create_test_samples(data_type: str, size: int, params: dict = None) -> Union[np.ndarray, pd.DataFrame, pd.Series]:
    """
    Create standardized test data samples for utility function testing.
    
    Parameters
    ----------
    data_type : str
        Type of data to generate:
        - 'returns': Financial return series
        - 'timeseries': Time series with trend and seasonality
        - 'matrix': Multi-dimensional array
    size : int
        Size/length of the test data
    params : dict, default=None
        Additional parameters for data generation
    
    Returns
    -------
    Union[np.ndarray, pd.DataFrame, pd.Series]
        Generated test data in requested format
    """
    if params is None:
        params = {}
    
    # Set random seed for reproducibility
    np.random.seed(params.get('seed', 42))
    
    if data_type == 'returns':
        # Generate random return data
        mean = params.get('mean', 0.0001)
        std = params.get('std', 0.01)
        data = np.random.normal(mean, std, size)
        
        if params.get('as_series', False):
            # Create a pandas Series with datetime index
            date_range = pd.date_range(start='2020-01-01', periods=size, freq='D')
            return pd.Series(data, index=date_range, name='returns')
        
        return data
    
    elif data_type == 'timeseries':
        # Generate time series data with trend and seasonality
        time = np.arange(size)
        trend = params.get('trend', 0.001) * time
        seasonality = params.get('seasonality', 0.1) * np.sin(2 * np.pi * time / params.get('period', 20))
        noise = params.get('noise', 0.02) * np.random.randn(size)
        
        data = trend + seasonality + noise
        
        if params.get('as_dataframe', False):
            # Create a pandas DataFrame with datetime index
            date_range = pd.date_range(start='2020-01-01', periods=size, freq='D')
            return pd.DataFrame({'value': data}, index=date_range)
        
        if params.get('as_series', False):
            # Create a pandas Series with datetime index
            date_range = pd.date_range(start='2020-01-01', periods=size, freq='D')
            return pd.Series(data, index=date_range, name='timeseries')
        
        return data
    
    elif data_type == 'matrix':
        # Generate matrix data
        rows = size
        cols = params.get('cols', 3)
        
        data = np.random.randn(rows, cols)
        
        if params.get('as_dataframe', False):
            # Create a pandas DataFrame with datetime index
            date_range = pd.date_range(start='2020-01-01', periods=rows, freq='D')
            columns = params.get('columns', [f'var_{i}' for i in range(cols)])
            return pd.DataFrame(data, index=date_range, columns=columns)
        
        return data
    
    else:
        raise ValueError(f"Unknown data_type: {data_type}")