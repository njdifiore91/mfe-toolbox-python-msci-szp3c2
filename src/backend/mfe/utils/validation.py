"""
MFE Toolbox - Input Validation Module

This module provides comprehensive validation functions for ensuring data integrity
and parameter constraints across the MFE Toolbox. It includes functions for validating
array shapes, types, numerical constraints, and model specifications.

The module implements Python 3.12 type hints and robust error handling for production-grade
input validation in econometric modeling applications.
"""

import numpy as np  # numpy 1.26.3
import pandas as pd  # pandas 2.1.4
from typing import Any, Dict, List, Optional, Tuple, Union, Type
import numbers  # Python 3.12
import logging  # Python 3.12
import scipy.linalg  # scipy 1.11.4

# Setup module logger
logger = logging.getLogger(__name__)


def validate_array(
    arr: Any,
    expected_shape: Optional[Tuple[int, ...]] = None,
    expected_dtype: Optional[np.dtype] = None,
    allow_none: bool = False,
    param_name: str = "array"
) -> np.ndarray:
    """
    Validates that an input is a proper numpy array with specified shape and dtype.
    
    Parameters
    ----------
    arr : Any
        The input to validate as a numpy array
    expected_shape : Optional[Tuple[int, ...]], default=None
        The expected shape of the array. If None, any shape is accepted.
    expected_dtype : Optional[np.dtype], default=None
        The expected dtype of the array. If None, any dtype is accepted.
    allow_none : bool, default=False
        If True, None is allowed as input and will be returned as is.
    param_name : str, default="array"
        Name of the parameter for error messages.
        
    Returns
    -------
    np.ndarray
        The validated numpy array.
        
    Raises
    ------
    TypeError
        If the input cannot be converted to a numpy array or has wrong dtype.
    ValueError
        If the array shape does not match the expected shape.
    """
    # Handle None case
    if arr is None:
        if allow_none:
            return None
        else:
            raise TypeError(f"Parameter '{param_name}' cannot be None")
    
    # Convert to numpy array if not already
    if not isinstance(arr, np.ndarray):
        try:
            arr = np.asarray(arr)
        except Exception as e:
            raise TypeError(f"Parameter '{param_name}' could not be converted to numpy array: {str(e)}")
    
    # Validate shape if expected_shape is provided
    if expected_shape is not None:
        if arr.shape != expected_shape:
            raise ValueError(
                f"Parameter '{param_name}' has incorrect shape. "
                f"Expected {expected_shape}, got {arr.shape}"
            )
    
    # Validate dtype if expected_dtype is provided
    if expected_dtype is not None:
        if not np.issubdtype(arr.dtype, expected_dtype):
            try:
                arr = arr.astype(expected_dtype)
                logger.debug(f"Converted '{param_name}' from {arr.dtype} to {expected_dtype}")
            except Exception as e:
                raise TypeError(
                    f"Parameter '{param_name}' has incorrect dtype and could not be converted. "
                    f"Expected {expected_dtype}, got {arr.dtype}. Error: {str(e)}"
                )
    
    return arr


def validate_data(
    data: Union[np.ndarray, pd.Series, pd.DataFrame, List],
    min_length: Optional[int] = None,
    ensure_2d: bool = False,
    ensure_contiguous: bool = False,
    dtype: Optional[np.dtype] = None,
    allow_none: bool = False,
    param_name: str = "data"
) -> np.ndarray:
    """
    Validates input data for time series analysis, handling numpy arrays, 
    pandas Series, DataFrames, and list inputs.
    
    Parameters
    ----------
    data : Union[np.ndarray, pd.Series, pd.DataFrame, List]
        The input data to validate
    min_length : Optional[int], default=None
        Minimum required length of the data. If None, no minimum is enforced.
    ensure_2d : bool, default=False
        If True, ensures the output is a 2D array
    ensure_contiguous : bool, default=False
        If True, ensures the memory layout is contiguous (C order)
    dtype : Optional[np.dtype], default=None
        The desired dtype for the output array
    allow_none : bool, default=False
        If True, None is allowed as input and will be returned as is
    param_name : str, default="data"
        Name of the parameter for error messages
        
    Returns
    -------
    np.ndarray
        The validated numpy array
        
    Raises
    ------
    TypeError
        If the input has incorrect type
    ValueError
        If the input does not meet the validation criteria
    """
    # Handle None case
    if data is None:
        if allow_none:
            return None
        else:
            raise TypeError(f"Parameter '{param_name}' cannot be None")
    
    # Convert pandas Series or DataFrame to numpy array
    if isinstance(data, (pd.Series, pd.DataFrame)):
        data = data.values
    
    # Convert list to numpy array
    if not isinstance(data, np.ndarray):
        try:
            data = np.asarray(data)
        except Exception as e:
            raise TypeError(f"Parameter '{param_name}' could not be converted to numpy array: {str(e)}")
    
    # Check minimum length
    if min_length is not None and len(data) < min_length:
        raise ValueError(
            f"Parameter '{param_name}' is too short. "
            f"Minimum length is {min_length}, got {len(data)}"
        )
    
    # Ensure 2D array if requested
    if ensure_2d:
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        elif data.ndim > 2:
            raise ValueError(
                f"Parameter '{param_name}' has too many dimensions. "
                f"Expected 1D or 2D array, got {data.ndim}D"
            )
    
    # Ensure contiguous memory layout if requested
    if ensure_contiguous and not data.flags.c_contiguous:
        data = np.ascontiguousarray(data)
    
    # Convert to specified dtype if provided
    if dtype is not None and not np.issubdtype(data.dtype, dtype):
        try:
            data = data.astype(dtype)
        except Exception as e:
            raise TypeError(
                f"Parameter '{param_name}' could not be converted to {dtype}: {str(e)}"
            )
    
    return data


def validate_params(
    params: Dict[str, Any],
    param_specs: Dict[str, Dict[str, Any]],
    strict: bool = True,
    raise_error: bool = True
) -> bool:
    """
    Validates parameter dictionaries against specifications with type and constraint checking.
    
    Parameters
    ----------
    params : Dict[str, Any]
        Dictionary of parameters to validate
    param_specs : Dict[str, Dict[str, Any]]
        Specifications for each parameter, with keys being parameter names and values
        being dictionaries with the following possible keys:
        - 'type': The expected type(s) of the parameter
        - 'required': Whether the parameter is required (default: True)
        - 'min': Minimum value for numeric parameters
        - 'max': Maximum value for numeric parameters
        - 'allowed_values': List of allowed values
        - 'validator': Custom validation function that takes the parameter value
                       and returns a boolean
    strict : bool, default=True
        If True, extra parameters not in param_specs will cause validation to fail
    raise_error : bool, default=True
        If True, raise detailed exceptions for validation failures
        
    Returns
    -------
    bool
        True if all parameters pass validation, otherwise False
        
    Raises
    ------
    TypeError
        If input params is not a dictionary or parameter types don't match specifications
    ValueError
        If required parameters are missing or parameter constraints are violated
    """
    if not isinstance(params, dict):
        if raise_error:
            raise TypeError("params must be a dictionary")
        return False
    
    # Check for required parameters
    for param_name, spec in param_specs.items():
        required = spec.get('required', True)
        if required and param_name not in params:
            if raise_error:
                raise ValueError(f"Required parameter '{param_name}' is missing")
            return False
    
    # Check strict mode - no extra parameters allowed
    if strict:
        extra_params = set(params.keys()) - set(param_specs.keys())
        if extra_params:
            if raise_error:
                raise ValueError(f"Unexpected parameters: {', '.join(extra_params)}")
            return False
    
    # Validate each parameter
    for param_name, param_value in params.items():
        # Skip parameters not in specs (already checked in strict mode)
        if param_name not in param_specs:
            continue
        
        spec = param_specs[param_name]
        
        # Type validation
        if 'type' in spec:
            expected_type = spec['type']
            if not validate_type(param_value, expected_type, param_name, raise_error):
                return False
        
        # Skip further validation if None and None is allowed
        if param_value is None and spec.get('allow_none', False):
            continue
        
        # Numeric constraints for numeric types
        if isinstance(param_value, (int, float, np.number)):
            # Min value
            if 'min' in spec:
                min_val = spec['min']
                min_inclusive = spec.get('min_inclusive', True)
                
                if min_inclusive and param_value < min_val:
                    if raise_error:
                        raise ValueError(
                            f"Parameter '{param_name}' must be >= {min_val}, got {param_value}"
                        )
                    return False
                elif not min_inclusive and param_value <= min_val:
                    if raise_error:
                        raise ValueError(
                            f"Parameter '{param_name}' must be > {min_val}, got {param_value}"
                        )
                    return False
            
            # Max value
            if 'max' in spec:
                max_val = spec['max']
                max_inclusive = spec.get('max_inclusive', True)
                
                if max_inclusive and param_value > max_val:
                    if raise_error:
                        raise ValueError(
                            f"Parameter '{param_name}' must be <= {max_val}, got {param_value}"
                        )
                    return False
                elif not max_inclusive and param_value >= max_val:
                    if raise_error:
                        raise ValueError(
                            f"Parameter '{param_name}' must be < {max_val}, got {param_value}"
                        )
                    return False
        
        # Allowed values
        if 'allowed_values' in spec:
            allowed_values = spec['allowed_values']
            if param_value not in allowed_values:
                if raise_error:
                    raise ValueError(
                        f"Parameter '{param_name}' must be one of {allowed_values}, got {param_value}"
                    )
                return False
        
        # Custom validator function
        if 'validator' in spec:
            validator = spec['validator']
            if not validator(param_value):
                if raise_error:
                    raise ValueError(
                        f"Parameter '{param_name}' failed custom validation: {param_value}"
                    )
                return False
    
    return True


def check_dimension(
    arr: np.ndarray,
    dim: int,
    param_name: str = "array",
    raise_error: bool = True
) -> bool:
    """
    Validates that an array has a specific dimension.
    
    Parameters
    ----------
    arr : np.ndarray
        The array to validate
    dim : int
        The expected dimension (number of axes)
    param_name : str, default="array"
        Name of the parameter for error messages
    raise_error : bool, default=True
        If True, raise ValueError for dimension mismatch
        
    Returns
    -------
    bool
        True if array has expected dimension, otherwise False
        
    Raises
    ------
    TypeError
        If arr is not a numpy array
    ValueError
        If array dimension doesn't match expected dimension
    """
    if not isinstance(arr, np.ndarray):
        if raise_error:
            raise TypeError(f"Parameter '{param_name}' must be a numpy array")
        return False
    
    if arr.ndim != dim:
        if raise_error:
            raise ValueError(
                f"Parameter '{param_name}' has incorrect dimension. "
                f"Expected {dim}D, got {arr.ndim}D"
            )
        return False
    
    return True


def check_dimensions(
    arr: np.ndarray,
    dims: Union[int, List[int], Tuple[int, ...]],
    param_name: str = "array",
    raise_error: bool = True
) -> bool:
    """
    Validates that an array has one of several allowed dimensions.
    
    Parameters
    ----------
    arr : np.ndarray
        The array to validate
    dims : Union[int, List[int], Tuple[int, ...]]
        The allowed dimensions (number of axes)
    param_name : str, default="array"
        Name of the parameter for error messages
    raise_error : bool, default=True
        If True, raise ValueError for dimension mismatch
        
    Returns
    -------
    bool
        True if array dimension is in allowed dimensions, otherwise False
        
    Raises
    ------
    TypeError
        If arr is not a numpy array
    ValueError
        If array dimension is not in allowed dimensions
    """
    if not isinstance(arr, np.ndarray):
        if raise_error:
            raise TypeError(f"Parameter '{param_name}' must be a numpy array")
        return False
    
    # Convert single dimension to list
    if isinstance(dims, int):
        dims = [dims]
    
    if arr.ndim not in dims:
        if raise_error:
            raise ValueError(
                f"Parameter '{param_name}' has incorrect dimension. "
                f"Expected one of {dims}, got {arr.ndim}"
            )
        return False
    
    return True


def is_positive_integer(
    value: Any,
    param_name: Optional[str] = None,
    raise_error: bool = True
) -> bool:
    """
    Checks if a value is a positive integer (> 0).
    
    Parameters
    ----------
    value : Any
        The value to check
    param_name : Optional[str], default=None
        Name of the parameter for error messages
    raise_error : bool, default=True
        If True, raise ValueError for validation failure
        
    Returns
    -------
    bool
        True if value is a positive integer, otherwise False
        
    Raises
    ------
    ValueError
        If value is not a positive integer
    """
    param_desc = f"Parameter '{param_name}'" if param_name else "Value"
    
    # Check if value is an integer
    if not isinstance(value, (int, np.integer)):
        if raise_error:
            raise ValueError(f"{param_desc} must be an integer, got {type(value).__name__}")
        return False
    
    # Check if value is positive
    if value <= 0:
        if raise_error:
            raise ValueError(f"{param_desc} must be positive (> 0), got {value}")
        return False
    
    return True


def is_non_negative_integer(
    value: Any,
    param_name: Optional[str] = None,
    raise_error: bool = True
) -> bool:
    """
    Checks if a value is a non-negative integer (≥ 0).
    
    Parameters
    ----------
    value : Any
        The value to check
    param_name : Optional[str], default=None
        Name of the parameter for error messages
    raise_error : bool, default=True
        If True, raise ValueError for validation failure
        
    Returns
    -------
    bool
        True if value is a non-negative integer, otherwise False
        
    Raises
    ------
    ValueError
        If value is not a non-negative integer
    """
    param_desc = f"Parameter '{param_name}'" if param_name else "Value"
    
    # Check if value is an integer
    if not isinstance(value, (int, np.integer)):
        if raise_error:
            raise ValueError(f"{param_desc} must be an integer, got {type(value).__name__}")
        return False
    
    # Check if value is non-negative
    if value < 0:
        if raise_error:
            raise ValueError(f"{param_desc} must be non-negative (≥ 0), got {value}")
        return False
    
    return True


def is_positive_float(
    value: Any,
    zero_allowed: bool = False,
    param_name: Optional[str] = None,
    raise_error: bool = True
) -> bool:
    """
    Checks if a value is a positive float (or integer).
    
    Parameters
    ----------
    value : Any
        The value to check
    zero_allowed : bool, default=False
        If True, zero is considered valid
    param_name : Optional[str], default=None
        Name of the parameter for error messages
    raise_error : bool, default=True
        If True, raise ValueError for validation failure
        
    Returns
    -------
    bool
        True if value is a positive number, otherwise False
        
    Raises
    ------
    ValueError
        If value is not a positive number
    """
    param_desc = f"Parameter '{param_name}'" if param_name else "Value"
    
    # Check if value is a number
    if not isinstance(value, (int, float, np.number)):
        if raise_error:
            raise ValueError(f"{param_desc} must be a number, got {type(value).__name__}")
        return False
    
    # Check if value is positive
    if zero_allowed:
        if value < 0:
            if raise_error:
                raise ValueError(f"{param_desc} must be non-negative (≥ 0), got {value}")
            return False
    else:
        if value <= 0:
            if raise_error:
                raise ValueError(f"{param_desc} must be positive (> 0), got {value}")
            return False
    
    return True


def is_probability(
    value: Any,
    param_name: Optional[str] = None,
    raise_error: bool = True
) -> bool:
    """
    Checks if a value is a valid probability (between 0 and 1 inclusive).
    
    Parameters
    ----------
    value : Any
        The value to check
    param_name : Optional[str], default=None
        Name of the parameter for error messages
    raise_error : bool, default=True
        If True, raise ValueError for validation failure
        
    Returns
    -------
    bool
        True if value is a valid probability, otherwise False
        
    Raises
    ------
    ValueError
        If value is not a valid probability
    """
    param_desc = f"Parameter '{param_name}'" if param_name else "Value"
    
    # Check if value is a number
    if not isinstance(value, (int, float, np.number)):
        if raise_error:
            raise ValueError(f"{param_desc} must be a number, got {type(value).__name__}")
        return False
    
    # Check if value is between 0 and 1
    if value < 0 or value > 1:
        if raise_error:
            raise ValueError(f"{param_desc} must be between 0 and 1, got {value}")
        return False
    
    return True


def check_lags(
    lags: Union[int, List[int], np.ndarray],
    max_lag: Optional[int] = None,
    raise_error: bool = True
) -> np.ndarray:
    """
    Validates lag specification for time series analysis.
    
    Parameters
    ----------
    lags : Union[int, List[int], np.ndarray]
        The lag or lags to validate, either a single integer or a list/array of integers
    max_lag : Optional[int], default=None
        Maximum allowed lag value. If None, no maximum is enforced.
    raise_error : bool, default=True
        If True, raise ValueError for validation failure
        
    Returns
    -------
    np.ndarray
        Validated and sorted array of lags
        
    Raises
    ------
    ValueError
        If lags are not valid
    """
    # Convert single integer to array
    if isinstance(lags, (int, np.integer)):
        lags = np.array([lags], dtype=int)
    
    # Convert list to array
    if isinstance(lags, list):
        try:
            lags = np.array(lags, dtype=int)
        except Exception as e:
            if raise_error:
                raise ValueError(f"Could not convert lags to integer array: {str(e)}")
            return np.array([], dtype=int)
    
    # Validate array type
    if not isinstance(lags, np.ndarray):
        if raise_error:
            raise ValueError(f"Lags must be an integer, list of integers, or numpy array, got {type(lags).__name__}")
        return np.array([], dtype=int)
    
    # Ensure integer type
    if not np.issubdtype(lags.dtype, np.integer):
        try:
            lags = lags.astype(int)
        except Exception as e:
            if raise_error:
                raise ValueError(f"Lags must be integers, got {lags.dtype}: {str(e)}")
            return np.array([], dtype=int)
    
    # Check for negative lags
    if np.any(lags < 0):
        if raise_error:
            raise ValueError(f"All lags must be non-negative, got {lags}")
        return np.array([], dtype=int)
    
    # Check maximum lag if specified
    if max_lag is not None and np.any(lags > max_lag):
        if raise_error:
            raise ValueError(f"All lags must be <= {max_lag}, got {lags}")
        return np.array([], dtype=int)
    
    # Sort lags in ascending order
    return np.sort(lags)


def check_range(
    value: Union[int, float, np.number],
    min_value: Union[int, float, np.number],
    max_value: Union[int, float, np.number],
    inclusive_min: bool = True,
    inclusive_max: bool = True,
    param_name: Optional[str] = None,
    raise_error: bool = True
) -> bool:
    """
    Checks if a value is within a specified range.
    
    Parameters
    ----------
    value : Union[int, float, np.number]
        The value to check
    min_value : Union[int, float, np.number]
        The minimum allowed value
    max_value : Union[int, float, np.number]
        The maximum allowed value
    inclusive_min : bool, default=True
        If True, minimum value is inclusive (>=)
    inclusive_max : bool, default=True
        If True, maximum value is inclusive (<=)
    param_name : Optional[str], default=None
        Name of the parameter for error messages
    raise_error : bool, default=True
        If True, raise ValueError for validation failure
        
    Returns
    -------
    bool
        True if value is within range, otherwise False
        
    Raises
    ------
    ValueError
        If value is not within the specified range
    """
    param_desc = f"Parameter '{param_name}'" if param_name else "Value"
    
    # Check if value is a number
    if not isinstance(value, (int, float, np.number)):
        if raise_error:
            raise ValueError(f"{param_desc} must be a number, got {type(value).__name__}")
        return False
    
    # Check minimum bound
    min_check = value >= min_value if inclusive_min else value > min_value
    if not min_check:
        if raise_error:
            operator = ">=" if inclusive_min else ">"
            raise ValueError(f"{param_desc} must be {operator} {min_value}, got {value}")
        return False
    
    # Check maximum bound
    max_check = value <= max_value if inclusive_max else value < max_value
    if not max_check:
        if raise_error:
            operator = "<=" if inclusive_max else "<"
            raise ValueError(f"{param_desc} must be {operator} {max_value}, got {value}")
        return False
    
    return True


def check_order(
    order: Union[Tuple[int, ...], List[int]],
    model_type: str,
    raise_error: bool = True
) -> bool:
    """
    Validates model order specifications for ARIMA/GARCH models.
    
    Parameters
    ----------
    order : Union[Tuple[int, ...], List[int]]
        The model order specification (e.g., (p, q) for GARCH, (p, d, q) for ARIMA)
    model_type : str
        Type of model to validate order for ('ARIMA', 'GARCH', 'ARFIMA', etc.)
    raise_error : bool, default=True
        If True, raise ValueError for validation failure
        
    Returns
    -------
    bool
        True if order is valid for the specified model type, otherwise False
        
    Raises
    ------
    ValueError
        If order is not valid for the specified model type
    """
    # Check if order is tuple or list
    if not isinstance(order, (tuple, list)):
        if raise_error:
            raise ValueError(f"Order must be a tuple or list, got {type(order).__name__}")
        return False
    
    # Check if all elements are integers
    if not all(isinstance(x, (int, np.integer)) for x in order):
        if raise_error:
            raise ValueError("All elements in order must be integers")
        return False
    
    # Check if all elements are non-negative
    if not all(x >= 0 for x in order):
        if raise_error:
            raise ValueError("All elements in order must be non-negative")
        return False
    
    # Check length based on model type
    model_type = model_type.upper()
    
    if model_type == 'ARIMA':
        if len(order) != 3:
            if raise_error:
                raise ValueError(f"ARIMA order must be (p, d, q), got {order}")
            return False
    elif model_type == 'ARFIMA':
        if len(order) != 3:
            if raise_error:
                raise ValueError(f"ARFIMA order must be (p, d, q), got {order}")
            return False
    elif model_type == 'GARCH':
        if len(order) != 2:
            if raise_error:
                raise ValueError(f"GARCH order must be (p, q), got {order}")
            return False
    elif model_type == 'EGARCH':
        if len(order) != 2:
            if raise_error:
                raise ValueError(f"EGARCH order must be (p, q), got {order}")
            return False
    elif model_type == 'FIGARCH':
        if len(order) != 3:
            if raise_error:
                raise ValueError(f"FIGARCH order must be (p, d, q), got {order}")
            return False
    elif model_type == 'ARIMAX':
        if len(order) != 3:
            if raise_error:
                raise ValueError(f"ARIMAX order must be (p, d, q), got {order}")
            return False
    else:
        if raise_error:
            raise ValueError(f"Unknown model type: {model_type}")
        return False
    
    return True


def check_time_series(
    data: Union[np.ndarray, pd.Series, pd.DataFrame, List],
    min_length: Optional[int] = None,
    allow_2d: bool = False,
    allow_missing: bool = False,
    raise_error: bool = True
) -> np.ndarray:
    """
    Validates time series data for econometric modeling.
    
    Parameters
    ----------
    data : Union[np.ndarray, pd.Series, pd.DataFrame, List]
        The time series data to validate
    min_length : Optional[int], default=None
        Minimum required length of the time series. If None, no minimum is enforced.
    allow_2d : bool, default=False
        If True, allows 2D arrays for multivariate time series
    allow_missing : bool, default=False
        If True, allows NaN and Inf values in the data
    raise_error : bool, default=True
        If True, raise ValueError for validation failure
        
    Returns
    -------
    np.ndarray
        Validated time series data as numpy array
        
    Raises
    ------
    ValueError
        If data does not meet the requirements for time series analysis
    """
    # Convert to numpy array using validate_data
    try:
        data_array = validate_data(
            data, 
            min_length=min_length,
            ensure_2d=False,  # We'll handle dimensions separately
            dtype=np.float64,
            param_name="time_series"
        )
    except Exception as e:
        if raise_error:
            raise ValueError(f"Invalid time series data: {str(e)}")
        return np.array([])
    
    # Check dimensions
    if not allow_2d and data_array.ndim > 1:
        if raise_error:
            raise ValueError(
                f"Time series must be 1-dimensional, got {data_array.ndim}-dimensional data. "
                f"Use allow_2d=True for multivariate time series."
            )
        return np.array([])
    
    if data_array.ndim > 2:
        if raise_error:
            raise ValueError(
                f"Time series can have at most 2 dimensions, got {data_array.ndim}"
            )
        return np.array([])
    
    # Check for NaN and Inf values
    if not allow_missing and not np.isfinite(data_array).all():
        if raise_error:
            raise ValueError(
                "Time series contains NaN or Inf values. "
                "Use allow_missing=True to allow missing values."
            )
        return np.array([])
    
    return data_array


def is_float_array(
    arr: np.ndarray,
    raise_error: bool = True
) -> bool:
    """
    Checks if an array contains only floating-point values.
    
    Parameters
    ----------
    arr : np.ndarray
        The array to check
    raise_error : bool, default=True
        If True, raise TypeError for validation failure
        
    Returns
    -------
    bool
        True if array contains only floats, otherwise False
        
    Raises
    ------
    TypeError
        If arr is not a numpy array or not a floating-point array
    """
    if not isinstance(arr, np.ndarray):
        if raise_error:
            raise TypeError("Input must be a numpy array")
        return False
    
    if not np.issubdtype(arr.dtype, np.floating):
        if raise_error:
            raise TypeError(
                f"Array must have floating-point dtype, got {arr.dtype}"
            )
        return False
    
    return True


def is_int_array(
    arr: np.ndarray,
    raise_error: bool = True
) -> bool:
    """
    Checks if an array contains only integer values.
    
    Parameters
    ----------
    arr : np.ndarray
        The array to check
    raise_error : bool, default=True
        If True, raise TypeError for validation failure
        
    Returns
    -------
    bool
        True if array contains only integers, otherwise False
        
    Raises
    ------
    TypeError
        If arr is not a numpy array or not an integer array
    """
    if not isinstance(arr, np.ndarray):
        if raise_error:
            raise TypeError("Input must be a numpy array")
        return False
    
    if not np.issubdtype(arr.dtype, np.integer):
        if raise_error:
            raise TypeError(
                f"Array must have integer dtype, got {arr.dtype}"
            )
        return False
    
    return True


def is_positive_definite(
    matrix: np.ndarray,
    raise_error: bool = True
) -> bool:
    """
    Checks if a matrix is positive definite (all eigenvalues > 0).
    
    Parameters
    ----------
    matrix : np.ndarray
        The matrix to check
    raise_error : bool, default=True
        If True, raise ValueError for validation failure
        
    Returns
    -------
    bool
        True if matrix is positive definite, otherwise False
        
    Raises
    ------
    ValueError
        If matrix is not positive definite or not a square matrix
    """
    # Check if matrix is a 2D array
    if not isinstance(matrix, np.ndarray) or matrix.ndim != 2:
        if raise_error:
            raise ValueError("Input must be a 2D numpy array")
        return False
    
    # Check if matrix is square
    if matrix.shape[0] != matrix.shape[1]:
        if raise_error:
            raise ValueError(
                f"Matrix must be square, got shape {matrix.shape}"
            )
        return False
    
    # Check if matrix is symmetric
    if not is_symmetric(matrix, raise_error=False):
        if raise_error:
            raise ValueError("Matrix must be symmetric to check positive definiteness")
        return False
    
    try:
        # Compute eigenvalues using scipy.linalg.eigvalsh (efficient for symmetric matrices)
        eigenvalues = scipy.linalg.eigvalsh(matrix)
        
        # Check if all eigenvalues are positive
        if np.all(eigenvalues > 0):
            return True
        else:
            if raise_error:
                min_eig = np.min(eigenvalues)
                raise ValueError(
                    f"Matrix is not positive definite. Smallest eigenvalue: {min_eig}"
                )
            return False
    except Exception as e:
        if raise_error:
            raise ValueError(f"Failed to check positive definiteness: {str(e)}")
        return False


def is_symmetric(
    matrix: np.ndarray,
    tolerance: float = 1e-8,
    raise_error: bool = True
) -> bool:
    """
    Checks if a matrix is symmetric (equal to its transpose).
    
    Parameters
    ----------
    matrix : np.ndarray
        The matrix to check
    tolerance : float, default=1e-8
        Tolerance for floating-point comparison
    raise_error : bool, default=True
        If True, raise ValueError for validation failure
        
    Returns
    -------
    bool
        True if matrix is symmetric, otherwise False
        
    Raises
    ------
    ValueError
        If matrix is not symmetric or not a square matrix
    """
    # Check if matrix is a 2D array
    if not isinstance(matrix, np.ndarray) or matrix.ndim != 2:
        if raise_error:
            raise ValueError("Input must be a 2D numpy array")
        return False
    
    # Check if matrix is square
    if matrix.shape[0] != matrix.shape[1]:
        if raise_error:
            raise ValueError(
                f"Matrix must be square to be symmetric, got shape {matrix.shape}"
            )
        return False
    
    # Calculate the difference between matrix and its transpose
    diff = matrix - matrix.T
    
    # Check if maximum absolute difference is within tolerance
    max_diff = np.max(np.abs(diff))
    if max_diff <= tolerance:
        return True
    else:
        if raise_error:
            raise ValueError(
                f"Matrix is not symmetric. Maximum difference: {max_diff}"
            )
        return False


def check_strictly_increasing(
    array: np.ndarray,
    raise_error: bool = True
) -> bool:
    """
    Checks if an array is strictly increasing (each element > previous).
    
    Parameters
    ----------
    array : np.ndarray
        The array to check
    raise_error : bool, default=True
        If True, raise ValueError for validation failure
        
    Returns
    -------
    bool
        True if array is strictly increasing, otherwise False
        
    Raises
    ------
    ValueError
        If array is not strictly increasing or not a 1D array
    """
    # Check if array is a 1D numpy array
    if not isinstance(array, np.ndarray) or array.ndim != 1:
        if raise_error:
            raise ValueError("Input must be a 1D numpy array")
        return False
    
    # Check array length
    if len(array) <= 1:
        return True  # Single element or empty array is trivially strictly increasing
    
    # Compare each element to the previous one
    is_increasing = np.all(array[1:] > array[:-1])
    
    if not is_increasing:
        if raise_error:
            # Find the first non-increasing index
            for i in range(1, len(array)):
                if array[i] <= array[i-1]:
                    raise ValueError(
                        f"Array is not strictly increasing at index {i}. "
                        f"Values: {array[i-1]} and {array[i]}"
                    )
        return False
    
    return True


def check_shape_compatibility(
    arrays: List[np.ndarray],
    operation: str,
    raise_error: bool = True
) -> bool:
    """
    Validates that array shapes are compatible for specified operations.
    
    Parameters
    ----------
    arrays : List[np.ndarray]
        List of arrays to check for compatibility
    operation : str
        Type of operation to check compatibility for:
        - 'addition': All arrays must have the same shape
        - 'multiplication': Matrix multiplication compatibility
        - 'broadcasting': NumPy broadcasting rules
    raise_error : bool, default=True
        If True, raise ValueError for validation failure
        
    Returns
    -------
    bool
        True if shapes are compatible for operation, otherwise False
        
    Raises
    ------
    ValueError
        If shapes are not compatible for the specified operation
    """
    # Check if all inputs are numpy arrays
    if not all(isinstance(arr, np.ndarray) for arr in arrays):
        if raise_error:
            raise ValueError("All inputs must be numpy arrays")
        return False
    
    # Check compatibility based on operation
    operation = operation.lower()
    
    if operation == 'addition':
        # For addition, all arrays must have the same shape
        first_shape = arrays[0].shape
        
        for i, arr in enumerate(arrays[1:], 1):
            if arr.shape != first_shape:
                if raise_error:
                    raise ValueError(
                        f"Arrays not compatible for addition. "
                        f"Shape of array 0: {first_shape}, "
                        f"shape of array {i}: {arr.shape}"
                    )
                return False
        
        return True
    
    elif operation == 'multiplication':
        # For matrix multiplication (A @ B), the last dimension of A must match
        # the second-to-last dimension of B
        if len(arrays) != 2:
            if raise_error:
                raise ValueError(
                    "Matrix multiplication compatibility check requires exactly 2 arrays"
                )
            return False
        
        a, b = arrays
        
        # Special case for 1D arrays
        if a.ndim == 1 and b.ndim == 1:
            # Vector dot product: lengths must match
            if len(a) != len(b):
                if raise_error:
                    raise ValueError(
                        f"Incompatible shapes for vector dot product: {a.shape} and {b.shape}"
                    )
                return False
            return True
        
        # Handle general case
        if a.ndim >= 1 and b.ndim >= 2:
            if a.shape[-1] != b.shape[-2]:
                if raise_error:
                    raise ValueError(
                        f"Incompatible shapes for matrix multiplication: {a.shape} and {b.shape}"
                    )
                return False
            return True
        else:
            if raise_error:
                raise ValueError(
                    f"Arrays must have sufficient dimensions for matrix multiplication, "
                    f"got shapes {a.shape} and {b.shape}"
                )
            return False
    
    elif operation == 'broadcasting':
        # Check NumPy broadcasting rules
        try:
            # Use numpy.broadcast_shapes to check if shapes can be broadcast together
            final_shape = np.broadcast_shapes(*(arr.shape for arr in arrays))
            return True
        except ValueError as e:
            if raise_error:
                raise ValueError(
                    f"Arrays not compatible for broadcasting: {str(e)}"
                )
            return False
    
    else:
        if raise_error:
            raise ValueError(
                f"Unknown operation: {operation}. "
                f"Supported operations: 'addition', 'multiplication', 'broadcasting'"
            )
        return False


def validate_type(
    value: Any,
    expected_type: Union[type, Tuple[type, ...]],
    param_name: Optional[str] = None,
    raise_error: bool = True
) -> bool:
    """
    Validates that a value is of the expected type.
    
    Parameters
    ----------
    value : Any
        The value to validate
    expected_type : Union[type, Tuple[type, ...]]
        The expected type or tuple of allowed types
    param_name : Optional[str], default=None
        Name of the parameter for error messages
    raise_error : bool, default=True
        If True, raise TypeError for validation failure
        
    Returns
    -------
    bool
        True if value is of expected type, otherwise False
        
    Raises
    ------
    TypeError
        If value is not of the expected type
    """
    param_desc = f"Parameter '{param_name}'" if param_name else "Value"
    
    # Handle special case for None
    if value is None:
        if type(None) in expected_type if isinstance(expected_type, tuple) else expected_type is type(None):
            return True
        else:
            if raise_error:
                raise TypeError(f"{param_desc} cannot be None, expected {expected_type.__name__}")
            return False
    
    # Handle numeric types more flexibly
    if expected_type in (float, numbers.Real) or (isinstance(expected_type, tuple) and 
                                                 any(t in (float, numbers.Real) for t in expected_type)):
        if isinstance(value, (int, float, np.number)) and not isinstance(value, bool):
            return True
    
    # Handle integer types more flexibly
    if expected_type in (int, numbers.Integral) or (isinstance(expected_type, tuple) and 
                                                    any(t in (int, numbers.Integral) for t in expected_type)):
        if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
            return True
    
    # Handle NumPy array type with dtype checking
    if expected_type is np.ndarray or (isinstance(expected_type, tuple) and np.ndarray in expected_type):
        if isinstance(value, np.ndarray):
            return True
    
    # Standard isinstance check
    if isinstance(value, expected_type):
        return True
    
    # Type validation failed
    if raise_error:
        if isinstance(expected_type, tuple):
            type_names = ", ".join(t.__name__ for t in expected_type)
            raise TypeError(
                f"{param_desc} has incorrect type. "
                f"Expected one of: {type_names}, got {type(value).__name__}"
            )
        else:
            raise TypeError(
                f"{param_desc} has incorrect type. "
                f"Expected {expected_type.__name__}, got {type(value).__name__}"
            )
    
    return False