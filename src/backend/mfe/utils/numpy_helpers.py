"""
MFE Toolbox - NumPy Helpers Module

This module provides utility functions for NumPy array operations used throughout the MFE Toolbox.
It contains optimized, type-safe implementations of common numerical array operations required
for financial econometric analysis.

The functions in this module leverage Numba JIT compilation where appropriate to achieve
high performance for numerical computations.
"""

import numpy as np  # numpy 1.26.3
import numba  # numba 0.59.0
from typing import Any, List, Optional, Tuple, Union
import logging  # Python standard library

from .validation import validate_type

# Setup module logger
logger = logging.getLogger(__name__)


def ensure_array(
    data: Any,
    dtype: Optional[np.dtype] = None,
    shape: Optional[Tuple[int, ...]] = None,
    allow_none: bool = False
) -> Optional[np.ndarray]:
    """
    Ensures the input is a valid NumPy array with specified type and dimensions.
    
    Parameters
    ----------
    data : Any
        Input data to be converted to a NumPy array
    dtype : Optional[np.dtype], default=None
        The desired data type for the array. If None, uses NumPy's default inference.
    shape : Optional[Tuple[int, ...]], default=None
        The expected shape for the array. If provided, validates that the array has this shape.
    allow_none : bool, default=False
        If True, allows None as a valid input and returns None. If False, raises an error for None.
    
    Returns
    -------
    np.ndarray or None
        Validated NumPy array meeting the specified requirements, or None if input is None 
        and allow_none is True.
    
    Raises
    ------
    TypeError
        If data cannot be converted to a NumPy array or is None when allow_none is False.
    ValueError
        If the resulting array does not match the expected shape.
    """
    # Check if data is None and if allow_none is True
    if data is None:
        if allow_none:
            return None
        else:
            raise TypeError("Input data cannot be None")
    
    # Convert input data to numpy array with specified dtype if provided
    try:
        if dtype is not None:
            arr = np.asarray(data, dtype=dtype)
        else:
            arr = np.asarray(data)
    except Exception as e:
        logger.error(f"Failed to convert input to numpy array: {str(e)}")
        raise TypeError(f"Failed to convert input to numpy array: {str(e)}")
    
    # Validate shape if provided
    if shape is not None:
        if arr.shape != shape:
            raise ValueError(
                f"Array shape mismatch. Expected shape {shape}, got {arr.shape}."
            )
    
    return arr


def validate_dimensions(
    arrays: List[np.ndarray],
    operation: Optional[str] = None
) -> bool:
    """
    Validates that arrays have compatible dimensions for operations.
    
    Parameters
    ----------
    arrays : List[np.ndarray]
        List of arrays to check for dimensional compatibility
    operation : Optional[str], default=None
        The operation to check compatibility for. Options include:
        - 'add': All arrays must have the same shape
        - 'matmul': Arrays must have compatible shapes for matrix multiplication
        - 'broadcast': Arrays must be broadcastable together
        If None, checks for exact shape matching.
    
    Returns
    -------
    bool
        True if dimensions are compatible, raises ValueError otherwise
    
    Raises
    ------
    ValueError
        If dimensions are incompatible for the specified operation
    TypeError
        If any input is not a NumPy array
    """
    if not arrays:
        return True
    
    # Check if all inputs are NumPy arrays
    for i, arr in enumerate(arrays):
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"Input {i} is not a NumPy array")
    
    # Default to exact shape matching if no operation specified
    if operation is None:
        first_shape = arrays[0].shape
        for i, arr in enumerate(arrays[1:], 1):
            if arr.shape != first_shape:
                raise ValueError(
                    f"Arrays have incompatible shapes: {first_shape} vs {arr.shape} for array {i}"
                )
        return True
    
    # Check compatibility based on operation
    operation = operation.lower()
    
    if operation == 'add':
        # For addition, all arrays must have the same shape
        first_shape = arrays[0].shape
        for i, arr in enumerate(arrays[1:], 1):
            if arr.shape != first_shape:
                raise ValueError(
                    f"Arrays not compatible for addition. Shape of array 0: {first_shape}, "
                    f"shape of array {i}: {arr.shape}"
                )
        return True
    
    elif operation == 'matmul':
        # For matrix multiplication (A @ B), the last dimension of A must match
        # the second-to-last dimension of B
        if len(arrays) != 2:
            raise ValueError(
                "Matrix multiplication compatibility check requires exactly 2 arrays"
            )
        
        a, b = arrays
        
        # Special case for 1D arrays
        if a.ndim == 1 and b.ndim == 1:
            # Vector dot product: lengths must match
            if len(a) != len(b):
                raise ValueError(
                    f"Incompatible shapes for vector dot product: {a.shape} and {b.shape}"
                )
            return True
        
        # Handle general case
        if a.ndim >= 1 and b.ndim >= 2:
            if a.shape[-1] != b.shape[-2]:
                raise ValueError(
                    f"Incompatible shapes for matrix multiplication: {a.shape} and {b.shape}"
                )
            return True
        else:
            raise ValueError(
                f"Arrays must have sufficient dimensions for matrix multiplication, "
                f"got shapes {a.shape} and {b.shape}"
            )
    
    elif operation == 'broadcast':
        # Check NumPy broadcasting rules
        try:
            # Use numpy.broadcast_shapes to check if shapes can be broadcast together
            np.broadcast_shapes(*(arr.shape for arr in arrays))
            return True
        except ValueError as e:
            raise ValueError(f"Arrays not compatible for broadcasting: {str(e)}")
    
    else:
        raise ValueError(
            f"Unknown operation: {operation}. "
            f"Supported operations: 'add', 'matmul', 'broadcast'"
        )


@numba.jit(nopython=True)
def safe_divide(
    numerator: np.ndarray,
    denominator: np.ndarray,
    fill_value: float = 0.0
) -> np.ndarray:
    """
    Performs division with customizable handling for division by zero.
    
    Parameters
    ----------
    numerator : np.ndarray
        Numerator array for division
    denominator : np.ndarray
        Denominator array for division
    fill_value : float, default=0.0
        Value to use where denominator is zero
    
    Returns
    -------
    np.ndarray
        Result of division with zeros handled according to fill_value
    
    Notes
    -----
    This function is optimized with Numba for performance.
    """
    # Create output array of same shape as inputs
    result = np.empty_like(numerator, dtype=np.float64)
    
    # Perform safe division
    for i in range(numerator.size):
        num = numerator.flat[i]
        den = denominator.flat[i]
        
        if den == 0:
            result.flat[i] = fill_value
        else:
            result.flat[i] = num / den
    
    return result.reshape(numerator.shape)


def rolling_window(
    arr: np.ndarray,
    window: int,
    step: int = 1
) -> np.ndarray:
    """
    Creates rolling window views of input arrays for efficient computation.
    
    Parameters
    ----------
    arr : np.ndarray
        The input array
    window : int
        Size of the rolling window
    step : int, default=1
        Step size between windows
    
    Returns
    -------
    np.ndarray
        Array of rolling windows
    
    Notes
    -----
    This function creates views without copying data when possible, making
    it memory-efficient for large arrays.
    
    Examples
    --------
    >>> x = np.array([1, 2, 3, 4, 5, 6])
    >>> rolling_window(x, 3, 1)
    array([[1, 2, 3],
           [2, 3, 4],
           [3, 4, 5],
           [4, 5, 6]])
    """
    # Validate input parameters
    if not isinstance(arr, np.ndarray):
        raise TypeError("Input must be a numpy array")
    
    if not isinstance(window, (int, np.integer)):
        raise TypeError("Window size must be an integer")
    
    if not isinstance(step, (int, np.integer)):
        raise TypeError("Step size must be an integer")
    
    if window <= 0:
        raise ValueError("Window size must be positive")
    
    if step <= 0:
        raise ValueError("Step size must be positive")
    
    if window > arr.shape[0]:
        raise ValueError(f"Window size {window} exceeds array length {arr.shape[0]}")
    
    # Calculate output shape based on input array, window size, and step
    num_windows = max(0, (arr.shape[0] - window) // step + 1)
    
    if num_windows == 0:
        return np.array([])
    
    # Get strides for original array
    orig_stride = arr.strides[0]
    
    # Create strided view
    out_shape = (num_windows, window) + arr.shape[1:]
    out_strides = (orig_stride * step, orig_stride) + arr.strides[1:]
    
    # Create strided view of the input array to form windows
    return np.lib.stride_tricks.as_strided(
        arr, shape=out_shape, strides=out_strides, writeable=False
    )


@numba.jit(nopython=True)
def fast_cov(
    x: np.ndarray,
    rowvar: bool = False,
    bias: bool = False
) -> np.ndarray:
    """
    Calculates covariance matrices with optimized performance.
    
    Parameters
    ----------
    x : np.ndarray
        Input array of observations. If rowvar is True, each row represents a
        variable and each column represents an observation. Otherwise, the
        relationship is transposed.
    rowvar : bool, default=False
        If True, each row is a variable and each column is an observation.
        If False, each column is a variable and each row is an observation.
    bias : bool, default=False
        If False, normalization is by (N-1), giving unbiased estimator.
        If True, normalization is by N, giving the maximum likelihood estimator.
    
    Returns
    -------
    np.ndarray
        Covariance matrix
    
    Notes
    -----
    This function is optimized with Numba for performance.
    """
    # Validate input dimensions
    if x.ndim > 2:
        raise ValueError(f"Input array must be 1D or 2D, got {x.ndim}D")
    
    # Reshape 1D array to 2D
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    
    # Transpose if rowvar is True (default assumes observations in rows)
    if rowvar:
        x = x.T
    
    # Get number of observations and variables
    n_obs, n_vars = x.shape
    
    # Check for sufficient observations
    if n_obs <= 1:
        raise ValueError(f"Need at least 2 observations, got {n_obs}")
    
    # Center the data by subtracting the mean
    means = np.zeros(n_vars)
    for i in range(n_vars):
        means[i] = np.mean(x[:, i])
    
    # Create centered data
    x_centered = np.empty_like(x)
    for i in range(n_obs):
        for j in range(n_vars):
            x_centered[i, j] = x[i, j] - means[j]
    
    # Calculate covariance using optimized matrix operations
    cov = np.zeros((n_vars, n_vars))
    for i in range(n_vars):
        for j in range(n_vars):
            for k in range(n_obs):
                cov[i, j] += x_centered[k, i] * x_centered[k, j]
    
    # Apply bias correction if specified
    if bias:
        # Maximum likelihood estimator (divide by n)
        cov /= n_obs
    else:
        # Unbiased estimator (divide by n-1)
        cov /= (n_obs - 1)
    
    return cov


@numba.jit(nopython=True)
def fast_corr(
    x: np.ndarray,
    rowvar: bool = False
) -> np.ndarray:
    """
    Calculates correlation matrices with optimized performance.
    
    Parameters
    ----------
    x : np.ndarray
        Input array of observations. If rowvar is True, each row represents a
        variable and each column represents an observation. Otherwise, the
        relationship is transposed.
    rowvar : bool, default=False
        If True, each row is a variable and each column is an observation.
        If False, each column is a variable and each row is an observation.
    
    Returns
    -------
    np.ndarray
        Correlation matrix
    
    Notes
    -----
    This function is optimized with Numba for performance.
    """
    # Calculate covariance matrix using fast_cov
    cov = fast_cov(x, rowvar=rowvar, bias=False)
    
    # Get number of variables
    n_vars = cov.shape[0]
    
    # Extract diagonal elements as standard deviations
    std_devs = np.zeros(n_vars)
    for i in range(n_vars):
        std_devs[i] = np.sqrt(cov[i, i])
    
    # Normalize covariance by outer product of standard deviations
    corr = np.zeros_like(cov)
    for i in range(n_vars):
        for j in range(n_vars):
            if std_devs[i] > 0 and std_devs[j] > 0:
                corr[i, j] = cov[i, j] / (std_devs[i] * std_devs[j])
            else:
                corr[i, j] = 0.0
    
    return corr


def ensure_finite(
    arr: np.ndarray,
    fill_value: Optional[float] = None,
    raise_error: bool = True
) -> np.ndarray:
    """
    Checks for and handles NaN and Inf values in arrays.
    
    Parameters
    ----------
    arr : np.ndarray
        Input array to be checked
    fill_value : Optional[float], default=None
        Value to use for replacing NaN and Inf values. If None and raise_error
        is False, NaN and Inf values remain unchanged.
    raise_error : bool, default=True
        If True, raises ValueError if NaN or Inf values are found.
        If False, replaces NaN and Inf values with fill_value if provided.
    
    Returns
    -------
    np.ndarray
        Array with NaN and Inf values handled according to parameters
    
    Raises
    ------
    ValueError
        If raise_error is True and NaN or Inf values are found
    """
    # Check if array contains NaN or Inf values
    if not np.isfinite(arr).all():
        # If raise_error is True, raise ValueError on finding NaN/Inf
        if raise_error:
            nan_count = np.isnan(arr).sum()
            inf_count = np.isinf(arr).sum()
            
            if nan_count > 0 and inf_count > 0:
                raise ValueError(
                    f"Array contains {nan_count} NaN and {inf_count} Inf values"
                )
            elif nan_count > 0:
                raise ValueError(f"Array contains {nan_count} NaN values")
            else:
                raise ValueError(f"Array contains {inf_count} Inf values")
        
        # If fill_value is provided, replace NaN/Inf with fill_value
        if fill_value is not None:
            result = arr.copy()
            mask = ~np.isfinite(result)
            result[mask] = fill_value
            return result
    
    # Return the original array if no NaN/Inf or if no action was taken
    return arr


@numba.jit(nopython=True)
def lag_matrix(
    y: np.ndarray,
    lags: int
) -> np.ndarray:
    """
    Creates a matrix of lagged values for time series analysis.
    
    Parameters
    ----------
    y : np.ndarray
        The time series data (1D or 2D array)
    lags : int
        Number of lags to include
    
    Returns
    -------
    np.ndarray
        Matrix containing original and lagged series
    
    Notes
    -----
    This function is optimized with Numba for performance.
    
    For a univariate time series, the output is a matrix with rows equal to 
    (length - lags) and columns equal to (lags + 1), where each column
    corresponds to a lagged version of the original series.
    
    For a multivariate time series, the output includes lags for each variable.
    """
    # Validate input array is 1D or 2D
    if y.ndim > 2:
        raise ValueError("Input array must be 1D or 2D")
    
    # Convert 1D array to 2D column vector if needed
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    
    # Get dimensions
    T, K = y.shape  # T: time periods, K: number of variables
    
    # Check if we have enough observations
    if T <= lags:
        raise ValueError(
            f"Number of observations ({T}) must be greater than lags ({lags})"
        )
    
    # Create output array with appropriate dimensions
    # Output will have (T - lags) rows and K * (lags + 1) columns
    result = np.zeros((T - lags, K * (lags + 1)))
    
    # Fill the output array with lagged values
    for t in range(lags, T):
        # Current observation (t) for all variables
        for k in range(K):
            result[t - lags, k] = y[t, k]
        
        # Add lagged observations
        for lag in range(1, lags + 1):
            for k in range(K):
                # Each lag of each variable gets its own column
                result[t - lags, K * lag + k] = y[t - lag, k]
    
    return result


def block_diagonal(
    arrays: List[np.ndarray]
) -> np.ndarray:
    """
    Creates a block diagonal matrix from a sequence of arrays.
    
    Parameters
    ----------
    arrays : List[np.ndarray]
        List of 2D arrays to arrange as blocks on the diagonal
    
    Returns
    -------
    np.ndarray
        Block diagonal matrix
    
    Notes
    -----
    Each input array must be 2D. The resulting matrix will have
    blocks of the input arrays on the main diagonal, with zeros elsewhere.
    
    Examples
    --------
    >>> a = np.array([[1, 2], [3, 4]])
    >>> b = np.array([[5, 6], [7, 8]])
    >>> block_diagonal([a, b])
    array([[1, 2, 0, 0],
           [3, 4, 0, 0],
           [0, 0, 5, 6],
           [0, 0, 7, 8]])
    """
    # Calculate dimensions of output matrix
    if not arrays:
        return np.array([])
    
    # Check that all arrays are 2D
    for i, arr in enumerate(arrays):
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"Input {i} is not a NumPy array")
        if arr.ndim != 2:
            raise ValueError(f"Input array {i} must be 2D, got {arr.ndim}D")
    
    # Get shapes of all arrays
    shapes = np.array([arr.shape for arr in arrays])
    
    # Calculate output shape
    out_shape = np.sum(shapes, axis=0)
    
    # Create output matrix filled with zeros
    result = np.zeros(out_shape)
    
    # Insert each input array along the diagonal
    row_start = 0
    col_start = 0
    
    for arr in arrays:
        rows, cols = arr.shape
        result[row_start:row_start+rows, col_start:col_start+cols] = arr
        row_start += rows
        col_start += cols
    
    return result


@numba.jit(nopython=True)
def outer_product(
    a: np.ndarray,
    b: np.ndarray
) -> np.ndarray:
    """
    Computes the outer product of two vectors with Numba optimization.
    
    Parameters
    ----------
    a : np.ndarray
        First vector (1D array)
    b : np.ndarray
        Second vector (1D array)
    
    Returns
    -------
    np.ndarray
        Outer product matrix
    
    Notes
    -----
    This function is optimized with Numba for performance.
    The outer product of vectors a and b is a matrix M where M_{ij} = a_i * b_j.
    """
    # Validate input arrays are 1D vectors
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("Input arrays must be 1D vectors")
    
    # Get dimensions
    m = a.shape[0]
    n = b.shape[0]
    
    # Create output matrix
    result = np.zeros((m, n))
    
    # Compute outer product using optimized multiplication
    for i in range(m):
        for j in range(n):
            result[i, j] = a[i] * b[j]
    
    return result


@numba.jit(nopython=True)
def efficient_diff(
    arr: np.ndarray,
    n: int = 1,
    axis: int = 0
) -> np.ndarray:
    """
    Efficiently calculates the differences between consecutive elements with optimized performance.
    
    Parameters
    ----------
    arr : np.ndarray
        Input array
    n : int, default=1
        Number of times to difference
    axis : int, default=0
        Axis along which to difference
    
    Returns
    -------
    np.ndarray
        Array of differences
    
    Notes
    -----
    This function is optimized with Numba for performance.
    It computes the n-th difference along the specified axis.
    
    Examples
    --------
    >>> x = np.array([1, 3, 6, 10, 15])
    >>> efficient_diff(x)
    array([2, 3, 4, 5])
    >>> efficient_diff(x, n=2)
    array([1, 1, 1])
    """
    # Validate input parameters
    if n < 1:
        raise ValueError(f"Number of differences n must be >= 1, got {n}")
    
    if axis < 0:
        axis = arr.ndim + axis
    
    if axis >= arr.ndim or axis < 0:
        raise ValueError(f"Axis {axis} is out of bounds for array of dimension {arr.ndim}")
    
    # Handle simple cases quickly
    if n == 1:
        # Handle 1D arrays more efficiently
        if arr.ndim == 1:
            result = np.empty(arr.shape[0] - 1)
            for i in range(arr.shape[0] - 1):
                result[i] = arr[i + 1] - arr[i]
            return result
        
        # Handle N-dimensional arrays
        result_shape = list(arr.shape)
        result_shape[axis] -= 1
        result = np.empty(result_shape)
        
        # Single pass differencing
        if axis == 0:
            for i in range(1, arr.shape[0]):
                for j in range(arr.shape[1]):
                    result[i-1, j] = arr[i, j] - arr[i-1, j]
        else:  # axis == 1
            for i in range(arr.shape[0]):
                for j in range(1, arr.shape[1]):
                    result[i, j-1] = arr[i, j] - arr[i, j-1]
        
        return result
    
    # For multiple differences, apply recursively
    return efficient_diff(efficient_diff(arr, 1, axis), n-1, axis)