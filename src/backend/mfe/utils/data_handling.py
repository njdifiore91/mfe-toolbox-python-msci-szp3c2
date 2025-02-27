"""
MFE Toolbox - Data Handling Module

This module provides comprehensive utilities for data manipulation, transformation, 
and preprocessing specifically for financial time series analysis in the MFE Toolbox.
It implements functions for efficient data conversion between various formats, time 
series manipulation, and specialized financial data handling with robust error 
handling and type safety.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import numpy as np  # numpy 1.26.3
import pandas as pd  # pandas 2.1.4
import scipy.signal  # scipy 1.11.4
import asyncio  # Python 3.12

from .validation import validate_array, check_time_series, is_float_array
from .numpy_helpers import ensure_array, ensure_finite
from .pandas_helpers import convert_to_dataframe, convert_to_series, ensure_datetime_index
from .numba_helpers import optimized_jit

# Setup module logger
logger = logging.getLogger(__name__)

# List of exported functions
__all__ = [
    'convert_time_series', 'convert_frequency', 'load_financial_data', 
    'save_financial_data', 'filter_time_series', 'detect_outliers', 
    'handle_missing_values', 'calculate_financial_returns', 'adjust_for_corporate_actions', 
    'split_train_test', 'convert_to_log_returns', 'convert_to_simple_returns', 
    'standardize_data', 'handle_date_range', 'resample_high_frequency', 
    'align_multiple_series', 'merge_time_series'
]


def convert_time_series(
    data: Any,
    output_type: str,
    time_index: Optional[Any] = None,
    ensure_finite: bool = True
) -> Union[np.ndarray, pd.Series, pd.DataFrame]:
    """
    Converts time series data between different formats (NumPy array, Pandas Series, 
    Pandas DataFrame) with appropriate handling of time indices.
    
    Parameters
    ----------
    data : Any
        The input data to convert (array, list, Series, DataFrame, etc.)
    output_type : str
        Target format: 'numpy', 'series', or 'dataframe'
    time_index : Optional[Any], default=None
        Time index to use for pandas objects. Can be a list of dates, 
        DatetimeIndex, or string column name.
    ensure_finite : bool, default=True
        If True, checks for and potentially raises on NaN and Inf values
        
    Returns
    -------
    Union[np.ndarray, pd.Series, pd.DataFrame]
        Converted time series data in the specified format
        
    Raises
    ------
    ValueError
        If data cannot be converted to the specified format
    TypeError
        If output_type is not one of the supported types
    """
    # Validate input data
    check_time_series(data, allow_2d=True, allow_missing=not ensure_finite)
    
    # Check output_type
    if output_type not in ['numpy', 'series', 'dataframe']:
        raise TypeError(f"Invalid output_type: {output_type}. Must be 'numpy', 'series', or 'dataframe'")
    
    # Convert to numpy array
    if output_type == 'numpy':
        result = ensure_array(data, allow_none=False)
        if ensure_finite:
            result = ensure_finite(result, raise_error=True)
        return result
    
    # Convert to pandas Series
    elif output_type == 'series':
        result = convert_to_series(data, index=time_index)
        if ensure_finite:
            if not result.isna().any() and not np.isinf(result).any():
                return result
            else:
                raise ValueError("Series contains NaN or Inf values")
        return result
    
    # Convert to pandas DataFrame
    elif output_type == 'dataframe':
        result = convert_to_dataframe(data, index=time_index)
        if ensure_finite:
            if not result.isna().any().any() and not np.isinf(result).any().any():
                return result
            else:
                raise ValueError("DataFrame contains NaN or Inf values")
        return result


def convert_frequency(
    data: Any,
    freq: str,
    method: str = 'mean',
    include_start: Optional[bool] = True,
    include_end: Optional[bool] = True
) -> Union[pd.Series, pd.DataFrame]:
    """
    Resamples time series data to a different frequency with support for 
    various aggregation methods.
    
    Parameters
    ----------
    data : Any
        Time series data to resample (will be converted to pandas object)
    freq : str
        Frequency string in pandas format (e.g., 'D' for daily, 'M' for monthly)
    method : str, default='mean'
        Aggregation method: 'mean', 'sum', 'first', 'last', 'min', 'max', 'median', 'ohlc'
    include_start : bool, default=True
        Whether to include the start of the period in resampling
    include_end : bool, default=True
        Whether to include the end of the period in resampling
        
    Returns
    -------
    Union[pd.Series, pd.DataFrame]
        Resampled time series data
        
    Raises
    ------
    ValueError
        If the resampling method is not supported or data cannot be resampled
    """
    # Convert input to pandas object (Series or DataFrame)
    if isinstance(data, np.ndarray) or not isinstance(data, (pd.Series, pd.DataFrame)):
        if data.ndim == 1 or (isinstance(data, np.ndarray) and data.ndim == 1):
            # Convert to Series for 1D data
            pandas_data = convert_to_series(data)
        else:
            # Convert to DataFrame for 2D+ data
            pandas_data = convert_to_dataframe(data)
    else:
        pandas_data = data
    
    # Ensure data has DatetimeIndex
    pandas_data = ensure_datetime_index(pandas_data)
    
    # Set up resampling boundaries based on inclusion parameters
    closed = None
    if include_start and include_end:
        closed = None  # pandas default behavior
    elif include_start and not include_end:
        closed = 'left'
    elif not include_start and include_end:
        closed = 'right'
    else:  # not include_start and not include_end
        # This is a special case, need to handle differently
        closed = 'neither'
    
    # Create resampler object
    resampler = pandas_data.resample(freq, closed=closed)
    
    # Apply aggregation method
    if method == 'mean':
        result = resampler.mean()
    elif method == 'sum':
        result = resampler.sum()
    elif method == 'first':
        result = resampler.first()
    elif method == 'last':
        result = resampler.last()
    elif method == 'min':
        result = resampler.min()
    elif method == 'max':
        result = resampler.max()
    elif method == 'median':
        result = resampler.median()
    elif method == 'ohlc':
        result = resampler.ohlc()
    else:
        raise ValueError(f"Unsupported aggregation method: {method}")
    
    return result


def load_financial_data(
    file_path: str,
    file_format: Optional[str] = None,
    date_column: Optional[str] = None,
    **kwargs
) -> Union[pd.DataFrame, pd.Series]:
    """
    Loads financial time series data from various file formats with proper handling
    of date columns and data types.
    
    Parameters
    ----------
    file_path : str
        Path to the file to load
    file_format : Optional[str], default=None
        Format of the file: 'csv', 'excel', 'parquet', 'hdf', 'json', 'pickle'.
        If None, inferred from file extension.
    date_column : Optional[str], default=None
        Name of the column to convert to DatetimeIndex
    **kwargs : dict
        Additional keyword arguments for the specific pandas reader function
        
    Returns
    -------
    Union[pd.DataFrame, pd.Series]
        Loaded financial data as pandas DataFrame or Series
        
    Raises
    ------
    ValueError
        If file format is not supported or file cannot be loaded
    """
    # Determine file format if not specified
    if file_format is None:
        file_extension = file_path.split('.')[-1].lower()
        if file_extension in ['csv', 'txt']:
            file_format = 'csv'
        elif file_extension in ['xls', 'xlsx', 'xlsm']:
            file_format = 'excel'
        elif file_extension == 'parquet':
            file_format = 'parquet'
        elif file_extension == 'hdf':
            file_format = 'hdf'
        elif file_extension == 'json':
            file_format = 'json'
        elif file_extension in ['pkl', 'pickle']:
            file_format = 'pickle'
        else:
            raise ValueError(f"Could not determine file format from extension: {file_extension}")
    
    try:
        # Load data using appropriate pandas reader
        if file_format == 'csv':
            data = pd.read_csv(file_path, **kwargs)
        elif file_format == 'excel':
            data = pd.read_excel(file_path, **kwargs)
        elif file_format == 'parquet':
            data = pd.read_parquet(file_path, **kwargs)
        elif file_format == 'hdf':
            key = kwargs.pop('key', None)
            if key is None:
                raise ValueError("'key' parameter required for HDF5 files")
            data = pd.read_hdf(file_path, key=key, **kwargs)
        elif file_format == 'json':
            data = pd.read_json(file_path, **kwargs)
        elif file_format == 'pickle':
            data = pd.read_pickle(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
        
        # Convert date column to DatetimeIndex if specified
        if date_column is not None:
            if date_column in data.columns:
                data[date_column] = pd.to_datetime(data[date_column])
                data = data.set_index(date_column)
            else:
                logger.warning(f"Date column '{date_column}' not found in the data")
        
        # If data has a single column and index, convert to Series
        if isinstance(data, pd.DataFrame) and data.shape[1] == 1:
            return data.iloc[:, 0]
        
        return data
    
    except Exception as e:
        logger.error(f"Error loading financial data from {file_path}: {str(e)}")
        raise ValueError(f"Failed to load data from {file_path}: {str(e)}")


def save_financial_data(
    data: Union[pd.DataFrame, pd.Series],
    file_path: str,
    file_format: Optional[str] = None,
    **kwargs
) -> bool:
    """
    Saves financial time series data to various file formats with proper handling
    of date formatting.
    
    Parameters
    ----------
    data : Union[pd.DataFrame, pd.Series]
        Financial data to save
    file_path : str
        Path where the file will be saved
    file_format : Optional[str], default=None
        Format to save the file: 'csv', 'excel', 'parquet', 'hdf', 'json', 'pickle'.
        If None, inferred from file extension.
    **kwargs : dict
        Additional keyword arguments for the specific pandas writer function
        
    Returns
    -------
    bool
        True if save operation was successful
        
    Raises
    ------
    ValueError
        If file format is not supported or data cannot be saved
    TypeError
        If data is not a pandas DataFrame or Series
    """
    # Validate input data
    if not isinstance(data, (pd.DataFrame, pd.Series)):
        raise TypeError("Input data must be a pandas DataFrame or Series")
    
    # Determine file format if not specified
    if file_format is None:
        file_extension = file_path.split('.')[-1].lower()
        if file_extension in ['csv', 'txt']:
            file_format = 'csv'
        elif file_extension in ['xls', 'xlsx', 'xlsm']:
            file_format = 'excel'
        elif file_extension == 'parquet':
            file_format = 'parquet'
        elif file_extension == 'hdf':
            file_format = 'hdf'
        elif file_extension == 'json':
            file_format = 'json'
        elif file_extension in ['pkl', 'pickle']:
            file_format = 'pickle'
        else:
            raise ValueError(f"Could not determine file format from extension: {file_extension}")
    
    try:
        # Save data using appropriate pandas writer
        if file_format == 'csv':
            data.to_csv(file_path, **kwargs)
        elif file_format == 'excel':
            data.to_excel(file_path, **kwargs)
        elif file_format == 'parquet':
            data.to_parquet(file_path, **kwargs)
        elif file_format == 'hdf':
            key = kwargs.pop('key', 'data')
            data.to_hdf(file_path, key=key, **kwargs)
        elif file_format == 'json':
            data.to_json(file_path, **kwargs)
        elif file_format == 'pickle':
            data.to_pickle(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error saving financial data to {file_path}: {str(e)}")
        raise ValueError(f"Failed to save data to {file_path}: {str(e)}")


@optimized_jit
def filter_time_series(
    data: Any,
    filter_type: str,
    filter_params: Optional[dict] = None
) -> Union[np.ndarray, pd.Series, pd.DataFrame]:
    """
    Applies various filtering techniques to time series data to remove noise
    or extract specific components.
    
    Parameters
    ----------
    data : Any
        Time series data to filter (array, Series, DataFrame)
    filter_type : str
        Type of filter to apply: 'moving_average', 'ewma', 'butterworth',
        'hodrick_prescott', 'wavelet'
    filter_params : Optional[dict], default=None
        Parameters specific to the chosen filter:
        - 'moving_average': {'window': int, 'center': bool}
        - 'ewma': {'span': int, 'alpha': float}
        - 'butterworth': {'N': int, 'Wn': float, 'btype': str}
        - 'hodrick_prescott': {'lambda': float}
        - 'wavelet': {'wavelet': str, 'level': int}
        
    Returns
    -------
    Union[np.ndarray, pd.Series, pd.DataFrame]
        Filtered time series data in the same format as input
        
    Raises
    ------
    ValueError
        If filter_type is not supported or filter parameters are invalid
    """
    # Set default filter parameters if not provided
    if filter_params is None:
        filter_params = {}
    
    # Determine input type to preserve it for output
    is_pandas_series = isinstance(data, pd.Series)
    is_pandas_dataframe = isinstance(data, pd.DataFrame)
    original_index = None
    original_columns = None
    
    if is_pandas_series:
        original_index = data.index
    elif is_pandas_dataframe:
        original_index = data.index
        original_columns = data.columns
    
    # Convert to numpy array for filtering
    array_data = convert_time_series(data, output_type='numpy')
    
    # Apply the specified filter
    if filter_type == 'moving_average':
        window = filter_params.get('window', 5)
        center = filter_params.get('center', True)
        
        if array_data.ndim == 1:
            # 1D array - simple 1D moving average
            weights = np.ones(window) / window
            filtered_data = np.convolve(array_data, weights, mode='same')
        else:
            # 2D array - apply moving average to each column
            filtered_data = np.zeros_like(array_data)
            weights = np.ones(window) / window
            for col in range(array_data.shape[1]):
                filtered_data[:, col] = np.convolve(array_data[:, col], weights, mode='same')
        
        # Adjust edges if not centered
        if not center:
            if array_data.ndim == 1:
                filtered_data[:window//2] = array_data[:window//2]
                filtered_data[-(window//2):] = array_data[-(window//2):]
            else:
                for col in range(array_data.shape[1]):
                    filtered_data[:window//2, col] = array_data[:window//2, col]
                    filtered_data[-(window//2):, col] = array_data[-(window//2):, col]
    
    elif filter_type == 'ewma':
        span = filter_params.get('span', 5)
        alpha = filter_params.get('alpha', None)
        
        # Convert back to pandas for EWMA
        if is_pandas_series:
            filtered_data = pd.Series(array_data, index=original_index).ewm(
                span=span, alpha=alpha, adjust=True
            ).mean().values
        elif is_pandas_dataframe:
            temp_df = pd.DataFrame(array_data, index=original_index, columns=original_columns)
            filtered_data = temp_df.ewm(span=span, alpha=alpha, adjust=True).mean().values
        else:
            # For numpy arrays, need to implement EWMA manually
            if alpha is None:
                alpha = 2.0 / (span + 1.0)
            
            filtered_data = np.zeros_like(array_data)
            if array_data.ndim == 1:
                filtered_data[0] = array_data[0]
                for i in range(1, len(array_data)):
                    filtered_data[i] = alpha * array_data[i] + (1 - alpha) * filtered_data[i-1]
            else:
                filtered_data[0, :] = array_data[0, :]
                for i in range(1, array_data.shape[0]):
                    filtered_data[i, :] = alpha * array_data[i, :] + (1 - alpha) * filtered_data[i-1, :]
    
    elif filter_type == 'butterworth':
        N = filter_params.get('N', 2)
        Wn = filter_params.get('Wn', 0.1)
        btype = filter_params.get('btype', 'lowpass')
        
        # Create Butterworth filter
        b, a = scipy.signal.butter(N, Wn, btype=btype)
        
        # Apply filter
        if array_data.ndim == 1:
            filtered_data = scipy.signal.filtfilt(b, a, array_data)
        else:
            filtered_data = np.zeros_like(array_data)
            for col in range(array_data.shape[1]):
                filtered_data[:, col] = scipy.signal.filtfilt(b, a, array_data[:, col])
    
    elif filter_type == 'hodrick_prescott':
        lambda_param = filter_params.get('lambda', 1600)
        
        if array_data.ndim == 1:
            cycle, trend = scipy.signal.hpfilter(array_data, lambda_param)
            filtered_data = trend
        else:
            filtered_data = np.zeros_like(array_data)
            for col in range(array_data.shape[1]):
                cycle, trend = scipy.signal.hpfilter(array_data[:, col], lambda_param)
                filtered_data[:, col] = trend
    
    elif filter_type == 'wavelet':
        try:
            import pywt
        except ImportError:
            raise ImportError("Wavelet filtering requires the PyWavelets (pywt) package. Install it with 'pip install PyWavelets'.")
        
        wavelet = filter_params.get('wavelet', 'db4')
        level = filter_params.get('level', 3)
        
        if array_data.ndim == 1:
            # Perform wavelet decomposition
            coeffs = pywt.wavedec(array_data, wavelet, level=level)
            
            # Modify coefficients (e.g., remove detail coefficients for denoising)
            for i in range(1, len(coeffs)):
                coeffs[i] = np.zeros_like(coeffs[i])
            
            # Reconstruct signal
            filtered_data = pywt.waverec(coeffs, wavelet)
            
            # Ensure same length as input (waverec might return slightly different length)
            if len(filtered_data) > len(array_data):
                filtered_data = filtered_data[:len(array_data)]
            elif len(filtered_data) < len(array_data):
                padding = np.zeros(len(array_data) - len(filtered_data))
                filtered_data = np.concatenate([filtered_data, padding])
        else:
            filtered_data = np.zeros_like(array_data)
            for col in range(array_data.shape[1]):
                # Perform wavelet decomposition
                coeffs = pywt.wavedec(array_data[:, col], wavelet, level=level)
                
                # Modify coefficients
                for i in range(1, len(coeffs)):
                    coeffs[i] = np.zeros_like(coeffs[i])
                
                # Reconstruct signal
                col_filtered = pywt.waverec(coeffs, wavelet)
                
                # Ensure same length as input
                if len(col_filtered) > array_data.shape[0]:
                    col_filtered = col_filtered[:array_data.shape[0]]
                elif len(col_filtered) < array_data.shape[0]:
                    padding = np.zeros(array_data.shape[0] - len(col_filtered))
                    col_filtered = np.concatenate([col_filtered, padding])
                
                filtered_data[:, col] = col_filtered
    
    else:
        raise ValueError(f"Unsupported filter_type: {filter_type}")
    
    # Convert back to original format
    if is_pandas_series:
        return pd.Series(filtered_data, index=original_index)
    elif is_pandas_dataframe:
        return pd.DataFrame(filtered_data, index=original_index, columns=original_columns)
    else:
        return filtered_data


def detect_outliers(
    data: Any,
    method: str = 'z_score',
    threshold: float = 3.0,
    return_mask: bool = False
) -> Union[np.ndarray, pd.Series, pd.DataFrame, Tuple]:
    """
    Detects outliers in financial time series data using various statistical methods.
    
    Parameters
    ----------
    data : Any
        Time series data to check for outliers (array, Series, DataFrame)
    method : str, default='z_score'
        Method to use for outlier detection:
        - 'z_score': Standard deviations from mean
        - 'iqr': Interquartile range
        - 'modified_z': Modified Z-score with median absolute deviation
        - 'isolation_forest': Isolation Forest algorithm (requires scikit-learn)
    threshold : float, default=3.0
        Threshold for outlier detection (interpretation depends on method)
    return_mask : bool, default=False
        If True, returns a tuple (cleaned_data, outlier_mask), otherwise just cleaned_data
        
    Returns
    -------
    Union[np.ndarray, pd.Series, pd.DataFrame, Tuple]
        Cleaned data with outliers removed or replaced, or tuple with data and outlier mask
        
    Raises
    ------
    ValueError
        If method is not supported or data is invalid
    """
    # Determine input type to preserve it for output
    is_pandas_series = isinstance(data, pd.Series)
    is_pandas_dataframe = isinstance(data, pd.DataFrame)
    original_index = None
    original_columns = None
    
    if is_pandas_series:
        original_index = data.index
    elif is_pandas_dataframe:
        original_index = data.index
        original_columns = data.columns
    
    # Convert to numpy array for processing
    array_data = convert_time_series(data, output_type='numpy')
    
    # Initialize outlier mask
    if array_data.ndim == 1:
        outlier_mask = np.zeros(array_data.shape, dtype=bool)
    else:
        outlier_mask = np.zeros(array_data.shape, dtype=bool)
    
    # Apply the specified outlier detection method
    if method == 'z_score':
        if array_data.ndim == 1:
            # 1D array
            mean = np.nanmean(array_data)
            std = np.nanstd(array_data)
            outlier_mask = np.abs(array_data - mean) > threshold * std
        else:
            # 2D array - apply to each column
            for col in range(array_data.shape[1]):
                col_data = array_data[:, col]
                mean = np.nanmean(col_data)
                std = np.nanstd(col_data)
                outlier_mask[:, col] = np.abs(col_data - mean) > threshold * std
    
    elif method == 'iqr':
        if array_data.ndim == 1:
            # 1D array
            q1 = np.nanpercentile(array_data, 25)
            q3 = np.nanpercentile(array_data, 75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            outlier_mask = (array_data < lower_bound) | (array_data > upper_bound)
        else:
            # 2D array - apply to each column
            for col in range(array_data.shape[1]):
                col_data = array_data[:, col]
                q1 = np.nanpercentile(col_data, 25)
                q3 = np.nanpercentile(col_data, 75)
                iqr = q3 - q1
                lower_bound = q1 - threshold * iqr
                upper_bound = q3 + threshold * iqr
                outlier_mask[:, col] = (col_data < lower_bound) | (col_data > upper_bound)
    
    elif method == 'modified_z':
        if array_data.ndim == 1:
            # 1D array
            median = np.nanmedian(array_data)
            # MAD: median absolute deviation
            mad = np.nanmedian(np.abs(array_data - median))
            # Modified Z-score (robust to outliers)
            modified_z = 0.6745 * np.abs(array_data - median) / mad if mad > 0 else np.zeros_like(array_data)
            outlier_mask = modified_z > threshold
        else:
            # 2D array - apply to each column
            for col in range(array_data.shape[1]):
                col_data = array_data[:, col]
                median = np.nanmedian(col_data)
                mad = np.nanmedian(np.abs(col_data - median))
                modified_z = 0.6745 * np.abs(col_data - median) / mad if mad > 0 else np.zeros_like(col_data)
                outlier_mask[:, col] = modified_z > threshold
    
    elif method == 'isolation_forest':
        try:
            from sklearn.ensemble import IsolationForest
        except ImportError:
            raise ImportError("Isolation Forest method requires scikit-learn. Install it with 'pip install scikit-learn'.")
        
        # Prepare data for Isolation Forest
        if array_data.ndim == 1:
            X = array_data.reshape(-1, 1)
        else:
            X = array_data
        
        # Handle NaN values
        nan_mask = np.isnan(X).any(axis=1) if X.ndim > 1 else np.isnan(X)
        X_clean = X[~nan_mask] if X.ndim > 1 else X[~nan_mask].reshape(-1, 1)
        
        # Fit Isolation Forest
        clf = IsolationForest(contamination=min(0.1, float(threshold) / 10), random_state=42)
        clf.fit(X_clean)
        
        # Predict outliers
        y_pred = clf.predict(X_clean)
        is_outlier = y_pred == -1  # -1 for outliers, 1 for inliers
        
        # Update outlier mask, preserving NaN locations
        if array_data.ndim == 1:
            outlier_mask[~nan_mask] = is_outlier
            outlier_mask[nan_mask] = True  # Consider NaNs as outliers
        else:
            temp_mask = np.zeros(array_data.shape[0], dtype=bool)
            temp_mask[~nan_mask] = is_outlier
            temp_mask[nan_mask] = True  # Consider NaNs as outliers
            
            for col in range(array_data.shape[1]):
                outlier_mask[:, col] = temp_mask
    
    else:
        raise ValueError(f"Unsupported outlier detection method: {method}")
    
    # Create cleaned data (with outliers removed/replaced)
    cleaned_data = array_data.copy()
    if array_data.ndim == 1:
        cleaned_data[outlier_mask] = np.nan
    else:
        for col in range(array_data.shape[1]):
            cleaned_data[:, col][outlier_mask[:, col]] = np.nan
    
    # Convert back to original format
    if is_pandas_series:
        cleaned_series = pd.Series(cleaned_data, index=original_index)
        if return_mask:
            mask_series = pd.Series(outlier_mask, index=original_index)
            return cleaned_series, mask_series
        return cleaned_series
    elif is_pandas_dataframe:
        cleaned_df = pd.DataFrame(cleaned_data, index=original_index, columns=original_columns)
        if return_mask:
            mask_df = pd.DataFrame(outlier_mask, index=original_index, columns=original_columns)
            return cleaned_df, mask_df
        return cleaned_df
    else:
        if return_mask:
            return cleaned_data, outlier_mask
        return cleaned_data


def handle_missing_values(
    data: Any,
    method: str = 'ffill',
    method_params: Optional[dict] = None
) -> Union[np.ndarray, pd.Series, pd.DataFrame]:
    """
    Handles missing values in time series data through various imputation or removal methods.
    
    Parameters
    ----------
    data : Any
        Time series data containing missing values (array, Series, DataFrame)
    method : str, default='ffill'
        Method for handling missing values:
        - 'drop': Remove rows with missing values
        - 'ffill': Forward fill (propagate last valid observation forward)
        - 'bfill': Backward fill (use next valid observation to fill gap)
        - 'linear': Linear interpolation
        - 'cubic': Cubic spline interpolation
        - 'mean': Replace with mean value
        - 'median': Replace with median value
        - 'mode': Replace with mode (most frequent) value
        - 'knn': k-nearest neighbors imputation (requires scikit-learn)
    method_params : Optional[dict], default=None
        Additional parameters for the selected method:
        - For 'knn': {'n_neighbors': int, 'weights': str}
        - For 'linear'/'cubic': {'limit': int}
        
    Returns
    -------
    Union[np.ndarray, pd.Series, pd.DataFrame]
        Time series with missing values handled
        
    Raises
    ------
    ValueError
        If method is not supported or missing values cannot be handled
    """
    # Set default method parameters if not provided
    if method_params is None:
        method_params = {}
    
    # Determine input type to preserve it for output
    is_pandas_series = isinstance(data, pd.Series)
    is_pandas_dataframe = isinstance(data, pd.DataFrame)
    is_numpy_array = isinstance(data, np.ndarray)
    
    # For numpy arrays, try to detect missing values
    if is_numpy_array:
        # Convert to pandas for easier handling of missing values
        if data.ndim == 1:
            temp_data = pd.Series(data)
        else:
            temp_data = pd.DataFrame(data)
        data = temp_data
        is_pandas_series = data.ndim == 1
        is_pandas_dataframe = data.ndim > 1
    
    # Apply the specified missing value handling method
    if method == 'drop':
        if is_pandas_series or is_pandas_dataframe:
            result = data.dropna().copy()
        else:
            # This should not happen as we converted numpy arrays above
            mask = ~np.isnan(data).any(axis=1) if data.ndim > 1 else ~np.isnan(data)
            result = data[mask].copy()
    
    elif method == 'ffill':
        if is_pandas_series or is_pandas_dataframe:
            limit = method_params.get('limit', None)
            result = data.ffill(limit=limit).copy()
        else:
            # This should not happen as we converted numpy arrays above
            raise ValueError("Unexpected data type for ffill method")
    
    elif method == 'bfill':
        if is_pandas_series or is_pandas_dataframe:
            limit = method_params.get('limit', None)
            result = data.bfill(limit=limit).copy()
        else:
            # This should not happen as we converted numpy arrays above
            raise ValueError("Unexpected data type for bfill method")
    
    elif method == 'linear':
        if is_pandas_series or is_pandas_dataframe:
            limit = method_params.get('limit', None)
            result = data.interpolate(method='linear', limit=limit).copy()
        else:
            # This should not happen as we converted numpy arrays above
            raise ValueError("Unexpected data type for linear interpolation method")
    
    elif method == 'cubic':
        if is_pandas_series or is_pandas_dataframe:
            limit = method_params.get('limit', None)
            result = data.interpolate(method='cubic', limit=limit).copy()
        else:
            # This should not happen as we converted numpy arrays above
            raise ValueError("Unexpected data type for cubic interpolation method")
    
    elif method == 'mean':
        if is_pandas_series:
            mean_value = data.mean()
            result = data.fillna(mean_value).copy()
        elif is_pandas_dataframe:
            # Fill with column means
            result = data.fillna(data.mean()).copy()
        else:
            # This should not happen as we converted numpy arrays above
            raise ValueError("Unexpected data type for mean imputation method")
    
    elif method == 'median':
        if is_pandas_series:
            median_value = data.median()
            result = data.fillna(median_value).copy()
        elif is_pandas_dataframe:
            # Fill with column medians
            result = data.fillna(data.median()).copy()
        else:
            # This should not happen as we converted numpy arrays above
            raise ValueError("Unexpected data type for median imputation method")
    
    elif method == 'mode':
        if is_pandas_series:
            mode_value = data.mode().iloc[0] if not data.mode().empty else np.nan
            result = data.fillna(mode_value).copy()
        elif is_pandas_dataframe:
            # Fill with column modes (most frequent values)
            modes = data.mode().iloc[0] if not data.mode().empty else pd.Series(np.nan, index=data.columns)
            result = data.fillna(modes).copy()
        else:
            # This should not happen as we converted numpy arrays above
            raise ValueError("Unexpected data type for mode imputation method")
    
    elif method == 'knn':
        try:
            from sklearn.impute import KNNImputer
        except ImportError:
            raise ImportError("KNN imputation requires scikit-learn. Install it with 'pip install scikit-learn'.")
        
        n_neighbors = method_params.get('n_neighbors', 5)
        weights = method_params.get('weights', 'uniform')
        
        # Create and fit KNN imputer
        imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights)
        
        if is_pandas_series:
            # For Series, reshape to 2D for KNNImputer
            values = imputer.fit_transform(data.values.reshape(-1, 1))
            result = pd.Series(values.ravel(), index=data.index)
        elif is_pandas_dataframe:
            # For DataFrame, apply directly
            values = imputer.fit_transform(data.values)
            result = pd.DataFrame(values, index=data.index, columns=data.columns)
        else:
            # This should not happen as we converted numpy arrays above
            raise ValueError("Unexpected data type for KNN imputation method")
    
    else:
        raise ValueError(f"Unsupported missing value handling method: {method}")
    
    # Convert back to numpy array if original was numpy
    if is_numpy_array:
        result = result.values
    
    return result


def calculate_financial_returns(
    prices: Any,
    method: str = 'simple',
    periods: int = 1,
    fill_na: bool = True
) -> Union[np.ndarray, pd.Series, pd.DataFrame]:
    """
    Calculates financial returns from price series with various methods such as 
    simple, log, or percentage returns.
    
    Parameters
    ----------
    prices : Any
        Price series data (array, Series, DataFrame)
    method : str, default='simple'
        Return calculation method:
        - 'simple': (p_t / p_{t-periods}) - 1
        - 'log': log(p_t / p_{t-periods})
        - 'percentage': (p_t - p_{t-periods}) / p_{t-periods} * 100
    periods : int, default=1
        Number of periods to use in return calculation
    fill_na : bool, default=True
        Whether to fill NaN values at the beginning of the series
        
    Returns
    -------
    Union[np.ndarray, pd.Series, pd.DataFrame]
        Calculated returns in same format as input
        
    Raises
    ------
    ValueError
        If method is not supported or prices contain invalid values
    """
    # Validate input prices
    if periods <= 0:
        raise ValueError("periods must be positive")
    
    # Determine input type to preserve it for output
    is_pandas_series = isinstance(prices, pd.Series)
    is_pandas_dataframe = isinstance(prices, pd.DataFrame)
    original_index = None
    original_columns = None
    
    if is_pandas_series:
        original_index = prices.index
    elif is_pandas_dataframe:
        original_index = prices.index
        original_columns = prices.columns
    
    # Convert to numpy array for calculations
    array_prices = convert_time_series(prices, output_type='numpy')
    
    # Validate that prices are positive
    if np.any(array_prices <= 0):
        raise ValueError("Price series contains non-positive values which cannot be used for return calculation")
    
    # Calculate returns based on specified method
    if method == 'simple':
        # Simple returns: (p_t / p_{t-periods}) - 1
        returns = np.empty_like(array_prices)
        returns[:periods] = np.nan  # First 'periods' elements will be NaN
        
        if array_prices.ndim == 1:
            # 1D array
            returns[periods:] = array_prices[periods:] / array_prices[:-periods] - 1
        else:
            # 2D array
            for col in range(array_prices.shape[1]):
                returns[periods:, col] = array_prices[periods:, col] / array_prices[:-periods, col] - 1
    
    elif method == 'log':
        # Log returns: log(p_t / p_{t-periods})
        returns = np.empty_like(array_prices)
        returns[:periods] = np.nan  # First 'periods' elements will be NaN
        
        if array_prices.ndim == 1:
            # 1D array
            returns[periods:] = np.log(array_prices[periods:] / array_prices[:-periods])
        else:
            # 2D array
            for col in range(array_prices.shape[1]):
                returns[periods:, col] = np.log(array_prices[periods:, col] / array_prices[:-periods, col])
    
    elif method == 'percentage':
        # Percentage returns: (p_t - p_{t-periods}) / p_{t-periods} * 100
        returns = np.empty_like(array_prices)
        returns[:periods] = np.nan  # First 'periods' elements will be NaN
        
        if array_prices.ndim == 1:
            # 1D array
            returns[periods:] = (array_prices[periods:] - array_prices[:-periods]) / array_prices[:-periods] * 100
        else:
            # 2D array
            for col in range(array_prices.shape[1]):
                returns[periods:, col] = (array_prices[periods:, col] - array_prices[:-periods, col]) / array_prices[:-periods, col] * 100
    
    else:
        raise ValueError(f"Unsupported return calculation method: {method}")
    
    # Handle NaN values at beginning if requested
    if fill_na:
        if array_prices.ndim == 1:
            returns[:periods] = 0
        else:
            returns[:periods, :] = 0
    
    # Convert back to original format
    if is_pandas_series:
        return pd.Series(returns, index=original_index)
    elif is_pandas_dataframe:
        return pd.DataFrame(returns, index=original_index, columns=original_columns)
    else:
        return returns


def adjust_for_corporate_actions(
    prices: Any,
    corporate_actions: pd.DataFrame,
    action_types: List[str]
) -> Union[np.ndarray, pd.Series, pd.DataFrame]:
    """
    Adjusts price series for corporate actions such as dividends, splits, and mergers.
    
    Parameters
    ----------
    prices : Any
        Price series data (array, Series, DataFrame)
    corporate_actions : pd.DataFrame
        DataFrame with corporate actions information. Must contain the following columns:
        - 'date': Date of the corporate action
        - 'action_type': Type of action ('dividend', 'split', 'merger', etc.)
        - 'value': Value relevant to the action (dividend amount, split ratio, etc.)
    action_types : List[str]
        Types of corporate actions to adjust for, e.g., ['dividend', 'split']
        
    Returns
    -------
    Union[np.ndarray, pd.Series, pd.DataFrame]
        Adjusted price series
        
    Raises
    ------
    ValueError
        If corporate_actions DataFrame is invalid or unsupported action types are specified
    """
    # Validate corporate_actions DataFrame
    required_columns = ['date', 'action_type', 'value']
    if not all(col in corporate_actions.columns for col in required_columns):
        raise ValueError(f"corporate_actions DataFrame must contain the following columns: {required_columns}")
    
    # Convert prices to pandas object for easier handling of dates
    if isinstance(prices, np.ndarray):
        if prices.ndim == 1:
            prices_pd = pd.Series(prices)
        else:
            prices_pd = pd.DataFrame(prices)
    else:
        prices_pd = prices.copy()
    
    # Ensure prices has a DatetimeIndex
    if not isinstance(prices_pd.index, pd.DatetimeIndex):
        raise ValueError("Price series must have a DatetimeIndex for corporate action adjustments")
    
    # Ensure corporate_actions 'date' column is datetime type
    corporate_actions['date'] = pd.to_datetime(corporate_actions['date'])
    
    # Sort corporate_actions by date (oldest to newest)
    corporate_actions = corporate_actions.sort_values('date')
    
    # Adjustment factor to apply to prices (start with 1.0)
    if isinstance(prices_pd, pd.Series):
        adjustment_factors = pd.Series(1.0, index=prices_pd.index)
    else:
        adjustment_factors = pd.DataFrame(1.0, index=prices_pd.index, columns=prices_pd.columns)
    
    # Process each action type
    for action_type in action_types:
        if action_type == 'dividend':
            # Filter actions to include only dividends
            dividend_actions = corporate_actions[corporate_actions['action_type'] == 'dividend']
            
            for _, row in dividend_actions.iterrows():
                action_date = row['date']
                dividend_amount = row['value']
                
                # Find dates prior to ex-dividend date
                if isinstance(prices_pd, pd.Series):
                    prior_dates = prices_pd.index < action_date
                    # Calculate adjustment factor (price + dividend) / price
                    earliest_date_after = prices_pd.index[prices_pd.index >= action_date].min()
                    if pd.notna(earliest_date_after):
                        dividend_ratio = (prices_pd[earliest_date_after] + dividend_amount) / prices_pd[earliest_date_after]
                        adjustment_factors[prior_dates] *= dividend_ratio
                else:
                    # For DataFrames, adjust each column separately if there's a symbol column
                    if 'symbol' in row:
                        symbol = row['symbol']
                        if symbol in prices_pd.columns:
                            prior_dates = prices_pd.index < action_date
                            earliest_date_after = prices_pd.index[prices_pd.index >= action_date].min()
                            if pd.notna(earliest_date_after):
                                dividend_ratio = (prices_pd.loc[earliest_date_after, symbol] + dividend_amount) / prices_pd.loc[earliest_date_after, symbol]
                                adjustment_factors.loc[prior_dates, symbol] *= dividend_ratio
                    else:
                        # Apply to all columns if no symbol specified
                        prior_dates = prices_pd.index < action_date
                        for col in prices_pd.columns:
                            earliest_date_after = prices_pd.index[prices_pd.index >= action_date].min()
                            if pd.notna(earliest_date_after) and pd.notna(prices_pd.loc[earliest_date_after, col]):
                                dividend_ratio = (prices_pd.loc[earliest_date_after, col] + dividend_amount) / prices_pd.loc[earliest_date_after, col]
                                adjustment_factors.loc[prior_dates, col] *= dividend_ratio
        
        elif action_type == 'split':
            # Filter actions to include only splits
            split_actions = corporate_actions[corporate_actions['action_type'] == 'split']
            
            for _, row in split_actions.iterrows():
                action_date = row['date']
                split_ratio = row['value']  # e.g., 2 for 2-for-1 split
                
                # Find dates prior to split date
                if isinstance(prices_pd, pd.Series):
                    prior_dates = prices_pd.index < action_date
                    adjustment_factors[prior_dates] *= split_ratio
                else:
                    # For DataFrames, adjust each column separately if there's a symbol column
                    if 'symbol' in row:
                        symbol = row['symbol']
                        if symbol in prices_pd.columns:
                            prior_dates = prices_pd.index < action_date
                            adjustment_factors.loc[prior_dates, symbol] *= split_ratio
                    else:
                        # Apply to all columns if no symbol specified
                        prior_dates = prices_pd.index < action_date
                        for col in prices_pd.columns:
                            adjustment_factors.loc[prior_dates, col] *= split_ratio
        
        elif action_type == 'merger':
            # Filter actions to include only mergers
            merger_actions = corporate_actions[corporate_actions['action_type'] == 'merger']
            
            for _, row in merger_actions.iterrows():
                action_date = row['date']
                merger_ratio = row['value']  # Ratio of old to new shares
                
                # Require additional columns for mergers
                if 'old_symbol' not in row or 'new_symbol' not in row:
                    logger.warning("Skipping merger action due to missing 'old_symbol' or 'new_symbol' information")
                    continue
                
                old_symbol = row['old_symbol']
                new_symbol = row['new_symbol']
                
                # Handle merger - only applies to DataFrames
                if isinstance(prices_pd, pd.DataFrame):
                    if old_symbol in prices_pd.columns and new_symbol in prices_pd.columns:
                        prior_dates = prices_pd.index < action_date
                        # Adjust old symbol prices relative to new symbol
                        adjustment_ratio = merger_ratio * prices_pd.loc[prices_pd.index[prices_pd.index >= action_date].min(), new_symbol] / prices_pd.loc[prices_pd.index[prices_pd.index < action_date].max(), old_symbol]
                        adjustment_factors.loc[prior_dates, old_symbol] *= adjustment_ratio
        
        else:
            logger.warning(f"Ignoring unsupported corporate action type: {action_type}")
    
    # Apply adjustment factors to prices
    adjusted_prices = prices_pd * adjustment_factors
    
    # Convert back to original format if input was numpy array
    if isinstance(prices, np.ndarray):
        return adjusted_prices.values
    
    return adjusted_prices


def split_train_test(
    data: Any,
    split_point: Union[float, str, pd.Timestamp],
    shuffle: Optional[bool] = False
) -> Tuple[Union[np.ndarray, pd.Series, pd.DataFrame], Union[np.ndarray, pd.Series, pd.DataFrame]]:
    """
    Splits time series data into training and testing sets with support for various 
    splitting methods.
    
    Parameters
    ----------
    data : Any
        Time series data to split (array, Series, DataFrame)
    split_point : Union[float, str, pd.Timestamp]
        Point to split the data:
        - If float (0 < split_point < 1): Fraction of data to use for training
        - If str or pd.Timestamp: Date to use as split point
    shuffle : bool, default=False
        Whether to shuffle the training data (only valid for non-time-based splitting)
        
    Returns
    -------
    Tuple[Union[np.ndarray, pd.Series, pd.DataFrame], Union[np.ndarray, pd.Series, pd.DataFrame]]
        Tuple of (train_data, test_data) in same format as input
        
    Raises
    ------
    ValueError
        If split_point is invalid or data cannot be split
    """
    # Determine input type to preserve it for output
    is_pandas_series = isinstance(data, pd.Series)
    is_pandas_dataframe = isinstance(data, pd.DataFrame)
    has_time_index = False
    
    if is_pandas_series or is_pandas_dataframe:
        has_time_index = isinstance(data.index, pd.DatetimeIndex)
    
    # Handle fractional split_point (percentage-based split)
    if isinstance(split_point, float):
        if not 0 < split_point < 1:
            raise ValueError("Fractional split_point must be between 0 and 1")
        
        if is_pandas_series or is_pandas_dataframe:
            n = len(data)
            split_idx = int(n * split_point)
            
            train_data = data.iloc[:split_idx].copy()
            test_data = data.iloc[split_idx:].copy()
            
            # Shuffle training data if requested (and if not time-indexed)
            if shuffle and not has_time_index:
                train_data = train_data.sample(frac=1.0).copy()
        
        else:
            # NumPy array
            n = len(data)
            split_idx = int(n * split_point)
            
            train_data = data[:split_idx].copy()
            test_data = data[split_idx:].copy()
            
            # Shuffle training data if requested
            if shuffle:
                rng = np.random.default_rng()
                train_data = rng.permutation(train_data)
    
    # Handle date-based split_point (for time-indexed data)
    elif isinstance(split_point, (str, pd.Timestamp)):
        if not has_time_index:
            raise ValueError("Date-based split requires data with DatetimeIndex")
        
        # Convert string to Timestamp if needed
        if isinstance(split_point, str):
            split_date = pd.Timestamp(split_point)
        else:
            split_date = split_point
        
        train_data = data.loc[:split_date].copy()
        test_data = data.loc[split_date:].copy()
        
        # Don't include split_date in both train and test
        if not train_data.empty and not test_data.empty:
            if train_data.index[-1] == test_data.index[0]:
                test_data = test_data.iloc[1:].copy()
    
    else:
        raise ValueError(f"Unsupported split_point type: {type(split_point)}")
    
    # Check if we got valid splits
    if len(train_data) == 0:
        raise ValueError("Training set is empty. Adjust split_point.")
    
    if len(test_data) == 0:
        raise ValueError("Test set is empty. Adjust split_point.")
    
    return train_data, test_data


def convert_to_log_returns(
    prices: Any,
    periods: int = 1
) -> Union[np.ndarray, pd.Series, pd.DataFrame]:
    """
    Converts price series to logarithmic returns with proper handling of time indices.
    
    Parameters
    ----------
    prices : Any
        Price series data (array, Series, DataFrame)
    periods : int, default=1
        Number of periods to use in return calculation
        
    Returns
    -------
    Union[np.ndarray, pd.Series, pd.DataFrame]
        Log returns in same format as input
        
    Raises
    ------
    ValueError
        If prices contain invalid values
    """
    return calculate_financial_returns(prices, method='log', periods=periods)


def convert_to_simple_returns(
    prices: Any,
    periods: int = 1
) -> Union[np.ndarray, pd.Series, pd.DataFrame]:
    """
    Converts price series to simple returns with proper handling of time indices.
    
    Parameters
    ----------
    prices : Any
        Price series data (array, Series, DataFrame)
    periods : int, default=1
        Number of periods to use in return calculation
        
    Returns
    -------
    Union[np.ndarray, pd.Series, pd.DataFrame]
        Simple returns in same format as input
        
    Raises
    ------
    ValueError
        If prices contain invalid values
    """
    return calculate_financial_returns(prices, method='simple', periods=periods)


def standardize_data(
    data: Any,
    robust: bool = False,
    fit_params: Optional[dict] = None
) -> Union[np.ndarray, pd.Series, pd.DataFrame]:
    """
    Standardizes time series data (z-score normalization) with optional robust scaling.
    
    Parameters
    ----------
    data : Any
        Time series data to standardize (array, Series, DataFrame)
    robust : bool, default=False
        If True, use median and IQR for scaling (robust to outliers)
        If False, use mean and standard deviation
    fit_params : Optional[dict], default=None
        Pre-computed parameters for standardization:
        - For robust=False: {'mean': float or array, 'std': float or array}
        - For robust=True: {'median': float or array, 'iqr': float or array}
        
    Returns
    -------
    Union[np.ndarray, pd.Series, pd.DataFrame]
        Standardized data in same format as input
        
    Raises
    ------
    ValueError
        If data cannot be standardized or fit_params is invalid
    """
    # Determine input type to preserve it for output
    is_pandas_series = isinstance(data, pd.Series)
    is_pandas_dataframe = isinstance(data, pd.DataFrame)
    original_index = None
    original_columns = None
    
    if is_pandas_series:
        original_index = data.index
    elif is_pandas_dataframe:
        original_index = data.index
        original_columns = data.columns
    
    # Convert to numpy array for standardization
    array_data = convert_time_series(data, output_type='numpy')
    
    # Set default fit_params if not provided
    if fit_params is None:
        fit_params = {}
    
    # Standardize using robust or non-robust method
    if robust:
        if 'median' in fit_params and 'iqr' in fit_params:
            median = fit_params['median']
            iqr = fit_params['iqr']
        else:
            if array_data.ndim == 1:
                median = np.nanmedian(array_data)
                q75, q25 = np.nanpercentile(array_data, [75, 25])
                iqr = q75 - q25
                # Handle potential zero IQR
                if iqr == 0:
                    iqr = np.nanstd(array_data) * 1.349  # Approximation
                    if iqr == 0:
                        iqr = 1.0  # Fallback
            else:
                # Calculate median and IQR for each column
                median = np.nanmedian(array_data, axis=0)
                q75 = np.nanpercentile(array_data, 75, axis=0)
                q25 = np.nanpercentile(array_data, 25, axis=0)
                iqr = q75 - q25
                # Handle potential zero IQR
                zero_iqr_mask = iqr == 0
                if np.any(zero_iqr_mask):
                    std = np.nanstd(array_data, axis=0)
                    iqr[zero_iqr_mask] = std[zero_iqr_mask] * 1.349
                    if np.any(iqr == 0):
                        iqr[iqr == 0] = 1.0
        
        # Standardize using median and IQR
        if array_data.ndim == 1:
            standardized_data = (array_data - median) / iqr
        else:
            standardized_data = np.zeros_like(array_data)
            for col in range(array_data.shape[1]):
                standardized_data[:, col] = (array_data[:, col] - median[col]) / iqr[col]
    
    else:
        if 'mean' in fit_params and 'std' in fit_params:
            mean = fit_params['mean']
            std = fit_params['std']
        else:
            if array_data.ndim == 1:
                mean = np.nanmean(array_data)
                std = np.nanstd(array_data)
                # Handle potential zero std
                if std == 0:
                    std = 1.0
            else:
                # Calculate mean and std for each column
                mean = np.nanmean(array_data, axis=0)
                std = np.nanstd(array_data, axis=0)
                # Handle potential zero std
                std[std == 0] = 1.0
        
        # Standardize using mean and std
        if array_data.ndim == 1:
            standardized_data = (array_data - mean) / std
        else:
            standardized_data = np.zeros_like(array_data)
            for col in range(array_data.shape[1]):
                standardized_data[:, col] = (array_data[:, col] - mean[col]) / std[col]
    
    # Convert back to original format
    if is_pandas_series:
        return pd.Series(standardized_data, index=original_index)
    elif is_pandas_dataframe:
        return pd.DataFrame(standardized_data, index=original_index, columns=original_columns)
    else:
        return standardized_data


def handle_date_range(
    data: Any,
    start_date: Optional[Union[str, pd.Timestamp]] = None,
    end_date: Optional[Union[str, pd.Timestamp]] = None,
    timezone: Optional[str] = None
) -> Union[pd.Series, pd.DataFrame]:
    """
    Filters time series data to a specific date range with proper handling of time zones.
    
    Parameters
    ----------
    data : Any
        Time series data to filter (Series, DataFrame with DatetimeIndex)
    start_date : Optional[Union[str, pd.Timestamp]], default=None
        Start date for the range (inclusive). If None, no start date filtering.
    end_date : Optional[Union[str, pd.Timestamp]], default=None
        End date for the range (inclusive). If None, no end date filtering.
    timezone : Optional[str], default=None
        Timezone to convert dates to (e.g., 'UTC', 'US/Eastern')
        
    Returns
    -------
    Union[pd.Series, pd.DataFrame]
        Time series data filtered to specified date range
        
    Raises
    ------
    ValueError
        If data doesn't have DatetimeIndex or dates are invalid
    """
    # Convert to pandas object for date handling
    if isinstance(data, np.ndarray):
        raise ValueError("NumPy arrays must have a DatetimeIndex for date range filtering. Convert to pandas first.")
    
    # Make a copy to avoid modifying the original
    result = data.copy()
    
    # Ensure data has DatetimeIndex
    if not isinstance(result.index, pd.DatetimeIndex):
        result = ensure_datetime_index(result)
    
    # Apply timezone if specified
    if timezone is not None:
        if result.index.tz is None:
            # Localize if index has no timezone
            result.index = result.index.tz_localize(timezone)
        else:
            # Convert if index already has a timezone
            result.index = result.index.tz_convert(timezone)
    
    # Convert start_date and end_date to Timestamp if provided as strings
    if start_date is not None and isinstance(start_date, str):
        start_date = pd.Timestamp(start_date)
        if timezone is not None and start_date.tz is None:
            start_date = start_date.tz_localize(timezone)
    
    if end_date is not None and isinstance(end_date, str):
        end_date = pd.Timestamp(end_date)
        if timezone is not None and end_date.tz is None:
            end_date = end_date.tz_localize(timezone)
    
    # Filter by start_date
    if start_date is not None:
        result = result.loc[result.index >= start_date]
    
    # Filter by end_date
    if end_date is not None:
        result = result.loc[result.index <= end_date]
    
    return result


def resample_high_frequency(
    data: Any,
    freq: str,
    price_column: str = 'price',
    volume_column: str = 'volume',
    method: str = 'ohlc'
) -> pd.DataFrame:
    """
    Resamples high-frequency financial data with specialized methods for intraday data.
    
    Parameters
    ----------
    data : Any
        High-frequency data to resample (must be convertible to DataFrame)
    freq : str
        Frequency string in pandas format (e.g., '1min', '5min', 'H')
    price_column : str, default='price'
        Name of the column containing price data
    volume_column : str, default='volume'
        Name of the column containing volume data
    method : str, default='ohlc'
        Resampling method:
        - 'ohlc': Open, High, Low, Close prices
        - 'vwap': Volume-weighted average price
        - 'twap': Time-weighted average price
        
    Returns
    -------
    pd.DataFrame
        Resampled high-frequency data
        
    Raises
    ------
    ValueError
        If required columns are missing or method is not supported
    """
    # Convert to pandas DataFrame
    df = convert_to_dataframe(data)
    
    # Ensure data has DatetimeIndex
    df = ensure_datetime_index(df)
    
    # Validate required columns
    if price_column not in df.columns:
        raise ValueError(f"Price column '{price_column}' not found in data")
    
    # Create resampler object
    resampler = df.resample(freq)
    
    # Apply the specified resampling method
    if method == 'ohlc':
        # Open, High, Low, Close prices
        result = pd.DataFrame(index=resampler.indices.keys())
        
        # Extract price OHLC
        price_ohlc = resampler[price_column].ohlc()
        result[f'{price_column}_open'] = price_ohlc['open']
        result[f'{price_column}_high'] = price_ohlc['high']
        result[f'{price_column}_low'] = price_ohlc['low']
        result[f'{price_column}_close'] = price_ohlc['close']
        
        # Add volume sum if volume column exists
        if volume_column in df.columns:
            result[volume_column] = resampler[volume_column].sum()
    
    elif method == 'vwap':
        # Volume-weighted average price
        if volume_column not in df.columns:
            raise ValueError(f"Volume column '{volume_column}' required for VWAP calculation")
        
        # Create price * volume column for VWAP calculation
        df['_price_volume'] = df[price_column] * df[volume_column]
        
        # Calculate VWAP and volume sum
        result = pd.DataFrame(index=resampler.indices.keys())
        result[volume_column] = resampler[volume_column].sum()
        sum_price_volume = resampler['_price_volume'].sum()
        
        # Calculate VWAP as sum(price * volume) / sum(volume)
        # Handle division by zero
        result[f'{price_column}_vwap'] = sum_price_volume / result[volume_column].replace(0, np.nan)
        
        # Add standard OHLC data too
        price_ohlc = resampler[price_column].ohlc()
        result[f'{price_column}_open'] = price_ohlc['open']
        result[f'{price_column}_high'] = price_ohlc['high']
        result[f'{price_column}_low'] = price_ohlc['low']
        result[f'{price_column}_close'] = price_ohlc['close']
    
    elif method == 'twap':
        # Time-weighted average price (simple average within each interval)
        result = pd.DataFrame(index=resampler.indices.keys())
        
        # Calculate mean price for each interval
        result[f'{price_column}_twap'] = resampler[price_column].mean()
        
        # Add standard OHLC data too
        price_ohlc = resampler[price_column].ohlc()
        result[f'{price_column}_open'] = price_ohlc['open']
        result[f'{price_column}_high'] = price_ohlc['high']
        result[f'{price_column}_low'] = price_ohlc['low']
        result[f'{price_column}_close'] = price_ohlc['close']
        
        # Add volume sum if volume column exists
        if volume_column in df.columns:
            result[volume_column] = resampler[volume_column].sum()
    
    else:
        raise ValueError(f"Unsupported resampling method: {method}")
    
    return result


def align_multiple_series(
    series_list: List[Any],
    method: str = 'outer',
    index: Optional[Union[str, pd.DatetimeIndex]] = None
) -> List[pd.Series]:
    """
    Aligns multiple time series to a common time index with proper handling of missing values.
    
    Parameters
    ----------
    series_list : List[Any]
        List of time series data to align
    method : str, default='outer'
        Join method:
        - 'outer': Union of all time indices (includes all timestamps)
        - 'inner': Intersection of all time indices (only common timestamps)
        - 'forward': Forward-fill missing values after alignment
        - 'backward': Backward-fill missing values after alignment
    index : Optional[Union[str, pd.DatetimeIndex]], default=None
        Common index to use. If None, derived from the series based on join method.
        
    Returns
    -------
    List[pd.Series]
        List of aligned time series
        
    Raises
    ------
    ValueError
        If series_list is empty or method is not supported
    """
    if not series_list:
        return []
    
    # Convert all series to pandas Series with DatetimeIndex
    pandas_series = []
    for series in series_list:
        if isinstance(series, pd.Series):
            # Ensure DatetimeIndex
            if not isinstance(series.index, pd.DatetimeIndex):
                series = ensure_datetime_index(series)
            pandas_series.append(series)
        else:
            # Convert to Series
            pandas_series.append(convert_to_series(series))
            # Ensure DatetimeIndex
            if not isinstance(pandas_series[-1].index, pd.DatetimeIndex):
                pandas_series[-1] = ensure_datetime_index(pandas_series[-1])
    
    # Determine common index if not provided
    if index is None:
        if method == 'outer':
            # Union of all time indices
            all_indices = [s.index for s in pandas_series]
            common_index = pd.DatetimeIndex(sorted(set().union(*all_indices)))
        elif method == 'inner':
            # Intersection of all time indices
            all_indices = [set(s.index) for s in pandas_series]
            common_timestamps = all_indices[0].intersection(*all_indices[1:])
            common_index = pd.DatetimeIndex(sorted(common_timestamps))
        else:
            # Default to outer join for other methods
            all_indices = [s.index for s in pandas_series]
            common_index = pd.DatetimeIndex(sorted(set().union(*all_indices)))
    elif isinstance(index, str):
        # Try to interpret string as frequency for date_range
        start_date = min(s.index.min() for s in pandas_series)
        end_date = max(s.index.max() for s in pandas_series)
        common_index = pd.date_range(start=start_date, end=end_date, freq=index)
    else:
        # Use provided DatetimeIndex
        common_index = index
    
    # Align all series to the common index
    aligned_series = []
    
    if method == 'forward' or method == 'ffill':
        # Forward-fill after reindexing
        for series in pandas_series:
            aligned_series.append(series.reindex(common_index).ffill())
    elif method == 'backward' or method == 'bfill':
        # Backward-fill after reindexing
        for series in pandas_series:
            aligned_series.append(series.reindex(common_index).bfill())
    elif method == 'outer' or method == 'inner':
        # Just reindex without filling
        for series in pandas_series:
            aligned_series.append(series.reindex(common_index))
    else:
        raise ValueError(f"Unsupported alignment method: {method}")
    
    return aligned_series


def merge_time_series(
    series_list: List[Any],
    names: Optional[List[str]] = None,
    join: str = 'outer',
    fill_method: str = 'ffill'
) -> pd.DataFrame:
    """
    Merges multiple time series into a single DataFrame with proper handling of time indices.
    
    Parameters
    ----------
    series_list : List[Any]
        List of time series data to merge
    names : Optional[List[str]], default=None
        Column names for each series. If None, uses numbered columns.
    join : str, default='outer'
        Join method for merging indices:
        - 'outer': Union of all indices
        - 'inner': Intersection of all indices
    fill_method : str, default='ffill'
        Method for filling missing values after merge:
        - 'ffill': Forward fill
        - 'bfill': Backward fill
        - None: No filling (leave as NaN)
        
    Returns
    -------
    pd.DataFrame
        Merged DataFrame containing all time series
        
    Raises
    ------
    ValueError
        If series_list is empty or join/fill_method is not supported
    """
    if not series_list:
        return pd.DataFrame()
    
    # Generate column names if not provided
    if names is None:
        names = [f'series_{i}' for i in range(len(series_list))]
    elif len(names) != len(series_list):
        raise ValueError("Length of names must match length of series_list")
    
    # Align series using align_multiple_series
    aligned_series = align_multiple_series(series_list, method=join)
    
    # Create empty DataFrame with proper index
    result = pd.DataFrame(index=aligned_series[0].index)
    
    # Add each series as a column
    for i, series in enumerate(aligned_series):
        result[names[i]] = series
    
    # Apply fill method if specified
    if fill_method == 'ffill' or fill_method == 'forward':
        result = result.ffill()
    elif fill_method == 'bfill' or fill_method == 'backward':
        result = result.bfill()
    elif fill_method is not None:
        raise ValueError(f"Unsupported fill method: {fill_method}")
    
    return result


async def async_convert_time_series(
    data: Any,
    output_type: str,
    time_index: Optional[Any] = None,
    ensure_finite: bool = True
) -> Union[np.ndarray, pd.Series, pd.DataFrame]:
    """
    Asynchronous version of convert_time_series for handling large datasets without blocking.
    
    Parameters
    ----------
    data : Any
        The input data to convert (array, list, Series, DataFrame, etc.)
    output_type : str
        Target format: 'numpy', 'series', or 'dataframe'
    time_index : Optional[Any], default=None
        Time index to use for pandas objects. Can be a list of dates, 
        DatetimeIndex, or string column name.
    ensure_finite : bool, default=True
        If True, checks for and potentially raises on NaN and Inf values
        
    Returns
    -------
    Union[np.ndarray, pd.Series, pd.DataFrame]
        Converted time series data in the specified format
        
    Notes
    -----
    This function performs the same operation as convert_time_series but
    asynchronously using asyncio, making it suitable for large datasets or when
    used in asynchronous applications.
    """
    # Use a thread pool executor to run the CPU-bound conversion without blocking the event loop
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,  # Default executor
        lambda: convert_time_series(data, output_type, time_index, ensure_finite)
    )