"""
Utility functions for working with pandas DataFrames and Series,
focusing on time series manipulation for financial econometric analysis.
"""

import logging
from typing import Any, List, Optional, Tuple, Union

import numpy as np  # version 1.26.3
import pandas as pd  # version 2.1.4

# Set up logger
logger = logging.getLogger(__name__)


def convert_to_dataframe(
    data: Any, 
    column_names: Optional[list] = None,
    index: Optional[Union[str, pd.DatetimeIndex]] = None
) -> pd.DataFrame:
    """
    Convert various data types to a pandas DataFrame.
    
    Parameters
    ----------
    data : Any
        Data to convert, can be DataFrame, Series, NumPy array, list, etc.
    column_names : list, optional
        Column names to use if data is not already a DataFrame. If None and data
        is not a DataFrame, numeric column names will be used.
    index : str or DatetimeIndex, optional
        Index to use for the DataFrame. Can be either a column name in the data
        to use as index, or a DatetimeIndex to directly set.
        
    Returns
    -------
    pd.DataFrame
        Pandas DataFrame with properly formatted data.
        
    Raises
    ------
    TypeError
        If the data cannot be converted to a DataFrame.
    """
    try:
        # If already a DataFrame
        if isinstance(data, pd.DataFrame):
            df = data.copy() if index is not None else data
        
        # If Series
        elif isinstance(data, pd.Series):
            name = data.name if data.name is not None else 0
            df = pd.DataFrame({name: data})
        
        # If NumPy array
        elif isinstance(data, np.ndarray):
            if data.ndim == 1:
                # 1D array to single column DataFrame
                col_name = column_names[0] if column_names and len(column_names) > 0 else 0
                df = pd.DataFrame({col_name: data})
            else:
                # 2D array to multi-column DataFrame
                cols = column_names if column_names else range(data.shape[1])
                df = pd.DataFrame(data, columns=cols)
        
        # If list
        elif isinstance(data, list):
            if not data:
                return pd.DataFrame()
                
            if all(isinstance(item, (list, tuple)) for item in data):
                # List of lists
                cols = column_names if column_names else range(len(data[0]))
                df = pd.DataFrame(data, columns=cols)
            else:
                # Simple list
                col_name = column_names[0] if column_names and len(column_names) > 0 else 0
                df = pd.DataFrame({col_name: data})
        
        # If dictionary
        elif isinstance(data, dict):
            df = pd.DataFrame(data)
            
        else:
            raise TypeError(f"Cannot convert data of type {type(data)} to DataFrame")
        
        # Apply index if provided
        if index is not None:
            if isinstance(index, (pd.DatetimeIndex, pd.Index)):
                df.index = index
            elif isinstance(index, str) and index in df.columns:
                df = df.set_index(index)
                
        return df
                
    except Exception as e:
        logger.error(f"Error converting to DataFrame: {str(e)}")
        raise TypeError(f"Failed to convert to DataFrame: {str(e)}")


def convert_to_series(
    data: Any, 
    index: Optional[Union[str, pd.DatetimeIndex]] = None,
    name: Optional[str] = None
) -> pd.Series:
    """
    Convert various data types to a pandas Series.
    
    Parameters
    ----------
    data : Any
        Data to convert, can be Series, DataFrame, NumPy array, list, etc.
    index : str or DatetimeIndex, optional
        Index to use for the Series. Can be either a column name in the data
        to use as index, or a DatetimeIndex to directly set.
    name : str, optional
        Name to assign to the resulting Series.
        
    Returns
    -------
    pd.Series
        Pandas Series with properly formatted data.
        
    Raises
    ------
    TypeError
        If the data cannot be converted to a Series.
    ValueError
        If the data is a DataFrame with multiple columns and no column is specified.
    """
    try:
        # If already a Series
        if isinstance(data, pd.Series):
            series = data.copy() if index is not None or name is not None else data
            
            if name is not None:
                series.name = name
                
        # If DataFrame
        elif isinstance(data, pd.DataFrame):
            if data.shape[1] == 1:
                # Single column DataFrame
                series = data.iloc[:, 0]
            else:
                raise ValueError("DataFrame has multiple columns. Specify a column name to convert to Series.")
                
        # If NumPy array
        elif isinstance(data, np.ndarray):
            if data.ndim > 1 and data.shape[1] > 1:
                raise ValueError("Multi-dimensional array with multiple columns. Cannot convert to Series.")
            
            # Flatten array if multidimensional
            data_flat = data.flatten() if data.ndim > 1 else data
            series = pd.Series(data_flat, name=name)
            
        # If list
        elif isinstance(data, list):
            series = pd.Series(data, name=name)
            
        else:
            raise TypeError(f"Cannot convert data of type {type(data)} to Series")
        
        # Apply index if provided
        if index is not None:
            if isinstance(index, (pd.DatetimeIndex, pd.Index)):
                series.index = index
            elif isinstance(index, str) and isinstance(data, pd.DataFrame) and index in data.columns:
                series.index = data[index]
                
        return series
                
    except Exception as e:
        logger.error(f"Error converting to Series: {str(e)}")
        raise TypeError(f"Failed to convert to Series: {str(e)}")


def ensure_datetime_index(
    data: Union[pd.DataFrame, pd.Series],
    date_column: Optional[str] = None,
    inplace: bool = False
) -> Union[pd.DataFrame, pd.Series]:
    """
    Ensure a pandas DataFrame or Series has a DatetimeIndex, converting if necessary.
    
    Parameters
    ----------
    data : DataFrame or Series
        The data object to check or convert.
    date_column : str, optional
        If provided and data is a DataFrame, use this column to create the DatetimeIndex.
    inplace : bool, default False
        If True, modify the original object, otherwise return a copy.
        
    Returns
    -------
    DataFrame or Series
        The original data with DatetimeIndex if inplace is True, otherwise a copy.
        
    Raises
    ------
    TypeError
        If the index cannot be converted to a DatetimeIndex.
    ValueError
        If date_column is provided but not found in the DataFrame.
    """
    # Create a copy if not inplace
    df = data if inplace else data.copy()
    
    # Already has DatetimeIndex
    if isinstance(df.index, pd.DatetimeIndex):
        return df
    
    try:
        # If date_column is provided, use it to create DatetimeIndex
        if date_column is not None:
            if not isinstance(df, pd.DataFrame):
                raise TypeError("date_column can only be used with DataFrame")
                
            if date_column not in df.columns:
                raise ValueError(f"Column '{date_column}' not found in DataFrame")
                
            df = df.set_index(pd.DatetimeIndex(pd.to_datetime(df[date_column])))
            
        # Try to convert current index to DatetimeIndex
        else:
            df.index = pd.DatetimeIndex(pd.to_datetime(df.index))
            
        return df
        
    except Exception as e:
        logger.error(f"Error ensuring DatetimeIndex: {str(e)}")
        raise TypeError(f"Failed to convert to DatetimeIndex: {str(e)}")


def resample_time_series(
    data: Union[pd.DataFrame, pd.Series],
    freq: str,
    method: str = 'mean'
) -> Union[pd.DataFrame, pd.Series]:
    """
    Resample time series data to a different frequency using pandas capabilities.
    
    Parameters
    ----------
    data : DataFrame or Series
        The time series data to resample.
    freq : str
        The frequency to resample to, using pandas frequency strings
        (e.g., 'D' for daily, 'M' for month-end, 'H' for hourly).
    method : str, default 'mean'
        The resampling method to use. Options include:
        - 'mean': Average values within each time bin
        - 'sum': Sum values within each time bin
        - 'first': First value in each time bin
        - 'last': Last value in each time bin
        - 'min': Minimum value in each time bin
        - 'max': Maximum value in each time bin
        - 'median': Median value in each time bin
        - 'ohlc': For OHLC resampling (only applicable for DataFrames with price data)
        
    Returns
    -------
    DataFrame or Series
        Resampled time series data.
        
    Raises
    ------
    ValueError
        If the method is not supported or if the data doesn't have a DatetimeIndex.
    """
    # Ensure data has DatetimeIndex
    if not isinstance(data.index, pd.DatetimeIndex):
        data = ensure_datetime_index(data)
    
    # Create resample object
    resampler = data.resample(freq)
    
    # Apply resampling method
    try:
        if method == 'mean':
            return resampler.mean()
        elif method == 'sum':
            return resampler.sum()
        elif method == 'first':
            return resampler.first()
        elif method == 'last':
            return resampler.last()
        elif method == 'min':
            return resampler.min()
        elif method == 'max':
            return resampler.max()
        elif method == 'median':
            return resampler.median()
        elif method == 'ohlc':
            if isinstance(data, pd.Series):
                # Convert Series to DataFrame for OHLC
                df = pd.DataFrame(data)
                result = resampler.ohlc()
                # Flatten column names
                result.columns = [f"{data.name}_{col}" if data.name else col for col in result.columns]
                return result
            else:
                return resampler.ohlc()
        else:
            raise ValueError(f"Unsupported resampling method: {method}")
    
    except Exception as e:
        logger.error(f"Error resampling time series: {str(e)}")
        raise ValueError(f"Failed to resample time series: {str(e)}")


def align_time_series(
    data_list: List[Union[pd.DataFrame, pd.Series]],
    method: Optional[str] = None
) -> List[Union[pd.DataFrame, pd.Series]]:
    """
    Align multiple time series to a common DatetimeIndex.
    
    Parameters
    ----------
    data_list : list of DataFrame or Series
        List of time series data objects to align.
    method : str, optional
        Method to use for filling missing values after alignment:
        - None: Don't fill missing values (default)
        - 'ffill': Forward fill
        - 'bfill': Backward fill
        - 'nearest': Nearest value
        - 'zero': Fill with zeros
        
    Returns
    -------
    list of DataFrame or Series
        List of aligned time series with common DatetimeIndex.
        
    Raises
    ------
    ValueError
        If any of the inputs cannot be aligned.
    """
    if not data_list:
        return []
    
    try:
        # Ensure all data have DatetimeIndex
        aligned_data = []
        
        for data in data_list:
            if not isinstance(data.index, pd.DatetimeIndex):
                aligned_data.append(ensure_datetime_index(data))
            else:
                aligned_data.append(data)
        
        # Find all unique timestamps across all series
        all_timestamps = sorted(set().union(*[data.index for data in aligned_data]))
        common_index = pd.DatetimeIndex(all_timestamps)
        
        # Reindex all series to the common index
        result = []
        for data in aligned_data:
            if method is None:
                result.append(data.reindex(common_index))
            elif method == 'ffill':
                result.append(data.reindex(common_index).ffill())
            elif method == 'bfill':
                result.append(data.reindex(common_index).bfill())
            elif method == 'nearest':
                result.append(data.reindex(common_index, method='nearest'))
            elif method == 'zero':
                reindexed = data.reindex(common_index)
                reindexed.fillna(0, inplace=True)
                result.append(reindexed)
            else:
                raise ValueError(f"Unsupported fill method: {method}")
        
        return result
    
    except Exception as e:
        logger.error(f"Error aligning time series: {str(e)}")
        raise ValueError(f"Failed to align time series: {str(e)}")


def rolling_window_pandas(
    data: Union[pd.DataFrame, pd.Series],
    window_size: int,
    window_type: Optional[str] = None
) -> pd.core.window.rolling.Rolling:
    """
    Create rolling windows of data using pandas' rolling function with enhanced features.
    
    Parameters
    ----------
    data : DataFrame or Series
        The time series data to create rolling windows from.
    window_size : int
        Size of the rolling window.
    window_type : str, optional
        Type of window to use:
        - None: Standard rolling window (default)
        - 'gaussian': Gaussian window
        - 'exponential': Exponentially weighted window
        - 'triangular': Triangular window
        - 'triang': Triangular window (alternative)
        - 'blackman': Blackman window
        - 'hamming': Hamming window
        - 'bartlett': Bartlett window
        - 'parzen': Parzen window
        
    Returns
    -------
    pandas.core.window.rolling.Rolling
        Rolling window object for further operations.
        
    Raises
    ------
    ValueError
        If window_size is invalid or window_type is not supported.
    """
    if window_size <= 0:
        raise ValueError("Window size must be positive")
    
    try:
        if window_type is None:
            return data.rolling(window=window_size)
        elif window_type == 'exponential':
            return data.ewm(span=window_size)
        else:
            # Other window types use the win_type parameter
            valid_windows = ['gaussian', 'triangular', 'triang', 'blackman', 'hamming', 'bartlett', 'parzen']
            if window_type not in valid_windows:
                raise ValueError(f"Unsupported window type: {window_type}. Valid options are: {', '.join(valid_windows)}")
            
            return data.rolling(window=window_size, win_type=window_type)
    
    except Exception as e:
        if "win_type" in str(e):
            logger.error(f"Window type '{window_type}' requires scipy. Install scipy package.")
            raise ValueError(f"Window type '{window_type}' requires scipy package")
        logger.error(f"Error creating rolling window: {str(e)}")
        raise ValueError(f"Failed to create rolling window: {str(e)}")


def handle_missing_data(
    data: Union[pd.DataFrame, pd.Series],
    method: str = 'ffill',
    inplace: bool = False
) -> Union[pd.DataFrame, pd.Series]:
    """
    Handle missing data in pandas objects using various methods.
    
    Parameters
    ----------
    data : DataFrame or Series
        The data containing missing values to handle.
    method : str, default 'ffill'
        Method to use for handling missing values:
        - 'ffill': Forward fill (propagate last valid observation forward)
        - 'bfill': Backward fill (use next valid observation to fill gap)
        - 'interpolate': Interpolate values
        - 'drop': Remove rows with any missing values
        - 'fill_value': Fill with a specific value (0 by default)
        - 'mean': Fill with mean value (column-wise for DataFrame)
        - 'median': Fill with median value (column-wise for DataFrame)
        - 'mode': Fill with mode value (column-wise for DataFrame)
    inplace : bool, default False
        If True, modify the original object, otherwise return a copy.
        
    Returns
    -------
    DataFrame or Series
        Data with missing values handled.
        
    Raises
    ------
    ValueError
        If the method is not supported.
    """
    # Create a copy if not inplace
    df = data if inplace else data.copy()
    
    try:
        if method == 'ffill':
            return df.fillna(method='ffill')
        elif method == 'bfill':
            return df.fillna(method='bfill')
        elif method == 'interpolate':
            return df.interpolate()
        elif method == 'drop':
            return df.dropna()
        elif method == 'fill_value':
            return df.fillna(0)  # Default to 0
        elif method == 'mean':
            if isinstance(df, pd.DataFrame):
                return df.fillna(df.mean())
            else:
                return df.fillna(df.mean())
        elif method == 'median':
            if isinstance(df, pd.DataFrame):
                return df.fillna(df.median())
            else:
                return df.fillna(df.median())
        elif method == 'mode':
            if isinstance(df, pd.DataFrame):
                # Mode can return multiple values, take the first
                mode_values = df.mode().iloc[0]
                return df.fillna(mode_values)
            else:
                mode_value = df.mode().iloc[0]
                return df.fillna(mode_value)
        else:
            raise ValueError(f"Unsupported method for handling missing data: {method}")
    
    except Exception as e:
        logger.error(f"Error handling missing data: {str(e)}")
        raise ValueError(f"Failed to handle missing data: {str(e)}")


def extract_time_features(
    data: Union[pd.DataFrame, pd.Series],
    features: List[str]
) -> pd.DataFrame:
    """
    Extract time-based features from DatetimeIndex for time series analysis.
    
    Parameters
    ----------
    data : DataFrame or Series
        Time series data with DatetimeIndex.
    features : list of str
        List of time features to extract. Options include:
        - 'year': Year
        - 'month': Month (1-12)
        - 'day': Day of month
        - 'dayofweek': Day of week (0=Monday, 6=Sunday)
        - 'dayofyear': Day of year
        - 'quarter': Quarter (1-4)
        - 'hour': Hour
        - 'minute': Minute
        - 'second': Second
        - 'week': Week number
        - 'is_month_start': Is first day of month
        - 'is_month_end': Is last day of month
        - 'is_quarter_start': Is first day of quarter
        - 'is_quarter_end': Is last day of quarter
        - 'is_year_start': Is first day of year
        - 'is_year_end': Is last day of year
        - 'weekday_name': Day name (Monday, Tuesday, etc.)
        
    Returns
    -------
    DataFrame
        Original data with additional time feature columns.
        
    Raises
    ------
    ValueError
        If the data doesn't have a DatetimeIndex or features list is empty.
    """
    # Ensure data has DatetimeIndex
    if not isinstance(data.index, pd.DatetimeIndex):
        data = ensure_datetime_index(data)
    
    if not features:
        raise ValueError("Features list cannot be empty")
    
    # Convert Series to DataFrame if needed
    if isinstance(data, pd.Series):
        df = data.to_frame()
    else:
        df = data.copy()
    
    try:
        valid_features = [
            'year', 'month', 'day', 'dayofweek', 'dayofyear',
            'quarter', 'hour', 'minute', 'second', 'week',
            'is_month_start', 'is_month_end', 'is_quarter_start',
            'is_quarter_end', 'is_year_start', 'is_year_end'
        ]
        
        for feature in features:
            if feature == 'weekday_name':
                # Get day name (Monday, Tuesday, etc.)
                df[feature] = df.index.day_name()
            elif feature in valid_features:
                # Get attribute directly from DatetimeIndex
                df[feature] = getattr(df.index, feature)
            else:
                logger.warning(f"Unknown time feature: {feature}")
        
        return df
    
    except Exception as e:
        logger.error(f"Error extracting time features: {str(e)}")
        raise ValueError(f"Failed to extract time features: {str(e)}")


def split_time_series(
    data: Union[pd.DataFrame, pd.Series],
    split_point: Union[str, pd.Timestamp, float]
) -> Tuple[Union[pd.DataFrame, pd.Series], Union[pd.DataFrame, pd.Series]]:
    """
    Split time series data into train and test sets based on time.
    
    Parameters
    ----------
    data : DataFrame or Series
        Time series data to split.
    split_point : str, Timestamp, or float
        Point to split the data:
        - If str or Timestamp: Date to use as cutoff
        - If float between 0 and 1: Fraction of data to use for training
        
    Returns
    -------
    tuple of (DataFrame or Series, DataFrame or Series)
        Train and test datasets.
        
    Raises
    ------
    ValueError
        If split_point is invalid or data doesn't have a DatetimeIndex.
    """
    # Ensure data has DatetimeIndex
    if not isinstance(data.index, pd.DatetimeIndex):
        data = ensure_datetime_index(data)
    
    try:
        # Handle float split point (fraction of data)
        if isinstance(split_point, float):
            if not 0.0 < split_point < 1.0:
                raise ValueError("Float split_point must be between 0 and 1")
                
            split_idx = int(len(data) * split_point)
            train = data.iloc[:split_idx]
            test = data.iloc[split_idx:]
            return train, test
        
        # Handle date split point
        else:
            if isinstance(split_point, str):
                split_date = pd.Timestamp(split_point)
            else:
                split_date = split_point
                
            train = data.loc[:split_date]
            test = data.loc[split_date:]
            
            # Remove overlap (split_date appears in both)
            if not train.empty and not test.empty and train.index[-1] == test.index[0]:
                test = test.iloc[1:]
                
            return train, test
    
    except Exception as e:
        logger.error(f"Error splitting time series: {str(e)}")
        raise ValueError(f"Failed to split time series: {str(e)}")


def calculate_returns(
    prices: Union[pd.DataFrame, pd.Series],
    method: str = 'simple',
    periods: int = 1
) -> Union[pd.DataFrame, pd.Series]:
    """
    Calculate returns from price series with various methods.
    
    Parameters
    ----------
    prices : DataFrame or Series
        Price data to calculate returns from.
    method : str, default 'simple'
        Method to calculate returns:
        - 'simple': (p_t / p_{t-periods}) - 1
        - 'log': log(p_t / p_{t-periods})
        - 'pct': (p_t - p_{t-periods}) / p_{t-periods} * 100
    periods : int, default 1
        Number of periods to use for return calculation.
        
    Returns
    -------
    DataFrame or Series
        Calculated returns.
        
    Raises
    ------
    ValueError
        If the method is not supported or periods is invalid.
    """
    if periods <= 0:
        raise ValueError("Periods must be positive")
    
    try:
        if method == 'simple':
            returns = prices / prices.shift(periods) - 1
        elif method == 'log':
            returns = np.log(prices / prices.shift(periods))
        elif method == 'pct':
            returns = (prices - prices.shift(periods)) / prices.shift(periods) * 100
        else:
            raise ValueError(f"Unsupported return calculation method: {method}")
            
        return returns
    
    except Exception as e:
        logger.error(f"Error calculating returns: {str(e)}")
        raise ValueError(f"Failed to calculate returns: {str(e)}")