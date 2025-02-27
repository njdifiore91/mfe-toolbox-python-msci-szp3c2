"""
MFE Toolbox - Realized Volatility Module

This module implements realized volatility measures for high-frequency financial data analysis,
including kernel-based estimation and noise filtering techniques.
"""

import numpy as np  # numpy 1.26.3
from numba import jit  # numba 0.59.0
import pandas as pd  # pandas 2.1.4
from scipy import stats  # scipy 1.11.4
from statsmodels import kernel_regression  # statsmodels 0.14.1

from ..utils.validation import validate_input
from ..utils.numba_helpers import jit_check
from ..utils.data_handling import convert_time_series
from ..utils.numpy_helpers import ensure_array
from ..utils.pandas_helpers import to_datetime_index
from ..utils.async_helpers import async_process

# Default parameters
DEFAULT_SAMPLING_INTERVAL = 300  # Default sampling interval in seconds
DEFAULT_KERNEL_TYPE = 'bartlett'  # Default kernel type for realized kernel estimation

def realized_variance(
    prices: np.ndarray,
    times: np.ndarray or pd.Series,
    time_type: str,
    sampling_type: str,
    sampling_interval=DEFAULT_SAMPLING_INTERVAL,
    noise_adjust: bool = False
) -> tuple[float, float]:
    """
    Computes the realized variance of a high-frequency price series with optional noise filtering.
    
    Parameters
    ----------
    prices : np.ndarray
        High-frequency price series
    times : np.ndarray or pd.Series
        Timestamps corresponding to each price observation
    time_type : str
        Type of time data: 'datetime', 'seconds', or 'businesstime'
    sampling_type : str
        Method for sampling: 'CalendarTime', 'BusinessTime', 'CalendarUniform', 
        'BusinessUniform', or 'Fixed'
    sampling_interval : int or tuple, default=DEFAULT_SAMPLING_INTERVAL
        Sampling interval specification, interpretation depends on sampling_type
    noise_adjust : bool, default=False
        If True, applies noise filtering to the returns
        
    Returns
    -------
    tuple[float, float]
        Realized variance and subsampled realized variance
    """
    # Validate inputs
    validate_input({
        'time_type': {'type': str, 'allowed_values': ['datetime', 'seconds', 'businesstime']},
        'sampling_type': {'type': str, 'allowed_values': 
                         ['CalendarTime', 'BusinessTime', 'CalendarUniform', 'BusinessUniform', 'Fixed']}
    })
    
    # Convert time series to appropriate format
    prices = ensure_array(prices)
    times = ensure_array(times)
    
    # Sample the price data
    sampled_prices, sampled_times = sampling_scheme(
        prices, times, time_type, sampling_type, sampling_interval
    )
    
    # Calculate returns (log price differences)
    returns = np.diff(np.log(sampled_prices))
    
    # Apply noise filtering if requested
    if noise_adjust:
        returns = noise_filter(sampled_prices[:-1], returns, 'MA', {'window': 2})
    
    # Compute realized variance as sum of squared returns
    rv = np.sum(returns ** 2)
    
    # Calculate subsampled version for comparison
    # Use a different sampling interval for subsampling
    if isinstance(sampling_interval, tuple):
        subsample_interval = (sampling_interval[0] * 2, 
                             sampling_interval[1] * 2 if len(sampling_interval) > 1 else None)
    else:
        subsample_interval = sampling_interval * 2
        
    subsample_prices, subsample_times = sampling_scheme(
        prices, times, time_type, sampling_type, subsample_interval
    )
    subsample_returns = np.diff(np.log(subsample_prices))
    
    if noise_adjust:
        subsample_returns = noise_filter(subsample_prices[:-1], subsample_returns, 'MA', {'window': 2})
        
    rv_ss = np.sum(subsample_returns ** 2)
    
    return rv, rv_ss

def realized_kernel(
    prices: np.ndarray,
    times: np.ndarray or pd.Series,
    time_type: str,
    kernel_type: str = DEFAULT_KERNEL_TYPE,
    bandwidth: float = None
) -> float:
    """
    Implements kernel-based estimation of realized volatility using various kernel functions.
    
    Parameters
    ----------
    prices : np.ndarray
        High-frequency price series
    times : np.ndarray or pd.Series
        Timestamps corresponding to each price observation
    time_type : str
        Type of time data: 'datetime', 'seconds', or 'businesstime'
    kernel_type : str, default=DEFAULT_KERNEL_TYPE
        Type of kernel function: 'bartlett', 'parzen', 'tukey-hanning', etc.
    bandwidth : float, default=None
        Bandwidth parameter for the kernel. If None, determined automatically.
    
    Returns
    -------
    float
        Realized kernel estimate of volatility
    """
    # Validate inputs
    validate_input({
        'time_type': {'type': str, 'allowed_values': ['datetime', 'seconds', 'businesstime']},
        'kernel_type': {'type': str, 'allowed_values': 
                       ['bartlett', 'parzen', 'tukey-hanning', 'qs', 'truncated']}
    })
    
    # Convert time series to appropriate format
    prices = ensure_array(prices)
    times = ensure_array(times)
    
    # Calculate returns (log price differences)
    returns = np.diff(np.log(prices))
    n = len(returns)
    
    # Determine bandwidth if not provided
    if bandwidth is None:
        # Scale-dependent bandwidth selection (Barndorff-Nielsen et al., 2009)
        c = 0.5  # Constant for bandwidth selection
        q = 0.5  # Quantile for robust scale estimation
        scale = np.percentile(np.abs(returns), q * 100) / 0.6745
        bandwidth = c * n**(1/3) * scale
    
    # Compute autocovariance matrices up to bandwidth lags
    max_lag = int(bandwidth)
    gamma0 = np.sum(returns**2)  # Realized variance
    
    # Initialize kernel weights
    kernel_weights = np.zeros(max_lag + 1)
    kernel_weights[0] = 1.0  # Weight for lag 0
    
    # Apply kernel weights based on kernel_type
    if kernel_type == 'bartlett':
        # Bartlett kernel: k(x) = 1 - |x| for |x| <= 1, 0 otherwise
        for h in range(1, max_lag + 1):
            kernel_weights[h] = 1.0 - (h / max_lag)
    elif kernel_type == 'parzen':
        # Parzen kernel
        for h in range(1, max_lag + 1):
            x = h / max_lag
            if x <= 0.5:
                kernel_weights[h] = 1 - 6 * x**2 + 6 * x**3
            else:
                kernel_weights[h] = 2 * (1 - x)**3
    elif kernel_type == 'tukey-hanning':
        # Tukey-Hanning kernel
        for h in range(1, max_lag + 1):
            x = h / max_lag
            kernel_weights[h] = 0.5 * (1 + np.cos(np.pi * x))
    elif kernel_type == 'qs':
        # Quadratic Spectral kernel
        for h in range(1, max_lag + 1):
            x = h / max_lag * bandwidth
            if x == 0:
                kernel_weights[h] = 1.0
            else:
                kernel_weights[h] = (25 / (12 * np.pi**2 * x**2)) * (
                    np.sin(6 * np.pi * x / 5) / (6 * np.pi * x / 5) - np.cos(6 * np.pi * x / 5)
                )
    elif kernel_type == 'truncated':
        # Truncated kernel (uniform weights up to bandwidth)
        kernel_weights = np.ones(max_lag + 1)
    
    # Compute realized kernel estimate
    rk = gamma0  # Start with the realized variance
    
    # Add weighted autocovariances
    for h in range(1, max_lag + 1):
        if h < n:
            # Compute h-th order autocovariance
            gamma_h = 0
            for t in range(n - h):
                gamma_h += returns[t] * returns[t + h]
            
            # Add weighted autocovariance to realized kernel
            rk += 2 * kernel_weights[h] * gamma_h
    
    return rk

def sampling_scheme(
    prices: np.ndarray,
    times: np.ndarray or pd.Series,
    time_type: str,
    sampling_type: str,
    sampling_interval: int or tuple
) -> tuple[np.ndarray, np.ndarray]:
    """
    Implements different sampling schemes for intraday data including calendar time and business time sampling.
    
    Parameters
    ----------
    prices : np.ndarray
        High-frequency price series
    times : np.ndarray or pd.Series
        Timestamps corresponding to each price observation
    time_type : str
        Type of time data: 'datetime', 'seconds', or 'businesstime'
    sampling_type : str
        Method for sampling: 'CalendarTime', 'BusinessTime', 'CalendarUniform', 
        'BusinessUniform', or 'Fixed'
    sampling_interval : int or tuple
        Sampling interval specification, interpretation depends on sampling_type
        
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Sampled prices and times
    """
    # Validate inputs
    validate_input({
        'time_type': {'type': str, 'allowed_values': ['datetime', 'seconds', 'businesstime']},
        'sampling_type': {'type': str, 'allowed_values': 
                         ['CalendarTime', 'BusinessTime', 'CalendarUniform', 'BusinessUniform', 'Fixed']}
    })
    
    # Convert to numpy arrays
    prices = ensure_array(prices)
    times = ensure_array(times)
    
    # Convert times to a standard format based on time_type
    if time_type == 'datetime':
        # Convert to pandas DatetimeIndex
        times = pd.DatetimeIndex(times)
        # Convert to seconds since midnight
        seconds = (times.hour * 3600 + times.minute * 60 + times.second + 
                  times.microsecond / 1e6)
    elif time_type == 'seconds':
        # Already in seconds format
        seconds = times
    elif time_type == 'businesstime':
        # Business time - no conversion needed
        seconds = times
    else:
        raise ValueError(f"Unsupported time_type: {time_type}")
    
    n = len(prices)
    sampled_indices = []
    
    # Apply the specified sampling scheme
    if sampling_type == 'CalendarTime':
        # Calendar time sampling: sample every X seconds
        interval = sampling_interval if isinstance(sampling_interval, (int, float)) else sampling_interval[0]
        
        # Find time range
        start_time = np.min(seconds)
        end_time = np.max(seconds)
        
        # Create sampling grid
        grid = np.arange(start_time, end_time + interval, interval)
        
        # For each grid point, find the closest observation
        last_idx = 0
        for t in grid:
            # Find index of the closest time point >= t
            idx = last_idx
            while idx < n and seconds[idx] < t:
                idx += 1
            
            if idx < n:
                sampled_indices.append(idx)
                last_idx = idx
    
    elif sampling_type == 'BusinessTime':
        # Business time sampling: sample every X observations
        if isinstance(sampling_interval, tuple):
            min_interval, max_interval = sampling_interval[0], sampling_interval[1]
        else:
            min_interval = max_interval = sampling_interval
        
        idx = 0
        while idx < n:
            sampled_indices.append(idx)
            interval = np.random.randint(min_interval, max_interval + 1)
            idx += interval
    
    elif sampling_type == 'CalendarUniform':
        # Uniform sampling in calendar time
        if isinstance(sampling_interval, tuple):
            num_points = sampling_interval[0]
        else:
            num_points = sampling_interval
        
        # Generate uniformly spaced points in time
        uniform_times = np.linspace(np.min(seconds), np.max(seconds), num_points)
        
        # For each uniform time, find the closest observation
        for t in uniform_times:
            idx = np.argmin(np.abs(seconds - t))
            if idx not in sampled_indices:
                sampled_indices.append(idx)
    
    elif sampling_type == 'BusinessUniform':
        # Uniform sampling in observation count (business time)
        if isinstance(sampling_interval, tuple):
            num_points = sampling_interval[0]
        else:
            num_points = sampling_interval
        
        # Generate uniformly spaced indices
        sampled_indices = np.linspace(0, n - 1, num_points, dtype=int).tolist()
    
    elif sampling_type == 'Fixed':
        # Fixed number of observations with equal spacing
        num_points = sampling_interval
        # Use systematic sampling
        step = max(1, n // num_points)
        sampled_indices = list(range(0, n, step))
    
    else:
        raise ValueError(f"Unsupported sampling_type: {sampling_type}")
    
    # Ensure sampled_indices is sorted and unique
    sampled_indices = sorted(set(sampled_indices))
    
    # Extract sampled prices and times
    sampled_prices = prices[sampled_indices]
    sampled_times = times[sampled_indices]
    
    return sampled_prices, sampled_times

@jit(nopython=True)
def noise_filter(
    prices: np.ndarray,
    returns: np.ndarray,
    filter_type: str,
    filter_params: dict = None
) -> np.ndarray:
    """
    Filters microstructure noise from high-frequency price data.
    
    Parameters
    ----------
    prices : np.ndarray
        Price series corresponding to the returns
    returns : np.ndarray
        Returns series to be filtered
    filter_type : str
        Type of filter to apply: 'MA', 'Kernel', 'HodrickPrescott', 'WaveletThresholding'
    filter_params : dict, default=None
        Parameters specific to the chosen filter
        
    Returns
    -------
    np.ndarray
        Noise-filtered returns
    """
    # Set default parameters if not provided
    if filter_params is None:
        filter_params = {}
    
    n = len(returns)
    filtered_returns = np.zeros_like(returns)
    
    if filter_type == 'MA':
        # Moving average filter
        window = filter_params.get('window', 2)
        
        # Simple centered moving average
        for i in range(n):
            window_start = max(0, i - window // 2)
            window_end = min(n, i + window // 2 + 1)
            filtered_returns[i] = np.mean(returns[window_start:window_end])
    
    elif filter_type == 'Kernel':
        # Kernel smoother
        bandwidth = filter_params.get('bandwidth', 0.1)
        
        # Gaussian kernel
        for i in range(n):
            weights = np.exp(-0.5 * ((np.arange(n) - i) / bandwidth) ** 2)
            weights /= np.sum(weights)
            filtered_returns[i] = np.sum(weights * returns)
    
    elif filter_type == 'HodrickPrescott':
        # Hodrick-Prescott filter
        # This is a simple implementation as Numba doesn't support scipy.signal.hpfilter
        lambda_param = filter_params.get('lambda', 1600)
        
        # Build the HP filter matrix (simplified version)
        eye = np.eye(n)
        D = np.zeros((n-2, n))
        
        for i in range(n-2):
            D[i, i] = 1
            D[i, i+1] = -2
            D[i, i+2] = 1
        
        HP = eye + lambda_param * D.T @ D
        
        # Solve the system (approximate in Numba)
        # In full implementation, this would use scipy.linalg.solve
        filtered_returns = returns.copy()
        for _ in range(10):  # Simplified iterative solution
            for i in range(n):
                s = returns[i]
                for j in range(n):
                    if i != j:
                        s -= HP[i, j] * filtered_returns[j]
                filtered_returns[i] = s / HP[i, i]
    
    elif filter_type == 'WaveletThresholding':
        # Simplified wavelet thresholding for Numba compatibility
        threshold = filter_params.get('threshold', 0.05)
        
        # Apply simple hard thresholding to returns
        # In a full implementation, this would use proper wavelet transforms
        mean_abs_return = np.mean(np.abs(returns))
        threshold_value = threshold * mean_abs_return
        
        for i in range(n):
            if np.abs(returns[i]) < threshold_value:
                filtered_returns[i] = 0
            else:
                filtered_returns[i] = returns[i]
    
    else:
        # Default: return unfiltered
        filtered_returns = returns.copy()
    
    return filtered_returns

def preprocess_price_data(
    prices: np.ndarray or pd.Series,
    times: np.ndarray or pd.Series,
    time_type: str,
    detect_outliers: bool = True,
    threshold: float = 3.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Preprocesses price data for realized volatility computation, including outlier detection and handling.
    
    Parameters
    ----------
    prices : np.ndarray or pd.Series
        High-frequency price series
    times : np.ndarray or pd.Series
        Timestamps corresponding to each price observation
    time_type : str
        Type of time data: 'datetime', 'seconds', or 'businesstime'
    detect_outliers : bool, default=True
        If True, detects and removes outliers
    threshold : float, default=3.0
        Threshold for outlier detection (number of standard deviations)
        
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Preprocessed prices and times
    """
    # Validate inputs
    validate_input({
        'time_type': {'type': str, 'allowed_values': ['datetime', 'seconds', 'businesstime']}
    })
    
    # Convert to numpy arrays
    prices = ensure_array(prices)
    times = ensure_array(times)
    
    # Ensure all prices are positive
    if np.any(prices <= 0):
        raise ValueError("Price data contains zero or negative values")
    
    # Convert prices to log scale
    log_prices = np.log(prices)
    
    # Calculate returns for outlier detection
    returns = np.diff(log_prices)
    
    # Detect and handle outliers if requested
    if detect_outliers:
        # Calculate z-scores for returns
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        z_scores = np.abs((returns - mean_return) / std_return)
        
        # Identify outliers
        outlier_indices = np.where(z_scores > threshold)[0]
        
        if len(outlier_indices) > 0:
            # Create mask for good data points
            good_mask = np.ones(len(prices), dtype=bool)
            
            # Mark outlier points and the subsequent points (affected by the outlier return)
            for idx in outlier_indices:
                good_mask[idx] = False
                good_mask[idx + 1] = False
            
            # Filter out outliers
            filtered_prices = prices[good_mask]
            filtered_times = times[good_mask]
            
            # Log the outlier removal
            outlier_percent = 100 * len(outlier_indices) / len(returns)
            print(f"Removed {len(outlier_indices)} outliers ({outlier_percent:.2f}% of data)")
            
            return filtered_prices, filtered_times
    
    # If no outlier detection or no outliers found, return original data
    return prices, times

async def async_realized_variance(
    prices: np.ndarray,
    times: np.ndarray or pd.Series,
    time_type: str,
    sampling_type: str,
    sampling_interval=DEFAULT_SAMPLING_INTERVAL,
    noise_adjust: bool = False
) -> tuple[float, float]:
    """
    Asynchronous version of realized_variance for handling long-running computations.
    
    Parameters
    ----------
    prices : np.ndarray
        High-frequency price series
    times : np.ndarray or pd.Series
        Timestamps corresponding to each price observation
    time_type : str
        Type of time data: 'datetime', 'seconds', or 'businesstime'
    sampling_type : str
        Method for sampling: 'CalendarTime', 'BusinessTime', 'CalendarUniform', 
        'BusinessUniform', or 'Fixed'
    sampling_interval : int or tuple, default=DEFAULT_SAMPLING_INTERVAL
        Sampling interval specification, interpretation depends on sampling_type
    noise_adjust : bool, default=False
        If True, applies noise filtering to the returns
        
    Returns
    -------
    tuple[float, float]
        Realized variance and subsampled realized variance
    """
    # Validate inputs asynchronously
    def run_calculation():
        return realized_variance(prices, times, time_type, sampling_type, 
                               sampling_interval, noise_adjust)
    
    # Run the calculation in a separate thread to avoid blocking
    return await async_process(run_calculation)

def realized_covariance(
    prices_1: np.ndarray,
    prices_2: np.ndarray,
    times: np.ndarray or pd.Series,
    time_type: str,
    sampling_type: str,
    sampling_interval=DEFAULT_SAMPLING_INTERVAL
) -> float:
    """
    Computes the realized covariance between multiple high-frequency price series.
    
    Parameters
    ----------
    prices_1 : np.ndarray
        First high-frequency price series
    prices_2 : np.ndarray
        Second high-frequency price series
    times : np.ndarray or pd.Series
        Timestamps corresponding to each price observation
    time_type : str
        Type of time data: 'datetime', 'seconds', or 'businesstime'
    sampling_type : str
        Method for sampling: 'CalendarTime', 'BusinessTime', 'CalendarUniform', 
        'BusinessUniform', or 'Fixed'
    sampling_interval : int or tuple, default=DEFAULT_SAMPLING_INTERVAL
        Sampling interval specification, interpretation depends on sampling_type
        
    Returns
    -------
    float
        Realized covariance estimate
    """
    # Validate inputs
    validate_input({
        'time_type': {'type': str, 'allowed_values': ['datetime', 'seconds', 'businesstime']},
        'sampling_type': {'type': str, 'allowed_values': 
                         ['CalendarTime', 'BusinessTime', 'CalendarUniform', 'BusinessUniform', 'Fixed']}
    })
    
    # Convert to numpy arrays
    prices_1 = ensure_array(prices_1)
    prices_2 = ensure_array(prices_2)
    times = ensure_array(times)
    
    # Ensure all series have the same length
    if len(prices_1) != len(prices_2) or len(prices_1) != len(times):
        raise ValueError("All input series must have the same length")
    
    # Sample the price data
    sampled_prices_1, sampled_times = sampling_scheme(
        prices_1, times, time_type, sampling_type, sampling_interval
    )
    
    sampled_prices_2, _ = sampling_scheme(
        prices_2, times, time_type, sampling_type, sampling_interval
    )
    
    # Calculate returns (log price differences)
    returns_1 = np.diff(np.log(sampled_prices_1))
    returns_2 = np.diff(np.log(sampled_prices_2))
    
    # Compute product of returns
    return_products = returns_1 * returns_2
    
    # Sum products to get realized covariance
    rcov = np.sum(return_products)
    
    return rcov

def realized_volatility(
    prices: np.ndarray,
    times: np.ndarray or pd.Series,
    time_type: str,
    sampling_type: str,
    sampling_interval=DEFAULT_SAMPLING_INTERVAL,
    noise_adjust: bool = False,
    annualize: bool = False,
    scale: float = 252
) -> tuple[float, float]:
    """
    Computes the realized volatility (square root of realized variance) of a price series.
    
    Parameters
    ----------
    prices : np.ndarray
        High-frequency price series
    times : np.ndarray or pd.Series
        Timestamps corresponding to each price observation
    time_type : str
        Type of time data: 'datetime', 'seconds', or 'businesstime'
    sampling_type : str
        Method for sampling: 'CalendarTime', 'BusinessTime', 'CalendarUniform', 
        'BusinessUniform', or 'Fixed'
    sampling_interval : int or tuple, default=DEFAULT_SAMPLING_INTERVAL
        Sampling interval specification, interpretation depends on sampling_type
    noise_adjust : bool, default=False
        If True, applies noise filtering to the returns
    annualize : bool, default=False
        If True, annualizes the volatility
    scale : float, default=252
        Annualization factor (252 for daily data -> annual)
        
    Returns
    -------
    tuple[float, float]
        Realized volatility and its subsampled version
    """
    # Get variance measures
    rv, rv_ss = realized_variance(
        prices, times, time_type, sampling_type, sampling_interval, noise_adjust
    )
    
    # Take square root to get volatility
    vol = np.sqrt(rv)
    vol_ss = np.sqrt(rv_ss)
    
    # Annualize if requested
    if annualize:
        if time_type == 'datetime':
            # Determine the sampling frequency in days
            times_pd = pd.DatetimeIndex(times)
            time_range = (times_pd.max() - times_pd.min()).total_seconds() / 86400  # Convert to days
            n_periods = len(prices) / time_range  # Average number of observations per day
            
            # Adjust the scale based on the sampling frequency
            adj_scale = scale / n_periods
        else:
            # For other time types, use the provided scale directly
            adj_scale = scale
        
        vol *= np.sqrt(adj_scale)
        vol_ss *= np.sqrt(adj_scale)
    
    return vol, vol_ss

def get_realized_measures(
    prices: np.ndarray or pd.Series,
    times: np.ndarray or pd.Series,
    time_type: str,
    measures: list,
    params: dict = None
) -> dict:
    """
    Main function for computing various realized measures including variance, volatility, and kernel estimates.
    
    Parameters
    ----------
    prices : np.ndarray or pd.Series
        High-frequency price series
    times : np.ndarray or pd.Series
        Timestamps corresponding to each price observation
    time_type : str
        Type of time data: 'datetime', 'seconds', or 'businesstime'
    measures : list
        List of measures to compute: 'variance', 'volatility', 'kernel', 'covariance'
    params : dict, default=None
        Dictionary of parameters for each measure
        
    Returns
    -------
    dict
        Dictionary of computed realized measures
    """
    # Validate inputs
    validate_input({
        'time_type': {'type': str, 'allowed_values': ['datetime', 'seconds', 'businesstime']},
        'measures': {'type': list}
    })
    
    # Set default parameters if not provided
    if params is None:
        params = {}
    
    # Preprocess price data
    detect_outliers = params.get('detect_outliers', True)
    threshold = params.get('outlier_threshold', 3.0)
    
    preprocessed_prices, preprocessed_times = preprocess_price_data(
        prices, times, time_type, detect_outliers, threshold
    )
    
    # Initialize results dictionary
    results = {}
    
    # Compute requested measures
    if 'variance' in measures:
        sampling_type = params.get('sampling_type', 'CalendarTime')
        sampling_interval = params.get('sampling_interval', DEFAULT_SAMPLING_INTERVAL)
        noise_adjust = params.get('noise_adjust', False)
        
        rv, rv_ss = realized_variance(
            preprocessed_prices, preprocessed_times, time_type, 
            sampling_type, sampling_interval, noise_adjust
        )
        
        results['variance'] = rv
        results['variance_subsampled'] = rv_ss
    
    if 'volatility' in measures:
        sampling_type = params.get('sampling_type', 'CalendarTime')
        sampling_interval = params.get('sampling_interval', DEFAULT_SAMPLING_INTERVAL)
        noise_adjust = params.get('noise_adjust', False)
        annualize = params.get('annualize', False)
        scale = params.get('scale', 252)
        
        vol, vol_ss = realized_volatility(
            preprocessed_prices, preprocessed_times, time_type, 
            sampling_type, sampling_interval, noise_adjust, annualize, scale
        )
        
        results['volatility'] = vol
        results['volatility_subsampled'] = vol_ss
    
    if 'kernel' in measures:
        kernel_type = params.get('kernel_type', DEFAULT_KERNEL_TYPE)
        bandwidth = params.get('bandwidth', None)
        
        rk = realized_kernel(
            preprocessed_prices, preprocessed_times, time_type, kernel_type, bandwidth
        )
        
        results['kernel'] = rk
    
    if 'covariance' in measures:
        if 'prices_2' not in params:
            raise ValueError("Second price series (prices_2) required for covariance calculation")
        
        prices_2 = params['prices_2']
        sampling_type = params.get('sampling_type', 'CalendarTime')
        sampling_interval = params.get('sampling_interval', DEFAULT_SAMPLING_INTERVAL)
        
        # Preprocess the second price series
        preprocessed_prices_2, _ = preprocess_price_data(
            prices_2, times, time_type, detect_outliers, threshold
        )
        
        rcov = realized_covariance(
            preprocessed_prices, preprocessed_prices_2, preprocessed_times, 
            time_type, sampling_type, sampling_interval
        )
        
        results['covariance'] = rcov
    
    return results

class RealizedVolatility:
    """
    Class for computing and analyzing realized volatility measures from high-frequency financial data.
    
    This class provides a convenient interface for calculating various realized volatility
    measures, with support for different sampling schemes and noise filtering methods.
    """
    
    def __init__(self):
        """
        Initializes the RealizedVolatility class with default parameters.
        """
        # Initialize empty properties
        self._prices = None
        self._times = None
        self._time_type = None
        self._params = {
            'sampling_type': 'CalendarTime',
            'sampling_interval': DEFAULT_SAMPLING_INTERVAL,
            'noise_adjust': False,
            'kernel_type': DEFAULT_KERNEL_TYPE,
            'detect_outliers': True,
            'outlier_threshold': 3.0,
            'annualize': False,
            'scale': 252
        }
        self._results = {}
    
    def set_data(self, prices, times, time_type):
        """
        Sets the price and time data for analysis.
        
        Parameters
        ----------
        prices : np.ndarray or pd.Series
            High-frequency price series
        times : np.ndarray or pd.Series
            Timestamps corresponding to each price observation
        time_type : str
            Type of time data: 'datetime', 'seconds', or 'businesstime'
            
        Returns
        -------
        None
        """
        # Validate inputs
        validate_input({
            'time_type': {'type': str, 'allowed_values': ['datetime', 'seconds', 'businesstime']}
        })
        
        # Convert time series to appropriate format
        self._prices = ensure_array(prices)
        self._times = ensure_array(times)
        self._time_type = time_type
        
        # Reset results when new data is set
        self._results = {}
    
    def set_params(self, params):
        """
        Sets parameters for realized volatility computation.
        
        Parameters
        ----------
        params : dict
            Dictionary of parameters for volatility calculations
            
        Returns
        -------
        None
        """
        # Validate that params is a dictionary
        if not isinstance(params, dict):
            raise TypeError("params must be a dictionary")
        
        # Update parameters
        self._params.update(params)
    
    def compute(self, measures):
        """
        Computes specified realized measures.
        
        Parameters
        ----------
        measures : list
            List of measures to compute: 'variance', 'volatility', 'kernel', 'covariance'
            
        Returns
        -------
        dict
            Dictionary of computed realized measures
        """
        # Validate that data has been set
        if self._prices is None or self._times is None or self._time_type is None:
            raise ValueError("Data must be set using set_data() before computation")
        
        # Call get_realized_measures with instance data and parameters
        results = get_realized_measures(
            self._prices, self._times, self._time_type, measures, self._params
        )
        
        # Store results
        self._results.update(results)
        
        return results
    
    async def compute_async(self, measures):
        """
        Asynchronously computes specified realized measures.
        
        Parameters
        ----------
        measures : list
            List of measures to compute: 'variance', 'volatility', 'kernel', 'covariance'
            
        Returns
        -------
        dict
            Async coroutine returning dictionary of computed realized measures
        """
        # Validate that data has been set
        if self._prices is None or self._times is None or self._time_type is None:
            raise ValueError("Data must be set using set_data() before computation")
        
        # Define function to run in a separate thread
        def run_calculation():
            return get_realized_measures(
                self._prices, self._times, self._time_type, measures, self._params
            )
        
        # Run the calculation asynchronously
        results = await async_process(run_calculation)
        
        # Store results
        self._results.update(results)
        
        return results
    
    def get_results(self):
        """
        Returns the computed realized measures.
        
        Returns
        -------
        dict
            Dictionary of computed realized measures
        """
        return self._results.copy()
    
    def clear(self):
        """
        Clears all data and results.
        
        Returns
        -------
        None
        """
        self._prices = None
        self._times = None
        self._time_type = None
        self._results = {}
        # Reset parameters to defaults
        self._params = {
            'sampling_type': 'CalendarTime',
            'sampling_interval': DEFAULT_SAMPLING_INTERVAL,
            'noise_adjust': False,
            'kernel_type': DEFAULT_KERNEL_TYPE,
            'detect_outliers': True,
            'outlier_threshold': 3.0,
            'annualize': False,
            'scale': 252
        }