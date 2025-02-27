"""
MFE Toolbox - Testing Framework

This module provides the core testing infrastructure for the MFE Toolbox, 
including environment setup, path configuration, shared test fixtures, and 
utility functions for testing Numba-optimized implementations.

The testing framework is built on pytest with hypothesis for property-based
testing and numba.testing for validating performance-critical routines.
"""

import os
import sys
import logging
from typing import Any, Callable, Dict, Optional, Union, TypeVar

import pytest
import numpy as np  # numpy 1.26.3
import pandas as pd  # pandas 2.1.4
from hypothesis import strategies as st
from pathlib import Path

# Try to import numba.testing, but don't fail if numba is not available
try:
    import numba.testing
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

# Import internal utilities
from ..mfe.utils.validation import validate_params, validate_array
from ..mfe.initialize import check_numba_availability
from ..mfe.utils.data_handling import to_numpy_array

# Define constants for test paths
TEST_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(TEST_ROOT_DIR, 'test_data')

# Set up test logger
TEST_LOGGER = logging.getLogger('mfe.tests')

# Define exported functions
__all__ = [
    'check_test_environment', 
    'get_test_data_path', 
    'setup_test_environment', 
    'prepare_test_data', 
    'requires_numba'
]

# Type variable for function decorators
F = TypeVar('F', bound=Callable[..., Any])


def check_test_environment(check_numba: bool = True) -> bool:
    """
    Verifies that the test environment is properly configured for MFE Toolbox testing.
    
    Parameters
    ----------
    check_numba : bool, default=True
        Whether to check if Numba is available and functioning
        
    Returns
    -------
    bool
        True if the environment is properly configured, False otherwise
    """
    # Check Python version
    python_version = sys.version_info[:2]
    if python_version < (3, 12):
        TEST_LOGGER.error(f"Python version check failed. Minimum required: 3.12, got {python_version[0]}.{python_version[1]}")
        return False
    
    # Check required packages
    required_packages = {
        'numpy': '1.26.3',
        'pandas': '2.1.4', 
        'pytest': '7.4.3',
        'hypothesis': '6.92.1'
    }
    
    for package_name, min_version in required_packages.items():
        try:
            module = __import__(package_name)
            if not hasattr(module, '__version__'):
                TEST_LOGGER.warning(f"Package {package_name} has no version attribute")
                continue
                
            version = module.__version__
            if version < min_version:
                TEST_LOGGER.warning(f"Package {package_name} version {version} is below minimum required {min_version}")
        except ImportError:
            TEST_LOGGER.error(f"Required package {package_name} is not installed")
            return False
    
    # Check Numba if requested
    if check_numba:
        if not HAS_NUMBA:
            TEST_LOGGER.warning("Numba is not available. Tests requiring Numba will be skipped.")
        elif not check_numba_availability():
            TEST_LOGGER.warning("Numba is installed but JIT compilation is not working properly")
    
    # Verify test data directory exists
    if not os.path.exists(TEST_DATA_DIR):
        try:
            os.makedirs(TEST_DATA_DIR)
            TEST_LOGGER.info(f"Created test data directory: {TEST_DATA_DIR}")
        except Exception as e:
            TEST_LOGGER.error(f"Failed to create test data directory: {str(e)}")
            return False
    
    TEST_LOGGER.info("Test environment check completed successfully")
    return True


def get_test_data_path(filename: str, subdir: str = '') -> str:
    """
    Returns the absolute path to a test data file.
    
    Parameters
    ----------
    filename : str
        Name of the test data file
    subdir : str, default=''
        Subdirectory within the test data directory
        
    Returns
    -------
    str
        Absolute path to the test data file
    """
    if subdir:
        path = os.path.join(TEST_DATA_DIR, subdir, filename)
    else:
        path = os.path.join(TEST_DATA_DIR, filename)
    
    if not os.path.exists(path):
        TEST_LOGGER.warning(f"Test data file not found: {path}")
    
    return os.path.abspath(path)


def setup_test_environment(verbose: bool = False) -> bool:
    """
    Sets up the test environment for the MFE Toolbox tests.
    
    Parameters
    ----------
    verbose : bool, default=False
        If True, configures more verbose logging
        
    Returns
    -------
    bool
        True if setup was successful, False otherwise
    """
    # Configure logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=log_level, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Verify environment configuration
    if not check_test_environment():
        TEST_LOGGER.error("Test environment verification failed")
        return False
    
    # Create test data directory if it doesn't exist
    if not os.path.exists(TEST_DATA_DIR):
        try:
            os.makedirs(TEST_DATA_DIR)
        except Exception as e:
            TEST_LOGGER.error(f"Failed to create test data directory: {str(e)}")
            return False
    
    # Configure pytest warnings
    if 'PYTHONWARNINGS' not in os.environ:
        os.environ['PYTHONWARNINGS'] = 'ignore::DeprecationWarning'
    
    TEST_LOGGER.info("Test environment setup completed successfully")
    return True


def prepare_test_data(
    data_type: str, 
    params: Dict[str, Any] = None
) -> Union[np.ndarray, pd.DataFrame]:
    """
    Prepares test data for a specific test case.
    
    Parameters
    ----------
    data_type : str
        Type of test data to prepare:
        - 'returns': Financial return series
        - 'prices': Price series
        - 'volatility': Volatility series
        - 'timeseries': Generic time series
        - 'high_frequency': High-frequency data
    params : Dict[str, Any], default=None
        Parameters for data generation:
        - 'size': int, number of observations
        - 'dimensions': int, number of variables
        - 'seed': int, random seed
        - 'frequencies': List[str], for time series data
        - Other parameters specific to each data_type
        
    Returns
    -------
    Union[np.ndarray, pd.DataFrame]
        Prepared test data
    """
    if params is None:
        params = {}
    
    # Set default parameters
    size = params.get('size', 1000)
    dimensions = params.get('dimensions', 1)
    seed = params.get('seed', 42)
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    if data_type == 'returns':
        # Generate financial return series
        mean = params.get('mean', 0.0001)
        std = params.get('std', 0.01)
        
        if dimensions == 1:
            # Univariate returns
            data = np.random.normal(mean, std, size)
            
            if params.get('as_dataframe', False):
                date_range = pd.date_range(start='2020-01-01', periods=size, freq='D')
                return pd.DataFrame({'returns': data}, index=date_range)
            return data
        else:
            # Multivariate returns
            data = np.random.normal(mean, std, (size, dimensions))
            
            if params.get('as_dataframe', False):
                date_range = pd.date_range(start='2020-01-01', periods=size, freq='D')
                columns = [f'asset_{i}' for i in range(dimensions)]
                return pd.DataFrame(data, index=date_range, columns=columns)
            return data
            
    elif data_type == 'prices':
        # Generate price series
        start_price = params.get('start_price', 100.0)
        volatility = params.get('volatility', 0.01)
        drift = params.get('drift', 0.0001)
        
        if dimensions == 1:
            # Univariate price series
            returns = np.random.normal(drift, volatility, size)
            prices = start_price * np.cumprod(1 + returns)
            
            if params.get('as_dataframe', False):
                date_range = pd.date_range(start='2020-01-01', periods=size, freq='D')
                return pd.DataFrame({'price': prices}, index=date_range)
            return prices
        else:
            # Multivariate price series
            prices = np.zeros((size, dimensions))
            for i in range(dimensions):
                asset_drift = drift * (0.8 + 0.4 * np.random.random())
                asset_vol = volatility * (0.8 + 0.4 * np.random.random())
                asset_start = start_price * (0.8 + 0.4 * np.random.random())
                
                returns = np.random.normal(asset_drift, asset_vol, size)
                prices[:, i] = asset_start * np.cumprod(1 + returns)
            
            if params.get('as_dataframe', False):
                date_range = pd.date_range(start='2020-01-01', periods=size, freq='D')
                columns = [f'asset_{i}' for i in range(dimensions)]
                return pd.DataFrame(prices, index=date_range, columns=columns)
            return prices
            
    elif data_type == 'volatility':
        # Generate volatility series with GARCH-like properties
        persistence = params.get('persistence', 0.95)
        baseline = params.get('baseline', 0.0001)
        shock_size = params.get('shock_size', 0.05)
        
        if dimensions == 1:
            # Univariate volatility
            volatility = np.zeros(size)
            volatility[0] = baseline
            
            for t in range(1, size):
                # Random shock
                if np.random.random() < 0.05:
                    shock = shock_size
                else:
                    shock = 0
                
                # GARCH-like process
                volatility[t] = baseline + persistence * volatility[t-1] + shock
            
            if params.get('as_dataframe', False):
                date_range = pd.date_range(start='2020-01-01', periods=size, freq='D')
                return pd.DataFrame({'volatility': volatility}, index=date_range)
            return volatility
        else:
            # Multivariate volatility
            volatility = np.zeros((size, dimensions))
            
            for i in range(dimensions):
                asset_persistence = persistence * (0.9 + 0.1 * np.random.random())
                asset_baseline = baseline * (0.9 + 0.1 * np.random.random())
                
                volatility[0, i] = asset_baseline
                
                for t in range(1, size):
                    if np.random.random() < 0.05:
                        shock = shock_size * (0.9 + 0.2 * np.random.random())
                    else:
                        shock = 0
                    
                    volatility[t, i] = asset_baseline + asset_persistence * volatility[t-1, i] + shock
            
            if params.get('as_dataframe', False):
                date_range = pd.date_range(start='2020-01-01', periods=size, freq='D')
                columns = [f'asset_{i}' for i in range(dimensions)]
                return pd.DataFrame(volatility, index=date_range, columns=columns)
            return volatility
            
    elif data_type == 'timeseries':
        # Generate generic time series with trend, seasonality, and noise
        trend = params.get('trend', 0.001)
        seasonality = params.get('seasonality', 0.1)
        cycle_length = params.get('cycle_length', 20)
        noise_level = params.get('noise_level', 0.02)
        
        if dimensions == 1:
            # Generate components
            time = np.arange(size)
            trend_component = trend * time
            seasonal_component = seasonality * np.sin(2 * np.pi * time / cycle_length)
            noise_component = noise_level * np.random.randn(size)
            
            # Combine components
            series = trend_component + seasonal_component + noise_component
            
            if params.get('as_dataframe', False):
                date_range = pd.date_range(start='2020-01-01', periods=size, freq='D')
                return pd.DataFrame({'value': series}, index=date_range)
            return series
        else:
            # Multivariate time series
            series = np.zeros((size, dimensions))
            
            for i in range(dimensions):
                # Slightly different parameters for each dimension
                dim_trend = trend * (0.8 + 0.4 * np.random.random())
                dim_seasonality = seasonality * (0.8 + 0.4 * np.random.random())
                dim_cycle = cycle_length * (0.8 + 0.4 * np.random.random())
                dim_noise = noise_level * (0.8 + 0.4 * np.random.random())
                
                # Generate components
                time = np.arange(size)
                trend_component = dim_trend * time
                seasonal_component = dim_seasonality * np.sin(2 * np.pi * time / dim_cycle)
                noise_component = dim_noise * np.random.randn(size)
                
                # Combine components
                series[:, i] = trend_component + seasonal_component + noise_component
            
            if params.get('as_dataframe', False):
                date_range = pd.date_range(start='2020-01-01', periods=size, freq='D')
                columns = [f'series_{i}' for i in range(dimensions)]
                return pd.DataFrame(series, index=date_range, columns=columns)
            return series
            
    elif data_type == 'high_frequency':
        # Generate high-frequency financial data
        frequencies = params.get('frequencies', ['1min', '5min', '15min', '30min', 'H'])
        frequency = params.get('frequency', '1min')
        trading_hours = params.get('trading_hours', 8)  # hours per day
        days = params.get('days', 5)
        
        if frequency not in frequencies:
            frequency = '1min'
        
        # Calculate total observations based on frequency and period
        freq_minutes = {'1min': 1, '5min': 5, '15min': 15, '30min': 30, 'H': 60}
        minutes_per_day = trading_hours * 60
        obs_per_day = minutes_per_day // freq_minutes[frequency]
        total_obs = obs_per_day * days
        
        # Generate timestamps
        start_date = params.get('start_date', '2020-01-01 09:30:00')
        timestamps = pd.date_range(start=start_date, periods=total_obs, freq=frequency)
        
        # Generate price data (random walk with drift)
        start_price = params.get('start_price', 100.0)
        volatility = params.get('volatility', 0.0005)  # Reduced for high-frequency
        drift = params.get('drift', 0.00001)  # Reduced for high-frequency
        
        if dimensions == 1:
            # Univariate high-frequency data
            returns = np.random.normal(drift, volatility, total_obs)
            prices = start_price * np.cumprod(1 + returns)
            
            # Generate volume
            vol_mean = params.get('volume_mean', 1000)
            vol_std = params.get('volume_std', 200)
            volume = np.abs(np.random.normal(vol_mean, vol_std, total_obs)).astype(int)
            
            return pd.DataFrame({
                'price': prices,
                'volume': volume,
                'return': returns
            }, index=timestamps)
        else:
            # Multivariate high-frequency data
            data = {}
            
            for i in range(dimensions):
                asset_drift = drift * (0.8 + 0.4 * np.random.random())
                asset_vol = volatility * (0.8 + 0.4 * np.random.random())
                asset_start = start_price * (0.8 + 0.4 * np.random.random())
                
                returns = np.random.normal(asset_drift, asset_vol, total_obs)
                prices = asset_start * np.cumprod(1 + returns)
                
                # Generate volume
                vol_mean = params.get('volume_mean', 1000) * (0.8 + 0.4 * np.random.random())
                vol_std = params.get('volume_std', 200) * (0.8 + 0.4 * np.random.random())
                volume = np.abs(np.random.normal(vol_mean, vol_std, total_obs)).astype(int)
                
                data[f'price_{i}'] = prices
                data[f'volume_{i}'] = volume
                data[f'return_{i}'] = returns
            
            return pd.DataFrame(data, index=timestamps)
    
    else:
        raise ValueError(f"Unknown data_type: {data_type}")


def requires_numba(test_func: F) -> F:
    """
    Decorator for tests that require Numba functionality.
    
    Parameters
    ----------
    test_func : Callable
        Test function that requires Numba
        
    Returns
    -------
    Callable
        Wrapped test function that skips if Numba is unavailable
    """
    # Check if Numba is available and working
    skip_test = not (HAS_NUMBA and check_numba_availability())
    reason = "Numba is not available or JIT compilation is not working"
    
    # Apply pytest.mark.skipif decorator if Numba is not available
    return pytest.mark.skipif(skip_test, reason=reason)(test_func)


# Initialize test environment when module is imported
setup_test_environment()