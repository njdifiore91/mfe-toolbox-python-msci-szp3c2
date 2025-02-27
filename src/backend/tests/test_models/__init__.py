"""
MFE Toolbox - Models Test Package

This module provides testing utilities and fixtures specifically for financial
model tests in the MFE Toolbox. It configures pytest markers for model-specific
tests including time series, volatility, and high-frequency model tests.
"""

import logging
import numpy as np  # numpy 1.26.3
import pytest  # pytest 7.4.3
from typing import Any, Callable, Dict, List, Optional, Union

# Import test utilities from parent package
from .. import check_test_environment, requires_numba, get_test_data_path

# Set up logger for this test package
logger = logging.getLogger('mfe.tests.models')

# Exported functions
__all__ = [
    'parametrize_model_tests',
    'validate_model_params', 
    'check_model_convergence',
    'create_model_test_data'
]

# Mark all tests in this package with 'models' marker
pytestmark = [pytest.mark.models]


def parametrize_model_tests(model_configs: list, include_numba_tests: bool = True) -> Callable:
    """
    Helper function for parametrizing model test cases with common test configurations.
    
    Parameters
    ----------
    model_configs : list
        List of model configurations to test
    include_numba_tests : bool, default=True
        Whether to include tests that require Numba
        
    Returns
    -------
    Callable
        Decorator function that can be applied to test functions
    """
    def decorator(func):
        # Apply pytest parametrize decorator
        parametrized_func = pytest.mark.parametrize('model_config', model_configs)(func)
        
        # Apply appropriate model-specific marker based on the function name or docstring
        if any(kw in func.__name__ for kw in ['arma', 'arima', 'time_series']):
            parametrized_func = pytest.mark.time_series(parametrized_func)
        elif any(kw in func.__name__ for kw in ['garch', 'arch', 'volatility']):
            parametrized_func = pytest.mark.volatility(parametrized_func)
        elif any(kw in func.__name__ for kw in ['realized', 'high_frequency', 'hf']):
            parametrized_func = pytest.mark.high_frequency(parametrized_func)
        
        # Apply Numba decorator if needed
        if include_numba_tests:
            parametrized_func = requires_numba(parametrized_func)
        
        return parametrized_func
    
    return decorator


def validate_model_params(params: dict, model_type: str) -> bool:
    """
    Validate model parameters for testing to ensure they're within reasonable bounds.
    
    Parameters
    ----------
    params : dict
        Dictionary of model parameters to validate
    model_type : str
        Type of model ('time_series', 'volatility', 'high_frequency')
        
    Returns
    -------
    bool
        True if parameters are valid, otherwise raises appropriate exception
    """
    if not isinstance(params, dict):
        raise TypeError("Parameters must be provided as a dictionary")
    
    # Validate based on model type
    if model_type.lower() in ('time_series', 'arma', 'arima', 'armax'):
        # Validate AR/MA orders, constants, etc.
        if 'ar_order' in params and not isinstance(params['ar_order'], int):
            raise ValueError(f"AR order must be an integer, got {type(params['ar_order'])}")
        
        if 'ma_order' in params and not isinstance(params['ma_order'], int):
            raise ValueError(f"MA order must be an integer, got {type(params['ma_order'])}")
        
        if 'ar_order' in params and params['ar_order'] < 0:
            raise ValueError(f"AR order must be non-negative, got {params['ar_order']}")
        
        if 'ma_order' in params and params['ma_order'] < 0:
            raise ValueError(f"MA order must be non-negative, got {params['ma_order']}")
        
        if 'constant' in params and not isinstance(params['constant'], bool):
            raise ValueError(f"Constant parameter must be boolean, got {type(params['constant'])}")
        
    elif model_type.lower() in ('volatility', 'garch', 'arch', 'egarch'):
        # Validate GARCH parameters
        if 'p' in params and not isinstance(params['p'], int):
            raise ValueError(f"GARCH p parameter must be an integer, got {type(params['p'])}")
        
        if 'q' in params and not isinstance(params['q'], int):
            raise ValueError(f"GARCH q parameter must be an integer, got {type(params['q'])}")
        
        if 'p' in params and params['p'] < 0:
            raise ValueError(f"GARCH p parameter must be non-negative, got {params['p']}")
        
        if 'q' in params and params['q'] < 0:
            raise ValueError(f"GARCH q parameter must be non-negative, got {params['q']}")
        
        if 'distribution' in params and not isinstance(params['distribution'], str):
            raise ValueError(f"Distribution parameter must be a string, got {type(params['distribution'])}")
        
    elif model_type.lower() in ('high_frequency', 'realized', 'rv', 'realized_volatility'):
        # Validate high-frequency parameters
        if 'sampling_type' in params and not isinstance(params['sampling_type'], str):
            raise ValueError(f"Sampling type must be a string, got {type(params['sampling_type'])}")
            
        if 'kernel_type' in params and not isinstance(params['kernel_type'], str):
            raise ValueError(f"Kernel type must be a string, got {type(params['kernel_type'])}")
            
        if 'bandwidth' in params and not isinstance(params['bandwidth'], (int, float)):
            raise ValueError(f"Bandwidth must be numeric, got {type(params['bandwidth'])}")
            
        if 'bandwidth' in params and params['bandwidth'] <= 0:
            raise ValueError(f"Bandwidth must be positive, got {params['bandwidth']}")
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return True


def check_model_convergence(model_result: object, tolerance: float = 1e-6) -> bool:
    """
    Verify model optimization convergence and parameter stability.
    
    Parameters
    ----------
    model_result : object
        Model result object containing convergence information
    tolerance : float, default=1e-6
        Numerical tolerance for convergence checks
        
    Returns
    -------
    bool
        True if model converged successfully within tolerance
    """
    # Check for convergence attribute
    if hasattr(model_result, 'converged'):
        if not model_result.converged:
            logger.warning("Model did not converge according to convergence flag")
            return False
    
    # Check number of iterations if available
    if hasattr(model_result, 'iterations'):
        max_iterations = getattr(model_result, 'max_iterations', 1000)
        if model_result.iterations >= max_iterations:
            logger.warning(f"Model reached maximum iterations ({max_iterations})")
            return False
    
    # Check parameter standard errors if available
    if hasattr(model_result, 'std_errors'):
        if np.any(np.isnan(model_result.std_errors)):
            logger.warning("Some parameter standard errors are NaN")
            return False
        
        if np.any(np.abs(model_result.std_errors) > 1/tolerance):
            logger.warning("Some parameter standard errors are extremely large")
            return False
    
    # Check log-likelihood if available
    if hasattr(model_result, 'loglikelihood'):
        if not np.isfinite(model_result.loglikelihood):
            logger.warning("Log-likelihood is not finite")
            return False
    
    return True


def create_model_test_data(model_type: str, sample_size: int, properties: dict = None) -> np.ndarray:
    """
    Create standardized test data for different model types.
    
    Parameters
    ----------
    model_type : str
        Type of model to generate data for ('arma', 'garch', 'realized')
    sample_size : int
        Size of the test data to generate
    properties : dict, default=None
        Additional properties for the generated data
        
    Returns
    -------
    np.ndarray
        Generated test data appropriate for the specified model type
    """
    if properties is None:
        properties = {}
    
    # Seed for reproducibility
    seed = properties.get('seed', 12345)
    np.random.seed(seed)
    
    if model_type.lower() in ('arma', 'arima', 'armax', 'time_series'):
        # Create time series with AR/MA properties
        ar_coefs = properties.get('ar_coefs', [0.7])
        ma_coefs = properties.get('ma_coefs', [0.3])
        const = properties.get('const', 0.0)
        sigma = properties.get('sigma', 1.0)
        
        # Create innovation process
        innovations = np.random.normal(0, sigma, sample_size + max(len(ar_coefs), len(ma_coefs)))
        data = np.zeros_like(innovations)
        
        # Apply ARMA process
        for t in range(max(len(ar_coefs), len(ma_coefs)), len(data)):
            # AR component
            ar_component = sum(ar_coefs[i] * data[t-i-1] for i in range(min(t, len(ar_coefs))))
            
            # MA component
            ma_component = sum(ma_coefs[i] * innovations[t-i-1] for i in range(min(t, len(ma_coefs))))
            
            # Combine components
            data[t] = const + ar_component + ma_component + innovations[t]
        
        # Return only the valid part of the data
        return data[max(len(ar_coefs), len(ma_coefs)):]
        
    elif model_type.lower() in ('garch', 'arch', 'egarch', 'volatility'):
        # Create data with volatility clustering
        alpha = properties.get('alpha', 0.1)
        beta = properties.get('beta', 0.8)
        omega = properties.get('omega', 0.1)
        
        # Check GARCH stationarity
        if alpha + beta >= 1:
            logger.warning("Non-stationary GARCH parameters: alpha + beta >= 1")
            alpha = 0.1
            beta = 0.8
        
        # Generate GARCH process
        data = np.zeros(sample_size)
        variance = np.zeros(sample_size)
        
        # Initialize with unconditional variance
        variance[0] = omega / (1 - alpha - beta)
        data[0] = np.sqrt(variance[0]) * np.random.normal(0, 1)
        
        for t in range(1, sample_size):
            # GARCH variance recursion
            variance[t] = omega + alpha * data[t-1]**2 + beta * variance[t-1]
            
            # Generate return
            data[t] = np.sqrt(variance[t]) * np.random.normal(0, 1)
        
        return data
        
    elif model_type.lower() in ('realized', 'rv', 'high_frequency', 'realized_volatility'):
        # Create high-frequency price data with microstructure noise
        volatility = properties.get('volatility', 0.20)  # Annualized volatility
        daily_vol = volatility / np.sqrt(252)  # Daily volatility
        
        obs_per_day = properties.get('obs_per_day', 78)  # 5-minute data for 6.5 hour trading day
        n_days = sample_size // obs_per_day
        
        if n_days < 1:
            n_days = 1
        
        # Efficient price process
        efficient_prices = np.zeros(n_days * obs_per_day)
        efficient_prices[0] = 100.0  # Initial price
        
        # Random walk with drift for efficient price
        for t in range(1, len(efficient_prices)):
            # Divide daily vol by sqrt of obs to get per-observation volatility
            period_vol = daily_vol / np.sqrt(obs_per_day)
            efficient_prices[t] = efficient_prices[t-1] * np.exp(np.random.normal(0, period_vol))
        
        # Add microstructure noise
        noise_magnitude = properties.get('noise_magnitude', 0.0001)
        observed_prices = efficient_prices * (1 + np.random.normal(0, noise_magnitude, len(efficient_prices)))
        
        return observed_prices
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")