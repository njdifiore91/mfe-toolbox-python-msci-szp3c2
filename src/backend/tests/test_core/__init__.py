"""
MFE Toolbox - Test Core Package

This package contains tests for core statistical and computational modules of the MFE Toolbox,
including bootstrap, distributions, optimization, cross-section, and testing utilities.
"""

import pytest
import numpy as np
import pytest_asyncio
import numba.testing
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Import test environment utilities from parent package
from .. import check_test_environment, requires_numba, get_test_data_path

# Set up logger for test_core
logger = logging.getLogger('mfe.tests.core')

# Define exports
__all__ = [
    'parametrize_distribution_tests',
    'parametrize_bootstrap_tests', 
    'parametrize_optimization_tests',
    'validate_core_imports',
    'create_distribution_samples'
]

# Add pytest mark for core tests
pytestmark = [pytest.mark.core]

def parametrize_distribution_tests(distributions: list, test_params: dict) -> Callable:
    """
    Helper function for parametrizing statistical distribution test cases.
    
    Parameters
    ----------
    distributions : list
        List of distribution functions or names to test
    test_params : dict
        Dictionary of test parameters for each distribution
        
    Returns
    -------
    Callable
        Decorator function that can be applied to test functions
    """
    # Create parameter combinations for testing
    test_cases = []
    ids = []
    
    for dist in distributions:
        dist_name = dist.__name__ if hasattr(dist, '__name__') else str(dist)
        dist_params = test_params.get(dist_name, {})
        
        # Add test case with distribution and its parameters
        test_cases.append((dist, dist_params))
        ids.append(dist_name)
    
    # Create pytest parametrize decorator
    def decorator(func):
        return pytest.mark.parametrize('distribution,params', test_cases, ids=ids)(func)
    
    return decorator

def parametrize_bootstrap_tests(bootstrap_methods: list, test_params: dict) -> Callable:
    """
    Helper function for parametrizing bootstrap method test cases.
    
    Parameters
    ----------
    bootstrap_methods : list
        List of bootstrap methods to test
    test_params : dict
        Dictionary of test parameters for each bootstrap method
        
    Returns
    -------
    Callable
        Decorator function that can be applied to test functions
    """
    # Create parameter combinations for testing
    test_cases = []
    ids = []
    has_async = False
    
    for method in bootstrap_methods:
        method_name = method.__name__ if hasattr(method, '__name__') else str(method)
        method_params = test_params.get(method_name, {})
        
        # Check if method is async (from params or method attribute)
        is_async = method_params.get('is_async', False) or getattr(method, 'is_async', False)
        if is_async:
            has_async = True
        
        # Add test case with method and its parameters
        test_cases.append((method, method_params))
        ids.append(method_name)
    
    # Create pytest parametrize decorator
    def decorator(func):
        parametrized_func = pytest.mark.parametrize(
            'bootstrap_method,params', test_cases, ids=ids
        )(func)
        
        # Add async mark if needed
        if has_async:
            return pytest.mark.asyncio(parametrized_func)
        return parametrized_func
    
    return decorator

def parametrize_optimization_tests(algorithms: list, test_functions: dict) -> Callable:
    """
    Helper function for parametrizing optimization algorithm test cases.
    
    Parameters
    ----------
    algorithms : list
        List of optimization algorithms to test
    test_functions : dict
        Dictionary of test functions for optimization algorithms
        
    Returns
    -------
    Callable
        Decorator function that can be applied to test functions
    """
    # Create parameter combinations for testing
    test_cases = []
    ids = []
    needs_numba = False
    
    for alg in algorithms:
        alg_name = alg.__name__ if hasattr(alg, '__name__') else str(alg)
        functions = test_functions.get(alg_name, {})
        
        # Check if algorithm requires numba
        requires_numba_opt = functions.get('requires_numba', False)
        if requires_numba_opt:
            needs_numba = True
        
        # Add test case with algorithm and functions
        test_cases.append((alg, functions))
        ids.append(alg_name)
    
    # Create pytest parametrize decorator
    def decorator(func):
        parametrized_func = pytest.mark.parametrize(
            'algorithm,test_functions', test_cases, ids=ids
        )(func)
        
        # Add numba mark if needed
        if needs_numba:
            return requires_numba(parametrized_func)
        return parametrized_func
    
    return decorator

def validate_core_imports() -> bool:
    """
    Verify that core modules can be correctly imported.
    
    Returns
    -------
    bool
        True if all imports are successful, False otherwise
    """
    core_modules = [
        ('mfe.core.bootstrap', ['block_bootstrap', 'stationary_bootstrap']),
        ('mfe.core.distributions', ['ged_pdf', 'skewed_t_pdf', 'jarque_bera']),
        ('mfe.core.optimization', ['minimize', 'maximize']),
        ('mfe.core.testing', ['ljung_box', 'arch_test']),
        ('mfe.core.cross_section', ['ols', 'pca'])
    ]
    
    all_imports_successful = True
    
    for module_name, expected_functions in core_modules:
        try:
            # Dynamically import the module
            module = __import__(module_name, fromlist=expected_functions)
            
            # Check if expected functions are present
            for func_name in expected_functions:
                if not hasattr(module, func_name):
                    logger.warning(f"Function {func_name} not found in {module_name}")
                    all_imports_successful = False
                    
            logger.debug(f"Successfully imported {module_name}")
            
        except ImportError as e:
            logger.error(f"Failed to import {module_name}: {str(e)}")
            all_imports_successful = False
            
    return all_imports_successful

def create_distribution_samples(dist_type: str, size: int, params: dict) -> np.ndarray:
    """
    Create standardized test data samples for distribution function testing.
    
    Parameters
    ----------
    dist_type : str
        Type of distribution to generate ('normal', 'skewt', 'ged')
    size : int
        Size of the test data sample
    params : dict
        Parameters specific to the distribution type
        
    Returns
    -------
    np.ndarray
        Generated test data following specified distribution
    """
    # Set random seed for reproducibility if provided
    seed = params.get('seed', 42)
    np.random.seed(seed)
    
    if dist_type == 'normal':
        # Generate normal distribution data
        mu = params.get('mu', 0.0)
        sigma = params.get('sigma', 1.0)
        return np.random.normal(mu, sigma, size)
        
    elif dist_type == 'skewt':
        # Generate skewed t-distribution data
        # This is a simplified approximation
        nu = params.get('nu', 5.0)  # Degrees of freedom
        lam = params.get('lambda', 0.5)  # Skewness parameter between 0 and 1
        
        # Generate t-distributed values
        t_values = np.random.standard_t(nu, size)
        
        # Apply skewness adjustment
        skew_mask = np.random.random(size) < lam
        t_values[skew_mask] = np.abs(t_values[skew_mask])
        t_values[~skew_mask] = -np.abs(t_values[~skew_mask])
        
        return t_values
        
    elif dist_type == 'ged':
        # Generate generalized error distribution data
        # This is a simplified approximation
        mu = params.get('mu', 0.0)
        sigma = params.get('sigma', 1.0)
        nu = params.get('nu', 2.0)  # Shape parameter (2 = normal)
        
        # Using normal distribution for nu=2, Laplace for nu=1, and various in between
        if abs(nu - 2.0) < 1e-5:
            # Standard normal
            data = np.random.normal(0, 1, size)
        elif abs(nu - 1.0) < 1e-5:
            # Laplace distribution
            data = np.random.laplace(0, 1/np.sqrt(2), size)
        else:
            # For other nu values, use a mixture approach
            gamma_scale = np.sqrt(2**(-2/nu) * 
                               np.math.gamma(1/nu) / np.math.gamma(3/nu))
            data = np.random.normal(0, 1, size)
            data = np.sign(data) * np.abs(data)**nu
            data = data * gamma_scale
        
        # Scale and shift
        return mu + sigma * data
        
    else:
        raise ValueError(f"Unsupported distribution type: {dist_type}")