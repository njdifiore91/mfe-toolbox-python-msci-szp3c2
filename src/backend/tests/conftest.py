"""
Central pytest configuration file for the MFE Toolbox testing framework.

Defines test fixtures, utility functions, and setup routines shared across all test modules.
Provides data generators, environment validators, and specialized fixtures for testing
Numba-optimized functions, asynchronous operations, and statistical models.
"""

import os
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable

import pytest
import numpy as np  # numpy 1.26.3
import pandas as pd  # pandas 2.1.4
import pytest_asyncio  # pytest-asyncio 0.21.1
from hypothesis import strategies as st  # hypothesis 6.92.1

# Try to import numba.testing, but don't fail if numba is not available
try:
    import numba  # numba 0.59.0
    import numba.testing  # numba 0.59.0
    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False

import scipy  # scipy 1.11.4
import statsmodels  # statsmodels 0.14.1
import asyncio  # Python 3.12

# Import internal utilities
from . import check_test_environment, setup_test_environment, get_test_data_path
from ..mfe.initialize import check_numba_availability
from ..mfe.utils.validation import validate_array

# Define constants
logger = logging.getLogger('mfe.tests')
TEST_DATA_DIR = Path(__file__).parent / 'test_data'

# Pytest hooks
def pytest_configure(config):
    """
    Pytest hook to configure the test environment before any tests are run.
    
    Parameters
    ----------
    config : pytest.Config
        Pytest configuration object
    """
    # Setup test environment
    setup_test_environment()
    
    # Register custom markers
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "numba: marks tests that require Numba")
    config.addinivalue_line("markers", "async_test: marks tests that use async/await")
    config.addinivalue_line("markers", "models: marks tests related to statistical models")
    config.addinivalue_line("markers", "core: marks tests of core functionality")
    config.addinivalue_line("markers", "integration: marks integration tests")
    
    # Configure pytest
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Check Numba availability and log
    if not HAVE_NUMBA or not check_numba_availability():
        logger.warning("Numba is not available. Tests requiring Numba will be skipped.")

def pytest_collection_modifyitems(config, items):
    """
    Pytest hook to modify collected test items.
    
    Parameters
    ----------
    config : pytest.Config
        Pytest configuration object
    items : List[pytest.Item]
        List of collected test items
    """
    # Add appropriate markers based on test path
    for item in items:
        # Mark slow tests
        if "test_performance" in item.nodeid or "test_benchmark" in item.nodeid:
            item.add_marker(pytest.mark.slow)
        
        # Mark Numba tests
        if "test_numba" in item.nodeid or "test_jit" in item.nodeid:
            item.add_marker(pytest.mark.numba)
        
        # Skip Numba tests if Numba is not available
        if "test_numba" in item.nodeid or "test_jit" in item.nodeid:
            if not HAVE_NUMBA or not check_numba_availability():
                item.add_marker(pytest.mark.skip(reason="Numba not available"))
        
        # Add model markers
        if "test_models" in item.nodeid:
            item.add_marker(pytest.mark.models)
        
        # Add core markers
        if "test_core" in item.nodeid:
            item.add_marker(pytest.mark.core)
        
        # Add integration markers
        if "test_integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)

# Fixtures
@pytest.fixture(scope="session")
def sample_returns_small():
    """
    Fixture providing a small sample of synthetic return data for model testing.
    
    Returns
    -------
    np.ndarray
        Array of synthetic returns (100 elements)
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate synthetic return data
    return np.random.normal(0, 0.01, 100)

@pytest.fixture(scope="session")
def sample_returns_large():
    """
    Fixture providing a large sample of synthetic return data for model testing.
    
    Returns
    -------
    np.ndarray
        Array of synthetic returns (1000 elements)
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate synthetic return data
    return np.random.normal(0, 0.01, 1000)

@pytest.fixture(scope="session")
def time_series_fixture():
    """
    Fixture providing a pandas Series with datetime index for time series tests.
    
    Returns
    -------
    pd.Series
        Time series data with datetime index
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create date range
    dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
    
    # Create returns series
    returns = np.random.normal(0, 0.01, 100)
    
    # Create pandas Series with datetime index
    return pd.Series(returns, index=dates)

@pytest.fixture(scope="session")
def market_data():
    """
    Fixture providing real market benchmark data loaded from test_data directory.
    
    Returns
    -------
    np.ndarray
        Array of market data loaded from file
    """
    # Construct path to test data file
    file_path = get_test_data_path('market_benchmark.npy')
    
    # Load data if file exists
    try:
        data = np.load(file_path)
        # Validate the loaded array
        validate_array(data)
        return data
    except (FileNotFoundError, ValueError) as e:
        logger.warning(f"Could not load market benchmark data: {str(e)}")
        # Return synthetic data as fallback
        np.random.seed(42)
        return np.random.normal(0, 0.01, 1000)

@pytest.fixture(scope="session")
def hypothesis_float_array_strategy():
    """
    Fixture providing hypothesis strategies for generating arrays of float values.
    
    Returns
    -------
    st.SearchStrategy
        Hypothesis strategy for generating float arrays
    """
    # Create a strategy for arrays of reasonable size with values in typical financial range
    return st.arrays(
        np.float64,
        st.integers(min_value=10, max_value=500),
        elements=st.floats(min_value=-0.2, max_value=0.2, allow_nan=False, allow_infinity=False)
    )

@pytest.fixture
async def async_fixture():
    """
    Fixture for supporting async function testing with pytest_asyncio.
    
    Returns
    -------
    asyncio.AbstractEventLoop
        Asyncio event loop
    """
    # Create a new event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Yield the loop
    yield loop
    
    # Close the loop after test
    loop.close()

@pytest.fixture(scope="session")
def numba_test_functions():
    """
    Fixture providing sample functions for testing Numba optimization.
    
    Returns
    -------
    Dict[str, Callable]
        Dictionary of test functions with varying Numba compatibility
    """
    def simple_add(x, y):
        return x + y
    
    def array_sum(arr):
        return np.sum(arr)
    
    def array_mean(arr):
        return np.mean(arr)
    
    def matrix_multiply(a, b):
        return a @ b
    
    def vectorizable_func(x):
        return x * x + 2 * x - 1
    
    return {
        'simple_add': simple_add,
        'array_sum': array_sum,
        'array_mean': array_mean,
        'matrix_multiply': matrix_multiply,
        'vectorizable_func': vectorizable_func
    }