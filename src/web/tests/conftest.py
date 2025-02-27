import pytest
import pytest_asyncio
from PyQt6 import QtWidgets, QtCore, QtTest
import numpy as np
import pandas as pd
import asyncio

# Register pytest_asyncio plugin for asynchronous test support
pytest_plugins = ['pytest_asyncio']

def pytest_configure(config):
    """
    Configure pytest for UI testing, including plugins and marker registration.
    
    Args:
        config: pytest.Config object for configuration
    """
    # Register custom markers for UI tests
    config.addinivalue_line("markers", "ui: marks tests as ui tests")
    config.addinivalue_line("markers", "async_ui: marks tests as asynchronous ui tests")
    config.addinivalue_line("markers", "slow: marks tests as slow running")
    
    # Set up pytest-qt integration
    config.addinivalue_line("markers", "qt: mark a test as requiring qt")
    
    # Configure asyncio event loop policy for testing
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

@pytest.fixture(scope='session')
def qapp():
    """
    Fixture that provides a QApplication instance for PyQt6 tests.
    
    Returns:
        PyQt6.QtWidgets.QApplication: A QApplication instance for UI tests
    """
    # Check if QApplication already exists
    app = QtWidgets.QApplication.instance()
    if app is None:
        # Create new instance if none exists
        app = QtWidgets.QApplication([])
    
    # Set application name for consistent behavior
    app.setApplicationName("MFE_Test")
    
    yield app
    
    # Cleanup after tests complete
    app.quit()

@pytest.fixture(scope='function')
def event_loop():
    """
    Fixture that sets up and tears down an event loop for async testing.
    
    Returns:
        asyncio.AbstractEventLoop: An event loop for async tests
    """
    # Create a new event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    yield loop
    
    # Close the event loop after test completes
    loop.close()

@pytest.fixture
def sample_data():
    """
    Fixture that provides sample numerical data for testing visualizations and models.
    
    Returns:
        numpy.ndarray: A NumPy array containing sample data
    """
    # Set fixed random seed for reproducibility
    np.random.seed(42)
    
    # Generate sample data - 1000 points of normally distributed returns
    return np.random.randn(1000)

@pytest.fixture
def sample_time_series():
    """
    Fixture that provides sample time series data for testing time series components.
    
    Returns:
        pandas.DataFrame: A Pandas DataFrame containing time series data
    """
    # Set fixed random seed for reproducibility
    np.random.seed(42)
    
    # Generate date range for daily data
    dates = pd.date_range(start='2020-01-01', periods=500, freq='D')
    
    # Generate random values with some autocorrelation for realism
    values = np.zeros(500)
    values[0] = np.random.randn()
    for i in range(1, 500):
        values[i] = 0.7 * values[i-1] + np.random.randn()
    
    # Create DataFrame with date index
    return pd.DataFrame({'value': values}, index=dates)

@pytest.fixture
def mock_model_params():
    """
    Fixture that provides mock model parameters for testing parameter displays.
    
    Returns:
        dict: A dictionary containing mock model parameters
    """
    return {
        'params': {
            'AR(1)': 0.756,
            'MA(1)': -0.243,
            'Constant': 0.002
        },
        'std_errors': {
            'AR(1)': 0.045,
            'MA(1)': 0.067,
            'Constant': 0.001
        },
        't_stats': {
            'AR(1)': 16.8,
            'MA(1)': -3.6,
            'Constant': 2.0
        },
        'p_values': {
            'AR(1)': 0.000,
            'MA(1)': 0.000,
            'Constant': 0.045
        },
        'info_criteria': {
            'AIC': -2.34,
            'BIC': -2.28,
            'Log-Likelihood': -245.67
        }
    }

@pytest.fixture
def mock_volatility_data():
    """
    Fixture that provides mock volatility data for testing volatility visualizations.
    
    Returns:
        dict: A dictionary containing volatility series data
    """
    # Set fixed random seed for reproducibility
    np.random.seed(42)
    
    # Create time index
    dates = pd.date_range(start='2020-01-01', periods=500, freq='D')
    
    # Generate returns with volatility clustering
    returns = np.zeros(500)
    volatility = np.zeros(500)
    volatility[0] = 0.01
    
    for i in range(1, 500):
        # GARCH-like process for volatility
        volatility[i] = 0.0001 + 0.15 * returns[i-1]**2 + 0.8 * volatility[i-1]
        returns[i] = np.random.randn() * np.sqrt(volatility[i])
    
    return {
        'dates': dates,
        'returns': returns,
        'volatility': volatility
    }