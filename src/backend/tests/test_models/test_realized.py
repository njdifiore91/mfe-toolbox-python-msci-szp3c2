import pytest
import numpy as np
import pandas as pd
from hypothesis import given, strategies as st

# Import functions to be tested
from mfe.models.realized import realized_variance, realized_kernel_variance, realized_covariance
from mfe.models.high_frequency import high_frequency_sampling
from mfe.utils.numba_helpers import is_numba_optimized

# Import test fixtures
from ..conftest import sample_returns, market_benchmark

def test_realized_variance_basic():
    """Test that realized_variance correctly calculates variance with basic parameters."""
    # Generate test data
    np.random.seed(42)
    prices = np.exp(np.cumsum(np.random.normal(0, 0.01, 1000)))
    times = np.arange(1000)
    
    # Calculate realized variance
    rv, rv_ss = realized_variance(prices, times, 'seconds', 'CalendarTime')
    
    # Assertions
    assert isinstance(rv, float)
    assert rv > 0
    assert isinstance(rv_ss, float)
    assert rv_ss > 0

@pytest.mark.parametrize('samplingType, samplingInterval', [
    ('CalendarTime', (60, 300)),
    ('CalendarUniform', (78, 390)), 
    ('BusinessTime', (1, 50, 300)),
    ('BusinessUniform', (68, 390)),
    ('Fixed', 30)
])
def test_realized_variance_sampling_types(samplingType, samplingInterval):
    """Test realized_variance calculation with different sampling methods."""
    # Generate test data
    np.random.seed(42)
    prices = np.exp(np.cumsum(np.random.normal(0, 0.01, 1000)))
    times = np.arange(1000)
    
    # Calculate realized variance with the specific sampling type
    rv, rv_ss = realized_variance(prices, times, 'seconds', samplingType, samplingInterval)
    
    # Assertions
    assert isinstance(rv, float)
    assert rv > 0
    assert isinstance(rv_ss, float)
    assert rv_ss > 0

@pytest.mark.parametrize('kernel_type', ['Bartlett', 'Parzen', 'QS', 'Truncated', 'Tukey-Hanning'])
def test_realized_kernel_variance(kernel_type):
    """Test kernel-based realized variance estimation with different kernel types."""
    # Generate test data
    np.random.seed(42)
    prices = np.exp(np.cumsum(np.random.normal(0, 0.01, 1000)))
    times = np.arange(1000)
    
    # Calculate realized kernel variance with the specified kernel
    rk = realized_kernel_variance(prices, times, 'seconds', kernel_type)
    
    # Assertions
    assert isinstance(rk, float)
    assert rk > 0

def test_realized_covariance():
    """Test that realized_covariance correctly calculates covariance between two price series."""
    # Generate correlated test data
    np.random.seed(42)
    common_factor = np.random.normal(0, 0.01, 1000)
    specific_factor1 = np.random.normal(0, 0.005, 1000)
    specific_factor2 = np.random.normal(0, 0.005, 1000)
    
    # Create two correlated price series
    returns1 = 0.7 * common_factor + 0.3 * specific_factor1
    returns2 = 0.7 * common_factor + 0.3 * specific_factor2
    
    prices1 = np.exp(np.cumsum(returns1))
    prices2 = np.exp(np.cumsum(returns2))
    times = np.arange(1000)
    
    # Calculate realized covariance
    rcov = realized_covariance(prices1, prices2, times, 'seconds', 'CalendarTime')
    
    # Assertions
    assert isinstance(rcov, float)
    # Covariance should be positive since we generated positively correlated returns
    assert rcov > 0

@pytest.mark.parametrize('scheme', ['calendar', 'business', 'tick'])
def test_high_frequency_sampling(scheme):
    """Test various sampling schemes for high-frequency data."""
    # Generate high-frequency data
    np.random.seed(42)
    prices = np.exp(np.cumsum(np.random.normal(0, 0.001, 10000)))
    timestamps = pd.date_range(start='2020-01-01 9:30', periods=10000, freq='1min')
    
    # Apply sampling scheme
    sampled_prices, sampled_times = high_frequency_sampling(prices, timestamps, scheme)
    
    # Assertions
    assert len(sampled_prices) > 0
    assert len(sampled_prices) == len(sampled_times)
    assert len(sampled_prices) <= len(prices)  # Sampling should reduce or maintain size

def test_numba_optimization():
    """Test that realized volatility functions are properly optimized with Numba."""
    # Check that key functions are Numba-optimized
    assert is_numba_optimized(realized_variance)
    assert is_numba_optimized(realized_kernel_variance)
    assert is_numba_optimized(realized_covariance)

def test_realized_variance_performance(benchmark):
    """Benchmark performance of realized variance calculation."""
    # Generate test data
    np.random.seed(42)
    prices = np.exp(np.cumsum(np.random.normal(0, 0.01, 5000)))
    times = np.arange(5000)
    
    # Benchmark the function
    result = benchmark(lambda: realized_variance(prices, times, 'seconds', 'CalendarTime'))
    
    # Assertions - just verify the result is valid
    assert isinstance(result[0], float)
    assert result[0] > 0

@given(st.arrays(dtype=np.float64, shape=st.integers(min_value=10, max_value=1000), 
                 elements=st.floats(min_value=0.1, max_value=1000.0)))
def test_property_positive_variance(prices):
    """Property-based test ensuring realized variance is always positive."""
    # Create timestamps
    times = np.arange(len(prices))
    
    # Calculate realized variance
    rv, _ = realized_variance(prices, times, 'seconds', 'Fixed')
    
    # Assertions
    assert rv >= 0

@given(st.arrays(dtype=np.float64, shape=st.integers(min_value=10, max_value=100), 
                 elements=st.floats(min_value=0.1, max_value=100.0)),
       st.floats(min_value=0.5, max_value=2.0))
def test_property_scaling(prices, scale):
    """Property-based test verifying scaling properties of realized variance."""
    # Create timestamps
    times = np.arange(len(prices))
    
    # Calculate realized variance for original and scaled prices
    rv_original, _ = realized_variance(prices, times, 'seconds', 'Fixed')
    rv_scaled, _ = realized_variance(prices * scale, times, 'seconds', 'Fixed')
    
    # Variance should scale as square of the scale factor (within tolerance)
    assert np.isclose(rv_scaled, rv_original * scale**2, rtol=1e-3, atol=1e-3)