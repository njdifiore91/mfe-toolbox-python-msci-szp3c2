"""
Tests for the bootstrap module of the MFE Toolbox.

This module contains tests for the bootstrap functionality, including
synchronous and asynchronous implementations, with a focus on validating
the Numba optimization for time series resampling methods.
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st

from mfe.core.bootstrap import (
    block_bootstrap, stationary_bootstrap, moving_block_bootstrap,
    block_bootstrap_async, stationary_bootstrap_async, moving_block_bootstrap_async,
    Bootstrap, BootstrapResult, calculate_bootstrap_ci, calculate_bootstrap_pvalue
)
from mfe.utils.numba_helpers import optimized_jit, fallback_to_python
from mfe.utils.validation import validate_array, is_positive_integer, is_probability
from mfe.utils.async_helpers import AsyncTask

# Constants for tests
SAMPLE_SIZE = 100
NUM_BOOTSTRAP = 500
BLOCK_SIZE = 10
PROBABILITY = 0.2

# Utility functions for tests
def mean_func(data):
    """Simple statistic function that calculates the mean of input data."""
    return np.mean(data)

def generate_test_data(size=SAMPLE_SIZE, random_seed=True):
    """Generate synthetic time series data for bootstrap tests."""
    if random_seed:
        np.random.seed(42)  # For reproducibility
    
    # Generate an AR(1) process with persistence
    data = np.zeros(size)
    data[0] = np.random.randn()
    for i in range(1, size):
        data[i] = 0.7 * data[i-1] + np.random.randn()
    
    return data

# Global variable to track progress in async tests
progress_value = 0

def progress_callback(progress):
    """Simple callback function to track progress during async bootstrap operations."""
    global progress_value
    progress_value = progress

# Fixtures for tests
@pytest.fixture
def bootstrap_data():
    """Fixture providing synthetic data for bootstrap tests."""
    return generate_test_data()

@pytest.fixture(scope="module")
def statistic_functions():
    """Fixture providing various statistic functions for bootstrap tests."""
    return {
        'mean': lambda x: np.mean(x),
        'median': lambda x: np.median(x),
        'variance': lambda x: np.var(x),
        'std': lambda x: np.std(x),
        'quantile25': lambda x: np.percentile(x, 25),
        'quantile75': lambda x: np.percentile(x, 75)
    }

@pytest.fixture
def progress_tracker():
    """Fixture for tracking progress in async bootstrap tests."""
    tracker = {'value': 0.0}
    
    def track_progress(progress):
        tracker['value'] = progress
    
    tracker['callback'] = track_progress
    return tracker

# Basic bootstrap tests
@pytest.mark.core
@pytest.mark.bootstrap
def test_block_bootstrap_basic():
    """Basic test for block bootstrap functionality."""
    # Generate test data
    data = generate_test_data()
    
    # Run block bootstrap
    bootstrap_statistics = block_bootstrap(
        data=data,
        statistic_func=mean_func,
        block_size=BLOCK_SIZE,
        num_bootstrap=NUM_BOOTSTRAP
    )
    
    # Verify output shape
    assert isinstance(bootstrap_statistics, np.ndarray)
    assert bootstrap_statistics.shape == (NUM_BOOTSTRAP,)
    
    # Verify statistical properties
    assert np.isfinite(bootstrap_statistics).all()
    
    # The mean of bootstrap statistics should be close to the original statistic
    original_mean = mean_func(data)
    bootstrap_mean = np.mean(bootstrap_statistics)
    assert np.abs(bootstrap_mean - original_mean) < 0.2  # Reasonable tolerance

@pytest.mark.core
@pytest.mark.bootstrap
def test_stationary_bootstrap_basic():
    """Basic test for stationary bootstrap functionality."""
    # Generate test data
    data = generate_test_data()
    
    # Run stationary bootstrap
    bootstrap_statistics = stationary_bootstrap(
        data=data,
        statistic_func=mean_func,
        probability=PROBABILITY,
        num_bootstrap=NUM_BOOTSTRAP
    )
    
    # Verify output shape
    assert isinstance(bootstrap_statistics, np.ndarray)
    assert bootstrap_statistics.shape == (NUM_BOOTSTRAP,)
    
    # Verify statistical properties
    assert np.isfinite(bootstrap_statistics).all()
    
    # The mean of bootstrap statistics should be close to the original statistic
    original_mean = mean_func(data)
    bootstrap_mean = np.mean(bootstrap_statistics)
    assert np.abs(bootstrap_mean - original_mean) < 0.2  # Reasonable tolerance

@pytest.mark.core
@pytest.mark.bootstrap
def test_moving_block_bootstrap_basic():
    """Basic test for moving block bootstrap functionality."""
    # Generate test data
    data = generate_test_data()
    
    # Run moving block bootstrap
    bootstrap_statistics = moving_block_bootstrap(
        data=data,
        statistic_func=mean_func,
        block_size=BLOCK_SIZE,
        num_bootstrap=NUM_BOOTSTRAP
    )
    
    # Verify output shape
    assert isinstance(bootstrap_statistics, np.ndarray)
    assert bootstrap_statistics.shape == (NUM_BOOTSTRAP,)
    
    # Verify statistical properties
    assert np.isfinite(bootstrap_statistics).all()
    
    # The mean of bootstrap statistics should be close to the original statistic
    original_mean = mean_func(data)
    bootstrap_mean = np.mean(bootstrap_statistics)
    assert np.abs(bootstrap_mean - original_mean) < 0.2  # Reasonable tolerance

# Bootstrap class tests
@pytest.mark.core
@pytest.mark.bootstrap
def test_bootstrap_class_basic():
    """Test the Bootstrap class with different bootstrap methods."""
    # Generate test data
    data = generate_test_data()
    
    # Test block bootstrap
    bootstrap = Bootstrap(method='block', params={'block_size': BLOCK_SIZE}, num_bootstrap=NUM_BOOTSTRAP)
    result = bootstrap.run(data, mean_func)
    
    # Verify results
    assert isinstance(result, BootstrapResult)
    assert isinstance(result.bootstrap_statistics, np.ndarray)
    assert result.bootstrap_statistics.shape == (NUM_BOOTSTRAP,)
    assert isinstance(result.original_statistic, float)
    
    # Test stationary bootstrap
    bootstrap = Bootstrap(method='stationary', params={'probability': PROBABILITY}, num_bootstrap=NUM_BOOTSTRAP)
    result = bootstrap.run(data, mean_func)
    
    # Verify results
    assert isinstance(result, BootstrapResult)
    assert isinstance(result.bootstrap_statistics, np.ndarray)
    assert result.bootstrap_statistics.shape == (NUM_BOOTSTRAP,)
    
    # Test moving block bootstrap
    bootstrap = Bootstrap(method='moving', params={'block_size': BLOCK_SIZE}, num_bootstrap=NUM_BOOTSTRAP)
    result = bootstrap.run(data, mean_func)
    
    # Verify results
    assert isinstance(result, BootstrapResult)
    assert isinstance(result.bootstrap_statistics, np.ndarray)
    assert result.bootstrap_statistics.shape == (NUM_BOOTSTRAP,)

@pytest.mark.core
@pytest.mark.bootstrap
def test_bootstrap_result_class():
    """Test the BootstrapResult class methods."""
    # Generate test data
    data = generate_test_data()
    
    # Create a Bootstrap object and run analysis
    bootstrap = Bootstrap(method='block', params={'block_size': BLOCK_SIZE}, num_bootstrap=NUM_BOOTSTRAP)
    result = bootstrap.run(data, mean_func)
    
    # Test bootstrap_statistics property
    assert isinstance(result.bootstrap_statistics, np.ndarray)
    assert result.bootstrap_statistics.shape == (NUM_BOOTSTRAP,)
    
    # Test standard_error method
    std_error = result.standard_error()
    assert isinstance(std_error, float)
    assert std_error > 0
    
    # Test confidence_interval method
    ci_percentile = result.confidence_interval(method='percentile', alpha=0.05)
    assert isinstance(ci_percentile, tuple)
    assert len(ci_percentile) == 2
    assert ci_percentile[0] < ci_percentile[1]  # Lower bound should be less than upper bound
    
    ci_bca = result.confidence_interval(method='bca', alpha=0.05)
    assert isinstance(ci_bca, tuple)
    assert len(ci_bca) == 2
    assert ci_bca[0] < ci_bca[1]
    
    # Test calculate_pvalue method
    null_value = result.original_statistic
    p_value_two_sided = result.calculate_pvalue(null_value, alternative='two-sided')
    assert isinstance(p_value_two_sided, float)
    assert 0 <= p_value_two_sided <= 1
    
    p_value_less = result.calculate_pvalue(null_value, alternative='less')
    assert isinstance(p_value_less, float)
    assert 0 <= p_value_less <= 1
    
    p_value_greater = result.calculate_pvalue(null_value, alternative='greater')
    assert isinstance(p_value_greater, float)
    assert 0 <= p_value_greater <= 1

@pytest.mark.core
@pytest.mark.bootstrap
def test_bootstrap_ci_calculation():
    """Test bootstrap confidence interval calculation."""
    # Generate test data
    data = generate_test_data()
    
    # Run bootstrap
    bootstrap_statistics = block_bootstrap(
        data=data,
        statistic_func=mean_func,
        block_size=BLOCK_SIZE,
        num_bootstrap=NUM_BOOTSTRAP
    )
    
    original_statistic = mean_func(data)
    
    # Test percentile method
    ci_percentile = calculate_bootstrap_ci(
        bootstrap_statistics=bootstrap_statistics,
        original_statistic=original_statistic,
        method='percentile',
        alpha=0.05
    )
    
    assert isinstance(ci_percentile, tuple)
    assert len(ci_percentile) == 2
    assert ci_percentile[0] < ci_percentile[1]
    
    # Test bca method
    ci_bca = calculate_bootstrap_ci(
        bootstrap_statistics=bootstrap_statistics,
        original_statistic=original_statistic,
        method='bca',
        alpha=0.05
    )
    
    assert isinstance(ci_bca, tuple)
    assert len(ci_bca) == 2
    assert ci_bca[0] < ci_bca[1]
    
    # Original statistic should be within the confidence interval with high probability
    assert ci_percentile[0] <= original_statistic <= ci_percentile[1] or \
           ci_bca[0] <= original_statistic <= ci_bca[1]

@pytest.mark.core
@pytest.mark.bootstrap
def test_bootstrap_pvalue_calculation():
    """Test bootstrap p-value calculation."""
    # Generate test data
    data = generate_test_data()
    
    # Run bootstrap
    bootstrap_statistics = block_bootstrap(
        data=data,
        statistic_func=mean_func,
        block_size=BLOCK_SIZE,
        num_bootstrap=NUM_BOOTSTRAP
    )
    
    original_statistic = mean_func(data)
    
    # Test two-sided p-value
    p_value_two_sided = calculate_bootstrap_pvalue(
        bootstrap_statistics=bootstrap_statistics,
        original_statistic=original_statistic,
        null_value=original_statistic,  # Using original statistic as null, should give high p-value
        alternative='two-sided'
    )
    
    assert isinstance(p_value_two_sided, float)
    assert 0 <= p_value_two_sided <= 1
    
    # Test one-sided p-values
    p_value_less = calculate_bootstrap_pvalue(
        bootstrap_statistics=bootstrap_statistics,
        original_statistic=original_statistic,
        null_value=original_statistic,
        alternative='less'
    )
    
    assert isinstance(p_value_less, float)
    assert 0 <= p_value_less <= 1
    
    p_value_greater = calculate_bootstrap_pvalue(
        bootstrap_statistics=bootstrap_statistics,
        original_statistic=original_statistic,
        null_value=original_statistic,
        alternative='greater'
    )
    
    assert isinstance(p_value_greater, float)
    assert 0 <= p_value_greater <= 1

@pytest.mark.core
@pytest.mark.bootstrap
def test_hypothesis_test():
    """Test hypothesis testing using Bootstrap class."""
    # Generate test data
    data = generate_test_data()
    
    # Create Bootstrap object
    bootstrap = Bootstrap(method='block', params={'block_size': BLOCK_SIZE}, num_bootstrap=NUM_BOOTSTRAP)
    
    # Test hypothesis_test method
    original_statistic = mean_func(data)
    
    # Test with null value equal to original statistic (should have high p-value)
    result, p_value = bootstrap.hypothesis_test(
        data=data,
        statistic_func=mean_func,
        null_value=original_statistic,
        alternative='two-sided'
    )
    
    assert isinstance(result, BootstrapResult)
    assert isinstance(p_value, float)
    assert 0 <= p_value <= 1
    assert p_value > 0.05  # Should not reject null when it's true

# Async bootstrap tests
@pytest.mark.core
@pytest.mark.bootstrap
@pytest.mark.asyncio
async def test_async_block_bootstrap():
    """Test async implementation of block bootstrap."""
    # Generate test data
    data = generate_test_data()
    
    # Reset progress tracker
    global progress_value
    progress_value = 0
    
    # Run async block bootstrap
    bootstrap_statistics = await block_bootstrap_async(
        data=data,
        statistic_func=mean_func,
        block_size=BLOCK_SIZE,
        num_bootstrap=NUM_BOOTSTRAP,
        progress_callback=progress_callback
    )
    
    # Verify output shape
    assert isinstance(bootstrap_statistics, np.ndarray)
    assert bootstrap_statistics.shape == (NUM_BOOTSTRAP,)
    
    # Verify statistical properties
    assert np.isfinite(bootstrap_statistics).all()
    
    # The mean of bootstrap statistics should be close to the original statistic
    original_mean = mean_func(data)
    bootstrap_mean = np.mean(bootstrap_statistics)
    assert np.abs(bootstrap_mean - original_mean) < 0.2
    
    # Verify progress was reported
    assert progress_value == 100.0

@pytest.mark.core
@pytest.mark.bootstrap
@pytest.mark.asyncio
async def test_async_stationary_bootstrap():
    """Test async implementation of stationary bootstrap."""
    # Generate test data
    data = generate_test_data()
    
    # Reset progress tracker
    global progress_value
    progress_value = 0
    
    # Run async stationary bootstrap
    bootstrap_statistics = await stationary_bootstrap_async(
        data=data,
        statistic_func=mean_func,
        probability=PROBABILITY,
        num_bootstrap=NUM_BOOTSTRAP,
        progress_callback=progress_callback
    )
    
    # Verify output shape
    assert isinstance(bootstrap_statistics, np.ndarray)
    assert bootstrap_statistics.shape == (NUM_BOOTSTRAP,)
    
    # Verify statistical properties
    assert np.isfinite(bootstrap_statistics).all()
    
    # The mean of bootstrap statistics should be close to the original statistic
    original_mean = mean_func(data)
    bootstrap_mean = np.mean(bootstrap_statistics)
    assert np.abs(bootstrap_mean - original_mean) < 0.2
    
    # Verify progress was reported
    assert progress_value == 100.0

@pytest.mark.core
@pytest.mark.bootstrap
@pytest.mark.asyncio
async def test_async_moving_block_bootstrap():
    """Test async implementation of moving block bootstrap."""
    # Generate test data
    data = generate_test_data()
    
    # Reset progress tracker
    global progress_value
    progress_value = 0
    
    # Run async moving block bootstrap
    bootstrap_statistics = await moving_block_bootstrap_async(
        data=data,
        statistic_func=mean_func,
        block_size=BLOCK_SIZE,
        num_bootstrap=NUM_BOOTSTRAP,
        progress_callback=progress_callback
    )
    
    # Verify output shape
    assert isinstance(bootstrap_statistics, np.ndarray)
    assert bootstrap_statistics.shape == (NUM_BOOTSTRAP,)
    
    # Verify statistical properties
    assert np.isfinite(bootstrap_statistics).all()
    
    # The mean of bootstrap statistics should be close to the original statistic
    original_mean = mean_func(data)
    bootstrap_mean = np.mean(bootstrap_statistics)
    assert np.abs(bootstrap_mean - original_mean) < 0.2
    
    # Verify progress was reported
    assert progress_value == 100.0

@pytest.mark.core
@pytest.mark.bootstrap
@pytest.mark.asyncio
async def test_bootstrap_class_async():
    """Test the Bootstrap class with async methods."""
    # Generate test data
    data = generate_test_data()
    
    # Reset progress tracker
    global progress_value
    
    # Test block bootstrap async
    progress_value = 0
    bootstrap = Bootstrap(method='block', params={'block_size': BLOCK_SIZE}, num_bootstrap=NUM_BOOTSTRAP)
    result = await bootstrap.run_async(data, mean_func, progress_callback)
    
    # Verify results
    assert isinstance(result, BootstrapResult)
    assert isinstance(result.bootstrap_statistics, np.ndarray)
    assert result.bootstrap_statistics.shape == (NUM_BOOTSTRAP,)
    assert progress_value == 100.0
    
    # Test stationary bootstrap async
    progress_value = 0
    bootstrap = Bootstrap(method='stationary', params={'probability': PROBABILITY}, num_bootstrap=NUM_BOOTSTRAP)
    result = await bootstrap.run_async(data, mean_func, progress_callback)
    
    # Verify results
    assert isinstance(result, BootstrapResult)
    assert progress_value == 100.0
    
    # Test moving block bootstrap async
    progress_value = 0
    bootstrap = Bootstrap(method='moving', params={'block_size': BLOCK_SIZE}, num_bootstrap=NUM_BOOTSTRAP)
    result = await bootstrap.run_async(data, mean_func, progress_callback)
    
    # Verify results
    assert isinstance(result, BootstrapResult)
    assert progress_value == 100.0

@pytest.mark.core
@pytest.mark.bootstrap
@pytest.mark.asyncio
async def test_async_hypothesis_test():
    """Test async hypothesis testing using Bootstrap class."""
    # Generate test data
    data = generate_test_data()
    
    # Create Bootstrap object
    bootstrap = Bootstrap(method='block', params={'block_size': BLOCK_SIZE}, num_bootstrap=NUM_BOOTSTRAP)
    
    # Reset progress tracker
    global progress_value
    progress_value = 0
    
    # Test hypothesis_test_async method
    original_statistic = mean_func(data)
    
    # Test with null value equal to original statistic (should have high p-value)
    result, p_value = await bootstrap.hypothesis_test_async(
        data=data,
        statistic_func=mean_func,
        null_value=original_statistic,
        alternative='two-sided',
        progress_callback=progress_callback
    )
    
    assert isinstance(result, BootstrapResult)
    assert isinstance(p_value, float)
    assert 0 <= p_value <= 1
    assert progress_value == 100.0

@pytest.mark.core
@pytest.mark.bootstrap
def test_input_validation():
    """Test input validation in bootstrap functions."""
    # Test invalid data type
    with pytest.raises(TypeError):
        block_bootstrap("not_an_array", mean_func, BLOCK_SIZE, NUM_BOOTSTRAP)
    
    # Test empty array
    with pytest.raises(ValueError):
        block_bootstrap(np.array([]), mean_func, BLOCK_SIZE, NUM_BOOTSTRAP)
    
    # Test invalid block size
    with pytest.raises(ValueError):
        block_bootstrap(generate_test_data(), mean_func, -1, NUM_BOOTSTRAP)
    
    with pytest.raises(ValueError):
        block_bootstrap(generate_test_data(), mean_func, 0, NUM_BOOTSTRAP)
    
    with pytest.raises(TypeError):
        block_bootstrap(generate_test_data(), mean_func, "not_an_integer", NUM_BOOTSTRAP)
    
    # Test invalid num_bootstrap
    with pytest.raises(ValueError):
        block_bootstrap(generate_test_data(), mean_func, BLOCK_SIZE, -1)
    
    with pytest.raises(ValueError):
        block_bootstrap(generate_test_data(), mean_func, BLOCK_SIZE, 0)
    
    with pytest.raises(TypeError):
        block_bootstrap(generate_test_data(), mean_func, BLOCK_SIZE, "not_an_integer")
    
    # Test invalid probability
    with pytest.raises(ValueError):
        stationary_bootstrap(generate_test_data(), mean_func, -0.1, NUM_BOOTSTRAP)
    
    with pytest.raises(ValueError):
        stationary_bootstrap(generate_test_data(), mean_func, 1.1, NUM_BOOTSTRAP)
    
    with pytest.raises(TypeError):
        stationary_bootstrap(generate_test_data(), mean_func, "not_a_float", NUM_BOOTSTRAP)
    
    # Test invalid data dimension
    with pytest.raises(ValueError):
        block_bootstrap(np.random.rand(10, 2), mean_func, BLOCK_SIZE, NUM_BOOTSTRAP)
    
    # Test invalid method in Bootstrap class
    with pytest.raises(ValueError):
        Bootstrap(method='invalid_method')

@pytest.mark.core
@pytest.mark.bootstrap
@pytest.mark.slow
def test_bootstrap_numba_optimization():
    """Test Numba optimization in bootstrap functions."""
    import time
    
    # Generate larger test data for better performance comparison
    data = generate_test_data(size=1000)
    
    # Create unoptimized version by removing the optimized_jit decorator
    def unoptimized_block_bootstrap(data, statistic_func, block_size, num_bootstrap, replace=True):
        """Unoptimized version of block_bootstrap for comparison."""
        # Copy the implementation from block_bootstrap but without Numba optimization
        validate_array(data, param_name="data")
        is_positive_integer(block_size, param_name="block_size")
        is_positive_integer(num_bootstrap, param_name="num_bootstrap")
        
        if data.ndim != 1:
            raise ValueError("data must be a 1-dimensional array")
        
        n = len(data)
        
        if block_size > n:
            raise ValueError(f"block_size ({block_size}) cannot be larger than data length ({n})")
        
        num_blocks = int(np.ceil(n / block_size))
        n_possible_blocks = n - block_size + 1
        
        bootstrap_statistics = np.zeros(num_bootstrap)
        
        for i in range(num_bootstrap):
            bootstrap_sample = np.zeros(num_blocks * block_size)
            
            if replace:
                block_indices = np.random.randint(0, n_possible_blocks, size=num_blocks)
            else:
                if num_blocks > n_possible_blocks:
                    raise ValueError(
                        f"Cannot sample {num_blocks} blocks without replacement "
                        f"when only {n_possible_blocks} blocks are available"
                    )
                block_indices = np.random.choice(n_possible_blocks, size=num_blocks, replace=False)
            
            for j, idx in enumerate(block_indices):
                start_pos = j * block_size
                end_pos = min((j + 1) * block_size, num_blocks * block_size)
                sample_length = end_pos - start_pos
                
                bootstrap_sample[start_pos:end_pos] = data[idx:idx + sample_length]
            
            bootstrap_sample = bootstrap_sample[:n]
            
            bootstrap_statistics[i] = statistic_func(bootstrap_sample)
        
        return bootstrap_statistics
    
    # Benchmark optimized vs unoptimized
    # Warm-up run
    _ = block_bootstrap(data, mean_func, BLOCK_SIZE, 10)
    _ = unoptimized_block_bootstrap(data, mean_func, BLOCK_SIZE, 10)
    
    # Time optimized version
    start_time = time.time()
    optimized_stats = block_bootstrap(data, mean_func, BLOCK_SIZE, NUM_BOOTSTRAP)
    optimized_time = time.time() - start_time
    
    # Time unoptimized version
    start_time = time.time()
    unoptimized_stats = unoptimized_block_bootstrap(data, mean_func, BLOCK_SIZE, NUM_BOOTSTRAP)
    unoptimized_time = time.time() - start_time
    
    # Just check that results are statistically equivalent
    np.testing.assert_allclose(np.mean(optimized_stats), np.mean(unoptimized_stats), rtol=0.1)
    np.testing.assert_allclose(np.std(optimized_stats), np.std(unoptimized_stats), rtol=0.1)

@pytest.mark.core
@pytest.mark.bootstrap
def test_bootstrap_with_different_statistics():
    """Test bootstrap functions with various statistic functions."""
    # Generate test data
    data = generate_test_data()
    
    # Define different statistic functions
    def mean_func(x):
        return np.mean(x)
    
    def median_func(x):
        return np.median(x)
    
    def variance_func(x):
        return np.var(x)
    
    def quantile_func(x):
        return np.percentile(x, 75)
    
    # Test with different statistics
    for func in [mean_func, median_func, variance_func, quantile_func]:
        # Calculate original statistic
        original_stat = func(data)
        
        # Run bootstrap
        bootstrap_statistics = block_bootstrap(
            data=data,
            statistic_func=func,
            block_size=BLOCK_SIZE,
            num_bootstrap=NUM_BOOTSTRAP
        )
        
        # Verify output shape
        assert bootstrap_statistics.shape == (NUM_BOOTSTRAP,)
        
        # Verify statistical properties
        assert np.isfinite(bootstrap_statistics).all()
        
        # The mean of bootstrap statistics should be reasonably close to the original
        bootstrap_mean = np.mean(bootstrap_statistics)
        # Using a larger tolerance since some statistics (like variance) can be more variable
        assert np.abs(bootstrap_mean - original_stat) < max(0.5, 0.5 * abs(original_stat))

@pytest.mark.core
@pytest.mark.bootstrap
@given(st.lists(st.floats(min_value=-100, max_value=100), min_size=20, max_size=100))
def test_property_based(data_list):
    """Property-based tests for bootstrap functions using hypothesis."""
    # Convert list to numpy array
    data = np.array(data_list)
    
    # Skip tests if there are any non-finite values
    if not np.isfinite(data).all():
        return
    
    # Run bootstrap with smaller number of samples for speed
    bootstrap_statistics = block_bootstrap(
        data=data,
        statistic_func=mean_func,
        block_size=5,  # Smaller block size for efficiency
        num_bootstrap=50  # Fewer bootstraps for efficiency
    )
    
    # Basic properties that should hold
    assert bootstrap_statistics.shape == (50,)
    assert np.isfinite(bootstrap_statistics).all()
    
    # The bootstrap mean should be somewhat close to the original mean
    # but we need a large tolerance due to the random nature
    original_mean = mean_func(data)
    bootstrap_mean = np.mean(bootstrap_statistics)
    
    # For very variable or extreme data, this might not hold strictly,
    # so we'll use a loose tolerance or skip extreme cases
    if abs(original_mean) > 1e-10:  # Non-zero mean
        # Allow for 100% deviation given the small sample and bootstrap size
        rel_error = abs((bootstrap_mean - original_mean) / original_mean)
        assert rel_error < 1.0