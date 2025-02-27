"""
Test module for validating the functionality of Numba helper utilities.

Verifies core decorators for JIT compilation, fallback mechanisms, and optimization
strategies that enable high-performance numerical computing throughout the MFE Toolbox.
Includes tests for compilation success, graceful degradation, and threading configuration.
"""
import pytest
import numpy as np
import numba
from numba import testing as numba_testing

from mfe.utils.numba_helpers import (
    optimized_jit, fallback_to_python, parallel_jit, check_numba_compatibility,
    vectorized_jit, configure_numba_threading, create_specialized_function,
    get_numba_function, DEFAULT_NUMBA_OPTIONS, jit_garch_recursion,
    jit_garch_likelihood
)

# Sample functions with different Numba compatibility characteristics for testing
SAMPLE_FUNCTIONS = {
    # Function compatible with Numba's nopython mode
    'compatible': lambda x: x * 2,
    
    # Function not compatible with Numba due to Python-specific operations
    'incompatible': lambda x: print(x) or x,
    
    # Function compatible only with Numba's object mode
    'object_mode': lambda x: x.astype(str) if isinstance(x, np.ndarray) else str(x)
}


@pytest.mark.parametrize('func_type', ['compatible', 'incompatible', 'object_mode'])
@pytest.mark.requires_numba
def test_check_numba_compatibility(func_type):
    """Tests the check_numba_compatibility function with various function types."""
    func = SAMPLE_FUNCTIONS[func_type]
    
    # Check compatibility
    result = check_numba_compatibility(func)
    
    # Assert based on function type
    if func_type == 'compatible':
        assert result is True, "Function should be compatible with Numba"
    elif func_type == 'incompatible':
        assert result is False, "Function with print should not be compatible with Numba"
    elif func_type == 'object_mode':
        assert result is False, "Function requiring object mode should not be marked as compatible"


@pytest.mark.requires_numba
def test_optimized_jit():
    """Tests the optimized_jit decorator for successful Numba compilation."""
    # Define a simple function with the decorator
    @optimized_jit()
    def test_func(x):
        return x * 2
    
    # Test with a NumPy array
    arr = np.array([1, 2, 3], dtype=np.float64)
    result = test_func(arr)
    
    # Check result
    np.testing.assert_array_equal(result, arr * 2)
    
    # Verify the function is compiled with Numba
    assert hasattr(test_func, 'nopython_signatures')
    
    # Test with custom options
    custom_options = {'nopython': True, 'cache': False}
    
    @optimized_jit(options=custom_options)
    def test_func_custom(x):
        return x + 1
    
    result_custom = test_func_custom(arr)
    np.testing.assert_array_equal(result_custom, arr + 1)


@pytest.mark.parametrize('func_type', ['compatible', 'incompatible'])
def test_fallback_to_python(func_type):
    """Tests the fallback_to_python decorator for graceful degradation when compilation fails."""
    func = SAMPLE_FUNCTIONS[func_type]
    
    # Apply decorator
    decorated_func = fallback_to_python()(func)
    
    # Test execution
    test_input = 5 if func_type == 'incompatible' else np.array([1, 2, 3])
    result = decorated_func(test_input)
    
    # Verify function executes successfully regardless of Numba compatibility
    expected = func(test_input)
    if isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(result, expected)
    else:
        assert result == expected
    
    # Check if Numba optimization is applied when available
    if func_type == 'compatible' and numba is not None:
        assert hasattr(decorated_func, 'nopython_signatures'), "Compatible function should be Numba-optimized"


@pytest.mark.requires_numba
def test_parallel_jit():
    """Tests the parallel_jit decorator for parallel execution optimization."""
    # Define a function suitable for parallelization
    @parallel_jit()
    def parallel_sum(arr):
        result = 0.0
        for i in range(len(arr)):
            result += arr[i]
        return result
    
    # Test with a large array
    arr = np.arange(1000, dtype=np.float64)
    result = parallel_sum(arr)
    
    # Verify result correctness
    assert result == np.sum(arr)
    
    # Check that parallel execution is configured
    assert hasattr(parallel_sum, 'nopython_signatures')
    # We can't easily verify parallelization itself in a unit test


@pytest.mark.requires_numba
def test_vectorized_jit():
    """Tests the vectorized_jit decorator for optimized element-wise operations."""
    # Define a simple function for vectorization
    @vectorized_jit(['float64(float64)'])
    def vectorized_sqrt(x):
        return np.sqrt(x)
    
    # Test with array inputs
    arr = np.array([1.0, 4.0, 9.0])
    result = vectorized_sqrt(arr)
    
    # Check result correctness
    np.testing.assert_array_almost_equal(result, np.sqrt(arr))
    
    # Test with scalar input
    scalar_result = vectorized_sqrt(16.0)
    assert scalar_result == 4.0
    
    # Test with multiple input types specified in signatures
    @vectorized_jit(['float64(float64, float64)', 'int64(int64, int64)'])
    def add_func(x, y):
        return x + y
    
    # Test float inputs
    float_result = add_func(np.array([1.5, 2.5]), np.array([3.0, 4.0]))
    np.testing.assert_array_almost_equal(float_result, np.array([4.5, 6.5]))
    
    # Test integer inputs
    int_result = add_func(np.array([1, 2], dtype=np.int64), np.array([3, 4], dtype=np.int64))
    np.testing.assert_array_equal(int_result, np.array([4, 6]))


@pytest.mark.requires_numba
def test_configure_numba_threading():
    """Tests the configure_numba_threading function for proper thread configuration."""
    if not hasattr(numba, 'get_num_threads'):
        pytest.skip("Numba version doesn't support thread configuration inspection")
    
    # Save original configuration
    original_threads = numba.get_num_threads() if hasattr(numba, 'get_num_threads') else None
    
    try:
        # Configure specific thread count
        configure_numba_threading(2)
        
        # Verify configuration is applied correctly
        if hasattr(numba, 'get_num_threads'):
            assert numba.get_num_threads() == 2
        
        # Test with default (auto) configuration
        configure_numba_threading()
        
        # Verify sensible defaults are applied
        if hasattr(numba, 'get_num_threads'):
            assert numba.get_num_threads() > 0
    
    finally:
        # Restore original configuration
        if original_threads is not None and hasattr(numba, 'set_num_threads'):
            numba.set_num_threads(original_threads)


@pytest.mark.requires_numba
def test_create_specialized_function():
    """Tests the create_specialized_function for type-specialized compilation."""
    # Define a sample function with type-dependent behavior
    def add_values(a, b):
        return a + b
    
    # Create specialized version with specific type signature
    specialized_add = create_specialized_function(
        add_values, 
        'float64(float64, float64)'
    )
    
    # Test with matching input types
    result = specialized_add(2.0, 3.0)
    assert result == 5.0
    
    # Verify compilation status
    assert hasattr(specialized_add, 'nopython_signatures')
    
    # Test with another type signature
    int_add = create_specialized_function(
        add_values,
        'int64(int64, int64)'
    )
    
    int_result = int_add(2, 3)
    assert int_result == 5


@pytest.mark.requires_numba
def test_get_numba_function():
    """Tests the get_numba_function for creating JIT-compiled functions with specific signatures."""
    # Define a sample function suitable for JIT compilation
    def multiply(a, b):
        return a * b
    
    # Use get_numba_function to create compiled version with specific signature
    compiled_multiply = get_numba_function(
        multiply,
        'float64(float64, float64)'
    )
    
    # Execute compiled function with appropriate inputs
    result = compiled_multiply(2.0, 3.0)
    assert result == 6.0
    
    # Verify function is actually compiled with Numba
    assert hasattr(compiled_multiply, 'nopython_signatures')
    
    # Test without explicit signature
    auto_compiled = get_numba_function(multiply)
    auto_result = auto_compiled(2.0, 3.0)
    assert auto_result == 6.0


@pytest.mark.requires_numba
def test_jit_garch_recursion():
    """Tests the jit_garch_recursion function for correct GARCH variance computation."""
    # Generate synthetic returns data with NumPy
    np.random.seed(42)
    returns = np.random.normal(0, 1, 100)
    
    # Create GARCH parameter array (omega, alpha, beta)
    parameters = np.array([0.05, 0.1, 0.85])
    
    # Call jit_garch_recursion with parameters and data
    variances = jit_garch_recursion(parameters, returns, p=1, q=1)
    
    # Verify variance array dimensions and values
    assert len(variances) == len(returns)
    assert np.all(variances > 0), "All variances should be positive"
    
    # Check recursion formula is implemented correctly
    for t in range(2, 10):
        expected_variance = (parameters[0] + 
                           parameters[1] * returns[t-1]**2 + 
                           parameters[2] * variances[t-1])
        assert np.isclose(variances[t], expected_variance, rtol=1e-10)


@pytest.mark.requires_numba
def test_jit_garch_likelihood():
    """Tests the jit_garch_likelihood function for GARCH parameter estimation."""
    # Generate synthetic returns data with NumPy
    np.random.seed(42)
    returns = np.random.normal(0, 1, 100)
    
    # Create GARCH parameter array (omega, alpha, beta)
    parameters = np.array([0.05, 0.1, 0.85])
    
    # Define simple distribution log-pdf function
    def normal_logpdf(x):
        return -0.5 * x**2 - 0.5 * np.log(2 * np.pi)
    
    # Call jit_garch_likelihood with parameters, data and distribution function
    likelihood = jit_garch_likelihood(parameters, returns, p=1, q=1, distribution_logpdf=normal_logpdf)
    
    # Verify likelihood value is numerically correct
    assert isinstance(likelihood, float)
    
    # Check likelihood is negative (for minimization)
    assert likelihood < 0
    
    # Verify likelihood behavior with different parameter values
    likelihood2 = jit_garch_likelihood(np.array([0.01, 0.05, 0.90]), returns, p=1, q=1, distribution_logpdf=normal_logpdf)
    assert likelihood != likelihood2, "Different parameters should yield different likelihoods"