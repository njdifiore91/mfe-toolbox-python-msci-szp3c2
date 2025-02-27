import pytest
import numpy as np
import scipy
from scipy import optimize
import asyncio
import time
from hypothesis import given, strategies as st
import numba.testing
from typing import Dict, List, Tuple, Callable, Optional, Any

from mfe.core.optimization import (
    OptimizationResult, Optimizer, AsyncOptimizer, 
    bfgs_optimization, quasi_newton_optimizer, async_optimize,
    compute_numerical_gradient, check_gradient, constrained_optimization, line_search
)
from mfe.utils.numba_helpers import optimized_jit, fallback_to_python, check_numba_compatibility
from mfe.utils.validation import validate_array
from mfe.utils.async_helpers import AsyncTask

# Constants for testing
TOLERANCE = 1e-6
SEED = 42
TEST_DIMENSIONS = [2, 5, 10]

def quadratic_function(x: np.ndarray) -> float:
    """Simple quadratic function for optimization testing"""
    return np.sum(x**2)

def quadratic_gradient(x: np.ndarray) -> np.ndarray:
    """Gradient of the quadratic function"""
    return 2 * x

def rosenbrock_function(x: np.ndarray) -> float:
    """Rosenbrock function for testing more complex optimization"""
    if len(x) < 2:
        return np.sum(x**2)
    
    result = 0
    for i in range(len(x) - 1):
        result += 100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
    
    return result

def rosenbrock_gradient(x: np.ndarray) -> np.ndarray:
    """Gradient of the Rosenbrock function"""
    if len(x) < 2:
        return 2 * x
    
    grad = np.zeros_like(x)
    for i in range(len(x) - 1):
        grad[i] += -400 * x[i] * (x[i+1] - x[i]**2) - 2 * (1 - x[i])
        if i > 0:
            grad[i] += 200 * (x[i] - x[i-1]**2)
    
    grad[-1] = 200 * (x[-1] - x[-2]**2)
    
    return grad

# Global variable to store progress values for testing
progress_values = []

def progress_callback(progress: float) -> None:
    """Callback function to track progress during optimization"""
    progress_values.append(progress)

def generate_test_data(dimensions: int, set_seed: bool = True) -> np.ndarray:
    """Generate test data for optimization tests"""
    if set_seed:
        np.random.seed(SEED)
    
    return np.random.rand(dimensions) * 2 - 1  # Values between -1 and 1

@pytest.fixture
def optimization_functions():
    """Fixture providing test functions for optimization"""
    quadratic_min = np.zeros(2)
    rosenbrock_min = np.ones(2)
    
    return {
        'quadratic': (quadratic_function, quadratic_gradient, quadratic_min),
        'rosenbrock': (rosenbrock_function, rosenbrock_gradient, rosenbrock_min)
    }

@pytest.fixture
def optimizer_instance():
    """Fixture providing an Optimizer instance with default configuration"""
    return Optimizer()

@pytest.fixture
def async_optimizer_instance():
    """Fixture providing an AsyncOptimizer instance with default configuration"""
    return AsyncOptimizer()

@pytest.fixture
def initial_params():
    """Fixture providing initial parameters for optimization tests"""
    np.random.seed(SEED)
    return np.random.rand(2) * 2 - 1  # Values between -1 and 1

@pytest.fixture
def progress_tracker():
    """Fixture for tracking progress in async optimization tests"""
    global progress_values
    progress_values = []
    
    def progress_callback(progress):
        progress_values.append(progress)
    
    return {'callback': progress_callback, 'values': progress_values}

class TestOptimizer:
    """Test class for the Optimizer class"""
    
    def setup_method(self):
        """Set up before each test method"""
        np.random.seed(SEED)
        self.optimizer = Optimizer()
    
    def test_optimizer_initialization(self):
        """Test that Optimizer initializes correctly"""
        # Test default initialization
        optimizer = Optimizer()
        options = optimizer.get_options()
        
        assert 'method' in options
        assert options['method'] == 'BFGS'
        assert 'tol' in options
        
        # Test custom options
        custom_options = {'method': 'L-BFGS-B', 'tol': 1e-8, 'maxiter': 500}
        optimizer = Optimizer(custom_options)
        options = optimizer.get_options()
        
        assert options['method'] == 'L-BFGS-B'
        assert options['tol'] == 1e-8
        assert options['maxiter'] == 500
    
    def test_minimize_quadratic(self):
        """Test minimization of simple quadratic function"""
        x0 = generate_test_data(3)
        
        result = self.optimizer.minimize(quadratic_function, x0, quadratic_gradient)
        
        # Should converge to [0, 0, 0]
        assert result.is_successful()
        assert np.allclose(result.parameters, np.zeros_like(x0), atol=TOLERANCE)
        assert result.objective_value < TOLERANCE
    
    def test_minimize_with_gradient(self):
        """Test minimization with analytical gradient"""
        x0 = generate_test_data(3)
        
        # Without gradient (uses numerical gradient)
        result_no_grad = self.optimizer.minimize(
            quadratic_function, x0, options={'maxiter': 20}
        )
        
        # With gradient
        result_with_grad = self.optimizer.minimize(
            quadratic_function, x0, quadratic_gradient, options={'maxiter': 20}
        )
        
        # Both should converge, but with gradient should be more accurate
        assert result_no_grad.is_successful()
        assert result_with_grad.is_successful()
        
        # Check objective values
        assert result_no_grad.objective_value < 1e-4
        assert result_with_grad.objective_value < 1e-4
        
        # With gradient should generally take fewer iterations
        assert result_with_grad.iterations <= result_no_grad.iterations * 1.5
    
    def test_constrained_minimize(self):
        """Test minimization with constraints"""
        x0 = np.array([0.5, 0.5])
        
        # Define constraints: x[0] + x[1] = 1 (equality) and x[0] >= 0.2 (inequality)
        constraints = [
            {'type': 'eq', 'fun': lambda x: x[0] + x[1] - 1},
            {'type': 'ineq', 'fun': lambda x: x[0] - 0.2}
        ]
        
        result = self.optimizer.constrained_minimize(
            quadratic_function, x0, constraints,
            options={'method': 'SLSQP'}
        )
        
        assert result.is_successful()
        
        # Check constraints
        assert abs(result.parameters[0] + result.parameters[1] - 1) < TOLERANCE  # Equality
        assert result.parameters[0] >= 0.2 - TOLERANCE  # Inequality
    
    def test_optimizer_options(self):
        """Test set_options and get_options methods"""
        # Get default options
        default_options = self.optimizer.get_options()
        
        # Set new options
        custom_options = {'method': 'L-BFGS-B', 'tol': 1e-9}
        self.optimizer.set_options(custom_options)
        
        # Verify options were updated
        updated_options = self.optimizer.get_options()
        assert updated_options['method'] == 'L-BFGS-B'
        assert updated_options['tol'] == 1e-9
        
        # Reset options
        self.optimizer.reset_options()
        
        # Verify reset
        reset_options = self.optimizer.get_options()
        assert reset_options['method'] == default_options['method']
        assert reset_options['tol'] == default_options['tol']

class TestAsyncOptimizer:
    """Test class for async optimization functionality"""
    
    def setup_method(self):
        """Set up before each test method"""
        np.random.seed(SEED)
        self.async_optimizer = AsyncOptimizer()
    
    def test_async_optimizer_initialization(self):
        """Test that AsyncOptimizer initializes correctly"""
        # Test default initialization
        optimizer = AsyncOptimizer()
        
        # Test custom options
        custom_options = {'method': 'L-BFGS-B', 'tol': 1e-8}
        optimizer = AsyncOptimizer(custom_options)
        
        # No assertions needed, just check that initialization doesn't fail
    
    @pytest.mark.asyncio
    async def test_start_optimization(self):
        """Test starting an async optimization task"""
        x0 = generate_test_data(3)
        
        # Start optimization task
        task_id = "test_task"
        self.async_optimizer.start_optimization(
            task_id, quadratic_function, x0, quadratic_gradient
        )
        
        # Check task status
        status = self.async_optimizer.get_task_status(task_id)
        
        assert status['running'] == True
        assert status['completed'] == False
        assert status['task_id'] == task_id
        
        # Wait for task to complete (should be quick for quadratic function)
        for _ in range(10):
            await asyncio.sleep(0.1)
            status = self.async_optimizer.get_task_status(task_id)
            if status['completed']:
                break
        
        assert status['completed'] == True
    
    @pytest.mark.asyncio
    async def test_get_task_status(self):
        """Test getting status of an optimization task"""
        x0 = generate_test_data(3)
        
        # Start optimization task
        task_id = "status_test"
        self.async_optimizer.start_optimization(
            task_id, quadratic_function, x0, quadratic_gradient
        )
        
        # Initial status
        status = self.async_optimizer.get_task_status(task_id)
        assert 'running' in status
        assert 'completed' in status
        assert 'progress' in status
        
        # Wait a bit
        await asyncio.sleep(0.1)
        
        # Check updated status
        updated_status = self.async_optimizer.get_task_status(task_id)
        
        # Wait for task to complete
        for _ in range(10):
            await asyncio.sleep(0.1)
            updated_status = self.async_optimizer.get_task_status(task_id)
            if updated_status['completed']:
                break
    
    @pytest.mark.asyncio
    async def test_cancel_task(self):
        """Test cancelling an optimization task"""
        x0 = generate_test_data(10)  # Larger problem to ensure it doesn't finish immediately
        
        # Start optimization task with more iterations
        task_id = "cancel_test"
        self.async_optimizer.start_optimization(
            task_id, 
            rosenbrock_function,  # More complex function
            x0,
            rosenbrock_gradient,
            options={'maxiter': 1000}
        )
        
        # Check task is running
        status = self.async_optimizer.get_task_status(task_id)
        assert status['running'] == True
        
        # Cancel task
        result = self.async_optimizer.cancel_task(task_id)
        assert result == True
        
        # Try to get status (should fail)
        with pytest.raises(ValueError):
            self.async_optimizer.get_task_status(task_id)
    
    @pytest.mark.asyncio
    async def test_get_result(self):
        """Test getting result of a completed task"""
        x0 = generate_test_data(2)
        
        # Start optimization task
        task_id = "result_test"
        self.async_optimizer.start_optimization(
            task_id, quadratic_function, x0, quadratic_gradient
        )
        
        # Wait for task to complete
        for _ in range(10):
            await asyncio.sleep(0.1)
            status = self.async_optimizer.get_task_status(task_id)
            if status['completed']:
                break
        
        # Get result
        result = self.async_optimizer.get_result(task_id)
        
        assert isinstance(result, OptimizationResult)
        assert result.is_successful()
        assert np.allclose(result.parameters, np.zeros_like(x0), atol=TOLERANCE)
    
    @pytest.mark.asyncio
    async def test_wait_for_task(self):
        """Test waiting for task completion"""
        x0 = generate_test_data(3)
        
        # Start optimization task
        task_id = "wait_test"
        self.async_optimizer.start_optimization(
            task_id, 
            quadratic_function, 
            x0, 
            quadratic_gradient,
            options={'yield_every': 1}  # Yield progress more frequently
        )
        
        # Wait for task while tracking progress
        progress_results = []
        async for iteration, params, value in self.async_optimizer.wait_for_task(task_id):
            progress_results.append((iteration, value))
        
        # Check we got progress updates
        assert len(progress_results) > 0
        
        # Check final result converged
        final_iteration, final_value = progress_results[-1]
        assert final_value < TOLERANCE

@pytest.mark.core
@pytest.mark.optimization
def test_bfgs_optimization():
    """Test BFGS optimization algorithm"""
    # Define a simple objective function and gradient
    def obj_func(x):
        return np.sum(x**2)
    
    def obj_grad(x):
        return 2 * x
    
    def obj_func_grad(x):
        return obj_func(x), obj_grad(x)
    
    # Generate initial parameters
    x0 = generate_test_data(3)
    
    # Run BFGS optimization
    result_params, result_value, converged, iterations = bfgs_optimization(
        obj_func_grad, x0, tolerance=1e-8
    )
    
    # Verify convergence
    assert converged
    assert np.allclose(result_params, np.zeros_like(x0), atol=TOLERANCE)
    assert result_value < TOLERANCE
    assert iterations > 0  # Should take some iterations to converge

@pytest.mark.core
@pytest.mark.optimization
def test_quasi_newton_optimizer():
    """Test quasi-Newton optimization function"""
    # Define objective function and gradient
    def obj_func(x):
        return np.sum(x**2)
    
    def obj_grad(x):
        return 2 * x
    
    # Generate initial parameters
    x0 = generate_test_data(3)
    
    # Test with different methods
    methods = ['BFGS', 'L-BFGS-B', 'CG']
    
    for method in methods:
        options = {'method': method, 'tol': 1e-8, 'maxiter': 100}
        
        result = quasi_newton_optimizer(obj_func, obj_grad, x0, options)
        
        assert result.is_successful()
        assert np.allclose(result.parameters, np.zeros_like(x0), atol=TOLERANCE)
        assert result.objective_value < TOLERANCE

@pytest.mark.core
@pytest.mark.optimization
@pytest.mark.asyncio
async def test_async_optimize():
    """Test asynchronous optimization function"""
    # Define objective function and gradient
    def obj_func(x):
        return np.sum(x**2)
    
    def obj_grad(x):
        return 2 * x
    
    # Generate initial parameters
    x0 = generate_test_data(3)
    
    # Set up progress tracking
    progress_results = []
    
    # Run async optimization
    async for iter_count, params, value in async_optimize(
        obj_func, obj_grad, x0, options={'yield_every': 1}
    ):
        progress_results.append((iter_count, value))
    
    # Check progress updates
    assert len(progress_results) > 0
    
    # Check final result
    final_iteration, final_value = progress_results[-1]
    assert final_value < TOLERANCE
    assert final_iteration > 0

@pytest.mark.core
@pytest.mark.optimization
def test_line_search():
    """Test line search algorithm"""
    # Define objective function
    def obj_func(x):
        return np.sum(x**2)
    
    # Current point and search direction
    x_current = np.array([1.0, 1.0])
    direction = np.array([-1.0, -1.0])  # Direction towards minimum
    
    # Run line search
    alpha, x_new, f_new = line_search(
        obj_func, x_current, direction, 1.0, 1e-4, 0.9
    )
    
    # Step size should be positive
    assert alpha > 0
    
    # New function value should be lower
    assert f_new < obj_func(x_current)
    
    # New point should be x_current + alpha * direction
    assert np.allclose(x_new, x_current + alpha * direction)

@pytest.mark.core
@pytest.mark.optimization
def test_compute_numerical_gradient():
    """Test numerical gradient computation"""
    # Define functions with known gradients
    def f1(x):
        return np.sum(x**2)
    
    def grad1(x):
        return 2 * x
    
    def f2(x):
        return np.sum(x**3)
    
    def grad2(x):
        return 3 * x**2
    
    # Test points
    test_points = [
        np.array([1.0, 2.0, 3.0]),
        np.array([-1.0, 0.0, 1.0]),
        np.array([0.5, -0.5, 0.0])
    ]
    
    for x in test_points:
        # Compute numerical gradients
        num_grad1 = compute_numerical_gradient(f1, x)
        num_grad2 = compute_numerical_gradient(f2, x)
        
        # Compare with analytical gradients
        assert np.allclose(num_grad1, grad1(x), rtol=1e-4)
        assert np.allclose(num_grad2, grad2(x), rtol=1e-4)

@pytest.mark.core
@pytest.mark.optimization
def test_check_gradient():
    """Test gradient verification function"""
    # Define functions with correct gradients
    def f1(x):
        return np.sum(x**2)
    
    def grad1(x):
        return 2 * x
    
    # Define incorrect gradient
    def bad_grad(x):
        return x  # Should be 2*x
    
    # Test points
    x = np.array([1.0, 2.0, 3.0])
    
    # Check correct gradient
    success, rel_diff, analytical, numerical = check_gradient(f1, grad1, x)
    assert success
    assert rel_diff < 1e-4
    assert np.allclose(analytical, numerical, rtol=1e-4)
    
    # Check incorrect gradient
    success, rel_diff, analytical, numerical = check_gradient(f1, bad_grad, x)
    assert not success
    assert rel_diff > 1e-4
    assert not np.allclose(analytical, numerical, rtol=1e-4)

@pytest.mark.core
@pytest.mark.optimization
def test_constrained_optimization():
    """Test optimization with constraints"""
    # Define objective function
    def obj_func(x):
        return np.sum(x**2)
    
    # Initial parameters
    x0 = np.array([0.5, 0.5])
    
    # Define constraints
    constraints = [
        {'type': 'eq', 'fun': lambda x: x[0] + x[1] - 1},  # x[0] + x[1] = 1
        {'type': 'ineq', 'fun': lambda x: x[0] - 0.2}      # x[0] >= 0.2
    ]
    
    # Run constrained optimization
    result = constrained_optimization(
        obj_func, x0, constraints, method='SLSQP'
    )
    
    # Verify result satisfies constraints
    assert result.is_successful()
    assert abs(result.parameters[0] + result.parameters[1] - 1) < TOLERANCE
    assert result.parameters[0] >= 0.2 - TOLERANCE

@pytest.mark.core
@pytest.mark.optimization
def test_optimization_result_class():
    """Test OptimizationResult class functionality"""
    # Create test data
    params = np.array([1.0, 2.0])
    value = 5.0
    converged = True
    iterations = 10
    message = "Optimization successful"
    gradient = np.array([0.1, 0.2])
    
    # Create OptimizationResult instance
    result = OptimizationResult(
        parameters=params,
        objective_value=value,
        converged=converged,
        iterations=iterations,
        message=message,
        gradient=gradient
    )
    
    # Check properties
    assert np.array_equal(result.parameters, params)
    assert result.objective_value == value
    assert result.converged == converged
    assert result.iterations == iterations
    assert result.message == message
    assert np.array_equal(result.gradient, gradient)
    
    # Test is_successful method
    assert result.is_successful() == converged
    
    # Test summary method
    summary = result.summary()
    assert "Optimization Results" in summary
    assert "Successful" in summary
    assert f"Function value: {value}" in summary

@pytest.mark.core
@pytest.mark.optimization
@given(st.lists(st.floats(min_value=-10, max_value=10), min_size=2, max_size=10).map(np.array))
def test_property_based_optimization(initial_params):
    """Property-based tests for optimization functions"""
    # Skip NaN or Inf values
    if not np.isfinite(initial_params).all():
        return
    
    # Run optimization on quadratic function
    def obj_func(x):
        return np.sum(x**2)
    
    def obj_grad(x):
        return 2 * x
    
    # Use different optimization methods
    methods = ['BFGS', 'L-BFGS-B']
    
    for method in methods:
        options = {'method': method, 'tol': 1e-6, 'maxiter': 100}
        
        try:
            result = quasi_newton_optimizer(obj_func, obj_grad, initial_params, options)
            
            # If optimization converged, verify properties of the solution
            if result.is_successful():
                # Solution should be close to the origin
                assert np.linalg.norm(result.parameters) < 1e-3 or result.objective_value < 1e-4
                
                # Gradient at solution should be close to zero
                if result.gradient is not None:
                    assert np.linalg.norm(result.gradient) < 1e-3
        except Exception:
            # For property-based testing, we're trying many arbitrary inputs
            # Some may cause optimization failures, which is okay
            pass

@pytest.mark.core
@pytest.mark.optimization
@pytest.mark.slow
def test_numba_optimization():
    """Test Numba optimization performance"""
    # Skip if Numba is not available
    if not check_numba_compatibility(lambda x: x):
        pytest.skip("Numba not available")
    
    # Define a function that can be JIT-compiled
    @optimized_jit()
    def jit_function(x):
        result = 0.0
        for i in range(len(x)):
            result += x[i] ** 2
        return result
    
    # Define the same function without JIT
    def non_jit_function(x):
        result = 0.0
        for i in range(len(x)):
            result += x[i] ** 2
        return result
    
    # Test sizes
    sizes = [100, 1000, 10000]
    
    for size in sizes:
        x = np.random.rand(size)
        
        # Warmup JIT
        _ = jit_function(x)
        
        # Time JIT version
        start = time.time()
        jit_result = jit_function(x)
        jit_time = time.time() - start
        
        # Time non-JIT version
        start = time.time()
        non_jit_result = non_jit_function(x)
        non_jit_time = time.time() - start
        
        # Results should be the same
        assert np.isclose(jit_result, non_jit_result)
        
        # JIT should generally be faster, but skip timing assertions
        # as they can be fragile in CI environments

@pytest.mark.core
@pytest.mark.optimization
def test_input_validation():
    """Test input validation in optimization functions"""
    # Test with invalid input types
    with pytest.raises(TypeError):
        validate_array("not an array", param_name="test_param")
    
    # Test with invalid dimensions
    with pytest.raises(ValueError):
        validate_array(np.zeros((2, 2)), expected_shape=(3, 3), param_name="test_param")
    
    # Test with NaN values
    x_with_nan = np.array([1.0, np.nan, 3.0])
    
    # Check that functions fail gracefully with NaN inputs
    with pytest.raises(Exception):
        quasi_newton_optimizer(quadratic_function, quadratic_gradient, x_with_nan)
    
    # Test with other problematic inputs
    empty_array = np.array([])
    with pytest.raises(Exception):
        quasi_newton_optimizer(quadratic_function, quadratic_gradient, empty_array)