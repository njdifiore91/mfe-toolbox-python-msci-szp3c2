"""
MFE Toolbox - Core Optimization Module

This module provides optimization routines for financial econometric models, leveraging
Numba for performance and providing both synchronous and asynchronous interfaces.

The module implements BFGS and other quasi-Newton optimization methods, numerical gradient
computation, line search algorithms, and constrained optimization, supporting both
synchronous and asynchronous execution patterns.
"""

import asyncio  # Python 3.12
import numpy as np  # numpy 1.26.3 
from scipy import optimize  # scipy 1.11.4
from numba import jit  # numba 0.59.0
from typing import AsyncIterator, Dict, List, Optional, Protocol, Tuple, TypeVar, Union, cast  # Python 3.12
from dataclasses import dataclass  # Python 3.12

# Internal imports
from ..utils.validation import validate_array
from ..utils.numba_helpers import jit_decorator
from ..utils.async_helpers import AsyncOperation

# Type variable for generic functions
T = TypeVar('T')

# Constants for optimization
CONVERGENCE_ATOL = 1e-8
MAX_ITERATIONS = 1000
OPTIMIZATION_METHODS = [
    'BFGS', 'CG', 'Newton-CG', 'L-BFGS-B', 'TNC', 'SLSQP', 
    'trust-constr', 'dogleg', 'trust-ncg', 'trust-exact', 'trust-krylov'
]


@jit(nopython=True)
def bfgs_optimization(
    objective_gradient,
    initial_params: np.ndarray,
    bounds: Optional[np.ndarray] = None,
    max_iterations: Optional[int] = None,
    tolerance: Optional[float] = None
) -> Tuple[np.ndarray, float, bool, int]:
    """
    BFGS optimization algorithm implementation with Numba acceleration.
    
    Parameters
    ----------
    objective_gradient : Callable[[np.ndarray], Tuple[float, np.ndarray]]
        Function that returns both objective value and gradient
    initial_params : np.ndarray
        Initial parameter values
    bounds : Optional[np.ndarray], default=None
        Parameter bounds as a 2D array of shape (n_params, 2)
    max_iterations : Optional[int], default=None
        Maximum number of iterations (defaults to MAX_ITERATIONS)
    tolerance : Optional[float], default=None
        Convergence tolerance (defaults to CONVERGENCE_ATOL)
        
    Returns
    -------
    Tuple[np.ndarray, float, bool, int]
        Optimized parameters, function value, convergence flag, and iteration count
    """
    # Set default values
    if max_iterations is None:
        max_iterations = 1000
    if tolerance is None:
        tolerance = 1e-8
        
    n = len(initial_params)
    x = initial_params.copy()
    
    # Initialize BFGS inverse Hessian approximation to identity matrix
    H = np.eye(n)
    
    # Initial function value and gradient
    f, g = objective_gradient(x)
    
    # Initialize iteration counter
    iterations = 0
    converged = False
    
    while iterations < max_iterations:
        # Search direction is -H*g
        p = -np.dot(H, g)
        
        # Apply bounds if provided
        if bounds is not None:
            for i in range(n):
                if p[i] < 0 and x[i] <= bounds[i, 0]:
                    p[i] = 0
                elif p[i] > 0 and x[i] >= bounds[i, 1]:
                    p[i] = 0
        
        # Line search to find step size
        alpha = 1.0
        c1 = 1e-4
        
        # Backtracking line search
        while True:
            x_new = x + alpha * p
            
            # Apply bounds if provided
            if bounds is not None:
                for i in range(n):
                    if x_new[i] < bounds[i, 0]:
                        x_new[i] = bounds[i, 0]
                    elif x_new[i] > bounds[i, 1]:
                        x_new[i] = bounds[i, 1]
            
            f_new, g_new = objective_gradient(x_new)
            
            if f_new <= f + c1 * alpha * np.dot(g, p):
                break
                
            alpha *= 0.5
            if alpha < 1e-10:
                # Line search failed, return current best
                return x, f, False, iterations
        
        # Compute step and gradient differences
        s = x_new - x
        y = g_new - g
        
        # Update only if sTy > 0 (curvature condition)
        sTy = np.dot(s, y)
        if sTy > 1e-10:
            # BFGS update formula
            Hy = np.dot(H, y)
            HysT = np.outer(Hy, s)
            syT = np.outer(s, y)
            ssT = np.outer(s, s)
            
            H = H + (1.0 + np.dot(y, Hy) / sTy) * ssT / sTy - (HysT + HysT.T) / sTy
        
        # Check for convergence
        if np.linalg.norm(g_new) < tolerance:
            converged = True
            x = x_new
            f = f_new
            break
            
        # Update current position, value, and gradient
        x = x_new
        f = f_new
        g = g_new
        
        iterations += 1
    
    return x, f, converged, iterations


def quasi_newton_optimizer(
    objective,
    gradient,
    initial_params: np.ndarray,
    options: Optional[Dict[str, Any]] = None
) -> 'OptimizationResult':
    """
    Quasi-Newton optimization with flexible configuration.
    
    Parameters
    ----------
    objective : Callable[[np.ndarray], float]
        Objective function to minimize
    gradient : Callable[[np.ndarray], np.ndarray]
        Gradient function of the objective
    initial_params : np.ndarray
        Initial parameter values
    options : Optional[Dict[str, Any]], default=None
        Optimization options including:
        - method: Optimization method ('BFGS', 'L-BFGS-B', etc.)
        - bounds: Parameter bounds as a list of (min, max) tuples
        - tol: Tolerance for convergence
        - maxiter: Maximum number of iterations
        - disp: Boolean for verbose output
        
    Returns
    -------
    OptimizationResult
        Result object containing optimized parameters and convergence information
    """
    # Process options
    if options is None:
        options = {}
    
    method = options.get('method', 'BFGS')
    bounds = options.get('bounds', None)
    tol = options.get('tol', CONVERGENCE_ATOL)
    maxiter = options.get('maxiter', MAX_ITERATIONS)
    disp = options.get('disp', False)
    
    # Validate initial parameters
    initial_params = validate_array(initial_params, param_name="initial_params")
    
    # Handle the case where we use the custom BFGS implementation
    if method == 'BFGS' and options.get('use_numba', True) and bounds is None:
        # Define a function that returns both value and gradient
        def objective_gradient(params):
            return objective(params), gradient(params)
        
        # Run the custom BFGS implementation
        result_params, result_value, converged, iterations = bfgs_optimization(
            objective_gradient, 
            initial_params,
            max_iterations=maxiter,
            tolerance=tol
        )
        
        # Create result object
        return OptimizationResult(
            parameters=result_params,
            objective_value=result_value,
            converged=converged,
            iterations=iterations,
            message="Optimization successful" if converged else "Maximum iterations reached",
            gradient=gradient(result_params),
            hessian=None,
            metadata={'method': 'BFGS (Numba)', 'iterations': iterations}
        )
    
    # Otherwise, use SciPy's implementation
    scipy_options = {
        'gtol': tol,
        'maxiter': maxiter,
        'disp': disp
    }
    
    # Prepare bounds for SciPy
    scipy_bounds = None
    if bounds is not None:
        if isinstance(bounds, list):
            scipy_bounds = bounds
        elif isinstance(bounds, np.ndarray):
            scipy_bounds = [(bounds[i, 0], bounds[i, 1]) for i in range(bounds.shape[0])]
    
    # Run the optimization
    result = optimize.minimize(
        objective,
        initial_params,
        method=method,
        jac=gradient,
        bounds=scipy_bounds,
        options=scipy_options
    )
    
    # Create result object
    return OptimizationResult.from_scipy_result(result)


async def async_optimize(
    objective,
    gradient,
    initial_params: np.ndarray,
    options: Optional[Dict[str, Any]] = None
) -> AsyncIterator[Tuple[int, np.ndarray, float]]:
    """
    Asynchronous optimization function for non-blocking operation.
    
    Parameters
    ----------
    objective : Callable[[np.ndarray], float]
        Objective function to minimize
    gradient : Callable[[np.ndarray], np.ndarray]
        Gradient function of the objective
    initial_params : np.ndarray
        Initial parameter values
    options : Optional[Dict[str, Any]], default=None
        Optimization options
        
    Returns
    -------
    AsyncIterator[Tuple[int, np.ndarray, float]]
        Async iterator yielding iteration count, current parameters, and function value
    """
    # Process options
    if options is None:
        options = {}
    
    method = options.get('method', 'BFGS')
    bounds = options.get('bounds', None)
    tol = options.get('tol', CONVERGENCE_ATOL)
    maxiter = options.get('maxiter', MAX_ITERATIONS)
    yield_every = options.get('yield_every', 1)  # How often to yield progress
    
    # Validate initial parameters
    initial_params = validate_array(initial_params, param_name="initial_params")
    
    # Custom implementation for async BFGS
    if method == 'BFGS':
        n = len(initial_params)
        x = initial_params.copy()
        
        # Initialize BFGS inverse Hessian approximation to identity matrix
        H = np.eye(n)
        
        # Initial function value and gradient
        f = objective(x)
        g = gradient(x)
        
        # Yield initial state
        yield 0, x.copy(), f
        
        # Initialize iteration counter
        iterations = 0
        
        while iterations < maxiter:
            # Allow other tasks to run
            await asyncio.sleep(0)
            
            # Search direction is -H*g
            p = -np.dot(H, g)
            
            # Apply bounds if provided
            if bounds is not None:
                for i in range(n):
                    if isinstance(bounds[i], tuple):
                        if p[i] < 0 and x[i] <= bounds[i][0]:
                            p[i] = 0
                        elif p[i] > 0 and x[i] >= bounds[i][1]:
                            p[i] = 0
            
            # Line search to find step size
            alpha, x_new, f_new = line_search(objective, x, p, 1.0, 1e-4, 0.9)
            g_new = gradient(x_new)
            
            # Compute step and gradient differences
            s = x_new - x
            y = g_new - g
            
            # Update only if sTy > 0 (curvature condition)
            sTy = np.dot(s, y)
            if sTy > 1e-10:
                # BFGS update formula
                Hy = np.dot(H, y)
                H = H + (1.0 + np.dot(y, Hy) / sTy) * np.outer(s, s) / sTy - (np.outer(Hy, s) + np.outer(s, Hy)) / sTy
            
            # Update current position, value, and gradient
            x = x_new
            f = f_new
            g = g_new
            
            iterations += 1
            
            # Yield progress every yield_every iterations
            if iterations % yield_every == 0:
                yield iterations, x.copy(), f
            
            # Check for convergence
            if np.linalg.norm(g) < tol:
                break
        
        # Yield final result if not already yielded
        if iterations % yield_every != 0:
            yield iterations, x, f
    else:
        # Use SciPy's optimization but in a non-blocking manner
        
        # Run the optimization in a non-blocking manner
        result = await asyncio.to_thread(
            optimize.minimize,
            objective,
            initial_params,
            method=method,
            jac=gradient,
            bounds=bounds,
            options={'gtol': tol, 'maxiter': maxiter}
        )
        
        # Yield final result
        yield result.nit, result.x, result.fun


@jit_decorator()
def line_search(
    objective,
    x_current: np.ndarray,
    direction: np.ndarray,
    initial_step_size: float,
    c1: float,
    c2: float
) -> Tuple[float, np.ndarray, float]:
    """
    Line search algorithm to find step size in gradient-based optimization.
    
    Parameters
    ----------
    objective : Callable[[np.ndarray], float]
        Objective function to minimize
    x_current : np.ndarray
        Current parameter values
    direction : np.ndarray
        Search direction
    initial_step_size : float
        Initial step size to try
    c1 : float
        Parameter for Armijo condition
    c2 : float
        Parameter for curvature condition
        
    Returns
    -------
    Tuple[float, np.ndarray, float]
        Step size, new parameters, and function value at new parameters
    """
    # Current function value
    f_current = objective(x_current)
    
    # Initial step size and new point
    alpha = initial_step_size
    x_new = x_current + alpha * direction
    f_new = objective(x_new)
    
    # Backtracking line search
    max_iter = 25
    iter_count = 0
    
    while f_new > f_current + c1 * alpha * np.dot(direction, direction) and iter_count < max_iter:
        # Reduce step size
        alpha *= 0.5
        
        # Compute new point and function value
        x_new = x_current + alpha * direction
        f_new = objective(x_new)
        
        iter_count += 1
    
    return alpha, x_new, f_new


@jit_decorator()
def compute_numerical_gradient(
    func,
    x: np.ndarray,
    epsilon: Optional[float] = None
) -> np.ndarray:
    """
    Compute numerical gradient using central difference method.
    
    Parameters
    ----------
    func : Callable[[np.ndarray], float]
        Function to differentiate
    x : np.ndarray
        Point at which to compute the gradient
    epsilon : Optional[float], default=None
        Step size for finite difference (defaults to sqrt(machine epsilon))
        
    Returns
    -------
    np.ndarray
        Numerical gradient vector
    """
    # Set default epsilon if not provided
    if epsilon is None:
        epsilon = np.sqrt(np.finfo(float).eps)
    
    n = len(x)
    grad = np.zeros_like(x)
    
    # Central difference method
    for i in range(n):
        x_plus = x.copy()
        x_minus = x.copy()
        
        x_plus[i] += epsilon
        x_minus[i] -= epsilon
        
        grad[i] = (func(x_plus) - func(x_minus)) / (2.0 * epsilon)
    
    return grad


def check_gradient(
    func,
    grad,
    x: np.ndarray,
    epsilon: Optional[float] = None,
    rtol: Optional[float] = None
) -> Tuple[bool, float, np.ndarray, np.ndarray]:
    """
    Compare analytical gradient with numerical gradient for verification.
    
    Parameters
    ----------
    func : Callable[[np.ndarray], float]
        Function to check gradient for
    grad : Callable[[np.ndarray], np.ndarray]
        Analytical gradient function
    x : np.ndarray
        Point at which to check the gradient
    epsilon : Optional[float], default=None
        Step size for numerical gradient
    rtol : Optional[float], default=None
        Relative tolerance for comparison (defaults to 1e-4)
        
    Returns
    -------
    Tuple[bool, float, np.ndarray, np.ndarray]
        Success flag, maximum relative error, analytical gradient, and numerical gradient
    """
    # Set default relative tolerance
    if rtol is None:
        rtol = 1e-4
    
    # Compute analytical gradient
    analytical_grad = grad(x)
    
    # Compute numerical gradient
    numerical_grad = compute_numerical_gradient(func, x, epsilon)
    
    # Calculate relative difference
    abs_diff = np.abs(analytical_grad - numerical_grad)
    abs_vals = np.maximum(np.abs(analytical_grad), np.abs(numerical_grad))
    
    # Handle zeros in denominator
    abs_vals = np.where(abs_vals < 1e-10, 1.0, abs_vals)
    
    # Relative error
    rel_diff = abs_diff / abs_vals
    max_rel_diff = np.max(rel_diff)
    
    # Check if gradient is correct within tolerance
    success = max_rel_diff <= rtol
    
    return success, max_rel_diff, analytical_grad, numerical_grad


def constrained_optimization(
    objective,
    initial_params: np.ndarray,
    constraints: Optional[List[Dict[str, Any]]] = None,
    bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    method: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None
) -> 'OptimizationResult':
    """
    Optimization with inequality and equality constraints.
    
    Parameters
    ----------
    objective : Callable[[np.ndarray], float]
        Objective function to minimize
    initial_params : np.ndarray
        Initial parameter values
    constraints : Optional[List[Dict[str, Any]]], default=None
        List of constraint dictionaries, each containing:
        - type: 'eq' for equality or 'ineq' for inequality
        - fun: constraint function
        - jac: jacobian of constraint function (optional)
    bounds : Optional[Tuple[np.ndarray, np.ndarray]], default=None
        Bounds for parameters as (lower_bounds, upper_bounds)
    method : Optional[str], default=None
        Optimization method (defaults to 'SLSQP' if None)
    options : Optional[Dict[str, Any]], default=None
        Additional options for the optimizer
        
    Returns
    -------
    OptimizationResult
        Result object containing optimized parameters and convergence information
    """
    # Process method
    if method is None:
        # Choose appropriate method based on constraints
        if constraints is not None:
            method = 'SLSQP'
        else:
            method = 'L-BFGS-B'
    
    # Process options
    if options is None:
        options = {}
    
    # Convert bounds to SciPy format if needed
    scipy_bounds = None
    if bounds is not None:
        if isinstance(bounds, tuple) and len(bounds) == 2:
            lower_bounds, upper_bounds = bounds
            scipy_bounds = [(lb, ub) for lb, ub in zip(lower_bounds, upper_bounds)]
    
    # Run the optimization
    result = optimize.minimize(
        objective,
        initial_params,
        method=method,
        bounds=scipy_bounds,
        constraints=constraints,
        options=options
    )
    
    # Create result object
    return OptimizationResult.from_scipy_result(result)


@dataclass
class OptimizationResult:
    """
    Container for optimization results with comprehensive metadata.
    """
    parameters: np.ndarray
    objective_value: float
    converged: bool
    iterations: int
    message: str
    gradient: Optional[np.ndarray] = None
    hessian: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_scipy_result(cls, scipy_result):
        """
        Create OptimizationResult from SciPy OptimizeResult.
        
        Parameters
        ----------
        scipy_result : Any
            SciPy optimization result object
            
        Returns
        -------
        OptimizationResult
            New OptimizationResult instance
        """
        # Extract metadata from SciPy result
        metadata = {}
        for key in scipy_result.__dict__:
            if key not in ['x', 'fun', 'success', 'nit', 'message', 'jac', 'hess']:
                metadata[key] = getattr(scipy_result, key)
        
        # Create and return result
        return cls(
            parameters=scipy_result.x,
            objective_value=scipy_result.fun,
            converged=scipy_result.success,
            iterations=scipy_result.nit if hasattr(scipy_result, 'nit') else 0,
            message=str(scipy_result.message),
            gradient=scipy_result.jac if hasattr(scipy_result, 'jac') else None,
            hessian=scipy_result.hess if hasattr(scipy_result, 'hess') else None,
            metadata=metadata
        )
    
    def is_successful(self) -> bool:
        """
        Check if optimization completed successfully.
        
        Returns
        -------
        bool
            True if optimization converged successfully
        """
        return self.converged
    
    def summary(self) -> str:
        """
        Generate human-readable summary of optimization result.
        
        Returns
        -------
        str
            Formatted summary string
        """
        lines = [
            "Optimization Results:",
            f"  Status: {'Successful' if self.converged else 'Failed'}",
            f"  Message: {self.message}",
            f"  Iterations: {self.iterations}",
            f"  Function value: {self.objective_value:.6g}",
            "  Parameters:"
        ]
        
        for i, param in enumerate(self.parameters):
            lines.append(f"    {i}: {param:.8g}")
        
        if self.gradient is not None:
            lines.append("  Gradient norm: {:.6g}".format(np.linalg.norm(self.gradient)))
        
        if self.metadata:
            lines.append("  Additional information:")
            for key, value in self.metadata.items():
                lines.append(f"    {key}: {value}")
        
        return "\n".join(lines)


class Optimizer:
    """
    Configurable optimization engine with multiple algorithms and asynchronous support.
    """
    
    def __init__(self, options: Optional[Dict[str, Any]] = None):
        """
        Initialize the Optimizer with default configuration.
        
        Parameters
        ----------
        options : Optional[Dict[str, Any]], default=None
            Default options for optimization
        """
        # Default optimization options
        self._default_options = {
            'method': 'BFGS',
            'tol': CONVERGENCE_ATOL,
            'maxiter': MAX_ITERATIONS,
            'disp': False,
            'use_numba': True
        }
        
        # Update with user-provided options
        if options is not None:
            self._default_options.update(options)
        
        # Store current options
        self._current_options = self._default_options.copy()
        
        # Supported optimization methods
        self._supported_methods = OPTIMIZATION_METHODS
    
    def minimize(
        self,
        objective,
        initial_params: np.ndarray,
        gradient: Optional[callable] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> OptimizationResult:
        """
        Minimize objective function with optional gradient.
        
        Parameters
        ----------
        objective : Callable[[np.ndarray], float]
            Objective function to minimize
        initial_params : np.ndarray
            Initial parameter values
        gradient : Optional[Callable[[np.ndarray], np.ndarray]], default=None
            Gradient function of the objective (if None, numerical gradient is used)
        options : Optional[Dict[str, Any]], default=None
            Optimization options
            
        Returns
        -------
        OptimizationResult
            Optimization result container
        """
        # Merge options
        merged_options = self._current_options.copy()
        if options is not None:
            merged_options.update(options)
        
        # Validate initial parameters
        initial_params = validate_array(initial_params, param_name="initial_params")
        
        # Use numerical gradient if not provided
        if gradient is None:
            gradient = lambda x: compute_numerical_gradient(objective, x)
        
        # Select appropriate method
        method = merged_options.get('method', 'BFGS')
        
        if method not in self._supported_methods:
            raise ValueError(f"Unsupported optimization method: {method}. "
                            f"Supported methods: {', '.join(self._supported_methods)}")
        
        # Run optimization
        return quasi_newton_optimizer(
            objective,
            gradient,
            initial_params,
            merged_options
        )
    
    async def async_minimize(
        self,
        objective,
        initial_params: np.ndarray,
        gradient: Optional[callable] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> AsyncIterator[Tuple[int, np.ndarray, float]]:
        """
        Asynchronous version of minimize function.
        
        Parameters
        ----------
        objective : Callable[[np.ndarray], float]
            Objective function to minimize
        initial_params : np.ndarray
            Initial parameter values
        gradient : Optional[Callable[[np.ndarray], np.ndarray]], default=None
            Gradient function of the objective (if None, numerical gradient is used)
        options : Optional[Dict[str, Any]], default=None
            Optimization options
            
        Returns
        -------
        AsyncIterator[Tuple[int, np.ndarray, float]]
            Async iterator yielding optimization progress
        """
        # Merge options
        merged_options = self._current_options.copy()
        if options is not None:
            merged_options.update(options)
        
        # Validate initial parameters
        initial_params = validate_array(initial_params, param_name="initial_params")
        
        # Use numerical gradient if not provided
        if gradient is None:
            gradient = lambda x: compute_numerical_gradient(objective, x)
        
        # Run asynchronous optimization
        async for result in async_optimize(
            objective,
            gradient,
            initial_params,
            merged_options
        ):
            yield result
    
    def constrained_minimize(
        self,
        objective,
        initial_params: np.ndarray,
        constraints: List[Dict[str, Any]],
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> OptimizationResult:
        """
        Minimize with inequality and equality constraints.
        
        Parameters
        ----------
        objective : Callable[[np.ndarray], float]
            Objective function to minimize
        initial_params : np.ndarray
            Initial parameter values
        constraints : List[Dict[str, Any]]
            List of constraint dictionaries
        bounds : Optional[Tuple[np.ndarray, np.ndarray]], default=None
            Bounds for parameters as (lower_bounds, upper_bounds)
        options : Optional[Dict[str, Any]], default=None
            Optimization options
            
        Returns
        -------
        OptimizationResult
            Optimization result container
        """
        # Merge options
        merged_options = self._current_options.copy()
        if options is not None:
            merged_options.update(options)
        
        # Use SLSQP method for constrained optimization
        method = merged_options.get('method', 'SLSQP')
        
        # Run constrained optimization
        return constrained_optimization(
            objective,
            initial_params,
            constraints,
            bounds,
            method,
            merged_options
        )
    
    def set_options(self, options: Dict[str, Any]) -> None:
        """
        Update optimizer options.
        
        Parameters
        ----------
        options : Dict[str, Any]
            Options to update
        """
        self._current_options.update(options)
    
    def get_options(self) -> Dict[str, Any]:
        """
        Get current optimizer options.
        
        Returns
        -------
        Dict[str, Any]
            Current options dictionary
        """
        return self._current_options.copy()
    
    def reset_options(self) -> None:
        """
        Reset options to default values.
        """
        self._current_options = self._default_options.copy()


class AsyncOptimizer:
    """
    Asynchronous optimization manager for long-running optimization tasks.
    """
    
    def __init__(self, options: Optional[Dict[str, Any]] = None):
        """
        Initialize AsyncOptimizer with optional configuration.
        
        Parameters
        ----------
        options : Optional[Dict[str, Any]], default=None
            Default options for optimization
        """
        self._optimizer = Optimizer(options)
        self._tasks = {}
    
    def start_optimization(
        self,
        task_id: str,
        objective,
        initial_params: np.ndarray,
        gradient: Optional[callable] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Start an asynchronous optimization task.
        
        Parameters
        ----------
        task_id : str
            Unique identifier for the task
        objective : Callable[[np.ndarray], float]
            Objective function to minimize
        initial_params : np.ndarray
            Initial parameter values
        gradient : Optional[Callable[[np.ndarray], np.ndarray]], default=None
            Gradient function of the objective
        options : Optional[Dict[str, Any]], default=None
            Optimization options
        """
        if task_id in self._tasks:
            raise ValueError(f"Task with ID '{task_id}' already exists")
        
        # Create an async generator for optimization progress
        async def optimization_task():
            async for iter_count, params, value in self._optimizer.async_minimize(
                objective, initial_params, gradient, options
            ):
                yield iter_count, params.copy(), value
        
        # Wrap in AsyncOperation
        task = AsyncOperation(optimization_task())
        self._tasks[task_id] = task
        
        # Start the task
        task.run()
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        Get status of an optimization task.
        
        Parameters
        ----------
        task_id : str
            Task identifier
            
        Returns
        -------
        Dict[str, Any]
            Task status information
        """
        if task_id not in self._tasks:
            raise ValueError(f"No task found with ID '{task_id}'")
        
        task = self._tasks[task_id]
        
        return {
            'running': task.is_running(),
            'completed': task.is_completed(),
            'progress': task.get_progress(),
            'task_id': task_id
        }
    
    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a running optimization task.
        
        Parameters
        ----------
        task_id : str
            Task identifier
            
        Returns
        -------
        bool
            True if task was cancelled successfully
        """
        if task_id not in self._tasks:
            return False
        
        task = self._tasks[task_id]
        success = task.cancel()
        
        if success:
            self._tasks.pop(task_id)
        
        return success
    
    def get_result(self, task_id: str) -> Optional[OptimizationResult]:
        """
        Get result of a completed optimization task.
        
        Parameters
        ----------
        task_id : str
            Task identifier
            
        Returns
        -------
        Optional[OptimizationResult]
            Optimization result if task completed, None otherwise
        """
        if task_id not in self._tasks:
            raise ValueError(f"No task found with ID '{task_id}'")
        
        task = self._tasks[task_id]
        
        if not task.is_completed():
            return None
        
        # Get the final result
        result = task.get_result()
        
        # If the result is already an OptimizationResult, return it
        if isinstance(result, OptimizationResult):
            return result
        
        # Otherwise, convert the final iteration result to OptimizationResult
        if isinstance(result, tuple) and len(result) == 3:
            iter_count, params, value = result
            
            return OptimizationResult(
                parameters=params,
                objective_value=value,
                converged=True,  # Assume converged if completed
                iterations=iter_count,
                message="Optimization completed",
                gradient=None,
                hessian=None,
                metadata={'task_id': task_id}
            )
        
        return None
    
    async def wait_for_task(self, task_id: str) -> AsyncIterator[Tuple[int, np.ndarray, float]]:
        """
        Wait for task completion asynchronously.
        
        Parameters
        ----------
        task_id : str
            Task identifier
            
        Returns
        -------
        AsyncIterator[Tuple[int, np.ndarray, float]]
            Async iterator yielding optimization progress
        """
        if task_id not in self._tasks:
            raise ValueError(f"No task found with ID '{task_id}'")
        
        task = self._tasks[task_id]
        
        # Get progress updates from the task
        async for progress in task.get_progress_iterator():
            yield progress