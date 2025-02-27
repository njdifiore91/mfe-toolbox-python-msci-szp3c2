"""
Numba optimization helpers for the MFE Toolbox.

This module provides decorators and utilities for applying Numba JIT compilation
to performance-critical functions across the MFE Toolbox, serving as a critical
performance layer replacing legacy MEX functions with Python-native optimizations.
"""
import functools
import logging
import warnings
import multiprocessing
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union, cast

# Try to import numba, but handle the case where it's not available
try:
    import numba
    from numba import jit, njit, vectorize
    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False
    warnings.warn(
        "Numba not available. Performance will be degraded for computationally intensive operations."
    )

import numpy as np  # numpy 1.26.3

# Set up module logger
logger = logging.getLogger(__name__)

# Default Numba JIT compilation options
DEFAULT_NUMBA_OPTIONS = {
    'nopython': True,
    'cache': True,
    'fastmath': True
}

# Type variables for better type hints
F = TypeVar('F', bound=Callable[..., Any])

# List of exports
__all__ = [
    'optimized_jit', 
    'fallback_to_python', 
    'parallel_jit', 
    'check_numba_compatibility',
    'get_numba_function', 
    'vectorized_jit', 
    'configure_numba_threading',
    'create_specialized_function', 
    'DEFAULT_NUMBA_OPTIONS',
    'jit_garch_recursion',
    'jit_garch_likelihood'
]


def check_numba_compatibility(func: Callable) -> bool:
    """
    Checks if Numba is available and determines whether a function can be JIT-compiled.
    
    Parameters
    ----------
    func : callable
        The function to check for Numba compatibility
        
    Returns
    -------
    bool
        True if the function can be compiled with Numba, False otherwise
    """
    if not HAVE_NUMBA:
        return False
    
    try:
        # Try to compile the function with nopython mode
        if HAVE_NUMBA:
            numba.njit(func)
        return True
    except Exception as e:
        logger.debug(f"Function {func.__name__} cannot be compiled with Numba: {str(e)}")
        return False


def optimized_jit(options: Optional[Dict[str, Any]] = None) -> Callable[[F], F]:
    """
    Decorator that applies Numba JIT compilation to a function with optimal settings.
    
    Parameters
    ----------
    options : dict, optional
        Options to pass to numba.jit. If None, DEFAULT_NUMBA_OPTIONS will be used.
        
    Returns
    -------
    callable
        Decorator function that applies JIT optimization to the target function if
        Numba is available, otherwise returns the original function.
    
    Examples
    --------
    >>> @optimized_jit()
    ... def fast_function(x, y):
    ...     return x + y
    """
    if options is None:
        options = DEFAULT_NUMBA_OPTIONS.copy()
    
    def decorator(func: F) -> F:
        if HAVE_NUMBA:
            try:
                return cast(F, numba.jit(**options)(func))
            except Exception as e:
                logger.warning(f"Failed to apply Numba JIT to {func.__name__}: {str(e)}")
                return func
        else:
            warnings.warn(
                f"Numba not available. Function {func.__name__} will run in pure Python mode."
            )
            return func
    
    return decorator


def fallback_to_python(options: Optional[Dict[str, Any]] = None) -> Callable[[F], F]:
    """
    Decorator that attempts to apply Numba JIT compilation but gracefully falls back
    to pure Python if compilation fails.
    
    Parameters
    ----------
    options : dict, optional
        Options to pass to numba.jit. If None, DEFAULT_NUMBA_OPTIONS will be used.
        
    Returns
    -------
    callable
        Decorator function that applies JIT optimization when possible, or returns the
        original function as fallback.
    
    Examples
    --------
    >>> @fallback_to_python()
    ... def computation(array):
    ...     # Complex computation that might not be Numba-compatible
    ...     return result
    """
    if options is None:
        options = DEFAULT_NUMBA_OPTIONS.copy()
    
    def decorator(func: F) -> F:
        if HAVE_NUMBA:
            try:
                return cast(F, numba.jit(**options)(func))
            except Exception as e:
                logger.warning(
                    f"Numba compilation failed for {func.__name__}, falling back to Python implementation: {str(e)}"
                )
                return func
        else:
            warnings.warn(
                f"Numba not available. Function {func.__name__} will run in pure Python mode."
            )
            return func
    
    return decorator


def parallel_jit(options: Optional[Dict[str, Any]] = None) -> Callable[[F], F]:
    """
    Decorator that applies Numba JIT compilation with parallel execution enabled
    for suitable functions.
    
    Parameters
    ----------
    options : dict, optional
        Options to pass to numba.jit. If None, DEFAULT_NUMBA_OPTIONS will be used
        with parallel=True added.
        
    Returns
    -------
    callable
        Decorator function that applies parallel JIT optimization to the target function
        if Numba is available, otherwise returns the original function.
    
    Examples
    --------
    >>> @parallel_jit()
    ... def parallel_computation(array):
    ...     # Computation that can benefit from parallelization
    ...     return result
    """
    if options is None:
        options = DEFAULT_NUMBA_OPTIONS.copy()
    options['parallel'] = True
    
    def decorator(func: F) -> F:
        if HAVE_NUMBA:
            try:
                return cast(F, numba.jit(**options)(func))
            except Exception as e:
                logger.warning(
                    f"Failed to apply parallel Numba JIT to {func.__name__}: {str(e)}"
                )
                # Try without parallel as fallback
                try:
                    non_parallel_options = options.copy()
                    non_parallel_options['parallel'] = False
                    return cast(F, numba.jit(**non_parallel_options)(func))
                except Exception:
                    return func
        else:
            warnings.warn(
                f"Numba not available. Function {func.__name__} will run in pure Python mode without parallelization."
            )
            return func
    
    return decorator


def vectorized_jit(signatures: Optional[List[str]] = None, 
                  options: Optional[Dict[str, Any]] = None) -> Callable[[F], F]:
    """
    Decorator that applies Numba vectorization to elementwise functions for
    optimized array operations.
    
    Parameters
    ----------
    signatures : list of str, optional
        Type signatures for vectorization in the format accepted by numba.vectorize.
    options : dict, optional
        Options to pass to numba.vectorize.
        
    Returns
    -------
    callable
        Decorator function that applies vectorized optimization if available.
    
    Examples
    --------
    >>> @vectorized_jit(['float64(float64, float64)'])
    ... def fast_add(x, y):
    ...     return x + y
    """
    if options is None:
        options = {'nopython': True, 'cache': True}
    
    def decorator(func: F) -> Callable:
        if HAVE_NUMBA and signatures:
            try:
                return numba.vectorize(signatures, **options)(func)
            except Exception as e:
                logger.warning(
                    f"Numba vectorization failed for {func.__name__}, using numpy.vectorize as fallback: {str(e)}"
                )
                return np.vectorize(func)
        elif HAVE_NUMBA:
            # No signatures provided, try to infer
            try:
                return numba.vectorize(**options)(func)
            except Exception as e:
                logger.warning(
                    f"Numba vectorization failed for {func.__name__}, using numpy.vectorize as fallback: {str(e)}"
                )
                return np.vectorize(func)
        else:
            warnings.warn(
                f"Numba not available. Function {func.__name__} will use numpy.vectorize instead."
            )
            return np.vectorize(func)
    
    return decorator


def get_numba_function(func: Callable, signature: Optional[str] = None,
                      options: Optional[Dict[str, Any]] = None) -> Callable:
    """
    Creates a JIT-compiled version of a function with specific type signatures.
    
    Parameters
    ----------
    func : callable
        The function to compile
    signature : str, optional
        Type signature string in the format accepted by numba.njit
    options : dict, optional
        Options to pass to numba.njit
        
    Returns
    -------
    callable
        JIT-compiled function with specified signature if Numba is available,
        otherwise the original function
    
    Examples
    --------
    >>> def add(a, b):
    ...     return a + b
    ...
    >>> fast_add = get_numba_function(add, "float64(float64, float64)")
    """
    if options is None:
        options = DEFAULT_NUMBA_OPTIONS.copy()
    
    if HAVE_NUMBA:
        try:
            if signature:
                return numba.njit(signature, **options)(func)
            else:
                return numba.njit(**options)(func)
        except Exception as e:
            logger.warning(
                f"Failed to create Numba function for {func.__name__}: {str(e)}"
            )
            return func
    else:
        warnings.warn(
            f"Numba not available. Function {func.__name__} will run in pure Python mode."
        )
        return func


def configure_numba_threading(num_threads: Optional[int] = None) -> None:
    """
    Configures Numba's threading behavior for optimal performance.
    
    Parameters
    ----------
    num_threads : int, optional
        Number of threads for Numba to use. If None, uses CPU count.
    
    Returns
    -------
    None
        Configuration is applied directly.
    """
    if not HAVE_NUMBA:
        warnings.warn("Numba not available. Threading configuration has no effect.")
        return
    
    # Determine number of threads if not provided
    if num_threads is None:
        try:
            num_threads = multiprocessing.cpu_count()
        except Exception:
            num_threads = 1
    
    try:
        # Set threading layer
        if hasattr(numba, 'set_threading_layer'):
            numba.set_threading_layer('threadsafe')
        
        # Configure number of threads
        if hasattr(numba, 'set_num_threads'):
            numba.set_num_threads(num_threads)
        
        logger.info(f"Numba configured to use {num_threads} threads")
    except Exception as e:
        logger.warning(f"Failed to configure Numba threading: {str(e)}")


def create_specialized_function(func: Callable, signature: str,
                               options: Optional[Dict[str, Any]] = None) -> Callable:
    """
    Creates a specialized Numba function for specific input types.
    
    Parameters
    ----------
    func : callable
        The function to specialize
    signature : str
        Type signature string specifying input and output types
    options : dict, optional
        Options to pass to numba.njit
        
    Returns
    -------
    callable
        Type-specialized JIT-compiled function if Numba is available,
        otherwise the original function
    
    Examples
    --------
    >>> def compute_stats(array):
    ...     return array.mean(), array.std()
    ...
    >>> fast_stats = create_specialized_function(
    ...     compute_stats, 
    ...     "Tuple(float64, float64)(float64[:])"
    ... )
    """
    if options is None:
        options = DEFAULT_NUMBA_OPTIONS.copy()
    
    if HAVE_NUMBA:
        try:
            return numba.njit(signature, **options)(func)
        except Exception as e:
            logger.warning(
                f"Failed to create specialized function for {func.__name__}: {str(e)}"
            )
            return func
    else:
        warnings.warn(
            f"Numba not available. Function {func.__name__} will run in pure Python mode."
        )
        return func


@optimized_jit()
def jit_garch_recursion(parameters: np.ndarray, data: np.ndarray, 
                       p: int, q: int) -> np.ndarray:
    """
    Numba-optimized implementation of GARCH variance recursion for
    performance-critical volatility calculations.
    
    Parameters
    ----------
    parameters : ndarray
        Array of GARCH parameters [omega, alpha_1, ..., alpha_q, beta_1, ..., beta_p]
    data : ndarray
        Array of return data for GARCH modeling
    p : int
        Number of GARCH lags (beta parameters)
    q : int
        Number of ARCH lags (alpha parameters)
        
    Returns
    -------
    ndarray
        Array of conditional variances
        
    Notes
    -----
    This is an optimized implementation that replaces legacy MEX functions
    with Numba-accelerated Python code for the core GARCH recursion.
    """
    T = len(data)
    omega = parameters[0]
    alpha = parameters[1:q+1]
    beta = parameters[q+1:q+p+1]
    
    # Initialize variance array
    variance = np.zeros_like(data)
    
    # Compute unconditional variance for initialization
    uncond_var = omega / (1.0 - np.sum(alpha) - np.sum(beta))
    
    # Set initial values using unconditional variance
    variance[:max(p, q)] = uncond_var
    
    # Main recursion loop
    for t in range(max(p, q), T):
        # ARCH component (alpha * squared returns)
        arch_component = 0.0
        for i in range(q):
            if t-i-1 >= 0:
                arch_component += alpha[i] * data[t-i-1]**2
        
        # GARCH component (beta * past variances)
        garch_component = 0.0
        for j in range(p):
            if t-j-1 >= 0:
                garch_component += beta[j] * variance[t-j-1]
        
        # Combine components for conditional variance
        variance[t] = omega + arch_component + garch_component
    
    return variance


@optimized_jit()
def jit_garch_likelihood(parameters: np.ndarray, data: np.ndarray, 
                        p: int, q: int, 
                        distribution_logpdf: Callable) -> float:
    """
    Numba-optimized implementation of GARCH likelihood calculation for
    parameter estimation.
    
    Parameters
    ----------
    parameters : ndarray
        Array of GARCH parameters [omega, alpha_1, ..., alpha_q, beta_1, ..., beta_p]
    data : ndarray
        Array of return data for GARCH modeling
    p : int
        Number of GARCH lags (beta parameters)
    q : int
        Number of ARCH lags (alpha parameters)
    distribution_logpdf : callable
        Function to compute log PDF of the error distribution
        
    Returns
    -------
    float
        Negative log-likelihood value (for minimization)
        
    Notes
    -----
    This function computes the GARCH log-likelihood with a specified error
    distribution, using Numba optimization for performance.
    """
    # Compute conditional variances
    variance = jit_garch_recursion(parameters, data, p, q)
    
    # Skip the burn-in period
    effective_T = len(data) - max(p, q)
    
    # Initialize log-likelihood
    loglike = 0.0
    
    # Compute log-likelihood using standardized residuals
    for t in range(max(p, q), len(data)):
        if variance[t] <= 0:
            return 1e10  # Large penalty for invalid variances
        
        # Standardized residual
        std_resid = data[t] / np.sqrt(variance[t])
        
        # Apply distribution log-pdf
        term = distribution_logpdf(std_resid)
        
        # Add variance adjustment
        loglike += term - 0.5 * np.log(variance[t])
    
    # Return negative log-likelihood for minimization
    return -loglike