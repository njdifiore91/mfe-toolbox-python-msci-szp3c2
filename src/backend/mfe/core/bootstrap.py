"""
MFE Toolbox - Bootstrap Module

This module provides robust resampling methods for dependent time series data,
including block bootstrap, stationary bootstrap, and moving block bootstrap.
These methods are essential for statistical inference with time series data
where traditional i.i.d. assumptions are violated.

The module includes both synchronous and asynchronous implementations, with
performance-critical functions optimized using Numba JIT compilation. The
asynchronous functions enable non-blocking execution for computationally
intensive bootstrap operations.

Classes:
    Bootstrap: Main interface for bootstrap analysis with various methods
    BootstrapResult: Container for bootstrap results with utilities for
                    inference and visualization

Functions:
    block_bootstrap: Block bootstrap resampling for dependent data
    stationary_bootstrap: Bootstrap with random block lengths
    moving_block_bootstrap: Bootstrap with overlapping blocks
    block_bootstrap_async: Asynchronous block bootstrap
    stationary_bootstrap_async: Asynchronous stationary bootstrap
    moving_block_bootstrap_async: Asynchronous moving block bootstrap
    calculate_bootstrap_ci: Calculate bootstrap confidence intervals
    calculate_bootstrap_pvalue: Calculate p-values for hypothesis tests
"""

import logging
import warnings
import asyncio
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, TypeVar, Coroutine

import numpy as np  # numpy 1.26.3
import scipy.stats  # scipy 1.11.4
import numba  # numba 0.59.0

# Import internal helpers
from ..utils.numba_helpers import optimized_jit, fallback_to_python
from ..utils.validation import (
    validate_array, validate_type, is_positive_integer, is_probability
)
from ..utils.async_helpers import AsyncTask, async_progress

# Set up module logger
logger = logging.getLogger(__name__)


@optimized_jit()
def block_bootstrap(
    data: np.ndarray,
    statistic_func: Callable,
    block_size: int,
    num_bootstrap: int,
    replace: bool = True
) -> np.ndarray:
    """
    Perform block bootstrap resampling for dependent time series data.
    
    Block bootstrap resamples blocks of consecutive observations to preserve
    the dependence structure within each block. This method is suitable for
    stationary time series with short-range dependence.
    
    Parameters
    ----------
    data : numpy.ndarray
        Time series data array (1-dimensional)
    statistic_func : callable
        Function that computes the statistic of interest from a data array
    block_size : int
        Size of blocks to resample
    num_bootstrap : int
        Number of bootstrap samples to generate
    replace : bool, default=True
        Whether to sample blocks with replacement
        
    Returns
    -------
    numpy.ndarray
        Array of bootstrap statistics with shape (num_bootstrap,)
        
    Notes
    -----
    The function is optimized using Numba JIT compilation for performance.
    
    References
    ----------
    Kunsch, H.R. (1989) "The Jackknife and the Bootstrap for General
    Stationary Observations," Annals of Statistics, 17, 1217-1241.
    """
    # Validate inputs
    validate_array(data, param_name="data")
    is_positive_integer(block_size, param_name="block_size")
    is_positive_integer(num_bootstrap, param_name="num_bootstrap")
    
    # Ensure data is 1D array
    if data.ndim != 1:
        raise ValueError("data must be a 1-dimensional array")
    
    # Get data length
    n = len(data)
    
    # Check if block_size is valid
    if block_size > n:
        raise ValueError(f"block_size ({block_size}) cannot be larger than data length ({n})")
    
    # Calculate number of blocks needed to cover the data length
    num_blocks = int(np.ceil(n / block_size))
    
    # Number of possible blocks for sampling
    n_possible_blocks = n - block_size + 1
    
    # Initialize array to store bootstrap statistics
    bootstrap_statistics = np.zeros(num_bootstrap)
    
    # Generate bootstrap samples and compute statistics
    for i in range(num_bootstrap):
        # Create bootstrap sample array
        bootstrap_sample = np.zeros(num_blocks * block_size)
        
        # Generate random block starting indices
        if replace:
            # Sample with replacement
            block_indices = np.random.randint(0, n_possible_blocks, size=num_blocks)
        else:
            # Sample without replacement
            if num_blocks > n_possible_blocks:
                raise ValueError(
                    f"Cannot sample {num_blocks} blocks without replacement "
                    f"when only {n_possible_blocks} blocks are available"
                )
            block_indices = np.random.choice(n_possible_blocks, size=num_blocks, replace=False)
        
        # Construct bootstrap sample from blocks
        for j, idx in enumerate(block_indices):
            start_pos = j * block_size
            end_pos = min((j + 1) * block_size, num_blocks * block_size)
            sample_length = end_pos - start_pos
            
            # Copy block data
            bootstrap_sample[start_pos:end_pos] = data[idx:idx + sample_length]
        
        # Trim bootstrap sample to match original length
        bootstrap_sample = bootstrap_sample[:n]
        
        # Compute statistic
        bootstrap_statistics[i] = statistic_func(bootstrap_sample)
    
    return bootstrap_statistics


@optimized_jit()
def stationary_bootstrap(
    data: np.ndarray,
    statistic_func: Callable,
    probability: float,
    num_bootstrap: int
) -> np.ndarray:
    """
    Perform stationary bootstrap resampling for dependent time series.
    
    Stationary bootstrap uses random block lengths drawn from a geometric
    distribution. This ensures the stationarity of the resampled series
    while preserving the dependence structure.
    
    Parameters
    ----------
    data : numpy.ndarray
        Time series data array (1-dimensional)
    statistic_func : callable
        Function that computes the statistic of interest from a data array
    probability : float
        Probability parameter for the geometric distribution (0 < p ≤ 1)
        which determines the expected block length (1/p)
    num_bootstrap : int
        Number of bootstrap samples to generate
        
    Returns
    -------
    numpy.ndarray
        Array of bootstrap statistics with shape (num_bootstrap,)
        
    Notes
    -----
    The function is optimized using Numba JIT compilation for performance.
    
    References
    ----------
    Politis, D.N. and Romano, J.P. (1994) "The Stationary Bootstrap,"
    Journal of the American Statistical Association, 89, 1303-1313.
    """
    # Validate inputs
    validate_array(data, param_name="data")
    is_probability(probability, param_name="probability")
    is_positive_integer(num_bootstrap, param_name="num_bootstrap")
    
    # Ensure data is 1D array
    if data.ndim != 1:
        raise ValueError("data must be a 1-dimensional array")
    
    # Get data length
    n = len(data)
    
    # Issue warning if probability is too small or too large
    if probability < 0.01:
        warnings.warn(
            f"Small probability value ({probability}) will result in very long blocks "
            f"(average length: {1/probability:.1f}), which may not be suitable "
            f"for data of length {n}"
        )
    elif probability > 0.5:
        warnings.warn(
            f"Large probability value ({probability}) will result in very short blocks "
            f"(average length: {1/probability:.1f}), which may not preserve "
            f"the dependence structure adequately"
        )
    
    # Initialize array to store bootstrap statistics
    bootstrap_statistics = np.zeros(num_bootstrap)
    
    # Generate bootstrap samples and compute statistics
    for i in range(num_bootstrap):
        # Create bootstrap sample array
        bootstrap_sample = np.zeros(n)
        
        # Generate bootstrap sample
        idx = 0
        while idx < n:
            # Random starting position
            start_pos = np.random.randint(0, n)
            
            # Random block length (geometric distribution)
            block_length = np.random.geometric(probability)
            
            # Copy block data (with circular wrapping if needed)
            for j in range(block_length):
                if idx >= n:
                    break
                pos = (start_pos + j) % n  # Circular wrapping
                bootstrap_sample[idx] = data[pos]
                idx += 1
        
        # Compute statistic
        bootstrap_statistics[i] = statistic_func(bootstrap_sample)
    
    return bootstrap_statistics


@optimized_jit()
def moving_block_bootstrap(
    data: np.ndarray,
    statistic_func: Callable,
    block_size: int,
    num_bootstrap: int
) -> np.ndarray:
    """
    Perform moving block bootstrap resampling for dependent time series.
    
    Moving block bootstrap resamples overlapping blocks of consecutive observations
    to preserve the dependence structure. This method is suitable for
    stationary time series with short-range dependence.
    
    Parameters
    ----------
    data : numpy.ndarray
        Time series data array (1-dimensional)
    statistic_func : callable
        Function that computes the statistic of interest from a data array
    block_size : int
        Size of blocks to resample
    num_bootstrap : int
        Number of bootstrap samples to generate
        
    Returns
    -------
    numpy.ndarray
        Array of bootstrap statistics with shape (num_bootstrap,)
        
    Notes
    -----
    The function is optimized using Numba JIT compilation for performance.
    Unlike the standard block bootstrap, moving block bootstrap always uses
    overlapping blocks and samples with replacement.
    
    References
    ----------
    Kunsch, H.R. (1989) "The Jackknife and the Bootstrap for General
    Stationary Observations," Annals of Statistics, 17, 1217-1241.
    """
    # Validate inputs
    validate_array(data, param_name="data")
    is_positive_integer(block_size, param_name="block_size")
    is_positive_integer(num_bootstrap, param_name="num_bootstrap")
    
    # Ensure data is 1D array
    if data.ndim != 1:
        raise ValueError("data must be a 1-dimensional array")
    
    # Get data length
    n = len(data)
    
    # Check if block_size is valid
    if block_size > n:
        raise ValueError(f"block_size ({block_size}) cannot be larger than data length ({n})")
    
    # Calculate number of blocks needed
    num_blocks = int(np.ceil(n / block_size))
    
    # Number of possible blocks (overlapping)
    n_possible_blocks = n - block_size + 1
    
    # Initialize array to store bootstrap statistics
    bootstrap_statistics = np.zeros(num_bootstrap)
    
    # Generate bootstrap samples and compute statistics
    for i in range(num_bootstrap):
        # Create bootstrap sample array
        bootstrap_sample = np.zeros(num_blocks * block_size)
        
        # Generate random block starting indices (with replacement)
        block_indices = np.random.randint(0, n_possible_blocks, size=num_blocks)
        
        # Construct bootstrap sample from blocks
        for j, idx in enumerate(block_indices):
            start_pos = j * block_size
            end_pos = min((j + 1) * block_size, num_blocks * block_size)
            sample_length = end_pos - start_pos
            
            # Copy block data
            bootstrap_sample[start_pos:end_pos] = data[idx:idx + sample_length]
        
        # Trim bootstrap sample to match original length
        bootstrap_sample = bootstrap_sample[:n]
        
        # Compute statistic
        bootstrap_statistics[i] = statistic_func(bootstrap_sample)
    
    return bootstrap_statistics


@async_progress()
async def block_bootstrap_async(
    data: np.ndarray,
    statistic_func: Callable,
    block_size: int,
    num_bootstrap: int,
    replace: bool = True,
    progress_callback: Optional[Callable[[float], None]] = None
) -> np.ndarray:
    """
    Asynchronous implementation of block bootstrap resampling.
    
    This function provides the same functionality as block_bootstrap but
    executes asynchronously, allowing for non-blocking operation and
    progress reporting during long computations.
    
    Parameters
    ----------
    data : numpy.ndarray
        Time series data array (1-dimensional)
    statistic_func : callable
        Function that computes the statistic of interest from a data array
    block_size : int
        Size of blocks to resample
    num_bootstrap : int
        Number of bootstrap samples to generate
    replace : bool, default=True
        Whether to sample blocks with replacement
    progress_callback : callable, optional
        Function to call with progress updates (0-100)
        
    Returns
    -------
    numpy.ndarray
        Array of bootstrap statistics with shape (num_bootstrap,)
        
    Notes
    -----
    This async function allows for non-blocking execution and progress
    reporting. It's particularly useful for computationally intensive 
    bootstrap operations in interactive environments.
    """
    # Validate inputs
    validate_array(data, param_name="data")
    is_positive_integer(block_size, param_name="block_size")
    is_positive_integer(num_bootstrap, param_name="num_bootstrap")
    
    # Ensure data is 1D array
    if data.ndim != 1:
        raise ValueError("data must be a 1-dimensional array")
    
    # Get data length
    n = len(data)
    
    # Check if block_size is valid
    if block_size > n:
        raise ValueError(f"block_size ({block_size}) cannot be larger than data length ({n})")
    
    # Calculate number of blocks needed to cover the data length
    num_blocks = int(np.ceil(n / block_size))
    
    # Number of possible blocks for sampling
    n_possible_blocks = n - block_size + 1
    
    # Initialize array to store bootstrap statistics
    bootstrap_statistics = np.zeros(num_bootstrap)
    
    # Report initial progress
    if progress_callback:
        progress_callback(0)
    
    # Generate bootstrap samples and compute statistics asynchronously
    for i in range(num_bootstrap):
        # Create bootstrap sample array
        bootstrap_sample = np.zeros(num_blocks * block_size)
        
        # Generate random block starting indices
        if replace:
            # Sample with replacement
            block_indices = np.random.randint(0, n_possible_blocks, size=num_blocks)
        else:
            # Sample without replacement
            if num_blocks > n_possible_blocks:
                raise ValueError(
                    f"Cannot sample {num_blocks} blocks without replacement "
                    f"when only {n_possible_blocks} blocks are available"
                )
            block_indices = np.random.choice(n_possible_blocks, size=num_blocks, replace=False)
        
        # Construct bootstrap sample from blocks
        for j, idx in enumerate(block_indices):
            start_pos = j * block_size
            end_pos = min((j + 1) * block_size, num_blocks * block_size)
            sample_length = end_pos - start_pos
            
            # Copy block data
            bootstrap_sample[start_pos:end_pos] = data[idx:idx + sample_length]
        
        # Trim bootstrap sample to match original length
        bootstrap_sample = bootstrap_sample[:n]
        
        # Compute statistic
        bootstrap_statistics[i] = statistic_func(bootstrap_sample)
        
        # Report progress
        if progress_callback:
            progress_callback((i + 1) / num_bootstrap * 100)
        
        # Yield to event loop occasionally to prevent blocking
        if (i + 1) % 10 == 0:
            await asyncio.sleep(0)
    
    # Report completion
    if progress_callback:
        progress_callback(100)
    
    return bootstrap_statistics


@async_progress()
async def stationary_bootstrap_async(
    data: np.ndarray,
    statistic_func: Callable,
    probability: float,
    num_bootstrap: int,
    progress_callback: Optional[Callable[[float], None]] = None
) -> np.ndarray:
    """
    Asynchronous implementation of stationary bootstrap resampling.
    
    This function provides the same functionality as stationary_bootstrap but
    executes asynchronously, allowing for non-blocking operation and
    progress reporting during long computations.
    
    Parameters
    ----------
    data : numpy.ndarray
        Time series data array (1-dimensional)
    statistic_func : callable
        Function that computes the statistic of interest from a data array
    probability : float
        Probability parameter for the geometric distribution (0 < p ≤ 1)
        which determines the expected block length (1/p)
    num_bootstrap : int
        Number of bootstrap samples to generate
    progress_callback : callable, optional
        Function to call with progress updates (0-100)
        
    Returns
    -------
    numpy.ndarray
        Array of bootstrap statistics with shape (num_bootstrap,)
        
    Notes
    -----
    This async function allows for non-blocking execution and progress
    reporting. It's particularly useful for computationally intensive 
    bootstrap operations in interactive environments.
    """
    # Validate inputs
    validate_array(data, param_name="data")
    is_probability(probability, param_name="probability")
    is_positive_integer(num_bootstrap, param_name="num_bootstrap")
    
    # Ensure data is 1D array
    if data.ndim != 1:
        raise ValueError("data must be a 1-dimensional array")
    
    # Get data length
    n = len(data)
    
    # Issue warning if probability is too small or too large
    if probability < 0.01:
        warnings.warn(
            f"Small probability value ({probability}) will result in very long blocks "
            f"(average length: {1/probability:.1f}), which may not be suitable "
            f"for data of length {n}"
        )
    elif probability > 0.5:
        warnings.warn(
            f"Large probability value ({probability}) will result in very short blocks "
            f"(average length: {1/probability:.1f}), which may not preserve "
            f"the dependence structure adequately"
        )
    
    # Initialize array to store bootstrap statistics
    bootstrap_statistics = np.zeros(num_bootstrap)
    
    # Report initial progress
    if progress_callback:
        progress_callback(0)
    
    # Generate bootstrap samples and compute statistics asynchronously
    for i in range(num_bootstrap):
        # Create bootstrap sample array
        bootstrap_sample = np.zeros(n)
        
        # Generate bootstrap sample
        idx = 0
        while idx < n:
            # Random starting position
            start_pos = np.random.randint(0, n)
            
            # Random block length (geometric distribution)
            block_length = np.random.geometric(probability)
            
            # Copy block data (with circular wrapping if needed)
            for j in range(block_length):
                if idx >= n:
                    break
                pos = (start_pos + j) % n  # Circular wrapping
                bootstrap_sample[idx] = data[pos]
                idx += 1
        
        # Compute statistic
        bootstrap_statistics[i] = statistic_func(bootstrap_sample)
        
        # Report progress
        if progress_callback:
            progress_callback((i + 1) / num_bootstrap * 100)
        
        # Yield to event loop occasionally to prevent blocking
        if (i + 1) % 10 == 0:
            await asyncio.sleep(0)
    
    # Report completion
    if progress_callback:
        progress_callback(100)
    
    return bootstrap_statistics


@async_progress()
async def moving_block_bootstrap_async(
    data: np.ndarray,
    statistic_func: Callable,
    block_size: int,
    num_bootstrap: int,
    progress_callback: Optional[Callable[[float], None]] = None
) -> np.ndarray:
    """
    Asynchronous implementation of moving block bootstrap resampling.
    
    This function provides the same functionality as moving_block_bootstrap but
    executes asynchronously, allowing for non-blocking operation and
    progress reporting during long computations.
    
    Parameters
    ----------
    data : numpy.ndarray
        Time series data array (1-dimensional)
    statistic_func : callable
        Function that computes the statistic of interest from a data array
    block_size : int
        Size of blocks to resample
    num_bootstrap : int
        Number of bootstrap samples to generate
    progress_callback : callable, optional
        Function to call with progress updates (0-100)
        
    Returns
    -------
    numpy.ndarray
        Array of bootstrap statistics with shape (num_bootstrap,)
        
    Notes
    -----
    This async function allows for non-blocking execution and progress
    reporting. It's particularly useful for computationally intensive 
    bootstrap operations in interactive environments.
    """
    # Validate inputs
    validate_array(data, param_name="data")
    is_positive_integer(block_size, param_name="block_size")
    is_positive_integer(num_bootstrap, param_name="num_bootstrap")
    
    # Ensure data is 1D array
    if data.ndim != 1:
        raise ValueError("data must be a 1-dimensional array")
    
    # Get data length
    n = len(data)
    
    # Check if block_size is valid
    if block_size > n:
        raise ValueError(f"block_size ({block_size}) cannot be larger than data length ({n})")
    
    # Calculate number of blocks needed
    num_blocks = int(np.ceil(n / block_size))
    
    # Number of possible blocks (overlapping)
    n_possible_blocks = n - block_size + 1
    
    # Initialize array to store bootstrap statistics
    bootstrap_statistics = np.zeros(num_bootstrap)
    
    # Report initial progress
    if progress_callback:
        progress_callback(0)
    
    # Generate bootstrap samples and compute statistics asynchronously
    for i in range(num_bootstrap):
        # Create bootstrap sample array
        bootstrap_sample = np.zeros(num_blocks * block_size)
        
        # Generate random block starting indices (with replacement)
        block_indices = np.random.randint(0, n_possible_blocks, size=num_blocks)
        
        # Construct bootstrap sample from blocks
        for j, idx in enumerate(block_indices):
            start_pos = j * block_size
            end_pos = min((j + 1) * block_size, num_blocks * block_size)
            sample_length = end_pos - start_pos
            
            # Copy block data
            bootstrap_sample[start_pos:end_pos] = data[idx:idx + sample_length]
        
        # Trim bootstrap sample to match original length
        bootstrap_sample = bootstrap_sample[:n]
        
        # Compute statistic
        bootstrap_statistics[i] = statistic_func(bootstrap_sample)
        
        # Report progress
        if progress_callback:
            progress_callback((i + 1) / num_bootstrap * 100)
        
        # Yield to event loop occasionally to prevent blocking
        if (i + 1) % 10 == 0:
            await asyncio.sleep(0)
    
    # Report completion
    if progress_callback:
        progress_callback(100)
    
    return bootstrap_statistics


def calculate_bootstrap_ci(
    bootstrap_statistics: np.ndarray,
    original_statistic: float,
    method: str = 'percentile',
    alpha: float = 0.05
) -> Tuple[float, float]:
    """
    Calculate bootstrap confidence interval using various methods.
    
    Parameters
    ----------
    bootstrap_statistics : numpy.ndarray
        Array of bootstrap statistics
    original_statistic : float
        Original statistic computed from the data
    method : str, default='percentile'
        Method to calculate confidence interval:
        - 'percentile': Uses percentiles of bootstrap distribution
        - 'bca': Bias-corrected and accelerated bootstrap
    alpha : float, default=0.05
        Significance level (e.g., 0.05 for 95% confidence interval)
        
    Returns
    -------
    tuple
        Lower and upper bounds of the confidence interval (lower, upper)
        
    Notes
    -----
    The 'percentile' method directly uses quantiles of the bootstrap distribution.
    The 'bca' method adjusts for bias and skewness in the bootstrap distribution.
    
    References
    ----------
    Efron, B. and Tibshirani, R. (1993) "An Introduction to the Bootstrap",
    Chapman & Hall/CRC.
    """
    # Validate inputs
    validate_array(bootstrap_statistics, param_name="bootstrap_statistics")
    if not isinstance(original_statistic, (int, float)):
        raise TypeError("original_statistic must be a number")
    is_probability(alpha, param_name="alpha")
    
    # Check method
    method = method.lower()
    if method not in ['percentile', 'bca']:
        raise ValueError("method must be 'percentile' or 'bca'")
    
    # Get number of bootstrap samples
    n_boot = len(bootstrap_statistics)
    
    if method == 'percentile':
        # Percentile method
        lower_percentile = alpha / 2 * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower = np.percentile(bootstrap_statistics, lower_percentile)
        upper = np.percentile(bootstrap_statistics, upper_percentile)
        
    elif method == 'bca':
        # Bias-corrected and accelerated bootstrap
        
        # Calculate bias-correction factor
        z0 = scipy.stats.norm.ppf(np.mean(bootstrap_statistics < original_statistic))
        
        # Calculate acceleration factor (jackknife method not implemented here)
        # This is a simplified approach
        accel = 0.0  # Assuming zero acceleration
        
        # Calculate adjusted percentiles
        alpha1 = alpha / 2
        alpha2 = 1 - alpha / 2
        
        z1 = scipy.stats.norm.ppf(alpha1)
        z2 = scipy.stats.norm.ppf(alpha2)
        
        # Apply BCa transformation
        p1 = scipy.stats.norm.cdf(z0 + (z0 + z1) / (1 - accel * (z0 + z1)))
        p2 = scipy.stats.norm.cdf(z0 + (z0 + z2) / (1 - accel * (z0 + z2)))
        
        # Get percentiles
        lower = np.percentile(bootstrap_statistics, p1 * 100)
        upper = np.percentile(bootstrap_statistics, p2 * 100)
    
    return (lower, upper)


def calculate_bootstrap_pvalue(
    bootstrap_statistics: np.ndarray,
    original_statistic: float,
    null_value: float,
    alternative: str = 'two-sided'
) -> float:
    """
    Calculate p-value for hypothesis test using bootstrap distribution.
    
    Parameters
    ----------
    bootstrap_statistics : numpy.ndarray
        Array of bootstrap statistics
    original_statistic : float
        Original statistic computed from the data
    null_value : float
        Null hypothesis value
    alternative : str, default='two-sided'
        Type of alternative hypothesis:
        - 'two-sided': H1: statistic != null_value
        - 'less': H1: statistic < null_value
        - 'greater': H1: statistic > null_value
        
    Returns
    -------
    float
        p-value for the hypothesis test
        
    Notes
    -----
    The p-value is calculated based on the position of the original
    statistic in the bootstrap distribution, adjusted for the null hypothesis.
    
    References
    ----------
    Davison, A.C. and Hinkley, D.V. (1997) "Bootstrap Methods and Their
    Application", Cambridge University Press.
    """
    # Validate inputs
    validate_array(bootstrap_statistics, param_name="bootstrap_statistics")
    if not isinstance(original_statistic, (int, float)):
        raise TypeError("original_statistic must be a number")
    if not isinstance(null_value, (int, float)):
        raise TypeError("null_value must be a number")
    
    # Check alternative
    alternative = alternative.lower()
    if alternative not in ['two-sided', 'less', 'greater']:
        raise ValueError("alternative must be 'two-sided', 'less', or 'greater'")
    
    # Center bootstrap distribution around null value
    centered_boot = bootstrap_statistics - (original_statistic - null_value)
    
    # Calculate p-value based on alternative
    if alternative == 'two-sided':
        # Two-sided p-value
        p_value = np.mean(np.abs(centered_boot - null_value) >= np.abs(original_statistic - null_value))
    elif alternative == 'less':
        # One-sided p-value (less than)
        p_value = np.mean(centered_boot <= original_statistic)
    elif alternative == 'greater':
        # One-sided p-value (greater than)
        p_value = np.mean(centered_boot >= original_statistic)
    
    return p_value


class BootstrapResult:
    """
    Container class for bootstrap analysis results with methods for
    confidence intervals, hypothesis testing, and visualization.
    
    This class stores bootstrap statistics and provides methods for
    inference and visualization of bootstrap results.
    
    Attributes
    ----------
    bootstrap_statistics : numpy.ndarray
        Array of bootstrap statistics
    original_statistic : float
        Original statistic computed from the data
    
    Methods
    -------
    standard_error()
        Calculate bootstrap standard error
    confidence_interval(method='percentile', alpha=0.05)
        Calculate bootstrap confidence interval
    calculate_pvalue(null_value, alternative='two-sided')
        Calculate p-value for hypothesis test
    plot_distribution(ax=None, show_ci=True, show_original=True)
        Plot bootstrap distribution
    summary()
        Generate summary statistics
    """
    
    def __init__(
        self,
        bootstrap_statistics: np.ndarray,
        original_statistic: float,
        ci_method: Optional[str] = None,
        alpha: Optional[float] = None
    ):
        """
        Initialize bootstrap result object.
        
        Parameters
        ----------
        bootstrap_statistics : numpy.ndarray
            Array of bootstrap statistics
        original_statistic : float
            Original statistic computed from the data
        ci_method : str, optional
            Method to calculate confidence interval ('percentile' or 'bca')
        alpha : float, optional
            Significance level for confidence interval
        """
        # Validate inputs
        validate_array(bootstrap_statistics, param_name="bootstrap_statistics")
        if not isinstance(original_statistic, (int, float)):
            raise TypeError("original_statistic must be a number")
        
        self.bootstrap_statistics = bootstrap_statistics
        self.original_statistic = original_statistic
        self._ci_method = ci_method
        self._alpha = alpha
        self._confidence_interval = None
        
        # Calculate confidence interval if method and alpha are provided
        if ci_method is not None and alpha is not None:
            self._confidence_interval = calculate_bootstrap_ci(
                bootstrap_statistics, original_statistic, ci_method, alpha
            )
    
    def standard_error(self) -> float:
        """
        Calculate bootstrap standard error of the statistic.
        
        Returns
        -------
        float
            Bootstrap standard error
        """
        return np.std(self.bootstrap_statistics, ddof=1)
    
    def confidence_interval(
        self,
        method: Optional[str] = None,
        alpha: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Calculate or return bootstrap confidence interval.
        
        Parameters
        ----------
        method : str, optional
            Method to calculate confidence interval:
            - 'percentile': Uses percentiles of bootstrap distribution
            - 'bca': Bias-corrected and accelerated bootstrap
            If None, uses stored method or defaults to 'percentile'
        alpha : float, optional
            Significance level (e.g., 0.05 for 95% confidence interval)
            If None, uses stored alpha or defaults to 0.05
            
        Returns
        -------
        tuple
            Lower and upper bounds of the confidence interval (lower, upper)
        """
        # Use provided parameters or fall back to stored ones
        method = method or self._ci_method or 'percentile'
        alpha = alpha or self._alpha or 0.05
        
        # Check if we can reuse stored CI
        if (self._confidence_interval is not None and 
            method == self._ci_method and 
            alpha == self._alpha):
            return self._confidence_interval
        
        # Calculate new CI
        ci = calculate_bootstrap_ci(
            self.bootstrap_statistics, self.original_statistic, method, alpha
        )
        
        # Update stored values
        self._ci_method = method
        self._alpha = alpha
        self._confidence_interval = ci
        
        return ci
    
    def calculate_pvalue(
        self,
        null_value: float,
        alternative: str = 'two-sided'
    ) -> float:
        """
        Calculate p-value for hypothesis test using bootstrap distribution.
        
        Parameters
        ----------
        null_value : float
            Null hypothesis value
        alternative : str, default='two-sided'
            Type of alternative hypothesis:
            - 'two-sided': H1: statistic != null_value
            - 'less': H1: statistic < null_value
            - 'greater': H1: statistic > null_value
            
        Returns
        -------
        float
            p-value for the hypothesis test
        """
        return calculate_bootstrap_pvalue(
            self.bootstrap_statistics, self.original_statistic,
            null_value, alternative
        )
    
    def plot_distribution(
        self,
        ax=None,
        show_ci: bool = True,
        show_original: bool = True,
        title: Optional[str] = None
    ):
        """
        Plot the bootstrap distribution with confidence interval and original statistic.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes object to plot on. If None, a new one is created.
        show_ci : bool, default=True
            Whether to show confidence interval lines
        show_original : bool, default=True
            Whether to show original statistic line
        title : str, optional
            Title for the plot
            
        Returns
        -------
        matplotlib.axes.Axes
            Axes object with the plot
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "Matplotlib is required for plotting. "
                "Install it with 'pip install matplotlib'."
            )
        
        # Create axes if not provided
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))
        
        # Plot histogram of bootstrap statistics
        ax.hist(self.bootstrap_statistics, bins=30, alpha=0.7, density=True)
        
        # Show confidence interval if requested
        if show_ci:
            lower, upper = self.confidence_interval()
            ax.axvline(lower, color='r', linestyle='--', alpha=0.8, 
                      label=f"{(1 - self._alpha) * 100:.1f}% CI Lower")
            ax.axvline(upper, color='r', linestyle='--', alpha=0.8,
                      label=f"{(1 - self._alpha) * 100:.1f}% CI Upper")
        
        # Show original statistic if requested
        if show_original:
            ax.axvline(self.original_statistic, color='k', linestyle='-', 
                      label="Original Statistic")
        
        # Add title and labels
        if title:
            ax.set_title(title)
        else:
            ax.set_title("Bootstrap Distribution")
        ax.set_xlabel("Statistic Value")
        ax.set_ylabel("Density")
        
        # Add legend
        ax.legend()
        
        return ax
    
    def summary(self) -> Dict[str, Any]:
        """
        Generate summary of bootstrap analysis results.
        
        Returns
        -------
        dict
            Dictionary with summary statistics including:
            - original_statistic: Original statistic value
            - bootstrap_mean: Mean of bootstrap statistics
            - bootstrap_median: Median of bootstrap statistics
            - standard_error: Bootstrap standard error
            - confidence_interval: Tuple of (lower, upper) CI bounds
            - ci_level: Confidence level (e.g., 0.95)
            - ci_method: Method used for CI calculation
            - num_bootstrap: Number of bootstrap samples
        """
        # Calculate statistics
        mean = np.mean(self.bootstrap_statistics)
        median = np.median(self.bootstrap_statistics)
        std_error = self.standard_error()
        
        # Get confidence interval
        lower, upper = self.confidence_interval()
        
        # Compile summary
        summary_dict = {
            'original_statistic': self.original_statistic,
            'bootstrap_mean': mean,
            'bootstrap_median': median,
            'standard_error': std_error,
            'confidence_interval': (lower, upper),
            'ci_level': 1 - (self._alpha or 0.05),
            'ci_method': self._ci_method or 'percentile',
            'num_bootstrap': len(self.bootstrap_statistics)
        }
        
        return summary_dict


class Bootstrap:
    """
    Main class for bootstrap analysis with support for different bootstrap
    methods and both synchronous and asynchronous execution.
    
    This class provides a unified interface for various bootstrap methods
    and facilitates both synchronous and asynchronous bootstrap analysis.
    
    Attributes
    ----------
    method : str
        Bootstrap method to use ('block', 'stationary', 'moving')
    params : dict
        Parameters for the bootstrap method
    num_bootstrap : int
        Number of bootstrap samples
    ci_method : str
        Method to calculate confidence interval
    alpha : float
        Significance level for confidence interval
    
    Methods
    -------
    run(data, statistic_func)
        Run bootstrap analysis synchronously
    run_async(data, statistic_func, progress_callback=None)
        Run bootstrap analysis asynchronously
    hypothesis_test(data, statistic_func, null_value, alternative)
        Perform hypothesis test
    hypothesis_test_async(data, statistic_func, null_value, alternative, progress_callback=None)
        Perform hypothesis test asynchronously
    """
    
    def __init__(
        self,
        method: str = 'block',
        params: Dict[str, Any] = None,
        num_bootstrap: int = 1000,
        ci_method: str = 'percentile',
        alpha: float = 0.05
    ):
        """
        Initialize bootstrap object with method and parameters.
        
        Parameters
        ----------
        method : str, default='block'
            Bootstrap method to use:
            - 'block': Block bootstrap
            - 'stationary': Stationary bootstrap
            - 'moving': Moving block bootstrap
        params : dict, optional
            Parameters for the bootstrap method:
            - 'block': {'block_size': int, 'replace': bool}
            - 'stationary': {'probability': float}
            - 'moving': {'block_size': int}
        num_bootstrap : int, default=1000
            Number of bootstrap samples
        ci_method : str, default='percentile'
            Method to calculate confidence interval ('percentile' or 'bca')
        alpha : float, default=0.05
            Significance level for confidence interval
        """
        # Validate method
        method = method.lower()
        if method not in ['block', 'stationary', 'moving']:
            raise ValueError("method must be 'block', 'stationary', or 'moving'")
        
        # Set up default parameters if not provided
        if params is None:
            params = {}
        
        # Check and set default parameters based on method
        if method == 'block':
            if 'block_size' not in params:
                params['block_size'] = 20
            if 'replace' not in params:
                params['replace'] = True
                
            # Validate parameters
            is_positive_integer(params['block_size'], param_name="block_size")
            if not isinstance(params['replace'], bool):
                raise TypeError("replace must be a boolean")
                
        elif method == 'stationary':
            if 'probability' not in params:
                params['probability'] = 0.1
                
            # Validate parameters
            is_probability(params['probability'], param_name="probability")
                
        elif method == 'moving':
            if 'block_size' not in params:
                params['block_size'] = 20
                
            # Validate parameters
            is_positive_integer(params['block_size'], param_name="block_size")
        
        # Validate other parameters
        is_positive_integer(num_bootstrap, param_name="num_bootstrap")
        
        ci_method = ci_method.lower()
        if ci_method not in ['percentile', 'bca']:
            raise ValueError("ci_method must be 'percentile' or 'bca'")
        
        is_probability(alpha, param_name="alpha")
        
        # Store parameters
        self.method = method
        self.params = params
        self.num_bootstrap = num_bootstrap
        self.ci_method = ci_method
        self.alpha = alpha
    
    def run(
        self,
        data: np.ndarray,
        statistic_func: Callable
    ) -> BootstrapResult:
        """
        Run bootstrap analysis using the configured method.
        
        Parameters
        ----------
        data : numpy.ndarray
            Time series data array
        statistic_func : callable
            Function that computes the statistic of interest from a data array
            
        Returns
        -------
        BootstrapResult
            Bootstrap result object with statistics and confidence interval
        """
        # Validate inputs
        validate_array(data, param_name="data")
        if not callable(statistic_func):
            raise TypeError("statistic_func must be callable")
        
        # Calculate original statistic
        original_statistic = statistic_func(data)
        
        # Run appropriate bootstrap method
        if self.method == 'block':
            bootstrap_statistics = block_bootstrap(
                data, statistic_func, 
                self.params['block_size'], self.num_bootstrap,
                self.params.get('replace', True)
            )
        elif self.method == 'stationary':
            bootstrap_statistics = stationary_bootstrap(
                data, statistic_func,
                self.params['probability'], self.num_bootstrap
            )
        elif self.method == 'moving':
            bootstrap_statistics = moving_block_bootstrap(
                data, statistic_func,
                self.params['block_size'], self.num_bootstrap
            )
        
        # Create and return result object
        return BootstrapResult(
            bootstrap_statistics, original_statistic,
            self.ci_method, self.alpha
        )
    
    async def run_async(
        self,
        data: np.ndarray,
        statistic_func: Callable,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> BootstrapResult:
        """
        Run bootstrap analysis asynchronously.
        
        Parameters
        ----------
        data : numpy.ndarray
            Time series data array
        statistic_func : callable
            Function that computes the statistic of interest from a data array
        progress_callback : callable, optional
            Function to call with progress updates (0-100)
            
        Returns
        -------
        BootstrapResult
            Bootstrap result object with statistics and confidence interval
        """
        # Validate inputs
        validate_array(data, param_name="data")
        if not callable(statistic_func):
            raise TypeError("statistic_func must be callable")
        
        # Calculate original statistic
        original_statistic = statistic_func(data)
        
        # Run appropriate bootstrap method asynchronously
        if self.method == 'block':
            bootstrap_statistics = await block_bootstrap_async(
                data, statistic_func, 
                self.params['block_size'], self.num_bootstrap,
                self.params.get('replace', True),
                progress_callback
            )
        elif self.method == 'stationary':
            bootstrap_statistics = await stationary_bootstrap_async(
                data, statistic_func,
                self.params['probability'], self.num_bootstrap,
                progress_callback
            )
        elif self.method == 'moving':
            bootstrap_statistics = await moving_block_bootstrap_async(
                data, statistic_func,
                self.params['block_size'], self.num_bootstrap,
                progress_callback
            )
        
        # Create and return result object
        return BootstrapResult(
            bootstrap_statistics, original_statistic,
            self.ci_method, self.alpha
        )
    
    def hypothesis_test(
        self,
        data: np.ndarray,
        statistic_func: Callable,
        null_value: float,
        alternative: str = 'two-sided'
    ) -> Tuple[BootstrapResult, float]:
        """
        Perform hypothesis test using bootstrap.
        
        Parameters
        ----------
        data : numpy.ndarray
            Time series data array
        statistic_func : callable
            Function that computes the statistic of interest from a data array
        null_value : float
            Null hypothesis value
        alternative : str, default='two-sided'
            Type of alternative hypothesis:
            - 'two-sided': H1: statistic != null_value
            - 'less': H1: statistic < null_value
            - 'greater': H1: statistic > null_value
            
        Returns
        -------
        tuple
            BootstrapResult object and p-value for the hypothesis test
        """
        # Run bootstrap analysis
        bootstrap_result = self.run(data, statistic_func)
        
        # Calculate p-value
        p_value = bootstrap_result.calculate_pvalue(null_value, alternative)
        
        return bootstrap_result, p_value
    
    async def hypothesis_test_async(
        self,
        data: np.ndarray,
        statistic_func: Callable,
        null_value: float,
        alternative: str = 'two-sided',
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> Tuple[BootstrapResult, float]:
        """
        Perform hypothesis test using bootstrap asynchronously.
        
        Parameters
        ----------
        data : numpy.ndarray
            Time series data array
        statistic_func : callable
            Function that computes the statistic of interest from a data array
        null_value : float
            Null hypothesis value
        alternative : str, default='two-sided'
            Type of alternative hypothesis:
            - 'two-sided': H1: statistic != null_value
            - 'less': H1: statistic < null_value
            - 'greater': H1: statistic > null_value
        progress_callback : callable, optional
            Function to call with progress updates (0-100)
            
        Returns
        -------
        tuple
            BootstrapResult object and p-value for the hypothesis test
        """
        # Run bootstrap analysis asynchronously
        bootstrap_result = await self.run_async(data, statistic_func, progress_callback)
        
        # Calculate p-value
        p_value = bootstrap_result.calculate_pvalue(null_value, alternative)
        
        return bootstrap_result, p_value


# Define exports
__all__ = [
    'block_bootstrap',
    'stationary_bootstrap',
    'moving_block_bootstrap',
    'block_bootstrap_async',
    'stationary_bootstrap_async',
    'moving_block_bootstrap_async',
    'calculate_bootstrap_ci',
    'calculate_bootstrap_pvalue',
    'Bootstrap',
    'BootstrapResult'
]