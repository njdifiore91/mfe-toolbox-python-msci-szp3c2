"""
MFE Toolbox - Bootstrap Benchmarking Module

This module provides benchmarking tools to measure the performance of bootstrap methods
in the MFE Toolbox, with a focus on comparing Numba-optimized implementations against
their pure Python counterparts across different data sizes and bootstrap configurations.
"""

import numpy as np  # numpy 1.26.3
import time  # standard library
import timeit  # standard library
import matplotlib.pyplot as plt  # matplotlib 3.8.0
import asyncio  # Python 3.12
from typing import Any, Callable, Dict, List, Optional, Tuple, Union  # Python 3.12
import argparse  # standard library
import logging  # standard library
import os  # standard library
import sys  # standard library
from tqdm import tqdm  # tqdm 4.66.1

# Internal imports
from ../../mfe.core.bootstrap import (
    block_bootstrap, stationary_bootstrap, moving_block_bootstrap,
    block_bootstrap_async, stationary_bootstrap_async, moving_block_bootstrap_async,
    Bootstrap, BootstrapResult
)
from ../../mfe.utils.numba_helpers import (
    optimized_jit, fallback_to_python, check_numba_compatibility
)
from ../../mfe.utils.numpy_helpers import ensure_array
from ../benchmark_numba import BenchmarkResult, benchmark_function

# Set up logger
logger = logging.getLogger(__name__)

# Define global constants
DATA_SIZES = [100, 1000, 10000, 50000]
NUM_BOOTSTRAP = 1000
BLOCK_SIZES = [5, 10, 20]
PROBABILITY_VALUES = [0.1, 0.2, 0.5]
REPETITIONS = 5
RESULTS_DIR = 'results/bootstrap'


def setup_benchmark_environment():
    """
    Initializes the benchmark environment and ensures Numba JIT is properly configured.
    """
    # Check if Numba is available
    if not check_numba_compatibility():
        logger.warning("Numba is not available or properly configured.")
        logger.warning("Benchmarks will run in non-optimized mode.")
    
    # Create result directories if they don't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(RESULTS_DIR, "benchmark.log")),
            logging.StreamHandler()
        ]
    )
    
    # Set numpy random seed for reproducibility
    np.random.seed(42)
    
    logger.info("Benchmark environment setup complete.")


def generate_test_data(size: int, add_dependence: bool = True) -> np.ndarray:
    """
    Generates synthetic time series data for bootstrap benchmarking.
    
    Parameters
    ----------
    size : int
        Size of the dataset to generate
    add_dependence : bool, default=True
        Whether to add temporal dependence to the data
        
    Returns
    -------
    numpy.ndarray
        Generated time series data with optional dependence structure
    """
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Generate random data
    data = np.random.randn(size)
    
    # Add temporal dependence if requested (AR(1) process)
    if add_dependence:
        # Create an AR(1) process with strong persistence
        phi = 0.7  # AR parameter
        for i in range(1, size):
            data[i] = phi * data[i-1] + np.random.randn()
    
    return data


def mean_statistic(data: np.ndarray) -> float:
    """
    Simple statistic function that calculates the mean of input data for bootstrap benchmarks.
    
    Parameters
    ----------
    data : numpy.ndarray
        Input data array
        
    Returns
    -------
    float
        Mean of the data
    """
    return np.mean(data)


def non_optimized_block_bootstrap(
    data: np.ndarray,
    statistic_func: Callable,
    block_size: int,
    num_bootstrap: int,
    replace: bool = True
) -> np.ndarray:
    """
    Pure Python implementation of block bootstrap without Numba optimization for performance comparison.
    
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
    """
    # Validate input
    data = ensure_array(data)
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


def non_optimized_stationary_bootstrap(
    data: np.ndarray,
    statistic_func: Callable,
    probability: float,
    num_bootstrap: int
) -> np.ndarray:
    """
    Pure Python implementation of stationary bootstrap without Numba optimization for performance comparison.
    
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
    """
    # Validate input
    data = ensure_array(data)
    if data.ndim != 1:
        raise ValueError("data must be a 1-dimensional array")
    
    # Get data length
    n = len(data)
    
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


def non_optimized_moving_block_bootstrap(
    data: np.ndarray,
    statistic_func: Callable,
    block_size: int,
    num_bootstrap: int
) -> np.ndarray:
    """
    Pure Python implementation of moving block bootstrap without Numba optimization for performance comparison.
    
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
    """
    # Validate input
    data = ensure_array(data)
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


def benchmark_block_bootstrap(
    data_size: int,
    block_size: int,
    num_bootstrap: int,
    repetitions: int
) -> BenchmarkResult:
    """
    Benchmarks the block bootstrap function comparing optimized and non-optimized versions.
    
    Parameters
    ----------
    data_size : int
        Size of the data to benchmark with
    block_size : int
        Size of blocks to use for bootstrapping
    num_bootstrap : int
        Number of bootstrap samples to generate
    repetitions : int
        Number of times to repeat the benchmark
        
    Returns
    -------
    BenchmarkResult
        Benchmark results for block bootstrap
    """
    # Generate test data
    data = generate_test_data(data_size)
    
    # Setup benchmark parameters
    params = {
        'statistic_func': mean_statistic,
        'block_size': block_size,
        'num_bootstrap': num_bootstrap,
        'replace': True
    }
    
    # Benchmark optimized version
    opt_time = benchmark_function(
        block_bootstrap,
        data,
        params,
        repetitions
    )
    
    # Benchmark non-optimized version
    nonopt_time = benchmark_function(
        non_optimized_block_bootstrap,
        data,
        params,
        repetitions
    )
    
    # Create benchmark result
    result = BenchmarkResult("Block Bootstrap")
    result.add_result(data_size, nonopt_time, opt_time)
    
    return result


def benchmark_stationary_bootstrap(
    data_size: int,
    probability: float,
    num_bootstrap: int,
    repetitions: int
) -> BenchmarkResult:
    """
    Benchmarks the stationary bootstrap function comparing optimized and non-optimized versions.
    
    Parameters
    ----------
    data_size : int
        Size of the data to benchmark with
    probability : float
        Probability parameter for bootstrapping
    num_bootstrap : int
        Number of bootstrap samples to generate
    repetitions : int
        Number of times to repeat the benchmark
        
    Returns
    -------
    BenchmarkResult
        Benchmark results for stationary bootstrap
    """
    # Generate test data
    data = generate_test_data(data_size)
    
    # Setup benchmark parameters
    params = {
        'statistic_func': mean_statistic,
        'probability': probability,
        'num_bootstrap': num_bootstrap
    }
    
    # Benchmark optimized version
    opt_time = benchmark_function(
        stationary_bootstrap,
        data,
        params,
        repetitions
    )
    
    # Benchmark non-optimized version
    nonopt_time = benchmark_function(
        non_optimized_stationary_bootstrap,
        data,
        params,
        repetitions
    )
    
    # Create benchmark result
    result = BenchmarkResult("Stationary Bootstrap")
    result.add_result(data_size, nonopt_time, opt_time)
    
    return result


def benchmark_moving_block_bootstrap(
    data_size: int,
    block_size: int,
    num_bootstrap: int,
    repetitions: int
) -> BenchmarkResult:
    """
    Benchmarks the moving block bootstrap function comparing optimized and non-optimized versions.
    
    Parameters
    ----------
    data_size : int
        Size of the data to benchmark with
    block_size : int
        Size of blocks to use for bootstrapping
    num_bootstrap : int
        Number of bootstrap samples to generate
    repetitions : int
        Number of times to repeat the benchmark
        
    Returns
    -------
    BenchmarkResult
        Benchmark results for moving block bootstrap
    """
    # Generate test data
    data = generate_test_data(data_size)
    
    # Setup benchmark parameters
    params = {
        'statistic_func': mean_statistic,
        'block_size': block_size,
        'num_bootstrap': num_bootstrap
    }
    
    # Benchmark optimized version
    opt_time = benchmark_function(
        moving_block_bootstrap,
        data,
        params,
        repetitions
    )
    
    # Benchmark non-optimized version
    nonopt_time = benchmark_function(
        non_optimized_moving_block_bootstrap,
        data,
        params,
        repetitions
    )
    
    # Create benchmark result
    result = BenchmarkResult("Moving Block Bootstrap")
    result.add_result(data_size, nonopt_time, opt_time)
    
    return result


async def benchmark_async_bootstrap(
    data_size: int,
    block_size: int,
    probability: float,
    num_bootstrap: int,
    repetitions: int
) -> Dict:
    """
    Benchmarks the asynchronous bootstrap implementations.
    
    Parameters
    ----------
    data_size : int
        Size of the data to benchmark with
    block_size : int
        Size of blocks to use for bootstrapping
    probability : float
        Probability parameter for stationary bootstrap
    num_bootstrap : int
        Number of bootstrap samples to generate
    repetitions : int
        Number of times to repeat the benchmark
        
    Returns
    -------
    dict
        Dictionary of benchmark results for async bootstrap functions
    """
    # Generate test data
    data = generate_test_data(data_size)
    
    # Define benchmark wrapper for async functions
    async def benchmark_async_func(
        func,
        data,
        params,
        repetitions
    ):
        start_time = time.time()
        for _ in range(repetitions):
            await func(
                data,
                params['statistic_func'],
                *params['args'],
                num_bootstrap=params['num_bootstrap']
            )
        elapsed = (time.time() - start_time) / repetitions
        return elapsed
    
    # Define parameters for each bootstrap method
    block_params = {
        'statistic_func': mean_statistic,
        'args': [block_size, True],
        'num_bootstrap': num_bootstrap
    }
    
    stationary_params = {
        'statistic_func': mean_statistic,
        'args': [probability],
        'num_bootstrap': num_bootstrap
    }
    
    moving_params = {
        'statistic_func': mean_statistic,
        'args': [block_size],
        'num_bootstrap': num_bootstrap
    }
    
    # Benchmark async functions
    block_time = await benchmark_async_func(
        block_bootstrap_async,
        data,
        block_params,
        repetitions
    )
    
    stationary_time = await benchmark_async_func(
        stationary_bootstrap_async,
        data,
        stationary_params,
        repetitions
    )
    
    moving_time = await benchmark_async_func(
        moving_block_bootstrap_async,
        data,
        moving_params,
        repetitions
    )
    
    # Return results
    return {
        'block_bootstrap_async': block_time,
        'stationary_bootstrap_async': stationary_time,
        'moving_block_bootstrap_async': moving_time,
        'data_size': data_size
    }


def benchmark_bootstrap_class(
    data_size: int,
    block_size: int,
    probability: float,
    num_bootstrap: int,
    repetitions: int
) -> Dict:
    """
    Benchmarks the Bootstrap class with different methods.
    
    Parameters
    ----------
    data_size : int
        Size of the data to benchmark with
    block_size : int
        Size of blocks to use for bootstrapping
    probability : float
        Probability parameter for stationary bootstrap
    num_bootstrap : int
        Number of bootstrap samples to generate
    repetitions : int
        Number of times to repeat the benchmark
        
    Returns
    -------
    dict
        Dictionary of benchmark results for Bootstrap class methods
    """
    # Generate test data
    data = generate_test_data(data_size)
    
    # Create Bootstrap instances for each method
    block_bootstrap_obj = Bootstrap(
        'block',
        {'block_size': block_size, 'replace': True},
        num_bootstrap
    )
    
    stationary_bootstrap_obj = Bootstrap(
        'stationary',
        {'probability': probability},
        num_bootstrap
    )
    
    moving_bootstrap_obj = Bootstrap(
        'moving',
        {'block_size': block_size},
        num_bootstrap
    )
    
    # Benchmark each method
    block_time = timeit.timeit(
        lambda: block_bootstrap_obj.run(data, mean_statistic),
        number=repetitions
    ) / repetitions
    
    stationary_time = timeit.timeit(
        lambda: stationary_bootstrap_obj.run(data, mean_statistic),
        number=repetitions
    ) / repetitions
    
    moving_time = timeit.timeit(
        lambda: moving_bootstrap_obj.run(data, mean_statistic),
        number=repetitions
    ) / repetitions
    
    # Return results
    return {
        'block_bootstrap_class': block_time,
        'stationary_bootstrap_class': stationary_time,
        'moving_bootstrap_class': moving_time,
        'data_size': data_size
    }


async def benchmark_bootstrap_class_async(
    data_size: int,
    block_size: int,
    probability: float,
    num_bootstrap: int,
    repetitions: int
) -> Dict:
    """
    Benchmarks the async methods of the Bootstrap class.
    
    Parameters
    ----------
    data_size : int
        Size of the data to benchmark with
    block_size : int
        Size of blocks to use for bootstrapping
    probability : float
        Probability parameter for stationary bootstrap
    num_bootstrap : int
        Number of bootstrap samples to generate
    repetitions : int
        Number of times to repeat the benchmark
        
    Returns
    -------
    dict
        Dictionary of benchmark results for Bootstrap class async methods
    """
    # Generate test data
    data = generate_test_data(data_size)
    
    # Create Bootstrap instances for each method
    block_bootstrap_obj = Bootstrap(
        'block',
        {'block_size': block_size, 'replace': True},
        num_bootstrap
    )
    
    stationary_bootstrap_obj = Bootstrap(
        'stationary',
        {'probability': probability},
        num_bootstrap
    )
    
    moving_bootstrap_obj = Bootstrap(
        'moving',
        {'block_size': block_size},
        num_bootstrap
    )
    
    # Define benchmark wrapper for async methods
    async def benchmark_async_method(bootstrap_obj, repetitions):
        start_time = time.time()
        for _ in range(repetitions):
            await bootstrap_obj.run_async(data, mean_statistic)
        elapsed = (time.time() - start_time) / repetitions
        return elapsed
    
    # Benchmark each method asynchronously
    block_time = await benchmark_async_method(block_bootstrap_obj, repetitions)
    stationary_time = await benchmark_async_method(stationary_bootstrap_obj, repetitions)
    moving_time = await benchmark_async_method(moving_bootstrap_obj, repetitions)
    
    # Return results
    return {
        'block_bootstrap_class_async': block_time,
        'stationary_bootstrap_class_async': stationary_time,
        'moving_bootstrap_class_async': moving_time,
        'data_size': data_size
    }


def plot_benchmark_results(
    results: Dict,
    method: str,
    filename: str
):
    """
    Creates visualizations of bootstrap benchmark results.
    
    Parameters
    ----------
    results : dict
        Dictionary of benchmark results
    method : str
        Name of the method being plotted
    filename : str
        Filename to save the plot as
    """
    # Create figure
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Extract data from results
    data_sizes = []
    python_times = []
    numba_times = []
    speedups = []
    
    for key, result in results.items():
        if isinstance(key, str) and key.startswith("data_size_"):
            data_sizes.append(result['data_size'])
            python_times.append(result['python_time'])
            numba_times.append(result['numba_time'])
            speedups.append(result['speedup'])
    
    # Sort by data size
    data_sizes, python_times, numba_times, speedups = zip(*sorted(
        zip(data_sizes, python_times, numba_times, speedups)
    ))
    
    # Plot execution times
    ax1.plot(data_sizes, python_times, 'o-', color='blue', label='Pure Python')
    ax1.plot(data_sizes, numba_times, 'o-', color='red', label='Numba Optimized')
    ax1.set_xlabel('Data Size')
    ax1.set_ylabel('Execution Time (seconds)')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    # Create second y-axis for speedup
    ax2 = ax1.twinx()
    ax2.plot(data_sizes, speedups, 'o--', color='green', label='Speedup Factor')
    ax2.set_ylabel('Speedup Factor (x)')
    
    # Create combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
    
    # Set title
    plt.title(f'Bootstrap Performance: {method}')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename))
    plt.close()


def run_all_bootstrap_benchmarks(
    data_sizes: List[int],
    block_sizes: List[int],
    probabilities: List[float],
    num_bootstrap: int,
    repetitions: int,
    run_async: bool = True,
    verbose: bool = True
) -> Dict:
    """
    Runs all bootstrap benchmarks and collects results.
    
    Parameters
    ----------
    data_sizes : list
        List of data sizes to benchmark
    block_sizes : list
        List of block sizes to benchmark
    probabilities : list
        List of probability values to benchmark
    num_bootstrap : int
        Number of bootstrap samples to generate
    repetitions : int
        Number of times to repeat each benchmark
    run_async : bool, default=True
        Whether to run async benchmarks
    verbose : bool, default=True
        Whether to print verbose output
        
    Returns
    -------
    dict
        Comprehensive dictionary of all benchmark results
    """
    # Initialize benchmark environment
    setup_benchmark_environment()
    
    # Initialize results dictionary
    results = {}
    
    # Run block bootstrap benchmarks
    block_results = {}
    for data_size in tqdm(data_sizes, desc="Block Bootstrap"):
        for block_size in block_sizes:
            result = benchmark_block_bootstrap(
                data_size, block_size, num_bootstrap, repetitions
            )
            key = f"data_size_{data_size}_block_{block_size}"
            block_results[key] = {
                'data_size': data_size,
                'block_size': block_size,
                'python_time': result.python_times[data_size],
                'numba_time': result.numba_times[data_size],
                'speedup': result.speedup_factors[data_size]
            }
            
            if verbose:
                logger.info(f"Block Bootstrap (size={data_size}, block={block_size}): "
                           f"Python: {result.python_times[data_size]:.6f}s, "
                           f"Numba: {result.numba_times[data_size]:.6f}s, "
                           f"Speedup: {result.speedup_factors[data_size]:.2f}x")
    
    results['block_bootstrap'] = block_results
    
    # Run stationary bootstrap benchmarks
    stationary_results = {}
    for data_size in tqdm(data_sizes, desc="Stationary Bootstrap"):
        for probability in probabilities:
            result = benchmark_stationary_bootstrap(
                data_size, probability, num_bootstrap, repetitions
            )
            key = f"data_size_{data_size}_prob_{probability}"
            stationary_results[key] = {
                'data_size': data_size,
                'probability': probability,
                'python_time': result.python_times[data_size],
                'numba_time': result.numba_times[data_size],
                'speedup': result.speedup_factors[data_size]
            }
            
            if verbose:
                logger.info(f"Stationary Bootstrap (size={data_size}, prob={probability}): "
                           f"Python: {result.python_times[data_size]:.6f}s, "
                           f"Numba: {result.numba_times[data_size]:.6f}s, "
                           f"Speedup: {result.speedup_factors[data_size]:.2f}x")
    
    results['stationary_bootstrap'] = stationary_results
    
    # Run moving block bootstrap benchmarks
    moving_results = {}
    for data_size in tqdm(data_sizes, desc="Moving Block Bootstrap"):
        for block_size in block_sizes:
            result = benchmark_moving_block_bootstrap(
                data_size, block_size, num_bootstrap, repetitions
            )
            key = f"data_size_{data_size}_block_{block_size}"
            moving_results[key] = {
                'data_size': data_size,
                'block_size': block_size,
                'python_time': result.python_times[data_size],
                'numba_time': result.numba_times[data_size],
                'speedup': result.speedup_factors[data_size]
            }
            
            if verbose:
                logger.info(f"Moving Block Bootstrap (size={data_size}, block={block_size}): "
                           f"Python: {result.python_times[data_size]:.6f}s, "
                           f"Numba: {result.numba_times[data_size]:.6f}s, "
                           f"Speedup: {result.speedup_factors[data_size]:.2f}x")
    
    results['moving_block_bootstrap'] = moving_results
    
    # Run async benchmarks if requested
    if run_async:
        async_results = {}
        bootstrap_class_async_results = {}
        
        # Run in asyncio event loop
        async def run_async_benchmarks():
            for data_size in tqdm(data_sizes, desc="Async Benchmarks"):
                # Use first block size and probability for simplicity
                block_size = block_sizes[0]
                probability = probabilities[0]
                
                # Run async function benchmarks
                async_result = await benchmark_async_bootstrap(
                    data_size, block_size, probability, num_bootstrap, repetitions
                )
                async_results[f"data_size_{data_size}"] = async_result
                
                # Run async class benchmarks
                class_async_result = await benchmark_bootstrap_class_async(
                    data_size, block_size, probability, num_bootstrap, repetitions
                )
                bootstrap_class_async_results[f"data_size_{data_size}"] = class_async_result
        
        # Run async benchmarks
        asyncio.run(run_async_benchmarks())
        
        # Store async results
        results['async_bootstrap'] = async_results
        results['bootstrap_class_async'] = bootstrap_class_async_results
    
    # Run bootstrap class benchmarks
    bootstrap_class_results = {}
    for data_size in tqdm(data_sizes, desc="Bootstrap Class"):
        # Use first block size and probability for simplicity
        block_size = block_sizes[0]
        probability = probabilities[0]
        
        class_result = benchmark_bootstrap_class(
            data_size, block_size, probability, num_bootstrap, repetitions
        )
        bootstrap_class_results[f"data_size_{data_size}"] = class_result
    
    results['bootstrap_class'] = bootstrap_class_results
    
    # Print summary if verbose
    if verbose:
        logger.info("Benchmark Summary:")
        logger.info(format_benchmark_results(results))
    
    # Create visualization plots
    plot_benchmark_results(
        results['block_bootstrap'],
        "Block Bootstrap",
        "block_bootstrap_benchmarks.png"
    )
    
    plot_benchmark_results(
        results['stationary_bootstrap'],
        "Stationary Bootstrap",
        "stationary_bootstrap_benchmarks.png"
    )
    
    plot_benchmark_results(
        results['moving_block_bootstrap'],
        "Moving Block Bootstrap",
        "moving_block_bootstrap_benchmarks.png"
    )
    
    return results


def format_benchmark_results(results: Dict) -> str:
    """
    Formats benchmark results into readable string representation.
    
    Parameters
    ----------
    results : dict
        Dictionary of benchmark results
        
    Returns
    -------
    str
        Formatted string with benchmark results in tabular format
    """
    output = "\n" + "=" * 80 + "\n"
    output += "BOOTSTRAP BENCHMARK RESULTS\n"
    output += "=" * 80 + "\n\n"
    
    # Format block bootstrap results
    output += "BLOCK BOOTSTRAP\n"
    output += "-" * 80 + "\n"
    output += f"{'Data Size':<10} {'Block Size':<10} {'Python (s)':<12} {'Numba (s)':<12} {'Speedup':<10}\n"
    output += "-" * 80 + "\n"
    
    for key, result in sorted(results['block_bootstrap'].items()):
        output += f"{result['data_size']:<10} {result['block_size']:<10} {result['python_time']:<12.6f} {result['numba_time']:<12.6f} {result['speedup']:<10.2f}x\n"
    
    # Calculate average speedup
    avg_speedup = sum(r['speedup'] for r in results['block_bootstrap'].values()) / len(results['block_bootstrap'])
    output += "-" * 80 + "\n"
    output += f"Average Speedup: {avg_speedup:.2f}x\n\n"
    
    # Format stationary bootstrap results
    output += "STATIONARY BOOTSTRAP\n"
    output += "-" * 80 + "\n"
    output += f"{'Data Size':<10} {'Probability':<10} {'Python (s)':<12} {'Numba (s)':<12} {'Speedup':<10}\n"
    output += "-" * 80 + "\n"
    
    for key, result in sorted(results['stationary_bootstrap'].items()):
        output += f"{result['data_size']:<10} {result['probability']:<10.2f} {result['python_time']:<12.6f} {result['numba_time']:<12.6f} {result['speedup']:<10.2f}x\n"
    
    # Calculate average speedup
    avg_speedup = sum(r['speedup'] for r in results['stationary_bootstrap'].values()) / len(results['stationary_bootstrap'])
    output += "-" * 80 + "\n"
    output += f"Average Speedup: {avg_speedup:.2f}x\n\n"
    
    # Format moving block bootstrap results
    output += "MOVING BLOCK BOOTSTRAP\n"
    output += "-" * 80 + "\n"
    output += f"{'Data Size':<10} {'Block Size':<10} {'Python (s)':<12} {'Numba (s)':<12} {'Speedup':<10}\n"
    output += "-" * 80 + "\n"
    
    for key, result in sorted(results['moving_block_bootstrap'].items()):
        output += f"{result['data_size']:<10} {result['block_size']:<10} {result['python_time']:<12.6f} {result['numba_time']:<12.6f} {result['speedup']:<10.2f}x\n"
    
    # Calculate average speedup
    avg_speedup = sum(r['speedup'] for r in results['moving_block_bootstrap'].values()) / len(results['moving_block_bootstrap'])
    output += "-" * 80 + "\n"
    output += f"Average Speedup: {avg_speedup:.2f}x\n\n"
    
    # Include async benchmark results if available
    if 'async_bootstrap' in results:
        output += "ASYNC BOOTSTRAP IMPLEMENTATIONS\n"
        output += "-" * 80 + "\n"
        output += f"{'Data Size':<10} {'Block':<12} {'Stationary':<12} {'Moving':<12}\n"
        output += "-" * 80 + "\n"
        
        for key, result in sorted(results['async_bootstrap'].items()):
            data_size = result['data_size']
            block_time = result['block_bootstrap_async']
            stationary_time = result['stationary_bootstrap_async']
            moving_time = result['moving_block_bootstrap_async']
            
            output += f"{data_size:<10} {block_time:<12.6f} {stationary_time:<12.6f} {moving_time:<12.6f}\n"
    
    # Include overall summary
    output += "\n" + "=" * 80 + "\n"
    output += "OVERALL SUMMARY\n"
    output += "=" * 80 + "\n"
    
    # Calculate aggregate statistics
    all_speedups = []
    all_speedups.extend([r['speedup'] for r in results['block_bootstrap'].values()])
    all_speedups.extend([r['speedup'] for r in results['stationary_bootstrap'].values()])
    all_speedups.extend([r['speedup'] for r in results['moving_block_bootstrap'].values()])
    
    overall_avg = sum(all_speedups) / len(all_speedups)
    overall_min = min(all_speedups)
    overall_max = max(all_speedups)
    
    output += f"Overall Average Speedup: {overall_avg:.2f}x\n"
    output += f"Minimum Speedup: {overall_min:.2f}x\n"
    output += f"Maximum Speedup: {overall_max:.2f}x\n"
    
    return output


def main():
    """
    Main function serving as entry point for bootstrap benchmarks.
    
    Returns
    -------
    int
        Exit code (0 for success)
    """
    # Set up command line argument parser
    parser = argparse.ArgumentParser(
        description='MFE Toolbox Bootstrap Benchmarking Tool'
    )
    
    # Add command line arguments
    parser.add_argument(
        '--data-sizes',
        type=int,
        nargs='+',
        default=DATA_SIZES,
        help='Data sizes to benchmark (default: %(default)s)'
    )
    
    parser.add_argument(
        '--block-sizes',
        type=int,
        nargs='+',
        default=BLOCK_SIZES,
        help='Block sizes for block bootstrap (default: %(default)s)'
    )
    
    parser.add_argument(
        '--probs',
        type=float,
        nargs='+',
        default=PROBABILITY_VALUES,
        help='Probability values for stationary bootstrap (default: %(default)s)'
    )
    
    parser.add_argument(
        '--num-bootstrap',
        type=int,
        default=NUM_BOOTSTRAP,
        help='Number of bootstrap samples (default: %(default)s)'
    )
    
    parser.add_argument(
        '--repetitions',
        type=int,
        default=REPETITIONS,
        help='Number of benchmark repetitions (default: %(default)s)'
    )
    
    parser.add_argument(
        '--no-async',
        action='store_true',
        help='Disable async benchmarks'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )
    
    # Parse command line arguments
    args = parser.parse_args()
    
    # Run all benchmarks
    results = run_all_bootstrap_benchmarks(
        data_sizes=args.data_sizes,
        block_sizes=args.block_sizes,
        probabilities=args.probs,
        num_bootstrap=args.num_bootstrap,
        repetitions=args.repetitions,
        run_async=not args.no_async,
        verbose=not args.quiet
    )
    
    # Print formatted results if not quiet
    if not args.quiet:
        print(format_benchmark_results(results))
    
    return 0


if __name__ == "__main__":
    sys.exit(main())