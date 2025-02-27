"""
MFE Toolbox - Realized Volatility Benchmarking Suite

This module provides a benchmarking suite to measure the performance of
realized volatility estimation and high-frequency financial data analysis functions.
It compares execution time across different data sizes, sampling schemes, and
computation methods with a focus on Numba optimization efficiency.
"""

import os  # standard library
import sys  # standard library
import time  # standard library
import timeit  # standard library
from typing import Any, Callable, Dict, List, Optional, Tuple, Union  # Python 3.12
import argparse  # standard library

import numpy as np  # numpy 1.26.3
import numba  # numba 0.59.0
import pandas as pd  # pandas 2.1.4
import matplotlib.pyplot as plt  # matplotlib 3.8.0
from tqdm import tqdm  # tqdm 4.66.1

# Internal imports
from ..mfe.models.realized import realized_variance  # The main realized variance function to benchmark
from ..mfe.models.realized import realized_kernel  # The kernel-based realized volatility function to benchmark
from ..mfe.models.realized import realized_covariance  # The realized covariance function to benchmark
from ..mfe.models.realized import sampling_scheme  # The sampling scheme function to benchmark
from ..mfe.models.realized import noise_filter  # The noise filtering function to benchmark
from ..mfe.utils.numba_helpers import ensure_jit_enabled  # Utility to ensure JIT compilation is properly enabled for benchmarking
from .benchmark_numba import BenchmarkResult  # Reuse benchmark result class for consistent result handling

# Global constants
SIZES = [100, 1000, 10000, 100000]
REPETITIONS = 10
RESULTS_DIR = 'results'
SAMPLING_TYPES = ['CalendarTime', 'BusinessTime', 'CalendarUniform', 'BusinessUniform', 'Fixed']
KERNEL_TYPES = ['Bartlett', 'Parzen', 'QS', 'Truncated', 'Tukey-Hanning']


def setup_benchmark_environment() -> None:
    """
    Initializes the benchmark environment and ensures Numba JIT is properly configured
    """
    # Ensure results directory exists
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    # Check if Numba is available using ensure_jit_enabled
    try:
        ensure_jit_enabled()
        print("Numba JIT compilation is enabled.")
    except AttributeError:
        print("Warning: ensure_jit_enabled function not found. Continuing anyway.")

    # Set up matplotlib configuration for visualizations
    plt.style.use('ggplot')

    # Configure Numba compilation options if needed
    print("Benchmark environment setup complete.")
    print(f"Data sizes to benchmark: {SIZES}")
    print(f"Repetitions per benchmark: {REPETITIONS}")
    print("\n")


def generate_price_data(size: int, volatility: float, include_jumps: bool) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates synthetic price data for benchmarking realized volatility functions

    Args:
        size (int): size
        volatility (float): volatility
        include_jumps (bool): include_jumps

    Returns:
        tuple[np.ndarray, np.ndarray]: Generated prices and timestamps
    """
    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate random log-returns based on specified volatility
    log_returns = np.random.normal(0, volatility / np.sqrt(252), size)

    # Add jumps if include_jumps is True
    if include_jumps:
        jump_indices = np.random.choice(size, int(size * 0.01), replace=False)
        log_returns[jump_indices] += np.random.normal(0, volatility * 5, len(jump_indices))

    # Convert returns to price levels
    prices = np.exp(np.cumsum(log_returns))

    # Generate timestamps at regular intervals
    timestamps = pd.date_range(start='2024-01-01', periods=size, freq='1min').values.astype('datetime64[s]').astype(np.int64)

    # Return tuple of prices and timestamps
    return prices, timestamps


def python_realized_variance(prices: np.ndarray, times: np.ndarray, sampling_type: str, sampling_interval: int or tuple) -> float:
    """
    Pure Python implementation of realized variance for benchmarking against Numba-optimized version

    Args:
        prices (np.ndarray): prices
        times (np.ndarray): times
        sampling_type (str): sampling_type
        sampling_interval (int or tuple): sampling_interval

    Returns:
        float: Realized variance
    """
    # Sample prices according to sampling_type
    sampled_prices, _ = sampling_scheme(prices, times, 'seconds', sampling_type, sampling_interval)

    # Calculate returns from sampled prices
    returns = np.diff(np.log(sampled_prices))

    # Compute sum of squared returns
    realized_variance_value = np.sum(returns ** 2)

    # Return realized variance
    return realized_variance_value


def python_realized_kernel(prices: np.ndarray, times: np.ndarray, kernel_type: str, bandwidth: float) -> float:
    """
    Pure Python implementation of kernel-based realized variance for benchmarking

    Args:
        prices (np.ndarray): prices
        times (np.ndarray): times
        kernel_type (str): kernel_type
        bandwidth (float): bandwidth

    Returns:
        float: Kernel-based realized variance
    """
    # Calculate returns from prices
    returns = np.diff(np.log(prices))
    n = len(returns)

    # Compute autocovariance matrices
    max_lag = int(bandwidth)
    gamma0 = np.sum(returns**2)

    # Apply kernel weights based on kernel_type
    kernel_weights = np.zeros(max_lag + 1)
    kernel_weights[0] = 1.0

    if kernel_type == 'bartlett':
        for h in range(1, max_lag + 1):
            kernel_weights[h] = 1.0 - (h / max_lag)
    elif kernel_type == 'parzen':
        for h in range(1, max_lag + 1):
            x = h / max_lag
            if x <= 0.5:
                kernel_weights[h] = 1 - 6 * x**2 + 6 * x**3
            else:
                kernel_weights[h] = 2 * (1 - x)**3
    else:
        raise ValueError(f"Unsupported kernel type: {kernel_type}")

    # Sum weighted autocovariances
    rk = gamma0
    for h in range(1, max_lag + 1):
        if h < n:
            gamma_h = np.sum(returns[:-h] * returns[h:])
            rk += 2 * kernel_weights[h] * gamma_h

    # Return kernel-based realized variance
    return rk


def benchmark_function(func: Callable, args: list, kwargs: dict, repetitions: int) -> float:
    """
    Benchmarks a function by running it multiple times and measuring execution time

    Args:
        func (callable): func
        args (list): args
        kwargs (dict): kwargs
        repetitions (int): repetitions

    Returns:
        float: Average execution time in seconds
    """
    # Initialize timing variables
    total_time = 0

    # Execute warm-up run to account for JIT compilation
    func(*args, **kwargs)

    # Run the function repetitions times
    for _ in range(repetitions):
        start_time = time.perf_counter()
        func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time += (end_time - start_time)

    # Calculate average execution time
    average_time = total_time / repetitions

    # Return average execution time
    return average_time


def benchmark_realized_variance() -> dict:
    """
    Benchmarks the realized variance function with different data sizes and sampling schemes

    Returns:
        dict: Benchmark results for realized variance function
    """
    # Create a BenchmarkResult instance for realized variance
    benchmark_result = BenchmarkResult("Realized Variance")

    # For each data size in SIZES, generate price data
    for size in SIZES:
        prices, times = generate_price_data(size, volatility=0.01, include_jumps=False)

        # For each sampling type in SAMPLING_TYPES, benchmark realized_variance
        for sampling_type in SAMPLING_TYPES:
            # Benchmark realized_variance
            numba_time = benchmark_function(
                realized_variance,
                args=[prices, times, 'seconds'],
                kwargs={'sampling_type': sampling_type, 'sampling_interval': 60},
                repetitions=REPETITIONS
            )

            # Benchmark python_realized_variance for comparison
            python_time = benchmark_function(
                python_realized_variance,
                args=[prices, times, sampling_type, 60],
                kwargs={},
                repetitions=REPETITIONS
            )

            # Calculate speedup from Numba optimization
            benchmark_result.add_result(
                size=size,
                python_time=python_time,
                numba_time=numba_time
            )

    # Plot and save performance comparison
    plot_results(
        results=benchmark_result,
        title="Realized Variance: Numba vs Python",
        filename="realized_variance_benchmark.png"
    )

    # Return benchmark results
    return benchmark_result.to_dict()


def benchmark_realized_kernel() -> dict:
    """
    Benchmarks the kernel-based realized variance function with different kernel types

    Returns:
        dict: Benchmark results for kernel-based realized variance
    """
    # Create a BenchmarkResult instance for realized kernel
    benchmark_result = BenchmarkResult("Realized Kernel")

    # For each data size in SIZES, generate price data
    for size in SIZES:
        prices, times = generate_price_data(size, volatility=0.01, include_jumps=False)

        # For each kernel type in KERNEL_TYPES, benchmark realized_kernel
        for kernel_type in KERNEL_TYPES:
            # Benchmark realized_kernel
            numba_time = benchmark_function(
                realized_kernel,
                args=[prices, times, 'seconds'],
                kwargs={'kernel_type': kernel_type, 'bandwidth': 30},
                repetitions=REPETITIONS
            )

            # Benchmark python_realized_kernel for comparison
            python_time = benchmark_function(
                python_realized_kernel,
                args=[prices, times, kernel_type, 30],
                kwargs={},
                repetitions=REPETITIONS
            )

            # Calculate speedup from Numba optimization
            benchmark_result.add_result(
                size=size,
                python_time=python_time,
                numba_time=numba_time
            )

    # Plot and save performance comparison
    plot_results(
        results=benchmark_result,
        title="Realized Kernel: Numba vs Python",
        filename="realized_kernel_benchmark.png"
    )

    # Return benchmark results
    return benchmark_result.to_dict()


def benchmark_sampling_schemes() -> dict:
    """
    Benchmarks the sampling scheme function with different sampling types

    Returns:
        dict: Benchmark results for sampling scheme function
    """
    # Create a BenchmarkResult instance for sampling schemes
    benchmark_result = BenchmarkResult("Sampling Schemes")

    # For each data size in SIZES, generate price data
    for size in SIZES:
        prices, times = generate_price_data(size, volatility=0.01, include_jumps=False)

        # For each sampling type in SAMPLING_TYPES, benchmark sampling_scheme
        for sampling_type in SAMPLING_TYPES:
            # Benchmark sampling_scheme
            numba_time = benchmark_function(
                sampling_scheme,
                args=[prices, times, 'seconds', sampling_type, 60],
                kwargs={},
                repetitions=REPETITIONS
            )

            # Implement and benchmark Python sampling scheme for comparison
            def python_sampling_scheme(prices, times, sampling_type, sampling_interval):
                # This is a placeholder for a pure Python implementation of the sampling scheme
                # In a real implementation, this would contain the same logic as the numba-optimized version
                return prices[::10], times[::10]

            python_time = benchmark_function(
                python_sampling_scheme,
                args=[prices, times, sampling_type, 60],
                kwargs={},
                repetitions=REPETITIONS
            )

            # Calculate speedup from Numba optimization
            benchmark_result.add_result(
                size=size,
                python_time=python_time,
                numba_time=numba_time
            )

    # Plot and save performance comparison
    plot_results(
        results=benchmark_result,
        title="Sampling Schemes: Numba vs Python",
        filename="sampling_schemes_benchmark.png"
    )

    # Return benchmark results
    return benchmark_result.to_dict()


def benchmark_noise_filtering() -> dict:
    """
    Benchmarks the noise filtering function with different filter types

    Returns:
        dict: Benchmark results for noise filtering function
    """
    # Create a BenchmarkResult instance for noise filtering
    benchmark_result = BenchmarkResult("Noise Filtering")

    # For each data size in SIZES, generate price data with microstructure noise
    for size in SIZES:
        prices, times = generate_price_data(size, volatility=0.01, include_jumps=False)

        # For different filter types, benchmark noise_filter
        filter_type = 'MA'  # Example filter type
        # Benchmark noise_filter
        numba_time = benchmark_function(
            noise_filter,
            args=[prices, prices, filter_type],
            kwargs={'filter_params': {'window': 5}},
            repetitions=REPETITIONS
        )

        # Implement and benchmark Python noise filtering for comparison
        def python_noise_filter(prices, returns, filter_type, filter_params):
            # This is a placeholder for a pure Python implementation of the noise filter
            # In a real implementation, this would contain the same logic as the numba-optimized version
            return returns

        python_time = benchmark_function(
            python_noise_filter,
            args=[prices, prices, filter_type],
            kwargs={'filter_params': {'window': 5}},
            repetitions=REPETITIONS
        )

        # Calculate speedup from Numba optimization
        benchmark_result.add_result(
            size=size,
            python_time=python_time,
            numba_time=numba_time
        )

    # Plot and save performance comparison
    plot_results(
        results=benchmark_result,
        title="Noise Filtering: Numba vs Python",
        filename="noise_filtering_benchmark.png"
    )

    # Return benchmark results
    return benchmark_result.to_dict()


def plot_results(results: dict, title: str, filename: str) -> None:
    """
    Creates visualizations of benchmark results

    Args:
        results (dict): results
        title (str): title
        filename (str): filename
    """
    # Create a new matplotlib figure
    plt.figure(figsize=(10, 6))

    # Plot execution times for both implementations across different data sizes
    sizes = list(results['python_times'].keys())
    plt.plot(sizes, list(results['python_times'].values()), marker='o', label='Python')
    plt.plot(sizes, list(results['numba_times'].values()), marker='o', label='Numba')

    # Plot speedup factors on a secondary axis
    speedup_factors = list(results['speedup_factors'].values())
    plt.xlabel('Data Size')
    plt.ylabel('Execution Time (s)')
    plt.title(title)
    plt.legend()
    plt.grid(True)

    # Save the figure to the results directory
    plt.savefig(os.path.join(RESULTS_DIR, filename))
    plt.close()


def run_all_benchmarks() -> dict:
    """
    Executes all benchmark tests for realized volatility functions and compiles results

    Returns:
        dict: Comprehensive benchmark results
    """
    # Set up the benchmark environment
    setup_benchmark_environment()

    # Run benchmark_realized_variance
    realized_variance_results = benchmark_realized_variance()

    # Run benchmark_realized_kernel
    realized_kernel_results = benchmark_realized_kernel()

    # Run benchmark_sampling_schemes
    sampling_schemes_results = benchmark_sampling_schemes()

    # Run benchmark_noise_filtering
    noise_filtering_results = benchmark_noise_filtering()

    # Compile all results into a single report
    all_results = {
        "realized_variance": realized_variance_results,
        "realized_kernel": realized_kernel_results,
        "sampling_schemes": sampling_schemes_results,
        "noise_filtering": noise_filtering_results,
    }

    # Save comprehensive results to disk
    with open(os.path.join(RESULTS_DIR, "all_results.txt"), "w") as f:
        for name, result in all_results.items():
            f.write(f"Benchmark: {name}\n")
            for key, value in result.items():
                f.write(f"\t{key}: {value}\n")
            f.write("\n")

    # Return the consolidated benchmark data
    return all_results


def main() -> int:
    """
    Main entry point for the benchmark script

    Returns:
        int: Exit code (0 for success)
    """
    # Parse command-line arguments using argparse
    parser = argparse.ArgumentParser(description="MFE Toolbox - Realized Volatility Benchmarking Suite")
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    args = parser.parse_args()

    # Set up the benchmark environment
    setup_benchmark_environment()

    # Determine which benchmarks to run based on arguments
    if args.all:
        # Execute the appropriate benchmark functions
        all_results = run_all_benchmarks()
        print(all_results)
    else:
        print("No benchmarks specified. Use --all to run all benchmarks.")

    # Display summary results
    print("Benchmark completed.")

    # Return exit code 0 for success
    return 0


if __name__ == "__main__":
    sys.exit(main())