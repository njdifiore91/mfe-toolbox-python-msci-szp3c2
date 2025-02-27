"""
MFE Toolbox - Numba Benchmarking Suite

This module provides a benchmarking suite to compare the performance of
Numba-optimized functions against their pure Python counterparts within the MFE Toolbox.
It measures execution times across various data sizes and calculates speedup factors to
quantify the performance improvements achieved through Numba's JIT compilation.

The benchmark suite includes tests for core computational functions across multiple areas:
- GARCH model estimation and likelihood calculation
- Bootstrap resampling methods
- Optimization algorithms
- Volatility calculations

Results are presented both numerically and visually, allowing for easy assessment of
Numba's impact on computational performance.
"""

import os
import sys
import time
import timeit
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np  # numpy 1.26.3
import numba  # numba 0.59.0
import matplotlib.pyplot as plt  # matplotlib 3.8.0
from tqdm import tqdm  # tqdm 4.66.1

# Internal imports
from mfe.utils.numba_helpers import ensure_jit_enabled
from mfe.core.optimization import optimize_garch
from mfe.core.bootstrap import stationary_bootstrap
from mfe.models.garch import garch_likelihood
from mfe.models.volatility import compute_volatility

# Global constants
SIZES = [100, 1000, 10000, 100000]  # Data sizes to benchmark
REPETITIONS = 10  # Number of repetitions for each benchmark
RESULTS_DIR = 'results'  # Directory to save results


class BenchmarkResult:
    """
    Class to store and analyze benchmark results.
    """
    
    def __init__(self, name: str):
        """
        Initialize a new BenchmarkResult instance.
        
        Parameters
        ----------
        name : str
            Name of the benchmark
        """
        self.name = name
        self.python_times = {}
        self.numba_times = {}
        self.speedup_factors = {}
    
    def add_result(self, size: int, python_time: float, numba_time: float):
        """
        Add a new benchmark result for a specific data size.
        
        Parameters
        ----------
        size : int
            Size of the data used in the benchmark
        python_time : float
            Execution time of the pure Python implementation
        numba_time : float
            Execution time of the Numba-optimized implementation
        """
        self.python_times[size] = python_time
        self.numba_times[size] = numba_time
        
        # Calculate speedup factor (how many times faster Numba is)
        if numba_time > 0:
            self.speedup_factors[size] = python_time / numba_time
        else:
            self.speedup_factors[size] = float('inf')
    
    def summary(self) -> str:
        """
        Generate a summary of the benchmark results.
        
        Returns
        -------
        str
            A formatted summary of the benchmark results
        """
        summary = f"Benchmark Results for {self.name}\n"
        summary += "=" * (22 + len(self.name)) + "\n"
        summary += f"{'Size':<10} {'Python (s)':<12} {'Numba (s)':<12} {'Speedup':<10}\n"
        summary += "-" * 44 + "\n"
        
        for size in sorted(self.python_times.keys()):
            python_time = self.python_times[size]
            numba_time = self.numba_times[size]
            speedup = self.speedup_factors[size]
            
            summary += f"{size:<10} {python_time:<12.6f} {numba_time:<12.6f} {speedup:<10.2f}x\n"
        
        # Calculate average speedup
        avg_speedup = sum(self.speedup_factors.values()) / len(self.speedup_factors)
        summary += "-" * 44 + "\n"
        summary += f"Average Speedup: {avg_speedup:.2f}x\n"
        
        return summary
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the benchmark results to a dictionary.
        
        Returns
        -------
        dict
            Dictionary representation of the benchmark results
        """
        return {
            'name': self.name,
            'python_times': self.python_times,
            'numba_times': self.numba_times,
            'speedup_factors': self.speedup_factors,
            'average_speedup': sum(self.speedup_factors.values()) / len(self.speedup_factors)
        }


def setup_benchmark_environment():
    """
    Initialize the benchmark environment and ensure Numba JIT is properly configured.
    """
    # Check Numba availability
    if not hasattr(numba, 'jit'):
        print("ERROR: Numba is not available or properly installed.")
        print("Please ensure Numba is installed: pip install numba==0.59.0")
        sys.exit(1)
    
    # Ensure JIT is enabled (if function exists)
    try:
        ensure_jit_enabled()
        print("Numba JIT compilation is enabled.")
    except AttributeError:
        print("Warning: ensure_jit_enabled function not found. Continuing anyway.")
    
    # Create results directory if it doesn't exist
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    
    print("Benchmark environment setup complete.")
    print(f"Numba version: {numba.__version__}")
    print(f"NumPy version: {np.__version__}")
    print(f"Data sizes to benchmark: {SIZES}")
    print(f"Repetitions per benchmark: {REPETITIONS}")
    print("")


def generate_test_data(size: int, data_type: str) -> np.ndarray:
    """
    Generate synthetic test data for benchmarking.
    
    Parameters
    ----------
    size : int
        Size of the data to generate
    data_type : str
        Type of data to generate ('returns', 'prices', 'parameters', etc.)
    
    Returns
    -------
    np.ndarray
        Generated test data
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    if data_type == 'returns':
        # Generate random returns data (approximately normal)
        return np.random.normal(0, 0.01, size)
    
    elif data_type == 'prices':
        # Generate random price data (random walk)
        returns = np.random.normal(0.0005, 0.01, size)
        prices = 100 * np.cumprod(1 + returns)
        return prices
    
    elif data_type == 'parameters':
        # Generate random GARCH model parameters
        # Structure: [omega, alpha, beta]
        return np.array([0.01, 0.1, 0.85])
    
    elif data_type == 'optimization_params':
        # Initial parameters for optimization benchmarks
        return np.random.uniform(0, 1, size)
    
    else:
        # Default to random array
        return np.random.random(size)


# Pure Python implementations of functions that have Numba-optimized versions

def python_stationary_bootstrap(data: np.ndarray, statistic_func: Callable, 
                              probability: float, num_bootstrap: int) -> np.ndarray:
    """
    Pure Python implementation of stationary bootstrap.
    
    Parameters
    ----------
    data : np.ndarray
        Time series data array
    statistic_func : callable
        Function to compute the statistic of interest
    probability : float
        Probability parameter for geometric distribution
    num_bootstrap : int
        Number of bootstrap samples to generate
    
    Returns
    -------
    np.ndarray
        Array of bootstrap statistics
    """
    # Get data length
    n = len(data)
    
    # Initialize array to store bootstrap statistics
    bootstrap_statistics = np.zeros(num_bootstrap)
    
    # Generate bootstrap samples
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


def python_garch_likelihood(parameters: np.ndarray, data: np.ndarray, p: int, q: int) -> float:
    """
    Pure Python implementation of GARCH likelihood calculation.
    
    Parameters
    ----------
    parameters : np.ndarray
        Model parameters [omega, alpha_1, ..., alpha_q, beta_1, ..., beta_p]
    data : np.ndarray
        Return data for GARCH modeling
    p : int
        Number of GARCH lags
    q : int
        Number of ARCH lags
    
    Returns
    -------
    float
        Negative log-likelihood value
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
    
    # Calculate log-likelihood
    loglike = 0.0
    
    # Skip the burn-in period
    for t in range(max(p, q), T):
        if variance[t] <= 0:
            return 1e10  # Large penalty for invalid variances
        
        # Normal distribution log-likelihood
        loglike += -0.5 * np.log(2 * np.pi) - 0.5 * np.log(variance[t]) - 0.5 * data[t]**2 / variance[t]
    
    # Return negative log-likelihood for minimization
    return -loglike


def python_optimize_garch(data: np.ndarray, initial_params: np.ndarray, p: int = 1, q: int = 1) -> np.ndarray:
    """
    Pure Python implementation of GARCH parameter optimization.
    
    Parameters
    ----------
    data : np.ndarray
        Return data for GARCH modeling
    initial_params : np.ndarray
        Initial parameter values
    p : int, default=1
        Number of GARCH lags
    q : int, default=1
        Number of ARCH lags
    
    Returns
    -------
    np.ndarray
        Optimized parameters
    """
    # Define the objective function (negative log-likelihood)
    def objective(params):
        return python_garch_likelihood(params, data, p, q)
    
    # Simple optimizer implementation (gradient descent)
    params = initial_params.copy()
    learning_rate = 0.01
    n_iterations = 50  # Reduced for benchmarking purposes
    
    for _ in range(n_iterations):
        # Calculate gradient numerically
        gradient = np.zeros_like(params)
        epsilon = 1e-6
        
        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += epsilon
            
            params_minus = params.copy()
            params_minus[i] -= epsilon
            
            gradient[i] = (objective(params_plus) - objective(params_minus)) / (2 * epsilon)
        
        # Update parameters using gradient descent
        params -= learning_rate * gradient
    
    return params


def python_compute_volatility(returns: np.ndarray, alpha: float = 0.06) -> np.ndarray:
    """
    Pure Python implementation of EWMA volatility calculation.
    
    Parameters
    ----------
    returns : np.ndarray
        Array of asset returns
    alpha : float, default=0.06
        Decay factor parameter
    
    Returns
    -------
    np.ndarray
        Conditional volatility estimates
    """
    n = len(returns)
    variance = np.zeros(n)
    
    # Initialize with squared return
    variance[0] = returns[0]**2
    
    # EWMA recursion
    for t in range(1, n):
        variance[t] = alpha * returns[t-1]**2 + (1 - alpha) * variance[t-1]
    
    # Convert variance to volatility (standard deviation)
    volatility = np.sqrt(variance)
    
    return volatility


def benchmark_function(func: Callable, data: np.ndarray, 
                      params: Dict[str, Any], repetitions: int = 5) -> float:
    """
    Benchmark a function by running it multiple times and measuring execution time.
    
    Parameters
    ----------
    func : callable
        Function to benchmark
    data : np.ndarray
        Input data to pass to the function
    params : dict
        Parameters to pass to the function
    repetitions : int, default=5
        Number of repetitions to run
        
    Returns
    -------
    float
        Average execution time in seconds
    """
    # Create a timer function depending on function signature
    if func.__name__ == 'stationary_bootstrap' or func.__name__ == 'python_stationary_bootstrap':
        # Stationary bootstrap has a specific signature
        timer_func = lambda: func(
            data, 
            params.get('statistic_func', lambda x: np.mean(x)),
            params.get('probability', 0.1),
            params.get('num_bootstrap', 100)
        )
    
    elif func.__name__ == 'garch_likelihood' or func.__name__ == 'python_garch_likelihood':
        # GARCH likelihood function 
        timer_func = lambda: func(
            params.get('parameters', np.array([0.01, 0.1, 0.85])),
            data,
            params.get('p', 1),
            params.get('q', 1)
        )
    
    elif func.__name__ == 'optimize_garch' or func.__name__ == 'python_optimize_garch':
        # GARCH optimization function
        timer_func = lambda: func(
            data,
            params.get('initial_params', np.array([0.01, 0.1, 0.85])),
            params.get('p', 1),
            params.get('q', 1)
        )
    
    elif func.__name__ == 'compute_volatility' or func.__name__ == 'python_compute_volatility':
        # Volatility function
        timer_func = lambda: func(
            data,
            params.get('alpha', 0.06)
        )
    
    else:
        # Generic function call with data as first argument
        timer_func = lambda: func(data, **params)
    
    # Warm-up run to trigger Numba compilation if needed
    timer_func()
    
    # Time execution
    start_time = time.time()
    for _ in range(repetitions):
        timer_func()
    execution_time = (time.time() - start_time) / repetitions
    
    return execution_time


def compare_implementations(numba_func: Callable, python_func: Callable, 
                           func_name: str, data_sizes: List[int], 
                           params: Dict[str, Any]) -> BenchmarkResult:
    """
    Compare performance between Numba-optimized and pure Python implementations.
    
    Parameters
    ----------
    numba_func : callable
        Numba-optimized implementation
    python_func : callable
        Pure Python implementation
    func_name : str
        Name of the function being benchmarked
    data_sizes : list
        List of data sizes to benchmark
    params : dict
        Parameters for the functions
        
    Returns
    -------
    BenchmarkResult
        Performance comparison results
    """
    # Create result object
    result = BenchmarkResult(func_name)
    
    print(f"\nBenchmarking {func_name}...")
    
    # Benchmark each data size
    for size in tqdm(data_sizes, desc="Data sizes"):
        # Generate appropriate test data
        if 'data_type' in params:
            data = generate_test_data(size, params['data_type'])
        else:
            data = generate_test_data(size, 'returns')
        
        try:
            # Benchmark Python implementation
            python_time = benchmark_function(
                python_func, 
                data, 
                params, 
                repetitions=REPETITIONS
            )
            
            # Benchmark Numba implementation
            numba_time = benchmark_function(
                numba_func, 
                data, 
                params, 
                repetitions=REPETITIONS
            )
            
            # Store results
            result.add_result(size, python_time, numba_time)
            
            # Print immediate results
            speedup = result.speedup_factors[size]
            print(f"  Size {size}: Python: {python_time:.6f}s, Numba: {numba_time:.6f}s, Speedup: {speedup:.2f}x")
            
        except Exception as e:
            print(f"  Error benchmarking size {size}: {str(e)}")
            print(f"  Skipping this size...")
            continue
    
    # Print summary
    if result.speedup_factors:
        print(result.summary())
    else:
        print(f"No valid benchmarks completed for {func_name}")
    
    return result


def plot_results(results: BenchmarkResult, title: str, filename: str):
    """
    Create visualizations of benchmark results.
    
    Parameters
    ----------
    results : BenchmarkResult
        Benchmark results to visualize
    title : str
        Plot title
    filename : str
        Filename to save the plot
    """
    # Check if there are results to plot
    if not results.speedup_factors:
        print(f"No results to plot for {results.name}")
        return
    
    plt.figure(figsize=(12, 6))
    
    # Create primary axis for execution times
    ax1 = plt.gca()
    
    # Get data for plotting
    sizes = sorted(results.python_times.keys())
    python_times = [results.python_times[size] for size in sizes]
    numba_times = [results.numba_times[size] for size in sizes]
    
    # Plot execution times
    ax1.plot(sizes, python_times, 'o-', color='blue', label='Python')
    ax1.plot(sizes, numba_times, 'o-', color='red', label='Numba')
    ax1.set_xlabel('Data Size')
    ax1.set_ylabel('Execution Time (s)')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    # Create secondary axis for speedup factor
    ax2 = ax1.twinx()
    speedups = [results.speedup_factors[size] for size in sizes]
    ax2.plot(sizes, speedups, 'o--', color='green', label='Speedup')
    ax2.set_ylabel('Speedup Factor (x)')
    ax2.set_yscale('log')
    
    # Add horizontal line at speedup = 1
    ax2.axhline(y=1, color='black', linestyle='--', alpha=0.3)
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # Add title and adjust layout
    plt.title(title)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(RESULTS_DIR, filename))
    plt.close()


def run_garch_benchmark() -> BenchmarkResult:
    """
    Run benchmarks specifically for GARCH model functions.
    
    Returns
    -------
    BenchmarkResult
        Benchmark results for GARCH functions
    """
    # Set up parameters for GARCH likelihood benchmark
    params = {
        'data_type': 'returns',
        'parameters': np.array([0.01, 0.1, 0.85]),  # [omega, alpha, beta]
        'p': 1,  # GARCH order
        'q': 1   # ARCH order
    }
    
    # Use smaller sizes for GARCH to avoid excessive benchmarking time
    garch_sizes = [100, 500, 1000, 5000]
    
    try:
        # Compare implementations
        results = compare_implementations(
            garch_likelihood,
            python_garch_likelihood,
            "GARCH Likelihood",
            garch_sizes,
            params
        )
        
        # Plot results
        plot_results(
            results,
            "GARCH Likelihood: Python vs. Numba Performance",
            "garch_likelihood_benchmark.png"
        )
        
        return results
    except Exception as e:
        print(f"Error during GARCH benchmark: {str(e)}")
        return BenchmarkResult("GARCH Likelihood (Failed)")


def run_bootstrap_benchmark() -> BenchmarkResult:
    """
    Run benchmarks specifically for bootstrap functions.
    
    Returns
    -------
    BenchmarkResult
        Benchmark results for bootstrap functions
    """
    # Define a simple statistic function
    def mean_statistic(data):
        return np.mean(data)
    
    # Set up parameters for bootstrap benchmark
    params = {
        'data_type': 'returns',
        'statistic_func': mean_statistic,
        'probability': 0.1,
        'num_bootstrap': 100
    }
    
    # Use smaller data sizes for bootstrap as it's more intensive
    bootstrap_sizes = [100, 500, 1000, 2000]
    
    try:
        # Compare implementations
        results = compare_implementations(
            stationary_bootstrap,
            python_stationary_bootstrap,
            "Stationary Bootstrap",
            bootstrap_sizes,
            params
        )
        
        # Plot results
        plot_results(
            results,
            "Stationary Bootstrap: Python vs. Numba Performance",
            "bootstrap_benchmark.png"
        )
        
        return results
    except Exception as e:
        print(f"Error during bootstrap benchmark: {str(e)}")
        return BenchmarkResult("Stationary Bootstrap (Failed)")


def run_optimization_benchmark() -> BenchmarkResult:
    """
    Run benchmarks specifically for optimization algorithms.
    
    Returns
    -------
    BenchmarkResult
        Benchmark results for optimization functions
    """
    # Set up parameters for optimization benchmark
    params = {
        'data_type': 'returns',
        'initial_params': np.array([0.01, 0.1, 0.85]),  # [omega, alpha, beta]
        'p': 1,  # GARCH order
        'q': 1   # ARCH order
    }
    
    # Use smaller data sizes for optimization as it's more intensive
    opt_sizes = [100, 500, 1000, 2000]
    
    try:
        # Compare implementations
        results = compare_implementations(
            optimize_garch,
            python_optimize_garch,
            "GARCH Optimization",
            opt_sizes,
            params
        )
        
        # Plot results
        plot_results(
            results,
            "GARCH Optimization: Python vs. Numba Performance",
            "optimization_benchmark.png"
        )
        
        return results
    except Exception as e:
        print(f"Error during optimization benchmark: {str(e)}")
        return BenchmarkResult("GARCH Optimization (Failed)")


def run_volatility_benchmark() -> BenchmarkResult:
    """
    Run benchmarks specifically for volatility calculation functions.
    
    Returns
    -------
    BenchmarkResult
        Benchmark results for volatility functions
    """
    # Set up parameters for volatility benchmark
    params = {
        'data_type': 'returns',
        'alpha': 0.06  # EWMA decay factor
    }
    
    try:
        # Compare implementations
        results = compare_implementations(
            compute_volatility,
            python_compute_volatility,
            "EWMA Volatility",
            SIZES,
            params
        )
        
        # Plot results
        plot_results(
            results,
            "EWMA Volatility: Python vs. Numba Performance",
            "volatility_benchmark.png"
        )
        
        return results
    except Exception as e:
        print(f"Error during volatility benchmark: {str(e)}")
        return BenchmarkResult("EWMA Volatility (Failed)")


def run_all_benchmarks() -> Dict[str, BenchmarkResult]:
    """
    Execute all benchmark tests and compile results.
    
    Returns
    -------
    dict
        Comprehensive benchmark results for all functions
    """
    # Set up benchmark environment
    setup_benchmark_environment()
    
    # Initialize results dictionary
    all_results = {}
    
    # Run all benchmarks
    print("Running GARCH benchmarks...")
    all_results['garch'] = run_garch_benchmark()
    
    print("Running bootstrap benchmarks...")
    all_results['bootstrap'] = run_bootstrap_benchmark()
    
    print("Running optimization benchmarks...")
    all_results['optimization'] = run_optimization_benchmark()
    
    print("Running volatility benchmarks...")
    all_results['volatility'] = run_volatility_benchmark()
    
    # Create summary visualization for successful benchmarks
    successful_results = {k: v for k, v in all_results.items() if v.speedup_factors}
    if successful_results:
        create_summary_visualization(successful_results)
    
    # Print overall summary
    print("\nOverall Benchmark Summary:")
    print("=========================")
    
    any_success = False
    for name, result in all_results.items():
        if result.speedup_factors:
            avg_speedup = sum(result.speedup_factors.values()) / len(result.speedup_factors)
            print(f"{name}: Average Speedup: {avg_speedup:.2f}x")
            any_success = True
        else:
            print(f"{name}: Failed to complete")
    
    if not any_success:
        print("No benchmarks completed successfully.")
    
    return all_results


def create_summary_visualization(all_results: Dict[str, BenchmarkResult]):
    """
    Create a summary visualization comparing speedups across all benchmarks.
    
    Parameters
    ----------
    all_results : dict
        Dictionary of benchmark results
    """
    # Check if there are results to visualize
    if not all_results:
        print("No results to visualize in summary")
        return
    
    plt.figure(figsize=(12, 8))
    
    # Prepare data for plotting
    benchmark_names = list(all_results.keys())
    avg_speedups = []
    
    for name in benchmark_names:
        result = all_results[name]
        if result.speedup_factors:
            avg_speedup = sum(result.speedup_factors.values()) / len(result.speedup_factors)
            avg_speedups.append(avg_speedup)
        else:
            avg_speedups.append(0)  # No speedup for failed benchmarks
    
    # Sort by average speedup
    sorted_indices = np.argsort(avg_speedups)
    benchmark_names = [benchmark_names[i] for i in sorted_indices]
    avg_speedups = [avg_speedups[i] for i in sorted_indices]
    
    # Create bar chart
    plt.barh(benchmark_names, avg_speedups, color='steelblue')
    
    # Add labels and values
    for i, v in enumerate(avg_speedups):
        plt.text(v + 0.5, i, f"{v:.2f}x", va='center')
    
    # Set labels and title
    plt.xlabel('Average Speedup Factor (x)')
    plt.ylabel('Benchmark Category')
    plt.title('Numba Performance Improvement by Category')
    
    # Add horizontal line at speedup = 1
    plt.axvline(x=1, color='red', linestyle='--', alpha=0.7)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'summary_speedups.png'))
    plt.close()


def main():
    """
    Main entry point for the benchmark script.
    
    Returns
    -------
    int
        Exit code (0 for success)
    """
    try:
        run_all_benchmarks()
        return 0
    except Exception as e:
        print(f"Error during benchmarking: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())