"""
MFE Toolbox - GARCH Benchmarking Module

This module contains benchmarks for comparing the performance of Numba-optimized
GARCH implementations against equivalent pure Python implementations. It evaluates
the computational efficiency of different GARCH variants across various data sizes 
and model configurations.

The benchmarks include:
- GARCH variance recursion
- GARCH likelihood calculation
- Full GARCH model estimation
- GARCH forecasting
- GARCH simulation
"""

import numpy as np  # numpy 1.26.3
import time  # Python standard library
import matplotlib.pyplot as plt  # matplotlib 3.8.0
from tqdm import tqdm  # tqdm 4.66.1
import os  # Python standard library

# Internal imports
from .benchmark_numba import (
    BenchmarkResult, 
    setup_benchmark_environment, 
    benchmark_function, 
    plot_results
)
from ..mfe.models.garch import GARCH
from ..mfe.utils.numba_helpers import (
    jit_garch_recursion, 
    jit_garch_likelihood, 
    optimized_jit
)
from ..mfe.utils.numpy_helpers import ensure_array

# Global constants
SIZES = [100, 500, 1000, 5000, 10000]  # Data sizes to benchmark
REPETITIONS = 10  # Number of times to repeat each benchmark for statistical validity
GARCH_ORDERS = [{'p': 1, 'q': 1}, {'p': 1, 'q': 2}, {'p': 2, 'q': 1}]  # Different GARCH orders to benchmark
RESULTS_DIR = 'results/garch'  # Directory to save benchmark results


def generate_garch_data(size: int, p: int, q: int, omega: float, alpha: np.ndarray, beta: np.ndarray) -> tuple:
    """
    Generates synthetic financial return data following a GARCH process.
    
    Parameters
    ----------
    size : int
        Number of observations to generate
    p : int
        GARCH order (number of lagged variance terms)
    q : int
        ARCH order (number of lagged squared return terms)
    omega : float
        Constant term in the variance equation
    alpha : numpy.ndarray
        ARCH coefficients (q)
    beta : numpy.ndarray
        GARCH coefficients (p)
        
    Returns
    -------
    tuple
        (returns, volatility) - Generated return series and corresponding volatility
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Initialize arrays for returns and volatility
    returns = np.zeros(size)
    volatility = np.zeros(size)
    
    # Generate random innovations from standard normal distribution
    innovations = np.random.normal(0, 1, size)
    
    # Calculate unconditional variance for initialization
    uncond_var = omega / (1.0 - np.sum(alpha) - np.sum(beta))
    
    # Set initial values for volatility
    volatility[:max(p, q)] = uncond_var
    
    # Generate GARCH process
    for t in range(max(p, q), size):
        # ARCH component (alpha * squared returns)
        arch_component = 0.0
        for i in range(q):
            if t-i-1 >= 0:
                arch_component += alpha[i] * returns[t-i-1]**2
        
        # GARCH component (beta * past variances)
        garch_component = 0.0
        for j in range(p):
            if t-j-1 >= 0:
                garch_component += beta[j] * volatility[t-j-1]
        
        # Combine components for conditional variance
        volatility[t] = omega + arch_component + garch_component
        
        # Generate return as volatility * innovation
        returns[t] = np.sqrt(volatility[t]) * innovations[t]
    
    return returns, volatility


def garch_recursion_python(parameters: np.ndarray, data: np.ndarray, p: int, q: int) -> np.ndarray:
    """
    Pure Python implementation of GARCH variance recursion for benchmark comparison.
    
    Parameters
    ----------
    parameters : numpy.ndarray
        Model parameters [omega, alpha_1, ..., alpha_q, beta_1, ..., beta_p]
    data : numpy.ndarray
        Return data for GARCH modeling
    p : int
        GARCH order (number of lagged variance terms)
    q : int
        ARCH order (number of lagged squared return terms)
        
    Returns
    -------
    numpy.ndarray
        Conditional variance series
    """
    # Extract parameters
    omega = parameters[0]
    alpha = parameters[1:q+1]
    beta = parameters[q+1:q+p+1]
    
    # Get data length
    T = len(data)
    
    # Initialize variance array
    variance = np.zeros_like(data)
    
    # Calculate unconditional variance for initialization
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


def garch_likelihood_python(parameters: np.ndarray, data: np.ndarray, p: int, q: int) -> float:
    """
    Pure Python implementation of GARCH likelihood calculation for benchmark comparison.
    
    Parameters
    ----------
    parameters : numpy.ndarray
        Model parameters [omega, alpha_1, ..., alpha_q, beta_1, ..., beta_p]
    data : numpy.ndarray
        Return data for GARCH modeling
    p : int
        GARCH order (number of lagged variance terms)
    q : int
        ARCH order (number of lagged squared return terms)
        
    Returns
    -------
    float
        Negative log-likelihood value
    """
    # Calculate conditional variances
    variance = garch_recursion_python(parameters, data, p, q)
    
    # Initialize log-likelihood
    loglike = 0.0
    
    # Skip the burn-in period
    T = len(data)
    for t in range(max(p, q), T):
        # Check for valid variance
        if variance[t] <= 0:
            return 1e10  # Large penalty for invalid variance
        
        # Normal distribution log-likelihood
        loglike += -0.5 * np.log(2 * np.pi) - 0.5 * np.log(variance[t]) - 0.5 * data[t]**2 / variance[t]
    
    # Return negative log-likelihood for minimization
    return -loglike


def benchmark_garch_recursion() -> BenchmarkResult:
    """
    Benchmarks GARCH variance recursion comparing Numba-optimized vs. pure Python implementation.
    
    Returns
    -------
    BenchmarkResult
        Benchmark results for GARCH variance recursion
    """
    # Create result container
    result = BenchmarkResult("GARCH Variance Recursion")
    
    print("\nBenchmarking GARCH variance recursion...")
    
    # Loop through different data sizes
    for size in tqdm(SIZES, desc="Data sizes"):
        # Generate test data
        alpha = np.array([0.1])
        beta = np.array([0.85])
        omega = 0.01
        returns, _ = generate_garch_data(size, 1, 1, omega, alpha, beta)
        
        # Ensure data is proper numpy array
        returns = ensure_array(returns)
        
        # Create parameters array
        parameters = np.array([omega, alpha[0], beta[0]])
        
        # Benchmark Python implementation
        start_time = time.time()
        for _ in range(REPETITIONS):
            garch_recursion_python(parameters, returns, 1, 1)
        python_time = (time.time() - start_time) / REPETITIONS
        
        # Benchmark Numba implementation
        # First run to compile
        jit_garch_recursion(parameters, returns, 1, 1)
        
        start_time = time.time()
        for _ in range(REPETITIONS):
            jit_garch_recursion(parameters, returns, 1, 1)
        numba_time = (time.time() - start_time) / REPETITIONS
        
        # Add result
        result.add_result(size, python_time, numba_time)
        
        # Print immediate results
        print(f"  Size {size}: Python: {python_time:.6f}s, Numba: {numba_time:.6f}s, " 
              f"Speedup: {python_time/numba_time:.2f}x")
    
    # Plot the results
    plot_results(
        result,
        "GARCH Variance Recursion: Python vs. Numba Performance",
        os.path.join(RESULTS_DIR, "garch_recursion_benchmark.png")
    )
    
    return result


def benchmark_garch_likelihood() -> BenchmarkResult:
    """
    Benchmarks GARCH likelihood calculation comparing Numba-optimized vs. pure Python implementation.
    
    Returns
    -------
    BenchmarkResult
        Benchmark results for GARCH likelihood calculation
    """
    # Create result container
    result = BenchmarkResult("GARCH Likelihood Calculation")
    
    print("\nBenchmarking GARCH likelihood calculation...")
    
    # Loop through different data sizes
    for size in tqdm(SIZES, desc="Data sizes"):
        # Generate test data
        alpha = np.array([0.1])
        beta = np.array([0.85])
        omega = 0.01
        returns, _ = generate_garch_data(size, 1, 1, omega, alpha, beta)
        
        # Ensure data is proper numpy array
        returns = ensure_array(returns)
        
        # Create parameters array
        parameters = np.array([omega, alpha[0], beta[0]])
        
        # Benchmark Python implementation
        start_time = time.time()
        for _ in range(REPETITIONS):
            garch_likelihood_python(parameters, returns, 1, 1)
        python_time = (time.time() - start_time) / REPETITIONS
        
        # Benchmark Numba implementation
        # First run to compile
        jit_garch_likelihood(parameters, returns, 1, 1, np.log)
        
        start_time = time.time()
        for _ in range(REPETITIONS):
            jit_garch_likelihood(parameters, returns, 1, 1, np.log)
        numba_time = (time.time() - start_time) / REPETITIONS
        
        # Add result
        result.add_result(size, python_time, numba_time)
        
        # Print immediate results
        print(f"  Size {size}: Python: {python_time:.6f}s, Numba: {numba_time:.6f}s, " 
              f"Speedup: {python_time/numba_time:.2f}x")
    
    # Plot the results
    plot_results(
        result,
        "GARCH Likelihood Calculation: Python vs. Numba Performance",
        os.path.join(RESULTS_DIR, "garch_likelihood_benchmark.png")
    )
    
    return result


def benchmark_garch_estimation() -> dict:
    """
    Benchmarks full GARCH model estimation using the GARCH class.
    
    Returns
    -------
    dict
        Benchmark results for GARCH model estimation with different orders
    """
    # Initialize results dictionary
    results = {}
    
    print("\nBenchmarking GARCH model estimation...")
    
    # Loop through different GARCH orders
    for order in GARCH_ORDERS:
        p = order['p']
        q = order['q']
        
        print(f"\nGARCH({p},{q}) estimation:")
        
        # Create result container
        order_name = f"GARCH({p},{q}) Estimation"
        result = BenchmarkResult(order_name)
        
        # Loop through different data sizes
        for size in tqdm(SIZES[:3], desc="Data sizes"):  # Using smaller subset for full estimation
            # Generate test data
            alpha = np.array([0.1 / q] * q)  # Normalize alpha parameters
            beta = np.array([0.85 / p] * p)  # Normalize beta parameters
            omega = 0.01
            returns, _ = generate_garch_data(size, p, q, omega, alpha, beta)
            
            # Ensure data is proper numpy array
            returns = ensure_array(returns)
            
            # Create GARCH model
            model = GARCH(p=p, q=q)
            
            # Benchmark estimation
            try:
                start_time = time.time()
                model.fit(returns)
                estimation_time = time.time() - start_time
                
                # Add result (no Numba/Python comparison here, just timing GARCH estimation)
                result.add_result(size, estimation_time, estimation_time)  # Same value used twice
                
                # Print immediate results
                print(f"  Size {size}: Estimation time: {estimation_time:.6f}s")
            except Exception as e:
                print(f"  Error estimating GARCH({p},{q}) with size {size}: {str(e)}")
        
        # Plot the results
        plot_results(
            result,
            f"GARCH({p},{q}) Estimation Performance",
            os.path.join(RESULTS_DIR, f"garch_{p}_{q}_estimation_benchmark.png")
        )
        
        # Store results for this order
        results[f"GARCH({p},{q})"] = result
    
    return results


def benchmark_garch_forecast() -> BenchmarkResult:
    """
    Benchmarks GARCH volatility forecasting performance.
    
    Returns
    -------
    BenchmarkResult
        Benchmark results for GARCH forecasting
    """
    # Create result container
    result = BenchmarkResult("GARCH Forecasting")
    
    print("\nBenchmarking GARCH forecasting...")
    
    # Define forecast horizons
    horizons = [1, 5, 10, 20, 50]
    
    # Generate fixed-sized dataset for all tests
    size = 1000
    alpha = np.array([0.1])
    beta = np.array([0.85])
    omega = 0.01
    returns, _ = generate_garch_data(size, 1, 1, omega, alpha, beta)
    
    # Ensure data is proper numpy array
    returns = ensure_array(returns)
    
    # Create and fit GARCH model
    model = GARCH(p=1, q=1)
    try:
        model.fit(returns)
        
        # Loop through different forecast horizons
        for horizon in tqdm(horizons, desc="Forecast horizons"):
            # Benchmark forecasting
            start_time = time.time()
            for _ in range(REPETITIONS):
                model.forecast(horizon)
            forecast_time = (time.time() - start_time) / REPETITIONS
            
            # Add result
            result.add_result(horizon, forecast_time, forecast_time)  # Same value used twice
            
            # Print immediate results
            print(f"  Horizon {horizon}: Forecast time: {forecast_time:.6f}s")
    
    except Exception as e:
        print(f"Error during GARCH model fitting for forecasting benchmark: {str(e)}")
    
    # Plot the results
    plot_results(
        result,
        "GARCH Forecasting Performance",
        os.path.join(RESULTS_DIR, "garch_forecast_benchmark.png")
    )
    
    return result


def benchmark_garch_simulation() -> BenchmarkResult:
    """
    Benchmarks GARCH simulation performance.
    
    Returns
    -------
    BenchmarkResult
        Benchmark results for GARCH simulation
    """
    # Create result container
    result = BenchmarkResult("GARCH Simulation")
    
    print("\nBenchmarking GARCH simulation...")
    
    # Define simulation sizes
    sim_sizes = [100, 500, 1000, 5000, 10000]
    
    # Generate fixed-sized dataset for model fitting
    size = 1000
    alpha = np.array([0.1])
    beta = np.array([0.85])
    omega = 0.01
    returns, _ = generate_garch_data(size, 1, 1, omega, alpha, beta)
    
    # Ensure data is proper numpy array
    returns = ensure_array(returns)
    
    # Create and fit GARCH model
    model = GARCH(p=1, q=1)
    try:
        model.fit(returns)
        
        # Loop through different simulation sizes
        for sim_size in tqdm(sim_sizes, desc="Simulation sizes"):
            # Benchmark simulation
            start_time = time.time()
            for _ in range(REPETITIONS):
                model.simulate(sim_size)
            simulation_time = (time.time() - start_time) / REPETITIONS
            
            # Add result
            result.add_result(sim_size, simulation_time, simulation_time)  # Same value used twice
            
            # Print immediate results
            print(f"  Size {sim_size}: Simulation time: {simulation_time:.6f}s")
    
    except Exception as e:
        print(f"Error during GARCH model fitting for simulation benchmark: {str(e)}")
    
    # Plot the results
    plot_results(
        result,
        "GARCH Simulation Performance",
        os.path.join(RESULTS_DIR, "garch_simulation_benchmark.png")
    )
    
    return result


def run_all_garch_benchmarks() -> dict:
    """
    Main function to run all GARCH-related benchmarks.
    
    Returns
    -------
    dict
        Comprehensive benchmark results for all GARCH functions
    """
    # Set up the benchmark environment
    setup_benchmark_environment()
    
    # Create results directory if it doesn't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Run all benchmarks
    print("Starting GARCH benchmarks...")
    
    recursion_results = benchmark_garch_recursion()
    likelihood_results = benchmark_garch_likelihood()
    estimation_results = benchmark_garch_estimation()
    forecast_results = benchmark_garch_forecast()
    simulation_results = benchmark_garch_simulation()
    
    # Compile all results
    all_results = {
        'recursion': recursion_results,
        'likelihood': likelihood_results,
        'estimation': estimation_results,
        'forecast': forecast_results,
        'simulation': simulation_results
    }
    
    # Save full summary
    with open(os.path.join(RESULTS_DIR, "garch_benchmark_summary.txt"), "w") as f:
        f.write("GARCH Benchmarking Summary\n")
        f.write("==========================\n\n")
        
        f.write(recursion_results.summary())
        f.write("\n\n")
        
        f.write(likelihood_results.summary())
        f.write("\n\n")
        
        f.write("GARCH Estimation Results:\n")
        for name, result in estimation_results.items():
            f.write(f"\n{result.summary()}")
        f.write("\n\n")
        
        f.write(forecast_results.summary())
        f.write("\n\n")
        
        f.write(simulation_results.summary())
    
    print("\nAll GARCH benchmarks completed.")
    print(f"Results saved to {os.path.join(RESULTS_DIR, 'garch_benchmark_summary.txt')}")
    
    return all_results


def main() -> int:
    """
    Main entry point for the GARCH benchmarking script.
    
    Returns
    -------
    int
        Exit code (0 for success)
    """
    try:
        run_all_garch_benchmarks()
        return 0
    except Exception as e:
        print(f"Error during benchmarking: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())