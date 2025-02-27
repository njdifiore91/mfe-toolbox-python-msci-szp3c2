"""
Benchmark module for ARMA model performance testing, comparing Numba-optimized
implementations against pure Python equivalents. Measures execution time and
performance improvement across different data sizes and model configurations.
"""

import time  # standard library
from typing import Dict  # Python 3.12

import numpy as np  # numpy 1.26.3
from tqdm import tqdm  # tqdm 4.66.1

# Internal imports
from .benchmark_numba import (
    BenchmarkResult,
    benchmark_function,
    plot_results,
    setup_benchmark_environment,
)
from ..mfe.models.arma import ARMA, compute_arma_residuals, arma_forecast
from ..mfe.utils.numba_helpers import optimized_jit  # Numba decorator

# Global constants
SIZES = [100, 500, 1000, 5000, 10000]  # Data sizes to benchmark
REPETITIONS = 10  # Number of times to repeat each benchmark for statistical validity
ARMA_ORDERS = [
    {"p": 1, "q": 1},
    {"p": 2, "q": 2},
    {"p": 5, "q": 5},
]  # Different ARMA orders to benchmark


def generate_arma_data(
    size: int, p: int, q: int, ar_coef: float, ma_coef: float
) -> np.ndarray:
    """
    Generates synthetic time series data from an ARMA process for benchmarking

    Parameters
    ----------
    size : int
        Size of the time series data to generate
    p : int
        Autoregressive order
    q : int
        Moving average order
    ar_coef : float
        Coefficient for the AR component
    ma_coef : float
        Coefficient for the MA component

    Returns
    -------
    np.ndarray
        Generated ARMA process time series data
    """
    # Set random seed for reproducibility
    np.random.seed(42)

    # Initialize the time series array with proper length including burn-in period
    burn_in = 100
    y = np.zeros(size + burn_in)

    # Generate random innovations from normal distribution
    errors = np.random.normal(0, 1, size + burn_in)

    # Apply AR and MA components recursively to generate the ARMA process
    for t in range(max(p, q), size + burn_in):
        ar_part = sum(ar_coef * y[t - i - 1] for i in range(p))
        ma_part = sum(ma_coef * errors[t - i - 1] for i in range(q))
        y[t] = ar_part + ma_part + errors[t]

    # Discard burn-in period from the beginning
    y = y[burn_in:]

    # Return the generated time series data
    return y


@optimized_jit()
def compute_arma_residuals_python(
    y: np.ndarray, ar_params: np.ndarray, ma_params: np.ndarray, constant: float
) -> np.ndarray:
    """
    Pure Python implementation of ARMA residuals computation for benchmark comparison

    Parameters
    ----------
    y : np.ndarray
        Time series data
    ar_params : np.ndarray
        Autoregressive parameters
    ma_params : np.ndarray
        Moving average parameters
    constant : float
        Constant term in the model

    Returns
    -------
    np.ndarray
        Array of residuals from the ARMA model
    """
    # Initialize residuals array with zeros
    n = len(y)
    residuals = np.zeros(n)

    # Loop through each time point in the series
    for t in range(n):
        # Initialize predicted value with constant
        y_pred = constant

        # Apply AR component using for loops and simple calculations
        for i in range(len(ar_params)):
            if t - i - 1 >= 0:
                y_pred += ar_params[i] * y[t - i - 1]

        # Apply MA component using for loops and simple calculations
        for j in range(len(ma_params)):
            if t - j - 1 >= 0:
                y_pred += ma_params[j] * residuals[t - j - 1]

        # Compute residual
        residuals[t] = y[t] - y_pred

    # Return computed residuals without using Numba optimization
    return residuals


@optimized_jit()
def arma_forecast_python(
    y: np.ndarray,
    ar_params: np.ndarray,
    ma_params: np.ndarray,
    constant: float,
    steps: int,
    residuals: np.ndarray,
) -> np.ndarray:
    """
    Pure Python implementation of ARMA forecasting for benchmark comparison

    Parameters
    ----------
    y : np.ndarray
        Time series data
    ar_params : np.ndarray
        Autoregressive parameters
    ma_params : np.ndarray
        Moving average parameters
    constant : float
        Constant term in the model
    steps : int
        Number of steps ahead to forecast
    residuals : np.ndarray
        Array of residuals from the ARMA model

    Returns
    -------
    np.ndarray
        Array of forecasted values for specified steps ahead
    """
    # Initialize forecast array with zeros
    forecasts = np.zeros(steps)
    n = len(y)

    # For each forecast step:
    for h in range(steps):
        # Initialize forecast with constant
        forecast = constant

        # Apply AR component using past actual and forecasted values with Python loops
        for i in range(len(ar_params)):
            if h - i - 1 >= 0:
                forecast += ar_params[i] * forecasts[h - i - 1]
            else:
                forecast += ar_params[i] * y[n - i - 1]

        # Apply MA component using past residuals (zeros for future periods) with Python loops
        for j in range(len(ma_params)):
            if h - j - 1 >= 0:
                forecast += ma_params[j] * 0  # Future residuals are zero
            else:
                forecast += ma_params[j] * residuals[n - j - 1]

        # Store the forecasted value
        forecasts[h] = forecast

    # Return forecasted values array
    return forecasts


def benchmark_arma_residuals() -> BenchmarkResult:
    """
    Benchmarks ARMA residual computation comparing Numba-optimized vs. pure Python implementation

    Returns
    -------
    BenchmarkResult
        Benchmark results for ARMA residual computation
    """
    # Create BenchmarkResult object for storing results
    result = BenchmarkResult("ARMA Residuals")

    # For each data size in SIZES:
    for size in SIZES:
        # Generate appropriate ARMA test data
        y = generate_arma_data(size, p=2, q=2, ar_coef=0.5, ma_coef=0.3)

        # Create AR and MA parameters for testing
        ar_params = np.array([0.5, 0.2])
        ma_params = np.array([0.3, 0.1])
        constant = 0.1

        # Benchmark Numba-optimized compute_arma_residuals function
        numba_time = benchmark_function(
            compute_arma_residuals,
            y,
            {"ar_params": ar_params, "ma_params": ma_params, "constant": constant},
            repetitions=REPETITIONS,
        )

        # Benchmark pure Python compute_arma_residuals_python function
        python_time = benchmark_function(
            compute_arma_residuals_python,
            y,
            {"ar_params": ar_params, "ma_params": ma_params, "constant": constant},
            repetitions=REPETITIONS,
        )

        # Add results with size, Python time, and Numba time
        result.add_result(size, python_time, numba_time)

    # Plot the benchmark results
    plot_results(
        result,
        "ARMA Residuals: Python vs. Numba Performance",
        "arma_residuals_benchmark.png",
    )

    # Return the benchmark results object
    return result


def benchmark_arma_forecast() -> BenchmarkResult:
    """
    Benchmarks ARMA forecasting comparing Numba-optimized vs. pure Python implementation

    Returns
    -------
    BenchmarkResult
        Benchmark results for ARMA forecasting
    """
    # Create BenchmarkResult object for storing results
    result = BenchmarkResult("ARMA Forecast")

    # For each data size in SIZES:
    for size in SIZES:
        # Generate appropriate ARMA test data
        y = generate_arma_data(size, p=2, q=2, ar_coef=0.5, ma_coef=0.3)

        # Create AR and MA parameters for testing
        ar_params = np.array([0.5, 0.2])
        ma_params = np.array([0.3, 0.1])
        constant = 0.1
        steps = 5  # Number of steps to forecast

        # Compute residuals for forecast input
        residuals = compute_arma_residuals(y, ar_params, ma_params, constant)

        # Benchmark Numba-optimized arma_forecast function
        numba_time = benchmark_function(
            arma_forecast,
            y,
            {
                "ar_params": ar_params,
                "ma_params": ma_params,
                "constant": constant,
                "steps": steps,
                "residuals": residuals,
            },
            repetitions=REPETITIONS,
        )

        # Benchmark pure Python arma_forecast_python function
        python_time = benchmark_function(
            arma_forecast_python,
            y,
            {
                "ar_params": ar_params,
                "ma_params": ma_params,
                "constant": constant,
                "steps": steps,
                "residuals": residuals,
            },
            repetitions=REPETITIONS,
        )

        # Add results with size, Python time, and Numba time
        result.add_result(size, python_time, numba_time)

    # Plot the benchmark results
    plot_results(
        result,
        "ARMA Forecast: Python vs. Numba Performance",
        "arma_forecast_benchmark.png",
    )

    # Return the benchmark results object
    return result


def benchmark_arma_estimation() -> dict:
    """
    Benchmarks full ARMA model estimation using the ARMA class

    Returns
    -------
    dict
        Benchmark results for ARMA model estimation with different orders
    """
    # Initialize results dictionary
    results = {}

    # For each ARMA order in ARMA_ORDERS:
    for order in ARMA_ORDERS:
        # Create BenchmarkResult object for current order
        p = order["p"]
        q = order["q"]
        result = BenchmarkResult(f"ARMA({p},{q}) Estimation")

        # For each data size in SIZES:
        for size in SIZES:
            # Generate appropriate ARMA test data
            y = generate_arma_data(size, p=p, q=q, ar_coef=0.5, ma_coef=0.3)

            # Create ARMA model instance with current orders
            model = ARMA(p=p, q=q)

            # Measure time to estimate model
            start_time = time.time()
            model.estimate(y)
            estimation_time = time.time() - start_time

            # Add estimation time to results
            result.add_result(size, estimation_time, estimation_time)  # Same time for both

        # Plot the benchmark results for current order
        plot_results(
            result,
            f"ARMA({p},{q}) Estimation: Python Performance",
            f"arma_estimation_benchmark_{p}_{q}.png",

        )

        # Add current order results to overall results dictionary
        results[f"ARMA({p},{q})"] = result

    # Return the complete results dictionary
    return results


def run_all_arma_benchmarks() -> dict:
    """
    Main function to run all ARMA-related benchmarks

    Returns
    -------
    dict
        Comprehensive benchmark results for all ARMA functions
    """
    # Initialize the benchmark environment
    setup_benchmark_environment()

    # Run benchmark_arma_residuals
    residuals_result = benchmark_arma_residuals()

    # Run benchmark_arma_forecast
    forecast_result = benchmark_arma_forecast()

    # Run benchmark_arma_estimation
    estimation_results = benchmark_arma_estimation()

    # Compile all results into a single report
    all_results = {
        "residuals": residuals_result,
        "forecast": forecast_result,
        "estimation": estimation_results,
    }

    # Print summary of benchmarks
    print("\nARMA Benchmark Summary:")
    print("=====================")
    for name, result in all_results.items():
        if isinstance(result, dict):
            for model_name, model_result in result.items():
                print(f"\n{model_name}:")
                print(model_result.summary())
        else:
            print(f"\n{name}:")
            print(result.summary())

    # Return compiled benchmark results
    return all_results


def main() -> int:
    """
    Main entry point for the ARMA benchmarking script

    Returns
    -------
    int
        Exit code (0 for success)
    """
    # Parse any command-line arguments
    # (Currently, no arguments are defined, but this can be extended)

    # Call run_all_arma_benchmarks
    run_all_arma_benchmarks()

    # Print summary of results
    # (Summary is already printed in run_all_arma_benchmarks)

    # Return exit code 0 for success
    return 0


if __name__ == "__main__":
    main()