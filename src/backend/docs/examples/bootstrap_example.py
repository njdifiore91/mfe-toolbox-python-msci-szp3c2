"""
Example script demonstrating the use of bootstrap analysis functionality from the MFE Toolbox.
This educational example shows how to use different bootstrap methods for financial time series analysis,
compute confidence intervals, and conduct hypothesis tests using the Python implementation of the toolbox.
"""

import time  # Python 3.12
import asyncio  # Python 3.12
import numpy as np  # numpy 1.26.3
import matplotlib.pyplot as plt  # matplotlib 3.8.2
from scipy import stats  # scipy 1.11.4

# Import internal modules from the MFE Toolbox
from mfe.core.bootstrap import Bootstrap, BootstrapResult  # src/backend/mfe/core/bootstrap.py
from mfe.utils.numpy_helpers import ensure_array  # src/backend/mfe/utils/numpy_helpers.py
from mfe.utils.data_handling import load_financial_data  # src/backend/mfe/utils/data_handling.py

# Define global constants for the example
RANDOM_SEED = 42
FIGURE_SIZE = (10, 6)


def generate_sample_data(n_samples: int, mean: float, std: float) -> np.ndarray:
    """
    Generates a sample financial time series with typical characteristics for bootstrap demonstration

    Parameters
    ----------
    n_samples : int
        Number of data points in the time series
    mean : float
        Mean of the generated time series
    std : float
        Standard deviation of the generated time series

    Returns
    -------
    numpy.ndarray
        Simulated financial return series
    """
    # Set random seed for reproducibility
    np.random.seed(RANDOM_SEED)

    # Generate random normal returns with specified mean and std
    returns = np.random.normal(mean, std, n_samples)

    # Add autocorrelation to simulate time series dependence
    for i in range(1, n_samples):
        returns[i] += 0.5 * returns[i - 1]

    # Add volatility clustering to simulate GARCH effects
    volatility = np.random.normal(0, 0.1, n_samples)
    for i in range(1, n_samples):
        volatility[i] = 0.8 * volatility[i - 1] + 0.2 * np.random.normal(0, 0.1, 1)
        returns[i] *= np.abs(volatility[i])

    # Return the simulated return series
    return returns


def mean_statistic(data: np.ndarray) -> float:
    """
    Calculates the mean of a data series, used as example statistic for bootstrap

    Parameters
    ----------
    data : numpy.ndarray
        Input data series

    Returns
    -------
    float
        Mean of the input array
    """
    # Ensure input is a NumPy array
    data = ensure_array(data)

    # Calculate mean using numpy.mean
    mean = np.mean(data)

    # Return the computed mean
    return mean


def sharpe_ratio(returns: np.ndarray, risk_free_rate: float) -> float:
    """
    Calculates the Sharpe ratio of returns, used as complex statistic example

    Parameters
    ----------
    returns : numpy.ndarray
        Array of returns
    risk_free_rate : float
        Risk-free rate of return

    Returns
    -------
    float
        Sharpe ratio
    """
    # Ensure input is a NumPy array
    returns = ensure_array(returns)

    # Calculate mean excess return (mean - risk_free_rate)
    mean_excess_return = np.mean(returns) - risk_free_rate

    # Calculate standard deviation of returns
    std_dev = np.std(returns)

    # Compute Sharpe ratio as excess return divided by standard deviation
    sharpe_ratio = mean_excess_return / std_dev

    # Return the Sharpe ratio
    return sharpe_ratio


def run_block_bootstrap_example(returns: np.ndarray, block_size: int, num_bootstrap: int) -> None:
    """
    Demonstrates the block bootstrap method for time series analysis

    Parameters
    ----------
    returns : numpy.ndarray
        Financial return series
    block_size : int
        Size of the blocks for resampling
    num_bootstrap : int
        Number of bootstrap samples
    """
    # Print explanation of block bootstrap method
    print("\nRunning Block Bootstrap Example:")
    print("Block bootstrap resamples blocks of consecutive observations to preserve time series structure.")

    # Initialize Bootstrap object with method='block'
    bootstrap = Bootstrap(method='block', params={'block_size': block_size}, num_bootstrap=num_bootstrap)

    # Run bootstrap on returns using mean_statistic
    result = bootstrap.run(returns, mean_statistic)

    # Print original statistic value and bootstrap statistics summary
    print(f"Original Mean: {result.original_statistic:.4f}")
    summary = result.summary()
    print(f"Bootstrap Mean: {summary['bootstrap_mean']:.4f}")
    print(f"Bootstrap Standard Error: {summary['standard_error']:.4f}")

    # Calculate and print 95% confidence interval
    lower_ci, upper_ci = result.confidence_interval()
    print(f"95% Confidence Interval: ({lower_ci:.4f}, {upper_ci:.4f})")

    # Plot bootstrap distribution with matplotlib
    plt.figure(figsize=FIGURE_SIZE)
    ax = result.plot_distribution()
    ax.set_title("Block Bootstrap Distribution of Mean Returns")
    ax.axvline(0, color='red', linestyle='--', label='Null Hypothesis (Mean = 0)')
    plt.legend()

    # Display the figure with explanatory annotations
    print("Displaying Block Bootstrap Distribution...")
    plt.show()


def run_stationary_bootstrap_example(returns: np.ndarray, probability: float, num_bootstrap: int) -> None:
    """
    Demonstrates the stationary bootstrap method with random block lengths

    Parameters
    ----------
    returns : numpy.ndarray
        Financial return series
    probability : float
        Probability for geometric distribution of block lengths
    num_bootstrap : int
        Number of bootstrap samples
    """
    # Print explanation of stationary bootstrap method
    print("\nRunning Stationary Bootstrap Example:")
    print("Stationary bootstrap uses random block lengths drawn from a geometric distribution.")

    # Initialize Bootstrap object with method='stationary'
    bootstrap = Bootstrap(method='stationary', params={'probability': probability}, num_bootstrap=num_bootstrap)

    # Run bootstrap on returns using mean_statistic
    result = bootstrap.run(returns, mean_statistic)

    # Print original statistic value and bootstrap statistics summary
    print(f"Original Mean: {result.original_statistic:.4f}")
    summary = result.summary()
    print(f"Bootstrap Mean: {summary['bootstrap_mean']:.4f}")
    print(f"Bootstrap Standard Error: {summary['standard_error']:.4f}")

    # Calculate and print 95% confidence interval
    lower_ci, upper_ci = result.confidence_interval()
    print(f"95% Confidence Interval: ({lower_ci:.4f}, {upper_ci:.4f})")

    # Plot bootstrap distribution with matplotlib
    plt.figure(figsize=FIGURE_SIZE)
    ax = result.plot_distribution()
    ax.set_title("Stationary Bootstrap Distribution of Mean Returns")
    ax.axvline(0, color='red', linestyle='--', label='Null Hypothesis (Mean = 0)')
    plt.legend()

    # Display the figure with explanatory annotations
    print("Displaying Stationary Bootstrap Distribution...")
    plt.show()


def run_async_bootstrap_example(returns: np.ndarray, block_size: int, num_bootstrap: int) -> None:
    """
    Demonstrates the asynchronous bootstrap execution with progress reporting

    Parameters
    ----------
    returns : numpy.ndarray
        Financial return series
    block_size : int
        Size of the blocks for resampling
    num_bootstrap : int
        Number of bootstrap samples
    """
    # Print explanation of asynchronous bootstrap execution
    print("\nRunning Asynchronous Bootstrap Example:")
    print("Asynchronous bootstrap execution with progress reporting using async/await.")

    # Define progress callback function to print progress updates
    def progress_callback(progress: float) -> None:
        print(f"Bootstrap Progress: {progress:.2f}%")

    # Initialize Bootstrap object with method='block'
    bootstrap = Bootstrap(method='block', params={'block_size': block_size}, num_bootstrap=num_bootstrap)

    # Setup asyncio event loop
    loop = asyncio.get_event_loop()

    # Execute bootstrap.run_async with the progress callback
    start_time = time.time()
    async_result = loop.run_until_complete(bootstrap.run_async(returns, mean_statistic, progress_callback))
    async_time = time.time() - start_time

    # Run synchronous version for comparison
    start_time = time.time()
    sync_result = bootstrap.run(returns, mean_statistic)
    sync_time = time.time() - start_time

    # Print execution time comparison with synchronous version
    print(f"Asynchronous Bootstrap Time: {async_time:.4f} seconds")
    print(f"Synchronous Bootstrap Time: {sync_time:.4f} seconds")

    # Plot bootstrap distribution to demonstrate results are identical
    plt.figure(figsize=FIGURE_SIZE)
    ax1 = async_result.plot_distribution(title="Asynchronous Bootstrap Distribution")
    ax2 = sync_result.plot_distribution(ax=ax1, show_ci=False, show_original=False)  # Overlay synchronous result
    ax2.set_title("Comparison of Async and Sync Bootstrap Distributions")
    plt.legend(["Asynchronous", "Synchronous"])

    # Show execution advantages of async implementation
    print("Displaying Asynchronous Bootstrap Distribution...")
    plt.show()


def run_hypothesis_test_example(returns: np.ndarray, null_value: float, num_bootstrap: int) -> None:
    """
    Demonstrates bootstrap-based hypothesis testing

    Parameters
    ----------
    returns : numpy.ndarray
        Financial return series
    null_value : float
        Null hypothesis value for the mean return
    num_bootstrap : int
        Number of bootstrap samples
    """
    # Print explanation of bootstrap hypothesis testing
    print("\nRunning Bootstrap Hypothesis Test Example:")
    print("Bootstrap-based hypothesis testing for mean return against a null value.")

    # Initialize Bootstrap object with method='block'
    bootstrap = Bootstrap(method='block', params={'block_size': 20}, num_bootstrap=num_bootstrap)

    # Run hypothesis test for mean return against null_value
    print("\nTwo-Sided Test:")
    result_two_sided, p_value_two_sided = bootstrap.hypothesis_test(returns, mean_statistic, null_value, alternative='two-sided')
    print(f"P-value (Two-Sided): {p_value_two_sided:.4f}")

    print("\nGreater Than Test:")
    result_greater, p_value_greater = bootstrap.hypothesis_test(returns, mean_statistic, null_value, alternative='greater')
    print(f"P-value (Greater Than): {p_value_greater:.4f}")

    print("\nLess Than Test:")
    result_less, p_value_less = bootstrap.hypothesis_test(returns, mean_statistic, null_value, alternative='less')
    print(f"P-value (Less Than): {p_value_less:.4f}")

    # Plot bootstrap distribution with null value reference line
    plt.figure(figsize=FIGURE_SIZE)
    ax = result_two_sided.plot_distribution()
    ax.set_title("Bootstrap Hypothesis Test Distribution")
    ax.axvline(null_value, color='red', linestyle='--', label='Null Hypothesis')
    plt.legend()

    # Display the figure with explanatory annotations
    print("Displaying Bootstrap Hypothesis Test Distribution...")
    plt.show()


def run_complex_statistic_example(returns: np.ndarray, num_bootstrap: int) -> None:
    """
    Demonstrates bootstrap analysis for complex financial statistics

    Parameters
    ----------
    returns : numpy.ndarray
        Financial return series
    num_bootstrap : int
        Number of bootstrap samples
    """
    # Print explanation of complex financial statistics bootstrap
    print("\nRunning Complex Statistic (Sharpe Ratio) Bootstrap Example:")
    print("Bootstrap analysis for complex financial statistics like Sharpe Ratio.")

    # Define sharpe_ratio as the complex statistic function
    risk_free_rate = 0.02  # Example risk-free rate
    sharpe_ratio_func = lambda x: sharpe_ratio(x, risk_free_rate)

    # Initialize Bootstrap object with method='block'
    bootstrap = Bootstrap(method='block', params={'block_size': 20}, num_bootstrap=num_bootstrap)

    # Run bootstrap on returns using sharpe_ratio
    result = bootstrap.run(returns, sharpe_ratio_func)

    # Print original Sharpe ratio and bootstrap statistics summary
    print(f"Original Sharpe Ratio: {result.original_statistic:.4f}")
    summary = result.summary()
    print(f"Bootstrap Mean Sharpe Ratio: {summary['bootstrap_mean']:.4f}")
    print(f"Bootstrap Standard Error: {summary['standard_error']:.4f}")

    # Calculate and print 95% confidence interval for Sharpe ratio
    lower_ci, upper_ci = result.confidence_interval()
    print(f"95% Confidence Interval: ({lower_ci:.4f}, {upper_ci:.4f})")

    # Plot bootstrap distribution of Sharpe ratio estimates
    plt.figure(figsize=FIGURE_SIZE)
    ax = result.plot_distribution()
    ax.set_title("Bootstrap Distribution of Sharpe Ratio Estimates")
    plt.legend()

    # Display the figure with explanatory annotations
    print("Displaying Bootstrap Distribution of Sharpe Ratio Estimates...")
    plt.show()


def main() -> None:
    """
    Main function that runs all bootstrap examples sequentially
    """
    # Print introduction to MFE Toolbox bootstrap examples
    print("MFE Toolbox - Bootstrap Examples")
    print("This script demonstrates various bootstrap methods for financial time series analysis.")

    # Generate sample financial return data
    n_samples = 1000
    mean_return = 0.001
    std_dev = 0.02
    returns = generate_sample_data(n_samples, mean_return, std_dev)

    # Run block bootstrap example with appropriate parameters
    block_size = 20
    num_bootstrap = 500
    run_block_bootstrap_example(returns, block_size, num_bootstrap)

    # Run stationary bootstrap example with appropriate parameters
    probability = 0.1
    num_bootstrap = 500
    run_stationary_bootstrap_example(returns, probability, num_bootstrap)

    # Run async bootstrap example with appropriate parameters
    block_size = 20
    num_bootstrap = 500
    run_async_bootstrap_example(returns, block_size, num_bootstrap)

    # Run hypothesis testing example with appropriate parameters
    null_value = 0.0
    num_bootstrap = 500
    run_hypothesis_test_example(returns, null_value, num_bootstrap)

    # Run complex statistic (Sharpe ratio) example
    num_bootstrap = 500
    run_complex_statistic_example(returns, num_bootstrap)

    # Print summary and conclusion of bootstrap capabilities
    print("\nSummary:")
    print("This script demonstrated various bootstrap methods for financial time series analysis.")
    print("Bootstrap is a powerful tool for statistical inference when traditional assumptions are violated.")


if __name__ == '__main__':
    main()