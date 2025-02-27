"""
MFE Toolbox - Bootstrap Analysis Example

This script demonstrates bootstrap techniques for financial time series analysis.
It showcases different bootstrap methods (block, stationary, moving block), 
their performance characteristics, and applications for statistical inference.
"""

import numpy as np                # numpy 1.26.3
import pandas as pd               # pandas 2.1.4
import matplotlib.pyplot as plt   # matplotlib 3.8.2
import scipy.stats                # scipy 1.11.4
import time                       # Python 3.12
import asyncio                    # Python 3.12
import argparse                   # Python 3.12

# Import from MFE Toolbox
from ..mfe.core.bootstrap import Bootstrap, BootstrapResult
from ..mfe.utils.data_handling import load_financial_data, calculate_financial_returns
from ..mfe.utils.numpy_helpers import ensure_array

# Global constants
RANDOM_SEED = 42
FIGURE_SIZE = (12, 8)
DEFAULT_BOOTSTRAP_SIZE = 500


def generate_simulated_returns(n_samples=1000, mean=0.0001, std=0.01, seed=RANDOM_SEED):
    """
    Generates simulated financial return series with stylized facts for bootstrap demonstration.
    
    Parameters
    ----------
    n_samples : int
        Number of samples to generate
    mean : float
        Target mean of returns
    std : float
        Target standard deviation of returns
    seed : int
        Random seed for reproducibility
        
    Returns
    -------
    np.ndarray
        Simulated financial return series
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Generate baseline random normal returns
    returns = np.random.normal(0, 1, n_samples)
    
    # Add autocorrelation pattern to simulate persistence
    for i in range(1, n_samples):
        returns[i] += 0.05 * returns[i-1]
    
    # Add volatility clustering (GARCH-like effect)
    volatility = np.ones(n_samples)
    for i in range(1, n_samples):
        volatility[i] = 0.9 * volatility[i-1] + 0.1 * returns[i-1]**2
    volatility = np.sqrt(volatility)
    
    # Apply volatility to returns
    returns = returns * volatility
    
    # Scale to specified mean and standard deviation
    returns = (returns - np.mean(returns)) / np.std(returns) * std + mean
    
    return returns


def load_example_data(data_source='stock', use_simulated=False):
    """
    Loads real or simulated financial data for bootstrap analysis.
    
    Parameters
    ----------
    data_source : str
        Type of data to load ('stock', 'forex', 'crypto')
    use_simulated : bool
        If True, generate simulated data instead of loading real data
        
    Returns
    -------
    np.ndarray
        Financial return series
    """
    if use_simulated:
        print("Using simulated financial data...")
        return generate_simulated_returns(n_samples=2000)
    
    print(f"Loading {data_source} data...")
    try:
        # Load appropriate dataset based on data_source
        if data_source == 'stock':
            # Load sample stock data
            prices = load_financial_data(f"data/{data_source}_prices.csv")
        elif data_source == 'forex':
            # Load sample forex data
            prices = load_financial_data(f"data/{data_source}_prices.csv")
        elif data_source == 'crypto':
            # Load sample crypto data
            prices = load_financial_data(f"data/{data_source}_prices.csv")
        else:
            print(f"Unknown data source: {data_source}, using simulated data instead")
            return generate_simulated_returns(n_samples=2000)
        
        # Convert prices to returns
        returns = calculate_financial_returns(prices, method='simple')
        
        # Remove missing values and convert to numpy array
        returns = ensure_array(returns.dropna())
        
        # Winsorize to remove extreme outliers
        returns = np.clip(returns, 
                         np.percentile(returns, 0.5), 
                         np.percentile(returns, 99.5))
        
        return returns
    
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        print("Falling back to simulated data")
        return generate_simulated_returns(n_samples=2000)


def calculate_mean(x):
    """
    Calculates mean of a series, used as a statistic function for bootstrap.
    
    Parameters
    ----------
    x : np.ndarray
        Input array
        
    Returns
    -------
    float
        Mean of the input array
    """
    if not isinstance(x, np.ndarray):
        x = np.asarray(x)
    return np.mean(x)


def calculate_variance(x):
    """
    Calculates variance of a series, used as a statistic function for bootstrap.
    
    Parameters
    ----------
    x : np.ndarray
        Input array
        
    Returns
    -------
    float
        Variance of the input array
    """
    if not isinstance(x, np.ndarray):
        x = np.asarray(x)
    return np.var(x)


def calculate_quantile(x, q=0.95):
    """
    Calculates a specified quantile of a series, used as a statistic function for bootstrap.
    
    Parameters
    ----------
    x : np.ndarray
        Input array
    q : float
        Quantile to calculate (between 0 and 1)
        
    Returns
    -------
    float
        Specified quantile of the input array
    """
    if not isinstance(x, np.ndarray):
        x = np.asarray(x)
    if not 0 <= q <= 1:
        raise ValueError("Quantile must be between 0 and 1")
    return np.quantile(x, q)


def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    """
    Calculates Sharpe ratio, used as a complex statistic for bootstrap.
    
    Parameters
    ----------
    returns : np.ndarray
        Return series
    risk_free_rate : float
        Risk-free rate
        
    Returns
    -------
    float
        Sharpe ratio
    """
    if not isinstance(returns, np.ndarray):
        returns = np.asarray(returns)
    excess_return = np.mean(returns) - risk_free_rate
    std_dev = np.std(returns)
    if std_dev == 0:
        return 0
    return excess_return / std_dev


def demonstrate_block_bootstrap(returns, block_size=20, num_bootstrap=DEFAULT_BOOTSTRAP_SIZE):
    """
    Demonstrates block bootstrap method for time series data.
    
    Parameters
    ----------
    returns : np.ndarray
        Financial return series
    block_size : int
        Size of blocks for bootstrap
    num_bootstrap : int
        Number of bootstrap samples
        
    Returns
    -------
    BootstrapResult
        Bootstrap analysis results
    """
    print("\n" + "=" * 80)
    print("BLOCK BOOTSTRAP DEMONSTRATION")
    print("=" * 80)
    
    # Initialize Bootstrap object with block method
    bootstrap = Bootstrap(method='block', 
                         params={'block_size': block_size}, 
                         num_bootstrap=num_bootstrap)
    
    # Run bootstrap analysis for mean return
    print(f"\nPerforming block bootstrap with block_size={block_size}, samples={num_bootstrap}...")
    result = bootstrap.run(returns, calculate_mean)
    
    # Print summary
    summary = result.summary()
    print("\nBootstrap Summary:")
    print(f"Original Statistic (Mean Return): {summary['original_statistic']:.6f}")
    print(f"Bootstrap Mean: {summary['bootstrap_mean']:.6f}")
    print(f"Bootstrap Standard Error: {summary['standard_error']:.6f}")
    ci_lower, ci_upper = summary['confidence_interval']
    print(f"95% Confidence Interval: [{ci_lower:.6f}, {ci_upper:.6f}]")
    
    # Plot bootstrap distribution
    plt.figure(figsize=FIGURE_SIZE)
    result.plot_distribution(title="Block Bootstrap Distribution of Mean Return")
    plt.tight_layout()
    plt.show()
    
    return result


def demonstrate_stationary_bootstrap(returns, probability=0.1, num_bootstrap=DEFAULT_BOOTSTRAP_SIZE):
    """
    Demonstrates stationary bootstrap method with random block lengths.
    
    Parameters
    ----------
    returns : np.ndarray
        Financial return series
    probability : float
        Probability parameter for the geometric distribution
    num_bootstrap : int
        Number of bootstrap samples
        
    Returns
    -------
    BootstrapResult
        Bootstrap analysis results
    """
    print("\n" + "=" * 80)
    print("STATIONARY BOOTSTRAP DEMONSTRATION")
    print("=" * 80)
    
    # Initialize Bootstrap object with stationary method
    bootstrap = Bootstrap(method='stationary', 
                         params={'probability': probability}, 
                         num_bootstrap=num_bootstrap)
    
    # Run bootstrap analysis for mean return
    print(f"\nPerforming stationary bootstrap with probability={probability}, samples={num_bootstrap}...")
    result = bootstrap.run(returns, calculate_mean)
    
    # Print summary
    summary = result.summary()
    print("\nBootstrap Summary:")
    print(f"Original Statistic (Mean Return): {summary['original_statistic']:.6f}")
    print(f"Bootstrap Mean: {summary['bootstrap_mean']:.6f}")
    print(f"Bootstrap Standard Error: {summary['standard_error']:.6f}")
    ci_lower, ci_upper = summary['confidence_interval']
    print(f"95% Confidence Interval: [{ci_lower:.6f}, {ci_upper:.6f}]")
    
    # Plot bootstrap distribution
    plt.figure(figsize=FIGURE_SIZE)
    result.plot_distribution(title="Stationary Bootstrap Distribution of Mean Return")
    plt.tight_layout()
    plt.show()
    
    return result


def demonstrate_moving_block_bootstrap(returns, block_size=20, num_bootstrap=DEFAULT_BOOTSTRAP_SIZE):
    """
    Demonstrates moving block bootstrap method with overlapping blocks.
    
    Parameters
    ----------
    returns : np.ndarray
        Financial return series
    block_size : int
        Size of blocks for bootstrap
    num_bootstrap : int
        Number of bootstrap samples
        
    Returns
    -------
    BootstrapResult
        Bootstrap analysis results
    """
    print("\n" + "=" * 80)
    print("MOVING BLOCK BOOTSTRAP DEMONSTRATION")
    print("=" * 80)
    
    # Initialize Bootstrap object with moving method
    bootstrap = Bootstrap(method='moving', 
                         params={'block_size': block_size}, 
                         num_bootstrap=num_bootstrap)
    
    # Run bootstrap analysis for mean return
    print(f"\nPerforming moving block bootstrap with block_size={block_size}, samples={num_bootstrap}...")
    result = bootstrap.run(returns, calculate_mean)
    
    # Print summary
    summary = result.summary()
    print("\nBootstrap Summary:")
    print(f"Original Statistic (Mean Return): {summary['original_statistic']:.6f}")
    print(f"Bootstrap Mean: {summary['bootstrap_mean']:.6f}")
    print(f"Bootstrap Standard Error: {summary['standard_error']:.6f}")
    ci_lower, ci_upper = summary['confidence_interval']
    print(f"95% Confidence Interval: [{ci_lower:.6f}, {ci_upper:.6f}]")
    
    # Plot bootstrap distribution
    plt.figure(figsize=FIGURE_SIZE)
    result.plot_distribution(title="Moving Block Bootstrap Distribution of Mean Return")
    plt.tight_layout()
    plt.show()
    
    return result


def compare_bootstrap_methods(returns, block_size=20, probability=0.1, 
                             num_bootstrap=DEFAULT_BOOTSTRAP_SIZE, statistic_func=calculate_mean):
    """
    Compares different bootstrap methods using the same dataset and statistic.
    
    Parameters
    ----------
    returns : np.ndarray
        Financial return series
    block_size : int
        Size of blocks for block bootstrap methods
    probability : float
        Probability parameter for stationary bootstrap
    num_bootstrap : int
        Number of bootstrap samples
    statistic_func : callable
        Function to compute the statistic of interest
        
    Returns
    -------
    dict
        Dictionary of bootstrap results for each method
    """
    print("\n" + "=" * 80)
    print("COMPARISON OF BOOTSTRAP METHODS")
    print("=" * 80)
    
    results = {}
    
    # Initialize Bootstrap objects for each method
    bootstrap_block = Bootstrap(method='block', 
                               params={'block_size': block_size}, 
                               num_bootstrap=num_bootstrap)
    
    bootstrap_stationary = Bootstrap(method='stationary', 
                                    params={'probability': probability}, 
                                    num_bootstrap=num_bootstrap)
    
    bootstrap_moving = Bootstrap(method='moving', 
                                params={'block_size': block_size}, 
                                num_bootstrap=num_bootstrap)
    
    # Time and execute each method
    methods = [
        ('block', bootstrap_block),
        ('stationary', bootstrap_stationary),
        ('moving', bootstrap_moving)
    ]
    
    for name, bootstrap in methods:
        print(f"\nExecuting {name} bootstrap method...")
        start_time = time.time()
        result = bootstrap.run(returns, statistic_func)
        execution_time = time.time() - start_time
        
        # Store results with execution time
        results[name] = {
            'result': result,
            'execution_time': execution_time,
            'summary': result.summary()
        }
        
        print(f"Execution time: {execution_time:.4f} seconds")
        
    # Print comparison
    print("\nComparison of Bootstrap Methods:")
    print("-" * 80)
    print(f"{'Method':<15} {'Original':<10} {'Bootstrap Mean':<15} {'Std Error':<10} {'95% CI':<20} {'Time (s)':<10}")
    print("-" * 80)
    
    for name, res in results.items():
        summary = res['summary']
        ci_lower, ci_upper = summary['confidence_interval']
        print(f"{name:<15} {summary['original_statistic']:<10.6f} {summary['bootstrap_mean']:<15.6f} "
              f"{summary['standard_error']:<10.6f} [{ci_lower:<8.6f}, {ci_upper:<8.6f}] {res['execution_time']:<10.4f}")
    
    # Create comparison plot
    plt.figure(figsize=FIGURE_SIZE)
    
    for name, res in results.items():
        bootstrap_stats = res['result'].bootstrap_statistics
        plt.hist(bootstrap_stats, bins=30, alpha=0.5, label=name)
    
    plt.axvline(statistic_func(returns), color='k', linestyle='-', label='Original')
    plt.title("Comparison of Bootstrap Distributions")
    plt.xlabel("Statistic Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return results


def demonstrate_async_bootstrap(returns, block_size=20, num_bootstrap=DEFAULT_BOOTSTRAP_SIZE):
    """
    Demonstrates asynchronous bootstrap analysis for non-blocking execution.
    
    Parameters
    ----------
    returns : np.ndarray
        Financial return series
    block_size : int
        Size of blocks for bootstrap
    num_bootstrap : int
        Number of bootstrap samples
        
    Returns
    -------
    dict
        Asynchronous bootstrap results
    """
    print("\n" + "=" * 80)
    print("ASYNCHRONOUS BOOTSTRAP DEMONSTRATION")
    print("=" * 80)
    
    # Define progress callback function
    def progress_callback(progress):
        if int(progress) % 10 == 0:
            print(f"Progress: {progress:.1f}%")
    
    # Initialize Bootstrap object
    bootstrap = Bootstrap(method='block', 
                         params={'block_size': block_size}, 
                         num_bootstrap=num_bootstrap)
    
    # Asynchronous execution
    async def run_async_bootstrap():
        print(f"\nPerforming asynchronous block bootstrap with block_size={block_size}, samples={num_bootstrap}...")
        
        # Measure time
        start_time = time.time()
        
        # Run async bootstrap
        async_result = await bootstrap.run_async(returns, calculate_mean, progress_callback)
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        print(f"\nAsync execution completed in {execution_time:.4f} seconds")
        
        # Return result with time
        return {
            'result': async_result,
            'execution_time': execution_time
        }
    
    # Run synchronous version for comparison
    print("Running synchronous bootstrap for comparison...")
    sync_start = time.time()
    sync_result = bootstrap.run(returns, calculate_mean)
    sync_time = time.time() - sync_start
    print(f"Synchronous execution completed in {sync_time:.4f} seconds")
    
    # Run asynchronous version
    asyncio_result = asyncio.run(run_async_bootstrap())
    async_result = asyncio_result['result']
    async_time = asyncio_result['execution_time']
    
    # Compare results
    print("\nComparing Sync vs Async Results:")
    print(f"Original statistic: {sync_result.original_statistic:.6f}")
    print(f"Sync bootstrap mean: {np.mean(sync_result.bootstrap_statistics):.6f}")
    print(f"Async bootstrap mean: {np.mean(async_result.bootstrap_statistics):.6f}")
    print(f"Sync standard error: {sync_result.standard_error():.6f}")
    print(f"Async standard error: {async_result.standard_error():.6f}")
    
    # Plot comparison
    plt.figure(figsize=FIGURE_SIZE)
    
    plt.hist(sync_result.bootstrap_statistics, bins=30, alpha=0.5, label='Synchronous')
    plt.hist(async_result.bootstrap_statistics, bins=30, alpha=0.5, label='Asynchronous')
    
    plt.axvline(sync_result.original_statistic, color='k', linestyle='-', label='Original')
    plt.title("Comparison of Sync vs Async Bootstrap Distributions")
    plt.xlabel("Statistic Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Return comparison data
    return {
        'sync_result': sync_result,
        'sync_time': sync_time,
        'async_result': async_result,
        'async_time': async_time
    }


def demonstrate_confidence_intervals(returns, num_bootstrap=DEFAULT_BOOTSTRAP_SIZE):
    """
    Demonstrates calculation and interpretation of bootstrap confidence intervals.
    
    Parameters
    ----------
    returns : np.ndarray
        Financial return series
    num_bootstrap : int
        Number of bootstrap samples
        
    Returns
    -------
    dict
        Different confidence intervals with various methods
    """
    print("\n" + "=" * 80)
    print("BOOTSTRAP CONFIDENCE INTERVALS DEMONSTRATION")
    print("=" * 80)
    
    # Initialize Bootstrap object
    bootstrap = Bootstrap(method='block', 
                         params={'block_size': 20}, 
                         num_bootstrap=num_bootstrap)
    
    # Run bootstrap analysis
    print(f"\nPerforming bootstrap for confidence interval calculation...")
    result = bootstrap.run(returns, calculate_mean)
    
    # Calculate different types of confidence intervals
    confidence_levels = [0.90, 0.95, 0.99]
    methods = ['percentile', 'bca']
    
    ci_results = {}
    
    print("\nBootstrap Confidence Intervals:")
    print("-" * 80)
    print(f"{'Confidence Level':<20} {'Method':<15} {'Lower':<10} {'Upper':<10} {'Width':<10}")
    print("-" * 80)
    
    for level in confidence_levels:
        ci_results[level] = {}
        
        for method in methods:
            alpha = 1 - level
            ci_lower, ci_upper = result.confidence_interval(method=method, alpha=alpha)
            ci_width = ci_upper - ci_lower
            
            ci_results[level][method] = {
                'lower': ci_lower,
                'upper': ci_upper,
                'width': ci_width
            }
            
            print(f"{level*100:.1f}%{'':<14} {method:<15} {ci_lower:<10.6f} {ci_upper:<10.6f} {ci_width:<10.6f}")
    
    # Visualization of different confidence intervals
    plt.figure(figsize=FIGURE_SIZE)
    
    # Plot bootstrap distribution
    bootstrap_stats = result.bootstrap_statistics
    plt.hist(bootstrap_stats, bins=30, density=True, alpha=0.6, label='Bootstrap Distribution')
    
    # Plot confidence intervals
    y_pos = 0.2
    y_step = 0.05
    
    for level in confidence_levels:
        for method in methods:
            ci = ci_results[level][method]
            plt.plot([ci['lower'], ci['upper']], [y_pos, y_pos], 'o-', 
                    label=f"{level*100:.0f}% {method} CI")
            y_pos += y_step
    
    # Plot original statistic
    plt.axvline(result.original_statistic, color='k', linestyle='-', label='Original')
    
    plt.title("Bootstrap Confidence Intervals for Mean Return")
    plt.xlabel("Mean Return")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return ci_results


def demonstrate_hypothesis_testing(returns, null_value=0, num_bootstrap=DEFAULT_BOOTSTRAP_SIZE):
    """
    Demonstrates bootstrap-based hypothesis testing.
    
    Parameters
    ----------
    returns : np.ndarray
        Financial return series
    null_value : float
        Null hypothesis value for the mean
    num_bootstrap : int
        Number of bootstrap samples
        
    Returns
    -------
    dict
        Hypothesis test results
    """
    print("\n" + "=" * 80)
    print("BOOTSTRAP HYPOTHESIS TESTING DEMONSTRATION")
    print("=" * 80)
    
    # Initialize Bootstrap object
    bootstrap = Bootstrap(method='block', 
                         params={'block_size': 20}, 
                         num_bootstrap=num_bootstrap)
    
    # Print hypothesis information
    print(f"\nNull Hypothesis: Mean return = {null_value}")
    print(f"Actual mean return: {np.mean(returns):.6f}")
    
    # Perform hypothesis tests with different alternatives
    alternatives = ['two-sided', 'less', 'greater']
    test_results = {}
    
    print("\nBootstrap Hypothesis Test Results:")
    print("-" * 80)
    print(f"{'Alternative':<15} {'p-value':<10} {'Conclusion (alpha=0.05)':<30}")
    print("-" * 80)
    
    for alternative in alternatives:
        # Run bootstrap test
        result, p_value = bootstrap.hypothesis_test(
            returns, calculate_mean, null_value, alternative
        )
        
        # Store result
        test_results[alternative] = {
            'result': result,
            'p_value': p_value,
            'reject_null': p_value < 0.05
        }
        
        # Print result
        conclusion = "Reject H0" if p_value < 0.05 else "Fail to reject H0"
        print(f"{alternative:<15} {p_value:<10.6f} {conclusion:<30}")
    
    # Visualize test results
    plt.figure(figsize=FIGURE_SIZE)
    
    # Plot bootstrap distribution
    result = test_results['two-sided']['result']
    bootstrap_stats = result.bootstrap_statistics
    plt.hist(bootstrap_stats, bins=30, alpha=0.6, label='Bootstrap Distribution')
    
    # Plot null value and original statistic
    plt.axvline(null_value, color='r', linestyle='--', label=f'Null Value: {null_value}')
    plt.axvline(result.original_statistic, color='k', linestyle='-', 
               label=f'Original Value: {result.original_statistic:.6f}')
    
    # Add p-value annotation
    two_sided_p = test_results['two-sided']['p_value']
    plt.annotate(f'Two-sided p-value: {two_sided_p:.6f}', 
                xy=(0.05, 0.95), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))
    
    plt.title("Bootstrap Hypothesis Test for Mean Return")
    plt.xlabel("Mean Return")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return test_results


def demonstrate_complex_statistics(returns, num_bootstrap=DEFAULT_BOOTSTRAP_SIZE):
    """
    Demonstrates bootstrap analysis for complex financial statistics like Sharpe ratio.
    
    Parameters
    ----------
    returns : np.ndarray
        Financial return series
    num_bootstrap : int
        Number of bootstrap samples
        
    Returns
    -------
    dict
        Bootstrap results for complex statistics
    """
    print("\n" + "=" * 80)
    print("BOOTSTRAP ANALYSIS FOR COMPLEX STATISTICS")
    print("=" * 80)
    
    # Define complex statistics to analyze
    statistics = {
        'Sharpe Ratio': lambda x: calculate_sharpe_ratio(x),
        '95% VaR': lambda x: -1 * calculate_quantile(x, q=0.05),
        'Skewness': lambda x: scipy.stats.skew(x),
        'Kurtosis': lambda x: scipy.stats.kurtosis(x)
    }
    
    # Initialize Bootstrap object
    bootstrap = Bootstrap(method='block', 
                         params={'block_size': 20}, 
                         num_bootstrap=num_bootstrap)
    
    # Run bootstrap analysis for each statistic
    results = {}
    
    for name, stat_func in statistics.items():
        print(f"\nAnalyzing {name}...")
        
        # Calculate original statistic
        original_value = stat_func(returns)
        print(f"Original value: {original_value:.6f}")
        
        # Perform bootstrap
        bootstrap_result = bootstrap.run(returns, stat_func)
        
        # Calculate confidence interval
        ci_lower, ci_upper = bootstrap_result.confidence_interval()
        print(f"95% confidence interval: [{ci_lower:.6f}, {ci_upper:.6f}]")
        
        # Store results
        results[name] = {
            'original': original_value,
            'bootstrap_result': bootstrap_result,
            'ci': (ci_lower, ci_upper)
        }
        
        # Visualize
        plt.figure(figsize=FIGURE_SIZE)
        bootstrap_result.plot_distribution(title=f"Bootstrap Distribution of {name}")
        plt.tight_layout()
        plt.show()
    
    # Create summary table
    print("\nSummary of Complex Statistics Bootstrap Analysis:")
    print("-" * 80)
    print(f"{'Statistic':<15} {'Original':<10} {'Bootstrap Mean':<15} {'Std Error':<10} {'95% CI':<20}")
    print("-" * 80)
    
    for name, result in results.items():
        bootstrap_result = result['bootstrap_result']
        summary = bootstrap_result.summary()
        ci_lower, ci_upper = summary['confidence_interval']
        
        print(f"{name:<15} {summary['original_statistic']:<10.6f} {summary['bootstrap_mean']:<15.6f} "
              f"{summary['standard_error']:<10.6f} [{ci_lower:<8.6f}, {ci_upper:<8.6f}]")
    
    return results


def run_bootstrap_examples(use_async=False, data_source='stock', use_simulated=False):
    """
    Main function to run all bootstrap examples.
    
    Parameters
    ----------
    use_async : bool
        Whether to demonstrate asynchronous bootstrap
    data_source : str
        Type of data to use ('stock', 'forex', 'crypto')
    use_simulated : bool
        Whether to use simulated data instead of real data
    """
    print("=" * 80)
    print("MFE TOOLBOX - BOOTSTRAP ANALYSIS EXAMPLES")
    print("=" * 80)
    print("\nThis example demonstrates various bootstrap techniques for financial time series analysis.")
    print("It covers block bootstrap, stationary bootstrap, and moving block bootstrap methods.")
    print("We'll analyze confidence intervals, hypothesis tests, and complex statistics.")
    
    # Load example data
    returns = load_example_data(data_source, use_simulated)
    print(f"Loaded return series with {len(returns)} observations")
    print(f"Mean: {np.mean(returns):.6f}, Std Dev: {np.std(returns):.6f}")
    
    # Demonstrate block bootstrap
    block_result = demonstrate_block_bootstrap(returns)
    
    # Demonstrate stationary bootstrap
    stationary_result = demonstrate_stationary_bootstrap(returns)
    
    # Demonstrate moving block bootstrap
    moving_result = demonstrate_moving_block_bootstrap(returns)
    
    # Compare bootstrap methods
    comparison_results = compare_bootstrap_methods(returns)
    
    # Demonstrate async bootstrap if requested
    if use_async:
        async_results = demonstrate_async_bootstrap(returns)
    
    # Demonstrate confidence intervals
    ci_results = demonstrate_confidence_intervals(returns)
    
    # Demonstrate hypothesis testing
    hypothesis_results = demonstrate_hypothesis_testing(returns)
    
    # Demonstrate complex statistics
    complex_results = demonstrate_complex_statistics(returns)
    
    print("\n" + "=" * 80)
    print("BOOTSTRAP ANALYSIS EXAMPLES COMPLETED")
    print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MFE Toolbox Bootstrap Analysis Example')
    parser.add_argument('--async', dest='use_async', action='store_true', help='Demonstrate async bootstrap methods')
    parser.add_argument('--data', choices=['stock', 'forex', 'crypto'], default='stock', help='Type of data to use')
    parser.add_argument('--simulated', action='store_true', help='Use simulated data instead of real data')
    args = parser.parse_args()
    
    run_bootstrap_examples(args.use_async, args.data, args.simulated)