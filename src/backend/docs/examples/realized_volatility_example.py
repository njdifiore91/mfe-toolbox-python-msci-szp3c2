"""
MFE Toolbox - Realized Volatility Example

This example demonstrates how to calculate and analyze realized volatility measures
using the MFE Toolbox. It covers basic concepts of realized volatility estimation,
kernel-based methods, sampling schemes, noise filtering, and visualization of results.
"""

import numpy as np  # numpy 1.26.3
import pandas as pd  # pandas 2.1.4
import matplotlib.pyplot as plt  # matplotlib 3.8.2
import asyncio  # Python 3.12
import os  # Python 3.12
import time  # Python 3.12

# Import MFE Toolbox functions
from mfe.models.realized import (
    realized_variance, realized_kernel, realized_volatility,
    RealizedVolatility, preprocess_price_data
)
from mfe.utils.data_handling import (
    load_financial_data, convert_time_series, calculate_financial_returns
)

# Path to sample data
SAMPLE_DATA_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'tests', 'test_data', 'market_benchmark.npy')

# Sampling intervals to demonstrate
SAMPLING_INTERVALS = [60, 300, 600, 1800]

# Kernel types to demonstrate
KERNEL_TYPES = ['bartlett', 'parzen', 'QS', 'tukey-hanning', 'truncated']

# Path for saving figures
FIGURE_PATH = os.path.join(os.path.dirname(__file__), 'figures')


def generate_synthetic_data(n_points=1000, volatility=0.01, drift=0.0):
    """
    Generates synthetic high-frequency price data for demonstration purposes when real data is not available.
    
    Parameters
    ----------
    n_points : int
        Number of data points to generate
    volatility : float
        Annualized volatility parameter
    drift : float
        Annualized drift parameter
        
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Tuple containing price series and corresponding timestamps
    """
    # Generate timestamps spanning a trading day (6.5 hours)
    trading_seconds = 6.5 * 3600  # 6.5 hours in seconds
    # Create random timestamps within the trading day
    timestamps = np.sort(np.random.uniform(0, trading_seconds, n_points))
    
    # Convert to seconds since midnight for a simulated trading day
    seconds = timestamps + 9.5 * 3600  # Starting at 9:30 AM
    
    # Calculate time differences for scaling volatility
    dt = np.diff(timestamps, prepend=0) / (252 * trading_seconds)
    dt[0] = dt[1]  # Fix the first element
    
    # Generate price increments using Brownian motion with drift
    increments = np.random.normal(
        loc=drift * dt,
        scale=volatility * np.sqrt(dt),
        size=n_points
    )
    
    # Add some microstructure noise to simulate high-frequency effects
    noise = np.random.normal(0, 0.0001, n_points)
    increments = increments + noise
    
    # Cumulate to create a price path, starting at 100
    log_prices = 100 * np.exp(np.cumsum(increments))
    
    return log_prices, seconds


def load_example_data():
    """
    Loads sample high-frequency data from file or generates synthetic data if file doesn't exist.
    
    Returns
    -------
    tuple[np.ndarray, pd.Series]
        Tuple containing price data and timestamp series
    """
    try:
        # Try to load sample data file
        print("Loading sample high-frequency data...")
        data = load_financial_data(SAMPLE_DATA_PATH)
        
        # Extract prices and times from the loaded data
        if isinstance(data, pd.DataFrame):
            prices = data.iloc[:, 0].values
            times = data.index.values
        else:
            prices = data
            # Generate synthetic timestamps if not available
            times = np.linspace(0, len(prices) - 1, len(prices))
        
    except (FileNotFoundError, ValueError):
        # Generate synthetic data if file doesn't exist
        print("Sample data file not found. Generating synthetic high-frequency data...")
        prices, times = generate_synthetic_data(n_points=1000, volatility=0.015)
    
    # Convert to appropriate formats
    times_series = pd.Series(times)
    
    # Print some summary statistics
    print(f"Loaded {len(prices)} high-frequency observations")
    print(f"Price range: [{prices.min():.2f}, {prices.max():.2f}]")
    
    return prices, times_series


def example_basic_usage(prices, times):
    """
    Demonstrates basic usage of realized volatility measures with simple API calls.
    
    Parameters
    ----------
    prices : np.ndarray
        High-frequency price data
    times : pd.Series
        Corresponding timestamps
        
    Returns
    -------
    dict
        Dictionary of results from various methods
    """
    print("\n" + "="*80)
    print("BASIC USAGE OF REALIZED VOLATILITY FUNCTIONS")
    print("="*80)
    print("This example demonstrates the basic API calls for realized volatility measures,")
    print("showing how to compute realized variance, volatility, and kernel estimators.")
    
    # Initialize results dictionary
    results = {}
    
    # Convert time series for use with realized functions
    time_type = 'seconds'  # Assuming times are in seconds format
    
    # Compute realized variance with calendar time sampling
    print("\nComputing realized variance with calendar time sampling...")
    rv, rv_ss = realized_variance(
        prices, times, time_type=time_type,
        sampling_type='CalendarTime', sampling_interval=300  # 5-minute sampling
    )
    results['rv_calendar'] = rv
    
    # Compute realized volatility with business time sampling
    print("Computing realized volatility with business time sampling...")
    vol, vol_ss = realized_volatility(
        prices, times, time_type=time_type,
        sampling_type='BusinessTime', sampling_interval=50,  # Every 50 observations
        annualize=True  # Annualize the result
    )
    results['vol_business'] = vol
    
    # Compute realized kernel with QS kernel
    print("Computing realized kernel with QS kernel...")
    rk = realized_kernel(
        prices, times, time_type=time_type,
        kernel_type='qs',  # Quadratic Spectral kernel
    )
    results['kernel_QS'] = rk
    
    # Print comparison of results
    print("\nResults comparison:")
    print(f"Realized variance (Calendar time, 5-min):     {results['rv_calendar']:.6f}")
    print(f"Realized volatility (Business time, ann.):    {results['vol_business']:.6f}")
    print(f"Realized kernel (QS kernel):                 {np.sqrt(results['kernel_QS']):.6f}")
    
    return results


def example_sampling_schemes(prices, times):
    """
    Demonstrates the impact of different sampling schemes on realized volatility estimation.
    
    Parameters
    ----------
    prices : np.ndarray
        High-frequency price data
    times : pd.Series
        Corresponding timestamps
        
    Returns
    -------
    dict
        Dictionary of results from various sampling schemes
    """
    print("\n" + "="*80)
    print("IMPACT OF DIFFERENT SAMPLING SCHEMES")
    print("="*80)
    print("This example demonstrates how different sampling schemes affect realized volatility estimates,")
    print("comparing calendar time, business time, and fixed sampling approaches.")
    
    # Initialize results dictionary
    results = {}
    
    # Time type for the data
    time_type = 'seconds'
    
    # Test different sampling schemes
    schemes = [
        ('CalendarTime', 300),           # 5-minute calendar time sampling
        ('BusinessTime', (10, 50)),      # Business time with 10-50 observations per sample
        ('Fixed', 50)                    # Fixed sampling with 50 total samples
    ]
    
    for scheme_name, interval in schemes:
        print(f"\nComputing realized variance with {scheme_name} sampling...")
        rv, _ = realized_variance(
            prices, times, time_type=time_type,
            sampling_type=scheme_name, sampling_interval=interval
        )
        
        # Calculate annualized volatility (assuming daily data)
        vol = np.sqrt(rv) * np.sqrt(252)  # Annualize
        
        results[scheme_name] = vol
        print(f"Annualized volatility: {vol:.4f}")
    
    # Create a comparison plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(results.keys(), results.values())
    
    # Customize plot
    plt.title('Impact of Sampling Schemes on Realized Volatility Estimates', fontsize=14)
    plt.ylabel('Annualized Volatility', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{height:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    print("\nObservations:")
    print("- Different sampling schemes can lead to varying volatility estimates")
    print("- Calendar time sampling is commonly used for its economic interpretation")
    print("- Business time sampling can better capture market activity patterns")
    print("- The choice of sampling scheme should align with research questions and data characteristics")
    
    return results


def example_noise_filtering(prices, times):
    """
    Demonstrates the effect of noise filtering on realized volatility estimation.
    
    Parameters
    ----------
    prices : np.ndarray
        High-frequency price data
    times : pd.Series
        Corresponding timestamps
        
    Returns
    -------
    dict
        Dictionary comparing results with and without noise filtering
    """
    print("\n" + "="*80)
    print("EFFECT OF NOISE FILTERING")
    print("="*80)
    print("This example shows how noise filtering affects realized volatility estimates,")
    print("demonstrating its importance in high-frequency financial data analysis.")
    
    # Initialize results dictionary
    results = {}
    
    # Time type for the data
    time_type = 'seconds'
    
    # Preprocess price data for analysis (with and without outlier detection)
    clean_prices, clean_times = preprocess_price_data(
        prices, times, time_type, detect_outliers=True, threshold=3.0
    )
    
    # Common parameters for both calculations
    sampling_type = 'CalendarTime'
    sampling_interval = 300  # 5-minute sampling
    
    # Compute realized volatility WITHOUT noise filtering
    print("\nComputing realized volatility WITHOUT noise filtering...")
    vol_no_filter, _ = realized_volatility(
        clean_prices, clean_times, time_type=time_type,
        sampling_type=sampling_type, sampling_interval=sampling_interval,
        noise_adjust=False, annualize=True
    )
    results['no_filter'] = vol_no_filter
    
    # Compute realized volatility WITH noise filtering
    print("Computing realized volatility WITH noise filtering...")
    vol_with_filter, _ = realized_volatility(
        clean_prices, clean_times, time_type=time_type,
        sampling_type=sampling_type, sampling_interval=sampling_interval,
        noise_adjust=True, annualize=True
    )
    results['with_filter'] = vol_with_filter
    
    # Calculate percentage difference
    pct_diff = ((vol_with_filter - vol_no_filter) / vol_no_filter) * 100
    results['pct_diff'] = pct_diff
    
    print("\nResults comparison:")
    print(f"Realized volatility WITHOUT noise filtering: {vol_no_filter:.6f}")
    print(f"Realized volatility WITH noise filtering:    {vol_with_filter:.6f}")
    print(f"Percentage difference:                      {pct_diff:.2f}%")
    
    # Create a comparison plot
    labels = ['Without Filtering', 'With Filtering']
    values = [vol_no_filter, vol_with_filter]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, values, color=['#3498db', '#2ecc71'])
    
    # Customize plot
    plt.title('Effect of Noise Filtering on Realized Volatility Estimates', fontsize=14)
    plt.ylabel('Annualized Volatility', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{height:.4f}', ha='center', va='bottom', fontsize=10)
    
    # Add arrow showing percentage difference
    plt.annotate(f'{pct_diff:.2f}% difference',
                xy=(1, vol_with_filter),
                xytext=(0.5, (vol_no_filter + vol_with_filter)/2),
                arrowprops=dict(facecolor='red', shrink=0.05, width=1.5, headwidth=8),
                ha='center', va='center', fontsize=11,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", lw=1))
    
    plt.tight_layout()
    plt.show()
    
    print("\nObservations:")
    print("- Market microstructure noise can significantly impact realized volatility estimates")
    print("- Noise filtering helps obtain more accurate measures by removing high-frequency artifacts")
    print("- The difference between filtered and unfiltered estimates can be substantial")
    print("- Noise filtering is especially important for very high-frequency data")
    
    return results


def example_kernel_methods(prices, times):
    """
    Demonstrates kernel-based realized volatility estimation with different kernel functions.
    
    Parameters
    ----------
    prices : np.ndarray
        High-frequency price data
    times : pd.Series
        Corresponding timestamps
        
    Returns
    -------
    dict
        Dictionary of results from various kernel methods
    """
    print("\n" + "="*80)
    print("KERNEL-BASED REALIZED VOLATILITY ESTIMATION")
    print("="*80)
    print("This example demonstrates kernel-based estimation methods for realized volatility,")
    print("comparing different kernel functions and their impact on volatility estimates.")
    
    # Initialize results dictionary
    results = {}
    
    # Time type for the data
    time_type = 'seconds'
    
    # Test different kernel types
    for kernel_type in KERNEL_TYPES:
        print(f"\nComputing realized kernel with {kernel_type} kernel...")
        
        # Compute realized kernel
        rk = realized_kernel(
            prices, times, time_type=time_type,
            kernel_type=kernel_type.lower()  # Ensure lowercase for function
        )
        
        # Calculate annualized volatility
        vol = np.sqrt(rk) * np.sqrt(252)  # Annualize
        
        results[kernel_type] = vol
        print(f"Annualized volatility: {vol:.4f}")
    
    # Create a comparison plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(results.keys(), results.values())
    
    # Customize plot
    plt.title('Comparison of Kernel-Based Realized Volatility Estimators', fontsize=14)
    plt.ylabel('Annualized Volatility', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{height:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    print("\nObservations:")
    print("- Different kernel functions produce varying volatility estimates")
    print("- Kernel-based estimation helps address market microstructure noise and sampling issues")
    print("- The Bartlett kernel is often used as a standard approach")
    print("- Parzen and QS kernels can provide more robust estimates in some cases")
    print("- The choice of kernel should consider data characteristics and research objectives")
    
    return results


def example_sampling_frequency(prices, times):
    """
    Demonstrates the effect of sampling frequency on realized volatility estimates (signature plot).
    
    Parameters
    ----------
    prices : np.ndarray
        High-frequency price data
    times : pd.Series
        Corresponding timestamps
        
    Returns
    -------
    dict
        Dictionary mapping sampling intervals to volatility estimates
    """
    print("\n" + "="*80)
    print("EFFECT OF SAMPLING FREQUENCY (SIGNATURE PLOT)")
    print("="*80)
    print("This example creates a signature plot to visualize how sampling frequency")
    print("affects realized volatility estimates, highlighting microstructure noise effects.")
    
    # Initialize results dictionary
    results = {}
    
    # Time type for the data
    time_type = 'seconds'
    
    # Compute realized volatility for different sampling intervals
    for interval in SAMPLING_INTERVALS:
        print(f"\nComputing realized volatility with {interval}-second sampling...")
        
        vol, _ = realized_volatility(
            prices, times, time_type=time_type,
            sampling_type='CalendarTime', sampling_interval=interval,
            noise_adjust=False, annualize=True
        )
        
        results[interval] = vol
        print(f"Annualized volatility: {vol:.4f}")
    
    # Create signature plot
    intervals = list(results.keys())
    vols = list(results.values())
    
    plt.figure(figsize=(12, 6))
    plt.plot(intervals, vols, 'o-', linewidth=2, markersize=8)
    
    # Customize plot
    plt.title('Signature Plot: Realized Volatility vs. Sampling Frequency', fontsize=14)
    plt.xlabel('Sampling Interval (seconds)', fontsize=12)
    plt.ylabel('Annualized Volatility', fontsize=12)
    plt.grid(linestyle='--', alpha=0.7)
    
    # Add data labels
    for i, (interval, vol) in enumerate(zip(intervals, vols)):
        plt.annotate(f'{vol:.4f}', 
                    xy=(interval, vol),
                    xytext=(0, 10),
                    textcoords='offset points',
                    ha='center', va='bottom')
    
    # Add explanation
    plt.figtext(0.5, 0.01, 
               'Note: The signature plot shows how realized volatility estimates vary with sampling frequency.\n'
               'Higher volatility at very high frequencies often indicates market microstructure noise.',
               ha='center', fontsize=10, bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow", ec="orange", lw=1))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.show()
    
    print("\nObservations:")
    print("- Realized volatility estimates typically increase at higher sampling frequencies")
    print("- This pattern, known as the 'signature plot', reveals market microstructure noise")
    print("- The bias-variance tradeoff: higher frequency means more data (less variance) but more noise (more bias)")
    print("- The optimal sampling frequency balances these competing effects")
    print("- 5-minute (300-second) sampling is a common choice in the literature")
    
    return results


def example_class_interface(prices, times):
    """
    Demonstrates using the RealizedVolatility class for comprehensive analysis.
    
    Parameters
    ----------
    prices : np.ndarray
        High-frequency price data
    times : pd.Series
        Corresponding timestamps
        
    Returns
    -------
    dict
        Dictionary of results from class-based analysis
    """
    print("\n" + "="*80)
    print("USING THE REALIZEDVOLATILITY CLASS")
    print("="*80)
    print("This example demonstrates the RealizedVolatility class interface,")
    print("which provides a comprehensive and convenient API for realized measures.")
    
    # Initialize RealizedVolatility object
    print("\nInitializing RealizedVolatility object...")
    rv_analyzer = RealizedVolatility()
    
    # Set data
    time_type = 'seconds'
    rv_analyzer.set_data(prices, times, time_type)
    
    # Configure parameters
    print("Setting parameters for analysis...")
    rv_analyzer.set_params({
        'sampling_type': 'CalendarTime',
        'sampling_interval': 300,  # 5-minute sampling
        'noise_adjust': True,
        'kernel_type': 'bartlett',
        'detect_outliers': True,
        'annualize': True,
        'scale': 252  # Annualization factor
    })
    
    # Compute multiple measures at once
    print("Computing multiple realized measures...")
    measures_to_compute = ['variance', 'volatility', 'kernel']
    results = rv_analyzer.compute(measures_to_compute)
    
    # Display results
    print("\nResults from RealizedVolatility class:")
    print(f"Realized variance:    {results.get('variance', 'N/A'):.6f}")
    print(f"Realized volatility:  {results.get('volatility', 'N/A'):.6f}")
    print(f"Realized kernel:      {np.sqrt(results.get('kernel', 0)):.6f} (volatility)")
    
    # Demonstrate parameter adjustment and recomputation
    print("\nAdjusting parameters and recomputing...")
    rv_analyzer.set_params({
        'sampling_interval': 600,  # 10-minute sampling
        'noise_adjust': False
    })
    
    # Recompute with new parameters
    new_results = rv_analyzer.compute(['volatility'])
    
    print(f"Realized volatility (10-min, no filtering): {new_results.get('volatility', 'N/A'):.6f}")
    
    # Get all current results
    all_results = rv_analyzer.get_results()
    
    return all_results


async def async_compute_measures(prices, times, measures, params):
    """
    Asynchronous function to compute multiple realized measures concurrently.
    
    Parameters
    ----------
    prices : np.ndarray
        High-frequency price data
    times : pd.Series
        Corresponding timestamps
    measures : list
        List of measures to compute
    params : dict
        Parameters for computation
        
    Returns
    -------
    dict
        Dictionary of computed realized measures
    """
    # Initialize RealizedVolatility object
    rv_analyzer = RealizedVolatility()
    
    # Set data
    rv_analyzer.set_data(prices, times, 'seconds')
    
    # Set parameters
    rv_analyzer.set_params(params)
    
    # Compute measures asynchronously
    results = await rv_analyzer.compute_async(measures)
    
    return results


def example_async_computation(prices, times):
    """
    Demonstrates asynchronous computation of multiple realized measures.
    
    Parameters
    ----------
    prices : np.ndarray
        High-frequency price data
    times : pd.Series
        Corresponding timestamps
        
    Returns
    -------
    dict
        Dictionary of results from asynchronous computation
    """
    print("\n" + "="*80)
    print("ASYNCHRONOUS COMPUTATION")
    print("="*80)
    print("This example demonstrates asynchronous computation of realized measures")
    print("using Python's async/await pattern for improved performance.")
    
    # Define parameters for computation
    measures = ['variance', 'volatility', 'kernel']
    params = {
        'sampling_type': 'CalendarTime',
        'sampling_interval': 300,
        'noise_adjust': True,
        'kernel_type': 'bartlett',
        'detect_outliers': True,
        'annualize': True
    }
    
    # Sequential (synchronous) computation for comparison
    print("\nPerforming synchronous computation for comparison...")
    rv_analyzer_sync = RealizedVolatility()
    rv_analyzer_sync.set_data(prices, times, 'seconds')
    rv_analyzer_sync.set_params(params)
    
    start_time = time.time()
    sync_results = rv_analyzer_sync.compute(measures)
    sync_time = time.time() - start_time
    
    print(f"Synchronous computation time: {sync_time:.4f} seconds")
    
    # Asynchronous computation
    print("\nPerforming asynchronous computation...")
    start_time = time.time()
    async_results = asyncio.run(async_compute_measures(prices, times, measures, params))
    async_time = time.time() - start_time
    
    print(f"Asynchronous computation time: {async_time:.4f} seconds")
    print(f"Performance improvement: {(sync_time - async_time) / sync_time * 100:.2f}%")
    
    # Compare results
    print("\nResults comparison:")
    print(f"Sync realized volatility:  {sync_results.get('volatility', 'N/A'):.6f}")
    print(f"Async realized volatility: {async_results.get('volatility', 'N/A'):.6f}")
    
    print("\nObservations:")
    print("- Asynchronous computation can improve performance for multiple measures")
    print("- The async/await pattern allows non-blocking execution")
    print("- For simple cases, the overhead might outweigh the benefits")
    print("- For complex analyses or multiple assets, async can provide significant gains")
    
    return async_results


def create_volatility_plot(results, plot_type, title, save_fig=False):
    """
    Creates visualization of realized volatility results.
    
    Parameters
    ----------
    results : dict
        Dictionary containing results to plot
    plot_type : str
        Type of plot to create
    title : str
        Plot title
    save_fig : bool
        Whether to save the figure to a file
    
    Returns
    -------
    None
        Displays or saves the plot
    """
    plt.figure(figsize=(12, 6))
    
    if plot_type == 'bar':
        bars = plt.bar(results.keys(), results.values())
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=10)
    
    elif plot_type == 'line':
        plt.plot(list(results.keys()), list(results.values()), 'o-', 
                linewidth=2, markersize=8)
        
        # Add data labels
        for x, y in zip(results.keys(), results.values()):
            plt.annotate(f'{y:.4f}', 
                        xy=(x, y),
                        xytext=(0, 10),
                        textcoords='offset points',
                        ha='center', va='bottom')
    
    # Customize plot
    plt.title(title, fontsize=14)
    plt.ylabel('Annualized Volatility', fontsize=12)
    plt.grid(linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save figure if requested
    if save_fig:
        if not os.path.exists(FIGURE_PATH):
            os.makedirs(FIGURE_PATH)
        
        filename = title.lower().replace(' ', '_') + '.png'
        filepath = os.path.join(FIGURE_PATH, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {filepath}")
    else:
        plt.show()


def main():
    """
    Main function to run all examples sequentially.
    
    Returns
    -------
    None
    """
    print("\n" + "="*80)
    print("MFE TOOLBOX - REALIZED VOLATILITY EXAMPLE")
    print("="*80)
    print("This example demonstrates various techniques for calculating and analyzing")
    print("realized volatility using high-frequency financial data.")
    print("\nKey concepts covered:")
    print("1. Basic realized volatility estimation")
    print("2. Different sampling schemes (calendar time, business time)")
    print("3. Noise filtering techniques")
    print("4. Kernel-based estimation methods")
    print("5. Impact of sampling frequency (signature plot)")
    print("6. Comprehensive class-based interface")
    print("7. Asynchronous computation with async/await patterns")
    
    # Load or generate example data
    prices, times = load_example_data()
    
    # Run examples
    example_basic_usage(prices, times)
    example_sampling_schemes(prices, times)
    example_noise_filtering(prices, times)
    example_kernel_methods(prices, times)
    example_sampling_frequency(prices, times)
    example_class_interface(prices, times)
    example_async_computation(prices, times)
    
    print("\n" + "="*80)
    print("REALIZED VOLATILITY EXAMPLE COMPLETE")
    print("="*80)
    print("This example demonstrated various techniques for calculating and analyzing")
    print("realized volatility using high-frequency financial data.")
    print("\nKey takeaways:")
    print("1. Realized volatility provides a powerful non-parametric way to measure volatility")
    print("2. The choice of sampling scheme and frequency significantly impacts estimates")
    print("3. Noise filtering is essential for high-frequency data analysis")
    print("4. Kernel-based methods can provide more robust estimates")
    print("5. The MFE Toolbox provides comprehensive tools for realized volatility analysis")
    print("6. Asynchronous computation can improve performance for complex analyses")


if __name__ == '__main__':
    main()