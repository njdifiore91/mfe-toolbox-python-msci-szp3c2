"""
MFE Toolbox - Realized Measures Example

This example demonstrates how to use the realized volatility measures and
high-frequency data analysis capabilities of the MFE Toolbox.

The example covers:
1. Computing realized variance with different sampling methods
2. Kernel-based realized volatility estimation
3. Effects of sampling frequency on volatility estimates
4. Noise filtering in high-frequency data
5. Using the HighFrequencyData class for integrated analysis
6. Asynchronous computation of realized measures
"""

import os
import time
import numpy as np  # numpy 1.26.3
import pandas as pd  # pandas 2.1.4
import matplotlib.pyplot as plt  # matplotlib 3.8.2
import asyncio  # Python 3.12

from mfe.models.realized import (
    realized_variance,
    realized_kernel,
    realized_volatility,
    RealizedVolatility,
    preprocess_price_data
)
from mfe.models.high_frequency import HighFrequencyData
from mfe.utils.data_handling import load_financial_data, convert_time_series

# Define paths and constants
SAMPLE_DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'high_frequency_sample.csv')
SAMPLING_INTERVALS = [60, 300, 600, 1800]  # Sampling intervals in seconds
KERNEL_TYPES = ['Bartlett', 'Parzen', 'QS', 'Tukey-Hanning', 'Truncated']


def generate_sample_data(n_points: int, volatility: float, drift: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates synthetic high-frequency price data for demonstration.
    
    Parameters
    ----------
    n_points : int
        Number of data points to generate
    volatility : float
        Annualized volatility for price process
    drift : float
        Annualized drift for price process
        
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Arrays of prices and timestamps
    """
    # Generate timestamps at random intervals (mimicking real trading data)
    # Start at 9:30 AM (market open) and end at 4:00 PM (market close)
    open_time = 9.5 * 3600  # 9:30 AM in seconds
    close_time = 16 * 3600  # 4:00 PM in seconds
    
    # Timestamps will be in seconds since midnight
    timestamps = np.sort(np.random.uniform(open_time, close_time, n_points))
    
    # Convert volatility and drift to per-second
    seconds_per_year = 252 * 6.5 * 3600  # 252 trading days, 6.5 hours per day
    vol_per_second = volatility / np.sqrt(seconds_per_year)
    drift_per_second = drift / seconds_per_year
    
    # Generate price path
    price = 100.0  # Starting price
    prices = np.zeros(n_points)
    prices[0] = price
    
    for i in range(1, n_points):
        dt = timestamps[i] - timestamps[i-1]  # Time increment in seconds
        
        # Random price increment based on geometric Brownian motion
        price *= np.exp((drift_per_second - 0.5 * vol_per_second**2) * dt + 
                         vol_per_second * np.sqrt(dt) * np.random.standard_normal())
        
        # Add microstructure noise to mimic high-frequency market data
        noise = np.random.normal(0, 0.0001 * price)
        prices[i] = price + noise
    
    return prices, timestamps


def load_sample_data() -> tuple[np.ndarray, np.ndarray]:
    """
    Loads sample high-frequency data from file or generates synthetic data if file doesn't exist.
    
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Arrays of prices and timestamps
    """
    try:
        if os.path.exists(SAMPLE_DATA_PATH):
            print(f"Loading sample data from {SAMPLE_DATA_PATH}")
            data = load_financial_data(SAMPLE_DATA_PATH, date_column='time')
            prices = data['price'].values
            times = data.index.values
        else:
            print("Sample data file not found. Generating synthetic data...")
            prices, times = generate_sample_data(n_points=5000, volatility=0.2, drift=0.05)
            
            # Optionally save the generated data
            # df = pd.DataFrame({'price': prices, 'time': times})
            # os.makedirs(os.path.dirname(SAMPLE_DATA_PATH), exist_ok=True)
            # df.to_csv(SAMPLE_DATA_PATH, index=False)
        
        # Display basic info
        print(f"Loaded {len(prices)} price observations")
        print(f"Price range: [{np.min(prices):.2f}, {np.max(prices):.2f}]")
        
        return prices, times
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        print("Falling back to synthetic data generation")
        return generate_sample_data(n_points=5000, volatility=0.2, drift=0.05)


def demonstrate_realized_variance(prices: np.ndarray, times: np.ndarray) -> dict:
    """
    Demonstrates computation of realized variance with different sampling methods.
    
    Parameters
    ----------
    prices : np.ndarray
        High-frequency price series
    times : np.ndarray
        Timestamps corresponding to each price observation
        
    Returns
    -------
    dict
        Dictionary of results from various methods
    """
    print("\n" + "="*80)
    print("DEMONSTRATION 1: Realized Variance with Different Sampling Methods")
    print("="*80)
    
    results = {}
    
    # Calendar time sampling (regular time intervals)
    print("\nCalendar Time Sampling:")
    rv_calendar, rv_ss_calendar = realized_variance(
        prices, times, time_type='seconds', sampling_type='CalendarTime',
        sampling_interval=300  # 5-minute intervals
    )
    print(f"  5-minute realized variance: {rv_calendar:.8f}")
    print(f"  Subsampled realized variance: {rv_ss_calendar:.8f}")
    results['calendar'] = {'rv': rv_calendar, 'rv_ss': rv_ss_calendar}
    
    # Business time sampling (based on number of observations)
    print("\nBusiness Time Sampling:")
    rv_business, rv_ss_business = realized_variance(
        prices, times, time_type='seconds', sampling_type='BusinessTime',
        sampling_interval=(100, 200)  # Random between 100-200 observations
    )
    print(f"  Business time realized variance: {rv_business:.8f}")
    print(f"  Subsampled realized variance: {rv_ss_business:.8f}")
    results['business'] = {'rv': rv_business, 'rv_ss': rv_ss_business}
    
    # Fixed sampling (equal number of observations)
    print("\nFixed Sampling:")
    rv_fixed, rv_ss_fixed = realized_variance(
        prices, times, time_type='seconds', sampling_type='Fixed',
        sampling_interval=50  # 50 equal-spaced observations
    )
    print(f"  Fixed sampling realized variance: {rv_fixed:.8f}")
    print(f"  Subsampled realized variance: {rv_ss_fixed:.8f}")
    results['fixed'] = {'rv': rv_fixed, 'rv_ss': rv_ss_fixed}
    
    # Compare results in a table
    print("\nComparison of Realized Variance Methods:")
    print("-" * 60)
    print(f"{'Method':<20} {'Realized Variance':<20} {'Subsampled':<20}")
    print("-" * 60)
    print(f"{'Calendar Time':<20} {rv_calendar:<20.8f} {rv_ss_calendar:<20.8f}")
    print(f"{'Business Time':<20} {rv_business:<20.8f} {rv_ss_business:<20.8f}")
    print(f"{'Fixed':<20} {rv_fixed:<20.8f} {rv_ss_fixed:<20.8f}")
    
    return results


def demonstrate_kernel_estimation(prices: np.ndarray, times: np.ndarray) -> dict:
    """
    Demonstrates kernel-based realized volatility estimation.
    
    Parameters
    ----------
    prices : np.ndarray
        High-frequency price series
    times : np.ndarray
        Timestamps corresponding to each price observation
        
    Returns
    -------
    dict
        Dictionary of results from various kernel types
    """
    print("\n" + "="*80)
    print("DEMONSTRATION 2: Kernel-Based Realized Volatility Estimation")
    print("="*80)
    
    results = {}
    
    print("\nComparing Different Kernel Estimators:")
    print("-" * 60)
    print(f"{'Kernel Type':<15} {'Bandwidth':<10} {'Realized Kernel':<20}")
    print("-" * 60)
    
    # Compare different kernel types
    for kernel_type in KERNEL_TYPES:
        # Compute kernel with automatic bandwidth selection
        rk = realized_kernel(
            prices, times, time_type='seconds', kernel_type=kernel_type.lower()
        )
        
        # Store result
        results[kernel_type] = rk
        
        # Compute implied volatility (annualized)
        vol = np.sqrt(rk) * np.sqrt(252)  # Annualized volatility
        
        print(f"{kernel_type:<15} {'Auto':<10} {rk:<20.8f} (Vol: {vol:.4f})")
    
    # Create a bar plot comparing kernel estimates
    plt.figure(figsize=(10, 6))
    plt.bar(results.keys(), [np.sqrt(val) * np.sqrt(252) for val in results.values()])
    plt.title('Annualized Volatility by Kernel Type')
    plt.ylabel('Annualized Volatility')
    plt.xlabel('Kernel Type')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    return results


def demonstrate_sampling_effects(prices: np.ndarray, times: np.ndarray) -> dict:
    """
    Demonstrates effects of different sampling frequencies on realized volatility.
    
    Parameters
    ----------
    prices : np.ndarray
        High-frequency price series
    times : np.ndarray
        Timestamps corresponding to each price observation
        
    Returns
    -------
    dict
        Dictionary of results from various sampling intervals
    """
    print("\n" + "="*80)
    print("DEMONSTRATION 3: Effects of Sampling Frequency on Volatility Estimates")
    print("="*80)
    
    results = {}
    
    print("\nComparing Different Sampling Frequencies:")
    print("-" * 70)
    print(f"{'Sampling Interval (s)':<20} {'Realized Variance':<20} {'Annualized Vol (%)':<20}")
    print("-" * 70)
    
    # Iterate through different sampling intervals
    for interval in SAMPLING_INTERVALS:
        # Compute realized variance with calendar time sampling
        rv, _ = realized_variance(
            prices, times, time_type='seconds', sampling_type='CalendarTime',
            sampling_interval=interval
        )
        
        # Store result
        results[interval] = rv
        
        # Compute annualized volatility
        vol_annual = np.sqrt(rv) * np.sqrt(252) * 100  # Convert to percentage
        
        print(f"{interval:<20} {rv:<20.8f} {vol_annual:<20.2f}")
    
    # Create signature plot (sampling interval vs. volatility)
    plt.figure(figsize=(10, 6))
    x = np.array(list(results.keys()))
    y = np.array([np.sqrt(val) * np.sqrt(252) * 100 for val in results.values()])
    
    plt.semilogx(x, y, 'o-', linewidth=2, markersize=8)
    plt.title('Signature Plot: Effect of Sampling Frequency on Volatility')
    plt.xlabel('Sampling Interval (seconds, log scale)')
    plt.ylabel('Annualized Volatility (%)')
    plt.grid(True, alpha=0.3)
    
    # Add annotations
    for i, (interval, rv) in enumerate(results.items()):
        vol = np.sqrt(rv) * np.sqrt(252) * 100
        plt.annotate(f"{interval}s: {vol:.2f}%", 
                     xy=(interval, vol), 
                     xytext=(5, 5),
                     textcoords='offset points')
    
    plt.tight_layout()
    plt.show()
    
    # Explain the signature plot
    print("\nSignature Plot Interpretation:")
    print("-" * 70)
    print("The signature plot shows how volatility estimates change with sampling frequency.")
    print("At very high frequencies (small intervals), microstructure noise often inflates")
    print("volatility estimates. As sampling frequency decreases, estimates stabilize.")
    print("The optimal sampling frequency is often at the 'elbow' of this curve.")
    
    return results


def demonstrate_noise_filtering(prices: np.ndarray, times: np.ndarray) -> dict:
    """
    Demonstrates effects of noise filtering on realized volatility estimation.
    
    Parameters
    ----------
    prices : np.ndarray
        High-frequency price series
    times : np.ndarray
        Timestamps corresponding to each price observation
        
    Returns
    -------
    dict
        Dictionary of results with and without noise filtering
    """
    print("\n" + "="*80)
    print("DEMONSTRATION 4: Effects of Noise Filtering on Volatility Estimates")
    print("="*80)
    
    results = {}
    
    # Compute realized variance without noise filtering
    rv_raw, _ = realized_variance(
        prices, times, time_type='seconds', sampling_type='CalendarTime',
        sampling_interval=60,  # 1-minute intervals
        noise_adjust=False
    )
    
    vol_raw = np.sqrt(rv_raw) * np.sqrt(252) * 100  # Annualized, percentage
    results['raw'] = {'rv': rv_raw, 'vol': vol_raw}
    
    # Compute realized variance with noise filtering
    rv_filtered, _ = realized_variance(
        prices, times, time_type='seconds', sampling_type='CalendarTime',
        sampling_interval=60,  # 1-minute intervals
        noise_adjust=True
    )
    
    vol_filtered = np.sqrt(rv_filtered) * np.sqrt(252) * 100  # Annualized, percentage
    results['filtered'] = {'rv': rv_filtered, 'vol': vol_filtered}
    
    print("\nComparison of Volatility Estimates With and Without Noise Filtering:")
    print("-" * 70)
    print(f"{'Method':<20} {'Realized Variance':<20} {'Annualized Vol (%)':<20}")
    print("-" * 70)
    print(f"{'Raw (No Filtering)':<20} {rv_raw:<20.8f} {vol_raw:<20.2f}")
    print(f"{'With Noise Filtering':<20} {rv_filtered:<20.8f} {vol_filtered:<20.2f}")
    
    # Calculate percentage difference
    pct_diff = abs(vol_filtered - vol_raw) / vol_raw * 100
    print(f"\nPercentage Difference: {pct_diff:.2f}%")
    
    # Demonstration for different sampling intervals with and without filtering
    sampling_intervals = [30, 60, 120, 300, 600]
    filtered_results = []
    raw_results = []
    
    print("\nEffect of Noise Filtering Across Sampling Frequencies:")
    print("-" * 80)
    print(f"{'Interval (s)':<12} {'Raw RV':<15} {'Filtered RV':<15} {'Raw Vol (%)':<15} {'Filtered Vol (%)':<15} {'% Diff':<10}")
    print("-" * 80)
    
    for interval in sampling_intervals:
        # Without noise filtering
        rv_raw, _ = realized_variance(
            prices, times, time_type='seconds', sampling_type='CalendarTime',
            sampling_interval=interval, noise_adjust=False
        )
        vol_raw = np.sqrt(rv_raw) * np.sqrt(252) * 100
        raw_results.append(vol_raw)
        
        # With noise filtering
        rv_filtered, _ = realized_variance(
            prices, times, time_type='seconds', sampling_type='CalendarTime',
            sampling_interval=interval, noise_adjust=True
        )
        vol_filtered = np.sqrt(rv_filtered) * np.sqrt(252) * 100
        filtered_results.append(vol_filtered)
        
        # Calculate percentage difference
        pct_diff = abs(vol_filtered - vol_raw) / vol_raw * 100
        
        print(f"{interval:<12} {rv_raw:<15.8f} {rv_filtered:<15.8f} {vol_raw:<15.2f} {vol_filtered:<15.2f} {pct_diff:<10.2f}")
    
    # Plot comparative results
    plt.figure(figsize=(10, 6))
    plt.plot(sampling_intervals, raw_results, 'o-', label='Without Noise Filtering')
    plt.plot(sampling_intervals, filtered_results, 's-', label='With Noise Filtering')
    plt.title('Effect of Noise Filtering on Volatility Estimates')
    plt.xlabel('Sampling Interval (seconds)')
    plt.ylabel('Annualized Volatility (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return results


def demonstrate_high_frequency_class(prices: np.ndarray, times: np.ndarray) -> dict:
    """
    Demonstrates usage of the HighFrequencyData class for integrated analysis.
    
    Parameters
    ----------
    prices : np.ndarray
        High-frequency price series
    times : np.ndarray
        Timestamps corresponding to each price observation
        
    Returns
    -------
    dict
        Dictionary of results from HighFrequencyData analysis
    """
    print("\n" + "="*80)
    print("DEMONSTRATION 5: Using the HighFrequencyData Class for Integrated Analysis")
    print("="*80)
    
    results = {}
    
    # Initialize HighFrequencyData object
    hf_data = HighFrequencyData()
    
    # Preprocess the data
    preprocessed_prices, preprocessed_times = preprocess_price_data(
        prices, times, time_type='seconds', detect_outliers=True
    )
    
    # Set data in HighFrequencyData object
    hf_data.set_data(preprocessed_prices, preprocessed_times, time_type='seconds')
    
    # Compute returns
    print("\nComputing Returns:")
    returns = hf_data.compute_returns(method='log')
    print(f"Computed {len(returns)} returns")
    results['returns'] = returns
    
    # Filter data
    print("\nFiltering Data:")
    filtered_data = hf_data.filter_data(method='z_score', threshold=3.0)
    print(f"Filtered data contains {len(filtered_data)} observations")
    results['filtered_data'] = filtered_data
    
    # Sample data using calendar time
    print("\nSampling Data:")
    sampled_data = hf_data.sample_data(
        sampling_type='CalendarTime',
        sampling_interval=300  # 5-minute intervals
    )
    print(f"Created {len(sampled_data)} sampled data points")
    results['sampled_data'] = sampled_data
    
    # Estimate volatility using various methods
    print("\nEstimating Volatility:")
    
    # Using standard realized volatility
    rv_results = hf_data.estimate_volatility(
        method='realized_variance',
        sampling_type='CalendarTime',
        sampling_interval=300,  # 5-minute intervals
        annualize=True
    )
    volatility = np.sqrt(rv_results) * np.sqrt(252)  # Annualized
    print(f"Realized volatility (standard): {volatility:.4f}")
    results['realized_volatility'] = volatility
    
    # Using asynchronous estimation
    print("\nAsynchronous Volatility Estimation:")
    async_result = asyncio.run(
        hf_data.async_estimate_volatility(
            method='realized_kernel',
            kernel_type='bartlett',
            sampling_type='CalendarTime',
            sampling_interval=300,
            annualize=True
        )
    )
    print(f"Asynchronous volatility estimate: {async_result:.4f}")
    results['async_volatility'] = async_result
    
    # Print summary of results
    print("\nSummary of Results from HighFrequencyData Analysis:")
    print("-" * 70)
    print(f"{'Metric':<25} {'Value':<15}")
    print("-" * 70)
    print(f"{'Original data points':<25} {len(prices):<15}")
    print(f"{'After preprocessing':<25} {len(preprocessed_prices):<15}")
    print(f"{'After filtering':<25} {len(filtered_data):<15}")
    print(f"{'Sampled data points':<25} {len(sampled_data):<15}")
    print(f"{'Realized volatility':<25} {volatility:<15.4f}")
    print(f"{'Async kernel volatility':<25} {async_result:<15.4f}")
    
    return results


async def async_compute_measures(prices: np.ndarray, times: np.ndarray, measures: list) -> dict:
    """
    Asynchronous function to compute multiple realized measures concurrently.
    
    Parameters
    ----------
    prices : np.ndarray
        High-frequency price series
    times : np.ndarray
        Timestamps corresponding to each price observation
    measures : list
        List of measures to compute
        
    Returns
    -------
    dict
        Dictionary of computed measures
    """
    # Initialize RealizedVolatility object
    rv = RealizedVolatility()
    
    # Set data and parameters
    rv.set_data(prices, times, time_type='seconds')
    rv.set_params({
        'sampling_type': 'CalendarTime',
        'sampling_interval': 300,  # 5-minute intervals
        'kernel_type': 'bartlett',  # For kernel estimation
        'noise_adjust': True,       # Enable noise filtering
        'annualize': True,          # Annualize results
        'scale': 252                # Trading days per year
    })
    
    # Compute measures asynchronously
    result = await rv.compute_async(measures)
    return result


def demonstrate_async_computation(prices: np.ndarray, times: np.ndarray) -> dict:
    """
    Demonstrates asynchronous computation of realized measures.
    
    Parameters
    ----------
    prices : np.ndarray
        High-frequency price series
    times : np.ndarray
        Timestamps corresponding to each price observation
        
    Returns
    -------
    dict
        Dictionary of results from asynchronous computation
    """
    print("\n" + "="*80)
    print("DEMONSTRATION 6: Asynchronous Computation of Realized Measures")
    print("="*80)
    
    results = {}
    
    async def async_demo():
        print("\nRunning Asynchronous Computations:")
        
        # Define measures to compute
        measures_list = [
            ['variance', 'volatility'],
            ['kernel', 'volatility'],
            ['variance', 'kernel', 'volatility']
        ]
        
        # Run multiple computations concurrently
        start_time = time.time()
        
        # Create tasks
        tasks = [
            async_compute_measures(prices, times, measures)
            for measures in measures_list
        ]
        
        # Execute all tasks concurrently
        all_results = await asyncio.gather(*tasks)
        
        execution_time = time.time() - start_time
        
        print(f"\nAll async computations completed in {execution_time:.4f} seconds")
        
        # Print results
        print("\nResults from Asynchronous Computation:")
        print("-" * 70)
        
        for i, result in enumerate(all_results):
            print(f"\nTask {i+1} - Measures: {measures_list[i]}")
            for key, value in result.items():
                if key.startswith('vol'):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value:.8f}")
        
        # For comparison, run sequentially
        print("\nComparison with Sequential Execution:")
        start_time = time.time()
        
        sequential_results = []
        for measures in measures_list:
            rv = RealizedVolatility()
            rv.set_data(prices, times, time_type='seconds')
            rv.set_params({
                'sampling_type': 'CalendarTime',
                'sampling_interval': 300,
                'annualize': True,
                'scale': 252
            })
            sequential_results.append(rv.compute(measures))
        
        seq_execution_time = time.time() - start_time
        
        print(f"Sequential computation completed in {seq_execution_time:.4f} seconds")
        print(f"Speedup factor: {seq_execution_time / execution_time:.2f}x")
        
        return {
            'async_results': all_results,
            'async_time': execution_time,
            'sequential_time': seq_execution_time,
            'speedup': seq_execution_time / execution_time
        }
    
    # Run the async demo using asyncio
    results = asyncio.run(async_demo())
    
    return results


def plot_results(results: dict, plot_type: str, title: str) -> None:
    """
    Creates visualization of realized volatility results.
    
    Parameters
    ----------
    results : dict
        Dictionary containing results to plot
    plot_type : str
        Type of plot to create ('bar', 'line', 'comparison')
    title : str
        Title for the plot
    """
    plt.figure(figsize=(10, 6))
    
    if plot_type == 'bar':
        # For bar charts (e.g., comparing different methods)
        labels = list(results.keys())
        values = [results[k] if isinstance(results[k], float) else results[k].get('volatility', 0) 
                  for k in labels]
        
        plt.bar(labels, values)
        plt.ylabel('Value')
        
    elif plot_type == 'line':
        # For line charts (e.g., time series)
        x = list(results.keys())
        y = [results[k] if isinstance(results[k], float) else results[k].get('volatility', 0) 
             for k in x]
        
        plt.plot(x, y, 'o-', linewidth=2)
        plt.ylabel('Value')
        
    elif plot_type == 'comparison':
        # For comparing multiple series
        for label, data in results.items():
            if isinstance(data, dict) and 'x' in data and 'y' in data:
                plt.plot(data['x'], data['y'], label=label)
            else:
                # Skip if data format doesn't match expected structure
                continue
        
        plt.legend()
    
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def main() -> None:
    """
    Main function to run all demonstrations.
    """
    print("="*80)
    print("MFE TOOLBOX - REALIZED MEASURES DEMONSTRATION")
    print("="*80)
    print("\nThis example demonstrates how to use the realized volatility measures")
    print("and high-frequency data analysis capabilities of the MFE Toolbox.")
    print("\nThe example includes:")
    print("1. Calculation of realized variance with different sampling methods")
    print("2. Kernel-based estimation of realized volatility")
    print("3. Analysis of sampling frequency effects on volatility estimates")
    print("4. Noise filtering techniques for high-frequency data")
    print("5. Using the HighFrequencyData class for integrated analysis")
    print("6. Asynchronous computation of multiple realized measures")
    
    # Load or generate sample data
    prices, times = load_sample_data()
    
    # Run demonstrations
    demo1_results = demonstrate_realized_variance(prices, times)
    demo2_results = demonstrate_kernel_estimation(prices, times)
    demo3_results = demonstrate_sampling_effects(prices, times)
    demo4_results = demonstrate_noise_filtering(prices, times)
    demo5_results = demonstrate_high_frequency_class(prices, times)
    demo6_results = demonstrate_async_computation(prices, times)
    
    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETE")
    print("="*80)
    print("\nSummary of Key Findings:")
    print("- Different sampling methods can produce varying realized variance estimates")
    print("- Kernel-based estimators provide robust volatility measures")
    print("- Sampling frequency significantly impacts volatility estimates due to microstructure noise")
    print("- Noise filtering is essential for accurate high-frequency volatility estimation")
    print("- The HighFrequencyData class provides an integrated approach to analysis")
    print("- Asynchronous computation enables efficient processing of multiple measures")


if __name__ == '__main__':
    main()