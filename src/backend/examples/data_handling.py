"""
MFE Toolbox - Data Handling Examples

This module demonstrates the data handling capabilities of the MFE Toolbox
for financial time series data, showcasing loading, preprocessing, transformation,
and visualization of data for financial econometric analysis.
"""

import os  # Python 3.12
import numpy as np  # numpy 1.26.3
import pandas as pd  # pandas 2.1.4
import matplotlib.pyplot as plt  # matplotlib 3.8.0
from pathlib import Path  # Python 3.12
import statsmodels.api as sm  # statsmodels 0.14.1
import statsmodels.graphics.tsaplots as tsaplots  # statsmodels 0.14.1

# Import MFE Toolbox utilities
from mfe.utils.data_handling import (
    load_financial_data, convert_time_series, calculate_financial_returns,
    split_train_test, handle_missing_values, detect_outliers, standardize_data
)
from mfe.utils.validation import check_time_series, is_float_array
from mfe.utils.pandas_helpers import ensure_datetime_index, extract_time_features
from mfe.models.realized import RealizedVolatility

# Define paths for sample data and output
SAMPLE_DATA_PATH = Path(__file__).parent.parent / 'tests' / 'test_data' / 'market_benchmark.npy'
EXAMPLE_OUTPUT_DIR = Path(__file__).parent / 'output'

def load_example_data(file_path=None, as_dataframe=True):
    """
    Loads example financial data from the sample data path or a specified file path.
    
    Parameters
    ----------
    file_path : Optional[str]
        Path to the financial data file. If None, uses the default sample data.
    as_dataframe : bool
        If True, returns the data as a pandas DataFrame, otherwise as a numpy array.
        
    Returns
    -------
    Union[pd.DataFrame, np.ndarray]
        The loaded financial time series data.
    """
    # Use default sample data path if not provided
    if file_path is None:
        file_path = SAMPLE_DATA_PATH
    
    # Validate that the file exists
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    # Load data based on file extension
    if file_path.suffix == '.npy':
        data = np.load(file_path)
        if as_dataframe:
            # Create a simple DataFrame with a datetime index
            dates = pd.date_range(start='2022-01-01', periods=len(data))
            data = pd.DataFrame(data, index=dates, columns=['price'])
    else:
        # Use the data_handling utility for other file formats
        data = load_financial_data(
            str(file_path),
            file_format=None,  # Auto-detect format from extension
            date_column='date' if as_dataframe else None
        )
    
    print(f"Loaded data of shape: {data.shape}")
    return data

def prepare_returns_data(prices, method='simple', periods=1):
    """
    Prepares returns data from price data using various calculation methods.
    
    Parameters
    ----------
    prices : Union[pd.DataFrame, np.ndarray]
        Price data to calculate returns from.
    method : str
        Method for return calculation: 'simple', 'log', or 'percentage'.
    periods : int
        Number of periods to use in return calculation.
        
    Returns
    -------
    Union[pd.DataFrame, np.ndarray]
        Calculated returns data in the same format as input.
    """
    # Validate input data
    check_time_series(prices, allow_2d=True)
    
    # Calculate returns using data_handling utility
    returns = calculate_financial_returns(
        prices, method=method, periods=periods, fill_na=True
    )
    
    # Print summary statistics for returns
    if isinstance(returns, pd.DataFrame) or isinstance(returns, pd.Series):
        print(f"\n{method.capitalize()} returns summary statistics ({periods}-period):")
        print(returns.describe())
    else:
        print(f"\n{method.capitalize()} returns summary statistics ({periods}-period):")
        print(f"Mean: {np.mean(returns):.6f}, Std Dev: {np.std(returns):.6f}")
        print(f"Min: {np.min(returns):.6f}, Max: {np.max(returns):.6f}")
    
    return returns

def clean_and_preprocess_data(data, missing_method='ffill', handle_outliers=True, outlier_threshold=3.0):
    """
    Cleans and preprocesses financial time series data by handling missing values and outliers.
    
    Parameters
    ----------
    data : Union[pd.DataFrame, np.ndarray]
        Time series data to clean and preprocess.
    missing_method : str
        Method for handling missing values: 'ffill', 'bfill', 'linear', 'cubic', etc.
    handle_outliers : bool
        If True, detects and handles outliers.
    outlier_threshold : float
        Threshold for outlier detection (number of standard deviations).
        
    Returns
    -------
    Union[pd.DataFrame, np.ndarray]
        Cleaned and preprocessed data.
    """
    # Validate input data
    check_time_series(data, allow_2d=True, allow_missing=True)
    
    # Keep track of original format
    is_dataframe = isinstance(data, pd.DataFrame)
    is_series = isinstance(data, pd.Series)
    
    # Check for missing values
    if is_dataframe:
        missing_count = data.isna().sum().sum()
    elif is_series:
        missing_count = data.isna().sum()
    else:
        missing_count = np.isnan(data).sum()
    
    if missing_count > 0:
        print(f"Found {missing_count} missing values in data")
        
        # Handle missing values
        processed_data = handle_missing_values(data, method=missing_method)
        
        if is_dataframe:
            post_missing = processed_data.isna().sum().sum()
        elif is_series:
            post_missing = processed_data.isna().sum()
        else:
            post_missing = np.isnan(processed_data).sum()
            
        print(f"After '{missing_method}' method: {post_missing} missing values remain")
    else:
        processed_data = data
        print("No missing values found in data")
    
    # Handle outliers if requested
    if handle_outliers:
        print("\nChecking for outliers...")
        
        # Detect outliers and get both cleaned data and outlier mask
        if is_dataframe or is_series:
            cleaned_data, outlier_mask = detect_outliers(
                processed_data, method='z_score', threshold=outlier_threshold, return_mask=True
            )
            outlier_count = outlier_mask.sum().sum() if is_dataframe else outlier_mask.sum()
        else:
            cleaned_data, outlier_mask = detect_outliers(
                processed_data, method='z_score', threshold=outlier_threshold, return_mask=True
            )
            outlier_count = np.sum(outlier_mask)
            
        print(f"Detected {outlier_count} outliers using z-score method with threshold {outlier_threshold}")
        
        # If there are outliers, replace them with NaN and then handle missing values again
        if outlier_count > 0:
            processed_data = handle_missing_values(cleaned_data, method=missing_method)
            print(f"Replaced outliers and filled using '{missing_method}' method")
    
    return processed_data

def split_data_for_modeling(data, split_point=0.8):
    """
    Splits time series data into training and testing sets for model development.
    
    Parameters
    ----------
    data : Union[pd.DataFrame, np.ndarray]
        Time series data to split.
    split_point : Union[float, str, pd.Timestamp]
        Point to split the data:
        - If float (0 < split_point < 1): Fraction of data to use for training
        - If str or pd.Timestamp: Date to use as split point
        
    Returns
    -------
    Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.DataFrame, np.ndarray]]
        Tuple containing training and testing datasets.
    """
    # Validate input data
    check_time_series(data, allow_2d=True)
    
    # Use data_handling utility to split data
    train_data, test_data = split_train_test(data, split_point=split_point)
    
    # Print information about the split
    print("\nData split for modeling:")
    if isinstance(data, (pd.DataFrame, pd.Series)):
        print(f"Full dataset: {len(data)} observations")
        print(f"Training set: {len(train_data)} observations ({len(train_data)/len(data)*100:.1f}%)")
        print(f"Testing set:  {len(test_data)} observations ({len(test_data)/len(data)*100:.1f}%)")
        
        if isinstance(data.index, pd.DatetimeIndex):
            print(f"\nTime ranges:")
            print(f"Training: {train_data.index[0]} to {train_data.index[-1]}")
            print(f"Testing:  {test_data.index[0]} to {test_data.index[-1]}")
    else:
        print(f"Full dataset: {len(data)} observations")
        print(f"Training set: {len(train_data)} observations ({len(train_data)/len(data)*100:.1f}%)")
        print(f"Testing set:  {len(test_data)} observations ({len(test_data)/len(data)*100:.1f}%)")
    
    return train_data, test_data

def visualize_financial_data(data, title=None, show_returns=True, show_stats=True, show_acf_pacf=True):
    """
    Creates comprehensive visualizations of financial time series data.
    
    Parameters
    ----------
    data : Union[pd.DataFrame, pd.Series]
        Financial time series data to visualize.
    title : Optional[str]
        Title for the visualization.
    show_returns : bool
        If True, calculates and shows returns.
    show_stats : bool
        If True, displays descriptive statistics.
    show_acf_pacf : bool
        If True, shows autocorrelation and partial autocorrelation plots.
    
    Returns
    -------
    None
    """
    # Validate input data is pandas object
    if not isinstance(data, (pd.DataFrame, pd.Series)):
        raise TypeError("Data must be a pandas DataFrame or Series for visualization")
    
    # Convert Series to DataFrame if needed
    if isinstance(data, pd.Series):
        data = data.to_frame(data.name if data.name else "Value")
    
    # Determine the number of plots needed
    num_plots = 1  # Original time series
    if show_returns:
        num_plots += 1
    if show_acf_pacf:
        num_plots += 2  # ACF and PACF plots
    
    # Create figure with appropriate number of subplots
    fig = plt.figure(figsize=(12, num_plots * 4))
    
    # Plot original time series
    ax1 = fig.add_subplot(num_plots, 1, 1)
    data.plot(ax=ax1)
    ax1.set_title(title if title else "Financial Time Series")
    ax1.set_ylabel("Value")
    ax1.grid(True)
    
    # Add plot number tracker
    plot_num = 2
    
    # Calculate and plot returns if requested
    if show_returns:
        ax2 = fig.add_subplot(num_plots, 1, plot_num)
        plot_num += 1
        
        # Calculate returns for the first column if multiple columns exist
        if data.shape[1] > 1:
            returns = data.iloc[:, 0].pct_change().dropna()
            returns.name = f"{data.columns[0]} Returns"
        else:
            returns = data.pct_change().dropna()
            returns.columns = [f"{col} Returns" for col in returns.columns]
        
        returns.plot(ax=ax2)
        ax2.set_title(f"Returns")
        ax2.set_ylabel("Return")
        ax2.grid(True)
        
        # Show descriptive statistics if requested
        if show_stats:
            stats_text = (
                f"Mean: {returns.mean().values[0]:.6f}\n"
                f"Std Dev: {returns.std().values[0]:.6f}\n"
                f"Skewness: {returns.skew().values[0]:.6f}\n"
                f"Kurtosis: {returns.kurtosis().values[0]:.6f}"
            )
            ax2.text(
                0.02, 0.95, stats_text, transform=ax2.transAxes,
                bbox=dict(facecolor='white', alpha=0.7), verticalalignment='top'
            )
    
    # Add ACF and PACF plots if requested
    if show_acf_pacf:
        # Determine which data to use for ACF/PACF
        if show_returns and 'returns' in locals():
            acf_data = returns.iloc[:, 0] if isinstance(returns, pd.DataFrame) else returns
        else:
            acf_data = data.iloc[:, 0] if data.shape[1] > 1 else data.iloc[:, 0]
        
        # ACF plot
        ax3 = fig.add_subplot(num_plots, 1, plot_num)
        plot_num += 1
        tsaplots.plot_acf(acf_data, lags=30, ax=ax3)
        ax3.set_title("Autocorrelation Function")
        ax3.grid(True)
        
        # PACF plot
        ax4 = fig.add_subplot(num_plots, 1, plot_num)
        tsaplots.plot_pacf(acf_data, lags=30, ax=ax4)
        ax4.set_title("Partial Autocorrelation Function")
        ax4.grid(True)
    
    # Adjust layout and display
    plt.tight_layout()
    plt.show()

def demonstrate_high_frequency_analysis(prices, times):
    """
    Demonstrates high-frequency data analysis using the RealizedVolatility class.
    
    Parameters
    ----------
    prices : Union[pd.DataFrame, np.ndarray]
        High-frequency price data.
    times : Union[pd.Series, np.ndarray]
        Corresponding timestamps.
        
    Returns
    -------
    dict
        Dictionary of computed realized volatility measures.
    """
    # Validate input data
    if isinstance(prices, pd.DataFrame):
        prices_array = prices.values
        if prices.shape[1] > 1:
            # Use the first column if multiple columns
            prices_array = prices.iloc[:, 0].values
    else:
        prices_array = np.asarray(prices)
    
    if isinstance(times, pd.Series):
        times_array = times.values
    else:
        times_array = np.asarray(times)
    
    # Ensure we have sufficient data for high-frequency analysis
    if len(prices_array) < 100:
        print("Warning: Limited data for high-frequency analysis. Using simplified example.")
    
    # Create RealizedVolatility instance
    rv = RealizedVolatility()
    
    # Determine time_type based on times
    if isinstance(times, pd.DatetimeIndex) or (isinstance(times, pd.Series) and pd.api.types.is_datetime64_dtype(times)):
        time_type = 'datetime'
    else:
        # Assume seconds format for simplicity
        time_type = 'seconds'
    
    # Set data for analysis
    rv.set_data(prices_array, times_array, time_type)
    
    # Configure parameters for high-frequency analysis
    rv.set_params({
        'sampling_type': 'CalendarTime',
        'sampling_interval': 300,  # 5-minute intervals
        'noise_adjust': True,
        'kernel_type': 'bartlett',
        'detect_outliers': True,
        'outlier_threshold': 3.0,
        'annualize': True
    })
    
    print("\nPerforming high-frequency analysis...")
    
    # Compute realized measures
    measures = ['variance', 'volatility', 'kernel']
    results = rv.compute(measures)
    
    # Print results summary
    print("\nRealized Volatility Measures:")
    print(f"Realized Variance: {results.get('variance', 'N/A'):.6f}")
    print(f"Realized Volatility: {results.get('volatility', 'N/A'):.6f}")
    print(f"Realized Kernel: {results.get('kernel', 'N/A'):.6f}")
    
    return results

def extract_features_from_time_series(data, time_features=['year', 'month', 'day', 'dayofweek']):
    """
    Demonstrates extracting meaningful features from time series for modeling.
    
    Parameters
    ----------
    data : pd.DataFrame
        Time series data with datetime index.
    time_features : List[str]
        List of time features to extract.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with extracted time series features.
    """
    # Ensure data has datetime index
    data = ensure_datetime_index(data)
    
    # Extract time features
    enhanced_data = extract_time_features(data, time_features)
    
    # Add some derived features
    if 'month' in enhanced_data.columns and 'dayofweek' in enhanced_data.columns:
        # Create month-end indicator
        if 'day' in enhanced_data.columns:
            # This is a simplistic approach - a more accurate one would use calendar logic
            enhanced_data['is_month_end'] = (enhanced_data['day'] >= 28)
        
        # Create weekend indicator
        enhanced_data['is_weekend'] = enhanced_data['dayofweek'] >= 5
        
        # Create quarter indicator
        if 'month' in enhanced_data.columns:
            enhanced_data['quarter'] = ((enhanced_data['month'] - 1) // 3) + 1
    
    # Print feature information
    print("\nExtracted time features:")
    print(f"Data shape: {enhanced_data.shape}")
    print(f"Features added: {[col for col in enhanced_data.columns if col not in data.columns]}")
    
    # Show a sample of the enhanced data
    print("\nSample of enhanced data with time features:")
    print(enhanced_data.head())
    
    return enhanced_data

def run_data_handling_examples():
    """
    Main function that runs a comprehensive set of data handling examples.
    """
    # Create output directory if it doesn't exist
    EXAMPLE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("MFE TOOLBOX - DATA HANDLING EXAMPLES")
    print("=" * 80)
    
    # Example 1: Load data
    print("\n1. LOADING FINANCIAL DATA")
    print("-" * 50)
    data = load_example_data()
    
    # Example 2: Data format conversion
    print("\n2. DATA FORMAT CONVERSION")
    print("-" * 50)
    print("Converting data between DataFrame, Series, and NumPy array formats:")
    
    # Convert to numpy array
    numpy_data = convert_time_series(data, output_type='numpy')
    print(f"- Converted to NumPy array of shape {numpy_data.shape}")
    
    # Convert to pandas Series (first column if multiple)
    series_data = convert_time_series(
        data.iloc[:, 0] if data.shape[1] > 1 else data, 
        output_type='series'
    )
    print(f"- Converted to Pandas Series of length {len(series_data)}")
    
    # Convert back to DataFrame
    dataframe_data = convert_time_series(series_data, output_type='dataframe')
    print(f"- Converted back to DataFrame of shape {dataframe_data.shape}")
    
    # Example 3: Prepare returns data
    print("\n3. PREPARING RETURNS DATA")
    print("-" * 50)
    simple_returns = prepare_returns_data(data, method='simple', periods=1)
    log_returns = prepare_returns_data(data, method='log', periods=1)
    
    # Visualize returns
    visualize_financial_data(
        log_returns, 
        title="Log Returns Analysis",
        show_returns=False,  # Already showing returns
        show_stats=True,
        show_acf_pacf=True
    )
    
    # Example 4: Data cleaning and preprocessing
    print("\n4. DATA CLEANING AND PREPROCESSING")
    print("-" * 50)
    
    # Introduce some missing values for demonstration
    if isinstance(data, pd.DataFrame):
        missing_data = data.copy()
        missing_data.iloc[10:15, :] = np.nan
        missing_data.iloc[50:52, :] = np.nan
    else:
        missing_data = np.copy(data)
        missing_data[10:15] = np.nan
        missing_data[50:52] = np.nan
    
    print("Cleaning and preprocessing data with missing values and outliers:")
    cleaned_data = clean_and_preprocess_data(
        missing_data, 
        missing_method='linear', 
        handle_outliers=True,
        outlier_threshold=3.0
    )
    
    # Example 5: Split data for modeling
    print("\n5. SPLITTING DATA FOR MODELING")
    print("-" * 50)
    train_data, test_data = split_data_for_modeling(cleaned_data, split_point=0.8)
    
    # Example 6: Extract features from time series
    print("\n6. EXTRACTING TIME FEATURES")
    print("-" * 50)
    if isinstance(data, pd.DataFrame) and isinstance(data.index, pd.DatetimeIndex):
        enhanced_data = extract_features_from_time_series(
            data,
            time_features=['year', 'month', 'day', 'dayofweek', 'quarter', 'is_month_start', 'is_month_end']
        )
    else:
        print("Skipping time feature extraction - data doesn't have DatetimeIndex")
    
    # Example 7: High-frequency analysis (if appropriate data available)
    print("\n7. HIGH-FREQUENCY ANALYSIS")
    print("-" * 50)
    try:
        # For demonstration, generate some high-frequency data if real data is not available
        if len(data) < 100 or not isinstance(data.index, pd.DatetimeIndex):
            print("Generating synthetic high-frequency data for demonstration...")
            # Generate 1000 price points with 1-minute intervals
            syn_dates = pd.date_range(start='2022-01-01 09:30:00', periods=1000, freq='1min')
            # Generate geometric Brownian motion for prices
            np.random.seed(42)
            price = 100.0
            returns = np.random.normal(0.0001, 0.001, 1000)
            prices = np.exp(np.cumsum(returns)) * price
            hf_data = pd.DataFrame(prices, index=syn_dates, columns=['price'])
            
            # Run high-frequency analysis
            hf_results = demonstrate_high_frequency_analysis(hf_data, hf_data.index)
        else:
            # Use real data if available
            hf_results = demonstrate_high_frequency_analysis(data, data.index)
    except Exception as e:
        print(f"Error in high-frequency analysis: {str(e)}")
        print("Skipping high-frequency analysis")
    
    print("\nAll data handling examples completed successfully!")

if __name__ == "__main__":
    run_data_handling_examples()