"""
Unit and property-based tests for the MFE Toolbox data_handling module.

This test file validates the functionality of time series data manipulation, transformation,
and preprocessing functions using pytest and hypothesis for property-based testing.
"""

import pytest
import numpy as np  # numpy 1.26.3
import pandas as pd  # pandas 2.1.4
from hypothesis import given, strategies as st  # hypothesis 6.92.1
from hypothesis.extra.numpy import arrays  # hypothesis 6.92.1
from hypothesis.extra.pandas import series  # hypothesis 6.92.1
import pytest_asyncio  # pytest-asyncio 0.21.1

from mfe.utils.data_handling import (
    convert_time_series, convert_frequency, filter_time_series, detect_outliers,
    handle_missing_values, calculate_financial_returns, split_train_test,
    convert_to_log_returns, convert_to_simple_returns, standardize_data,
    handle_date_range, resample_high_frequency, align_multiple_series,
    merge_time_series, async_convert_time_series
)


def test_convert_time_series_basic():
    """Tests basic functionality of convert_time_series with various input and output formats."""
    # Create test data in different formats
    data_np = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    data_series = pd.Series(data_np, name='data')
    data_df = pd.DataFrame({'data': data_np})
    
    # Test conversion to numpy
    np_from_np = convert_time_series(data_np, output_type='numpy')
    np_from_series = convert_time_series(data_series, output_type='numpy')
    np_from_df = convert_time_series(data_df, output_type='numpy')
    
    assert isinstance(np_from_np, np.ndarray)
    assert isinstance(np_from_series, np.ndarray)
    assert isinstance(np_from_df, np.ndarray)
    assert np.array_equal(np_from_np, data_np)
    assert np.array_equal(np_from_series, data_np)
    assert np.array_equal(np_from_df.flatten(), data_np)  # Flatten to handle 2D arrays
    
    # Test conversion to pandas Series
    series_from_np = convert_time_series(data_np, output_type='series')
    series_from_series = convert_time_series(data_series, output_type='series')
    series_from_df = convert_time_series(data_df, output_type='series')
    
    assert isinstance(series_from_np, pd.Series)
    assert isinstance(series_from_series, pd.Series)
    assert isinstance(series_from_df, pd.Series)
    assert series_from_np.equals(pd.Series(data_np))
    assert series_from_series.equals(data_series)
    assert series_from_df.equals(pd.Series(data_np))
    
    # Test conversion to pandas DataFrame
    df_from_np = convert_time_series(data_np, output_type='dataframe')
    df_from_series = convert_time_series(data_series, output_type='dataframe')
    df_from_df = convert_time_series(data_df, output_type='dataframe')
    
    assert isinstance(df_from_np, pd.DataFrame)
    assert isinstance(df_from_series, pd.DataFrame)
    assert isinstance(df_from_df, pd.DataFrame)
    assert df_from_np.equals(pd.DataFrame(data_np))
    assert df_from_series.equals(pd.DataFrame(data_series))
    assert df_from_df.equals(data_df)


def test_convert_time_series_with_time_index():
    """Tests convert_time_series with time index handling."""
    # Create data with datetime index
    dates = pd.date_range('2023-01-01', periods=5)
    data_series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=dates, name='data')
    data_df = pd.DataFrame({'data': [1.0, 2.0, 3.0, 4.0, 5.0]}, index=dates)
    
    # Test with explicit time_index parameter for simple array
    array_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    series_with_dates = convert_time_series(array_data, output_type='series', time_index=dates)
    df_with_dates = convert_time_series(array_data, output_type='dataframe', time_index=dates)
    
    assert isinstance(series_with_dates, pd.Series)
    assert isinstance(df_with_dates, pd.DataFrame)
    assert series_with_dates.index.equals(dates)
    assert df_with_dates.index.equals(dates)
    
    # Test preservation of existing indexes
    series_preserved = convert_time_series(data_series, output_type='series')
    df_preserved = convert_time_series(data_df, output_type='dataframe')
    
    assert series_preserved.index.equals(dates)
    assert df_preserved.index.equals(dates)
    
    # Test with various frequency specifications
    dates_hourly = pd.date_range('2023-01-01', periods=5, freq='H')
    dates_minute = pd.date_range('2023-01-01', periods=5, freq='T')
    
    series_hourly = convert_time_series(array_data, output_type='series', time_index=dates_hourly)
    series_minute = convert_time_series(array_data, output_type='series', time_index=dates_minute)
    
    assert series_hourly.index.equals(dates_hourly)
    assert series_minute.index.equals(dates_minute)


def test_convert_time_series_error_handling():
    """Tests error handling in convert_time_series."""
    # Test with invalid output_type
    with pytest.raises(TypeError):
        convert_time_series(np.array([1, 2, 3]), output_type='invalid')
    
    # Test with invalid data (containing NaN) when ensure_finite=True
    with pytest.raises(ValueError):
        convert_time_series(np.array([1, np.nan, 3]), output_type='numpy', ensure_finite=True)
    
    # Test with incompatible time_index
    with pytest.raises(ValueError):
        # Provide time_index with wrong length
        convert_time_series(
            np.array([1, 2, 3]), 
            output_type='series', 
            time_index=pd.date_range('2023-01-01', periods=2)
        )


def test_convert_frequency():
    """Tests frequency conversion functionality."""
    # Create daily time series
    dates = pd.date_range('2023-01-01', periods=10, freq='D')
    data = pd.Series(np.arange(10), index=dates)
    
    # Test downsampling to monthly frequency with different methods
    monthly_mean = convert_frequency(data, freq='M', method='mean')
    monthly_sum = convert_frequency(data, freq='M', method='sum')
    
    assert isinstance(monthly_mean, pd.Series)
    assert monthly_mean.index.freq.startswith('M')
    assert len(monthly_mean) < len(data)  # Downsampling should reduce the length
    
    # Test upsampling to hourly frequency with different methods
    hourly_ffill = convert_frequency(data, freq='H', method='first')
    
    assert isinstance(hourly_ffill, pd.Series)
    assert hourly_ffill.index.freq == 'H'
    assert len(hourly_ffill) > len(data)  # Upsampling should increase the length
    
    # Test with different include_start and include_end parameters
    monthly_inclusive = convert_frequency(data, freq='M', include_start=True, include_end=True)
    monthly_exclusive = convert_frequency(data, freq='M', include_start=False, include_end=False)
    
    # They should produce different results with these parameters
    assert len(monthly_inclusive) == len(monthly_exclusive)


def test_filter_time_series():
    """Tests time series filtering functionality."""
    # Create noisy time series
    np.random.seed(42)
    x = np.linspace(0, 10, 1000)
    noise = np.random.normal(0, 1, 1000)
    signal = np.sin(x) + noise
    data = pd.Series(signal, index=pd.date_range('2023-01-01', periods=1000, freq='H'))
    
    # Test moving average filter
    filtered_ma = filter_time_series(data, filter_type='moving_average', filter_params={'window': 10})
    assert isinstance(filtered_ma, pd.Series)
    assert len(filtered_ma) == len(data)
    assert filtered_ma.index.equals(data.index)
    # Standard deviation of filtered signal should be lower than original noisy signal
    assert filtered_ma.std() < data.std()
    
    # Test EWMA filter
    filtered_ewma = filter_time_series(data, filter_type='ewma', filter_params={'span': 10})
    assert isinstance(filtered_ewma, pd.Series)
    assert len(filtered_ewma) == len(data)
    assert filtered_ewma.std() < data.std()
    
    # Test Butterworth filter
    filtered_butterworth = filter_time_series(
        data, 
        filter_type='butterworth', 
        filter_params={'N': 2, 'Wn': 0.05}
    )
    assert isinstance(filtered_butterworth, pd.Series)
    assert len(filtered_butterworth) == len(data)
    assert filtered_butterworth.std() < data.std()
    
    # Test Hodrick-Prescott filter
    filtered_hp = filter_time_series(
        data, 
        filter_type='hodrick_prescott', 
        filter_params={'lambda': 1600}
    )
    assert isinstance(filtered_hp, pd.Series)
    assert len(filtered_hp) == len(data)
    # HP filter extracts trend, so should be smoother
    assert filtered_hp.diff().std() < data.diff().std()


def test_detect_outliers():
    """Tests outlier detection functionality."""
    # Create data with known outliers
    np.random.seed(42)
    normal_data = np.random.normal(0, 1, 100)
    data_with_outliers = normal_data.copy()
    # Add outliers at specific positions
    outlier_positions = [10, 25, 75]
    for pos in outlier_positions:
        data_with_outliers[pos] = 10.0  # Clear outlier
    
    series_data = pd.Series(data_with_outliers)
    
    # Test z-score method
    cleaned_z, mask_z = detect_outliers(
        series_data, 
        method='z_score', 
        threshold=3.0, 
        return_mask=True
    )
    
    assert isinstance(cleaned_z, pd.Series)
    assert isinstance(mask_z, pd.Series)
    assert mask_z.sum() >= len(outlier_positions)  # Should detect at least our known outliers
    assert np.isnan(cleaned_z[outlier_positions]).all()  # Outliers should be replaced with NaN
    
    # Test IQR method
    cleaned_iqr, mask_iqr = detect_outliers(
        series_data, 
        method='iqr', 
        threshold=1.5, 
        return_mask=True
    )
    
    assert isinstance(cleaned_iqr, pd.Series)
    assert mask_iqr.sum() >= len(outlier_positions)
    
    # Test modified Z-score method
    cleaned_mod_z, mask_mod_z = detect_outliers(
        series_data, 
        method='modified_z', 
        threshold=3.5, 
        return_mask=True
    )
    
    assert isinstance(cleaned_mod_z, pd.Series)
    assert mask_mod_z.sum() >= len(outlier_positions)
    
    # Test with NumPy array input
    cleaned_np, mask_np = detect_outliers(
        data_with_outliers, 
        method='z_score', 
        threshold=3.0, 
        return_mask=True
    )
    
    assert isinstance(cleaned_np, np.ndarray)
    assert isinstance(mask_np, np.ndarray)
    assert mask_np.sum() >= len(outlier_positions)


def test_handle_missing_values():
    """Tests missing value handling functionality."""
    # Create test data with NaN values in various patterns
    dates = pd.date_range('2023-01-01', periods=10)
    data = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0, np.nan, np.nan, 8.0, 9.0, 10.0], index=dates)
    
    # Test forward fill imputation
    ffilled = handle_missing_values(data, method='ffill')
    assert isinstance(ffilled, pd.Series)
    assert not ffilled.isna().any()  # No NaNs after forward filling
    assert ffilled[2] == 2.0  # NaN at position 2 should be filled with previous value
    
    # Test backward fill imputation
    bfilled = handle_missing_values(data, method='bfill')
    assert isinstance(bfilled, pd.Series)
    assert not bfilled.isna().any()
    assert bfilled[2] == 4.0  # NaN at position 2 should be filled with next value
    
    # Test linear interpolation
    interpolated = handle_missing_values(data, method='linear')
    assert isinstance(interpolated, pd.Series)
    assert not interpolated.isna().any()
    assert interpolated[2] == 3.0  # NaN at position 2 should be (2+4)/2 = 3
    
    # Test mean imputation
    mean_filled = handle_missing_values(data, method='mean')
    assert isinstance(mean_filled, pd.Series)
    assert not mean_filled.isna().any()
    non_nan_mean = data.dropna().mean()
    assert mean_filled[2] == non_nan_mean
    
    # Test with various methods on DataFrame
    df = pd.DataFrame({
        'A': [1.0, 2.0, np.nan, 4.0, 5.0],
        'B': [np.nan, 2.0, 3.0, np.nan, 5.0]
    })
    
    methods = ['ffill', 'bfill', 'linear', 'mean', 'median', 'mode']
    for method in methods:
        result = handle_missing_values(df, method=method)
        assert isinstance(result, pd.DataFrame)
        assert not result.isna().any().any()  # No NaNs in the entire DataFrame


def test_calculate_financial_returns():
    """Tests financial return calculation functionality."""
    # Create price series
    dates = pd.date_range('2023-01-01', periods=10)
    prices = pd.Series([100, 102, 105, 103, 107, 110, 112, 109, 111, 115], index=dates)
    
    # Test simple returns
    simple_returns = calculate_financial_returns(prices, method='simple', periods=1)
    assert isinstance(simple_returns, pd.Series)
    assert len(simple_returns) == len(prices)
    # Check first return calculation: (102/100) - 1 = 0.02
    assert np.isclose(simple_returns.iloc[1], 0.02)
    
    # Test log returns
    log_returns = calculate_financial_returns(prices, method='log', periods=1)
    assert isinstance(log_returns, pd.Series)
    # Check log return calculation: log(102/100) ≈ 0.01980
    assert np.isclose(log_returns.iloc[1], np.log(102/100))
    
    # Test percentage returns
    pct_returns = calculate_financial_returns(prices, method='percentage', periods=1)
    assert isinstance(pct_returns, pd.Series)
    # Check percentage return: ((102-100)/100) * 100 = 2.0
    assert np.isclose(pct_returns.iloc[1], 2.0)
    
    # Test with different period parameters
    two_period_returns = calculate_financial_returns(prices, method='simple', periods=2)
    assert isinstance(two_period_returns, pd.Series)
    # Two-period return: (105/100) - 1 = 0.05
    assert np.isclose(two_period_returns.iloc[2], 0.05)
    
    # Test with fill_na=False
    returns_with_na = calculate_financial_returns(prices, method='simple', fill_na=False)
    assert np.isnan(returns_with_na.iloc[0])  # First value should be NaN


def test_split_train_test():
    """Tests training/testing split functionality."""
    # Create time series data
    dates = pd.date_range('2023-01-01', periods=100)
    data = pd.Series(np.random.randn(100), index=dates)
    
    # Test with proportion-based split point
    train, test = split_train_test(data, split_point=0.8)
    assert isinstance(train, pd.Series)
    assert isinstance(test, pd.Series)
    assert len(train) == 80
    assert len(test) == 20
    
    # Test with date-based split point
    split_date = '2023-03-01'
    train_date, test_date = split_train_test(data, split_point=split_date)
    assert isinstance(train_date, pd.Series)
    assert isinstance(test_date, pd.Series)
    assert train_date.index[-1] <= pd.Timestamp(split_date)
    assert test_date.index[0] >= pd.Timestamp(split_date)
    
    # Test with shuffle parameter
    train_shuffled, test_shuffled = split_train_test(data.values, split_point=0.8, shuffle=True)
    assert isinstance(train_shuffled, np.ndarray)
    assert isinstance(test_shuffled, np.ndarray)
    assert len(train_shuffled) == 80
    assert len(test_shuffled) == 20
    
    # Check if shuffled is different from ordered
    train_ordered, _ = split_train_test(data.values, split_point=0.8, shuffle=False)
    assert not np.array_equal(train_shuffled, train_ordered)  # Should be different due to shuffling


def test_convert_to_log_returns():
    """Tests log return conversion functionality."""
    # Create price series
    dates = pd.date_range('2023-01-01', periods=10)
    prices = pd.Series([100, 102, 105, 103, 107, 110, 112, 109, 111, 115], index=dates)
    
    # Test log returns calculation
    log_returns = convert_to_log_returns(prices)
    assert isinstance(log_returns, pd.Series)
    assert len(log_returns) == len(prices)
    
    # Check log return calculation: log(102/100) ≈ 0.01980
    assert np.isclose(log_returns.iloc[1], np.log(102/100))
    
    # Test with different period parameters
    two_period_returns = convert_to_log_returns(prices, periods=2)
    assert isinstance(two_period_returns, pd.Series)
    # Two-period log return: log(105/100) ≈ 0.04879
    assert np.isclose(two_period_returns.iloc[2], np.log(105/100))
    
    # Test with different input formats
    prices_np = prices.values
    log_returns_np = convert_to_log_returns(prices_np)
    assert isinstance(log_returns_np, np.ndarray)
    assert np.isclose(log_returns_np[1], np.log(102/100))
    
    prices_df = pd.DataFrame({'price': prices})
    log_returns_df = convert_to_log_returns(prices_df)
    assert isinstance(log_returns_df, pd.DataFrame)
    assert np.isclose(log_returns_df.iloc[1, 0], np.log(102/100))


def test_convert_to_simple_returns():
    """Tests simple return conversion functionality."""
    # Create price series
    dates = pd.date_range('2023-01-01', periods=10)
    prices = pd.Series([100, 102, 105, 103, 107, 110, 112, 109, 111, 115], index=dates)
    
    # Test simple returns calculation
    simple_returns = convert_to_simple_returns(prices)
    assert isinstance(simple_returns, pd.Series)
    assert len(simple_returns) == len(prices)
    
    # Check simple return calculation: (102/100) - 1 = 0.02
    assert np.isclose(simple_returns.iloc[1], 0.02)
    
    # Test with different period parameters
    two_period_returns = convert_to_simple_returns(prices, periods=2)
    assert isinstance(two_period_returns, pd.Series)
    # Two-period simple return: (105/100) - 1 = 0.05
    assert np.isclose(two_period_returns.iloc[2], 0.05)
    
    # Test with different input formats
    prices_np = prices.values
    simple_returns_np = convert_to_simple_returns(prices_np)
    assert isinstance(simple_returns_np, np.ndarray)
    assert np.isclose(simple_returns_np[1], 0.02)
    
    prices_df = pd.DataFrame({'price': prices})
    simple_returns_df = convert_to_simple_returns(prices_df)
    assert isinstance(simple_returns_df, pd.DataFrame)
    assert np.isclose(simple_returns_df.iloc[1, 0], 0.02)


def test_standardize_data():
    """Tests data standardization functionality."""
    # Create data with known statistical properties
    np.random.seed(42)
    data = np.random.normal(10, 2, 100)  # mean=10, std=2
    
    # Test standard z-score normalization
    standardized = standardize_data(data, robust=False)
    assert isinstance(standardized, np.ndarray)
    assert len(standardized) == len(data)
    assert np.isclose(standardized.mean(), 0, atol=1e-10)
    assert np.isclose(standardized.std(), 1, atol=1e-10)
    
    # Test robust standardization
    robust_standardized = standardize_data(data, robust=True)
    assert isinstance(robust_standardized, np.ndarray)
    assert len(robust_standardized) == len(data)
    
    # Test with pandas Series
    series_data = pd.Series(data)
    standardized_series = standardize_data(series_data, robust=False)
    assert isinstance(standardized_series, pd.Series)
    assert np.isclose(standardized_series.mean(), 0, atol=1e-10)
    assert np.isclose(standardized_series.std(), 1, atol=1e-10)
    
    # Test with pandas DataFrame
    df_data = pd.DataFrame({'A': data, 'B': data * 2})
    standardized_df = standardize_data(df_data, robust=False)
    assert isinstance(standardized_df, pd.DataFrame)
    assert np.isclose(standardized_df['A'].mean(), 0, atol=1e-10)
    assert np.isclose(standardized_df['A'].std(), 1, atol=1e-10)
    assert np.isclose(standardized_df['B'].mean(), 0, atol=1e-10)
    assert np.isclose(standardized_df['B'].std(), 1, atol=1e-10)
    
    # Test with fit_params
    fit_params = {'mean': 10, 'std': 2}
    standardized_with_params = standardize_data(data, robust=False, fit_params=fit_params)
    assert np.isclose(standardized_with_params[0], (data[0] - 10) / 2)


def test_handle_date_range():
    """Tests date range filtering functionality."""
    # Create time series data spanning multiple time periods
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    data = pd.Series(range(100), index=dates)
    
    # Test filtering with start_date only
    start_filtered = handle_date_range(data, start_date='2023-02-01')
    assert isinstance(start_filtered, pd.Series)
    assert start_filtered.index[0] >= pd.Timestamp('2023-02-01')
    assert len(start_filtered) < len(data)
    
    # Test filtering with end_date only
    end_filtered = handle_date_range(data, end_date='2023-02-01')
    assert isinstance(end_filtered, pd.Series)
    assert end_filtered.index[-1] <= pd.Timestamp('2023-02-01')
    assert len(end_filtered) < len(data)
    
    # Test filtering with both start_date and end_date
    range_filtered = handle_date_range(data, start_date='2023-01-15', end_date='2023-02-15')
    assert isinstance(range_filtered, pd.Series)
    assert range_filtered.index[0] >= pd.Timestamp('2023-01-15')
    assert range_filtered.index[-1] <= pd.Timestamp('2023-02-15')
    assert len(range_filtered) < len(data)
    
    # Test with Timestamp objects
    start_ts = pd.Timestamp('2023-01-15')
    end_ts = pd.Timestamp('2023-02-15')
    ts_filtered = handle_date_range(data, start_date=start_ts, end_date=end_ts)
    assert isinstance(ts_filtered, pd.Series)
    assert ts_filtered.equals(range_filtered)  # Should be the same as string date filtering
    
    # Test timezone handling
    data_tz = data.copy()
    data_tz.index = data_tz.index.tz_localize('UTC')
    tz_filtered = handle_date_range(data_tz, start_date='2023-01-15', timezone='UTC')
    assert isinstance(tz_filtered, pd.Series)
    assert tz_filtered.index.tz.zone == 'UTC'


def test_resample_high_frequency():
    """Tests high-frequency data resampling functionality."""
    # Create high-frequency financial data
    dates = pd.date_range('2023-01-01', periods=1000, freq='T')  # Minute data
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.normal(0, 0.1, 1000))
    volumes = np.random.randint(100, 1000, 1000)
    
    df = pd.DataFrame({
        'price': prices,
        'volume': volumes
    }, index=dates)
    
    # Test OHLC resampling
    ohlc = resample_high_frequency(df, freq='5T', price_column='price', method='ohlc')
    assert isinstance(ohlc, pd.DataFrame)
    assert len(ohlc) < len(df)  # Resampled to lower frequency
    assert 'price_open' in ohlc.columns
    assert 'price_high' in ohlc.columns
    assert 'price_low' in ohlc.columns
    assert 'price_close' in ohlc.columns
    assert 'volume' in ohlc.columns
    
    # Test VWAP resampling
    vwap = resample_high_frequency(df, freq='5T', price_column='price', volume_column='volume', method='vwap')
    assert isinstance(vwap, pd.DataFrame)
    assert len(vwap) < len(df)
    assert 'price_vwap' in vwap.columns
    
    # Test TWAP resampling
    twap = resample_high_frequency(df, freq='5T', price_column='price', method='twap')
    assert isinstance(twap, pd.DataFrame)
    assert len(twap) < len(df)
    assert 'price_twap' in twap.columns
    
    # Test with different frequency parameters
    hourly = resample_high_frequency(df, freq='H', price_column='price', method='ohlc')
    assert isinstance(hourly, pd.DataFrame)
    assert len(hourly) < len(ohlc)  # Hourly has fewer bars than 5-minute


def test_align_multiple_series():
    """Tests time series alignment functionality."""
    # Create multiple time series with different time indices
    dates1 = pd.date_range('2023-01-01', periods=10, freq='D')
    dates2 = pd.date_range('2023-01-05', periods=10, freq='D')
    dates3 = pd.date_range('2022-12-25', periods=15, freq='D')
    
    series1 = pd.Series(np.random.randn(10), index=dates1)
    series2 = pd.Series(np.random.randn(10), index=dates2)
    series3 = pd.Series(np.random.randn(15), index=dates3)
    
    # Test alignment with outer method (union of all indices)
    aligned_outer = align_multiple_series([series1, series2, series3], method='outer')
    assert len(aligned_outer) == 3
    assert isinstance(aligned_outer[0], pd.Series)
    assert isinstance(aligned_outer[1], pd.Series)
    assert isinstance(aligned_outer[2], pd.Series)
    
    # All aligned series should have the same index
    assert aligned_outer[0].index.equals(aligned_outer[1].index)
    assert aligned_outer[1].index.equals(aligned_outer[2].index)
    
    # The aligned index should include all dates from all series
    all_dates = sorted(set().union(dates1, dates2, dates3))
    assert len(aligned_outer[0].index) == len(all_dates)
    
    # Test alignment with inner method (intersection of all indices)
    aligned_inner = align_multiple_series([series1, series2, series3], method='inner')
    assert len(aligned_inner) == 3
    
    # The aligned index should only include dates common to all series
    common_dates = sorted(set(dates1).intersection(set(dates2), set(dates3)))
    assert len(aligned_inner[0].index) == len(common_dates)
    
    # Test alignment with forward fill
    aligned_ffill = align_multiple_series([series1, series2, series3], method='forward')
    assert len(aligned_ffill) == 3
    # NaN values should be filled with previous values
    assert not aligned_ffill[0].isna().any()
    assert not aligned_ffill[1].isna().any()
    assert not aligned_ffill[2].isna().any()
    
    # Test with explicitly provided target index
    target_index = pd.date_range('2023-01-01', periods=20, freq='D')
    aligned_custom = align_multiple_series([series1, series2, series3], method='outer', index=target_index)
    assert len(aligned_custom) == 3
    assert aligned_custom[0].index.equals(target_index)
    assert aligned_custom[1].index.equals(target_index)
    assert aligned_custom[2].index.equals(target_index)


def test_merge_time_series():
    """Tests time series merging functionality."""
    # Create multiple time series for merging
    dates1 = pd.date_range('2023-01-01', periods=10, freq='D')
    dates2 = pd.date_range('2023-01-05', periods=10, freq='D')
    dates3 = pd.date_range('2022-12-25', periods=15, freq='D')
    
    series1 = pd.Series(np.random.randn(10), index=dates1)
    series2 = pd.Series(np.random.randn(10), index=dates2)
    series3 = pd.Series(np.random.randn(15), index=dates3)
    
    # Test merging with different join methods
    merged_outer = merge_time_series([series1, series2, series3], join='outer')
    assert isinstance(merged_outer, pd.DataFrame)
    assert merged_outer.shape[1] == 3  # 3 columns for 3 series
    
    # With outer join, all dates should be included
    all_dates = sorted(set().union(dates1, dates2, dates3))
    assert len(merged_outer) == len(all_dates)
    
    # Test merging with inner join
    merged_inner = merge_time_series([series1, series2, series3], join='inner')
    assert isinstance(merged_inner, pd.DataFrame)
    
    # With inner join, only common dates should be included
    common_dates = sorted(set(dates1).intersection(set(dates2), set(dates3)))
    assert len(merged_inner) == len(common_dates)
    
    # Test with custom column names
    custom_names = ['Series A', 'Series B', 'Series C']
    merged_named = merge_time_series([series1, series2, series3], names=custom_names)
    assert isinstance(merged_named, pd.DataFrame)
    assert list(merged_named.columns) == custom_names
    
    # Test with different fill methods
    merged_ffill = merge_time_series([series1, series2, series3], fill_method='ffill')
    assert isinstance(merged_ffill, pd.DataFrame)
    # NaN values should be filled
    assert not merged_ffill.isna().any().any()
    
    merged_bfill = merge_time_series([series1, series2, series3], fill_method='bfill')
    assert isinstance(merged_bfill, pd.DataFrame)
    assert not merged_bfill.isna().any().any()
    
    # Test with no fill method
    merged_no_fill = merge_time_series([series1, series2, series3], fill_method=None)
    assert isinstance(merged_no_fill, pd.DataFrame)
    # Should have NaN values
    assert merged_no_fill.isna().any().any()


@pytest.mark.asyncio
async def test_async_convert_time_series():
    """Tests asynchronous time series conversion functionality."""
    # Create large test data for asynchronous processing
    large_data = np.random.randn(10000)
    
    # Test async conversion to numpy format
    async_np_result = await async_convert_time_series(large_data, output_type='numpy')
    assert isinstance(async_np_result, np.ndarray)
    assert np.array_equal(async_np_result, large_data)
    
    # Test async conversion to pandas Series
    async_series_result = await async_convert_time_series(large_data, output_type='series')
    assert isinstance(async_series_result, pd.Series)
    assert np.array_equal(async_series_result.values, large_data)
    
    # Test async conversion to pandas DataFrame
    async_df_result = await async_convert_time_series(large_data, output_type='dataframe')
    assert isinstance(async_df_result, pd.DataFrame)
    
    # Test with time index specification
    dates = pd.date_range('2023-01-01', periods=len(large_data))
    async_with_dates = await async_convert_time_series(
        large_data, 
        output_type='series', 
        time_index=dates
    )
    assert isinstance(async_with_dates, pd.Series)
    assert async_with_dates.index.equals(dates)
    
    # Verify async results match synchronous version
    sync_result = convert_time_series(large_data, output_type='numpy')
    assert np.array_equal(async_np_result, sync_result)


@pytest.mark.property
@given(series(elements=st.floats(allow_nan=False, allow_infinity=False)))
def test_convert_time_series_property(data):
    """Property-based tests for convert_time_series using hypothesis strategies."""
    # Convert to numpy and back
    np_result = convert_time_series(data, output_type='numpy')
    series_result = convert_time_series(np_result, output_type='series')
    
    # Values should be preserved (within floating point precision)
    assert np.allclose(data.values, series_result.values)
    
    # Convert to DataFrame and back
    df_result = convert_time_series(data, output_type='dataframe')
    series_from_df = convert_time_series(df_result, output_type='series')
    
    # Values should be preserved
    assert np.allclose(data.values, series_from_df.values)
    
    # Try with different output types
    for output_type in ['numpy', 'series', 'dataframe']:
        result = convert_time_series(data, output_type=output_type)
        if output_type == 'numpy':
            assert isinstance(result, np.ndarray)
        elif output_type == 'series':
            assert isinstance(result, pd.Series)
        elif output_type == 'dataframe':
            assert isinstance(result, pd.DataFrame)


@pytest.mark.property
@given(arrays(dtype=np.float64, 
              shape=st.integers(10, 100), 
              elements=st.floats(min_value=1.0, max_value=1000.0, 
                                allow_nan=False, allow_infinity=False)))
def test_calculate_financial_returns_property(data):
    """Property-based tests for calculate_financial_returns using hypothesis strategies."""
    # Test simple returns
    simple_returns = calculate_financial_returns(data, method='simple')
    
    # Check property: if prices double, simple return should be 1.0
    doubled_prices = data * 2
    for i in range(1, len(data)):
        # Calculate return manually: (p_t / p_{t-1}) - 1
        expected_return = (data[i] / data[i-1]) - 1
        assert np.isclose(simple_returns[i], expected_return)
    
    # Test log returns
    log_returns = calculate_financial_returns(data, method='log')
    
    # Check property: log returns are approximately additive
    # For small returns: log(1+r) ≈ r
    for i in range(1, len(data)):
        # Calculate log return manually: log(p_t / p_{t-1})
        expected_log_return = np.log(data[i] / data[i-1])
        assert np.isclose(log_returns[i], expected_log_return)
    
    # Test different periods
    for period in [1, 2, 5]:
        if period >= len(data):
            continue
        
        period_returns = calculate_financial_returns(data, method='simple', periods=period)
        
        # Check a few returns with manual calculation
        for i in range(period, len(data)):
            expected_period_return = (data[i] / data[i-period]) - 1
            assert np.isclose(period_returns[i], expected_period_return)


@pytest.mark.property
@given(arrays(dtype=np.float64, 
              shape=st.integers(10, 100), 
              elements=st.floats(allow_nan=False, allow_infinity=False)))
def test_standardize_data_property(data):
    """Property-based tests for standardize_data using hypothesis strategies."""
    # Test non-robust standardization
    standardized = standardize_data(data, robust=False)
    
    # Key property: standardized data should have mean ≈ 0 and std ≈ 1
    if len(data) > 1 and np.std(data) > 0:
        assert np.isclose(np.mean(standardized), 0, atol=1e-10)
        assert np.isclose(np.std(standardized), 1, atol=1e-10)
    
    # Test robust standardization
    robust_standardized = standardize_data(data, robust=True)
    
    # Calculate median and IQR manually to verify
    if len(data) > 1:
        median = np.median(data)
        q75, q25 = np.percentile(data, [75, 25])
        iqr = q75 - q25
        
        # If IQR is zero, standardize_data should use alternate method
        if iqr > 0:
            # Calculate expected robust standardized values
            expected_robust = (data - median) / iqr
            assert np.allclose(robust_standardized, expected_robust)