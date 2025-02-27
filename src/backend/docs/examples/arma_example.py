"""
Example demonstrating ARMA time series modeling using the MFE Toolbox.

This script provides a comprehensive example of how to use the MFE Toolbox
for ARMA time series modeling, including data generation, model fitting,
diagnostic testing, and forecasting. It showcases both synchronous and
asynchronous execution patterns.
"""

import numpy as np  # numpy 1.26.3
import pandas as pd  # pandas 2.1.4
import matplotlib.pyplot as plt  # matplotlib 3.8.2
import statsmodels.graphics.tsaplots as smt  # statsmodels 0.14.1
import asyncio  # built-in
from typing import List  # Python 3.12

# Internal imports from the MFE Toolbox
from mfe.models.arma import ARMA  # Import the ARMA model class
from mfe.models.armax import ARMAX  # Import the ARMAX model class
from mfe.utils.data_handling import load_dataset  # Function to load example datasets
from mfe.utils.validation import validate_timeseries  # Function to validate time series data
from mfe.utils.numpy_helpers import to_numpy_array  # Function to convert inputs to NumPy arrays
from mfe.core.testing import ljung_box_test  # Function to perform Ljung-Box test


def generate_example_data(n_samples: int, ar_params: List[float], ma_params: List[float], sigma: float) -> pd.DataFrame:
    """
    Generates synthetic time series data for ARMA model demonstration.

    Parameters
    ----------
    n_samples : int
        Number of data points to generate.
    ar_params : List[float]
        List of AR parameters.
    ma_params : List[float]
        List of MA parameters.
    sigma : float
        Standard deviation of the white noise.

    Returns
    -------
    pandas.DataFrame
        Time series data with date index.
    """
    # Import necessary libraries
    # Create random number generator
    rng = np.random.default_rng(seed=42)
    # Generate white noise innovations
    white_noise = sigma * rng.standard_normal(size=n_samples)
    # Initialize time series with zeros
    time_series = np.zeros(n_samples)
    # Implement AR component by iterating through AR lags
    for t in range(len(ar_params), n_samples):
        ar_component = np.sum(ar_params * time_series[t - len(ar_params):t][::-1])
        time_series[t] += ar_component
    # Implement MA component by iterating through MA lags
    for t in range(len(ma_params), n_samples):
        ma_component = np.sum(ma_params * white_noise[t - len(ma_params):t][::-1])
        time_series[t] += ma_component
    # Create date range index
    date_index = pd.date_range(start='2023-01-01', periods=n_samples, freq='D')
    # Return data as Pandas DataFrame with date index
    return pd.DataFrame(time_series, index=date_index, columns=['Example Data'])


def run_arma_example():
    """
    Demonstrates complete workflow for ARMA model fitting, diagnostics and forecasting.
    """
    # Generate example time series data
    data = generate_example_data(n_samples=200, ar_params=[0.6, 0.3], ma_params=[0.4], sigma=0.5)
    # Validate time series data for ARMA modeling
    validated_data = validate_timeseries(data['Example Data'].values)
    # Create and configure ARMA model
    arma_model = ARMA(p=2, q=1)
    # Fit model to data
    arma_model.estimate(validated_data)
    # Display model summary and parameters
    print("\nARMA Model Summary:")
    print(arma_model.summary())
    # Perform diagnostic tests on residuals
    print("\nDiagnostic Tests:")
    lb_statistic, lb_pvalue = ljung_box_test(arma_model.residuals, lags=10)
    print(f"Ljung-Box Test: statistic={lb_statistic:.4f}, p-value={lb_pvalue:.4f}")
    # Plot diagnostic plots (residuals, ACF, PACF)
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    axes[0].plot(arma_model.residuals)
    axes[0].set_title('Residuals')
    smt.plot_acf(arma_model.residuals, ax=axes[1], lags=30, title='ACF')
    smt.plot_pacf(arma_model.residuals, ax=axes[2], lags=30, title='PACF')
    plt.tight_layout()
    plt.show()
    # Perform forecasting
    forecast = arma_model.forecast(validated_data, steps=20)
    # Plot original data with forecasts
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data['Example Data'], label='Original Data')
    forecast_index = pd.date_range(start=data.index[-1], periods=21, freq='D')[1:]
    plt.plot(forecast_index, forecast, label='Forecast', color='red')
    plt.title('ARMA Model Forecast')
    plt.legend()
    plt.show()

def run_armax_example():
    """
    Demonstrates ARMAX model with exogenous variables.
    """
    # Generate example time series data
    data = generate_example_data(n_samples=200, ar_params=[0.6, 0.3], ma_params=[0.4], sigma=0.5)
    # Generate exogenous variable data
    exog_data = np.random.rand(200, 1)
    # Validate time series and exogenous data
    validated_data = validate_timeseries(data['Example Data'].values)
    validated_exog = validate_timeseries(exog_data)
    # Create and configure ARMAX model
    armax_model = ARMAX(p=1, q=1)
    # Fit model to data with exogenous variables
    armax_model.estimate(validated_data, exog=validated_exog)
    # Display model summary and parameters
    print("\nARMAX Model Summary:")
    print(armax_model.summary())
    # Perform diagnostic tests on residuals
    print("\nDiagnostic Tests:")
    lb_statistic, lb_pvalue = ljung_box_test(armax_model.residuals, lags=10)
    print(f"Ljung-Box Test: statistic={lb_statistic:.4f}, p-value={lb_pvalue:.4f}")
    # Generate future exogenous variables
    exog_forecast = np.random.rand(20, 1)
    # Perform forecasting with exogenous variables
    forecast = armax_model.forecast(validated_data, steps=20, exog_forecast=exog_forecast)
    # Plot original data with forecasts
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data['Example Data'], label='Original Data')
    forecast_index = pd.date_range(start=data.index[-1], periods=21, freq='D')[1:]
    plt.plot(forecast_index, forecast, label='Forecast', color='red')
    plt.title('ARMAX Model Forecast with Exogenous Variables')
    plt.legend()
    plt.show()

async def async_arma_example():
    """
    Demonstrates asynchronous ARMA model fitting and forecasting using async/await pattern.
    """
    # Generate example time series data
    data = generate_example_data(n_samples=200, ar_params=[0.6, 0.3], ma_params=[0.4], sigma=0.5)
    # Validate time series data
    validated_data = validate_timeseries(data['Example Data'].values)
    # Create and configure ARMA model
    arma_model = ARMA(p=2, q=1)
    # Asynchronously fit model to data using await
    print("\nAsynchronous ARMA Model Fitting:")
    await arma_model.estimate_async(validated_data)
    # Display progress during fitting
    # Display model summary when complete
    print(arma_model.summary())
    # Asynchronously generate forecasts
    forecast = await arma_model.forecast_async(validated_data, steps=20)
    # Plot original data with forecasts
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data['Example Data'], label='Original Data')
    forecast_index = pd.date_range(start=data.index[-1], periods=21, freq='D')[1:]
    plt.plot(forecast_index, forecast, label='Forecast', color='red')
    plt.title('Asynchronous ARMA Model Forecast')
    plt.legend()
    plt.show()

def main():
    """
    Main execution function that runs all examples.
    """
    # Print welcome message and explanation
    print("Running ARMA and ARMAX examples using the MFE Toolbox...")
    # Run standard ARMA example
    run_arma_example()
    # Run ARMAX example with exogenous variables
    run_armax_example()
    # Set up and run async examples using asyncio event loop
    asyncio.run(async_arma_example())
    # Show completion message
    print("\nARMA and ARMAX examples completed.")

if __name__ == '__main__':
    main()