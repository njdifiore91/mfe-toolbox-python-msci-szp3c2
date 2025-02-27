import asyncio  # version: standard library
from datetime import datetime  # version: standard library
from typing import Dict  # version: standard library

import numpy as np  # version: 1.26.3
import pandas as pd  # version: 2.1.4
import scipy  # version: 1.11.4
import statsmodels.api as statsmodels  # version: 0.14.1
from matplotlib import pyplot as plt  # version: 3.8.2

# Internal imports
from mfe.models.bekk import BEKKModel  # Import the BEKK multivariate GARCH model for demonstration
from mfe.models.ccc import CCCModel  # Import the Constant Conditional Correlation multivariate GARCH model
from mfe.models.dcc import DCCModel  # Import the Dynamic Conditional Correlation multivariate GARCH model
from mfe.utils.validation import validate_multivariate_returns  # Utility for validating multivariate financial returns data
from mfe.utils.data_handling import generate_multivariate_returns  # Utility for generating synthetic multivariate return data

# Global constants
SEED = 42
NUM_ASSETS = 3
SAMPLE_SIZE = 1000
FORECAST_HORIZON = 10


def setup_data(n_assets: int, sample_size: int, seed: int) -> pd.DataFrame:
    """
    Generates synthetic multivariate return data for model testing.

    Parameters
    ----------
    n_assets : int
        Number of assets in the multivariate return series
    sample_size : int
        Number of observations to generate
    seed : int
        Random seed for reproducibility

    Returns
    -------
    pd.DataFrame
        DataFrame containing multivariate asset returns
    """
    # Set random seed for reproducibility
    np.random.seed(seed)

    # Generate date range
    dates = pd.date_range(start='2020-01-01', periods=sample_size, freq='B')

    # Call generate_multivariate_returns to create synthetic return data with correlation structure
    returns_data = generate_multivariate_returns(n_assets, sample_size)

    # Create a DataFrame with generated returns and date index
    returns = pd.DataFrame(returns_data, index=dates)

    # Validate the returns data
    validate_multivariate_returns(returns)

    # Return the DataFrame
    return returns


def run_bekk_example(returns: pd.DataFrame) -> Dict:
    """
    Demonstrates the BEKK multivariate GARCH model estimation and forecasting.

    Parameters
    ----------
    returns : pd.DataFrame
        Multivariate return series

    Returns
    -------
    dict
        Results dictionary containing fitted model, forecasts, and performance metrics
    """
    # Print section header for BEKK model demonstration
    print("\n--- BEKK Model Example ---")

    # Initialize BEKK model with p=1, q=1, k=NUM_ASSETS parameters
    model = BEKKModel(n_assets=NUM_ASSETS)

    # Fit the model to the returns data
    results = model.fit(returns)

    # Print model summary, parameters, and log-likelihood
    print(results.summary())

    # Generate volatility forecasts for FORECAST_HORIZON periods
    forecasts = model.forecast(horizon=FORECAST_HORIZON)
    print("\nBEKK Forecasts:\n", forecasts.forecast_covariances)

    # Plot the forecasted conditional variances using matplotlib
    plt.figure(figsize=(10, 6))
    for i in range(NUM_ASSETS):
        plt.plot(forecasts.forecast_covariances[:, i, i], label=f'Asset {i+1}')
    plt.title('BEKK Forecasted Conditional Variances')
    plt.xlabel('Forecast Horizon')
    plt.ylabel('Conditional Variance')
    plt.legend()
    plt.show()

    # Return dictionary with model, forecasts, and metrics
    return {'model': model, 'results': results, 'forecasts': forecasts}


def run_ccc_example(returns: pd.DataFrame) -> Dict:
    """
    Demonstrates the Constant Conditional Correlation (CCC) multivariate GARCH model.

    Parameters
    ----------
    returns : pd.DataFrame
        Multivariate return series

    Returns
    -------
    dict
        Results dictionary containing fitted model, forecasts, and performance metrics
    """
    # Print section header for CCC model demonstration
    print("\n--- CCC Model Example ---")

    # Initialize CCC model with p=1, q=1, k=NUM_ASSETS parameters
    model = CCCModel(num_assets=NUM_ASSETS)

    # Fit the model to the returns data
    results = model.fit(returns)

    # Print model summary, constant correlation matrix, and log-likelihood
    print(results.summary())

    # Generate volatility forecasts for FORECAST_HORIZON periods
    forecasts = model.forecast(horizon=FORECAST_HORIZON)
    print("\nCCC Forecasts:\n", forecasts.forecast_covariances)

    # Plot the forecasted conditional variances using matplotlib
    plt.figure(figsize=(10, 6))
    for i in range(NUM_ASSETS):
        plt.plot(forecasts.forecast_covariances[:, i, i], label=f'Asset {i+1}')
    plt.title('CCC Forecasted Conditional Variances')
    plt.xlabel('Forecast Horizon')
    plt.ylabel('Conditional Variance')
    plt.legend()
    plt.show()

    # Return dictionary with model, forecasts, and metrics
    return {'model': model, 'results': results, 'forecasts': forecasts}


def run_dcc_example(returns: pd.DataFrame) -> Dict:
    """
    Demonstrates the Dynamic Conditional Correlation (DCC) multivariate GARCH model.

    Parameters
    ----------
    returns : pd.DataFrame
        Multivariate return series

    Returns
    -------
    dict
        Results dictionary containing fitted model, forecasts, and performance metrics
    """
    # Print section header for DCC model demonstration
    print("\n--- DCC Model Example ---")

    # Initialize DCC model with p=1, q=1, k=NUM_ASSETS parameters
    model = DCCModel(p=1, q=1)

    # Fit the model to the returns data
    results = model.fit(returns)

    # Print model summary, parameters, and log-likelihood
    print(results.summary())

    # Generate volatility forecasts for FORECAST_HORIZON periods
    forecasts = model.forecast(horizon=FORECAST_HORIZON)
    print("\nDCC Forecasts:\n", forecasts)

    # Plot the dynamic conditional correlations using matplotlib
    plt.figure(figsize=(10, 6))
    for i in range(NUM_ASSETS):
        for j in range(i, NUM_ASSETS):
            plt.plot(forecasts[:, i, j], label=f'Corr({i+1},{j+1})')
    plt.title('DCC Forecasted Dynamic Conditional Correlations')
    plt.xlabel('Forecast Horizon')
    plt.ylabel('Correlation')
    plt.legend()
    plt.show()

    # Return dictionary with model, forecasts, and metrics
    return {'model': model, 'results': results, 'forecasts': forecasts}


def compare_models(bekk_results: Dict, ccc_results: Dict, dcc_results: Dict) -> pd.DataFrame:
    """
    Compares the forecasting performance of different multivariate GARCH models.

    Parameters
    ----------
    bekk_results : dict
        Results from the BEKK model
    ccc_results : dict
        Results from the CCC model
    dcc_results : dict
        Results from the DCC model

    Returns
    -------
    pd.DataFrame
        DataFrame containing comparison metrics
    """
    # Extract log-likelihood values from each model result
    bekk_ll = bekk_results['results'].log_likelihood
    ccc_ll = ccc_results['results'].log_likelihood
    dcc_ll = dcc_results['results'].likelihood

    # Extract information criteria (AIC, BIC) from each model result
    bekk_aic = bekk_results['results'].information_criteria['aic']
    ccc_aic = ccc_results['results'].information_criteria['aic']
    dcc_aic = dcc_results['results'].diagnostics['information_criteria']['aic']

    bekk_bic = bekk_results['results'].information_criteria['bic']
    ccc_bic = ccc_results['results'].information_criteria['bic']
    dcc_bic = dcc_results['results'].diagnostics['information_criteria']['bic']

    # Calculate number of parameters for each model
    num_assets = bekk_results['model'].n_assets
    bekk_params = num_assets * (num_assets + 1) // 2 + 2 * num_assets**2
    ccc_params = num_assets * (num_assets - 1) // 2 + 2 * num_assets
    dcc_params = 2  # DCC parameters a and b

    # Create a comparison DataFrame with all metrics
    comparison_data = {
        'Log-Likelihood': [bekk_ll, ccc_ll, dcc_ll],
        'AIC': [bekk_aic, ccc_aic, dcc_aic],
        'BIC': [bekk_bic, ccc_bic, dcc_bic],
        'Num Params': [bekk_params, ccc_params, dcc_params]
    }
    comparison_df = pd.DataFrame(comparison_data, index=['BEKK', 'CCC', 'DCC'])

    # Print the comparison table
    print("\n--- Model Comparison ---")
    print(comparison_df)

    # Return the comparison DataFrame
    return comparison_df


async def async_model_estimation(returns: pd.DataFrame) -> Dict:
    """
    Demonstrates asynchronous estimation of multiple multivariate volatility models.

    Parameters
    ----------
    returns : pd.DataFrame
        Multivariate return series

    Returns
    -------
    dict
        Dictionary containing all model results
    """
    # Print section header for async estimation example
    print("\n--- Asynchronous Model Estimation Example ---")

    # Initialize all three model types (BEKK, CCC, DCC)
    bekk_model = BEKKModel(n_assets=NUM_ASSETS)
    ccc_model = CCCModel(num_assets=NUM_ASSETS)
    dcc_model = DCCModel(p=1, q=1)

    # Create async tasks for fitting each model using fit_async method
    bekk_task = asyncio.create_task(bekk_model.fit_async(returns), name="BEKK")
    ccc_task = asyncio.create_task(ccc_model.fit_async(returns), name="CCC")
    dcc_task = asyncio.create_task(dcc_model.fit_async(returns), name="DCC")

    # Use asyncio.gather to execute all fitting tasks concurrently
    start_time = asyncio.get_event_loop().time()
    bekk_results, ccc_results, dcc_results = await asyncio.gather(bekk_task, ccc_task, dcc_task)
    end_time = asyncio.get_event_loop().time()

    # Collect results and compute execution time
    execution_time = end_time - start_time
    print(f"\nAsync Estimation Time: {execution_time:.2f} seconds")

    # Compare with sequential execution time
    start_time_seq = asyncio.get_event_loop().time()
    bekk_results_seq = BEKKModel(n_assets=NUM_ASSETS).fit(returns)
    ccc_results_seq = CCCModel(n_assets=NUM_ASSETS).fit(returns)
    dcc_results_seq = DCCModel(p=1, q=1).fit(returns)
    end_time_seq = asyncio.get_event_loop().time()
    execution_time_seq = end_time_seq - start_time_seq
    print(f"Sequential Estimation Time: {execution_time_seq:.2f} seconds")

    # Return dictionary of all fitted models
    return {'bekk': bekk_results, 'ccc': ccc_results, 'dcc': dcc_results}


def plot_forecasts_comparison(bekk_results: Dict, ccc_results: Dict, dcc_results: Dict, asset_index: int):
    """
    Creates comparative plots of forecasts from different multivariate models.

    Parameters
    ----------
    bekk_results : dict
        Results from the BEKK model
    ccc_results : dict
        Results from the CCC model
    dcc_results : dict
        Results from the DCC model
    asset_index : int
        Index of the asset to plot
    """
    # Create a new matplotlib figure with appropriate size
    plt.figure(figsize=(12, 8))

    # Extract conditional variance forecasts for the specified asset from each model
    bekk_forecasts = bekk_results['forecasts'].forecast_covariances[:, asset_index, asset_index]
    ccc_forecasts = ccc_results['forecasts'].forecast_covariances[:, asset_index, asset_index]
    dcc_forecasts = dcc_results['forecasts'][:, asset_index, asset_index]

    # Plot the forecast series on the same graph with different colors/styles
    plt.plot(bekk_forecasts, label='BEKK', color='blue')
    plt.plot(ccc_forecasts, label='CCC', color='red')
    plt.plot(dcc_forecasts, label='DCC', color='green')

    # Add legend, labels, and title to the plot
    plt.title(f'Volatility Forecasts Comparison - Asset {asset_index + 1}')
    plt.xlabel('Forecast Horizon')
    plt.ylabel('Conditional Variance')
    plt.legend()

    # Display the comparative forecast plot
    plt.show()


def main():
    """
    Main execution function that runs all examples and demonstrates async capabilities.
    """
    # Setup synthetic multivariate return data
    returns = setup_data(n_assets=NUM_ASSETS, sample_size=SAMPLE_SIZE, seed=SEED)

    # Run sequential model examples (BEKK, CCC, DCC)
    bekk_results = run_bekk_example(returns)
    ccc_results = run_ccc_example(returns)
    dcc_results = run_dcc_example(returns)

    # Compare sequential model results
    compare_models(bekk_results, ccc_results, dcc_results)

    # Run async example using asyncio.run(async_model_estimation(returns))
    async_results = asyncio.run(async_model_estimation(returns))

    # Create comparative forecast plots
    plot_forecasts_comparison(bekk_results, ccc_results, dcc_results, asset_index=0)

    # Print final conclusions and observations
    print("\n--- Conclusions ---")
    print("The examples demonstrate the use of BEKK, CCC, and DCC models for multivariate volatility modeling.")
    print("Asynchronous execution can significantly reduce computation time when fitting multiple models.")


if __name__ == "__main__":
    main()