"""
Example script demonstrating multivariate volatility modeling capabilities of the MFE Toolbox,
showcasing BEKK, CCC, and DCC models for estimating, forecasting, and simulating
multivariate financial time series with correlated volatility dynamics.
"""

import argparse  # Python 3.12
import asyncio  # Python 3.12
import logging  # Python 3.12

import matplotlib.pyplot as plt  # matplotlib 3.8.0
import numpy as np  # numpy 1.26.3
import pandas as pd  # pandas 2.1.4
import seaborn as sns  # seaborn 0.13.0

# Internal imports
from mfe.models.bekk import BEKKModel  # BEKK multivariate GARCH model implementation
from mfe.models.ccc import CCCModel  # CCC multivariate volatility model implementation
from mfe.models.dcc import DCCModel  # DCC multivariate volatility model implementation
from mfe.models.multivariate import (  # Core multivariate volatility functions and classes
    MultivariateType,
    async_estimate_multivariate_volatility,
    estimate_multivariate_volatility,
    forecast_multivariate_volatility,
    simulate_multivariate_volatility,
)
from mfe.utils.data_handling import (  # Data loading and transformation utilities
    convert_to_log_returns,
    load_financial_data,
)

# Configure logging for the module
logger = logging.getLogger(__name__)

# Define global constants for data paths and default parameters
EXAMPLE_DATA_PATH = "../tests/test_data/market_benchmark.npy"
DEFAULT_MODEL_PARAMS = {
    "BEKK": {"allow_diagonal": True},
    "CCC": {"garch_orders": [(1, 1), (1, 1), (1, 1)]},
    "DCC": {"p": 1, "q": 1},
}
DISTRIBUTION_PARAMS = {"dist_type": "student", "dist_params": {"df": 8}}


def load_multivariate_data(data_path: str, n_assets: int) -> pd.DataFrame:
    """
    Loads example multivariate financial data for volatility modeling demonstration

    Parameters
    ----------
    data_path : str
        Path to the data file
    n_assets : int
        Number of assets to load

    Returns
    -------
    pd.DataFrame
        Log returns of multivariate financial data
    """
    # Load financial data using load_financial_data function with specified path
    data = load_financial_data(data_path)

    # If data is univariate, generate synthetic multivariate data with correlations
    if isinstance(data, pd.Series) or data.shape[1] == 1:
        # Generate synthetic multivariate data with correlations
        corr = np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.2], [0.3, 0.2, 1.0]])
        rng = np.random.default_rng(42)
        data = rng.multivariate_normal([0, 0, 0], corr, size=len(data))
        data = pd.DataFrame(data, columns=["Asset1", "Asset2", "Asset3"])

    # If data has more columns than n_assets, select the first n_assets columns
    if data.shape[1] > n_assets:
        data = data.iloc[:, :n_assets]

    # Calculate log returns using convert_to_log_returns
    log_returns = convert_to_log_returns(data)

    # Return pandas DataFrame of multivariate log returns with proper column names
    log_returns.columns = [f"Asset{i+1}" for i in range(n_assets)]
    return log_returns


def run_bekk_example(returns: pd.DataFrame, model_params: dict, use_async: bool) -> tuple:
    """
    Demonstrates BEKK multivariate volatility model estimation, forecasting, and analysis

    Parameters
    ----------
    returns : pd.DataFrame
        Multivariate return series
    model_params : dict
        Dictionary of BEKK model parameters
    use_async : bool
        Flag to use asynchronous estimation

    Returns
    -------
    tuple
        (model, result, forecast) - Estimated model, estimation result, and forecast
    """
    # Print section header for BEKK example
    print("\n--- BEKK Model Example ---")

    # Extract BEKK-specific parameters from model_params
    bekk_params = model_params.get("BEKK", {})

    # If use_async is True, use async_estimate_multivariate_volatility and await result
    if use_async:
        print("Estimating BEKK model asynchronously...")
        result = asyncio.run(async_estimate_multivariate_volatility(returns, MultivariateType.BEKK.value, bekk_params))
    # Otherwise, use synchronous estimate_multivariate_volatility function
    else:
        print("Estimating BEKK model synchronously...")
        result = estimate_multivariate_volatility(returns, MultivariateType.BEKK.value, bekk_params)

    # Print model summary information including parameters and statistics
    print("\nBEKK Model Summary:")
    print(result.summary())

    # Generate volatility forecast for 10 periods ahead using forecast_multivariate_volatility
    forecast = forecast_multivariate_volatility(result, horizon=10)

    # Print forecast summary information
    print("\nBEKK Model Forecast:")
    print(forecast.forecast_covariances)

    # Return the model, estimation result, and forecast as a tuple
    return BEKKModel, result, forecast


def run_ccc_example(returns: pd.DataFrame, model_params: dict, use_async: bool) -> tuple:
    """
    Demonstrates CCC multivariate volatility model estimation, forecasting, and analysis

    Parameters
    ----------
    returns : pd.DataFrame
        Multivariate return series
    model_params : dict
        Dictionary of CCC model parameters
    use_async : bool
        Flag to use asynchronous estimation

    Returns
    -------
    tuple
        (model, result, forecast) - Estimated model, estimation result, and forecast
    """
    # Print section header for CCC example
    print("\n--- CCC Model Example ---")

    # Extract CCC-specific parameters from model_params
    ccc_params = model_params.get("CCC", {})

    # If use_async is True, use async_estimate_multivariate_volatility and await result
    if use_async:
        print("Estimating CCC model asynchronously...")
        result = asyncio.run(async_estimate_multivariate_volatility(returns, MultivariateType.CCC.value, ccc_params))
    # Otherwise, use synchronous estimate_multivariate_volatility function
    else:
        print("Estimating CCC model synchronously...")
        result = estimate_multivariate_volatility(returns, MultivariateType.CCC.value, ccc_params)

    # Print model summary information including parameters and statistics
    print("\nCCC Model Summary:")
    print(result.summary())

    # Generate volatility forecast for 10 periods ahead using forecast_multivariate_volatility
    forecast = forecast_multivariate_volatility(result, horizon=10)

    # Print correlation matrix and other CCC-specific information
    print("\nCCC Model Forecast:")
    print(forecast.forecast_covariances)

    # Return the model, estimation result, and forecast as a tuple
    return CCCModel, result, forecast


def run_dcc_example(returns: pd.DataFrame, model_params: dict, use_async: bool) -> tuple:
    """
    Demonstrates DCC multivariate volatility model estimation, forecasting, and analysis

    Parameters
    ----------
    returns : pd.DataFrame
        Multivariate return series
    model_params : dict
        Dictionary of DCC model parameters
    use_async : bool
        Flag to use asynchronous estimation

    Returns
    -------
    tuple
        (model, result, forecast) - Estimated model, estimation result, and forecast
    """
    # Print section header for DCC example
    print("\n--- DCC Model Example ---")

    # Extract DCC-specific parameters from model_params
    dcc_params = model_params.get("DCC", {})

    # If use_async is True, use async_estimate_multivariate_volatility and await result
    if use_async:
        print("Estimating DCC model asynchronously...")
        result = asyncio.run(async_estimate_multivariate_volatility(returns, MultivariateType.DCC.value, dcc_params))
    # Otherwise, use synchronous estimate_multivariate_volatility function
    else:
        print("Estimating DCC model synchronously...")
        result = estimate_multivariate_volatility(returns, MultivariateType.DCC.value, dcc_params)

    # Print model summary information including parameters and statistics
    print("\nDCC Model Summary:")
    print(result.summary())

    # Generate volatility forecast for 10 periods ahead using forecast_multivariate_volatility
    forecast = forecast_multivariate_volatility(result, horizon=10)

    # Print time-varying correlation dynamics and DCC-specific information
    print("\nDCC Model Forecast:")
    print(forecast.forecast_covariances)

    # Return the model, estimation result, and forecast as a tuple
    return DCCModel, result, forecast


def run_simulation_example(model_type: str, model_params: dict, n_assets: int, n_periods: int) -> tuple:
    """
    Demonstrates simulation capabilities of multivariate volatility models

    Parameters
    ----------
    model_type : str
        Type of multivariate model to simulate
    model_params : dict
        Dictionary of model parameters
    n_assets : int
        Number of assets to simulate
    n_periods : int
        Number of periods to simulate

    Returns
    -------
    tuple
        (simulated_returns, simulated_covariances) - Simulated data
    """
    # Print section header for simulation example
    print("\n--- Simulation Example ---")

    # Extract model-specific parameters from model_params
    simulation_params = model_params.get(model_type, {})

    # Configure simulation parameters including distribution type
    distribution_params = DISTRIBUTION_PARAMS

    # Call simulate_multivariate_volatility with appropriate parameters
    simulated_returns, simulated_covariances = simulate_multivariate_volatility(
        model_type=model_type,
        params=simulation_params,
        n_obs=n_periods,
        n_assets=n_assets,
        dist_type=distribution_params["dist_type"],
        dist_params=distribution_params["dist_params"],
    )

    # Print summary statistics of simulated data
    print("\nSimulated Returns:")
    print(simulated_returns)
    print("\nSimulated Covariances:")
    print(simulated_covariances)

    # Return the simulated returns and covariances as a tuple
    return simulated_returns, simulated_covariances


def visualize_multivariate_results(returns: pd.DataFrame, bekk_results: object, ccc_results: object, dcc_results: object, save_plots: bool) -> None:
    """
    Creates and displays visualizations of multivariate volatility model results

    Parameters
    ----------
    returns : pd.DataFrame
        Original multivariate returns series
    bekk_results : object
        Results from BEKK model estimation
    ccc_results : object
        Results from CCC model estimation
    dcc_results : object
        Results from DCC model estimation
    save_plots : bool
        Flag to save plots to files
    """
    # Create a figure layout using matplotlib with multiple subplots
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 12))
    fig.suptitle("Multivariate Volatility Model Results", fontsize=16)

    # Plot 1: Original multivariate returns series
    returns.plot(ax=axes[0, 0], title="Original Returns Series")
    axes[0, 0].set_ylabel("Returns")

    # Plot 2: Conditional volatilities from each model for comparison
    # (Simplified for demonstration - plot a single asset's volatility)
    asset_index = 0  # Choose the first asset for demonstration
    axes[0, 1].plot(bekk_results.volatilities[:, asset_index], label="BEKK", color="blue")
    axes[0, 1].plot(ccc_results.volatilities[:, asset_index], label="CCC", color="red")
    axes[0, 1].plot(dcc_results.volatilities[:, asset_index], label="DCC", color="green")
    axes[0, 1].set_title("Conditional Volatilities (Asset 1)")
    axes[0, 1].set_ylabel("Volatility")
    axes[0, 1].legend()

    # Plot 3: Correlation matrices/dynamics visualization
    # (Simplified for demonstration - plot a heatmap of the last correlation matrix)
    sns.heatmap(ccc_results.parameters.constant_correlation, annot=True, cmap="coolwarm", ax=axes[0, 2])
    axes[0, 2].set_title("Constant Correlation Matrix (CCC)")

    # Plot 4: Standardized residuals analysis
    # (Simplified for demonstration - plot a histogram of standardized residuals for a single asset)
    sns.histplot(bekk_results.standardized_residuals[:, asset_index], ax=axes[1, 0], kde=True)
    axes[1, 0].set_title("Standardized Residuals (BEKK - Asset 1)")
    axes[1, 0].set_xlabel("Standardized Residuals")

    # Plot 5: Forecast comparison between models
    # (Simplified for demonstration - plot a single element of the forecast covariance)
    horizon = 10
    bekk_forecast = forecast_multivariate_volatility(bekk_results, horizon=horizon).forecast_covariances[:, 0, 0]
    ccc_forecast = forecast_multivariate_volatility(ccc_results, horizon=horizon).forecast_covariances[:, 0, 0]
    dcc_forecast = forecast_multivariate_volatility(dcc_results, horizon=horizon).forecast_covariances[:, 0, 0]
    axes[1, 1].plot(bekk_forecast, label="BEKK", color="blue")
    axes[1, 1].plot(ccc_forecast, label="CCC", color="red")
    axes[1, 1].plot(dcc_forecast, label="DCC", color="green")
    axes[1, 1].set_title("Forecast Comparison (Covariance 1,1)")
    axes[1, 1].set_xlabel("Horizon")
    axes[1, 1].set_ylabel("Covariance")
    axes[1, 1].legend()

    # Plot 6: Impulse response analysis if relevant
    # (Placeholder - implement if impulse response analysis is added)
    axes[1, 2].text(0.5, 0.5, "Impulse Response Analysis (Future)", ha="center", va="center")
    axes[1, 2].axis("off")

    # If save_plots is True, save figures to files
    if save_plots:
        plt.savefig("multivariate_volatility_results.png")

    # Display plots if not in non-interactive mode
    plt.show()


def main() -> int:
    """
    Main entry point for the multivariate volatility example script
    """
    # Parse command line arguments using argparse
    parser = argparse.ArgumentParser(description="Multivariate Volatility Model Example")

    # Set up argument options for data path, model parameters, async mode, and saving plots
    parser.add_argument("--data_path", type=str, default=EXAMPLE_DATA_PATH, help="Path to the data file")
    parser.add_argument("--n_assets", type=int, default=3, help="Number of assets to model")
    parser.add_argument("--model_params", type=str, default=DEFAULT_MODEL_PARAMS, help="Model parameters (JSON string)")
    parser.add_argument("--use_async", action="store_true", help="Use asynchronous estimation")
    parser.add_argument("--save_plots", action="store_true", help="Save plots to files")

    # Parse arguments
    args = parser.parse_args()

    # Load example multivariate data using load_multivariate_data function
    returns = load_multivariate_data(args.data_path, args.n_assets)

    # Run BEKK example using run_bekk_example function
    BEKKModel, bekk_result, bekk_forecast = run_bekk_example(returns, args.model_params, args.use_async)

    # Run CCC example using run_ccc_example function
    CCCModel, ccc_result, ccc_forecast = run_ccc_example(returns, args.model_params, args.use_async)

    # Run DCC example using run_dcc_example function
    DCCModel, dcc_result, dcc_forecast = run_dcc_example(returns, args.model_params, args.use_async)

    # Run simulation example using run_simulation_example function
    simulated_returns, simulated_covariances = run_simulation_example("BEKK", args.model_params, args.n_assets, len(returns))

    # Visualize results using visualize_multivariate_results function
    visualize_multivariate_results(returns, bekk_result, ccc_result, dcc_result, args.save_plots)

    # Return exit code 0 for successful execution
    return 0


@asyncio.coroutine
def async_main() -> int:
    """
    Asynchronous version of the main entry point for the multivariate volatility example
    """
    # Parse command line arguments using argparse
    parser = argparse.ArgumentParser(description="Multivariate Volatility Model Example")

    # Set up argument options for data path, model parameters, async mode, and saving plots
    parser.add_argument("--data_path", type=str, default=EXAMPLE_DATA_PATH, help="Path to the data file")
    parser.add_argument("--n_assets", type=int, default=3, help="Number of assets to model")
    parser.add_argument("--model_params", type=str, default=DEFAULT_MODEL_PARAMS, help="Model parameters (JSON string)")
    parser.add_argument("--use_async", action="store_true", help="Use asynchronous estimation")
    parser.add_argument("--save_plots", action="store_true", help="Save plots to files")

    # Parse arguments
    args = parser.parse_args()

    # Load example multivariate data using load_multivariate_data function
    returns = load_multivariate_data(args.data_path, args.n_assets)

    # Run BEKK example using run_bekk_example function with use_async=True
    bekk_task = asyncio.create_task(run_bekk_example(returns, args.model_params, True))

    # Run CCC example using run_ccc_example function with use_async=True
    ccc_task = asyncio.create_task(run_ccc_example(returns, args.model_params, True))

    # Run DCC example using run_dcc_example function with use_async=True
    dcc_task = asyncio.create_task(run_dcc_example(returns, args.model_params, True))

    # Await all asynchronous operations using asyncio.gather
    BEKKModel, bekk_result, bekk_forecast = await bekk_task
    CCCModel, ccc_result, ccc_forecast = await ccc_task
    DCCModel, dcc_result, dcc_forecast = await dcc_task

    # Run simulation example using run_simulation_example function
    simulated_returns, simulated_covariances = run_simulation_example("BEKK", args.model_params, args.n_assets, len(returns))

    # Visualize results using visualize_multivariate_results function
    visualize_multivariate_results(returns, bekk_result, ccc_result, dcc_result, args.save_plots)

    # Return exit code 0 for successful execution
    return 0


if __name__ == "__main__":
    # Parse command line arguments to check for async mode
    parser = argparse.ArgumentParser(description="Multivariate Volatility Model Example")
    parser.add_argument("--use_async", action="store_true", help="Use asynchronous estimation")
    args, unknown = parser.parse_known_args()

    try:
        # If async mode is requested, run asyncio.run(async_main())
        if args.use_async:
            asyncio.run(async_main())
        # Otherwise, run synchronous main() function
        else:
            main()
    except Exception as e:
        print(f"An error occurred: {e}")
        exit(1)

    # Exit with appropriate status code
    exit(0)