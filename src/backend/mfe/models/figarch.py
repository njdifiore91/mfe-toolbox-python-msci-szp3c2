"""
MFE Toolbox - Fractionally Integrated GARCH (FIGARCH) Model

This module implements the Fractionally Integrated Generalized Autoregressive 
Conditional Heteroskedasticity (FIGARCH) model for volatility in financial time series.
The FIGARCH model allows for long memory in volatility processes, which is characterized 
by hyperbolic decay in the autocorrelation function.

The implementation uses Numba for performance optimization and provides async support
for long-running estimation tasks.
"""

import numpy as np  # numpy 1.26.3
from numba import jit  # numba 0.59.0
from dataclasses import dataclass, field  # Python 3.12
from scipy.special import gamma, gammaln  # scipy 1.11.4
from scipy.optimize import minimize  # scipy 1.11.4
from typing import Optional, Union, Dict, List, Tuple, Callable  # Python 3.12
import copy  # Python 3.12
import asyncio  # Python 3.12

# Internal imports
from .volatility import UnivariateVolatilityModel, calculate_log_likelihood
from ..utils.validation import validate_array, is_positive_float, check_range
from ..utils.numba_helpers import optimized_jit
from ..utils.async_helpers import run_in_executor, AsyncTask
from ..core.distributions import distribution_forecast
from ..core.optimization import Optimizer


@optimized_jit(nopython=True)
def figarch_recursion(d: float, phi: float, beta: float, max_lag: int) -> np.ndarray:
    """
    Compute FIGARCH recursion coefficients for volatility calculation. Optimized with Numba for performance.
    
    Parameters
    ----------
    d : float
        Fractional integration parameter (0 < d < 1)
    phi : float
        GARCH autoregressive parameter
    beta : float
        GARCH moving average parameter
    max_lag : int
        Maximum lag for recursion (truncation point)
        
    Returns
    -------
    np.ndarray
        Array of recursion coefficients for FIGARCH process
    """
    # Initialize coefficient array
    lambda_coef = np.zeros(max_lag)
    
    # Set first coefficient
    lambda_coef[0] = (1 - phi - beta) * d
    
    # Implement recursive formula for fractional differencing
    for i in range(1, max_lag):
        # Calculate lambda_k using the formula with gamma functions
        frac_diff = d * gamma(i - d) / (gamma(1 - d) * gamma(i + 1))
        
        # Apply FIGARCH-specific polynomial adjustments with phi and beta parameters
        lambda_coef[i] = frac_diff
        
        if i == 1:
            lambda_coef[i] += (phi - beta) * lambda_coef[0]
        else:
            lambda_coef[i] += phi * lambda_coef[i-1] - beta * frac_diff
    
    return lambda_coef


@optimized_jit(nopython=True)
def figarch_variance(returns: np.ndarray, omega: float, d: float, phi: float, beta: float, max_lag: int) -> np.ndarray:
    """
    Calculate conditional variance series for FIGARCH model based on returns data. Optimized with Numba for performance.
    
    Parameters
    ----------
    returns : np.ndarray
        Array of return data
    omega : float
        Constant term in the variance equation (must be positive)
    d : float
        Fractional integration parameter (0 < d < 1)
    phi : float
        GARCH autoregressive parameter (0 <= phi <= 1)
    beta : float
        GARCH moving average parameter (0 <= beta <= 1)
    max_lag : int
        Maximum lag for truncation of infinite ARCH representation
        
    Returns
    -------
    np.ndarray
        Conditional variance series for FIGARCH model
    """
    # Compute squared returns from input data
    T = len(returns)
    eps_squared = returns * returns
    
    # Compute unconditional variance using model parameters
    uncond_var = omega / (1 - (1 - phi) * d / (1 - phi * (1 - d) - beta))
    
    # Calculate FIGARCH recursion coefficients using figarch_recursion
    lambda_coef = figarch_recursion(d, phi, beta, max_lag)
    
    # Initialize variance array with same length as returns
    variance = np.zeros(T)
    
    # Set initial variance values based on unconditional variance
    for t in range(min(max_lag, T)):
        variance[t] = uncond_var
    
    # Implement FIGARCH recursion formula for all time periods
    for t in range(max_lag, T):
        variance[t] = omega
        
        # Add ARCH terms with fractional coefficients
        for i in range(max_lag):
            if t - i - 1 >= 0:
                variance[t] += lambda_coef[i] * eps_squared[t - i - 1]
    
    return variance


def figarch_likelihood(params: np.ndarray, returns: np.ndarray, distribution: str, 
                      dist_params: dict, max_lag: int) -> float:
    """
    Objective function for FIGARCH parameter estimation that calculates the negative log-likelihood.
    
    Parameters
    ----------
    params : np.ndarray
        Parameter vector [omega, d, phi, beta]
    returns : np.ndarray
        Array of return data
    distribution : str
        Error distribution type
    dist_params : dict
        Parameters for the error distribution
    max_lag : int
        Maximum lag for FIGARCH recursion
        
    Returns
    -------
    float
        Negative log-likelihood value for optimization
    """
    # Extract FIGARCH parameters (omega, d, phi, beta) from params array
    omega, d, phi, beta = params
    
    # Validate parameter constraints for FIGARCH process
    if (omega <= 0 or d <= 0 or d >= 1 or phi < 0 or phi >= 1 or
        beta < 0 or beta >= 1 or phi + beta >= 1):
        return 1e10  # Large penalty for invalid parameters
    
    # Calculate conditional variances using figarch_variance function
    variance = figarch_variance(returns, omega, d, phi, beta, max_lag)
    
    # Compute log-likelihood using appropriate distribution
    log_likelihood = calculate_log_likelihood(returns, variance, distribution, dist_params)
    
    # Return negative log-likelihood for minimization
    return -log_likelihood


@dataclass
class FIGARCH(UnivariateVolatilityModel):
    """
    Fractionally Integrated GARCH (FIGARCH) model for long memory volatility modeling with parameter estimation,
    forecasting, and simulation capabilities.
    """
    
    # Model parameters
    omega: float = 0.1
    d: float = 0.4
    phi: float = 0.2
    beta: float = 0.3
    max_lag: int = 1000
    
    # Storage for estimation results
    variance: np.ndarray = None
    returns: np.ndarray = None
    fit_stats: dict = None
    std_errors: np.ndarray = None
    is_fitted: bool = False
    
    def __init__(self, omega: float = 0.1, d: float = 0.4, phi: float = 0.2, 
                 beta: float = 0.3, max_lag: int = 1000, distribution: str = 'normal',
                 dist_params: dict = None):
        """
        Initialize FIGARCH model with parameters
        
        Parameters
        ----------
        omega : float, optional
            Constant term in the variance equation, by default 0.1
        d : float, optional
            Fractional integration parameter (0 < d < 1), by default 0.4
        phi : float, optional
            GARCH autoregressive parameter (0 <= phi <= 1), by default 0.2
        beta : float, optional
            GARCH moving average parameter (0 <= beta <= 1), by default 0.3
        max_lag : int, optional
            Maximum lag for truncation of infinite ARCH representation, by default 1000
        distribution : str, optional
            Error distribution type, by default 'normal'
        dist_params : dict, optional
            Parameters for the error distribution, by default None
        """
        # Call parent constructor to initialize UnivariateVolatilityModel
        super().__init__(distribution=distribution, distribution_params=dist_params)
        
        # Initialize FIGARCH-specific parameters (omega, d, phi, beta)
        self.omega = omega
        self.d = d
        self.phi = phi
        self.beta = beta
        self.max_lag = max_lag
        
        # Validate parameter ranges using validate_parameters method
        self.validate_parameters(omega, d, phi, beta)
        
        # Initialize model properties (variance, returns, fit_stats)
        self.variance = None
        self.returns = None
        self.fit_stats = {}
        self.std_errors = None
        self.is_fitted = False
    
    def validate_parameters(self, omega: float, d: float, phi: float, beta: float) -> bool:
        """
        Validate FIGARCH model parameters within valid ranges
        
        Parameters
        ----------
        omega : float
            Constant term in the variance equation
        d : float
            Fractional integration parameter
        phi : float
            GARCH autoregressive parameter
        beta : float
            GARCH moving average parameter
            
        Returns
        -------
        bool
            True if parameters are valid, raises ValueError otherwise
        """
        # Check omega > 0 using is_positive_float
        if not is_positive_float(omega):
            raise ValueError(f"omega must be positive, got {omega}")
        
        # Check 0 < d < 1 for fractional integration parameter
        if not check_range(d, 0, 1, inclusive_min=False, inclusive_max=False):
            raise ValueError(f"d must be between 0 and 1, got {d}")
        
        # Check 0 <= phi <= 1 for autoregressive parameter
        if not check_range(phi, 0, 1, inclusive_min=True, inclusive_max=False):
            raise ValueError(f"phi must be between 0 and 1, got {phi}")
        
        # Check 0 <= beta <= 1 for GARCH parameter
        if not check_range(beta, 0, 1, inclusive_min=True, inclusive_max=False):
            raise ValueError(f"beta must be between 0 and 1, got {beta}")
        
        # Check phi + beta < 1 for stationarity
        if phi + beta >= 1:
            raise ValueError(f"phi + beta must be less than 1, got {phi + beta}")
        
        return True
    
    def calculate_variance(self, returns: np.ndarray) -> np.ndarray:
        """
        Calculate conditional variance series for FIGARCH model
        
        Parameters
        ----------
        returns : np.ndarray
            Array of return data
            
        Returns
        -------
        np.ndarray
            Conditional variance series
        """
        # Validate input returns array
        returns = validate_array(returns, param_name="returns")
        
        # Call figarch_variance function with model parameters
        variance = figarch_variance(returns, self.omega, self.d, self.phi, self.beta, self.max_lag)
        
        # Store variance in model object
        self.variance = variance
        
        return variance
    
    def estimate(self, returns: np.ndarray, initial_params: dict = None, 
                optimizer_params: dict = None) -> 'FIGARCH':
        """
        Estimate FIGARCH model parameters from return data using maximum likelihood
        
        Parameters
        ----------
        returns : np.ndarray
            Return data for estimation
        initial_params : dict, optional
            Initial parameter values, by default None
        optimizer_params : dict, optional
            Options for optimizer, by default None
            
        Returns
        -------
        FIGARCH
            Self for method chaining
        """
        # Validate input returns array
        returns = validate_array(returns, param_name="returns")
        self.returns = returns
        
        # Set default initial parameters if not provided
        if initial_params is None:
            initial_params = {
                'omega': self.omega,
                'd': self.d,
                'phi': self.phi,
                'beta': self.beta
            }
        
        # Configure optimizer settings
        if optimizer_params is None:
            optimizer_params = {
                'method': 'L-BFGS-B',
                'tol': 1e-8,
                'options': {'maxiter': 1000}
            }
        
        # Create parameter bounds for constrained optimization
        param_bounds = [
            (1e-6, 10.0),      # omega bounds
            (1e-6, 0.999),     # d bounds
            (0.0, 0.999),      # phi bounds
            (0.0, 0.999)       # beta bounds
        ]
        
        # Initialize optimizer
        optimizer = Optimizer()
        
        # Define objective function for likelihood maximization
        def obj_function(params):
            return figarch_likelihood(
                params, returns, self.distribution, self.distribution_params, self.max_lag
            )
        
        # Run optimization
        initial_values = [
            initial_params.get('omega', self.omega),
            initial_params.get('d', self.d),
            initial_params.get('phi', self.phi),
            initial_params.get('beta', self.beta)
        ]
        
        # Call optimization routine with Optimizer.minimize
        result = optimizer.minimize(
            obj_function, 
            np.array(initial_values),
            options={'bounds': param_bounds, **optimizer_params}
        )
        
        # Extract and store estimated parameters
        params = result.parameters
        self.omega, self.d, self.phi, self.beta = params
        
        # Calculate variance with optimized parameters
        self.variance = self.calculate_variance(returns)
        
        # Compute standard errors for parameter estimates
        self.std_errors = np.zeros_like(params)  # This would be improved in a production implementation
        
        # Calculate fit statistics (log-likelihood, AIC, BIC)
        log_likelihood = -result.objective_value
        n_params = 4  # omega, d, phi, beta
        aic = -2 * log_likelihood + 2 * n_params
        bic = -2 * log_likelihood + n_params * np.log(len(returns))
        
        self.fit_stats = {
            'log_likelihood': log_likelihood,
            'aic': aic,
            'bic': bic,
            'iterations': result.iterations,
            'converged': result.converged
        }
        
        # Set is_fitted flag to True
        self.is_fitted = True
        
        return self
    
    async def estimate_async(self, returns: np.ndarray, initial_params: dict = None,
                            optimizer_params: dict = None, 
                            progress_callback: Callable[[float], None] = None) -> 'FIGARCH':
        """
        Asynchronously estimate FIGARCH model parameters with progress reporting
        
        Parameters
        ----------
        returns : np.ndarray
            Return data for estimation
        initial_params : dict, optional
            Initial parameter values, by default None
        optimizer_params : dict, optional
            Options for optimizer, by default None
        progress_callback : callable, optional
            Function to report progress, by default None
            
        Returns
        -------
        FIGARCH
            Self for method chaining
        """
        # Validate input returns array
        returns = validate_array(returns, param_name="returns")
        self.returns = returns
        
        # Set default initial parameters if not provided
        if initial_params is None:
            initial_params = {
                'omega': self.omega,
                'd': self.d,
                'phi': self.phi,
                'beta': self.beta
            }
        
        # Configure optimizer settings
        if optimizer_params is None:
            optimizer_params = {
                'method': 'L-BFGS-B',
                'tol': 1e-8,
                'options': {'maxiter': 1000}
            }
        
        # Create parameter bounds for constrained optimization
        param_bounds = [
            (1e-6, 10.0),      # omega bounds
            (1e-6, 0.999),     # d bounds
            (0.0, 0.999),      # phi bounds
            (0.0, 0.999)       # beta bounds
        ]
        
        # Create async task using AsyncTask class
        task = AsyncTask(asyncio.sleep(0), progress_callback=progress_callback)
        
        # Define objective function for likelihood maximization
        def obj_function(params):
            return figarch_likelihood(
                params, returns, self.distribution, self.distribution_params, self.max_lag
            )
        
        # Initial parameter vector
        initial_values = [
            initial_params.get('omega', self.omega),
            initial_params.get('d', self.d),
            initial_params.get('phi', self.phi),
            initial_params.get('beta', self.beta)
        ]
        
        # Call Optimizer.async_minimize for asynchronous optimization
        async for iteration, params, obj_value in Optimizer().async_minimize(
            obj_function, 
            np.array(initial_values),
            options={'bounds': param_bounds, **optimizer_params}
        ):
            # Report progress during optimization if callback provided
            if progress_callback:
                progress = min(100 * iteration / optimizer_params.get('maxiter', 1000), 99)
                task.report_progress(progress)
        
        # Report completion
        if progress_callback:
            task.report_progress(100)
        
        # Get final result
        result = Optimizer().minimize(
            obj_function, 
            np.array(initial_values),
            options={'bounds': param_bounds, **optimizer_params}
        )
        
        # Extract and store final parameters when complete
        params = result.parameters
        self.omega, self.d, self.phi, self.beta = params
        
        # Compute standard errors and fit statistics
        self.variance = self.calculate_variance(returns)
        self.std_errors = np.zeros_like(params)
        
        log_likelihood = -result.objective_value
        n_params = 4
        aic = -2 * log_likelihood + 2 * n_params
        bic = -2 * log_likelihood + n_params * np.log(len(returns))
        
        self.fit_stats = {
            'log_likelihood': log_likelihood,
            'aic': aic,
            'bic': bic,
            'iterations': result.iterations,
            'converged': result.converged
        }
        
        # Set is_fitted flag to True
        self.is_fitted = True
        
        return self
    
    def forecast(self, horizon: int, returns: np.ndarray = None) -> np.ndarray:
        """
        Forecast conditional variance for future periods
        
        Parameters
        ----------
        horizon : int
            Number of periods to forecast
        returns : np.ndarray, optional
            Return data to use for initial conditions, by default None
            
        Returns
        -------
        np.ndarray
            Forecasted variance series
        """
        # Check if model has been fitted
        if not self.is_fitted and returns is None:
            raise ValueError("Model must be fitted before forecasting or returns must be provided")
        
        # Validate horizon parameter
        if horizon <= 0:
            raise ValueError("Forecast horizon must be positive")
        
        # If returns not provided, use stored returns from estimation
        if returns is not None:
            returns = validate_array(returns, param_name="returns")
        else:
            returns = self.returns
        
        # Calculate current conditional variance
        current_var = self.calculate_variance(returns)
        T = len(returns)
        
        # Compute FIGARCH recursion coefficients
        lambda_coef = figarch_recursion(self.d, self.phi, self.beta, self.max_lag)
        
        # Initialize forecast array
        forecasts = np.zeros(horizon)
        
        # Compute the long-run average variance
        long_run_var = self.omega / (1 - (1 - self.phi) * self.d / (1 - self.phi * (1 - self.d) - self.beta))
        
        # Implement FIGARCH forecasting formula with fractional weights
        for h in range(horizon):
            fc = self.omega
            
            for i in range(min(self.max_lag, T)):
                lag_idx = T - i - 1
                if lag_idx >= 0:
                    fc += lambda_coef[i] * returns[lag_idx]**2
            
            forecasts[h] = fc if fc > 0 else long_run_var
            
        return forecasts
    
    async def forecast_async(self, horizon: int, returns: np.ndarray = None,
                            progress_callback: Callable[[float], None] = None) -> np.ndarray:
        """
        Asynchronously forecast conditional variance with progress reporting
        
        Parameters
        ----------
        horizon : int
            Number of periods to forecast
        returns : np.ndarray, optional
            Return data to use for initial conditions, by default None
        progress_callback : callable, optional
            Function to report progress, by default None
            
        Returns
        -------
        np.ndarray
            Forecasted variance series
        """
        # Check if model has been fitted
        if not self.is_fitted and returns is None:
            raise ValueError("Model must be fitted before forecasting or returns must be provided")
        
        # Validate horizon parameter
        if horizon <= 0:
            raise ValueError("Forecast horizon must be positive")
        
        # Create async task for forecasting operation
        task = AsyncTask(asyncio.sleep(0), progress_callback=progress_callback)
        
        # Run forecast calculation in executor to avoid blocking
        def _forecast():
            if progress_callback:
                task.report_progress(10)
                
            result = self.forecast(horizon, returns)
            
            if progress_callback:
                task.report_progress(100)
                
            return result
        
        # Return completed forecasts when done
        return await run_in_executor(_forecast)
    
    def simulate(self, n_steps: int, initial_returns: np.ndarray = None,
                burn_in: int = 500, seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate returns from fitted FIGARCH model using Monte Carlo methods
        
        Parameters
        ----------
        n_steps : int
            Number of time steps to simulate
        initial_returns : np.ndarray, optional
            Initial returns for the simulation, by default None
        burn_in : int, optional
            Number of initial observations to discard, by default 500
        seed : int, optional
            Random seed for reproducibility, by default None
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Tuple containing simulated returns and variance series
        """
        # Check if model has been fitted
        if not self.is_fitted:
            raise ValueError("Model must be fitted before simulation")
        
        # Validate input parameters
        if n_steps <= 0:
            raise ValueError("Number of steps must be positive")
        
        if burn_in < 0:
            raise ValueError("Burn-in period must be non-negative")
        
        # Set random seed for reproducibility if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize arrays for simulated returns and variance
        total_steps = n_steps + burn_in
        sim_returns = np.zeros(total_steps)
        sim_variance = np.zeros(total_steps)
        
        # Compute unconditional variance
        uncond_var = self.omega / (1 - (1 - self.phi) * self.d / (1 - self.phi * (1 - self.d) - self.beta))
        
        # If initial_returns provided, use them for initial conditions
        if initial_returns is not None:
            initial_returns = validate_array(initial_returns, param_name="initial_returns")
            T_init = min(len(initial_returns), self.max_lag)
            sim_returns[:T_init] = initial_returns[-T_init:]
        
        # Compute FIGARCH recursion coefficients
        lambda_coef = figarch_recursion(self.d, self.phi, self.beta, self.max_lag)
        
        # Generate innovations from specified distribution
        if self.distribution.lower() == 'normal':
            innovations = np.random.standard_normal(total_steps)
        elif self.distribution.lower() == 'studentt':
            df = self.distribution_params.get('df', 5)
            innovations = np.random.standard_t(df, total_steps)
        elif self.distribution.lower() == 'ged':
            # Approximation for GED distribution
            innovations = np.random.standard_normal(total_steps)
        else:
            # Default to normal
            innovations = np.random.standard_normal(total_steps)
        
        # Generate burn-in period to reach stable state
        for t in range(total_steps):
            # Set initial variance
            if t < self.max_lag:
                sim_variance[t] = uncond_var
                continue
            
            # Calculate variance using FIGARCH recursive formula
            var_t = self.omega
            
            for i in range(self.max_lag):
                if t - i - 1 >= 0:
                    var_t += lambda_coef[i] * sim_returns[t - i - 1]**2
            
            # Store variance and generate return
            sim_variance[t] = var_t
            sim_returns[t] = np.sqrt(var_t) * innovations[t]
        
        # Return tuple of (simulated_returns, simulated_variance)
        return sim_returns[burn_in:], sim_variance[burn_in:]
    
    async def simulate_async(self, n_steps: int, initial_returns: np.ndarray = None,
                           burn_in: int = 500, seed: int = None,
                           progress_callback: Callable[[float], None] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Asynchronously simulate returns from FIGARCH model with progress reporting
        
        Parameters
        ----------
        n_steps : int
            Number of time steps to simulate
        initial_returns : np.ndarray, optional
            Initial returns for the simulation, by default None
        burn_in : int, optional
            Number of initial observations to discard, by default 500
        seed : int, optional
            Random seed for reproducibility, by default None
        progress_callback : callable, optional
            Function to report progress, by default None
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Tuple containing simulated returns and variance series
        """
        # Check if model has been fitted
        if not self.is_fitted:
            raise ValueError("Model must be fitted before simulation")
        
        # Validate input parameters
        if n_steps <= 0:
            raise ValueError("Number of steps must be positive")
        
        # Create async task for simulation operation
        task = AsyncTask(asyncio.sleep(0), progress_callback=progress_callback)
        
        # Run simulation in executor to avoid blocking
        def _simulate():
            if progress_callback:
                task.report_progress(10)
                
            sim_returns, sim_variance = self.simulate(n_steps, initial_returns, burn_in, seed)
            
            if progress_callback:
                task.report_progress(100)
                
            return sim_returns, sim_variance
        
        # Return completed simulation results when done
        return await run_in_executor(_simulate)
    
    def calculate_forecast_distribution(self, horizon: int, quantiles: List[float],
                                       returns: np.ndarray = None) -> Dict:
        """
        Calculate return distribution forecasts for specified quantiles
        
        Parameters
        ----------
        horizon : int
            Forecast horizon
        quantiles : List[float]
            List of quantiles to calculate (between 0 and 1)
        returns : np.ndarray, optional
            Returns data, by default None
            
        Returns
        -------
        Dict
            Dictionary of forecasted quantiles for each horizon
        """
        # Check if model has been fitted
        if not self.is_fitted and returns is None:
            raise ValueError("Model must be fitted before forecasting or returns must be provided")
        
        # Validate input parameters
        if horizon <= 0:
            raise ValueError("Forecast horizon must be positive")
        
        for q in quantiles:
            if not (0 <= q <= 1):
                raise ValueError(f"Quantiles must be between 0 and 1, got {q}")
        
        # Generate variance forecasts using forecast method
        variance_forecasts = self.forecast(horizon, returns)
        
        # For each forecast horizon, calculate return distribution
        forecasts = {}
        for h in range(horizon):
            # Standard deviation forecast for this horizon
            std_forecast = np.sqrt(variance_forecasts[h])
            
            # Distribution parameters
            dist_params = {
                'mu': 0,
                'sigma': std_forecast,
                **self.distribution_params or {}
            }
            
            # Use distribution_forecast with model's distribution type
            quantile_forecasts = distribution_forecast(
                self.distribution, 
                dist_params, 
                quantiles
            )
            
            forecasts[h+1] = {
                'variance': variance_forecasts[h],
                'std': std_forecast,
                'quantiles': dict(zip(quantiles, quantile_forecasts['values']))
            }
        
        return forecasts
    
    def half_life(self) -> float:
        """
        Calculate the volatility half-life implied by the FIGARCH process
        
        Returns
        -------
        float
            Half-life in time periods
        """
        # Check if model has been fitted
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calculating half-life")
        
        # Calculate persistence based on fractional integration parameter d
        persistence = 1 - self.d / (1 - self.phi * (1 - self.d) - self.beta)
        
        # Compute half-life using log(0.5)/log(persistence) formula
        if persistence >= 1:
            return float('inf')  # Infinite half-life for integrated processes
        elif persistence <= 0:
            return 0  # No persistence
        else:
            return np.log(0.5) / np.log(persistence)
    
    def summary(self) -> str:
        """
        Generate summary of FIGARCH model estimation results
        
        Returns
        -------
        str
            Formatted summary string with model parameters and statistics
        """
        # Check if model has been fitted
        if not self.is_fitted:
            return "FIGARCH model has not been fitted yet"
        
        # Format model parameters with standard errors
        summary_lines = []
        summary_lines.append("=" * 70)
        summary_lines.append("FIGARCH Model Summary")
        summary_lines.append("=" * 70)
        
        # Include model specification details (distribution, max_lag)
        summary_lines.append(f"Distribution:          {self.distribution}")
        summary_lines.append(f"Max Lag:               {self.max_lag}")
        summary_lines.append("-" * 70)
        
        # Format model parameters with standard errors
        summary_lines.append("Parameter Estimates:")
        summary_lines.append("-" * 70)
        summary_lines.append(f"omega:                 {self.omega:.6f}")
        summary_lines.append(f"d:                     {self.d:.6f}")
        summary_lines.append(f"phi:                   {self.phi:.6f}")
        summary_lines.append(f"beta:                  {self.beta:.6f}")
        summary_lines.append("-" * 70)
        
        # Add fit statistics (log-likelihood, AIC, BIC)
        summary_lines.append("Fit Statistics:")
        summary_lines.append("-" * 70)
        summary_lines.append(f"Log-Likelihood:        {self.fit_stats['log_likelihood']:.6f}")
        summary_lines.append(f"AIC:                   {self.fit_stats['aic']:.6f}")
        summary_lines.append(f"BIC:                   {self.fit_stats['bic']:.6f}")
        summary_lines.append(f"Convergence:           {'Yes' if self.fit_stats['converged'] else 'No'}")
        summary_lines.append(f"Iterations:            {self.fit_stats['iterations']}")
        
        # Include half-life information
        half_life_val = self.half_life()
        summary_lines.append("-" * 70)
        summary_lines.append(f"Volatility Half-Life:  {half_life_val:.2f} periods")
        summary_lines.append("=" * 70)
        
        return "\n".join(summary_lines)