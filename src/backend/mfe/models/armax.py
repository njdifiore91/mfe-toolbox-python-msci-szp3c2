"""
MFE Toolbox - ARMAX Module

This module implements AutoRegressive Moving Average with eXogenous variables (ARMAX)
time series models with robust parameter estimation, diagnostic tools, and forecasting
capabilities. It extends the ARMA model to include exogenous regressors.

The module leverages Python's scientific stack (NumPy, SciPy, Pandas, Statsmodels)
with Numba optimization for performance-critical calculations.
"""

import numpy as np  # numpy 1.26.3
import scipy.stats as stats  # scipy 1.11.4
import scipy.optimize as optimize  # scipy 1.11.4
import statsmodels.api as sm  # statsmodels 0.14.1
import pandas as pd  # pandas 2.1.4
import numba  # numba 0.59.0
from typing import Any, Dict, List, Optional, Tuple, Union, cast  # Python 3.12
from dataclasses import dataclass, field  # Python 3.12
import asyncio  # Python 3.12
import logging  # Python 3.12

# Internal imports
from .arma import ARMA, arma_forecast, compute_arma_residuals
from ..utils.numba_helpers import jit_decorator
from ..utils.validation import validate_array, check_time_series
from ..core.optimization import OptimizationResult, Optimizer
from ..core.distributions import normal_loglikelihood

# Module constants
ARMAX_MAX_LAG = 100  # Maximum lag order for ARMAX models
DEFAULT_OPTIMIZER = 'SLSQP'  # Default optimization method for parameter estimation

# Set up module logger
logger = logging.getLogger(__name__)


@jit_decorator(nopython=True)
def compute_armax_residuals(
    y: np.ndarray,
    ar_params: np.ndarray,
    ma_params: np.ndarray,
    constant: float = 0.0,
    exog: Optional[np.ndarray] = None,
    exog_params: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Computes residuals from an ARMAX model with given parameters and data.
    
    Parameters
    ----------
    y : ndarray
        Time series data
    ar_params : ndarray
        Autoregressive parameters (phi coefficients)
    ma_params : ndarray
        Moving average parameters (theta coefficients)
    constant : float, default=0.0
        Constant term in the model
    exog : ndarray, optional
        Exogenous variables
    exog_params : ndarray, optional
        Parameters for exogenous variables
        
    Returns
    -------
    ndarray
        Array of residuals from the ARMAX model
    
    Notes
    -----
    This function is optimized with Numba for performance.
    """
    # Get dimensions
    n = len(y)
    p = len(ar_params)
    q = len(ma_params)
    
    # Initialize arrays
    residuals = np.zeros(n)
    
    # Set up exog effect
    exog_effect = np.zeros(n)
    if exog is not None and exog_params is not None:
        for i in range(n):
            for j in range(len(exog_params)):
                exog_effect[i] += exog[i, j] * exog_params[j]
    
    # Compute residuals using recursion
    for t in range(n):
        # Initialize predicted value with constant
        y_pred = constant
        
        # Add AR component
        for i in range(p):
            if t - i - 1 >= 0:
                y_pred += ar_params[i] * y[t - i - 1]
        
        # Add MA component using previous residuals
        for j in range(q):
            if t - j - 1 >= 0:
                y_pred += ma_params[j] * residuals[t - j - 1]
        
        # Add exogenous effect
        if exog is not None and exog_params is not None:
            y_pred += exog_effect[t]
        
        # Compute residual
        residuals[t] = y[t] - y_pred
    
    return residuals


@jit_decorator(nopython=True)
def compute_armax_loglikelihood(
    params: np.ndarray,
    y: np.ndarray,
    p: int,
    q: int,
    include_constant: bool = True,
    exog: Optional[np.ndarray] = None
) -> float:
    """
    Computes the log-likelihood of an ARMAX model for given parameters.
    
    Parameters
    ----------
    params : ndarray
        Parameter vector [ar_params, ma_params, constant, exog_params]
    y : ndarray
        Time series data
    p : int
        Order of AR component
    q : int
        Order of MA component
    include_constant : bool, default=True
        Whether to include a constant term in the model
    exog : ndarray, optional
        Exogenous variables
    
    Returns
    -------
    float
        Negative log-likelihood value (for minimization)
    
    Notes
    -----
    This function is optimized with Numba for performance.
    """
    # Determine number of parameters and extract them
    k_exog = 0 if exog is None else exog.shape[1]
    
    # Extract parameters
    ar_params = params[:p] if p > 0 else np.array([])
    ma_params = params[p:p+q] if q > 0 else np.array([])
    
    index = p + q
    constant = params[index] if include_constant else 0.0
    index += include_constant
    
    exog_params = params[index:index+k_exog] if k_exog > 0 else None
    
    # Compute residuals
    residuals = compute_armax_residuals(y, ar_params, ma_params, constant, exog, exog_params)
    
    # Compute negative log-likelihood
    neg_ll = -normal_loglikelihood(residuals)
    
    return neg_ll


@jit_decorator(nopython=True)
def armax_forecast(
    y: np.ndarray,
    ar_params: np.ndarray,
    ma_params: np.ndarray,
    constant: float = 0.0,
    steps: int = 1,
    residuals: Optional[np.ndarray] = None,
    exog: Optional[np.ndarray] = None,
    exog_forecast: Optional[np.ndarray] = None,
    exog_params: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Generates forecasts from an ARMAX model with given parameters.
    
    Parameters
    ----------
    y : ndarray
        Time series data
    ar_params : ndarray
        Autoregressive parameters
    ma_params : ndarray
        Moving average parameters
    constant : float, default=0.0
        Constant term in the model
    steps : int, default=1
        Number of steps ahead to forecast
    residuals : ndarray, optional
        Residuals from the model, used for MA component
    exog : ndarray, optional
        Exogenous variables in the history
    exog_forecast : ndarray, optional
        Exogenous variables for the forecast period
    exog_params : ndarray, optional
        Parameters for exogenous variables
    
    Returns
    -------
    ndarray
        Array of forecasted values for specified steps ahead
    
    Notes
    -----
    This function is optimized with Numba for performance.
    """
    # Get dimensions
    n = len(y)
    p = len(ar_params)
    q = len(ma_params)
    
    # If no residuals provided, compute them
    if residuals is None:
        residuals = compute_armax_residuals(y, ar_params, ma_params, constant, exog, exog_params)
    
    # Initialize forecast array
    forecasts = np.zeros(steps)
    
    # Create extended series for easier forecasting
    y_extended = np.concatenate([y, np.zeros(steps)])
    resid_extended = np.concatenate([residuals, np.zeros(steps)])
    
    # Prepare exogenous effect for forecast horizon
    exog_effect = np.zeros(steps)
    if exog_forecast is not None and exog_params is not None:
        for i in range(steps):
            for j in range(len(exog_params)):
                exog_effect[i] += exog_forecast[i, j] * exog_params[j]
    
    # Generate forecasts recursively
    for h in range(steps):
        # Initialize forecast with constant
        forecast = constant
        
        # Add AR component
        for i in range(p):
            if h - i - 1 >= 0:
                # Use previous forecasts
                forecast += ar_params[i] * y_extended[n + h - i - 1]
            else:
                # Use actual data
                forecast += ar_params[i] * y[n + h - i - 1]
        
        # Add MA component (only observed residuals, future innovations are zero)
        for j in range(q):
            if h - j - 1 >= 0:
                # Future innovations are zero
                forecast += ma_params[j] * resid_extended[n + h - j - 1]
            else:
                # Use known residuals
                forecast += ma_params[j] * residuals[n + h - j - 1]
        
        # Add exogenous effect if provided
        if exog_forecast is not None and exog_params is not None:
            forecast += exog_effect[h]
        
        # Store forecast
        forecasts[h] = forecast
        y_extended[n + h] = forecast
    
    return forecasts


def armax_order_select(
    y: np.ndarray,
    exog: np.ndarray,
    max_ar: int = 5,
    max_ma: int = 5,
    ic: str = 'aic',
    include_constant: bool = True
) -> Tuple[int, int]:
    """
    Selects optimal ARMAX order based on information criteria.
    
    Parameters
    ----------
    y : ndarray
        Time series data
    exog : ndarray
        Exogenous variables
    max_ar : int, default=5
        Maximum AR order to consider
    max_ma : int, default=5
        Maximum MA order to consider
    ic : str, default='aic'
        Information criterion ('aic', 'bic', 'hqic')
    include_constant : bool, default=True
        Whether to include a constant term in the model
    
    Returns
    -------
    tuple
        Tuple containing optimal AR and MA orders
    """
    # Validate input data
    y = validate_array(y)
    if y.ndim != 1:
        raise ValueError("Input y must be a 1D array")
    
    # Validate parameters
    if max_ar < 0 or max_ma < 0:
        raise ValueError("Maximum AR and MA orders must be non-negative")
    
    if max_ar > ARMAX_MAX_LAG or max_ma > ARMAX_MAX_LAG:
        raise ValueError(f"Maximum AR and MA orders must not exceed {ARMAX_MAX_LAG}")
    
    if max_ar == 0 and max_ma == 0:
        logger.warning("Both max_ar and max_ma are zero, returning (0, 0)")
        return 0, 0
    
    if ic.lower() not in ['aic', 'bic', 'hqic']:
        raise ValueError("Information criterion must be one of: 'aic', 'bic', 'hqic'")
    
    # Initialize variables for optimal order
    best_ic = np.inf
    best_p = 0
    best_q = 0
    
    # Loop through different orders
    for p in range(max_ar + 1):
        for q in range(max_ma + 1):
            # Skip if both p and q are zero (pure white noise model)
            if p == 0 and q == 0:
                if not include_constant and exog is None:
                    continue
            
            try:
                # Create and estimate model
                model = ARMAX(p=p, q=q, include_constant=include_constant)
                model.estimate(y, exog=exog)
                
                # Get information criterion value
                ic_value = model.information_criteria([ic.lower()])[ic.lower()]
                
                # Update best model if current is better
                if ic_value < best_ic:
                    best_ic = ic_value
                    best_p = p
                    best_q = q
            except Exception as e:
                logger.debug(f"Error estimating ARMAX({p}, {q}): {str(e)}")
                continue
    
    logger.info(f"Best ARMAX order selected: ({best_p}, {best_q}) with {ic}={best_ic:.4f}")
    return best_p, best_q


@dataclass
class ARMAX:
    """
    AutoRegressive Moving Average with eXogenous variables time series model, extending ARMA with additional regressors.
    
    Parameters
    ----------
    p : int
        Order of AR component
    q : int
        Order of MA component
    include_constant : bool, default=True
        Whether to include a constant term in the model
    optimizer : str, default='SLSQP'
        Optimization method for parameter estimation
    
    Attributes
    ----------
    ar_params : ndarray
        Estimated AR parameters
    ma_params : ndarray
        Estimated MA parameters
    constant : float
        Estimated constant term
    exog_params : ndarray
        Estimated exogenous parameters
    residuals : ndarray
        Model residuals
    sigma2 : float
        Residual variance
    loglikelihood : float
        Log-likelihood of the estimated model
    standard_errors : ndarray
        Standard errors of parameter estimates
    results_summary : DataFrame
        Summary of estimation results
    """
    p: int = 0
    q: int = 0
    include_constant: bool = True
    optimizer: str = DEFAULT_OPTIMIZER
    
    # Estimated parameters (populated after estimation)
    ar_params: Optional[np.ndarray] = None
    ma_params: Optional[np.ndarray] = None
    constant: Optional[float] = None
    exog_params: Optional[np.ndarray] = None
    
    # Results (populated after estimation)
    residuals: Optional[np.ndarray] = None
    sigma2: Optional[float] = None
    loglikelihood: Optional[float] = None
    standard_errors: Optional[np.ndarray] = None
    results_summary: Optional[pd.DataFrame] = None
    
    def __post_init__(self):
        """
        Validate model parameters after initialization.
        """
        # Validate AR and MA orders
        if self.p < 0 or self.q < 0:
            raise ValueError("AR and MA orders must be non-negative")
        
        if self.p > ARMAX_MAX_LAG or self.q > ARMAX_MAX_LAG:
            raise ValueError(f"AR and MA orders must not exceed {ARMAX_MAX_LAG}")
        
        # Validate optimizer
        if self.optimizer not in ['BFGS', 'L-BFGS-B', 'Powell', 'CG', 'Newton-CG', 'SLSQP']:
            logger.warning(f"Optimizer '{self.optimizer}' may not be supported. Using '{DEFAULT_OPTIMIZER}'.")
            self.optimizer = DEFAULT_OPTIMIZER
    
    def estimate(
        self,
        y: np.ndarray,
        exog: Optional[np.ndarray] = None,
        optimizer_kwargs: Optional[Dict[str, Any]] = None
    ) -> 'ARMAX':
        """
        Estimates ARMAX model parameters using maximum likelihood.
        
        Parameters
        ----------
        y : ndarray
            Time series data
        exog : ndarray, optional
            Exogenous variables
        optimizer_kwargs : dict, optional
            Additional arguments for the optimizer
        
        Returns
        -------
        self
            Returns self for method chaining
        """
        # Validate input data
        y = validate_array(y)
        if y.ndim != 1:
            raise ValueError("Input y must be a 1D array")
        
        n = len(y)
        
        # Validate exogenous variables
        k_exog = 0
        if exog is not None:
            exog = validate_array(exog)
            if exog.ndim == 1:
                exog = exog.reshape(-1, 1)
            
            if len(exog) != n:
                raise ValueError(f"Length of exog ({len(exog)}) must match length of y ({n})")
            
            k_exog = exog.shape[1]
        
        # Determine number of parameters
        n_params = self.p + self.q + (1 if self.include_constant else 0) + k_exog
        
        if n_params == 0:
            raise ValueError("Model has no parameters. At least one of p, q, include_constant, or exog should be provided.")
        
        # Set up initial parameter values
        initial_params = np.zeros(n_params)
        
        # Initialize AR parameters (close to stationarity)
        if self.p > 0:
            # Use small positive values for stability
            initial_params[:self.p] = 0.1 / np.arange(1, self.p + 1)
        
        # Initialize MA parameters (close to invertibility)
        if self.q > 0:
            # Use small negative values for stability
            initial_params[self.p:self.p + self.q] = -0.1 / np.arange(1, self.q + 1)
        
        # Initialize constant term (to sample mean)
        if self.include_constant:
            initial_params[self.p + self.q] = np.mean(y)
        
        # Initialize exogenous parameters (to OLS estimates)
        if k_exog > 0:
            # Simple OLS for initial exog parameters
            idx = self.p + self.q + (1 if self.include_constant else 0)
            X = exog
            ols_params = np.linalg.lstsq(X, y, rcond=None)[0]
            initial_params[idx:idx + k_exog] = ols_params
        
        # Set up bounds for parameters
        bounds = []
        
        # Bounds for AR parameters (for stationarity)
        for _ in range(self.p):
            bounds.append((-0.99, 0.99))  # Keep inside unit circle
        
        # Bounds for MA parameters (for invertibility)
        for _ in range(self.q):
            bounds.append((-0.99, 0.99))  # Keep inside unit circle
        
        # Bounds for constant
        if self.include_constant:
            mean_abs = np.abs(np.mean(y))
            bounds.append((-10 * mean_abs, 10 * mean_abs))  # Wide range centered on zero
        
        # Bounds for exogenous parameters
        for _ in range(k_exog):
            bounds.append((-10, 10))  # Wide range centered on zero
        
        # Define objective function (negative log-likelihood)
        def objective(params):
            return compute_armax_loglikelihood(params, y, self.p, self.q, 
                                             self.include_constant, exog)
        
        # Prepare optimizer settings
        optimizer_options = {
            'maxiter': 1000,
            'disp': False
        }
        
        if optimizer_kwargs is not None:
            optimizer_options.update(optimizer_kwargs)
        
        # Create optimizer
        opt = Optimizer()
        
        # Run optimization
        try:
            result = opt.minimize(
                objective, 
                initial_params,
                options={
                    'method': self.optimizer,
                    'bounds': bounds,
                    **optimizer_options
                }
            )
            
            # Check if optimization was successful
            if not result.converged:
                logger.warning(f"Optimization did not converge: {result.message}")
        
            # Extract parameters
            params = result.parameters
            std_errors = result.standard_errors
            
            # Store parameters
            idx = 0
            if self.p > 0:
                self.ar_params = params[idx:idx + self.p]
                idx += self.p
            else:
                self.ar_params = np.array([])
            
            if self.q > 0:
                self.ma_params = params[idx:idx + self.q]
                idx += self.q
            else:
                self.ma_params = np.array([])
            
            if self.include_constant:
                self.constant = params[idx]
                idx += 1
            else:
                self.constant = 0.0
            
            if k_exog > 0:
                self.exog_params = params[idx:idx + k_exog]
            else:
                self.exog_params = None
            
            # Compute and store residuals
            self.residuals = compute_armax_residuals(
                y, self.ar_params, self.ma_params, 
                self.constant, exog, self.exog_params
            )
            
            # Compute and store other results
            self.sigma2 = np.var(self.residuals)
            self.loglikelihood = -result.objective_value
            self.standard_errors = std_errors
            
            # Generate results summary
            self._create_summary()
            
            return self
        
        except Exception as e:
            logger.error(f"Error estimating ARMAX model: {str(e)}")
            raise
    
    async def estimate_async(
        self,
        y: np.ndarray,
        exog: Optional[np.ndarray] = None,
        optimizer_kwargs: Optional[Dict[str, Any]] = None
    ) -> 'ARMAX':
        """
        Asynchronously estimates ARMAX model parameters.
        
        Parameters
        ----------
        y : ndarray
            Time series data
        exog : ndarray, optional
            Exogenous variables
        optimizer_kwargs : dict, optional
            Additional arguments for the optimizer
        
        Returns
        -------
        self
            Returns self for method chaining
        
        Notes
        -----
        This method uses asyncio for non-blocking estimation of model parameters.
        """
        # Validate input data (same as synchronous version)
        y = validate_array(y)
        if y.ndim != 1:
            raise ValueError("Input y must be a 1D array")
        
        n = len(y)
        
        # Validate exogenous variables
        k_exog = 0
        if exog is not None:
            exog = validate_array(exog)
            if exog.ndim == 1:
                exog = exog.reshape(-1, 1)
            
            if len(exog) != n:
                raise ValueError(f"Length of exog ({len(exog)}) must match length of y ({n})")
            
            k_exog = exog.shape[1]
        
        # Determine number of parameters
        n_params = self.p + self.q + (1 if self.include_constant else 0) + k_exog
        
        if n_params == 0:
            raise ValueError("Model has no parameters. At least one of p, q, include_constant, or exog should be provided.")
        
        # Set up initial parameter values (same as synchronous version)
        initial_params = np.zeros(n_params)
        
        # Initialize AR parameters
        if self.p > 0:
            initial_params[:self.p] = 0.1 / np.arange(1, self.p + 1)
        
        # Initialize MA parameters
        if self.q > 0:
            initial_params[self.p:self.p + self.q] = -0.1 / np.arange(1, self.q + 1)
        
        # Initialize constant term
        if self.include_constant:
            initial_params[self.p + self.q] = np.mean(y)
        
        # Initialize exogenous parameters
        if k_exog > 0:
            idx = self.p + self.q + (1 if self.include_constant else 0)
            X = exog
            ols_params = np.linalg.lstsq(X, y, rcond=None)[0]
            initial_params[idx:idx + k_exog] = ols_params
        
        # Set up bounds for parameters (same as synchronous version)
        bounds = []
        
        # Bounds for AR parameters
        for _ in range(self.p):
            bounds.append((-0.99, 0.99))
        
        # Bounds for MA parameters
        for _ in range(self.q):
            bounds.append((-0.99, 0.99))
        
        # Bounds for constant
        if self.include_constant:
            mean_abs = np.abs(np.mean(y))
            bounds.append((-10 * mean_abs, 10 * mean_abs))
        
        # Bounds for exogenous parameters
        for _ in range(k_exog):
            bounds.append((-10, 10))
        
        # Define objective function
        def objective(params):
            return compute_armax_loglikelihood(params, y, self.p, self.q, 
                                             self.include_constant, exog)
        
        # Prepare optimizer settings
        optimizer_options = {
            'maxiter': 1000,
            'disp': False
        }
        
        if optimizer_kwargs is not None:
            optimizer_options.update(optimizer_kwargs)
        
        # Create optimizer
        opt = Optimizer()
        
        # Run optimization asynchronously
        try:
            # Get result from async optimization
            result = await asyncio.to_thread(
                opt.minimize,
                objective, 
                initial_params,
                options={
                    'method': self.optimizer,
                    'bounds': bounds,
                    **optimizer_options
                }
            )
            
            # Check if optimization was successful
            if not result.converged:
                logger.warning(f"Optimization did not converge: {result.message}")
        
            # Extract parameters
            params = result.parameters
            std_errors = result.standard_errors
            
            # Store parameters
            idx = 0
            if self.p > 0:
                self.ar_params = params[idx:idx + self.p]
                idx += self.p
            else:
                self.ar_params = np.array([])
            
            if self.q > 0:
                self.ma_params = params[idx:idx + self.q]
                idx += self.q
            else:
                self.ma_params = np.array([])
            
            if self.include_constant:
                self.constant = params[idx]
                idx += 1
            else:
                self.constant = 0.0
            
            if k_exog > 0:
                self.exog_params = params[idx:idx + k_exog]
            else:
                self.exog_params = None
            
            # Compute and store residuals asynchronously
            residuals_task = asyncio.create_task(
                asyncio.to_thread(
                    compute_armax_residuals,
                    y, self.ar_params, self.ma_params, 
                    self.constant, exog, self.exog_params
                )
            )
            
            self.residuals = await residuals_task
            
            # Compute and store other results
            self.sigma2 = np.var(self.residuals)
            self.loglikelihood = -result.objective_value
            self.standard_errors = std_errors
            
            # Generate results summary
            self._create_summary()
            
            return self
        
        except Exception as e:
            logger.error(f"Error estimating ARMAX model asynchronously: {str(e)}")
            raise
    
    def forecast(
        self,
        y: np.ndarray,
        steps: int = 1,
        exog_forecast: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Generates forecasts from estimated ARMAX model.
        
        Parameters
        ----------
        y : ndarray
            Time series data
        steps : int, default=1
            Number of steps ahead to forecast
        exog_forecast : ndarray, optional
            Exogenous variables for the forecast period
        
        Returns
        -------
        ndarray
            Array of point forecasts for specified steps ahead
        """
        # Check if model has been estimated
        if self.ar_params is None or self.ma_params is None:
            raise ValueError("Model must be estimated before forecasting")
        
        # Validate input data
        y = validate_array(y)
        if y.ndim != 1:
            raise ValueError("Input y must be a 1D array")
        
        # Validate steps
        if steps <= 0:
            raise ValueError("steps must be a positive integer")
        
        # Validate exogenous forecast data
        if self.exog_params is not None and exog_forecast is None:
            raise ValueError("exog_forecast is required for models with exogenous variables")
        
        if exog_forecast is not None:
            exog_forecast = validate_array(exog_forecast)
            if exog_forecast.ndim == 1:
                exog_forecast = exog_forecast.reshape(-1, 1)
            
            if len(exog_forecast) < steps:
                raise ValueError(f"exog_forecast must have at least {steps} observations")
            
            if exog_forecast.shape[1] != len(self.exog_params):
                raise ValueError(f"exog_forecast must have {len(self.exog_params)} columns")
            
            # Truncate if longer than needed
            exog_forecast = exog_forecast[:steps]
        
        # Generate forecasts
        forecasts = armax_forecast(
            y, self.ar_params, self.ma_params, self.constant, steps,
            self.residuals, None, exog_forecast, self.exog_params
        )
        
        return forecasts
    
    async def forecast_async(
        self,
        y: np.ndarray,
        steps: int = 1,
        exog_forecast: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Asynchronously generates forecasts from estimated ARMAX model.
        
        Parameters
        ----------
        y : ndarray
            Time series data
        steps : int, default=1
            Number of steps ahead to forecast
        exog_forecast : ndarray, optional
            Exogenous variables for the forecast period
        
        Returns
        -------
        ndarray
            Array of point forecasts for specified steps ahead
        
        Notes
        -----
        This method uses asyncio for non-blocking forecast generation.
        """
        # Check if model has been estimated
        if self.ar_params is None or self.ma_params is None:
            raise ValueError("Model must be estimated before forecasting")
        
        # Validate input data
        y = validate_array(y)
        if y.ndim != 1:
            raise ValueError("Input y must be a 1D array")
        
        # Validate steps
        if steps <= 0:
            raise ValueError("steps must be a positive integer")
        
        # Validate exogenous forecast data
        if self.exog_params is not None and exog_forecast is None:
            raise ValueError("exog_forecast is required for models with exogenous variables")
        
        if exog_forecast is not None:
            exog_forecast = validate_array(exog_forecast)
            if exog_forecast.ndim == 1:
                exog_forecast = exog_forecast.reshape(-1, 1)
            
            if len(exog_forecast) < steps:
                raise ValueError(f"exog_forecast must have at least {steps} observations")
            
            if exog_forecast.shape[1] != len(self.exog_params):
                raise ValueError(f"exog_forecast must have {len(self.exog_params)} columns")
            
            # Truncate if longer than needed
            exog_forecast = exog_forecast[:steps]
        
        # Run forecast asynchronously
        forecasts = await asyncio.to_thread(
            armax_forecast,
            y, self.ar_params, self.ma_params, self.constant, steps,
            self.residuals, None, exog_forecast, self.exog_params
        )
        
        return forecasts
    
    def forecast_variance(self, steps: int = 1) -> np.ndarray:
        """
        Computes variance of forecasts for uncertainty quantification.
        
        Parameters
        ----------
        steps : int, default=1
            Number of steps ahead for which to compute forecast variances
        
        Returns
        -------
        ndarray
            Array of forecast variances for each step ahead
        """
        # Check if model has been estimated
        if self.ar_params is None or self.ma_params is None or self.sigma2 is None:
            raise ValueError("Model must be estimated before computing forecast variance")
        
        # Validate steps
        if steps <= 0:
            raise ValueError("steps must be a positive integer")
        
        # Initialize array for forecast variances
        variances = np.zeros(steps)
        
        # For MA(∞) representation, need to compute psi-weights
        psi_weights = np.zeros(steps + 1)
        psi_weights[0] = 1.0
        
        # Compute psi-weights recursively
        for i in range(1, steps + 1):
            # MA component contribution
            if i <= len(self.ma_params):
                psi_weights[i] += self.ma_params[i-1]
            
            # AR component contribution
            for j in range(1, min(i, len(self.ar_params) + 1)):
                psi_weights[i] += self.ar_params[j-1] * psi_weights[i-j]
        
        # Calculate forecast variances
        for h in range(steps):
            # Sum squared psi-weights
            variances[h] = self.sigma2 * np.sum(psi_weights[:h+1]**2)
        
        return variances
    
    def simulate(
        self,
        nsimulations: int = 100,
        burn: int = 50,
        initial_values: Optional[np.ndarray] = None,
        exog_sim: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Simulates time series from ARMAX model with given parameters.
        
        Parameters
        ----------
        nsimulations : int, default=100
            Number of time periods to simulate
        burn : int, default=50
            Number of initial observations to discard (burn-in period)
        initial_values : ndarray, optional
            Initial values for the simulation. If None, zeros are used.
        exog_sim : ndarray, optional
            Exogenous variables for the simulation period
        
        Returns
        -------
        ndarray
            Simulated time series data
        """
        # Check if model has been estimated
        if self.ar_params is None or self.ma_params is None or self.sigma2 is None:
            raise ValueError("Model must be estimated before simulation")
        
        # Validate parameters
        if nsimulations <= 0:
            raise ValueError("nsimulations must be a positive integer")
        
        if burn < 0:
            raise ValueError("burn must be a non-negative integer")
        
        # Get dimensions
        p = len(self.ar_params)
        q = len(self.ma_params)
        
        # Determine total length of simulation including burn-in
        total_length = nsimulations + burn
        
        # Set up initial values if not provided
        if initial_values is not None:
            initial_values = validate_array(initial_values)
            if len(initial_values) < max(p, q):
                raise ValueError(f"initial_values must have length at least max(p, q) = {max(p, q)}")
            
            y_init = initial_values[:max(p, q)]
        else:
            y_init = np.zeros(max(p, q))
        
        # Validate exogenous simulation data
        k_exog = 0
        if self.exog_params is not None:
            k_exog = len(self.exog_params)
            
            if exog_sim is None:
                raise ValueError("exog_sim is required for models with exogenous variables")
            
            exog_sim = validate_array(exog_sim)
            if exog_sim.ndim == 1:
                exog_sim = exog_sim.reshape(-1, 1)
            
            if len(exog_sim) < total_length:
                raise ValueError(f"exog_sim must have at least {total_length} observations")
            
            if exog_sim.shape[1] != k_exog:
                raise ValueError(f"exog_sim must have {k_exog} columns")
            
            # Truncate if longer than needed
            exog_sim = exog_sim[:total_length]
        
        # Set up result array
        y_sim = np.zeros(total_length)
        residuals_sim = np.zeros(total_length)
        
        # Set initial values
        if len(y_init) > 0:
            y_sim[:len(y_init)] = y_init
        
        # Generate random innovations
        np.random.seed(None)  # Use current system time for seed
        innovations = np.random.normal(0, np.sqrt(self.sigma2), total_length)
        
        # Simulate ARMAX process
        for t in range(max(p, q), total_length):
            # Initialize with constant
            y_sim[t] = self.constant
            
            # Add AR component
            for i in range(p):
                y_sim[t] += self.ar_params[i] * y_sim[t - i - 1]
            
            # Add MA component
            for j in range(q):
                y_sim[t] += self.ma_params[j] * residuals_sim[t - j - 1]
            
            # Add exogenous effect
            if k_exog > 0:
                for k in range(k_exog):
                    y_sim[t] += self.exog_params[k] * exog_sim[t, k]
            
            # Add innovation
            y_sim[t] += innovations[t]
            
            # Store residual for MA component
            residuals_sim[t] = innovations[t]
        
        # Return simulated series excluding burn-in
        return y_sim[burn:]
    
    def diagnostic_tests(self, tests: List[str] = ['ljung_box', 'jarque_bera']) -> Dict[str, Any]:
        """
        Performs diagnostic tests on model residuals.
        
        Parameters
        ----------
        tests : list, default=['ljung_box', 'jarque_bera']
            List of tests to perform. Options include:
            - 'ljung_box': Ljung-Box test for autocorrelation
            - 'jarque_bera': Jarque-Bera test for normality
            - 'arch_lm': ARCH-LM test for conditional heteroskedasticity
            - 'adf': Augmented Dickey-Fuller test for stationarity
        
        Returns
        -------
        dict
            Dictionary of test results
        """
        # Check if model has been estimated
        if self.residuals is None:
            raise ValueError("Model must be estimated before performing diagnostic tests")
        
        # Initialize results dictionary
        results = {}
        
        # Perform requested tests
        for test in tests:
            if test.lower() == 'ljung_box':
                # Compute lags as min(10, n/5) where n is sample size
                lags = min(10, len(self.residuals) // 5)
                
                # Import function from statsmodels
                from statsmodels.stats.diagnostic import acorr_ljungbox
                
                # Run test
                lb_results = acorr_ljungbox(self.residuals, lags=[lags], model_df=self.p + self.q)
                results['ljung_box'] = {
                    'statistic': float(lb_results.iloc[0, 0]),
                    'p_value': float(lb_results.iloc[0, 1]),
                    'lags': lags
                }
            
            elif test.lower() == 'jarque_bera':
                # Import function from scipy
                from scipy.stats import jarque_bera
                
                # Run test
                jb_stat, jb_pval, _, _ = jarque_bera(self.residuals)
                results['jarque_bera'] = {
                    'statistic': jb_stat,
                    'p_value': jb_pval
                }
            
            elif test.lower() == 'arch_lm':
                # Import function from statsmodels
                from statsmodels.stats.diagnostic import het_arch
                
                # Run test with 5 lags
                lags = min(5, len(self.residuals) // 10)
                arch_lm_stat, arch_lm_pval, _, _ = het_arch(self.residuals, nlags=lags)
                results['arch_lm'] = {
                    'statistic': arch_lm_stat,
                    'p_value': arch_lm_pval,
                    'lags': lags
                }
            
            elif test.lower() == 'adf':
                # Run Augmented Dickey-Fuller test
                from statsmodels.tsa.stattools import adfuller
                
                adf_results = adfuller(self.residuals)
                results['adf'] = {
                    'statistic': adf_results[0],
                    'p_value': adf_results[1],
                    'lags': adf_results[2],
                    'critical_values': adf_results[4]
                }
            
            else:
                logger.warning(f"Unknown test: {test}")
        
        return results
    
    def summary(self) -> pd.DataFrame:
        """
        Generates summary of model estimation results.
        
        Returns
        -------
        DataFrame
            DataFrame containing parameter estimates and diagnostics
        """
        # Check if model has been estimated
        if self.results_summary is None:
            if self.ar_params is not None and self.ma_params is not None:
                self._create_summary()
            else:
                raise ValueError("Model must be estimated before generating summary")
        
        return self.results_summary
    
    def information_criteria(self, criteria: List[str] = ['aic', 'bic', 'hqic']) -> Dict[str, float]:
        """
        Computes information criteria for model selection.
        
        Parameters
        ----------
        criteria : list, default=['aic', 'bic', 'hqic']
            List of information criteria to compute. Options include:
            - 'aic': Akaike Information Criterion
            - 'bic': Bayesian Information Criterion
            - 'hqic': Hannan-Quinn Information Criterion
        
        Returns
        -------
        dict
            Dictionary of information criteria values
        """
        # Check if model has been estimated
        if self.loglikelihood is None or self.residuals is None:
            raise ValueError("Model must be estimated before computing information criteria")
        
        # Calculate number of parameters
        k = self.p + self.q + (1 if self.include_constant else 0)
        if self.exog_params is not None:
            k += len(self.exog_params)
        
        # Calculate number of observations
        n = len(self.residuals)
        
        # Initialize results dictionary
        results = {}
        
        # Compute requested criteria
        for criterion in criteria:
            if criterion.lower() == 'aic':
                # AIC = -2 * log(L) + 2 * k
                results['aic'] = -2 * self.loglikelihood + 2 * k
            
            elif criterion.lower() == 'bic':
                # BIC = -2 * log(L) + k * log(n)
                results['bic'] = -2 * self.loglikelihood + k * np.log(n)
            
            elif criterion.lower() == 'hqic':
                # HQIC = -2 * log(L) + 2 * k * log(log(n))
                results['hqic'] = -2 * self.loglikelihood + 2 * k * np.log(np.log(n))
            
            else:
                logger.warning(f"Unknown information criterion: {criterion}")
        
        return results
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Converts model parameters and results to dictionary format.
        
        Returns
        -------
        dict
            Dictionary containing model specification and results
        """
        model_dict = {
            'model': 'ARMAX',
            'p': self.p,
            'q': self.q,
            'include_constant': self.include_constant,
            'optimizer': self.optimizer
        }
        
        # Add parameters if model has been estimated
        if self.ar_params is not None and self.ma_params is not None:
            model_dict['ar_params'] = self.ar_params.tolist() if len(self.ar_params) > 0 else []
            model_dict['ma_params'] = self.ma_params.tolist() if len(self.ma_params) > 0 else []
            
            if self.constant is not None:
                model_dict['constant'] = self.constant
            
            if self.exog_params is not None:
                model_dict['exog_params'] = self.exog_params.tolist()
            
            if self.sigma2 is not None:
                model_dict['sigma2'] = self.sigma2
            
            if self.loglikelihood is not None:
                model_dict['loglikelihood'] = self.loglikelihood
            
            # Add information criteria
            info_criteria = self.information_criteria()
            model_dict.update(info_criteria)
        
        return model_dict
    
    @classmethod
    def from_dict(cls, model_dict: Dict[str, Any]) -> 'ARMAX':
        """
        Creates ARMAX model instance from dictionary specification.
        
        Parameters
        ----------
        model_dict : dict
            Dictionary containing model specification
        
        Returns
        -------
        ARMAX
            ARMAX model instance with loaded parameters
        """
        # Check model type
        if model_dict.get('model') != 'ARMAX':
            raise ValueError("Dictionary does not contain an ARMAX model specification")
        
        # Create model instance
        model = cls(
            p=model_dict.get('p', 0),
            q=model_dict.get('q', 0),
            include_constant=model_dict.get('include_constant', True),
            optimizer=model_dict.get('optimizer', DEFAULT_OPTIMIZER)
        )
        
        # Load parameters if available
        if 'ar_params' in model_dict and 'ma_params' in model_dict:
            model.ar_params = np.array(model_dict['ar_params'])
            model.ma_params = np.array(model_dict['ma_params'])
            
            if 'constant' in model_dict:
                model.constant = model_dict['constant']
            
            if 'exog_params' in model_dict:
                model.exog_params = np.array(model_dict['exog_params'])
            
            if 'sigma2' in model_dict:
                model.sigma2 = model_dict['sigma2']
            
            if 'loglikelihood' in model_dict:
                model.loglikelihood = model_dict['loglikelihood']
        
        return model
    
    def _create_summary(self) -> None:
        """
        Creates summary DataFrame with parameter estimates and statistics.
        """
        # Check if model has been estimated
        if self.ar_params is None or self.ma_params is None:
            raise ValueError("Model must be estimated before creating summary")
        
        # Create list for parameter names and values
        param_names = []
        param_values = []
        
        # Add AR parameters
        for i in range(len(self.ar_params)):
            param_names.append(f"AR({i+1})")
            param_values.append(self.ar_params[i])
        
        # Add MA parameters
        for i in range(len(self.ma_params)):
            param_names.append(f"MA({i+1})")
            param_values.append(self.ma_params[i])
        
        # Add constant
        if self.include_constant:
            param_names.append("Constant")
            param_values.append(self.constant)
        
        # Add exogenous parameters
        if self.exog_params is not None:
            for i in range(len(self.exog_params)):
                param_names.append(f"Exog({i+1})")
                param_values.append(self.exog_params[i])
        
        # Create DataFrame for parameters
        params_df = pd.DataFrame({
            'Parameter': param_names,
            'Value': param_values
        })
        
        # Add standard errors if available
        if self.standard_errors is not None and len(self.standard_errors) == len(param_values):
            params_df['Std Error'] = self.standard_errors
            
            # Add t-statistics and p-values
            params_df['t-statistic'] = params_df['Value'] / params_df['Std Error']
            params_df['p-value'] = 2 * (1 - stats.t.cdf(np.abs(params_df['t-statistic']), 
                                                     len(self.residuals) - len(param_values)))
        
        # Add model information
        model_info = {
            'Model': f"ARMAX({self.p}, {self.q}){' with constant' if self.include_constant else ''}{' with exogenous variables' if self.exog_params is not None else ''}",
            'Log-likelihood': self.loglikelihood,
            'AIC': self.information_criteria(['aic'])['aic'],
            'BIC': self.information_criteria(['bic'])['bic'],
            'HQIC': self.information_criteria(['hqic'])['hqic'],
            'Residual Variance': self.sigma2
        }
        
        # Combine into a single DataFrame
        model_info_df = pd.DataFrame(list(model_info.items()), columns=['Statistic', 'Value'])
        
        # Create a comprehensive summary DataFrame
        summary_df = pd.concat([
            pd.DataFrame([{'Parameter': 'Model Information', 'Value': ''}]),
            model_info_df.rename(columns={'Statistic': 'Parameter'}),
            pd.DataFrame([{'Parameter': 'Parameter Estimates', 'Value': ''}]),
            params_df
        ], ignore_index=True)
        
        # Store summary
        self.results_summary = summary_df