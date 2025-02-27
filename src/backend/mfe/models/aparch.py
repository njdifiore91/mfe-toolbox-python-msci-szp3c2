"""
MFE Toolbox - APARCH Model

This module implements the Asymmetric Power ARCH (APARCH) model for volatility modeling
in financial time series. It features power transformations and asymmetric volatility 
response to positive and negative returns.

The APARCH model is defined by:
    σ_t^δ = ω + Σ α_i (|r_{t-i}| - γ_i r_{t-i})^δ + Σ β_j σ_{t-j}^δ

Where:
- δ is the power parameter
- γ_i controls asymmetric response to positive vs negative returns
- α_i and β_j are the ARCH and GARCH parameters

References:
    Ding, Z., Granger, C. W., & Engle, R. F. (1993). A long memory property of
    stock market returns and a new model. Journal of Empirical Finance, 1(1), 83-106.
"""

import numpy as np  # numpy 1.26.3
import scipy.optimize  # scipy 1.11.4
from typing import Optional, Dict, Any, Tuple, List, Union, Callable  # Python 3.12
from dataclasses import dataclass  # Python 3.12
import logging  # Python 3.12

# Internal imports
from .volatility import UnivariateVolatilityModel, VOLATILITY_MODELS
from ..utils.validation import validate_array, is_positive_float, check_range
from ..utils.numba_helpers import optimized_jit
from ..utils.numpy_helpers import ensure_array
from ..utils.async_helpers import run_in_executor
from ..utils.statsmodels_helpers import calculate_information_criteria
from ..core.distributions import GeneralizedErrorDistribution, SkewedTDistribution
from ..core.optimization import Optimizer

# Set up module logger
logger = logging.getLogger(__name__)


@optimized_jit
def jit_aparch_recursion(parameters: np.ndarray, data: np.ndarray, delta: float, p: int, q: int) -> np.ndarray:
    """
    Numba-optimized implementation of APARCH variance recursion.
    
    Parameters
    ----------
    parameters : np.ndarray
        Array of model parameters [omega, alpha_1, ..., alpha_q, gamma_1, ..., gamma_q, beta_1, ..., beta_p]
    data : np.ndarray
        Array of return data
    delta : float
        Power parameter
    p : int
        Number of GARCH lags (β parameters)
    q : int
        Number of ARCH lags (α parameters)
        
    Returns
    -------
    np.ndarray
        Array of conditional variance to the power of delta/2
    """
    T = len(data)
    
    # Extract parameters
    omega = parameters[0]
    alpha = parameters[1:q+1]
    gamma = parameters[q+1:2*q+1]  
    beta = parameters[2*q+1:2*q+p+1]
    
    # Initialize power variance array
    power_var = np.zeros(T)
    
    # Set initial values (use unconditional variance estimate)
    unconditional = np.mean(np.abs(data) ** delta)
    power_var[:max(p, q)] = unconditional
    
    # Main APARCH recursion loop
    for t in range(max(p, q), T):
        # Initialize with constant term
        power_var[t] = omega
        
        # ARCH terms with asymmetry (α_i * (|r_{t-i}| - γ_i * r_{t-i})^δ)
        for i in range(q):
            if t-i-1 >= 0:  # Ensure we don't go out of bounds
                abs_ret = np.abs(data[t-i-1])
                asym_term = abs_ret - gamma[i] * data[t-i-1]
                power_var[t] += alpha[i] * (asym_term ** delta)
        
        # GARCH terms (β_j * σ_{t-j}^δ)
        for j in range(p):
            if t-j-1 >= 0:  # Ensure we don't go out of bounds
                power_var[t] += beta[j] * power_var[t-j-1]
    
    return power_var


@optimized_jit
def jit_aparch_likelihood(parameters: np.ndarray, data: np.ndarray, delta: float, 
                         p: int, q: int, distribution_logpdf: Callable) -> float:
    """
    Numba-optimized implementation of APARCH likelihood calculation.
    
    Parameters
    ----------
    parameters : np.ndarray
        Array of model parameters [omega, alpha_1, ..., alpha_q, gamma_1, ..., gamma_q, beta_1, ..., beta_p]
    data : np.ndarray
        Array of return data
    delta : float
        Power parameter
    p : int
        Number of GARCH lags (β parameters)
    q : int
        Number of ARCH lags (α parameters)
    distribution_logpdf : callable
        Function to compute log PDF of the error distribution
        
    Returns
    -------
    float
        Negative log-likelihood value (for minimization)
    """
    # Compute conditional power variances
    power_var = jit_aparch_recursion(parameters, data, delta, p, q)
    
    # Calculate conditional standard deviations
    std_dev = power_var ** (1.0 / delta)
    
    # Compute standardized residuals
    std_resid = data / np.sqrt(std_dev)
    
    # Apply the distribution log-pdf function
    ll_terms = distribution_logpdf(std_resid) - 0.5 * np.log(std_dev)
    
    # Exclude burn-in period from log-likelihood calculation
    valid_idx = np.isfinite(ll_terms)[max(p, q):]
    
    # Sum valid log-likelihood terms
    if np.sum(valid_idx) > 0:
        loglikelihood = np.sum(ll_terms[max(p, q):][valid_idx])
    else:
        # Return large negative value if no valid terms
        return -1e10
    
    # Return negative log-likelihood for minimization
    return -loglikelihood


@dataclass
class APARCH(UnivariateVolatilityModel):
    """
    Asymmetric Power ARCH (APARCH) volatility model.
    
    The APARCH model extends the GARCH model with a flexible power parameter (delta)
    and asymmetric response to positive and negative returns:
    
    σ_t^δ = ω + Σ α_i (|r_{t-i}| - γ_i r_{t-i})^δ + Σ β_j σ_{t-j}^δ
    
    Parameters
    ----------
    p : int
        GARCH order (number of lagged variance terms)
    q : int
        ARCH order (number of lagged squared return terms)
    delta : Optional[float], default=None
        Power parameter (default is 2.0 for standard GARCH-like behavior)
    distribution : str, default='normal'
        Error distribution ('normal', 'student', 'ged', 'skewt')
    distribution_params : dict, default=None
        Additional parameters for the error distribution
    mean_adjustment : bool, default=True
        Whether to adjust returns by subtracting the mean
    """
    p: int
    q: int
    delta: Optional[float] = None  # Default will be set to 2.0 in __init__
    distribution: str = 'normal'
    distribution_params: Dict[str, Any] = None
    mean_adjustment: bool = True
    
    def __init__(self, p: int, q: int, delta: Optional[float] = None, 
                distribution: str = 'normal', distribution_params: Optional[Dict[str, Any]] = None,
                mean_adjustment: bool = True):
        """
        Initialize the APARCH model.
        
        Parameters
        ----------
        p : int
            GARCH order (number of lagged variance terms)
        q : int
            ARCH order (number of lagged squared return terms)
        delta : Optional[float], default=None
            Power parameter (default is 2.0 for standard GARCH-like behavior)
        distribution : str, default='normal'
            Error distribution ('normal', 'student', 'ged', 'skewt')
        distribution_params : Optional[Dict[str, Any]], default=None
            Additional parameters for the error distribution
        mean_adjustment : bool, default=True
            Whether to adjust returns by subtracting the mean
        """
        # Call parent constructor
        super().__init__(distribution=distribution, 
                        distribution_params=distribution_params or {},
                        mean_adjustment=mean_adjustment)
        
        # Validate and store p and q
        if not isinstance(p, int) or p < 0:
            raise ValueError(f"GARCH order (p) must be a non-negative integer, got {p}")
        self.p = p
        
        if not isinstance(q, int) or q < 0:
            raise ValueError(f"ARCH order (q) must be a non-negative integer, got {q}")
        self.q = q
        
        # Set delta (default to 2.0 if not provided)
        if delta is None:
            self.delta = 2.0
        else:
            if not is_positive_float(delta):
                raise ValueError(f"Power parameter (delta) must be positive, got {delta}")
            self.delta = delta
        
        # Initialize model state
        self.parameters = None
        self.conditional_variances = None
        self.fit_stats = None
    
    def validate_parameters(self, parameters: np.ndarray) -> bool:
        """
        Validate the APARCH model parameters.
        
        Parameters
        ----------
        parameters : np.ndarray
            Array of model parameters [omega, alpha_1, ..., alpha_q, gamma_1, ..., gamma_q, beta_1, ..., beta_p]
            
        Returns
        -------
        bool
            True if parameters are valid, False otherwise
        """
        # Validate parameters array
        if not isinstance(parameters, np.ndarray):
            return False
        
        # Check parameter length
        expected_length = 1 + self.q + self.q + self.p  # omega, alphas, gammas, betas
        if len(parameters) != expected_length:
            logger.warning(f"Expected {expected_length} parameters, got {len(parameters)}")
            return False
        
        # Extract parameters
        omega = parameters[0]
        alpha = parameters[1:self.q+1]
        gamma = parameters[self.q+1:2*self.q+1]
        beta = parameters[2*self.q+1:2*self.q+self.p+1]
        
        # Positivity constraint on omega
        if omega <= 0:
            return False
        
        # Positivity constraint on alpha
        if np.any(alpha < 0):
            return False
        
        # Asymmetry constraint on gamma (-1 < gamma < 1)
        if np.any(np.abs(gamma) >= 1):
            return False
        
        # Positivity constraint on beta
        if np.any(beta < 0):
            return False
        
        # Stationarity constraint: sum(alpha_i*(1+gamma_i)^delta/2 + beta_i < 1)
        alpha_sum = np.sum(alpha * np.power((1 + gamma), self.delta/2) + 
                          alpha * np.power((1 - gamma), self.delta/2)) / 2
        beta_sum = np.sum(beta)
        
        if alpha_sum + beta_sum >= 1:
            return False
        
        return True
    
    def _calculate_variance(self, parameters: np.ndarray, data: np.ndarray) -> np.ndarray:
        """
        Calculate conditional variance series for APARCH model.
        
        Parameters
        ----------
        parameters : np.ndarray
            Array of model parameters
        data : np.ndarray
            Return data
            
        Returns
        -------
        np.ndarray
            Conditional variance series
        """
        # Validate parameter array shape and values
        if not self.validate_parameters(parameters):
            raise ValueError("Invalid APARCH parameters")
        
        # Call Numba-optimized recursion for efficient computation
        power_var = jit_aparch_recursion(parameters, data, self.delta, self.p, self.q)
        
        # Convert power variances to standard variances by raising to 2/delta
        variance = power_var ** (2.0 / self.delta)
        
        return variance
    
    def calculate_variance(self, returns: np.ndarray) -> np.ndarray:
        """
        Calculate conditional variance series for fitted APARCH model.
        
        Parameters
        ----------
        returns : np.ndarray
            Return data
            
        Returns
        -------
        np.ndarray
            Conditional variance series
        """
        # Check if model has been fitted
        if self.parameters is None:
            raise RuntimeError("Model must be fitted before calculating variance")
        
        # Preprocess returns to handle mean adjustment
        returns = self.preprocess_returns(returns)
        
        # Call _calculate_variance with fitted parameters
        variance = self._calculate_variance(self.parameters, returns)
        
        return variance
    
    def log_likelihood(self, parameters: np.ndarray, data: np.ndarray) -> float:
        """
        Calculate log-likelihood for the APARCH model.
        
        Parameters
        ----------
        parameters : np.ndarray
            Model parameters
        data : np.ndarray
            Return data
            
        Returns
        -------
        float
            Negative log-likelihood value (for minimization)
        """
        # Validate parameters
        if not self.validate_parameters(parameters):
            return 1e10  # Large penalty for invalid parameters
        
        # Select appropriate distribution log-pdf based on model's distribution
        if self.distribution == 'normal':
            def distribution_logpdf(x):
                return -0.5 * x**2 - 0.5 * np.log(2 * np.pi)
        elif self.distribution == 'student':
            df = self.distribution_params.get('df', 5)
            student_t = getattr(scipy.stats, 't')
            def distribution_logpdf(x):
                return student_t.logpdf(x, df)
        elif self.distribution == 'ged':
            ged = GeneralizedErrorDistribution(**self.distribution_params)
            distribution_logpdf = ged.loglikelihood
        elif self.distribution == 'skewt':
            skewt = SkewedTDistribution(**self.distribution_params)
            distribution_logpdf = skewt.loglikelihood
        else:
            raise ValueError(f"Unsupported distribution: {self.distribution}")
        
        # Call jit_aparch_likelihood with parameters, data, and distribution function
        return jit_aparch_likelihood(parameters, data, self.delta, self.p, self.q, distribution_logpdf)
    
    def negative_log_likelihood(self, parameters: np.ndarray, data: np.ndarray) -> float:
        """
        Objective function for optimization (negative log-likelihood).
        
        Parameters
        ----------
        parameters : np.ndarray
            Model parameters
        data : np.ndarray
            Return data
            
        Returns
        -------
        float
            Negative log-likelihood value
        """
        # If parameters violate constraints, return large positive value (penalty)
        if not self.validate_parameters(parameters):
            return 1e10
        
        try:
            # Otherwise, call log_likelihood and return the result
            return self.log_likelihood(parameters, data)
        except Exception as e:
            # Handle any numerical errors by returning large positive value with logging
            logger.warning(f"Log-likelihood computation failed: {str(e)}")
            return 1e10
    
    def fit(self, returns: np.ndarray, initial_parameters: Optional[np.ndarray] = None,
           options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Fit APARCH model to the return data.
        
        Parameters
        ----------
        returns : np.ndarray
            Return data
        initial_parameters : Optional[np.ndarray], default=None
            Initial parameter values for optimization
        options : Optional[Dict[str, Any]], default=None
            Additional options for optimization
            
        Returns
        -------
        Dict[str, Any]
            Estimation results including parameters and diagnostics
        """
        # Preprocess returns data to handle mean adjustment
        returns = self.preprocess_returns(returns)
        
        # Set default initial parameters if not provided
        if initial_parameters is None:
            # Default initial values based on GARCH-like structure
            omega_init = np.var(returns) * 0.05
            alpha_init = np.ones(self.q) * 0.05
            gamma_init = np.zeros(self.q)  # Start with symmetric model
            beta_init = np.ones(self.p) * 0.85
            
            initial_parameters = np.concatenate(([omega_init], alpha_init, gamma_init, beta_init))
        
        # Validate initial parameters
        if not self.validate_parameters(initial_parameters):
            logger.warning("Initial parameters are invalid, adjusting to valid values...")
            # Adjust to valid values
            omega_init = np.var(returns) * 0.05
            alpha_init = np.ones(self.q) * 0.05
            gamma_init = np.zeros(self.q)  # Start with symmetric model
            beta_init = np.ones(self.p) * 0.85
            
            initial_parameters = np.concatenate(([omega_init], alpha_init, gamma_init, beta_init))
        
        # Configure optimizer options
        if options is None:
            options = {}
        
        # Create objective function from negative_log_likelihood
        def objective(params):
            return self.negative_log_likelihood(params, returns)
        
        # Use Optimizer to minimize negative log-likelihood
        optimizer = Optimizer()
        try:
            optimization_result = optimizer.minimize(
                objective,
                initial_parameters,
                options=options
            )
            
            # Store estimated parameters
            self.parameters = optimization_result.parameters
            
            # Calculate conditional variances with fitted parameters
            self.conditional_variances = self._calculate_variance(self.parameters, returns)
            
            # Compute fit statistics (log-likelihood, AIC, BIC)
            ll = -optimization_result.objective_value
            n_params = len(self.parameters)
            n_obs = len(returns)
            
            aic = 2 * n_params - 2 * ll
            bic = n_params * np.log(n_obs) - 2 * ll
            
            self.fit_stats = {
                'log_likelihood': ll,
                'aic': aic,
                'bic': bic,
                'n_params': n_params,
                'n_observations': n_obs,
                'converged': optimization_result.converged,
                'message': optimization_result.message,
                'iterations': optimization_result.iterations
            }
            
            # Extract parameter components for return
            omega = self.parameters[0]
            alpha = self.parameters[1:self.q+1]
            gamma = self.parameters[self.q+1:2*self.q+1]
            beta = self.parameters[2*self.q+1:2*self.q+self.p+1]
            
            # Return comprehensive results dictionary
            results = {
                'parameters': {
                    'omega': omega,
                    'alpha': alpha.tolist() if hasattr(alpha, 'tolist') else alpha,
                    'gamma': gamma.tolist() if hasattr(gamma, 'tolist') else gamma,
                    'beta': beta.tolist() if hasattr(beta, 'tolist') else beta,
                    'delta': self.delta
                },
                'p': self.p,
                'q': self.q,
                'delta': self.delta,
                'distribution': self.distribution,
                'distribution_params': self.distribution_params,
                'fit_stats': self.fit_stats,
                'conditional_variances': self.conditional_variances,
                'persistence': self.get_persistence(),
                'half_life': self.half_life()
            }
            
            return results
            
        except Exception as e:
            logger.error(f"APARCH model estimation failed: {str(e)}")
            raise RuntimeError(f"APARCH model estimation failed: {str(e)}")
    
    async def fit_async(self, returns: np.ndarray, initial_parameters: Optional[np.ndarray] = None,
                      options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Asynchronously fit APARCH model to the return data.
        
        Parameters
        ----------
        returns : np.ndarray
            Return data
        initial_parameters : Optional[np.ndarray], default=None
            Initial parameter values for optimization
        options : Optional[Dict[str, Any]], default=None
            Additional options for optimization
            
        Returns
        -------
        Dict[str, Any]
            Estimation results including parameters and diagnostics
        """
        # Preprocess returns data to handle mean adjustment
        returns = self.preprocess_returns(returns)
        
        # Set default initial parameters if not provided
        if initial_parameters is None:
            # Default initial values
            omega_init = np.var(returns) * 0.05
            alpha_init = np.ones(self.q) * 0.05
            gamma_init = np.zeros(self.q)  # Start with symmetric model
            beta_init = np.ones(self.p) * 0.85
            
            initial_parameters = np.concatenate(([omega_init], alpha_init, gamma_init, beta_init))
        
        # Configure optimizer options
        if options is None:
            options = {}
        
        # Create objective function from negative_log_likelihood
        def objective(params):
            return self.negative_log_likelihood(params, returns)
        
        # Use Optimizer.async_minimize for asynchronous optimization
        optimizer = Optimizer()
        try:
            # Collect results from the async iterator
            async_result = optimizer.async_minimize(
                objective,
                initial_parameters,
                options=options
            )
            
            iterations, parameters, objective_value = None, None, None
            async for iter_count, params, value in async_result:
                iterations = iter_count
                parameters = params
                objective_value = value
            
            # Process optimization results when completed
            self.parameters = parameters
            
            # Calculate conditional variances
            self.conditional_variances = self._calculate_variance(self.parameters, returns)
            
            # Compute fit statistics
            ll = -objective_value
            n_params = len(self.parameters)
            n_obs = len(returns)
            
            aic = 2 * n_params - 2 * ll
            bic = n_params * np.log(n_obs) - 2 * ll
            
            self.fit_stats = {
                'log_likelihood': ll,
                'aic': aic,
                'bic': bic,
                'n_params': n_params,
                'n_observations': n_obs,
                'converged': True,  # Assume convergence if we got results
                'message': "Optimization completed asynchronously",
                'iterations': iterations
            }
            
            # Extract parameter components
            omega = self.parameters[0]
            alpha = self.parameters[1:self.q+1]
            gamma = self.parameters[self.q+1:2*self.q+1]
            beta = self.parameters[2*self.q+1:2*self.q+self.p+1]
            
            # Return comprehensive results dictionary
            results = {
                'parameters': {
                    'omega': omega,
                    'alpha': alpha.tolist() if hasattr(alpha, 'tolist') else alpha,
                    'gamma': gamma.tolist() if hasattr(gamma, 'tolist') else gamma,
                    'beta': beta.tolist() if hasattr(beta, 'tolist') else beta,
                    'delta': self.delta
                },
                'p': self.p,
                'q': self.q,
                'delta': self.delta,
                'distribution': self.distribution,
                'distribution_params': self.distribution_params,
                'fit_stats': self.fit_stats,
                'conditional_variances': self.conditional_variances,
                'persistence': self.get_persistence(),
                'half_life': self.half_life()
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Async APARCH model estimation failed: {str(e)}")
            raise RuntimeError(f"Async APARCH model estimation failed: {str(e)}")
    
    def forecast(self, returns: np.ndarray, horizon: int) -> np.ndarray:
        """
        Generate volatility forecasts for a specified horizon.
        
        Parameters
        ----------
        returns : np.ndarray
            Array of volatility forecasts
        horizon : int
            Forecast horizon
            
        Returns
        -------
        np.ndarray
            Array of volatility forecasts
        """
        # Check if model has been fitted
        if self.parameters is None:
            raise RuntimeError("Model must be fitted before forecasting")
        
        # Validate horizon
        if not isinstance(horizon, int) or horizon <= 0:
            raise ValueError(f"Horizon must be a positive integer, got {horizon}")
        
        # Preprocess returns
        returns = self.preprocess_returns(returns)
        
        # Extract model parameters
        omega = self.parameters[0]
        alpha = self.parameters[1:self.q+1]
        gamma = self.parameters[self.q+1:2*self.q+1]
        beta = self.parameters[2*self.q+1:2*self.q+self.p+1]
        
        # Calculate current variances
        current_power_var = jit_aparch_recursion(self.parameters, returns, self.delta, self.p, self.q)
        
        # Initialize forecast array
        forecast = np.zeros(horizon)
        
        # Get the last q returns and p variances for initial conditions
        last_returns = returns[-self.q:]
        last_power_var = current_power_var[-self.p:]
        
        # Implement multi-step ahead forecasting recursively
        for h in range(horizon):
            # Initialize with constant term
            forecast[h] = omega
            
            # ARCH terms with asymmetry
            for i in range(min(h, self.q)):
                if i < len(last_returns):
                    abs_ret = np.abs(last_returns[-(i+1)])
                    asym_term = abs_ret - gamma[i] * last_returns[-(i+1)]
                    forecast[h] += alpha[i] * (asym_term ** self.delta)
                else:
                    # For steps beyond available data, use expected value
                    # Expected value of |ε_t| for standard normal is √(2/π)
                    E_abs_eps = np.sqrt(2/np.pi)
                    
                    # Approximate expected value of asymmetric term
                    E_asym_term = E_abs_eps ** self.delta * (forecast[h-i-1] ** (self.delta/2))
                    forecast[h] += alpha[i] * E_asym_term
            
            # GARCH terms
            for j in range(min(h, self.p)):
                if j < len(last_power_var):
                    forecast[h] += beta[j] * last_power_var[-(j+1)]
                else:
                    forecast[h] += beta[j] * forecast[h-j-1]
        
        # Convert from power variance to standard variance (σ^δ -> σ^2)
        forecast_var = forecast ** (2.0 / self.delta)
        
        return forecast_var
    
    def simulate(self, n_periods: int, initial_values: Optional[np.ndarray] = None,
                seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate returns from the APARCH process.
        
        Parameters
        ----------
        n_periods : int
            Number of periods to simulate
        initial_values : Optional[np.ndarray], default=None
            Initial values for the simulation
        seed : Optional[int], default=None
            Random seed for reproducibility
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Tuple containing (simulated_returns, simulated_variances)
        """
        # Check if model has been fitted
        if self.parameters is None:
            raise RuntimeError("Model must be fitted before simulation")
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Extract model parameters
        omega = self.parameters[0]
        alpha = self.parameters[1:self.q+1]
        gamma = self.parameters[self.q+1:2*self.q+1]
        beta = self.parameters[2*self.q+1:2*self.q+self.p+1]
        
        # Initialize arrays for returns and variances
        max_lag = max(self.p, self.q)
        sim_length = n_periods + max_lag
        
        power_var = np.zeros(sim_length)
        returns = np.zeros(sim_length)
        
        # Set up initial values for recursion
        if initial_values is not None:
            if len(initial_values) < max_lag:
                raise ValueError(f"initial_values must have at least {max_lag} elements")
            returns[:max_lag] = initial_values[:max_lag]
        else:
            # Use random initial values based on unconditional variance
            uncond_var = (omega / (1 - np.sum(alpha) - np.sum(beta))) ** (2.0 / self.delta)
            returns[:max_lag] = np.random.randn(max_lag) * np.sqrt(uncond_var)
        
        # Initialize power variance with unconditional estimate
        power_var[:max_lag] = omega / (1 - np.sum(alpha) - np.sum(beta))
        
        # Generate random innovations based on the specified distribution
        if self.distribution == 'normal':
            innovations = np.random.randn(n_periods)
        elif self.distribution == 'student':
            df = self.distribution_params.get('df', 5)
            innovations = np.random.standard_t(df, size=n_periods)
        elif self.distribution == 'ged':
            ged = GeneralizedErrorDistribution(**self.distribution_params)
            innovations = ged.random(n_periods)
        elif self.distribution == 'skewt':
            skewt = SkewedTDistribution(**self.distribution_params)
            innovations = skewt.random(n_periods)
        else:
            raise ValueError(f"Unsupported distribution: {self.distribution}")
        
        # Implement APARCH recursion to generate variances
        for t in range(max_lag, sim_length):
            # Compute conditional power variance
            power_var[t] = omega
            
            # ARCH terms with asymmetry
            for i in range(self.q):
                abs_ret = np.abs(returns[t-i-1])
                asym_term = abs_ret - gamma[i] * returns[t-i-1]
                power_var[t] += alpha[i] * (asym_term ** self.delta)
            
            # GARCH terms
            for j in range(self.p):
                power_var[t] += beta[j] * power_var[t-j-1]
            
            # Calculate returns as sqrt(variance) * innovation
            std_dev = power_var[t] ** (1.0 / self.delta)
            returns[t] = np.sqrt(std_dev) * innovations[t - max_lag]
        
        # Return simulated returns and variances
        simulated_returns = returns[max_lag:]
        simulated_variances = power_var[max_lag:] ** (2.0 / self.delta)
        
        return simulated_returns, simulated_variances
    
    def summary(self) -> Dict[str, Any]:
        """
        Generate a summary of the fitted APARCH model.
        
        Returns
        -------
        Dict[str, Any]
            Model summary information
        """
        # Check if model has been fitted
        if self.parameters is None:
            raise RuntimeError("Model must be fitted before generating a summary")
        
        # Extract parameter estimates and create parameter dictionary
        omega = self.parameters[0]
        alpha = self.parameters[1:self.q+1]
        gamma = self.parameters[self.q+1:2*self.q+1]
        beta = self.parameters[2*self.q+1:2*self.q+self.p+1]
        
        # Parameter names with indices
        param_names = ['omega']
        for i in range(self.q):
            param_names.append(f'alpha[{i+1}]')
        for i in range(self.q):
            param_names.append(f'gamma[{i+1}]')
        for i in range(self.p):
            param_names.append(f'beta[{i+1}]')
        
        # Create parameters dictionary with names
        parameters_dict = dict(zip(param_names, self.parameters))
        
        # Calculate persistence measure
        persistence = self.get_persistence()
        half_life = self.half_life()
        
        # Return comprehensive summary dictionary
        summary = {
            'model': 'APARCH',
            'p': self.p,
            'q': self.q,
            'delta': self.delta,
            'distribution': self.distribution,
            'distribution_params': self.distribution_params,
            'parameters': parameters_dict,
            'persistence': persistence,
            'half_life': half_life,
            'fit_stats': self.fit_stats
        }
        
        return summary
    
    def get_persistence(self) -> float:
        """
        Calculate the volatility persistence of the APARCH process.
        
        Returns
        -------
        float
            Persistence measure
        """
        # Check if model has been fitted
        if self.parameters is None:
            raise RuntimeError("Model must be fitted before calculating persistence")
        
        # Extract alpha (ARCH), gamma (asymmetry), and beta (GARCH) parameters
        alpha = self.parameters[1:self.q+1]
        gamma = self.parameters[self.q+1:2*self.q+1]
        beta = self.parameters[2*self.q+1:2*self.q+self.p+1]
        
        # Calculate persistence as sum of alpha_i*(1+gamma_i)^delta/2 + sum of beta_i
        alpha_persistence = np.sum(alpha * np.power((1 + gamma), self.delta/2) + 
                                 alpha * np.power((1 - gamma), self.delta/2)) / 2
        beta_persistence = np.sum(beta)
        
        return alpha_persistence + beta_persistence
    
    def half_life(self) -> float:
        """
        Calculate the volatility half-life of the APARCH process.
        
        Returns
        -------
        float
            Half-life in periods
        """
        # Check if model has been fitted
        if self.parameters is None:
            raise RuntimeError("Model must be fitted before calculating half-life")
        
        # Calculate persistence
        persistence = self.get_persistence()
        
        # Check for unit root or near-unit root process
        if persistence >= 1.0:
            return np.inf
        elif persistence <= 0.0:
            return 0.0
        
        # Calculate half-life as log(0.5) / log(persistence)
        return np.log(0.5) / np.log(persistence)


# Register the APARCH model in the VOLATILITY_MODELS registry
VOLATILITY_MODELS['APARCH'] = APARCH