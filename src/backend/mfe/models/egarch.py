"""
MFE Toolbox - Exponential GARCH (EGARCH) Model

This module implements the Exponential GARCH (EGARCH) model for volatility modeling
in financial time series. The EGARCH model captures asymmetric effects in volatility
and uses a logarithmic variance formulation to ensure positivity without parameter
constraints.

The implementation includes robust parameter estimation, forecasting, simulation,
and diagnostic methods, optimized with Numba for high performance.

Key features:
- Logarithmic variance formulation ensuring positive variance
- Asymmetric effects through leverage parameters
- Persistence and half-life calculation
- Numba-optimized core functions
- Asynchronous model fitting with Python's async/await pattern
"""

import numpy as np  # numpy 1.26.3
import scipy.optimize  # scipy 1.11.4
from typing import Dict, List, Optional, Tuple, Union, Any, Callable  # Python 3.12
from dataclasses import dataclass  # Python 3.12
import logging  # Python 3.12
import math  # Python 3.12

# Internal imports
from .volatility import UnivariateVolatilityModel, VOLATILITY_MODELS
from ..utils.validation import validate_parameter, is_positive_float
from ..utils.numba_helpers import optimized_jit
from ..utils.numpy_helpers import ensure_array
from ..utils.async_helpers import run_in_executor
from ..utils.statsmodels_helpers import calculate_information_criteria

# Set up logger
logger = logging.getLogger(__name__)


@optimized_jit
def jit_egarch_recursion(parameters: np.ndarray, data: np.ndarray, p: int, o: int, q: int) -> np.ndarray:
    """
    Numba-optimized implementation of EGARCH log-variance recursion for performance-critical volatility calculations.
    
    Parameters
    ----------
    parameters : ndarray
        Array of EGARCH parameters [omega, alpha_1, ..., alpha_q, gamma_1, ..., gamma_o, beta_1, ..., beta_p]
    data : ndarray
        Array of return data for EGARCH modeling
    p : int
        Number of GARCH lags (beta parameters)
    o : int
        Number of leverage lags (gamma parameters)
    q : int
        Number of ARCH lags (alpha parameters)
        
    Returns
    -------
    ndarray
        Array of conditional variances
    """
    T = len(data)
    
    # Extract parameters
    omega = parameters[0]
    alpha = parameters[1:q+1]
    gamma = parameters[q+1:q+o+1]
    beta = parameters[q+o+1:q+o+p+1]
    
    # Initialize log-variance and variance arrays
    log_variance = np.zeros(T)
    variance = np.zeros(T)
    
    # Calculate unconditional variance for initialization
    # For EGARCH, the unconditional variance is more complex due to the log transformation
    # We use a simple approximation for initialization
    log_uncond_var = omega / (1.0 - np.sum(beta))
    uncond_var = np.exp(log_uncond_var)
    
    # Initialize first max(p, q, o) variances with unconditional variance
    max_lag = max(p, q, o)
    variance[:max_lag] = uncond_var
    log_variance[:max_lag] = log_uncond_var
    
    # Compute expected absolute value of z (for normal distribution, this is sqrt(2/pi))
    expected_abs_z = np.sqrt(2.0 / np.pi)
    
    # Main recursion loop
    for t in range(max_lag, T):
        # Previous standardized residuals (z_t)
        z = np.zeros(max(q, o))
        for i in range(max(q, o)):
            if t-i-1 >= 0 and variance[t-i-1] > 0:
                z[i] = data[t-i-1] / np.sqrt(variance[t-i-1])
        
        # Calculate log-variance using EGARCH formula
        log_var_t = omega
        
        # ARCH component with absolute value adjustment (|z|-E[|z|])
        for i in range(q):
            if t-i-1 >= 0:
                log_var_t += alpha[i] * (np.abs(z[i]) - expected_abs_z)
        
        # Leverage component (z itself)
        for i in range(o):
            if t-i-1 >= 0:
                log_var_t += gamma[i] * z[i]
        
        # GARCH component (previous log-variances)
        for i in range(p):
            if t-i-1 >= 0:
                log_var_t += beta[i] * log_variance[t-i-1]
        
        # Store log-variance and convert to variance
        log_variance[t] = log_var_t
        variance[t] = np.exp(log_var_t)
    
    return variance


@optimized_jit
def jit_egarch_likelihood(parameters: np.ndarray, data: np.ndarray, p: int, o: int, q: int, 
                        distribution_logpdf: Callable) -> float:
    """
    Numba-optimized implementation of EGARCH likelihood calculation for parameter estimation.
    
    Parameters
    ----------
    parameters : ndarray
        Array of EGARCH parameters [omega, alpha_1, ..., alpha_q, gamma_1, ..., gamma_o, beta_1, ..., beta_p]
    data : ndarray
        Array of return data for EGARCH modeling
    p : int
        Number of GARCH lags (beta parameters)
    o : int
        Number of leverage lags (gamma parameters)
    q : int
        Number of ARCH lags (alpha parameters)
    distribution_logpdf : callable
        Function to compute log PDF for the error distribution
        
    Returns
    -------
    float
        Negative log-likelihood value
    """
    # Compute conditional variances
    variance = jit_egarch_recursion(parameters, data, p, o, q)
    
    # Initialize log-likelihood
    loglike = 0.0
    
    # Determine effective sample size (excluding burn-in period)
    max_lag = max(p, q, o)
    T = len(data)
    
    # Calculate log-likelihood using standardized residuals
    for t in range(max_lag, T):
        if variance[t] <= 0:
            return 1e10  # Large penalty for invalid variances
        
        # Calculate standardized residual
        std_resid = data[t] / np.sqrt(variance[t])
        
        # Calculate log density for the standardized residual
        term = distribution_logpdf(std_resid)
        
        # Add log density and adjustment for variance transformation
        loglike += term - 0.5 * np.log(variance[t])
    
    # Return negative log-likelihood for minimization
    return -loglike


def calculate_expected_abs_z(distribution: str, dist_params: Dict[str, float]) -> float:
    """
    Calculate the expected value of absolute standardized residuals for various distributions.
    
    Parameters
    ----------
    distribution : str
        Distribution type ('normal', 'student', 'ged', 'skewed_t')
    dist_params : dict
        Parameters for the specified distribution
        
    Returns
    -------
    float
        Expected value of |z|
    """
    # For normal distribution, E[|z|] = sqrt(2/π)
    if distribution.lower() == 'normal':
        return np.sqrt(2.0 / np.pi)
    
    # For Student's t-distribution with v degrees of freedom
    elif distribution.lower() == 'student' or distribution.lower() == 't':
        v = dist_params.get('nu', dist_params.get('df', 5.0))
        if v <= 2:
            v = 5.0  # Default to reasonable value if invalid
            logger.warning("Invalid degrees of freedom for t-distribution. Using v=5.0")
            
        # E[|z|] for t-distribution = Γ((v-1)/2) / (Γ(v/2) * sqrt(π * (v-2)))
        return (math.gamma((v - 1) / 2) / (math.gamma(v / 2) * np.sqrt(np.pi * (v - 2))))
    
    # For Generalized Error Distribution (GED)
    elif distribution.lower() == 'ged':
        v = dist_params.get('nu', 2.0)
        if v <= 0:
            v = 2.0  # Default to normal distribution if invalid
            logger.warning("Invalid shape parameter for GED. Using v=2.0")
            
        # E[|z|] for GED = 2^(1/v) * Γ(2/v) / Γ(1/v)
        return (2 ** (1 / v) * math.gamma(2 / v) / math.gamma(1 / v))
    
    # For Hansen's skewed t-distribution
    elif distribution.lower() == 'skewed_t':
        v = dist_params.get('nu', 5.0)
        lambda_ = dist_params.get('lambda', 0.0)
        
        if v <= 2:
            v = 5.0  # Default to reasonable value if invalid
            logger.warning("Invalid degrees of freedom for skewed t-distribution. Using v=5.0")
        
        # For skewed t, the calculation is complex. We use an approximation here
        # For lambda = 0 (symmetric case), this reduces to the t formula above
        t_factor = (math.gamma((v - 1) / 2) / (math.gamma(v / 2) * np.sqrt(np.pi * (v - 2))))
        skew_adjustment = 1.0 + 0.5 * abs(lambda_)  # Approximate adjustment for skewness
        
        return t_factor * skew_adjustment
    
    # Default to normal distribution if unknown
    else:
        logger.warning(f"Unknown distribution: {distribution}. Using normal distribution.")
        return np.sqrt(2.0 / np.pi)


@dataclass
class EGARCH(UnivariateVolatilityModel):
    """
    Implementation of the Exponential GARCH (p,o,q) model that captures asymmetric effects
    in volatility using a logarithmic variance formulation.
    
    The EGARCH model uses the following formulation for the conditional variance:
    log(σ²_t) = ω + Σα_i(|z_{t-i}|-E[|z_{t-i}|]) + Σγ_i*z_{t-i} + Σβ_j*log(σ²_{t-j})
    
    where:
    - z_t = ε_t/σ_t is the standardized residual
    - ω is the constant term
    - α_i are the ARCH parameters capturing magnitude effects
    - γ_i are the leverage parameters capturing sign effects
    - β_j are the GARCH parameters capturing persistence
    
    The log-variance formulation ensures positive variance without constraints on parameters.
    
    Parameters
    ----------
    p : int
        Number of GARCH lags (β parameters)
    o : int
        Number of leverage lags (γ parameters)
    q : int
        Number of ARCH lags (α parameters)
    distribution : str, default='normal'
        Error distribution: 'normal', 'student', 'ged', or 'skewed_t'
    distribution_params : dict, default=None
        Parameters for the error distribution
    """
    p: int
    o: int
    q: int
    distribution: str = 'normal'
    distribution_params: Dict[str, float] = None
    
    def __init__(self, p: int = 1, o: int = 1, q: int = 1, 
                distribution: str = 'normal', 
                distribution_params: Optional[Dict[str, float]] = None):
        """
        Initialize EGARCH model with specified orders and distribution.
        
        Parameters
        ----------
        p : int, default=1
            Number of GARCH lags (β parameters)
        o : int, default=1
            Number of leverage lags (γ parameters)
        q : int, default=1
            Number of ARCH lags (α parameters)
        distribution : str, default='normal'
            Error distribution: 'normal', 'student', 'ged', or 'skewed_t'
        distribution_params : dict, default=None
            Parameters for the error distribution
        """
        # Call parent constructor
        super().__init__(distribution=distribution, distribution_params=distribution_params or {})
        
        # Validate and set model orders
        if not isinstance(p, int) or p < 0:
            raise ValueError("GARCH order (p) must be a non-negative integer")
        if not isinstance(o, int) or o < 0:
            raise ValueError("Leverage order (o) must be a non-negative integer")
        if not isinstance(q, int) or q < 0:
            raise ValueError("ARCH order (q) must be a non-negative integer")
        
        self.p = p
        self.o = o
        self.q = q
        
        # Initialize additional EGARCH attributes
        self.parameters = None
        self.fit_stats = {}
        self.conditional_variances = None
        
        # Parameter components
        self.omega = None
        self.alpha = None
        self.gamma = None
        self.beta = None
        
        # Calculate expected absolute value of standardized residuals
        self.expected_abs_z = calculate_expected_abs_z(
            self.distribution, self.distribution_params
        )

    def _validate_parameters(self, params: np.ndarray) -> bool:
        """
        Validate parameter values for the EGARCH model.
        
        Parameters
        ----------
        params : ndarray
            Array of EGARCH parameters [omega, alpha_1, ..., alpha_q, gamma_1, ..., gamma_o, beta_1, ..., beta_p]
            
        Returns
        -------
        bool
            True if parameters are valid
        """
        # Check correct parameter array length
        expected_length = 1 + self.q + self.o + self.p
        if len(params) != expected_length:
            logger.warning(f"Parameter array length is {len(params)}, expected {expected_length}")
            return False
        
        # Extract parameters
        omega, alpha, gamma, beta = self._extract_parameters(params)
        
        # For EGARCH, we don't require positivity of alpha and beta
        # The main requirement is that the model is stationary: sum(beta) < 1
        if np.sum(beta) >= 1:
            logger.warning(f"Sum of beta parameters ({np.sum(beta)}) must be < 1 for stationarity")
            return False
        
        return True
    
    def _extract_parameters(self, params: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract individual parameter components from parameter vector.
        
        Parameters
        ----------
        params : ndarray
            Array of EGARCH parameters
            
        Returns
        -------
        tuple
            (omega, alpha, gamma, beta)
        """
        omega = params[0]
        alpha = params[1:self.q+1]
        gamma = params[self.q+1:self.q+self.o+1]
        beta = params[self.q+self.o+1:]
        
        return omega, alpha, gamma, beta
    
    def _variance(self, parameters: np.ndarray, data: np.ndarray) -> np.ndarray:
        """
        Calculate conditional variance series for EGARCH(p,o,q) model.
        
        Parameters
        ----------
        parameters : ndarray
            Array of model parameters
        data : ndarray
            Array of return data
            
        Returns
        -------
        ndarray
            Conditional variance series
        """
        # Validate parameter array
        if not self._validate_parameters(parameters):
            raise ValueError("Invalid parameters for EGARCH model")
        
        # Use Numba-optimized function for efficient computation
        variance = jit_egarch_recursion(parameters, data, self.p, self.o, self.q)
        
        # Store the computed variances
        self.conditional_variances = variance
        
        return variance
    
    def _log_likelihood(self, parameters: np.ndarray, data: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the EGARCH model.
        
        Parameters
        ----------
        parameters : ndarray
            Array of model parameters
        data : ndarray
            Array of return data
            
        Returns
        -------
        float
            Negative log-likelihood value
        """
        # Get the appropriate distribution log PDF function
        if self.distribution.lower() == 'normal':
            log_pdf = lambda x: -0.5 * x**2 - 0.5 * np.log(2 * np.pi)
        elif self.distribution.lower() in ['student', 't']:
            nu = self.distribution_params.get('nu', self.distribution_params.get('df', 5.0))
            log_pdf = lambda x: scipy.stats.t.logpdf(x, df=nu)
        elif self.distribution.lower() == 'ged':
            nu = self.distribution_params.get('nu', 2.0)
            log_pdf = lambda x: scipy.stats.gennorm.logpdf(x, beta=nu)
        elif self.distribution.lower() == 'skewed_t':
            nu = self.distribution_params.get('nu', 5.0)
            lambda_ = self.distribution_params.get('lambda', 0.0)
            log_pdf = lambda x: scipy.stats.skewnorm.logpdf(x, a=lambda_, loc=0, scale=1) # Approximation
        else:
            # Default to normal distribution
            log_pdf = lambda x: -0.5 * x**2 - 0.5 * np.log(2 * np.pi)
        
        # Calculate negative log-likelihood
        return jit_egarch_likelihood(parameters, data, self.p, self.o, self.q, log_pdf)
    
    def _forecast(self, horizon: int) -> np.ndarray:
        """
        Generate volatility forecasts for a specified horizon.
        
        Parameters
        ----------
        horizon : int
            Forecast horizon
            
        Returns
        -------
        ndarray
            Forecasted conditional variances
        """
        if self.parameters is None:
            raise ValueError("Model must be fitted before forecasting")
        
        # Extract model parameters
        omega, alpha, gamma, beta = self._extract_parameters(self.parameters)
        
        # Initialize forecast array
        forecast = np.zeros(horizon)
        
        # Set up initial values for forecast recursion
        if self.conditional_variances is None:
            raise ValueError("Conditional variances are not available")
        
        # Get the most recent variance values for initialization
        T = len(self.conditional_variances)
        
        # Extract necessary history for recursion, reversing order for convenient indexing
        recent_log_var = np.log(self.conditional_variances[max(0, T-self.p):T])[::-1]
        
        # Need the standardized residuals for the ARCH and leverage components
        # We need data at the same indices as the recent_log_var
        data_needed = T - len(recent_log_var)
        if data_needed > 0:
            raise ValueError(f"Not enough data for forecasting. Need {data_needed} more observations.")
        
        # Calculate standardized residuals
        recent_std_resids = np.zeros(max(self.q, self.o))
        for i in range(min(max(self.q, self.o), T)):
            if self.conditional_variances[T-i-1] > 0:
                recent_std_resids[i] = self.data[T-i-1] / np.sqrt(self.conditional_variances[T-i-1])
        
        # Initialize first forecast using the EGARCH formula
        log_var_t = omega
        
        # ARCH component with absolute value adjustment (|z|-E[|z|])
        for i in range(min(self.q, len(recent_std_resids))):
            log_var_t += alpha[i] * (np.abs(recent_std_resids[i]) - self.expected_abs_z)
        
        # Leverage component (z itself)
        for i in range(min(self.o, len(recent_std_resids))):
            log_var_t += gamma[i] * recent_std_resids[i]
        
        # GARCH component (previous log-variances)
        for i in range(min(self.p, len(recent_log_var))):
            log_var_t += beta[i] * recent_log_var[i]
        
        # Store first forecast
        forecast[0] = np.exp(log_var_t)
        
        # Forecast for remaining horizon
        for h in range(1, horizon):
            # For multi-step forecasts, we don't have standardized residuals, so we use their expectations
            # The expectation of z_t is 0
            # The expectation of |z_t| depends on the distribution and is stored in self.expected_abs_z
            
            log_var_t = omega
            
            # ARCH component with absolute value adjustment
            # For h > 1, we use E[|z|] for both |z| and E[|z|], which nets to 0
            if h <= self.q:
                # For short horizons, we still have some known standardized residuals
                for i in range(min(self.q, h-1)):
                    # Note that we use expectation of 0 for unknown future standardized residuals
                    log_var_t += 0  # Zero because E[|z|] - E[|z|] = 0
                
                # Use known values for the most recent observations
                for i in range(h-1, self.q):
                    log_var_t += alpha[i] * (np.abs(recent_std_resids[i-h+1]) - self.expected_abs_z)
            
            # Leverage component
            # For h > 1, we use E[z] = 0
            if h <= self.o:
                # For short horizons, we still have some known standardized residuals
                for i in range(min(self.o, h-1)):
                    # Note that we use expectation of 0 for unknown future standardized residuals
                    log_var_t += 0  # Zero because E[z] = 0
                
                # Use known values for the most recent observations
                for i in range(h-1, self.o):
                    log_var_t += gamma[i] * recent_std_resids[i-h+1]
            
            # GARCH component
            for i in range(self.p):
                if i < h:
                    # Use previously forecasted log-variances
                    log_var_t += beta[i] * np.log(forecast[h-i-1])
                else:
                    # Use actual log-variances from history
                    log_var_t += beta[i] * recent_log_var[i-h]
            
            # Store forecast
            forecast[h] = np.exp(log_var_t)
        
        return forecast
    
    def _simulate(self, n_periods: int, initial_data: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate returns from the EGARCH process.
        
        Parameters
        ----------
        n_periods : int
            Number of periods to simulate
        initial_data : ndarray, optional
            Initial data for the simulation
            
        Returns
        -------
        tuple
            (simulated_returns, simulated_variances)
        """
        if self.parameters is None:
            raise ValueError("Model must be fitted before simulation")
        
        # Extract model parameters
        omega, alpha, gamma, beta = self._extract_parameters(self.parameters)
        
        # Set up simulation arrays
        returns = np.zeros(n_periods)
        log_variance = np.zeros(n_periods)
        variance = np.zeros(n_periods)
        
        # Determine the appropriate distribution for innovations
        if self.distribution.lower() == 'normal':
            random_generator = lambda size: np.random.standard_normal(size)
        elif self.distribution.lower() in ['student', 't']:
            nu = self.distribution_params.get('nu', self.distribution_params.get('df', 5.0))
            random_generator = lambda size: np.random.standard_t(df=nu, size=size)
        elif self.distribution.lower() == 'ged':
            nu = self.distribution_params.get('nu', 2.0)
            random_generator = lambda size: np.random.normal(size=size)  # Approximation
        elif self.distribution.lower() == 'skewed_t':
            random_generator = lambda size: np.random.normal(size=size)  # Approximation
        else:
            # Default to normal distribution
            random_generator = lambda size: np.random.standard_normal(size)
        
        # Generate random innovations
        z = random_generator(n_periods)
        
        # Initialize first max(p,q,o) values using unconditional variance
        max_lag = max(self.p, self.q, self.o)
        log_uncond_var = omega / (1.0 - np.sum(beta))
        uncond_var = np.exp(log_uncond_var)
        
        # If initial data is provided, use it for initialization
        if initial_data is not None:
            initial_data = ensure_array(initial_data)
            if len(initial_data) < max_lag:
                raise ValueError(f"Initial data length ({len(initial_data)}) must be at least max_lag ({max_lag})")
            
            # Use initial data to compute initial values
            for t in range(max_lag):
                returns[t] = initial_data[t]
                variance[t] = uncond_var
                log_variance[t] = log_uncond_var
        else:
            # Initialize with unconditional variance
            for t in range(max_lag):
                # Generate returns based on unconditional variance
                variance[t] = uncond_var
                log_variance[t] = log_uncond_var
                returns[t] = np.sqrt(variance[t]) * z[t]
        
        # Main simulation loop
        for t in range(max_lag, n_periods):
            # Calculate log-variance using EGARCH formula
            log_var_t = omega
            
            # ARCH component with absolute value adjustment (|z|-E[|z|])
            for i in range(self.q):
                if t-i-1 >= 0:
                    z_lag = returns[t-i-1] / np.sqrt(variance[t-i-1])
                    log_var_t += alpha[i] * (np.abs(z_lag) - self.expected_abs_z)
            
            # Leverage component (z itself)
            for i in range(self.o):
                if t-i-1 >= 0:
                    z_lag = returns[t-i-1] / np.sqrt(variance[t-i-1])
                    log_var_t += gamma[i] * z_lag
            
            # GARCH component (previous log-variances)
            for i in range(self.p):
                if t-i-1 >= 0:
                    log_var_t += beta[i] * log_variance[t-i-1]
            
            # Store log-variance and convert to variance
            log_variance[t] = log_var_t
            variance[t] = np.exp(log_var_t)
            
            # Generate return
            returns[t] = np.sqrt(variance[t]) * z[t]
        
        return returns, variance
    
    async def fit_async(self, data: np.ndarray, initial_params: Optional[np.ndarray] = None,
                     options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Asynchronous version of the fit method for EGARCH model.
        
        Parameters
        ----------
        data : ndarray
            Return data for volatility modeling
        initial_params : ndarray, optional
            Initial parameter values for optimization
        options : dict, optional
            Additional options for optimization
            
        Returns
        -------
        dict
            Optimization results
        """
        # Store data for later use
        self.data = ensure_array(data)
        
        # Preprocess returns for volatility modeling
        preprocessed_data = self.preprocess_returns(self.data)
        
        # Set up default options
        if options is None:
            options = {}
        
        # Set default optimization method
        if 'method' not in options:
            options['method'] = 'SLSQP'
        
        # Generate initial parameters if not provided
        if initial_params is None:
            # Sensible defaults for EGARCH
            omega = -0.1  # Small negative value often works well for EGARCH
            alpha = np.ones(self.q) * 0.1
            gamma = np.ones(self.o) * -0.1  # Negative value for leverage effect
            beta = np.ones(self.p) * 0.8
            
            initial_params = np.concatenate([[omega], alpha, gamma, beta])
        
        # Ensure initial parameters are an array
        initial_params = ensure_array(initial_params)
        
        # Check if initial parameters are valid
        if not self._validate_parameters(initial_params):
            logger.warning("Initial parameters are invalid. Using default values.")
            # Use default parameters
            omega = -0.1
            alpha = np.ones(self.q) * 0.1
            gamma = np.ones(self.o) * -0.1
            beta = np.ones(self.p) * 0.8
            
            initial_params = np.concatenate([[omega], alpha, gamma, beta])
        
        # Define the objective function for optimization
        def objective(params):
            return self._log_likelihood(params, preprocessed_data)
        
        # Set up bounds for optimization
        # In EGARCH, parameters don't need to be positive, but we do need stationarity
        # We can set loose bounds to help the optimization
        bounds = []
        # Omega can be positive or negative
        bounds.append((-10.0, 10.0))
        # Alpha can be positive or negative, but usually positive
        for _ in range(self.q):
            bounds.append((-1.0, 2.0))
        # Gamma can be positive or negative for leverage effects
        for _ in range(self.o):
            bounds.append((-2.0, 2.0))
        # Beta should be positive and sum to less than 1 for stationarity
        for _ in range(self.p):
            bounds.append((0.01, 0.999))
        
        # Add bounds to options
        options['bounds'] = bounds
        
        # Define wrapper for async optimization
        async def async_optimize():
            try:
                # Run optimization in executor to avoid blocking
                return await run_in_executor(
                    scipy.optimize.minimize,
                    objective,
                    initial_params,
                    method=options.get('method', 'SLSQP'),
                    bounds=options.get('bounds'),
                    options={
                        'maxiter': options.get('maxiter', 1000),
                        'disp': options.get('disp', False)
                    }
                )
            except Exception as e:
                logger.error(f"Optimization failed: {str(e)}")
                raise ValueError(f"EGARCH optimization failed: {str(e)}")
        
        # Run optimization asynchronously
        try:
            result = await async_optimize()
            
            # Check if optimization was successful
            if not result.success:
                logger.warning(f"Optimization did not converge: {result.message}")
            
            # Store optimized parameters
            self.parameters = result.x
            
            # Extract individual parameters
            self.omega, self.alpha, self.gamma, self.beta = self._extract_parameters(self.parameters)
            
            # Calculate conditional variances
            self.conditional_variances = self._variance(self.parameters, preprocessed_data)
            
            # Calculate log-likelihood, AIC, and BIC
            likelihood = -result.fun
            n_params = len(self.parameters)
            n_observations = len(data) - max(self.p, self.q, self.o)
            
            aic, bic = calculate_information_criteria(likelihood, n_params, n_observations)
            
            # Store fit statistics
            self.fit_stats = {
                'log_likelihood': likelihood,
                'aic': aic,
                'bic': bic,
                'num_params': n_params,
                'converged': result.success,
                'message': result.message,
                'iterations': result.nit
            }
            
            # Convert optimization result to dictionary for return
            return {
                'parameters': self.parameters,
                'log_likelihood': likelihood,
                'aic': aic,
                'bic': bic,
                'converged': result.success,
                'message': result.message,
                'iterations': result.nit,
                'status': result.status
            }
        
        except Exception as e:
            logger.error(f"Error during EGARCH fitting: {str(e)}")
            raise RuntimeError(f"EGARCH fitting failed: {str(e)}")
    
    def summary(self) -> Dict[str, Any]:
        """
        Generate a summary of the fitted EGARCH model.
        
        Returns
        -------
        dict
            Model summary information
        """
        if self.parameters is None:
            raise ValueError("Model has not been fitted")
        
        # Extract parameter estimates
        omega, alpha, gamma, beta = self._extract_parameters(self.parameters)
        
        # Create parameter names
        param_names = ['omega']
        param_names.extend([f'alpha[{i+1}]' for i in range(len(alpha))])
        param_names.extend([f'gamma[{i+1}]' for i in range(len(gamma))])
        param_names.extend([f'beta[{i+1}]' for i in range(len(beta))])
        
        # Combine parameter values
        param_values = [omega]
        param_values.extend(alpha)
        param_values.extend(gamma)
        param_values.extend(beta)
        
        # Create parameter summary
        params = {}
        for name, value in zip(param_names, param_values):
            params[name] = value
        
        # Calculate persistence
        persistence = np.sum(beta)
        
        # Calculate half-life
        half_life = np.log(0.5) / np.log(persistence) if persistence < 1 else np.inf
        
        # Compile summary dictionary
        summary = {
            'model': 'EGARCH',
            'orders': {
                'p': self.p,
                'o': self.o,
                'q': self.q
            },
            'distribution': self.distribution,
            'distribution_params': self.distribution_params,
            'parameters': params,
            'fit_stats': self.fit_stats,
            'persistence': persistence,
            'half_life': half_life
        }
        
        return summary
    
    def get_persistence(self) -> float:
        """
        Calculate the persistence of the EGARCH process.
        
        Returns
        -------
        float
            Persistence value
        """
        if self.parameters is None:
            raise ValueError("Model has not been fitted")
        
        # Extract parameters
        _, _, _, beta = self._extract_parameters(self.parameters)
        
        # Persistence is the sum of beta coefficients
        persistence = np.sum(beta)
        
        return persistence
    
    def half_life(self) -> float:
        """
        Calculate the volatility half-life of the EGARCH process.
        
        Returns
        -------
        float
            Half-life in periods
        """
        # Get persistence
        persistence = self.get_persistence()
        
        # Calculate half-life
        if persistence >= 1:
            return np.inf
        else:
            return np.log(0.5) / np.log(persistence)
    
    def get_news_impact_curve(self, z_range: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate the news impact curve showing the effect of shocks on volatility.
        
        Parameters
        ----------
        z_range : ndarray, optional
            Range of standardized residuals (z) to evaluate the curve.
            If None, a default range of -5 to 5 is used.
            
        Returns
        -------
        ndarray
            Impact on log-variance for each z value
        """
        if self.parameters is None:
            raise ValueError("Model has not been fitted")
        
        # Extract parameters
        _, alpha, gamma, _ = self._extract_parameters(self.parameters)
        
        # Create default z_range if not provided
        if z_range is None:
            z_range = np.linspace(-5, 5, 100)
        
        # Ensure z_range is a NumPy array
        z_range = ensure_array(z_range)
        
        # Calculate news impact for each value of z
        # We focus on the impact in log-variance space
        impact = np.zeros_like(z_range)
        
        for i, z in enumerate(z_range):
            # Calculate ARCH component for each alpha
            arch_impact = 0
            for j in range(len(alpha)):
                arch_impact += alpha[j] * (np.abs(z) - self.expected_abs_z)
            
            # Calculate leverage component for each gamma
            leverage_impact = 0
            for j in range(len(gamma)):
                leverage_impact += gamma[j] * z
            
            # Calculate total impact
            impact[i] = arch_impact + leverage_impact
        
        return impact

# Register the EGARCH model in the global registry
VOLATILITY_MODELS['EGARCH'] = EGARCH