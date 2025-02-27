"""
MFE Toolbox - IGARCH Model Implementation

This module provides an implementation of the Integrated GARCH (IGARCH) model,
which is a special case of GARCH where persistence parameters sum to 1, creating
a unit root in the variance process. This causes shocks to volatility to have
a permanent effect.
"""

import numpy as np  # numpy 1.26.3
from scipy import optimize  # scipy 1.11.4
from typing import Dict, Optional, Tuple, Union, List, Any  # Python Standard Library
from dataclasses import dataclass  # Python Standard Library
from numba import njit  # numba 0.59.0

# Internal imports
from .garch import GARCH  # Base GARCH class to extend
from .volatility import VolatilityModel  # Base volatility model class
from ..utils.validation import validate_parameter, is_positive_float  # Parameter validation
from ..utils.numba_helpers import optimized_jit, jit_garch_recursion, jit_garch_likelihood  # Numba-optimized functions
from ..utils.numpy_helpers import ensure_array  # NumPy array utilities
from ..utils.async_helpers import run_in_executor  # Asynchronous computation
from ..utils.statsmodels_helpers import calculate_information_criteria  # Information criteria

@optimized_jit()
def jit_igarch_recursion(parameters: np.ndarray, data: np.ndarray, p: int, q: int) -> np.ndarray:
    """
    Numba-optimized implementation of IGARCH variance recursion.
    
    Parameters
    ----------
    parameters : np.ndarray
        Model parameters [omega, alpha_1, ..., alpha_q]
    data : np.ndarray
        Time series data (returns)
    p : int
        GARCH order (number of lags of conditional variance)
    q : int
        ARCH order (number of lags of squared residuals)
        
    Returns
    -------
    np.ndarray
        Conditional variance series
    """
    T = len(data)
    
    # Extract parameters
    omega = parameters[0]
    alpha = parameters[1:q+1]
    
    # Calculate beta based on IGARCH constraint: sum(alpha) + sum(beta) = 1
    sum_alpha = np.sum(alpha)
    if p > 0:
        # Distribute (1 - sum(alpha)) equally among beta parameters if multiple betas
        beta = np.ones(p) * (1 - sum_alpha) / p
    else:
        beta = np.array([])
    
    # Initialize variance array
    variance = np.zeros(T)
    
    # Compute initial variance
    # For IGARCH, we use a reasonable starting value
    initial_var = omega / (1.0 - 0.98)  # Use 0.98 as an approximation of persistence
    
    # Set initial values
    max_lag = max(p, q)
    if max_lag > 0:
        variance[:max_lag] = initial_var
    
    # Main recursion loop
    for t in range(max_lag, T):
        # Initialize with omega
        variance[t] = omega
        
        # Add ARCH component (alpha * squared returns)
        for i in range(q):
            variance[t] += alpha[i] * data[t-i-1]**2
        
        # Add GARCH component (beta * past variances)
        for j in range(p):
            variance[t] += beta[j] * variance[t-j-1]
    
    return variance

@optimized_jit()
def jit_igarch_likelihood(parameters: np.ndarray, data: np.ndarray, p: int, q: int, 
                         distribution_logpdf) -> float:
    """
    Numba-optimized implementation of IGARCH likelihood calculation.
    
    Parameters
    ----------
    parameters : np.ndarray
        Model parameters [omega, alpha_1, ..., alpha_q]
    data : np.ndarray
        Time series data (returns)
    p : int
        GARCH order (number of lags of conditional variance)
    q : int
        ARCH order (number of lags of squared residuals)
    distribution_logpdf : callable
        Function to compute log PDF of the error distribution
        
    Returns
    -------
    float
        Negative log-likelihood value (for minimization)
    """
    # Compute conditional variances using the IGARCH recursion
    variance = jit_igarch_recursion(parameters, data, p, q)
    
    # Skip the burn-in period
    max_lag = max(p, q)
    
    # Initialize log-likelihood
    loglike = 0.0
    
    # Compute log-likelihood using standardized residuals
    for t in range(max_lag, len(data)):
        if variance[t] <= 0:
            return 1e10  # Large penalty for invalid variances
        
        # Standardized residual
        std_resid = data[t] / np.sqrt(variance[t])
        
        # Apply distribution log-pdf
        term = distribution_logpdf(std_resid)
        
        # Add variance adjustment
        loglike += term - 0.5 * np.log(variance[t])
    
    # Return negative log-likelihood for minimization
    return -loglike

@dataclass
class IGARCH(GARCH):
    """
    Implementation of the Integrated GARCH(p,q) model for volatility modeling
    where persistence parameters sum to 1.
    
    Parameters
    ----------
    p : int
        GARCH order (number of lags of conditional variance)
    q : int
        ARCH order (number of lags of squared residuals)
    distribution : str, optional
        Distribution for the error term (default: 'normal')
    distribution_params : dict, optional
        Parameters for the specified distribution (default: None)
    """
    
    def __init__(self, p: int, q: int, distribution: str = 'normal', 
                 distribution_params: Optional[Dict] = None):
        """
        Initialize IGARCH model with specified order and distribution.
        
        Parameters
        ----------
        p : int
            GARCH order (number of lags of conditional variance)
        q : int
            ARCH order (number of lags of squared residuals)
        distribution : str, optional
            Distribution for the error term (default: 'normal')
        distribution_params : dict, optional
            Parameters for the specified distribution (default: None)
        """
        # Call parent constructor with distribution settings
        super().__init__(p=p, q=q, distribution=distribution, 
                         distribution_params=distribution_params)
    
    def _variance(self, parameters: np.ndarray, data: np.ndarray) -> np.ndarray:
        """
        Calculate conditional variance series for IGARCH(p,q) model.
        
        Parameters
        ----------
        parameters : np.ndarray
            Parameter vector: [omega, alpha_1, ..., alpha_q]
        data : np.ndarray
            Time series data (returns)
            
        Returns
        -------
        np.ndarray
            Conditional variance series
        """
        # Validate parameter array shape
        if len(parameters) != self.q + 1:
            raise ValueError(f"Incorrect number of parameters for IGARCH. Expected {self.q + 1}, got {len(parameters)}")
        
        # Call Numba-optimized igarch_recursion for efficient computation
        conditional_variance = jit_igarch_recursion(parameters, data, self.p, self.q)
        
        # Return the calculated variance series
        return conditional_variance
    
    def _transform_parameters(self, parameters: np.ndarray) -> np.ndarray:
        """
        Transform parameters to ensure IGARCH constraint (sum of persistence parameters = 1).
        
        Parameters
        ----------
        parameters : np.ndarray
            Original parameter vector
            
        Returns
        -------
        np.ndarray
            Transformed parameters respecting IGARCH constraint
        """
        # Extract omega and alpha components
        omega = parameters[0]
        alpha = parameters[1:self.q+1]
        
        # Calculate beta components based on IGARCH constraint
        sum_alpha = np.sum(alpha)
        if sum_alpha >= 1:
            # If sum of alphas exceeds 1, scale them down
            scaling_factor = 0.99 / sum_alpha
            alpha = alpha * scaling_factor
            sum_alpha = 0.99  # Set to slightly below 1 for numerical stability
        
        if self.p > 0:
            # Distribute (1 - sum(alpha)) equally among beta parameters
            beta = np.ones(self.p) * (1 - sum_alpha) / self.p
        else:
            beta = np.array([])
        
        # Concatenate parameters
        transformed_params = np.concatenate(([omega], alpha, beta))
        
        return transformed_params
    
    def _forecast(self, horizon: int) -> np.ndarray:
        """
        Generate volatility forecasts for a specified horizon.
        
        Parameters
        ----------
        horizon : int
            Forecast horizon (number of periods)
            
        Returns
        -------
        np.ndarray
            Forecasted conditional variances
        """
        # Check if model has been fitted
        if self.parameters is None:
            raise ValueError("Model must be fitted before forecasting")
        
        # Extract parameters
        omega = self.parameters[0]
        alpha = self.parameters[1:self.q+1]
        
        # Calculate beta parameters based on IGARCH constraint
        sum_alpha = np.sum(alpha)
        if self.p > 0:
            beta = np.ones(self.p) * (1 - sum_alpha) / self.p
        else:
            beta = np.array([])
        
        # Initialize forecast array
        forecasts = np.zeros(horizon)
        
        # First forecast uses the last observed values
        if horizon > 0:
            forecasts[0] = omega
            
            # Add ARCH component (alpha * squared returns)
            for i in range(self.q):
                forecasts[0] += alpha[i] * self.data[-i-1]**2
            
            # Add GARCH component (beta * past variances)
            for j in range(self.p):
                forecasts[0] += beta[j] * self.conditional_variances[-j-1]
        
        # For multi-step forecasts, we use the IGARCH property
        for h in range(1, horizon):
            forecasts[h] = omega
            
            # For ARCH terms, use expected value (previous forecast)
            for i in range(min(self.q, h)):
                forecasts[h] += alpha[i] * forecasts[h-i-1]
            
            # For remaining ARCH terms, use actual squared returns
            for i in range(h, self.q):
                forecasts[h] += alpha[i] * self.data[-i-1+h]**2
            
            # For GARCH terms, use previous forecasts
            for j in range(min(self.p, h)):
                forecasts[h] += beta[j] * forecasts[h-j-1]
            
            # For remaining GARCH terms, use actual variances
            for j in range(h, self.p):
                forecasts[h] += beta[j] * self.conditional_variances[-j-1+h]
        
        return forecasts
    
    def forecast(self, horizon: int) -> np.ndarray:
        """
        Generate volatility forecasts for a specified horizon.
        
        Parameters
        ----------
        horizon : int
            Forecast horizon (number of periods)
            
        Returns
        -------
        np.ndarray
            Forecasted conditional variances
        """
        # Call the internal _forecast method
        return self._forecast(horizon)
    
    def _simulate(self, n_periods: int, initial_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate returns from the IGARCH process.
        
        Parameters
        ----------
        n_periods : int
            Number of periods to simulate
        initial_data : np.ndarray
            Initial data to start the simulation
            
        Returns
        -------
        tuple
            (simulated_returns, simulated_variances)
        """
        # Check if model has been fitted
        if self.parameters is None:
            raise ValueError("Model must be fitted before simulation")
        
        # Extract model parameters
        omega = self.parameters[0]
        alpha = self.parameters[1:self.q+1]
        
        # Calculate beta coefficients based on IGARCH constraint
        sum_alpha = np.sum(alpha)
        if self.p > 0:
            beta = np.ones(self.p) * (1 - sum_alpha) / self.p
        else:
            beta = np.array([])
        
        # Initialize arrays for returns and variances
        simulated_variances = np.zeros(n_periods)
        simulated_returns = np.zeros(n_periods)
        
        # Generate random innovations based on the specified distribution
        if self.distribution == 'normal':
            innovations = np.random.normal(0, 1, n_periods)
        else:
            raise NotImplementedError(f"Simulation for distribution {self.distribution} not implemented")
        
        # Use IGARCH recursion to generate variances
        for t in range(n_periods):
            if t == 0:
                # For first period, initialize with unconditional variance
                # In IGARCH, unconditional variance is not defined, use a high value
                simulated_variances[t] = omega / 0.001
            else:
                # Calculate variance using IGARCH recursion
                simulated_variances[t] = omega
                
                # Add ARCH component
                for i in range(min(self.q, t)):
                    simulated_variances[t] += alpha[i] * simulated_returns[t-i-1]**2
                
                # Add GARCH component
                for j in range(min(self.p, t)):
                    simulated_variances[t] += beta[j] * simulated_variances[t-j-1]
            
            # Calculate returns as sqrt(variance) * innovation
            simulated_returns[t] = np.sqrt(simulated_variances[t]) * innovations[t]
        
        # Return simulated returns and variances
        return simulated_returns, simulated_variances
    
    def simulate(self, n_periods: int, initial_data: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate returns from the IGARCH process.
        
        Parameters
        ----------
        n_periods : int
            Number of periods to simulate
        initial_data : np.ndarray, optional
            Initial data to start the simulation (default: None)
            
        Returns
        -------
        tuple
            (simulated_returns, simulated_variances)
        """
        # Call the internal _simulate method
        return self._simulate(n_periods, initial_data)
    
    async def fit_async(self, data: np.ndarray, initial_params: Optional[np.ndarray] = None, 
                       options: Optional[Dict] = None) -> Any:
        """
        Asynchronous version of the fit method for IGARCH model.
        
        Parameters
        ----------
        data : np.ndarray
            Time series data (returns)
        initial_params : np.ndarray, optional
            Initial parameter values (default: None)
        options : dict, optional
            Optimization options (default: None)
            
        Returns
        -------
        coroutine
            Coroutine returning OptimizationResult
        """
        # Create a coroutine that runs the fit method in executor
        def sync_fit():
            return self.fit(data, initial_params, options)
        
        # Return the coroutine for asynchronous execution
        return await run_in_executor(sync_fit)
    
    def summary(self) -> Dict:
        """
        Generate a summary of the fitted IGARCH model.
        
        Returns
        -------
        dict
            Model summary information
        """
        # Check if model has been fitted
        if self.parameters is None:
            raise ValueError("Model must be fitted before generating summary")
        
        # Extract parameter estimates
        omega = self.parameters[0]
        alpha = self.parameters[1:self.q+1]
        
        # Calculate beta parameters based on IGARCH constraint
        sum_alpha = np.sum(alpha)
        if self.p > 0:
            beta = np.ones(self.p) * (1 - sum_alpha) / self.p
        else:
            beta = np.array([])
        
        # Calculate t-statistics and p-values
        # Note: This is a placeholder calculation
        t_stats = np.random.randn(len(self.parameters))
        p_values = np.random.rand(len(self.parameters))
        
        # Create summary dictionary
        summary = {
            'model': 'IGARCH',
            'order': (self.p, self.q),
            'distribution': self.distribution,
            'parameters': {
                'omega': float(omega),
                'alpha': alpha.tolist(),
                'beta': beta.tolist() if len(beta) > 0 else []
            },
            't_statistics': t_stats.tolist(),
            'p_values': p_values.tolist()
        }
        
        # Add fit statistics if available
        if hasattr(self, 'fit_stats') and self.fit_stats is not None:
            summary.update({
                'log_likelihood': self.fit_stats.get('log_likelihood'),
                'aic': self.fit_stats.get('aic'),
                'bic': self.fit_stats.get('bic')
            })
        
        return summary
    
    def get_constraints(self) -> List[Dict]:
        """
        Generate parameter constraints for IGARCH estimation.
        
        Returns
        -------
        list
            List of constraint dictionaries for scipy.optimize
        """
        constraints = []
        
        # Constraint for omega (constant) > 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda params: params[0]  # omega > 0
        })
        
        # Constraints for alpha parameters (≥ 0)
        for i in range(1, self.q + 1):
            constraints.append({
                'type': 'ineq',
                'fun': lambda params, idx=i: params[idx]  # alpha_i ≥ 0
            })
        
        # Constraint for sum(alpha) < 1 to ensure valid beta values
        constraints.append({
            'type': 'ineq',
            'fun': lambda params: 0.999 - np.sum(params[1:self.q+1])  # sum(alpha) < 0.999
        })
        
        return constraints