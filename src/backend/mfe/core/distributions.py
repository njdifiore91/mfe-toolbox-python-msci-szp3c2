"""
MFE Toolbox - Core Distribution Functions

This module provides core statistical distribution functions and tests for 
financial time series analysis. It includes implementations of advanced distributions
commonly used in financial econometrics, optimized with Numba for performance.

Key components:
1. Generalized Error Distribution (GED) 
2. Hansen's skewed t-distribution
3. Statistical normality and distribution tests
4. Distribution parameter estimation
5. Log-likelihood calculations for distributions

The module leverages SciPy and Statsmodels for underlying statistical operations
and provides Numba-optimized versions of performance-critical functions.
"""

import numpy as np  # numpy 1.26.3
import scipy.stats as stats  # scipy 1.11.4
import scipy.special as special  # scipy 1.11.4
import scipy.optimize as optimize  # scipy 1.11.4
from statsmodels.distributions import ECDF  # statsmodels 0.14.1
import statsmodels.api as sm  # statsmodels 0.14.1
import numba  # numba 0.59.0
from typing import Any, Dict, List, Optional, Tuple, Union, Callable  # Python 3.12

from ..utils.numba_helpers import optimized_jit, fallback_to_python
from ..utils.validation import validate_data, is_probability
from ..utils.numpy_helpers import ensure_array, ensure_finite

# Set up module logger
import logging
logger = logging.getLogger(__name__)

# ===========================
# Generalized Error Distribution (GED) Functions
# ===========================

@numba.jit(nopython=True)
def ged_pdf(x: np.ndarray, mu: float, sigma: float, nu: float) -> np.ndarray:
    """
    Probability density function for the Generalized Error Distribution (GED), optimized with Numba.
    
    Parameters
    ----------
    x : ndarray
        Points at which to evaluate the PDF
    mu : float
        Location parameter
    sigma : float
        Scale parameter (must be positive)
    nu : float
        Shape parameter (must be positive)
        
    Returns
    -------
    ndarray
        GED probability density values for given inputs
    """
    # Calculate standardized values
    z = (x - mu) / sigma
    
    # Special cases for efficiency and accuracy
    if nu == 2.0:  # Normal distribution
        pdf = np.exp(-0.5 * z**2) / (sigma * np.sqrt(2.0 * np.pi))
    elif nu == 1.0:  # Laplace distribution
        pdf = np.exp(-np.abs(z)) / (2.0 * sigma)
    else:
        # For other nu values, use a simplified direct calculation
        # This is an approximation for Numba compatibility
        A = 0.5 * nu / (sigma * np.exp(np.log(2.0) * (1.0/nu)) * np.exp(np.log(special.gamma(1.0/nu))))
        pdf = A * np.exp(-(np.abs(z) ** nu) / 2.0)
    
    return pdf

@numba.jit(nopython=True)
def ged_cdf(x: np.ndarray, mu: float, sigma: float, nu: float) -> np.ndarray:
    """
    Cumulative distribution function for the Generalized Error Distribution (GED), optimized with Numba.
    
    Parameters
    ----------
    x : ndarray
        Points at which to evaluate the CDF
    mu : float
        Location parameter
    sigma : float
        Scale parameter (must be positive)
    nu : float
        Shape parameter (must be positive)
        
    Returns
    -------
    ndarray
        GED cumulative probability values for given inputs
    """
    # Calculate standardized values
    z = (x - mu) / sigma
    
    # Special case for normal distribution (nu=2)
    if nu == 2.0:
        # Approximate normal CDF for Numba compatibility
        cdf = 0.5 * (1.0 + np.sign(z) * (1.0 - np.exp(-0.5 * z * z * (4.0/np.pi))))
        return cdf
    
    # For other nu values, we approximate using a numerical approach
    # This is a simplified version for Numba compatibility
    
    # Determine sign for later use
    sign_z = np.sign(z)
    abs_z = np.abs(z)
    
    # For Laplace distribution (nu=1)
    if nu == 1.0:
        cdf = 0.5 + 0.5 * sign_z * (1.0 - np.exp(-abs_z))
    else:
        # Approximate for other nu values
        k = abs_z ** nu / 2.0
        cdf = 0.5 + 0.5 * sign_z * (1.0 - 1.0 / (1.0 + k))
    
    return cdf

def ged_ppf(q: np.ndarray, mu: float, sigma: float, nu: float) -> np.ndarray:
    """
    Percent point function (inverse CDF) for the Generalized Error Distribution (GED).
    
    Parameters
    ----------
    q : ndarray
        Probability points at which to evaluate the PPF
    mu : float
        Location parameter
    sigma : float
        Scale parameter (must be positive)
    nu : float
        Shape parameter (must be positive)
        
    Returns
    -------
    ndarray
        GED quantile values for given probability inputs
    """
    # Validate inputs
    q = ensure_array(q)
    if not np.all((q >= 0) & (q <= 1)):
        raise ValueError("All probabilities must be between 0 and 1")
    
    # Special cases
    if nu == 2.0:  # Normal distribution
        return mu + sigma * stats.norm.ppf(q)
    if nu == 1.0:  # Laplace distribution
        return mu + sigma * stats.laplace.ppf(q)
    
    # For other nu values, use numerical inversion
    result = np.empty_like(q, dtype=np.float64)
    
    for i in range(len(q.flat)):
        qi = q.flat[i]
        
        # Handle edge cases
        if qi == 0:
            result.flat[i] = -np.inf
        elif qi == 1:
            result.flat[i] = np.inf
        else:
            # Use scipy.optimize for numerical inversion
            def objective(x):
                x_arr = np.array([x]) if np.isscalar(x) else np.asarray(x)
                return float(ged_cdf(x_arr, mu, sigma, nu)[0] - qi)
            
            # Initial guess based on normal distribution
            x0 = mu + sigma * stats.norm.ppf(qi)
            
            # Find root of objective function
            try:
                result.flat[i] = optimize.brentq(objective, mu - 10*sigma, mu + 10*sigma)
            except:
                # Fallback to normal approximation if optimization fails
                result.flat[i] = x0
    
    return result

def ged_random(size: int, mu: float, sigma: float, nu: float, seed: Optional[int] = None) -> np.ndarray:
    """
    Random number generator for the Generalized Error Distribution (GED).
    
    Parameters
    ----------
    size : int
        Number of random samples to generate
    mu : float
        Location parameter
    sigma : float
        Scale parameter (must be positive)
    nu : float
        Shape parameter (must be positive)
    seed : Optional[int], default=None
        Random seed for reproducibility
        
    Returns
    -------
    ndarray
        Array of random numbers from the GED distribution
    """
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Generate uniform random numbers
    u = np.random.random(size)
    
    # Transform to GED using inverse CDF
    return ged_ppf(u, mu, sigma, nu)

# ===========================
# Hansen's Skewed t-distribution Functions
# ===========================

@numba.jit(nopython=True)
def skewt_pdf(x: np.ndarray, mu: float, sigma: float, nu: float, lambda_: float) -> np.ndarray:
    """
    Probability density function for Hansen's skewed t-distribution, optimized with Numba.
    
    Parameters
    ----------
    x : ndarray
        Points at which to evaluate the PDF
    mu : float
        Location parameter
    sigma : float
        Scale parameter (must be positive)
    nu : float
        Degrees of freedom (must be > 2)
    lambda_ : float
        Skewness parameter (must be between -1 and 1)
        
    Returns
    -------
    ndarray
        Skewed t probability density values for given inputs
    """
    # Constants for the distribution (pre-calculated for Numba compatibility)
    const1 = special.gamma((nu + 1) / 2) / (special.gamma(nu / 2) * np.sqrt(np.pi * (nu - 2)))
    a = 4 * lambda_ * const1 * ((nu - 2) / (nu - 1))
    b = np.sqrt(1 + 3 * lambda_**2 - a**2)
    
    # Standardize the input
    z = (x - mu) / sigma
    
    # Initialize output array
    pdf = np.zeros_like(z)
    
    # Calculate PDF for each value
    for i in range(len(z)):
        zi = z[i]
        
        # Determine which region we're in
        if lambda_ * zi < -a/b:
            # Left tail
            s = -1
        else:
            # Right tail
            s = 1
        
        # Calculate components of the PDF
        d = (b * zi + a) / (1 + s * lambda_)
        t_density = const1 * (1 + d**2 / (nu - 2))**(-0.5 * (nu + 1)) / np.sqrt(nu - 2)
        
        # Combine for final PDF value
        pdf[i] = b * t_density / sigma
    
    return pdf

def skewt_cdf(x: np.ndarray, mu: float, sigma: float, nu: float, lambda_: float) -> np.ndarray:
    """
    Cumulative distribution function for Hansen's skewed t-distribution.
    
    Parameters
    ----------
    x : ndarray
        Points at which to evaluate the CDF
    mu : float
        Location parameter
    sigma : float
        Scale parameter (must be positive)
    nu : float
        Degrees of freedom (must be > 2)
    lambda_ : float
        Skewness parameter (must be between -1 and 1)
        
    Returns
    -------
    ndarray
        Skewed t cumulative probability values for given inputs
    """
    # Ensure x is an array
    x = ensure_array(x)
    
    # Constants for the distribution
    const1 = special.gamma((nu + 1) / 2) / (special.gamma(nu / 2) * np.sqrt(np.pi * (nu - 2)))
    a = 4 * lambda_ * const1 * ((nu - 2) / (nu - 1))
    b = np.sqrt(1 + 3 * lambda_**2 - a**2)
    
    # Standardize the input
    z = (x - mu) / sigma
    
    # Initialize output array
    cdf = np.zeros_like(z)
    
    # Calculate CDF for each value
    for i in range(len(z)):
        zi = z[i]
        
        if lambda_ * zi < -a/b:
            # Left tail
            d = (b * zi + a) / (1 - lambda_)
            cdf[i] = 0.5 * stats.t.cdf(d, nu)
        else:
            # Right tail
            d = (b * zi + a) / (1 + lambda_)
            cdf[i] = 0.5 + 0.5 * stats.t.cdf(d, nu)
    
    return cdf

def skewt_ppf(q: np.ndarray, mu: float, sigma: float, nu: float, lambda_: float) -> np.ndarray:
    """
    Percent point function (inverse CDF) for Hansen's skewed t-distribution.
    
    Parameters
    ----------
    q : ndarray
        Probability points at which to evaluate the PPF
    mu : float
        Location parameter
    sigma : float
        Scale parameter (must be positive)
    nu : float
        Degrees of freedom (must be > 2)
    lambda_ : float
        Skewness parameter (must be between -1 and 1)
        
    Returns
    -------
    ndarray
        Skewed t quantile values for given probability inputs
    """
    # Validate inputs
    q = ensure_array(q)
    if not np.all((q >= 0) & (q <= 1)):
        raise ValueError("All probabilities must be between 0 and 1")
    
    # Initialize output array
    result = np.empty_like(q, dtype=np.float64)
    
    # Process each probability value
    for i in range(len(q.flat)):
        qi = q.flat[i]
        
        # Handle edge cases
        if qi == 0:
            result.flat[i] = -np.inf
        elif qi == 1:
            result.flat[i] = np.inf
        else:
            # Use scipy.optimize for numerical inversion
            def objective(x):
                x_arr = np.array([x]) if np.isscalar(x) else np.asarray(x)
                return float(skewt_cdf(x_arr, mu, sigma, nu, lambda_)[0] - qi)
            
            # Initial guess based on normal distribution
            x0 = mu + sigma * stats.norm.ppf(qi)
            
            # Find root of objective function
            try:
                result.flat[i] = optimize.brentq(objective, mu - 10*sigma, mu + 10*sigma)
            except:
                # Fallback to normal approximation if optimization fails
                result.flat[i] = x0
    
    return result

def skewt_random(size: int, mu: float, sigma: float, nu: float, lambda_: float, seed: Optional[int] = None) -> np.ndarray:
    """
    Random number generator for Hansen's skewed t-distribution.
    
    Parameters
    ----------
    size : int
        Number of random samples to generate
    mu : float
        Location parameter
    sigma : float
        Scale parameter (must be positive)
    nu : float
        Degrees of freedom (must be > 2)
    lambda_ : float
        Skewness parameter (must be between -1 and 1)
    seed : Optional[int], default=None
        Random seed for reproducibility
        
    Returns
    -------
    ndarray
        Array of random numbers from the skewed t-distribution
    """
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Generate uniform random numbers
    u = np.random.random(size)
    
    # Transform to skewed t using inverse CDF
    return skewt_ppf(u, mu, sigma, nu, lambda_)

# ===========================
# Statistical Tests
# ===========================

def jarque_bera(data: np.ndarray) -> Tuple[float, float]:
    """
    Jarque-Bera test for normality based on skewness and kurtosis.
    
    Parameters
    ----------
    data : ndarray
        Input data array
        
    Returns
    -------
    tuple
        Returns the test statistic and p-value
    """
    # Validate and prepare data
    data = validate_data(data)
    n = len(data)
    
    # Calculate sample skewness and kurtosis
    s = stats.skew(data)
    k = stats.kurtosis(data)
    
    # Calculate test statistic
    jb = n/6 * (s**2 + k**2/4)
    
    # Compute p-value
    p_value = 1 - stats.chi2.cdf(jb, df=2)
    
    return jb, p_value

def lilliefors(data: np.ndarray) -> Tuple[float, float]:
    """
    Lilliefors test for normality, an adaptation of the Kolmogorov-Smirnov test.
    
    Parameters
    ----------
    data : ndarray
        Input data array
        
    Returns
    -------
    tuple
        Returns the test statistic and p-value
    """
    # Validate and prepare data
    data = validate_data(data)
    
    # Use statsmodels implementation
    from statsmodels.stats.diagnostic import lilliefors as sm_lilliefors
    D, p_value = sm_lilliefors(data)
    
    return D, p_value

def shapiro_wilk(data: np.ndarray) -> Tuple[float, float]:
    """
    Wrapper for the Shapiro-Wilk test for normality using SciPy's implementation.
    
    Parameters
    ----------
    data : ndarray
        Input data array
        
    Returns
    -------
    tuple
        Returns the test statistic and p-value
    """
    # Validate and prepare data
    data = validate_data(data)
    
    # Check sample size limitations
    if len(data) < 3 or len(data) > 5000:
        raise ValueError("Shapiro-Wilk test requires sample size between 3 and 5000")
    
    # Perform test
    W, p_value = stats.shapiro(data)
    
    return W, p_value

def ks_test(data: np.ndarray, distribution: str, params: Optional[Dict[str, float]] = None) -> Tuple[float, float]:
    """
    Kolmogorov-Smirnov test for evaluating if a sample comes from a specific distribution.
    
    Parameters
    ----------
    data : ndarray
        Input data array
    distribution : str
        The distribution to test against ('norm', 'ged', 'skewt', 't', etc.)
    params : Optional[dict], default=None
        Dictionary of parameters for the specified distribution. If None,
        parameters will be estimated from the data.
        
    Returns
    -------
    tuple
        Returns the test statistic and p-value
    """
    # Validate and prepare data
    data = validate_data(data)
    
    # Select the appropriate distribution CDF function
    if distribution.lower() == 'norm':
        if params is None:
            mu, sigma = np.mean(data), np.std(data, ddof=1)
        else:
            mu = params.get('mu', params.get('loc', 0))
            sigma = params.get('sigma', params.get('scale', 1))
        cdf = lambda x: stats.norm.cdf(x, loc=mu, scale=sigma)
    
    elif distribution.lower() == 'ged':
        if params is None:
            # Estimate parameters using function defined later
            params = estimate_ged_params(data)
        mu = params.get('mu', 0)
        sigma = params.get('sigma', 1)
        nu = params.get('nu', 2)
        cdf = lambda x: ged_cdf(np.asarray(x), mu, sigma, nu)
    
    elif distribution.lower() == 'skewt':
        if params is None:
            # Estimate parameters using function defined later
            params = estimate_skewt_params(data)
        mu = params.get('mu', 0)
        sigma = params.get('sigma', 1)
        nu = params.get('nu', 5)
        lambda_ = params.get('lambda', 0)
        cdf = lambda x: skewt_cdf(np.asarray(x), mu, sigma, nu, lambda_)
    
    elif distribution.lower() == 't':
        if params is None:
            # Estimate t distribution parameters
            df, loc, scale = stats.t.fit(data)
        else:
            df = params.get('df', 5)
            loc = params.get('mu', params.get('loc', 0))
            scale = params.get('sigma', params.get('scale', 1))
        cdf = lambda x: stats.t.cdf(x, df=df, loc=loc, scale=scale)
    
    else:
        raise ValueError(f"Unsupported distribution: {distribution}")
    
    # Calculate empirical CDF
    ecdf = ECDF(data)
    
    # Get sorted data points
    x = np.sort(data)
    
    # Calculate theoretical CDF values
    y_theo = np.array([cdf(xi) for xi in x])
    
    # Calculate empirical CDF values
    y_emp = ecdf(x)
    
    # Calculate the test statistic: maximum absolute difference
    D = np.max(np.abs(y_emp - y_theo))
    
    # Calculate p-value using asymptotic distribution
    n = len(data)
    p_value = np.exp(-2 * n * D**2)
    
    return D, p_value

# ===========================
# Likelihood Functions
# ===========================

def normal_loglikelihood(data: np.ndarray, mu: Optional[float] = None, sigma: Optional[float] = None) -> float:
    """
    Calculates the log-likelihood of data under a normal distribution.
    
    Parameters
    ----------
    data : ndarray
        Input data array
    mu : Optional[float], default=None
        Mean parameter. If None, estimated from the data.
    sigma : Optional[float], default=None
        Standard deviation parameter. If None, estimated from the data.
        
    Returns
    -------
    float
        Log-likelihood value
    """
    # Validate and prepare data
    data = validate_data(data)
    
    # Estimate parameters if not provided
    if mu is None:
        mu = np.mean(data)
    if sigma is None:
        sigma = np.std(data, ddof=1)
    
    # Calculate log-likelihood
    n = len(data)
    ll = -0.5 * n * np.log(2 * np.pi) - n * np.log(sigma) - np.sum((data - mu)**2) / (2 * sigma**2)
    
    return ll

def t_loglikelihood(data: np.ndarray, mu: Optional[float] = None, 
                    sigma: Optional[float] = None, nu: Optional[float] = None) -> float:
    """
    Calculates the log-likelihood of data under a t-distribution.
    
    Parameters
    ----------
    data : ndarray
        Input data array
    mu : Optional[float], default=None
        Location parameter. If None, estimated from the data.
    sigma : Optional[float], default=None
        Scale parameter. If None, estimated from the data.
    nu : Optional[float], default=None
        Degrees of freedom parameter. If None, estimated from the data.
        
    Returns
    -------
    float
        Log-likelihood value
    """
    # Validate and prepare data
    data = validate_data(data)
    
    # Estimate parameters if not provided
    if mu is None or sigma is None or nu is None:
        params = stats.t.fit(data)
        if nu is None:
            nu = params[0]  # df
        if mu is None:
            mu = params[1]  # loc
        if sigma is None:
            sigma = params[2]  # scale
    
    # Calculate log-likelihood
    z = (data - mu) / sigma
    log_const = special.gammaln((nu + 1) / 2) - special.gammaln(nu / 2) - 0.5 * np.log(nu * np.pi) - np.log(sigma)
    log_kernel = -((nu + 1) / 2) * np.sum(np.log(1 + z**2 / nu))
    ll = len(data) * log_const + log_kernel
    
    return ll

def ged_loglikelihood(data: np.ndarray, mu: Optional[float] = None,
                     sigma: Optional[float] = None, nu: Optional[float] = None) -> float:
    """
    Calculates the log-likelihood of data under a Generalized Error Distribution.
    
    Parameters
    ----------
    data : ndarray
        Input data array
    mu : Optional[float], default=None
        Location parameter. If None, estimated from the data.
    sigma : Optional[float], default=None
        Scale parameter. If None, estimated from the data.
    nu : Optional[float], default=None
        Shape parameter. If None, estimated from the data.
        
    Returns
    -------
    float
        Log-likelihood value
    """
    # Validate and prepare data
    data = validate_data(data)
    
    # Estimate parameters if not provided
    if mu is None or sigma is None or nu is None:
        params = estimate_ged_params(data)
        if mu is None:
            mu = params['mu']
        if sigma is None:
            sigma = params['sigma']
        if nu is None:
            nu = params['nu']
    
    # Calculate log-likelihood
    z = (data - mu) / sigma
    log_A = np.log(nu) - np.log(2) - np.log(special.gamma(1/nu))
    log_term1 = len(data) * log_A - len(data) * np.log(sigma)
    log_term2 = -0.5 * np.sum(np.abs(z)**nu)
    ll = log_term1 + log_term2
    
    return ll

def skewt_loglikelihood(data: np.ndarray, mu: Optional[float] = None, sigma: Optional[float] = None,
                       nu: Optional[float] = None, lambda_: Optional[float] = None) -> float:
    """
    Calculates the log-likelihood of data under Hansen's skewed t-distribution.
    
    Parameters
    ----------
    data : ndarray
        Input data array
    mu : Optional[float], default=None
        Location parameter. If None, estimated from the data.
    sigma : Optional[float], default=None
        Scale parameter. If None, estimated from the data.
    nu : Optional[float], default=None
        Degrees of freedom parameter. If None, estimated from the data.
    lambda_ : Optional[float], default=None
        Skewness parameter. If None, estimated from the data.
        
    Returns
    -------
    float
        Log-likelihood value
    """
    # Validate and prepare data
    data = validate_data(data)
    
    # Estimate parameters if not provided
    if mu is None or sigma is None or nu is None or lambda_ is None:
        params = estimate_skewt_params(data)
        if mu is None:
            mu = params['mu']
        if sigma is None:
            sigma = params['sigma']
        if nu is None:
            nu = params['nu']
        if lambda_ is None:
            lambda_ = params['lambda']
    
    # Calculate PDF values
    pdf_values = skewt_pdf(data, mu, sigma, nu, lambda_)
    
    # Calculate log-likelihood with protection against log(0)
    eps = np.finfo(float).eps  # Small constant
    ll = np.sum(np.log(pdf_values + eps))
    
    return ll

# ===========================
# Parameter Estimation Functions
# ===========================

def estimate_ged_params(data: np.ndarray, starting_values: Optional[Dict[str, float]] = None,
                      bounds: Optional[Dict[str, Tuple[float, float]]] = None) -> Dict[str, Any]:
    """
    Estimates parameters of the Generalized Error Distribution from data using maximum likelihood.
    
    Parameters
    ----------
    data : ndarray
        Input data array
    starting_values : Optional[dict], default=None
        Initial parameter values for optimization. If None, defaults are used.
    bounds : Optional[dict], default=None
        Parameter bounds for optimization. If None, defaults are used.
        
    Returns
    -------
    dict
        Estimated parameters (mu, sigma, nu) and optimization results
    """
    # Validate and prepare data
    data = validate_data(data)
    
    # Set default starting values if not provided
    if starting_values is None:
        starting_values = {
            'mu': np.mean(data),
            'sigma': np.std(data, ddof=1),
            'nu': 2.0  # Start with normal distribution
        }
    
    # Set default bounds if not provided
    if bounds is None:
        bounds = {
            'mu': (np.min(data) - 5 * np.std(data), np.max(data) + 5 * np.std(data)),
            'sigma': (1e-6, 10 * np.std(data)),
            'nu': (0.5, 10.0)  # Allow for very peaked (nu < 1) distributions
        }
    
    # Define negative log-likelihood function
    def neg_loglikelihood(params):
        mu, sigma, nu = params
        if sigma <= 0 or nu <= 0:
            return 1e10  # Large penalty for invalid parameters
        return -ged_loglikelihood(data, mu, sigma, nu)
    
    # Initial parameter vector
    x0 = [starting_values['mu'], starting_values['sigma'], starting_values['nu']]
    
    # Parameter bounds for optimization
    param_bounds = [
        bounds['mu'],
        bounds['sigma'],
        bounds['nu']
    ]
    
    # Perform optimization
    result = optimize.minimize(
        neg_loglikelihood,
        x0,
        bounds=param_bounds,
        method='L-BFGS-B'
    )
    
    # Extract estimated parameters
    mu_hat, sigma_hat, nu_hat = result.x
    
    # Calculate log-likelihood
    ll = -result.fun
    
    # Calculate information criteria
    n = len(data)
    k = 3  # Number of parameters
    aic = -2 * ll + 2 * k
    bic = -2 * ll + k * np.log(n)
    
    # Return results
    return {
        'mu': mu_hat,
        'sigma': sigma_hat,
        'nu': nu_hat,
        'loglikelihood': ll,
        'aic': aic,
        'bic': bic,
        'converged': result.success,
        'message': result.message
    }

def estimate_skewt_params(data: np.ndarray, starting_values: Optional[Dict[str, float]] = None,
                         bounds: Optional[Dict[str, Tuple[float, float]]] = None) -> Dict[str, Any]:
    """
    Estimates parameters of Hansen's skewed t-distribution from data using maximum likelihood.
    
    Parameters
    ----------
    data : ndarray
        Input data array
    starting_values : Optional[dict], default=None
        Initial parameter values for optimization. If None, defaults are used.
    bounds : Optional[dict], default=None
        Parameter bounds for optimization. If None, defaults are used.
        
    Returns
    -------
    dict
        Estimated parameters (mu, sigma, nu, lambda) and optimization results
    """
    # Validate and prepare data
    data = validate_data(data)
    
    # Set default starting values if not provided
    if starting_values is None:
        starting_values = {
            'mu': np.mean(data),
            'sigma': np.std(data, ddof=1),
            'nu': 5.0,  # Default degrees of freedom
            'lambda': 0.0  # Start with symmetric t
        }
    
    # Set default bounds if not provided
    if bounds is None:
        bounds = {
            'mu': (np.min(data) - 5 * np.std(data), np.max(data) + 5 * np.std(data)),
            'sigma': (1e-6, 10 * np.std(data)),
            'nu': (2.1, 30.0),  # nu > 2 for finite variance
            'lambda': (-0.99, 0.99)  # lambda must be in (-1, 1)
        }
    
    # Define negative log-likelihood function
    def neg_loglikelihood(params):
        mu, sigma, nu, lambda_ = params
        if sigma <= 0 or nu <= 2 or abs(lambda_) >= 1:
            return 1e10  # Large penalty for invalid parameters
        return -skewt_loglikelihood(data, mu, sigma, nu, lambda_)
    
    # Initial parameter vector
    x0 = [
        starting_values['mu'],
        starting_values['sigma'],
        starting_values['nu'],
        starting_values['lambda']
    ]
    
    # Parameter bounds for optimization
    param_bounds = [
        bounds['mu'],
        bounds['sigma'],
        bounds['nu'],
        bounds['lambda']
    ]
    
    # Perform optimization
    result = optimize.minimize(
        neg_loglikelihood,
        x0,
        bounds=param_bounds,
        method='L-BFGS-B'
    )
    
    # Extract estimated parameters
    mu_hat, sigma_hat, nu_hat, lambda_hat = result.x
    
    # Calculate log-likelihood
    ll = -result.fun
    
    # Calculate information criteria
    n = len(data)
    k = 4  # Number of parameters
    aic = -2 * ll + 2 * k
    bic = -2 * ll + k * np.log(n)
    
    # Return results
    return {
        'mu': mu_hat,
        'sigma': sigma_hat,
        'nu': nu_hat,
        'lambda': lambda_hat,
        'loglikelihood': ll,
        'aic': aic,
        'bic': bic,
        'converged': result.success,
        'message': result.message
    }

def distribution_fit(data: np.ndarray, dist_type: str, starting_values: Optional[Dict[str, float]] = None,
                    bounds: Optional[Dict[str, Tuple[float, float]]] = None) -> Dict[str, Any]:
    """
    Generic distribution fitting function that supports multiple distribution types.
    
    Parameters
    ----------
    data : ndarray
        Input data array
    dist_type : str
        Distribution type ('norm', 'ged', 'skewt', 't')
    starting_values : Optional[dict], default=None
        Initial parameter values for optimization. If None, defaults are used.
    bounds : Optional[dict], default=None
        Parameter bounds for optimization. If None, defaults are used.
        
    Returns
    -------
    dict
        Estimated parameters and goodness-of-fit measures
    """
    # Validate and prepare data
    data = validate_data(data)
    
    # Fit the specified distribution
    if dist_type.lower() == 'norm':
        # Estimate normal distribution parameters
        mu = np.mean(data)
        sigma = np.std(data, ddof=1)
        
        # Calculate log-likelihood and other metrics
        ll = normal_loglikelihood(data, mu, sigma)
        n = len(data)
        k = 2  # Number of parameters
        aic = -2 * ll + 2 * k
        bic = -2 * ll + k * np.log(n)
        
        # Run diagnostic tests
        jb_stat, jb_pval = jarque_bera(data)
        
        try:
            sw_stat, sw_pval = shapiro_wilk(data)
        except:
            sw_stat, sw_pval = np.nan, np.nan
            
        ks_stat, ks_pval = ks_test(data, 'norm', {'mu': mu, 'sigma': sigma})
        
        return {
            'distribution': 'normal',
            'mu': mu,
            'sigma': sigma,
            'loglikelihood': ll,
            'aic': aic,
            'bic': bic,
            'tests': {
                'jarque_bera': {'statistic': jb_stat, 'p_value': jb_pval},
                'shapiro_wilk': {'statistic': sw_stat, 'p_value': sw_pval},
                'ks_test': {'statistic': ks_stat, 'p_value': ks_pval}
            }
        }
    
    elif dist_type.lower() == 'ged':
        # Estimate GED parameters
        result = estimate_ged_params(data, starting_values, bounds)
        
        # Run KS test
        ks_stat, ks_pval = ks_test(data, 'ged', {
            'mu': result['mu'],
            'sigma': result['sigma'],
            'nu': result['nu']
        })
        
        result['tests'] = {
            'ks_test': {'statistic': ks_stat, 'p_value': ks_pval}
        }
        
        result['distribution'] = 'ged'
        return result
    
    elif dist_type.lower() == 'skewt':
        # Estimate skewed t parameters
        result = estimate_skewt_params(data, starting_values, bounds)
        
        # Run KS test
        ks_stat, ks_pval = ks_test(data, 'skewt', {
            'mu': result['mu'],
            'sigma': result['sigma'],
            'nu': result['nu'],
            'lambda': result['lambda']
        })
        
        result['tests'] = {
            'ks_test': {'statistic': ks_stat, 'p_value': ks_pval}
        }
        
        result['distribution'] = 'skewt'
        return result
    
    elif dist_type.lower() == 't':
        # Estimate Student's t parameters
        df, loc, scale = stats.t.fit(data)
        
        # Calculate log-likelihood and other metrics
        ll = t_loglikelihood(data, loc, scale, df)
        n = len(data)
        k = 3  # Number of parameters
        aic = -2 * ll + 2 * k
        bic = -2 * ll + k * np.log(n)
        
        # Run KS test
        ks_stat, ks_pval = ks_test(data, 't', {'df': df, 'loc': loc, 'scale': scale})
        
        return {
            'distribution': 't',
            'df': df,
            'mu': loc,
            'sigma': scale,
            'loglikelihood': ll,
            'aic': aic,
            'bic': bic,
            'tests': {
                'ks_test': {'statistic': ks_stat, 'p_value': ks_pval}
            }
        }
    
    else:
        raise ValueError(f"Unsupported distribution type: {dist_type}")

def distribution_forecast(dist_type: str, params: Dict[str, float], quantiles: List[float]) -> Dict[str, np.ndarray]:
    """
    Generates forecasted quantiles from a fitted distribution.
    
    Parameters
    ----------
    dist_type : str
        Distribution type ('norm', 'ged', 'skewt', 't')
    params : dict
        Parameters of the fitted distribution
    quantiles : list
        List of probability levels for which to compute quantiles
        
    Returns
    -------
    dict
        Forecasted quantiles for the specified distribution
    """
    # Validate quantiles
    quantiles = np.asarray(quantiles)
    if np.any((quantiles < 0) | (quantiles > 1)):
        raise ValueError("Quantiles must be between 0 and 1")
    
    # Generate quantile forecasts based on distribution type
    if dist_type.lower() == 'norm':
        mu = params.get('mu', 0)
        sigma = params.get('sigma', 1)
        quantile_values = stats.norm.ppf(quantiles, loc=mu, scale=sigma)
    
    elif dist_type.lower() == 'ged':
        mu = params.get('mu', 0)
        sigma = params.get('sigma', 1)
        nu = params.get('nu', 2)
        quantile_values = ged_ppf(quantiles, mu, sigma, nu)
    
    elif dist_type.lower() == 'skewt':
        mu = params.get('mu', 0)
        sigma = params.get('sigma', 1)
        nu = params.get('nu', 5)
        lambda_ = params.get('lambda', 0)
        quantile_values = skewt_ppf(quantiles, mu, sigma, nu, lambda_)
    
    elif dist_type.lower() == 't':
        mu = params.get('mu', 0)
        sigma = params.get('sigma', 1)
        df = params.get('df', 5)
        quantile_values = stats.t.ppf(quantiles, df=df, loc=mu, scale=sigma)
    
    else:
        raise ValueError(f"Unsupported distribution type: {dist_type}")
    
    # Return results
    return {
        'distribution': dist_type,
        'quantiles': quantiles,
        'values': quantile_values,
        'parameters': params
    }

# ===========================
# Information Criteria Functions
# ===========================

def akaike_information_criterion(log_likelihood: float, num_params: int) -> float:
    """
    Calculates the Akaike Information Criterion (AIC) for model selection.
    
    Parameters
    ----------
    log_likelihood : float
        Log-likelihood value of the model
    num_params : int
        Number of parameters in the model
        
    Returns
    -------
    float
        AIC value
    """
    return -2 * log_likelihood + 2 * num_params

def bayesian_information_criterion(log_likelihood: float, num_params: int, num_observations: int) -> float:
    """
    Calculates the Bayesian Information Criterion (BIC) for model selection.
    
    Parameters
    ----------
    log_likelihood : float
        Log-likelihood value of the model
    num_params : int
        Number of parameters in the model
    num_observations : int
        Number of observations used for estimation
        
    Returns
    -------
    float
        BIC value
    """
    return -2 * log_likelihood + num_params * np.log(num_observations)

# ===========================
# Distribution Classes
# ===========================

class GeneralizedErrorDistribution:
    """
    Class implementation of the Generalized Error Distribution (GED) with methods for 
    PDF, CDF, PPF, and random sampling.
    """
    
    def __init__(self, mu: float = 0.0, sigma: float = 1.0, nu: float = 2.0):
        """
        Initialize a GED distribution with location, scale, and shape parameters.
        
        Parameters
        ----------
        mu : float, default=0.0
            Location parameter
        sigma : float, default=1.0
            Scale parameter (must be positive)
        nu : float, default=2.0
            Shape parameter (must be positive)
        """
        # Validate parameters
        if sigma <= 0:
            raise ValueError("Scale parameter sigma must be positive")
        if nu <= 0:
            raise ValueError("Shape parameter nu must be positive")
        
        self.mu = mu
        self.sigma = sigma
        self.nu = nu
    
    def pdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Probability density function for the GED distribution.
        
        Parameters
        ----------
        x : Union[float, ndarray]
            Points at which to evaluate the PDF
            
        Returns
        -------
        Union[float, ndarray]
            PDF values for given inputs
        """
        x_array = np.asarray(x)
        return ged_pdf(x_array, self.mu, self.sigma, self.nu)
    
    def cdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Cumulative distribution function for the GED distribution.
        
        Parameters
        ----------
        x : Union[float, ndarray]
            Points at which to evaluate the CDF
            
        Returns
        -------
        Union[float, ndarray]
            CDF values for given inputs
        """
        x_array = np.asarray(x)
        return ged_cdf(x_array, self.mu, self.sigma, self.nu)
    
    def ppf(self, q: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Percent point function (inverse CDF) for the GED distribution.
        
        Parameters
        ----------
        q : Union[float, ndarray]
            Probability points at which to evaluate the PPF
            
        Returns
        -------
        Union[float, ndarray]
            Quantile values for given probabilities
        """
        q_array = np.asarray(q)
        return ged_ppf(q_array, self.mu, self.sigma, self.nu)
    
    def random(self, size: int, seed: Optional[int] = None) -> np.ndarray:
        """
        Generate random samples from the GED distribution.
        
        Parameters
        ----------
        size : int
            Number of random samples to generate
        seed : Optional[int], default=None
            Random seed for reproducibility
            
        Returns
        -------
        ndarray
            Array of random samples
        """
        return ged_random(size, self.mu, self.sigma, self.nu, seed)
    
    @classmethod
    def fit(cls, data: np.ndarray, starting_values: Optional[Dict[str, float]] = None,
           bounds: Optional[Dict[str, Tuple[float, float]]] = None) -> 'GeneralizedErrorDistribution':
        """
        Fit distribution parameters to data using maximum likelihood estimation.
        
        Parameters
        ----------
        data : ndarray
            Input data array
        starting_values : Optional[dict], default=None
            Initial parameter values for optimization. If None, defaults are used.
        bounds : Optional[dict], default=None
            Parameter bounds for optimization. If None, defaults are used.
            
        Returns
        -------
        GeneralizedErrorDistribution
            New instance with fitted parameters
        """
        params = estimate_ged_params(data, starting_values, bounds)
        return cls(params['mu'], params['sigma'], params['nu'])
    
    def loglikelihood(self, data: np.ndarray) -> float:
        """
        Calculate log-likelihood of data under the GED distribution.
        
        Parameters
        ----------
        data : ndarray
            Input data array
            
        Returns
        -------
        float
            Log-likelihood value
        """
        return ged_loglikelihood(data, self.mu, self.sigma, self.nu)


class SkewedTDistribution:
    """
    Class implementation of Hansen's skewed t-distribution with methods for 
    PDF, CDF, PPF, and random sampling.
    """
    
    def __init__(self, mu: float = 0.0, sigma: float = 1.0, nu: float = 5.0, lambda_: float = 0.0):
        """
        Initialize a skewed t-distribution with location, scale, degrees of freedom, and skewness parameters.
        
        Parameters
        ----------
        mu : float, default=0.0
            Location parameter
        sigma : float, default=1.0
            Scale parameter (must be positive)
        nu : float, default=5.0
            Degrees of freedom (must be > 2)
        lambda_ : float, default=0.0
            Skewness parameter (must be between -1 and 1)
        """
        # Validate parameters
        if sigma <= 0:
            raise ValueError("Scale parameter sigma must be positive")
        if nu <= 2:
            raise ValueError("Degrees of freedom nu must be > 2")
        if abs(lambda_) >= 1:
            raise ValueError("Skewness parameter lambda must be between -1 and 1")
        
        self.mu = mu
        self.sigma = sigma
        self.nu = nu
        self.lambda_ = lambda_
    
    def pdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Probability density function for the skewed t-distribution.
        
        Parameters
        ----------
        x : Union[float, ndarray]
            Points at which to evaluate the PDF
            
        Returns
        -------
        Union[float, ndarray]
            PDF values for given inputs
        """
        x_array = np.asarray(x)
        return skewt_pdf(x_array, self.mu, self.sigma, self.nu, self.lambda_)
    
    def cdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Cumulative distribution function for the skewed t-distribution.
        
        Parameters
        ----------
        x : Union[float, ndarray]
            Points at which to evaluate the CDF
            
        Returns
        -------
        Union[float, ndarray]
            CDF values for given inputs
        """
        x_array = np.asarray(x)
        return skewt_cdf(x_array, self.mu, self.sigma, self.nu, self.lambda_)
    
    def ppf(self, q: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Percent point function (inverse CDF) for the skewed t-distribution.
        
        Parameters
        ----------
        q : Union[float, ndarray]
            Probability points at which to evaluate the PPF
            
        Returns
        -------
        Union[float, ndarray]
            Quantile values for given probabilities
        """
        q_array = np.asarray(q)
        return skewt_ppf(q_array, self.mu, self.sigma, self.nu, self.lambda_)
    
    def random(self, size: int, seed: Optional[int] = None) -> np.ndarray:
        """
        Generate random samples from the skewed t-distribution.
        
        Parameters
        ----------
        size : int
            Number of random samples to generate
        seed : Optional[int], default=None
            Random seed for reproducibility
            
        Returns
        -------
        ndarray
            Array of random samples
        """
        return skewt_random(size, self.mu, self.sigma, self.nu, self.lambda_, seed)
    
    @classmethod
    def fit(cls, data: np.ndarray, starting_values: Optional[Dict[str, float]] = None,
          bounds: Optional[Dict[str, Tuple[float, float]]] = None) -> 'SkewedTDistribution':
        """
        Fit distribution parameters to data using maximum likelihood estimation.
        
        Parameters
        ----------
        data : ndarray
            Input data array
        starting_values : Optional[dict], default=None
            Initial parameter values for optimization. If None, defaults are used.
        bounds : Optional[dict], default=None
            Parameter bounds for optimization. If None, defaults are used.
            
        Returns
        -------
        SkewedTDistribution
            New instance with fitted parameters
        """
        params = estimate_skewt_params(data, starting_values, bounds)
        return cls(params['mu'], params['sigma'], params['nu'], params['lambda'])
    
    def loglikelihood(self, data: np.ndarray) -> float:
        """
        Calculate log-likelihood of data under the skewed t-distribution.
        
        Parameters
        ----------
        data : ndarray
            Input data array
            
        Returns
        -------
        float
            Log-likelihood value
        """
        return skewt_loglikelihood(data, self.mu, self.sigma, self.nu, self.lambda_)


class DistributionTest:
    """
    Class for performing various distribution tests on financial time series data.
    """
    
    def __init__(self, data: np.ndarray):
        """
        Initialize a distribution test object with data.
        
        Parameters
        ----------
        data : ndarray
            Time series data to test
        """
        # Validate and store data
        self.data = validate_data(data)
        
        # Initialize results dictionary
        self.test_results = {}
    
    def run_normality_tests(self) -> Dict[str, Dict[str, float]]:
        """
        Run multiple normality tests on the data.
        
        Returns
        -------
        dict
            Dictionary of test results
        """
        # Run Jarque-Bera test
        jb_stat, jb_pval = jarque_bera(self.data)
        
        # Run Shapiro-Wilk test if applicable
        try:
            sw_stat, sw_pval = shapiro_wilk(self.data)
        except:
            sw_stat, sw_pval = np.nan, np.nan
        
        # Run Lilliefors test
        lf_stat, lf_pval = lilliefors(self.data)
        
        # Store results
        results = {
            'jarque_bera': {'statistic': jb_stat, 'p_value': jb_pval},
            'shapiro_wilk': {'statistic': sw_stat, 'p_value': sw_pval},
            'lilliefors': {'statistic': lf_stat, 'p_value': lf_pval}
        }
        
        self.test_results['normality'] = results
        return results
    
    def run_distribution_tests(self, dist_type: str, params: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Test if data follows a specific distribution type.
        
        Parameters
        ----------
        dist_type : str
            Distribution type ('norm', 'ged', 'skewt', 't')
        params : Optional[dict], default=None
            Distribution parameters. If None, parameters are estimated from the data.
            
        Returns
        -------
        dict
            Dictionary of test results
        """
        # Validate distribution type
        if dist_type.lower() not in ['norm', 'ged', 'skewt', 't']:
            raise ValueError(f"Unsupported distribution type: {dist_type}")
        
        # Fit distribution if params not provided
        if params is None:
            fit_results = distribution_fit(self.data, dist_type)
            params = {k: v for k, v in fit_results.items() if k not in ['tests', 'distribution', 'loglikelihood', 'aic', 'bic']}
        
        # Run Kolmogorov-Smirnov test
        ks_stat, ks_pval = ks_test(self.data, dist_type, params)
        
        # Calculate log-likelihood
        if dist_type.lower() == 'norm':
            ll = normal_loglikelihood(self.data, params.get('mu'), params.get('sigma'))
        elif dist_type.lower() == 'ged':
            ll = ged_loglikelihood(self.data, params.get('mu'), params.get('sigma'), params.get('nu'))
        elif dist_type.lower() == 'skewt':
            ll = skewt_loglikelihood(self.data, params.get('mu'), params.get('sigma'), params.get('nu'), params.get('lambda'))
        elif dist_type.lower() == 't':
            ll = t_loglikelihood(self.data, params.get('mu'), params.get('sigma'), params.get('df'))
        
        # Calculate information criteria
        n = len(self.data)
        k = len(params)
        aic = akaike_information_criterion(ll, k)
        bic = bayesian_information_criterion(ll, k, n)
        
        # Store results
        results = {
            'distribution': dist_type,
            'parameters': params,
            'loglikelihood': ll,
            'aic': aic,
            'bic': bic,
            'ks_test': {'statistic': ks_stat, 'p_value': ks_pval}
        }
        
        self.test_results[f'distribution_{dist_type}'] = results
        return results
    
    def compare_distributions(self, dist_types: List[str]) -> Dict[str, Any]:
        """
        Compare multiple distribution fits to find the best model.
        
        Parameters
        ----------
        dist_types : list
            List of distribution types to compare
            
        Returns
        -------
        dict
            Comparison results with best fit information
        """
        # Validate distribution types
        for dist in dist_types:
            if dist.lower() not in ['norm', 'ged', 'skewt', 't']:
                raise ValueError(f"Unsupported distribution type: {dist}")
        
        # Fit each distribution
        results = {}
        for dist in dist_types:
            fit = self.run_distribution_tests(dist)
            results[dist] = fit
        
        # Find best model based on AIC
        aic_values = {dist: results[dist]['aic'] for dist in dist_types}
        best_aic = min(aic_values, key=aic_values.get)
        
        # Find best model based on BIC
        bic_values = {dist: results[dist]['bic'] for dist in dist_types}
        best_bic = min(bic_values, key=bic_values.get)
        
        # Compare p-values from KS tests
        ks_pvals = {dist: results[dist]['ks_test']['p_value'] for dist in dist_types}
        best_ks = max(ks_pvals, key=ks_pvals.get)
        
        # Store comparison results
        comparison = {
            'distributions': results,
            'aic_values': aic_values,
            'bic_values': bic_values,
            'ks_pvalues': ks_pvals,
            'best_aic': best_aic,
            'best_bic': best_bic,
            'best_ks': best_ks,
            'overall_best': best_aic if aic_values[best_aic] <= bic_values[best_bic] else best_bic
        }
        
        self.test_results['comparison'] = comparison
        return comparison
    
    def plot_fit_comparison(self, dist_types: List[str], show_pdf: bool = True,
                           show_cdf: bool = True, show_qq: bool = False) -> Tuple:
        """
        Generate plots comparing empirical distribution with fitted distributions.
        
        Parameters
        ----------
        dist_types : list
            List of distribution types to compare
        show_pdf : bool, default=True
            Whether to show PDF comparison
        show_cdf : bool, default=True
            Whether to show CDF comparison
        show_qq : bool, default=False
            Whether to show QQ-plot comparison
            
        Returns
        -------
        tuple
            Figure and axes objects from matplotlib
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("Matplotlib is required for plotting")
        
        # Determine number of subplots
        num_plots = sum([show_pdf, show_cdf, show_qq])
        if num_plots == 0:
            raise ValueError("At least one plot type must be selected")
        
        # Create figure and subplots
        fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))
        if num_plots == 1:
            axes = [axes]  # Ensure axes is a list for consistent indexing
        
        # Fit distributions if not already done
        for dist in dist_types:
            if f'distribution_{dist}' not in self.test_results:
                self.run_distribution_tests(dist)
        
        # Define common x-range for plots
        x_min = np.min(self.data) - 2 * np.std(self.data)
        x_max = np.max(self.data) + 2 * np.std(self.data)
        x = np.linspace(x_min, x_max, 1000)
        
        # Define colors for different distributions
        colors = {
            'norm': 'blue',
            'ged': 'red',
            't': 'green',
            'skewt': 'purple'
        }
        
        plot_idx = 0
        
        # Plot PDF comparison
        if show_pdf:
            ax = axes[plot_idx]
            
            # Plot histogram of empirical data
            ax.hist(self.data, bins=30, density=True, alpha=0.5, color='gray', label='Data')
            
            # Plot fitted PDFs
            for dist in dist_types:
                params = self.test_results[f'distribution_{dist}']['parameters']
                
                if dist.lower() == 'norm':
                    y = stats.norm.pdf(x, loc=params.get('mu'), scale=params.get('sigma'))
                elif dist.lower() == 'ged':
                    y = ged_pdf(x, params.get('mu'), params.get('sigma'), params.get('nu'))
                elif dist.lower() == 'skewt':
                    y = skewt_pdf(x, params.get('mu'), params.get('sigma'), params.get('nu'), params.get('lambda'))
                elif dist.lower() == 't':
                    y = stats.t.pdf(x, df=params.get('df'), loc=params.get('mu'), scale=params.get('sigma'))
                
                ax.plot(x, y, label=dist, color=colors.get(dist.lower(), 'black'))
            
            ax.set_title('PDF Comparison')
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
            ax.legend()
            plot_idx += 1
        
        # Plot CDF comparison
        if show_cdf:
            ax = axes[plot_idx]
            
            # Plot empirical CDF
            ecdf = ECDF(self.data)
            ax.step(np.sort(self.data), ecdf(np.sort(self.data)), label='Empirical', color='gray')
            
            # Plot fitted CDFs
            for dist in dist_types:
                params = self.test_results[f'distribution_{dist}']['parameters']
                
                if dist.lower() == 'norm':
                    y = stats.norm.cdf(x, loc=params.get('mu'), scale=params.get('sigma'))
                elif dist.lower() == 'ged':
                    y = ged_cdf(x, params.get('mu'), params.get('sigma'), params.get('nu'))
                elif dist.lower() == 'skewt':
                    y = skewt_cdf(x, params.get('mu'), params.get('sigma'), params.get('nu'), params.get('lambda'))
                elif dist.lower() == 't':
                    y = stats.t.cdf(x, df=params.get('df'), loc=params.get('mu'), scale=params.get('sigma'))
                
                ax.plot(x, y, label=dist, color=colors.get(dist.lower(), 'black'))
            
            ax.set_title('CDF Comparison')
            ax.set_xlabel('Value')
            ax.set_ylabel('Probability')
            ax.legend()
            plot_idx += 1
        
        # Plot QQ comparison if requested
        if show_qq:
            ax = axes[plot_idx]
            
            # Create reference line
            min_q = np.min(self.data)
            max_q = np.max(self.data)
            ref_line = np.linspace(min_q, max_q, 100)
            ax.plot(ref_line, ref_line, 'k--', label='Reference Line')
            
            # Plot QQ plots for each distribution
            for dist in dist_types:
                params = self.test_results[f'distribution_{dist}']['parameters']
                
                # Get theoretical quantiles
                p = np.linspace(0.01, 0.99, 100)
                
                if dist.lower() == 'norm':
                    q_theo = stats.norm.ppf(p, loc=params.get('mu'), scale=params.get('sigma'))
                elif dist.lower() == 'ged':
                    q_theo = ged_ppf(p, params.get('mu'), params.get('sigma'), params.get('nu'))
                elif dist.lower() == 'skewt':
                    q_theo = skewt_ppf(p, params.get('mu'), params.get('sigma'), params.get('nu'), params.get('lambda'))
                elif dist.lower() == 't':
                    q_theo = stats.t.ppf(p, df=params.get('df'), loc=params.get('mu'), scale=params.get('sigma'))
                
                # Get empirical quantiles
                q_emp = np.quantile(self.data, p)
                
                ax.scatter(q_theo, q_emp, label=dist, color=colors.get(dist.lower(), 'black'), alpha=0.7)
            
            ax.set_title('Q-Q Plot Comparison')
            ax.set_xlabel('Theoretical Quantiles')
            ax.set_ylabel('Empirical Quantiles')
            ax.legend()
        
        plt.tight_layout()
        return fig, axes