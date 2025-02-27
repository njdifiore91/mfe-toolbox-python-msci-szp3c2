"""
MFE Toolbox - Statistical Testing Module

This module provides a comprehensive suite of econometric tests and model diagnostics
for financial time series analysis. It includes tests for autocorrelation, normality,
unit roots, independence, heteroskedasticity, and model comparison.

The implementation leverages SciPy's statistical functions and Statsmodels' econometric
capabilities, with performance-critical routines optimized using Numba.
"""

import numpy as np  # numpy 1.26.3
from scipy import stats  # scipy 1.11.4
import statsmodels.api as sm  # statsmodels 0.14.1
import pandas as pd  # pandas 2.1.4
from numba import jit  # numba 0.59.0
from typing import Tuple, List, Optional, Union, Any  # Python 3.12
import asyncio  # Python 3.12
from dataclasses import dataclass  # Python 3.12

from ..utils.validation import validate_array, validate_int_range, validate_bool

# Export list
__all__ = [
    "ljung_box", "jarque_bera", "adf_test", "bds_test", 
    "white_reality_check", "berkowitz_test", "durbin_watson", 
    "engle_arch_test", "white_test", "breusch_pagan_test", 
    "ramsey_reset", "model_confidence_set"
]


@dataclass
class TestResult:
    """
    Class to store and format statistical test results.
    
    Attributes
    ----------
    name : str
        Name of the statistical test
    statistic : float
        Test statistic value
    pvalue : float
        P-value of the test
    additional_info : Optional[dict]
        Additional information about the test result
    """
    name: str
    statistic: float
    pvalue: float
    additional_info: Optional[dict] = None
    
    def __post_init__(self):
        """Initialize additional_info to empty dict if None provided."""
        if self.additional_info is None:
            self.additional_info = {}
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert test result to pandas DataFrame.
        
        Returns
        -------
        pd.DataFrame
            DataFrame representation of test result
        """
        # Create base DataFrame with test statistics
        df = pd.DataFrame({
            'Test': [self.name],
            'Statistic': [self.statistic],
            'P-value': [self.pvalue]
        })
        
        # Add any additional info as columns
        for key, value in self.additional_info.items():
            df[key] = [value]
            
        return df
    
    def __str__(self) -> str:
        """
        String representation of test result.
        
        Returns
        -------
        str
            Formatted string representation
        """
        # Create basic result string
        result = f"{self.name}: Statistic = {self.statistic:.4f}, P-value = {self.pvalue:.4f}"
        
        # Add significance asterisks
        if self.pvalue < 0.001:
            result += " ***"
        elif self.pvalue < 0.01:
            result += " **"
        elif self.pvalue < 0.05:
            result += " *"
            
        # Add additional info if present
        if self.additional_info:
            add_info = ", ".join(f"{k} = {v}" for k, v in self.additional_info.items())
            result += f" ({add_info})"
            
        return result


@jit(nopython=True, cache=True)
def _ljung_box_compute(residuals: np.ndarray, lags: int, 
                       model_df: int = 0) -> Tuple[float, float]:
    """
    Numba optimized computation of Ljung-Box test statistic.
    
    Parameters
    ----------
    residuals : np.ndarray
        Time series residuals to test for autocorrelation
    lags : int
        Number of lags to include in the test
    model_df : int, default=0
        Degrees of freedom used in the model that generated the residuals
    
    Returns
    -------
    Tuple[float, float]
        Test statistic, p-value
    """
    n = len(residuals)
    
    # Compute autocorrelations up to specified lag
    acf = np.zeros(lags)
    residuals_mean = np.mean(residuals)
    residuals_centered = residuals - residuals_mean
    variance = np.sum(residuals_centered**2)
    
    for k in range(1, lags + 1):
        cross_product = 0.0
        for i in range(k, n):
            cross_product += residuals_centered[i] * residuals_centered[i - k]
        acf[k-1] = cross_product / variance
    
    # Calculate Ljung-Box Q statistic
    q_stat = n * (n + 2) * np.sum(acf**2 / (n - np.arange(1, lags + 1)))
    
    # Adjust degrees of freedom
    df = max(1, lags - model_df)
    
    # Calculate p-value from chi-square distribution
    p_value = 1.0 - stats.chi2.cdf(q_stat, df)
    
    return q_stat, p_value


def ljung_box(residuals: np.ndarray, lags: int, model_df: Optional[int] = 0, 
              return_df: bool = False) -> Union[Tuple[float, float], pd.DataFrame]:
    """
    Ljung-Box test for autocorrelation in time series residuals.
    
    Parameters
    ----------
    residuals : np.ndarray
        Time series residuals to test for autocorrelation
    lags : int
        Number of lags to include in the test
    model_df : Optional[int], default=0
        Degrees of freedom used in the model that generated the residuals
    return_df : bool, default=False
        If True, return results as a pandas DataFrame
    
    Returns
    -------
    Union[Tuple[float, float], pd.DataFrame]
        Test statistic, p-value or DataFrame of results
    
    Notes
    -----
    The Ljung-Box test examines whether there is autocorrelation in a time
    series at specified lags. The null hypothesis is that the data are 
    independently distributed.
    """
    # Validate input parameters
    residuals = validate_array(residuals, param_name="residuals")
    
    if residuals.ndim != 1:
        raise ValueError("residuals must be a 1D array")
    
    validate_int_range(lags, 1, len(residuals) - 1, param_name="lags")
    
    if model_df is not None:
        validate_int_range(model_df, 0, lags - 1, param_name="model_df")
    else:
        model_df = 0
    
    validate_bool(return_df, param_name="return_df")
    
    # Compute Ljung-Box statistic and p-value
    q_stat, p_value = _ljung_box_compute(residuals, lags, model_df)
    
    # Return results based on format requested
    if return_df:
        result = TestResult(
            name="Ljung-Box Q Test",
            statistic=q_stat,
            pvalue=p_value,
            additional_info={
                "lags": lags,
                "df": lags - model_df
            }
        )
        return result.to_dataframe()
    else:
        return q_stat, p_value


@jit(nopython=True, cache=True)
def jarque_bera(data: np.ndarray) -> Tuple[float, float]:
    """
    Jarque-Bera test for normality of a distribution.
    
    Parameters
    ----------
    data : np.ndarray
        Data to test for normality
    
    Returns
    -------
    Tuple[float, float]
        Test statistic, p-value
    
    Notes
    -----
    The Jarque-Bera test examines whether sample data have skewness and kurtosis
    matching a normal distribution. The null hypothesis is that the data are normally
    distributed.
    """
    # Basic validation - full validation in the wrapper
    if data.ndim != 1:
        raise ValueError("data must be a 1D array")
    
    n = len(data)
    
    # Calculate mean and center data
    data_mean = np.mean(data)
    data_centered = data - data_mean
    
    # Calculate variance, skewness, and kurtosis
    m2 = np.sum(data_centered**2) / n  # Variance
    m3 = np.sum(data_centered**3) / n  # Third moment
    m4 = np.sum(data_centered**4) / n  # Fourth moment
    
    skewness = m3 / (m2**1.5)
    kurtosis = m4 / (m2**2) - 3.0  # Excess kurtosis
    
    # Compute Jarque-Bera test statistic
    jb_stat = n / 6 * (skewness**2 + kurtosis**2 / 4)
    
    # Calculate p-value from chi-square distribution with 2 degrees of freedom
    p_value = 1.0 - stats.chi2.cdf(jb_stat, 2)
    
    return jb_stat, p_value


def adf_test(data: np.ndarray, regression_type: str = 'c', 
             lags: Optional[int] = None, return_df: bool = False) -> Union[Tuple[float, float], pd.DataFrame]:
    """
    Augmented Dickey-Fuller test for unit roots in time series.
    
    Parameters
    ----------
    data : np.ndarray
        Time series data to test for unit roots
    regression_type : str, default='c'
        Regression model specification:
        - 'c' : constant (default)
        - 'ct' : constant and trend
        - 'ctt' : constant, trend, and quadratic trend
        - 'n' : no constant, no trend
    lags : Optional[int], default=None
        Number of lags to include in the test. If None, determines optimal lag
        order based on AIC or BIC.
    return_df : bool, default=False
        If True, return results as a pandas DataFrame
    
    Returns
    -------
    Union[Tuple[float, float], pd.DataFrame]
        Test statistic, p-value or DataFrame of results
    
    Notes
    -----
    The Augmented Dickey-Fuller test examines whether a time series has a unit root.
    The null hypothesis is that the series has a unit root (is non-stationary).
    """
    # Validate input parameters
    data = validate_array(data, param_name="data")
    
    if data.ndim != 1:
        raise ValueError("data must be a 1D array")
    
    if regression_type not in ['c', 'ct', 'ctt', 'n']:
        raise ValueError("regression_type must be one of: 'c', 'ct', 'ctt', 'n'")
    
    if lags is not None:
        validate_int_range(lags, 0, None, param_name="lags")
    
    validate_bool(return_df, param_name="return_df")
    
    # Use statsmodels implementation for ADF test
    result = sm.tsa.stattools.adfuller(
        data, 
        maxlag=lags, 
        regression=regression_type,
        autolag='AIC' if lags is None else None
    )
    
    # Extract results
    adf_stat = result[0]  # ADF statistic
    p_value = result[1]   # p-value
    used_lags = result[2]  # Used lag order
    n_obs = result[3]      # Number of observations
    critical_values = result[4]  # Critical values
    
    # Return test statistic and p-value or DataFrame
    if return_df:
        additional_info = {
            "lags": used_lags,
            "n_obs": n_obs,
            "critical_1%": critical_values['1%'],
            "critical_5%": critical_values['5%'],
            "critical_10%": critical_values['10%'],
            "regression": regression_type
        }
        
        result = TestResult(
            name="Augmented Dickey-Fuller Test",
            statistic=adf_stat,
            pvalue=p_value,
            additional_info=additional_info
        )
        return result.to_dataframe()
    else:
        return adf_stat, p_value


def bds_test(data: np.ndarray, max_dim: int = 3, epsilon: float = 0.7, 
             return_df: bool = False) -> Union[List[Tuple[float, float]], pd.DataFrame]:
    """
    BDS test for independence of time series based on correlation dimension.
    
    Parameters
    ----------
    data : np.ndarray
        Time series data to test for independence
    max_dim : int, default=3
        Maximum embedding dimension
    epsilon : float, default=0.7
        Distance for detecting close points, usually expressed as a multiple
        of the standard deviation of the data
    return_df : bool, default=False
        If True, return results as a pandas DataFrame
    
    Returns
    -------
    Union[List[Tuple[float, float]], pd.DataFrame]
        List of test statistics and p-values for each dimension or DataFrame
    
    Notes
    -----
    The BDS test examines whether a time series is independent and identically 
    distributed (i.i.d.). The null hypothesis is that the series is i.i.d.
    This test is useful for detecting nonlinear dependence in time series.
    """
    # Validate input parameters
    data = validate_array(data, param_name="data")
    
    if data.ndim != 1:
        raise ValueError("data must be a 1D array")
    
    validate_int_range(max_dim, 2, None, param_name="max_dim")
    
    if not isinstance(epsilon, (int, float)) or epsilon <= 0:
        raise ValueError("epsilon must be a positive number")
    
    validate_bool(return_df, param_name="return_df")
    
    # Use statsmodels implementation for BDS test
    results = []
    std_dev = np.std(data)
    actual_epsilon = epsilon * std_dev  # Standard practice to scale by std dev
    
    for dimension in range(2, max_dim + 1):
        result = sm.tsa.stattools.bds(data, dimension, distance=actual_epsilon)
        bds_stat = result[0]
        p_value = result[1]
        results.append((bds_stat, p_value))
    
    # Return list of test statistics and p-values or DataFrame
    if return_df:
        dimensions = list(range(2, max_dim + 1))
        statistics = [r[0] for r in results]
        p_values = [r[1] for r in results]
        
        df = pd.DataFrame({
            'Dimension': dimensions,
            'Statistic': statistics,
            'P-value': p_values,
            'Epsilon': [epsilon] * len(dimensions)
        })
        
        return df
    else:
        return results


@jit(nopython=True, cache=True)
def _stationary_bootstrap_indices(n: int, block_size: int) -> np.ndarray:
    """
    Generate indices for stationary bootstrap resampling.
    
    Parameters
    ----------
    n : int
        Length of original series
    block_size : int
        Average block size
    
    Returns
    -------
    np.ndarray
        Bootstrap indices
    
    Notes
    -----
    The stationary bootstrap uses random block sizes with a geometric distribution.
    """
    # Probability of starting a new block
    p = 1.0 / block_size
    
    # Initialize indices
    indices = np.zeros(n, dtype=np.int64)
    
    # Generate bootstrap indices
    t = 0
    while t < n:
        # Draw a random starting point
        start = np.random.randint(0, n)
        
        # Draw a random block length from geometric distribution
        # For numba compatibility, we implement a basic geometric sampler
        block_length = 1
        while np.random.random() > p and block_length < n:
            block_length += 1
        
        # Fill indices for this block
        for i in range(block_length):
            if t + i < n:
                indices[t + i] = (start + i) % n
            else:
                break
                
        # Move to next position
        t += block_length
    
    return indices


def white_reality_check(benchmark_losses: np.ndarray, model_losses: np.ndarray, 
                       bootstrap_reps: int = 1000, 
                       block_size: Optional[int] = None) -> Tuple[float, float]:
    """
    White's Reality Check for testing superior predictive ability across multiple models.
    
    Parameters
    ----------
    benchmark_losses : np.ndarray
        Loss series for the benchmark model
    model_losses : np.ndarray
        Loss series for competing models, with shape (n_models, n_observations)
    bootstrap_reps : int, default=1000
        Number of bootstrap replications
    block_size : Optional[int], default=None
        Block size for stationary bootstrap. If None, an optimal size is selected.
    
    Returns
    -------
    Tuple[float, float]
        Test statistic, p-value
    
    Notes
    -----
    White's Reality Check tests whether any of the competing models have superior 
    predictive ability relative to a benchmark model. The null hypothesis is that 
    the benchmark model is not outperformed by any of the competing models.
    """
    # Validate input parameters
    benchmark_losses = validate_array(benchmark_losses, param_name="benchmark_losses")
    model_losses = validate_array(model_losses, param_name="model_losses")
    
    if benchmark_losses.ndim != 1:
        raise ValueError("benchmark_losses must be a 1D array")
    
    if model_losses.ndim != 2:
        raise ValueError("model_losses must be a 2D array with shape (n_models, n_observations)")
    
    n_obs = len(benchmark_losses)
    n_models, n_model_obs = model_losses.shape
    
    if n_model_obs != n_obs:
        raise ValueError(
            f"model_losses must have the same number of observations as benchmark_losses. "
            f"Got {n_model_obs} and {n_obs}."
        )
    
    validate_int_range(bootstrap_reps, 1, None, param_name="bootstrap_reps")
    
    if block_size is not None:
        validate_int_range(block_size, 1, None, param_name="block_size")
    else:
        # Default: approximately the square root of the sample size
        block_size = int(np.sqrt(n_obs))
    
    # Calculate performance differences between models and benchmark
    # Positive values indicate the model outperforms the benchmark
    perf_diffs = benchmark_losses - model_losses
    
    # Compute White's Reality Check statistic
    # Maximum of the normalized sum of performance differences
    mean_diffs = np.mean(perf_diffs, axis=1)
    max_mean_diff = np.max(mean_diffs)
    test_stat = np.sqrt(n_obs) * max_mean_diff
    
    # Generate bootstrap distribution
    bootstrap_stats = np.zeros(bootstrap_reps)
    
    for b in range(bootstrap_reps):
        # Generate bootstrap sample using stationary bootstrap
        indices = _stationary_bootstrap_indices(n_obs, block_size)
        bootstrap_diffs = perf_diffs[:, indices]
        
        # Calculate centered bootstrap sample
        centered_diffs = bootstrap_diffs - np.mean(perf_diffs, axis=1).reshape(-1, 1)
        
        # Calculate bootstrap statistic
        bootstrap_means = np.mean(centered_diffs, axis=1)
        max_bootstrap_mean = np.max(bootstrap_means)
        bootstrap_stats[b] = np.sqrt(n_obs) * max_bootstrap_mean
    
    # Calculate p-value from bootstrap distribution
    p_value = np.mean(bootstrap_stats >= test_stat)
    
    return test_stat, p_value


async def async_white_reality_check(benchmark_losses: np.ndarray, model_losses: np.ndarray, 
                                  bootstrap_reps: int = 1000, 
                                  block_size: Optional[int] = None) -> Tuple[float, float]:
    """
    Asynchronous version of White's Reality Check for testing superior predictive ability.
    
    Parameters
    ----------
    benchmark_losses : np.ndarray
        Loss series for the benchmark model
    model_losses : np.ndarray
        Loss series for competing models, with shape (n_models, n_observations)
    bootstrap_reps : int, default=1000
        Number of bootstrap replications
    block_size : Optional[int], default=None
        Block size for stationary bootstrap. If None, an optimal size is selected.
    
    Returns
    -------
    Tuple[float, float]
        Test statistic, p-value
    
    Notes
    -----
    This is an asynchronous implementation of White's Reality Check that can be used
    with Python's async/await syntax for improved performance when dealing with a large
    number of bootstrap replications.
    """
    # Validate input parameters (same as synchronous version)
    benchmark_losses = validate_array(benchmark_losses, param_name="benchmark_losses")
    model_losses = validate_array(model_losses, param_name="model_losses")
    
    if benchmark_losses.ndim != 1:
        raise ValueError("benchmark_losses must be a 1D array")
    
    if model_losses.ndim != 2:
        raise ValueError("model_losses must be a 2D array with shape (n_models, n_observations)")
    
    n_obs = len(benchmark_losses)
    n_models, n_model_obs = model_losses.shape
    
    if n_model_obs != n_obs:
        raise ValueError(
            f"model_losses must have the same number of observations as benchmark_losses. "
            f"Got {n_model_obs} and {n_obs}."
        )
    
    validate_int_range(bootstrap_reps, 1, None, param_name="bootstrap_reps")
    
    if block_size is not None:
        validate_int_range(block_size, 1, None, param_name="block_size")
    else:
        # Default: approximately the square root of the sample size
        block_size = int(np.sqrt(n_obs))
    
    # Calculate performance differences between models and benchmark
    perf_diffs = benchmark_losses - model_losses
    
    # Compute White's Reality Check statistic
    mean_diffs = np.mean(perf_diffs, axis=1)
    max_mean_diff = np.max(mean_diffs)
    test_stat = np.sqrt(n_obs) * max_mean_diff
    
    # Helper function to compute a single bootstrap sample
    async def compute_bootstrap_sample():
        # Generate bootstrap indices
        indices = _stationary_bootstrap_indices(n_obs, block_size)
        bootstrap_diffs = perf_diffs[:, indices]
        
        # Calculate centered bootstrap sample
        centered_diffs = bootstrap_diffs - np.mean(perf_diffs, axis=1).reshape(-1, 1)
        
        # Calculate bootstrap statistic
        bootstrap_means = np.mean(centered_diffs, axis=1)
        max_bootstrap_mean = np.max(bootstrap_means)
        return np.sqrt(n_obs) * max_bootstrap_mean
    
    # Generate bootstrap distribution asynchronously
    tasks = [compute_bootstrap_sample() for _ in range(bootstrap_reps)]
    bootstrap_stats = await asyncio.gather(*tasks)
    bootstrap_stats = np.array(bootstrap_stats)
    
    # Calculate p-value from bootstrap distribution
    p_value = np.mean(bootstrap_stats >= test_stat)
    
    return test_stat, p_value


def berkowitz_test(pits: np.ndarray, lags: int = 1) -> Tuple[float, float]:
    """
    Berkowitz test for evaluating density forecasts using PITs transformation.
    
    Parameters
    ----------
    pits : np.ndarray
        Probability Integral Transforms (PITs) values to test
    lags : int, default=1
        Number of lags to include in the AR model
    
    Returns
    -------
    Tuple[float, float]
        Test statistic, p-value
    
    Notes
    -----
    The Berkowitz test evaluates the accuracy of density forecasts by testing whether 
    the probability integral transforms (PITs) of the data under the forecast 
    distributions are i.i.d. uniform. The null hypothesis is that the density
    forecasts are correctly specified.
    """
    # Validate input parameters
    pits = validate_array(pits, param_name="pits")
    
    if pits.ndim != 1:
        raise ValueError("pits must be a 1D array")
    
    if np.any((pits < 0) | (pits > 1)):
        raise ValueError("pits values must be between 0 and 1")
    
    validate_int_range(lags, 1, None, param_name="lags")
    
    # Transform PITs to normal using inverse normal CDF
    norm_pits = stats.norm.ppf(pits)
    
    # Fit AR model with specified lags
    y = norm_pits[lags:]
    X = sm.add_constant(
        np.column_stack([norm_pits[i:-lags+i] for i in range(lags)])
    )
    
    # Restricted model (independent standard normal)
    ll_restrict = np.sum(stats.norm.logpdf(norm_pits))
    
    # Unrestricted model (AR with intercept and variance)
    model = sm.OLS(y, X)
    results = model.fit()
    resid = results.resid
    sigma2 = np.sum(resid**2) / len(resid)
    ll_unrestrict = -0.5 * len(resid) * (np.log(2 * np.pi * sigma2) + 1)
    
    # Compute likelihood ratio test statistic
    lr_stat = 2 * (ll_unrestrict - ll_restrict)
    
    # Calculate p-value from chi-square distribution with lags+2 degrees of freedom
    # (lags AR coefficients, intercept, and variance)
    p_value = 1.0 - stats.chi2.cdf(lr_stat, lags + 2)
    
    return lr_stat, p_value


@jit(nopython=True, cache=True)
def durbin_watson(residuals: np.ndarray) -> float:
    """
    Durbin-Watson test for autocorrelation in regression residuals.
    
    Parameters
    ----------
    residuals : np.ndarray
        Regression residuals to test for autocorrelation
    
    Returns
    -------
    float
        Durbin-Watson statistic
    
    Notes
    -----
    The Durbin-Watson test examines whether there is first-order autocorrelation
    in regression residuals. The statistic is approximately 2 for no autocorrelation,
    less than 2 for positive autocorrelation, and greater than 2 for negative
    autocorrelation.
    """
    n = len(residuals)
    
    # Compute sum of squared differences between consecutive residuals
    diff_sum = 0.0
    for i in range(1, n):
        diff_sum += (residuals[i] - residuals[i-1])**2
    
    # Compute sum of squared residuals
    sq_sum = np.sum(residuals**2)
    
    # Calculate Durbin-Watson statistic
    if sq_sum == 0:
        raise ValueError("Sum of squared residuals is zero")
    
    dw_stat = diff_sum / sq_sum
    
    return dw_stat


def engle_arch_test(residuals: np.ndarray, lags: int = 1) -> Tuple[float, float]:
    """
    Engle's ARCH test for heteroskedasticity in time series residuals.
    
    Parameters
    ----------
    residuals : np.ndarray
        Time series residuals to test for ARCH effects
    lags : int, default=1
        Number of lags to include in the test
    
    Returns
    -------
    Tuple[float, float]
        Test statistic, p-value
    
    Notes
    -----
    Engle's ARCH test examines whether there is autoregressive conditional 
    heteroskedasticity (ARCH) in time series residuals. The null hypothesis is 
    that there is no ARCH effect up to the specified order.
    """
    # Validate input parameters
    residuals = validate_array(residuals, param_name="residuals")
    
    if residuals.ndim != 1:
        raise ValueError("residuals must be a 1D array")
    
    validate_int_range(lags, 1, None, param_name="lags")
    
    n = len(residuals)
    if n <= lags:
        raise ValueError(f"Number of observations ({n}) must be greater than lags ({lags})")
    
    # Calculate squared residuals
    residuals_sq = residuals**2
    
    # Set up regression matrix: constant and lagged squared residuals
    X = np.ones((n - lags, lags + 1))
    for i in range(1, lags + 1):
        X[:, i] = residuals_sq[lags - i:-i]
    
    # Dependent variable: current squared residuals
    y = residuals_sq[lags:]
    
    # Run regression
    model = sm.OLS(y, X)
    results = model.fit()
    
    # Calculate LM test statistic (T*R²)
    lm_stat = n * results.rsquared
    
    # Calculate p-value from chi-square distribution with lags degrees of freedom
    p_value = 1.0 - stats.chi2.cdf(lm_stat, lags)
    
    return lm_stat, p_value


def white_test(residuals: np.ndarray, X: np.ndarray, 
               include_cross_terms: bool = True) -> Tuple[float, float]:
    """
    White's test for heteroskedasticity in regression residuals.
    
    Parameters
    ----------
    residuals : np.ndarray
        Regression residuals to test for heteroskedasticity
    X : np.ndarray
        Explanatory variables used in the original regression
    include_cross_terms : bool, default=True
        If True, include cross-products of explanatory variables
    
    Returns
    -------
    Tuple[float, float]
        Test statistic, p-value
    
    Notes
    -----
    White's test examines whether the variance of the errors in a regression model
    is constant (homoskedastic). The null hypothesis is that the variance is constant.
    """
    # Validate input parameters
    residuals = validate_array(residuals, param_name="residuals")
    X = validate_array(X, param_name="X")
    
    if residuals.ndim != 1:
        raise ValueError("residuals must be a 1D array")
    
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    
    n, k = X.shape
    if n != len(residuals):
        raise ValueError(f"X and residuals must have the same number of observations")
    
    validate_bool(include_cross_terms, param_name="include_cross_terms")
    
    # Calculate squared residuals
    residuals_sq = residuals**2
    
    # Construct regressors for auxiliary regression
    # Start with constant term
    aux_X = [np.ones(n)]
    
    # Add original regressors
    for j in range(k):
        aux_X.append(X[:, j])
    
    # Add squared terms
    for j in range(k):
        aux_X.append(X[:, j]**2)
    
    # Add cross-product terms if requested
    if include_cross_terms and k > 1:
        for i in range(k):
            for j in range(i+1, k):
                aux_X.append(X[:, i] * X[:, j])
    
    # Combine all terms into the design matrix
    aux_X = np.column_stack(aux_X)
    
    # Run auxiliary regression of squared residuals on constructed regressors
    model = sm.OLS(residuals_sq, aux_X)
    results = model.fit()
    
    # Calculate LM test statistic (T*R²)
    lm_stat = n * results.rsquared
    
    # Calculate p-value from chi-square distribution
    df = aux_X.shape[1] - 1  # Degrees of freedom (number of regressors - 1)
    p_value = 1.0 - stats.chi2.cdf(lm_stat, df)
    
    return lm_stat, p_value


def breusch_pagan_test(residuals: np.ndarray, X: np.ndarray) -> Tuple[float, float]:
    """
    Breusch-Pagan test for heteroskedasticity in regression residuals.
    
    Parameters
    ----------
    residuals : np.ndarray
        Regression residuals to test for heteroskedasticity
    X : np.ndarray
        Explanatory variables used in the original regression
    
    Returns
    -------
    Tuple[float, float]
        Test statistic, p-value
    
    Notes
    -----
    The Breusch-Pagan test examines whether the variance of the errors in a regression
    model is constant (homoskedastic). The null hypothesis is that the variance is constant.
    Unlike White's test, it assumes a specific form for heteroskedasticity.
    """
    # Validate input parameters
    residuals = validate_array(residuals, param_name="residuals")
    X = validate_array(X, param_name="X")
    
    if residuals.ndim != 1:
        raise ValueError("residuals must be a 1D array")
    
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    
    n, k = X.shape
    if n != len(residuals):
        raise ValueError(f"X and residuals must have the same number of observations")
    
    # Calculate squared residuals and scale by their variance
    sigma2 = np.mean(residuals**2)
    scaled_residuals_sq = residuals**2 / sigma2
    
    # Run auxiliary regression of scaled squared residuals on X
    model = sm.OLS(scaled_residuals_sq, X)
    results = model.fit()
    
    # Calculate LM test statistic (half the regression ESS)
    lm_stat = 0.5 * results.ess
    
    # Calculate p-value from chi-square distribution
    df = k  # Degrees of freedom (number of regressors)
    p_value = 1.0 - stats.chi2.cdf(lm_stat, df)
    
    return lm_stat, p_value


def ramsey_reset(y: np.ndarray, X: np.ndarray, power: int = 4) -> Tuple[float, float]:
    """
    Ramsey's RESET test for functional form misspecification in regression models.
    
    Parameters
    ----------
    y : np.ndarray
        Dependent variable
    X : np.ndarray
        Explanatory variables
    power : int, default=4
        Highest power of fitted values to include in the test
    
    Returns
    -------
    Tuple[float, float]
        Test statistic, p-value
    
    Notes
    -----
    Ramsey's RESET (Regression Equation Specification Error Test) tests whether
    non-linear combinations of the fitted values help explain the response variable.
    The null hypothesis is that the model is correctly specified.
    """
    # Validate input parameters
    y = validate_array(y, param_name="y")
    X = validate_array(X, param_name="X")
    
    if y.ndim != 1:
        raise ValueError("y must be a 1D array")
    
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    
    n, k = X.shape
    if n != len(y):
        raise ValueError(f"y and X must have the same number of observations")
    
    validate_int_range(power, 2, None, param_name="power")
    
    # Fit original model y = X*b
    model1 = sm.OLS(y, X)
    results1 = model1.fit()
    
    # Generate fitted values
    y_hat = results1.fittedvalues
    
    # Create powers of fitted values
    fitted_powers = np.column_stack([y_hat**(i+2) for i in range(power-1)])
    
    # Create augmented regressors
    X_augmented = np.column_stack([X, fitted_powers])
    
    # Fit augmented model
    model2 = sm.OLS(y, X_augmented)
    results2 = model2.fit()
    
    # Calculate F-test for joint significance of added terms
    f_stat = ((results1.ssr - results2.ssr) / (power - 1)) / (results2.ssr / (n - k - (power - 1)))
    
    # Calculate p-value from F distribution
    p_value = 1.0 - stats.f.cdf(f_stat, power - 1, n - k - (power - 1))
    
    return f_stat, p_value


class ModelConfidenceSet:
    """
    Implementation of Model Confidence Set procedure of Hansen et al.
    
    The Model Confidence Set (MCS) procedure identifies the set of models that
    contains the best model(s) with a given level of confidence. It is useful
    for model selection when multiple models are being compared.
    
    References
    ----------
    Hansen, P. R., Lunde, A., & Nason, J. M. (2011). The model confidence set.
    Econometrica, 79(2), 453-497.
    """
    
    def __init__(self, losses: np.ndarray, alpha: float = 0.05, 
                 bootstrap_reps: int = 1000, test_type: str = 'TR'):
        """
        Initialize ModelConfidenceSet with loss matrix and parameters.
        
        Parameters
        ----------
        losses : np.ndarray
            Loss matrix with shape (n_models, n_observations)
        alpha : float, default=0.05
            Significance level
        bootstrap_reps : int, default=1000
            Number of bootstrap replications
        test_type : str, default='TR'
            Type of test statistic:
            - 'TR': Range statistic
            - 'TSQ': Semi-quadratic statistic
        """
        # Validate input parameters
        self._losses = validate_array(losses, param_name="losses")
        
        if losses.ndim != 2:
            raise ValueError("losses must be a 2D array with shape (n_models, n_observations)")
        
        if not isinstance(alpha, float) or alpha <= 0 or alpha >= 1:
            raise ValueError("alpha must be a float between 0 and 1")
        
        validate_int_range(bootstrap_reps, 1, None, param_name="bootstrap_reps")
        
        if test_type not in ['TR', 'TSQ']:
            raise ValueError("test_type must be either 'TR' or 'TSQ'")
        
        self._alpha = alpha
        self._bootstrap_reps = bootstrap_reps
        self._test_type = test_type
        self._survived = list(range(self._losses.shape[0]))  # Initialize with all models
    
    def run(self) -> List[int]:
        """
        Run the Model Confidence Set procedure.
        
        Returns
        -------
        List[int]
            Indices of models in the confidence set
        """
        n_models, n_obs = self._losses.shape
        
        # Initialize model set with all models
        model_set = list(range(n_models))
        
        # Compute loss differences between all pairs of models
        loss_diffs = np.zeros((n_models, n_models, n_obs))
        for i in range(n_models):
            for j in range(n_models):
                loss_diffs[i, j] = self._losses[i] - self._losses[j]
        
        # Implement iterative elimination procedure
        while len(model_set) > 1:
            # Compute test statistic for current model set
            current_loss_diffs = loss_diffs[np.ix_(model_set, model_set)]
            d_bar = np.mean(current_loss_diffs, axis=2)
            
            if self._test_type == 'TR':
                # Range statistic
                t_stats = np.max(np.abs(d_bar), axis=1)
                test_stat = np.max(t_stats)
            else:  # 'TSQ'
                # Semi-quadratic statistic
                t_stats = np.sum(d_bar**2, axis=1)
                test_stat = np.max(t_stats)
            
            # Generate bootstrap distribution
            bootstrap_stats = np.zeros(self._bootstrap_reps)
            
            for b in range(self._bootstrap_reps):
                # Generate bootstrap sample using stationary bootstrap
                indices = _stationary_bootstrap_indices(n_obs, int(np.sqrt(n_obs)))
                bootstrap_diffs = current_loss_diffs[:, :, indices]
                
                # Calculate centered bootstrap sample
                centered_diffs = bootstrap_diffs - d_bar.reshape(len(model_set), len(model_set), 1)
                
                # Calculate bootstrap statistic
                bootstrap_means = np.mean(centered_diffs, axis=2)
                
                if self._test_type == 'TR':
                    # Range statistic
                    bootstrap_t_stats = np.max(np.abs(bootstrap_means), axis=1)
                    bootstrap_stats[b] = np.max(bootstrap_t_stats)
                else:  # 'TSQ'
                    # Semi-quadratic statistic
                    bootstrap_t_stats = np.sum(bootstrap_means**2, axis=1)
                    bootstrap_stats[b] = np.max(bootstrap_t_stats)
            
            # Calculate p-value
            p_value = np.mean(bootstrap_stats >= test_stat)
            
            # If test fails to reject, exit with current model set
            if p_value >= self._alpha:
                break
            
            # If test rejects, eliminate worst model
            if self._test_type == 'TR':
                worst_idx = np.argmax(t_stats)
            else:  # 'TSQ'
                worst_idx = np.argmax(t_stats)
            
            # Map back to original index
            worst_model = model_set[worst_idx]
            
            # Remove the worst model
            model_set.remove(worst_model)
        
        # Save the final model set
        self._survived = model_set
        
        return model_set
    
    async def run_async(self) -> List[int]:
        """
        Run the Model Confidence Set procedure asynchronously.
        
        Returns
        -------
        List[int]
            Indices of models in the confidence set
        """
        n_models, n_obs = self._losses.shape
        
        # Initialize model set with all models
        model_set = list(range(n_models))
        
        # Compute loss differences between all pairs of models
        loss_diffs = np.zeros((n_models, n_models, n_obs))
        for i in range(n_models):
            for j in range(n_models):
                loss_diffs[i, j] = self._losses[i] - self._losses[j]
        
        # Helper function to compute bootstrap statistics asynchronously
        async def compute_bootstrap_stats(current_loss_diffs, d_bar, n_obs, test_type):
            # Generate bootstrap sample using stationary bootstrap
            indices = _stationary_bootstrap_indices(n_obs, int(np.sqrt(n_obs)))
            bootstrap_diffs = current_loss_diffs[:, :, indices]
            
            # Calculate centered bootstrap sample
            centered_diffs = bootstrap_diffs - d_bar.reshape(d_bar.shape[0], d_bar.shape[1], 1)
            
            # Calculate bootstrap statistic
            bootstrap_means = np.mean(centered_diffs, axis=2)
            
            if test_type == 'TR':
                # Range statistic
                bootstrap_t_stats = np.max(np.abs(bootstrap_means), axis=1)
                return np.max(bootstrap_t_stats)
            else:  # 'TSQ'
                # Semi-quadratic statistic
                bootstrap_t_stats = np.sum(bootstrap_means**2, axis=1)
                return np.max(bootstrap_t_stats)
        
        # Implement iterative elimination procedure
        while len(model_set) > 1:
            # Compute test statistic for current model set
            current_loss_diffs = loss_diffs[np.ix_(model_set, model_set)]
            d_bar = np.mean(current_loss_diffs, axis=2)
            
            if self._test_type == 'TR':
                # Range statistic
                t_stats = np.max(np.abs(d_bar), axis=1)
                test_stat = np.max(t_stats)
            else:  # 'TSQ'
                # Semi-quadratic statistic
                t_stats = np.sum(d_bar**2, axis=1)
                test_stat = np.max(t_stats)
            
            # Generate bootstrap distribution asynchronously
            tasks = [compute_bootstrap_stats(current_loss_diffs, d_bar, n_obs, self._test_type)
                    for _ in range(self._bootstrap_reps)]
            bootstrap_stats = await asyncio.gather(*tasks)
            bootstrap_stats = np.array(bootstrap_stats)
            
            # Calculate p-value
            p_value = np.mean(bootstrap_stats >= test_stat)
            
            # If test fails to reject, exit with current model set
            if p_value >= self._alpha:
                break
            
            # If test rejects, eliminate worst model
            if self._test_type == 'TR':
                worst_idx = np.argmax(t_stats)
            else:  # 'TSQ'
                worst_idx = np.argmax(t_stats)
            
            # Map back to original index
            worst_model = model_set[worst_idx]
            
            # Remove the worst model
            model_set.remove(worst_model)
        
        # Save the final model set
        self._survived = model_set
        
        return model_set
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert results to pandas DataFrame.
        
        Returns
        -------
        pd.DataFrame
            DataFrame representation of results
        """
        n_models = self._losses.shape[0]
        model_indices = list(range(n_models))
        
        # Create DataFrame with model indices and inclusion status
        df = pd.DataFrame({
            'Model': model_indices,
            'In_MCS': [idx in self._survived for idx in model_indices]
        })
        
        # Add mean loss for each model
        df['Mean_Loss'] = [np.mean(self._losses[i]) for i in model_indices]
        
        # Sort by inclusion status and mean loss
        df = df.sort_values(['In_MCS', 'Mean_Loss'], ascending=[False, True])
        
        return df


def model_confidence_set(losses: np.ndarray, alpha: float = 0.05, 
                         bootstrap_reps: int = 1000, test_type: str = 'TR',
                         return_df: bool = False) -> Union[List[int], pd.DataFrame]:
    """
    Model Confidence Set procedure for selecting superior models based on loss functions.
    
    Parameters
    ----------
    losses : np.ndarray
        Loss matrix with shape (n_models, n_observations)
    alpha : float, default=0.05
        Significance level
    bootstrap_reps : int, default=1000
        Number of bootstrap replications
    test_type : str, default='TR'
        Type of test statistic ('TR' or 'TSQ')
    return_df : bool, default=False
        If True, return results as a pandas DataFrame
    
    Returns
    -------
    Union[List[int], pd.DataFrame]
        Indices of models in the confidence set or DataFrame
    
    Notes
    -----
    The Model Confidence Set (MCS) procedure identifies the set of models that
    contains the best model(s) with a given level of confidence. It is useful
    for model selection when multiple models are being compared.
    """
    # Create ModelConfidenceSet instance
    mcs = ModelConfidenceSet(
        losses=losses, 
        alpha=alpha, 
        bootstrap_reps=bootstrap_reps, 
        test_type=test_type
    )
    
    # Run the procedure
    result = mcs.run()
    
    # Return the results
    if return_df:
        return mcs.to_dataframe()
    else:
        return result


async def async_model_confidence_set(losses: np.ndarray, alpha: float = 0.05, 
                                   bootstrap_reps: int = 1000, test_type: str = 'TR',
                                   return_df: bool = False) -> Union[List[int], pd.DataFrame]:
    """
    Asynchronous implementation of Model Confidence Set procedure.
    
    Parameters
    ----------
    losses : np.ndarray
        Loss matrix with shape (n_models, n_observations)
    alpha : float, default=0.05
        Significance level
    bootstrap_reps : int, default=1000
        Number of bootstrap replications
    test_type : str, default='TR'
        Type of test statistic ('TR' or 'TSQ')
    return_df : bool, default=False
        If True, return results as a pandas DataFrame
    
    Returns
    -------
    Union[List[int], pd.DataFrame]
        Indices of models in the confidence set or DataFrame
    
    Notes
    -----
    This is an asynchronous implementation of the Model Confidence Set procedure
    that can be used with Python's async/await syntax for improved performance
    when dealing with a large number of bootstrap replications.
    """
    # Create ModelConfidenceSet instance
    mcs = ModelConfidenceSet(
        losses=losses, 
        alpha=alpha, 
        bootstrap_reps=bootstrap_reps, 
        test_type=test_type
    )
    
    # Run the procedure asynchronously
    result = await mcs.run_async()
    
    # Return the results
    if return_df:
        return mcs.to_dataframe()
    else:
        return result