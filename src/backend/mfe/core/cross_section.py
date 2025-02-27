"""
MFE Toolbox - Cross-Sectional Analysis Module

This module provides implementations of cross-sectional econometric analysis tools
including regression models and principal component analysis. The module is
optimized with Numba for high-performance numerical computations and includes
asynchronous support for long-running analyses.

The functions in this module are designed to work with NumPy arrays and integrate
with the broader MFE Toolbox ecosystem for financial econometrics.
"""

import numpy as np  # numpy 1.26.3
import scipy.linalg  # scipy 1.11.4
import scipy.stats  # scipy 1.11.4
import pandas as pd  # pandas 2.1.4
import statsmodels.api as sm  # statsmodels 0.14.1
import numba  # numba 0.59.0
from typing import Dict, List, Optional, Tuple, Union, Any

# Internal imports
from ..utils.validation import validate_array, validate_type
from ..utils.numba_helpers import jit_compiler
from ..utils.numpy_helpers import safe_matrix_inverse
from ..utils.async_helpers import async_task
from ..utils.statsmodels_helpers import create_statsmodels_model

# Global constants
DEFAULT_TOLERANCE = 1e-8
MAX_ITERATIONS = 100


@jit_compiler(nopython=False)
def cross_sectional_regression(
    y: np.ndarray,
    X: np.ndarray,
    robust: bool = True,
    cov_type: str = 'HC3'
) -> Dict[str, Any]:
    """
    Performs cross-sectional regression analysis using OLS with robust standard errors.
    
    Parameters
    ----------
    y : np.ndarray
        Dependent variable array (n_samples,)
    X : np.ndarray
        Independent variables array (n_samples, n_features)
    robust : bool, default=True
        If True, computes robust standard errors
    cov_type : str, default='HC3'
        Type of robust covariance estimator to use if robust=True.
        Options include 'HC0', 'HC1', 'HC2', 'HC3', 'HAC'
        
    Returns
    -------
    dict
        Regression results including parameters, standard errors, t-statistics, 
        and p-values
    """
    # Validate input arrays
    y = validate_array(y, param_name='y')
    X = validate_array(X, param_name='X')
    
    # Ensure y is a vector
    if y.ndim > 1:
        if y.shape[1] == 1:
            y = y.flatten()
        else:
            raise ValueError("y must be a vector or a column matrix")
    
    # Ensure X has at least 2 dimensions
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    # Check dimensions compatibility
    if len(y) != X.shape[0]:
        raise ValueError(f"Mismatch in number of samples: y has {len(y)} samples, X has {X.shape[0]} samples")
    
    # Create OLS model using statsmodels
    model = sm.OLS(y, X)
    
    # Fit model with appropriate covariance type
    if robust:
        results = model.fit(cov_type=cov_type)
    else:
        results = model.fit()
    
    # Extract results
    params = results.params
    std_errors = results.bse
    t_stats = results.tvalues
    p_values = results.pvalues
    
    # Calculate additional statistics
    r_squared = results.rsquared
    adj_r_squared = results.rsquared_adj
    residuals = results.resid
    fvalue = results.fvalue
    f_pvalue = results.f_pvalue
    aic = results.aic
    bic = results.bic
    
    # Create result dictionary
    result_dict = {
        'params': params,
        'std_errors': std_errors,
        't_stats': t_stats,
        'p_values': p_values,
        'r_squared': r_squared,
        'adj_r_squared': adj_r_squared,
        'residuals': residuals,
        'fvalue': fvalue,
        'f_pvalue': f_pvalue,
        'aic': aic,
        'bic': bic,
        'n_observations': len(y),
        'df_model': results.df_model,
        'df_resid': results.df_resid
    }
    
    return result_dict


@jit_compiler(nopython=True)
def principal_component_analysis(
    X: np.ndarray,
    n_components: int = None,
    standardize: bool = True
) -> Dict[str, Any]:
    """
    Performs principal component analysis on the input data matrix with Numba optimization.
    
    Parameters
    ----------
    X : np.ndarray
        Input data matrix (n_samples, n_features)
    n_components : int, default=None
        Number of components to retain. If None, all components are kept.
    standardize : bool, default=True
        Whether to standardize the input data (subtract mean, divide by std)
    
    Returns
    -------
    dict
        PCA results including eigenvalues, eigenvectors, explained variance, 
        and projected data
    """
    # Validate input array
    X = validate_array(X, param_name='X')
    
    # Ensure X is a 2D array
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    elif X.ndim > 2:
        raise ValueError("Input data must be 1D or 2D")
    
    n_samples, n_features = X.shape
    
    # Validate n_components
    if n_components is None:
        n_components = min(n_samples, n_features)
    else:
        if not isinstance(n_components, int) or n_components <= 0:
            raise ValueError("n_components must be a positive integer")
        n_components = min(n_components, min(n_samples, n_features))
    
    # Center and optionally standardize the data
    X_centered = X.copy()
    means = np.mean(X, axis=0)
    
    for i in range(n_features):
        X_centered[:, i] = X[:, i] - means[i]
    
    if standardize:
        stds = np.std(X, axis=0, ddof=1)
        for i in range(n_features):
            if stds[i] > 0:  # Avoid division by zero
                X_centered[:, i] = X_centered[:, i] / stds[i]
    
    # Compute covariance matrix
    cov_matrix = np.dot(X_centered.T, X_centered) / (n_samples - 1)
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sort eigenvalues and eigenvectors in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Select top n_components
    eigenvalues = eigenvalues[:n_components]
    eigenvectors = eigenvectors[:, :n_components]
    
    # Project data onto principal components
    projected_data = np.dot(X_centered, eigenvectors)
    
    # Calculate explained variance ratio
    total_variance = np.sum(eigenvalues)
    explained_variance_ratio = eigenvalues / total_variance
    
    # Create result dictionary
    result_dict = {
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors,
        'explained_variance': eigenvalues,
        'explained_variance_ratio': explained_variance_ratio,
        'cumulative_variance_ratio': np.cumsum(explained_variance_ratio),
        'projected_data': projected_data,
        'n_components': n_components,
        'n_samples': n_samples,
        'n_features': n_features,
        'means': means
    }
    
    if standardize:
        result_dict['stds'] = stds
    
    return result_dict


@jit_compiler(nopython=True)
def cross_sectional_stats(
    X: np.ndarray,
    robust: bool = False
) -> Dict[str, Any]:
    """
    Computes basic cross-sectional statistics for input data.
    
    Parameters
    ----------
    X : np.ndarray
        Input data matrix (n_samples, n_features) or vector
    robust : bool, default=False
        If True, computes robust statistics using median and IQR
    
    Returns
    -------
    dict
        Statistical measures including mean, median, standard deviation, 
        skewness, kurtosis
    """
    # Validate input array
    X = validate_array(X, param_name='X')
    
    # Ensure X is a 2D array
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    n_samples, n_features = X.shape
    
    # Initialize output containers
    means = np.zeros(n_features)
    medians = np.zeros(n_features)
    stds = np.zeros(n_features)
    mins = np.zeros(n_features)
    maxs = np.zeros(n_features)
    skewness = np.zeros(n_features)
    kurtosis = np.zeros(n_features)
    
    # Compute statistics for each feature
    for i in range(n_features):
        x = X[:, i]
        
        # Basic statistics
        means[i] = np.mean(x)
        medians[i] = np.median(x)
        stds[i] = np.std(x, ddof=1)
        mins[i] = np.min(x)
        maxs[i] = np.max(x)
        
        # Higher moments (skewness and kurtosis)
        if stds[i] > 0:
            m3 = np.mean((x - means[i])**3)
            m4 = np.mean((x - means[i])**4)
            skewness[i] = m3 / (stds[i]**3)
            kurtosis[i] = m4 / (stds[i]**4) - 3  # Excess kurtosis
    
    # Create result dictionary
    result_dict = {
        'mean': means,
        'median': medians,
        'std': stds,
        'min': mins,
        'max': maxs,
        'range': maxs - mins,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'n_samples': n_samples
    }
    
    # Add robust statistics if requested
    if robust:
        q1 = np.zeros(n_features)
        q3 = np.zeros(n_features)
        iqr = np.zeros(n_features)
        mad = np.zeros(n_features)
        
        for i in range(n_features):
            x = X[:, i]
            q1[i] = np.percentile(x, 25)
            q3[i] = np.percentile(x, 75)
            iqr[i] = q3[i] - q1[i]
            mad[i] = np.median(np.abs(x - medians[i]))
        
        result_dict.update({
            'q1': q1,
            'q3': q3,
            'iqr': iqr,
            'mad': mad
        })
    
    return result_dict


@async_task
async def async_regression(
    y: np.ndarray,
    X: np.ndarray,
    robust: bool = True,
    cov_type: str = 'HC3'
) -> Dict[str, Any]:
    """
    Asynchronous version of cross-sectional regression for long-running analyses.
    
    Parameters
    ----------
    y : np.ndarray
        Dependent variable array (n_samples,)
    X : np.ndarray
        Independent variables array (n_samples, n_features)
    robust : bool, default=True
        If True, computes robust standard errors
    cov_type : str, default='HC3'
        Type of robust covariance estimator to use if robust=True
    
    Returns
    -------
    dict
        Regression results including parameters, standard errors, t-statistics, 
        and p-values
    """
    # Validate input arrays
    y = validate_array(y, param_name='y')
    X = validate_array(X, param_name='X')
    
    # Run the synchronous regression function inside the async context
    return cross_sectional_regression(y, X, robust, cov_type)


@async_task
async def async_pca(
    X: np.ndarray,
    n_components: int = None,
    standardize: bool = True
) -> Dict[str, Any]:
    """
    Asynchronous version of principal component analysis for large datasets.
    
    Parameters
    ----------
    X : np.ndarray
        Input data matrix (n_samples, n_features)
    n_components : int, default=None
        Number of components to retain. If None, all components are kept.
    standardize : bool, default=True
        Whether to standardize the input data (subtract mean, divide by std)
    
    Returns
    -------
    dict
        PCA results including eigenvalues, eigenvectors, explained variance, 
        and projected data
    """
    # Validate input array
    X = validate_array(X, param_name='X')
    
    # Run the synchronous PCA function inside the async context
    return principal_component_analysis(X, n_components, standardize)


@jit_compiler(nopython=True)
def compute_cross_correlation(
    X: np.ndarray,
    method: str = 'pearson'
) -> np.ndarray:
    """
    Computes pairwise cross-sectional correlations between variables with Numba optimization.
    
    Parameters
    ----------
    X : np.ndarray
        Input data matrix (n_samples, n_features)
    method : str, default='pearson'
        Correlation method: 'pearson', 'spearman', or 'kendall'
    
    Returns
    -------
    np.ndarray
        Correlation matrix for the input variables
    """
    # Validate input array
    X = validate_array(X, param_name='X')
    
    # Ensure X is a 2D array
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    n_samples, n_features = X.shape
    
    # Check method parameter
    method = method.lower()
    if method not in ['pearson', 'spearman', 'kendall']:
        raise ValueError("method must be one of 'pearson', 'spearman', or 'kendall'")
    
    # For Spearman correlation, convert to ranks
    if method == 'spearman':
        X_ranked = np.zeros_like(X)
        for i in range(n_features):
            X_ranked[:, i] = scipy.stats.rankdata(X[:, i])
        X = X_ranked
    
    # Initialize correlation matrix
    corr_matrix = np.eye(n_features)
    
    # Compute correlation matrix
    if method in ['pearson', 'spearman']:
        # Standardize the data
        X_std = np.zeros_like(X)
        for i in range(n_features):
            x = X[:, i]
            mean_x = np.mean(x)
            std_x = np.std(x, ddof=1)
            if std_x > 0:
                X_std[:, i] = (x - mean_x) / std_x
            else:
                X_std[:, i] = 0
        
        # Compute correlation using dot product of standardized variables
        for i in range(n_features):
            for j in range(i+1, n_features):
                corr = np.sum(X_std[:, i] * X_std[:, j]) / (n_samples - 1)
                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr
    
    elif method == 'kendall':
        # Compute Kendall's tau correlation
        for i in range(n_features):
            for j in range(i+1, n_features):
                x = X[:, i]
                y = X[:, j]
                
                # Count concordant and discordant pairs
                n_concordant = 0
                n_discordant = 0
                
                for k in range(n_samples):
                    for l in range(k+1, n_samples):
                        a = x[k] - x[l]
                        b = y[k] - y[l]
                        
                        if a * b > 0:
                            n_concordant += 1
                        elif a * b < 0:
                            n_discordant += 1
                
                # Compute tau
                tau = (n_concordant - n_discordant) / (0.5 * n_samples * (n_samples - 1))
                corr_matrix[i, j] = tau
                corr_matrix[j, i] = tau
    
    return corr_matrix


def compute_heteroscedasticity_test(
    residuals: np.ndarray,
    X: np.ndarray,
    test_type: str = 'white'
) -> Dict[str, Any]:
    """
    Performs tests for heteroscedasticity in cross-sectional regression residuals.
    
    Parameters
    ----------
    residuals : np.ndarray
        Residuals from a fitted regression model
    X : np.ndarray
        Independent variables used in the regression
    test_type : str, default='white'
        Type of heteroscedasticity test: 'white', 'breusch_pagan', or 'goldfeld_quandt'
    
    Returns
    -------
    dict
        Test results including test statistic, p-value, and test conclusion
    """
    # Validate input arrays
    residuals = validate_array(residuals, param_name='residuals')
    X = validate_array(X, param_name='X')
    
    # Ensure residuals is a vector
    if residuals.ndim > 1:
        if residuals.shape[1] == 1:
            residuals = residuals.flatten()
        else:
            raise ValueError("residuals must be a vector or a column matrix")
    
    # Ensure X has at least 2 dimensions
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    # Check dimensions compatibility
    if len(residuals) != X.shape[0]:
        raise ValueError(f"Mismatch in number of samples: residuals has {len(residuals)} samples, X has {X.shape[0]} samples")
    
    # Check test_type parameter
    test_type = test_type.lower()
    if test_type not in ['white', 'breusch_pagan', 'goldfeld_quandt']:
        raise ValueError("test_type must be one of 'white', 'breusch_pagan', or 'goldfeld_quandt'")
    
    # Initialize result dictionary
    result = {
        'test_type': test_type,
        'n_observations': len(residuals)
    }
    
    # Perform the specified test
    if test_type == 'white':
        # White's test for heteroscedasticity
        
        # Squared residuals
        residuals_sq = residuals ** 2
        
        # Create auxiliary regressors (including cross products)
        n_vars = X.shape[1]
        aux_X = np.ones((X.shape[0], 1 + 2*n_vars + n_vars*(n_vars-1)//2))
        
        col_idx = 1
        
        # Add linear terms
        for i in range(n_vars):
            aux_X[:, col_idx] = X[:, i]
            col_idx += 1
        
        # Add squared terms
        for i in range(n_vars):
            aux_X[:, col_idx] = X[:, i] ** 2
            col_idx += 1
        
        # Add cross-product terms
        for i in range(n_vars):
            for j in range(i+1, n_vars):
                aux_X[:, col_idx] = X[:, i] * X[:, j]
                col_idx += 1
        
        # Run auxiliary regression
        aux_model = sm.OLS(residuals_sq, aux_X)
        aux_results = aux_model.fit()
        
        # Calculate test statistic (nR²)
        n = len(residuals)
        test_stat = n * aux_results.rsquared
        
        # Calculate p-value (chi-square distribution)
        df = aux_X.shape[1] - 1  # degrees of freedom
        p_value = scipy.stats.chi2.sf(test_stat, df)
        
        # Store results
        result.update({
            'statistic': test_stat,
            'p_value': p_value,
            'df': df,
            'critical_value_1pct': scipy.stats.chi2.ppf(0.99, df),
            'critical_value_5pct': scipy.stats.chi2.ppf(0.95, df),
            'critical_value_10pct': scipy.stats.chi2.ppf(0.90, df),
            'null_hypothesis': 'Homoscedasticity',
            'alternative_hypothesis': 'Heteroscedasticity'
        })
        
    elif test_type == 'breusch_pagan':
        # Breusch-Pagan test for heteroscedasticity
        
        # Squared residuals
        residuals_sq = residuals ** 2
        
        # Standardize squared residuals by their mean
        residuals_sq_std = residuals_sq / np.mean(residuals_sq)
        
        # Run auxiliary regression
        aux_model = sm.OLS(residuals_sq_std, X)
        aux_results = aux_model.fit()
        
        # Calculate test statistic (0.5 * ESS)
        explained_sum_sq = aux_results.ess
        test_stat = 0.5 * explained_sum_sq
        
        # Calculate p-value (chi-square distribution)
        df = X.shape[1] - 1  # degrees of freedom
        p_value = scipy.stats.chi2.sf(test_stat, df)
        
        # Store results
        result.update({
            'statistic': test_stat,
            'p_value': p_value,
            'df': df,
            'critical_value_1pct': scipy.stats.chi2.ppf(0.99, df),
            'critical_value_5pct': scipy.stats.chi2.ppf(0.95, df),
            'critical_value_10pct': scipy.stats.chi2.ppf(0.90, df),
            'null_hypothesis': 'Homoscedasticity',
            'alternative_hypothesis': 'Heteroscedasticity'
        })
        
    elif test_type == 'goldfeld_quandt':
        # Goldfeld-Quandt test for heteroscedasticity
        
        # Determine the variable for sorting (use the first column of X)
        sort_var = X[:, 0]
        
        # Sort data by the sort variable
        sort_idx = np.argsort(sort_var)
        X_sorted = X[sort_idx]
        residuals_sorted = residuals[sort_idx]
        
        # Split the data into two parts (skip middle 20% if sample size is large enough)
        n = len(residuals)
        if n >= 30:
            # Skip middle 20%
            skip_pct = 0.2
            skip_n = int(n * skip_pct)
            start_idx = int(n * (1 - skip_pct) / 2)
            end_idx = n - start_idx
            
            X1 = X_sorted[:start_idx]
            X2 = X_sorted[end_idx:]
            residuals1 = residuals_sorted[:start_idx]
            residuals2 = residuals_sorted[end_idx:]
        else:
            # Split in half
            mid_idx = n // 2
            X1 = X_sorted[:mid_idx]
            X2 = X_sorted[mid_idx:]
            residuals1 = residuals_sorted[:mid_idx]
            residuals2 = residuals_sorted[mid_idx:]
        
        # Compute variances of residuals in each group
        var1 = np.var(residuals1, ddof=1)
        var2 = np.var(residuals2, ddof=1)
        
        # Calculate test statistic (F-ratio of variances)
        if var1 > 0 and var2 > 0:
            # Use the larger variance in the numerator
            if var2 > var1:
                test_stat = var2 / var1
            else:
                test_stat = var1 / var2
            
            # Calculate degrees of freedom
            df1 = len(residuals1) - X.shape[1]
            df2 = len(residuals2) - X.shape[1]
            
            # Calculate p-value (F-distribution)
            p_value = 2 * min(scipy.stats.f.sf(test_stat, df1, df2), 
                             scipy.stats.f.sf(test_stat, df2, df1))
            
            # Store results
            result.update({
                'statistic': test_stat,
                'p_value': p_value,
                'df1': max(df1, df2),
                'df2': min(df1, df2),
                'var_ratio': max(var1, var2) / min(var1, var2),
                'critical_value_1pct': scipy.stats.f.ppf(0.99, max(df1, df2), min(df1, df2)),
                'critical_value_5pct': scipy.stats.f.ppf(0.95, max(df1, df2), min(df1, df2)),
                'critical_value_10pct': scipy.stats.f.ppf(0.90, max(df1, df2), min(df1, df2)),
                'null_hypothesis': 'Homoscedasticity',
                'alternative_hypothesis': 'Heteroscedasticity'
            })
        else:
            result.update({
                'error': 'Cannot compute test statistic - zero variance in one or both groups'
            })
    
    # Add conclusion
    if 'p_value' in result:
        if result['p_value'] < 0.01:
            result['conclusion'] = 'Strong evidence against the null hypothesis of homoscedasticity (p < 0.01)'
        elif result['p_value'] < 0.05:
            result['conclusion'] = 'Evidence against the null hypothesis of homoscedasticity (p < 0.05)'
        elif result['p_value'] < 0.10:
            result['conclusion'] = 'Weak evidence against the null hypothesis of homoscedasticity (p < 0.10)'
        else:
            result['conclusion'] = 'Insufficient evidence against the null hypothesis of homoscedasticity (p >= 0.10)'
    
    return result


class CrossSectionalRegression:
    """
    Class for performing cross-sectional regression analysis with various enhancements.
    
    This class provides a comprehensive interface for cross-sectional regression analysis,
    including model fitting, prediction, diagnostic testing, and result summarization.
    
    Attributes
    ----------
    results_ : dict
        Dictionary containing regression results after fitting
    coef_ : np.ndarray
        Array of estimated coefficients after fitting
    resid_ : np.ndarray
        Array of residuals after fitting
    r_squared_ : float
        R-squared value after fitting
    """
    
    def __init__(
        self,
        robust: bool = True,
        cov_type: str = 'HC3',
        constant: bool = True
    ):
        """
        Initializes the CrossSectionalRegression class.
        
        Parameters
        ----------
        robust : bool, default=True
            If True, computes robust standard errors
        cov_type : str, default='HC3'
            Type of robust covariance estimator to use if robust=True
        constant : bool, default=True
            If True, adds a constant term to the regression
        """
        # Initialize parameters
        self.robust = robust
        self.cov_type = cov_type
        self.constant = constant
        
        # Initialize attributes that will be set after fitting
        self.results_ = None
        self.coef_ = None
        self.resid_ = None
        self.r_squared_ = None
        self.model_ = None
        self.sm_results_ = None
    
    def fit(
        self,
        y: np.ndarray,
        X: np.ndarray
    ) -> 'CrossSectionalRegression':
        """
        Fits the regression model to the provided data.
        
        Parameters
        ----------
        y : np.ndarray
            Dependent variable array
        X : np.ndarray
            Independent variables array
        
        Returns
        -------
        self
            The fitted model instance for method chaining
        """
        # Validate input arrays
        y = validate_array(y, param_name='y')
        X = validate_array(X, param_name='X')
        
        # Ensure y is a vector
        if y.ndim > 1:
            if y.shape[1] == 1:
                y = y.flatten()
            else:
                raise ValueError("y must be a vector or a column matrix")
        
        # Ensure X has at least 2 dimensions
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Check dimensions compatibility
        if len(y) != X.shape[0]:
            raise ValueError(f"Mismatch in number of samples: y has {len(y)} samples, X has {X.shape[0]} samples")
        
        # Add constant to X if specified
        if self.constant:
            X_with_const = sm.add_constant(X)
        else:
            X_with_const = X
        
        # Create and fit the OLS model
        self.model_ = sm.OLS(y, X_with_const)
        
        if self.robust:
            self.sm_results_ = self.model_.fit(cov_type=self.cov_type)
        else:
            self.sm_results_ = self.model_.fit()
        
        # Extract and store results
        self.coef_ = self.sm_results_.params
        self.resid_ = self.sm_results_.resid
        self.r_squared_ = self.sm_results_.rsquared
        
        # Store comprehensive results
        self.results_ = {
            'params': self.sm_results_.params,
            'std_errors': self.sm_results_.bse,
            't_stats': self.sm_results_.tvalues,
            'p_values': self.sm_results_.pvalues,
            'r_squared': self.sm_results_.rsquared,
            'adj_r_squared': self.sm_results_.rsquared_adj,
            'residuals': self.sm_results_.resid,
            'fvalue': self.sm_results_.fvalue,
            'f_pvalue': self.sm_results_.f_pvalue,
            'aic': self.sm_results_.aic,
            'bic': self.sm_results_.bic,
            'n_observations': len(y),
            'df_model': self.sm_results_.df_model,
            'df_resid': self.sm_results_.df_resid,
            'cov_type': self.sm_results_.cov_type,
        }
        
        return self
    
    def predict(
        self,
        X_new: np.ndarray
    ) -> np.ndarray:
        """
        Makes predictions using the fitted model.
        
        Parameters
        ----------
        X_new : np.ndarray
            New independent variables for prediction
        
        Returns
        -------
        np.ndarray
            Predicted values
        """
        # Check if model has been fitted
        if self.coef_ is None:
            raise RuntimeError("Model has not been fitted. Call fit() before predict()")
        
        # Validate input array
        X_new = validate_array(X_new, param_name='X_new')
        
        # Ensure X_new has at least 2 dimensions
        if X_new.ndim == 1:
            X_new = X_new.reshape(-1, 1)
        
        # Add constant to X_new if the model was fitted with a constant
        if self.constant:
            X_new_with_const = sm.add_constant(X_new)
        else:
            X_new_with_const = X_new
        
        # Check compatibility with fitted model
        if X_new_with_const.shape[1] != len(self.coef_):
            raise ValueError(f"X_new has {X_new.shape[1]} features, but the model was fitted with {len(self.coef_) - int(self.constant)} features")
        
        # Make predictions
        return self.sm_results_.predict(X_new_with_const)
    
    def summary(self) -> str:
        """
        Returns a comprehensive summary of regression results.
        
        Returns
        -------
        str
            Formatted summary string
        """
        # Check if model has been fitted
        if self.results_ is None:
            raise RuntimeError("Model has not been fitted. Call fit() before summary()")
        
        # Get the statsmodels summary
        sm_summary = self.sm_results_.summary()
        
        # Return the summary as a string
        return str(sm_summary)
    
    def test_heteroscedasticity(
        self,
        test_type: str = 'white'
    ) -> Dict[str, Any]:
        """
        Tests for heteroscedasticity in the regression residuals.
        
        Parameters
        ----------
        test_type : str, default='white'
            Type of heteroscedasticity test: 'white', 'breusch_pagan', or 'goldfeld_quandt'
        
        Returns
        -------
        dict
            Test results
        """
        # Check if model has been fitted
        if self.results_ is None:
            raise RuntimeError("Model has not been fitted. Call fit() before test_heteroscedasticity()")
        
        # Get the design matrix without the constant term
        if self.constant:
            X = self.model_.exog[:, 1:]
        else:
            X = self.model_.exog
        
        # Perform the heteroscedasticity test
        return compute_heteroscedasticity_test(self.resid_, X, test_type)
    
    async def async_fit(
        self,
        y: np.ndarray,
        X: np.ndarray
    ) -> 'CrossSectionalRegression':
        """
        Asynchronous version of fit method for long-running regressions.
        
        Parameters
        ----------
        y : np.ndarray
            Dependent variable array
        X : np.ndarray
            Independent variables array
        
        Returns
        -------
        self
            The fitted model instance for method chaining
        """
        # Use async_regression to perform the regression
        results = await async_regression(y, X, self.robust, self.cov_type)
        
        # Store the results
        self.results_ = results
        self.coef_ = results['params']
        self.resid_ = results['residuals']
        self.r_squared_ = results['r_squared']
        
        return self


class PrincipalComponentAnalysis:
    """
    Class for performing principal component analysis with Numba optimization.
    
    This class provides a comprehensive interface for principal component analysis,
    including data transformation, inverse transformation, and variance explanation.
    
    Attributes
    ----------
    components_ : np.ndarray
        Principal components (eigenvectors) after fitting
    explained_variance_ : np.ndarray
        Variance explained by each component after fitting
    explained_variance_ratio_ : np.ndarray
        Percentage of variance explained by each component after fitting
    singular_values_ : np.ndarray
        Singular values after fitting
    """
    
    def __init__(
        self,
        n_components: int = None,
        standardize: bool = True
    ):
        """
        Initializes the PrincipalComponentAnalysis class.
        
        Parameters
        ----------
        n_components : int, default=None
            Number of components to retain. If None, all components are kept.
        standardize : bool, default=True
            Whether to standardize the input data (subtract mean, divide by std)
        """
        # Initialize parameters
        self.n_components = n_components
        self.standardize = standardize
        
        # Initialize attributes that will be set after fitting
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.singular_values_ = None
        self.mean_ = None
        self.scale_ = None
        self.n_samples_ = None
        self.n_features_ = None
    
    def fit(
        self,
        X: np.ndarray
    ) -> 'PrincipalComponentAnalysis':
        """
        Fits the PCA model to the provided data.
        
        Parameters
        ----------
        X : np.ndarray
            Input data matrix
        
        Returns
        -------
        self
            The fitted model instance for method chaining
        """
        # Validate input array
        X = validate_array(X, param_name='X')
        
        # Ensure X is a 2D array
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Store dimensions
        self.n_samples_, self.n_features_ = X.shape
        
        # Determine number of components
        if self.n_components is None:
            self.n_components = min(self.n_samples_, self.n_features_)
        else:
            if not isinstance(self.n_components, int) or self.n_components <= 0:
                raise ValueError("n_components must be a positive integer")
            self.n_components = min(self.n_components, min(self.n_samples_, self.n_features_))
        
        # Center the data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # Standardize if requested
        if self.standardize:
            self.scale_ = np.std(X, axis=0, ddof=1)
            # Avoid division by zero
            self.scale_[self.scale_ < 1e-10] = 1.0
            X_centered = X_centered / self.scale_
        else:
            self.scale_ = np.ones(self.n_features_)
        
        # Compute covariance matrix
        cov_matrix = np.dot(X_centered.T, X_centered) / (self.n_samples_ - 1)
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort eigenvalues and eigenvectors in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Select top n_components
        eigenvalues = eigenvalues[:self.n_components]
        eigenvectors = eigenvectors[:, :self.n_components]
        
        # Store components and explained variance
        self.components_ = eigenvectors
        self.explained_variance_ = eigenvalues
        total_variance = np.sum(eigenvalues)
        self.explained_variance_ratio_ = eigenvalues / total_variance
        self.singular_values_ = np.sqrt(eigenvalues * (self.n_samples_ - 1))
        
        return self
    
    def transform(
        self,
        X: np.ndarray
    ) -> np.ndarray:
        """
        Transforms data using the fitted PCA model.
        
        Parameters
        ----------
        X : np.ndarray
            Input data matrix
        
        Returns
        -------
        np.ndarray
            Transformed data
        """
        # Check if model has been fitted
        if self.components_ is None:
            raise RuntimeError("Model has not been fitted. Call fit() before transform()")
        
        # Validate input array
        X = validate_array(X, param_name='X')
        
        # Ensure X is a 2D array
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Check compatibility
        if X.shape[1] != self.n_features_:
            raise ValueError(f"X has {X.shape[1]} features, but the model was fitted with {self.n_features_} features")
        
        # Center and optionally standardize the data
        X_centered = X - self.mean_
        if self.standardize:
            X_centered = X_centered / self.scale_
        
        # Project the data onto the principal components
        return np.dot(X_centered, self.components_)
    
    def fit_transform(
        self,
        X: np.ndarray
    ) -> np.ndarray:
        """
        Fits the model and transforms the data in one step.
        
        Parameters
        ----------
        X : np.ndarray
            Input data matrix
        
        Returns
        -------
        np.ndarray
            Transformed data
        """
        # Fit the model
        self.fit(X)
        
        # Transform the data
        return self.transform(X)
    
    def inverse_transform(
        self,
        X_transformed: np.ndarray
    ) -> np.ndarray:
        """
        Transforms data back to original space.
        
        Parameters
        ----------
        X_transformed : np.ndarray
            Data in transformed space
        
        Returns
        -------
        np.ndarray
            Data in original space
        """
        # Check if model has been fitted
        if self.components_ is None:
            raise RuntimeError("Model has not been fitted. Call fit() before inverse_transform()")
        
        # Validate input array
        X_transformed = validate_array(X_transformed, param_name='X_transformed')
        
        # Ensure X_transformed is a 2D array
        if X_transformed.ndim == 1:
            X_transformed = X_transformed.reshape(-1, 1)
        
        # Check compatibility
        if X_transformed.shape[1] != self.n_components:
            raise ValueError(f"X_transformed has {X_transformed.shape[1]} components, but the model has {self.n_components} components")
        
        # Project back to original space
        X_original = np.dot(X_transformed, self.components_.T)
        
        # Undo standardization if it was applied
        if self.standardize:
            X_original = X_original * self.scale_
        
        # Undo centering
        X_original = X_original + self.mean_
        
        return X_original
    
    async def async_fit(
        self,
        X: np.ndarray
    ) -> 'PrincipalComponentAnalysis':
        """
        Asynchronous version of fit method for large datasets.
        
        Parameters
        ----------
        X : np.ndarray
            Input data matrix
        
        Returns
        -------
        self
            The fitted model instance for method chaining
        """
        # Use async_pca to perform the PCA
        results = await async_pca(X, self.n_components, self.standardize)
        
        # Store the results
        self.components_ = results['eigenvectors']
        self.explained_variance_ = results['explained_variance']
        self.explained_variance_ratio_ = results['explained_variance_ratio']
        self.singular_values_ = np.sqrt(results['explained_variance'] * (results['n_samples'] - 1))
        self.mean_ = results['means']
        if self.standardize:
            self.scale_ = results['stds']
        else:
            self.scale_ = np.ones(results['n_features'])
        self.n_samples_ = results['n_samples']
        self.n_features_ = results['n_features']
        
        return self
    
    async def async_transform(
        self,
        X: np.ndarray
    ) -> np.ndarray:
        """
        Asynchronous version of transform method for large datasets.
        
        Parameters
        ----------
        X : np.ndarray
            Input data matrix
        
        Returns
        -------
        np.ndarray
            Transformed data
        """
        # Check if model has been fitted
        if self.components_ is None:
            raise RuntimeError("Model has not been fitted. Call fit() before async_transform()")
        
        # Validate input array
        X = validate_array(X, param_name='X')
        
        # Run transformation asynchronously
        async def _transform():
            return self.transform(X)
        
        return await _transform()