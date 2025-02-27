"""
MFE Toolbox - Statsmodels Helper Module

This module provides helper functions and utilities for integrating with the Statsmodels
library. It includes wrapper classes and utility functions that simplify the process
of creating, fitting, and analyzing statistical models using Statsmodels within the
MFE Toolbox ecosystem.

The module leverages async/await patterns for long-running operations and provides
robust error handling and data validation for econometric modeling applications.
"""

import numpy as np  # numpy 1.26.3
import pandas as pd  # pandas 2.1.4
import statsmodels.api as sm  # statsmodels 0.14.1
import statsmodels.tsa.api as tsa  # statsmodels 0.14.1
from statsmodels.stats import diagnostic as diag  # statsmodels 0.14.1
from statsmodels.tsa.stattools import acf, pacf  # statsmodels 0.14.1
import scipy.stats  # scipy 1.11.4
from typing import Any, Awaitable, Dict, List, Optional, Tuple, Union, Callable
import asyncio  # Python 3.12
import logging  # Python 3.12

# Internal imports
from .validation import validate_data, validate_model_params
from .numpy_helpers import ensure_array

# Set up module logger
logger = logging.getLogger(__name__)


def create_statsmodel(
    data: np.ndarray,
    model_type: str,
    params: Optional[dict] = None
) -> sm.base.model.Model:
    """
    Creates a Statsmodels model with validated inputs and proper configuration for MFE requirements.
    
    Parameters
    ----------
    data : np.ndarray
        Input data for the model
    model_type : str
        Type of model to create (e.g., 'OLS', 'ARIMA', 'VAR', 'GARCH')
    params : Optional[dict], default=None
        Additional parameters for model initialization
        
    Returns
    -------
    statsmodels.base.model.Model
        Initialized Statsmodels model instance
        
    Raises
    ------
    ValueError
        If model_type is not supported or parameters are invalid
    TypeError
        If data has incorrect type
    """
    # Validate input data
    data = validate_data(data, min_length=2, allow_none=False)
    
    # Initialize default params if None
    if params is None:
        params = {}
    
    # Create model based on model_type
    model_type = model_type.upper()
    
    try:
        if model_type == 'OLS':
            # Extract endog and exog
            if 'exog' in params:
                exog = ensure_array(params.pop('exog'))
            else:
                if data.ndim == 2 and data.shape[1] > 1:
                    # Assume first column is endog, rest are exog
                    exog = data[:, 1:]
                    data = data[:, 0]
                else:
                    # No exogenous variables, add constant
                    exog = sm.add_constant(np.ones(len(data)))
            
            return sm.OLS(data, exog, **params)
        
        elif model_type == 'ARIMA' or model_type == 'ARIMAX':
            order = params.pop('order', (1, 0, 0))
            seasonal_order = params.pop('seasonal_order', None)
            
            if 'exog' in params:
                exog = ensure_array(params.pop('exog'))
                return tsa.SARIMAX(data, exog=exog, order=order, 
                                 seasonal_order=seasonal_order, **params)
            else:
                return tsa.SARIMAX(data, order=order, 
                                 seasonal_order=seasonal_order, **params)
        
        elif model_type == 'VAR':
            lags = params.pop('lags', 1)
            return tsa.VAR(data, **params)
        
        elif model_type == 'GARCH':
            p = params.pop('p', 1)
            o = params.pop('o', 0)
            q = params.pop('q', 1)
            
            # Handle arch_model import only when needed to avoid unnecessary dependency
            try:
                from arch import arch_model
                return arch_model(data, p=p, o=o, q=q, **params)
            except ImportError:
                raise ImportError("The 'arch' package is required for GARCH modeling. "
                                 "Install it using: pip install arch")
        
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    except Exception as e:
        logger.error(f"Error creating {model_type} model: {str(e)}")
        raise ValueError(f"Failed to create {model_type} model: {str(e)}")


def fit_statsmodel(
    model: sm.base.model.Model,
    fit_params: Optional[dict] = None,
    async_fit: bool = False
) -> Union[sm.base.model.Results, Awaitable[sm.base.model.Results]]:
    """
    Fits a Statsmodels model with robust error handling and optional async execution.
    
    Parameters
    ----------
    model : statsmodels.base.model.Model
        The Statsmodels model to fit
    fit_params : Optional[dict], default=None
        Additional parameters for model fitting
    async_fit : bool, default=False
        If True, performs fitting asynchronously
        
    Returns
    -------
    Union[statsmodels.base.model.Results, Awaitable[statsmodels.base.model.Results]]
        Model fitting results or awaitable for async execution
        
    Raises
    ------
    ValueError
        If model fitting fails
    TypeError
        If model is not a Statsmodels model
    """
    # Validate model input
    if not isinstance(model, sm.base.model.Model) and not hasattr(model, 'fit'):
        raise TypeError("Input must be a Statsmodels model or have a 'fit' method")
    
    # Initialize fit params if None
    if fit_params is None:
        fit_params = {}
    
    # Define async fit function for long-running operations
    async def _async_fit():
        loop = asyncio.get_running_loop()
        try:
            # Run model fitting in an executor to avoid blocking
            return await loop.run_in_executor(
                None, lambda: model.fit(**fit_params)
            )
        except Exception as e:
            logger.error(f"Async model fitting failed: {str(e)}")
            raise ValueError(f"Failed to fit model asynchronously: {str(e)}")
    
    # Define synchronous fit function with error handling
    def _sync_fit():
        try:
            return model.fit(**fit_params)
        except Exception as e:
            logger.error(f"Model fitting failed: {str(e)}")
            raise ValueError(f"Failed to fit model: {str(e)}")
    
    # Return appropriate result based on async_fit flag
    if async_fit:
        return _async_fit()
    else:
        return _sync_fit()


def extract_statsmodel_results(
    results: sm.base.model.Results,
    metrics: Optional[list] = None
) -> dict:
    """
    Extracts and formats results from a fitted Statsmodels model into a standardized MFE format.
    
    Parameters
    ----------
    results : statsmodels.base.model.Results
        The fitted model results
    metrics : Optional[list], default=None
        List of specific metrics to extract. If None, extracts all available metrics.
        
    Returns
    -------
    dict
        Formatted results with selected metrics
        
    Raises
    ------
    TypeError
        If results is not a Statsmodels results object
    ValueError
        If requested metrics are not available
    """
    # Validate results input
    if not isinstance(results, sm.base.model.Results) and not hasattr(results, 'params'):
        raise TypeError("Input must be a Statsmodels results object or have a 'params' attribute")
    
    # Define default metrics if none provided
    if metrics is None:
        metrics = ['params', 'bse', 'tvalues', 'pvalues', 'llf', 'aic', 'bic']
    
    # Initialize results dictionary
    extracted_results = {}
    
    # Extract basic information
    try:
        # Get coefficient estimates and statistics
        if hasattr(results, 'params') and 'params' in metrics:
            extracted_results['params'] = results.params.tolist() if hasattr(results.params, 'tolist') else results.params
        
        if hasattr(results, 'bse') and 'bse' in metrics:
            extracted_results['std_errors'] = results.bse.tolist() if hasattr(results.bse, 'tolist') else results.bse
        
        if hasattr(results, 'tvalues') and 'tvalues' in metrics:
            extracted_results['t_stats'] = results.tvalues.tolist() if hasattr(results.tvalues, 'tolist') else results.tvalues
        
        if hasattr(results, 'pvalues') and 'pvalues' in metrics:
            extracted_results['p_values'] = results.pvalues.tolist() if hasattr(results.pvalues, 'tolist') else results.pvalues
        
        # Get likelihood information
        if hasattr(results, 'llf') and 'llf' in metrics:
            extracted_results['log_likelihood'] = results.llf
        
        # Get information criteria
        if hasattr(results, 'aic') and 'aic' in metrics:
            extracted_results['aic'] = results.aic
        
        if hasattr(results, 'bic') and 'bic' in metrics:
            extracted_results['bic'] = results.bic
        
        # Get residuals
        if hasattr(results, 'resid') and 'resid' in metrics:
            extracted_results['residuals'] = results.resid.tolist() if hasattr(results.resid, 'tolist') else results.resid
        
        # Get fitted values
        if hasattr(results, 'fittedvalues') and 'fittedvalues' in metrics:
            extracted_results['fitted_values'] = results.fittedvalues.tolist() if hasattr(results.fittedvalues, 'tolist') else results.fittedvalues
        
        # Special handling for various model types
        if hasattr(results, 'model') and hasattr(results.model, 'endog_names'):
            extracted_results['dependent_variable'] = results.model.endog_names
        
        if hasattr(results, 'model') and hasattr(results.model, 'exog_names'):
            extracted_results['independent_variables'] = results.model.exog_names
        
    except Exception as e:
        logger.warning(f"Error extracting some results: {str(e)}")
    
    return extracted_results


async def async_statsmodel_forecast(
    results: sm.base.model.Results,
    steps: int,
    forecast_params: Optional[dict] = None
) -> np.ndarray:
    """
    Performs forecast using a fitted Statsmodels model with async support for long-running operations.
    
    Parameters
    ----------
    results : statsmodels.base.model.Results
        The fitted model results
    steps : int
        Number of steps to forecast
    forecast_params : Optional[dict], default=None
        Additional parameters for forecasting
        
    Returns
    -------
    np.ndarray
        Forecast results
        
    Raises
    ------
    TypeError
        If results is not a Statsmodels results object
    ValueError
        If forecasting fails or if steps is not positive
    """
    # Validate inputs
    if not isinstance(results, sm.base.model.Results) and not hasattr(results, 'forecast'):
        raise TypeError("Results must be a Statsmodels results object or have a 'forecast' method")
    
    if not isinstance(steps, (int, np.integer)) or steps <= 0:
        raise ValueError(f"Steps must be a positive integer, got {steps}")
    
    # Initialize forecast params if None
    if forecast_params is None:
        forecast_params = {}
    
    # Get running event loop
    loop = asyncio.get_running_loop()
    
    try:
        # Run forecasting in an executor to avoid blocking
        def _forecast():
            try:
                if hasattr(results, 'forecast'):
                    return results.forecast(steps=steps, **forecast_params)
                elif hasattr(results, 'get_forecast'):
                    forecast = results.get_forecast(steps=steps, **forecast_params)
                    return forecast.predicted_mean
                elif hasattr(results, 'predict'):
                    return results.predict(steps=steps, **forecast_params)
                else:
                    raise ValueError("Results object does not support forecasting")
            except Exception as e:
                logger.error(f"Forecasting failed: {str(e)}")
                raise ValueError(f"Failed to perform forecast: {str(e)}")
        
        # Execute forecasting in the executor
        forecast_result = await loop.run_in_executor(None, _forecast)
        
        # Convert to numpy array if necessary
        if not isinstance(forecast_result, np.ndarray):
            forecast_result = np.asarray(forecast_result)
        
        return forecast_result
    
    except Exception as e:
        logger.error(f"Async forecasting failed: {str(e)}")
        raise ValueError(f"Failed to perform async forecast: {str(e)}")


def create_arima_spec(
    p: int,
    d: int,
    q: int,
    additional_params: Optional[dict] = None
) -> dict:
    """
    Creates an ARIMA model specification with proper order parameters and validation.
    
    Parameters
    ----------
    p : int
        Autoregressive order
    d : int
        Integration order
    q : int
        Moving average order
    additional_params : Optional[dict], default=None
        Additional model parameters
        
    Returns
    -------
    dict
        ARIMA specification dictionary
        
    Raises
    ------
    ValueError
        If any order parameter is negative
    """
    # Validate order parameters
    if not isinstance(p, (int, np.integer)) or p < 0:
        raise ValueError(f"AR order (p) must be a non-negative integer, got {p}")
    
    if not isinstance(d, (int, np.integer)) or d < 0:
        raise ValueError(f"Integration order (d) must be a non-negative integer, got {d}")
    
    if not isinstance(q, (int, np.integer)) or q < 0:
        raise ValueError(f"MA order (q) must be a non-negative integer, got {q}")
    
    # Create specification dictionary
    spec = {
        'order': (p, d, q)
    }
    
    # Add additional parameters if provided
    if additional_params is not None:
        if not isinstance(additional_params, dict):
            raise TypeError("additional_params must be a dictionary")
        
        # Handle seasonal parameters specially
        if 'seasonal_order' in additional_params:
            seasonal_order = additional_params.pop('seasonal_order')
            if not isinstance(seasonal_order, tuple) or len(seasonal_order) != 4:
                raise ValueError("seasonal_order must be a tuple of length 4: (P, D, Q, s)")
            
            spec['seasonal_order'] = seasonal_order
        
        # Add remaining parameters
        spec.update(additional_params)
    
    return spec


def create_var_spec(
    lags: int,
    additional_params: Optional[dict] = None
) -> dict:
    """
    Creates a Vector Autoregression (VAR) model specification with proper lag parameters.
    
    Parameters
    ----------
    lags : int
        Number of lags for VAR model
    additional_params : Optional[dict], default=None
        Additional model parameters
        
    Returns
    -------
    dict
        VAR specification dictionary
        
    Raises
    ------
    ValueError
        If lags is not positive
    """
    # Validate lag parameter
    if not isinstance(lags, (int, np.integer)) or lags <= 0:
        raise ValueError(f"Lags must be a positive integer, got {lags}")
    
    # Create specification dictionary
    spec = {
        'lags': lags
    }
    
    # Add additional parameters if provided
    if additional_params is not None:
        if not isinstance(additional_params, dict):
            raise TypeError("additional_params must be a dictionary")
        
        # Add remaining parameters
        spec.update(additional_params)
    
    return spec


def calculate_information_criteria(
    results: sm.base.model.Results,
    criteria: list
) -> dict:
    """
    Calculates various information criteria (AIC, BIC, HQIC) for model selection.
    
    Parameters
    ----------
    results : statsmodels.base.model.Results
        The fitted model results
    criteria : list
        List of criteria to calculate. Valid options: 'aic', 'bic', 'hqic', 'aicc'
        
    Returns
    -------
    dict
        Dictionary of calculated information criteria
        
    Raises
    ------
    TypeError
        If results is not a Statsmodels results object
    ValueError
        If requested criteria are invalid
    """
    # Validate results input
    if not isinstance(results, sm.base.model.Results) and not hasattr(results, 'params'):
        raise TypeError("Input must be a Statsmodels results object or have a 'params' attribute")
    
    # Validate criteria input
    valid_criteria = ['aic', 'bic', 'hqic', 'aicc']
    for criterion in criteria:
        if criterion.lower() not in valid_criteria:
            raise ValueError(f"Invalid criterion: {criterion}. Valid options: {valid_criteria}")
    
    # Initialize output dictionary
    info_criteria = {}
    
    try:
        # Calculate requested information criteria
        for criterion in criteria:
            criterion = criterion.lower()
            
            if criterion == 'aic' and hasattr(results, 'aic'):
                info_criteria['aic'] = results.aic
            
            elif criterion == 'bic' and hasattr(results, 'bic'):
                info_criteria['bic'] = results.bic
            
            elif criterion == 'hqic' and hasattr(results, 'hqic'):
                info_criteria['hqic'] = results.hqic
            
            elif criterion == 'aicc':
                # Calculate AICc if not directly available
                if hasattr(results, 'aicc'):
                    info_criteria['aicc'] = results.aicc
                elif hasattr(results, 'aic') and hasattr(results, 'model'):
                    # AICc = AIC + 2k(k+1)/(n-k-1) where k is num parameters and n is sample size
                    k = len(results.params)
                    n = results.model.nobs
                    
                    if n > k + 1:  # Avoid division by zero
                        aicc_correction = 2 * k * (k + 1) / (n - k - 1)
                        info_criteria['aicc'] = results.aic + aicc_correction
                    else:
                        logger.warning("Cannot calculate AICc: sample size too small relative to parameters")
            
            else:
                logger.warning(f"Information criterion '{criterion}' not available for this model")
    
    except Exception as e:
        logger.warning(f"Error calculating information criteria: {str(e)}")
    
    return info_criteria


def perform_statistical_tests(
    residuals: np.ndarray,
    tests: list
) -> dict:
    """
    Performs statistical tests on model residuals and outputs test results.
    
    Parameters
    ----------
    residuals : np.ndarray
        Residuals from a fitted model
    tests : list
        List of tests to perform. Valid options: 'normality', 'serial_correlation',
        'heteroskedasticity', 'stationarity'
        
    Returns
    -------
    dict
        Dictionary of test results with test statistics and p-values
        
    Raises
    ------
    ValueError
        If residuals array is invalid or tests are not supported
    """
    # Validate residuals input
    residuals = ensure_array(residuals)
    
    if residuals.size == 0:
        raise ValueError("Residuals array cannot be empty")
    
    # Validate tests input
    valid_tests = ['normality', 'serial_correlation', 'heteroskedasticity', 'stationarity']
    for test in tests:
        if test.lower() not in valid_tests:
            raise ValueError(f"Invalid test: {test}. Valid options: {valid_tests}")
    
    # Initialize results dictionary
    test_results = {}
    
    try:
        # Perform requested tests
        for test in tests:
            test = test.lower()
            
            if test == 'normality':
                # Jarque-Bera test for normality
                jb_stat, jb_pval = diag.jarque_bera(residuals)
                test_results['normality'] = {
                    'test': 'Jarque-Bera',
                    'statistic': float(jb_stat),
                    'p_value': float(jb_pval),
                    'null_hypothesis': 'Residuals are normally distributed'
                }
            
            elif test == 'serial_correlation':
                # Ljung-Box test for serial correlation
                lags = min(10, int(len(residuals) / 5))
                if lags > 0:
                    lb_stat, lb_pval = diag.acorr_ljungbox(residuals, lags=[lags])
                    test_results['serial_correlation'] = {
                        'test': 'Ljung-Box Q',
                        'statistic': float(lb_stat[0]),
                        'p_value': float(lb_pval[0]),
                        'lags': lags,
                        'null_hypothesis': 'No serial correlation'
                    }
                else:
                    logger.warning("Not enough observations for serial correlation test")
            
            elif test == 'heteroskedasticity':
                # White test for heteroskedasticity
                try:
                    # Create a simple regression for heteroskedasticity test
                    resid_sq = residuals ** 2
                    x = np.column_stack((np.ones_like(residuals), residuals, residuals ** 2))
                    model = sm.OLS(resid_sq, x).fit()
                    
                    # LM test statistic: n*R^2
                    lm_stat = len(residuals) * model.rsquared
                    lm_pval = 1 - scipy.stats.chi2.cdf(lm_stat, 2)  # df = 2
                    
                    test_results['heteroskedasticity'] = {
                        'test': 'White',
                        'statistic': float(lm_stat),
                        'p_value': float(lm_pval),
                        'null_hypothesis': 'Homoskedasticity'
                    }
                except Exception as e:
                    logger.warning(f"Heteroskedasticity test failed: {str(e)}")
            
            elif test == 'stationarity':
                # Augmented Dickey-Fuller test for stationarity
                try:
                    from statsmodels.tsa.stattools import adfuller
                    adf_result = adfuller(residuals)
                    
                    test_results['stationarity'] = {
                        'test': 'Augmented Dickey-Fuller',
                        'statistic': float(adf_result[0]),
                        'p_value': float(adf_result[1]),
                        'null_hypothesis': 'Unit root present (non-stationary)',
                        'critical_values': adf_result[4]
                    }
                except Exception as e:
                    logger.warning(f"Stationarity test failed: {str(e)}")
    
    except Exception as e:
        logger.warning(f"Error performing statistical tests: {str(e)}")
    
    return test_results


class StatsmodelsWrapper:
    """
    A wrapper class for Statsmodels models providing simplified interface and MFE integration.
    
    This class provides a unified interface to create, fit, and analyze Statsmodels models
    with enhanced error handling, comprehensive validation, and async support.
    
    Attributes
    ----------
    model : statsmodels.base.model.Model
        The underlying Statsmodels model
    results : statsmodels.base.model.Results
        Results from fitted model
    specification : dict
        Model specification parameters
    """
    
    def __init__(self, specification: Optional[dict] = None):
        """
        Initializes the wrapper with optional model specification.
        
        Parameters
        ----------
        specification : Optional[dict], default=None
            Initial model specification parameters
        """
        self.model = None
        self.results = None
        self.specification = specification or {}
    
    def create_model(self, data: np.ndarray, model_type: str, params: Optional[dict] = None) -> None:
        """
        Creates a Statsmodels model based on specification and data.
        
        Parameters
        ----------
        data : np.ndarray
            Input data for the model
        model_type : str
            Type of model to create (e.g., 'OLS', 'ARIMA', 'VAR')
        params : Optional[dict], default=None
            Additional parameters for model initialization
            
        Returns
        -------
        None
            Sets the model attribute internally
            
        Raises
        ------
        ValueError
            If model creation fails
        """
        # Combine specification and params
        combined_params = self.specification.copy()
        if params is not None:
            combined_params.update(params)
        
        # Create the model
        self.model = create_statsmodel(data, model_type, combined_params)
    
    def fit(self, fit_params: Optional[dict] = None, async_fit: bool = False) -> Union[dict, Awaitable[dict]]:
        """
        Fits the model with optional async support.
        
        Parameters
        ----------
        fit_params : Optional[dict], default=None
            Parameters for model fitting
        async_fit : bool, default=False
            If True, performs fitting asynchronously
            
        Returns
        -------
        Union[dict, Awaitable[dict]]
            Formatted results dictionary or awaitable for async execution
            
        Raises
        ------
        ValueError
            If no model has been created or fitting fails
        """
        # Check if model exists
        if self.model is None:
            raise ValueError("No model has been created. Call create_model() first.")
        
        # Define async wrapper
        async def _async_fit_wrapper():
            # Fit the model asynchronously
            self.results = await fit_statsmodel(self.model, fit_params, async_fit=True)
            # Extract and return formatted results
            return extract_statsmodel_results(self.results)
        
        # Synchronous or asynchronous execution
        if async_fit:
            return _async_fit_wrapper()
        else:
            # Fit the model synchronously
            self.results = fit_statsmodel(self.model, fit_params, async_fit=False)
            # Extract and return formatted results
            return extract_statsmodel_results(self.results)
    
    def forecast(self, steps: int, forecast_params: Optional[dict] = None, 
                async_forecast: bool = False) -> Union[np.ndarray, Awaitable[np.ndarray]]:
        """
        Generates forecasts from the fitted model.
        
        Parameters
        ----------
        steps : int
            Number of steps to forecast
        forecast_params : Optional[dict], default=None
            Additional parameters for forecasting
        async_forecast : bool, default=False
            If True, performs forecasting asynchronously
            
        Returns
        -------
        Union[np.ndarray, Awaitable[np.ndarray]]
            Forecast array or awaitable
            
        Raises
        ------
        ValueError
            If no model has been fitted or forecasting fails
        """
        # Check if model has been fitted
        if self.results is None:
            raise ValueError("Model has not been fitted. Call fit() first.")
        
        if async_forecast:
            return async_statsmodel_forecast(self.results, steps, forecast_params)
        else:
            # Synchronous forecasting
            if forecast_params is None:
                forecast_params = {}
            
            try:
                if hasattr(self.results, 'forecast'):
                    return self.results.forecast(steps=steps, **forecast_params)
                elif hasattr(self.results, 'get_forecast'):
                    forecast = self.results.get_forecast(steps=steps, **forecast_params)
                    return forecast.predicted_mean
                elif hasattr(self.results, 'predict'):
                    return self.results.predict(steps=steps, **forecast_params)
                else:
                    raise ValueError("Results object does not support forecasting")
            except Exception as e:
                logger.error(f"Forecasting failed: {str(e)}")
                raise ValueError(f"Failed to perform forecast: {str(e)}")
    
    def get_diagnostics(self, diagnostics: list) -> dict:
        """
        Retrieves diagnostic information from fitted model.
        
        Parameters
        ----------
        diagnostics : list
            List of diagnostic metrics to retrieve
            
        Returns
        -------
        dict
            Diagnostic results dictionary
            
        Raises
        ------
        ValueError
            If no model has been fitted or diagnostics are invalid
        """
        # Check if model has been fitted
        if self.results is None:
            raise ValueError("Model has not been fitted. Call fit() first.")
        
        diagnostic_results = {}
        
        # Process different types of diagnostics
        for diagnostic in diagnostics:
            if diagnostic.lower() == 'information_criteria':
                diagnostic_results['information_criteria'] = calculate_information_criteria(
                    self.results, ['aic', 'bic', 'hqic']
                )
            
            elif diagnostic.lower() == 'residual_tests':
                if hasattr(self.results, 'resid'):
                    diagnostic_results['residual_tests'] = perform_statistical_tests(
                        self.results.resid, ['normality', 'serial_correlation', 'heteroskedasticity']
                    )
                else:
                    logger.warning("Residuals not available for this model")
            
            elif diagnostic.lower() == 'summary_statistics':
                # Get basic summary statistics
                if hasattr(self.results, 'summary'):
                    try:
                        # Extract key statistics from summary
                        diagnostic_results['summary_statistics'] = {
                            'r_squared': self.results.rsquared if hasattr(self.results, 'rsquared') else None,
                            'adj_r_squared': self.results.rsquared_adj if hasattr(self.results, 'rsquared_adj') else None,
                            'f_statistic': self.results.fvalue if hasattr(self.results, 'fvalue') else None,
                            'f_pvalue': self.results.f_pvalue if hasattr(self.results, 'f_pvalue') else None,
                            'log_likelihood': self.results.llf if hasattr(self.results, 'llf') else None,
                            'nobs': self.results.nobs if hasattr(self.results, 'nobs') else None
                        }
                    except Exception as e:
                        logger.warning(f"Error extracting summary statistics: {str(e)}")
            
            else:
                logger.warning(f"Unknown diagnostic: {diagnostic}")
        
        return diagnostic_results


class ARIMAWrapper(StatsmodelsWrapper):
    """
    Specialized wrapper for ARIMA models with simplified interface.
    
    This class extends StatsmodelsWrapper with ARIMA-specific functionality,
    providing streamlined parameter handling and diagnostic plotting.
    
    Attributes
    ----------
    order : tuple
        The (p, d, q) order of the ARIMA model
    seasonal_order : tuple
        The seasonal order of the ARIMA model, default is None (no seasonality)
    """
    
    def __init__(self, p: int, d: int, q: int, seasonal_order: Optional[tuple] = None,
                additional_params: Optional[dict] = None):
        """
        Initializes the ARIMA wrapper with order parameters.
        
        Parameters
        ----------
        p : int
            Autoregressive order
        d : int
            Integration order
        q : int
            Moving average order
        seasonal_order : Optional[tuple], default=None
            Seasonal order in the form (P, D, Q, s)
        additional_params : Optional[dict], default=None
            Additional ARIMA model parameters
        """
        # Create ARIMA specification
        arima_spec = create_arima_spec(p, d, q, additional_params)
        
        # If seasonal_order provided, add it to the specification
        if seasonal_order is not None:
            arima_spec['seasonal_order'] = seasonal_order
        
        # Initialize parent with the specification
        super().__init__(arima_spec)
        
        # Store order parameters as properties
        self.order = (p, d, q)
        self.seasonal_order = seasonal_order
    
    def fit_model(self, data: np.ndarray, fit_params: Optional[dict] = None,
                async_fit: bool = False) -> Union[dict, Awaitable[dict]]:
        """
        Specialized method to fit ARIMA model to time series data.
        
        Parameters
        ----------
        data : np.ndarray
            Time series data
        fit_params : Optional[dict], default=None
            Additional fitting parameters
        async_fit : bool, default=False
            If True, performs fitting asynchronously
            
        Returns
        -------
        Union[dict, Awaitable[dict]]
            Results dictionary or awaitable
            
        Raises
        ------
        ValueError
            If fitting fails
        """
        # Create ARIMA model with data
        self.create_model(data, 'ARIMA', self.specification)
        
        # Fit the model using parent method
        return self.fit(fit_params, async_fit)
    
    def plot_diagnostics(self) -> dict:
        """
        Generates standard diagnostic plots for ARIMA model.
        
        Returns
        -------
        dict
            Dictionary with plot data for the following diagnostics:
            - Residual plot
            - ACF plot
            - PACF plot
            - Q-Q plot
            
        Raises
        ------
        ValueError
            If model has not been fitted
        """
        # Check if model has been fitted
        if self.results is None:
            raise ValueError("Model has not been fitted. Call fit_model() first.")
        
        plot_data = {}
        
        try:
            # Residual plot data
            if hasattr(self.results, 'resid'):
                residuals = self.results.resid
                plot_data['residuals'] = {
                    'x': np.arange(len(residuals)),
                    'y': residuals,
                    'title': 'Residuals Plot',
                    'xlabel': 'Time',
                    'ylabel': 'Residuals'
                }
            
            # ACF plot data
            if hasattr(self.results, 'resid'):
                # Calculate ACF values
                acf_values = acf(self.results.resid, nlags=min(40, len(self.results.resid) // 4))
                plot_data['acf'] = {
                    'x': np.arange(len(acf_values)),
                    'y': acf_values,
                    'title': 'Autocorrelation Function',
                    'xlabel': 'Lag',
                    'ylabel': 'ACF'
                }
            
            # PACF plot data
            if hasattr(self.results, 'resid'):
                # Calculate PACF values
                pacf_values = pacf(self.results.resid, nlags=min(40, len(self.results.resid) // 4))
                plot_data['pacf'] = {
                    'x': np.arange(len(pacf_values)),
                    'y': pacf_values,
                    'title': 'Partial Autocorrelation Function',
                    'xlabel': 'Lag',
                    'ylabel': 'PACF'
                }
            
            # Q-Q plot data
            if hasattr(self.results, 'resid'):
                residuals = self.results.resid
                # Generate theoretical quantiles
                theoretical_quantiles = scipy.stats.norm.ppf(
                    np.linspace(0.01, 0.99, len(residuals))
                )
                # Sort residuals to represent empirical quantiles
                empirical_quantiles = np.sort(residuals)
                
                plot_data['qq'] = {
                    'x': theoretical_quantiles,
                    'y': empirical_quantiles,
                    'title': 'Q-Q Plot',
                    'xlabel': 'Theoretical Quantiles',
                    'ylabel': 'Empirical Quantiles'
                }
            
        except Exception as e:
            logger.warning(f"Error generating diagnostic plots: {str(e)}")
        
        return plot_data


class VARWrapper(StatsmodelsWrapper):
    """
    Specialized wrapper for Vector Autoregression (VAR) models.
    
    This class extends StatsmodelsWrapper with VAR-specific functionality,
    providing streamlined parameter handling and impulse response analysis.
    
    Attributes
    ----------
    lags : int
        Number of lags in the VAR model
    variables : list
        List of variable names in the VAR model
    """
    
    def __init__(self, lags: int, additional_params: Optional[dict] = None):
        """
        Initializes the VAR wrapper with lag parameter.
        
        Parameters
        ----------
        lags : int
            Number of lags for VAR model
        additional_params : Optional[dict], default=None
            Additional VAR model parameters
        """
        # Create VAR specification
        var_spec = create_var_spec(lags, additional_params)
        
        # Initialize parent with the specification
        super().__init__(var_spec)
        
        # Store lag parameter as property
        self.lags = lags
        self.variables = []
    
    def fit_model(self, data: np.ndarray, variable_names: Optional[list] = None,
                fit_params: Optional[dict] = None, async_fit: bool = False) -> Union[dict, Awaitable[dict]]:
        """
        Specialized method to fit VAR model to multivariate time series data.
        
        Parameters
        ----------
        data : np.ndarray
            Multivariate time series data
        variable_names : Optional[list], default=None
            Names of the variables in the data
        fit_params : Optional[dict], default=None
            Additional fitting parameters
        async_fit : bool, default=False
            If True, performs fitting asynchronously
            
        Returns
        -------
        Union[dict, Awaitable[dict]]
            Results dictionary or awaitable
            
        Raises
        ------
        ValueError
            If fitting fails
        """
        # Store variable names if provided
        if variable_names is not None:
            self.variables = variable_names
        
        # Create VAR model with data
        self.create_model(data, 'VAR', self.specification)
        
        # Fit the model using parent method
        return self.fit(fit_params, async_fit)
    
    def impulse_response(self, periods: int) -> np.ndarray:
        """
        Computes impulse response functions from fitted VAR model.
        
        Parameters
        ----------
        periods : int
            Number of periods for impulse response analysis
            
        Returns
        -------
        np.ndarray
            Impulse response function array
            
        Raises
        ------
        ValueError
            If model has not been fitted or periods is not positive
        """
        # Check if model has been fitted
        if self.results is None:
            raise ValueError("Model has not been fitted. Call fit_model() first.")
        
        # Validate periods
        if not isinstance(periods, (int, np.integer)) or periods <= 0:
            raise ValueError(f"Periods must be a positive integer, got {periods}")
        
        try:
            # Check if impulse response method exists
            if hasattr(self.results, 'irf'):
                # Get impulse response
                irf = self.results.irf(periods)
                
                # Return impulse response array
                if hasattr(irf, 'irfs'):
                    return irf.irfs
                else:
                    return irf
            else:
                raise ValueError("Impulse response function not available for this model")
        
        except Exception as e:
            logger.error(f"Error computing impulse response: {str(e)}")
            raise ValueError(f"Failed to compute impulse response: {str(e)}")
    
    def forecast_error_variance_decomposition(self, periods: int) -> np.ndarray:
        """
        Computes forecast error variance decomposition.
        
        Parameters
        ----------
        periods : int
            Number of periods for variance decomposition
            
        Returns
        -------
        np.ndarray
            Variance decomposition array
            
        Raises
        ------
        ValueError
            If model has not been fitted or periods is not positive
        """
        # Check if model has been fitted
        if self.results is None:
            raise ValueError("Model has not been fitted. Call fit_model() first.")
        
        # Validate periods
        if not isinstance(periods, (int, np.integer)) or periods <= 0:
            raise ValueError(f"Periods must be a positive integer, got {periods}")
        
        try:
            # Check if forecast error variance decomposition method exists
            if hasattr(self.results, 'fevd'):
                # Get variance decomposition
                fevd = self.results.fevd(periods)
                
                # Return variance decomposition array
                if hasattr(fevd, 'decomp'):
                    return fevd.decomp
                else:
                    return fevd
            else:
                raise ValueError("Forecast error variance decomposition not available for this model")
        
        except Exception as e:
            logger.error(f"Error computing variance decomposition: {str(e)}")
            raise ValueError(f"Failed to compute variance decomposition: {str(e)}")