"""
Utilities for formatted printing and displaying of model results, statistical summaries,
and diagnostic information for the MFE Toolbox.

This module provides functions for consistent, well-formatted output of numerical results,
parameter estimates, statistical tests, and tables.
"""

import logging
from typing import Dict, List, Optional, Any, Union
import numpy as np
import pandas as pd
from tabulate import tabulate  # version 0.9.0

# Initialize module logger
logger = logging.getLogger(__name__)

def format_parameter_table(params_df: pd.DataFrame, title: str = None, 
                          float_format: Optional[str] = "%.4f") -> str:
    """
    Creates a formatted table of parameter estimates with statistics such as
    standard errors, t-statistics, and p-values.
    
    Parameters
    ----------
    params_df : pandas.DataFrame
        DataFrame containing parameter estimates and statistics.
        Expected columns: ['Parameter', 'Estimate', 'Std.Error', 't-stat', 'p-value']
    title : str, optional
        Title for the table
    float_format : str, optional
        Format string for floating point numbers, default is "%.4f"
        
    Returns
    -------
    str
        Formatted table as a string
    """
    # Validate DataFrame structure
    required_cols = ['Parameter', 'Estimate', 'Std.Error', 't-stat', 'p-value']
    for col in required_cols:
        if col not in params_df.columns:
            raise ValueError(f"Parameter DataFrame must contain column: {col}")
    
    # Make a copy to avoid modifying the original
    df = params_df.copy()
    
    # Format numeric columns
    numeric_cols = ['Estimate', 'Std.Error', 't-stat']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: float_format % x if pd.notnull(x) else "")
    
    # Format p-values with stars for significance
    if 'p-value' in df.columns:
        df['p-value'] = df['p-value'].apply(
            lambda p: f"{float_format % p}{'***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''}"
            if pd.notnull(p) else ""
        )
    
    # Generate table
    table_str = tabulate(df, headers='keys', tablefmt='psql', showindex=False)
    
    # Add title if provided
    if title:
        table_str = f"{title}\n{table_str}"
        
    # Add significance legend
    table_str += "\nSignificance: *** p<0.01, ** p<0.05, * p<0.1"
    
    return table_str

def format_information_criteria(criteria: Dict[str, float], float_format: Optional[str] = "%.4f") -> str:
    """
    Formats information criteria (AIC, BIC, etc.) for model comparison.
    
    Parameters
    ----------
    criteria : Dict[str, float]
        Dictionary of information criteria, with keys as criterion names and
        values as criterion values (e.g., {'AIC': -123.45, 'BIC': -120.67})
    float_format : str, optional
        Format string for floating point numbers, default is "%.4f"
        
    Returns
    -------
    str
        Formatted information criteria as a string
    """
    # Validate input
    if not isinstance(criteria, dict):
        raise TypeError("Criteria must be provided as a dictionary")
    
    if not criteria:
        return "No information criteria available"
    
    # Format criterion values
    formatted_criteria = [(name, float_format % value) for name, value in criteria.items()]
    
    # Generate table
    table = tabulate(formatted_criteria, headers=['Criterion', 'Value'], tablefmt='psql')
    
    return table

def format_model_summary(model_results: Dict[str, Any], model_name: str, 
                        float_format: Optional[str] = "%.4f") -> str:
    """
    Creates a comprehensive formatted summary of model results including
    parameters, statistics, and diagnostics.
    
    Parameters
    ----------
    model_results : Dict[str, Any]
        Dictionary containing model results including:
        - 'params': DataFrame of parameter estimates
        - 'loglikelihood': Log-likelihood value
        - 'aic', 'bic', etc.: Information criteria values
        - 'diagnostics': Dictionary of diagnostic test results
    model_name : str
        Name or identifier for the model
    float_format : str, optional
        Format string for floating point numbers, default is "%.4f"
        
    Returns
    -------
    str
        Complete formatted model summary as a string
    """
    # Extract components from model results
    summary_parts = []
    
    # Add model header
    header = f"=== {model_name} Results ===\n"
    summary_parts.append(header)
    
    # Format parameter table if available
    if 'params' in model_results and isinstance(model_results['params'], pd.DataFrame):
        param_table = format_parameter_table(
            model_results['params'],
            title="Parameter Estimates",
            float_format=float_format
        )
        summary_parts.append(param_table)
    
    # Format information criteria if available
    criteria = {}
    for key in ['aic', 'bic', 'hqic', 'AIC', 'BIC', 'HQIC']:
        if key.lower() in model_results:
            criteria[key.upper()] = model_results[key.lower()]
        elif key in model_results:
            criteria[key] = model_results[key]
    
    if criteria:
        criteria_table = format_information_criteria(criteria, float_format)
        summary_parts.append("\nInformation Criteria:")
        summary_parts.append(criteria_table)
    
    # Add model statistics
    stats = []
    if 'loglikelihood' in model_results:
        stats.append(('Log-Likelihood', float_format % model_results['loglikelihood']))
    if 'num_obs' in model_results:
        stats.append(('Number of Observations', str(model_results['num_obs'])))
    if 'df_model' in model_results:
        stats.append(('Degrees of Freedom (Model)', str(model_results['df_model'])))
    if 'df_resid' in model_results:
        stats.append(('Degrees of Freedom (Residuals)', str(model_results['df_resid'])))
    
    if stats:
        stats_table = tabulate(stats, headers=['Statistic', 'Value'], tablefmt='psql')
        summary_parts.append("\nModel Statistics:")
        summary_parts.append(stats_table)
    
    # Format diagnostic tests if available
    if 'diagnostics' in model_results and isinstance(model_results['diagnostics'], dict):
        diag_parts = ["\nDiagnostic Tests:"]
        for test_name, test_results in model_results['diagnostics'].items():
            if isinstance(test_results, dict) and 'stat' in test_results and 'pvalue' in test_results:
                test_str = format_statistical_test(
                    test_name,
                    test_results['stat'],
                    test_results['pvalue'],
                    float_format=float_format
                )
                diag_parts.append(test_str)
        
        if len(diag_parts) > 1:  # Only add if we have actual tests
            summary_parts.append("\n".join(diag_parts))
    
    # Combine all parts
    return "\n\n".join(summary_parts)

def print_model_equation(model_spec: Dict[str, Any], parameters: Dict[str, float] = None,
                        include_values: Optional[bool] = False) -> str:
    """
    Generates a formatted string representation of a model equation (e.g., ARMA, GARCH)
    with parameter values.
    
    Parameters
    ----------
    model_spec : Dict[str, Any]
        Dictionary containing model specification including:
        - 'type': Model type (e.g., 'ARMA', 'GARCH')
        - 'order': Model order (e.g., [1, 1] for ARMA(1,1))
        - Additional model-specific parameters
    parameters : Dict[str, float], optional
        Dictionary of parameter values, with keys matching parameter names
    include_values : bool, optional
        Whether to include parameter values in the equation, default is False
        
    Returns
    -------
    str
        Formatted model equation as a string
    """
    if 'type' not in model_spec:
        raise ValueError("Model specification must include 'type'")
    
    model_type = model_spec['type'].upper()
    equation = ""
    
    if model_type == 'ARMA' or model_type == 'ARMAX':
        # Extract order
        if 'order' not in model_spec:
            raise ValueError("ARMA model specification must include 'order'")
        
        p, q = model_spec['order']
        
        # Build equation
        y_term = "y_t"
        
        # Constant term
        has_constant = model_spec.get('constant', False)
        if has_constant:
            const_term = "c"
            if include_values and parameters and 'const' in parameters:
                const_term = f"{parameters['const']:.4f}"
            equation += const_term
        
        # AR terms
        for i in range(1, p + 1):
            param_name = f"ar_{i}"
            param_value = f"φ_{i}"
            
            if include_values and parameters and param_name in parameters:
                param_value = f"{parameters[param_name]:.4f}"
            
            if equation:
                equation += f" + {param_value} y_{{t-{i}}}"
            else:
                equation += f"{param_value} y_{{t-{i}}}"
        
        # MA terms
        for i in range(1, q + 1):
            param_name = f"ma_{i}"
            param_value = f"θ_{i}"
            
            if include_values and parameters and param_name in parameters:
                param_value = f"{parameters[param_name]:.4f}"
            
            if equation:
                equation += f" + {param_value} ε_{{t-{i}}}"
            else:
                equation += f"{param_value} ε_{{t-{i}}}"
        
        # Add error term
        equation += " + ε_t"
        
        # Complete equation
        equation = f"{y_term} = {equation}"
        
    elif model_type in ['GARCH', 'EGARCH', 'AGARCH', 'TARCH', 'APARCH']:
        # Extract order
        if 'order' not in model_spec:
            raise ValueError(f"{model_type} model specification must include 'order'")
        
        p, q = model_spec['order']
        
        # For GARCH-type models, the equation represents the conditional variance
        if model_type == 'GARCH':
            # Standard GARCH(p,q) model
            equation = "σ²_t = ω"
            
            # ARCH terms
            for i in range(1, q + 1):
                param_name = f"alpha_{i}"
                param_value = f"α_{i}"
                
                if include_values and parameters and param_name in parameters:
                    param_value = f"{parameters[param_name]:.4f}"
                
                equation += f" + {param_value} ε²_{{t-{i}}}"
            
            # GARCH terms
            for i in range(1, p + 1):
                param_name = f"beta_{i}"
                param_value = f"β_{i}"
                
                if include_values and parameters and param_name in parameters:
                    param_value = f"{parameters[param_name]:.4f}"
                
                equation += f" + {param_value} σ²_{{t-{i}}}"
                
        elif model_type == 'EGARCH':
            # EGARCH model with asymmetric terms
            equation = "ln(σ²_t) = ω"
            
            # ARCH terms with asymmetry
            for i in range(1, q + 1):
                alpha_param = f"alpha_{i}"
                alpha_value = f"α_{i}"
                gamma_param = f"gamma_{i}"
                gamma_value = f"γ_{i}"
                
                if include_values and parameters:
                    if alpha_param in parameters:
                        alpha_value = f"{parameters[alpha_param]:.4f}"
                    if gamma_param in parameters:
                        gamma_value = f"{parameters[gamma_param]:.4f}"
                
                equation += f" + {alpha_value} |ε_{{t-{i}}}/σ_{{t-{i}}}|"
                equation += f" + {gamma_value} (ε_{{t-{i}}}/σ_{{t-{i}}})"
            
            # GARCH terms
            for i in range(1, p + 1):
                param_name = f"beta_{i}"
                param_value = f"β_{i}"
                
                if include_values and parameters and param_name in parameters:
                    param_value = f"{parameters[param_name]:.4f}"
                
                equation += f" + {param_value} ln(σ²_{{t-{i}}})"
                
        # Add other GARCH variant equations as needed
        else:
            equation = f"{model_type}({p},{q}) equation not implemented"
    
    else:
        equation = f"{model_type} equation not implemented"
    
    return equation

def format_matrix(matrix: np.ndarray, row_labels: Optional[List[str]] = None,
                 col_labels: Optional[List[str]] = None, title: Optional[str] = None,
                 float_format: Optional[str] = "%.4f") -> str:
    """
    Creates a formatted string representation of a matrix (e.g., correlation matrix).
    
    Parameters
    ----------
    matrix : numpy.ndarray
        Matrix to be formatted
    row_labels : List[str], optional
        Labels for rows
    col_labels : List[str], optional
        Labels for columns
    title : str, optional
        Title for the matrix
    float_format : str, optional
        Format string for floating point numbers, default is "%.4f"
        
    Returns
    -------
    str
        Formatted matrix as a string
    """
    # Validate matrix dimensions
    if not isinstance(matrix, np.ndarray):
        raise TypeError("Matrix must be a numpy ndarray")
    
    if matrix.ndim != 2:
        raise ValueError("Matrix must be 2-dimensional")
    
    # Validate labels
    n_rows, n_cols = matrix.shape
    
    if row_labels is not None and len(row_labels) != n_rows:
        raise ValueError(f"Row labels length ({len(row_labels)}) does not match matrix rows ({n_rows})")
    
    if col_labels is not None and len(col_labels) != n_cols:
        raise ValueError(f"Column labels length ({len(col_labels)}) does not match matrix columns ({n_cols})")
    
    # Convert to DataFrame if labels are provided
    if row_labels is not None or col_labels is not None:
        df = pd.DataFrame(
            matrix,
            index=row_labels if row_labels is not None else None,
            columns=col_labels if col_labels is not None else None
        )
        
        # Apply float formatting
        formatted_df = df.applymap(lambda x: float_format % x)
        
        # Generate table
        table_str = tabulate(formatted_df, headers='keys', tablefmt='psql')
    else:
        # Format numpy array directly
        formatted_matrix = np.array([[float_format % val for val in row] for row in matrix])
        table_str = tabulate(formatted_matrix, tablefmt='psql')
    
    # Add title if provided
    if title:
        table_str = f"{title}\n{table_str}"
    
    return table_str

def format_statistical_test(test_name: str, statistic: float, p_value: float,
                           interpretation: Optional[str] = None, critical_value: Optional[float] = None,
                           float_format: Optional[str] = "%.4f") -> str:
    """
    Formats the results of a statistical test with test statistic, p-value, and interpretation.
    
    Parameters
    ----------
    test_name : str
        Name of the statistical test
    statistic : float
        Test statistic value
    p_value : float
        P-value of the test
    interpretation : str, optional
        Custom interpretation of the test result
    critical_value : float, optional
        Critical value for the test if applicable
    float_format : str, optional
        Format string for floating point numbers, default is "%.4f"
        
    Returns
    -------
    str
        Formatted test result as a string
    """
    # Format statistic and p-value
    formatted_stat = float_format % statistic
    
    # Add significance stars to p-value
    stars = '***' if p_value < 0.01 else '**' if p_value < 0.05 else '*' if p_value < 0.1 else ''
    formatted_p = f"{float_format % p_value}{stars}"
    
    # Add critical value if provided
    critical_str = f", Critical value: {float_format % critical_value}" if critical_value is not None else ""
    
    # Add or generate interpretation
    if interpretation is None:
        if p_value < 0.05:
            interpretation = "Significant at 5% level"
        else:
            interpretation = "Not significant at 5% level"
    
    # Generate formatted test result
    result = f"{test_name}: Statistic = {formatted_stat}, p-value = {formatted_p}{critical_str}\n"
    result += f"Interpretation: {interpretation}"
    
    return result

def format_bootstrap_results(bootstrap_results: Dict[str, Any], alpha: Optional[float] = 0.05,
                            float_format: Optional[str] = "%.4f") -> str:
    """
    Formats bootstrap results including parameter estimates and confidence intervals.
    
    Parameters
    ----------
    bootstrap_results : Dict[str, Any]
        Dictionary containing bootstrap results including:
        - 'estimates': Parameter point estimates
        - 'ci_lower': Lower confidence interval bounds
        - 'ci_upper': Upper confidence interval bounds
    alpha : float, optional
        Significance level (default is 0.05 for 95% confidence intervals)
    float_format : str, optional
        Format string for floating point numbers, default is "%.4f"
        
    Returns
    -------
    str
        Formatted bootstrap results as a string
    """
    # Validate input
    required_keys = ['estimates', 'ci_lower', 'ci_upper']
    for key in required_keys:
        if key not in bootstrap_results:
            raise ValueError(f"Bootstrap results must contain '{key}'")
    
    # Extract results
    estimates = bootstrap_results['estimates']
    ci_lower = bootstrap_results['ci_lower']
    ci_upper = bootstrap_results['ci_upper']
    n_bootstrap = bootstrap_results.get('n_bootstrap', 'Unknown')
    
    # Create result table
    table_data = []
    for param in estimates:
        estimate = float_format % estimates[param]
        lower = float_format % ci_lower[param]
        upper = float_format % ci_upper[param]
        ci_str = f"[{lower}, {upper}]"
        table_data.append([param, estimate, ci_str])
    
    # Generate table
    confidence_level = (1 - alpha) * 100
    title = f"Bootstrap Results ({n_bootstrap} replications, {confidence_level:.0f}% Confidence Intervals)"
    table = tabulate(table_data, headers=['Parameter', 'Estimate', f'{confidence_level:.0f}% CI'], tablefmt='psql')
    
    return f"{title}\n{table}"

def generate_log_message(message: str, context: Optional[Dict[str, Any]] = None,
                        level: Optional[str] = 'INFO') -> str:
    """
    Generates a structured log message for consistent logging output.
    
    Parameters
    ----------
    message : str
        Main message content
    context : Dict[str, Any], optional
        Additional context information as key-value pairs
    level : str, optional
        Log level (e.g., 'INFO', 'WARNING', 'ERROR'), default is 'INFO'
        
    Returns
    -------
    str
        Formatted log message
    """
    # Construct base message
    log_msg = message
    
    # Add context if provided
    if context:
        context_str = ", ".join([f"{k}={v}" for k, v in context.items()])
        log_msg = f"{log_msg} [{context_str}]"
    
    # Add level-specific formatting
    level = level.upper()
    if level == 'WARNING':
        log_msg = f"WARNING: {log_msg}"
    elif level == 'ERROR':
        log_msg = f"ERROR: {log_msg}"
    elif level == 'DEBUG':
        log_msg = f"DEBUG: {log_msg}"
    
    return log_msg

def log_model_estimation(model_type: str, params: Dict[str, Any], likelihood: Optional[float] = None,
                        iteration: Optional[int] = None, converged: Optional[bool] = None) -> None:
    """
    Logs model estimation progress and results to the specified logger.
    
    Parameters
    ----------
    model_type : str
        Type of model being estimated (e.g., 'ARMA', 'GARCH')
    params : Dict[str, Any]
        Model parameters
    likelihood : float, optional
        Current log-likelihood value
    iteration : int, optional
        Current iteration number
    converged : bool, optional
        Whether the model has converged
        
    Returns
    -------
    None
        Logs message but returns nothing
    """
    # Construct message components
    params_str = ", ".join([f"{k}={v}" for k, v in params.items()])
    context = {}
    
    if likelihood is not None:
        context['likelihood'] = f"{likelihood:.6f}"
    
    if iteration is not None:
        context['iteration'] = iteration
        
    if converged is not None:
        context['converged'] = converged
    
    # Build message
    if iteration is not None:
        # Progress message for iterations
        msg = f"{model_type} estimation [iteration {iteration}]: {params_str}"
        logger.debug(generate_log_message(msg, context, 'DEBUG'))
    else:
        # Summary message for final result
        msg = f"{model_type} estimation: {params_str}"
        logger.info(generate_log_message(msg, context, 'INFO'))

def format_warning(message: str, context: Optional[Dict[str, Any]] = None) -> str:
    """
    Formats a warning message with consistent styling.
    
    Parameters
    ----------
    message : str
        Warning message content
    context : Dict[str, Any], optional
        Additional context information as key-value pairs
        
    Returns
    -------
    str
        Formatted warning message
    """
    # Prefix with WARNING
    warning_msg = f"WARNING: {message}"
    
    # Add context if provided
    if context:
        context_str = ", ".join([f"{k}={v}" for k, v in context.items()])
        warning_msg = f"{warning_msg} [{context_str}]"
    
    # Add consistent styling
    warning_msg = f"! {warning_msg} !"
    
    return warning_msg

def format_error(message: str, context: Optional[Dict[str, Any]] = None,
                exception: Optional[Exception] = None) -> str:
    """
    Formats an error message with consistent styling.
    
    Parameters
    ----------
    message : str
        Error message content
    context : Dict[str, Any], optional
        Additional context information as key-value pairs
    exception : Exception, optional
        Exception object if applicable
        
    Returns
    -------
    str
        Formatted error message
    """
    # Prefix with ERROR
    error_msg = f"ERROR: {message}"
    
    # Add context if provided
    if context:
        context_str = ", ".join([f"{k}={v}" for k, v in context.items()])
        error_msg = f"{error_msg} [{context_str}]"
    
    # Add exception info if provided
    if exception:
        error_msg = f"{error_msg}\nException: {type(exception).__name__}: {str(exception)}"
    
    # Add consistent styling
    error_msg = f"!!! {error_msg} !!!"
    
    return error_msg

def setup_logger(name: str, level: Optional[str] = 'INFO',
                format_string: Optional[str] = None,
                log_file: Optional[str] = None) -> logging.Logger:
    """
    Configures a logger with appropriate formatting and handlers.
    
    Parameters
    ----------
    name : str
        Name for the logger
    level : str, optional
        Log level (e.g., 'INFO', 'DEBUG', 'WARNING'), default is 'INFO'
    format_string : str, optional
        Custom format string for log messages
    log_file : str, optional
        Path to log file (if None, only console logging is enabled)
        
    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    
    # Set log level
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    log_level = level_map.get(level.upper(), logging.INFO)
    logger.setLevel(log_level)
    
    # Clear any existing handlers
    if logger.handlers:
        logger.handlers = []
    
    # Create formatter
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(format_string)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log_file is specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger