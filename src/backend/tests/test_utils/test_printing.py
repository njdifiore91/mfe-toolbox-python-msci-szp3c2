"""
Test module for the MFE Toolbox's printing utilities.
Contains comprehensive unit tests for various formatted output functions
including parameter tables, model summaries, statistical test formatting,
matrix formatting, logging functions, and warning/error message formatting.
"""

import pytest  # pytest 7.4.3
import numpy as np  # numpy 1.26.3
import pandas as pd  # pandas 2.1.4
import logging  # Python 3.12
import io  # Python 3.12
import re  # Python 3.12

from mfe.utils.printing import (  # src/backend/mfe/utils/printing.py
    format_parameter_table,
    format_information_criteria,
    format_model_summary,
    format_matrix,
    format_statistical_test,
    format_bootstrap_results,
    print_model_equation,
    generate_log_message,
    log_model_estimation,
    format_warning,
    format_error,
    setup_logger
)
from . import validate_test_data  # src/backend/tests/test_utils/__init__.py

# Apply utils marker to all tests in this module
pytestmark = [pytest.mark.utils]


def test_format_parameter_table():
    """
    Tests the format_parameter_table function with various input DataFrame structures
    """
    # Create test DataFrame with parameter estimates, standard errors, t-statistics, and p-values
    data = {
        'Parameter': ['AR(1)', 'MA(1)', 'Constant'],
        'Estimate': [0.5, 0.2, 0.1],
        'Std.Error': [0.1, 0.05, 0.02],
        't-stat': [5.0, 4.0, 5.0],
        'p-value': [0.001, 0.01, 0.05]
    }
    df = pd.DataFrame(data)

    # Call format_parameter_table with the test DataFrame
    table_str = format_parameter_table(df)

    # Verify table formatting matches expected pattern
    assert "Parameter  Estimate    Std.Error    t-stat    p-value" in table_str
    assert "| AR(1)      0.5000      0.1000         5.0000    0.0010***" in table_str
    assert "| MA(1)      0.2000      0.0500         4.0000    0.0100** " in table_str
    assert "| Constant   0.1000      0.0200         5.0000    0.0500*  " in table_str
    assert "Significance: *** p<0.01, ** p<0.05, * p<0.1" in table_str

    # Check title inclusion when specified
    title = "ARIMA Model Parameters"
    table_str_with_title = format_parameter_table(df, title=title)
    assert title in table_str_with_title
    assert "Parameter  Estimate    Std.Error    t-stat    p-value" in table_str_with_title

    # Verify proper formatting with different float_format options
    float_format = "%.2f"
    table_str_custom_format = format_parameter_table(df, float_format=float_format)
    assert "| AR(1)      0.50        0.10         5.00    0.00***" in table_str_custom_format

    # Test error handling with invalid input
    with pytest.raises(ValueError):
        format_parameter_table("invalid input")


def test_format_information_criteria():
    """
    Tests the format_information_criteria function with different criteria dictionaries
    """
    # Create test dictionary with AIC, BIC, and other information criteria
    criteria = {'AIC': 10.5, 'BIC': 12.0, 'HQIC': 11.5}

    # Call format_information_criteria with the test dictionary
    table_str = format_information_criteria(criteria)

    # Verify formatting matches expected pattern
    assert "Criterion      Value" in table_str
    assert "| AIC            10.5000" in table_str
    assert "| BIC            12.0000" in table_str
    assert "| HQIC           11.5000" in table_str

    # Check numeric formatting with different float_format options
    float_format = "%.2f"
    table_str_custom_format = format_information_criteria(criteria, float_format=float_format)
    assert "| AIC            10.50" in table_str_custom_format

    # Test empty dictionary handling
    empty_criteria = {}
    table_str_empty = format_information_criteria(empty_criteria)
    assert table_str_empty == "No information criteria available"

    # Test error handling with invalid input
    with pytest.raises(TypeError):
        format_information_criteria("invalid input")


def test_format_model_summary():
    """
    Tests the format_model_summary function for creating comprehensive model result summaries
    """
    # Create test model results dictionary with parameters, statistics, and diagnostics
    data = {
        'Parameter': ['AR(1)', 'MA(1)', 'Constant'],
        'Estimate': [0.5, 0.2, 0.1],
        'Std.Error': [0.1, 0.05, 0.02],
        't-stat': [5.0, 4.0, 5.0],
        'p-value': [0.001, 0.01, 0.05]
    }
    params_df = pd.DataFrame(data)

    model_results = {
        'params': params_df,
        'loglikelihood': -100.0,
        'aic': 202.0,
        'bic': 205.0,
        'diagnostics': {
            'LB Test': {'stat': 2.5, 'pvalue': 0.10},
            'JB Test': {'stat': 0.5, 'pvalue': 0.50}
        }
    }

    # Call format_model_summary with the test model results
    model_name = "ARIMA Model"
    summary_str = format_model_summary(model_results, model_name)

    # Verify parameter table inclusion in summary
    assert "Parameter Estimates" in summary_str
    assert "Parameter  Estimate    Std.Error    t-stat    p-value" in summary_str

    # Check information criteria inclusion
    assert "Information Criteria:" in summary_str
    assert "| Criterion   Value" in summary_str
    assert "| AIC         202.0000" in summary_str
    assert "| BIC         205.0000" in summary_str

    # Verify diagnostic test results inclusion when available
    assert "Diagnostic Tests:" in summary_str
    assert "LB Test: Statistic = 2.5000, p-value = 0.1000" in summary_str
    assert "JB Test: Statistic = 0.5000, p-value = 0.5000" in summary_str

    # Test different model_name and float_format options
    model_name = "GARCH Model"
    float_format = "%.2f"
    summary_str_custom = format_model_summary(model_results, model_name, float_format=float_format)
    assert "=== GARCH Model Results ===" in summary_str_custom
    assert "| AIC         202.00" in summary_str_custom

    # Test error handling with invalid input
    with pytest.raises(TypeError):
        format_model_summary("invalid input", model_name)


def test_print_model_equation():
    """
    Tests the print_model_equation function for generating formatted model equations
    """
    # Create test model specification dictionaries for different model types (ARMA, GARCH, etc.)
    arma_spec = {'type': 'ARMA', 'order': [1, 1], 'constant': True}
    garch_spec = {'type': 'GARCH', 'order': [1, 1]}
    egarch_spec = {'type': 'EGARCH', 'order': [1, 1]}

    # Create parameter dictionaries with estimated values
    arma_params = {'const': 0.1, 'ar_1': 0.5, 'ma_1': 0.2}
    garch_params = {'omega': 0.01, 'alpha_1': 0.1, 'beta_1': 0.8}
    egarch_params = {'omega': 0.01, 'alpha_1': 0.1, 'gamma_1': 0.05, 'beta_1': 0.8}

    # Call print_model_equation with test specifications and parameters
    arma_equation = print_model_equation(arma_spec, arma_params, include_values=True)
    garch_equation = print_model_equation(garch_spec, garch_params, include_values=True)
    egarch_equation = print_model_equation(egarch_spec, egarch_params, include_values=True)

    # Verify equation structures match expected patterns
    assert "y_t = 0.1000 + 0.5000 y_{t-1} + 0.2000 \u03b5_{t-1} + \u03b5_t" in arma_equation
    assert "\u03c3\u00b2_t = \u03c9 + \u03b1_1 \u03b5\u00b2_{t-1} + \u03b2_1 \u03c3\u00b2_{t-1}" in garch_equation
    assert "ln(\u03c3\u00b2_t) = \u03c9 + \u03b1_1 |\u03b5_{t-1}/\u03c3_{t-1}| + \u03b3_1 (\u03b5_{t-1}/\u03c3_{t-1}) + \u03b2_1 ln(\u03c3\u00b2_{t-1})" in egarch_equation

    # Check parameter value inclusion when include_values=True
    arma_equation_no_values = print_model_equation(arma_spec)
    assert "y_t = c + \u03c6_1 y_{t-1} + \u03b8_1 \u03b5_{t-1} + \u03b5_t" in arma_equation_no_values

    # Test without parameter values when include_values=False
    garch_equation_no_values = print_model_equation(garch_spec)
    assert "\u03c3\u00b2_t = \u03c9 + \u03b1_1 \u03b5\u00b2_{t-1} + \u03b2_1 \u03c3\u00b2_{t-1}" in garch_equation_no_values

    # Test error handling with invalid model specifications
    with pytest.raises(ValueError):
        print_model_equation({'type': 'Invalid'})


def test_format_matrix():
    """
    Tests the format_matrix function for formatting matrices as readable tables
    """
    # Create test numpy arrays of different shapes
    matrix1 = np.array([[1, 2], [3, 4]])
    matrix2 = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])

    # Call format_matrix with test arrays
    table_str1 = format_matrix(matrix1)
    table_str2 = format_matrix(matrix2)

    # Verify matrix formatting matches expected pattern
    assert "|   1 |   2 |" in table_str1
    assert "|   3 |   4 |" in table_str1
    assert "|   0.1 |   0.2 |   0.3 |" in table_str2
    assert "|   0.4 |   0.5 |   0.6 |" in table_str2
    assert "|   0.7 |   0.8 |   0.9 |" in table_str2

    # Check row and column label inclusion when provided
    row_labels = ['Row1', 'Row2']
    col_labels = ['Col1', 'Col2']
    table_str_labels = format_matrix(matrix1, row_labels=row_labels, col_labels=col_labels)
    assert "Col1    Col2" in table_str_labels
    assert "Row1       1       2" in table_str_labels
    assert "Row2       3       4" in table_str_labels

    # Verify title inclusion when specified
    title = "Correlation Matrix"
    table_str_title = format_matrix(matrix1, title=title)
    assert title in table_str_title

    # Test different float_format options
    float_format = "%.1f"
    table_str_custom_format = format_matrix(matrix1, float_format=float_format)
    assert "|   1.0 |   2.0 |" in table_str_custom_format

    # Test error handling with invalid dimensions or mismatched labels
    with pytest.raises(ValueError):
        format_matrix(np.array([1, 2, 3]))
    with pytest.raises(ValueError):
        format_matrix(matrix1, row_labels=['Row1'], col_labels=['Col1', 'Col2'])


def test_format_statistical_test():
    """
    Tests the format_statistical_test function for formatting statistical test results
    """
    # Create test statistical test parameters (name, statistic, p-value, etc.)
    test_name = "Jarque-Bera Test"
    statistic = 1.5
    p_value = 0.20

    # Call format_statistical_test with test parameters
    test_str = format_statistical_test(test_name, statistic, p_value)

    # Verify test result formatting matches expected pattern
    assert "Jarque-Bera Test: Statistic = 1.5000, p-value = 0.2000" in test_str
    assert "Interpretation: Not significant at 5% level" in test_str

    # Check significance stars for different p-value thresholds
    test_str_significant = format_statistical_test(test_name, statistic, 0.005)
    assert "p-value = 0.0050***" in test_str_significant
    assert "Interpretation: Significant at 5% level" in test_str_significant

    # Verify interpretation inclusion when provided
    interpretation = "Test is not significant"
    test_str_interpretation = format_statistical_test(test_name, statistic, p_value, interpretation=interpretation)
    assert "Interpretation: Test is not significant" in test_str_interpretation

    # Check critical value inclusion when provided
    critical_value = 3.84
    test_str_critical = format_statistical_test(test_name, statistic, p_value, critical_value=critical_value)
    assert ", Critical value: 3.8400" in test_str_critical

    # Test different float_format options
    float_format = "%.1f"
    test_str_custom_format = format_statistical_test(test_name, statistic, p_value, float_format=float_format)
    assert "Statistic = 1.5, p-value = 0.2" in test_str_custom_format

    # Test error handling with invalid input
    with pytest.raises(TypeError):
        format_statistical_test("invalid input", statistic, p_value)


def test_format_bootstrap_results():
    """
    Tests the format_bootstrap_results function for formatting bootstrap analysis results
    """
    # Create test bootstrap results dictionary with parameter estimates and confidence intervals
    bootstrap_results = {
        'estimates': {'param1': 0.5, 'param2': 0.2},
        'ci_lower': {'param1': 0.4, 'param2': 0.1},
        'ci_upper': {'param1': 0.6, 'param2': 0.3},
        'n_bootstrap': 1000
    }

    # Call format_bootstrap_results with test results
    results_str = format_bootstrap_results(bootstrap_results)

    # Verify results formatting matches expected pattern
    assert "Bootstrap Results (1000 replications, 95% Confidence Intervals)" in results_str
    assert "Parameter  Estimate    95% CI" in results_str
    assert "| param1     0.5000      [0.4000, 0.6000]" in results_str
    assert "| param2     0.2000      [0.1000, 0.3000]" in results_str

    # Check confidence interval formatting for different alpha levels
    results_str_alpha = format_bootstrap_results(bootstrap_results, alpha=0.10)
    assert "Bootstrap Results (1000 replications, 90% Confidence Intervals)" in results_str_alpha

    # Test different float_format options
    float_format = "%.1f"
    results_str_custom_format = format_bootstrap_results(bootstrap_results, float_format=float_format)
    assert "| param1     0.5      [0.4, 0.6]" in results_str_custom_format

    # Test error handling with invalid input
    with pytest.raises(ValueError):
        format_bootstrap_results("invalid input")


def test_generate_log_message():
    """
    Tests the generate_log_message function for creating structured log messages
    """
    # Create test messages and context dictionaries
    message = "Model estimation completed"
    context = {'model': 'ARIMA', 'likelihood': -100.0}

    # Call generate_log_message with test inputs
    log_msg = generate_log_message(message, context)

    # Verify message formatting matches expected pattern
    assert message in log_msg
    assert "model=ARIMA, likelihood=-100.0" in log_msg

    # Check context inclusion in formatted message
    log_msg_no_context = generate_log_message(message)
    assert message in log_msg_no_context

    # Test different log level options
    log_msg_warning = generate_log_message(message, level='WARNING')
    assert "WARNING: Model estimation completed" in log_msg_warning

    # Test error handling with invalid input
    with pytest.raises(TypeError):
        generate_log_message(123)


def test_log_model_estimation():
    """
    Tests the log_model_estimation function for logging model estimation progress
    """
    # Set up test logger with StringIO capture
    logger = logging.getLogger('test_logger')
    logger.setLevel(logging.DEBUG)
    output = io.StringIO()
    handler = logging.StreamHandler(output)
    logger.addHandler(handler)

    # Create test model parameters
    model_type = "ARIMA"
    params = {'ar_1': 0.5, 'ma_1': 0.2}

    # Call log_model_estimation with test parameters
    log_model_estimation(model_type, params, likelihood=-100.0, iteration=10, converged=False)

    # Verify log message content matches expected pattern
    log_contents = output.getvalue()
    assert "ARIMA estimation [iteration 10]: ar_1=0.5, ma_1=0.2 [likelihood=-100.000000, iteration=10, converged=False]" in log_contents

    # Check iteration inclusion when provided
    log_model_estimation(model_type, params, likelihood=-100.0)
    log_contents_no_iteration = output.getvalue()
    assert "ARIMA estimation: ar_1=0.5, ma_1=0.2 [likelihood=-100.000000]" in log_contents_no_iteration

    # Verify likelihood value inclusion when provided
    log_model_estimation(model_type, params)
    log_contents_no_likelihood = output.getvalue()
    assert "ARIMA estimation: ar_1=0.5, ma_1=0.2" in log_contents_no_likelihood

    # Test convergence status inclusion when provided
    log_model_estimation(model_type, params, converged=True)
    log_contents_converged = output.getvalue()
    assert "ARIMA estimation: ar_1=0.5, ma_1=0.2 [converged=True]" in log_contents_converged

    # Test log level selection based on parameters
    log_model_estimation(model_type, params, iteration=1)
    log_contents_debug = output.getvalue()
    assert "ARIMA estimation [iteration 1]: ar_1=0.5, ma_1=0.2" in log_contents_debug

    # Clean up logger
    logger.removeHandler(handler)


def test_format_warning():
    """
    Tests the format_warning function for consistent warning message formatting
    """
    # Create test warning messages and context dictionaries
    message = "Data contains missing values"
    context = {'column': 'returns', 'method': 'ffill'}

    # Call format_warning with test inputs
    warning_msg = format_warning(message, context)

    # Verify warning prefix inclusion
    assert "WARNING: Data contains missing values" in warning_msg

    # Check context inclusion in formatted message
    assert "column=returns, method=ffill" in warning_msg

    # Verify consistent styling of warning messages
    assert "! WARNING: Data contains missing values [column=returns, method=ffill] !" in warning_msg

    # Test error handling with invalid input
    with pytest.raises(TypeError):
        format_warning(123)


def test_format_error():
    """
    Tests the format_error function for consistent error message formatting
    """
    # Create test error messages, context dictionaries, and exception objects
    message = "Model estimation failed"
    context = {'model': 'GARCH', 'iteration': 50}
    exception = ValueError("Invalid parameters")

    # Call format_error with test inputs
    error_msg = format_error(message, context, exception)

    # Verify error prefix inclusion
    assert "ERROR: Model estimation failed" in error_msg

    # Check context inclusion in formatted message
    assert "model=GARCH, iteration=50" in error_msg

    # Verify exception information inclusion when provided
    assert "Exception: ValueError: Invalid parameters" in error_msg

    # Test consistent styling of error messages
    assert "!!! ERROR: Model estimation failed [model=GARCH, iteration=50]\nException: ValueError: Invalid parameters !!!" in error_msg

    # Test error handling with invalid input
    with pytest.raises(TypeError):
        format_error(123)


def test_setup_logger():
    """
    Tests the setup_logger function for configuring logging with appropriate formatting
    """
    # Call setup_logger with test name
    logger = setup_logger('test_logger')

    # Verify logger has correct name
    assert logger.name == 'test_logger'

    # Check default log level when not specified
    assert logger.level == logging.INFO

    # Verify custom log level when specified
    logger_debug = setup_logger('test_logger_debug', level='DEBUG')
    assert logger_debug.level == logging.DEBUG

    # Test custom format string usage
    format_string = '%(levelname)s - %(message)s'
    logger_custom_format = setup_logger('test_logger_format', format_string=format_string)
    handler = logger_custom_format.handlers[0]
    assert handler.formatter._fmt == format_string

    # Check file handler creation when log_file is specified
    log_file = 'test.log'
    logger_file = setup_logger('test_logger_file', log_file=log_file)
    assert any(isinstance(h, logging.FileHandler) for h in logger_file.handlers)

    # Verify handlers are not duplicated when called multiple times
    logger_multiple_calls = setup_logger('test_logger_multiple')
    num_handlers = len(logger_multiple_calls.handlers)
    setup_logger('test_logger_multiple')
    assert len(logger_multiple_calls.handlers) == num_handlers

    # Test error handling with invalid input
    with pytest.raises(TypeError):
        setup_logger(123)