"""
MFE Toolbox - Test Module for Residual Plot Component

This module tests the ResidualPlot widget's capabilities including statistical analysis
of residuals, visualization, asynchronous updates, and integration with PyQt6.
"""

import pytest  # pytest 7.4.3
import numpy as np  # numpy 1.26.3
import matplotlib.figure  # matplotlib 3.7.1
from PyQt6.QtTest import QTest  # PyQt6 6.6.1
import pytest_asyncio  # pytest-asyncio 0.21.1
import os  # Python standard library
import tempfile  # Python standard library
import scipy.stats as stats  # scipy 1.11.4

from mfe.ui.plots.residual_plot import (
    ResidualPlot,
    analyze_residuals,
    create_residual_figure,
    DEFAULT_PLOT_PARAMS
)

# Constants for testing
TEST_DATA_LENGTH = 100
MODEL_SIZES = [(1, 1), (2, 2), (3, 0), (0, 3)]


def generate_test_data(length=TEST_DATA_LENGTH, model_size=(1, 1)):
    """
    Creates synthetic time series data with model residuals and fitted values for residual plot testing.
    
    Parameters
    ----------
    length : int
        Length of the test data series
    model_size : tuple
        Size of the model (p, q) for AR and MA components
        
    Returns
    -------
    tuple
        Tuple of (residuals, fitted_values) arrays for testing
    """
    # Set random seed for reproducible tests
    np.random.seed(42)
    
    # Generate simulated AR process
    ar_params = np.array([0.7]) if model_size[0] > 0 else np.array([])
    ma_params = np.array([0.3]) if model_size[1] > 0 else np.array([])
    
    # Generate innovations (residuals)
    residuals = np.random.normal(0, 1, length)
    
    # Generate fitted values through a simple model
    y = np.zeros(length)
    
    # Fill initial values
    for i in range(max(len(ar_params), len(ma_params))):
        y[i] = residuals[i]
    
    # Generate ARMA process
    for t in range(max(len(ar_params), len(ma_params)), length):
        ar_component = np.sum(ar_params * y[t-len(ar_params):t][::-1]) if len(ar_params) > 0 else 0
        ma_component = np.sum(ma_params * residuals[t-len(ma_params):t][::-1]) if len(ma_params) > 0 else 0
        y[t] = ar_component + ma_component + residuals[t]
    
    # Fitted values are observed values minus residuals
    fitted_values = y - residuals
    
    return residuals, fitted_values


def check_residual_stats(stats):
    """
    Validates the statistical metrics returned by analyze_residuals function.
    
    Parameters
    ----------
    stats : dict
        Dictionary of residual statistics
        
    Returns
    -------
    bool
        True if statistics are valid, False otherwise
    """
    # Check if stats is a dictionary
    if not isinstance(stats, dict):
        return False
    
    # Check required keys
    required_keys = ['mean', 'std', 'min', 'max', 'jarque_bera', 'durbin_watson']
    if not all(key in stats for key in required_keys):
        return False
    
    # Check types of values
    if not isinstance(stats['mean'], float):
        return False
    if not isinstance(stats['std'], float):
        return False
    if not isinstance(stats['min'], float):
        return False
    if not isinstance(stats['max'], float):
        return False
    if not isinstance(stats['durbin_watson'], float):
        return False
    
    # Check jarque_bera is a tuple with two elements
    if not isinstance(stats['jarque_bera'], tuple) or len(stats['jarque_bera']) != 2:
        return False
    
    # Check ranges of statistical values
    if stats['std'] < 0:  # Standard deviation must be non-negative
        return False
    if stats['min'] > stats['max']:  # Min cannot be greater than max
        return False
    if stats['durbin_watson'] < 0 or stats['durbin_watson'] > 4:  # DW stat range
        return False
    
    # Check p-value for Jarque-Bera test
    if not 0 <= stats['jarque_bera'][1] <= 1:  # p-value must be in [0, 1]
        return False
    
    return True


@pytest.mark.parametrize('model_size', MODEL_SIZES)
def test_analyze_residuals_function(model_size):
    """
    Tests if the analyze_residuals function correctly computes residual statistics.
    """
    # Generate test data
    residuals, _ = generate_test_data(model_size=model_size)
    
    # Call analyze_residuals function
    result = analyze_residuals(residuals)
    
    # Verify function returns a dictionary with proper structure
    assert isinstance(result, dict)
    assert check_residual_stats(result)
    
    # Verify key statistics are correctly computed
    assert abs(result['mean'] - np.mean(residuals)) < 1e-10
    assert abs(result['std'] - np.std(residuals)) < 1e-10
    assert abs(result['min'] - np.min(residuals)) < 1e-10
    assert abs(result['max'] - np.max(residuals)) < 1e-10
    
    # Verify Jarque-Bera calculation
    jb_stat, jb_pval = stats.jarque_bera(residuals)
    assert abs(result['jarque_bera'][0] - jb_stat) < 1e-10
    assert abs(result['jarque_bera'][1] - jb_pval) < 1e-10
    
    # Verify Durbin-Watson calculation
    dw_stat = np.sum(np.diff(residuals) ** 2) / np.sum(residuals ** 2)
    assert abs(result['durbin_watson'] - dw_stat) < 1e-10
    
    # Test with special case: all zeros
    zero_residuals = np.zeros(10)
    zero_result = analyze_residuals(zero_residuals)
    assert zero_result['mean'] == 0
    assert zero_result['std'] == 0
    
    # Test with special case: single value
    single_residual = np.array([1.0])
    with pytest.raises(ValueError):
        analyze_residuals(single_residual)  # Should raise error (not enough data)
    
    # Test with invalid input
    with pytest.raises(ValueError):
        analyze_residuals("not an array")
    with pytest.raises(ValueError):
        analyze_residuals(None)


@pytest.mark.parametrize('model_size', MODEL_SIZES)
def test_create_residual_figure(model_size):
    """
    Tests if the create_residual_figure function generates a correct matplotlib figure.
    """
    # Generate test data
    residuals, fitted_values = generate_test_data(model_size=model_size)
    
    # Create figure with default parameters
    fig = create_residual_figure(residuals, fitted_values, DEFAULT_PLOT_PARAMS)
    
    # Verify the function returns a matplotlib Figure object
    assert isinstance(fig, matplotlib.figure.Figure)
    
    # Verify the figure contains expected subplots
    assert len(fig.axes) == 3  # Should have 3 subplots
    
    # Test with custom plot parameters
    custom_params = DEFAULT_PLOT_PARAMS.copy()
    custom_params.update({
        'title': 'Custom Title',
        'residual_color': 'red',
        'hist_bins': 15,
        'alpha': 0.5
    })
    
    custom_fig = create_residual_figure(residuals, fitted_values, custom_params)
    assert isinstance(custom_fig, matplotlib.figure.Figure)
    
    # Test with dimension mismatch
    mismatched_fitted = np.zeros(len(residuals) + 5)
    with pytest.raises(ValueError):
        create_residual_figure(residuals, mismatched_fitted, DEFAULT_PLOT_PARAMS)
    
    # Test with invalid inputs
    with pytest.raises(ValueError):
        create_residual_figure(None, fitted_values, DEFAULT_PLOT_PARAMS)
    with pytest.raises(ValueError):
        create_residual_figure(residuals, None, DEFAULT_PLOT_PARAMS)
    with pytest.raises(TypeError):
        create_residual_figure(residuals, fitted_values, "not a dict")


@pytest.mark.gui
def test_residual_plot_widget_initialization(qtbot):
    """
    Tests if the ResidualPlot widget initializes correctly.
    """
    # Create widget
    widget = ResidualPlot()
    qtbot.addWidget(widget)
    
    # Verify widget is created without errors
    assert widget is not None
    
    # Verify widget has expected initial state
    assert widget._residuals is None
    assert widget._fitted_values is None
    assert widget._plot_params == DEFAULT_PLOT_PARAMS
    assert not widget._initialized
    
    # Verify widget layout and components
    assert widget.layout() is not None
    assert widget._plot_widget is not None
    
    # Verify default size is reasonable
    assert widget.sizeHint().width() > 0
    assert widget.sizeHint().height() > 0


@pytest.mark.gui
@pytest.mark.parametrize('model_size', MODEL_SIZES)
def test_residual_plot_widget_set_residuals(qtbot, model_size):
    """
    Tests if the set_residuals method properly updates the widget with residual data.
    """
    # Create widget and test data
    widget = ResidualPlot()
    qtbot.addWidget(widget)
    residuals, _ = generate_test_data(model_size=model_size)
    
    # Call set_residuals
    widget.set_residuals(residuals)
    
    # Verify residuals are stored
    assert widget._residuals is not None
    assert np.array_equal(widget._residuals, residuals)
    
    # Verify residual stats are calculated
    assert widget._residual_stats is not None
    assert check_residual_stats(widget._residual_stats)
    
    # Test with invalid inputs
    with pytest.raises(ValueError):
        widget.set_residuals(None)
    with pytest.raises(ValueError):
        widget.set_residuals("not an array")
    
    # Test with unusual but valid input: empty array
    empty_array = np.array([])
    with pytest.raises(ValueError):  # Should raise error (not enough data)
        widget.set_residuals(empty_array)


@pytest.mark.gui
@pytest.mark.parametrize('model_size', MODEL_SIZES)
def test_residual_plot_widget_set_fitted_values(qtbot, model_size):
    """
    Tests if the set_fitted_values method properly updates the widget with fitted value data.
    """
    # Create widget and test data
    widget = ResidualPlot()
    qtbot.addWidget(widget)
    residuals, fitted_values = generate_test_data(model_size=model_size)
    
    # Set residuals first
    widget.set_residuals(residuals)
    
    # Call set_fitted_values
    widget.set_fitted_values(fitted_values)
    
    # Verify fitted values are stored
    assert widget._fitted_values is not None
    assert np.array_equal(widget._fitted_values, fitted_values)
    
    # Test with dimension mismatch
    mismatched_fitted = np.zeros(len(residuals) + 5)
    with pytest.raises(ValueError):
        widget.set_fitted_values(mismatched_fitted)
    
    # Test setting fitted values before residuals
    new_widget = ResidualPlot()
    qtbot.addWidget(new_widget)
    new_widget.set_fitted_values(fitted_values)  # Should work without residuals
    assert np.array_equal(new_widget._fitted_values, fitted_values)
    
    # Test with invalid inputs
    with pytest.raises(ValueError):
        widget.set_fitted_values(None)
    with pytest.raises(ValueError):
        widget.set_fitted_values("not an array")


@pytest.mark.gui
def test_residual_plot_widget_set_plot_params(qtbot):
    """
    Tests if set_plot_params correctly updates widget configuration.
    """
    # Create widget
    widget = ResidualPlot()
    qtbot.addWidget(widget)
    
    # Default parameters
    assert widget._plot_params == DEFAULT_PLOT_PARAMS
    
    # Update with custom parameters
    custom_params = {
        'title': 'Custom Title',
        'residual_color': 'red',
        'hist_bins': 15,
        'alpha': 0.5
    }
    
    widget.set_plot_params(custom_params)
    
    # Verify parameters were updated
    assert widget._plot_params['title'] == 'Custom Title'
    assert widget._plot_params['residual_color'] == 'red'
    assert widget._plot_params['hist_bins'] == 15
    assert widget._plot_params['alpha'] == 0.5
    
    # Verify parameters not in custom_params remain unchanged
    assert widget._plot_params['grid'] == DEFAULT_PLOT_PARAMS['grid']
    
    # Test with invalid input
    with pytest.raises(TypeError):
        widget.set_plot_params("not a dict")
    
    # Test with empty dict (valid, but no changes)
    widget.set_plot_params({})
    assert widget._plot_params['title'] == 'Custom Title'  # Still has previous values


@pytest.mark.gui
@pytest.mark.parametrize('model_size', MODEL_SIZES)
def test_residual_plot_get_residual_stats(qtbot, model_size):
    """
    Tests if get_residual_stats returns correct statistical analysis.
    """
    # Create widget and test data
    widget = ResidualPlot()
    qtbot.addWidget(widget)
    residuals, _ = generate_test_data(model_size=model_size)
    
    # Test before setting residuals
    empty_stats = widget.get_residual_stats()
    assert isinstance(empty_stats, dict)
    assert len(empty_stats) == 0
    
    # Set residuals and get stats
    widget.set_residuals(residuals)
    stats = widget.get_residual_stats()
    
    # Verify stats dictionary
    assert isinstance(stats, dict)
    assert check_residual_stats(stats)
    
    # Verify is a copy, not a reference
    stats_copy = widget.get_residual_stats()
    stats_copy['mean'] = 999.999
    assert widget.get_residual_stats()['mean'] != 999.999
    
    # Verify identical to stats calculated directly
    direct_stats = analyze_residuals(residuals)
    for key in direct_stats:
        if isinstance(direct_stats[key], tuple):
            assert direct_stats[key][0] == stats[key][0]
            assert direct_stats[key][1] == stats[key][1]
        else:
            assert direct_stats[key] == stats[key]


@pytest.mark.gui
@pytest.mark.parametrize('model_size', MODEL_SIZES)
def test_residual_plot_update(qtbot, model_size):
    """
    Tests if the update_plot method correctly refreshes the plot display.
    """
    # Create widget and test data
    widget = ResidualPlot()
    qtbot.addWidget(widget)
    residuals, fitted_values = generate_test_data(model_size=model_size)
    
    # Update without data should not raise errors but log a warning
    widget.update_plot()  # Should not crash
    assert not widget._initialized
    
    # Set data and update plot
    widget.set_residuals(residuals)
    widget.set_fitted_values(fitted_values)
    widget.update_plot()
    
    # Verify initialized flag is set
    assert widget._initialized
    
    # Test update with only residuals (no fitted values)
    new_widget = ResidualPlot()
    qtbot.addWidget(new_widget)
    new_widget.set_residuals(residuals)
    new_widget.update_plot()
    assert new_widget._initialized
    
    # Test multiple updates
    widget.update_plot()  # Should not crash


@pytest.mark.gui
def test_residual_plot_save_figure(qtbot, tmp_path):
    """
    Tests if the save_figure method correctly saves the plot to a file.
    """
    # Create widget and test data
    widget = ResidualPlot()
    qtbot.addWidget(widget)
    residuals, fitted_values = generate_test_data()
    
    # Try to save before initialization
    filepath = os.path.join(tmp_path, "uninit_test.png")
    assert not widget.save_figure(filepath)
    assert not os.path.exists(filepath)
    
    # Set data and update plot
    widget.set_residuals(residuals)
    widget.set_fitted_values(fitted_values)
    widget.update_plot()
    
    # Save figure
    filepath = os.path.join(tmp_path, "test.png")
    success = widget.save_figure(filepath)
    
    # Verify save was successful
    assert success
    assert os.path.exists(filepath)
    
    # Check that file is a valid image (at least non-empty)
    assert os.path.getsize(filepath) > 0
    
    # Test save with invalid path
    invalid_filepath = "/invalid/path/test.png"
    assert not widget.save_figure(invalid_filepath)
    
    # Test save with different formats
    pdf_filepath = os.path.join(tmp_path, "test.pdf")
    assert widget.save_figure(pdf_filepath)
    assert os.path.exists(pdf_filepath)
    
    svg_filepath = os.path.join(tmp_path, "test.svg")
    assert widget.save_figure(svg_filepath)
    assert os.path.exists(svg_filepath)


@pytest.mark.gui
def test_residual_plot_clear(qtbot):
    """
    Tests if the clear method correctly resets the widget state.
    """
    # Create widget and test data
    widget = ResidualPlot()
    qtbot.addWidget(widget)
    residuals, fitted_values = generate_test_data()
    
    # Set data and update plot
    widget.set_residuals(residuals)
    widget.set_fitted_values(fitted_values)
    widget.update_plot()
    
    # Verify widget is initialized
    assert widget._initialized
    assert widget._residuals is not None
    assert widget._fitted_values is not None
    assert len(widget._residual_stats) > 0
    
    # Clear widget
    widget.clear()
    
    # Verify widget state is reset
    assert not widget._initialized
    assert widget._residuals is None
    assert widget._fitted_values is None
    assert len(widget._residual_stats) == 0
    
    # Test clearing already cleared widget
    widget.clear()  # Should not crash
    
    # Verify widget can be reused after clearing
    widget.set_residuals(residuals)
    widget.update_plot()
    assert widget._initialized


@pytest.mark.asyncio
@pytest.mark.gui
async def test_residual_plot_async_operations(qtbot):
    """
    Tests asynchronous behavior of the ResidualPlot widget.
    """
    # Create widget and test data
    widget = ResidualPlot()
    qtbot.addWidget(widget)
    residuals, fitted_values = generate_test_data()
    
    # Set data
    widget.set_residuals(residuals)
    widget.set_fitted_values(fitted_values)
    
    # Update asynchronously
    await widget.async_update_plot()
    
    # Verify widget is initialized
    assert widget._initialized
    
    # Try async update with no data
    empty_widget = ResidualPlot()
    qtbot.addWidget(empty_widget)
    await empty_widget.async_update_plot()  # Should not crash
    
    # Test clearing after async update
    widget.clear()
    assert not widget._initialized
    
    # Test multiple async updates
    widget.set_residuals(residuals)
    await widget.async_update_plot()
    await widget.async_update_plot()  # Should not crash
    assert widget._initialized