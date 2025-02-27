"""
Test cases for PACFPlot component that validates the functionality of the 
partial autocorrelation function visualization widget used in the MFE Toolbox UI.

These tests verify the PACF visualization capabilities, input validation, and
asynchronous update functionality of the component.
"""

import pytest
import pytest_asyncio
import numpy as np
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QWidget
import matplotlib.pyplot as plt

# Import components under test
from mfe.ui.plots.pacf_plot import PACFPlot, calculate_pacf, create_pacf_figure
from mfe.ui.plot_widgets import PACFPlotWidget

# Test constants
TEST_NLAGS = 20
TEST_ALPHA = 0.05
DEFAULT_PLOT_PARAMS = {'title': 'Test PACF Plot', 'bar_color': 'blue', 'grid': True}


@pytest.fixture
def sample_data():
    """Generate time series sample data for testing."""
    np.random.seed(42)  # Ensure reproducible results
    # Generate AR(1) process with phi=0.7 - good for PACF visualization
    n = 500
    data = np.zeros(n)
    data[0] = np.random.normal(0, 1)
    for t in range(1, n):
        data[t] = 0.7 * data[t-1] + np.random.normal(0, 1)
    return data


@pytest.fixture
def qapp():
    """Create a Qt application for testing UI components."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app
    # Ensure we don't call quit() to avoid stopping the current test session


def test_calculate_pacf(sample_data):
    """Tests the PACF calculation utility function."""
    # Call calculate_pacf with sample data
    pacf_values, conf_intervals, lags = calculate_pacf(sample_data, nlags=TEST_NLAGS, alpha=TEST_ALPHA)
    
    # Verify output types
    assert isinstance(pacf_values, np.ndarray)
    assert isinstance(conf_intervals, np.ndarray)
    assert isinstance(lags, np.ndarray)
    
    # Verify dimensions
    assert len(pacf_values) == TEST_NLAGS + 1  # +1 for lag 0
    assert conf_intervals.shape[0] == 2  # Upper and lower bounds
    assert conf_intervals.shape[1] == TEST_NLAGS + 1
    assert len(lags) == TEST_NLAGS + 1
    
    # Verify lag 0 autocorrelation is always 1
    assert np.isclose(pacf_values[0], 1.0)
    
    # Verify lag values are as expected
    assert np.array_equal(lags, np.arange(TEST_NLAGS + 1))
    
    # Verify confidence intervals are symmetric around 0
    assert np.allclose(conf_intervals[0, :], -conf_intervals[1, :])


def test_create_pacf_figure(sample_data):
    """Tests the creation of a PACF figure."""
    # Call create_pacf_figure with sample data and default parameters
    fig = create_pacf_figure(sample_data, nlags=TEST_NLAGS, alpha=TEST_ALPHA, plot_params=DEFAULT_PLOT_PARAMS)
    
    # Verify figure is created correctly
    assert isinstance(fig, plt.Figure)
    
    # Check that figure has exactly one axis
    assert len(fig.get_axes()) == 1
    
    # Get the axis for further inspection
    ax = fig.get_axes()[0]
    
    # Check that title matches what we set
    assert ax.get_title() == DEFAULT_PLOT_PARAMS['title']
    
    # Check that the grid is enabled as specified
    assert ax.get_grid()
    
    # Check that x and y labels are as expected
    assert ax.get_xlabel() == 'Lag'
    assert ax.get_ylabel() == 'Partial Autocorrelation'
    
    # Check that there are bar patches (one for each lag)
    assert len(ax.patches) == TEST_NLAGS + 1
    
    # Check that there are confidence interval lines and zero line
    lines = ax.get_lines()
    assert len(lines) == 3  # 2 for confidence intervals and 1 for zero line
    
    plt.close(fig)  # Clean up


def test_pacf_plot_initialization(qapp):
    """Tests the initialization of the PACFPlot widget."""
    # Create a new PACFPlot instance
    plot = PACFPlot()
    
    # Verify default properties are correctly set
    assert plot._data is None
    assert plot._nlags == 40  # Default nlags value
    assert plot._alpha == 0.05  # Default alpha value
    assert isinstance(plot._plot_params, dict)
    assert not plot._initialized
    
    # Verify the widget's parent is properly set to None
    assert plot.parent() is None
    
    # Clean up
    plot.deleteLater()


def test_pacf_plot_set_data(qapp, sample_data):
    """Tests setting data on the PACFPlot widget."""
    # Create a new PACFPlot instance
    plot = PACFPlot()
    
    # Set up a slot to verify data_changed signal is emitted
    signal_received = False
    def on_data_changed():
        nonlocal signal_received
        signal_received = True
    
    plot.data_changed.connect(on_data_changed)
    
    # Call set_data with sample_data
    plot.set_data(sample_data)
    
    # Verify data was correctly stored
    assert plot._data is not None
    assert np.array_equal(plot._data, sample_data)
    
    # Check that signal was emitted
    assert signal_received
    
    # Clean up
    plot.deleteLater()


def test_pacf_plot_set_parameters(qapp, sample_data):
    """Tests setting PACF parameters on the widget."""
    # Create a new PACFPlot instance with sample data
    plot = PACFPlot()
    plot.set_data(sample_data)
    
    # Set custom nlags value
    custom_nlags = 15
    plot.set_nlags(custom_nlags)
    assert plot._nlags == custom_nlags
    
    # Set custom alpha value
    custom_alpha = 0.01
    plot.set_alpha(custom_alpha)
    assert plot._alpha == custom_alpha
    
    # Set custom plot parameters
    custom_params = {
        'title': 'Custom PACF Plot',
        'grid': False,
        'bar_color': 'red'
    }
    plot.set_plot_params(custom_params)
    
    # Verify all parameters are correctly stored
    for key, value in custom_params.items():
        assert plot._plot_params[key] == value
    
    # Clean up
    plot.deleteLater()


def test_pacf_plot_clear(qapp, sample_data):
    """Tests clearing the PACFPlot widget."""
    # Create a new PACFPlot instance with sample data
    plot = PACFPlot()
    plot.set_data(sample_data)
    
    # Update the plot to initialize it
    plot.update_plot()
    assert plot._initialized
    
    # Call clear() method
    plot.clear()
    
    # Verify data has been cleared
    assert plot._data is None
    assert not plot._initialized
    
    # Clean up
    plot.deleteLater()


@pytest.mark.asyncio
async def test_async_update_plot(qapp, sample_data, event_loop):
    """Tests asynchronous plot updates."""
    # Create a new PACFPlot instance with sample data
    plot = PACFPlot()
    plot.set_data(sample_data)
    
    # Await async_update_plot() method
    await plot.async_update_plot()
    
    # Verify plot has been updated
    assert plot._initialized
    
    # Check that figure contains expected elements
    figure = plot.get_figure()
    assert isinstance(figure, plt.Figure)
    
    # Get the axis for further inspection
    if len(figure.get_axes()) > 0:
        ax = figure.get_axes()[0]
        
        # Check that labels are set
        assert ax.get_xlabel() == 'Lag'
        assert ax.get_ylabel() == 'Partial Autocorrelation'
        
        # Check that there are bar patches (one for each lag)
        assert len(ax.patches) > 0
    
    # Clean up
    plot.deleteLater()


def test_pacf_plot_widget_integration(qapp, sample_data):
    """Tests integration with PACFPlotWidget."""
    # Create a new PACFPlotWidget instance
    widget = PACFPlotWidget()
    
    # Set data using set_data method
    widget.set_data(sample_data)
    
    # Set parameters using available methods
    widget.set_nlags(15)
    widget.set_alpha(0.01)
    widget.set_method("ywmle")  # PACFPlotWidget has this additional method
    
    # Call update_plot method
    widget.update_plot()
    
    # Verify widget contains the expected figure
    figure = widget.get_figure()
    assert isinstance(figure, plt.Figure)
    
    # Clean up
    widget.deleteLater()


@pytest.mark.asyncio
async def test_async_pacf_plot_widget(qapp, sample_data, event_loop):
    """Tests asynchronous updates in PACFPlotWidget."""
    # Create a new PACFPlotWidget instance
    widget = PACFPlotWidget()
    
    # Set data using set_data method
    widget.set_data(sample_data)
    
    # Set parameters for the plot
    widget.set_nlags(15)
    widget.set_alpha(0.01)
    widget.set_method("ywmle")
    
    # Set custom plot parameters
    widget.set_plot_params({
        'title': 'Async PACF Test',
        'grid': True,
        'bar_color': 'green'
    })
    
    # Await async_update_plot method
    await widget.async_update_plot()
    
    # Verify widget shows the expected plot
    figure = widget.get_figure()
    assert isinstance(figure, plt.Figure)
    
    # Check that the plot was initialized
    assert widget._initialized
    
    # Clean up
    widget.deleteLater()


def test_input_validation(qapp):
    """Tests input validation in PACFPlot."""
    # Create a new PACFPlot instance
    plot = PACFPlot()
    
    # Test invalid data types
    with pytest.raises(ValueError):
        plot.set_data(None)
    
    with pytest.raises(ValueError):
        plot.set_data(np.array([np.nan, 1, 2]))
    
    with pytest.raises(ValueError):
        plot.set_data(np.array([np.inf, 1, 2]))
    
    # Test invalid nlags
    with pytest.raises(ValueError):
        plot.set_nlags(-1)
    
    with pytest.raises(ValueError):
        plot.set_nlags(0)
    
    with pytest.raises(ValueError):
        plot.set_nlags("invalid")
    
    # Test invalid alpha
    with pytest.raises(ValueError):
        plot.set_alpha(-0.1)
    
    with pytest.raises(ValueError):
        plot.set_alpha(1.1)
    
    with pytest.raises(ValueError):
        plot.set_alpha("invalid")
    
    # Test invalid plot_params
    with pytest.raises(TypeError):
        plot.set_plot_params("invalid")
    
    # Clean up
    plot.deleteLater()