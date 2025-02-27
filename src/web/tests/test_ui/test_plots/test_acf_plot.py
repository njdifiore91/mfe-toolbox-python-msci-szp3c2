import pytest
import numpy as np  # version 1.26.3
from matplotlib.figure import Figure  # version 3.8.0
from PyQt6.QtTest import QTest  # version 6.6.1
import pytest_asyncio  # version 0.21.1
import os  # standard library
import tempfile  # standard library
import asyncio  # standard library

# Import components to test
from mfe.ui.plots.acf_plot import ACFPlot, calculate_acf, create_acf_figure

# Constants for testing
TEST_DATA_LENGTH = 100
DEFAULT_NLAGS = 20
DEFAULT_ALPHA = 0.05

def generate_test_data(length, ar_param):
    """
    Creates synthetic time series data for ACF plot testing
    
    Parameters
    ----------
    length : int
        Length of time series to generate
    ar_param : float
        AR(1) parameter for the synthetic data
    
    Returns
    -------
    np.ndarray
        Synthetic AR(1) time series for testing
    """
    # Set random seed for reproducible tests
    np.random.seed(42)
    
    # Generate random noise
    noise = np.random.randn(length)
    
    # Create an AR(1) process
    data = np.zeros(length)
    data[0] = noise[0]
    
    for i in range(1, length):
        data[i] = ar_param * data[i-1] + noise[i]
    
    return data

@pytest.mark.parametrize('nlags, alpha', [(10, 0.05), (20, 0.01), (40, 0.10)])
def test_calculate_acf_function(nlags, alpha):
    """
    Tests if the calculate_acf function correctly computes ACF values and confidence intervals
    """
    # Generate test data
    data = generate_test_data(TEST_DATA_LENGTH, ar_param=0.7)
    
    # Calculate ACF
    acf_vals, conf_intervals, lags = calculate_acf(data, nlags=nlags, alpha=alpha)
    
    # Verify the function returns a tuple with correct length
    assert isinstance(acf_vals, np.ndarray)
    assert isinstance(conf_intervals, np.ndarray)
    assert isinstance(lags, np.ndarray)
    
    # Verify ACF values are within expected range
    assert len(acf_vals) == nlags + 1  # +1 for lag 0
    assert acf_vals[0] == pytest.approx(1.0)  # First value should be 1.0
    assert all(-1.0 <= val <= 1.0 for val in acf_vals)  # Values between -1 and 1
    
    # Verify confidence intervals have correct dimensions
    assert conf_intervals.shape[0] == 2  # Lower and upper bounds
    assert conf_intervals.shape[1] == nlags + 1  # +1 for lag 0
    
    # Verify lags array has expected length
    assert len(lags) == nlags + 1  # +1 for lag 0
    assert list(lags) == list(range(nlags + 1))  # Should be [0, 1, 2, ..., nlags]

def test_create_acf_figure():
    """
    Tests if the create_acf_figure function generates a correct matplotlib figure
    """
    # Generate test data
    data = generate_test_data(TEST_DATA_LENGTH, ar_param=0.7)
    
    # Create ACF figure
    fig = create_acf_figure(data, nlags=DEFAULT_NLAGS, alpha=DEFAULT_ALPHA)
    
    # Verify the function returns a matplotlib Figure object
    assert isinstance(fig, Figure)
    
    # Verify the figure contains expected axes
    assert len(fig.axes) > 0
    ax = fig.axes[0]
    
    # Verify the plot contains expected elements
    assert ax.get_title() == 'Autocorrelation Function'
    assert ax.get_xlabel() == 'Lag'
    assert ax.get_ylabel() == 'Autocorrelation'
    
    # Check for bar elements representing ACF values
    bars = [c for c in ax.get_children() if hasattr(c, 'get_height')]
    assert len(bars) >= DEFAULT_NLAGS + 1

@pytest.mark.gui
def test_acf_plot_widget_initialization(qtbot):
    """
    Tests if the ACFPlot widget initializes correctly
    """
    # Create widget
    widget = ACFPlot()
    qtbot.addWidget(widget)
    
    # Verify widget creates without errors
    assert widget is not None
    
    # Verify widget has expected initial state
    assert widget._figure is None
    assert widget._data is None
    assert widget._nlags == 40  # Default value
    assert widget._alpha == 0.05  # Default value
    assert widget._initialized is False
    
    # Verify layout exists
    assert widget.layout() is not None

@pytest.mark.gui
def test_acf_plot_widget_set_data(qtbot):
    """
    Tests if the set_data method properly updates the widget with time series data
    """
    # Create widget and test data
    widget = ACFPlot()
    qtbot.addWidget(widget)
    data = generate_test_data(TEST_DATA_LENGTH, ar_param=0.7)
    
    # Set data and verify state changes
    widget.set_data(data)
    
    # Verify widget accepts data without errors
    assert widget._data is not None
    assert np.array_equal(widget._data, data)
    
    # Verify widget state is updated after setting data
    assert widget._initialized is True
    assert widget._figure is not None
    
    # Test with invalid inputs
    with pytest.raises(ValueError):
        widget.set_data(np.array([np.nan, 1, 2]))
        
    with pytest.raises((ValueError, TypeError)):
        widget.set_data([])  # Empty list should be rejected
        
    # Test with non-numeric values
    with pytest.raises(ValueError):
        widget.set_data(np.array(['a', 'b', 'c']))

@pytest.mark.gui
def test_acf_plot_widget_set_parameters(qtbot):
    """
    Tests if parameter setters correctly update widget configuration
    """
    # Create widget and test data
    widget = ACFPlot()
    qtbot.addWidget(widget)
    data = generate_test_data(TEST_DATA_LENGTH, ar_param=0.7)
    widget.set_data(data)
    
    # Test set_nlags
    widget.set_nlags(15)
    assert widget._nlags == 15
    
    # Test invalid nlags
    with pytest.raises(ValueError):
        widget.set_nlags(-1)
    
    with pytest.raises(ValueError):
        widget.set_nlags(0)
        
    # Test set_alpha
    widget.set_alpha(0.01)
    assert widget._alpha == 0.01
    
    # Test invalid alpha
    with pytest.raises(ValueError):
        widget.set_alpha(-0.5)
    
    with pytest.raises(ValueError):
        widget.set_alpha(1.5)
    
    # Test set_plot_params
    params = {
        'title': 'Custom Title',
        'grid': False,
        'bar_color': 'red'
    }
    widget.set_plot_params(params)
    
    # Verify parameters were updated
    assert widget._plot_params['title'] == 'Custom Title'
    assert widget._plot_params['grid'] is False
    assert widget._plot_params['bar_color'] == 'red'
    
    # Test invalid plot_params
    with pytest.raises(TypeError):
        widget.set_plot_params("not_a_dict")

@pytest.mark.gui
def test_acf_plot_update(qtbot):
    """
    Tests if the update_plot method correctly refreshes the plot display
    """
    # Create widget and test data
    widget = ACFPlot()
    qtbot.addWidget(widget)
    data = generate_test_data(TEST_DATA_LENGTH, ar_param=0.7)
    
    # Test update_plot with no data
    widget.update_plot()  # Should handle this gracefully without data
    assert widget._figure is None
    
    # Set data and update plot
    widget.set_data(data)
    
    # Store original figure for comparison
    orig_figure = widget._figure
    widget.clear()
    
    # Update plot manually
    widget._data = data  # Re-add data without triggering set_data
    widget.update_plot()
    
    # Verify update_plot creates a matplotlib figure
    assert widget._figure is not None
    assert isinstance(widget._figure, Figure)
    assert widget._figure is not orig_figure  # Should be a new figure
    
    # Verify widget layout has canvas embedded
    assert widget.layout().count() > 0

@pytest.mark.gui
def test_acf_plot_save_figure(qtbot, tmp_path):
    """
    Tests if the save_figure method correctly saves the plot to a file
    """
    # Create widget and test data
    widget = ACFPlot()
    qtbot.addWidget(widget)
    data = generate_test_data(TEST_DATA_LENGTH, ar_param=0.7)
    widget.set_data(data)
    
    # Create a temporary file path
    file_path = os.path.join(tmp_path, "acf_test.png")
    
    # Try to save figure
    result = widget.save_figure(file_path)
    
    # Verify save was successful
    assert result is True
    assert os.path.exists(file_path)
    assert os.path.getsize(file_path) > 0  # File should not be empty
    
    # Test save with invalid path
    invalid_path = "/nonexistent/directory/test.png"
    result = widget.save_figure(invalid_path)
    assert result is False
    
    # Test save with no figure
    widget.clear()
    result = widget.save_figure(file_path)
    assert result is False

@pytest.mark.gui
def test_acf_plot_clear(qtbot):
    """
    Tests if the clear method correctly resets the widget state
    """
    # Create widget and test data
    widget = ACFPlot()
    qtbot.addWidget(widget)
    data = generate_test_data(TEST_DATA_LENGTH, ar_param=0.7)
    widget.set_data(data)
    
    # Verify widget has plot before clearing
    assert widget._figure is not None
    assert widget._data is not None
    assert widget._initialized is True
    assert widget.layout().count() > 0
    
    # Clear the widget
    widget.clear()
    
    # Verify widget state is reset
    assert widget._figure is None
    assert widget._data is None
    assert widget._initialized is False
    
    # Test clearing an already cleared widget
    widget.clear()  # Should not raise errors
    assert widget._figure is None

@pytest.mark.asyncio
@pytest.mark.gui
async def test_acf_plot_async_operations(qtbot):
    """
    Tests asynchronous behavior of the ACFPlot widget
    """
    # Create widget
    widget = ACFPlot()
    qtbot.addWidget(widget)
    
    # Generate test data
    data = generate_test_data(TEST_DATA_LENGTH, ar_param=0.7)
    
    # Connect a signal to verify async operation completes
    plot_updated_signal_triggered = False
    
    def on_plot_updated():
        nonlocal plot_updated_signal_triggered
        plot_updated_signal_triggered = True
    
    widget.plot_updated.connect(on_plot_updated)
    
    # Set data and await a brief moment for async operations
    widget.set_data(data)
    await asyncio.sleep(0.1)  # Small delay to let async operations complete
    
    # Check if signal was triggered and widget was updated
    assert plot_updated_signal_triggered is True
    assert widget._initialized is True
    assert widget._figure is not None
    
    # Test that operations complete correctly in async mode
    widget.clear()
    widget.set_data(data)
    
    # Verify the figure was created again
    assert widget._figure is not None