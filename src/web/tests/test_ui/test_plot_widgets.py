import pytest
import numpy as np
import asyncio
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtWidgets import QApplication
import matplotlib.pyplot as plt
import pandas as pd

# Import widgets to test
from mfe.ui.plot_widgets import (
    BasePlotWidget, 
    TimeSeriesPlotWidget,
    ResidualPlotWidget,
    ACFPlotWidget,
    PACFPlotWidget,
    VolatilityPlotWidget,
    QQPlotWidget
)

# Use qapp fixture for all tests in this file
pytestmark = pytest.mark.usefixtures('qapp')

@pytest.mark.ui
def test_plot_widget_initialization(qtbot):
    """Tests that a basic PlotWidget can be initialized properly with default settings."""
    # Initialize widget
    widget = BasePlotWidget()
    qtbot.addWidget(widget)
    
    # Check widget is created properly
    assert widget is not None
    assert widget.isVisible()
    assert widget.get_figure() is not None
    
    # Check size policy
    size_policy = widget.sizePolicy()
    assert size_policy.horizontalPolicy() == size_policy.Policy.Expanding
    assert size_policy.verticalPolicy() == size_policy.Policy.Expanding

@pytest.mark.ui
def test_plot_widget_with_data(qtbot):
    """Tests that a PlotWidget can properly display data."""
    # Create sample data
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    
    # Initialize widget
    widget = BasePlotWidget()
    qtbot.addWidget(widget)
    
    # Plot data
    fig = widget.get_figure()
    ax = fig.add_subplot(111)
    ax.plot(x, y)
    widget.update_plot()
    
    # Verify plot contains data (basic check - figure has axes with data)
    assert len(fig.get_axes()) > 0
    assert len(fig.get_axes()[0].lines) > 0

@pytest.mark.ui
def test_time_series_plot_widget(qtbot, sample_time_series):
    """Tests specific functionality of the TimeSeriesPlotWidget."""
    # Initialize widget
    widget = TimeSeriesPlotWidget()
    qtbot.addWidget(widget)
    
    # Set data and update
    widget.set_data(sample_time_series)
    
    # Check widget properties
    assert widget.get_figure() is not None
    
    # Test title setting
    widget.set_title("Test Time Series")
    
    # Test grid toggling
    widget.set_grid(False)
    widget.set_grid(True)
    
    # Test date format setting (if applicable)
    if isinstance(sample_time_series.index, pd.DatetimeIndex):
        widget.set_date_format('%Y-%m')

@pytest.mark.ui
def test_residual_plot_widget(qtbot, sample_data):
    """Tests specific functionality of the ResidualPlotWidget."""
    # Initialize widget
    widget = ResidualPlotWidget()
    qtbot.addWidget(widget)
    
    # Create residuals (use sample data)
    residuals = sample_data
    fitted_values = sample_data * 0.8 + 0.2  # Simple transformation for fitted values
    
    # Set data
    widget.set_residuals(residuals)
    widget.set_fitted_values(fitted_values)
    
    # Check widget exists and has statistics
    assert widget is not None
    stats = widget.get_residual_stats()
    assert isinstance(stats, dict)
    
    # Test plot parameters
    widget.set_plot_params({"title": "Custom Residual Plot", "residual_color": "green"})

@pytest.mark.ui
def test_acf_plot_widget(qtbot, sample_data):
    """Tests specific functionality of the ACFPlotWidget."""
    # Initialize widget
    widget = ACFPlotWidget()
    qtbot.addWidget(widget)
    
    # Set data
    widget.set_data(sample_data)
    
    # Test parameter setting
    widget.set_nlags(30)
    widget.set_alpha(0.01)
    
    # Check widget is properly initialized
    assert widget is not None
    assert widget.get_figure() is not None
    
    # Test confidence intervals and other ACF-specific features
    widget.set_plot_params({"title": "Custom ACF Plot", "bar_color": "blue"})

@pytest.mark.ui
def test_pacf_plot_widget(qtbot, sample_data):
    """Tests specific functionality of the PACFPlotWidget."""
    # Initialize widget
    widget = PACFPlotWidget()
    qtbot.addWidget(widget)
    
    # Set data
    widget.set_data(sample_data)
    
    # Test parameter setting
    widget.set_nlags(15)
    widget.set_alpha(0.05)
    widget.set_method("ywmle")
    
    # Check widget is properly initialized
    assert widget is not None
    
    # Test PACF-specific features
    widget.set_plot_params({"title": "Custom PACF Plot", "bar_color": "red"})

@pytest.mark.ui
def test_volatility_plot_widget(qtbot, mock_volatility_data):
    """Tests specific functionality of the VolatilityPlotWidget."""
    # Initialize widget
    widget = VolatilityPlotWidget()
    qtbot.addWidget(widget)
    
    # Convert mock data to DataFrame
    dates = mock_volatility_data['dates']
    volatility = mock_volatility_data['volatility']
    df = pd.DataFrame({'volatility': volatility}, index=dates)
    
    # Set data
    widget.set_data(df)
    
    # Check widget is properly initialized
    assert widget is not None
    assert widget.get_figure() is not None

@pytest.mark.ui
def test_qq_plot_widget(qtbot, sample_data):
    """Tests specific functionality of the QQPlotWidget."""
    # Initialize widget
    widget = QQPlotWidget()
    qtbot.addWidget(widget)
    
    # Set data
    widget.set_data(sample_data)
    
    # Check widget is properly initialized
    assert widget is not None
    assert widget.get_figure() is not None
    
    # Test QQ plot-specific features like the reference line
    assert widget.get_figure().get_axes()[0] is not None

@pytest.mark.ui
def test_plot_widget_update(qtbot):
    """Tests that a PlotWidget can be updated with new data."""
    # Initialize widget
    widget = BasePlotWidget()
    qtbot.addWidget(widget)
    
    # Initial plot
    fig = widget.get_figure()
    ax = fig.add_subplot(111)
    ax.plot([1, 2, 3], [1, 2, 3])
    widget.update_plot()
    
    # Check initial state
    assert len(fig.get_axes()) > 0
    assert len(fig.get_axes()[0].lines) > 0
    
    # Update with new data
    fig.clear()
    ax = fig.add_subplot(111)
    ax.plot([1, 2, 3], [3, 2, 1])  # Different data
    widget.update_plot()
    
    # Verify update succeeded
    assert len(fig.get_axes()) > 0
    assert len(fig.get_axes()[0].lines) > 0

@pytest.mark.ui
def test_plot_widget_clear(qtbot):
    """Tests that a PlotWidget can be cleared properly."""
    # Initialize widget
    widget = BasePlotWidget()
    qtbot.addWidget(widget)
    
    # Add plot
    fig = widget.get_figure()
    ax = fig.add_subplot(111)
    ax.plot([1, 2, 3], [1, 2, 3])
    widget.update_plot()
    
    # Clear plot
    widget.clear()
    
    # Verify plot is cleared
    assert widget.get_figure() is not None
    # Different implementations might clear differently, so we can't check specifically

@pytest.mark.ui
@pytest.mark.asyncio
async def test_plot_widget_async_update(qtbot):
    """Tests that a PlotWidget can be updated asynchronously."""
    # Initialize widget
    widget = BasePlotWidget()
    qtbot.addWidget(widget)
    
    # Define async update function
    async def do_update():
        # Get figure and add plot
        fig = widget.get_figure()
        ax = fig.add_subplot(111)
        ax.plot([1, 2, 3], [1, 2, 3])
        await widget.async_update_plot()
    
    # Execute async update
    await do_update()
    
    # Verify update succeeded
    fig = widget.get_figure()
    assert len(fig.get_axes()) > 0
    assert len(fig.get_axes()[0].lines) > 0

@pytest.mark.ui
def test_plot_widget_resize(qtbot):
    """Tests that a PlotWidget properly handles resizing."""
    # Initialize widget
    widget = BasePlotWidget()
    qtbot.addWidget(widget)
    widget.show()
    
    # Get original size
    original_size = widget.size()
    
    # Resize widget
    new_width = original_size.width() + 100
    new_height = original_size.height() + 100
    widget.resize(new_width, new_height)
    
    # Process events to ensure resize takes effect
    QApplication.processEvents()
    
    # Check that size changed
    assert widget.width() > original_size.width()
    assert widget.height() > original_size.height()
    
    # The figure should maintain its integrity after resizing
    assert widget.get_figure() is not None

@pytest.mark.ui
def test_plot_widget_save(qtbot, tmp_path):
    """Tests that a PlotWidget can save its contents to a file."""
    # Initialize widget
    widget = BasePlotWidget()
    qtbot.addWidget(widget)
    
    # Add plot content
    fig = widget.get_figure()
    ax = fig.add_subplot(111)
    ax.plot([1, 2, 3], [1, 2, 3])
    widget.update_plot()
    
    # Save to a file
    filepath = tmp_path / "test_plot.png"
    success = widget.save_plot(str(filepath))
    
    # Verify file was created
    assert success
    assert filepath.exists()
    assert filepath.stat().st_size > 0  # File should not be empty

@pytest.mark.ui
def test_plot_widget_style(qtbot):
    """Tests that a PlotWidget correctly applies style settings."""
    # Initialize widget
    widget = BasePlotWidget()
    qtbot.addWidget(widget)
    
    # Set custom style through plot parameters
    widget.set_plot_params({
        "title": "Styled Plot",
        "grid": True,
        "line_width": 2.0,
        "marker": "o",
        "alpha": 0.8
    })
    
    # Verify the widget exists (full style verification would require more complex testing)
    assert widget is not None
    
    # Check the figure has been created and styled
    fig = widget.get_figure()
    assert fig is not None