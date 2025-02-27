"""
MFE Toolbox - Interactive Time Series Plot Component

This module provides an interactive PyQt6 widget for visualizing financial time series data
with support for asynchronous updates, customizable styling, and interactive features.
The component integrates with Matplotlib for high-quality financial data visualization
and leverages PyQt6 for GUI integration.
"""

import numpy as np  # version 1.26.3
import matplotlib.pyplot as plt  # version 3.7.1
import pandas as pd  # version 2.1.4
import matplotlib.dates as mdates  # version 3.7.1
from PyQt6.QtCore import QObject, pyqtSignal, QSize  # version 6.6.1
from PyQt6.QtWidgets import QWidget, QVBoxLayout  # version 6.6.1
import logging  # Python standard library
from typing import Optional, Dict, List, Union, Tuple, Any  # Python standard library
import asyncio  # Python standard library

# Internal imports
from .matplotlib_backend import AsyncPlotWidget
from ..async.signals import PlotSignals
from ....backend.mfe.utils.validation import validate_array

# Setup module logger
logger = logging.getLogger(__name__)

# Default plot parameters
DEFAULT_PLOT_PARAMS = {
    "title": "Time Series Plot",
    "grid": True,
    "line_width": 1.5,
    "marker": None,
    "alpha": 1.0,
    "date_format": "%Y-%m-%d",
    "legend": True,
    "color_map": "tab10"
}


def create_time_series_figure(data: pd.DataFrame, plot_params: Dict) -> plt.Figure:
    """
    Creates a matplotlib figure configured for time series visualization.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing time series data to plot, typically with a DatetimeIndex
    plot_params : dict
        Dictionary of plot parameters controlling appearance and behavior
        
    Returns
    -------
    matplotlib.figure.Figure
        A configured matplotlib figure with time series plot
    
    Notes
    -----
    This function creates a standalone figure for time series visualization,
    applying appropriate styling and date formatting for financial data.
    """
    # Validate data
    validate_array(data.values)
    
    # Create figure and axis
    fig = plt.figure(figsize=(10, 6), dpi=100)
    ax = fig.add_subplot(111)
    
    # Plot each column of the DataFrame
    for column in data.columns:
        ax.plot(data.index, data[column], 
                linewidth=plot_params.get("line_width", DEFAULT_PLOT_PARAMS["line_width"]),
                marker=plot_params.get("marker", DEFAULT_PLOT_PARAMS["marker"]),
                alpha=plot_params.get("alpha", DEFAULT_PLOT_PARAMS["alpha"]),
                label=column)
    
    # Set title if provided
    title = plot_params.get("title", DEFAULT_PLOT_PARAMS["title"])
    ax.set_title(title)
    
    # Configure grid
    grid = plot_params.get("grid", DEFAULT_PLOT_PARAMS["grid"])
    ax.grid(grid, linestyle='--', alpha=0.7)
    
    # Format date axis if the index contains datetime values
    if isinstance(data.index, pd.DatetimeIndex):
        date_format = plot_params.get("date_format", DEFAULT_PLOT_PARAMS["date_format"])
        format_date_axis(ax, data, date_format)
    
    # Add legend if requested
    if plot_params.get("legend", DEFAULT_PLOT_PARAMS["legend"]) and len(data.columns) > 1:
        ax.legend()
    
    # Adjust layout for optimal display
    fig.tight_layout()
    
    return fig


def format_date_axis(ax: plt.Axes, data: pd.DataFrame, date_format: str) -> None:
    """
    Configures the x-axis for proper date display in time series plots.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The matplotlib axes to configure
    data : pd.DataFrame
        The DataFrame with datetime index
    date_format : str
        Format string for date labels (e.g., '%Y-%m-%d')
        
    Notes
    -----
    This function automatically determines appropriate date locators and formatters
    based on the date range in the data.
    """
    # Verify data has a DatetimeIndex
    if not isinstance(data.index, pd.DatetimeIndex):
        logger.warning("DataFrame index is not a DatetimeIndex, skipping date formatting")
        return
    
    # Set the date formatter
    date_formatter = mdates.DateFormatter(date_format)
    ax.xaxis.set_major_formatter(date_formatter)
    
    # Calculate date range to determine appropriate locator
    date_range = (data.index.max() - data.index.min()).days
    
    # Set appropriate locators based on data range
    if date_range <= 7:  # Less than a week
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_minor_locator(mdates.HourLocator(interval=6))
    elif date_range <= 60:  # Less than two months
        ax.xaxis.set_major_locator(mdates.WeekdayLocator())
        ax.xaxis.set_minor_locator(mdates.DayLocator())
    elif date_range <= 365:  # Less than a year
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_minor_locator(mdates.WeekdayLocator())
    elif date_range <= 365 * 2:  # Less than two years
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_minor_locator(mdates.MonthLocator())
    else:  # More than two years
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=3))
    
    # Rotate date labels for better readability if needed
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Adjust bottom margin to accommodate rotated labels
    plt.subplots_adjust(bottom=0.15)


class TimeSeriesPlot(QWidget):
    """
    Interactive PyQt6 widget for displaying and manipulating financial time series data plots
    with asynchronous update capabilities.
    
    This widget provides a comprehensive interface for visualizing time series data
    with support for customizable styling, interactive updates, and asynchronous
    rendering to maintain UI responsiveness.
    
    Attributes
    ----------
    signals : PlotSignals
        PyQt signals for asynchronous plot update events
    """
    
    def __init__(self, parent=None, data: Optional[pd.DataFrame]=None, 
                 plot_params: Optional[Dict]=None):
        """
        Initializes the time series plot widget with default configuration.
        
        Parameters
        ----------
        parent : QWidget, optional
            Parent widget
        data : Optional[pd.DataFrame], default=None
            Initial time series data to plot
        plot_params : Optional[Dict], default=None
            Plot parameters to customize appearance
        """
        super().__init__(parent)
        
        # Initialize signals
        self.signals = PlotSignals()
        
        # Initialize properties
        self._data = data
        self._plot_params = DEFAULT_PLOT_PARAMS.copy()
        if plot_params:
            self._plot_params.update(plot_params)
        
        self._initialized = False
        
        # Setup layout
        self._layout = QVBoxLayout(self)
        self.setLayout(self._layout)
        
        # Create plot widget
        self._plot_widget = AsyncPlotWidget(self)
        self._layout.addWidget(self._plot_widget)
        
        # Plot data if provided
        if data is not None:
            self.update_plot()
        
        logger.debug("TimeSeriesPlot widget initialized")
    
    def set_data(self, data: pd.DataFrame) -> None:
        """
        Sets the time series data to be visualized.
        
        Parameters
        ----------
        data : pd.DataFrame
            DataFrame containing time series data, typically with DatetimeIndex
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Data must be a pandas DataFrame")
        
        self._data = data
        
        # Update plot if already initialized
        if self._initialized:
            self.update_plot()
        
        logger.debug(f"Set time series data with shape {data.shape}")
    
    def set_plot_params(self, params: Dict) -> None:
        """
        Sets plotting parameters to customize time series plot appearance.
        
        Parameters
        ----------
        params : Dict
            Dictionary of plot parameters to update
        """
        if not isinstance(params, dict):
            raise TypeError("Plot parameters must be a dictionary")
        
        # Update current parameters with new values
        self._plot_params.update(params)
        
        # Update plot if data is available
        if self._data is not None and self._initialized:
            self.update_plot()
        
        logger.debug(f"Updated plot parameters: {params.keys()}")
    
    def update_plot(self) -> None:
        """
        Updates the time series plot with current data and settings.
        
        This method performs a synchronous update of the plot, which may
        block the UI briefly for large datasets.
        """
        if self._data is None:
            logger.warning("Cannot update plot: No data available")
            return
        
        # Signal that update is starting
        self.signals.update_started.emit()
        
        try:
            # Clear existing plot
            self._plot_widget.clear_plot()
            
            # Create new figure for the time series data
            figure = create_time_series_figure(self._data, self._plot_params)
            
            # Set the figure in the plot widget
            self._plot_widget.get_figure().clear()
            
            # Copy the figure content to the widget's figure
            for ax in figure.get_axes():
                self._plot_widget.get_figure().add_subplot(ax)
            
            # Update the display
            self._plot_widget.update_plot()
            
            # Set initialized flag
            self._initialized = True
            
            # Signal that update is complete
            self.signals.update_complete.emit(self)
            
            logger.debug("Time series plot updated successfully")
        except Exception as e:
            logger.error(f"Error updating time series plot: {str(e)}")
            self.signals.update_error.emit(Exception(f"Plot update failed: {str(e)}"))
    
    async def async_update_plot(self) -> None:
        """
        Asynchronously updates the time series plot to prevent UI blocking.
        
        This method is particularly useful for large datasets or when
        frequent updates are needed without freezing the UI.
        """
        if self._data is None:
            logger.warning("Cannot update plot: No data available")
            return
        
        # Signal that update is starting
        self.signals.update_started.emit()
        
        try:
            # Yield control back to event loop to prevent UI freezing
            await asyncio.sleep(0)
            
            # Clear existing plot
            self._plot_widget.clear_plot()
            
            # Create new figure for the time series data
            figure = create_time_series_figure(self._data, self._plot_params)
            
            # Set the figure in the plot widget
            self._plot_widget.get_figure().clear()
            
            # Copy the figure content to the widget's figure
            for ax in figure.get_axes():
                self._plot_widget.get_figure().add_subplot(ax)
            
            # Update the display asynchronously
            await self._plot_widget.async_update_plot()
            
            # Set initialized flag
            self._initialized = True
            
            # Signal that update is complete
            self.signals.update_complete.emit(self)
            
            logger.debug("Time series plot updated asynchronously")
        except Exception as e:
            logger.error(f"Error in async update of time series plot: {str(e)}")
            self.signals.update_error.emit(Exception(f"Async plot update failed: {str(e)}"))
    
    def save_figure(self, filepath: str, dpi: Optional[int]=None) -> bool:
        """
        Saves the current time series plot to a file.
        
        Parameters
        ----------
        filepath : str
            Path where the figure should be saved
        dpi : Optional[int], default=None
            Resolution in dots per inch for the saved figure
            
        Returns
        -------
        bool
            True if save was successful, False otherwise
        """
        if not self._initialized:
            logger.warning("Cannot save figure: Plot not initialized")
            return False
        
        try:
            # Save the figure using the AsyncPlotWidget's save method
            success = self._plot_widget.save_plot(filepath, dpi or 300)
            
            if success:
                logger.debug(f"Time series plot saved to {filepath}")
            else:
                logger.warning(f"Failed to save time series plot to {filepath}")
            
            return success
        except Exception as e:
            logger.error(f"Error saving time series plot: {str(e)}")
            return False
    
    def clear(self) -> None:
        """
        Clears the current plot and resets widget state.
        """
        self._data = None
        self._plot_widget.clear_plot()
        self._initialized = False
        logger.debug("Time series plot cleared")
    
    def set_title(self, title: str) -> None:
        """
        Sets the plot title.
        
        Parameters
        ----------
        title : str
            Title to display on the plot
        """
        self._plot_params["title"] = title
        
        if self._initialized:
            self.update_plot()
            
        logger.debug(f"Time series plot title set to '{title}'")
    
    def set_grid(self, visible: bool) -> None:
        """
        Toggles the plot grid visibility.
        
        Parameters
        ----------
        visible : bool
            Whether to show grid lines
        """
        self._plot_params["grid"] = visible
        
        if self._initialized:
            self.update_plot()
            
        logger.debug(f"Time series plot grid visibility set to {visible}")
    
    def set_date_format(self, date_format: str) -> None:
        """
        Sets the date format for x-axis labels.
        
        Parameters
        ----------
        date_format : str
            Format string for date labels (e.g., '%Y-%m-%d')
        """
        self._plot_params["date_format"] = date_format
        
        if self._initialized and self._data is not None and isinstance(self._data.index, pd.DatetimeIndex):
            self.update_plot()
            
        logger.debug(f"Time series plot date format set to '{date_format}'")
    
    def get_figure(self) -> Optional[plt.Figure]:
        """
        Returns the matplotlib figure for direct access.
        
        Returns
        -------
        Optional[matplotlib.figure.Figure]
            The current figure or None if not initialized
        """
        if not self._initialized:
            return None
            
        return self._plot_widget.get_figure()
    
    def sizeHint(self) -> QSize:
        """
        Returns the recommended size for the widget.
        
        Returns
        -------
        QSize
            Recommended widget size
        """
        return QSize(800, 500)