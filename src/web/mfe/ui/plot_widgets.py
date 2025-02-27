"""
MFE Toolbox - Plot Widgets Module

This module provides high-level plot widget classes that integrate specific plot types
from the plots module with PyQt6's widget system. These widgets encapsulate common
plot configurations and interactions needed across the MFE Toolbox UI, supporting
asynchronous updates to maintain UI responsiveness during intensive econometric computations.
"""

import asyncio  # Python standard library
import logging  # Python standard library
import typing  # Python standard library
from typing import Dict, Optional, Union, Tuple, List, Any

import numpy as np  # numpy 1.26.3
import pandas as pd  # pandas 2.1.4
import matplotlib.pyplot as plt  # matplotlib 3.7.1
from matplotlib.figure import Figure  # matplotlib 3.7.1
from statsmodels.tsa.stattools import pacf  # statsmodels 0.14.1

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QSplitter,
    QSizePolicy, QPushButton, QToolBar
)  # PyQt6 6.6.1
from PyQt6.QtCore import Qt, QSize, pyqtSignal, pyqtSlot  # PyQt6 6.6.1

# Internal imports
from .plots.residual_plot import ResidualPlot
from .plots.acf_plot import ACFPlot
from .plots.time_series_plot import TimeSeriesPlot
from .plots.matplotlib_backend import AsyncPlotWidget
from .async.worker import AsyncWorker, WorkerManager

# Setup module logger
logger = logging.getLogger(__name__)

# Default constants
DEFAULT_PLOT_HEIGHT = 300
DEFAULT_PLOT_WIDTH = 400
TOOLBAR_ICON_SIZE = 24
PLOT_SPACING = 10
DEFAULT_PACF_PARAMS = {"title": "Partial Autocorrelation Function", "grid": True, "bar_color": "blue", "alpha": 0.05, "nlags": 20, "method": "ywmle"}


def create_plot_toolbar(plot_widget: AsyncPlotWidget) -> QToolBar:
    """
    Creates a toolbar with common plot manipulation actions.
    
    Parameters
    ----------
    plot_widget : AsyncPlotWidget
        The plot widget to associate with the toolbar
        
    Returns
    -------
    QToolBar
        Toolbar with plot manipulation buttons
    """
    toolbar = QToolBar()
    toolbar.setIconSize(QSize(TOOLBAR_ICON_SIZE, TOOLBAR_ICON_SIZE))
    
    # Add save action
    save_action = toolbar.addAction("Save")
    save_action.triggered.connect(lambda: plot_widget.save_plot("plot.png", 300))
    
    # Add zoom actions
    zoom_in_action = toolbar.addAction("Zoom In")
    zoom_out_action = toolbar.addAction("Zoom Out")
    
    # Add reset view action
    reset_action = toolbar.addAction("Reset View")
    
    # Add export action
    export_action = toolbar.addAction("Export")
    
    # Configure toolbar appearance
    toolbar.setOrientation(Qt.Orientation.Horizontal)
    toolbar.setFloatable(False)
    toolbar.setMovable(False)
    
    return toolbar


def apply_plot_stylesheet(widget: QWidget, style_name: str) -> None:
    """
    Applies a consistent style to a plot widget.
    
    Parameters
    ----------
    widget : QWidget
        The widget to style
    style_name : str
        The name of the style to apply
    """
    if style_name == "default":
        stylesheet = """
            QWidget {
                background-color: white;
                border: 1px solid #cccccc;
                border-radius: 4px;
            }
            QToolBar {
                background-color: #f0f0f0;
                border-bottom: 1px solid #cccccc;
                spacing: 2px;
                padding: 2px;
            }
            QPushButton {
                background-color: #f0f0f0;
                border: 1px solid #cccccc;
                border-radius: 2px;
                padding: 4px;
                min-width: 60px;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
            QPushButton:pressed {
                background-color: #d0d0d0;
            }
        """
    elif style_name == "dark":
        stylesheet = """
            QWidget {
                background-color: #2d2d30;
                border: 1px solid #3f3f46;
                border-radius: 4px;
                color: #f0f0f0;
            }
            QToolBar {
                background-color: #333337;
                border-bottom: 1px solid #3f3f46;
                spacing: 2px;
                padding: 2px;
            }
            QPushButton {
                background-color: #3f3f46;
                border: 1px solid #555555;
                border-radius: 2px;
                padding: 4px;
                min-width: 60px;
                color: #f0f0f0;
            }
            QPushButton:hover {
                background-color: #505050;
            }
            QPushButton:pressed {
                background-color: #606060;
            }
        """
    else:
        # Default to a minimal stylesheet
        stylesheet = """
            QWidget {
                border: 1px solid #cccccc;
            }
        """
    
    widget.setStyleSheet(stylesheet)
    widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)


def calculate_pacf(data: np.ndarray, nlags: int = 20, alpha: float = 0.05, method: str = "ywmle") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates the partial autocorrelation function for a given time series data using statsmodels.
    
    Parameters
    ----------
    data : np.ndarray
        The time series data for which to calculate PACF
    nlags : int, optional
        Number of lags to include in the PACF calculation, default is 20
    alpha : float, optional
        Significance level for the confidence intervals, default is 0.05
    method : str, optional
        Method to use for PACF calculation, default is "ywmle"
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Tuple containing PACF values, confidence intervals, and lags
    """
    # Validate input data
    if not isinstance(data, np.ndarray):
        data = np.asarray(data)
    
    if not np.isfinite(data).all():
        raise ValueError("Input data contains NaN or infinite values")
    
    # Calculate PACF using statsmodels
    pacf_values = pacf(data, nlags=nlags, alpha=alpha, method=method)
    
    # If alpha is provided, statsmodels returns a tuple with values and confidence intervals
    if isinstance(pacf_values, tuple):
        pacf_vals = pacf_values[0]
        conf_intervals = pacf_values[1]
    else:
        pacf_vals = pacf_values
        # Calculate approx confidence intervals manually if not provided
        conf_intervals = np.ones((2, nlags + 1)) * 1.96 / np.sqrt(len(data))
        conf_intervals[0, :] = -conf_intervals[0, :]
    
    # Create lag indices
    lags = np.arange(len(pacf_vals))
    
    return pacf_vals, conf_intervals, lags


def create_pacf_figure(data: np.ndarray, nlags: int = 20, alpha: float = 0.05, 
                      method: str = "ywmle", plot_params: Dict[str, Any] = None) -> Figure:
    """
    Creates a matplotlib figure with a PACF plot for the given data.
    
    Parameters
    ----------
    data : np.ndarray
        The time series data for which to create a PACF plot
    nlags : int, optional
        Number of lags to include in the plot, default is 20
    alpha : float, optional
        Significance level for the confidence intervals, default is 0.05
    method : str, optional
        Method to use for PACF calculation, default is "ywmle"
    plot_params : Dict[str, Any], optional
        Dictionary containing customization parameters for the plot
        
    Returns
    -------
    matplotlib.figure.Figure
        A matplotlib figure object containing the PACF plot
    """
    # Set default plot parameters if not provided
    if plot_params is None:
        plot_params = {}
    
    # Default parameters
    default_params = DEFAULT_PACF_PARAMS.copy()
    
    # Update with user-provided parameters
    for key, value in plot_params.items():
        default_params[key] = value
    
    # Calculate PACF values and confidence intervals
    pacf_vals, conf_intervals, lags = calculate_pacf(
        data, 
        nlags=nlags, 
        alpha=alpha, 
        method=method
    )
    
    # Create figure
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    
    # Plot PACF as bars
    ax.bar(lags, pacf_vals, width=0.3, 
           color=default_params["bar_color"], alpha=0.7)
    
    # Add confidence interval lines
    if isinstance(conf_intervals, np.ndarray) and conf_intervals.shape[0] == 2:
        ax.plot(lags, conf_intervals[0, :], color='r', 
                linestyle='--', linewidth=1)
        ax.plot(lags, conf_intervals[1, :], color='r', 
                linestyle='--', linewidth=1)
    
    # Add zero line for reference
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    
    # Add labels and title
    ax.set_title(default_params["title"])
    ax.set_xlabel("Lag")
    ax.set_ylabel("Partial Autocorrelation")
    
    # Add grid if specified
    ax.grid(default_params["grid"])
    
    # Tight layout for better appearance
    fig.tight_layout()
    
    return fig


class BasePlotWidget(QWidget):
    """
    Base class for all plot widgets providing common functionality.
    
    This class implements the core functionality shared by all plot widgets,
    including layout management, asynchronous updates, and toolbar integration.
    """
    
    # Define signals
    updated = pyqtSignal()
    cleared = pyqtSignal()
    updateStarted = pyqtSignal()
    error = pyqtSignal(Exception)
    
    def __init__(self, parent=None):
        """
        Initializes the base plot widget with layout and common components.
        
        Parameters
        ----------
        parent : QWidget
            Parent widget
        """
        super().__init__(parent)
        
        # Create main layout
        self._layout = QVBoxLayout(self)
        self.setLayout(self._layout)
        
        # Initialize plot widget
        self._plot_widget = AsyncPlotWidget(self)
        
        # Create toolbar
        self._toolbar = create_plot_toolbar(self._plot_widget)
        
        # Add toolbar and plot widget to layout
        self._layout.addWidget(self._toolbar)
        self._layout.addWidget(self._plot_widget)
        
        # Initialize worker manager for async operations
        self._worker_manager = WorkerManager()
        
        # Initialize state variables
        self._is_updating = False
        self._plot_params = {}
        
        # Set size policy and minimum size
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(DEFAULT_PLOT_WIDTH, DEFAULT_PLOT_HEIGHT)
    
    def get_figure(self) -> Figure:
        """
        Returns the matplotlib figure for direct manipulation.
        
        Returns
        -------
        matplotlib.figure.Figure
            The underlying matplotlib figure
        """
        return self._plot_widget.get_figure()
    
    def clear(self) -> None:
        """
        Clears the plot.
        """
        self._plot_widget.clear_plot()
        self._is_updating = False
        self.cleared.emit()
    
    def update_plot(self) -> None:
        """
        Updates the plot synchronously.
        
        Note: This may block the UI thread briefly. For non-blocking updates,
        use async_update_plot() or schedule_update().
        """
        if self._is_updating:
            logger.debug("Plot update already in progress, skipping")
            return
        
        self._is_updating = True
        try:
            self._plot_widget.update_plot()
            self._is_updating = False
            self.updated.emit()
        except Exception as e:
            self._is_updating = False
            logger.error(f"Error updating plot: {str(e)}")
            self.error.emit(e)
    
    async def async_update_plot(self) -> None:
        """
        Updates the plot asynchronously to prevent UI blocking.
        
        This is useful for computationally intensive plot updates.
        """
        if self._is_updating:
            logger.debug("Plot update already in progress, skipping")
            return
        
        self._is_updating = True
        try:
            await self._plot_widget.async_update_plot()
            self._is_updating = False
            self.updated.emit()
        except Exception as e:
            self._is_updating = False
            logger.error(f"Error in async plot update: {str(e)}")
            self.error.emit(e)
    
    def schedule_update(self) -> None:
        """
        Schedules an asynchronous update using the worker system.
        
        This is the recommended way to update plots in response to UI events.
        """
        if self._is_updating:
            logger.debug("Plot update already in progress, skipping")
            return
        
        # Create async worker for plot update
        worker = self._worker_manager.create_async_worker(self.async_update_plot)
        
        # Connect signals
        worker.signals.started.connect(lambda: self.updateStarted.emit())
        worker.signals.finished.connect(self._on_worker_finished)
        worker.signals.error.connect(self._on_worker_error)
        
        # Start the worker
        self._worker_manager.start_worker(worker)
        self.updateStarted.emit()
    
    def save_plot(self, filename: str, dpi: int = 300) -> bool:
        """
        Saves the current plot to a file.
        
        Parameters
        ----------
        filename : str
            Path to save the plot
        dpi : int, optional
            Resolution in dots per inch, default is 300
            
        Returns
        -------
        bool
            True if save was successful, False otherwise
        """
        return self._plot_widget.save_plot(filename, dpi)
    
    def set_plot_params(self, params: Dict[str, Any]) -> None:
        """
        Sets plot parameters and triggers update.
        
        Parameters
        ----------
        params : Dict[str, Any]
            Dictionary of plot parameters
        """
        if not isinstance(params, dict):
            raise TypeError("Plot parameters must be a dictionary")
        
        self._plot_params.update(params)
        self.schedule_update()
    
    def sizeHint(self) -> QSize:
        """
        Returns the recommended size for the widget.
        
        Returns
        -------
        QSize
            Recommended size
        """
        return QSize(DEFAULT_PLOT_WIDTH, DEFAULT_PLOT_HEIGHT)
    
    @pyqtSlot()
    def _on_worker_finished(self) -> None:
        """
        Handles worker completion.
        """
        self._is_updating = False
        self.updated.emit()
        logger.debug("Plot update completed")
    
    @pyqtSlot(Exception)
    def _on_worker_error(self, error: Exception) -> None:
        """
        Handles worker errors.
        
        Parameters
        ----------
        error : Exception
            Error that occurred during async update
        """
        self._is_updating = False
        logger.error(f"Error in plot worker: {str(error)}")
        self.error.emit(error)


class TimeSeriesPlotWidget(BasePlotWidget):
    """
    Widget for displaying financial time series data.
    
    This widget provides an interface for visualizing time series data
    with support for pandas DataFrames and asynchronous updates.
    """
    
    # Define signals
    dataChanged = pyqtSignal()
    
    def __init__(self, parent=None):
        """
        Initializes the time series plot widget.
        
        Parameters
        ----------
        parent : QWidget, optional
            Parent widget
        """
        super().__init__(parent)
        
        # Initialize data properties
        self._data = None
        self._dates = None
        
        # Create TimeSeriesPlot instance
        self._ts_plot = TimeSeriesPlot(self)
        
        # Replace the base AsyncPlotWidget with the specialized TimeSeriesPlot
        self._layout.removeWidget(self._plot_widget)
        self._plot_widget = self._ts_plot
        self._layout.addWidget(self._ts_plot)
        
        # Set default plot parameters
        self._plot_params = {
            "title": "Time Series Plot",
            "grid": True,
            "line_width": 1.5,
            "marker": None,
            "alpha": 1.0,
        }
    
    def set_data(self, data: np.ndarray, dates: pd.DatetimeIndex = None) -> None:
        """
        Sets the time series data to be visualized.
        
        Parameters
        ----------
        data : np.ndarray
            The time series data
        dates : pd.DatetimeIndex, optional
            The corresponding dates for the data
        """
        # Validate data
        if not isinstance(data, np.ndarray):
            data = np.asarray(data)
        
        if dates is not None and not isinstance(dates, pd.DatetimeIndex):
            raise TypeError("dates must be a pandas DatetimeIndex")
        
        # Store data and dates
        self._data = data
        self._dates = dates
        
        # Update time series plot
        self._ts_plot.set_data(data, dates)
        
        # Schedule update
        self.schedule_update()
        
        # Emit signal
        self.dataChanged.emit()
    
    def set_title(self, title: str) -> None:
        """
        Sets the plot title.
        
        Parameters
        ----------
        title : str
            The title for the plot
        """
        self._plot_params["title"] = title
        self._ts_plot.set_title(title)
        self.schedule_update()
    
    def update_plot(self) -> None:
        """
        Updates the time series plot.
        """
        if self._data is None:
            logger.warning("Cannot update plot: No data available")
            return
        
        self._ts_plot.update_plot()
        self.updated.emit()
    
    async def async_update_plot(self) -> None:
        """
        Updates the time series plot asynchronously.
        """
        if self._data is None:
            logger.warning("Cannot update plot: No data available")
            return
        
        await self._ts_plot.async_update_plot()
        self.updated.emit()
    
    def clear(self) -> None:
        """
        Clears the time series plot.
        """
        self._data = None
        self._dates = None
        self._ts_plot.clear()
        self.cleared.emit()


class ResidualPlotWidget(BasePlotWidget):
    """
    Widget for displaying model residual diagnostics.
    
    This widget provides comprehensive visualization of model residuals,
    including scatter plots, distribution analysis, and statistical tests.
    """
    
    # Define signals
    dataChanged = pyqtSignal()
    
    def __init__(self, parent=None):
        """
        Initializes the residual plot widget.
        
        Parameters
        ----------
        parent : QWidget, optional
            Parent widget
        """
        super().__init__(parent)
        
        # Initialize data properties
        self._residuals = None
        self._fitted_values = None
        
        # Create ResidualPlot instance
        self._residual_plot = ResidualPlot(self)
        
        # Replace the base AsyncPlotWidget with the specialized ResidualPlot
        self._layout.removeWidget(self._plot_widget)
        self._plot_widget = self._residual_plot
        self._layout.addWidget(self._residual_plot)
        
        # Initialize residual statistics
        self._residual_stats = {}
        
        # Set default plot parameters
        self._plot_params = {
            "title": "Residual Analysis",
            "grid": True,
            "residual_color": "blue",
            "hist_bins": 20,
            "alpha": 0.7,
        }
    
    def set_residuals(self, residuals: np.ndarray) -> None:
        """
        Sets the residual data to be visualized.
        
        Parameters
        ----------
        residuals : np.ndarray
            The model residuals
        """
        # Validate residuals
        if not isinstance(residuals, np.ndarray):
            residuals = np.asarray(residuals)
        
        # Store residuals
        self._residuals = residuals
        
        # Update residual plot
        self._residual_plot.set_residuals(residuals)
        
        # Update residual statistics
        self._residual_stats = self._residual_plot.get_residual_stats()
        
        # Schedule update
        self.schedule_update()
        
        # Emit signal
        self.dataChanged.emit()
    
    def set_fitted_values(self, fitted_values: np.ndarray) -> None:
        """
        Sets the fitted values for residual analysis.
        
        Parameters
        ----------
        fitted_values : np.ndarray
            The fitted values from the model
        """
        # Validate fitted values
        if not isinstance(fitted_values, np.ndarray):
            fitted_values = np.asarray(fitted_values)
        
        # Store fitted values
        self._fitted_values = fitted_values
        
        # Update residual plot
        self._residual_plot.set_fitted_values(fitted_values)
        
        # Schedule update
        self.schedule_update()
        
        # Emit signal
        self.dataChanged.emit()
    
    def get_residual_stats(self) -> Dict[str, Any]:
        """
        Gets the current residual statistics.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary of residual statistics
        """
        return self._residual_stats.copy()
    
    def update_plot(self) -> None:
        """
        Updates the residual plot.
        """
        if self._residuals is None:
            logger.warning("Cannot update plot: No residuals available")
            return
        
        self._residual_plot.update_plot()
        self.updated.emit()
    
    async def async_update_plot(self) -> None:
        """
        Updates the residual plot asynchronously.
        """
        if self._residuals is None:
            logger.warning("Cannot update plot: No residuals available")
            return
        
        await self._residual_plot.async_update_plot()
        self.updated.emit()
    
    def clear(self) -> None:
        """
        Clears the residual plot.
        """
        self._residuals = None
        self._fitted_values = None
        self._residual_stats = {}
        self._residual_plot.clear()
        self.cleared.emit()


class ACFPlotWidget(BasePlotWidget):
    """
    Widget for displaying autocorrelation function plots.
    
    This widget provides visualization of autocorrelation functions
    with configurable lags and confidence intervals.
    """
    
    # Define signals
    dataChanged = pyqtSignal()
    
    def __init__(self, parent=None):
        """
        Initializes the ACF plot widget.
        
        Parameters
        ----------
        parent : QWidget, optional
            Parent widget
        """
        super().__init__(parent)
        
        # Initialize data properties
        self._data = None
        self._nlags = 20
        self._alpha = 0.05
        
        # Create ACFPlot instance
        self._acf_plot = ACFPlot(self)
        
        # Replace the base AsyncPlotWidget with the specialized ACFPlot
        self._layout.removeWidget(self._plot_widget)
        self._plot_widget = self._acf_plot
        self._layout.addWidget(self._acf_plot)
        
        # Set default plot parameters
        self._plot_params = {
            "title": "Autocorrelation Function",
            "grid": True,
            "bar_color": "steelblue",
            "alpha": 0.05,
        }
    
    def set_data(self, data: np.ndarray) -> None:
        """
        Sets the data for ACF calculation.
        
        Parameters
        ----------
        data : np.ndarray
            The time series data
        """
        # Validate data
        if not isinstance(data, np.ndarray):
            data = np.asarray(data)
        
        # Store data
        self._data = data
        
        # Update ACF plot
        self._acf_plot.set_data(data)
        
        # Schedule update
        self.schedule_update()
        
        # Emit signal
        self.dataChanged.emit()
    
    def set_nlags(self, nlags: int) -> None:
        """
        Sets the number of lags for ACF calculation.
        
        Parameters
        ----------
        nlags : int
            Number of lags to display
        """
        # Validate nlags
        if not isinstance(nlags, int) or nlags <= 0:
            raise ValueError("Number of lags must be a positive integer")
        
        # Store nlags
        self._nlags = nlags
        
        # Update ACF plot
        self._acf_plot.set_nlags(nlags)
        
        # Schedule update if data is available
        if self._data is not None:
            self.schedule_update()
    
    def set_alpha(self, alpha: float) -> None:
        """
        Sets the significance level for confidence intervals.
        
        Parameters
        ----------
        alpha : float
            Significance level (between 0 and 1)
        """
        # Validate alpha
        if not isinstance(alpha, float) or not 0 < alpha < 1:
            raise ValueError("Alpha must be a float between 0 and 1")
        
        # Store alpha
        self._alpha = alpha
        
        # Update ACF plot
        self._acf_plot.set_alpha(alpha)
        
        # Schedule update if data is available
        if self._data is not None:
            self.schedule_update()
    
    def update_plot(self) -> None:
        """
        Updates the ACF plot.
        """
        if self._data is None:
            logger.warning("Cannot update plot: No data available")
            return
        
        self._acf_plot.update_plot()
        self.updated.emit()
    
    def clear(self) -> None:
        """
        Clears the ACF plot.
        """
        self._data = None
        self._acf_plot.clear()
        self.cleared.emit()


class PACFPlotWidget(BasePlotWidget):
    """
    Widget for displaying partial autocorrelation function plots.
    
    This widget provides visualization of partial autocorrelation functions
    with configurable lags, confidence intervals, and calculation methods.
    """
    
    # Define signals
    dataChanged = pyqtSignal()
    
    def __init__(self, parent=None):
        """
        Initializes the PACF plot widget.
        
        Parameters
        ----------
        parent : QWidget, optional
            Parent widget
        """
        super().__init__(parent)
        
        # Initialize data properties
        self._data = None
        self._nlags = 20
        self._alpha = 0.05
        self._method = "ywmle"
        
        # Load default parameters
        self._pacf_params = DEFAULT_PACF_PARAMS.copy()
        
        # Set initialization flag
        self._initialized = False
        
        # Initialize figure
        self._figure = None
        
        # Set default plot parameters
        self._plot_params = {
            "title": "Partial Autocorrelation Function",
            "grid": True,
            "bar_color": "blue",
            "alpha": 0.05,
        }
    
    def set_data(self, data: np.ndarray) -> None:
        """
        Sets the data for PACF calculation.
        
        Parameters
        ----------
        data : np.ndarray
            The time series data
        """
        # Validate data
        if not isinstance(data, np.ndarray):
            data = np.asarray(data)
        
        # Store data
        self._data = data
        
        # Schedule update
        self.schedule_update()
        
        # Emit signal
        self.dataChanged.emit()
    
    def set_nlags(self, nlags: int) -> None:
        """
        Sets the number of lags for PACF calculation.
        
        Parameters
        ----------
        nlags : int
            Number of lags to display
        """
        # Validate nlags
        if not isinstance(nlags, int) or nlags <= 0:
            raise ValueError("Number of lags must be a positive integer")
        
        # Store nlags
        self._nlags = nlags
        self._pacf_params["nlags"] = nlags
        
        # Schedule update if data is available
        if self._data is not None:
            self.schedule_update()
    
    def set_alpha(self, alpha: float) -> None:
        """
        Sets the significance level for confidence intervals.
        
        Parameters
        ----------
        alpha : float
            Significance level (between 0 and 1)
        """
        # Validate alpha
        if not isinstance(alpha, float) or not 0 < alpha < 1:
            raise ValueError("Alpha must be a float between 0 and 1")
        
        # Store alpha
        self._alpha = alpha
        self._pacf_params["alpha"] = alpha
        
        # Schedule update if data is available
        if self._data is not None:
            self.schedule_update()
    
    def set_method(self, method: str) -> None:
        """
        Sets the PACF calculation method.
        
        Parameters
        ----------
        method : str
            Method to use for PACF calculation
        """
        # Validate method
        valid_methods = ["ywmle", "ols", "yw", "ld", "ldb"]
        if method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")
        
        # Store method
        self._method = method
        self._pacf_params["method"] = method
        
        # Schedule update if data is available
        if self._data is not None:
            self.schedule_update()
    
    def update_plot(self) -> None:
        """
        Updates the PACF plot.
        """
        if self._data is None:
            logger.warning("Cannot update plot: No data available")
            return
        
        # Create PACF figure
        self._figure = create_pacf_figure(
            self._data,
            self._nlags,
            self._alpha,
            self._method,
            self._pacf_params
        )
        
        # Replace current figure in plot widget
        self._plot_widget.get_figure().clf()
        for ax in self._figure.get_axes():
            self._plot_widget.get_figure().add_subplot(ax)
        
        # Update the plot
        self._plot_widget.update_plot()
        self._initialized = True
        
        self.updated.emit()
    
    async def async_update_plot(self) -> None:
        """
        Updates the PACF plot asynchronously.
        """
        if self._data is None:
            logger.warning("Cannot update plot: No data available")
            return
        
        # Yield control back to event loop
        await asyncio.sleep(0)
        
        # Create PACF figure
        self._figure = create_pacf_figure(
            self._data,
            self._nlags,
            self._alpha,
            self._method,
            self._pacf_params
        )
        
        # Replace current figure in plot widget
        self._plot_widget.get_figure().clf()
        for ax in self._figure.get_axes():
            self._plot_widget.get_figure().add_subplot(ax)
        
        # Update the plot asynchronously
        await self._plot_widget.async_update_plot()
        self._initialized = True
        
        self.updated.emit()
    
    def clear(self) -> None:
        """
        Clears the PACF plot.
        """
        self._data = None
        self._initialized = False
        super().clear()
        self.cleared.emit()
    
    def set_plot_params(self, params: Dict[str, Any]) -> None:
        """
        Sets plot parameters for PACF visualization.
        
        Parameters
        ----------
        params : Dict[str, Any]
            Dictionary of plot parameters
        """
        # Validate params
        if not isinstance(params, dict):
            raise TypeError("Plot parameters must be a dictionary")
        
        # Update PACF parameters
        self._pacf_params.update(params)
        
        # Schedule update if data is available
        if self._data is not None:
            self.schedule_update()


class DiagnosticPlotsWidget(QWidget):
    """
    Composite widget that combines multiple diagnostic plots in a grid layout.
    
    This widget provides a comprehensive view of model diagnostics by combining
    residual plots, ACF, PACF, and time series plots in a single widget.
    """
    
    # Define signals
    dataChanged = pyqtSignal()
    residualsChanged = pyqtSignal()
    fittedValuesChanged = pyqtSignal()
    allPlotsUpdated = pyqtSignal()
    allPlotsCleared = pyqtSignal()
    updateStarted = pyqtSignal()
    
    def __init__(self, parent=None):
        """
        Initializes the diagnostic plots widget with a grid of plots.
        
        Parameters
        ----------
        parent : QWidget, optional
            Parent widget
        """
        super().__init__(parent)
        
        # Create grid layout
        self._grid_layout = QGridLayout(self)
        self.setLayout(self._grid_layout)
        
        # Initialize worker manager
        self._worker_manager = WorkerManager()
        
        # Create plot widgets
        self._residual_plot = ResidualPlotWidget(self)
        self._acf_plot = ACFPlotWidget(self)
        self._pacf_plot = PACFPlotWidget(self)
        self._ts_plot = TimeSeriesPlotWidget(self)
        
        # Add plots to grid layout
        self._grid_layout.addWidget(self._residual_plot, 0, 0)
        self._grid_layout.addWidget(self._acf_plot, 0, 1)
        self._grid_layout.addWidget(self._ts_plot, 1, 0)
        self._grid_layout.addWidget(self._pacf_plot, 1, 1)
        
        # Set layout spacing
        self._grid_layout.setSpacing(PLOT_SPACING)
        
        # Initialize data properties
        self._data = None
        self._residuals = None
        self._fitted_values = None
        
        # Connect signals
        self._residual_plot.dataChanged.connect(lambda: self.residualsChanged.emit())
        self._ts_plot.dataChanged.connect(lambda: self.dataChanged.emit())
        
        # Set size policy and minimum size
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(DEFAULT_PLOT_WIDTH * 2, DEFAULT_PLOT_HEIGHT * 2)
    
    def set_data(self, data: np.ndarray, dates: pd.DatetimeIndex = None) -> None:
        """
        Sets the original time series data.
        
        Parameters
        ----------
        data : np.ndarray
            The time series data
        dates : pd.DatetimeIndex, optional
            The corresponding dates for the data
        """
        # Validate data
        if not isinstance(data, np.ndarray):
            data = np.asarray(data)
        
        # Store data
        self._data = data
        
        # Update time series plot
        self._ts_plot.set_data(data, dates)
        
        # Update ACF and PACF plots if residuals are not available
        if self._residuals is None:
            self._acf_plot.set_data(data)
            self._pacf_plot.set_data(data)
        
        # Emit signal
        self.dataChanged.emit()
    
    def set_residuals(self, residuals: np.ndarray) -> None:
        """
        Sets the model residuals.
        
        Parameters
        ----------
        residuals : np.ndarray
            The model residuals
        """
        # Validate residuals
        if not isinstance(residuals, np.ndarray):
            residuals = np.asarray(residuals)
        
        # Store residuals
        self._residuals = residuals
        
        # Update residual plot
        self._residual_plot.set_residuals(residuals)
        
        # Update ACF and PACF plots
        self._acf_plot.set_data(residuals)
        self._pacf_plot.set_data(residuals)
        
        # Schedule updates
        self._residual_plot.schedule_update()
        self._acf_plot.schedule_update()
        self._pacf_plot.schedule_update()
        
        # Emit signal
        self.residualsChanged.emit()
    
    def set_fitted_values(self, fitted_values: np.ndarray) -> None:
        """
        Sets the model fitted values.
        
        Parameters
        ----------
        fitted_values : np.ndarray
            The fitted values from the model
        """
        # Validate fitted values
        if not isinstance(fitted_values, np.ndarray):
            fitted_values = np.asarray(fitted_values)
        
        # Store fitted values
        self._fitted_values = fitted_values
        
        # Update residual plot
        self._residual_plot.set_fitted_values(fitted_values)
        
        # Schedule update
        self._residual_plot.schedule_update()
        
        # Emit signal
        self.fittedValuesChanged.emit()
    
    def update_all_plots(self) -> None:
        """
        Updates all diagnostic plots.
        """
        # Update time series plot
        if self._data is not None:
            self._ts_plot.update_plot()
        
        # Update residual plot
        if self._residuals is not None:
            self._residual_plot.update_plot()
        
        # Update ACF plot
        if self._residuals is not None:
            self._acf_plot.update_plot()
        
        # Update PACF plot
        if self._residuals is not None:
            self._pacf_plot.update_plot()
        
        # Emit signal
        self.allPlotsUpdated.emit()
    
    async def async_update_all_plots(self) -> None:
        """
        Asynchronously updates all diagnostic plots.
        """
        # Update time series plot
        if self._data is not None:
            await self._ts_plot.async_update_plot()
        
        # Update residual plot
        if self._residuals is not None:
            await self._residual_plot.async_update_plot()
        
        # Update ACF plot
        if self._residuals is not None:
            await self._acf_plot.async_update_plot()
        
        # Update PACF plot
        if self._residuals is not None:
            await self._pacf_plot.async_update_plot()
        
        # Emit signal
        self.allPlotsUpdated.emit()
    
    def schedule_update_all(self) -> None:
        """
        Schedules asynchronous update of all plots.
        """
        # Create async worker for plot update
        worker = self._worker_manager.create_async_worker(self.async_update_all_plots)
        
        # Connect signals
        worker.signals.started.connect(lambda: self.updateStarted.emit())
        worker.signals.finished.connect(lambda: self.allPlotsUpdated.emit())
        worker.signals.error.connect(lambda e: logger.error(f"Error updating plots: {str(e)}"))
        
        # Start the worker
        self._worker_manager.start_worker(worker)
        self.updateStarted.emit()
    
    def clear_all(self) -> None:
        """
        Clears all diagnostic plots.
        """
        # Clear data
        self._data = None
        self._residuals = None
        self._fitted_values = None
        
        # Clear all plots
        self._ts_plot.clear()
        self._residual_plot.clear()
        self._acf_plot.clear()
        self._pacf_plot.clear()
        
        # Emit signal
        self.allPlotsCleared.emit()
    
    def get_residual_stats(self) -> Dict[str, Any]:
        """
        Gets statistical information about residuals.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary of residual statistics
        """
        return self._residual_plot.get_residual_stats()
    
    def save_plots(self, basename: str, format: str = "png", dpi: int = 300) -> Dict[str, bool]:
        """
        Saves all plots to files with a common basename.
        
        Parameters
        ----------
        basename : str
            Base filename to use for all plots
        format : str, optional
            File format (e.g., 'png', 'pdf', 'svg'), default is 'png'
        dpi : int, optional
            Resolution in dots per inch, default is 300
            
        Returns
        -------
        Dict[str, bool]
            Dictionary with success status for each plot
        """
        results = {}
        
        # Save time series plot
        ts_filename = f"{basename}_timeseries.{format}"
        results["timeseries"] = self._ts_plot.save_plot(ts_filename, dpi)
        
        # Save residual plot
        residual_filename = f"{basename}_residuals.{format}"
        results["residuals"] = self._residual_plot.save_plot(residual_filename, dpi)
        
        # Save ACF plot
        acf_filename = f"{basename}_acf.{format}"
        results["acf"] = self._acf_plot.save_plot(acf_filename, dpi)
        
        # Save PACF plot
        pacf_filename = f"{basename}_pacf.{format}"
        results["pacf"] = self._pacf_plot.save_plot(pacf_filename, dpi)
        
        return results