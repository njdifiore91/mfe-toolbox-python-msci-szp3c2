"""
MFE Toolbox - Volatility Plot Module

This module provides interactive PyQt6 widgets for visualizing volatility models and forecasts.
It includes components for displaying returns, volatility series, and forecasts with
confidence intervals, supporting both synchronous and asynchronous updates.

The module integrates with the MFE Toolbox's volatility models to provide a seamless
visualization experience for financial risk analysis.
"""

import numpy as np  # numpy 1.26.3
import matplotlib.pyplot as plt  # matplotlib 3.7.1
import pandas as pd  # pandas 2.1.4
from PyQt6.QtCore import QObject, pyqtSignal, QSize  # PyQt6 6.6.1
from PyQt6.QtWidgets import QWidget, QVBoxLayout  # PyQt6 6.6.1
import logging  # Python standard library
import asyncio  # Python standard library
from typing import Union, Optional, List, Tuple, Dict, Any  # Python standard library

# Internal imports
from ..plots.matplotlib_backend import AsyncPlotWidget
from ..async.signals import PlotSignals
from ...backend.mfe.utils.validation import validate_array
from ...backend.mfe.models.volatility import VolatilityModel

# Configure logger
logger = logging.getLogger(__name__)

# Default plot parameters
DEFAULT_VOLATILITY_PLOT_PARAMS = {
    "title": "Volatility Plot",
    "grid": True,
    "returns_color": "black",
    "volatility_color": "red",
    "forecast_color": "blue",
    "line_width": 1.5, 
    "alpha": 0.8,
    "show_returns": True,
    "show_forecast": True,
    "use_percent": True,
    "legend": True,
    "confidence_interval": 0.95,
    "display_range": None  # Optional date range to show
}


def create_volatility_figure(
    returns: np.ndarray,
    volatility: np.ndarray,
    forecast: Optional[np.ndarray] = None,
    dates: Optional[pd.DatetimeIndex] = None,
    plot_params: dict = None
) -> plt.Figure:
    """
    Creates a matplotlib figure configured for volatility visualization.
    
    Parameters
    ----------
    returns : np.ndarray
        Array of return series data
    volatility : np.ndarray
        Array of volatility series data
    forecast : Optional[np.ndarray], default=None
        Optional array of volatility forecast data
    dates : Optional[pd.DatetimeIndex], default=None
        Optional array of dates for x-axis
    plot_params : dict, default=None
        Dictionary of plot parameters
        
    Returns
    -------
    plt.Figure
        A configured matplotlib figure with volatility plot
    """
    # Validate inputs
    returns = validate_array(returns, param_name="returns")
    volatility = validate_array(volatility, param_name="volatility")
    if forecast is not None:
        forecast = validate_array(forecast, param_name="forecast")
    
    # Apply default parameters if not provided
    if plot_params is None:
        plot_params = DEFAULT_VOLATILITY_PLOT_PARAMS.copy()
    else:
        # Merge with defaults
        default_params = DEFAULT_VOLATILITY_PLOT_PARAMS.copy()
        default_params.update(plot_params)
        plot_params = default_params
    
    # Create figure and axis
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    
    # Set up x-axis values
    if dates is not None:
        x_values = dates
    else:
        x_values = np.arange(len(returns))
    
    # Scale to percentage if requested
    scale_factor = 100 if plot_params["use_percent"] else 1
    
    # Plot returns if show_returns is True
    if plot_params["show_returns"]:
        ax.plot(
            x_values,
            returns * scale_factor,
            color=plot_params["returns_color"],
            linewidth=plot_params["line_width"] * 0.5,
            alpha=plot_params["alpha"] * 0.5,
            label="Returns"
        )
    
    # Plot volatility
    ax.plot(
        x_values,
        volatility * scale_factor,
        color=plot_params["volatility_color"],
        linewidth=plot_params["line_width"],
        alpha=plot_params["alpha"],
        label="Volatility"
    )
    
    # Plot forecast if available and show_forecast is True
    if forecast is not None and plot_params["show_forecast"]:
        # For forecast, use the last len(forecast) data points
        if dates is not None:
            forecast_dates = dates[-len(forecast):]
        else:
            forecast_dates = np.arange(len(returns) - len(forecast), len(returns))
        
        ax.plot(
            forecast_dates,
            forecast * scale_factor,
            color=plot_params["forecast_color"],
            linewidth=plot_params["line_width"],
            alpha=plot_params["alpha"],
            label="Forecast"
        )
        
        # Add confidence intervals if specified
        if "confidence_interval" in plot_params and plot_params["confidence_interval"] > 0:
            ci = plot_params["confidence_interval"]
            lower, upper = calculate_confidence_intervals(forecast, ci, "normal")
            
            ax.fill_between(
                forecast_dates,
                lower * scale_factor,
                upper * scale_factor,
                color=plot_params["forecast_color"],
                alpha=0.2,
                label=f"{int(ci*100)}% Confidence"
            )
    
    # Set title and labels
    ax.set_title(plot_params["title"])
    ax.set_xlabel("Time" if dates is None else "")
    ax.set_ylabel(f"Value ({'%' if plot_params['use_percent'] else ''})")
    
    # Configure grid
    ax.grid(plot_params["grid"])
    
    # Add legend if requested
    if plot_params["legend"]:
        ax.legend()
    
    # Format date axis if dates provided
    if dates is not None:
        fig.autofmt_xdate()
    
    # Set display range if specified
    if plot_params["display_range"] is not None and dates is not None:
        start_date, end_date = plot_params["display_range"]
        ax.set_xlim(start_date, end_date)
    
    # Adjust layout for optimal display
    fig.tight_layout()
    
    return fig


def create_forecast_figure(
    forecast: np.ndarray,
    confidence_intervals: Optional[np.ndarray] = None,
    forecast_dates: Optional[pd.DatetimeIndex] = None,
    plot_params: dict = None
) -> plt.Figure:
    """
    Creates a figure specifically for volatility forecasting visualization.
    
    Parameters
    ----------
    forecast : np.ndarray
        Array of volatility forecast data
    confidence_intervals : Optional[np.ndarray], default=None
        Optional array of confidence interval bounds, shape (2, len(forecast))
        where [0, :] is lower bounds and [1, :] is upper bounds
    forecast_dates : Optional[pd.DatetimeIndex], default=None
        Optional array of dates for the forecast period
    plot_params : dict, default=None
        Dictionary of plot parameters
        
    Returns
    -------
    plt.Figure
        A configured matplotlib figure with volatility forecast
    """
    # Validate inputs
    forecast = validate_array(forecast, param_name="forecast")
    
    # Apply default parameters if not provided
    if plot_params is None:
        plot_params = DEFAULT_VOLATILITY_PLOT_PARAMS.copy()
    else:
        # Merge with defaults
        default_params = DEFAULT_VOLATILITY_PLOT_PARAMS.copy()
        default_params.update(plot_params)
        plot_params = default_params
    
    # Create figure and axis
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    
    # Set up x-axis values
    if forecast_dates is not None:
        x_values = forecast_dates
    else:
        x_values = np.arange(len(forecast))
    
    # Scale to percentage if requested
    scale_factor = 100 if plot_params["use_percent"] else 1
    
    # Plot forecast
    ax.plot(
        x_values,
        forecast * scale_factor,
        color=plot_params["forecast_color"],
        linewidth=plot_params["line_width"],
        alpha=plot_params["alpha"],
        label="Forecast"
    )
    
    # Add confidence intervals if provided
    if confidence_intervals is not None:
        lower, upper = confidence_intervals
        
        ax.fill_between(
            x_values,
            lower * scale_factor,
            upper * scale_factor,
            color=plot_params["forecast_color"],
            alpha=0.2,
            label=f"{int(plot_params['confidence_interval']*100)}% Confidence"
        )
    
    # Set title and labels
    ax.set_title(plot_params.get("title", "Volatility Forecast"))
    ax.set_xlabel("Time" if forecast_dates is None else "")
    ax.set_ylabel(f"Volatility ({'%' if plot_params['use_percent'] else ''})")
    
    # Configure grid
    ax.grid(plot_params["grid"])
    
    # Add legend if requested
    if plot_params["legend"]:
        ax.legend()
    
    # Format date axis if dates provided
    if forecast_dates is not None:
        fig.autofmt_xdate()
    
    # Adjust layout for optimal display
    fig.tight_layout()
    
    return fig


def calculate_confidence_intervals(
    forecast: np.ndarray,
    confidence_level: float,
    distribution: str = "normal"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates confidence intervals for volatility forecasts.
    
    Parameters
    ----------
    forecast : np.ndarray
        Array of forecast volatility values
    confidence_level : float
        Confidence level (0 to 1)
    distribution : str, default="normal"
        Distribution to use for calculating intervals
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Lower and upper confidence bounds
    """
    # Validate inputs
    forecast = validate_array(forecast, param_name="forecast")
    
    if not 0 <= confidence_level <= 1:
        raise ValueError(f"Confidence level must be between 0 and 1, got {confidence_level}")
    
    # Calculate alpha for two-tailed intervals
    alpha = 1 - confidence_level
    
    # Calculate distribution quantiles
    if distribution.lower() == "normal":
        # For normal distribution
        import scipy.stats as stats
        z_score = stats.norm.ppf(alpha/2)
        
        # Symmetrical intervals for normal distribution
        lower_bound = forecast * (1 + z_score)
        upper_bound = forecast * (1 - z_score)
        
    elif distribution.lower() == "t":
        # For t distribution
        import scipy.stats as stats
        df = 5  # Default degrees of freedom
        t_score = stats.t.ppf(alpha/2, df)
        
        # Apply t-distribution intervals
        lower_bound = forecast * (1 + t_score)
        upper_bound = forecast * (1 - t_score)
        
    elif distribution.lower() == "skewed_t":
        # For skewed t-distribution
        # This is a simplified approximation
        import scipy.stats as stats
        df = 5  # Default degrees of freedom
        skew = 0.1  # Default skewness
        
        lower_quantile = stats.t.ppf(alpha/2, df) * (1 - skew)
        upper_quantile = stats.t.ppf(1-alpha/2, df) * (1 + skew)
        
        lower_bound = forecast * (1 + lower_quantile/np.sqrt(df))
        upper_bound = forecast * (1 + upper_quantile/np.sqrt(df))
    
    else:
        raise ValueError(f"Unsupported distribution: {distribution}")
    
    # Ensure non-negative lower bounds (volatility can't be negative)
    lower_bound = np.maximum(lower_bound, 0)
    
    return lower_bound, upper_bound


class VolatilityPlot(QWidget):
    """
    Interactive PyQt6 widget for displaying and manipulating volatility plots with
    asynchronous update capabilities.
    """
    
    def __init__(
        self,
        parent: QWidget = None,
        returns: Optional[np.ndarray] = None,
        volatility: Optional[np.ndarray] = None,
        forecast: Optional[np.ndarray] = None,
        dates: Optional[pd.DatetimeIndex] = None,
        plot_params: Optional[dict] = None
    ):
        """
        Initializes the volatility plot widget with default configuration.
        
        Parameters
        ----------
        parent : QWidget, default=None
            Parent widget
        returns : Optional[np.ndarray], default=None
            Returns series data
        volatility : Optional[np.ndarray], default=None
            Volatility series data
        forecast : Optional[np.ndarray], default=None
            Volatility forecast data
        dates : Optional[pd.DatetimeIndex], default=None
            Dates for the time series
        plot_params : Optional[dict], default=None
            Plot configuration parameters
        """
        # Initialize parent QWidget
        super().__init__(parent)
        
        # Initialize signals
        self.signals = PlotSignals()
        
        # Initialize data properties
        self._returns = returns
        self._volatility = volatility
        self._forecast = forecast
        self._dates = dates
        
        # Initialize plot parameters
        self._plot_params = DEFAULT_VOLATILITY_PLOT_PARAMS.copy()
        if plot_params is not None:
            self._plot_params.update(plot_params)
        
        # Create layout for widget
        self._layout = QVBoxLayout(self)
        self.setLayout(self._layout)
        
        # Create plot widget for matplotlib integration
        self._plot_widget = AsyncPlotWidget(self)
        self._layout.addWidget(self._plot_widget)
        
        # Initialize plot control flag
        self._initialized = False
        
        # Plot data if all required data is provided
        if returns is not None and volatility is not None:
            self.update_plot()
        
        # Log initialization
        logger.debug("VolatilityPlot widget initialized")
    
    def set_data(
        self,
        returns: np.ndarray,
        volatility: np.ndarray,
        forecast: Optional[np.ndarray] = None,
        dates: Optional[pd.DatetimeIndex] = None
    ) -> None:
        """
        Sets the returns and volatility data to be visualized.
        
        Parameters
        ----------
        returns : np.ndarray
            Returns series data
        volatility : np.ndarray
            Volatility series data
        forecast : Optional[np.ndarray], default=None
            Volatility forecast data
        dates : Optional[pd.DatetimeIndex], default=None
            Dates for the time series
        """
        # Validate inputs
        self._returns = validate_array(returns, param_name="returns")
        self._volatility = validate_array(volatility, param_name="volatility")
        
        # Check that lengths match
        if len(self._returns) != len(self._volatility):
            raise ValueError(
                f"Returns and volatility must have the same length, got {len(self._returns)} and {len(self._volatility)}"
            )
        
        # Set optional forecast data
        if forecast is not None:
            self._forecast = validate_array(forecast, param_name="forecast")
        else:
            self._forecast = None
        
        # Set dates
        self._dates = dates
        
        # Update plot if already initialized
        if self._initialized:
            self.update_plot()
        
        # Log data update
        logger.debug(f"VolatilityPlot data updated: {len(self._returns)} data points")
    
    def set_from_model(
        self,
        model: VolatilityModel,
        returns: np.ndarray,
        forecast_horizon: Optional[int] = None,
        dates: Optional[pd.DatetimeIndex] = None
    ) -> None:
        """
        Sets plot data directly from a volatility model.
        
        Parameters
        ----------
        model : VolatilityModel
            The volatility model to use
        returns : np.ndarray
            Returns series data
        forecast_horizon : Optional[int], default=None
            Number of periods to forecast
        dates : Optional[pd.DatetimeIndex], default=None
            Dates for the time series
        """
        # Validate returns data
        returns = validate_array(returns, param_name="returns")
        
        # Calculate volatility from model
        volatility = model.calculate_variance(returns)
        if isinstance(volatility, np.ndarray):
            volatility = np.sqrt(volatility)  # Convert variance to volatility
        
        # Calculate forecast if requested
        forecast = None
        if forecast_horizon is not None and forecast_horizon > 0:
            forecast_variance = model.forecast(returns, forecast_horizon)
            if isinstance(forecast_variance, np.ndarray):
                forecast = np.sqrt(forecast_variance)  # Convert variance to volatility
        
        # Set the data
        self.set_data(returns, volatility, forecast, dates)
        
        # Log model data extraction
        logger.debug(f"Data extracted from volatility model with {len(returns)} data points")
    
    def set_plot_params(self, params: dict) -> None:
        """
        Sets plotting parameters to customize volatility plot appearance.
        
        Parameters
        ----------
        params : dict
            Dictionary of plot parameters to update
        """
        if not isinstance(params, dict):
            raise TypeError("Plot parameters must be a dictionary")
        
        # Update parameters
        self._plot_params.update(params)
        
        # Update plot if data is available
        if self._returns is not None and self._volatility is not None:
            self.update_plot()
        
        # Log parameter updates
        logger.debug(f"Plot parameters updated: {list(params.keys())}")
    
    def update_plot(self) -> None:
        """
        Updates the volatility plot with current data and settings.
        """
        # Check if required data is available
        if self._returns is None or self._volatility is None:
            logger.warning("Cannot update plot: missing required data")
            return
        
        # Signal update started
        self.signals.update_started.emit()
        
        try:
            # Clear existing plot
            self._plot_widget.clear_plot()
            
            # Create volatility figure
            fig = create_volatility_figure(
                self._returns,
                self._volatility,
                self._forecast,
                self._dates,
                self._plot_params
            )
            
            # Set the figure in the plot widget
            plot_fig = self._plot_widget.get_figure()
            plot_fig.clear()
            
            # Copy content from created figure to the widget's figure
            for ax in fig.get_axes():
                # Get axis' position
                pos = ax.get_position()
                # Create new axis in the same position
                new_ax = plot_fig.add_axes(pos)
                # Copy all artists
                for line in ax.get_lines():
                    new_ax.add_line(line)
                for collection in ax.collections:
                    new_ax.add_collection(collection)
                # Copy layout
                new_ax.set_title(ax.get_title())
                new_ax.set_xlabel(ax.get_xlabel())
                new_ax.set_ylabel(ax.get_ylabel())
                new_ax.grid(ax.get_grid())
                # Copy legend if exists
                if ax.get_legend() is not None:
                    new_ax.legend()
                # Copy limits
                new_ax.set_xlim(ax.get_xlim())
                new_ax.set_ylim(ax.get_ylim())
            
            # Update the plot widget display
            self._plot_widget.update_plot()
            
            # Set initialized flag
            self._initialized = True
            
            # Signal update complete
            self.signals.update_complete.emit(self)
            
            # Log update
            logger.debug("Volatility plot updated successfully")
            
        except Exception as e:
            # Signal update error
            self.signals.update_error.emit(e)
            
            # Log error
            logger.error(f"Error updating volatility plot: {str(e)}")
    
    def update_forecast_plot(
        self,
        forecast: np.ndarray,
        forecast_dates: Optional[pd.DatetimeIndex] = None
    ) -> None:
        """
        Updates only the forecast part of the volatility plot.
        
        Parameters
        ----------
        forecast : np.ndarray
            Forecast volatility data
        forecast_dates : Optional[pd.DatetimeIndex], default=None
            Dates for the forecast period
        """
        # Validate forecast array
        self._forecast = validate_array(forecast, param_name="forecast")
        
        # Update forecast dates if provided
        if forecast_dates is not None:
            # If main dates are not set, use forecast_dates as _dates
            if self._dates is None:
                self._dates = forecast_dates
        
        # Update the plot
        self.update_plot()
        
        # Log forecast update
        logger.debug(f"Forecast updated with {len(forecast)} data points")
    
    async def async_update_plot(self) -> None:
        """
        Asynchronously updates the volatility plot to prevent UI blocking.
        """
        # Check if required data is available
        if self._returns is None or self._volatility is None:
            logger.warning("Cannot update plot: missing required data")
            return
        
        # Signal update started
        self.signals.update_started.emit()
        
        try:
            # Yield control to allow UI updates
            await asyncio.sleep(0)
            
            # Clear existing plot
            self._plot_widget.clear_plot()
            
            # Create volatility figure
            fig = create_volatility_figure(
                self._returns,
                self._volatility,
                self._forecast,
                self._dates,
                self._plot_params
            )
            
            # Set the figure in the plot widget - same as in update_plot
            plot_fig = self._plot_widget.get_figure()
            plot_fig.clear()
            
            # Copy content from created figure to the widget's figure
            for ax in fig.get_axes():
                # Get axis' position
                pos = ax.get_position()
                # Create new axis in the same position
                new_ax = plot_fig.add_axes(pos)
                # Copy all artists
                for line in ax.get_lines():
                    new_ax.add_line(line)
                for collection in ax.collections:
                    new_ax.add_collection(collection)
                # Copy layout
                new_ax.set_title(ax.get_title())
                new_ax.set_xlabel(ax.get_xlabel())
                new_ax.set_ylabel(ax.get_ylabel())
                new_ax.grid(ax.get_grid())
                # Copy legend if exists
                if ax.get_legend() is not None:
                    new_ax.legend()
                # Copy limits
                new_ax.set_xlim(ax.get_xlim())
                new_ax.set_ylim(ax.get_ylim())
            
            # Update the plot widget display asynchronously
            await self._plot_widget.async_update_plot()
            
            # Set initialized flag
            self._initialized = True
            
            # Signal update complete
            self.signals.update_complete.emit(self)
            
            # Log async update
            logger.debug("Volatility plot updated asynchronously")
            
        except Exception as e:
            # Signal update error
            self.signals.update_error.emit(e)
            
            # Log error
            logger.error(f"Error in async update of volatility plot: {str(e)}")
    
    def save_figure(self, filepath: str, dpi: Optional[int] = None) -> bool:
        """
        Saves the current volatility plot to a file.
        
        Parameters
        ----------
        filepath : str
            Path to save the figure
        dpi : Optional[int], default=None
            Resolution for the saved figure
            
        Returns
        -------
        bool
            True if save was successful, False otherwise
        """
        # Check if plot is initialized
        if not self._initialized:
            logger.warning("Cannot save: plot not initialized")
            return False
        
        # Save the figure
        result = self._plot_widget.save_plot(filepath, dpi)
        
        # Log result
        if result:
            logger.info(f"Figure saved to {filepath}")
        else:
            logger.error(f"Failed to save figure to {filepath}")
        
        return result
    
    def clear(self) -> None:
        """
        Clears the current plot and resets widget state.
        """
        # Reset data
        self._returns = None
        self._volatility = None
        self._forecast = None
        self._dates = None
        
        # Clear plot
        self._plot_widget.clear_plot()
        
        # Reset initialized flag
        self._initialized = False
        
        # Log clear operation
        logger.debug("Volatility plot cleared")
    
    def set_title(self, title: str) -> None:
        """
        Sets the plot title.
        
        Parameters
        ----------
        title : str
            New plot title
        """
        self._plot_params["title"] = title
        
        # Update plot if initialized
        if self._initialized:
            self.update_plot()
        
        # Log title change
        logger.debug(f"Plot title set to: {title}")
    
    def set_grid(self, visible: bool) -> None:
        """
        Toggles the plot grid visibility.
        
        Parameters
        ----------
        visible : bool
            Whether to show grid
        """
        self._plot_params["grid"] = visible
        
        # Update plot if initialized
        if self._initialized:
            self.update_plot()
        
        # Log grid visibility change
        logger.debug(f"Grid visibility set to: {visible}")
    
    def toggle_returns_visibility(self, visible: bool) -> None:
        """
        Toggles the visibility of returns series in the plot.
        
        Parameters
        ----------
        visible : bool
            Whether to show returns
        """
        self._plot_params["show_returns"] = visible
        
        # Update plot if initialized
        if self._initialized:
            self.update_plot()
        
        # Log visibility change
        logger.debug(f"Returns visibility set to: {visible}")
    
    def toggle_forecast_visibility(self, visible: bool) -> None:
        """
        Toggles the visibility of forecast series in the plot.
        
        Parameters
        ----------
        visible : bool
            Whether to show forecast
        """
        self._plot_params["show_forecast"] = visible
        
        # Update plot if initialized and forecast exists
        if self._initialized and self._forecast is not None:
            self.update_plot()
        
        # Log visibility change
        logger.debug(f"Forecast visibility set to: {visible}")
    
    def set_confidence_interval(self, level: float) -> None:
        """
        Sets the confidence interval level for forecast visualization.
        
        Parameters
        ----------
        level : float
            Confidence level (0 to 1)
        """
        if not 0 <= level <= 1:
            raise ValueError(f"Confidence level must be between 0 and 1, got {level}")
        
        self._plot_params["confidence_interval"] = level
        
        # Update plot if initialized and forecast exists
        if self._initialized and self._forecast is not None:
            self.update_plot()
        
        # Log confidence interval change
        logger.debug(f"Confidence interval set to: {level}")
    
    def get_figure(self) -> Optional[plt.Figure]:
        """
        Returns the matplotlib figure for direct access.
        
        Returns
        -------
        Optional[plt.Figure]
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


class VolatilityForecastPlot(QWidget):
    """
    Specialized plot widget for visualizing volatility forecasts with confidence intervals.
    """
    
    def __init__(
        self,
        parent: QWidget = None,
        forecast: Optional[np.ndarray] = None,
        lower_bound: Optional[np.ndarray] = None,
        upper_bound: Optional[np.ndarray] = None,
        forecast_dates: Optional[pd.DatetimeIndex] = None,
        plot_params: Optional[dict] = None
    ):
        """
        Initializes the volatility forecast plot widget with default configuration.
        
        Parameters
        ----------
        parent : QWidget, default=None
            Parent widget
        forecast : Optional[np.ndarray], default=None
            Forecast volatility data
        lower_bound : Optional[np.ndarray], default=None
            Lower confidence bound
        upper_bound : Optional[np.ndarray], default=None
            Upper confidence bound
        forecast_dates : Optional[pd.DatetimeIndex], default=None
            Dates for the forecast period
        plot_params : Optional[dict], default=None
            Plot configuration parameters
        """
        # Initialize parent QWidget
        super().__init__(parent)
        
        # Initialize signals
        self.signals = PlotSignals()
        
        # Initialize data properties
        self._forecast = forecast
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        self._forecast_dates = forecast_dates
        
        # Initialize plot parameters
        self._plot_params = DEFAULT_VOLATILITY_PLOT_PARAMS.copy()
        if plot_params is not None:
            self._plot_params.update(plot_params)
        
        # Set default title for forecast plot
        if "title" not in self._plot_params or self._plot_params["title"] == DEFAULT_VOLATILITY_PLOT_PARAMS["title"]:
            self._plot_params["title"] = "Volatility Forecast"
        
        # Create layout for widget
        self._layout = QVBoxLayout(self)
        self.setLayout(self._layout)
        
        # Create plot widget for matplotlib integration
        self._plot_widget = AsyncPlotWidget(self)
        self._layout.addWidget(self._plot_widget)
        
        # Initialize plot control flag
        self._initialized = False
        
        # Plot data if forecast is provided
        if forecast is not None:
            self.update_plot()
        
        # Log initialization
        logger.debug("VolatilityForecastPlot widget initialized")
    
    def set_forecast_data(
        self,
        forecast: np.ndarray,
        lower_bound: Optional[np.ndarray] = None,
        upper_bound: Optional[np.ndarray] = None,
        forecast_dates: Optional[pd.DatetimeIndex] = None
    ) -> None:
        """
        Sets the forecast data and confidence intervals to be visualized.
        
        Parameters
        ----------
        forecast : np.ndarray
            Forecast volatility data
        lower_bound : Optional[np.ndarray], default=None
            Lower confidence bound
        upper_bound : Optional[np.ndarray], default=None
            Upper confidence bound
        forecast_dates : Optional[pd.DatetimeIndex], default=None
            Dates for the forecast period
        """
        # Validate forecast array
        self._forecast = validate_array(forecast, param_name="forecast")
        
        # Set confidence bounds if provided
        if lower_bound is not None and upper_bound is not None:
            self._lower_bound = validate_array(lower_bound, param_name="lower_bound")
            self._upper_bound = validate_array(upper_bound, param_name="upper_bound")
            
            # Check that lengths match
            if len(self._forecast) != len(self._lower_bound) or len(self._forecast) != len(self._upper_bound):
                raise ValueError("Forecast and confidence bounds must have the same length")
        else:
            # Calculate confidence intervals if confidence_interval is specified
            if "confidence_interval" in self._plot_params and self._plot_params["confidence_interval"] > 0:
                self._lower_bound, self._upper_bound = calculate_confidence_intervals(
                    self._forecast,
                    self._plot_params["confidence_interval"],
                    "normal"
                )
            else:
                self._lower_bound = None
                self._upper_bound = None
        
        # Set forecast dates
        self._forecast_dates = forecast_dates
        
        # Update plot if already initialized
        if self._initialized:
            self.update_plot()
        
        # Log data update
        logger.debug(f"VolatilityForecastPlot data updated: {len(self._forecast)} forecast points")
    
    def set_from_model(
        self,
        model: VolatilityModel,
        returns: np.ndarray,
        forecast_horizon: int,
        last_date: Optional[pd.Timestamp] = None,
        freq: Optional[str] = None
    ) -> None:
        """
        Sets forecast data directly from a volatility model.
        
        Parameters
        ----------
        model : VolatilityModel
            The volatility model to use
        returns : np.ndarray
            Returns series data
        forecast_horizon : int
            Number of periods to forecast
        last_date : Optional[pd.Timestamp], default=None
            Last date of the returns series
        freq : Optional[str], default=None
            Frequency for date generation (e.g., 'D', 'B', 'M')
        """
        # Validate returns data and forecast horizon
        returns = validate_array(returns, param_name="returns")
        if not isinstance(forecast_horizon, (int, np.integer)) or forecast_horizon <= 0:
            raise ValueError(f"Forecast horizon must be a positive integer, got {forecast_horizon}")
        
        # Calculate forecast
        forecast_variance = model.forecast(returns, forecast_horizon)
        if isinstance(forecast_variance, np.ndarray):
            forecast = np.sqrt(forecast_variance)  # Convert variance to volatility
        else:
            raise ValueError("Model did not return a valid forecast array")
        
        # Calculate confidence intervals
        lower_bound, upper_bound = calculate_confidence_intervals(
            forecast,
            self._plot_params.get("confidence_interval", 0.95),
            "normal"
        )
        
        # Generate forecast dates if last_date and freq are provided
        forecast_dates = None
        if last_date is not None and freq is not None:
            try:
                # Create date range starting from the day after last_date
                next_date = pd.Timestamp(last_date) + pd.Timedelta(days=1)
                forecast_dates = pd.date_range(start=next_date, periods=forecast_horizon, freq=freq)
            except Exception as e:
                logger.warning(f"Failed to generate forecast dates: {str(e)}")
        
        # Set the forecast data
        self.set_forecast_data(forecast, lower_bound, upper_bound, forecast_dates)
        
        # Log model forecast extraction
        logger.debug(f"Forecast extracted from volatility model: {forecast_horizon} periods")
    
    def update_plot(self) -> None:
        """
        Updates the forecast plot with current data and settings.
        """
        # Check if forecast data is available
        if self._forecast is None:
            logger.warning("Cannot update plot: missing forecast data")
            return
        
        # Signal update started
        self.signals.update_started.emit()
        
        try:
            # Clear existing plot
            self._plot_widget.clear_plot()
            
            # Prepare confidence intervals
            confidence_intervals = None
            if self._lower_bound is not None and self._upper_bound is not None:
                confidence_intervals = (self._lower_bound, self._upper_bound)
            
            # Create forecast figure
            fig = create_forecast_figure(
                self._forecast,
                confidence_intervals,
                self._forecast_dates,
                self._plot_params
            )
            
            # Set the figure in the plot widget
            plot_fig = self._plot_widget.get_figure()
            plot_fig.clear()
            
            # Copy content from created figure to the widget's figure
            for ax in fig.get_axes():
                # Get axis' position
                pos = ax.get_position()
                # Create new axis in the same position
                new_ax = plot_fig.add_axes(pos)
                # Copy all artists
                for line in ax.get_lines():
                    new_ax.add_line(line)
                for collection in ax.collections:
                    new_ax.add_collection(collection)
                # Copy layout
                new_ax.set_title(ax.get_title())
                new_ax.set_xlabel(ax.get_xlabel())
                new_ax.set_ylabel(ax.get_ylabel())
                new_ax.grid(ax.get_grid())
                # Copy legend if exists
                if ax.get_legend() is not None:
                    new_ax.legend()
                # Copy limits
                new_ax.set_xlim(ax.get_xlim())
                new_ax.set_ylim(ax.get_ylim())
            
            # Update the plot widget display
            self._plot_widget.update_plot()
            
            # Set initialized flag
            self._initialized = True
            
            # Signal update complete
            self.signals.update_complete.emit(self)
            
            # Log update
            logger.debug("Volatility forecast plot updated successfully")
            
        except Exception as e:
            # Signal update error
            self.signals.update_error.emit(e)
            
            # Log error
            logger.error(f"Error updating volatility forecast plot: {str(e)}")
    
    async def async_update_plot(self) -> None:
        """
        Asynchronously updates the forecast plot to prevent UI blocking.
        """
        # Check if forecast data is available
        if self._forecast is None:
            logger.warning("Cannot update plot: missing forecast data")
            return
        
        # Signal update started
        self.signals.update_started.emit()
        
        try:
            # Yield control to allow UI updates
            await asyncio.sleep(0)
            
            # Clear existing plot
            self._plot_widget.clear_plot()
            
            # Prepare confidence intervals
            confidence_intervals = None
            if self._lower_bound is not None and self._upper_bound is not None:
                confidence_intervals = (self._lower_bound, self._upper_bound)
            
            # Create forecast figure
            fig = create_forecast_figure(
                self._forecast,
                confidence_intervals,
                self._forecast_dates,
                self._plot_params
            )
            
            # Set the figure in the plot widget - same as in update_plot
            plot_fig = self._plot_widget.get_figure()
            plot_fig.clear()
            
            # Copy content from created figure to the widget's figure
            for ax in fig.get_axes():
                # Get axis' position
                pos = ax.get_position()
                # Create new axis in the same position
                new_ax = plot_fig.add_axes(pos)
                # Copy all artists
                for line in ax.get_lines():
                    new_ax.add_line(line)
                for collection in ax.collections:
                    new_ax.add_collection(collection)
                # Copy layout
                new_ax.set_title(ax.get_title())
                new_ax.set_xlabel(ax.get_xlabel())
                new_ax.set_ylabel(ax.get_ylabel())
                new_ax.grid(ax.get_grid())
                # Copy legend if exists
                if ax.get_legend() is not None:
                    new_ax.legend()
                # Copy limits
                new_ax.set_xlim(ax.get_xlim())
                new_ax.set_ylim(ax.get_ylim())
            
            # Update the plot widget display asynchronously
            await self._plot_widget.async_update_plot()
            
            # Set initialized flag
            self._initialized = True
            
            # Signal update complete
            self.signals.update_complete.emit(self)
            
            # Log async update
            logger.debug("Volatility forecast plot updated asynchronously")
            
        except Exception as e:
            # Signal update error
            self.signals.update_error.emit(e)
            
            # Log error
            logger.error(f"Error in async update of volatility forecast plot: {str(e)}")
    
    def set_confidence_interval(self, level: float, distribution: str = "normal") -> None:
        """
        Sets the confidence interval level and recalculates bounds.
        
        Parameters
        ----------
        level : float
            Confidence level (0 to 1)
        distribution : str, default="normal"
            Distribution to use for confidence intervals
        """
        if not 0 <= level <= 1:
            raise ValueError(f"Confidence level must be between 0 and 1, got {level}")
        
        # Update confidence interval setting
        self._plot_params["confidence_interval"] = level
        
        # Recalculate confidence bounds if forecast is available
        if self._forecast is not None:
            self._lower_bound, self._upper_bound = calculate_confidence_intervals(
                self._forecast,
                level,
                distribution
            )
            
            # Update plot if initialized
            if self._initialized:
                self.update_plot()
        
        # Log confidence interval change
        logger.debug(f"Confidence interval set to: {level} using {distribution} distribution")
    
    def save_figure(self, filepath: str, dpi: Optional[int] = None) -> bool:
        """
        Saves the current forecast plot to a file.
        
        Parameters
        ----------
        filepath : str
            Path to save the figure
        dpi : Optional[int], default=None
            Resolution for the saved figure
            
        Returns
        -------
        bool
            True if save was successful, False otherwise
        """
        # Check if plot is initialized
        if not self._initialized:
            logger.warning("Cannot save: plot not initialized")
            return False
        
        # Save the figure
        result = self._plot_widget.save_plot(filepath, dpi)
        
        # Log result
        if result:
            logger.info(f"Figure saved to {filepath}")
        else:
            logger.error(f"Failed to save figure to {filepath}")
        
        return result
    
    def clear(self) -> None:
        """
        Clears the current plot and resets widget state.
        """
        # Reset data
        self._forecast = None
        self._lower_bound = None
        self._upper_bound = None
        self._forecast_dates = None
        
        # Clear plot
        self._plot_widget.clear_plot()
        
        # Reset initialized flag
        self._initialized = False
        
        # Log clear operation
        logger.debug("Volatility forecast plot cleared")
    
    def get_figure(self) -> Optional[plt.Figure]:
        """
        Returns the matplotlib figure for direct access.
        
        Returns
        -------
        Optional[plt.Figure]
            The current figure or None if not initialized
        """
        if not self._initialized:
            return None
        
        return self._plot_widget.get_figure()