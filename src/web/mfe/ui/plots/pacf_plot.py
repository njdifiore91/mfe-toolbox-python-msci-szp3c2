"""
MFE Toolbox - PACF Plot Component

This module implements a Partial AutoCorrelation Function (PACF) visualization 
component using PyQt6 and matplotlib. It provides interactive plot display
for time series analysis and ARMA/ARMAX model diagnostics with support for
asynchronous updates.
"""

import numpy as np  # numpy 1.26.3
from statsmodels.tsa.stattools import pacf  # statsmodels 0.14.1
import matplotlib.pyplot as plt  # matplotlib 3.8.0
from matplotlib.figure import Figure  # matplotlib 3.8.0
from PyQt6.QtWidgets import QWidget  # PyQt6 6.6.1
from PyQt6.QtCore import pyqtSignal  # PyQt6 6.6.1
import logging  # Python standard library
from typing import Dict, Optional, Tuple, Any  # Python standard library
import asyncio  # Python standard library

# Internal imports
from ..plots.matplotlib_backend import ensure_matplotlib_qt
from ..plot_widgets import BasePlotWidget

# Set up module logger
logger = logging.getLogger(__name__)


def calculate_pacf(data: np.ndarray, nlags: int = 40, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates the partial autocorrelation function for a given time series data using statsmodels.
    
    Parameters
    ----------
    data : np.ndarray
        The time series data for which to calculate PACF
    nlags : int, optional
        Number of lags to include in the PACF calculation, default is 40
    alpha : float, optional
        Significance level for the confidence intervals, default is 0.05
        
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
    pacf_values = pacf(data, nlags=nlags, alpha=alpha)
    
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


def create_pacf_figure(data: np.ndarray, nlags: int = 40, alpha: float = 0.05, 
                      plot_params: Dict[str, Any] = None) -> Figure:
    """
    Creates a matplotlib figure with a PACF plot for the given data.
    
    Parameters
    ----------
    data : np.ndarray
        The time series data for which to create a PACF plot
    nlags : int, optional
        Number of lags to include in the plot, default is 40
    alpha : float, optional
        Significance level for the confidence intervals, default is 0.05
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
    default_params = {
        'title': 'Partial Autocorrelation Function',
        'grid': True,
        'bar_color': 'blue',
        'alpha': 0.7,
        'figsize': (8, 6),
        'bar_width': 0.3,
        'ci_color': 'r',
        'zero_color': 'k'
    }
    
    # Update with user-provided parameters
    for key, value in plot_params.items():
        default_params[key] = value
    
    # Calculate PACF values and confidence intervals
    pacf_vals, conf_intervals, lags = calculate_pacf(data, nlags, alpha)
    
    # Create figure
    fig = plt.figure(figsize=default_params['figsize'])
    ax = fig.add_subplot(111)
    
    # Plot PACF as bars
    ax.bar(lags, pacf_vals, width=default_params['bar_width'], 
           color=default_params['bar_color'], alpha=default_params['alpha'])
    
    # Add confidence interval lines
    if isinstance(conf_intervals, np.ndarray) and conf_intervals.shape[0] == 2:
        ax.plot(lags, conf_intervals[0, :], color=default_params['ci_color'], 
                linestyle='--', linewidth=1)
        ax.plot(lags, conf_intervals[1, :], color=default_params['ci_color'], 
                linestyle='--', linewidth=1)
    
    # Add zero line for reference
    ax.axhline(y=0, color=default_params['zero_color'], linestyle='-', linewidth=0.5)
    
    # Add labels and title
    ax.set_title(default_params['title'])
    ax.set_xlabel('Lag')
    ax.set_ylabel('Partial Autocorrelation')
    
    # Add grid if specified
    ax.grid(default_params['grid'])
    
    # Tight layout for better appearance
    fig.tight_layout()
    
    return fig


async def async_create_pacf_figure(data: np.ndarray, nlags: int = 40, alpha: float = 0.05, 
                                  plot_params: Dict[str, Any] = None) -> Figure:
    """
    Asynchronously creates a PACF plot figure to prevent UI blocking.
    
    Parameters
    ----------
    data : np.ndarray
        The time series data for which to create a PACF plot
    nlags : int, optional
        Number of lags to include in the plot, default is 40
    alpha : float, optional
        Significance level for the confidence intervals, default is 0.05
    plot_params : Dict[str, Any], optional
        Dictionary containing customization parameters for the plot
        
    Returns
    -------
    matplotlib.figure.Figure
        A matplotlib figure object containing the PACF plot
    """
    # Yield control to prevent UI blocking
    await asyncio.sleep(0)
    
    try:
        # Create PACF figure
        return create_pacf_figure(data, nlags, alpha, plot_params)
    except Exception as e:
        logger.error(f"Error creating PACF plot: {str(e)}")
        raise


class PACFPlot(BasePlotWidget):
    """
    PyQt6 widget that embeds a matplotlib PACF plot in a Qt application,
    providing both UI integration and interactive visualization.
    """
    
    # Define signals for events
    data_changed = pyqtSignal()
    
    def __init__(self, parent=None):
        """
        Initializes the PACF plot widget with default parameters.
        
        Parameters
        ----------
        parent : QWidget
            The parent widget
        """
        # Initialize parent class
        super().__init__(parent)
        
        # Initialize properties
        self._figure = None
        self._data = None
        self._nlags = 40
        self._alpha = 0.05
        self._plot_params = {
            'title': 'Partial Autocorrelation Function',
            'grid': True,
            'bar_color': 'blue',
            'alpha': 0.05
        }
        self._initialized = False
        
        logger.debug("PACFPlot widget initialized")
    
    def set_data(self, data: np.ndarray) -> None:
        """
        Sets the time series data to be plotted and updates the PACF plot.
        
        Parameters
        ----------
        data : np.ndarray
            The time series data for the PACF plot
        """
        # Validate input data
        if not isinstance(data, np.ndarray):
            data = np.asarray(data)
        
        if not np.isfinite(data).all():
            raise ValueError("Input data contains NaN or infinite values")
        
        # Store the data
        self._data = data
        
        # Update the PACF plot if already initialized
        if self._initialized:
            self.schedule_update()
        
        # Emit data_changed signal
        self.data_changed.emit()
        
        logger.debug(f"Set PACF data with shape {data.shape}")
    
    def set_nlags(self, nlags: int) -> None:
        """
        Sets the number of lags to display in the PACF plot.
        
        Parameters
        ----------
        nlags : int
            Number of lags to display
        """
        # Validate lag value
        if not isinstance(nlags, int) or nlags <= 0:
            raise ValueError("Number of lags must be a positive integer")
        
        # Store the nlags value
        self._nlags = nlags
        
        # Update plot if data exists
        if self._data is not None and self._initialized:
            self.schedule_update()
        
        logger.debug(f"Set PACF nlags to {nlags}")
    
    def set_alpha(self, alpha: float) -> None:
        """
        Sets the significance level for confidence intervals in the PACF plot.
        
        Parameters
        ----------
        alpha : float
            Significance level (between 0 and 1)
        """
        # Validate alpha value
        if not isinstance(alpha, float) or not 0 < alpha < 1:
            raise ValueError("Alpha must be a float between 0 and 1")
        
        # Store the alpha value
        self._alpha = alpha
        
        # Update plot if data exists
        if self._data is not None and self._initialized:
            self.schedule_update()
        
        logger.debug(f"Set PACF alpha to {alpha}")
    
    def set_plot_params(self, params: Dict[str, Any]) -> None:
        """
        Sets the plotting parameters for customizing the PACF plot appearance.
        
        Parameters
        ----------
        params : Dict[str, Any]
            Dictionary of plot parameters
        """
        # Validate parameters
        if not isinstance(params, dict):
            raise TypeError("Plot parameters must be a dictionary")
        
        # Update plot parameters
        for key, value in params.items():
            self._plot_params[key] = value
        
        # Update plot if data exists
        if self._data is not None and self._initialized:
            self.schedule_update()
        
        logger.debug(f"Updated PACF plot parameters: {list(params.keys())}")
    
    def _create_figure(self) -> Figure:
        """
        Creates the PACF figure for the current data and parameters.
        
        Returns
        -------
        matplotlib.figure.Figure
            Figure containing the PACF plot
        """
        # Check if data is available
        if self._data is None:
            logger.warning("Cannot create PACF figure: No data available")
            return None
        
        # Create PACF figure
        return create_pacf_figure(
            self._data, 
            self._nlags, 
            self._alpha, 
            self._plot_params
        )
    
    async def async_update_plot(self) -> None:
        """
        Asynchronously updates the PACF plot to prevent UI blocking.
        """
        # Check if data is available
        if self._data is None:
            logger.warning("Cannot update PACF plot: No data available")
            return
        
        try:
            # Create PACF figure asynchronously
            figure = await async_create_pacf_figure(
                self._data, 
                self._nlags, 
                self._alpha, 
                self._plot_params
            )
            
            # Update the figure in the plot widget
            self.get_figure().clear()
            for ax in figure.get_axes():
                self.get_figure().add_axes(ax)
            
            # Update the plot using parent method
            await super().async_update_plot()
            
            # Mark as initialized
            self._initialized = True
            
            logger.debug("PACF plot asynchronously updated")
        except Exception as e:
            logger.error(f"Error updating PACF plot: {str(e)}")
            raise
    
    def clear(self) -> None:
        """
        Clears the current PACF plot and resets the widget state.
        """
        # Call parent clear method
        super().clear()
        
        # Reset data to None
        self._data = None
        
        # Reset initialization state
        self._initialized = False
        
        logger.debug("PACF plot cleared")