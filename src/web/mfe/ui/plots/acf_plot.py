import numpy as np  # version 1.26.3
from statsmodels.tsa.stattools import acf  # version 0.14.1
import matplotlib.pyplot as plt  # version 3.8.0
from matplotlib.figure import Figure  # version 3.8.0
from PyQt6.QtWidgets import QWidget, QVBoxLayout  # version 6.6.1
from PyQt6.QtCore import pyqtSignal  # version 6.6.1
import logging  # standard library
from typing import Dict, Optional, Tuple, Union, List, Any  # standard library

# Import matplotlib backend utilities
from ..plots.matplotlib_backend import (
    create_figure, embed_figure, clear_figure,
    update_canvas, save_figure as backend_save_figure
)

# Set up logger
logger = logging.getLogger(__name__)


def calculate_acf(data: np.ndarray, nlags: int = 40, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates the autocorrelation function for a given time series data using statsmodels.
    
    Parameters
    ----------
    data : np.ndarray
        The time series data for which to calculate ACF
    nlags : int, optional
        Number of lags to include in the ACF calculation, default is 40
    alpha : float, optional
        Significance level for the confidence intervals, default is 0.05
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Tuple containing ACF values, confidence intervals, and lags
    """
    # Validate input data
    if not isinstance(data, np.ndarray):
        data = np.asarray(data)
    
    if not np.isfinite(data).all():
        raise ValueError("Input data contains NaN or infinite values")
    
    # Calculate ACF using statsmodels
    acf_values = acf(data, nlags=nlags, alpha=alpha, fft=True)
    
    # If alpha is provided, statsmodels returns a tuple with values and confidence intervals
    if isinstance(acf_values, tuple):
        acf_vals = acf_values[0]
        conf_intervals = acf_values[1]
    else:
        acf_vals = acf_values
        # Calculate approx confidence intervals manually if not provided
        conf_intervals = np.ones((2, nlags + 1)) * 1.96 / np.sqrt(len(data))
        conf_intervals[0, :] = -conf_intervals[0, :]
    
    # Create lag indices
    lags = np.arange(len(acf_vals))
    
    return acf_vals, conf_intervals, lags


def create_acf_figure(data: np.ndarray, nlags: int = 40, alpha: float = 0.05, 
                     plot_params: Dict[str, Any] = None) -> Figure:
    """
    Creates a matplotlib figure with an ACF plot for the given data.
    
    Parameters
    ----------
    data : np.ndarray
        The time series data for which to create an ACF plot
    nlags : int, optional
        Number of lags to include in the plot, default is 40
    alpha : float, optional
        Significance level for the confidence intervals, default is 0.05
    plot_params : Dict[str, Any], optional
        Dictionary containing customization parameters for the plot
        
    Returns
    -------
    matplotlib.figure.Figure
        A matplotlib figure object containing the ACF plot
    """
    # Set default plot parameters if not provided
    if plot_params is None:
        plot_params = {}
    
    # Default parameters
    default_params = {
        'title': 'Autocorrelation Function',
        'xlabel': 'Lag',
        'ylabel': 'Autocorrelation',
        'grid': True,
        'figsize': (8, 6),
        'bar_width': 0.3,
        'bar_color': 'steelblue',
        'ci_color': 'r',
        'zero_color': 'k'
    }
    
    # Update with user-provided parameters
    for key, value in plot_params.items():
        default_params[key] = value
    
    # Calculate ACF values and confidence intervals
    acf_vals, conf_intervals, lags = calculate_acf(data, nlags, alpha)
    
    # Create figure
    fig = create_figure(figsize=default_params['figsize'])
    ax = fig.add_subplot(111)
    
    # Plot ACF as bars
    ax.bar(lags, acf_vals, width=default_params['bar_width'], 
           color=default_params['bar_color'], alpha=0.7)
    
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
    ax.set_xlabel(default_params['xlabel'])
    ax.set_ylabel(default_params['ylabel'])
    
    # Add grid if specified
    ax.grid(default_params['grid'])
    
    # Tight layout for better appearance
    fig.tight_layout()
    
    return fig


class ACFPlot(QWidget):
    """
    PyQt6 widget that embeds a matplotlib ACF plot in a Qt application,
    providing both UI integration and interactive visualization.
    """
    
    # Define signals for events
    data_changed = pyqtSignal(np.ndarray)
    plot_updated = pyqtSignal()
    
    def __init__(self, parent=None):
        """
        Initializes the ACF plot widget with default parameters.
        
        Parameters
        ----------
        parent : QWidget, optional
            Parent widget, default is None
        """
        super().__init__(parent)
        
        # Create layout for the widget
        self._layout = QVBoxLayout(self)
        self.setLayout(self._layout)
        
        # Initialize properties
        self._figure = None
        self._data = None
        self._nlags = 40
        self._alpha = 0.05
        self._plot_params = {
            'title': 'Autocorrelation Function',
            'xlabel': 'Lag',
            'ylabel': 'Autocorrelation',
            'grid': True
        }
        self._initialized = False
        
        # Connect signals and slots
        self.data_changed.connect(lambda: self.update_plot())
        
        logger.debug("ACFPlot widget initialized")
    
    def set_data(self, data: np.ndarray) -> None:
        """
        Sets the time series data to be plotted and updates the ACF plot.
        
        Parameters
        ----------
        data : np.ndarray
            The time series data for the ACF plot
        """
        # Validate input data
        if not isinstance(data, np.ndarray):
            data = np.asarray(data)
        
        if not np.isfinite(data).all():
            raise ValueError("Input data contains NaN or infinite values")
        
        logger.debug(f"Setting ACF plot data with {len(data)} observations")
        
        # Store the data
        self._data = data
        
        # Update plot if already initialized or create a new one
        self.update_plot()
        
        # Emit signal that data has been updated
        self.data_changed.emit(data)
    
    def set_nlags(self, nlags: int) -> None:
        """
        Sets the number of lags to display in the ACF plot.
        
        Parameters
        ----------
        nlags : int
            Number of lags to display
        """
        # Validate lag value
        if not isinstance(nlags, int) or nlags <= 0:
            raise ValueError("Number of lags must be a positive integer")
        
        logger.debug(f"Setting ACF plot lags to {nlags}")
        
        # Store the nlags value
        self._nlags = nlags
        
        # Update plot if data exists
        if self._data is not None and self._initialized:
            self.update_plot()
    
    def set_alpha(self, alpha: float) -> None:
        """
        Sets the significance level for confidence intervals in the ACF plot.
        
        Parameters
        ----------
        alpha : float
            Significance level (between 0 and 1)
        """
        # Validate alpha value
        if not isinstance(alpha, float) or not 0 < alpha < 1:
            raise ValueError("Alpha must be a float between 0 and 1")
        
        logger.debug(f"Setting ACF plot alpha to {alpha}")
        
        # Store the alpha value
        self._alpha = alpha
        
        # Update plot if data exists
        if self._data is not None and self._initialized:
            self.update_plot()
    
    def set_plot_params(self, params: Dict[str, Any]) -> None:
        """
        Sets the plotting parameters for customizing the ACF plot appearance.
        
        Parameters
        ----------
        params : Dict[str, Any]
            Dictionary of plot parameters
        """
        # Validate parameters
        if not isinstance(params, dict):
            raise TypeError("Plot parameters must be a dictionary")
        
        logger.debug(f"Updating ACF plot parameters: {list(params.keys())}")
        
        # Update plot parameters
        for key, value in params.items():
            self._plot_params[key] = value
        
        # Update plot if data exists
        if self._data is not None and self._initialized:
            self.update_plot()
    
    def update_plot(self) -> None:
        """
        Updates the ACF plot with current data and parameters.
        """
        # Check if data is available
        if self._data is None:
            logger.warning("Cannot update ACF plot: No data available")
            return
        
        logger.debug("Updating ACF plot")
        
        # Create new ACF figure
        self._figure = create_acf_figure(
            self._data, 
            self._nlags, 
            self._alpha, 
            self._plot_params
        )
        
        # Clear existing layout items
        for i in reversed(range(self._layout.count())):
            item = self._layout.itemAt(i)
            if item and item.widget():
                item.widget().setParent(None)
        
        # Create and add the embedded figure to the layout
        embedded_widget = embed_figure(self._figure, self, with_toolbar=True)
        self._layout.addWidget(embedded_widget)
        
        # Mark as initialized
        self._initialized = True
        
        # Emit signal that plot has been updated
        self.plot_updated.emit()
        
        logger.debug("ACF plot updated successfully")
    
    def save_figure(self, filepath: str) -> bool:
        """
        Saves the current ACF plot to a file.
        
        Parameters
        ----------
        filepath : str
            Path to save the figure to
            
        Returns
        -------
        bool
            Success status of the save operation
        """
        # Validate the figure exists
        if not self._initialized or self._figure is None:
            logger.warning("Cannot save ACF plot: No figure exists")
            return False
        
        try:
            # Save figure to specified path
            return backend_save_figure(self._figure, filepath, dpi=300)
        except Exception as e:
            logger.error(f"Failed to save ACF plot: {str(e)}")
            return False
    
    def clear(self) -> None:
        """
        Clears the current ACF plot and resets the widget state.
        """
        # Clear the internal figure reference
        if self._figure:
            clear_figure(self._figure)
            self._figure = None
        
        # Reset the data property
        self._data = None
        
        # Clear the matplotlib widget if it exists
        for i in reversed(range(self._layout.count())):
            item = self._layout.itemAt(i)
            if item and item.widget():
                item.widget().setParent(None)
        
        # Reset initialization state
        self._initialized = False
        
        logger.debug("ACF plot cleared")