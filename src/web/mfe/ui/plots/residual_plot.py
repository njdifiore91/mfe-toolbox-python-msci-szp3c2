"""
MFE Toolbox - Residual Plot Component

This module implements an interactive PyQt6-based residual plot component that 
visualizes model residuals with comprehensive statistical diagnostics. It provides
rich visualization of residuals, distribution plots, and fitted value relationships
with asynchronous updating capabilities.

The module is designed for the MFE Toolbox UI, offering extensive diagnostic tools
for econometric model validation.
"""

import numpy as np  # numpy 1.26.3
import matplotlib.pyplot as plt  # matplotlib 3.7.1
import scipy.stats as stats  # scipy 1.11.4
from PyQt6.QtCore import QSize  # PyQt6 6.6.1
from PyQt6.QtWidgets import QWidget, QVBoxLayout  # PyQt6 6.6.1
import statsmodels.stats.diagnostic as smd  # statsmodels 0.14.1
import logging  # Python standard library
from typing import Dict, Optional, Any  # Python standard library
import asyncio  # Python standard library

from .matplotlib_backend import AsyncPlotWidget
from ....backend.mfe.utils.validation import validate_array

# Setup module logger
logger = logging.getLogger(__name__)

# Default plot configuration
DEFAULT_PLOT_PARAMS = {
    "title": "Residual Analysis",
    "grid": True,
    "residual_color": "blue",
    "fitted_color": "red",
    "hist_bins": 20,
    "alpha": 0.7,
    "marker": "o",
    "markersize": 4,
    "scatter_title": "Residuals vs. Fitted",
    "hist_title": "Residual Distribution",
    "qqplot_title": "Q-Q Plot"
}


def analyze_residuals(residuals: np.ndarray) -> Dict[str, Any]:
    """
    Performs statistical analysis on model residuals.
    
    This function calculates basic statistics and performs diagnostic tests on
    the residuals of a statistical model, including normality and autocorrelation tests.
    
    Parameters
    ----------
    residuals : np.ndarray
        The residuals from a statistical model
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing statistical metrics and test results:
        - 'mean': Mean value of residuals
        - 'std': Standard deviation of residuals
        - 'min': Minimum residual value
        - 'max': Maximum residual value
        - 'jarque_bera': Tuple of (statistic, p-value) for Jarque-Bera normality test
        - 'durbin_watson': Durbin-Watson statistic for autocorrelation
        - 'het_white': Tuple of (statistic, p-value) for White's heteroskedasticity test
        
    Raises
    ------
    ValueError
        If residuals array is invalid
    """
    # Validate residuals array
    residuals = validate_array(residuals, param_name="residuals")
    
    # Calculate basic statistics
    mean = np.mean(residuals)
    std = np.std(residuals)
    min_val = np.min(residuals)
    max_val = np.max(residuals)
    
    # Perform Jarque-Bera normality test
    jb_stat, jb_pval = stats.jarque_bera(residuals)
    
    # Calculate Durbin-Watson statistic for autocorrelation
    # Note: This requires contiguous residuals in time order
    dw_stat = np.sum(np.diff(residuals) ** 2) / np.sum(residuals ** 2)
    
    # Initialize result dictionary
    result = {
        'mean': mean,
        'std': std,
        'min': min_val,
        'max': max_val,
        'jarque_bera': (jb_stat, jb_pval),
        'durbin_watson': dw_stat,
    }
    
    # Perform heteroskedasticity test if residuals and fitted values are available
    # We'll do a simple test based on the squared residuals
    try:
        # Create arbitrary X for heteroskedasticity test (X = [1, range])
        # This is just a placeholder until we have fitted values
        X = np.column_stack((
            np.ones(len(residuals)),
            np.arange(len(residuals))
        ))
        het_stat, het_pval, _, _ = smd.het_white(residuals, X)
        result['het_white'] = (het_stat, het_pval)
    except Exception as e:
        logger.warning(f"Heteroskedasticity test failed: {str(e)}")
        result['het_white'] = (None, None)
    
    return result


def create_residual_figure(
    residuals: np.ndarray,
    fitted_values: np.ndarray,
    plot_params: Dict[str, Any]
) -> plt.Figure:
    """
    Creates a matplotlib figure with residual diagnostic plots.
    
    This function generates a comprehensive figure with three subplots:
    1. Residuals vs. fitted values scatter plot
    2. Residual distribution histogram with normal distribution overlay
    3. Normal Q-Q plot of residuals
    
    Parameters
    ----------
    residuals : np.ndarray
        The residuals from a statistical model
    fitted_values : np.ndarray
        The fitted values from the model
    plot_params : Dict[str, Any]
        Dictionary of plotting parameters to customize the appearance
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure containing residual diagnostic plots
        
    Raises
    ------
    ValueError
        If residuals array is invalid or dimensions don't match
    """
    # Validate residuals array
    residuals = validate_array(residuals, param_name="residuals")
    
    # Validate fitted_values array and check compatibility with residuals
    fitted_values = validate_array(fitted_values, param_name="fitted_values")
    if len(fitted_values) != len(residuals):
        raise ValueError(
            f"Length mismatch: residuals length {len(residuals)}, "
            f"fitted_values length {len(fitted_values)}"
        )
    
    # Create a figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(plot_params.get("title", "Residual Analysis"), fontsize=14)
    
    # 1. Residuals vs Fitted scatter plot
    ax1.scatter(
        fitted_values, 
        residuals, 
        color=plot_params.get("residual_color", "blue"),
        alpha=plot_params.get("alpha", 0.7),
        marker=plot_params.get("marker", "o"),
        s=plot_params.get("markersize", 4)**2
    )
    ax1.axhline(y=0, color='r', linestyle='-', linewidth=1)
    ax1.set_xlabel("Fitted Values")
    ax1.set_ylabel("Residuals")
    ax1.set_title(plot_params.get("scatter_title", "Residuals vs. Fitted"))
    if plot_params.get("grid", True):
        ax1.grid(alpha=0.3)
    
    # 2. Residual distribution histogram
    hist_bins = plot_params.get("hist_bins", 20)
    ax2.hist(
        residuals, 
        bins=hist_bins, 
        density=True, 
        alpha=plot_params.get("alpha", 0.7),
        color=plot_params.get("residual_color", "blue")
    )
    
    # Add normal distribution curve
    x = np.linspace(min(residuals), max(residuals), 100)
    mean, std = np.mean(residuals), np.std(residuals)
    pdf = stats.norm.pdf(x, mean, std)
    ax2.plot(x, pdf, 'r-', linewidth=2)
    
    ax2.set_xlabel("Residual Value")
    ax2.set_ylabel("Density")
    ax2.set_title(plot_params.get("hist_title", "Residual Distribution"))
    if plot_params.get("grid", True):
        ax2.grid(alpha=0.3)
    
    # 3. Normal Q-Q plot
    stats.probplot(residuals, dist="norm", plot=ax3)
    ax3.set_title(plot_params.get("qqplot_title", "Q-Q Plot"))
    if plot_params.get("grid", True):
        ax3.grid(alpha=0.3)
    
    # Calculate residual statistics
    residual_stats = analyze_residuals(residuals)
    
    # Add statistics as text annotations to the figure
    stats_text = (
        f"Mean: {residual_stats['mean']:.4f}\n"
        f"Std Dev: {residual_stats['std']:.4f}\n"
        f"Jarque-Bera p-value: {residual_stats['jarque_bera'][1]:.4f}\n"
        f"Durbin-Watson: {residual_stats['durbin_watson']:.4f}"
    )
    
    # Add heteroskedasticity test result if available
    if residual_stats.get('het_white', (None, None))[1] is not None:
        stats_text += f"\nHet. Test p-value: {residual_stats['het_white'][1]:.4f}"
    
    fig.text(0.01, 0.01, stats_text, fontsize=9, verticalalignment='bottom')
    
    # Adjust layout
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    return fig


class ResidualPlot(QWidget):
    """
    A PyQt6 widget that displays interactive diagnostic plots for model residuals 
    with asynchronous updating capabilities.
    
    This widget provides comprehensive visualization and analysis of model residuals,
    including scatter plots, distribution analysis, and statistical tests. It supports
    asynchronous updates to maintain UI responsiveness during computation-intensive tasks.
    
    Parameters
    ----------
    parent : QWidget, optional
        The parent widget
    """
    
    def __init__(self, parent: Optional[QWidget] = None):
        """
        Initializes the residual plot widget with default parameters.
        
        Parameters
        ----------
        parent : QWidget
            The parent widget
        """
        super().__init__(parent)
        
        # Initialize properties
        self._residuals = None
        self._fitted_values = None
        self._plot_params = DEFAULT_PLOT_PARAMS.copy()
        self._initialized = False
        self._residual_stats = {}
        
        # Create layout
        self._layout = QVBoxLayout(self)
        self.setLayout(self._layout)
        
        # Create plot widget
        self._plot_widget = AsyncPlotWidget(self)
        self._layout.addWidget(self._plot_widget)
        
        logger.debug("ResidualPlot widget initialized")
    
    def set_residuals(self, residuals: np.ndarray) -> None:
        """
        Sets the residual data to be visualized.
        
        Parameters
        ----------
        residuals : np.ndarray
            The residuals from a statistical model
            
        Raises
        ------
        ValueError
            If residuals array is invalid
        """
        # Validate residuals
        residuals = validate_array(residuals, param_name="residuals")
        
        # Store residuals
        self._residuals = residuals
        
        # Calculate residual statistics
        self._residual_stats = analyze_residuals(residuals)
        
        # Update plot if we already have data
        if self._initialized:
            self.update_plot()
        
        logger.debug(f"Set residuals with shape {residuals.shape}")
    
    def set_fitted_values(self, fitted_values: np.ndarray) -> None:
        """
        Sets the fitted values for residual vs. fitted plot.
        
        Parameters
        ----------
        fitted_values : np.ndarray
            The fitted values from the model
            
        Raises
        ------
        ValueError
            If fitted_values array is invalid or dimensions don't match residuals
        """
        # Validate fitted_values
        fitted_values = validate_array(fitted_values, param_name="fitted_values")
        
        # Check dimensions match existing residuals if available
        if self._residuals is not None and len(fitted_values) != len(self._residuals):
            raise ValueError(
                f"Length mismatch: residuals length {len(self._residuals)}, "
                f"fitted_values length {len(fitted_values)}"
            )
        
        # Store fitted values
        self._fitted_values = fitted_values
        
        # Update plot if we already have residuals and the plot is initialized
        if self._residuals is not None and self._initialized:
            self.update_plot()
        
        logger.debug(f"Set fitted values with shape {fitted_values.shape}")
    
    def set_plot_params(self, params: Dict[str, Any]) -> None:
        """
        Sets plotting parameters to customize residual plot appearance.
        
        Parameters
        ----------
        params : Dict[str, Any]
            Dictionary of plot parameters to override defaults
            
        Raises
        ------
        TypeError
            If params is not a dictionary
        """
        if not isinstance(params, dict):
            raise TypeError("Plot parameters must be a dictionary")
        
        # Update plot parameters
        self._plot_params.update(params)
        
        # Update plot if data is already available
        if self._residuals is not None and self._initialized:
            self.update_plot()
        
        logger.debug(f"Updated plot parameters: {list(params.keys())}")
    
    def get_residual_stats(self) -> Dict[str, Any]:
        """
        Returns the statistical analysis results for the residuals.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary of residual statistics and test results
        """
        return self._residual_stats.copy() if self._residuals is not None else {}
    
    def update_plot(self) -> None:
        """
        Updates the residual plot with current data and settings.
        
        This method updates the plot synchronously, which may block the UI
        during the update process. For non-blocking updates, use async_update_plot.
        """
        if self._residuals is None:
            logger.warning("Cannot update plot: No residuals data available")
            return
        
        # Clear existing plot
        self._plot_widget.clear_plot()
        
        # Create residual figure
        fig = create_residual_figure(
            self._residuals,
            self._fitted_values if self._fitted_values is not None else np.arange(len(self._residuals)),
            self._plot_params
        )
        
        # Set figure in plot widget
        self._plot_widget.get_figure().clf()
        for ax in fig.get_axes():
            self._plot_widget.get_figure().add_axes(ax)
        
        # Update the plot
        self._plot_widget.update_plot()
        
        # Set initialized flag
        self._initialized = True
        
        logger.debug("Residual plot updated")
    
    async def async_update_plot(self) -> None:
        """
        Asynchronously updates the residual plot to prevent UI blocking.
        
        This method performs the plot update asynchronously, yielding control
        back to the event loop to maintain UI responsiveness.
        """
        if self._residuals is None:
            logger.warning("Cannot update plot: No residuals data available")
            return
        
        # Yield control to event loop
        await asyncio.sleep(0)
        
        # Clear existing plot
        self._plot_widget.clear_plot()
        
        # Create residual figure
        fig = create_residual_figure(
            self._residuals,
            self._fitted_values if self._fitted_values is not None else np.arange(len(self._residuals)),
            self._plot_params
        )
        
        # Set figure in plot widget and update asynchronously
        self._plot_widget.get_figure().clf()
        for ax in fig.get_axes():
            self._plot_widget.get_figure().add_axes(ax)
        
        await self._plot_widget.async_update_plot()
        
        # Set initialized flag
        self._initialized = True
        
        logger.debug("Residual plot asynchronously updated")
    
    def save_figure(self, filepath: str, dpi: int = 300) -> bool:
        """
        Saves the current residual plot to a file.
        
        Parameters
        ----------
        filepath : str
            The file path to save the figure to
        dpi : int, default=300
            Resolution in dots per inch
            
        Returns
        -------
        bool
            True if save was successful, False otherwise
        """
        if not self._initialized:
            logger.warning("Cannot save: Plot not yet initialized")
            return False
        
        # Try to save figure
        success = self._plot_widget.save_plot(filepath, dpi)
        
        if success:
            logger.info(f"Saved residual plot to {filepath}")
        else:
            logger.error(f"Failed to save residual plot to {filepath}")
        
        return success
    
    def clear(self) -> None:
        """
        Clears the current plot and resets widget state.
        """
        self._residuals = None
        self._fitted_values = None
        self._residual_stats = {}
        
        if self._initialized:
            self._plot_widget.clear_plot()
            self._initialized = False
        
        logger.debug("Residual plot cleared")
    
    def sizeHint(self) -> QSize:
        """
        Returns the recommended size for the widget.
        
        Returns
        -------
        QSize
            Recommended widget size
        """
        return QSize(800, 400)