"""
MFE Toolbox - Plot Components

This module provides a unified interface for accessing various plot types
including time series, ACF, PACF, and residual analysis plots with PyQt6 integration.
The plot components support asynchronous updates and are designed for financial
time series visualization and statistical analysis.
"""

import logging
from typing import Optional, Union, Dict, Any

# Configure logger
logger = logging.getLogger(__name__)

# Import plot components from submodules
from .matplotlib_backend import (
    MatplotlibBackend, 
    AsyncPlotWidget,
    create_figure,
    create_canvas,
    create_navigation_toolbar,
    embed_figure,
    clear_figure,
    update_canvas,
    async_update_canvas,
    save_figure,
    set_plot_style
)

from .acf_plot import ACFPlot
from .pacf_plot import PACFPlot
from .residual_plot import ResidualPlot
from .time_series_plot import TimeSeriesPlot

# Define version
__version__ = "4.0.0"

# Define exports
__all__ = [
    "MatplotlibBackend",
    "AsyncPlotWidget",
    "ACFPlot",
    "PACFPlot", 
    "ResidualPlot",
    "TimeSeriesPlot",
    "create_figure",
    "create_canvas",
    "create_navigation_toolbar",
    "embed_figure",
    "clear_figure",
    "update_canvas",
    "async_update_canvas",
    "save_figure",
    "set_plot_style",
    "init_matplotlib_backend"
]

def init_matplotlib_backend() -> None:
    """
    Initializes the matplotlib backend for PyQt6 integration.
    
    This function configures matplotlib to use the Qt5Agg backend,
    which is compatible with PyQt6, and sets up styles and default
    parameters for optimal visualization in the MFE Toolbox UI.
    
    Returns
    -------
    None
    """
    try:
        import matplotlib
        matplotlib.use('Qt5Agg')  # Set backend for PyQt6 compatibility
        
        # Configure matplotlib styles and defaults
        import matplotlib.pyplot as plt
        plt.style.use('ggplot')  # Default style for financial plots
        
        # Set default figure parameters
        matplotlib.rcParams['figure.figsize'] = (8, 6)
        matplotlib.rcParams['figure.dpi'] = 100
        matplotlib.rcParams['savefig.dpi'] = 300
        matplotlib.rcParams['font.size'] = 10
        matplotlib.rcParams['axes.titlesize'] = 12
        matplotlib.rcParams['axes.labelsize'] = 10
        
        logger.info("Matplotlib backend initialized successfully for PyQt6")
    except ImportError as e:
        logger.error(f"Failed to initialize matplotlib backend: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error during matplotlib backend initialization: {str(e)}")
        raise