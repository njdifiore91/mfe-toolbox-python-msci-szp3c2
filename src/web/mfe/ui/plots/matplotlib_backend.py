import matplotlib  # version 3.7.1
from matplotlib.figure import Figure  # version 3.7.1
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT  # version 3.7.1
import matplotlib.pyplot as plt  # version 3.7.1
import numpy as np  # version 1.26.3
import pandas as pd  # version 2.1.4
from PyQt6.QtCore import QObject, pyqtSignal  # version 6.6.1
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QApplication  # version 6.6.1
import asyncio  # Python 3.12
import uuid  # Python standard library

# Global constants
FIGURE_DPI = 100
DEFAULT_FIGURE_SIZE = (8, 6)
STYLE_SHEET = 'ggplot'

def create_figure(figsize=DEFAULT_FIGURE_SIZE, dpi=FIGURE_DPI):
    """
    Creates a matplotlib Figure instance with proper sizing for PyQt6 integration.
    
    Parameters
    ----------
    figsize : tuple
        Figure size in inches (width, height)
    dpi : int
        Dots per inch for figure resolution
        
    Returns
    -------
    matplotlib.figure.Figure
        A properly configured matplotlib figure
    """
    plt.style.use(STYLE_SHEET)
    figure = Figure(figsize=figsize, dpi=dpi)
    return figure

def create_canvas(figure):
    """
    Creates a QtAgg canvas for a matplotlib figure that can be embedded in PyQt6 widgets.
    
    Parameters
    ----------
    figure : matplotlib.figure.Figure
        The figure to create a canvas for
        
    Returns
    -------
    FigureCanvasQTAgg
        A Qt-compatible canvas containing the figure
    """
    canvas = FigureCanvasQTAgg(figure)
    # Configure canvas properties for PyQt6 compatibility
    canvas.setFocus()
    return canvas

def create_navigation_toolbar(canvas, parent):
    """
    Creates a navigation toolbar for interactive matplotlib figures in PyQt6.
    
    Parameters
    ----------
    canvas : FigureCanvasQTAgg
        The canvas to create a toolbar for
    parent : QWidget
        The parent widget for the toolbar
        
    Returns
    -------
    NavigationToolbar2QT
        A navigation toolbar for the figure
    """
    toolbar = NavigationToolbar2QT(canvas, parent)
    return toolbar

def embed_figure(figure, parent, with_toolbar=True):
    """
    Embeds a matplotlib figure into a PyQt6 widget with optional toolbar.
    
    Parameters
    ----------
    figure : matplotlib.figure.Figure
        The figure to embed
    parent : QWidget
        The parent widget
    with_toolbar : bool
        Whether to include a navigation toolbar
        
    Returns
    -------
    QWidget
        A widget containing the embedded figure
    """
    canvas = create_canvas(figure)
    container = QWidget(parent)
    layout = QVBoxLayout(container)
    layout.addWidget(canvas)
    
    if with_toolbar:
        toolbar = create_navigation_toolbar(canvas, container)
        layout.addWidget(toolbar)
    
    container.setLayout(layout)
    return container

def clear_figure(figure):
    """
    Clears a matplotlib figure to prepare it for new plots.
    
    Parameters
    ----------
    figure : matplotlib.figure.Figure
        The figure to clear
    """
    figure.clear()

def update_canvas(canvas):
    """
    Updates a matplotlib canvas to reflect changes in the figure.
    
    Parameters
    ----------
    canvas : FigureCanvasQTAgg
        The canvas to update
    """
    canvas.draw_idle()
    # Process any pending Qt events to ensure timely visual updates
    QApplication.processEvents()

async def async_update_canvas(canvas):
    """
    Asynchronously updates a matplotlib canvas to prevent UI blocking.
    
    Parameters
    ----------
    canvas : FigureCanvasQTAgg
        The canvas to update asynchronously
    """
    await asyncio.sleep(0)  # Yield to the event loop
    canvas.draw_idle()
    # Process any pending Qt events to ensure timely visual updates
    QApplication.processEvents()

def save_figure(figure, filename, dpi=300):
    """
    Saves a matplotlib figure to a file with appropriate resolution.
    
    Parameters
    ----------
    figure : matplotlib.figure.Figure
        The figure to save
    filename : str
        The filename to save to
    dpi : int
        The resolution for the saved figure
        
    Returns
    -------
    bool
        True if save was successful, False otherwise
    """
    try:
        figure.savefig(filename, dpi=dpi)
        return True
    except Exception as e:
        print(f"Error saving figure: {e}")
        return False

def set_plot_style(style_name):
    """
    Sets the global matplotlib style for consistent visual appearance.
    
    Parameters
    ----------
    style_name : str
        The name of the matplotlib style to use
    """
    plt.style.use(style_name)
    global STYLE_SHEET
    STYLE_SHEET = style_name


class MatplotlibBackend:
    """
    Core class that manages matplotlib integration with PyQt6 for MFE UI components.
    
    This class maintains a registry of figures and canvases, allowing for centralized
    management of matplotlib resources within the application.
    """
    
    def __init__(self, style=STYLE_SHEET):
        """
        Initializes the matplotlib backend with default configuration.
        
        Parameters
        ----------
        style : str
            The matplotlib style to use
        """
        self._figure_registry = {}
        self._canvas_registry = {}
        self.style = style
        plt.style.use(self.style)
    
    def create_figure(self, figure_id, figsize=DEFAULT_FIGURE_SIZE, dpi=FIGURE_DPI):
        """
        Creates and registers a new matplotlib figure with a unique identifier.
        
        Parameters
        ----------
        figure_id : str
            Unique identifier for the figure
        figsize : tuple
            Figure size in inches (width, height)
        dpi : int
            Dots per inch for figure resolution
            
        Returns
        -------
        matplotlib.figure.Figure
            The created figure
        """
        figure = create_figure(figsize, dpi)
        self._figure_registry[figure_id] = figure
        return figure
    
    def get_figure(self, figure_id):
        """
        Retrieves a registered figure by its identifier.
        
        Parameters
        ----------
        figure_id : str
            The figure identifier
            
        Returns
        -------
        matplotlib.figure.Figure
            The requested figure or None if not found
        """
        return self._figure_registry.get(figure_id)
    
    def create_canvas_for_figure(self, figure_id):
        """
        Creates and registers a Qt canvas for a figure.
        
        Parameters
        ----------
        figure_id : str
            The figure identifier
            
        Returns
        -------
        FigureCanvasQTAgg
            The created canvas
        """
        figure = self.get_figure(figure_id)
        if figure is None:
            return None
        
        canvas = create_canvas(figure)
        self._canvas_registry[figure_id] = canvas
        return canvas
    
    def get_canvas(self, figure_id):
        """
        Retrieves a registered canvas by its figure identifier.
        
        Parameters
        ----------
        figure_id : str
            The figure identifier
            
        Returns
        -------
        FigureCanvasQTAgg
            The requested canvas or None if not found
        """
        return self._canvas_registry.get(figure_id)
    
    def embed_figure_in_widget(self, figure_id, parent, with_toolbar=True):
        """
        Embeds a registered figure into a PyQt6 widget.
        
        Parameters
        ----------
        figure_id : str
            The figure identifier
        parent : QWidget
            The parent widget
        with_toolbar : bool
            Whether to include a navigation toolbar
            
        Returns
        -------
        QWidget
            Widget containing the embedded figure
        """
        figure = self.get_figure(figure_id)
        if figure is None:
            return None
        
        return embed_figure(figure, parent, with_toolbar)
    
    def update_figure(self, figure_id):
        """
        Triggers an update for a registered figure and its canvas.
        
        Parameters
        ----------
        figure_id : str
            The figure identifier
            
        Returns
        -------
        bool
            True if update was successful, False otherwise
        """
        canvas = self.get_canvas(figure_id)
        if canvas is None:
            return False
        
        update_canvas(canvas)
        return True
    
    async def async_update_figure(self, figure_id):
        """
        Asynchronously updates a registered figure and its canvas.
        
        Parameters
        ----------
        figure_id : str
            The figure identifier
            
        Returns
        -------
        bool
            True if update was successful, False otherwise
        """
        canvas = self.get_canvas(figure_id)
        if canvas is None:
            return False
        
        await async_update_canvas(canvas)
        return True
    
    def clear_figure(self, figure_id):
        """
        Clears a registered figure to prepare it for new plots.
        
        Parameters
        ----------
        figure_id : str
            The figure identifier
            
        Returns
        -------
        bool
            True if clearing was successful, False otherwise
        """
        figure = self.get_figure(figure_id)
        if figure is None:
            return False
        
        clear_figure(figure)
        canvas = self.get_canvas(figure_id)
        if canvas is not None:
            update_canvas(canvas)
        
        return True
    
    def save_figure(self, figure_id, filename, dpi=300):
        """
        Saves a registered figure to a file.
        
        Parameters
        ----------
        figure_id : str
            The figure identifier
        filename : str
            The filename to save to
        dpi : int
            The resolution for the saved figure
            
        Returns
        -------
        bool
            True if save was successful, False otherwise
        """
        figure = self.get_figure(figure_id)
        if figure is None:
            return False
        
        return save_figure(figure, filename, dpi)
    
    def set_style(self, style_name):
        """
        Sets the matplotlib style for all figures.
        
        Parameters
        ----------
        style_name : str
            The name of the matplotlib style to use
        """
        set_plot_style(style_name)
        self.style = style_name
        
        # Update all figures and canvases
        for figure_id in self._figure_registry:
            self.update_figure(figure_id)
    
    def cleanup(self):
        """
        Releases resources held by registered figures and canvases.
        """
        for figure in self._figure_registry.values():
            plt.close(figure)
        
        self._figure_registry.clear()
        self._canvas_registry.clear()


class AsyncPlotWidget(QWidget):
    """
    A PyQt6 widget that supports asynchronous matplotlib plotting operations.
    
    This widget encapsulates a matplotlib figure and canvas in a PyQt6 widget, providing
    methods for asynchronous plotting operations to maintain UI responsiveness.
    """
    
    def __init__(self, parent, figsize=DEFAULT_FIGURE_SIZE, dpi=FIGURE_DPI, with_toolbar=True):
        """
        Initializes an AsyncPlotWidget with a matplotlib figure and canvas.
        
        Parameters
        ----------
        parent : QWidget
            The parent widget
        figsize : tuple
            Figure size in inches (width, height)
        dpi : int
            Dots per inch for figure resolution
        with_toolbar : bool
            Whether to include a navigation toolbar
        """
        super().__init__(parent)
        
        self._backend = MatplotlibBackend()
        self._figure_id = str(uuid.uuid4())  # Generate a unique ID
        self._figure = self._backend.create_figure(self._figure_id, figsize, dpi)
        self._canvas = self._backend.create_canvas_for_figure(self._figure_id)
        
        self._layout = QVBoxLayout(self)
        self.setLayout(self._layout)
        
        self._canvas_container = self._backend.embed_figure_in_widget(self._figure_id, self, with_toolbar)
        self._layout.addWidget(self._canvas_container)
    
    def get_figure(self):
        """
        Returns the matplotlib figure for direct manipulation.
        
        Returns
        -------
        matplotlib.figure.Figure
            The figure contained in this widget
        """
        return self._figure
    
    def get_canvas(self):
        """
        Returns the matplotlib canvas for direct access.
        
        Returns
        -------
        FigureCanvasQTAgg
            The canvas displaying the figure
        """
        return self._canvas
    
    def clear_plot(self):
        """
        Clears the plot for new data.
        """
        self._backend.clear_figure(self._figure_id)
    
    def update_plot(self):
        """
        Triggers an update of the plot display.
        """
        self._backend.update_figure(self._figure_id)
    
    async def async_update_plot(self):
        """
        Asynchronously updates the plot display.
        """
        await self._backend.async_update_figure(self._figure_id)
    
    def save_plot(self, filename, dpi=300):
        """
        Saves the current plot to a file.
        
        Parameters
        ----------
        filename : str
            The filename to save to
        dpi : int
            The resolution for the saved figure
            
        Returns
        -------
        bool
            True if save was successful, False otherwise
        """
        return self._backend.save_figure(self._figure_id, filename, dpi)