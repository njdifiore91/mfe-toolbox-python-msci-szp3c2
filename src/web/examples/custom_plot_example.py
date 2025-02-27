import sys
from pathlib import Path
import asyncio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton
)
from PyQt6.QtCore import Qt, pyqtSignal

# Add necessary paths to sys.path to enable imports
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(str(SCRIPT_DIR.parent.parent))

# Import MFE Toolbox components
from web.mfe.ui.plots.matplotlib_backend import AsyncPlotWidget
from web.mfe.ui.plot_widgets import BasePlotWidget
from backend.mfe.utils.numpy_helpers import ensure_array

# Constants
DEFAULT_FIGURE_SIZE = (8, 6)
DEFAULT_DPI = 100

def generate_sample_data(n_points: int, data_type: str = 'random'):
    """
    Generates synthetic time series data for demonstrating custom plots.
    
    Parameters
    ----------
    n_points : int
        Number of data points to generate
    data_type : str, default='random'
        Type of data to generate: 'random', 'sine', 'trend', etc.
        
    Returns
    -------
    tuple
        (x_data, y_data) as numpy arrays
    """
    x_data = np.linspace(0, 10, n_points)
    
    if data_type == 'random':
        y_data = np.random.randn(n_points) * 0.5 + np.sin(x_data)
    elif data_type == 'sine':
        y_data = np.sin(x_data) + 0.1 * np.random.randn(n_points)
    elif data_type == 'trend':
        y_data = 0.5 * x_data + np.random.randn(n_points) * 0.5
    elif data_type == 'exponential':
        y_data = np.exp(x_data / 5) + np.random.randn(n_points) * 0.5
    else:
        # Default to random data
        y_data = np.random.randn(n_points)
    
    # Ensure the data is in the proper format using the MFE utility
    x_data = ensure_array(x_data)
    y_data = ensure_array(y_data)
    
    return x_data, y_data

def create_custom_plot(x_data, y_data, plot_params=None):
    """
    Creates a custom matplotlib figure with enhanced visual features.
    
    Parameters
    ----------
    x_data : np.ndarray
        X-axis data
    y_data : np.ndarray
        Y-axis data
    plot_params : dict, optional
        Customization parameters for the plot
        
    Returns
    -------
    matplotlib.figure.Figure
        Configured matplotlib figure
    """
    if plot_params is None:
        plot_params = {}
    
    # Get plot parameters with defaults
    figsize = plot_params.get('figsize', DEFAULT_FIGURE_SIZE)
    title = plot_params.get('title', 'Custom Plot Example')
    xlabel = plot_params.get('xlabel', 'X Axis')
    ylabel = plot_params.get('ylabel', 'Y Axis')
    grid = plot_params.get('grid', True)
    line_style = plot_params.get('line_style', '-')
    color = plot_params.get('color', 'blue')
    marker = plot_params.get('marker', 'o')
    alpha = plot_params.get('alpha', 0.7)
    
    # Create figure and axis
    fig = plt.figure(figsize=figsize, dpi=DEFAULT_DPI)
    ax = fig.add_subplot(111)
    
    # Plot the data
    ax.plot(x_data, y_data, linestyle=line_style, color=color, 
            marker=marker, alpha=alpha, label='Data')
    
    # Add title and labels
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    # Add grid if specified
    ax.grid(grid)
    
    # Add legend if a label was provided
    ax.legend()
    
    # Tight layout for better spacing
    fig.tight_layout()
    
    return fig

def create_multi_plot(x_data, y_data, plot_params=None):
    """
    Creates a figure with multiple subplots showing different 
    visualizations of the same data.
    
    Parameters
    ----------
    x_data : np.ndarray
        X-axis data
    y_data : np.ndarray
        Y-axis data
    plot_params : dict, optional
        Customization parameters for the plot
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure with multiple subplots
    """
    if plot_params is None:
        plot_params = {}
    
    # Get plot parameters with defaults
    figsize = plot_params.get('figsize', (10, 8))
    title = plot_params.get('title', 'Multi-Plot Visualization')
    
    # Create figure with a 2x2 grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=DEFAULT_DPI)
    
    # Flatten axes for easier indexing
    axes = axes.flatten()
    
    # Line plot (top-left)
    axes[0].plot(x_data, y_data, color='blue', marker='o', alpha=0.7)
    axes[0].set_title('Line Plot')
    axes[0].set_xlabel('X Axis')
    axes[0].set_ylabel('Y Axis')
    axes[0].grid(True)
    
    # Histogram (top-right)
    axes[1].hist(y_data, bins=20, color='green', alpha=0.7)
    axes[1].set_title('Histogram')
    axes[1].set_xlabel('Value')
    axes[1].set_ylabel('Frequency')
    axes[1].grid(True)
    
    # Scatter plot (bottom-left)
    axes[2].scatter(x_data, y_data, color='red', alpha=0.7)
    axes[2].set_title('Scatter Plot')
    axes[2].set_xlabel('X Axis')
    axes[2].set_ylabel('Y Axis')
    axes[2].grid(True)
    
    # Rolling mean (bottom-right)
    window_size = max(1, len(y_data) // 10)
    rolling_mean = np.convolve(y_data, np.ones(window_size)/window_size, mode='valid')
    rolling_x = x_data[window_size-1:]
    axes[3].plot(rolling_x, rolling_mean, color='purple', linestyle='-')
    axes[3].set_title(f'Rolling Mean (Window: {window_size})')
    axes[3].set_xlabel('X Axis')
    axes[3].set_ylabel('Value')
    axes[3].grid(True)
    
    # Add overall title to the figure
    fig.suptitle(title, fontsize=16)
    
    # Adjust layout for proper spacing
    fig.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust rect to make room for suptitle
    
    return fig

class CustomPlotWidget(BasePlotWidget):
    """
    A custom widget that extends BasePlotWidget to demonstrate 
    specialized plotting capabilities.
    """
    
    def __init__(self, parent=None):
        """
        Initializes the custom plot widget with default settings.
        
        Parameters
        ----------
        parent : QWidget, optional
            Parent widget
        """
        super().__init__(parent)
        
        # Initialize data properties
        self._x_data = None
        self._y_data = None
        self._plot_params = {
            'title': 'Custom Plot Example',
            'xlabel': 'X Axis',
            'ylabel': 'Y Axis',
            'grid': True,
            'line_style': '-',
            'color': 'blue',
            'marker': 'o',
            'alpha': 0.7
        }
        self._initialized = False
    
    def set_data(self, x_data, y_data):
        """
        Sets the data to be visualized in the custom plot.
        
        Parameters
        ----------
        x_data : np.ndarray
            X-axis data
        y_data : np.ndarray
            Y-axis data
        """
        # Validate and store the data
        self._x_data = ensure_array(x_data)
        self._y_data = ensure_array(y_data)
        
        # Update the plot
        if self._x_data is not None and self._y_data is not None:
            self.update_plot()
            self._initialized = True
    
    def set_plot_params(self, params):
        """
        Configures the plot appearance parameters.
        
        Parameters
        ----------
        params : dict
            Plot parameters to update
        """
        if not isinstance(params, dict):
            raise TypeError("Plot parameters must be a dictionary")
        
        # Update plot parameters
        self._plot_params.update(params)
        
        # Update the plot if initialized
        if self._initialized:
            self.update_plot()
    
    def update_plot(self):
        """
        Updates the plot with current data and parameters.
        """
        if self._x_data is None or self._y_data is None:
            return
        
        # Get the base figure
        fig = self.get_figure()
        fig.clf()  # Clear the figure
        
        # Create new axes
        ax = fig.add_subplot(111)
        
        # Plot the data
        ax.plot(self._x_data, self._y_data, 
                linestyle=self._plot_params.get('line_style', '-'),
                color=self._plot_params.get('color', 'blue'),
                marker=self._plot_params.get('marker', 'o'),
                alpha=self._plot_params.get('alpha', 0.7),
                label='Data')
        
        # Add title and labels
        ax.set_title(self._plot_params.get('title', 'Custom Plot Example'))
        ax.set_xlabel(self._plot_params.get('xlabel', 'X Axis'))
        ax.set_ylabel(self._plot_params.get('ylabel', 'Y Axis'))
        
        # Add grid if specified
        ax.grid(self._plot_params.get('grid', True))
        
        # Add legend if a label was provided
        ax.legend()
        
        # Tight layout for better spacing
        fig.tight_layout()
        
        # Update the plot widget
        super().update_plot()
    
    async def async_update_plot(self):
        """
        Asynchronously updates the plot to prevent UI blocking.
        """
        if self._x_data is None or self._y_data is None:
            return
        
        # Yield control to event loop
        await asyncio.sleep(0)
        
        # Get the base figure
        fig = self.get_figure()
        fig.clf()  # Clear the figure
        
        # Create new axes
        ax = fig.add_subplot(111)
        
        # Plot the data
        ax.plot(self._x_data, self._y_data, 
                linestyle=self._plot_params.get('line_style', '-'),
                color=self._plot_params.get('color', 'blue'),
                marker=self._plot_params.get('marker', 'o'),
                alpha=self._plot_params.get('alpha', 0.7),
                label='Data')
        
        # Add title and labels
        ax.set_title(self._plot_params.get('title', 'Custom Plot Example'))
        ax.set_xlabel(self._plot_params.get('xlabel', 'X Axis'))
        ax.set_ylabel(self._plot_params.get('ylabel', 'Y Axis'))
        
        # Add grid if specified
        ax.grid(self._plot_params.get('grid', True))
        
        # Add legend if a label was provided
        ax.legend()
        
        # Tight layout for better spacing
        fig.tight_layout()
        
        # Update the plot widget asynchronously
        await super().async_update_plot()
    
    def clear(self):
        """
        Clears the plot and resets data.
        """
        self._x_data = None
        self._y_data = None
        self._initialized = False
        super().clear()

class MultiPlotWidget(BasePlotWidget):
    """
    A widget demonstrating multiple plots in a single visualization.
    """
    
    def __init__(self, parent=None):
        """
        Initializes the multi-plot widget.
        
        Parameters
        ----------
        parent : QWidget, optional
            Parent widget
        """
        super().__init__(parent)
        
        # Initialize data properties
        self._x_data = None
        self._y_data = None
        self._plot_params = {
            'title': 'Multi-Plot Visualization',
            'figsize': (10, 8)
        }
    
    def set_data(self, x_data, y_data):
        """
        Sets the data to be visualized across multiple plots.
        
        Parameters
        ----------
        x_data : np.ndarray
            X-axis data
        y_data : np.ndarray
            Y-axis data
        """
        # Validate and store the data
        self._x_data = ensure_array(x_data)
        self._y_data = ensure_array(y_data)
        
        # Update the plots
        if self._x_data is not None and self._y_data is not None:
            self.update_plot()
    
    def update_plot(self):
        """
        Updates all subplots with current data.
        """
        if self._x_data is None or self._y_data is None:
            return
        
        # Get the base figure
        fig = self.get_figure()
        fig.clf()  # Clear the figure
        
        # Create a 2x2 grid of subplots
        axes = []
        for i in range(4):
            axes.append(fig.add_subplot(2, 2, i+1))
        
        # Line plot (top-left)
        axes[0].plot(self._x_data, self._y_data, color='blue', marker='o', alpha=0.7)
        axes[0].set_title('Line Plot')
        axes[0].set_xlabel('X Axis')
        axes[0].set_ylabel('Y Axis')
        axes[0].grid(True)
        
        # Histogram (top-right)
        axes[1].hist(self._y_data, bins=20, color='green', alpha=0.7)
        axes[1].set_title('Histogram')
        axes[1].set_xlabel('Value')
        axes[1].set_ylabel('Frequency')
        axes[1].grid(True)
        
        # Scatter plot (bottom-left)
        axes[2].scatter(self._x_data, self._y_data, color='red', alpha=0.7)
        axes[2].set_title('Scatter Plot')
        axes[2].set_xlabel('X Axis')
        axes[2].set_ylabel('Y Axis')
        axes[2].grid(True)
        
        # Rolling mean (bottom-right)
        window_size = max(1, len(self._y_data) // 10)
        rolling_mean = np.convolve(self._y_data, np.ones(window_size)/window_size, mode='valid')
        rolling_x = self._x_data[window_size-1:]
        axes[3].plot(rolling_x, rolling_mean, color='purple', linestyle='-')
        axes[3].set_title(f'Rolling Mean (Window: {window_size})')
        axes[3].set_xlabel('X Axis')
        axes[3].set_ylabel('Value')
        axes[3].grid(True)
        
        # Add overall title to the figure
        fig.suptitle(self._plot_params.get('title', 'Multi-Plot Visualization'), fontsize=16)
        
        # Adjust layout for proper spacing
        fig.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust rect to make room for suptitle
        
        # Update the plot widget
        super().update_plot()
    
    async def async_update_plot(self):
        """
        Asynchronously updates all plots.
        """
        if self._x_data is None or self._y_data is None:
            return
        
        # Yield control to event loop
        await asyncio.sleep(0)
        
        # Get the base figure
        fig = self.get_figure()
        fig.clf()  # Clear the figure
        
        # Create a 2x2 grid of subplots
        axes = []
        for i in range(4):
            axes.append(fig.add_subplot(2, 2, i+1))
        
        # Line plot (top-left)
        axes[0].plot(self._x_data, self._y_data, color='blue', marker='o', alpha=0.7)
        axes[0].set_title('Line Plot')
        axes[0].set_xlabel('X Axis')
        axes[0].set_ylabel('Y Axis')
        axes[0].grid(True)
        
        # Histogram (top-right)
        axes[1].hist(self._y_data, bins=20, color='green', alpha=0.7)
        axes[1].set_title('Histogram')
        axes[1].set_xlabel('Value')
        axes[1].set_ylabel('Frequency')
        axes[1].grid(True)
        
        # Scatter plot (bottom-left)
        axes[2].scatter(self._x_data, self._y_data, color='red', alpha=0.7)
        axes[2].set_title('Scatter Plot')
        axes[2].set_xlabel('X Axis')
        axes[2].set_ylabel('Y Axis')
        axes[2].grid(True)
        
        # Rolling mean (bottom-right)
        window_size = max(1, len(self._y_data) // 10)
        rolling_mean = np.convolve(self._y_data, np.ones(window_size)/window_size, mode='valid')
        rolling_x = self._x_data[window_size-1:]
        axes[3].plot(rolling_x, rolling_mean, color='purple', linestyle='-')
        axes[3].set_title(f'Rolling Mean (Window: {window_size})')
        axes[3].set_xlabel('X Axis')
        axes[3].set_ylabel('Value')
        axes[3].grid(True)
        
        # Add overall title to the figure
        fig.suptitle(self._plot_params.get('title', 'Multi-Plot Visualization'), fontsize=16)
        
        # Adjust layout for proper spacing
        fig.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust rect to make room for suptitle
        
        # Update the plot widget asynchronously
        await super().async_update_plot()

class MainWindow(QMainWindow):
    """
    Main application window demonstrating custom plot integration.
    """
    
    def __init__(self):
        """
        Initializes the main window with custom plot widgets and controls.
        """
        super().__init__()
        
        # Set window properties
        self.setWindowTitle("MFE Toolbox - Custom Plot Example")
        self.setGeometry(100, 100, 1000, 700)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self._main_layout = QVBoxLayout(central_widget)
        
        # Create custom plot widgets
        self._custom_plot = CustomPlotWidget()
        self._multi_plot = MultiPlotWidget()
        
        # Initially show only the custom plot
        self._main_layout.addWidget(self._custom_plot)
        self._main_layout.addWidget(self._multi_plot)
        self._multi_plot.hide()
        
        # Create buttons for controls
        self._buttons_layout = QHBoxLayout()
        
        # Update button
        self._update_button = QPushButton("Update Plots")
        self._update_button.clicked.connect(self.update_plots)
        self._buttons_layout.addWidget(self._update_button)
        
        # Clear button
        self._clear_button = QPushButton("Clear Plots")
        self._clear_button.clicked.connect(self.clear_plots)
        self._buttons_layout.addWidget(self._clear_button)
        
        # Switch button
        self._switch_button = QPushButton("Switch to Multi-Plot")
        self._switch_button.clicked.connect(self.switch_plot_type)
        self._buttons_layout.addWidget(self._switch_button)
        
        # Add buttons layout to main layout
        self._main_layout.addLayout(self._buttons_layout)
        
        # Generate initial data
        self.update_plots()
    
    def update_plots(self):
        """
        Updates all plots with new random data.
        """
        # Generate new sample data
        x_data, y_data = generate_sample_data(100, 'sine')
        
        # Update both plot widgets
        self._custom_plot.set_data(x_data, y_data)
        self._multi_plot.set_data(x_data, y_data)
    
    def clear_plots(self):
        """
        Clears all plots.
        """
        self._custom_plot.clear()
        self._multi_plot.clear()
    
    def switch_plot_type(self):
        """
        Toggles between different plot types.
        """
        if self._custom_plot.isVisible():
            self._custom_plot.hide()
            self._multi_plot.show()
            self._switch_button.setText("Switch to Custom Plot")
        else:
            self._custom_plot.show()
            self._multi_plot.hide()
            self._switch_button.setText("Switch to Multi-Plot")

def main():
    """
    Main entry point for the example application.
    """
    # Add necessary paths to sys.path for imports
    sys.path.append(str(SCRIPT_DIR.parent.parent))
    
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()

if __name__ == "__main__":
    main()