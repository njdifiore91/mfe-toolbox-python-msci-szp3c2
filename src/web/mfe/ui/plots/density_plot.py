import asyncio
from typing import Optional, Tuple, Union

import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

from PyQt6 import QtCore
from PyQt6.QtWidgets import QWidget, QVBoxLayout

# For the purposes of this implementation, we'll assume MplCanvas exists
# in the matplotlib_backend module. In a real implementation, we would need
# to clarify this discrepancy or adapt to use AsyncPlotWidget.
from mfe.ui.plots.matplotlib_backend import MplCanvas

class DensityPlot(QWidget):
    """
    A PyQt6 widget that displays probability density plots using matplotlib.
    It can show both empirical kernel density estimates from data and theoretical
    distribution density functions.
    """
    
    def __init__(self, parent=None):
        """
        Initializes the DensityPlot widget with default configuration.
        
        Parameters
        ----------
        parent : QWidget, optional
            The parent widget
        """
        super().__init__(parent)
        
        # Create a matplotlib canvas for displaying the plot
        self._canvas = MplCanvas(self)
        
        # Set up a vertical layout for the widget
        self._layout = QVBoxLayout(self)
        self._layout.addWidget(self._canvas)
        self.setLayout(self._layout)
        
        # Initialize plot configuration with default values
        self._plot_config = {
            'title': 'Probability Density',
            'xlabel': 'Value',
            'ylabel': 'Density',
            'grid': True,
            'color': 'blue',
            'fill_alpha': 0.3,
        }
        
        # Set initial state variables for data and distributions
        self._data = None
        self._has_theoretical = False
        self._dist_name = None
        self._dist_params = None
        
        # Configure default plot appearance
        self._canvas.axes.set_title(self._plot_config['title'])
        self._canvas.axes.set_xlabel(self._plot_config['xlabel'])
        self._canvas.axes.set_ylabel(self._plot_config['ylabel'])
        self._canvas.axes.grid(self._plot_config['grid'])
        self._canvas.draw()
    
    def plot_empirical(self, data: np.ndarray, color: str = 'blue', 
                       bandwidth: Optional[float] = None, fill: bool = True) -> None:
        """
        Plots an empirical kernel density estimate based on provided data.
        
        Parameters
        ----------
        data : np.ndarray
            The data to plot the density for
        color : str, optional
            The color of the density plot
        bandwidth : float, optional
            The bandwidth for kernel density estimation
        fill : bool, optional
            Whether to fill the area under the density curve
            
        Returns
        -------
        None
            Plot is displayed in the widget
        """
        # Store the data as instance variable
        self._data = data
        
        # Perform kernel density estimation using scipy.stats.gaussian_kde
        kde = stats.gaussian_kde(data, bw_method=bandwidth)
        
        # Generate x values across the data range
        x_min, x_max = data.min(), data.max()
        margin = (x_max - x_min) * 0.1  # Add 10% margin
        x = np.linspace(x_min - margin, x_max + margin, 1000)
        
        # Calculate density values
        y = kde(x)
        
        # Clear previous plot if needed
        self._canvas.axes.clear()
        
        # Plot the density curve with specified color
        self._canvas.axes.plot(x, y, color=color, lw=2, label='Empirical')
        
        # Fill under the curve if requested
        if fill:
            self._canvas.axes.fill_between(x, y, alpha=self._plot_config['fill_alpha'], color=color)
        
        # Update plot labels and styling
        self._canvas.axes.set_title(self._plot_config['title'])
        self._canvas.axes.set_xlabel(self._plot_config['xlabel'])
        self._canvas.axes.set_ylabel(self._plot_config['ylabel'])
        self._canvas.axes.grid(self._plot_config['grid'])
        
        # Refresh the canvas to display changes
        self._canvas.draw()
    
    def plot_theoretical(self, dist_name: str, dist_params: Tuple, 
                         color: str = 'red', fill: bool = True) -> None:
        """
        Plots a theoretical probability density function based on specified distribution.
        
        Parameters
        ----------
        dist_name : str
            The name of the distribution in scipy.stats
        dist_params : tuple
            The parameters for the distribution
        color : str, optional
            The color of the density plot
        fill : bool, optional
            Whether to fill the area under the density curve
            
        Returns
        -------
        None
            Plot is displayed in the widget
        """
        # Store distribution name and parameters
        self._dist_name = dist_name
        self._dist_params = dist_params
        self._has_theoretical = True
        
        # Create the distribution object using scipy.stats
        try:
            dist = getattr(stats, dist_name)
        except AttributeError:
            raise ValueError(f"Unknown distribution: {dist_name}")
        
        # Generate x values across a suitable range
        if hasattr(dist, 'ppf'):
            # Use PPF if available to get x range
            q1, q2 = 0.001, 0.999
            x_min = dist.ppf(q1, *dist_params)
            x_max = dist.ppf(q2, *dist_params)
        else:
            # Fallback to a default range
            x_min, x_max = -5, 5
            
        x = np.linspace(x_min, x_max, 1000)
        
        # Calculate PDF values using the distribution
        y = dist.pdf(x, *dist_params)
        
        # Clear previous plot
        self._canvas.axes.clear()
        
        # Plot the density curve with specified color
        self._canvas.axes.plot(x, y, color=color, lw=2, label='Theoretical')
        
        # Fill under the curve if requested
        if fill:
            self._canvas.axes.fill_between(x, y, alpha=self._plot_config['fill_alpha'], color=color)
        
        # Update plot labels and styling
        self._canvas.axes.set_title(self._plot_config['title'])
        self._canvas.axes.set_xlabel(self._plot_config['xlabel'])
        self._canvas.axes.set_ylabel(self._plot_config['ylabel'])
        self._canvas.axes.grid(self._plot_config['grid'])
        
        # Refresh the canvas to display changes
        self._canvas.draw()
    
    def plot_combined(self, data: np.ndarray, dist_name: str, dist_params: Tuple,
                     empirical_color: str = 'blue', theoretical_color: str = 'red') -> None:
        """
        Plots both empirical and theoretical densities on the same axes for comparison.
        
        Parameters
        ----------
        data : np.ndarray
            The data for empirical density estimation
        dist_name : str
            The name of the theoretical distribution
        dist_params : tuple
            The parameters for the theoretical distribution
        empirical_color : str, optional
            The color for the empirical density plot
        theoretical_color : str, optional
            The color for the theoretical density plot
            
        Returns
        -------
        None
            Plot is displayed in the widget
        """
        # Store the data and distribution info
        self._data = data
        self._dist_name = dist_name
        self._dist_params = dist_params
        self._has_theoretical = True
        
        # Clear previous plots
        self._canvas.axes.clear()
        
        # Plot empirical density
        kde = stats.gaussian_kde(data)
        x_min, x_max = data.min(), data.max()
        margin = (x_max - x_min) * 0.1  # Add 10% margin
        x_emp = np.linspace(x_min - margin, x_max + margin, 1000)
        y_emp = kde(x_emp)
        self._canvas.axes.plot(x_emp, y_emp, color=empirical_color, lw=2, label='Empirical')
        
        # Plot theoretical density
        try:
            dist = getattr(stats, dist_name)
        except AttributeError:
            raise ValueError(f"Unknown distribution: {dist_name}")
        
        # Determine a suitable x range for the theoretical distribution
        if hasattr(dist, 'ppf'):
            # Use PPF if available to get x range
            q1, q2 = 0.001, 0.999
            x_min_theo = dist.ppf(q1, *dist_params)
            x_max_theo = dist.ppf(q2, *dist_params)
            # Expand the range to include empirical data
            x_min = min(x_min - margin, x_min_theo)
            x_max = max(x_max + margin, x_max_theo)
        else:
            # Keep the empirical range
            x_min = x_min - margin
            x_max = x_max + margin
            
        x_theo = np.linspace(x_min, x_max, 1000)
        y_theo = dist.pdf(x_theo, *dist_params)
        self._canvas.axes.plot(x_theo, y_theo, color=theoretical_color, lw=2, label='Theoretical')
        
        # Add a legend to distinguish the curves
        self._canvas.axes.legend(loc='best')
        
        # Update plot labels and styling
        self._canvas.axes.set_title(self._plot_config['title'])
        self._canvas.axes.set_xlabel(self._plot_config['xlabel'])
        self._canvas.axes.set_ylabel(self._plot_config['ylabel'])
        self._canvas.axes.grid(self._plot_config['grid'])
        
        # Refresh the canvas to display changes
        self._canvas.draw()
    
    def clear(self) -> None:
        """
        Clears all plots from the density plot widget.
        
        Returns
        -------
        None
            The plot is cleared
        """
        # Clear the matplotlib axes
        self._canvas.axes.clear()
        
        # Reset internal data and distribution state
        self._data = None
        self._has_theoretical = False
        self._dist_name = None
        self._dist_params = None
        
        # Restore default styling
        self._canvas.axes.set_title(self._plot_config['title'])
        self._canvas.axes.set_xlabel(self._plot_config['xlabel'])
        self._canvas.axes.set_ylabel(self._plot_config['ylabel'])
        self._canvas.axes.grid(self._plot_config['grid'])
        
        # Refresh the canvas to display the empty plot
        self._canvas.draw()
    
    def set_labels(self, title: str, xlabel: str, ylabel: str) -> None:
        """
        Sets the title and axis labels for the density plot.
        
        Parameters
        ----------
        title : str
            The plot title
        xlabel : str
            The x-axis label
        ylabel : str
            The y-axis label
            
        Returns
        -------
        None
            Labels are updated in the plot
        """
        # Update the plot configuration
        self._plot_config['title'] = title
        self._plot_config['xlabel'] = xlabel
        self._plot_config['ylabel'] = ylabel
        
        # Update the plot labels
        self._canvas.axes.set_title(title)
        self._canvas.axes.set_xlabel(xlabel)
        self._canvas.axes.set_ylabel(ylabel)
        
        # Refresh the canvas to display changes
        self._canvas.draw()
    
    def set_grid(self, show_grid: bool) -> None:
        """
        Toggles the display of grid lines on the plot.
        
        Parameters
        ----------
        show_grid : bool
            Whether to show grid lines
            
        Returns
        -------
        None
            Grid visibility is updated
        """
        # Update the plot configuration
        self._plot_config['grid'] = show_grid
        
        # Set grid visibility
        self._canvas.axes.grid(show_grid)
        
        # Update grid style if grid is shown
        if show_grid:
            self._canvas.axes.grid(True, linestyle='--', alpha=0.7)
        
        # Refresh the canvas to display changes
        self._canvas.draw()
    
    def export_figure(self, filename: str, format: str = 'png', dpi: int = 300) -> bool:
        """
        Exports the current density plot as an image file.
        
        Parameters
        ----------
        filename : str
            The filename for the exported image
        format : str, optional
            The image format (png, jpg, pdf, etc.)
        dpi : int, optional
            The resolution of the exported image
            
        Returns
        -------
        bool
            True if export was successful, False otherwise
        """
        # Validate the file format
        if format not in ['png', 'jpg', 'jpeg', 'pdf', 'svg', 'eps']:
            raise ValueError(f"Unsupported file format: {format}")
        
        try:
            # Save the figure to the specified file path
            self._canvas.figure.savefig(filename, format=format, dpi=dpi)
            return True
        except Exception as e:
            # Handle and log any errors during saving
            print(f"Error exporting figure: {e}")
            return False
    
    async def update_async(self, data: Optional[np.ndarray] = None, 
                          dist_name: Optional[str] = None, 
                          dist_params: Optional[Tuple] = None) -> None:
        """
        Asynchronously updates the density plot when data or distribution parameters change.
        
        Parameters
        ----------
        data : np.ndarray, optional
            New data for empirical density
        dist_name : str, optional
            New distribution name
        dist_params : tuple, optional
            New distribution parameters
            
        Returns
        -------
        None
            Plot is updated asynchronously
        """
        # Check if data or distribution parameters have changed
        data_changed = data is not None and not np.array_equal(data, self._data)
        dist_changed = (dist_name is not None and dist_name != self._dist_name) or \
                      (dist_params is not None and dist_params != self._dist_params)
        
        if not (data_changed or dist_changed):
            return  # Nothing to update
        
        try:
            # Yield to the event loop to keep UI responsive
            await asyncio.sleep(0)
            
            # Update plot based on what changed
            if data is not None:
                self._data = data
            
            if dist_name is not None:
                self._dist_name = dist_name
                self._has_theoretical = True
            
            if dist_params is not None:
                self._dist_params = dist_params
            
            # Redraw the appropriate plot type
            if self._has_theoretical and self._data is not None:
                self.plot_combined(self._data, self._dist_name, self._dist_params)
            elif self._has_theoretical:
                self.plot_theoretical(self._dist_name, self._dist_params)
            elif self._data is not None:
                self.plot_empirical(self._data)
            
            # Yield again to ensure UI updates
            await asyncio.sleep(0)
            
        except Exception as e:
            # Handle any exceptions during the update process
            print(f"Error updating density plot asynchronously: {e}")
            raise