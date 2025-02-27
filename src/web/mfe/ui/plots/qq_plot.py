import numpy as np  # version 1.26.3
import pandas as pd  # version 2.1.4
import statsmodels.api as sm  # version 0.14.1
import scipy.stats  # version 1.11.4
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QLabel, QPushButton  # version 6.6.1
from PyQt6.QtCore import pyqtSignal, Qt, pyqtSlot  # version 6.6.1
import matplotlib.pyplot as plt  # version 3.7.1
from typing import Optional, Dict, Tuple  # standard library
import asyncio  # standard library
import logging  # standard library

# Internal imports
from .matplotlib_backend import AsyncPlotWidget, MatplotlibBackend  # Base matplotlib integration components for PyQt6
from ..plot_widgets import BasePlotWidget  # Base plot widget that provides common functionality
from ....backend.mfe.core.distributions import distribution_fit, jarque_bera, shapiro_wilk, ks_test, GeneralizedErrorDistribution, SkewedTDistribution  # Statistical distribution functions and tests for QQ plot generation

# Initialize logger
logger = logging.getLogger(__name__)

# Define supported distributions
DISTRIBUTIONS = {
    "normal": "Normal",
    "t": "Student's t",
    "ged": "Generalized Error",
    "skewed_t": "Skewed t"
}

# Default style options
DEFAULT_STYLE = {
    "grid": True,
    "line_color": "red",
    "marker_color": "blue",
    "marker_style": "o",
    "marker_size": 4
}

def get_qq_data(data: np.ndarray, dist_type: str, dist_params: Optional[dict] = None) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Generates data for QQ plots comparing empirical data against theoretical distributions.

    Args:
        data (np.ndarray): Input data array.
        dist_type (str): Type of theoretical distribution.
        dist_params (Optional[dict]): Distribution parameters.

    Returns:
        Tuple[np.ndarray, np.ndarray, dict]: A tuple containing theoretical quantiles, empirical quantiles, and distribution parameters.
    """
    # Validate the input data array and distribution type
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a NumPy array.")
    if dist_type not in DISTRIBUTIONS:
        raise ValueError(f"Distribution type must be one of: {', '.join(DISTRIBUTIONS.keys())}")

    # If dist_params not provided, fit the specified distribution to data
    if dist_params is None:
        fit_results = distribution_fit(data, dist_type)
        dist_params = {k: v for k, v in fit_results.items() if k not in ['tests', 'distribution', 'loglikelihood', 'aic', 'bic']}

    # Calculate empirical quantiles from data using statsmodels
    ecdf = sm.distributions.ECDF(data)
    empirical_quantiles = ecdf(np.sort(data))

    # Generate theoretical quantiles based on distribution type
    if dist_type == "normal":
        theoretical_quantiles = scipy.stats.norm.ppf(empirical_quantiles, loc=dist_params.get('mu', 0), scale=dist_params.get('sigma', 1))
    elif dist_type == "t":
        theoretical_quantiles = scipy.stats.t.ppf(empirical_quantiles, df=dist_params.get('df', 5), loc=dist_params.get('mu', 0), scale=dist_params.get('sigma', 1))
    elif dist_type == "ged":
        theoretical_quantiles = ged_ppf(empirical_quantiles, mu=dist_params.get('mu', 0), sigma=dist_params.get('sigma', 1), nu=dist_params.get('nu', 2))
    elif dist_type == "skewed_t":
        theoretical_quantiles = skewt_ppf(empirical_quantiles, mu=dist_params.get('mu', 0), sigma=dist_params.get('sigma', 1), nu=dist_params.get('nu', 5), lambda_=dist_params.get('lambda', 0))
    else:
        raise ValueError(f"Unsupported distribution type: {dist_type}")

    # Return theoretical quantiles, empirical quantiles, and distribution parameters
    return theoretical_quantiles, empirical_quantiles, dist_params

def create_qq_figure(data: np.ndarray, dist_type: str, dist_params: Optional[dict] = None, style_options: Optional[dict] = None) -> plt.figure.Figure:
    """
    Creates a matplotlib figure with a QQ plot.

    Args:
        data (np.ndarray): Input data array.
        dist_type (str): Type of theoretical distribution.
        dist_params (Optional[dict]): Distribution parameters.
        style_options (Optional[dict]): Styling options for the plot.

    Returns:
        plt.figure.Figure: Matplotlib figure containing QQ plot.
    """
    # Get QQ data using get_qq_data function
    theoretical_quantiles, empirical_quantiles, dist_params = get_qq_data(data, dist_type, dist_params)

    # Create a new matplotlib figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Set up axes and plotting area
    ax.scatter(theoretical_quantiles, empirical_quantiles, label="Data", color=style_options.get("marker_color", "blue"), marker=style_options.get("marker_style", "o"), s=style_options.get("marker_size", 4)**2)

    # Add reference line (y=x)
    min_val = min(np.min(theoretical_quantiles), np.min(empirical_quantiles))
    max_val = max(np.max(theoretical_quantiles), np.max(empirical_quantiles))
    ax.plot([min_val, max_val], [min_val, max_val], linestyle='--', color=style_options.get("line_color", "red"), label="Reference")

    # Add grid, labels, and title
    ax.grid(style_options.get("grid", True))
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Empirical Quantiles")
    ax.set_title(f"Q-Q Plot ({DISTRIBUTIONS[dist_type]})")
    ax.legend()

    # Return the created figure
    return fig

def calculate_qq_stats(theoretical_quantiles: np.ndarray, empirical_quantiles: np.ndarray) -> dict:
    """
    Calculates statistics for QQ plot to assess distributional fit.

    Args:
        theoretical_quantiles (np.ndarray): Theoretical quantiles.
        empirical_quantiles (np.ndarray): Empirical quantiles.

    Returns:
        dict: Dictionary with statistical measures of fit quality.
    """
    # Calculate mean squared error (MSE) between theoretical and empirical quantiles
    mse = np.mean((theoretical_quantiles - empirical_quantiles)**2)

    # Calculate R-squared (coefficient of determination)
    correlation_matrix = np.corrcoef(theoretical_quantiles, empirical_quantiles)
    correlation = correlation_matrix[0, 1]
    r_squared = correlation**2

    # Calculate maximum deviation point
    max_deviation = np.max(np.abs(theoretical_quantiles - empirical_quantiles))

    # Return statistics in a dictionary
    return {
        "mse": mse,
        "r_squared": r_squared,
        "max_deviation": max_deviation
    }

async def async_create_qq_figure(data: np.ndarray, dist_type: str, dist_params: Optional[dict] = None, style_options: Optional[dict] = None) -> plt.figure.Figure:
    """
    Asynchronously creates a QQ plot figure to prevent UI blocking.

    Args:
        data (np.ndarray): Input data array.
        dist_type (str): Type of theoretical distribution.
        dist_params (Optional[dict]): Distribution parameters.
        style_options (Optional[dict]): Styling options for the plot.

    Returns:
        plt.figure.Figure: Matplotlib figure containing QQ plot.
    """
    # Yield control to event loop with await asyncio.sleep(0)
    await asyncio.sleep(0)

    # Call create_qq_figure with provided parameters
    figure = create_qq_figure(data, dist_type, dist_params, style_options)

    # Return the created figure
    return figure

class QQPlot(BasePlotWidget):
    """
    Interactive QQ plot widget for distribution analysis and comparison.
    """

    distribution_changed = pyqtSignal()
    data_changed = pyqtSignal()

    def __init__(self, parent: QWidget):
        """
        Initializes the QQ plot widget with distribution selection controls.

        Args:
            parent (QWidget): Parent widget.
        """
        # Call BasePlotWidget constructor
        super().__init__(parent)

        # Initialize properties with default values
        self._data = None
        self._dist_type = "normal"
        self._dist_params = {}
        self._style_options = DEFAULT_STYLE
        self._stats = {}

        # Create distribution selector dropdown with available distributions
        self._dist_selector = QComboBox()
        for key, value in DISTRIBUTIONS.items():
            self._dist_selector.addItem(value, key)
        self._dist_selector.currentIndexChanged.connect(self._on_distribution_changed)

        # Set up QQ plot-specific UI controls
        dist_layout = QHBoxLayout()
        dist_layout.addWidget(QLabel("Distribution:"))
        dist_layout.addWidget(self._dist_selector)

        # Add distribution selection layout to the main layout
        self._layout.insertLayout(0, dist_layout)

        # Define signals for distribution and data changes
        self.distribution_changed.connect(self.async_update_plot)
        self.data_changed.connect(self.async_update_plot)

        # Initialize statistics display
        self._stats_panel = QWidget()
        self._stats_layout = QVBoxLayout()
        self._stats_panel.setLayout(self._stats_layout)
        self._layout.addWidget(self._stats_panel)
        self._stats_panel.hide()

        self._mse_label = QLabel()
        self._r_squared_label = QLabel()
        self._max_deviation_label = QLabel()

        self._stats_layout.addWidget(QLabel("Statistics:"))
        self._stats_layout.addWidget(self._mse_label)
        self._stats_layout.addWidget(self._r_squared_label)
        self._stats_layout.addWidget(self._max_deviation_label)

        # Add export button
        self._export_button = QPushButton("Export Statistics")
        self._export_button.clicked.connect(self._on_export_stats)
        self._stats_layout.addWidget(self._export_button)

        logger.debug("QQPlot widget initialized")

    def set_data(self, data: np.ndarray) -> None:
        """
        Sets the data to be visualized in the QQ plot.

        Args:
            data (np.ndarray): Input data array.
        """
        # Validate data as numeric array
        if not isinstance(data, np.ndarray):
            raise TypeError("Input data must be a NumPy array.")

        # Store data in _data property
        self._data = data

        # Reset distribution parameters
        self._dist_params = {}

        # Update plot if widget is already initialized
        if self._initialized:
            self.async_update_plot()

        # Emit data_changed signal
        self.data_changed.emit()

    def set_distribution(self, dist_type: str, dist_params: Optional[dict] = None) -> None:
        """
        Sets the theoretical distribution to compare against.

        Args:
            dist_type (str): Type of theoretical distribution.
            dist_params (Optional[dict]): Distribution parameters.
        """
        # Validate dist_type is one of the supported distributions
        if dist_type not in DISTRIBUTIONS:
            raise ValueError(f"Distribution type must be one of: {', '.join(DISTRIBUTIONS.keys())}")

        # Store distribution type and parameters
        self._dist_type = dist_type
        self._dist_params = dist_params

        # Update the distribution selector UI
        index = self._dist_selector.findData(dist_type)
        if index >= 0:
            self._dist_selector.setCurrentIndex(index)

        # Update plot if data is already available
        if self._data is not None and self._initialized:
            self.async_update_plot()

        # Emit distribution_changed signal
        self.distribution_changed.emit()

    def set_style_options(self, options: dict) -> None:
        """
        Sets style options for the QQ plot appearance.

        Args:
            options (dict): Dictionary of style options.
        """
        # Validate options dictionary
        if not isinstance(options, dict):
            raise TypeError("Style options must be a dictionary.")

        # Update _style_options with new values
        self._style_options.update(options)

        # Update plot if data is already available
        if self._data is not None and self._initialized:
            self.async_update_plot()

        # Log style changes
        logger.debug(f"Updated style options: {options.keys()}")

    def _create_figure(self) -> plt.figure.Figure:
        """
        Creates the QQ plot figure based on current settings.

        Returns:
            plt.figure.Figure: Created figure with QQ plot.
        """
        # Check if data is available
        if self._data is None:
            logger.warning("Cannot create plot: No data available.")
            return None

        # Call create_qq_figure with current parameters
        fig = create_qq_figure(self._data, self._dist_type, self._dist_params, self._style_options)

        # Calculate statistics using calculate_qq_stats
        theoretical_quantiles, empirical_quantiles, _ = get_qq_data(self._data, self._dist_type, self._dist_params)
        self._stats = calculate_qq_stats(theoretical_quantiles, empirical_quantiles)

        # Update stats display in UI
        self._update_stats_display()

        # Return the created figure
        return fig

    async def async_update_plot(self) -> None:
        """
        Asynchronously updates the QQ plot to prevent UI blocking.
        """
        # Check if data is available
        if self._data is None:
            logger.warning("Cannot update plot: No data available.")
            return

        # Yield control with await asyncio.sleep(0)
        await asyncio.sleep(0)

        # Get figure using await async_create_qq_figure
        figure = self._create_figure()

        # Apply styling to the figure
        if figure:
            self._plot_widget.get_figure().clear()
            for ax in figure.get_axes():
                self._plot_widget.get_figure().add_subplot(ax)
            await self._plot_widget.async_update_plot()

        # Log the async update operation
        logger.debug("QQPlot updated asynchronously.")

    def get_statistics(self) -> dict:
        """
        Returns statistics about the current QQ plot fit.

        Returns:
            dict: Dictionary with fit statistics.
        """
        # Return copy of _stats dictionary with fit metrics
        if self._stats:
            return self._stats.copy()
        else:
            return {}

    @pyqtSlot(str)
    def _on_distribution_changed(self, dist_type: str) -> None:
        """
        Slot for handling distribution selector changes.

        Args:
            dist_type (str): Selected distribution type.
        """
        # Convert UI selection to internal distribution type
        dist_type = self._dist_selector.itemData(self._dist_selector.currentIndex())

        # Clear existing distribution parameters
        self._dist_params = {}

        # Call set_distribution with new type
        self.set_distribution(dist_type, self._dist_params)

        # Log the distribution change
        logger.debug(f"Distribution changed to: {dist_type}")

    def _update_stats_display(self) -> None:
        """
        Updates the statistics display in the UI.
        """
        # Check if statistics are available
        if not self._stats:
            self._stats_panel.hide()
            return

        # Format statistics for display
        mse = self._stats.get("mse", 0)
        r_squared = self._stats.get("r_squared", 0)
        max_deviation = self._stats.get("max_deviation", 0)

        self._mse_label.setText(f"MSE: {mse:.4f}")
        self._r_squared_label.setText(f"R-squared: {r_squared:.4f}")
        self._max_deviation_label.setText(f"Max Deviation: {max_deviation:.4f}")

        # Show or hide statistics panel based on availability
        self._stats_panel.show()

    @pyqtSlot()
    def _on_export_stats(self) -> None:
        """
        Slot for handling export statistics button clicks.
        """
        # Show file dialog to get save location
        from PyQt6.QtWidgets import QFileDialog
        filepath, _ = QFileDialog.getSaveFileName(self, "Export Statistics", "", "CSV (*.csv);;JSON (*.json)")

        if not filepath:
            return

        # Export statistics to CSV or JSON based on selection
        import json
        try:
            if filepath.endswith(".csv"):
                import csv
                with open(filepath, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(["Statistic", "Value"])
                    for key, value in self._stats.items():
                        writer.writerow([key, value])
            elif filepath.endswith(".json"):
                with open(filepath, 'w') as jsonfile:
                    json.dump(self._stats, jsonfile, indent=4)
            else:
                raise ValueError("Unsupported file format. Please use .csv or .json.")

            # Show feedback on export result
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.information(self, "Export Successful", f"Statistics exported to {filepath}")

        except Exception as e:
            logger.error(f"Error exporting statistics: {str(e)}")
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Export Failed", f"Failed to export statistics: {str(e)}")