import asyncio  # standard library
import logging  # standard library
import os  # standard library
from pathlib import Path  # standard library
from typing import Any, Dict, Optional  # standard library

import numpy as np  # version 1.26.3
import pandas as pd  # version 2.1.4
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot  # version 6.6.1
from PyQt6.QtWidgets import (  # version 6.6.1
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..plot_widgets import MultiPlotWidget  # Base widget for organizing multiple plots in a grid layout
from ..plots.acf_plot import ACFPlot  # Provides autocorrelation function plot for time series diagnostics
from ..plots.pacf_plot import PACFPlot  # Provides partial autocorrelation function plot for time series diagnostics
from ..plots.qq_plot import QQPlot  # Provides quantile-quantile plot for residual distribution analysis
from ..plots.residual_plot import ResidualPlot  # Provides residual analysis plot for model diagnostics
from ..plots.time_series_plot import TimeSeriesPlot  # Provides time series visualization for data and fitted values
from .statistical_metrics import StatisticalMetrics  # Displays statistical test results and metrics for model diagnostics

# Initialize logger
logger = logging.getLogger(__name__)

# Default export prefix
DEFAULT_EXPORT_PREFIX = "diagnostic_"

# Tab names mapping
TAB_NAMES = {
    "time_series": "Time Series",
    "residual": "Residuals",
    "acf": "ACF",
    "pacf": "PACF",
    "qq": "Q-Q Plot",
    "statistics": "Statistics",
}


def combine_statistics(
    residual_stats: dict, qq_stats: dict, model_stats: dict
) -> dict:
    """Combines statistics from multiple diagnostic components into a unified dictionary"""
    # Create a new dictionary for combined statistics
    combined_stats = {}

    # Add residual statistics with 'residual_' prefix
    for key, value in residual_stats.items():
        combined_stats[f"residual_{key}"] = value

    # Add QQ plot statistics with 'qq_' prefix
    for key, value in qq_stats.items():
        combined_stats[f"qq_{key}"] = value

    # Add model statistics directly
    combined_stats.update(model_stats)

    # Return the combined dictionary
    return combined_stats


def ensure_export_directory(directory: str) -> bool:
    """Ensures that the export directory exists, creating it if necessary"""
    # Check if directory already exists
    if Path(directory).exists():
        return True

    # If not, create directory
    try:
        Path(directory).mkdir(parents=True, exist_ok=True)
        return True
    except OSError as e:
        logger.error(f"Failed to create directory {directory}: {e}")
        return False


class DiagnosticPanel(QWidget):
    """
    A comprehensive panel for displaying diagnostic plots and statistical information for time series and econometric models.
    Organizes multiple visualizations in a tabbed interface for effective model validation.
    """

    update_started = pyqtSignal()
    update_completed = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Initializes the diagnostic panel with tabs for different diagnostic plots and metrics"""
        super().__init__(parent)

        # Create update_started and update_completed signals for progress tracking
        self.update_started = pyqtSignal()
        self.update_completed = pyqtSignal()

        # Initialize main layout as QVBoxLayout
        self._main_layout = QVBoxLayout(self)

        # Create tab widget (QTabWidget) for organizing different diagnostic views
        self._tab_widget = QTabWidget()

        # Create each diagnostic component (TimeSeriesPlot, ResidualPlot, ACFPlot, PACFPlot, QQPlot, StatisticalMetrics)
        self._time_series_plot = TimeSeriesPlot()
        self._residual_plot = ResidualPlot()
        self._acf_plot = ACFPlot()
        self._pacf_plot = PACFPlot()
        self._qq_plot = QQPlot()
        self._statistical_metrics = StatisticalMetrics()

        # Add each component to its own tab in the tab widget
        self._tab_widget.addTab(self._time_series_plot, TAB_NAMES["time_series"])
        self._tab_widget.addTab(self._residual_plot, TAB_NAMES["residual"])
        self._tab_widget.addTab(self._acf_plot, TAB_NAMES["acf"])
        self._tab_widget.addTab(self._pacf_plot, TAB_NAMES["pacf"])
        self._tab_widget.addTab(self._qq_plot, TAB_NAMES["qq"])
        self._tab_widget.addTab(self._statistical_metrics, TAB_NAMES["statistics"])

        # Store references to all plot widgets in _plot_widgets dictionary for easy access
        self._plot_widgets: Dict[str, QWidget] = {
            "time_series": self._time_series_plot,
            "residual": self._residual_plot,
            "acf": self._acf_plot,
            "pacf": self._pacf_plot,
            "qq": self._qq_plot,
        }

        # Initialize _export_options dictionary with default export settings
        self._export_options: Dict[str, Any] = {}

        # Add the tab widget to the main layout
        self._main_layout.addWidget(self._tab_widget)

        # Set up control buttons for exporting plots and navigation
        self._setup_controls()

        # Connect signals and slots for tab changes and updates
        self._tab_widget.currentChanged.connect(self._on_tab_changed)

        # Set the layout for the widget
        self.setLayout(self._main_layout)

    def _setup_controls(self) -> None:
        """Sets up the control buttons for exporting plots and navigation"""
        # Create control layout
        control_layout = QHBoxLayout()

        # Create export button
        self._export_button = QPushButton("Export Plots")
        self._export_button.clicked.connect(self._on_export_clicked)
        control_layout.addWidget(self._export_button)

        # Add control layout to main layout
        self._main_layout.addLayout(control_layout)

    def update_plots(
        self,
        data: np.ndarray,
        residuals: np.ndarray,
        fitted_values: np.ndarray,
        model_results: dict,
    ) -> None:
        """Updates all diagnostic plots with new data and model results"""
        # Emit update_started signal
        self.update_started.emit()

        # Update time series plot with original data and fitted values
        self._time_series_plot.set_data(data)

        # Update residual plot with residuals and fitted values
        self._residual_plot.set_residuals(residuals)
        self._residual_plot.set_fitted_values(fitted_values)

        # Update ACF plot with residuals data
        self._acf_plot.set_data(residuals)

        # Update PACF plot with residuals data
        self._pacf_plot.set_data(residuals)

        # Update QQ plot with residuals for distribution analysis
        self._qq_plot.set_data(residuals)

        # Get residual statistics from residual plot
        residual_stats = self._residual_plot.get_residual_stats()

        # Get QQ plot statistics from QQ plot
        qq_stats = self._qq_plot.get_statistics()

        # Combine statistics with model_results using combine_statistics function
        combined_statistics = combine_statistics(residual_stats, qq_stats, model_results)

        # Update statistical metrics widget with combined statistics
        self._statistical_metrics.set_metrics_data(combined_statistics)

        # Emit update_completed signal
        self.update_completed.emit()

    async def async_update_plots(
        self,
        data: np.ndarray,
        residuals: np.ndarray,
        fitted_values: np.ndarray,
        model_results: dict,
    ) -> None:
        """Asynchronously updates all diagnostic plots to prevent UI blocking"""
        # Emit update_started signal
        self.update_started.emit()

        # Set time series data in time series plot
        self._time_series_plot.set_data(data)

        # Set residuals and fitted values in residual plot
        self._residual_plot.set_residuals(residuals)
        self._residual_plot.set_fitted_values(fitted_values)

        # Set residuals data in ACF plot
        self._acf_plot.set_data(residuals)

        # Set residuals data in PACF plot
        self._pacf_plot.set_data(residuals)

        # Set residuals data in QQ plot
        self._qq_plot.set_data(residuals)

        # Await each plot's async_update_plot method to prevent UI blocking
        await self._time_series_plot.async_update_plot()
        await self._residual_plot.async_update_plot()
        await self._acf_plot.async_update_plot()
        await self._pacf_plot.async_update_plot()
        await self._qq_plot.async_update_plot()

        # Get statistics from all components
        residual_stats = self._residual_plot.get_residual_stats()
        qq_stats = self._qq_plot.get_statistics()

        # Combine statistics with model results
        combined_statistics = combine_statistics(residual_stats, qq_stats, model_results)

        # Update statistical metrics widget with combined statistics
        self._statistical_metrics.set_metrics_data(combined_statistics)

        # Yield control with await asyncio.sleep(0) between updates
        await asyncio.sleep(0)

        # Emit update_completed signal
        self.update_completed.emit()

    def clear_plots(self) -> None:
        """Clears all plots and metrics in the diagnostic panel"""
        # Call clear method on time series plot
        self._time_series_plot.clear()

        # Call clear method on residual plot
        self._residual_plot.clear()

        # Call clear method on ACF plot
        self._acf_plot.clear()

        # Call clear method on PACF plot
        self._pacf_plot.clear()

        # Call clear method on QQ plot
        self._qq_plot.clear()

        # Call clear method on statistical metrics widget
        self._statistical_metrics.clear()

        # Log the clear operation
        logger.debug("Diagnostic panel plots cleared")

    def get_current_tab(self) -> Optional[str]:
        """Returns the currently active diagnostic tab"""
        # Get current tab index from _tab_widget
        current_index = self._tab_widget.currentIndex()

        # Return the tab name based on the index using the tab widget's tabText method
        if 0 <= current_index < self._tab_widget.count():
            return self._tab_widget.tabText(current_index)
        else:
            return None

    def set_current_tab(self, tab_name: str) -> bool:
        """Sets the currently active diagnostic tab"""
        # Convert tab_name to corresponding tab index
        tab_index = -1
        for i in range(self._tab_widget.count()):
            if self._tab_widget.tabText(i) == tab_name:
                tab_index = i
                break

        # If tab name is valid, set the current index of the tab widget
        if tab_index != -1:
            self._tab_widget.setCurrentIndex(tab_index)
            logger.info(f"Changed tab to {tab_name}")
            return True
        else:
            logger.warning(f"Tab name not found: {tab_name}")
            return False

    def export_plots(self, directory: str, prefix: str = "", options: dict = {}) -> dict:
        """Exports diagnostic plots to image files"""
        # Ensure export directory exists using ensure_export_directory function
        if not ensure_export_directory(directory):
            return {}

        # Create dictionary to track exported file paths
        exported_files = {}

        # Set default prefix to DEFAULT_EXPORT_PREFIX if not provided
        if not prefix:
            prefix = DEFAULT_EXPORT_PREFIX

        # Update _export_options with provided options if any
        self.set_export_options(options)

        # Create file paths for each plot type with naming pattern {prefix}_{plot_type}.png
        time_series_path = os.path.join(directory, f"{prefix}time_series.png")
        residual_path = os.path.join(directory, f"{prefix}residual.png")
        acf_path = os.path.join(directory, f"{prefix}acf.png")
        pacf_path = os.path.join(directory, f"{prefix}pacf.png")
        qq_path = os.path.join(directory, f"{prefix}qq.png")

        # Export time series plot to {directory}/{prefix}_time_series.png
        exported_files["time_series"] = self._time_series_plot.save_figure(time_series_path)

        # Export residual plot to {directory}/{prefix}_residual.png
        exported_files["residual"] = self._residual_plot.save_figure(residual_path)

        # Export ACF plot to {directory}/{prefix}_acf.png
        exported_files["acf"] = self._acf_plot.save_figure(acf_path)

        # Export PACF plot to {directory}/{prefix}_pacf.png
        exported_files["pacf"] = self._pacf_plot.save_figure(pacf_path)

        # Export QQ plot to {directory}/{prefix}_qq.png
        exported_files["qq"] = self._qq_plot.save_figure(qq_path)

        # Log each export operation
        for plot_type, file_path in exported_files.items():
            if file_path:
                logger.info(f"Exported {plot_type} plot to {file_path}")
            else:
                logger.warning(f"Failed to export {plot_type} plot")

        # Return dictionary mapping plot types to file paths
        return exported_files

    def export_statistics(self, filepath: str, format: str = "csv") -> bool:
        """Exports the statistical metrics to a file"""
        # Get statistics data from _statistical_metrics.get_metrics_data()
        statistics_data = self._statistical_metrics.get_metrics_data()

        # Ensure parent directory exists using ensure_export_directory
        if not ensure_export_directory(os.path.dirname(filepath)):
            return False

        try:
            # If format is 'csv', export statistics to CSV format
            if format == "csv":
                import csv

                with open(filepath, "w", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(["Metric", "Value"])
                    for key, value in statistics_data.items():
                        writer.writerow([key, value])

            # If format is 'json', export statistics to JSON format
            elif format == "json":
                import json

                with open(filepath, "w") as jsonfile:
                    json.dump(statistics_data, jsonfile, indent=4)

            # If format is 'txt', export statistics to text format
            elif format == "txt":
                with open(filepath, "w") as txtfile:
                    for key, value in statistics_data.items():
                        txtfile.write(f"{key}: {value}\n")

            else:
                raise ValueError("Unsupported format. Use 'csv', 'json', or 'txt'.")

            logger.info(f"Exported statistics to {filepath} in {format} format")
            return True

        except Exception as e:
            logger.error(f"Failed to export statistics: {str(e)}")
            return False

    def set_export_options(self, options: dict) -> None:
        """Sets options for plot and statistics exports"""
        # Validate options is a dictionary
        if not isinstance(options, dict):
            raise TypeError("Export options must be a dictionary")

        # Update _export_options with new values
        self._export_options.update(options)

        # Log the options update
        logger.debug(f"Updated export options: {options.keys()}")

    def get_combined_statistics(self) -> dict:
        """Gets combined statistics from all diagnostic components"""
        # Get residual statistics from _residual_plot.get_residual_stats()
        residual_stats = self._residual_plot.get_residual_stats()

        # Get QQ plot statistics from _qq_plot.get_statistics()
        qq_stats = self._qq_plot.get_statistics()

        # Get model statistics from _statistical_metrics.get_metrics_data()
        model_stats = self._statistical_metrics.get_metrics_data()

        # Combine statistics using combine_statistics function
        combined_statistics = combine_statistics(residual_stats, qq_stats, model_stats)

        # Return the combined dictionary
        return combined_statistics

    @pyqtSlot(int)
    def _on_tab_changed(self, index: int) -> None:
        """Slot for handling tab change events"""
        # Get tab name from index
        tab_name = self._tab_widget.tabText(index)

        # Log tab change event
        logger.info(f"Tab changed to: {tab_name}")

        # Perform any tab-specific updates if needed
        # (Currently no tab-specific updates are required)

    @pyqtSlot()
    def _on_export_clicked(self) -> None:
        """Slot for handling export button clicks"""
        # Show directory selection dialog
        from PyQt6.QtWidgets import QFileDialog, QMessageBox

        directory = QFileDialog.getExistingDirectory(self, "Select Export Directory")

        # If directory selected, call export_plots with the selected directory
        if directory:
            exported_files = self.export_plots(directory)

            # Show feedback on export result
            if exported_files:
                QMessageBox.information(
                    self, "Export Successful", f"Plots exported to {directory}"
                )
            else:
                QMessageBox.warning(self, "Export Failed", "Failed to export plots")

    def sizeHint(self) -> Qt.QSize:
        """Returns the recommended size for the widget"""
        # Return QSize with appropriate width and height for the diagnostic panel
        return Qt.QSize(800, 600)