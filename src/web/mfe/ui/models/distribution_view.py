#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A PyQt6-based UI component for visualizing and analyzing statistical distributions in the MFE Toolbox.
This view provides interactive tools for fitting various distributions to data, comparing distribution properties,
and conducting statistical tests for distribution goodness-of-fit.
"""
import asyncio  # standard library
import logging  # standard library
from typing import Any, Dict, Optional  # standard library

import numpy as np  # version 1.26.3
import pandas as pd  # version 2.1.4
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot  # version 6.6.1
from PyQt6.QtGui import QFont, QColor  # version 6.6.1
from PyQt6.QtWidgets import (  # version 6.6.1
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

# Internal imports
from '../../../backend/mfe/core/distributions import (  # Backend functionality for distribution analysis
    DistributionTest,
    GeneralizedErrorDistribution,
    SkewedTDistribution,
    distribution_fit,
    jarque_bera,
    ks_test,
    shapiro_wilk,
)
from '../../ui/async/signals import ModelSignals, PlotSignals  # Signal containers for asynchronous operations in the distribution view
from '../../ui/async/worker import AsyncWorker, Worker  # Background processing for non-blocking UI during distribution calculations
from '../../ui/components/model_equation import ModelEquation  # LaTeX rendering of distribution equations
from '../../ui/components/parameter_table import ParameterTable  # Table display for distribution parameters
from '../../ui/plots/density_plot import DensityPlot  # Visualization of probability density functions
from '../../ui/plots/qq_plot import QQPlot  # Quantile-Quantile plot for distribution comparison
from '../../ui/widgets import create_input_group, create_separator  # Utility functions for creating UI components

# Initialize logger
logger = logging.getLogger(__name__)

# Constants for distribution types and plot colors
DISTRIBUTION_TYPES = {'normal': 'Normal', 't': 'Student\'s t', 'ged': 'Generalized Error', 'skewed_t': 'Skewed t'}
DEFAULT_PLOT_COLORS = {'empirical': '#1f77b4', 'theoretical': '#ff7f0e', 'reference': '#2ca02c'}


class DistributionView(QWidget):
    """
    Main view component for statistical distribution analysis and visualization in the MFE Toolbox
    """

    def __init__(self, parent: QWidget) -> None:
        """
        Initialize the main distribution view component
        """
        super().__init__(parent)

        # Create ModelSignals instance
        self.signals = ModelSignals()

        # Create main layout with QSplitter for resizable panels
        main_layout = QVBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Create DistributionConfigPanel for settings
        self.config_panel = DistributionConfigPanel(self)
        splitter.addWidget(self.config_panel)

        # Create DistributionPlotPanel for visualization
        self.plot_panel = DistributionPlotPanel(self)
        splitter.addWidget(self.plot_panel)

        # Create DistributionStatsPanel for statistics
        self.stats_panel = DistributionStatsPanel(self)
        splitter.addWidget(self.stats_panel)

        # Set splitter stretch factors
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        splitter.setStretchFactor(2, 1)

        # Add splitter to main layout
        main_layout.addWidget(splitter)

        # Create control buttons (fit, reset, export)
        control_layout = QHBoxLayout()
        self.fit_button = QPushButton("Fit Distribution")
        self.reset_button = QPushButton("Reset")
        self.export_button = QPushButton("Export Results")
        control_layout.addWidget(self.fit_button)
        control_layout.addWidget(self.reset_button)
        control_layout.addWidget(self.export_button)
        main_layout.addLayout(control_layout)

        # Initialize data as None
        self.data: Optional[np.ndarray] = None

        # Set fitting_in_progress to False
        self.fitting_in_progress = False

        # Connect signal handlers
        self.fit_button.clicked.connect(self.fit_distribution)
        self.reset_button.clicked.connect(self.reset)
        self.export_button.clicked.connect(self.export_results)
        self.config_panel.signals.distribution_updated.connect(self._on_distribution_changed)

        # Apply styling
        self.setLayout(main_layout)

    def set_data(self, data: np.ndarray) -> None:
        """
        Sets the data for distribution analysis
        """
        # Validate data as numeric array
        if not isinstance(data, np.ndarray):
            raise TypeError("Input data must be a NumPy array")

        # Store data in instance variable
        self.data = data

        # Update plot_panel with new data
        self.plot_panel.set_data(data)

        # Update stats_panel with new data
        self.stats_panel.set_data(data)

        # Enable fit_button if data is valid
        self.fit_button.setEnabled(True)

        # Emit data_changed signal
        self.signals.data_changed.emit()

        # Log data update
        logger.debug("Data set for distribution analysis")

    @pyqtSlot()
    def fit_distribution(self) -> None:
        """
        Fits a distribution to the current data based on config panel settings
        """
        # Check if data is available
        if self.data is None:
            QMessageBox.warning(self, "No Data", "Please load data before fitting a distribution.")
            return

        # Set fitting_in_progress to True
        self.fitting_in_progress = True

        # Disable UI during fitting
        self.fit_button.setEnabled(False)
        self.reset_button.setEnabled(False)
        self.export_button.setEnabled(False)

        # Get distribution type from config_panel
        dist_type = self.config_panel.get_distribution_type()

        # Create AsyncWorker for fitting process
        worker = AsyncWorker(self._fit_distribution_task)

        # Connect worker signals
        worker.signals.result.connect(self._on_fit_complete)
        worker.signals.error.connect(self._on_fit_error)
        worker.signals.finished.connect(lambda: print("Fit task finished"))

        # Start worker thread
        QThreadPool.globalInstance().start(worker)

        # Log fit operation start
        logger.info(f"Fitting distribution: {dist_type}")

    async def _fit_distribution_task(self) -> Dict[str, Any]:
        """
        Asynchronous task for distribution fitting
        """
        # Get current distribution type from config panel
        dist_type = self.config_panel.get_distribution_type()

        # Call _fit_distribution_async with data and dist_type
        results = await _fit_distribution_async(self.data, dist_type)

        # Return fit results dictionary
        return results

    @pyqtSlot(object)
    def _on_fit_complete(self, results: Dict[str, Any]) -> None:
        """
        Handler for fit completion
        """
        # Set fitting_in_progress to False
        self.fitting_in_progress = False

        # Extract parameters from results
        parameters = {k: v for k, v in results.items() if k not in ['distribution', 'loglikelihood', 'aic', 'bic', 'tests']}

        # Update config_panel with fitted parameters
        self.config_panel.set_parameter_values(parameters)

        # Update plot_panel with distribution and parameters
        dist_type = self.config_panel.get_distribution_type()
        self.plot_panel.set_distribution(dist_type, parameters)

        # Update stats_panel with test statistics
        self.stats_panel.update_statistics(results['tests'])

        # Re-enable UI elements
        self.fit_button.setEnabled(True)
        self.reset_button.setEnabled(True)
        self.export_button.setEnabled(True)

        # Emit fit_complete signal
        self.signals.estimation_finished.emit()

        # Log fit completion
        logger.info("Distribution fitting complete")

    @pyqtSlot(Exception)
    def _on_fit_error(self, error: Exception) -> None:
        """
        Handler for fit errors
        """
        # Set fitting_in_progress to False
        self.fitting_in_progress = False

        # Show error message dialog
        QMessageBox.critical(self, "Fitting Error", str(error))

        # Re-enable UI elements
        self.fit_button.setEnabled(True)
        self.reset_button.setEnabled(True)
        self.export_button.setEnabled(True)

        # Emit fit_error signal
        self.signals.estimation_error.emit(error)

        # Log error details
        logger.error(f"Distribution fitting error: {str(error)}")

    @pyqtSlot()
    def reset(self) -> None:
        """
        Resets the view to initial state
        """
        # Clear plot_panel
        self.plot_panel.clear()

        # Clear stats_panel
        self.stats_panel.clear()

        # Reset config_panel to default distribution
        self.config_panel.set_distribution_type("normal")

        # Set data to None
        self.data = None

        # Disable fit_button
        self.fit_button.setEnabled(False)

        # Emit reset signal
        self.signals.estimation_started.emit()

        # Log reset operation
        logger.info("Distribution view reset")

    @pyqtSlot(object)
    def _on_distribution_changed(self, distribution_info: Dict[str, Any]) -> None:
        """
        Handler for distribution type or parameter changes
        """
        # Extract distribution type and parameters
        dist_type = distribution_info['type']
        params = distribution_info['parameters']

        # Update plot_panel with new distribution
        self.plot_panel.set_distribution(dist_type, params)

        # Update stats_panel with new distribution
        self.stats_panel.set_distribution(dist_type, params)

        # Log distribution change
        logger.debug(f"Distribution changed: {dist_type} with parameters {params}")

    @pyqtSlot()
    def export_results(self) -> bool:
        """
        Exports the current analysis results
        """
        # Show dialog for export type (plot, stats, or both)
        # Based on selection, call appropriate export method
        # Show success message
        # Return success status
        return True

    def get_current_distribution(self) -> Dict[str, Any]:
        """
        Gets the current distribution settings
        """
        # Get distribution type from config_panel
        dist_type = self.config_panel.get_distribution_type()

        # Get parameter values from config_panel
        params = self.config_panel.get_parameter_values()

        # Return dictionary with type and parameters
        return {'type': dist_type, 'parameters': params}

    def get_fit_statistics(self) -> Dict[str, Any]:
        """
        Gets the current fit statistics
        """
        # Get statistics from plot_panel and stats_panel
        # Combine into comprehensive statistics dictionary
        # Return statistics dictionary
        return {}


class DistributionConfigPanel(QWidget):
    """
    Panel for configuring statistical distribution settings and parameters
    """

    def __init__(self, parent: QWidget) -> None:
        """
        Initialize the distribution configuration panel
        """
        super().__init__(parent)

        # Create ModelSignals instance for communication
        self.signals = ModelSignals()

        # Create main layout for the panel
        main_layout = QVBoxLayout(self)

        # Set up distribution type selector with available distributions
        self.dist_type_selector = QComboBox()
        for key, value in DISTRIBUTION_TYPES.items():
            self.dist_type_selector.addItem(value, key)
        main_layout.addWidget(create_input_group("Distribution Type", self.dist_type_selector))

        # Initialize empty parameter controls dictionary
        self.parameter_controls: Dict[str, QDoubleSpinBox] = {}

        # Create parameter group box with initial controls for normal distribution
        self.params_group = QGroupBox("Distribution Parameters")
        main_layout.addWidget(self.params_group)

        # Set current_dist_type to 'normal'
        self.current_dist_type = "normal"

        # Initialize current_params
        self.current_params: Dict[str, Any] = {}

        # Call internal method to update parameter controls
        self._update_parameter_controls()

        # Connect signal handlers for UI interactions
        self.dist_type_selector.currentIndexChanged.connect(self._on_distribution_type_changed)

        # Apply styling from styles module
        self.setLayout(main_layout)

    def set_distribution_type(self, dist_type: str) -> None:
        """
        Sets the active distribution type and updates UI
        """
        # Update combo box selection if different
        if self.dist_type_selector.currentData() != dist_type:
            index = self.dist_type_selector.findData(dist_type)
            if index >= 0:
                self.dist_type_selector.setCurrentIndex(index)

        # Update current_dist_type
        self.current_dist_type = dist_type

        # Recreate parameter controls for the new distribution type
        self._update_parameter_controls()

        # Emit distribution_type_changed signal
        self.signals.distribution_updated.emit({'type': self.current_dist_type, 'parameters': self.current_params})

        # Log the distribution type change
        logger.info(f"Distribution type set to: {dist_type}")

    def set_parameter_values(self, params: Dict[str, Any]) -> None:
        """
        Sets parameter values in the UI controls
        """
        # For each parameter in params
        for param_name, param_value in params.items():
            # Find corresponding control in parameter_controls
            if param_name in self.parameter_controls:
                control = self.parameter_controls[param_name]
                # Set the control value if found
                control.setValue(param_value)

        # Update current_params with new values
        self.current_params.update(params)

        # Emit parameters_updated signal
        self.signals.distribution_updated.emit({'type': self.current_dist_type, 'parameters': self.current_params})

        # Log parameter update
        logger.debug(f"Parameters updated: {params}")

    def get_parameter_values(self) -> Dict[str, Any]:
        """
        Gets the current parameter values from UI controls
        """
        # Call _get_parameter_values with parameter_controls
        parameter_values = self._get_parameter_values(self.parameter_controls)

        # Return parameter values dictionary
        return parameter_values

    def get_distribution_type(self) -> str:
        """
        Gets the currently selected distribution type
        """
        # Return current_dist_type
        return self.current_dist_type

    @pyqtSlot(str)
    def _on_distribution_type_changed(self, dist_type: str) -> None:
        """
        Handler for distribution type change events
        """
        # Call set_distribution_type with the new dist_type
        self.set_distribution_type(self.dist_type_selector.itemData(self.dist_type_selector.currentIndex()))

        # Log the distribution change event
        logger.debug(f"Distribution type changed to: {dist_type}")

    @pyqtSlot()
    def _on_parameter_value_changed(self) -> None:
        """
        Handler for parameter value change events
        """
        # Get current parameter values
        current_values = self.get_parameter_values()

        # Update current_params
        self.current_params = current_values

        # Emit parameters_updated signal
        self.signals.distribution_updated.emit({'type': self.current_dist_type, 'parameters': self.current_params})

        # Log parameter value change
        logger.debug(f"Parameter value changed, new values: {self.current_params}")

    def _update_parameter_controls(self) -> None:
        """
        Updates the parameter UI controls for the current distribution
        """
        # Clear existing controls from params_group
        for i in reversed(range(self.params_group.layout().count())):
            self.params_group.layout().itemAt(i).widget().setParent(None)

        # Create new parameter controls for current_dist_type
        self.parameter_controls = self._create_distribution_parameter_controls(self.current_dist_type)

        # Add controls to params_group layout
        params_layout = QVBoxLayout()
        for param_name, control in self.parameter_controls.items():
            params_layout.addWidget(create_input_group(param_name, control))
        self.params_group.setLayout(params_layout)

        # Connect value changed signals
        for control in self.parameter_controls.values():
            control.valueChanged.connect(self._on_parameter_value_changed)

        # Update current_params from new controls
        self.current_params = self.get_parameter_values()

        # Log parameter controls update
        logger.debug(f"Parameter controls updated for distribution: {self.current_dist_type}")

    def _create_distribution_parameter_controls(self, dist_type: str) -> Dict[str, QDoubleSpinBox]:
        """
        Creates the UI controls for configuring distribution parameters
        """
        # Create a dictionary to store parameter controls
        parameter_controls = {}

        # For 'normal' distribution, create mu and sigma spinboxes
        if dist_type == "normal":
            mu_spinbox = QDoubleSpinBox()
            sigma_spinbox = QDoubleSpinBox()
            parameter_controls["mu"] = mu_spinbox
            parameter_controls["sigma"] = sigma_spinbox

        # For 't' distribution, create mu, sigma, and df spinboxes
        elif dist_type == "t":
            mu_spinbox = QDoubleSpinBox()
            sigma_spinbox = QDoubleSpinBox()
            df_spinbox = QDoubleSpinBox()
            parameter_controls["mu"] = mu_spinbox
            parameter_controls["sigma"] = sigma_spinbox
            parameter_controls["df"] = df_spinbox

        # For 'ged' distribution, create mu, sigma, and nu spinboxes
        elif dist_type == "ged":
            mu_spinbox = QDoubleSpinBox()
            sigma_spinbox = QDoubleSpinBox()
            nu_spinbox = QDoubleSpinBox()
            parameter_controls["mu"] = mu_spinbox
            parameter_controls["sigma"] = sigma_spinbox
            parameter_controls["nu"] = nu_spinbox

        # For 'skewed_t' distribution, create mu, sigma, df, and lambda spinboxes
        elif dist_type == "skewed_t":
            mu_spinbox = QDoubleSpinBox()
            sigma_spinbox = QDoubleSpinBox()
            df_spinbox = QDoubleSpinBox()
            lambda_spinbox = QDoubleSpinBox()
            parameter_controls["mu"] = mu_spinbox
            parameter_controls["sigma"] = sigma_spinbox
            parameter_controls["df"] = df_spinbox
            parameter_controls["lambda"] = lambda_spinbox

        # Apply appropriate ranges and tooltips to each control
        for param_name, control in parameter_controls.items():
            if param_name == "mu":
                control.setRange(-1000, 1000)
                control.setToolTip("Location parameter")
                control.setValue(0.0)
            elif param_name == "sigma":
                control.setRange(0.001, 1000)
                control.setToolTip("Scale parameter (must be positive)")
                control.setValue(1.0)
            elif param_name == "df":
                control.setRange(2.1, 100)
                control.setToolTip("Degrees of freedom (must be > 2)")
                control.setValue(5.0)
            elif param_name == "nu":
                control.setRange(0.5, 10)
                control.setToolTip("Shape parameter (must be positive)")
                control.setValue(2.0)
            elif param_name == "lambda":
                control.setRange(-0.99, 0.99)
                control.setToolTip("Skewness parameter (must be between -1 and 1)")
                control.setValue(0.0)

        # Return dictionary of parameter controls
        return parameter_controls

    def _get_parameter_values(self, parameter_controls: Dict[str, QDoubleSpinBox]) -> Dict[str, float]:
        """
        Extracts parameter values from UI controls
        """
        # Create an empty dictionary for parameter values
        parameter_values = {}

        # For each control in parameter_controls
        for param_name, control in parameter_controls.items():
            # Extract the parameter name and current value
            parameter_value = control.value()

            # Add to parameter values dictionary
            parameter_values[param_name] = parameter_value

        # Return the dictionary of parameter values
        return parameter_values

    def _format_distribution_equation(self, dist_type: str, params: Dict[str, float]) -> str:
        """
        Creates a LaTeX equation string for the selected distribution
        """
        # For 'normal' distribution, create Normal PDF equation
        # For 't' distribution, create Student's t PDF equation
        # For 'ged' distribution, create GED PDF equation
        # For 'skewed_t' distribution, create skewed t PDF equation
        # Substitute parameter values into equation
        # Return formatted LaTeX string
        return ""


async def _fit_distribution_async(data: np.ndarray, dist_type: str) -> Dict[str, Any]:
    """
    Asynchronously fits a distribution to provided data
    """
    # Validate input data and distribution type
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a NumPy array")
    if dist_type not in DISTRIBUTION_TYPES:
        raise ValueError(f"Invalid distribution type: {dist_type}")

    # Yield control to event loop with await asyncio.sleep(0)
    await asyncio.sleep(0)

    # Call distribution_fit from backend with appropriate parameters
    fit_results = distribution_fit(data, dist_type)

    # Calculate goodness-of-fit statistics
    # (This part is already included in distribution_fit)

    # Return dictionary with parameters and fit statistics
    return fit_results