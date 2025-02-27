#!/usr/bin/env python3
# -*- coding: utf-8 -*

"""
PyQt6-based view component for ARMA (AutoRegressive Moving Average) models,
providing an interactive interface for model configuration, estimation, and
result visualization.
"""

# Import necessary modules
import asyncio  # Asynchronous I/O, event loop, and coroutines
from dataclasses import dataclass  # Data class decorator for parameter storage
import logging  # Logging library for debugging and monitoring
from typing import (  # Type hints for better code documentation and static analysis
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np  # Numerical operations and array handling # numpy 1.26.3
import pandas as pd  # Data manipulation and time series handling # pandas 2.1.4
from PyQt6.QtCore import Qt, QSize, pyqtSignal  # Core PyQt6 functionality # PyQt6.QtCore 6.6.1
from PyQt6.QtWidgets import (  # Core PyQt6 widgets for building the user interface # PyQt6.QtWidgets 6.6.1
    QCheckBox,
    QGroupBox,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
    QHBoxLayout,
)

# Internal imports
from src.backend.mfe.models.arma import ARMAModel  # Access to the ARMA model implementation for time series analysis
from src.backend.mfe.utils.validation import validate_arma_parameters  # Validates ARMA model parameters
from src.web.mfe.ui.async.task_manager import TaskManager  # Manages asynchronous tasks for the UI
from src.web.mfe.ui.async.worker import AsyncWorker  # Handles asynchronous operations for model estimation
from src.web.mfe.ui.components.diagnostic_panel import DiagnosticPanel  # Displays diagnostic plots and visualizations
from src.web.mfe.ui.components.model_equation import ModelEquation  # Displays mathematical representation of the ARMA model
from src.web.mfe.ui.components.parameter_table import ParameterTable  # Displays model parameters in a table format
from src.web.mfe.ui.components.statistical_metrics import StatisticalMetrics  # Displays statistical metrics for the model
from src.web.mfe.ui.plots.acf_plot import ACFPlot  # Autocorrelation function plot for model diagnostics
from src.web.mfe.ui.plots.pacf_plot import PACFPlot  # Partial autocorrelation function plot for model diagnostics
from src.web.mfe.ui.plots.residual_plot import ResidualPlot  # Residual plot for model diagnostics
from src.web.mfe.ui.plots.time_series_plot import TimePlot  # Time series visualization

# Configure logger
logger = logging.getLogger(__name__)

# Constants for ARMA model
ARMA_MAX_ORDER = 30
ARMA_DEFAULT_AR_ORDER = 1
ARMA_DEFAULT_MA_ORDER = 1
ARMA_DEFAULT_INCLUDE_CONSTANT = True


@dataclass
class ARMAViewSettings:
    """
    Dataclass for storing ARMA view configuration settings
    """

    ar_order: int = ARMA_DEFAULT_AR_ORDER
    ma_order: int = ARMA_DEFAULT_MA_ORDER
    include_constant: bool = ARMA_DEFAULT_INCLUDE_CONSTANT
    data: Optional[pd.DataFrame] = None
    target_series: Optional[pd.Series] = None
    estimation_options: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """
        Initializes ARMA view settings with default values
        """
        self.ar_order = int(self.ar_order)
        self.ma_order = int(self.ma_order)
        self.include_constant = bool(self.include_constant)
        self.data = self.data if self.data is not None else None
        self.target_series = self.target_series if self.target_series is not None else None
        self.estimation_options = self.estimation_options if self.estimation_options is not None else {}


class ARMAView(QWidget):
    """
    PyQt6 widget providing an interactive view for ARMA model configuration,
    estimation, and visualization
    """

    def __init__(self, parent: Optional[QWidget] = None):
        """
        Initializes the ARMA view widget with UI components
        """
        super().__init__(parent)

        # Initialize settings
        self._settings = ARMAViewSettings()
        self._current_model: Optional[ARMAModel] = None
        self._estimation_task: Optional[asyncio.Task] = None
        self._task_manager = TaskManager()

        # Setup UI
        self._setup_ui()
        self._connect_signals()
        self._update_equation_display()

    def _setup_ui(self):
        """
        Sets up the UI components and layout
        """
        # Main layout
        self._main_layout = QVBoxLayout(self)

        # Configuration group
        self._config_group = QGroupBox("Model Configuration")
        config_layout = QVBoxLayout()

        # AR order
        ar_layout = QHBoxLayout()
        ar_label = QLabel("AR Order:")
        self._ar_order_spinbox = QSpinBox()
        self._ar_order_spinbox.setRange(0, ARMA_MAX_ORDER)
        self._ar_order_spinbox.setValue(self._settings.ar_order)
        ar_layout.addWidget(ar_label)
        ar_layout.addWidget(self._ar_order_spinbox)
        config_layout.addLayout(ar_layout)

        # MA order
        ma_layout = QHBoxLayout()
        ma_label = QLabel("MA Order:")
        self._ma_order_spinbox = QSpinBox()
        self._ma_order_spinbox.setRange(0, ARMA_MAX_ORDER)
        self._ma_order_spinbox.setValue(self._settings.ma_order)
        ma_layout.addWidget(ma_label)
        ma_layout.addWidget(self._ma_order_spinbox)
        config_layout.addLayout(ma_layout)

        # Include constant
        self._include_constant_checkbox = QCheckBox("Include Constant")
        self._include_constant_checkbox.setChecked(self._settings.include_constant)
        config_layout.addWidget(self._include_constant_checkbox)

        self._config_group.setLayout(config_layout)
        self._main_layout.addWidget(self._config_group)

        # Buttons
        button_layout = QHBoxLayout()
        self._estimate_button = QPushButton("Estimate Model")
        self._reset_button = QPushButton("Reset")
        self._export_button = QPushButton("Export Results")
        self._export_button.setEnabled(False)
        button_layout.addWidget(self._estimate_button)
        button_layout.addWidget(self._reset_button)
        button_layout.addWidget(self._export_button)
        self._main_layout.addLayout(button_layout)

        # Equation display
        self._model_equation = ModelEquation()
        self._main_layout.addWidget(self._model_equation)

        # Parameter table
        self._parameter_table = ParameterTable()
        self._main_layout.addWidget(self._parameter_table)

        # Diagnostic panel
        self._diagnostic_panel = DiagnosticPanel()
        self._main_layout.addWidget(self._diagnostic_panel)

        # Set layout
        self.setLayout(self._main_layout)

    def _connect_signals(self):
        """
        Connects UI signals to their respective slots
        """
        self._ar_order_spinbox.valueChanged.connect(self._on_ar_order_changed)
        self._ma_order_spinbox.valueChanged.connect(self._on_ma_order_changed)
        self._include_constant_checkbox.toggled.connect(self._on_include_constant_toggled)
        self._estimate_button.clicked.connect(self._on_estimate_clicked)
        self._reset_button.clicked.connect(self._on_reset_clicked)
        self._export_button.clicked.connect(self._on_export_clicked)

    def _on_ar_order_changed(self, value: int):
        """
        Handles changes to the AR order spinner value
        """
        self._settings.ar_order = value
        self._update_equation_display()
        self._estimate_button.setEnabled(True)

    def _on_ma_order_changed(self, value: int):
        """
        Handles changes to the MA order spinner value
        """
        self._settings.ma_order = value
        self._update_equation_display()
        self._estimate_button.setEnabled(True)

    def _on_include_constant_toggled(self, checked: bool):
        """
        Handles toggling of the include constant checkbox
        """
        self._settings.include_constant = checked
        self._update_equation_display()
        self._estimate_button.setEnabled(True)

    def _update_equation_display(self):
        """
        Updates the model equation display based on current settings
        """
        # Format equation
        latex_equation = _format_equation(
            self._settings.ar_order,
            self._settings.ma_order,
            self._settings.include_constant,
            parameters=None,
        )

        # Set equation
        self._model_equation.set_custom_equation(latex_equation)

    def _on_estimate_clicked(self):
        """
        Handles the estimate button click event, starting model estimation
        """
        if self._settings.data is None:
            QMessageBox.critical(self, "Error", "No data loaded. Please load data first.")
            return

        # Create ARMA model
        try:
            arma_model = _create_model_from_ui_settings(self._settings)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to create model: {e}")
            return

        # Disable estimate button, enable reset button
        self._estimate_button.setEnabled(False)
        self._reset_button.setEnabled(True)
        self._export_button.setEnabled(False)

        # Clear current results
        self._parameter_table.clear()
        self._diagnostic_panel.clear_plots()

        # Create task for model estimation
        async def estimate_task():
            try:
                # Estimate model
                self._current_model = await arma_model.estimate_async(self._settings.target_series.values)
                return self._current_model.summary().to_dict()
            except Exception as e:
                return str(e)

        # Submit task to task manager
        task_id = self._task_manager.create_task(
            name="Estimate ARMA Model",
            function=estimate_task,
            priority=Qt.Priority.HighPriority,
            task_type="model",
        )

        # Connect worker signals to update UI with results
        self._task_manager.signals.task_completed.connect(self._on_estimation_complete)
        self._task_manager.signals.task_failed.connect(self._on_estimation_error)
        self._task_manager.signals.task_progress.connect(self._on_estimation_progress)

    def _on_reset_clicked(self):
        """
        Handles the reset button click event, clearing current results
        """
        # Cancel any ongoing estimation task
        if self._estimation_task and not self._estimation_task.done():
            self._estimation_task.cancel()

        # Clear parameter table and diagnostic panel
        self._parameter_table.clear()
        self._diagnostic_panel.clear_plots()

        # Reset model equation to show only structure
        self._update_equation_display()

        # Reset current model
        self._current_model = None

        # Enable estimate button
        self._estimate_button.setEnabled(True)
        self._export_button.setEnabled(False)

    def _on_export_clicked(self):
        """
        Handles the export button click event, exporting results
        """
        if self._current_model is None:
            QMessageBox.warning(self, "Export Error", "No model estimated. Please estimate model first.")
            return

        # Open file dialog for save location
        from PyQt6.QtWidgets import QFileDialog
        filename, _ = QFileDialog.getSaveFileName(self, "Export Results", "", "CSV Files (*.csv);;All Files (*)")

        if not filename:
            return

        # Export model summary, parameters, and plots to selected location
        try:
            # Export model summary and parameters to CSV
            self._current_model.summary().to_csv(filename, index=False)

            # Export plots to directory with same name as CSV file
            plot_dir = filename.replace(".csv", "_plots")
            self._diagnostic_panel.export_plots(plot_dir)

            QMessageBox.information(self, "Export Successful", f"Results exported to {filename} and {plot_dir}")

        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export data: {e}")

    def _on_estimation_complete(self, results):
        """
        Handles the completion of model estimation, updating the UI with results
        """
        # Update model equation with estimated parameters
        latex_equation = _format_equation(
            self._settings.ar_order,
            self._settings.ma_order,
            self._settings.include_constant,
            parameters=results,
        )
        self._model_equation.set_custom_equation(latex_equation)

        # Update parameter table with parameter estimates
        self._parameter_table.set_parameter_data(results)

        # Update diagnostic panel with residual plots, ACF, and PACF
        self._diagnostic_panel.update_plots(
            data=self._settings.target_series.values,
            residuals=self._current_model.residuals,
            fitted_values=self._current_model.fittedvalues,
            model_results=results,
        )

        # Enable export button
        self._export_button.setEnabled(True)

    def _on_estimation_error(self, error_message):
        """
        Handles errors during model estimation
        """
        QMessageBox.critical(self, "Estimation Error", f"Failed to estimate model: {error_message}")
        self._estimate_button.setEnabled(True)
        self._reset_button.setEnabled(False)
        self._export_button.setEnabled(False)

    def _on_estimation_progress(self, progress, message):
        """
        Updates the UI with estimation progress information
        """
        # TODO: Implement progress indicator
        pass

    def set_data(self, data: pd.DataFrame, target_column: Optional[str] = None):
        """
        Sets the data to be used for model estimation
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Data must be a pandas DataFrame")

        self._settings.data = data

        if target_column:
            if target_column not in data.columns:
                raise ValueError(f"Target column '{target_column}' not found in data")
            self._settings.target_series = data[target_column]
        elif len(data.columns) == 1:
            self._settings.target_series = data.iloc[:, 0]
        else:
            raise ValueError("Missing target column specification for multi-column data")

        self._estimate_button.setEnabled(True)

    def get_current_model(self) -> Optional[ARMAModel]:
        """
        Returns the currently estimated ARMA model if available
        """
        return self._current_model

    def get_settings(self) -> ARMAViewSettings:
        """
        Returns the current view settings
        """
        return self._settings

    def apply_settings(self, settings: ARMAViewSettings):
        """
        Applies the provided settings to the view
        """
        self._settings = settings
        self._ar_order_spinbox.setValue(settings.ar_order)
        self._ma_order_spinbox.setValue(settings.ma_order)
        self._include_constant_checkbox.setChecked(settings.include_constant)
        self._update_equation_display()

def _create_model_from_ui_settings(ui_settings: ARMAViewSettings) -> ARMAModel:
    """
    Creates an ARMA model instance from the current UI settings
    """
    # Extract AR order, MA order, and constant flag from ui_settings
    ar_order = ui_settings.ar_order
    ma_order = ui_settings.ma_order
    include_constant = ui_settings.include_constant

    # Validate parameters using validate_arma_parameters function
    try:
        validate_arma_parameters(ar_order, ma_order, include_constant)
    except ValueError as e:
        raise ValueError(f"Invalid ARMA parameters: {e}")

    # Create and return a new ARMAModel instance with the validated parameters
    return ARMAModel(p=ar_order, q=ma_order, include_constant=include_constant)

def _format_equation(ar_order: int, ma_order: int, include_constant: bool, parameters: Optional[Dict[str, float]] = None) -> str:
    """
    Formats the ARMA model equation for LaTeX display
    """
    # Build the AR component of the equation
    ar_component = ""
    if ar_order > 0:
        ar_terms = []
        for i in range(1, ar_order + 1):
            if parameters and f"phi_{i}" in parameters:
                phi_value = parameters[f"phi_{i}"]
                phi_sign = "-" if phi_value < 0 else ""
                ar_terms.append(f"{phi_sign} {abs(phi_value):.3f} y_{{t-{i}}}")
            else:
                ar_terms.append(f"\\phi_{{{i}}} y_{{t-{i}}}")
        ar_component = " - ".join(ar_terms)

    # Build the MA component of the equation
    ma_component = ""
    if ma_order > 0:
        ma_terms = []
        for i in range(1, ma_order + 1):
            if parameters and f"theta_{i}" in parameters:
                theta_value = parameters[f"theta_{i}"]
                theta_sign = "+" if theta_value > 0 else ""
                ma_terms.append(f"{theta_sign} {abs(theta_value):.3f} \\varepsilon_{{t-{i}}}")
            else:
                ma_terms.append(f"\\theta_{{{i}}} \\varepsilon_{{t-{i}}}")
        ma_component = " + ".join(ma_terms)

    # Add the constant term if include_constant is True
    constant_term = ""
    if include_constant:
        if parameters and "constant" in parameters:
            constant_value = parameters["constant"]
            constant_term = f"{constant_value:.4f}"
        else:
            constant_term = "\\mu"

    # If parameters are provided, substitute the parameter values in the equation
    equation_parts = []
    equation_parts.append("y_t")

    if constant_term:
        equation_parts.append(f"= {constant_term}")
    else:
        equation_parts.append("=")

    if ar_component:
        if equation_parts[-1] != "=":
            equation_parts.append(f"+ {ar_component}")
        else:
            equation_parts.append(ar_component)

    if ma_component:
        equation_parts.append(f"+ {ma_component}")

    equation_parts.append("+ \\varepsilon_t")

    equation = " ".join(equation_parts)

    # Return the formatted LaTeX equation string
    return f"${equation}$"

def _generate_model_tooltip() -> str:
    """
    Generates a tooltip explaining the ARMA model parameters
    """
    # Create a detailed explanation of AR and MA components
    tooltip_text = "<b>Autoregressive Moving Average (ARMA) Model</b><br><br>"
    tooltip_text += "The ARMA model is a combination of Autoregressive (AR) and Moving Average (MA) models.<br><br>"
    tooltip_text += "<b>AR Component (p):</b><br>"
    tooltip_text += "The AR component models the current value as a linear combination of its past values. "
    tooltip_text += "The order 'p' represents the number of past values used in the model.<br><br>"
    tooltip_text += "<b>MA Component (q):</b><br>"
    tooltip_text += "The MA component models the current value as a linear combination of past error terms. "
    tooltip_text += "The order 'q' represents the number of past error terms used in the model.<br><br>"

    # Add explanation of the constant term
    tooltip_text += "<b>Constant Term:</b><br>"
    tooltip_text += "The constant term (often denoted as μ) represents the mean value of the time series. "
    tooltip_text += "Including a constant term allows the model to account for a non-zero mean.<br><br>"

    # Add details about suitable scenarios for ARMA models
    tooltip_text += "<b>Suitable Scenarios:</b><br>"
    tooltip_text += "ARMA models are suitable for stationary time series data. "
    tooltip_text += "They are commonly used in financial econometrics to model and forecast asset returns, "
    tooltip_text += "inflation rates, and other economic indicators."

    # Return the concatenated tooltip text
    return tooltip_text