#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implements the primary UI widgets for the MFE Toolbox, focusing on ARMAX model estimation and visualization.
Provides comprehensive controls, parameter input fields, and visualization components using PyQt6.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np  # version 1.26.3
from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QMessageBox,
    QDialog,
    QDialogButtonBox,
    QToolTip,
    QApplication
)

from .async.task_manager import TaskManager  # type: ignore
from .components.model_equation import ModelEquation  # type: ignore
from .components.parameter_table import ParameterTable  # type: ignore
from .components.progress_indicator import ProgressIndicator  # type: ignore
from .components.diagnostic_panel import DiagnosticPanel  # type: ignore
from .components.statistical_metrics import StatisticalMetrics  # type: ignore
from .plots.residual_plot import ResidualPlot  # type: ignore
from .plots.acf_plot import ACFPlot  # type: ignore
from .plots.pacf_plot import PACFPlot  # type: ignore
from '../../../backend/mfe/models/armax import ARMAX  # type: ignore

# Configure logger for this module
logger = logging.getLogger(__name__)

# Global constants for UI dimensions and spacing
DEFAULT_WINDOW_WIDTH = 800
DEFAULT_WINDOW_HEIGHT = 600
PARAMETER_SPACING = 10
CONTROL_BUTTON_SIZE = (120, 30)


def create_parameter_input(label: str, default_value: int, validator: Callable) -> Tuple[QLabel, QLineEdit, Any]:
    """
    Creates a labeled parameter input field with validation.

    Args:
        label: The label text for the input field.
        default_value: The default value for the input field.
        validator: A validator function to apply to the input.

    Returns:
        A tuple containing (QLabel, QLineEdit, QValidator).
    """
    # Create a QLabel with the provided label text
    label_widget = QLabel(label)

    # Create a QLineEdit with the default value converted to string
    input_widget = QLineEdit(str(default_value))

    # Apply the validator function to create a QValidator
    input_validator = validator()

    # Set the validator on the QLineEdit
    input_widget.setValidator(input_validator)

    # Return a tuple of the created widgets
    return label_widget, input_widget, input_validator


def create_button_row(button_configs: List[Dict[str, Any]]) -> QWidget:
    """
    Creates a row of buttons with consistent styling.

    Args:
        button_configs: A list of dictionaries, where each dictionary contains the 'text' and 'handler' for a button.

    Returns:
        A container widget with the buttons arranged horizontally.
    """
    # Create a container QWidget
    container = QWidget()

    # Create a horizontal layout for the container
    layout = QHBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)

    # For each button configuration, create a QPushButton with the specified text
    for config in button_configs:
        button = QPushButton(config['text'])

        # Connect each button's clicked signal to its corresponding handler function
        button.clicked.connect(config['handler'])

        # Apply standard styling to each button
        button.setFixedSize(QSize(*CONTROL_BUTTON_SIZE))
        button.setStyleSheet("""
            QPushButton {
                background-color: #e0e0e0;
                border: 1px solid #c0c0c0;
                border-radius: 4px;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #d0d0d0;
            }
        """)

        # Add each button to the horizontal layout
        layout.addWidget(button)

    # Add spacing between buttons
    layout.addSpacing(PARAMETER_SPACING)

    # Set the layout on the container widget
    container.setLayout(layout)

    # Return the container widget
    return container


def center_window(window: QWidget) -> None:
    """
    Centers a window on the screen.

    Args:
        window: The QWidget window to center.
    """
    # Calculate the geometry of the screen
    screen_geometry = QApplication.desktop().screenGeometry()

    # Calculate the center point of the screen
    center_x = (screen_geometry.width() - window.width()) // 2
    center_y = (screen_geometry.height() - window.height()) // 2

    # Move the window to center it on the screen
    window.move(center_x, center_y)


class ARMAXWidget(QWidget):
    """
    Main widget for ARMAX model estimation UI.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        """
        Initializes the ARMAXWidget with UI components.
        """
        super().__init__(parent)

        # Initialize the task manager for asynchronous operations
        self.task_manager = TaskManager()

        # Create the main layout as QVBoxLayout
        self.layout = QVBoxLayout(self)

        # Create and add the parameter input section
        self.parameter_widget = self.setup_parameter_section()
        self.layout.addWidget(self.parameter_widget)

        # Create and add the plot visualization section
        self.plot_widget = DiagnosticPanel()
        self.layout.addWidget(self.plot_widget)

        # Create and add the control buttons section
        self.control_widget = self.setup_control_section()
        self.layout.addWidget(self.control_widget)

        # Set up signal-slot connections for UI events
        self.setup_connections()

        # Initialize the model instance
        self.model = ARMAX()

        # Create a results storage dictionary
        self.results: Dict[str, Any] = {}

        # Set the layout for the widget
        self.setLayout(self.layout)

    def setup_parameter_section(self) -> QWidget:
        """
        Creates the parameter input section of the UI.
        """
        # Create a parameter container widget
        parameter_widget = QWidget()

        # Create a form layout for parameter inputs
        form_layout = QFormLayout(parameter_widget)
        form_layout.setContentsMargins(0, 0, 0, 0)

        # Add AR order input with validator
        self.ar_label, self.ar_input, self.ar_validator = create_parameter_input("AR Order (p)", 1, lambda: Qt.Validator.IntValidator(0, 100))
        form_layout.addRow(self.ar_label, self.ar_input)
        QToolTip.setFont(QFont('SansSerif', 10))
        self.ar_label.setToolTip("The order of the autoregressive (AR) part of the model.")

        # Add MA order input with validator
        self.ma_label, self.ma_input, self.ma_validator = create_parameter_input("MA Order (q)", 1, lambda: Qt.Validator.IntValidator(0, 100))
        form_layout.addRow(self.ma_label, self.ma_input)
        QToolTip.setFont(QFont('SansSerif', 10))
        self.ma_label.setToolTip("The order of the moving average (MA) part of the model.")

        # Add constant inclusion checkbox
        self.constant_checkbox = QCheckBox("Include Constant")
        self.constant_checkbox.setChecked(True)
        form_layout.addRow(self.constant_checkbox)
        QToolTip.setFont(QFont('SansSerif', 10))
        self.constant_checkbox.setToolTip("Whether to include a constant term in the model.")

        # Add exogenous variables selection combobox
        self.exogenous_combo = QComboBox()
        self.exogenous_combo.addItem("None")
        form_layout.addRow("Exogenous Variables", self.exogenous_combo)
        QToolTip.setFont(QFont('SansSerif', 10))
        self.exogenous_combo.setToolTip("Select exogenous variables to include in the model.")

        # Set the layout on the container widget
        parameter_widget.setLayout(form_layout)

        # Return the parameter widget
        return parameter_widget

    def setup_plot_section(self) -> QWidget:
        """
        Creates the plot visualization section of the UI.
        """
        # Create a plot container widget
        plot_widget = QWidget()

        # Create a grid layout for arranging plots
        grid_layout = QFormLayout(plot_widget)
        grid_layout.setContentsMargins(0, 0, 0, 0)

        # Add ResidualPlot to the top-left position
        self.residual_plot = ResidualPlot()
        grid_layout.addRow("Residual Plot", self.residual_plot)

        # Add ACFPlot to the top-right position
        self.acf_plot = ACFPlot()
        grid_layout.addRow("ACF Plot", self.acf_plot)

        # Add PACFPlot to the bottom-left position
        self.pacf_plot = PACFPlot()
        grid_layout.addRow("PACF Plot", self.pacf_plot)

        # Add DiagnosticPanel to the bottom-right position
        self.diagnostic_panel = DiagnosticPanel()
        grid_layout.addRow("Diagnostics", self.diagnostic_panel)

        # Set the layout on the container widget
        plot_widget.setLayout(grid_layout)

        # Return the plot widget
        return plot_widget

    def setup_control_section(self) -> QWidget:
        """
        Creates the control buttons section of the UI.
        """
        # Create a control container widget
        control_widget = QWidget()

        # Define button configurations (text and handler functions)
        button_configs = [
            {'text': 'Estimate Model', 'handler': self.on_estimate_clicked},
            {'text': 'Reset', 'handler': self.on_reset_clicked},
            {'text': 'View Results', 'handler': self.on_view_results_clicked},
            {'text': 'Close', 'handler': self.on_close_clicked}
        ]

        # Use create_button_row to generate buttons
        button_row = create_button_row(button_configs)

        # Create a progress indicator for estimation feedback
        self.progress_indicator = ProgressIndicator()

        # Create a horizontal layout for the container
        hbox = QVBoxLayout(control_widget)
        hbox.addWidget(button_row)
        hbox.addWidget(self.progress_indicator)

        # Return the control widget
        return control_widget

    def setup_connections(self):
        """
        Sets up signal-slot connections for UI events.
        """
        # Connect signals from the control widget
        # self.control_widget.estimateRequested.connect(self.on_estimate_clicked)
        # self.control_widget.resetRequested.connect(self.on_reset_clicked)
        # self.control_widget.viewResultsRequested.connect(self.on_view_results_clicked)
        # self.control_widget.closeRequested.connect(self.on_close_clicked)

        # Connect signals from the model to update the UI
        # self.model.estimationStarted.connect(self.progress_indicator.start_progress)
        # self.model.estimationFinished.connect(self.progress_indicator.complete)
        # self.model.estimationError.connect(self.progress_indicator.show_error)
        pass

    def on_estimate_clicked(self):
        """
        Handles the Estimate Model button click event.
        """
        # Validate all input parameters
        is_valid, params = self.validate_parameters()
        if not is_valid:
            return

        # Create an ARMAX model instance with the parameters
        ar_order = int(self.ar_input.text())
        ma_order = int(self.ma_input.text())
        include_constant = self.constant_checkbox.isChecked()

        # Show the progress indicator
        self.progress_indicator.start_progress("Estimating Model")

        # Create an asynchronous task for estimation using TaskManager
        self.task_manager.create_task(
            name="Estimate ARMAX Model",
            function=self.async_estimate,
            kwargs={
                'ar_order': ar_order,
                'ma_order': ma_order,
                'include_constant': include_constant,
                'exogenous_variables': []
            }
        )

    def async_estimate(self, ar_order: int, ma_order: int, include_constant: bool, exogenous_variables: List[str]) -> Dict[str, Any]:
        """
        Performs asynchronous model estimation.
        """
        # Create an ARMAX model with the specified parameters
        model = ARMAX(p=ar_order, q=ma_order, include_constant=include_constant)

        # Prepare the data for estimation
        data = np.random.randn(100)  # Replace with actual data loading

        # Await the model.estimate() async method
        model.estimate(data)

        # Process and format the results
        results = {
            'ar_params': model.ar_params,
            'ma_params': model.ma_params,
            'constant': model.constant,
            'residuals': model.residuals
        }

        # Update UI with results
        self.update_plots(results)

        # Hide the progress indicator
        self.progress_indicator.complete()

        # Return the formatted results dictionary
        return results

    def on_reset_clicked(self):
        """
        Handles the Reset button click event.
        """
        # Reset all parameter inputs to default values
        self.ar_input.setText("1")
        self.ma_input.setText("1")
        self.constant_checkbox.setChecked(True)
        self.exogenous_combo.setCurrentIndex(0)

        # Clear all plot visualizations
        self.plot_widget.clear()

        # Reset the progress indicator
        self.progress_indicator.reset()

        # Clear the results storage
        self.results.clear()

    def on_view_results_clicked(self):
        """
        Handles the View Results button click event.
        """
        # Check if results are available
        if not self.results:
            QMessageBox.warning(self, "No Results", "Please estimate the model first.")
            return

        # If available, create and show an ARMAXViewer dialog
        # viewer = ARMAXViewer(self.results, self)
        # viewer.show()
        pass

    def on_close_clicked(self):
        """
        Handles the Close button click event.
        """
        # Show the close confirmation dialog
        close_dialog = QDialog(self)
        close_dialog.setWindowTitle("Confirm Close")

        # Create layout for the dialog
        dialog_layout = QVBoxLayout(close_dialog)

        # Add a label to the dialog
        message_label = QLabel("Are you sure you want to close?\nUnsaved changes will be lost.")
        dialog_layout.addWidget(message_label)

        # Create button box for the dialog
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Yes | QDialogButtonBox.StandardButton.No)
        button_box.accepted.connect(self.close)
        button_box.rejected.connect(close_dialog.reject)
        dialog_layout.addWidget(button_box)

        # Set the layout for the dialog
        close_dialog.setLayout(dialog_layout)

        # Show the dialog
        close_dialog.exec()

    def update_plots(self, results):
        """
        Updates all plots with new estimation results.
        """
        # Extract data from results dictionary
        # residuals = results['residuals']

        # Update the residual plot with new residuals
        # self.residual_plot.set_residuals(residuals)

        # Update the ACF plot with autocorrelation values
        # self.acf_plot.set_data(residuals)

        # Update the PACF plot with partial autocorrelation values
        # self.pacf_plot.set_data(residuals)

        # Refresh all plot widgets
        # self.residual_plot.update()
        # self.acf_plot.update()
        # self.pacf_plot.update()
        pass

    def validate_parameters(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Validates all input parameters before estimation.
        """
        # Check AR order input for validity (positive integer)
        try:
            ar_order = int(self.ar_input.text())
            if ar_order < 0:
                raise ValueError("AR order must be a non-negative integer.")
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "AR order must be a non-negative integer.")
            return False, None

        # Check MA order input for validity (positive integer)
        try:
            ma_order = int(self.ma_input.text())
            if ma_order < 0:
                raise ValueError("MA order must be a non-negative integer.")
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "MA order must be a non-negative integer.")
            return False, None

        # Get the constant inclusion state from checkbox
        include_constant = self.constant_checkbox.isChecked()

        # Get selected exogenous variables from combobox
        exogenous_variables = []  # Replace with actual logic to retrieve selected variables

        # If all parameters are valid, return (True, parameter_dict)
        parameter_dict = {
            'ar_order': ar_order,
            'ma_order': ma_order,
            'include_constant': include_constant,
            'exogenous_variables': exogenous_variables
        }
        return True, parameter_dict

class ModelConfigurationWidget(QWidget):
    """
    Widget for configuring ARMAX model parameters.
    """
    def __init__(self, parent: Optional[QWidget] = None):
        """
        Initializes the model configuration widget.
        """
        super().__init__(parent)

        # Create a form layout for parameter inputs
        self.layout = QFormLayout(self)

        # Add AR order input with integer validator
        self.ar_order_input = QLineEdit()
        self.ar_order_input.setValidator(Qt.Validator.IntValidator())
        self.layout.addRow("AR Order (p):", self.ar_order_input)

        # Add MA order input with integer validator
        self.ma_order_input = QLineEdit()
        self.ma_order_input.setValidator(Qt.Validator.IntValidator())
        self.layout.addRow("MA Order (q):", self.ma_order_input)

        # Add constant inclusion checkbox
        self.constant_checkbox = QCheckBox("Include Constant")
        self.layout.addRow(self.constant_checkbox)

        # Add exogenous variables selection combobox
        self.exogenous_combo = QComboBox()
        self.exogenous_combo.addItem("None")
        self.layout.addRow("Exogenous Variables:", self.exogenous_combo)

        # Set default values for all inputs
        self.reset_parameters()

        # Connect input signals to validation slots
        # self.ar_order_input.textChanged.connect(self.validate_parameters)
        # self.ma_order_input.textChanged.connect(self.validate_parameters)

    def get_parameters(self) -> Dict[str, Any]:
        """
        Retrieves current parameter values from the widget.
        """
        # Get AR order from input field and convert to integer
        ar_order = int(self.ar_order_input.text())

        # Get MA order from input field and convert to integer
        ma_order = int(self.ma_order_input.text())

        # Get constant inclusion state from checkbox
        include_constant = self.constant_checkbox.isChecked()

        # Get selected exogenous variables from combobox
        exogenous_variables = []  # Replace with actual logic to retrieve selected variables

        # Return a dictionary with all parameter values
        return {
            'ar_order': ar_order,
            'ma_order': ma_order,
            'include_constant': include_constant,
            'exogenous_variables': exogenous_variables
        }

    def reset_parameters(self):
        """
        Resets all parameter inputs to default values.
        """
        # Set AR order input to default value (1)
        self.ar_order_input.setText("1")

        # Set MA order input to default value (1)
        self.ma_order_input.setText("1")

        # Set constant checkbox to default state (checked)
        self.constant_checkbox.setChecked(True)

        # Reset exogenous variables combobox selection
        self.exogenous_combo.setCurrentIndex(0)

        # Emit parameterChanged signal
        # self.parameterChanged.emit()

    def validate_parameters(self) -> bool:
        """
        Validates all parameter inputs.
        """
        # Check if AR order is a positive integer
        try:
            ar_order = int(self.ar_order_input.text())
            if ar_order < 0:
                return False
        except ValueError:
            return False

        # Check if MA order is a positive integer
        try:
            ma_order = int(self.ma_order_input.text())
            if ma_order < 0:
                return False
        except ValueError:
            return False

        # Return validation result
        return True

class DiagnosticVisualizationWidget(QWidget):
    """
    Widget for displaying model diagnostic visualizations.
    """
    def __init__(self, parent: Optional[QWidget] = None):
        """
        Initializes the diagnostic visualization widget.
        """
        super().__init__(parent)

        # Create a grid layout for arranging plots
        self.layout = QFormLayout(self)

        # Create and add ResidualPlot to the top-left position
        self.residual_plot = ResidualPlot()
        self.layout.addRow("Residual Plot", self.residual_plot)

        # Add ACFPlot to the top-right position
        self.acf_plot = ACFPlot()
        self.layout.addRow("ACF Plot", self.acf_plot)

        # Add PACFPlot to the bottom-left position
        self.pacf_plot = PACFPlot()
        self.layout.addRow("PACF Plot", self.pacf_plot)

        # Add DiagnosticPanel to the bottom-right position
        self.diagnostic_panel = DiagnosticPanel()
        self.layout.addRow("Diagnostics", self.diagnostic_panel)

    def update_visualizations(self, results: Dict[str, Any]):
        """
        Updates all visualizations with new model results.
        """
        # Extract data from results dictionary
        residuals = results['residuals']

        # Update the residual plot with new residuals
        self.residual_plot.set_residuals(residuals)

        # Update the ACF plot with autocorrelation values
        self.acf_plot.set_data(residuals)

        # Update the PACF plot with partial autocorrelation values
        self.pacf_plot.set_data(residuals)

        # Update the diagnostic panel with test results
        # self.diagnostic_panel.update_diagnostics(results['diagnostics'])

        # Refresh all visualization widgets
        self.residual_plot.update()
        self.acf_plot.update()
        self.pacf.update()
        # self.diagnostic_panel.update()

    def clear_visualizations(self):
        """
        Clears all visualizations.
        """
        # Clear the residual plot
        self.residual_plot.clear()

        # Clear the ACF plot
        self.acf_plot.clear()

        # Clear the PACF plot
        self.pacf_plot.clear()

        # Clear the diagnostic panel
        # self.diagnostic_panel.clear()

        # Refresh all visualization widgets
        self.residual_plot.update()
        self.acf_plot.update()
        self.pacf_plot.update()
        # self.diagnostic_panel.update()

class ControlButtonsWidget(QWidget):
    """
    Widget for control buttons in the ARMAX UI.
    """
    estimateRequested = pyqtSignal()
    resetRequested = pyqtSignal()
    viewResultsRequested = pyqtSignal()
    closeRequested = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None):
        """
        Initializes the control buttons widget.
        """
        super().__init__(parent)

        # Create a horizontal layout for buttons
        self.layout = QHBoxLayout(self)

        # Create and add Estimate Model button
        self.estimate_button = QPushButton("Estimate Model")
        self.estimate_button.clicked.connect(self.on_estimate_clicked)
        self.layout.addWidget(self.estimate_button)

        # Create and add Reset button
        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.on_reset_clicked)
        self.layout.addWidget(self.reset_button)

        # Create and add View Results button
        self.view_results_button = QPushButton("View Results")
        self.view_results_button.clicked.connect(self.on_view_results_clicked)
        self.layout.addWidget(self.view_results_button)

        # Create and add Close button
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.on_close_clicked)
        self.layout.addWidget(self.close_button)

        # Create and add progress indicator
        self.progress_indicator = ProgressIndicator()
        self.layout.addWidget(self.progress_indicator)

        # Apply styling to all buttons
        for button in [self.estimate_button, self.reset_button, self.view_results_button, self.close_button]:
            button.setFixedSize(QSize(*CONTROL_BUTTON_SIZE))
            button.setStyleSheet("""
                QPushButton {
                    background-color: #e0e0e0;
                    border: 1px solid #c0c0c0;
                    border-radius: 4px;
                    padding: 5px;
                }
                QPushButton:hover {
                    background-color: #d0d0d0;
                }
            """)

    def on_estimate_clicked(self):
        """
        Handles Estimate Model button click.
        """
        # Emit estimateRequested signal
        self.estimateRequested.emit()

        # Show progress indicator
        self.progress_indicator.start_progress("Estimating Model")

        # Disable estimate button during processing
        self.estimate_button.setEnabled(False)

    def on_reset_clicked(self):
        """
        Handles Reset button click.
        """
        # Emit resetRequested signal
        self.resetRequested.emit()

        # Reset progress indicator
        self.progress_indicator.reset()

        # Enable estimate button
        self.estimate_button.setEnabled(True)

    def on_view_results_clicked(self):
        """
        Handles View Results button click.
        """
        # Emit viewResultsRequested signal
        self.viewResultsRequested.emit()

    def on_close_clicked(self):
        """
        Handles Close button click.
        """
        # Emit closeRequested signal
        self.closeRequested.emit()

    def update_progress(self, value: float):
        """
        Updates the progress indicator.
        """
        # Update progress indicator with new value
        self.progress_indicator.update_progress(value)

        # If value reaches 100%, reset progress indicator
        if value >= 1.0:
            self.progress_indicator.reset()
            self.estimate_button.setEnabled(True)