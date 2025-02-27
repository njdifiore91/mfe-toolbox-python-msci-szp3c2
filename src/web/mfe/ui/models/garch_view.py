#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyQt6-based UI component for visualizing and interacting with GARCH volatility
models in the MFE Toolbox. Provides a dedicated interface for configuring,
estimating, diagnosing, and forecasting GARCH models with various orders and
distributions.
"""

import asyncio  # standard library
import logging  # standard library
from typing import Any, Dict, Optional, Union  # standard library

import numpy  # numpy 1.26.3
import pandas  # pandas 2.1.4
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot  # PyQt6 6.6.1
from PyQt6.QtWidgets import (  # PyQt6 6.6.1
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from dataclasses import dataclass  # standard library
import dataclasses
import typing

# Internal imports
from ..components.diagnostic_panel import DiagnosticPanel
from ..components.model_equation import ModelEquation
from ..components.parameter_table import ParameterTable
from ..plots.volatility_plot import VolatilityPlot
from ..async.worker import WorkerManager
from ...backend.mfe.models.garch import GARCH

# Module logger
logger = logging.getLogger(__name__)

DISTRIBUTION_TYPES = {"normal": "Normal", "student": "Student-t", "ged": "GED", "skewed-t": "Skewed Student-t"}


@dataclasses.dataclass
class GARCHViewSettings:
    """
    Data class for storing GARCH model view configuration settings
    """
    p: int = 1
    q: int = 1
    distribution: str = "normal"
    dist_params: dict = dataclasses.field(default_factory=dict)
    model_params: dict = dataclasses.field(default_factory=dict)
    forecast_horizon: int = 10
    forecast_alpha: float = 0.05


class GARCHView(QWidget):
    """
    PyQt6-based UI widget for GARCH volatility model configuration, estimation,
    and visualization
    """

    def __init__(self, parent: Optional[QWidget] = None):
        """
        Initialize the GARCH model view component
        """
        super().__init__(parent)

        # Initialize settings and data
        self._worker_manager = WorkerManager()
        self._settings = GARCHViewSettings()
        self._data: numpy.ndarray = None
        self._returns: numpy.ndarray = None
        self._model_fitted = False
        self._model: GARCH = None
        self._estimation_result: dict = None
        self._forecast_result: dict = None

        # Initialize UI components
        self.init_ui()

        # Set up signal connections
        self.setup_connections()

        logger.info("GARCHView initialized")

    def init_ui(self) -> None:
        """
        Initialize the user interface components
        """
        # Main layout
        layout = QVBoxLayout(self)

        # Configuration section
        config_group = QGroupBox("Model Configuration")
        config_layout = QHBoxLayout()

        # GARCH orders
        p_label = QLabel("GARCH (p):")
        self._p_spin = QSpinBox()
        self._p_spin.setRange(1, 10)
        self._p_spin.setValue(self._settings.p)
        q_label = QLabel("ARCH (q):")
        self._q_spin = QSpinBox()
        self._q_spin.setRange(1, 10)
        self._q_spin.setValue(self._settings.q)

        # Distribution type
        dist_label = QLabel("Distribution:")
        self._dist_type_combo = QComboBox()
        for key, value in DISTRIBUTION_TYPES.items():
            self._dist_type_combo.addItem(value, key)
        self._dist_type_combo.setCurrentIndex(self._dist_type_combo.findData(self._settings.distribution))

        config_layout.addWidget(p_label)
        config_layout.addWidget(self._p_spin)
        config_layout.addWidget(q_label)
        config_layout.addWidget(self._q_spin)
        config_layout.addWidget(dist_label)
        config_layout.addWidget(self._dist_type_combo)
        config_group.setLayout(config_layout)

        # Tab widget
        self._tab_widget = QTabWidget()

        # Parameter table
        self._parameter_table = ParameterTable()
        self._tab_widget.addTab(self._parameter_table, "Parameters")

        # Diagnostic panel
        self._diagnostic_panel = DiagnosticPanel()
        self._tab_widget.addTab(self._diagnostic_panel, "Diagnostics")

        # Model equation
        self._model_equation = ModelEquation()
        self._tab_widget.addTab(self._model_equation, "Equation")

        # Volatility plot
        self._volatility_plot = VolatilityPlot()
        self._tab_widget.addTab(self._volatility_plot, "Volatility Plot")

        # Buttons
        button_layout = QHBoxLayout()
        self._estimate_button = QPushButton("Estimate")
        self._forecast_button = QPushButton("Forecast")
        self._reset_button = QPushButton("Reset")
        self._forecast_button.setEnabled(False)  # Disable until model is fitted

        button_layout.addWidget(self._estimate_button)
        button_layout.addWidget(self._forecast_button)
        button_layout.addWidget(self._reset_button)

        # Progress bar
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.hide()

        # Add widgets to layout
        layout.addWidget(config_group)
        layout.addWidget(self._tab_widget)
        layout.addLayout(button_layout)
        layout.addWidget(self._progress_bar)

        self.setLayout(layout)

    def setup_connections(self) -> None:
        """
        Set up signal-slot connections for UI components
        """
        self._p_spin.valueChanged.connect(self.update_model_equation)
        self._q_spin.valueChanged.connect(self.update_model_equation)
        self._dist_type_combo.currentIndexChanged.connect(self.update_dist_parameters)
        self._estimate_button.clicked.connect(self.on_estimate_clicked)
        self._forecast_button.clicked.connect(self.on_forecast_clicked)
        self._reset_button.clicked.connect(self.reset_model)
        self._tab_widget.currentChanged.connect(self.on_tab_changed)

    def set_data(self, data: Union[numpy.ndarray, pandas.Series, pandas.DataFrame]) -> bool:
        """
        Set the financial time series data for analysis
        """
        try:
            # Convert data to numpy array if not already
            if isinstance(data, pandas.Series):
                data = data.values
            elif isinstance(data, pandas.DataFrame):
                data = data.values.flatten()
            self._data = numpy.asarray(data)

            # Calculate returns as percentage changes
            self._returns = (self._data[1:] - self._data[:-1]) / self._data[:-1]

            # Reset any existing model fit
            self.reset_model()

            # Update UI components with new data
            self._volatility_plot.set_data(self._returns)
            self.update_model_equation()

            return True
        except Exception as e:
            logger.error(f"Error setting data: {str(e)}")
            QMessageBox.critical(self, "Data Error", f"Failed to set data: {str(e)}")
            return False

    @pyqtSlot()
    def update_model_equation(self) -> None:
        """
        Update the displayed model equation based on current settings
        """
        p = self._p_spin.value()
        q = self._q_spin.value()
        self._model_equation.set_garch_equation(p, q)
        self._settings.p = p
        self._settings.q = q

    @pyqtSlot()
    def update_dist_parameters(self) -> None:
        """
        Update distribution parameters based on selected distribution type
        """
        dist_type = self._dist_type_combo.currentData()
        self._settings.distribution = dist_type

    @pyqtSlot()
    def on_estimate_clicked(self) -> None:
        """
        Handle estimate button click by starting model estimation
        """
        if self._data is None:
            QMessageBox.warning(self, "Data Error", "No data loaded. Please load data first.")
            return

        # Collect current model parameters from UI
        self._settings.p = self._p_spin.value()
        self._settings.q = self._q_spin.value()
        self._settings.distribution = self._dist_type_combo.currentData()

        # Disable UI inputs during estimation
        self.enable_ui(False)
        self._progress_bar.show()

        # Create async worker for estimation
        worker = self._worker_manager.create_async_worker(self.async_estimate_model)
        worker.signals.progress.connect(self.on_estimation_progress)
        worker.signals.result.connect(self.on_estimation_complete)
        worker.signals.error.connect(self.on_estimation_error)

        # Start worker for asynchronous estimation
        self._worker_manager.start_worker(worker)
        logger.info("Estimation started")

    @pyqtSlot(float)
    def on_estimation_progress(self, progress: float) -> None:
        """
        Handle progress updates during model estimation
        """
        self._progress_bar.setValue(int(progress))
        if progress == 100:
            self._progress_bar.setFormat("Finalizing...")

    @pyqtSlot(object)
    def on_estimation_complete(self, result: object) -> None:
        """
        Handle completion of model estimation
        """
        self._estimation_result = result
        self._model = result["model"]
        self._parameter_table.set_parameter_data(result["parameters"])
        self._model_equation.update_parameters(result["parameters"])
        self._diagnostic_panel.async_update_diagnostics(self._returns)
        self._volatility_plot.set_data(self._returns, result["conditional_variances"])
        self._model_fitted = True
        self.enable_ui(True)
        self._progress_bar.hide()
        self._tab_widget.setCurrentIndex(1)
        logger.info("Estimation complete")

    @pyqtSlot(Exception)
    def on_estimation_error(self, error: Exception) -> None:
        """
        Handle errors during model estimation
        """
        self.enable_ui(True)
        self._progress_bar.hide()
        QMessageBox.critical(self, "Estimation Error", f"Failed to estimate model: {str(error)}")
        logger.error(f"Estimation error: {str(error)}")

    async def async_estimate_model(self) -> dict:
        """
        Asynchronously estimate GARCH model
        """
        # Get current model settings from _settings
        p = self._settings.p
        q = self._settings.q
        distribution = self._settings.distribution

        # Create GARCH model with appropriate p, q, and distribution
        model = GARCH(p=p, q=q, distribution=distribution)

        # Await model.fit_async with returns data
        result = await model.fit_async(self._returns)

        # Get model summary and diagnostic information
        summary = model.summary()

        # Create result dictionary with model and diagnostics
        result_dict = {
            "model": model,
            "parameters": summary["parameters"],
            "conditional_variances": model._variance(model.parameters, self._returns)
        }

        # Return estimation result dictionary
        return result_dict

    @pyqtSlot()
    def on_forecast_clicked(self) -> None:
        """
        Handle forecast button click by generating volatility forecasts
        """
        if not self._model_fitted:
            QMessageBox.warning(self, "Model Error", "Model must be estimated before forecasting.")
            return

        # Get forecast horizon and alpha from UI
        horizon = self._forecast_horizon_spin.value()
        alpha = self._forecast_alpha_spin.value()

        # Update _settings with forecast settings
        self._settings.forecast_horizon = horizon
        self._settings.forecast_alpha = alpha

        # Generate forecast using model.forecast
        forecast = self._model.forecast(horizon)

        # Update volatility plot with forecast data
        self._volatility_plot.set_forecast_data(forecast)

        # Store forecast result
        self._forecast_result = forecast

        # Switch to volatility plot tab
        self._tab_widget.setCurrentIndex(3)
        logger.info("Forecast generated")

    @pyqtSlot()
    def reset_model(self) -> None:
        """
        Reset the model and clear all results
        """
        self._model_fitted = False
        self._model = None
        self._estimation_result = None
        self._forecast_result = None
        self._parameter_table.clear()
        self._diagnostic_panel.clear_plots()
        self._model_equation.clear()
        self._volatility_plot.clear()
        self.enable_ui(True)
        self._tab_widget.setCurrentIndex(0)
        logger.info("Model reset")

    @pyqtSlot(int)
    def on_tab_changed(self, index: int) -> None:
        """
        Handle tab widget tab changes
        """
        tab_text = self._tab_widget.tabText(index)
        if tab_text == "Parameters":
            self._parameter_table.refresh_parameter_display()
        elif tab_text == "Diagnostics":
            self._diagnostic_panel.async_update_diagnostics(self._returns)
        elif tab_text == "Volatility Plot":
            self._volatility_plot.async_update_plot()
        logger.info(f"Tab changed to: {tab_text}")

    def enable_ui(self, enabled: bool) -> None:
        """
        Enable or disable UI components
        """
        self._p_spin.setEnabled(enabled)
        self._q_spin.setEnabled(enabled)
        self._dist_type_combo.setEnabled(enabled)
        self._estimate_button.setEnabled(enabled)
        self._forecast_button.setEnabled(enabled and self._model_fitted)

    def get_settings(self) -> GARCHViewSettings:
        """
        Get current GARCH view settings
        """
        return dataclasses.replace(self._settings)

    def apply_settings(self, settings: GARCHViewSettings) -> None:
        """
        Apply settings to the GARCH view
        """
        self._settings = dataclasses.replace(settings)
        self._p_spin.setValue(settings.p)
        self._q_spin.setValue(settings.q)
        self._dist_type_combo.setCurrentIndex(self._dist_type_combo.findData(settings.distribution))
        self.update_dist_parameters()
        self.update_model_equation()

    def get_current_model(self) -> Union[GARCH, None]:
        """
        Get the currently estimated GARCH model
        """
        return self._model if self._model_fitted else None

    def get_estimation_summary(self) -> str:
        """
        Get text summary of GARCH estimation results
        """
        if not self._model_fitted:
            return "No model estimated"

        # Format GARCH type, orders, and distribution
        summary = f"GARCH({self._settings.p}, {self._settings.q}) with {self._settings.distribution} distribution\n"

        # Format parameter estimates with standard errors
        summary += "Parameter Estimates:\n"
        for param_name, param_values in self._estimation_result["parameters"].items():
            summary += f"  {param_name}: {param_values['estimate']:.4f} (Std. Err: {param_values['std_error']:.4f})\n"

        # Format log-likelihood and information criteria
        summary += f"Log-Likelihood: {self._estimation_result['log_likelihood']:.2f}\n"
        summary += f"AIC: {self._estimation_result['aic']:.2f}\n"
        summary += f"BIC: {self._estimation_result['bic']:.2f}\n"

        # Format diagnostics (persistence, half-life)
        summary += f"Persistence: {self._model.persistence:.4f}\n"
        summary += f"Half-Life: {format_half_life(self._model.half_life)}\n"

        return summary

    def export_results(self, filepath: str) -> bool:
        """
        Export estimation results to file
        """
        if not self._model_fitted:
            QMessageBox.warning(self, "Export Error", "No model estimated. Estimate model first.")
            return False

        try:
            # Generate summary text
            summary_text = self.get_estimation_summary()

            # Write summary to file
            with open(filepath, "w") as f:
                f.write(summary_text)

            # Export parameter table if available
            if self._parameter_table:
                self._parameter_table.export_table(filepath + "_parameters.csv")

            # Export volatility plot if available
            if self._volatility_plot:
                self._volatility_plot.save_figure(filepath + "_volatility.png")

            logger.info(f"Exported results to {filepath}")
            QMessageBox.information(self, "Export Successful", f"Results exported to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Error exporting results: {str(e)}")
            QMessageBox.critical(self, "Export Error", f"Failed to export results: {str(e)}")
            return False

    def has_results(self) -> bool:
        """
        Check if the view has estimation results
        """
        return self._model_fitted


def format_half_life(half_life: float) -> str:
    """
    Formats the half-life value with appropriate units for display
    """
    if half_life is None:
        return "-"
    if half_life < 1 / 24:
        return f"{half_life * 24 * 60:.0f} hours"
    elif half_life < 1:
        return f"{half_life * 24:.0f} days"
    elif half_life < 365:
        return f"{half_life / 7:.0f} weeks"
    else:
        return f"{half_life / 365:.0f} years"