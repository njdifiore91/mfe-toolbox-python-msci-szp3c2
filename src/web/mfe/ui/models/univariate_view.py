from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QPushButton, QLabel, QTabWidget, QGroupBox, QSpinBox, QDoubleSpinBox
)  # PyQt6 6.6.1
from PyQt6.QtCore import pyqtSignal, QThreadPool  # PyQt6 6.6.1
import numpy as np  # numpy 1.26.3
import pandas as pd  # pandas 2.1.4
import matplotlib.pyplot as plt  # matplotlib 3.7.2

from src.web.mfe.ui.async.worker import AsyncTaskWorker  # Handles asynchronous model estimation in the background
from src.web.mfe.ui.async.signals import TaskSignals  # Provides signals for async task communication
from src.web.mfe.ui.async.task_manager import TaskManager  # Manages asynchronous task execution
from src.web.mfe.ui.components.parameter_table import ParameterTable  # Displays and allows editing of model parameters
from src.web.mfe.ui.components.diagnostic_panel import DiagnosticPanel  # Displays diagnostic plots for model evaluation
from src.web.mfe.ui.components.model_equation import ModelEquation  # Renders LaTeX equations of model specifications
from src.web.mfe.ui.components.progress_indicator import ProgressIndicator  # Displays estimation progress
from src.web.mfe.ui.components.statistical_metrics import StatisticalMetrics  # Displays statistical metrics of the estimated model
from src.web.mfe.ui.components.error_display import ErrorDisplay  # Displays error messages to the user
from src.web.mfe.ui.plots.volatility_plot import VolatilityPlot  # Creates volatility plots
from src.web.mfe.ui.plots.residual_plot import ResidualPlot  # Displays residual plots
from src.web.mfe.ui.plots.time_series_plot import TimeSeriesPlot  # Creates time series plots
from src.web.mfe.ui.plots.density_plot import DensityPlot  # Creates density plots for residuals
from src.backend.mfe.models.garch import GARCH  # GARCH model implementation
from src.backend.mfe.models.egarch import EGARCH  # EGARCH model implementation
from src.backend.mfe.models.tarch import TARCH  # TARCH model implementation
from src.backend.mfe.models.agarch import AGARCH  # AGARCH model implementation
from src.backend.mfe.models.aparch import APARCH  # APARCH model implementation
from src.backend.mfe.models.figarch import FIGARCH  # FIGARCH model implementation
from src.backend.mfe.models.igarch import IGARCH  # IGARCH model implementation
import logging
class UnivariateView(QWidget):
    """
    A PyQt6 widget that provides an interface for configuring, estimating, and visualizing univariate volatility models.
    """
    
    def __init__(self, parent: QWidget):
        """
        Initializes the UnivariateView widget with UI components and model options.
        """
        super().__init__(parent)
        
        # Initialize model_options dict with univariate models
        self.model_options = {
            "GARCH": GARCH,
            "EGARCH": EGARCH,
            "TARCH": TARCH,
            "AGARCH": AGARCH,
            "APARCH": APARCH,
            "FIGARCH": FIGARCH,
            "IGARCH": IGARCH
        }
        
        # Initialize UI components
        self.initialize_ui()
        
        # Initialize current_results
        self.current_results = {}
        
        # Initialize data
        self.data = None
        
        # Initialize task_manager for async operations
        self.task_manager = TaskManager()

    def initialize_ui(self):
        """
        Creates and arranges all UI components for the UnivariateView.
        """
        # Create main layout (QVBoxLayout)
        main_layout = QVBoxLayout(self)
        
        # Create model selection section
        model_selection_widget = self.create_model_selection()
        main_layout.addWidget(model_selection_widget)
        
        # Create parameter input section using ParameterTable
        self.parameter_table = self.create_parameter_panel()
        main_layout.addWidget(self.parameter_table)
        
        # Create action buttons (Estimate, Reset, Export)
        action_layout = QHBoxLayout()
        self.estimate_button = QPushButton("Estimate")
        self.estimate_button.clicked.connect(self.estimate_model)
        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_view)

        self.export_button = QPushButton("Export Results")
        self.export_button.clicked.connect(self.export_results)
        self.export_button.setEnabled(False)
        
        action_layout.addWidget(self.estimate_button)
        action_layout.addWidget(self.reset_button)
        action_layout.addWidget(self.export_button)
        main_layout.addLayout(action_layout)
        
        # Create tabs for Results, Diagnostics, and Forecasts
        self.tab_widget = QTabWidget()
        self.results_tab = QWidget()
        self.diagnostics_tab = QWidget()
        
        self.tab_widget.addTab(self.results_tab, "Results")
        self.tab_widget.addTab(self.diagnostics_tab, "Diagnostics")
        
        # Results tab layout
        results_layout = QVBoxLayout()
        self.model_equation = ModelEquation()
        self.statistical_metrics = StatisticalMetrics()
        results_layout.addWidget(self.model_equation)
        results_layout.addWidget(self.statistical_metrics)
        self.results_tab.setLayout(results_layout)
        
        # Diagnostics tab layout
        diagnostics_layout = QVBoxLayout()
        self.diagnostic_panel = DiagnosticPanel()
        diagnostics_layout.addWidget(self.diagnostic_panel)
        self.diagnostics_tab.setLayout(diagnostics_layout)
        
        main_layout.addWidget(self.tab_widget)
        
        # Add ProgressIndicator for estimation progress
        self.progress_indicator = ProgressIndicator()
        main_layout.addWidget(self.progress_indicator)
        
        # Add ErrorDisplay for error messages
        self.error_display = ErrorDisplay()
        main_layout.addWidget(self.error_display)
        
        # Connect signals to slots for UI interactions
        self.setLayout(main_layout)

    def create_model_selection(self):
        """
        Creates the model selection dropdown and related UI elements.
        """
        # Create container widget
        container = QWidget()
        layout = QHBoxLayout()
        container.setLayout(layout)
        
        # Create QComboBox for model type selection
        self.model_combo = QComboBox()
        for model_name in self.model_options.keys():
            self.model_combo.addItem(model_name)
        layout.addWidget(QLabel("Select Model:"))
        layout.addWidget(self.model_combo)
        
        # Connect selection change signal to update_parameter_view
        self.model_combo.currentIndexChanged.connect(self.update_parameter_view)
        
        return container

    def create_parameter_panel(self):
        """
        Creates the parameter configuration panel for the selected model.
        """
        # Create ParameterTable instance
        parameter_table = ParameterTable()
        
        # Configure with default parameters for GARCH model
        default_params = {"omega": 0.1, "alpha": 0.2, "beta": 0.8}
        param_data = {param: {"value": value} for param, value in default_params.items()}
        
        return parameter_table

    def create_results_panel(self):
        """
        Creates the panel for displaying model estimation results.
        """
        # Create QTabWidget for organizing results
        tab_widget = QTabWidget()
        
        # Create ModelEquation widget for displaying model equation
        model_equation = ModelEquation()
        
        # Create StatisticalMetrics widget for displaying statistics
        statistical_metrics = StatisticalMetrics()
        
        # Create plot widgets for volatility and residuals
        volatility_plot = VolatilityPlot()
        residual_plot = ResidualPlot()
        
        # Organize widgets into tabs
        tab_widget.addTab(model_equation, "Model Equation")
        tab_widget.addTab(statistical_metrics, "Statistics")
        tab_widget.addTab(volatility_plot, "Volatility Plot")
        tab_widget.addTab(residual_plot, "Residual Plot")
        
        return tab_widget

    def update_parameter_view(self):
        """
        Updates the parameter view based on the selected model type.
        """
        # Get the currently selected model type
        model_name = self.model_combo.currentText()
        
        # Clear the parameter table
        self.parameter_table.clear()
        
        # Get default parameters for the selected model type
        if model_name == "GARCH":
            default_params = {"omega": 0.1, "alpha": 0.2, "beta": 0.8}
        elif model_name == "EGARCH":
            default_params = {"omega": 0.1, "alpha": 0.2, "beta": 0.8, "gamma": 0.1}
        else:
            default_params = {"omega": 0.1, "alpha": 0.2, "beta": 0.8}
        
        # Update the parameter table with model-specific parameters
        param_data = {param: {"value": value} for param, value in default_params.items()}
        self.parameter_table.set_parameter_data(param_data)
        
        # Update model equation display
        self.model_equation.set_arma_equation(1, 1, True)

    def estimate_model(self):
        """
        Initiates the asynchronous estimation of the selected model with the configured parameters.
        """
        # Show progress indicator
        self.progress_indicator.show()
        
        # Get selected model type
        model_name = self.model_combo.currentText()
        
        # Get parameter values from parameter table
        params = self.parameter_table.get_parameters()
        
        # Create model instance based on selection (GARCH, EGARCH, etc.)
        if model_name == "GARCH":
            model = GARCH(1, 1)
        elif model_name == "EGARCH":
            model = EGARCH(1, 1, 1)
        else:
            model = GARCH(1, 1)
        
        # Prepare data for estimation
        data = np.random.randn(100)  # Replace with actual data loading
        
        # Create AsyncTaskWorker for model estimation
        worker = AsyncTaskWorker(model.fit, data, params)
        
        # Connect task signals to appropriate slots
        worker.signals.result.connect(self.handle_estimation_result)
        worker.signals.error.connect(self.handle_estimation_error)
        worker.signals.progress.connect(self.handle_estimation_progress)
        
        # Submit task to TaskManager for background execution
        self.task_manager.submit_task(worker)

    def handle_estimation_progress(self, progress):
        """
        Updates the progress indicator during model estimation.
        """
        # Update progress indicator with current progress value
        self.progress_indicator.set_progress(progress)

    def handle_estimation_result(self, results):
        """
        Processes and displays the results of a successful model estimation.
        """
        # Store results in current_results
        self.current_results = results
        
        # Hide progress indicator
        self.progress_indicator.hide()
        
        # Update model equation with estimated parameters
        self.model_equation.set_arma_equation(1, 1, True, results["parameters"])
        
        # Update statistical metrics with model statistics
        self.statistical_metrics.update_metrics(results["fit_stats"])
        
        # Update volatility plot with estimated volatility
        # self.volatility_plot.plot(results["volatility"])
        
        # Update residual plots
        # self.residual_plot.plot(results["residuals"])
        
        # Enable export button
        self.export_button.setEnabled(True)

    def handle_estimation_error(self, error):
        """
        Handles errors that occur during model estimation.
        """
        # Hide progress indicator
        self.progress_indicator.hide()
        
        # Display error message using ErrorDisplay
        self.error_display.show_error(str(error))
        
        # Log error details
        logging.error(f"Estimation error: {error}")

    def reset_view(self):
        """
        Resets the view to its initial state, clearing all results and plots.
        """
        # Clear current_results
        self.current_results = {}
        
        # Clear all plots
        # self.volatility_plot.clear()
        # self.residual_plot.clear()
        
        # Reset model equation
        self.model_equation.set_arma_equation(1, 1, True)
        
        # Reset statistical metrics
        self.statistical_metrics.clear()
        
        # Reset parameter table to default values
        self.update_parameter_view()
        
        # Disable export button
        self.export_button.setEnabled(False)

    def export_results(self):
        """
        Exports the current model results to a file.
        """
        # Show file dialog for saving location
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getSaveFileName(self, "Save Results", "", "CSV Files (*.csv)")
        
        # Format results for export
        results_string = str(self.current_results)  # Replace with actual formatting
        
        # Save results to selected file
        with open(file_path, "w") as file:
            file.write(results_string)
        
        # Show confirmation message on success
        QMessageBox.information(self, "Results Saved", "Results saved successfully!")

    def set_data(self, data: np.ndarray):
        """
        Sets the data to be used for model estimation.
        """
        # Store the data
        self.data = data
        
        # Update time series plot with new data
        # self.time_series_plot.plot(data)
        
        # Enable estimation button if data is valid
        self.estimate_button.setEnabled(True)

    def get_results(self):
        """
        Retrieves the current model estimation results.
        """
        # Return the current_results dictionary
        return self.current_results

    def update_plots(self):
        """
        Updates all plots with the current estimation results.
        """
        # Check if results are available
        if not self.current_results:
            return
        
        # Update time series plot with original data
        # self.time_series_plot.plot(self.data)
        
        # Update volatility plot with estimated volatility
        # self.volatility_plot.plot(self.current_results["volatility"])
        
        # Update residual plot with model residuals
        # self.residual_plot.plot(self.current_results["residuals"])
        
        # Refresh plot displays
        # self.time_series_plot.update()
        # self.volatility_plot.update()
        # self.residual_plot.update()