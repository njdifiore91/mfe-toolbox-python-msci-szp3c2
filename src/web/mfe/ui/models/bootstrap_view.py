"""
MFE Toolbox UI - Bootstrap View Module

This module provides a PyQt6-based UI component for Bootstrap analysis in the MFE Toolbox,
enabling users to configure bootstrap parameters, run analyses with real-time progress 
tracking, and visualize results through interactive charts and statistics.
"""

import logging
from typing import Optional, Dict, Any, List, Union, Tuple

import numpy as np
import pandas as pd
from PyQt6.QtCore import Qt, pyqtSignal, QObject
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox, 
    QSpinBox, QGroupBox, QDoubleSpinBox, QSizePolicy
)

# Internal imports
from ....backend.mfe.core.bootstrap import (
    block_bootstrap, stationary_bootstrap, block_bootstrap_async, 
    stationary_bootstrap_async, calculate_bootstrap_ci
)
from ....backend.mfe.utils.validation import validate_input
from ..async.task_manager import TaskManager
from ..async.signals import BaseTaskSignals
from ..components.progress_indicator import ProgressIndicator
from ..components.parameter_table import ParameterTable
from ..plots.time_series_plot import TimeSeriesPlot
from ..components.statistical_metrics import StatisticalMetrics

# Configure module logger
logger = logging.getLogger(__name__)


class BootstrapViewSignals(QObject):
    """Signal class for the bootstrap view to emit events."""
    bootstrap_started = pyqtSignal()
    bootstrap_completed = pyqtSignal()
    bootstrap_error = pyqtSignal(Exception)
    parameters_updated = pyqtSignal()
    
    def __init__(self):
        """Initializes the signal class."""
        super().__init__()


class BootstrapView(QWidget):
    """
    PyQt6-based UI component for Bootstrap analysis with configurable parameters
    and real-time visualization.
    """
    
    def __init__(self, parent: Optional[QWidget] = None):
        """
        Initializes the bootstrap view with UI components and signal connections.
        
        Parameters
        ----------
        parent : QWidget, optional
            Parent widget
        """
        super().__init__(parent)
        
        # Initialize signals
        self._signals = BootstrapViewSignals()
        
        # Initialize task manager for async operations
        self._task_manager = TaskManager()
        
        # Initialize state
        self._bootstrap_params = {
            "method": "block",
            "block_size": 20,
            "probability": 0.1,
            "num_bootstrap": 1000
        }
        self._data = None
        self._bootstrap_results = None
        self._is_running = False
        
        # Setup UI and connect signals
        self._setup_ui()
        self._connect_signals()
        
        logger.debug("Bootstrap view initialized")
    
    def _setup_ui(self):
        """Sets up the UI components for the bootstrap view."""
        # Create main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # ========== Parameter Input Section ==========
        param_group = QGroupBox("Bootstrap Parameters")
        param_layout = QVBoxLayout()
        param_layout.setContentsMargins(10, 15, 10, 10)
        param_layout.setSpacing(8)
        
        # Bootstrap method selection
        method_layout = QHBoxLayout()
        method_label = QLabel("Bootstrap Method:")
        method_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        
        self._method_combo = QComboBox()
        self._method_combo.addItems(["Block Bootstrap", "Stationary Bootstrap"])
        self._method_combo.setToolTip(
            "Block Bootstrap: Resamples blocks of consecutive observations\n"
            "Stationary Bootstrap: Uses random block lengths drawn from a geometric distribution"
        )
        
        method_layout.addWidget(method_label)
        method_layout.addWidget(self._method_combo)
        param_layout.addLayout(method_layout)
        
        # Parameter inputs
        param_inputs_layout = QHBoxLayout()
        
        # Block size input
        block_size_layout = QVBoxLayout()
        block_size_label = QLabel("Block Size:")
        self._block_size_spin = QSpinBox()
        self._block_size_spin.setRange(1, 1000)
        self._block_size_spin.setValue(self._bootstrap_params["block_size"])
        self._block_size_spin.setToolTip(
            "Size of consecutive blocks to resample\n"
            "Larger blocks preserve more of the dependence structure"
        )
        block_size_layout.addWidget(block_size_label)
        block_size_layout.addWidget(self._block_size_spin)
        param_inputs_layout.addLayout(block_size_layout)
        
        # Probability input (for stationary bootstrap)
        probability_layout = QVBoxLayout()
        probability_label = QLabel("Probability:")
        self._probability_spin = QDoubleSpinBox()
        self._probability_spin.setRange(0.01, 0.99)
        self._probability_spin.setSingleStep(0.05)
        self._probability_spin.setDecimals(2)
        self._probability_spin.setValue(self._bootstrap_params["probability"])
        self._probability_spin.setToolTip(
            "Probability parameter for the geometric distribution\n"
            "Determines the expected block length (1/p)"
        )
        probability_layout.addWidget(probability_label)
        probability_layout.addWidget(self._probability_spin)
        param_inputs_layout.addLayout(probability_layout)
        
        # Number of bootstrap samples input
        num_bootstrap_layout = QVBoxLayout()
        num_bootstrap_label = QLabel("Number of Samples:")
        self._num_bootstrap_spin = QSpinBox()
        self._num_bootstrap_spin.setRange(100, 10000)
        self._num_bootstrap_spin.setSingleStep(100)
        self._num_bootstrap_spin.setValue(self._bootstrap_params["num_bootstrap"])
        self._num_bootstrap_spin.setToolTip(
            "Number of bootstrap samples to generate\n"
            "More samples provide more accurate results but take longer to compute"
        )
        num_bootstrap_layout.addWidget(num_bootstrap_label)
        num_bootstrap_layout.addWidget(self._num_bootstrap_spin)
        param_inputs_layout.addLayout(num_bootstrap_layout)
        
        param_layout.addLayout(param_inputs_layout)
        param_group.setLayout(param_layout)
        main_layout.addWidget(param_group)
        
        # ========== Action Buttons ==========
        buttons_layout = QHBoxLayout()
        
        self._run_button = QPushButton("Run Bootstrap")
        self._run_button.setToolTip("Start bootstrap analysis with current parameters")
        
        self._stop_button = QPushButton("Cancel")
        self._stop_button.setToolTip("Cancel the current bootstrap analysis")
        
        self._reset_button = QPushButton("Reset")
        self._reset_button.setToolTip("Reset all parameters and clear results")
        
        # Disable stop button initially
        self._stop_button.setEnabled(False)
        
        buttons_layout.addWidget(self._run_button)
        buttons_layout.addWidget(self._stop_button)
        buttons_layout.addWidget(self._reset_button)
        buttons_layout.addStretch()
        
        main_layout.addLayout(buttons_layout)
        
        # ========== Progress Indicator ==========
        self._progress_indicator = ProgressIndicator(self, show_percentage=True, show_cancel_button=True)
        main_layout.addWidget(self._progress_indicator)
        
        # ========== Results Visualization ==========
        results_group = QGroupBox("Bootstrap Results")
        results_layout = QVBoxLayout()
        results_layout.setContentsMargins(10, 15, 10, 10)
        results_layout.setSpacing(10)
        
        # Parameter table for bootstrap stats
        self._parameter_table = ParameterTable()
        results_layout.addWidget(self._parameter_table)
        
        # Time series plot for bootstrap visualization
        self._time_series_plot = TimeSeriesPlot()
        results_layout.addWidget(self._time_series_plot)
        
        # Statistical metrics display
        self._statistical_metrics = StatisticalMetrics()
        results_layout.addWidget(self._statistical_metrics)
        
        results_group.setLayout(results_layout)
        main_layout.addWidget(results_group)
        
        # Set the layout
        self.setLayout(main_layout)
        
        # Update UI based on initial method
        self._update_method_inputs(self._bootstrap_params["method"])
    
    def _connect_signals(self):
        """Connects signals and slots for the bootstrap view."""
        # Connect method combo box
        self._method_combo.currentIndexChanged.connect(
            lambda idx: self._update_method_inputs("block" if idx == 0 else "stationary")
        )
        
        # Connect parameter inputs
        self._block_size_spin.valueChanged.connect(self.update_parameters)
        self._probability_spin.valueChanged.connect(self.update_parameters)
        self._num_bootstrap_spin.valueChanged.connect(self.update_parameters)
        
        # Connect action buttons
        self._run_button.clicked.connect(self.start_bootstrap)
        self._stop_button.clicked.connect(self.cancel_bootstrap)
        self._reset_button.clicked.connect(self.reset_view)
        
        # Connect progress indicator to cancel signal
        self._progress_indicator.cancel_requested.connect(self.cancel_bootstrap)
        
        # Connect bootstrap view signals
        self._signals.bootstrap_started.connect(
            lambda: self._progress_indicator.start_progress("Running Bootstrap Analysis")
        )
        self._signals.bootstrap_completed.connect(
            lambda: self._progress_indicator.complete()
        )
        self._signals.bootstrap_error.connect(
            lambda err: self._progress_indicator.complete(message=f"Error: {str(err)}")
        )
        
        # Connect task manager signals
        self._task_manager.signals.task_completed.connect(
            lambda task_id, result: self.handle_results(result)
        )
        self._task_manager.signals.task_failed.connect(
            lambda task_id, error: self.handle_error(error)
        )
    
    def set_data(self, data: np.ndarray):
        """
        Sets the data to be used for bootstrap analysis.
        
        Parameters
        ----------
        data : np.ndarray
            Time series data for bootstrap analysis
        """
        try:
            # Validate data
            if data is None:
                logger.warning("Cannot set empty data for bootstrap analysis")
                self._run_button.setEnabled(False)
                return
                
            if not isinstance(data, np.ndarray):
                data = np.asarray(data)
            
            # Ensure 1D array
            if data.ndim > 1:
                if data.shape[1] == 1:  # Column vector
                    data = data.flatten()
                else:
                    raise ValueError("Bootstrap data must be 1-dimensional")
            
            # Validate minimum length
            if len(data) < 5:  # Arbitrary minimum length
                raise ValueError("Data must have at least 5 observations for bootstrap")
            
            # Store data
            self._data = data
            
            # Update UI to reflect data availability
            self._run_button.setEnabled(True)
            
            logger.debug(f"Bootstrap data set with shape: {data.shape}")
        except Exception as e:
            logger.error(f"Error setting bootstrap data: {str(e)}")
            self._run_button.setEnabled(False)
    
    def update_parameters(self):
        """Updates the bootstrap parameters based on user input."""
        # Get current method
        method_idx = self._method_combo.currentIndex()
        method = "block" if method_idx == 0 else "stationary"
        
        # Get parameter values
        block_size = self._block_size_spin.value()
        probability = self._probability_spin.value()
        num_bootstrap = self._num_bootstrap_spin.value()
        
        # Update parameter dictionary
        self._bootstrap_params["method"] = method
        self._bootstrap_params["block_size"] = block_size
        self._bootstrap_params["probability"] = probability
        self._bootstrap_params["num_bootstrap"] = num_bootstrap
        
        # Emit signal that parameters were updated
        self._signals.parameters_updated.emit()
        
        # Update parameter table
        param_data = {
            "Method": {"estimate": method.capitalize()},
            "Block Size": {"estimate": block_size} if method == "block" else None,
            "Probability": {"estimate": probability} if method == "stationary" else None,
            "Number of Samples": {"estimate": num_bootstrap}
        }
        
        # Filter out None values
        param_data = {k: v for k, v in param_data.items() if v is not None}
        
        # Update parameter table
        self._parameter_table.set_parameter_data(param_data)
        
        logger.debug(f"Bootstrap parameters updated: {self._bootstrap_params}")
    
    def start_bootstrap(self):
        """Starts the bootstrap analysis with current parameters."""
        if self._data is None or len(self._data) == 0:
            logger.warning("Cannot start bootstrap: No data available")
            return
        
        if self._is_running:
            logger.warning("Bootstrap analysis already running")
            return
        
        # Set running state
        self._is_running = True
        self._signals.bootstrap_started.emit()
        
        # Update UI for running state
        self._run_button.setEnabled(False)
        self._stop_button.setEnabled(True)
        
        # Create and start task
        bootstrap_task_id = self._task_manager.create_task(
            "Bootstrap Analysis",
            self._run_bootstrap_task
        )
        
        logger.debug(f"Started bootstrap analysis with method: {self._bootstrap_params['method']}")
    
    def cancel_bootstrap(self):
        """Cancels any running bootstrap analysis."""
        if not self._is_running:
            return
        
        # Cancel all tasks
        self._task_manager.cancel_all()
        
        # Reset running state
        self._is_running = False
        
        # Update UI
        self._run_button.setEnabled(True)
        self._stop_button.setEnabled(False)
        
        logger.debug("Bootstrap analysis cancelled")
    
    def reset_view(self):
        """Resets the view to its initial state."""
        # Cancel any running task
        self.cancel_bootstrap()
        
        # Clear results
        self._bootstrap_results = None
        
        # Reset parameters to defaults
        self._bootstrap_params = {
            "method": "block",
            "block_size": 20,
            "probability": 0.1,
            "num_bootstrap": 1000
        }
        
        # Reset UI components
        self._method_combo.setCurrentIndex(0)
        self._block_size_spin.setValue(self._bootstrap_params["block_size"])
        self._probability_spin.setValue(self._bootstrap_params["probability"])
        self._num_bootstrap_spin.setValue(self._bootstrap_params["num_bootstrap"])
        
        # Clear results displays
        self._parameter_table.clear()
        self._time_series_plot.clear()
        self._statistical_metrics.clear()
        self._progress_indicator.reset()
        
        # Update parameters
        self.update_parameters()
        
        logger.debug("Bootstrap view reset to initial state")
    
    async def _run_bootstrap_task(self):
        """
        Asynchronous method to run the bootstrap analysis.
        
        Returns
        -------
        dict
            Bootstrap results including resampled data and statistics
        """
        try:
            # Get bootstrap parameters
            method = self._bootstrap_params["method"]
            num_bootstrap = self._bootstrap_params["num_bootstrap"]
            
            # Define a simple statistic function (e.g., mean)
            def statistic_func(x):
                return np.mean(x)
            
            # Run appropriate bootstrap method
            if method == "block":
                block_size = self._bootstrap_params["block_size"]
                
                # Run async block bootstrap
                bootstrap_stats = await block_bootstrap_async(
                    self._data,
                    statistic_func,
                    block_size,
                    num_bootstrap,
                    progress_callback=self._handle_progress
                )
            else:  # Stationary bootstrap
                probability = self._bootstrap_params["probability"]
                
                # Run async stationary bootstrap
                bootstrap_stats = await stationary_bootstrap_async(
                    self._data,
                    statistic_func,
                    probability,
                    num_bootstrap,
                    progress_callback=self._handle_progress
                )
            
            # Calculate original statistic
            original_stat = statistic_func(self._data)
            
            # Calculate confidence intervals
            ci_95 = calculate_bootstrap_ci(
                bootstrap_stats, 
                original_stat, 
                method="percentile", 
                alpha=0.05
            )
            
            ci_99 = calculate_bootstrap_ci(
                bootstrap_stats, 
                original_stat, 
                method="percentile", 
                alpha=0.01
            )
            
            # Create results dictionary
            results = {
                "bootstrap_stats": bootstrap_stats,
                "original_stat": original_stat,
                "ci_95": ci_95,
                "ci_99": ci_99,
                "method": method,
                "num_bootstrap": num_bootstrap,
                "block_size": self._bootstrap_params.get("block_size") if method == "block" else None,
                "probability": self._bootstrap_params.get("probability") if method == "stationary" else None,
                "standard_error": np.std(bootstrap_stats, ddof=1),
                "data_length": len(self._data)
            }
            
            return results
        
        except Exception as e:
            logger.error(f"Error in bootstrap analysis: {str(e)}")
            self._signals.bootstrap_error.emit(e)
            raise
    
    def _handle_progress(self, progress: float):
        """
        Handles progress updates from the bootstrap computation.
        
        Parameters
        ----------
        progress : float
            Progress percentage (0-100)
        """
        # Update progress indicator (normalize to 0-1 range)
        self._progress_indicator.update_progress(progress / 100)
    
    def handle_results(self, results: Dict):
        """
        Handles the completed bootstrap results.
        
        Parameters
        ----------
        results : dict
            Dictionary containing bootstrap results
        """
        if not results or not self._is_running:
            return
        
        try:
            # Store results
            self._bootstrap_results = results
            
            # 1. Create a DataFrame for the time series plot
            bootstrap_stats = results["bootstrap_stats"]
            histogram_data = pd.DataFrame({"Bootstrap Distribution": bootstrap_stats})
            
            # Update time series plot with histogram
            self._time_series_plot.set_data(histogram_data)
            plot_params = {
                "title": f"Bootstrap Distribution ({results['method'].title()} Bootstrap)",
                "grid": True
            }
            self._time_series_plot.set_plot_params(plot_params)
            self._time_series_plot.update_plot()
            
            # 2. Update parameter table with bootstrap statistics
            param_data = {
                "Original Statistic": {"estimate": results["original_stat"]},
                "Bootstrap Mean": {"estimate": np.mean(bootstrap_stats)},
                "Bootstrap Median": {"estimate": np.median(bootstrap_stats)},
                "Bootstrap Std. Error": {"estimate": results["standard_error"]},
                "95% CI Lower": {"estimate": results["ci_95"][0]},
                "95% CI Upper": {"estimate": results["ci_95"][1]},
                "99% CI Lower": {"estimate": results["ci_99"][0]},
                "99% CI Upper": {"estimate": results["ci_99"][1]}
            }
            self._parameter_table.set_parameter_data(param_data)
            
            # 3. Update statistical metrics
            metrics_data = {
                "bootstrap_samples": results["num_bootstrap"],
                "method": results["method"].capitalize(),
                "original_stat": results["original_stat"],
                "bootstrap_mean": np.mean(bootstrap_stats),
                "bootstrap_median": np.median(bootstrap_stats),
                "bootstrap_std_error": results["standard_error"],
                "ci_95_lower": results["ci_95"][0],
                "ci_95_upper": results["ci_95"][1],
                "ci_99_lower": results["ci_99"][0],
                "ci_99_upper": results["ci_99"][1],
                "ci_95_width": results["ci_95"][1] - results["ci_95"][0],
                "ci_99_width": results["ci_99"][1] - results["ci_99"][0],
                "data_length": results["data_length"]
            }
            
            # Add method-specific metrics
            if results["method"] == "block":
                metrics_data["block_size"] = results["block_size"]
            else:  # stationary
                metrics_data["probability"] = results["probability"]
                metrics_data["expected_block_length"] = 1.0 / results["probability"]
            
            self._statistical_metrics.set_metrics_data(metrics_data)
            
            # Reset UI state
            self._is_running = False
            self._run_button.setEnabled(True)
            self._stop_button.setEnabled(False)
            
            # Emit completion signal
            self._signals.bootstrap_completed.emit()
            
            logger.debug("Bootstrap analysis completed and results processed")
            
        except Exception as e:
            logger.error(f"Error processing bootstrap results: {str(e)}")
            self.handle_error(e)
    
    def handle_error(self, error: Exception):
        """
        Handles errors during bootstrap computation.
        
        Parameters
        ----------
        error : Exception
            The error that occurred
        """
        # Log the error
        logger.error(f"Bootstrap error: {str(error)}")
        
        # Reset UI state
        self._is_running = False
        self._run_button.setEnabled(True)
        self._stop_button.setEnabled(False)
        
        # Emit error signal
        self._signals.bootstrap_error.emit(error)
    
    def _update_method_inputs(self, method: str):
        """
        Updates the available parameter inputs based on selected bootstrap method.
        
        Parameters
        ----------
        method : str
            The selected bootstrap method ('block' or 'stationary')
        """
        self._bootstrap_params["method"] = method
        
        # Update UI based on selected method
        if method == "block":
            # Show block size, hide probability
            self._block_size_spin.setEnabled(True)
            self._probability_spin.setEnabled(False)
        else:  # Stationary bootstrap
            # Show probability, hide block size
            self._block_size_spin.setEnabled(False)
            self._probability_spin.setEnabled(True)
            
        # Update parameters after method change
        self.update_parameters()
        
        logger.debug(f"Updated UI for bootstrap method: {method}")
    
    def get_results(self):
        """
        Returns the current bootstrap results.
        
        Returns
        -------
        dict
            Current bootstrap results or None if not computed
        """
        return self._bootstrap_results