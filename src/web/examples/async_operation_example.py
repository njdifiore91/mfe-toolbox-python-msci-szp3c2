"""
MFE Toolbox - Asynchronous Operations Example

This example demonstrates how to perform computationally intensive financial
operations in background threads while maintaining UI responsiveness using
PyQt6's signal system and Python's async/await patterns.

The example includes:
1. Basic worker thread approach with Worker
2. Asynchronous function execution with AsyncWorker
3. Advanced task management with TaskManager
4. Real-time progress reporting and UI updates
5. Comprehensive error handling
"""

import sys
import os
import time
import random
import numpy as np
import asyncio
import logging
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QProgressBar, QComboBox, QMessageBox
)
from PyQt6.QtCore import Qt, QThreadPool

# Import Worker and AsyncWorker for background thread execution
from mfe.ui.async.worker import Worker, AsyncWorker

# Import TaskManager for advanced task management
from mfe.ui.async.task_manager import TaskManager, TaskPriority, TaskType

# Import async helpers
from mfe.utils.async_helpers import async_progress, handle_exceptions_async, run_in_executor

# Import GARCH model for demonstration
from mfe.models.garch import GARCH

# Configure logger
logger = logging.getLogger(__name__)


def simulate_time_series(length: int, with_progress: bool = False, progress_callback: callable = None) -> np.ndarray:
    """
    Simulates a financial time series with GARCH volatility characteristics.
    
    Parameters
    ----------
    length : int
        Length of the simulated time series
    with_progress : bool, default=False
        Whether to report progress during simulation
    progress_callback : callable, default=None
        Callback function for progress reporting
        
    Returns
    -------
    numpy.ndarray
        Simulated time series data with GARCH-like properties
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Initialize returns array
    returns = np.zeros(length)
    
    # Initial volatility
    volatility = 0.01
    
    # GARCH parameters
    omega = 0.00001
    alpha = 0.1
    beta = 0.85
    
    # Generate returns with GARCH process
    for t in range(length):
        # Calculate volatility using GARCH(1,1) process
        if t > 0:
            volatility = omega + alpha * returns[t-1]**2 + beta * volatility
        
        # Generate return with current volatility
        returns[t] = np.random.normal(0, np.sqrt(volatility))
        
        # Report progress if requested
        if with_progress and progress_callback and t % max(1, length // 100) == 0:
            progress = (t + 1) / length
            progress_callback(progress)
            time.sleep(0.01)  # Add small delay to simulate longer computation
    
    return returns


def cpu_intensive_calculation(iterations: int, complexity: float, progress_callback: callable = None) -> dict:
    """
    Simulates a CPU intensive calculation that would block the UI if run in the main thread.
    
    Parameters
    ----------
    iterations : int
        Number of calculation iterations
    complexity : float
        Complexity factor to adjust computation intensity
    progress_callback : callable, default=None
        Callback function for progress reporting
        
    Returns
    -------
    dict
        Results of the calculation with execution metrics
    """
    # Record start time
    start_time = time.time()
    
    # Initialize result
    result = 0
    
    # Perform calculations
    for i in range(iterations):
        # Simulate complex calculation based on complexity factor
        for j in range(int(complexity * 1000)):
            result += (np.sin(i * j / 1000) ** 2 + np.cos(i * j / 1000) ** 2)
        
        # Report progress periodically
        if progress_callback and i % max(1, iterations // 100) == 0:
            progress = (i + 1) / iterations
            progress_callback(progress)
        
        # Small sleep to simulate I/O operations
        time.sleep(0.001)
    
    # Calculate execution time
    execution_time = time.time() - start_time
    
    # Return results and metrics
    return {
        'result': result,
        'iterations': iterations,
        'complexity': complexity,
        'execution_time': execution_time
    }


@async_progress
@handle_exceptions_async
async def async_data_processing(data: np.ndarray) -> dict:
    """
    Asynchronous function that processes data with progress reporting.
    
    Parameters
    ----------
    data : numpy.ndarray
        Input data array
        
    Returns
    -------
    dict
        Processed data results with statistics
    """
    # Validate input
    if not isinstance(data, np.ndarray):
        raise ValueError("Input must be a numpy array")
    
    # Get progress_callback from context (injected by async_progress decorator)
    progress_callback = async_data_processing.report_progress
    
    # Step 1: Data validation and normalization
    await asyncio.sleep(0.2)  # Simulate I/O operations
    if progress_callback:
        progress_callback(0.1)
    
    # Step 2: Calculate moving averages and other indicators
    await asyncio.sleep(0.3)
    ma_5 = np.convolve(data, np.ones(5)/5, mode='valid')
    ma_20 = np.convolve(data, np.ones(20)/20, mode='valid')
    if progress_callback:
        progress_callback(0.3)
    
    # Step 3: Pattern detection
    await asyncio.sleep(0.3)
    diff = np.diff(data)
    if progress_callback:
        progress_callback(0.5)
    
    # Step 4: Statistical calculations
    mean = np.mean(data)
    variance = np.var(data)
    skewness = np.mean(((data - mean) / np.sqrt(variance)) ** 3)
    kurtosis = np.mean(((data - mean) / np.sqrt(variance)) ** 4) - 3
    if progress_callback:
        progress_callback(0.7)
    
    # Step 5: Anomaly detection
    await asyncio.sleep(0.2)
    threshold = mean + 2 * np.sqrt(variance)
    anomalies = data > threshold
    if progress_callback:
        progress_callback(0.9)
    
    # Compile results
    results = {
        'mean': mean,
        'variance': variance,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'anomalies_count': np.sum(anomalies),
        'max_value': np.max(data),
        'min_value': np.min(data),
        'data_length': len(data)
    }
    
    # Complete
    if progress_callback:
        progress_callback(1.0)
    
    return results


@async_progress
@handle_exceptions_async
async def estimate_garch_model(returns: np.ndarray) -> dict:
    """
    Asynchronous function to estimate a GARCH model on time series data.
    
    Parameters
    ----------
    returns : numpy.ndarray
        Return series data
        
    Returns
    -------
    dict
        GARCH model estimation results
    """
    # Validate input
    if not isinstance(returns, np.ndarray):
        raise ValueError("Returns must be a numpy array")
    
    # Get progress_callback from context (injected by async_progress decorator)
    progress_callback = estimate_garch_model.report_progress
    
    # Report initial progress
    if progress_callback:
        progress_callback(0.0)
    
    # Create GARCH(1,1) model
    model = GARCH(p=1, q=1)
    if progress_callback:
        progress_callback(0.1)
    
    # Convert the synchronous fit method to async using run_in_executor
    fit_result = await run_in_executor(model.fit, returns)
    if progress_callback:
        progress_callback(0.8)
    
    # Extract model parameters, standard errors, and goodness-of-fit statistics
    results = {
        'omega': model.parameters[0],
        'alpha': model.parameters[1],
        'beta': model.parameters[2],
        'log_likelihood': fit_result.get('log_likelihood', None),
        'aic': fit_result.get('aic', None),
        'bic': fit_result.get('bic', None),
        'persistence': model.parameters[1] + model.parameters[2],
        'unconditional_variance': model.get_unconditional_variance() if hasattr(model, 'get_unconditional_variance') else None,
        'half_life': model.half_life() if hasattr(model, 'half_life') else None
    }
    
    # Calculate volatility forecast
    forecast_horizon = 10
    volatility_forecast = model.forecast(forecast_horizon) if hasattr(model, 'forecast') else None
    results['forecast'] = volatility_forecast
    
    if progress_callback:
        progress_callback(1.0)
    
    return results


def configure_logging():
    """
    Configures the basic logging setup for the example.
    """
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        level=logging.INFO
    )
    logger.info("Async operations example application starting")


class AsyncOperationsExample(QMainWindow):
    """
    Main window demonstrating various approaches to asynchronous operations in PyQt6-based UI.
    """
    
    def __init__(self):
        """Initialize the async operations example window with UI components."""
        super().__init__()
        
        # Window configuration
        self.setWindowTitle("MFE Toolbox - Async Operations Example")
        self.setMinimumSize(800, 600)
        
        # Initialize state
        self.current_worker = None
        self.current_async_worker = None
        self.current_task_id = None
        
        # Get thread pool
        self.thread_pool = QThreadPool.globalInstance()
        
        # Create task manager
        self.task_manager = TaskManager()
        
        # Set up UI components
        self.setup_ui()
        
        # Connect signals
        self.start_button.clicked.connect(self.on_start_clicked)
        self.cancel_button.clicked.connect(self.on_cancel_clicked)
        
        # Connect task manager signals
        self.task_manager.signals.task_started.connect(self.on_task_started)
        self.task_manager.signals.task_completed.connect(self.on_task_completed)
        self.task_manager.signals.task_progress.connect(self.on_task_progress)
        self.task_manager.signals.task_failed.connect(self.on_task_failed)
        self.task_manager.signals.task_cancelled.connect(self.on_task_cancelled)
        
        # Initialize UI state
        self.reset_ui_state()
        
        logger.info("AsyncOperationsExample application initialized")
    
    def setup_ui(self):
        """Sets up the UI components for the example window."""
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)
        
        # Header section
        header_layout = QVBoxLayout()
        title_label = QLabel("MFE Toolbox - Asynchronous Operations Example")
        title_label.setStyleSheet("font-size: 18pt; font-weight: bold;")
        description_label = QLabel(
            "This example demonstrates different approaches to maintaining UI responsiveness\n"
            "while performing computationally intensive financial operations in the background."
        )
        description_label.setStyleSheet("font-size: 11pt;")
        header_layout.addWidget(title_label)
        header_layout.addWidget(description_label)
        main_layout.addLayout(header_layout)
        
        # Operation selection
        operation_layout = QVBoxLayout()
        operation_label = QLabel("Select Operation:")
        self.operation_combo = QComboBox()
        self.operation_combo.addItem("Worker: Simulate Time Series", "time_series_worker")
        self.operation_combo.addItem("Worker: CPU Intensive Calculation", "cpu_intensive_worker")
        self.operation_combo.addItem("AsyncWorker: Data Processing", "data_processing_async")
        self.operation_combo.addItem("AsyncWorker: GARCH Estimation", "garch_estimation_async")
        self.operation_combo.addItem("TaskManager: Time Series + GARCH", "task_manager_combined")
        operation_layout.addWidget(operation_label)
        operation_layout.addWidget(self.operation_combo)
        main_layout.addLayout(operation_layout)
        
        # Control buttons
        buttons_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Operation")
        self.start_button.setMinimumWidth(150)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setMinimumWidth(150)
        buttons_layout.addWidget(self.start_button)
        buttons_layout.addWidget(self.cancel_button)
        buttons_layout.addStretch()
        main_layout.addLayout(buttons_layout)
        
        # Progress section
        progress_layout = QVBoxLayout()
        progress_label = QLabel("Progress:")
        self.progress_bar = QProgressBar()
        self.status_label = QLabel("Ready")
        progress_layout.addWidget(progress_label)
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.status_label)
        main_layout.addLayout(progress_layout)
        
        # Results section
        results_layout = QVBoxLayout()
        results_label = QLabel("Results:")
        self.result_label = QLabel("")
        self.result_label.setStyleSheet("background-color: #f5f5f5; padding: 10px; border-radius: 5px;")
        self.result_label.setWordWrap(True)
        self.result_label.setMinimumHeight(200)
        results_layout.addWidget(results_label)
        results_layout.addWidget(self.result_label)
        main_layout.addLayout(results_layout)
        
        # Add stretch to push content to the top
        main_layout.addStretch()
    
    def on_start_clicked(self):
        """Handler for the start button click event."""
        # Get selected operation
        operation = self.operation_combo.currentData()
        
        # Reset UI
        self.reset_ui_state()
        self.result_label.setText("")
        self.status_label.setText("Initializing...")
        
        # Determine appropriate async approach
        if operation.endswith("_worker"):
            # Use Worker for synchronous functions
            self.use_worker_approach(operation)
        elif operation.endswith("_async"):
            # Use AsyncWorker for asynchronous functions
            self.use_async_worker_approach(operation)
        elif operation.startswith("task_manager"):
            # Use TaskManager for advanced task orchestration
            self.use_task_manager_approach(operation)
        else:
            self.status_label.setText("Unknown operation type")
            return
        
        # Update UI to show operation in progress
        self.status_label.setText("Operation started...")
        self.cancel_button.setEnabled(True)
        self.start_button.setEnabled(False)
        
        logger.info(f"Started operation: {operation}")
    
    def on_cancel_clicked(self):
        """Handler for the cancel button click event."""
        if self.current_worker:
            # Cancel Worker
            self.current_worker.cancel()
        elif self.current_async_worker:
            # Cancel AsyncWorker
            self.current_async_worker.cancel()
        elif self.current_task_id:
            # Cancel TaskManager task
            self.task_manager.cancel_task(self.current_task_id)
        
        self.status_label.setText("Operation cancelled")
        logger.info("Operation cancelled by user")
    
    def use_worker_approach(self, operation: str):
        """Executes operation using the Worker class for synchronous functions."""
        # Determine the function to execute based on operation name
        if operation == "time_series_worker":
            function = simulate_time_series
            args = (5000, True)  # length, with_progress
        elif operation == "cpu_intensive_worker":
            function = cpu_intensive_calculation
            args = (1000, 0.5)  # iterations, complexity
        else:
            self.status_label.setText(f"Unknown operation: {operation}")
            return
        
        # Create Worker instance with the function and parameters
        self.current_worker = Worker(function, *args)
        
        # Connect worker signals to handler methods
        self.current_worker.signals.started.connect(self.on_worker_started)
        self.current_worker.signals.result.connect(self.on_worker_result)
        self.current_worker.signals.progress.connect(self.on_worker_progress)
        self.current_worker.signals.error.connect(self.on_worker_error)
        self.current_worker.signals.finished.connect(self.on_worker_finished)
        self.current_worker.signals.cancelled.connect(self.on_worker_cancelled)
        
        # Start worker in the thread pool
        self.thread_pool.start(self.current_worker)
        logger.info(f"Started worker approach for operation: {operation}")
    
    def use_async_worker_approach(self, operation: str):
        """Executes operation using the AsyncWorker class for asynchronous coroutines."""
        # Determine the coroutine function to execute based on operation name
        if operation == "data_processing_async":
            function = async_data_processing
            # Generate sample data
            data = np.random.normal(0, 1, 1000)
            args = (data,)
        elif operation == "garch_estimation_async":
            function = estimate_garch_model
            # Generate sample returns
            returns = simulate_time_series(2000)
            args = (returns,)
        else:
            self.status_label.setText(f"Unknown operation: {operation}")
            return
        
        # Create AsyncWorker instance with the coroutine function and parameters
        self.current_async_worker = AsyncWorker(function, *args)
        
        # Connect worker signals to handler methods
        self.current_async_worker.signals.started.connect(self.on_worker_started)
        self.current_async_worker.signals.result.connect(self.on_worker_result)
        self.current_async_worker.signals.progress.connect(self.on_worker_progress)
        self.current_async_worker.signals.error.connect(self.on_worker_error)
        self.current_async_worker.signals.finished.connect(self.on_worker_finished)
        self.current_async_worker.signals.cancelled.connect(self.on_worker_cancelled)
        
        # Start worker in the thread pool
        self.thread_pool.start(self.current_async_worker)
        logger.info(f"Started async worker approach for operation: {operation}")
    
    def use_task_manager_approach(self, operation: str):
        """Executes operation using the TaskManager for advanced task orchestration."""
        # Determine the function to execute and task type based on operation name
        if operation == "task_manager_combined":
            # Create a task group for the combined analysis
            group_id = self.task_manager.create_task_group("Combined Analysis")
            
            # Task 1: Generate time series (high priority)
            data_task_id = self.task_manager.create_task(
                name="Generate Time Series",
                function=simulate_time_series,
                args=(3000, True),
                priority=TaskPriority.HIGH,
                task_type=TaskType.DATA_PROCESSING,
                group_id=group_id
            )
            
            # Task 2: Estimate GARCH model (depends on Task 1)
            self.current_task_id = self.task_manager.create_task(
                name="Estimate GARCH Model",
                function=estimate_garch_model,
                priority=TaskPriority.NORMAL,
                task_type=TaskType.MODEL,
                dependencies=[data_task_id],
                group_id=group_id
            )
            
            logger.info(f"Started task manager approach for operation: {operation}")
        else:
            self.status_label.setText(f"Unknown operation: {operation}")
    
    def on_worker_started(self):
        """Signal handler for worker started event."""
        self.status_label.setText("Operation in progress...")
        self.progress_bar.setValue(0)
        logger.debug("Worker started")
    
    def on_worker_result(self, result):
        """Signal handler for worker result event."""
        # Format result data as string for display
        if isinstance(result, np.ndarray):
            # Format array result
            result_str = (
                f"Array Result Summary:\n"
                f"Shape: {result.shape}\n"
                f"Mean: {np.mean(result):.6f}\n"
                f"Std Dev: {np.std(result):.6f}\n"
                f"Min: {np.min(result):.6f}\n"
                f"Max: {np.max(result):.6f}"
            )
        elif isinstance(result, dict):
            # Format dictionary result
            result_str = "Result:\n"
            for key, value in result.items():
                if isinstance(value, np.ndarray):
                    result_str += f"{key}: [Array of shape {value.shape}]\n"
                elif isinstance(value, (float, np.float64, np.float32)):
                    result_str += f"{key}: {value:.6f}\n"
                else:
                    result_str += f"{key}: {value}\n"
        else:
            # Generic string representation
            result_str = str(result)
        
        # Update result_label with formatted result
        self.result_label.setText(result_str)
        logger.debug("Worker result received")
    
    def on_worker_progress(self, progress):
        """Signal handler for worker progress event."""
        progress_percent = int(progress * 100)
        self.progress_bar.setValue(progress_percent)
        self.status_label.setText(f"Progress: {progress_percent}%")
        
        # Log progress update if significant change
        if progress_percent % 10 == 0:
            logger.debug(f"Worker progress: {progress_percent}%")
    
    def on_worker_error(self, error):
        """Signal handler for worker error event."""
        error_msg = str(error)
        logger.error(f"Worker error: {error_msg}")
        
        # Show error message box with details about the exception
        QMessageBox.critical(
            self,
            "Operation Error",
            f"An error occurred during operation:\n\n{error_msg}"
        )
        
        # Update status label to indicate error
        self.status_label.setText(f"Error: {error_msg}")
        
        # Reset UI state to allow new operations
        self.reset_ui_state()
        self.start_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
    
    def on_worker_finished(self):
        """Signal handler for worker finished event."""
        # Reset current_worker or current_async_worker to None
        if self.current_worker:
            self.current_worker = None
        elif self.current_async_worker:
            self.current_async_worker = None
        
        # Update UI state to allow new operations
        self.start_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        
        # Set status to completed if not already set
        if not self.status_label.text().startswith(("Error", "Operation cancelled")):
            self.status_label.setText("Operation completed")
        
        logger.debug("Worker completed")
    
    def on_worker_cancelled(self):
        """Signal handler for worker cancelled event."""
        # Update status label to indicate cancellation
        self.status_label.setText("Operation cancelled")
        
        # Reset UI state to allow new operations
        self.reset_ui_state()
        self.start_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        
        logger.debug("Worker cancelled")
    
    def on_task_started(self, task_id):
        """Signal handler for task manager task started event."""
        # Verify task_id matches current_task_id
        if task_id == self.current_task_id:
            self.status_label.setText("Task in progress...")
            self.progress_bar.setValue(0)
            logger.debug(f"Task started: {task_id}")
    
    def on_task_completed(self, task_id, result):
        """Signal handler for task manager task completed event."""
        # Verify task_id matches current_task_id
        if task_id == self.current_task_id:
            # Format result data as string for display
            self.on_worker_result(result)
            
            # Set progress bar to 100%
            self.progress_bar.setValue(100)
            
            # Update UI state to allow new operations
            self.start_button.setEnabled(True)
            self.cancel_button.setEnabled(False)
            
            # Set status to completed
            self.status_label.setText("Task completed")
            
            # Clear task ID
            self.current_task_id = None
            
            logger.debug(f"Task completed: {task_id}")
    
    def on_task_progress(self, task_id, progress):
        """Signal handler for task manager task progress event."""
        # Verify task_id matches current_task_id
        if task_id == self.current_task_id:
            progress_percent = int(progress * 100)
            self.progress_bar.setValue(progress_percent)
            self.status_label.setText(f"Progress: {progress_percent}%")
            
            # Log progress update if significant change
            if progress_percent % 10 == 0:
                logger.debug(f"Task progress: {task_id} = {progress_percent}%")
    
    def on_task_failed(self, task_id, error):
        """Signal handler for task manager task failed event."""
        # Verify task_id matches current_task_id
        if task_id == self.current_task_id:
            error_msg = str(error)
            logger.error(f"Task error: {task_id} - {error_msg}")
            
            # Show error message box with details about the exception
            QMessageBox.critical(
                self,
                "Task Error",
                f"An error occurred during task execution:\n\n{error_msg}"
            )
            
            # Update status label to indicate error
            self.status_label.setText(f"Error: {error_msg}")
            
            # Reset UI state to allow new operations
            self.reset_ui_state()
            self.start_button.setEnabled(True)
            self.cancel_button.setEnabled(False)
            
            # Clear task ID
            self.current_task_id = None
    
    def on_task_cancelled(self, task_id):
        """Signal handler for task manager task cancelled event."""
        # Verify task_id matches current_task_id
        if task_id == self.current_task_id:
            # Update status label to indicate cancellation
            self.status_label.setText("Task cancelled")
            
            # Reset UI state to allow new operations
            self.reset_ui_state()
            self.start_button.setEnabled(True)
            self.cancel_button.setEnabled(False)
            
            # Reset current_task_id to None
            self.current_task_id = None
            
            logger.debug(f"Task cancelled: {task_id}")
    
    def reset_ui_state(self):
        """Resets the UI components to their initial state."""
        # Reset progress bar to 0%
        self.progress_bar.setValue(0)
        
        # Clear status label
        self.status_label.setText("")
        
        # Clear result label
        self.result_label.setText("")
        
        # Enable start button
        self.start_button.setEnabled(True)
        
        # Disable cancel button
        self.cancel_button.setEnabled(False)
    
    def closeEvent(self, event):
        """Handler for window close events."""
        # Cancel any ongoing operations
        if self.current_worker:
            self.current_worker.cancel()
        elif self.current_async_worker:
            self.current_async_worker.cancel()
        elif self.current_task_id:
            self.task_manager.cancel_task(self.current_task_id)
        
        # Accept the close event
        event.accept()
        
        logger.info("AsyncOperationsExample application shutdown")


def main():
    """Main entry point to run the async operations example application."""
    # Configure logging setup
    configure_logging()
    
    # Create QApplication instance
    app = QApplication(sys.argv)
    
    # Create and show the AsyncOperationsExample window
    window = AsyncOperationsExample()
    window.show()
    
    # Execute the application main loop
    return app.exec()


if __name__ == "__main__":
    main()