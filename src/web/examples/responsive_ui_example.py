import asyncio
import logging
import sys
import time

import numpy as np  # version 1.26.3

from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot  # PyQt6 version: 6.6.1
from PyQt6.QtWidgets import (QApplication, QLabel, QMainWindow, QMessageBox,
                             QPushButton, QVBoxLayout, QWidget, QHBoxLayout, QSizePolicy)  # PyQt6 version: 6.6.1

from ..mfe.ui.async.task_manager import TaskManager, TaskPriority, TaskType  # internal
from ..mfe.ui.components.progress_indicator import ProgressIndicator, ProgressStyle  # internal
from ...backend.mfe.models.arma import ARMA  # internal
from ...backend.mfe.utils.async_helpers import async_progress, handle_exceptions_async  # internal

# Configure module logger
logger = logging.getLogger(__name__)


def generate_sample_data(size: int) -> np.ndarray:
    """Generates sample time series data for ARMA modeling"""
    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate AR parameters ensuring stationarity
    ar_params = np.array([0.6, 0.3])

    # Generate MA parameters ensuring invertibility
    ma_params = np.array([-0.4, 0.2])

    # Initialize data array
    data = np.zeros(size)

    # Add autoregressive components
    for t in range(2, size):
        data[t] += ar_params[0] * data[t - 1]
        data[t] += ar_params[1] * data[t - 2]

    # Add moving average components
    for t in range(2, size):
        data[t] += ma_params[0] * np.random.randn()
        data[t] += ma_params[1] * np.random.randn()

    # Add small random noise
    data += 0.1 * np.random.randn(size)

    # Return the generated time series
    return data


@async_progress
@handle_exceptions_async
async def intensive_computation(iterations: int) -> dict:
    """Asynchronous function that simulates an intensive computation"""
    # Get progress_callback from function context (injected by async_progress decorator)
    progress_callback = intensive_computation.__dict__['report_progress']

    # Record start time for performance measurement
    start_time = time.time()

    # Initialize result value
    result = 0

    # Loop through the specified number of iterations
    for i in range(iterations):
        # Perform complex mathematical operations to simulate CPU load
        result += np.sin(i) * np.cos(i) * np.arctan(i)

        # Report progress periodically via progress_callback
        if i % 1000 == 0:
            progress = i / iterations
            progress_callback(progress)

    # Calculate execution time
    execution_time = time.time() - start_time

    # Return dictionary with results and performance metrics
    return {"result": result, "execution_time": execution_time}


class ResponsiveUI(QMainWindow):
    """Main window class demonstrating responsive UI techniques in PyQt6"""

    def __init__(self):
        """Initializes the ResponsiveUI window and sets up UI components"""
        # Call parent constructor (QMainWindow.__init__)
        super().__init__()

        # Set window title and size
        self.setWindowTitle("MFE Toolbox - Responsive UI Example")
        self.resize(600, 400)

        # Initialize task manager for async operations
        self.task_manager = TaskManager()

        # Initialize current_task_id to None
        self.current_task_id = None

        # Create UI components with setup_ui method
        self.setup_ui()

        # Connect UI signals to slots
        self.task_manager.signals.task_started.connect(self.on_task_started)
        self.task_manager.signals.task_progress.connect(self.on_task_progress)
        self.task_manager.signals.task_completed.connect(self.on_task_completed)
        self.task_manager.signals.task_failed.connect(self.on_task_failed)
        self.task_manager.signals.task_cancelled.connect(self.on_task_cancelled)

        # Initialize ARMA model for time series modeling
        self.model = ARMA(p=2, q=2)

    def setup_ui(self):
        """Creates and arranges all UI components"""
        # Create central widget and main layout
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)

        # Create title label for the example
        title_label = QLabel("Responsive UI with Async Operations")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title_label)

        # Create descriptive text about responsive UI patterns
        description_label = QLabel(
            "This example demonstrates how to build responsive UIs in the MFE Toolbox using PyQt6 with async/await patterns. "
            "Click 'Start Computation' to begin a long-running task. "
            "The UI remains responsive, and progress is displayed using a progress indicator."
        )
        description_label.setWordWrap(True)
        main_layout.addWidget(description_label)

        # Create progress indicator component
        self.progress_indicator = ProgressIndicator(show_percentage=True, show_cancel_button=True)
        main_layout.addWidget(self.progress_indicator)

        # Create status label to show current operation
        self.status_label = QLabel("Ready.")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        main_layout.addWidget(self.status_label)

        # Create result label to display operation results
        self.result_label = QLabel("Results will be displayed here.")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        main_layout.addWidget(self.result_label)

        # Create start and cancel buttons
        button_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Computation")
        self.start_button.clicked.connect(self.on_start_clicked)
        button_layout.addWidget(self.start_button)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.on_cancel_clicked)
        self.cancel_button.setEnabled(False)  # Initially disabled
        button_layout.addWidget(self.cancel_button)

        main_layout.addLayout(button_layout)

        # Arrange components in layouts
        self.setCentralWidget(central_widget)

    @pyqtSlot()
    def on_start_clicked(self):
        """Slot for start button click, begins an async computation"""
        # Update UI to show computation is starting
        self.status_label.setText("Starting computation...")
        self.progress_indicator.start_progress("Running ARMA Estimation", ProgressStyle.INFO)

        # Disable start button to prevent multiple starts
        self.start_button.setEnabled(False)

        # Enable cancel button
        self.cancel_button.setEnabled(True)

        # Generate example data for ARMA model
        sample_data = generate_sample_data(size=1000)

        # Create a task for the async ARMA estimation
        self.current_task_id = self.task_manager.create_task(
            name="ARMA Estimation",
            function=self.run_arma_estimation,
            args=(sample_data,),
            priority=TaskPriority.HIGH,
            task_type=TaskType.MODEL
        )

        # Connect task signals to appropriate slots
        logger.info(f"Task {self.current_task_id} created and started.")

    @pyqtSlot()
    def on_cancel_clicked(self):
        """Slot for cancel button click, stops ongoing computation"""
        # If current_task_id exists, cancel that specific task
        if self.current_task_id:
            self.task_manager.cancel_task(self.current_task_id)
        # Otherwise, cancel all running tasks via the task manager
        else:
            self.task_manager.cancel_all_tasks()

        # Update UI to show operation was cancelled
        self.status_label.setText("Operation cancelled.")
        self.progress_indicator.complete(ProgressStyle.WARNING, "Cancelled")

        # Enable start button
        self.start_button.setEnabled(True)

        # Disable cancel button
        self.cancel_button.setEnabled(False)

        self.current_task_id = None

        # Log cancellation action
        logger.info("Computation cancelled by user.")

    @pyqtSlot(str)
    def on_task_started(self, task_id: str):
        """Slot for task started signal from TaskManager"""
        # Verify task_id matches current_task_id
        if task_id != self.current_task_id:
            return

        # Have progress indicator show task has started
        self.progress_indicator.start_progress("Running ARMA Estimation", ProgressStyle.INFO)

        # Update status label with operation information
        self.status_label.setText(f"Task {task_id}: ARMA estimation started...")

        # Log task started event
        logger.debug(f"Task {task_id} started.")

    @pyqtSlot(str, float)
    def on_task_progress(self, task_id: str, progress: float):
        """Slot for task progress signal from TaskManager"""
        # Verify task_id matches current_task_id
        if task_id != self.current_task_id:
            return

        # Update progress indicator with new progress value
        self.progress_indicator.update_progress(progress)

        # Update status label with progress percentage
        self.status_label.setText(f"Task {task_id}: ARMA estimation progress: {progress:.2f}%")

        # Log progress update if significant change
        if int(progress * 100) % 10 == 0:
            logger.debug(f"Task {task_id} progress: {progress:.2f}%")

    @pyqtSlot(str, object)
    def on_task_completed(self, task_id: str, result: object):
        """Slot for task completed signal from TaskManager"""
        # Verify task_id matches current_task_id
        if task_id != self.current_task_id:
            return

        # Set progress indicator to complete with success style
        self.progress_indicator.complete(ProgressStyle.SUCCESS)

        # Format result data as string for display
        formatted_result = self.format_result(result)

        # Update result label with formatted results
        self.result_label.setText(formatted_result)

        # Reset UI state for new operations
        self.reset_ui_state()

        # Enable start button
        self.start_button.setEnabled(True)

        # Disable cancel button
        self.cancel_button.setEnabled(False)

        # Log task completion with result summary
        logger.info(f"Task {task_id} completed with result: {result}")

    @pyqtSlot(str, Exception)
    def on_task_failed(self, task_id: str, error: Exception):
        """Slot for task failed signal from TaskManager"""
        # Verify task_id matches current_task_id
        if task_id != self.current_task_id:
            return

        # Set progress indicator to error state
        self.progress_indicator.complete(ProgressStyle.ERROR, f"Error: {str(error)}")

        # Display error message dialog with details
        QMessageBox.critical(self, "Task Failed", f"Task {task_id} failed: {str(error)}")

        # Update status label to indicate error
        self.status_label.setText(f"Task {task_id} failed: {str(error)}")

        # Reset UI state for new operations
        self.reset_ui_state()

        # Enable start button
        self.start_button.setEnabled(True)

        # Disable cancel button
        self.cancel_button.setEnabled(False)

        # Log error details
        logger.error(f"Task {task_id} failed with error: {str(error)}")

    @pyqtSlot(str)
    def on_task_cancelled(self, task_id: str):
        """Slot for task cancelled signal from TaskManager"""
        # Verify task_id matches current_task_id
        if task_id != self.current_task_id:
            return

        # Update progress indicator to show cancellation
        self.progress_indicator.complete(ProgressStyle.WARNING, "Cancelled")

        # Update status label to indicate cancellation
        self.status_label.setText(f"Task {task_id} cancelled.")

        # Reset UI state for new operations
        self.reset_ui_state()

        # Enable start button
        self.start_button.setEnabled(True)

        # Disable cancel button
        self.cancel_button.setEnabled(False)

        self.current_task_id = None

        # Log task cancellation
        logger.info(f"Task {task_id} cancelled by user.")

    def reset_ui_state(self):
        """Resets the UI components to their initial state"""
        # Reset progress indicator
        self.progress_indicator.reset()

        # Clear status label
        self.status_label.setText("Ready.")

        # Clear result label
        self.result_label.setText("Results will be displayed here.")

        # Enable start button
        self.start_button.setEnabled(True)

        # Disable cancel button
        self.cancel_button.setEnabled(False)

        self.current_task_id = None

    @async_progress
    @handle_exceptions_async
    async def run_arma_estimation(self, data: np.ndarray) -> dict:
        """Runs ARMA model estimation asynchronously"""
        # Get progress_callback from function context
        progress_callback = self.run_arma_estimation.__dict__['report_progress']

        # Create ARMA model with specified parameters
        model = ARMA(p=2, q=2)

        # Report initial progress (10%)
        progress_callback(0.1)

        # Perform model estimation using estimate_async method
        await asyncio.sleep(0)
        self.status_label.setText("Estimating ARMA model parameters...")
        await asyncio.sleep(0)
        estimation_result = await model.estimate_async(data)

        # Report progress to caller via progress_callback
        progress_callback(1.0)

        # Collect estimation results (parameters, standard errors, etc.)
        result = model.to_dict()

        # Report complete (100%)
        progress_callback(1.0)

        # Return estimation results dictionary
        return result

    def format_result(self, result: dict) -> str:
        """Formats model estimation results for display"""
        # Extract model parameters from result dictionary
        ar_params = result.get('ar_params', [])
        ma_params = result.get('ma_params', [])
        constant = result.get('constant', 0)
        loglikelihood = result.get('loglikelihood', 0)
        aic = result.get('aic', 0)
        bic = result.get('bic', 0)

        # Format AR parameters with precision
        ar_str = ", ".join([f"{param:.4f}" for param in ar_params])

        # Format MA parameters with precision
        ma_str = ", ".join([f"{param:.4f}" for param in ma_params])

        # Include log-likelihood and information criteria
        result_str = (
            f"ARMA(2,2) Model Estimation Results:\n"
            f"  AR Parameters: {ar_str}\n"
            f"  MA Parameters: {ma_str}\n"
            f"  Constant: {constant:.4f}\n"
            f"  Log-Likelihood: {loglikelihood:.4f}\n"
            f"  AIC: {aic:.4f}\n"
            f"  BIC: {bic:.4f}"
        )

        # Format standard errors and t-statistics
        if 'standard_errors' in result:
            std_errors = result['standard_errors']
            param_names = [f"AR{i+1}" for i in range(len(ar_params))] + \
                          [f"MA{i+1}" for i in range(len(ma_params))] + \
                          ["Constant"]
            stats_str = "\n".join([
                f"  {name}: StdErr={std:.4f}, t={val:.4f}"
                for name, std, val in zip(param_names, std_errors, [ar_params, ma_params, constant])
            ])
            result_str += f"\n\nParameter Statistics:\n{stats_str}"

        # Create a formatted multiline string with all results
        return result_str

    def closeEvent(self, event):
        """Handles the window close event"""
        # Cancel all tasks to ensure clean shutdown
        self.task_manager.cancel_all_tasks()

        # Accept the close event
        event.accept()

        # Log application shutdown
        logger.info("Application shutting down.")


def main() -> int:
    """Entry point for the application"""
    # Configure logging setup
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    # Create QApplication instance
    app = QApplication(sys.argv)

    # Create and show the ResponsiveUI window
    window = ResponsiveUI()
    window.show()

    # Execute the application event loop
    exit_code = app.exec()

    # Return the application exit code
    return exit_code


if __name__ == "__main__":
    # Run the main function if the script is executed directly
    sys.exit(main())