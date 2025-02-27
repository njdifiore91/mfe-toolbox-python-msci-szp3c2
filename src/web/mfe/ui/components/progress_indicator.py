"""
MFE Toolbox UI - Progress Indicator Component

This module provides reusable progress indicator components for displaying task progress
in the MFE Toolbox UI. It enables visual feedback for long-running operations like
model estimation, data processing, and plotting using PyQt6.

The module includes two main classes:
- ProgressIndicator: A customizable progress bar widget with styling options
- TaskProgressIndicator: A specialized progress indicator that integrates with the TaskManager

Both components support real-time progress updates and cancellation functionality.
"""

import enum
import logging
from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot, QSize
from PyQt6.QtWidgets import (
    QWidget, QProgressBar, QLabel, QVBoxLayout, 
    QHBoxLayout, QPushButton, QSizePolicy
)

from ..async.signals import TaskSignals
from ..async.task_manager import TaskStatus
from ..styles import StyleConfig

# Configure module logger
logger = logging.getLogger(__name__)


class ProgressStyle(enum.Enum):
    """
    Enumeration defining different visual styles for the progress indicator.
    """
    DEFAULT = "default"
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class ProgressIndicator(QWidget):
    """
    A PyQt6 widget for displaying task progress with customizable appearance
    and optional cancellation functionality.
    
    This widget provides a progress bar with a label and an optional cancel button,
    and can be styled to indicate different states (success, error, etc.).
    """
    
    # Signals
    cancel_requested = pyqtSignal()
    progress_changed = pyqtSignal(float)
    
    def __init__(self, parent: Optional[QWidget] = None, 
                 show_percentage: bool = True,
                 show_cancel_button: bool = False):
        """
        Initialize the progress indicator widget with optional configuration.
        
        Parameters
        ----------
        parent : QWidget, optional
            Parent widget
        show_percentage : bool
            Whether to display percentage in the progress bar
        show_cancel_button : bool
            Whether to show a cancel button
        """
        super().__init__(parent)
        
        # Initialize state
        self.is_active = False
        self.current_task = ""
        self.current_style = ProgressStyle.DEFAULT
        self.current_progress = 0.0
        self.show_percentage = show_percentage
        self.show_cancel_button = show_cancel_button
        
        # Set up layout
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create task label
        self.label = QLabel(self)
        self.label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        self.label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.main_layout.addWidget(self.label)
        
        # Create progress bar
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(show_percentage)
        self.progress_bar.setMinimumHeight(20)
        self.progress_bar.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        
        # Create horizontal layout for progress bar and cancel button
        self.progress_layout = QHBoxLayout()
        self.progress_layout.setContentsMargins(0, 0, 0, 0)
        self.progress_layout.addWidget(self.progress_bar)
        
        # Create cancel button if enabled
        self.cancel_button = None
        if show_cancel_button:
            self.cancel_button = QPushButton("Cancel", self)
            self.cancel_button.setToolTip("Cancel the current operation")
            self.cancel_button.clicked.connect(self._on_cancel_clicked)
            self.cancel_button.setFixedWidth(70)
            self.progress_layout.addWidget(self.cancel_button)
        
        self.main_layout.addLayout(self.progress_layout)
        
        # Set widget layout
        self.setLayout(self.main_layout)
        
        # Apply default styling
        self._apply_style()
        
        # Hide widget initially
        self.hide()
        
        logger.debug("ProgressIndicator initialized")
    
    def start_progress(self, task_name: str, 
                      style: ProgressStyle = ProgressStyle.DEFAULT,
                      initial_progress: float = 0.0):
        """
        Shows the progress indicator and sets initial task information.
        
        Parameters
        ----------
        task_name : str
            Name or description of the task
        style : ProgressStyle
            Visual style for the progress indicator
        initial_progress : float
            Initial progress value between 0.0 and 1.0
        """
        self.is_active = True
        self.current_task = task_name
        self.current_style = style
        
        # Set label and progress
        self.label.setText(task_name)
        self.progress_bar.setValue(int(initial_progress * 100))
        self.current_progress = initial_progress
        
        # Apply styling
        self._apply_style()
        
        # Show the widget
        self.show()
        
        logger.debug(f"Started progress for task: {task_name}")
        self.progress_changed.emit(initial_progress)
    
    def update_progress(self, progress: float, message: Optional[str] = None):
        """
        Updates the progress indicator with new progress value.
        
        Parameters
        ----------
        progress : float
            Progress value between 0.0 and 1.0
        message : str, optional
            Optional message to display instead of the current task name
        """
        if not self.is_active:
            return
        
        # Validate progress value
        if progress < 0.0:
            progress = 0.0
        elif progress > 1.0:
            progress = 1.0
        
        # Update progress bar
        self.progress_bar.setValue(int(progress * 100))
        self.current_progress = progress
        
        # Update label if message is provided
        if message:
            self.label.setText(message)
        
        # Log significant progress changes
        if int(progress * 100) % 10 == 0:  # Log at 10% intervals
            logger.debug(f"Progress updated: {progress:.2f} for {self.current_task}")
        
        # Emit signal
        self.progress_changed.emit(progress)
    
    def complete(self, style: ProgressStyle = ProgressStyle.SUCCESS, 
                message: Optional[str] = None):
        """
        Marks the progress as complete with optional success/error styling.
        
        Parameters
        ----------
        style : ProgressStyle
            Style to indicate completion status (SUCCESS, ERROR, etc.)
        message : str, optional
            Optional completion message
        """
        # Set progress to 100%
        self.progress_bar.setValue(100)
        self.current_progress = 1.0
        self.current_style = style
        
        # Update message if provided
        if message:
            self.label.setText(message)
        else:
            self.label.setText(f"{self.current_task} - Complete")
        
        # Apply styling
        self._apply_style()
        
        logger.debug(f"Task completed: {self.current_task} with style {style.name}")
        self.progress_changed.emit(1.0)
    
    def hide_progress(self):
        """
        Hides the progress indicator.
        """
        self.hide()
        self.is_active = False
        self.progress_bar.setValue(0)
        self.current_progress = 0.0
        
        logger.debug("Progress indicator hidden")
    
    def reset(self):
        """
        Resets the progress indicator to initial state.
        """
        self.progress_bar.setValue(0)
        self.current_progress = 0.0
        self.current_style = ProgressStyle.DEFAULT
        self.label.setText("")
        self._apply_style()
        self.hide()
        self.is_active = False
        
        logger.debug("Progress indicator reset")
    
    def is_showing(self) -> bool:
        """
        Checks if the progress indicator is currently visible.
        
        Returns
        -------
        bool
            True if the indicator is visible, False otherwise
        """
        return self.is_active
    
    def get_progress(self) -> float:
        """
        Gets the current progress value.
        
        Returns
        -------
        float
            Current progress value between 0.0 and 1.0
        """
        return self.current_progress
    
    def get_task_name(self) -> str:
        """
        Gets the current task name.
        
        Returns
        -------
        str
            Current task name or empty string if none
        """
        return self.current_task
    
    def set_style(self, style: ProgressStyle):
        """
        Sets the visual style of the progress indicator.
        
        Parameters
        ----------
        style : ProgressStyle
            The style to apply to the progress indicator
        """
        self.current_style = style
        self._apply_style()
        
        logger.debug(f"Style changed to {style.name}")
    
    def show_cancel_button(self, show: bool):
        """
        Shows or hides the cancel button.
        
        Parameters
        ----------
        show : bool
            True to show the cancel button, False to hide it
        """
        self.show_cancel_button = show
        
        if self.cancel_button:
            self.cancel_button.setVisible(show)
    
    def connect_to_task(self, task_name: str, signals: TaskSignals):
        """
        Connects the progress indicator to a task's signals.
        
        Parameters
        ----------
        task_name : str
            Name of the task to display
        signals : TaskSignals
            Signal container for the task
        """
        # Connect task signals to progress indicator methods
        signals.started.connect(lambda: self.start_progress(task_name))
        signals.progress.connect(self.update_progress)
        signals.finished.connect(
            lambda: self.complete(ProgressStyle.SUCCESS)
        )
        signals.error.connect(
            lambda: self.complete(ProgressStyle.ERROR)
        )
        signals.cancelled.connect(
            lambda: self.complete(ProgressStyle.WARNING, "Cancelled")
        )
        
        # Connect cancel button to task cancellation
        self.cancel_requested.connect(signals.cancelled)
        
        logger.debug(f"Connected progress indicator to task: {task_name}")
    
    @pyqtSlot()
    def _on_cancel_clicked(self):
        """
        Slot for handling the cancel button click.
        """
        self.cancel_requested.emit()
        
        # Disable cancel button to prevent multiple clicks
        if self.cancel_button:
            self.cancel_button.setEnabled(False)
            self.label.setText(f"{self.current_task} - Cancelling...")
        
        logger.debug(f"Cancellation requested for task: {self.current_task}")
    
    def _apply_style(self):
        """
        Internal method to apply styling based on the current style.
        """
        style_config = StyleConfig()
        
        # Common style for the progress indicator
        common_style = """
            QProgressBar {
                border-radius: 3px;
                text-align: center;
            }
            QProgressBar::chunk {
                border-radius: 3px;
            }
        """
        
        # Apply style based on current_style
        if self.current_style == ProgressStyle.SUCCESS:
            self.progress_bar.setStyleSheet(common_style + f"""
                QProgressBar {{
                    border: 1px solid {style_config.get_color('success').name()};
                }}
                QProgressBar::chunk {{
                    background-color: {style_config.get_color('success').name()};
                }}
            """)
            self.label.setStyleSheet(f"color: {style_config.get_color('success').name()};")
            
        elif self.current_style == ProgressStyle.ERROR:
            self.progress_bar.setStyleSheet(common_style + f"""
                QProgressBar {{
                    border: 1px solid {style_config.get_color('error').name()};
                }}
                QProgressBar::chunk {{
                    background-color: {style_config.get_color('error').name()};
                }}
            """)
            self.label.setStyleSheet(f"color: {style_config.get_color('error').name()};")
            
        elif self.current_style == ProgressStyle.WARNING:
            self.progress_bar.setStyleSheet(common_style + f"""
                QProgressBar {{
                    border: 1px solid {style_config.get_color('warning').name()};
                }}
                QProgressBar::chunk {{
                    background-color: {style_config.get_color('warning').name()};
                }}
            """)
            self.label.setStyleSheet(f"color: {style_config.get_color('warning').name()};")
            
        elif self.current_style == ProgressStyle.INFO:
            self.progress_bar.setStyleSheet(common_style + f"""
                QProgressBar {{
                    border: 1px solid {style_config.get_color('primary').name()};
                }}
                QProgressBar::chunk {{
                    background-color: {style_config.get_color('primary').name()};
                }}
            """)
            self.label.setStyleSheet(f"color: {style_config.get_color('primary').name()};")
            
        else:  # DEFAULT
            self.progress_bar.setStyleSheet(common_style + f"""
                QProgressBar {{
                    border: 1px solid {style_config.get_color('border').name()};
                }}
                QProgressBar::chunk {{
                    background-color: {style_config.get_color('primary').name()};
                }}
            """)
            self.label.setStyleSheet("")  # Default label style


class TaskProgressIndicator(QWidget):
    """
    A specialized progress indicator that integrates with the TaskManager system.
    
    This component automatically handles task lifecycle events from a TaskManager
    instance and displays progress updates accordingly.
    """
    
    def __init__(self, parent: Optional[QWidget] = None,
                 show_percentage: bool = True,
                 show_cancel_button: bool = True):
        """
        Initialize a progress indicator specifically for task management integration.
        
        Parameters
        ----------
        parent : QWidget, optional
            Parent widget
        show_percentage : bool
            Whether to display percentage in the progress bar
        show_cancel_button : bool
            Whether to show a cancel button
        """
        super().__init__(parent)
        
        # Set up layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create progress indicator
        self.progress_indicator = ProgressIndicator(
            self, show_percentage, show_cancel_button
        )
        
        layout.addWidget(self.progress_indicator)
        self.setLayout(layout)
        
        # Dictionary to track signal connections
        self._task_connections = {}
        
        # Apply styling from config
        style_config = StyleConfig()
        self.setStyleSheet(style_config.get_stylesheet('widget'))
        
        logger.debug("TaskProgressIndicator initialized")
    
    def connect_to_task_manager(self, task_manager):
        """
        Connects the indicator to a TaskManager instance to monitor tasks.
        
        Parameters
        ----------
        task_manager : TaskManager
            The task manager instance to monitor
        """
        # Connect to task manager signals
        self._task_connections['started'] = task_manager.signals.task_started.connect(
            self._on_task_started
        )
        self._task_connections['progress'] = task_manager.signals.task_progress.connect(
            self._on_task_progress
        )
        self._task_connections['completed'] = task_manager.signals.task_completed.connect(
            self._on_task_completed
        )
        self._task_connections['failed'] = task_manager.signals.task_failed.connect(
            self._on_task_failed
        )
        self._task_connections['cancelled'] = task_manager.signals.task_cancelled.connect(
            self._on_task_cancelled
        )
        
        # Store task manager reference for cancellation requests
        self._task_connections['task_manager'] = task_manager
        
        # Connect cancel button to request cancellation
        self.progress_indicator.cancel_requested.connect(self._on_cancel_requested)
        
        logger.debug("Connected to task manager")
    
    def disconnect_from_task_manager(self):
        """
        Disconnects from previously connected TaskManager.
        """
        # Disconnect all signal connections
        for key, connection in list(self._task_connections.items()):
            if key != 'task_manager' and hasattr(connection, 'disconnect'):
                connection.disconnect()
        
        # Clear connection tracking
        self._task_connections.clear()
        
        # Reset progress indicator
        self.progress_indicator.reset()
        
        logger.debug("Disconnected from task manager")
    
    @pyqtSlot(str)
    def _on_task_started(self, task_id: str):
        """
        Handler for task started signal.
        
        Parameters
        ----------
        task_id : str
            ID of the started task
        """
        # Store current task ID
        self._task_connections['current_task_id'] = task_id
        
        # Get task name from task manager
        task_manager = self._task_connections.get('task_manager')
        task_name = f"Task {task_id}"
        if task_manager:
            try:
                task = task_manager._tasks.get(task_id)
                if task:
                    task_name = task.name
            except (AttributeError, KeyError):
                pass
        
        # Start progress tracking
        self.progress_indicator.start_progress(task_name)
        
        # Show cancel button if task is cancellable
        self.progress_indicator.show_cancel_button(True)
        
        logger.debug(f"Monitoring task: {task_id} ({task_name})")
    
    @pyqtSlot(str, float)
    def _on_task_progress(self, task_id: str, progress: float):
        """
        Handler for task progress signal.
        
        Parameters
        ----------
        task_id : str
            ID of the task reporting progress
        progress : float
            Progress value between 0.0 and 1.0
        """
        # Only update if this is the current task we're monitoring
        current_task_id = self._task_connections.get('current_task_id')
        if current_task_id and current_task_id == task_id:
            self.progress_indicator.update_progress(progress)
            
            # Log significant progress changes
            if int(progress * 100) % 25 == 0:  # Log at 25% intervals
                logger.debug(f"Task {task_id} progress: {progress:.2f}")
    
    @pyqtSlot(str, object)
    def _on_task_completed(self, task_id: str, result):
        """
        Handler for task completed signal.
        
        Parameters
        ----------
        task_id : str
            ID of the completed task
        result : object
            Result of the task
        """
        # Only update if this is the current task we're monitoring
        current_task_id = self._task_connections.get('current_task_id')
        if current_task_id and current_task_id == task_id:
            self.progress_indicator.complete(ProgressStyle.SUCCESS)
            
            # Schedule hiding the progress indicator after a delay
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(2000, self.progress_indicator.hide_progress)
            
            logger.debug(f"Task {task_id} completed successfully")
    
    @pyqtSlot(str, Exception)
    def _on_task_failed(self, task_id: str, error: Exception):
        """
        Handler for task failed signal.
        
        Parameters
        ----------
        task_id : str
            ID of the failed task
        error : Exception
            Error that caused the failure
        """
        # Only update if this is the current task we're monitoring
        current_task_id = self._task_connections.get('current_task_id')
        if current_task_id and current_task_id == task_id:
            self.progress_indicator.complete(
                ProgressStyle.ERROR, f"Error: {str(error)}"
            )
            
            # Schedule hiding the progress indicator after a longer delay
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(5000, self.progress_indicator.hide_progress)
            
            logger.error(f"Task {task_id} failed: {str(error)}")
    
    @pyqtSlot(str)
    def _on_task_cancelled(self, task_id: str):
        """
        Handler for task cancelled signal.
        
        Parameters
        ----------
        task_id : str
            ID of the cancelled task
        """
        # Only update if this is the current task we're monitoring
        current_task_id = self._task_connections.get('current_task_id')
        if current_task_id and current_task_id == task_id:
            self.progress_indicator.complete(
                ProgressStyle.WARNING, "Cancelled"
            )
            
            # Schedule hiding the progress indicator after a delay
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(2000, self.progress_indicator.hide_progress)
            
            logger.debug(f"Task {task_id} was cancelled")
    
    @pyqtSlot()
    def _on_cancel_requested(self):
        """
        Handler for the cancel button click.
        """
        # Get current task ID
        current_task_id = self._task_connections.get('current_task_id')
        if not current_task_id:
            return
        
        # Get task manager
        task_manager = self._task_connections.get('task_manager')
        if not task_manager:
            return
        
        # Request task cancellation
        try:
            task_manager.cancel_task(current_task_id)
            self.progress_indicator.label.setText("Cancelling...")
            logger.debug(f"Requested cancellation of task {current_task_id}")
        except Exception as e:
            logger.error(f"Error cancelling task {current_task_id}: {str(e)}")