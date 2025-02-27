"""
Provides a comprehensive set of dialog components and utilities for the MFE Toolbox UI.
Implements a BaseDialog class that serves as the foundation for specialized dialogs
including confirmation, information, and error dialogs. Includes utility functions
for common dialog operations and standardized dialog styling.
"""

import asyncio
import logging
from typing import Any, Callable, Coroutine, List, Optional, Tuple, TypeVar, Union

from PyQt6.QtCore import Qt, QSize, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QIcon, QPixmap
from PyQt6.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QFrame, QSizePolicy, QMessageBox, QLayout, QLineEdit, QProgressBar
)

from .assets.__init__ import get_icon_path
from .styles import (
    apply_stylesheet, get_title_font, get_normal_font, 
    create_standard_button, style_group_box
)

# Configure logger
logger = logging.getLogger(__name__)

# Dialog constants
DEFAULT_DIALOG_WIDTH = 400
DEFAULT_DIALOG_HEIGHT = 300
BUTTON_SPACING = 10
DIALOG_MARGIN = 20

T = TypeVar('T')  # For generic return types

class BaseDialog(QDialog):
    """
    Base class for all dialogs in the MFE Toolbox, providing common functionality
    and consistent styling.
    """
    
    def __init__(self, parent: Optional[QWidget] = None, title: str = "Dialog"):
        """
        Initializes the base dialog with standard layout and styling.
        
        Args:
            parent: Optional parent widget
            title: Dialog title
        """
        super().__init__(parent)
        
        # Setup dialog properties
        self.setWindowTitle(title)
        self.setModal(True)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowContextHelpButtonHint)
        
        # Setup layout
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(DIALOG_MARGIN, DIALOG_MARGIN, DIALOG_MARGIN, DIALOG_MARGIN)
        self.main_layout.setSpacing(10)
        
        # Create content area
        self.content_widget = QWidget(self)
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(10)
        
        # Create button area
        self.button_widget = QWidget(self)
        self.button_layout = QHBoxLayout(self.button_widget)
        self.button_layout.setContentsMargins(0, 0, 0, 0)
        self.button_layout.setSpacing(BUTTON_SPACING)
        
        # Add widgets to main layout
        self.main_layout.addWidget(self.content_widget, 1)
        self.main_layout.addWidget(self.button_widget, 0)
        
        # Initialize dialog result
        self.dialog_result = None
        
        # Connect signals
        self.accepted.connect(self.handle_accepted)
        self.rejected.connect(self.handle_rejected)
        
        # Set initial size
        self.resize(DEFAULT_DIALOG_WIDTH, DEFAULT_DIALOG_HEIGHT)
        
        # Apply styling
        apply_stylesheet(self, "")  # Apply any global stylesheet

    def add_widget_to_content(self, widget: QWidget) -> None:
        """
        Adds a widget to the dialog's content area.
        
        Args:
            widget: The widget to add
        """
        self.content_layout.addWidget(widget)
        self.content_layout.update()

    def add_layout_to_content(self, layout: QLayout) -> None:
        """
        Adds a layout to the dialog's content area.
        
        Args:
            layout: The layout to add
        """
        self.content_layout.addLayout(layout)
        self.content_layout.update()

    def add_separator(self) -> QFrame:
        """
        Adds a horizontal separator line to the dialog.
        
        Returns:
            The created separator widget
        """
        separator = QFrame(self)
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        separator.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        separator.setMinimumHeight(1)
        separator.setMaximumHeight(1)
        self.content_layout.addWidget(separator)
        return separator

    def create_button_layout(self, buttons: List[QPushButton]) -> QWidget:
        """
        Creates a standardized button layout with the provided buttons.
        
        Args:
            buttons: List of buttons to add to the layout
            
        Returns:
            Widget containing the button layout
        """
        # Clear existing buttons
        for i in reversed(range(self.button_layout.count())):
            item = self.button_layout.itemAt(i)
            if item.widget():
                item.widget().deleteLater()
                
        # Add spacer to push buttons to the right
        self.button_layout.addStretch(1)
        
        # Add buttons
        for button in buttons:
            self.button_layout.addWidget(button)
            
        return self.button_widget

    def set_result(self, result: Any) -> None:
        """
        Sets the dialog result and accepts the dialog.
        
        Args:
            result: The result to store
        """
        self.dialog_result = result
        self.accept()

    def get_result(self) -> Any:
        """
        Gets the dialog result after execution.
        
        Returns:
            The dialog result value
        """
        return self.dialog_result

    @pyqtSlot()
    def handle_accepted(self) -> None:
        """
        Handles the dialog acceptance (OK button).
        """
        # Default implementation - can be overridden by subclasses
        if self.dialog_result is None:
            self.dialog_result = True
        logger.debug(f"Dialog {self.windowTitle()} accepted with result: {self.dialog_result}")

    @pyqtSlot()
    def handle_rejected(self) -> None:
        """
        Handles the dialog rejection (Cancel button or close).
        """
        self.dialog_result = False
        logger.debug(f"Dialog {self.windowTitle()} rejected")

    def set_fixed_size(self, width: int, height: int) -> None:
        """
        Sets the dialog to a fixed size.
        
        Args:
            width: Fixed width in pixels
            height: Fixed height in pixels
        """
        self.setFixedSize(width, height)
        self.updateGeometry()

    def center_on_parent(self) -> None:
        """
        Centers the dialog on its parent widget.
        """
        center_dialog(self, self.parent())
    
    def center_on_screen(self) -> None:
        """
        Centers the dialog on the screen.
        """
        center_dialog(self, None)
        
    def exec(self) -> Any:
        """
        Executes the dialog modally and returns the result.
        
        Returns:
            The dialog result
        """
        self.center_on_parent()
        super().exec()
        return self.dialog_result


class InputDialog(BaseDialog):
    """
    Dialog for getting text input from the user.
    """
    
    def __init__(self, 
                 parent: Optional[QWidget] = None,
                 title: str = "Input",
                 message: str = "Enter value:",
                 default_text: str = "") -> None:
        """
        Initializes the input dialog with message and input field.
        
        Args:
            parent: Optional parent widget
            title: Dialog title
            message: Prompt message
            default_text: Default text for the input field
        """
        super().__init__(parent, title)
        
        # Message label
        self.message_label = QLabel(message, self)
        self.message_label.setFont(get_normal_font())
        self.add_widget_to_content(self.message_label)
        
        # Input field
        self.input_field = QLineEdit(default_text, self)
        self.input_field.setFont(get_normal_font())
        self.input_field.selectAll()
        self.add_widget_to_content(self.input_field)
        
        # Buttons
        self.ok_button = create_standard_button("OK", True)
        self.cancel_button = create_standard_button("Cancel", False)
        
        self.ok_button.clicked.connect(self.handle_ok)
        self.cancel_button.clicked.connect(self.handle_cancel)
        
        buttons = [self.ok_button, self.cancel_button]
        self.create_button_layout(buttons)
        
        # Set OK button as default
        self.ok_button.setDefault(True)
        
        # Set input field as focus
        self.input_field.setFocus()
        
        # Adjust size
        self.set_fixed_size(DEFAULT_DIALOG_WIDTH, 150)
        
    def get_input_text(self) -> str:
        """
        Gets the text currently in the input field.
        
        Returns:
            The entered text
        """
        return self.input_field.text()
    
    @pyqtSlot()
    def handle_ok(self) -> None:
        """
        Handles the OK button click.
        """
        input_text = self.get_input_text()
        self.set_result(input_text)
        logger.debug(f"Input dialog: User entered '{input_text}'")
        self.accept()
    
    @pyqtSlot()
    def handle_cancel(self) -> None:
        """
        Handles the Cancel button click.
        """
        self.set_result("")
        logger.debug("Input dialog: User cancelled")
        self.reject()


class AsyncProgressDialog(BaseDialog):
    """
    Dialog displaying progress during asynchronous operations.
    """
    
    def __init__(self, 
                 parent: Optional[QWidget] = None,
                 title: str = "Processing",
                 message: str = "Please wait...") -> None:
        """
        Initializes the progress dialog for an asynchronous operation.
        
        Args:
            parent: Optional parent widget
            title: Dialog title
            message: Message to display
        """
        super().__init__(parent, title)
        
        # Message label
        self.message_label = QLabel(message, self)
        self.message_label.setFont(get_normal_font())
        self.add_widget_to_content(self.message_label)
        
        # Progress bar
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.add_widget_to_content(self.progress_bar)
        
        # Status label
        self.status_label = QLabel("", self)
        self.status_label.setFont(get_normal_font())
        self.add_widget_to_content(self.status_label)
        
        # Cancel button
        self.cancel_button = create_standard_button("Cancel", False)
        self.cancel_button.clicked.connect(self.handle_cancel)
        
        self.create_button_layout([self.cancel_button])
        
        # AsyncIO task properties
        self.task = None
        self.task_result = None
        self.cancelled = False
        
        # Adjust size
        self.set_fixed_size(DEFAULT_DIALOG_WIDTH, 180)
        
    def set_task(self, coroutine_func: Callable[..., Coroutine[Any, Any, T]]) -> None:
        """
        Sets the asynchronous task to be executed.
        
        Args:
            coroutine_func: A coroutine function to execute
        """
        self.task = coroutine_func
        
    @pyqtSlot(int, str)
    def update_progress(self, progress: int, status_text: str) -> None:
        """
        Updates the progress bar and status.
        
        Args:
            progress: Progress value (0-100)
            status_text: Status text to display
        """
        self.progress_bar.setValue(progress)
        self.status_label.setText(status_text)
        # Process events to update UI
        QApplication.processEvents()  # type: ignore
        
    async def run_task(self) -> T:
        """
        Runs the asynchronous task and handles its progress.
        
        Returns:
            Result from the asynchronous task
        
        Raises:
            ValueError: If no task has been set
        """
        if self.task is None:
            raise ValueError("No task has been set")
        
        try:
            # Show the dialog but don't make it modal
            self.show()
            
            # Execute the task
            self.task_result = await self.task(
                progress_callback=self.update_progress,
                is_cancelled=self.is_cancelled
            )
            
            # Close the dialog
            self.close()
            
            return self.task_result
            
        except Exception as e:
            logger.error(f"Error in async task: {str(e)}")
            self.close()
            raise
            
    @pyqtSlot()
    def handle_cancel(self) -> None:
        """
        Handles the Cancel button click.
        """
        self.cancelled = True
        self.status_label.setText("Cancelling...")
        logger.debug("Async operation cancelled by user")
        
    def is_cancelled(self) -> bool:
        """
        Checks if the operation has been cancelled.
        
        Returns:
            True if cancelled, False otherwise
        """
        return self.cancelled


def show_error_dialog(message: str, title: str = "Error", parent: Optional[QWidget] = None) -> None:
    """
    Shows an error dialog with the specified message.
    
    Args:
        message: Error message to display
        title: Dialog title
        parent: Optional parent widget
    """
    logger.error(f"Error: {message}")
    msg_box = QMessageBox(parent)
    msg_box.setIcon(QMessageBox.Icon.Critical)
    msg_box.setText(message)
    msg_box.setWindowTitle(title)
    msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
    msg_box.setStyleSheet("""
        QMessageBox {
            background-color: #f8f9fa;
        }
        QLabel {
            color: #e74c3c;
            font-weight: bold;
        }
    """)
    msg_box.exec()


def show_warning_dialog(message: str, title: str = "Warning", parent: Optional[QWidget] = None) -> None:
    """
    Shows a warning dialog with the specified message.
    
    Args:
        message: Warning message to display
        title: Dialog title
        parent: Optional parent widget
    """
    logger.warning(f"Warning: {message}")
    msg_box = QMessageBox(parent)
    msg_box.setIcon(QMessageBox.Icon.Warning)
    msg_box.setText(message)
    msg_box.setWindowTitle(title)
    msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
    msg_box.setStyleSheet("""
        QMessageBox {
            background-color: #f8f9fa;
        }
        QLabel {
            color: #f39c12;
            font-weight: bold;
        }
    """)
    msg_box.exec()


def show_info_dialog(message: str, title: str = "Information", parent: Optional[QWidget] = None) -> None:
    """
    Shows an information dialog with the specified message.
    
    Args:
        message: Information message to display
        title: Dialog title
        parent: Optional parent widget
    """
    logger.info(f"Info: {message}")
    msg_box = QMessageBox(parent)
    msg_box.setIcon(QMessageBox.Icon.Information)
    msg_box.setText(message)
    msg_box.setWindowTitle(title)
    msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
    msg_box.setStyleSheet("""
        QMessageBox {
            background-color: #f8f9fa;
        }
        QLabel {
            color: #2980b9;
            font-weight: bold;
        }
    """)
    msg_box.exec()


def show_confirmation_dialog(message: str, title: str = "Confirm", parent: Optional[QWidget] = None) -> bool:
    """
    Shows a confirmation dialog with Yes/No options and returns the user's choice.
    
    Args:
        message: Confirmation message to display
        title: Dialog title
        parent: Optional parent widget
        
    Returns:
        True if user confirmed (Yes), False otherwise (No)
    """
    msg_box = QMessageBox(parent)
    msg_box.setIcon(QMessageBox.Icon.Question)
    msg_box.setText(message)
    msg_box.setWindowTitle(title)
    msg_box.setStandardButtons(
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
    )
    msg_box.setDefaultButton(QMessageBox.StandardButton.No)
    msg_box.setStyleSheet("""
        QMessageBox {
            background-color: #f8f9fa;
        }
        QLabel {
            color: #2c3e50;
            font-weight: bold;
        }
    """)
    
    result = msg_box.exec()
    confirmed = result == QMessageBox.StandardButton.Yes
    
    logger.debug(f"Confirmation dialog: User selected {'Yes' if confirmed else 'No'}")
    return confirmed


def show_input_dialog(message: str, title: str = "Input", default_text: str = "", 
                      parent: Optional[QWidget] = None) -> Tuple[bool, str]:
    """
    Shows an input dialog to get a text value from the user.
    
    Args:
        message: Prompt message
        title: Dialog title
        default_text: Default text for the input field
        parent: Optional parent widget
        
    Returns:
        Tuple containing (success, text value)
    """
    dialog = InputDialog(parent, title, message, default_text)
    result = dialog.exec()
    
    if result:
        input_text = dialog.get_input_text()
        logger.debug(f"Input dialog: User entered '{input_text}'")
        return True, input_text
    else:
        logger.debug("Input dialog: User cancelled")
        return False, ""


async def show_async_progress_dialog(coroutine_func: Callable[..., Coroutine[Any, Any, T]], 
                                    title: str = "Processing", 
                                    message: str = "Please wait...",
                                    parent: Optional[QWidget] = None) -> T:
    """
    Shows a progress dialog during an asynchronous operation.
    
    Args:
        coroutine_func: The coroutine function to execute
        title: Dialog title
        message: Message to display
        parent: Optional parent widget
        
    Returns:
        Result from the coroutine operation
    """
    try:
        dialog = AsyncProgressDialog(parent, title, message)
        dialog.set_task(coroutine_func)
        return await dialog.run_task()
    except Exception as e:
        logger.error(f"Error in async operation: {str(e)}")
        show_error_dialog(f"Operation failed: {str(e)}", parent=parent)
        raise


def center_dialog(dialog: QDialog, parent: Optional[QWidget] = None) -> None:
    """
    Centers a dialog on the screen or parent widget.
    
    Args:
        dialog: The dialog to center
        parent: Optional parent widget
    """
    if parent:
        # Center on parent
        parent_geo = parent.geometry()
        dialog_geo = dialog.geometry()
        
        x = parent_geo.x() + (parent_geo.width() - dialog_geo.width()) // 2
        y = parent_geo.y() + (parent_geo.height() - dialog_geo.height()) // 2
        
        dialog.move(x, y)
    else:
        # Center on screen
        screen_geo = dialog.screen().geometry()
        dialog_geo = dialog.geometry()
        
        x = (screen_geo.width() - dialog_geo.width()) // 2
        y = (screen_geo.height() - dialog_geo.height()) // 2
        
        dialog.move(x, y)