"""
Implements a confirmation dialog that appears when attempting to close the application 
with unsaved changes. This PyQt6-based dialog provides users with options to save, 
discard, or cancel the close operation.
"""

import logging
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QWidget

from .assets import load_icon
from .dialogs import BaseDialog
from .styles import get_normal_font, apply_stylesheet

# Setup logger
logger = logging.getLogger(__name__)

# Dialog constants
DIALOG_TITLE = "Confirm Close"
DIALOG_MESSAGE = "Are you sure you want to close?\nUnsaved changes will be lost."


class CloseConfirmationDialog(BaseDialog):
    """
    Dialog that confirms if the user wants to close the application with unsaved changes.
    """
    
    def __init__(self, parent: Optional[QWidget] = None):
        """
        Initializes the close confirmation dialog with warning icon and message.
        
        Args:
            parent: Optional parent widget
        """
        super().__init__(parent, DIALOG_TITLE)
        
        # Set fixed size for consistent appearance
        self.set_fixed_size(400, 180)
        
        # Create main layout for dialog content
        main_layout = QVBoxLayout()
        
        # Add warning icon and message
        content_widget = self._create_content_section()
        main_layout.addWidget(content_widget)
        
        # Create buttons
        button_widget = self._create_button_section()
        main_layout.addWidget(button_widget)
        
        # Set the main layout on the content widget
        self.content_layout.addLayout(main_layout)
        
        # Apply styling
        apply_stylesheet(self, "")
        
        logger.debug("Close confirmation dialog initialized")
    
    def _create_content_section(self) -> QWidget:
        """
        Creates the section with warning icon and message.
        
        Returns:
            Widget containing the content section
        """
        content_widget = QWidget()
        content_layout = QHBoxLayout(content_widget)
        
        # Load warning icon
        warning_icon = load_icon('warning')
        
        # Create icon label
        self.icon_label = QLabel()
        self.icon_label.setPixmap(warning_icon.pixmap(32, 32))
        self.icon_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        # Create message label
        self.message_label = QLabel(DIALOG_MESSAGE)
        self.message_label.setFont(get_normal_font())
        self.message_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        
        # Add labels to layout
        content_layout.addWidget(self.icon_label)
        content_layout.addWidget(self.message_label, 1)
        
        return content_widget
    
    def _create_button_section(self) -> QWidget:
        """
        Creates the buttons for the dialog.
        
        Returns:
            Widget containing buttons
        """
        # Create buttons
        self.no_button = QPushButton("No")
        self.yes_button = QPushButton("Yes")
        
        # Set No button as default (focused when dialog opens)
        self.no_button.setDefault(True)
        
        # Connect button signals
        self.no_button.clicked.connect(self.handle_no)
        self.yes_button.clicked.connect(self.handle_yes)
        
        # Create button layout
        return self.create_button_layout([self.no_button, self.yes_button])
    
    def handle_yes(self) -> None:
        """
        Handles the Yes button click (confirm close).
        """
        logger.debug("User selected to close the application")
        self.set_result(True)
        self.accept()
    
    def handle_no(self) -> None:
        """
        Handles the No button click (cancel close).
        """
        logger.debug("User cancelled closing the application")
        self.set_result(False)
        self.reject()


def show_close_confirmation_dialog(parent: Optional[QWidget] = None) -> bool:
    """
    Shows the close confirmation dialog modally and returns whether the user confirmed closure.
    
    Args:
        parent: Optional parent widget
        
    Returns:
        True if user confirmed closing, False otherwise
    """
    dialog = CloseConfirmationDialog(parent)
    result = dialog.exec()
    
    logger.debug(f"Close confirmation dialog result: {result}")
    return result