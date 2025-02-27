from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QStyle, QApplication
from PyQt6.QtGui import QIcon
from PyQt6.QtCore import Qt
import logging

# Configure logging for this module
logger = logging.getLogger(__name__)

class ErrorDisplay(QWidget):
    """
    A PyQt6 widget for displaying error messages with appropriate styling and icons.
    
    This widget provides a consistent way to display error and warning messages
    in the MFE Toolbox application. It includes appropriate styling and icons
    based on the message severity.
    """
    
    def __init__(self, parent=None):
        """
        Initializes the error display component with layouts and child widgets.
        
        Args:
            parent (QWidget, optional): The parent widget, if any. Defaults to None.
        """
        # Call the parent QWidget constructor
        super().__init__(parent)
        
        # Initialize instance variables
        self.current_message = ""
        self.severity = ""
        self.is_visible = False
        
        # Set up the main vertical layout
        self.main_layout = QVBoxLayout()
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Create horizontal layout for icon and message
        self.content_layout = QHBoxLayout()
        
        # Create and configure icon label
        self.icon_label = QLabel()
        self.icon_label.setMinimumWidth(24)
        self.icon_label.setMaximumWidth(24)
        self.icon_label.setMinimumHeight(24)
        self.icon_label.setMaximumHeight(24)
        
        # Create and configure message label with appropriate styling
        self.message_label = QLabel()
        self.message_label.setWordWrap(True)
        self.message_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        
        # Add icon and message labels to horizontal layout
        self.content_layout.addWidget(self.icon_label)
        self.content_layout.addWidget(self.message_label, 1)  # 1 to give it stretch
        
        # Add horizontal layout to main layout
        self.main_layout.addLayout(self.content_layout)
        
        # Set the widget's layout to main_layout
        self.setLayout(self.main_layout)
        
        # Hide the widget initially as there's no error to display
        self.hide()
    
    def show_error(self, message):
        """
        Displays an error message with error styling and icon.
        
        Args:
            message (str): The error message to display.
        """
        # Log the error message
        logger.error(message)
        
        # Set current_message and severity to 'error'
        self.current_message = message
        self.severity = "error"
        
        # Update message label text with the error message
        self.message_label.setText(message)
        
        # Apply error styling to the component (red text, etc.)
        self.message_label.setStyleSheet("color: #d9534f; font-weight: bold;")
        self.setStyleSheet("background-color: #f2dede; border: 1px solid #ebccd1; border-radius: 4px;")
        
        # Set error icon using QApplication.style().standardIcon
        error_icon = QApplication.style().standardIcon(QStyle.StandardPixmap.SP_MessageBoxCritical)
        self.icon_label.setPixmap(error_icon.pixmap(24, 24))
        
        # Make the component visible
        self.show()
        
        # Set is_visible to True
        self.is_visible = True
    
    def show_warning(self, message):
        """
        Displays a warning message with warning styling and icon.
        
        Args:
            message (str): The warning message to display.
        """
        # Log the warning message
        logger.warning(message)
        
        # Set current_message and severity to 'warning'
        self.current_message = message
        self.severity = "warning"
        
        # Update message label text with the warning message
        self.message_label.setText(message)
        
        # Apply warning styling to the component (amber text, etc.)
        self.message_label.setStyleSheet("color: #8a6d3b; font-weight: bold;")
        self.setStyleSheet("background-color: #fcf8e3; border: 1px solid #faebcc; border-radius: 4px;")
        
        # Set warning icon using QApplication.style().standardIcon
        warning_icon = QApplication.style().standardIcon(QStyle.StandardPixmap.SP_MessageBoxWarning)
        self.icon_label.setPixmap(warning_icon.pixmap(24, 24))
        
        # Make the component visible
        self.show()
        
        # Set is_visible to True
        self.is_visible = True
    
    def hide_message(self):
        """
        Hides the currently displayed error or warning message.
        """
        # Hide the widget
        self.hide()
        
        # Set is_visible to False
        self.is_visible = False
        
        # Clear current_message and severity
        self.current_message = ""
        self.severity = ""
    
    def is_showing_message(self):
        """
        Checks if an error or warning is currently being displayed.
        
        Returns:
            bool: True if a message is currently displayed, False otherwise.
        """
        # Return the value of is_visible
        return self.is_visible
    
    def get_current_message(self):
        """
        Returns the currently displayed message text.
        
        Returns:
            str: The current error or warning message, or empty string if none.
        """
        # Return current_message or empty string if no message is displayed
        return self.current_message
    
    def get_severity(self):
        """
        Returns the severity level of the current message.
        
        Returns:
            str: Either 'error', 'warning', or empty string if no message.
        """
        # Return severity or empty string if no message is displayed
        return self.severity