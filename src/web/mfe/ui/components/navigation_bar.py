from typing import Optional
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtWidgets import QWidget, QPushButton, QLabel, QHBoxLayout, QStyle
from PyQt6.QtGui import QIcon


class NavigationBar(QWidget):
    """
    A PyQt6 widget that provides navigation controls with previous/next buttons 
    and page indicators
    """
    # Navigation signals
    navigated_previous = pyqtSignal()
    navigated_next = pyqtSignal()
    
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """
        Initializes the NavigationBar with previous/next buttons and page indicator
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        
        # Create buttons and indicators
        self._previous_button = QPushButton()
        self._previous_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowLeft))
        self._previous_button.setToolTip("Previous page")
        
        self._next_button = QPushButton()
        self._next_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowRight))
        self._next_button.setToolTip("Next page")
        
        self._page_indicator = QLabel("1/1")
        self._page_indicator.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Set up layout
        self._layout = QHBoxLayout(self)
        self._layout.addWidget(self._previous_button)
        self._layout.addWidget(self._page_indicator)
        self._layout.addWidget(self._next_button)
        self.setLayout(self._layout)
        
        # Connect signals
        self._previous_button.clicked.connect(self._previous_clicked)
        self._next_button.clicked.connect(self._next_clicked)
        
        # Set initial button states
        self._previous_button.setEnabled(False)
        self._next_button.setEnabled(True)
    
    def update_page_indicator(self, current_page: int, total_pages: int) -> None:
        """
        Updates the page indicator text with current page and total pages
        
        Args:
            current_page: Current page number (1-based index)
            total_pages: Total number of pages
        """
        if current_page < 1 or current_page > total_pages:
            raise ValueError(f"Current page {current_page} is out of range [1, {total_pages}]")
        
        self._page_indicator.setText(f"{current_page}/{total_pages}")
        
        # Update button states
        self._previous_button.setEnabled(current_page > 1)
        self._next_button.setEnabled(current_page < total_pages)
    
    def enable_previous(self, enabled: bool) -> None:
        """
        Enables or disables the previous button based on navigation state
        
        Args:
            enabled: Whether the previous button should be enabled
        """
        self._previous_button.setEnabled(enabled)
    
    def enable_next(self, enabled: bool) -> None:
        """
        Enables or disables the next button based on navigation state
        
        Args:
            enabled: Whether the next button should be enabled
        """
        self._next_button.setEnabled(enabled)
    
    def _previous_clicked(self) -> None:
        """Internal handler for previous button clicks"""
        self.navigated_previous.emit()
    
    def _next_clicked(self) -> None:
        """Internal handler for next button clicks"""
        self.navigated_next.emit()