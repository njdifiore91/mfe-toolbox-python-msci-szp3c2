"""
About dialog for the MFE Toolbox application.

This module implements the About dialog which displays version information,
credits, and links to website and documentation using PyQt6.
"""

import logging
from typing import Optional

from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QPixmap, QDesktopServices, QUrl
from PyQt6.QtWidgets import QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QWidget

from .dialogs import BaseDialog
from .assets import load_image
from .styles import get_title_font, get_normal_font, apply_stylesheet

# Configure logger
logger = logging.getLogger(__name__)

# Application constants
APP_NAME = "ARMAX Model Estimation"
VERSION = "4.0"
COPYRIGHT = "(c) 2009 Kevin Sheppard"
WEBSITE_URL = "https://github.com/username/mfe-toolbox"
DOCUMENTATION_URL = "https://mfe-toolbox.readthedocs.io"


class AboutDialog(BaseDialog):
    """
    Dialog displaying information about the MFE Toolbox application,
    including version, logo, and link buttons.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        """
        Initializes the about dialog with application information and visual elements.

        Args:
            parent: Optional parent widget
        """
        super().__init__(parent, "About ARMAX")
        logger.debug("Initializing About dialog")
        
        # Set fixed size for consistent appearance
        self.set_fixed_size(400, 300)
        
        # Create main layout
        main_layout = QVBoxLayout()
        main_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.setSpacing(10)
        
        # Add components
        main_layout.addWidget(self._create_logo_section())
        main_layout.addWidget(self._create_info_section())
        main_layout.addWidget(self._create_button_section())
        
        # Add main layout to dialog content area
        self.add_layout_to_content(main_layout)
        
        logger.debug("About dialog initialized")

    def _create_logo_section(self) -> QLabel:
        """
        Creates and returns a label containing the Oxford logo.

        Returns:
            Label with Oxford logo
        """
        logo_label = QLabel()
        logo_pixmap = load_image('oxlogo')
        
        # Scale logo to appropriate size
        scaled_pixmap = logo_pixmap.scaled(
            QSize(120, 120),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        logo_label.setPixmap(scaled_pixmap)
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo_label.setContentsMargins(0, 10, 0, 10)
        
        return logo_label

    def _create_info_section(self) -> QLabel:
        """
        Creates and returns a label with application information.

        Returns:
            Label with application information
        """
        info_text = f"{APP_NAME}\nVersion {VERSION}\n{COPYRIGHT}"
        
        info_label = QLabel(info_text)
        info_label.setFont(get_normal_font())
        info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info_label.setContentsMargins(0, 5, 0, 15)
        
        return info_label

    def _create_button_section(self) -> QWidget:
        """
        Creates button section with website, documentation and OK buttons.

        Returns:
            Widget containing buttons
        """
        # Create container widget
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create links layout
        links_layout = QHBoxLayout()
        links_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        links_layout.setSpacing(20)
        
        # Create website button styled as hyperlink
        website_button = QPushButton("Website")
        self._style_hyperlink_button(website_button)
        website_button.clicked.connect(self._open_website)
        
        # Create documentation button styled as hyperlink
        docs_button = QPushButton("Documentation")
        self._style_hyperlink_button(docs_button)
        docs_button.clicked.connect(self._open_documentation)
        
        # Add links to layout
        links_layout.addWidget(website_button)
        links_layout.addWidget(docs_button)
        
        # Create OK button using base dialog function
        ok_button = QPushButton("OK")
        ok_button.setDefault(True)
        ok_button.clicked.connect(self.accept)
        
        # Create button layout for OK button
        buttons = [ok_button]
        button_widget = self.create_button_layout(buttons)
        
        # Add layouts to container
        container_layout.addLayout(links_layout)
        container_layout.addWidget(button_widget)
        
        return container

    def _style_hyperlink_button(self, button: QPushButton) -> None:
        """
        Styles a button to look like a hyperlink.

        Args:
            button: Button to style
        """
        # Set flat style (no border)
        button.setFlat(True)
        
        # Apply hyperlink styling
        button.setStyleSheet("""
            QPushButton {
                color: blue;
                text-decoration: underline;
                background: transparent;
                border: none;
            }
            QPushButton:hover {
                color: #0066cc;
            }
            QPushButton:pressed {
                color: #003366;
            }
        """)
        
        # Set cursor to pointing hand
        button.setCursor(Qt.CursorShape.PointingHandCursor)

    def _open_website(self) -> None:
        """
        Opens the application website in the default browser.
        """
        try:
            logger.debug(f"Opening website URL: {WEBSITE_URL}")
            QDesktopServices.openUrl(QUrl(WEBSITE_URL))
        except Exception as e:
            logger.error(f"Failed to open website: {str(e)}")

    def _open_documentation(self) -> None:
        """
        Opens the application documentation in the default browser.
        """
        try:
            logger.debug(f"Opening documentation URL: {DOCUMENTATION_URL}")
            QDesktopServices.openUrl(QUrl(DOCUMENTATION_URL))
        except Exception as e:
            logger.error(f"Failed to open documentation: {str(e)}")


def show_about_dialog(parent: Optional[QWidget] = None) -> None:
    """
    Shows the About dialog modally and returns after the user dismisses it.

    Args:
        parent: Optional parent widget
    """
    dialog = AboutDialog(parent)
    dialog.exec()