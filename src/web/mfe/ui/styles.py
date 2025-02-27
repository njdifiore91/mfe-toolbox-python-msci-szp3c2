"""
Defines styling constants, themes, and helper functions for the PyQt6-based
user interface components of the MFE Toolbox. Ensures visual consistency
across the application.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Any, Union

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont, QPalette
from PyQt6.QtWidgets import QWidget, QPushButton, QTableView, QGroupBox

# Font constants
FONT_FAMILY = 'Arial'
TITLE_FONT_SIZE = 14
NORMAL_FONT_SIZE = 11
SMALL_FONT_SIZE = 9

# Color constants
PRIMARY_COLOR = QColor(42, 130, 218)  # Blue
SECONDARY_COLOR = QColor(240, 240, 240)  # Light Gray
SUCCESS_COLOR = QColor(39, 174, 96)  # Green
WARNING_COLOR = QColor(241, 196, 15)  # Yellow
ERROR_COLOR = QColor(231, 76, 60)  # Red
LIGHT_BG_COLOR = QColor(248, 249, 250)  # Very Light Gray
DARK_BG_COLOR = QColor(52, 58, 64)  # Dark Gray

# Layout constants
DEFAULT_MARGIN = 10
DEFAULT_SPACING = 6

# Default stylesheet
DEFAULT_STYLESHEET = f"QWidget {{ font-family: {FONT_FAMILY}; }}"


def apply_stylesheet(widget: QWidget, stylesheet: str) -> None:
    """
    Applies the specified stylesheet to the given widget.
    
    Args:
        widget: The widget to apply the stylesheet to
        stylesheet: The stylesheet string to apply
    """
    widget.setStyleSheet(stylesheet)


def get_title_font() -> QFont:
    """
    Returns a QFont configured for titles.
    
    Returns:
        A font configured for titles
    """
    font = QFont(FONT_FAMILY, TITLE_FONT_SIZE)
    font.setBold(True)
    return font


def get_normal_font() -> QFont:
    """
    Returns a QFont configured for normal text.
    
    Returns:
        A font configured for normal text
    """
    return QFont(FONT_FAMILY, NORMAL_FONT_SIZE)


def get_small_font() -> QFont:
    """
    Returns a QFont configured for small text.
    
    Returns:
        A font configured for small text
    """
    return QFont(FONT_FAMILY, SMALL_FONT_SIZE)


def create_standard_button(text: str, is_primary: bool = False) -> QPushButton:
    """
    Creates and returns a QPushButton with standard styling.
    
    Args:
        text: The button text
        is_primary: Whether this is a primary action button (True) or a secondary button (False)
    
    Returns:
        A styled button
    """
    button = QPushButton(text)
    
    if is_primary:
        button.setStyleSheet(f"""
            QPushButton {{
                background-color: {PRIMARY_COLOR.name()};
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
            }}
            QPushButton:hover {{
                background-color: {PRIMARY_COLOR.darker(110).name()};
            }}
            QPushButton:pressed {{
                background-color: {PRIMARY_COLOR.darker(120).name()};
            }}
        """)
    else:
        button.setStyleSheet(f"""
            QPushButton {{
                background-color: {SECONDARY_COLOR.name()};
                color: #333;
                border: 1px solid #ccc;
                padding: 8px 16px;
                border-radius: 4px;
            }}
            QPushButton:hover {{
                background-color: {SECONDARY_COLOR.darker(105).name()};
            }}
            QPushButton:pressed {{
                background-color: {SECONDARY_COLOR.darker(110).name()};
            }}
        """)
    
    button.setFont(get_normal_font())
    return button


def style_table_view(table_view: QTableView) -> None:
    """
    Applies standardized styling to a QTableView.
    
    Args:
        table_view: The table view to style
    """
    table_view.setAlternatingRowColors(True)
    table_view.setStyleSheet(f"""
        QTableView {{
            border: 1px solid #ddd;
            gridline-color: #ddd;
            background-color: white;
        }}
        QHeaderView::section {{
            background-color: {SECONDARY_COLOR.name()};
            color: #333;
            padding: 5px;
            border: 1px solid #ddd;
            font-weight: bold;
        }}
        QTableView::item {{
            padding: 5px;
        }}
        QTableView::item:selected {{
            background-color: {PRIMARY_COLOR.lighter(160).name()};
            color: black;
        }}
        QTableView::item:selected:active {{
            background-color: {PRIMARY_COLOR.lighter(140).name()};
            color: black;
        }}
    """)
    
    # Configure grid style
    table_view.setShowGrid(True)
    table_view.setGridStyle(Qt.PenStyle.SolidLine)
    
    # Adjust selection behavior
    table_view.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
    table_view.setSelectionMode(QTableView.SelectionMode.SingleSelection)


def get_plot_style() -> Dict[str, Any]:
    """
    Returns a dictionary of plot styling parameters for consistent visualization.
    
    Returns:
        A dictionary with plot style parameters
    """
    return {
        'figure.figsize': (8, 6),
        'font.family': FONT_FAMILY,
        'font.size': NORMAL_FONT_SIZE,
        'axes.titlesize': TITLE_FONT_SIZE,
        'axes.labelsize': NORMAL_FONT_SIZE,
        'xtick.labelsize': SMALL_FONT_SIZE,
        'ytick.labelsize': SMALL_FONT_SIZE,
        'legend.fontsize': SMALL_FONT_SIZE,
        'lines.linewidth': 2,
        'grid.linestyle': '--',
        'grid.alpha': 0.7,
        'axes.grid': True,
        'axes.edgecolor': '#333333',
        'axes.linewidth': 1.2,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.prop_cycle': 'cycler("color", ["#4878CF", "#6ACC65", "#D65F5F", "#B47CC7", "#C4AD66", "#77BEDB"])',
    }


def style_group_box(group_box: QGroupBox) -> None:
    """
    Applies standardized styling to a QGroupBox.
    
    Args:
        group_box: The group box to style
    """
    group_box.setStyleSheet(f"""
        QGroupBox {{
            font-weight: bold;
            border: 1px solid #ddd;
            border-radius: 5px;
            margin-top: 1.2em;
            padding-top: 0.8em;
            padding: {DEFAULT_MARGIN}px;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 3px;
            left: 10px;
        }}
    """)
    
    group_box.setFont(get_normal_font())
    
    # Configure margins and spacing if the group box contains a layout
    layout = group_box.layout()
    if layout:
        layout.setContentsMargins(DEFAULT_MARGIN, DEFAULT_MARGIN, DEFAULT_MARGIN, DEFAULT_MARGIN)
        layout.setSpacing(DEFAULT_SPACING)


def apply_theme(widget: QWidget, theme_name: str) -> None:
    """
    Applies the specified theme (light or dark) to the given widget and its children.
    
    Args:
        widget: The widget to apply the theme to
        theme_name: The name of the theme ('light' or 'dark')
    """
    if theme_name not in ['light', 'dark']:
        theme_name = 'light'  # Default to light theme
    
    palette = QPalette()
    
    if theme_name == 'light':
        # Light theme colors
        palette.setColor(QPalette.ColorRole.Window, QColor(248, 249, 250))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(33, 37, 41))
        palette.setColor(QPalette.ColorRole.Base, QColor(255, 255, 255))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(242, 242, 242))
        palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(255, 255, 255))
        palette.setColor(QPalette.ColorRole.ToolTipText, QColor(33, 37, 41))
        palette.setColor(QPalette.ColorRole.Text, QColor(33, 37, 41))
        palette.setColor(QPalette.ColorRole.Button, QColor(240, 240, 240))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor(33, 37, 41))
        palette.setColor(QPalette.ColorRole.Link, PRIMARY_COLOR)
        palette.setColor(QPalette.ColorRole.Highlight, PRIMARY_COLOR)
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
        
        widget.setStyleSheet(f"""
            QWidget {{
                font-family: {FONT_FAMILY};
                color: #212529;
                background-color: #f8f9fa;
            }}
            QLabel {{
                color: #212529;
            }}
            QToolTip {{
                background-color: white;
                color: #212529;
                border: 1px solid #ddd;
            }}
        """)
    else:
        # Dark theme colors
        palette.setColor(QPalette.ColorRole.Window, QColor(52, 58, 64))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(248, 249, 250))
        palette.setColor(QPalette.ColorRole.Base, QColor(33, 37, 41))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(46, 52, 58))
        palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(33, 37, 41))
        palette.setColor(QPalette.ColorRole.ToolTipText, QColor(248, 249, 250))
        palette.setColor(QPalette.ColorRole.Text, QColor(248, 249, 250))
        palette.setColor(QPalette.ColorRole.Button, QColor(52, 58, 64))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor(248, 249, 250))
        palette.setColor(QPalette.ColorRole.Link, QColor(77, 171, 247))
        palette.setColor(QPalette.ColorRole.Highlight, QColor(77, 171, 247))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor(33, 37, 41))
        
        widget.setStyleSheet(f"""
            QWidget {{
                font-family: {FONT_FAMILY};
                color: #f8f9fa;
                background-color: #343a40;
            }}
            QLabel {{
                color: #f8f9fa;
            }}
            QToolTip {{
                background-color: #212529;
                color: #f8f9fa;
                border: 1px solid #495057;
            }}
        """)
    
    widget.setPalette(palette)
    
    # Apply theme to all child widgets
    for child in widget.findChildren(QWidget):
        child.setPalette(palette)


class ColorPalette:
    """
    Class defining color palettes for light and dark themes.
    """
    
    # Light theme color palette
    LIGHT_PALETTE = {
        'background': QColor(248, 249, 250),
        'foreground': QColor(33, 37, 41),
        'surface': QColor(255, 255, 255),
        'surface_alt': QColor(242, 242, 242),
        'primary': PRIMARY_COLOR,
        'secondary': SECONDARY_COLOR,
        'success': SUCCESS_COLOR,
        'warning': WARNING_COLOR,
        'error': ERROR_COLOR,
        'border': QColor(222, 226, 230),
        'hover': QColor(232, 236, 240),
        'selected': PRIMARY_COLOR.lighter(140),
        'disabled': QColor(173, 181, 189),
    }
    
    # Dark theme color palette
    DARK_PALETTE = {
        'background': QColor(52, 58, 64),
        'foreground': QColor(248, 249, 250),
        'surface': QColor(33, 37, 41),
        'surface_alt': QColor(46, 52, 58),
        'primary': QColor(77, 171, 247),
        'secondary': QColor(73, 80, 87),
        'success': QColor(72, 187, 120),
        'warning': QColor(255, 205, 86),
        'error': QColor(255, 99, 132),
        'border': QColor(73, 80, 87),
        'hover': QColor(73, 80, 87).lighter(120),
        'selected': QColor(77, 171, 247).darker(120),
        'disabled': QColor(108, 117, 125),
    }
    
    def __init__(self):
        """
        Initializes the color palette.
        """
        pass
    
    @staticmethod
    def get_palette(theme_name: str) -> Dict[str, QColor]:
        """
        Returns the color palette for the specified theme.
        
        Args:
            theme_name: The name of the theme ('light' or 'dark')
        
        Returns:
            A dictionary of color values for the theme
        """
        if theme_name == 'dark':
            return ColorPalette.DARK_PALETTE
        # Default to light theme
        return ColorPalette.LIGHT_PALETTE


@dataclass
class StyleConfig:
    """
    Dataclass containing style configuration for the application.
    """
    
    theme_name: str
    colors: Dict[str, QColor]
    fonts: Dict[str, QFont]
    spacing: Dict[str, int]
    stylesheets: Dict[str, str]
    
    def __init__(self, theme_name: str = 'light'):
        """
        Initializes the style configuration with default values.
        
        Args:
            theme_name: The theme name ('light' or 'dark'), defaults to 'light'
        """
        self.theme_name = theme_name
        
        # Initialize colors from the palette for the selected theme
        self.colors = ColorPalette.get_palette(theme_name)
        
        # Initialize fonts
        self.fonts = {
            'title': get_title_font(),
            'normal': get_normal_font(),
            'small': get_small_font(),
        }
        
        # Initialize spacing
        self.spacing = {
            'margin': DEFAULT_MARGIN,
            'spacing': DEFAULT_SPACING,
            'small_margin': DEFAULT_MARGIN // 2,
            'large_margin': DEFAULT_MARGIN * 2,
            'small_spacing': DEFAULT_SPACING // 2,
            'large_spacing': DEFAULT_SPACING * 2,
        }
        
        # Initialize stylesheets
        self._initialize_stylesheets()
    
    def _initialize_stylesheets(self) -> None:
        """
        Initializes the stylesheet dictionary with widget-specific stylesheets.
        """
        self.stylesheets = {}
        
        # Base widget stylesheet
        self.stylesheets['widget'] = f"""
            QWidget {{
                font-family: {FONT_FAMILY};
                color: {self.colors['foreground'].name()};
                background-color: {self.colors['background'].name()};
            }}
        """
        
        # Button stylesheets
        self.stylesheets['button_primary'] = f"""
            QPushButton {{
                background-color: {self.colors['primary'].name()};
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
            }}
            QPushButton:hover {{
                background-color: {self.colors['primary'].darker(110).name()};
            }}
            QPushButton:pressed {{
                background-color: {self.colors['primary'].darker(120).name()};
            }}
            QPushButton:disabled {{
                background-color: {self.colors['disabled'].name()};
                color: {self.colors['background'].name()};
            }}
        """
        
        self.stylesheets['button_secondary'] = f"""
            QPushButton {{
                background-color: {self.colors['secondary'].name()};
                color: {self.colors['foreground'].name()};
                border: 1px solid {self.colors['border'].name()};
                padding: 8px 16px;
                border-radius: 4px;
            }}
            QPushButton:hover {{
                background-color: {self.colors['hover'].name()};
            }}
            QPushButton:pressed {{
                background-color: {self.colors['hover'].darker(110).name()};
            }}
            QPushButton:disabled {{
                background-color: {self.colors['disabled'].name()};
                color: {self.colors['background'].name()};
            }}
        """
        
        # Table view stylesheet
        self.stylesheets['table_view'] = f"""
            QTableView {{
                border: 1px solid {self.colors['border'].name()};
                gridline-color: {self.colors['border'].name()};
                background-color: {self.colors['surface'].name()};
            }}
            QHeaderView::section {{
                background-color: {self.colors['secondary'].name()};
                color: {self.colors['foreground'].name()};
                padding: 5px;
                border: 1px solid {self.colors['border'].name()};
                font-weight: bold;
            }}
            QTableView::item {{
                padding: 5px;
            }}
            QTableView::item:selected {{
                background-color: {self.colors['selected'].name()};
                color: {self.colors['foreground'].name()};
            }}
            QTableView::item:selected:active {{
                background-color: {self.colors['selected'].darker(110).name()};
                color: {self.colors['foreground'].name()};
            }}
        """
        
        # Group box stylesheet
        self.stylesheets['group_box'] = f"""
            QGroupBox {{
                font-weight: bold;
                border: 1px solid {self.colors['border'].name()};
                border-radius: 5px;
                margin-top: 1.2em;
                padding-top: 0.8em;
                padding: {self.spacing['margin']}px;
                background-color: {self.colors['surface'].name()};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 3px;
                left: 10px;
                color: {self.colors['foreground'].name()};
            }}
        """
        
        # Line edit stylesheet
        self.stylesheets['line_edit'] = f"""
            QLineEdit {{
                border: 1px solid {self.colors['border'].name()};
                border-radius: 4px;
                padding: 5px;
                background-color: {self.colors['surface'].name()};
                color: {self.colors['foreground'].name()};
            }}
            QLineEdit:focus {{
                border: 1px solid {self.colors['primary'].name()};
            }}
            QLineEdit:disabled {{
                background-color: {self.colors['disabled'].name()};
                color: {self.colors['foreground'].darker(120).name()};
            }}
        """
        
        # Combo box stylesheet
        self.stylesheets['combo_box'] = f"""
            QComboBox {{
                border: 1px solid {self.colors['border'].name()};
                border-radius: 4px;
                padding: 5px;
                background-color: {self.colors['surface'].name()};
                color: {self.colors['foreground'].name()};
            }}
            QComboBox:hover {{
                border: 1px solid {self.colors['primary'].name()};
            }}
            QComboBox:disabled {{
                background-color: {self.colors['disabled'].name()};
                color: {self.colors['foreground'].darker(120).name()};
            }}
            QComboBox::drop-down {{
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 20px;
                border-left: 1px solid {self.colors['border'].name()};
            }}
            QComboBox QAbstractItemView {{
                border: 1px solid {self.colors['border'].name()};
                background-color: {self.colors['surface'].name()};
                color: {self.colors['foreground'].name()};
            }}
        """
        
        # Check box stylesheet
        self.stylesheets['check_box'] = f"""
            QCheckBox {{
                color: {self.colors['foreground'].name()};
                spacing: 5px;
            }}
            QCheckBox::indicator {{
                width: 16px;
                height: 16px;
                border: 1px solid {self.colors['border'].name()};
                border-radius: 3px;
                background-color: {self.colors['surface'].name()};
            }}
            QCheckBox::indicator:checked {{
                background-color: {self.colors['primary'].name()};
                image: url(:/icons/check.png);
            }}
            QCheckBox::indicator:hover {{
                border: 1px solid {self.colors['primary'].name()};
            }}
            QCheckBox:disabled {{
                color: {self.colors['disabled'].name()};
            }}
        """
        
        # Tabs stylesheet
        self.stylesheets['tab_widget'] = f"""
            QTabWidget::pane {{
                border: 1px solid {self.colors['border'].name()};
                background-color: {self.colors['surface'].name()};
            }}
            QTabBar::tab {{
                background-color: {self.colors['secondary'].name()};
                border: 1px solid {self.colors['border'].name()};
                padding: 8px 12px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }}
            QTabBar::tab:selected {{
                background-color: {self.colors['surface'].name()};
                border-bottom-color: {self.colors['surface'].name()};
            }}
            QTabBar::tab:hover {{
                background-color: {self.colors['hover'].name()};
            }}
        """
    
    def get_color(self, color_name: str) -> QColor:
        """
        Returns the QColor for the specified color name.
        
        Args:
            color_name: The name of the color to retrieve
        
        Returns:
            The requested color or a default color if not found
        """
        return self.colors.get(color_name, QColor(0, 0, 0))
    
    def get_font(self, font_type: str) -> QFont:
        """
        Returns the QFont for the specified font type.
        
        Args:
            font_type: The type of font to retrieve
        
        Returns:
            The requested font or a default font if not found
        """
        return self.fonts.get(font_type, get_normal_font())
    
    def get_stylesheet(self, widget_type: str) -> str:
        """
        Returns the stylesheet for the specified widget type.
        
        Args:
            widget_type: The type of widget to get the stylesheet for
        
        Returns:
            The widget stylesheet or an empty string if not found
        """
        return self.stylesheets.get(widget_type, "")
    
    def switch_theme(self, new_theme: str) -> None:
        """
        Switches to the specified theme and updates all theme-related properties.
        
        Args:
            new_theme: The name of the new theme ('light' or 'dark')
        """
        if new_theme not in ['light', 'dark']:
            return
        
        self.theme_name = new_theme
        self.colors = ColorPalette.get_palette(new_theme)
        self._initialize_stylesheets()