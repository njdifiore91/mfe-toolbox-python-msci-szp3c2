"""
MFE Toolbox Asset Management

This module provides path resolution and resource loading utilities for UI assets 
including icons and images for the PyQt6-based GUI.
"""

from pathlib import Path
from PyQt6.QtGui import QIcon, QPixmap

# Directory paths
ASSETS_DIR = Path(__file__).parent
ICONS_DIR = ASSETS_DIR / 'icons'
IMAGES_DIR = ASSETS_DIR / 'images'

def get_icon_path(icon_name: str) -> Path:
    """
    Returns the full path to an icon file.
    
    Args:
        icon_name: Name of the icon file. If no extension is provided, '.png' is appended.
        
    Returns:
        Path object pointing to the icon file.
    """
    # Check if icon_name includes file extension
    if '.' not in icon_name:
        icon_name = f"{icon_name}.png"
    
    return ICONS_DIR / icon_name

def get_image_path(image_name: str) -> Path:
    """
    Returns the full path to an image file.
    
    Args:
        image_name: Name of the image file. If no extension is provided, '.png' is appended.
        
    Returns:
        Path object pointing to the image file.
    """
    # Check if image_name includes file extension
    if '.' not in image_name:
        image_name = f"{image_name}.png"
    
    return IMAGES_DIR / image_name

def load_icon(icon_name: str) -> QIcon:
    """
    Loads and returns a QIcon from the icons directory.
    
    Args:
        icon_name: Name of the icon file. If no extension is provided, '.png' is appended.
        
    Returns:
        QIcon object initialized with the specified icon file.
    """
    icon_path = get_icon_path(icon_name)
    return QIcon(str(icon_path))

def load_image(image_name: str) -> QPixmap:
    """
    Loads and returns a QPixmap from the images directory.
    
    Args:
        image_name: Name of the image file. If no extension is provided, '.png' is appended.
        
    Returns:
        QPixmap object initialized with the specified image file.
    """
    image_path = get_image_path(image_name)
    return QPixmap(str(image_path))