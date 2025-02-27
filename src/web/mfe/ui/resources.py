"""
MFE UI Resources Management

This module manages UI resources including icons, images, and theme assets for the MFE application 
using PyQt6. It provides centralized access to compiled Qt resources and file-based assets.
"""

import os
from pathlib import Path
from PyQt6 import QtCore, QtGui

# Import the assets module to access UI asset directory
from . import assets

# Constants for resource paths
RESOURCE_PATH = str(assets.ASSETS_DIR)
ICON_PATH = str(assets.ICONS_DIR)
IMAGE_PATH = str(assets.IMAGES_DIR)

# Resource initialization flag
RESOURCES_INITIALIZED = False


def initialize_resources() -> bool:
    """
    Initializes and registers Qt resources for the application.
    
    Returns:
        bool: True if resources were successfully initialized, False otherwise.
    """
    global RESOURCES_INITIALIZED
    
    # Check if resources are already initialized
    if RESOURCES_INITIALIZED:
        return True
    
    try:
        # Register Qt resources if a compiled resource file is available
        # This is optional and will fall back to file-based resources if not available
        resource_file = os.path.join(RESOURCE_PATH, "mfe_resources.rcc")
        if os.path.exists(resource_file):
            if not QtCore.QResource.registerResource(resource_file):
                print(f"Warning: Failed to register resource file: {resource_file}")
        
        # Verify that resource paths exist
        if not os.path.isdir(ICON_PATH):
            raise FileNotFoundError(f"Icon directory not found: {ICON_PATH}")
        
        if not os.path.isdir(IMAGE_PATH):
            raise FileNotFoundError(f"Image directory not found: {IMAGE_PATH}")
        
        # Set flag to indicate resources are initialized
        RESOURCES_INITIALIZED = True
        return True
        
    except Exception as e:
        print(f"Error initializing resources: {e}")
        return False


def get_icon(icon_name: str) -> QtGui.QIcon:
    """
    Returns a QIcon object for the specified icon name.
    
    Args:
        icon_name: Name of the icon file (without extension).
        
    Returns:
        QIcon: Icon object for the specified name.
    """
    # Ensure resources are initialized
    if not RESOURCES_INITIALIZED:
        initialize_resources()
    
    # Add extension if not present
    if '.' not in icon_name:
        icon_name = f"{icon_name}.png"
    
    # Try to load from Qt resources first
    qt_resource_path = f":/icons/{icon_name}"
    icon = QtGui.QIcon(qt_resource_path)
    if not icon.isNull():
        return icon
    
    # Fall back to file system
    file_path = os.path.join(ICON_PATH, icon_name)
    if os.path.exists(file_path):
        return QtGui.QIcon(file_path)
    
    # Return an empty icon if not found
    print(f"Warning: Icon not found: {icon_name}")
    return QtGui.QIcon()


def get_pixmap(image_name: str) -> QtGui.QPixmap:
    """
    Returns a QPixmap object for the specified image name.
    
    Args:
        image_name: Name of the image file (without extension).
        
    Returns:
        QPixmap: Pixmap object for the specified image.
    """
    # Ensure resources are initialized
    if not RESOURCES_INITIALIZED:
        initialize_resources()
    
    # Add extension if not present
    if '.' not in image_name:
        image_name = f"{image_name}.png"
    
    # Try to load from Qt resources first
    qt_resource_path = f":/images/{image_name}"
    pixmap = QtGui.QPixmap(qt_resource_path)
    if not pixmap.isNull():
        return pixmap
    
    # Fall back to file system
    file_path = os.path.join(IMAGE_PATH, image_name)
    if os.path.exists(file_path):
        return QtGui.QPixmap(file_path)
    
    # Return an empty pixmap if not found
    print(f"Warning: Image not found: {image_name}")
    return QtGui.QPixmap()


def get_resource_path(resource_name: str, resource_type: str) -> str:
    """
    Returns the absolute path to a resource file.
    
    Args:
        resource_name: Name of the resource file.
        resource_type: Type of resource ('icon', 'image', etc.)
        
    Returns:
        str: Absolute path to the resource file.
    """
    # Ensure resources are initialized
    if not RESOURCES_INITIALIZED:
        initialize_resources()
    
    # Add extension if not present
    if '.' not in resource_name:
        resource_name = f"{resource_name}.png"
    
    # Determine base path based on resource type
    if resource_type.lower() == 'icon':
        base_path = ICON_PATH
    elif resource_type.lower() == 'image':
        base_path = IMAGE_PATH
    else:
        base_path = RESOURCE_PATH
    
    # Construct and verify the path
    resource_path = os.path.join(base_path, resource_name)
    
    # Check if resource exists
    if not os.path.exists(resource_path):
        print(f"Warning: Resource not found: {resource_path}")
    
    return resource_path


def resource_exists(resource_name: str, resource_type: str) -> bool:
    """
    Checks if a resource exists either in Qt resources or the file system.
    
    Args:
        resource_name: Name of the resource file.
        resource_type: Type of resource ('icon', 'image', etc.)
        
    Returns:
        bool: True if resource exists, False otherwise.
    """
    # Ensure resources are initialized
    if not RESOURCES_INITIALIZED:
        initialize_resources()
    
    # Add extension if not present
    if '.' not in resource_name:
        resource_name = f"{resource_name}.png"
    
    # Check Qt resources first
    qt_resource_path = f":/{resource_type}s/{resource_name}"
    resource = QtCore.QResource(qt_resource_path)
    if resource.isValid():
        return True
    
    # Then check file system
    if resource_type.lower() == 'icon':
        file_path = os.path.join(ICON_PATH, resource_name)
    elif resource_type.lower() == 'image':
        file_path = os.path.join(IMAGE_PATH, resource_name)
    else:
        file_path = os.path.join(RESOURCE_PATH, resource_name)
    
    return os.path.exists(file_path)