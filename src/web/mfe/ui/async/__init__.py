"""
Initialization file for the async package that provides asynchronous operation support for the MFE Toolbox UI.
Re-exports key components from the async submodules to provide a clean interface for asynchronous operations
in the PyQt6-based GUI.
"""

from PyQt6.QtCore import QObject, pyqtSignal
from typing import Any, Optional

# Import existing components
from .worker import AsyncWorker
from .task_manager import TaskManager

# Define signal classes for asynchronous operation feedback
class ProgressSignal(QObject):
    """Signal class for tracking and reporting progress from asynchronous operations."""
    progress = pyqtSignal(float)
    
    def __init__(self) -> None:
        """Initialize the progress signal."""
        super().__init__()
        
    def emit(self, value: float) -> None:
        """
        Emit a progress update signal.
        
        Parameters
        ----------
        value : float
            Progress value between 0.0 and 1.0
        """
        self.progress.emit(value)

class ResultSignal(QObject):
    """Signal class for transmitting results from completed asynchronous operations."""
    result = pyqtSignal(object)
    
    def __init__(self) -> None:
        """Initialize the result signal."""
        super().__init__()
        
    def emit(self, result: Any) -> None:
        """
        Emit a result signal with the operation result.
        
        Parameters
        ----------
        result : Any
            Result of the asynchronous operation
        """
        self.result.emit(result)

class ErrorSignal(QObject):
    """Signal class for communicating errors from asynchronous operations."""
    error = pyqtSignal(Exception)
    
    def __init__(self) -> None:
        """Initialize the error signal."""
        super().__init__()
        
    def emit(self, error: Exception) -> None:
        """
        Emit an error signal with the exception.
        
        Parameters
        ----------
        error : Exception
            Exception from the asynchronous operation
        """
        self.error.emit(error)

# Define result container class for asynchronous operation outputs
class AsyncResult:
    """Result container class for asynchronous operation outputs."""
    
    def __init__(self, result: Any = None, error: Optional[Exception] = None) -> None:
        """
        Initialize an async result container.
        
        Parameters
        ----------
        result : Any, optional
            Result of the asynchronous operation
        error : Optional[Exception], optional
            Exception from the asynchronous operation, if any
        """
        self.result = result
        self.error = error
        self.is_success = error is None
        
    @property
    def succeeded(self) -> bool:
        """
        Check if the operation succeeded.
        
        Returns
        -------
        bool
            True if the operation succeeded, False otherwise
        """
        return self.is_success
    
    @property
    def failed(self) -> bool:
        """
        Check if the operation failed.
        
        Returns
        -------
        bool
            True if the operation failed, False otherwise
        """
        return not self.is_success

# Export the components
__all__ = [
    'ProgressSignal',
    'ResultSignal', 
    'ErrorSignal',
    'AsyncWorker',
    'AsyncResult',
    'TaskManager'
]