"""
PyQt6 signal classes for asynchronous UI operations in the MFE Toolbox.

This module defines signal container classes that facilitate communication between worker
threads and the main UI thread. These signals enable non-blocking UI updates during
long-running computational tasks, progress reporting, and error handling.
"""
from PyQt6.QtCore import QObject, pyqtSignal
import logging
import typing  # Used for type annotations

# Configure module logger
logger = logging.getLogger(__name__)


class TaskSignals(QObject):
    """
    Signal container for worker task lifecycle events, providing a communication 
    channel between worker threads and the main thread.
    """
    
    # Task lifecycle signals
    started = pyqtSignal()
    result = pyqtSignal(object)
    error = pyqtSignal(Exception)
    progress = pyqtSignal(float)
    finished = pyqtSignal()
    cancelled = pyqtSignal()
    
    def __init__(self) -> None:
        """Initialize the task signals with PyQt signal definitions."""
        super().__init__()


class ModelSignals(QObject):
    """
    Signal container for model estimation events, providing progress updates and
    results for econometric model computation.
    """
    
    # Model estimation signals
    estimation_started = pyqtSignal()
    estimation_result = pyqtSignal(object)
    estimation_progress = pyqtSignal(float)
    estimation_error = pyqtSignal(Exception)
    estimation_finished = pyqtSignal()
    
    def __init__(self) -> None:
        """Initialize the model signals with PyQt signal definitions."""
        super().__init__()


class PlotSignals(QObject):
    """
    Signal container for plot update events, enabling asynchronous UI updates
    during visualization operations.
    """
    
    # Plot update signals
    update_started = pyqtSignal()
    update_complete = pyqtSignal(object)
    update_error = pyqtSignal(Exception)
    rendering_progress = pyqtSignal(float)
    
    def __init__(self) -> None:
        """Initialize the plot signals with PyQt signal definitions."""
        super().__init__()


class DataProcessingSignals(QObject):
    """
    Signal container for data processing events, providing progress updates
    during data transformations and filtering operations.
    """
    
    # Data processing signals
    processing_started = pyqtSignal()
    processing_result = pyqtSignal(object)
    processing_progress = pyqtSignal(float)
    processing_error = pyqtSignal(Exception)
    processing_finished = pyqtSignal()
    
    def __init__(self) -> None:
        """Initialize the data processing signals with PyQt signal definitions."""
        super().__init__()


class SignalFactory(QObject):
    """
    Factory class for creating specialized signal containers based on task type.
    """
    
    def __init__(self) -> None:
        """Initialize the signal factory."""
        super().__init__()
    
    def create_task_signals(self) -> TaskSignals:
        """
        Create a TaskSignals instance.
        
        Returns:
            TaskSignals: A new TaskSignals instance
        """
        return TaskSignals()
    
    def create_model_signals(self) -> ModelSignals:
        """
        Create a ModelSignals instance.
        
        Returns:
            ModelSignals: A new ModelSignals instance
        """
        return ModelSignals()
    
    def create_plot_signals(self) -> PlotSignals:
        """
        Create a PlotSignals instance.
        
        Returns:
            PlotSignals: A new PlotSignals instance
        """
        return PlotSignals()
    
    def create_data_processing_signals(self) -> DataProcessingSignals:
        """
        Create a DataProcessingSignals instance.
        
        Returns:
            DataProcessingSignals: A new DataProcessingSignals instance
        """
        return DataProcessingSignals()