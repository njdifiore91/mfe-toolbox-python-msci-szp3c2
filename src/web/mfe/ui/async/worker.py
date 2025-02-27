"""
MFE Toolbox UI - Asynchronous Worker Module

This module implements Worker and AsyncWorker classes for executing computationally 
intensive tasks in background threads to maintain UI responsiveness in the MFE Toolbox.
It provides signal-based communication between worker threads and the main UI thread,
with support for progress reporting, error handling, and task cancellation.
"""

import asyncio
import enum
import logging
from typing import Any, Callable, Optional, Union

from PyQt6.QtCore import QObject, QRunnable, QThreadPool, pyqtSlot

from .signals import TaskSignals
from backend.mfe.utils.async_helpers import async_progress, handle_exceptions_async

# Configure module logger
logger = logging.getLogger(__name__)


class WorkerStatus(enum.Enum):
    """Enumeration of possible worker status values for tracking worker lifecycle state."""
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    FINISHED = "FINISHED"
    ERROR = "ERROR"
    CANCELLED = "CANCELLED"


class Worker(QRunnable):
    """
    Worker implementation for executing synchronous functions in a background thread 
    with signal-based communication to the UI thread.
    """
    
    def __init__(self, fn: Callable, *args: Any, **kwargs: Any):
        """
        Initialize a worker with the function to be executed and its arguments.
        
        Parameters
        ----------
        fn : callable
            The function to execute in the background thread
        *args : Any
            Positional arguments to pass to the function
        **kwargs : Any
            Keyword arguments to pass to the function
        """
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = TaskSignals()
        self.is_cancelled = False
        self.status = WorkerStatus.CREATED
        logger.debug(f"Worker initialized with function: {fn.__name__}")
    
    def run(self):
        """
        Execute the worker function in a background thread with progress, error handling, 
        and cancellation support.
        
        This method is called by QThreadPool when the worker is started.
        
        Returns
        -------
        Any
            Result of the function execution or None if error/cancelled
        """
        # Check if already cancelled
        if self.is_cancelled:
            self.signals.cancelled.emit()
            self.status = WorkerStatus.CANCELLED
            return None
        
        # Set status to running and emit started signal
        self.status = WorkerStatus.RUNNING
        self.signals.started.emit()
        logger.debug(f"Worker started: {self.fn.__name__}")
        
        result = None
        try:
            # Execute the function with args and kwargs
            result = self.fn(*self.args, **self.kwargs)
            
            # If not cancelled during execution, emit result signal
            if not self.is_cancelled:
                self.signals.result.emit(result)
                self.status = WorkerStatus.FINISHED
            else:
                self.status = WorkerStatus.CANCELLED
                self.signals.cancelled.emit()
        except Exception as e:
            # Handle exceptions and emit error signal
            logger.error(f"Worker error in {self.fn.__name__}: {str(e)}")
            self.signals.error.emit(e)
            self.status = WorkerStatus.ERROR
        finally:
            # Always emit finished signal
            if not self.is_cancelled:
                self.signals.finished.emit()
        
        return result
    
    def cancel(self):
        """
        Mark the worker as cancelled to prevent or stop execution.
        """
        self.is_cancelled = True
        logger.debug(f"Worker cancellation requested: {self.fn.__name__}")
        
        # If worker hasn't started yet, emit cancelled signal immediately
        if self.status == WorkerStatus.CREATED:
            self.signals.cancelled.emit()
        
        self.status = WorkerStatus.CANCELLED


class AsyncWorker(QRunnable):
    """
    Worker implementation for executing asynchronous coroutine functions in a background 
    thread with signal-based communication to the UI thread.
    """
    
    def __init__(self, coro_fn: Callable, *args: Any, **kwargs: Any):
        """
        Initialize an async worker with the coroutine function to be executed and its arguments.
        
        Parameters
        ----------
        coro_fn : callable
            The coroutine function to execute in the background thread
        *args : Any
            Positional arguments to pass to the coroutine function
        **kwargs : Any
            Keyword arguments to pass to the coroutine function
        
        Raises
        ------
        TypeError
            If coro_fn is not a coroutine function
        """
        super().__init__()
        
        # Verify that coro_fn is a coroutine function
        if not asyncio.iscoroutinefunction(coro_fn):
            raise TypeError(f"Expected a coroutine function, got {type(coro_fn).__name__}")
        
        self.coro_fn = coro_fn
        self.args = args
        self.kwargs = kwargs
        self.signals = TaskSignals()
        self.is_cancelled = False
        self.status = WorkerStatus.CREATED
        self._task = None
        self._cancel_event = None
        logger.debug(f"AsyncWorker initialized with coroutine: {coro_fn.__name__}")
    
    def run(self):
        """
        Execute the async coroutine function in a background thread with event loop, 
        progress, error handling, and cancellation support.
        
        This method is called by QThreadPool when the worker is started.
        
        Returns
        -------
        Any
            Result of the coroutine execution or None if error/cancelled
        """
        # Check if already cancelled
        if self.is_cancelled:
            self.signals.cancelled.emit()
            self.status = WorkerStatus.CANCELLED
            return None
        
        # Set status to running and emit started signal
        self.status = WorkerStatus.RUNNING
        self.signals.started.emit()
        logger.debug(f"AsyncWorker started: {self.coro_fn.__name__}")
        
        result = None
        
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Create cancel event for cooperative cancellation
            self._cancel_event = asyncio.Event()
            
            # Run the coroutine task
            self._task = loop.create_task(self._run_async_task())
            result = loop.run_until_complete(self._task)
            
            # If not cancelled during execution, emit result signal
            if not self.is_cancelled:
                self.signals.result.emit(result)
                self.status = WorkerStatus.FINISHED
            else:
                self.status = WorkerStatus.CANCELLED
                self.signals.cancelled.emit()
                
        except asyncio.CancelledError:
            # Handle cancellation
            logger.debug(f"AsyncWorker was cancelled: {self.coro_fn.__name__}")
            self.is_cancelled = True
            self.status = WorkerStatus.CANCELLED
            self.signals.cancelled.emit()
            
        except Exception as e:
            # Handle exceptions and emit error signal
            logger.error(f"AsyncWorker error in {self.coro_fn.__name__}: {str(e)}")
            self.signals.error.emit(e)
            self.status = WorkerStatus.ERROR
            
        finally:
            # Always emit finished signal
            if not self.is_cancelled:
                self.signals.finished.emit()
            
            # Clean up resources
            loop.close()
            self._task = None
            self._cancel_event = None
        
        return result
    
    def cancel(self):
        """
        Cancel the running async task cooperatively.
        """
        self.is_cancelled = True
        logger.debug(f"AsyncWorker cancellation requested: {self.coro_fn.__name__}")
        
        # Set cancel event to trigger cooperative cancellation
        if self._cancel_event:
            self._cancel_event.set()
        
        # Cancel the task if it exists
        if self._task:
            self._task.cancel()
        
        # If worker hasn't started yet, emit cancelled signal immediately
        if self.status == WorkerStatus.CREATED:
            self.signals.cancelled.emit()
        
        self.status = WorkerStatus.CANCELLED
    
    def _progress_callback(self, progress: float):
        """
        Callback function to receive progress updates from the coroutine.
        
        Parameters
        ----------
        progress : float
            Progress value between 0.0 and 1.0
        
        Raises
        ------
        asyncio.CancelledError
            If the worker has been cancelled
        ValueError
            If the progress value is outside the valid range
        """
        # Check if worker has been cancelled
        if self.is_cancelled:
            raise asyncio.CancelledError()
        
        # Validate progress value
        if not (0.0 <= progress <= 1.0):
            raise ValueError(f"Progress value must be between 0.0 and 1.0, got {progress}")
        
        # Emit progress signal
        self.signals.progress.emit(progress)
        logger.debug(f"AsyncWorker progress: {progress * 100:.1f}%")
    
    async def _run_async_task(self):
        """
        Internal method to run the async task and handle its lifecycle.
        
        Returns
        -------
        Any
            Result of the coroutine execution
        
        Raises
        ------
        asyncio.CancelledError
            If cancellation is requested during execution
        """
        # Set up progress callback for the coroutine
        if 'progress_callback' not in self.kwargs:
            self.kwargs['progress_callback'] = self._progress_callback
        
        # Create a wrapper to monitor cancellation
        async def wrapper():
            # Check for cancellation before starting
            if self.is_cancelled:
                raise asyncio.CancelledError()
            
            # Run the coroutine function
            task_result = await self.coro_fn(*self.args, **self.kwargs)
            
            # Periodically check for cancellation during execution
            while not self._cancel_event.is_set():
                return task_result
            
            # If cancel event is set, raise CancelledError
            raise asyncio.CancelledError()
        
        return await wrapper()


class WorkerManager(QObject):
    """
    Manager class for creating, tracking, and coordinating Worker and AsyncWorker instances.
    """
    
    def __init__(self):
        """
        Initialize the worker manager with access to the global thread pool.
        """
        super().__init__()
        self._active_workers = []
        self._thread_pool = QThreadPool.globalInstance()
        logger.debug(f"WorkerManager initialized with thread pool max count: {self._thread_pool.maxThreadCount()}")
    
    def create_worker(self, fn: Callable, *args: Any, **kwargs: Any) -> Worker:
        """
        Create a Worker instance for a synchronous function.
        
        Parameters
        ----------
        fn : callable
            Function to execute in a background thread
        *args : Any
            Positional arguments to pass to the function
        **kwargs : Any
            Keyword arguments to pass to the function
            
        Returns
        -------
        Worker
            Created Worker instance
        """
        worker = Worker(fn, *args, **kwargs)
        worker.signals.finished.connect(self._remove_worker)
        self._active_workers.append(worker)
        return worker
    
    def create_async_worker(self, coro_fn: Callable, *args: Any, **kwargs: Any) -> AsyncWorker:
        """
        Create an AsyncWorker instance for an asynchronous coroutine function.
        
        Parameters
        ----------
        coro_fn : callable
            Coroutine function to execute in a background thread
        *args : Any
            Positional arguments to pass to the coroutine function
        **kwargs : Any
            Keyword arguments to pass to the coroutine function
            
        Returns
        -------
        AsyncWorker
            Created AsyncWorker instance
        """
        worker = AsyncWorker(coro_fn, *args, **kwargs)
        worker.signals.finished.connect(self._remove_worker)
        self._active_workers.append(worker)
        return worker
    
    def start_worker(self, worker: Union[Worker, AsyncWorker]):
        """
        Start a worker in the thread pool.
        
        Parameters
        ----------
        worker : Union[Worker, AsyncWorker]
            Worker instance to start
            
        Raises
        ------
        TypeError
            If worker is not a Worker or AsyncWorker instance
        """
        if not isinstance(worker, (Worker, AsyncWorker)):
            raise TypeError(f"Expected Worker or AsyncWorker instance, got {type(worker).__name__}")
        
        self._thread_pool.start(worker)
        logger.debug(f"Worker started in thread pool")
    
    def cancel_all(self) -> int:
        """
        Cancel all active workers.
        
        Returns
        -------
        int
            Number of workers cancelled
        """
        cancelled_count = 0
        for worker in self._active_workers[:]:  # Use a copy of the list
            worker.cancel()
            cancelled_count += 1
        
        logger.debug(f"Requested cancellation of {cancelled_count} workers")
        return cancelled_count
    
    def active_count(self) -> int:
        """
        Get the count of currently active workers.
        
        Returns
        -------
        int
            Number of active workers
        """
        return len(self._active_workers)
    
    @pyqtSlot()
    def _remove_worker(self):
        """
        Remove a worker from the active workers list when finished.
        """
        sender = self.sender()
        if sender in self._active_workers:
            self._active_workers.remove(sender)
            logger.debug(f"Worker removed from active workers list")