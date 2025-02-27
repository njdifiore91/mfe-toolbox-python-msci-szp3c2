"""
MFE Toolbox - Asynchronous Helpers Module

This module provides utilities for asynchronous operations in the MFE Toolbox,
including task management, parallelization, exception handling, and progress
reporting for computationally intensive econometric operations.

The module implements Python 3.12 async/await patterns and strict type hints
for robust, production-ready asynchronous execution.
"""

import asyncio  # Python 3.12
import concurrent.futures  # Python 3.12
import functools  # Python 3.12
import inspect  # Python 3.12
import logging  # Python 3.12
import time  # Python 3.12
from typing import (  # Python 3.12
    Any, Callable, Coroutine, Dict, List, 
    Optional, TypeVar, Union, Type
)

from .validation import is_positive_integer

# Setup module logger
logger = logging.getLogger(__name__)

# Type variables for better type hinting
T = TypeVar('T')


def async_generator(generator_func: Callable) -> Callable:
    """
    Decorator to convert a synchronous generator to an asynchronous generator.
    
    Parameters
    ----------
    generator_func : Callable
        A synchronous generator function to convert
        
    Returns
    -------
    Callable
        Wrapped async generator function
    """
    if not callable(generator_func):
        raise TypeError("Expected a callable generator function")
    
    @functools.wraps(generator_func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        gen = generator_func(*args, **kwargs)
        
        if not inspect.isgenerator(gen):
            raise TypeError(f"{generator_func.__name__} must return a generator")
        
        for item in gen:
            await asyncio.sleep(0)  # Allow other tasks to run
            yield item
    
    return wrapper


async def run_in_executor(func: Callable, *args: Any, **kwargs: Any) -> Any:
    """
    Run a synchronous function in a thread pool executor to avoid blocking the event loop.
    
    This is useful for CPU-bound operations or operations that would block the event loop.
    
    Parameters
    ----------
    func : Callable
        The synchronous function to run in the executor
    *args : Any
        Positional arguments to pass to the function
    **kwargs : Any
        Keyword arguments to pass to the function
        
    Returns
    -------
    Any
        The result of the function call
    """
    if not callable(func):
        raise TypeError("Expected a callable function")
    
    loop = asyncio.get_event_loop()
    partial_func = functools.partial(func, *args, **kwargs)
    return await loop.run_in_executor(None, partial_func)


async def gather_with_concurrency(limit: int, *tasks: Coroutine) -> List[Any]:
    """
    Run multiple coroutines concurrently with a limit on the number of concurrent tasks.
    
    Parameters
    ----------
    limit : int
        Maximum number of coroutines to run concurrently
    *tasks : Coroutine
        Coroutines to run with the concurrency limit
        
    Returns
    -------
    List[Any]
        List of results from all coroutines in the order they were submitted
    """
    is_positive_integer(limit, param_name="limit")
    
    semaphore = asyncio.Semaphore(limit)
    
    async def sem_task(task: Coroutine) -> Any:
        async with semaphore:
            return await task
    
    return await asyncio.gather(*(sem_task(task) for task in tasks))


def retry_async(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions_to_retry: List[Type[Exception]] = None
) -> Callable:
    """
    Decorator for retrying an async function with exponential backoff.
    
    Parameters
    ----------
    max_retries : int, default=3
        Maximum number of retry attempts
    initial_delay : float, default=1.0
        Initial delay between retries in seconds
    backoff_factor : float, default=2.0
        Factor by which the delay increases with each retry
    exceptions_to_retry : List[Type[Exception]], default=None
        List of exception types to retry on. If None, retries on all exceptions.
        
    Returns
    -------
    Callable
        Decorator function that wraps an async function with retry logic
    """
    if not isinstance(max_retries, int) or max_retries < 0:
        raise ValueError("max_retries must be a non-negative integer")
    
    if not isinstance(initial_delay, (int, float)) or initial_delay <= 0:
        raise ValueError("initial_delay must be a positive number")
    
    if not isinstance(backoff_factor, (int, float)) or backoff_factor <= 0:
        raise ValueError("backoff_factor must be a positive number")
    
    if exceptions_to_retry is None:
        exceptions_to_retry = [Exception]
    elif not isinstance(exceptions_to_retry, list) or not all(issubclass(e, Exception) for e in exceptions_to_retry):
        raise ValueError("exceptions_to_retry must be a list of exception types")
    
    def decorator(func: Callable) -> Callable:
        if not callable(func):
            raise TypeError("Expected a callable function")
        
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception = None
            delay = initial_delay
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except tuple(exceptions_to_retry) as e:
                    last_exception = e
                    
                    if attempt < max_retries:
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries + 1} for {func.__name__} failed: {str(e)}. "
                            f"Retrying in {delay:.2f} seconds..."
                        )
                        await asyncio.sleep(delay)
                        delay *= backoff_factor
                    else:
                        logger.error(
                            f"All {max_retries + 1} attempts for {func.__name__} failed. "
                            f"Last error: {str(e)}"
                        )
            
            if last_exception:
                raise last_exception
            
            raise RuntimeError("Unexpected error in retry logic")
        
        return wrapper
    
    return decorator


def timeout_async(seconds: float) -> Callable:
    """
    Decorator to add a timeout to an async function.
    
    Parameters
    ----------
    seconds : float
        Timeout duration in seconds
        
    Returns
    -------
    Callable
        Decorator function that wraps an async function with timeout logic
    """
    if not isinstance(seconds, (int, float)) or seconds <= 0:
        raise ValueError("Timeout must be a positive number of seconds")
    
    def decorator(func: Callable) -> Callable:
        if not callable(func):
            raise TypeError("Expected a callable function")
        
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)
        
        return wrapper
    
    return decorator


def async_progress(
    progress_callback: Optional[Callable[[float], None]] = None
) -> Callable:
    """
    Decorator for reporting progress from an async function.
    
    The decorated function should accept a 'report_progress' callback as a keyword argument.
    
    Parameters
    ----------
    progress_callback : Optional[Callable[[float], None]], default=None
        Function to call with progress percentage (0-100)
        
    Returns
    -------
    Callable
        Decorator function that wraps an async function with progress reporting
    """
    if progress_callback is not None and not callable(progress_callback):
        raise TypeError("progress_callback must be callable if provided")
    
    def decorator(func: Callable) -> Callable:
        if not callable(func):
            raise TypeError("Expected a callable function")
        
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            if progress_callback:
                kwargs['report_progress'] = progress_callback
            return await func(*args, **kwargs)
        
        return wrapper
    
    return decorator


def handle_exceptions_async(
    handler: Optional[Callable[[Exception], None]] = None
) -> Callable:
    """
    Decorator for handling exceptions in async functions.
    
    Parameters
    ----------
    handler : Optional[Callable[[Exception], None]], default=None
        Function to call with the exception. If None, exceptions are logged and re-raised.
        
    Returns
    -------
    Callable
        Decorator function that wraps an async function with exception handling
    """
    if handler is not None and not callable(handler):
        raise TypeError("Exception handler must be callable if provided")
    
    def decorator(func: Callable) -> Callable:
        if not callable(func):
            raise TypeError("Expected a callable function")
        
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if handler:
                    handler(e)
                else:
                    logger.exception(f"Exception in {func.__name__}: {str(e)}")
                raise
        
        return wrapper
    
    return decorator


def create_task_with_name(coro: Coroutine, name: str) -> asyncio.Task:
    """
    Create a named asyncio task for better debugging.
    
    Parameters
    ----------
    coro : Coroutine
        Coroutine to convert to a task
    name : str
        Name to assign to the task
        
    Returns
    -------
    asyncio.Task
        Created task with the specified name
    """
    if not asyncio.iscoroutine(coro):
        raise TypeError("Expected a coroutine object")
    
    task = asyncio.create_task(coro)
    task.set_name(name)
    return task


def run_async(func: Callable, *args: Any, **kwargs: Any) -> Any:
    """
    Run an async function in a new event loop until complete.
    
    Useful for running async code from synchronous contexts.
    
    Parameters
    ----------
    func : Callable
        Async function to run
    *args : Any
        Positional arguments to pass to the function
    **kwargs : Any
        Keyword arguments to pass to the function
        
    Returns
    -------
    Any
        Result of the async function
    """
    if not callable(func):
        raise TypeError("Expected a callable function")
    
    # Create a new event loop
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(func(*args, **kwargs))
    finally:
        loop.close()


def async_to_sync(async_func: Callable) -> Callable:
    """
    Decorator to convert an async function to a synchronous function.
    
    Parameters
    ----------
    async_func : Callable
        Async function to convert
        
    Returns
    -------
    Callable
        Wrapped synchronous function
    """
    if not callable(async_func):
        raise TypeError("Expected a callable function")
    
    @functools.wraps(async_func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return run_async(async_func, *args, **kwargs)
    
    return wrapper


class AsyncTaskGroup:
    """
    Class for managing a group of related asyncio tasks.
    
    Provides methods for creating, tracking, and cancelling related tasks.
    Can be used as an async context manager.
    """
    
    def __init__(self) -> None:
        """
        Initialize an empty task group.
        """
        self._tasks: Dict[str, asyncio.Task] = {}
        self._entered = False
    
    def create_task(self, coro: Coroutine, name: str) -> asyncio.Task:
        """
        Create and add a named task to the group.
        
        Parameters
        ----------
        coro : Coroutine
            Coroutine to convert to a task
        name : str
            Name to assign to the task
            
        Returns
        -------
        asyncio.Task
            Created task
        """
        if not asyncio.iscoroutine(coro):
            raise TypeError("Expected a coroutine object")
        
        if name in self._tasks:
            raise RuntimeError(f"Task with name '{name}' already exists in this group")
        
        task = create_task_with_name(coro, name)
        self._tasks[name] = task
        return task
    
    def get_task(self, name: str) -> Optional[asyncio.Task]:
        """
        Get a task by name.
        
        Parameters
        ----------
        name : str
            Name of the task to retrieve
            
        Returns
        -------
        Optional[asyncio.Task]
            Task with the specified name or None if not found
        """
        return self._tasks.get(name)
    
    def running_tasks(self) -> int:
        """
        Count of currently running tasks.
        
        Returns
        -------
        int
            Number of running tasks
        """
        return sum(1 for task in self._tasks.values() if not task.done())
    
    def cancel_all(self) -> None:
        """
        Cancel all tasks in the group.
        """
        for name, task in list(self._tasks.items()):
            if not task.done():
                task.cancel()
        
        self._tasks.clear()
    
    async def __aenter__(self) -> 'AsyncTaskGroup':
        """
        Enter context manager (async with support).
        
        Returns
        -------
        AsyncTaskGroup
            This task group instance
        """
        self._entered = True
        return self
    
    async def __aexit__(
        self, 
        exc_type: Any, 
        exc_val: Any, 
        exc_tb: Any
    ) -> bool:
        """
        Exit context manager and cancel all tasks.
        
        Parameters
        ----------
        exc_type : Any
            Exception type if an exception was raised, None otherwise
        exc_val : Any
            Exception value if an exception was raised, None otherwise
        exc_tb : Any
            Exception traceback if an exception was raised, None otherwise
            
        Returns
        -------
        bool
            False to propagate exceptions
        """
        self.cancel_all()
        self._entered = False
        return False  # Propagate exceptions


class AsyncLimiter:
    """
    Semaphore-based limiter for controlling concurrency in async operations.
    """
    
    def __init__(self, limit: int) -> None:
        """
        Initialize limiter with specified concurrency limit.
        
        Parameters
        ----------
        limit : int
            Maximum number of concurrent operations allowed
        """
        is_positive_integer(limit, param_name="limit")
        
        self.limit = limit
        self._semaphore = asyncio.Semaphore(limit)
    
    def async_limit(self, func: Callable) -> Callable:
        """
        Decorator to limit concurrency of an async function.
        
        Parameters
        ----------
        func : Callable
            Async function to limit concurrency for
            
        Returns
        -------
        Callable
            Wrapped function with concurrency limiting
        """
        if not callable(func):
            raise TypeError("Expected a callable function")
        
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            async with self._semaphore:
                return await func(*args, **kwargs)
        
        return wrapper
    
    async def acquire(self) -> bool:
        """
        Acquire the semaphore.
        
        Returns
        -------
        bool
            True when the semaphore is acquired
        """
        return await self._semaphore.acquire()
    
    def release(self) -> None:
        """
        Release the semaphore.
        """
        self._semaphore.release()
    
    async def __aenter__(self) -> 'AsyncLimiter':
        """
        Enter context manager (async with support).
        
        Returns
        -------
        AsyncLimiter
            This limiter instance
        """
        await self.acquire()
        return self
    
    async def __aexit__(
        self, 
        exc_type: Any, 
        exc_val: Any, 
        exc_tb: Any
    ) -> bool:
        """
        Exit context manager and release semaphore.
        
        Parameters
        ----------
        exc_type : Any
            Exception type if an exception was raised, None otherwise
        exc_val : Any
            Exception value if an exception was raised, None otherwise
        exc_tb : Any
            Exception traceback if an exception was raised, None otherwise
            
        Returns
        -------
        bool
            False to propagate exceptions
        """
        self.release()
        return False  # Propagate exceptions


class AsyncTask:
    """
    Class representing an asynchronous task with progress tracking and cancellation support.
    """
    
    def __init__(
        self, 
        coro: Coroutine, 
        name: Optional[str] = None,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> None:
        """
        Initialize an async task with a coroutine.
        
        Parameters
        ----------
        coro : Coroutine
            Coroutine to convert to a task
        name : Optional[str], default=None
            Name to assign to the task. If None, a random name is generated.
        progress_callback : Optional[Callable[[float], None]], default=None
            Callback function for progress updates
        """
        if not asyncio.iscoroutine(coro):
            raise TypeError("Expected a coroutine object")
        
        import uuid
        self.name = name or f"task-{uuid.uuid4().hex[:8]}"
        self._progress_callback = progress_callback
        self._completed = False
        self._result = None
        self._exception = None
        
        # Create the task and add a done callback
        self._task = create_task_with_name(coro, self.name)
        self._task.add_done_callback(self._task_done_callback)
    
    def report_progress(self, percentage: float) -> None:
        """
        Report progress percentage.
        
        Parameters
        ----------
        percentage : float
            Progress percentage (0-100)
        """
        if not 0 <= percentage <= 100:
            raise ValueError("Progress percentage must be between 0 and 100")
        
        if self._progress_callback:
            self._progress_callback(percentage)
        else:
            logger.debug(f"Task '{self.name}' progress: {percentage:.1f}%")
    
    def cancel(self) -> bool:
        """
        Cancel the task.
        
        Returns
        -------
        bool
            True if task was canceled
        """
        return self._task.cancel()
    
    def done(self) -> bool:
        """
        Check if the task is done.
        
        Returns
        -------
        bool
            True if task is complete
        """
        return self._completed or self._task.done()
    
    def result(self) -> Any:
        """
        Get the result of the task.
        
        Returns
        -------
        Any
            Task result
        """
        if not self.done():
            raise RuntimeError("Task is not done, cannot get result")
        
        if self._exception:
            raise self._exception
        
        return self._result
    
    def exception(self) -> Optional[Exception]:
        """
        Get the exception raised by the task.
        
        Returns
        -------
        Optional[Exception]
            Exception raised or None
        """
        if not self.done():
            raise RuntimeError("Task is not done, cannot get exception")
        
        return self._exception
    
    def _task_done_callback(self, task: asyncio.Task) -> None:
        """
        Internal callback when task completes.
        
        Parameters
        ----------
        task : asyncio.Task
            The completed task
        """
        self._completed = True
        
        try:
            self._result = task.result()
            logger.debug(f"Task '{self.name}' completed successfully")
        except Exception as e:
            self._exception = e
            logger.debug(f"Task '{self.name}' failed with exception: {str(e)}")
    
    async def wait(self) -> Any:
        """
        Wait for the task to complete.
        
        Returns
        -------
        Any
            Task result
        """
        try:
            return await self._task
        except Exception:
            # The exception will be stored by the done callback
            raise