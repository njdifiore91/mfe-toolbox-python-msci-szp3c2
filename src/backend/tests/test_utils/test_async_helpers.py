"""
Tests for async_helpers module in the MFE Toolbox.

This module contains comprehensive tests for all asynchronous utilities
provided in the mfe.utils.async_helpers module, including:
- Async generator conversion
- Thread pool execution
- Concurrency limiting
- Retry mechanisms
- Timeout handling
- Progress reporting
- Exception handling
- Task management

All tests use pytest-asyncio for testing async functionality.
"""

import asyncio
import concurrent.futures
import logging
import numpy as np
import pytest
import time
from unittest.mock import Mock, patch

from mfe.utils.async_helpers import (
    async_generator, run_in_executor, gather_with_concurrency,
    retry_async, timeout_async, async_progress, handle_exceptions_async,
    create_task_with_name, run_async, async_to_sync,
    AsyncTaskGroup, AsyncLimiter, AsyncTask
)

# Setup test logger
logger = logging.getLogger('test_async_helpers')


@pytest.mark.asyncio
async def test_async_generator():
    """Test that async_generator correctly converts synchronous generators to asynchronous generators"""
    # Define a synchronous generator function
    def sync_generator():
        for i in range(5):
            yield i
    
    # Apply the decorator
    async_gen = async_generator(sync_generator)
    
    # Use the async generator
    result = []
    async for item in async_gen():
        result.append(item)
    
    # Verify the result
    assert result == [0, 1, 2, 3, 4]


@pytest.mark.asyncio
async def test_run_in_executor():
    """Test that run_in_executor correctly runs synchronous functions in a thread pool executor"""
    # Define a CPU-bound function
    def cpu_bound_function(x, y):
        time.sleep(0.01)  # Simulate CPU work
        return x * y
    
    # Test with a simple case
    result = await run_in_executor(cpu_bound_function, 5, 10)
    assert result == 50
    
    # Test with a function that raises an exception
    def raising_function():
        raise ValueError("Test error")
    
    with pytest.raises(ValueError, match="Test error"):
        await run_in_executor(raising_function)


@pytest.mark.asyncio
async def test_gather_with_concurrency():
    """Test that gather_with_concurrency correctly limits concurrency when executing coroutines"""
    # Track concurrent execution
    counter = 0
    max_concurrent = 0
    results = []
    
    async def test_coro(i, delay):
        nonlocal counter, max_concurrent
        counter += 1
        max_concurrent = max(max_concurrent, counter)
        await asyncio.sleep(delay)
        results.append(i)
        counter -= 1
        return i
    
    # Create coroutines with varying delays
    coros = [test_coro(i, 0.05) for i in range(10)]
    
    # Set concurrency limit
    concurrency_limit = 3
    
    # Run with concurrency limit
    result = await gather_with_concurrency(concurrency_limit, *coros)
    
    # Verify the concurrency limit was respected
    assert max_concurrent <= concurrency_limit
    
    # Verify all coroutines were executed and results returned in correct order
    assert len(result) == 10
    assert sorted(result) == list(range(10))


@pytest.mark.asyncio
async def test_retry_async_success():
    """Test that retry_async decorator successfully returns the value when no exceptions occur"""
    # Create a mock coroutine that succeeds
    mock_coro = Mock()
    mock_coro.return_value = "success"
    
    # Create an async function that uses the mock
    async def test_func():
        return mock_coro()
    
    # Apply the retry decorator
    decorated = retry_async()(test_func)
    
    # Run the decorated function
    result = await decorated()
    
    # Verify the function was called exactly once
    mock_coro.assert_called_once()
    
    # Verify the return value
    assert result == "success"


@pytest.mark.asyncio
async def test_retry_async_with_retries():
    """Test that retry_async correctly retries the function when exceptions occur"""
    # Create a mock that fails a few times then succeeds
    mock_coro = Mock()
    mock_coro.side_effect = [ValueError("Fail"), ValueError("Fail again"), "success"]
    
    # Create an async function that uses the mock
    async def test_func():
        return mock_coro()
    
    # Apply the retry decorator
    decorated = retry_async(
        max_retries=2,
        initial_delay=0.01,  # Short delay for testing
        exceptions_to_retry=[ValueError]
    )(test_func)
    
    # Run the decorated function
    result = await decorated()
    
    # Verify the function was called the expected number of times
    assert mock_coro.call_count == 3
    
    # Verify the final result
    assert result == "success"


@pytest.mark.asyncio
async def test_retry_async_max_retries_exceeded():
    """Test that retry_async raises the last exception when max retries are exceeded"""
    # Create a mock that always fails
    mock_coro = Mock()
    mock_coro.side_effect = ValueError("Always fail")
    
    # Create an async function that uses the mock
    async def test_func():
        return mock_coro()
    
    # Apply the retry decorator
    decorated = retry_async(
        max_retries=2,
        initial_delay=0.01,  # Short delay for testing
        exceptions_to_retry=[ValueError]
    )(test_func)
    
    # Run the decorated function and expect it to raise the exception
    with pytest.raises(ValueError, match="Always fail"):
        await decorated()
    
    # Verify the function was called the expected number of times
    assert mock_coro.call_count == 3


@pytest.mark.asyncio
async def test_timeout_async_success():
    """Test that timeout_async allows functions that complete within the timeout"""
    # Create a coroutine that completes quickly
    async def quick_func():
        await asyncio.sleep(0.05)
        return "success"
    
    # Apply the timeout decorator
    decorated = timeout_async(1.0)(quick_func)
    
    # Run the decorated function
    result = await decorated()
    
    # Verify the result
    assert result == "success"


@pytest.mark.asyncio
async def test_timeout_async_timeout_occurs():
    """Test that timeout_async raises TimeoutError when function exceeds timeout"""
    # Create a coroutine that takes too long
    async def slow_func():
        await asyncio.sleep(0.2)
        return "never reached"
    
    # Apply the timeout decorator with a short timeout
    decorated = timeout_async(0.05)(slow_func)
    
    # Run the decorated function and expect a timeout
    with pytest.raises(asyncio.TimeoutError):
        await decorated()


@pytest.mark.asyncio
async def test_async_progress():
    """Test that async_progress correctly reports progress during async function execution"""
    # Create a mock progress callback
    progress_callback = Mock()
    
    # Create a coroutine that reports progress
    async def process_with_progress(report_progress=None):
        if report_progress:
            report_progress(0)
            await asyncio.sleep(0.01)
            report_progress(50)
            await asyncio.sleep(0.01)
            report_progress(100)
        return "complete"
    
    # Apply the progress decorator
    decorated = async_progress(progress_callback)(process_with_progress)
    
    # Run the decorated function
    result = await decorated()
    
    # Verify the progress callback was called with the expected values
    assert progress_callback.call_count == 3
    progress_callback.assert_any_call(0)
    progress_callback.assert_any_call(50)
    progress_callback.assert_any_call(100)
    
    # Verify the result
    assert result == "complete"


@pytest.mark.asyncio
async def test_handle_exceptions_async_no_exception():
    """Test that handle_exceptions_async allows execution when no exceptions occur"""
    # Create a mock exception handler
    exception_handler = Mock()
    
    # Create a coroutine that succeeds
    async def successful_func():
        return "success"
    
    # Apply the exception handler decorator
    decorated = handle_exceptions_async(exception_handler)(successful_func)
    
    # Run the decorated function
    result = await decorated()
    
    # Verify the handler was not called
    exception_handler.assert_not_called()
    
    # Verify the result
    assert result == "success"


@pytest.mark.asyncio
async def test_handle_exceptions_async_with_exception():
    """Test that handle_exceptions_async correctly calls the handler when exceptions occur"""
    # Create a mock exception handler
    exception_handler = Mock()
    
    # Create a coroutine that raises an exception
    async def failing_func():
        raise ValueError("Test exception")
    
    # Apply the exception handler decorator
    decorated = handle_exceptions_async(exception_handler)(failing_func)
    
    # Run the decorated function and expect the exception to be re-raised
    with pytest.raises(ValueError, match="Test exception"):
        await decorated()
    
    # Verify the exception handler was called with the exception
    exception_handler.assert_called_once()
    args, kwargs = exception_handler.call_args
    assert isinstance(args[0], ValueError)
    assert str(args[0]) == "Test exception"


@pytest.mark.asyncio
async def test_create_task_with_name():
    """Test that create_task_with_name correctly creates a named asyncio task"""
    # Create a simple coroutine
    async def test_coro():
        return "task result"
    
    # Create a named task
    task_name = "test_task"
    task = create_task_with_name(test_coro(), task_name)
    
    # Verify the task has the specified name
    assert task.get_name() == task_name
    
    # Await the task and verify the result
    result = await task
    assert result == "task result"


def test_run_async():
    """Test that run_async correctly runs an async function in a new event loop"""
    # Create an async function
    async def test_func(x, y):
        await asyncio.sleep(0.01)
        return x + y
    
    # Run the async function using run_async
    result = run_async(test_func, 5, 10)
    
    # Verify the result
    assert result == 15
    
    # Test with a function that raises an exception
    async def raising_func():
        await asyncio.sleep(0.01)
        raise ValueError("Test error")
    
    with pytest.raises(ValueError, match="Test error"):
        run_async(raising_func)


def test_async_to_sync():
    """Test that async_to_sync correctly converts an async function to a synchronous function"""
    # Create an async function
    async def test_async_func(x, y):
        await asyncio.sleep(0.01)
        return x * y
    
    # Convert to synchronous function
    sync_func = async_to_sync(test_async_func)
    
    # Call the synchronous function directly (without await)
    result = sync_func(5, 10)
    
    # Verify the result
    assert result == 50
    
    # Test with a function that raises an exception
    async def raising_async_func():
        await asyncio.sleep(0.01)
        raise ValueError("Test error")
    
    sync_raising_func = async_to_sync(raising_async_func)
    
    with pytest.raises(ValueError, match="Test error"):
        sync_raising_func()


@pytest.mark.asyncio
async def test_async_task_group():
    """Test that AsyncTaskGroup correctly manages a group of related tasks"""
    # Create an AsyncTaskGroup
    task_group = AsyncTaskGroup()
    
    # Create test coroutines
    async def success_coro():
        await asyncio.sleep(0.05)
        return "success"
    
    async def failing_coro():
        await asyncio.sleep(0.05)
        raise ValueError("Task failed")
    
    async def long_running_coro():
        await asyncio.sleep(0.2)
        return "long task completed"
    
    # Add tasks to the group using create_task with specific names
    success_task = task_group.create_task(success_coro(), "success_task")
    failing_task = task_group.create_task(failing_coro(), "failing_task")
    long_task = task_group.create_task(long_running_coro(), "long_task")
    
    # Test get_task retrieves the correct task by name
    assert task_group.get_task("success_task") is success_task
    assert task_group.get_task("non_existent_task") is None
    
    # Wait for some tasks to complete
    await asyncio.sleep(0.1)
    
    # Test running_tasks returns the correct count of uncompleted tasks
    assert task_group.running_tasks() == 1  # Only the long task should still be running
    
    # Test cancel_all correctly cancels all unfinished tasks
    task_group.cancel_all()
    
    # Verify the long task was cancelled
    assert long_task.cancelled()
    
    # Test context manager behavior with 'async with' statement
    async with AsyncTaskGroup() as group:
        task = group.create_task(success_coro(), "test_task")
        # The task will be automatically cancelled when exiting the context


@pytest.mark.asyncio
async def test_async_limiter():
    """Test that AsyncLimiter correctly limits concurrency in async operations"""
    # Create an AsyncLimiter with a specific concurrency limit
    limit = 3
    limiter = AsyncLimiter(limit)
    
    # Track concurrent execution
    counter = 0
    max_concurrent = 0
    
    # Define a coroutine that tracks concurrent execution count
    async def tracked_coro(i):
        nonlocal counter, max_concurrent
        
        async with limiter:  # Test context manager
            counter += 1
            max_concurrent = max(max_concurrent, counter)
            await asyncio.sleep(0.05)
            counter -= 1
        
        return i
    
    # Apply the limiter's async_limit decorator to the coroutine
    @limiter.async_limit
    async def limited_coro(i):
        nonlocal counter, max_concurrent
        counter += 1
        max_concurrent = max(max_concurrent, counter)
        await asyncio.sleep(0.05)
        counter -= 1
        return i
    
    # Execute multiple instances of the decorated coroutine concurrently
    tasks = [asyncio.create_task(limited_coro(i)) for i in range(10)]
    results = await asyncio.gather(*tasks)
    
    # Verify the maximum concurrent execution count never exceeds the limit
    assert max_concurrent <= limit
    
    # Verify all coroutines were executed
    assert sorted(results) == list(range(10))
    
    # Reset tracking
    counter = 0
    max_concurrent = 0
    
    # Test manual acquire/release
    async def acquire_release_test(i):
        nonlocal counter, max_concurrent
        await limiter.acquire()
        try:
            counter += 1
            max_concurrent = max(max_concurrent, counter)
            await asyncio.sleep(0.05)
            counter -= 1
            return i
        finally:
            limiter.release()
    
    # Execute multiple coroutines with manual acquire/release
    tasks = [asyncio.create_task(acquire_release_test(i)) for i in range(10)]
    results = await asyncio.gather(*tasks)
    
    # Verify the concurrency limit was respected
    assert max_concurrent <= limit


@pytest.mark.asyncio
async def test_async_task():
    """Test that AsyncTask correctly tracks async task execution and reports progress"""
    # Create a mock progress callback
    progress_callback = Mock()
    
    # Define a coroutine that reports progress and returns a result
    async def test_coro():
        await asyncio.sleep(0.05)
        return "task completed"
    
    # Create an AsyncTask with the coroutine and progress callback
    task = AsyncTask(test_coro(), "test_task", progress_callback)
    
    # Call report_progress method to update progress
    task.report_progress(50)
    
    # Await the task's wait method
    result = await task.wait()
    
    # Verify the progress callback was called with the expected value
    progress_callback.assert_called_with(50)
    
    # Verify the task completes with the expected result
    assert result == "task completed"
    assert task.result() == "task completed"
    assert task.done()
    assert task.exception() is None
    
    # Test task with exception
    async def failing_coro():
        await asyncio.sleep(0.05)
        raise ValueError("Task failed")
    
    error_task = AsyncTask(failing_coro(), "error_task")
    
    # Wait for the task to complete (with error)
    with pytest.raises(ValueError, match="Task failed"):
        await error_task.wait()
    
    # Verify error handling
    assert error_task.done()
    with pytest.raises(ValueError, match="Task failed"):
        error_task.result()
    assert isinstance(error_task.exception(), ValueError)
    
    # Test task cancellation
    async def long_coro():
        await asyncio.sleep(0.5)
        return "never reached"
    
    cancel_task = AsyncTask(long_coro(), "cancel_task")
    
    # Cancel the task
    cancel_task.cancel()
    
    # Wait for the task to be marked as done
    await asyncio.sleep(0.1)
    
    # Verify the task was cancelled
    assert cancel_task.done()