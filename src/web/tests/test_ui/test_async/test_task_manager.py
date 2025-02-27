import pytest
import asyncio
import time
from unittest import mock
from PyQt6.QtCore import QCoreApplication, QThreadPool

from src.web.mfe.ui.async.task_manager import TaskManager, TaskStatus
from src.web.mfe.ui.async.worker import Worker


class AsyncTaskTestFixture:
    """Fixture class for async task tests"""
    
    def __init__(self):
        """Initialize test fixture with a task manager"""
        self.task_manager = TaskManager()
        self.results = []
        self.errors = []
        
        # Connect to task manager signals to collect results and errors
        self.task_manager.signals.task_completed.connect(self._handle_task_completed)
        self.task_manager.signals.task_failed.connect(self._handle_task_failed)
    
    def _handle_task_completed(self, task_id, result):
        """Internal handler for task completed signal"""
        self.results.append(result)
    
    def _handle_task_failed(self, task_id, error):
        """Internal handler for task failed signal"""
        self.errors.append(error)
    
    def handle_result(self, result):
        """Handler for task result signal"""
        self.results.append(result)
    
    def handle_error(self, error_info):
        """Handler for task error signal"""
        self.errors.append(error_info)
    
    def submit_test_task(self, func, args=(), kwargs={}):
        """Submit a test task and connect signals"""
        task_id = self.task_manager.create_task(
            name=func.__name__,
            function=func,
            args=args,
            kwargs=kwargs or {}
        )
        return task_id


def test_task_manager_init():
    """Test TaskManager initialization with default and custom thread counts"""
    # Create with default settings
    manager = TaskManager()
    assert manager._thread_pool is not None
    assert manager._max_concurrent_tasks == QThreadPool.globalInstance().maxThreadCount()
    
    # Create with custom max threads
    custom_threads = 4
    manager = TaskManager(max_concurrent_tasks=custom_threads)
    assert manager._max_concurrent_tasks == custom_threads


def test_submit_task(qtbot):
    """Test submitting a task to the TaskManager"""
    mock_fn = mock.Mock(return_value="result")
    task_manager = TaskManager()
    
    # Create task
    task_id = task_manager.create_task("test_task", mock_fn)
    
    # Wait for task completion
    with qtbot.waitSignal(task_manager.signals.task_completed, timeout=2000):
        pass
    
    # Verify task execution
    mock_fn.assert_called_once()
    assert task_manager.get_task_status(task_id) == TaskStatus.COMPLETED
    assert task_manager.get_task_result(task_id) == "result"


def test_submit_task_with_args(qtbot):
    """Test submitting a task with arguments"""
    mock_fn = mock.Mock(return_value="result")
    task_manager = TaskManager()
    
    # Create task with args and kwargs
    task_id = task_manager.create_task(
        "test_task", 
        mock_fn, 
        args=(1, 2),
        kwargs={"key": "value"}
    )
    
    # Wait for task completion
    with qtbot.waitSignal(task_manager.signals.task_completed, timeout=2000):
        pass
    
    # Verify function was called with correct arguments
    mock_fn.assert_called_once_with(1, 2, key="value")


def test_cancel_task(qtbot):
    """Test canceling a specific task"""
    # Create a function that sleeps to simulate long-running task
    def slow_function():
        time.sleep(1)
        return "done"
    
    task_manager = TaskManager()
    
    # Submit long-running task
    task_id = task_manager.create_task("slow_task", slow_function)
    
    # Wait a moment for task to start
    qtbot.wait(100)
    
    # Cancel the task
    result = task_manager.cancel_task(task_id)
    assert result is True
    
    # Wait for cancellation to process
    qtbot.wait(100)
    
    # Verify task was cancelled
    assert task_manager.get_task_status(task_id) == TaskStatus.CANCELLED


def test_cancel_all_tasks(qtbot):
    """Test canceling all active tasks"""
    # Create a function that sleeps to simulate long-running task
    def slow_function():
        time.sleep(1)
        return "done"
    
    task_manager = TaskManager()
    
    # Submit multiple tasks
    task_ids = [
        task_manager.create_task(f"slow_task_{i}", slow_function)
        for i in range(3)
    ]
    
    # Wait a moment for tasks to start
    qtbot.wait(100)
    
    # Cancel all tasks
    cancelled_count = task_manager.cancel_all_tasks()
    assert cancelled_count > 0
    
    # Check that tasks are cancelled
    for task_id in task_ids:
        assert task_manager.get_task_status(task_id) == TaskStatus.CANCELLED


def wait_for_tasks(task_manager, timeout=1000):
    """Helper function to wait for all tasks to complete"""
    start_time = time.time()
    while (time.time() - start_time) * 1000 < timeout:
        if task_manager.get_active_task_count() == 0:
            return True
        QCoreApplication.processEvents()
        time.sleep(0.01)
    return False


def test_wait_for_tasks(qtbot):
    """Test waiting for all tasks to complete"""
    # Create a function with short execution time
    def quick_function():
        time.sleep(0.1)
        return "done"
    
    task_manager = TaskManager()
    
    # Submit multiple tasks
    task_ids = [
        task_manager.create_task(f"quick_task_{i}", quick_function)
        for i in range(3)
    ]
    
    # Wait for tasks to complete
    result = wait_for_tasks(task_manager, timeout=2000)
    
    # Verify all tasks completed
    assert result is True
    for task_id in task_ids:
        assert task_manager.get_task_status(task_id) == TaskStatus.COMPLETED


def test_wait_for_tasks_timeout(qtbot):
    """Test timeout while waiting for tasks"""
    # Create a function with long execution time
    def slow_function():
        time.sleep(2)
        return "done"
    
    task_manager = TaskManager()
    
    # Submit long-running task
    task_id = task_manager.create_task("slow_task", slow_function)
    
    # Wait with short timeout
    result = wait_for_tasks(task_manager, timeout=100)
    
    # Verify timeout occurred
    assert result is False
    assert task_manager.get_task_status(task_id) == TaskStatus.RUNNING
    
    # Clean up
    task_manager.cancel_all_tasks()


def test_task_signals(qtbot):
    """Test task signals propagation from worker to connected slots"""
    # Create mock slots for signals
    mock_started = mock.Mock()
    mock_result = mock.Mock()
    mock_error = mock.Mock()
    mock_progress = mock.Mock()
    mock_finished = mock.Mock()
    
    # Function that emits progress and returns a result
    def task_with_progress(progress_callback=None):
        if progress_callback:
            progress_callback(0.5)
        return "result"
    
    task_manager = TaskManager()
    
    # Connect to task manager signals
    task_manager.signals.task_started.connect(mock_started)
    task_manager.signals.task_completed.connect(lambda tid, res: mock_result(res))
    task_manager.signals.task_failed.connect(lambda tid, err: mock_error(err))
    task_manager.signals.task_progress.connect(lambda tid, prog: mock_progress(prog))
    
    # Submit task
    task_id = task_manager.create_task(
        "progress_task", 
        task_with_progress
    )
    
    # Wait for task completion
    with qtbot.waitSignal(task_manager.signals.task_completed, timeout=2000):
        pass
    
    # Verify signals
    mock_started.assert_called_once()
    mock_result.assert_called_once_with("result")
    mock_progress.assert_called_once_with(0.5)
    mock_error.assert_not_called()


def test_task_error_handling(qtbot):
    """Test error handling in task execution"""
    # Create a function that raises an exception
    def error_function():
        raise ValueError("Test error")
    
    # Create mock slots for error
    mock_error = mock.Mock()
    
    task_manager = TaskManager()
    
    # Connect to signals
    task_manager.signals.task_failed.connect(lambda tid, err: mock_error(err))
    
    # Submit function as task
    task_id = task_manager.create_task("error_task", error_function)
    
    # Wait for task failure
    with qtbot.waitSignal(task_manager.signals.task_failed, timeout=2000):
        pass
    
    # Verify error signal
    mock_error.assert_called_once()
    error = mock_error.call_args[0][0]
    assert isinstance(error, ValueError)
    assert str(error) == "Test error"
    
    # Verify task status
    assert task_manager.get_task_status(task_id) == TaskStatus.FAILED
    assert task_manager.get_task_error(task_id) is not None


def test_async_task(qtbot):
    """Test task manager handling of async/await functions"""
    # Create an async function
    async def async_function():
        await asyncio.sleep(0.1)
        return "async result"
    
    task_manager = TaskManager()
    
    # Track the result
    result_value = None
    
    def on_task_completed(task_id, result):
        nonlocal result_value
        result_value = result
    
    task_manager.signals.task_completed.connect(on_task_completed)
    
    # Submit async function as task
    task_id = task_manager.create_task("async_task", async_function)
    
    # Wait for task completion
    with qtbot.waitSignal(task_manager.signals.task_completed, timeout=2000):
        pass
    
    # Verify result
    assert result_value == "async result"
    assert task_manager.get_task_result(task_id) == "async result"