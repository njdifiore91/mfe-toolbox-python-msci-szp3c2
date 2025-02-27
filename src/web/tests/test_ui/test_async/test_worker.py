"""
MFE Toolbox - Tests for Worker class

This module contains unit tests for the Worker class that handles
asynchronous operations in the UI using PyQt6.
"""

import pytest
from PyQt6.QtCore import QThreadPool
from PyQt6.QtWidgets import QApplication
import time

from mfe.ui.async.worker import Worker


@pytest.fixture
def qt_application():
    """Create a QApplication instance for tests."""
    app = QApplication([])
    yield app
    app.quit()


@pytest.fixture
def q_thread_pool():
    """Get the global thread pool instance."""
    return QThreadPool.globalInstance()


def test_worker_initialization():
    """Test that a Worker can be properly initialized."""
    # Create a simple test function
    def test_func():
        return 42
    
    # Initialize a Worker with the test function
    worker = Worker(test_func)
    
    # Verify that the Worker object is properly configured
    assert worker.fn == test_func
    assert worker.args == ()
    assert worker.kwargs == {}


def test_worker_signals():
    """Test that Worker properly creates and exposes signals."""
    # Create a test function
    def test_func():
        return 42
    
    # Initialize a Worker with the test function
    worker = Worker(test_func)
    
    # Verify that the worker has a signals attribute
    assert hasattr(worker, 'signals')
    
    # Assert that the signals attribute has the expected signal properties
    assert hasattr(worker.signals, 'started')
    assert hasattr(worker.signals, 'finished')
    assert hasattr(worker.signals, 'error')
    assert hasattr(worker.signals, 'result')
    assert hasattr(worker.signals, 'progress')
    assert hasattr(worker.signals, 'cancelled')


def test_worker_execution(qt_application, q_thread_pool):
    """Test that Worker correctly executes the supplied function."""
    # Create a test function that returns a specific value
    def test_func():
        return 42
    
    # Create a result storage variable
    result = None
    
    # Set up signal handlers to store the result
    def store_result(res):
        nonlocal result
        result = res
    
    # Initialize a Worker with the test function
    worker = Worker(test_func)
    worker.signals.result.connect(store_result)
    
    # Start the worker in the thread pool
    q_thread_pool.start(worker)
    
    # Wait for execution to complete
    q_thread_pool.waitForDone()
    
    # Assert that the result matches the expected return value
    assert result == 42


def test_worker_with_arguments(qt_application, q_thread_pool):
    """Test that Worker correctly passes arguments to the function."""
    # Create a test function that returns its arguments
    def test_func(arg1, arg2, kwarg1=None):
        return arg1, arg2, kwarg1
    
    # Create args and kwargs for testing
    arg1_val = "test"
    arg2_val = 123
    kwarg1_val = "kwarg_value"
    
    # Create result storage variables
    result = None
    
    # Set up signal handlers to store the results
    def store_result(res):
        nonlocal result
        result = res
    
    # Initialize a Worker with the test function and arguments
    worker = Worker(test_func, arg1_val, arg2_val, kwarg1=kwarg1_val)
    worker.signals.result.connect(store_result)
    
    # Start the worker in the thread pool
    q_thread_pool.start(worker)
    
    # Wait for execution to complete
    q_thread_pool.waitForDone()
    
    # Assert that the received arguments match the ones provided
    assert result == (arg1_val, arg2_val, kwarg1_val)


def test_worker_error_handling(qt_application, q_thread_pool):
    """Test that Worker correctly handles and reports errors in the function."""
    # Create a test function that raises an exception
    def test_func():
        raise ValueError("Test error")
    
    # Create a flag to track if error signal was emitted
    error_raised = False
    error_value = None
    
    # Set up signal handler for the error signal
    def handle_error(error):
        nonlocal error_raised, error_value
        error_raised = True
        error_value = error
    
    # Initialize a Worker with the test function
    worker = Worker(test_func)
    worker.signals.error.connect(handle_error)
    
    # Start the worker in the thread pool
    q_thread_pool.start(worker)
    
    # Wait for execution to complete
    q_thread_pool.waitForDone()
    
    # Assert that the error signal was emitted
    assert error_raised
    
    # Assert that the error contains the expected exception information
    assert isinstance(error_value, ValueError)
    assert str(error_value) == "Test error"


def test_worker_progress_reporting(qt_application, q_thread_pool):
    """Test that Worker can correctly report progress."""
    # Test that the progress signal can be connected and emitted
    progress_values = []
    
    def handle_progress(value):
        progress_values.append(value)
    
    # Create a function that can report progress
    def test_func(progress_callback):
        # Use the provided callback to report progress
        for i in range(5):
            progress_callback(i / 4)
            time.sleep(0.01)
        return "Done"
    
    # Create a worker that provides progress_callback to the function
    worker = Worker(test_func, progress_callback=lambda v: worker.signals.progress.emit(v))
    worker.signals.progress.connect(handle_progress)
    
    # Start the worker
    q_thread_pool.start(worker)
    q_thread_pool.waitForDone()
    
    # Assert that progress was reported
    assert len(progress_values) == 5
    assert progress_values == [0.0, 0.25, 0.5, 0.75, 1.0]


def test_worker_signal_order(qt_application, q_thread_pool):
    """Test that Worker signals are emitted in the correct order."""
    # Create a test function
    def test_func():
        time.sleep(0.01)  # Small delay to ensure signal order is observable
        return "Done"
    
    # Create a list to store signal order
    signal_order = []
    
    # Set up signal handlers to record emission order
    def handle_started():
        signal_order.append("started")
    
    def handle_result(res):
        signal_order.append("result")
    
    def handle_finished():
        signal_order.append("finished")
    
    # Initialize a Worker with the test function
    worker = Worker(test_func)
    worker.signals.started.connect(handle_started)
    worker.signals.result.connect(handle_result)
    worker.signals.finished.connect(handle_finished)
    
    # Start the worker in the thread pool
    q_thread_pool.start(worker)
    
    # Wait for execution to complete
    q_thread_pool.waitForDone()
    
    # Assert that signals were emitted in the order: started, result, finished
    assert signal_order == ["started", "result", "finished"]