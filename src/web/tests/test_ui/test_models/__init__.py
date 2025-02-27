"""
Test utilities for UI model components in the MFE Toolbox.

This module provides helper functions for asynchronous testing of
PyQt6-based model visualization interfaces, supporting the integrated
testing of asynchronous operations in the UI layer.
"""

import pytest  # pytest v7.4.3
import asyncio
from functools import wraps
from PyQt6.QtTest import QTest  # PyQt6 v6.6.1

# Default timeout for UI operations in milliseconds
UI_TIMEOUT = 5000

# Mark all tests in this package as UI tests
pytestmark = [pytest.mark.ui]

# Exported symbols
__all__ = ["async_test_helper", "UI_TIMEOUT"]


def async_test_helper(function):
    """
    Decorator for simplifying async UI tests that need Qt event processing.
    
    This decorator ensures that Qt events are processed during asyncio operations,
    which is essential for testing asynchronous UI components properly. It handles
    the complexities of integrating PyQt6's event loop with Python's asyncio,
    allowing tests to simulate user interactions with UI components that perform
    asynchronous operations.
    
    Args:
        function: The test coroutine function to be wrapped
        
    Returns:
        A wrapped test function that handles asyncio event loop and Qt event processing
    
    Example:
        @async_test_helper
        async def test_model_async_update():
            # Test code that interacts with PyQt6 UI and uses await
            model = ModelVisualizer()
            await model.update_async()
            assert model.is_updated
    """
    @wraps(function)
    def wrapper(*args, **kwargs):
        # Create event loop for test execution
        loop = asyncio.get_event_loop()
        
        # Create coroutine by calling the original test function
        coro = function(*args, **kwargs)
        
        # Define cleanup for the test
        def cleanup():
            # Process any pending Qt events before finishing
            QTest.qWait(100)
        
        # Run the test coroutine with proper event handling
        try:
            result = loop.run_until_complete(coro)
            cleanup()
            return result
        except Exception as e:
            cleanup()
            raise e
    
    return wrapper