"""
UI Components Test Package for MFE Toolbox.

This package contains tests for the UI components used in the MFE Toolbox interface,
focusing on parameter tables, diagnostic panels, and model equation displays.
Tests leverage pytest's dynamic test discovery and support PyQt6-based UI testing
with integration for async/await patterns.
"""

import pytest  # pytest 7.4.3

# Define the test modules exposed by this package for test discovery
__all__ = [
    'test_parameter_table',
    'test_diagnostic_panel',
    'test_model_equation'
]