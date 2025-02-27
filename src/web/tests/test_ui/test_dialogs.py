"""
Unit tests for dialog components in the MFE Toolbox UI.

Tests the initialization, functionality, event handling, and interaction patterns
of various dialog classes including BaseDialog, InputDialog, AsyncProgressDialog,
and utility functions for showing different types of dialogs.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock

from PyQt6.QtCore import Qt, QSize, pyqtSignal
from PyQt6.QtGui import QDesktopServices, QUrl
from PyQt6.QtWidgets import (
    QMessageBox, QDialog, QLineEdit, QLabel, QPushButton,
    QVBoxLayout, QWidget, QApplication
)
from PyQt6.QtTest import QSignalSpy, QTest

from ...mfe.ui.dialogs import (
    BaseDialog, InputDialog, AsyncProgressDialog,
    show_error_dialog, show_warning_dialog, show_info_dialog,
    show_confirmation_dialog, show_input_dialog, show_async_progress_dialog,
    center_dialog
)
from ...mfe.ui.about_dialog import AboutDialog, show_about_dialog
from ...mfe.ui.close_dialog import CloseConfirmationDialog, show_close_confirmation_dialog
from ..conftest import qapp, qtbot


@pytest.mark.gui
def test_base_dialog_initialization(qtbot):
    """Tests the initialization of the BaseDialog class with proper title and layout."""
    # Create BaseDialog instance
    test_title = "Test Dialog"
    dialog = BaseDialog(title=test_title)
    qtbot.addWidget(dialog)
    
    # Verify window title
    assert dialog.windowTitle() == test_title
    
    # Verify dialog is modal
    assert dialog.isModal()
    
    # Verify main layout is properly initialized
    assert dialog.main_layout is not None
    assert dialog.main_layout.spacing() == 10
    
    # Verify content widget and layout are created
    assert dialog.content_widget is not None
    assert dialog.content_layout is not None
    
    # Verify button widget and layout are created
    assert dialog.button_widget is not None
    assert dialog.button_layout is not None
    
    # Verify dialog_result is initially None
    assert dialog.dialog_result is None


@pytest.mark.gui
def test_base_dialog_content_methods(qtbot):
    """Tests the content management methods of BaseDialog."""
    # Create BaseDialog instance
    dialog = BaseDialog(title="Content Test")
    qtbot.addWidget(dialog)
    
    # Create test widgets and layouts
    test_widget = QLabel("Test Widget")
    test_layout = QVBoxLayout()
    
    # Test add_widget_to_content
    dialog.add_widget_to_content(test_widget)
    assert test_widget.parent() == dialog.content_widget
    
    # Test add_layout_to_content
    dialog.add_layout_to_content(test_layout)
    assert dialog.content_layout.indexOf(test_layout) >= 0
    
    # Test add_separator
    separator = dialog.add_separator()
    assert separator.parent() == dialog.content_widget
    
    # Test create_button_layout
    test_button1 = QPushButton("Button 1")
    test_button2 = QPushButton("Button 2")
    buttons = [test_button1, test_button2]
    button_widget = dialog.create_button_layout(buttons)
    
    # Verify buttons are in layout
    for button in buttons:
        assert button.parent() == dialog.button_widget
    
    # Verify button_widget is returned
    assert button_widget == dialog.button_widget


@pytest.mark.gui
def test_base_dialog_result_methods(qtbot):
    """Tests the result handling methods of BaseDialog."""
    # Create BaseDialog instance
    dialog = BaseDialog()
    qtbot.addWidget(dialog)
    
    # Test set_result and get_result
    test_value = "test_result"
    spy = QSignalSpy(dialog.accepted)
    dialog.set_result(test_value)
    
    # Verify result is set
    assert dialog.dialog_result == test_value
    
    # Verify accepted signal was emitted
    assert len(spy) == 1
    
    # Test get_result
    assert dialog.get_result() == test_value
    
    # Test handle_accepted with None result
    dialog.dialog_result = None
    dialog.handle_accepted()
    assert dialog.dialog_result is True
    
    # Test handle_rejected
    spy = QSignalSpy(dialog.rejected)
    dialog.handle_rejected()
    assert dialog.dialog_result is False
    assert len(spy) == 1


@pytest.mark.gui
def test_base_dialog_positioning(qtbot, monkeypatch):
    """Tests the positioning methods of BaseDialog."""
    # Create BaseDialog instance
    dialog = BaseDialog()
    qtbot.addWidget(dialog)
    
    # Test set_fixed_size
    test_width, test_height = 400, 300
    dialog.set_fixed_size(test_width, test_height)
    assert dialog.width() == test_width
    assert dialog.height() == test_height
    
    # Create mock parent for testing centering
    mock_parent = QWidget()
    mock_parent.geometry = lambda: Qt.QRect(100, 100, 500, 500)
    dialog.setParent(mock_parent)
    
    # Mock center_dialog to test it's called properly
    mock_center_dialog = Mock()
    monkeypatch.setattr("...mfe.ui.dialogs.center_dialog", mock_center_dialog)
    
    # Test center_on_parent
    dialog.center_on_parent()
    mock_center_dialog.assert_called_with(dialog, dialog.parent())
    
    # Test center_on_screen
    dialog.center_on_screen()
    mock_center_dialog.assert_called_with(dialog, None)
    
    # Test exec method
    with patch.object(BaseDialog, 'center_on_parent') as mock_center:
        with patch.object(QDialog, 'exec') as mock_super_exec:
            dialog.exec()
            mock_center.assert_called_once()
            mock_super_exec.assert_called_once()


@pytest.mark.gui
def test_input_dialog(qtbot):
    """Tests the InputDialog class functionality."""
    # Create InputDialog with test parameters
    test_title = "Test Input"
    test_message = "Enter test value:"
    test_default = "default text"
    
    dialog = InputDialog(title=test_title, message=test_message, default_text=test_default)
    qtbot.addWidget(dialog)
    
    # Verify dialog title is set correctly
    assert dialog.windowTitle() == test_title
    
    # Verify message label shows correct text
    assert dialog.message_label.text() == test_message
    
    # Verify input field contains default text
    assert dialog.input_field.text() == test_default
    
    # Verify OK and Cancel buttons are created
    assert hasattr(dialog, 'ok_button')
    assert hasattr(dialog, 'cancel_button')
    
    # Test get_input_text
    assert dialog.get_input_text() == test_default
    
    # Test handle_ok
    spy = QSignalSpy(dialog.accepted)
    dialog.handle_ok()
    assert len(spy) == 1
    assert dialog.get_result() == test_default
    
    # Test handle_cancel
    dialog.dialog_result = None  # Reset result
    spy = QSignalSpy(dialog.rejected)
    dialog.handle_cancel()
    assert len(spy) == 1
    assert dialog.get_result() == ""
    
    # Test with user input
    dialog = InputDialog(title=test_title, message=test_message)
    qtbot.addWidget(dialog)
    
    # Simulate typing in input field
    test_input = "user input text"
    qtbot.keyClicks(dialog.input_field, test_input)
    
    # Click OK button
    qtbot.mouseClick(dialog.ok_button, Qt.MouseButton.LeftButton)
    
    # Verify result
    assert dialog.get_result() == test_input


@pytest.mark.asyncio
@pytest.mark.gui
async def test_async_progress_dialog(qtbot):
    """Tests the AsyncProgressDialog class functionality."""
    # Create AsyncProgressDialog with test parameters
    test_title = "Test Progress"
    test_message = "Processing..."
    
    dialog = AsyncProgressDialog(title=test_title, message=test_message)
    qtbot.addWidget(dialog)
    
    # Verify dialog title is set correctly
    assert dialog.windowTitle() == test_title
    
    # Verify message label shows correct text
    assert dialog.message_label.text() == test_message
    
    # Verify progress bar is initialized to 0
    assert dialog.progress_bar.value() == 0
    
    # Verify status label is empty initially
    assert dialog.status_label.text() == ""
    
    # Verify cancel button is created
    assert hasattr(dialog, 'cancel_button')
    
    # Test set_task
    async def mock_coroutine(progress_callback=None, is_cancelled=None):
        return "test_result"
    
    dialog.set_task(mock_coroutine)
    assert dialog.task == mock_coroutine
    
    # Test update_progress
    test_progress = 50
    test_status = "Half complete"
    dialog.update_progress(test_progress, test_status)
    
    # Verify progress bar updated
    assert dialog.progress_bar.value() == test_progress
    
    # Verify status label updated
    assert dialog.status_label.text() == test_status
    
    # Test run_task
    with patch.object(dialog, 'show'), patch.object(dialog, 'close'):
        # Create mock coroutine that updates progress
        async def test_task(progress_callback=None, is_cancelled=None):
            progress_callback(25, "Quarter done")
            progress_callback(75, "Almost done")
            return "success"
        
        dialog.set_task(test_task)
        result = await dialog.run_task()
        
        # Verify progress updates were shown
        assert dialog.progress_bar.value() == 75
        assert dialog.status_label.text() == "Almost done"
        
        # Verify result
        assert result == "success"
    
    # Test cancellation
    dialog = AsyncProgressDialog(title=test_title, message=test_message)
    qtbot.addWidget(dialog)
    
    # Click cancel button
    qtbot.mouseClick(dialog.cancel_button, Qt.MouseButton.LeftButton)
    
    # Verify cancellation state
    assert dialog.is_cancelled() is True
    assert "Cancelling" in dialog.status_label.text()


@pytest.mark.gui
def test_show_error_dialog(qtbot, monkeypatch):
    """Tests the show_error_dialog utility function."""
    # Mock QMessageBox.critical to avoid showing actual dialog
    mock_critical = Mock()
    monkeypatch.setattr(QMessageBox, "critical", mock_critical)
    
    # Call show_error_dialog
    test_message = "Test error message"
    test_title = "Test Error"
    show_error_dialog(test_message, test_title)
    
    # Verify QMessageBox.critical was called with correct parameters
    mock_critical.assert_called_once()
    args = mock_critical.call_args[0]
    assert args[1] == test_title
    assert test_message in args[2]


@pytest.mark.gui
def test_show_warning_dialog(qtbot, monkeypatch):
    """Tests the show_warning_dialog utility function."""
    # Mock QMessageBox.warning to avoid showing actual dialog
    mock_warning = Mock()
    monkeypatch.setattr(QMessageBox, "warning", mock_warning)
    
    # Call show_warning_dialog
    test_message = "Test warning message"
    test_title = "Test Warning"
    show_warning_dialog(test_message, test_title)
    
    # Verify QMessageBox.warning was called with correct parameters
    mock_warning.assert_called_once()
    args = mock_warning.call_args[0]
    assert args[1] == test_title
    assert test_message in args[2]


@pytest.mark.gui
def test_show_info_dialog(qtbot, monkeypatch):
    """Tests the show_info_dialog utility function."""
    # Mock QMessageBox.information to avoid showing actual dialog
    mock_info = Mock()
    monkeypatch.setattr(QMessageBox, "information", mock_info)
    
    # Call show_info_dialog
    test_message = "Test info message"
    test_title = "Test Info"
    show_info_dialog(test_message, test_title)
    
    # Verify QMessageBox.information was called with correct parameters
    mock_info.assert_called_once()
    args = mock_info.call_args[0]
    assert args[1] == test_title
    assert test_message in args[2]


@pytest.mark.gui
def test_show_confirmation_dialog(qtbot, monkeypatch):
    """Tests the show_confirmation_dialog utility function."""
    # Mock QMessageBox.question to return QMessageBox.Yes
    mock_question = Mock(return_value=QMessageBox.StandardButton.Yes)
    monkeypatch.setattr(QMessageBox, "question", mock_question)
    
    # Call show_confirmation_dialog
    test_message = "Test confirmation message"
    test_title = "Test Confirmation"
    result = show_confirmation_dialog(test_message, test_title)
    
    # Verify QMessageBox.question was called with correct parameters
    mock_question.assert_called_once()
    args = mock_question.call_args[0]
    assert args[1] == test_title
    assert test_message in args[2]
    
    # Verify result
    assert result is True
    
    # Now test with No response
    mock_question.reset_mock()
    mock_question.return_value = QMessageBox.StandardButton.No
    
    result = show_confirmation_dialog(test_message, test_title)
    assert result is False


@pytest.mark.gui
def test_show_input_dialog(qtbot, monkeypatch):
    """Tests the show_input_dialog utility function."""
    # Mock InputDialog
    mock_input_dialog = Mock()
    mock_input_dialog.exec.return_value = True
    mock_input_dialog.get_input_text.return_value = "test input"
    
    # Mock InputDialog constructor
    mock_init = Mock(return_value=mock_input_dialog)
    monkeypatch.setattr(InputDialog, "__init__", lambda self, parent=None, title="", message="", default_text="": None)
    monkeypatch.setattr(InputDialog, "__new__", mock_init)
    
    # Call show_input_dialog
    test_message = "Test input message"
    test_title = "Test Input"
    test_default = "default value"
    success, text = show_input_dialog(test_message, test_title, test_default)
    
    # Verify InputDialog was created with correct parameters
    mock_init.assert_called_once()
    
    # Verify result
    assert success is True
    assert text == "test input"
    
    # Test with cancelled input
    mock_input_dialog.exec.return_value = False
    mock_init.reset_mock()
    
    success, text = show_input_dialog(test_message, test_title, test_default)
    assert success is False
    assert text == ""


@pytest.mark.asyncio
@pytest.mark.gui
async def test_show_async_progress_dialog(qtbot):
    """Tests the show_async_progress_dialog utility function."""
    # Create mock coroutine function
    async def mock_task(progress_callback=None, is_cancelled=None):
        progress_callback(50, "Halfway done")
        return "test_result"
    
    # Mock AsyncProgressDialog
    with patch('...mfe.ui.dialogs.AsyncProgressDialog') as MockAsyncProgressDialog:
        mock_dialog = Mock()
        mock_dialog.run_task = Mock(return_value=asyncio.Future())
        mock_dialog.run_task.return_value.set_result("test_result")
        
        MockAsyncProgressDialog.return_value = mock_dialog
        
        # Call show_async_progress_dialog
        result = await show_async_progress_dialog(mock_task, "Test Progress", "Processing...")
        
        # Verify AsyncProgressDialog was created correctly
        MockAsyncProgressDialog.assert_called_once()
        
        # Verify set_task was called
        mock_dialog.set_task.assert_called_once_with(mock_task)
        
        # Verify run_task was called
        mock_dialog.run_task.assert_called_once()
        
        # Verify result
        assert result == "test_result"
        
    # Test exception handling
    async def failing_task(progress_callback=None, is_cancelled=None):
        raise ValueError("Test error")
    
    with patch('...mfe.ui.dialogs.AsyncProgressDialog') as MockAsyncProgressDialog:
        mock_dialog = Mock()
        future = asyncio.Future()
        future.set_exception(ValueError("Test error"))
        mock_dialog.run_task = Mock(return_value=future)
        
        MockAsyncProgressDialog.return_value = mock_dialog
        
        with pytest.raises(ValueError):
            await show_async_progress_dialog(failing_task)


@pytest.mark.gui
def test_center_dialog(qtbot):
    """Tests the center_dialog utility function."""
    # Create test dialog
    dialog = QDialog()
    dialog.resize(200, 200)
    qtbot.addWidget(dialog)
    
    # Test centering on screen
    screen_geo = dialog.screen().geometry()
    center_dialog(dialog, None)
    
    dialog_geo = dialog.geometry()
    x_centered = abs((screen_geo.width() - dialog_geo.width()) // 2 - dialog_geo.x()) <= 1
    y_centered = abs((screen_geo.height() - dialog_geo.height()) // 2 - dialog_geo.y()) <= 1
    
    assert x_centered and y_centered
    
    # Test centering on parent
    parent = QWidget()
    parent.setGeometry(100, 100, 400, 400)
    qtbot.addWidget(parent)
    
    center_dialog(dialog, parent)
    
    dialog_geo = dialog.geometry()
    parent_geo = parent.geometry()
    
    x_centered = abs((parent_geo.x() + (parent_geo.width() - dialog_geo.width()) // 2) - dialog_geo.x()) <= 1
    y_centered = abs((parent_geo.y() + (parent_geo.height() - dialog_geo.height()) // 2) - dialog_geo.y()) <= 1
    
    assert x_centered and y_centered


@pytest.mark.gui
def test_about_dialog(qtbot, monkeypatch):
    """Tests the AboutDialog class functionality."""
    # Create AboutDialog
    dialog = AboutDialog()
    qtbot.addWidget(dialog)
    
    # Verify dialog title
    assert dialog.windowTitle() == "About ARMAX"
    
    # Verify dialog has fixed size
    assert dialog.width() > 0
    assert dialog.height() > 0
    
    # Check content structure by searching for key elements
    # Find logo section
    logo_label = None
    for child in dialog.findChildren(QLabel):
        if child.pixmap() is not None:
            logo_label = child
            break
    assert logo_label is not None
    
    # Find info section with version information
    info_label = None
    for child in dialog.findChildren(QLabel):
        if child.text() and "Version" in child.text():
            info_label = child
            break
    assert info_label is not None
    
    # Find buttons
    website_button = None
    docs_button = None
    ok_button = None
    
    for button in dialog.findChildren(QPushButton):
        if button.text() == "Website":
            website_button = button
        elif button.text() == "Documentation":
            docs_button = button
        elif button.text() == "OK":
            ok_button = button
    
    assert website_button is not None
    assert docs_button is not None
    assert ok_button is not None
    
    # Test URL opening functions
    # Mock QDesktopServices.openUrl to avoid actually opening URLs
    mock_open_url = Mock()
    monkeypatch.setattr(QDesktopServices, "openUrl", mock_open_url)
    
    # Test _open_website
    dialog._open_website()
    mock_open_url.assert_called_once()
    assert "github.com" in str(mock_open_url.call_args[0][0].toString())
    
    # Reset mock and test _open_documentation
    mock_open_url.reset_mock()
    dialog._open_documentation()
    mock_open_url.assert_called_once()
    assert "readthedocs" in str(mock_open_url.call_args[0][0].toString())


@pytest.mark.gui
def test_show_about_dialog(qtbot, monkeypatch):
    """Tests the show_about_dialog utility function."""
    # Mock AboutDialog
    mock_about_dialog = Mock()
    
    # Mock AboutDialog constructor
    mock_init = Mock(return_value=mock_about_dialog)
    monkeypatch.setattr(AboutDialog, "__init__", lambda self, parent=None: None)
    monkeypatch.setattr(AboutDialog, "__new__", mock_init)
    
    # Call show_about_dialog
    show_about_dialog()
    
    # Verify AboutDialog was created
    mock_init.assert_called_once()
    
    # Verify exec was called
    mock_about_dialog.exec.assert_called_once()


@pytest.mark.gui
def test_close_confirmation_dialog(qtbot):
    """Tests the CloseConfirmationDialog class functionality."""
    # Create CloseConfirmationDialog
    dialog = CloseConfirmationDialog()
    qtbot.addWidget(dialog)
    
    # Verify dialog title
    assert dialog.windowTitle() == "Confirm Close"
    
    # Verify dialog has fixed size
    assert dialog.width() > 0
    assert dialog.height() > 0
    
    # Check content structure
    # Find warning icon
    icon_label = dialog.icon_label
    assert icon_label is not None
    assert icon_label.pixmap() is not None
    
    # Find message label
    message_label = dialog.message_label
    assert message_label is not None
    assert "Are you sure you want to close?" in message_label.text()
    
    # Find buttons
    assert hasattr(dialog, 'yes_button')
    assert hasattr(dialog, 'no_button')
    assert dialog.no_button.isDefault()
    
    # Test handle_yes
    spy = QSignalSpy(dialog.accepted)
    dialog.handle_yes()
    assert len(spy) == 1
    assert dialog.get_result() is True
    
    # Test handle_no
    dialog = CloseConfirmationDialog()
    qtbot.addWidget(dialog)
    
    spy = QSignalSpy(dialog.rejected)
    dialog.handle_no()
    assert len(spy) == 1
    assert dialog.get_result() is False


@pytest.mark.gui
def test_show_close_confirmation_dialog(qtbot, monkeypatch):
    """Tests the show_close_confirmation_dialog utility function."""
    # Mock CloseConfirmationDialog
    mock_dialog = Mock()
    mock_dialog.exec.return_value = True
    
    # Mock CloseConfirmationDialog constructor
    mock_init = Mock(return_value=mock_dialog)
    monkeypatch.setattr(CloseConfirmationDialog, "__init__", lambda self, parent=None: None)
    monkeypatch.setattr(CloseConfirmationDialog, "__new__", mock_init)
    
    # Call show_close_confirmation_dialog
    result = show_close_confirmation_dialog()
    
    # Verify CloseConfirmationDialog was created
    mock_init.assert_called_once()
    
    # Verify exec was called
    mock_dialog.exec.assert_called_once()
    
    # Verify result
    assert result is True
    
    # Test with cancel result
    mock_dialog.exec.return_value = False
    mock_init.reset_mock()
    
    result = show_close_confirmation_dialog()
    assert result is False