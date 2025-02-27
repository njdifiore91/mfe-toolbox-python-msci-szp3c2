import sys  # Python standard library: Access command-line arguments and system-specific parameters
import os  # Python standard library: Interact with the operating system for path handling
import logging  # Python standard library: Configurable logging for the application
import argparse  # Python standard library: Parse command-line arguments for application configuration

from src.backend.mfe import initialize  # Initialize the MFE Toolbox environment and verify dependencies
from src.web.mfe.ui import run_application, setup_ui_logging, initialize_resources  # Run the MFE Toolbox application with the main window

# Initialize logger
logger = logging.getLogger(__name__)

# Define application version
VERSION = "4.0.0"

def setup_logging(log_level: str) -> None:
    """
    Configure logging for the application with appropriate formatters and handlers

    Parameters:
        log_level: Logging level string (e.g., "DEBUG", "INFO", "WARNING", "ERROR")

    Returns:
        None: No return value
    """
    # Convert log_level string to logging level constant
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % log_level)

    # Configure root logger with specified level
    logging.basicConfig(level=numeric_level)

    # Create console handler with appropriate formatter
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Set log format with timestamp, level, and message
    handler.setFormatter(formatter)

    # Add handler to root logger
    logging.getLogger().addHandler(handler)

    # Log application startup message
    logger.info("Application logging configured")

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for application configuration

    Parameters:
        None

    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    # Create ArgumentParser with description of MFE Toolbox
    parser = argparse.ArgumentParser(description="MFE Toolbox - Python Financial Econometrics")

    # Add --log-level argument for configuring logging level
    parser.add_argument("--log-level", default="INFO", help="Set the logging level (DEBUG, INFO, WARNING, ERROR)")

    # Add --quiet argument for minimizing output
    parser.add_argument("--quiet", action="store_true", help="Suppress non-essential output")

    # Add --version argument for displaying version information
    parser.add_argument("--version", action="store_true", help="Show application version and exit")

    # Parse arguments from sys.argv
    args = parser.parse_args()

    # Return parsed arguments namespace
    return args

def main() -> int:
    """
    Main entry point function that initializes and runs the MFE Toolbox application

    Parameters:
        None

    Returns:
        int: Exit code (0 for success, non-zero for failure)
    """
    # Parse command-line arguments using parse_arguments()
    args = parse_arguments()

    # Set up logging using setup_logging() with specified log level
    setup_logging(args.log_level)

    # If --version argument is provided, print version and exit
    if args.version:
        print(f"MFE Toolbox Version: {VERSION}")
        return 0

    # Initialize the MFE Toolbox environment using initialize() with quiet flag
    quiet_mode = args.quiet
    if not initialize(quiet=quiet_mode):
        logger.error("MFE Toolbox initialization failed.")
        return 1

    # Configure UI logging using setup_ui_logging()
    setup_ui_logging(args.log_level)

    # Initialize UI resources using initialize_resources()
    initialize_resources()

    # Run the application using run_application()
    exit_code = run_application(sys.argv)

    # Return exit code from application execution
    return exit_code

if __name__ == "__main__":
    main()