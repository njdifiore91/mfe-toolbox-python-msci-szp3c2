import venv
import subprocess
import sys
from pathlib import Path

def setup_test_environment():
    """
    Creates and configures a Python virtual environment for testing the MFE Toolbox.
    
    This function:
    - Creates a Python virtual environment in the '.venv' directory
    - Installs all test dependencies from requirements-test.txt
    - Installs the package itself in development mode for seamless testing
    """
    # Define the path for the virtual environment
    venv_path = Path('.venv')
    
    print(f"Creating Python virtual environment at {venv_path}...")
    
    # Create virtual environment with pip enabled
    venv.create(venv_path, with_pip=True)
    
    # Determine the pip executable path based on the platform
    if sys.platform == "win32":
        pip_executable = str(venv_path / "Scripts" / "pip")
    else:
        pip_executable = str(venv_path / "bin" / "pip")
    
    print("Installing test dependencies from requirements-test.txt...")
    
    # Install test dependencies
    try:
        subprocess.run(
            [pip_executable, 'install', '-r', 'requirements-test.txt'],
            check=True
        )
        print("Successfully installed test dependencies")
    except subprocess.CalledProcessError:
        print("Error installing test dependencies")
        sys.exit(1)
    
    print("Installing package in development mode...")
    
    # Install the package in development mode
    try:
        subprocess.run(
            [pip_executable, 'install', '-e', '.'],
            check=True
        )
        print("Successfully installed package in development mode")
    except subprocess.CalledProcessError:
        print("Error installing package")
        sys.exit(1)
    
    print("\nTest environment setup completed successfully!")
    print("You can activate the environment using:")
    if sys.platform == "win32":
        print("    .venv\\Scripts\\activate")
    else:
        print("    source .venv/bin/activate")

if __name__ == '__main__':
    setup_test_environment()