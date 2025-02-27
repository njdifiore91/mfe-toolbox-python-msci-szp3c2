import os
import sys
import re
from pathlib import Path
from setuptools import setup, find_packages
from setuptools.command.sdist import sdist
from setuptools.command.build_py import build_py
from typing import Dict, List, Optional, Union

# Path to the current directory containing setup.py
HERE = Path(__file__).resolve().parent
PACKAGE_NAME = "mfe-toolbox"

def get_version():
    """
    Extract package version from mfe/__init__.py using regular expressions.
    
    Returns:
        str: The package version string
    """
    init_file = HERE / "mfe" / "__init__.py"
    if not init_file.exists():
        return "0.0.0"
    
    with open(init_file, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Look for __version__ = "x.y.z" pattern
    version_match = re.search(r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]', content)
    if version_match:
        return version_match.group(1)
    
    return "0.0.0"

def read_requirements():
    """
    Read package requirements from requirements.txt file.
    
    Returns:
        List[str]: List of package dependencies with version specifiers
    """
    req_file = HERE / "requirements.txt"
    if not req_file.exists():
        return []
    
    with open(req_file, "r", encoding="utf-8") as f:
        requirements = [line.strip() for line in f 
                       if line.strip() and not line.startswith("#")]
    
    return requirements

def read_long_description():
    """
    Read the long description from README.md file.
    
    Returns:
        str: Content of README.md as string
    """
    readme_file = HERE / "README.md"
    if not readme_file.exists():
        return ""
    
    with open(readme_file, "r", encoding="utf-8") as f:
        return f.read()

# Load package information
LONG_DESCRIPTION = read_long_description()
REQUIREMENTS = read_requirements()

class CustomBuildPy(build_py):
    """
    Custom build_py command that ensures Numba module compilation during build.
    This helps with first-time package usage by pre-compiling JIT functions.
    """
    
    def run(self):
        """
        Extends the standard build_py command with Numba precompilation.
        """
        # Run the standard build_py steps
        super().run()
        
        # List of modules that use Numba for JIT compilation
        numba_modules = [
            "mfe.core.optimization",
            "mfe.models.garch",
            "mfe.models.volatility",
            "mfe.core.bootstrap",
            "mfe.models.realized",
            "mfe.core.distributions"
        ]
        
        # Attempt to pre-compile Numba modules
        try:
            import numba
            print("Pre-compiling Numba-optimized modules...")
            
            for module_name in numba_modules:
                try:
                    # Dynamic import of modules to trigger JIT compilation
                    module_parts = module_name.split('.')
                    module_path = '.'.join(module_parts[:-1])
                    module_obj = module_parts[-1]
                    
                    # Use __import__ for dynamic importing
                    if module_path:
                        module = __import__(module_path, fromlist=[module_obj])
                        getattr(module, module_obj)
                    else:
                        __import__(module_obj)
                    
                    print(f"  Successfully compiled {module_name}")
                except Exception as e:
                    print(f"  Warning: Could not pre-compile {module_name}: {str(e)}")
                    
        except ImportError:
            print("Numba not available during build. JIT compilation will occur at runtime.")

class CustomSDist(sdist):
    """
    Custom sdist command that ensures all necessary files are included in source distribution.
    This is important for ensuring a complete source package for PyPI distribution.
    """
    
    def make_release_tree(self, base_dir, files):
        """
        Extends the standard make_release_tree method to include additional files.
        
        Args:
            base_dir (str): Distribution directory
            files (list): List of files to include
        """
        # Call the standard method
        super().make_release_tree(base_dir, files)
        
        # Ensure important files are included
        additional_files = [
            "README.md",
            "LICENSE",
            "requirements.txt",
            "pyproject.toml"
        ]
        
        base_path = Path(base_dir)
        for filename in additional_files:
            src_file = HERE / filename
            if src_file.exists():
                dest_file = base_path / filename
                if not dest_file.exists():
                    print(f"Including additional file: {filename}")
                    self.copy_file(str(src_file), str(dest_file))

# Configure package setup
setup(
    name=PACKAGE_NAME,
    version=get_version(),
    description="A comprehensive suite of Python modules designed for modeling financial time series and conducting advanced econometric analyses",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author="Kevin Sheppard",
    author_email="kevin.sheppard@economics.ox.ac.uk",
    url="https://github.com/bashtage/mfe-toolbox",
    packages=find_packages(include=["mfe", "mfe.*"]),
    install_requires=REQUIREMENTS,
    python_requires=">=3.12",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research", 
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Office/Business :: Financial"
    ],
    cmdclass={
        "build_py": CustomBuildPy,
        "sdist": CustomSDist
    },
    include_package_data=True,
    zip_safe=False
)