# src/tools/package_tools.py
import sys
import subprocess
import logging
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

def install_package(package_name: str, pip_options: str = ""):
    """
    Installs a Python package using pip.
    
    Args:
        package_name: The name of the package to install
        pip_options: Additional pip options (e.g., '--force-reinstall')
        
    Returns:
        str: Success or error message
    """
    try:
        command = [sys.executable, '-m', 'pip', 'install'] + pip_options.split() + [package_name]
        subprocess.check_call(command)
        return f"Package '{package_name}' installed successfully."
    except Exception as e:
        return f"Failed to install package '{package_name}': {e}"

# Define the args schema for install_package
class InstallPackageArgs(BaseModel):
    package_name: str = Field(description="The name of the package to install.")
    pip_options: str = Field(default="", description="Additional pip options (e.g., '--force-reinstall').")

# Create the install_package_tool
install_package_tool = StructuredTool.from_function(
    func=install_package,
    name="install_package",
    description="Installs a Python package using pip. Use this tool if you encounter a ModuleNotFoundError or need a package that's not installed.",
    args_schema=InstallPackageArgs
)