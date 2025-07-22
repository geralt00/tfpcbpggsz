# __init__.py (within the "core" directory)

# Versioning read from version.py
from tfpcbpggsz.version import __version__
import importlib.resources

# Import core modules to make them easily accessible
#from .amp import *
#from .config_loader import ConfigLoader  # Adjust if needed
#from .core import (
#    Normalisation,  # Replace with the actual names of your core functions
#    DecayNLLCalculator,
#)
#from .fit import fit  # Replace with actual names
#from .masspdfs import *
#from .plotter import *
#from .angle import *
#from .generator import *

#from .plotter import generate_plot 
# Additional setup or configuration (optional)
# You might initialize logging, set default parameters, etc.
def get_assets_path() -> str:
    """
    Gets the filesystem path to the 'assets' directory.
    """
    # 1. Get a reference to the 'assets' directory within the 'my_package' package.
    #    The .files() API is preferred since Python 3.9.
    assets_ref = importlib.resources.files('tfpcbpggsz').joinpath('external')

    # 2. Use 'as_file' to get a concrete path on the filesystem.
    #    This is essential because the package could be a zip file.
    with importlib.resources.as_file(assets_ref) as assets_path:
        # assets_path is a pathlib.Path object.
        # It's only guaranteed to exist inside this 'with' block.
        print(f"Assets directory is at: {assets_path}")
        
        # 3. Convert to string and pass to C++.
        # my_cxx_module.initialize_assets(str(assets_path))
        
        # For demonstration, we'll just return the path.
        return str(assets_path)

if __name__ == "__main__":
    # Example usage
    assets_path = get_assets_path()
    print(f"Assets path: {assets_path}")
    
    # If you have a C++ module to initialize, you can call it here
    # my_cxx_module.initialize_assets(assets_path)