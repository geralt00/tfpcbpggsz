# __init__.py (within the "core" directory)

# Versioning (Optional but recommended for good practice)
__version__ = "0.1.0"

# Import core modules to make them easily accessible
from .amp import *
from .config_loader import ConfigLoader  # Adjust if needed
from .core import (
    Normalisation,  # Replace with the actual names of your core functions
    DecayNLLCalculator,
)
from .fit import fit  # Replace with actual names
from .masspdfs import *
from .plotter import *

#from .plotter import generate_plot 
# Additional setup or configuration (optional)
# You might initialize logging, set default parameters, etc.
