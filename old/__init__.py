import os

project_name = os.path.basename(os.path.dirname(__file__))  # Get directory name
print(project_name)
# tfpcbpggsz/__init__.py
# (This file is empty)
# tfpcbpggsz/__init__.py
from .core import core, tensorflow_wrapper

__all__ = ['core', 'tensorflow_wrapper']

VERSION = "1.0.0"
