# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import shutil
import subprocess
import sys

this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, this_dir + "/..")

#-- Project information -----------------------------------------------------

project = 'TFPCBPGGSZ'
copyright = '2024, Shenghui Zeng'
author = 'Shenghui Zeng'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.graphviz",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_gallery.gen_gallery",
    "matplotlib.sphinxext.plot_directive",
]
exclude_patterns = [
    ".DS_Store",
    "Thumbs.db",
    "_build",
    "*ipynb",

]
source_suffix = [
    ".rst",
]


# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_title = "TFPCBPGGSZ"
viewcode_follow_imported_members = True

# -- Options for API ---------------------------------------------------------
add_module_names = False
autodoc_mock_imports = [
    "iminuit",
    "tensorflow",
]

# Cross-referencing configuration
default_role = "py:obj"
primary_domain = "py"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

# -- Generate API skeleton ----------------------------------------------------
shutil.rmtree("api", ignore_errors=True)
subprocess.call(
    " ".join(
        [
            "sphinx-apidoc",
            "-o api/",
            "--force",
            "--no-toc",
            "--templatedir _templates",
            "--separate",
            "../tfpcbpggsz",
            # exclude_patterns
            "../tfpcbpggsz/amp",
            "../tfpcbpggsz/amp_ag",
            "../tfpcbpggsz/amp_test",
            "../tfpcbpggsz/version.py",
            "../tfpcbpggsz/fit.py",
            "../tfpcbpggsz/config_loader.py",
            "../tfpcbpggsz/masspdfs.py",
            "../tfpcbpggsz/plotter.py",



        ]
    ),
    shell=True,
)

sphinx_gallery_conf = {
    "examples_dirs": "../examples",  # path to your example scripts
    "gallery_dirs": "auto_examples",  # path to where to save gallery generated output
    "line_numbers": True,
    "run_stale_examples": True,
    "filename_pattern": "ex",
}

