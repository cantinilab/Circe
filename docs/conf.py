# docs/conf.py

import os
import sys
from datetime import datetime

# -- Path setup --------------------------------------------------------------
# Make your module importable by adding the project root to sys.path
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------
project = 'Circe'
author = 'Rémi Trimbour'
copyright = f'{datetime.now().year}, {author}'
release = '0.3.8-alpha'  # Or dynamically pull from package version

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',    # Automatically generate documentation from docstrings
    'sphinx.ext.napoleon',   # Support for NumPy/Google style docstrings
    'sphinx.ext.viewcode',   # Add links to highlighted source code
    'sphinx.ext.intersphinx', # Link to other projects' documentation
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"

html_title = "CIRCE"
html_static_path = ['_static']
