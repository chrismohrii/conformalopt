# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "conformalopt"
copyright = "2025, Christopher Mohri"
author = "Christopher Mohri"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "alabaster"
#html_static_path = ["_static"]

import os
import sys

sys.path.insert(0, os.path.abspath("../../conformalopt"))  # Ensure package is found

# Enable Sphinx extensions
extensions = [
    "sphinx.ext.autodoc",  # Automatically pull docstrings
    "sphinx.ext.napoleon",  # Google-style docstrings
    "sphinx.ext.viewcode",  # Show source code in docs
    "sphinx.ext.mathjax",
]

# Use Read the Docs theme
html_theme = "furo"

autodoc_default_options = {
    "automodule": True,
    "autoclass": True,
    "autodoc_inherit_docstrings": True,
    "autodoc_default_flags": ["members"],
    "autodoc_exclude_members": None,  # This will exclude methods without docstrings
}


def skip_methods(app, what, name, obj, skip, options):
    # Skip methods without docstrings
    if what == "method" and obj.__doc__ is None:
        return True  # Skip the method
    return skip


def setup(app):
    app.connect("autodoc-skip-member", skip_methods)
