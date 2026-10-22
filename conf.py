import os
import sys
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))


# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'tgfsearch'
copyright = '2026, Jacob Shearer'
author = 'Jacob Shearer'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.napoleon']

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
html_sidebars = {'**': ['localtoc.html', 'globaltoc.html', 'searchbox.html']}



# Strips the module prefixes from function signatures
add_module_names = False


# Removes whole-module docstrings from autodoc's output
def remove_module_docstring(app, what, name, obj, options, lines):
    if what == 'module':
        lines.clear()

def setup(app):
    app.connect('autodoc-process-docstring', remove_module_docstring)
