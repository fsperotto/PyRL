# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'PyRL'
copyright = '2023, Filipo S. Perotto, Aymane Ouhabi, Melvine Nargeot'
author = 'Filipo S. Perotto, Aymane Ouhabi, Melvine Nargeot'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

import sys, os
sys.path.insert(0, os.path.abspath('../../src'))
sys.path.insert(0, os.path.abspath('../../src/pyrl'))

extensions = [
    'sphinx.ext.githubpages',
    'sphinx.ext.autodoc',
    'sphinx.ext.mathjax'
]

templates_path = ['_templates']
exclude_patterns = []

with open("../../requirements.txt", "r") as fh:
    requirements = fh.read()
    requirements = requirements.split()

autodoc_mock_imports = ["numpy", "scipy", "numba", "pandas"] + requirements

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

#html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
