# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import time
import warnings
from pathlib import Path

import mml
import mml.configs

# During docs building, show deprecation messages, they are also used for doc generation
warnings.simplefilter(action="default", category=DeprecationWarning)

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "mml"
copyright = f"2021-{time.strftime('%Y')}, German Cancer Research Center (DKFZ), Heidelberg, Germany"
author = "Patrick Godau"
release = mml.__version__
version = release

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",  # generate documentation using doc strings
    "sphinx.ext.autosummary",  # used to generate overview tables
    "sphinx.ext.intersphinx",  # link between different python packages
    "sphinx.ext.ifconfig",  # allows conditional docs
    "sphinx.ext.viewcode",  # links to code snippets
    "myst_nb",  # allow jupyter notebooks to be included
    "sphinxcontrib.autoyaml",  # create documentation out of the config yaml files
]

templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"

html_theme_options = {
    "logo_only": False,
    "prev_next_buttons_location": "bottom",
    "style_external_links": False,
    # Toc options
    "collapse_navigation": True,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}

# further configuration possibilities:
# https://sphinx-rtd-theme.readthedocs.io/en/stable/configuring.html

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
html_title = "MML"

# A shorter title for the navigation bar.  Default is the same as html_title.
html_short_title = "MML documentation"

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = "_static/mml_logo.png"

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
html_favicon = "_static/mml_favicon.ico"

html_static_path = ["_static"]

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
html_last_updated_fmt = "%b %d, %Y"

# remove the module prefix from module members documentation
add_module_names = False

# other settings
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/main/", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

autoclass_content = "class"
autodoc_class_signature = "separated"
autodoc_inherit_docstrings = False

nb_execution_mode = "off"
# autoyaml options (https://github.com/Jakski/sphinxcontrib-autoyaml?tab=readme-ov-file#options)
autoyaml_root = str(Path(mml.configs.__file__).parent)
autoyaml_level = 3
