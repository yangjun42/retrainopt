"""Sphinx configuration; build from the installed RetrainOpt package."""

from importlib.metadata import version as package_version

project = "RetrainOpt"
author = "Jun Yang"
release = package_version("retrainopt")
version = release
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.doctest",
    "sphinx.ext.mathjax",
]
myst_enable_extensions = ["dollarmath"]
html_theme = "furo"
html_static_path = ["_static"]
exclude_patterns = ["_build"]
autodoc_member_order = "bysource"
nitpicky = False

