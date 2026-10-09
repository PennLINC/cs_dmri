# SPDX-License-Identifier: MIT OR Apache-2.0
"""Sphinx configuration for the cs_dmri documentation."""

import os
import sys

sys.path.insert(0, os.path.abspath("../python"))

project = "cs_dmri"
author = "the PennLINC developers team"
copyright = "2026, the PennLINC developers team"

try:  # the installed package knows its version; fall back to the workspace manifest
    from importlib.metadata import version as _version

    release = _version("cs_dmri")
except Exception:  # noqa: BLE001
    import re

    with open(os.path.join(os.path.dirname(__file__), "..", "Cargo.toml")) as f:
        release = re.search(r'^version\s*=\s*"([^"]+)"', f.read(), re.M).group(1)
version = ".".join(release.split(".")[:2])

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
]

# The compiled extension is not built on the documentation host; the public
# classes and their docstrings live in the pure-Python layer.
autodoc_mock_imports = ["cs_dmri._cs_dmri"]
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {"members": True, "show-inheritance": False}
napoleon_numpy_docstring = True
napoleon_google_docstring = False

myst_enable_extensions = ["dollarmath", "colon_fence", "deflist"]
myst_heading_anchors = 3

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "nibabel": ("https://nipy.org/nibabel", None),
}

templates_path = []
exclude_patterns = ["_build", "tools"]
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_title = f"cs_dmri {release}"
