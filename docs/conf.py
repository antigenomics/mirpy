"""Sphinx configuration for mirpy (mirpy-lib, import ``mir``)."""

import os
import sys
from importlib.metadata import PackageNotFoundError, version as _pkg_version

# mir is installed (pip install -e .), but add the repo root so autodoc works from a bare checkout.
sys.path.insert(0, os.path.abspath(".."))

project = "mirpy"
author = "ISALGO lab"
copyright = "2026, ISALGO lab"

try:
    release = _pkg_version("mirpy-lib")
except PackageNotFoundError:
    release = "3.1.0"
version = ".".join(release.split(".")[:2])

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "nbsphinx",
]

# Stub RST files are maintained by hand (docs/mir*.rst); do not auto-generate.
autosummary_generate = False
autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = False
nbsphinx_execute = "never"

# Heavy / optional dependencies not needed to read docstrings — mock so autodoc imports every module
# on a docs-only environment (core deps numpy/polars/scipy/scikit-learn/seqtree/vdjtools are real).
autodoc_mock_imports = [
    "torch",
    "Bio",
    "arda",
    "vdjmatch",
    "kneed",
    "pynndescent",
    "matplotlib",
    "seaborn",
    "umap",
    "marimo",
    "huggingface_hub",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_title = "mirpy"
html_theme_options = {
    "github_url": "https://github.com/antigenomics/mirpy",
    "show_prev_next": False,
    "navigation_with_keys": False,
}
