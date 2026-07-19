"""Sphinx configuration for the mirpy documentation."""

import os
import sys
from importlib.metadata import PackageNotFoundError, version as _pkg_version

# mir is installed (pip install -e .), but add the repo root so autodoc works from a bare checkout.
sys.path.insert(0, os.path.abspath(".."))

project = "mirpy"
author = "ISALGO laboratory"
copyright = "2026, ISALGO laboratory"
try:
    version = release = _pkg_version("mirpy-lib")
except PackageNotFoundError:
    version = release = "3.4.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "nbsphinx",
]

# mir is pure Python; core deps (numpy/polars/scipy/scikit-learn/seqtree/vdjtools) import in the docs
# build env. The heavy optional deps (only imported by mir.ml / build-time / bench viz) are mocked.
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
autodoc_typehints = "description"
autodoc_member_order = "bysource"

# Render napoleon ``Attributes:`` sections as :ivar: fields so a dataclass's Attributes docstring
# does not duplicate its autodoc'd fields.
napoleon_use_ivar = True

intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}

templates_path = ["_templates"]
html_static_path = ["_static"]
html_css_files = ["custom.css"]
exclude_patterns = ["_build", "**.ipynb_checkpoints"]

html_theme = "pydata_sphinx_theme"
html_title = f"mirpy {release}"
html_theme_options = {
    # Version shown in the navbar brand on every page (no image logo → text brand).
    "logo": {"text": f"mirpy {release}"},
    "github_url": "https://github.com/antigenomics/mirpy",
    "navigation_with_keys": True,
}
nbsphinx_execute = "never"
