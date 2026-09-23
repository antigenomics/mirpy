"""Sphinx configuration for the mirpy documentation."""

import os
import sys

# mir is installed (pip install -e .), but add the src dir so autodoc works from a bare checkout.
sys.path.insert(0, os.path.abspath("../src"))
sys.path.insert(0, os.path.abspath("_ext"))   # local Sphinx extensions

from mir import __version__  # noqa: E402  (needs the sys.path line above)

project = "mirpy"
author = "ISALGO laboratory"
copyright = "2026, ISALGO laboratory"
version = release = __version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx.ext.mathjax",     # the math section renders client-side; no LaTeX in the build
    "sphinx.ext.graphviz",    # schematics are dot, drawn natively (a TikZ pass would need LaTeX)
    "themed_graphviz",        # docs/_ext: one dot source -> a light and a dark rendering
]

# SVG so the schematics stay sharp and selectable; needs the `dot` binary at build time.
graphviz_output_format = "svg"

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
exclude_patterns = ["_build"]

html_theme = "pydata_sphinx_theme"
html_title = f"mirpy {release}"
html_theme_options = {
    # Version shown in the navbar brand on every page (no image logo -> text brand).
    "logo": {"text": f"mirpy {release}"},
    "github_url": "https://github.com/antigenomics/mirpy",
    "navigation_with_keys": True,
    "show_prev_next": False,
    # **The sidebar was a flat wall of page titles** in source order, with no grouping and nothing
    # marking where the reader is -- so it was scenery, not navigation, and `signature` sat in it
    # indistinguishable from everything else. `index.rst` now carries captioned toctrees; these
    # options make the theme render that structure rather than flatten it. Same settings as
    # `seqtree` and `mhcmatch`.
    "show_nav_level": 2,        # open each caption's pages, do not collapse to the caption alone
    "navigation_depth": 3,      # let a page's own sections show under it
    "collapse_navigation": False,
    "header_links_before_dropdown": 4,
    "show_toc_level": 2,        # right-hand "On this page": subsections too, not just top level
}

# The stock sidebar renders only children of the current top-level page, and these pages are all
# top-level siblings -- so it renders nothing. `site-nav` renders the full captioned tree.
html_sidebars = {
    "**": ["site-nav"],
    "index": [],
}
