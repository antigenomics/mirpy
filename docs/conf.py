import os
import sys


ROOT = os.path.abspath("..")
sys.path.insert(0, ROOT)

project = "mirpy"
copyright = "2026, Immunosequencing Algorithms Laboratory (ISALGO lab)"
author = "Immunosequencing Algorithms Laboratory (ISALGO lab)"
release = "1.3.1"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.githubpages",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "nbsphinx",
]

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
]
autosummary_generate = False
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_mock_imports = []

html_baseurl = "https://docs.isalgo.dev/mirpy/"
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_title = "mirpy documentation"
html_css_files = ["custom.css"]
html_sidebars = {
    "index": [],
}
html_theme_options = {
    "navbar_align": "content",
    "show_prev_next": False,
    "secondary_sidebar_items": ["page-toc"],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/antigenomics/mirpy",
            "icon": "fa-brands fa-github",
        },
    ],
}
nbsphinx_execute = "never"
