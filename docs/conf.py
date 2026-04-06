import os
import sys


ROOT = os.path.abspath("..")
sys.path.insert(0, ROOT)

project = "mirpy"
copyright = "2026, antigenomics"
author = "antigenomics"
release = "1.0.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.githubpages",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
autosummary_generate = False
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_mock_imports = [
    "Bio",
    "Bio.Align",
    "Bio.Seq",
    "igraph",
    "matplotlib",
    "matplotlib.pyplot",
    "multipy",
    "multipy.fdr",
    "numpy",
    "olga",
    "olga.generation_probability",
    "olga.load_model",
    "olga.sequence_generation",
    "olga.utils",
    "pandas",
    "plotnine",
    "pympler",
    "pympler.asizeof",
    "pyparsing",
    "scipy",
    "scipy.cluster",
    "scipy.cluster.hierarchy",
    "scipy.sparse",
    "scipy.stats",
    "seaborn",
    "sklearn",
    "sklearn.base",
    "sklearn.decomposition",
    "sklearn.linear_model",
    "sklearn.manifold",
    "sklearn.metrics",
    "sklearn.model_selection",
    "sklearn.neighbors",
    "sklearn.pipeline",
    "sklearn.preprocessing",
    "sklearn.tree",
    "statsmodels",
    "statsmodels.stats",
    "statsmodels.stats.multitest",
    "textdistance",
    "tcrtrie",
    "tqdm",
    "tqdm.contrib",
    "tqdm.contrib.concurrent",
    "umap",
]

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_title = "mirpy documentation"
