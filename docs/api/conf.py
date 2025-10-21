import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, List

import tomli

# import tlcpack_sphinx_addon
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

root = Path(__file__).parents[1].resolve()
sys.path.insert(0, str(root))
os.environ["BUILD_DOC"] = "1"

project = "FlashInfer-Bench"
author = "FlashInfer Community"
copyright = f"2025-{datetime.now().year}, {author}"

try:
    from setuptools_scm import get_version as scm_get_version
except Exception:
    scm_get_version = None


def _compute_version() -> str:
    v = os.environ.get("FLASHINFER_BENCH_VERSION")
    if v:
        return v
    if scm_get_version is not None:
        try:
            return scm_get_version(root=str(root), fallback_version="0.0.0")
        except Exception:
            pass
    try:
        with open("../../pyproject.toml", "rb") as f:
            data = tomli.load(f)
        v = data.get("project", {}).get("version")
        if v:
            return v
    except Exception:
        pass
    return "0.0.0+docs"


__version__ = _compute_version()


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx_tabs.tabs",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinxcontrib.autodoc_pydantic",
]

autodoc_mock_imports = [
    "torch",
    "triton",
    "flashinfer._build_meta",
    "cuda",
    "numpy",
    "einops",
    "mpi4py",
    "safetensors",
]
autodoc_default_flags = ["members"]
autodoc_class_signature = "separated"
autodoc_member_order = "bysource"
autodoc_default_options = {"exclude-members": "model_config"}
autodoc_typehints = "both"

autodoc_pydantic_model_show_validator_summary = False
autodoc_pydantic_model_show_validator_members = False
autodoc_pydantic_model_show_config_summary = False
autodoc_pydantic_field_list_validators = False
autodoc_pydantic_model_summary_list_order = "bysource"
autodoc_pydantic_model_member_order = "bysource"

autosummary_generate = True

source_suffix = [".rst", ".md"]

language = "en"

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", ".sphinx-deps", ".sphinx-deps/**"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# A list of ignored prefixes for module index sorting.
# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "deflist",
    "colon_fence",
    "html_image",
    "linkify",
    "substitution",
]

myst_heading_anchors = 3
myst_ref_domains = ["std", "py"]
myst_all_links_external = False

# -- Options for HTML output ----------------------------------------------

html_theme = "furo"  # "sphinx_rtd_theme"

templates_path: List[Any] = []

html_static_path = ["_static"]

html_theme_options = {
    "light_logo": "FlashInfer-white-background.png",
    "dark_logo": "FlashInfer-black-background.png",
}
