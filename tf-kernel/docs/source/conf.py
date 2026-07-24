# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import re
from pathlib import Path

# -- Mock modules for autodoc on ReadTheDocs/CI (no CUDA) --------------------
autodoc_mock_imports = os.environ.get('SPHINX_AUTODOC_MOCK_MODULES', '').split(',')
autodoc_mock_imports = [m.strip() for m in autodoc_mock_imports if m.strip()]

# -- Project information -----------------------------------------------------
project = 'tf-kernel'
copyright = '2026, TeleFuser Team'
author = 'TeleFuser Team'
pyproject_contents = (Path(__file__).resolve().parents[2] / 'pyproject.toml').read_text(encoding='utf-8')
version_match = re.search(r'^version = "([^"]+)"$', pyproject_contents, re.MULTILINE)
if version_match is None:
    raise RuntimeError('Could not read tf-kernel version from pyproject.toml')
release = version_match.group(1)

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = []

# -- Extension configuration -------------------------------------------------
autodoc_member_order = 'bysource'
autodoc_typehints = 'description'

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'torch': ('https://docs.pytorch.org/docs/stable', None),
    'numpy': ('https://numpy.org/doc/stable', None),
}

# Todo settings
todo_include_todos = True
