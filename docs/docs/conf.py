# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('../'))


# -- Project information -----------------------------------------------------

project = 'TrafPy'
copyright = '2020, Christopher W. F. Parsonson'
author = 'Christopher W. F. Parsonson'

# The full version, including alpha/beta/rc tags
release = '0.0.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
import sphinx_rtd_theme

extensions = [
        'sphinx_rtd_theme',
        'sphinx.ext.autodoc',
        'nb2plots',
        'sphinx.ext.napoleon',
        'sphinx.ext.viewcode',
        'texext',
        'rst2pdf.pdfbuilder'
]
pdf_documents = [('index', u'trafpy_documentation', u'TafPy Documentation', u'C W F Parsonson'),]
# index - master document
# rst2pdf - name of the generated pdf
# Sample rst2pdf doc - title of the pdf
# Your Name - author name in the pdf

#napoleon_google_docstring = True
napoleon_numpy_docstring = False
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

# set sphinx master doc for readthedocs to find (otherwise get contents.rst not found)
master_doc = 'index'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['config.rst', 'modules.rst']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# The name of the Pygments (syntax highlighting) style to use.
# pygments_style = 'sphinx'
pygments_style = 'friendly'
#pygments_style = 'colorful'

# file-wide GitHub URL
github_url = 'https://github.com/cwfparsonson/trafpy'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']
html_static_path = [] # don't have _static file atm so removing this to remove warnings
