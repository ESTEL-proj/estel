"""Sphinx configuration of the user documentation of ESTEL (make docs; .readthedocs.yaml)."""

from __future__ import annotations

import contextlib
import os
import posixpath
import re
import subprocess
import sys
from importlib.metadata import version as _version
from pathlib import Path
from typing import Any

from docutils.nodes import Element
from sphinx.addnodes import pending_xref
from sphinx.application import Sphinx
from sphinx.environment import BuildEnvironment
from sphinx.util import logging

logger = logging.getLogger(__name__)
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
GITHUB = "https://github.com/ESTEL-proj/estel"
sys.path.insert(0, str(HERE / "_ext"))

from estel_api import source_lines  # noqa: E402  (in _ext/, on the path from the line above)

# ---------------------------------------------------------------- project

project = "ESTEL"
author = "Erwan Allys, Jean-Marc Delouis, Pablo Richard"
copyright = "2026, the ESTEL authors"  # as NOTICE
release = version = _version("estel")

# ---------------------------------------------------------------- extensions

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "numpydoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",  # the [source] links of the API reference: linkcode_resolve below
    "sphinx_design",
    "sphinx_copybutton",
    "estel_api",  # _ext/estel_api.py: the API reference, from api/manifest.json
]
exclude_patterns = ["_build", "_ext"]

# MyST: Markdown pages, and the anchors of the headings of docs/maths.md, down to its ####.
# attrs_block: a class on a block, such as estel-home on the section of index.md, whose title
# the hero stands in for (_static/estel.css).
myst_enable_extensions = ["colon_fence", "attrs_block"]
myst_heading_anchors = 4

# myst_nb: the notebooks run at build time, their outputs cached; an error fails the build.
nb_execution_mode = "cache"
nb_execution_raise_on_error = True
nb_execution_timeout = 120
nb_execution_show_tb = True
nb_merge_streams = True

# autodoc and autosummary: _ext/estel_api.py writes the pages, autosummary only lists them.
autosummary_generate = False
# numpydoc: the pages list the members of the manifest; numpydoc's own tables would list them all.
numpydoc_show_class_members = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "torch": ("https://docs.pytorch.org/docs/stable", None),
}

# ---------------------------------------------------------------- HTML

html_theme = "pydata_sphinx_theme"
html_title = "ESTEL"
# The logos are those of the repository, assets/logo/, copied into _static/ with no copy here.
html_static_path = ["_static", "../../assets/logo"]
html_css_files = ["estel.css"]
html_theme_options = {
    # The square logo, on light and dark backgrounds alike, then the name.
    "logo": {
        "text": "ESTEL",
        "image_light": "../../assets/logo/estel-logo-square.svg",
        "image_dark": "../../assets/logo/estel-logo-square.svg",
    },
    "icon_links": [
        {"name": "GitHub", "url": GITHUB, "icon": "fa-brands fa-github", "type": "fontawesome"},
    ],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
}

# ---------------------------------------------------------------- source links


def _commit() -> str:
    """The commit being built, so that the lines of the links are those of the documented code:
    the one Read the Docs names, else the HEAD of git, else main."""
    commit = os.environ.get("READTHEDOCS_GIT_COMMIT_HASH", "")
    if not commit:
        with contextlib.suppress(OSError):  # no git
            git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True)
            commit = git.stdout.decode() if git.returncode == 0 else ""
    return commit.strip() or "main"


SOURCE = f"{GITHUB}/blob/{_commit()}"


def linkcode_resolve(domain: str, info: dict[str, str]) -> str | None:
    """sphinx.ext.linkcode: the [source] link of a documented Python name, to the lines of its
    definition on GitHub (source_lines, in _ext/estel_api.py). A name it cannot locate is a
    warning, so an error in `make docs`."""
    if domain != "py":
        return None
    name = f"{info['module']}.{info['fullname']}"
    try:
        path, start, end = source_lines(info["module"], info["fullname"])
    except Exception as error:  # every failure is reported, whatever its type
        logger.warning(
            "No source link for %s (%s: %s). Define it in src/estel/ with a def, a class or an "
            "assignment, or teach source_lines in docs/user/_ext/estel_api.py to find it.",
            name,
            type(error).__name__,
            error,
            type="estel",
            subtype="source",
        )
        return None
    return f"{SOURCE}/{path}#L{start}-L{end}"


# ---------------------------------------------------------------- included files

# A Markdown link: its target, unless it is an anchor of the page or a URL with a scheme.
_RELATIVE_LINK = re.compile(r"\]\((?!#|[A-Za-z][A-Za-z0-9+.-]*:)([^)\s]+)\)")
# The opening fence of a block of LaTeX, as GitHub renders formulas.
_LATEX_FENCE = re.compile(r"^```latex[ \t]*$", re.MULTILINE)


def adapt_files_outside_the_site(
    app: Sphinx, relative_path: Path, parent_docname: str, content: list[str]
) -> None:
    """A file included from outside the site, such as ``docs/maths.md``, is written for GitHub:
    its relative links point to the files of the repository on GitHub, which are not pages of
    the site, and its ```` ```latex ```` blocks become ```` ```{math} ```` blocks, which MathJax
    renders."""
    source = (Path(app.srcdir) / relative_path).resolve()
    if source.is_relative_to(Path(app.srcdir).resolve()):
        return
    base = source.parent.relative_to(REPO).as_posix()

    def on_github(match: re.Match[str]) -> str:
        return f"]({GITHUB}/blob/main/{posixpath.normpath(posixpath.join(base, match[1]))})"

    content[0] = _LATEX_FENCE.sub("```{math}", _RELATIVE_LINK.sub(on_github, content[0]))


# ---------------------------------------------------------------- cross-references


def require_page_references(
    app: Sphinx, env: BuildEnvironment, node: pending_xref, contnode: Element
) -> None:
    """A Python role written in a page of the site, such as {py:class}`~estel.STOperator`,
    must name a documented object: an unresolved one is a warning, so an error in `make docs`.

    nitpicky would check the docstrings too, which cite private names (``_build_filters``,
    ``PlanarPlan``) that the site does not document. Runs after intersphinx (priority 900)."""
    source = Path((node.source or "").partition(":")[0])
    if node.get("refdomain") == "py" and source.suffix == ".md" and source.is_relative_to(HERE):
        logger.warning(
            "%s reference target not found: %s. A page names a public name of "
            "api/manifest.json, in full (estel.STOperator, estel.STStatistics.full_grid).",
            node["reftype"],
            node["reftarget"],
            location=node,
            type="ref",
        )


def setup(app: Sphinx) -> dict[str, Any]:
    app.connect("include-read", adapt_files_outside_the_site)
    app.connect("missing-reference", require_page_references, priority=900)
    return {"parallel_read_safe": True}
