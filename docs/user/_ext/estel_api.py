"""The API reference of the user documentation, generated from ``api/manifest.json``.

Only the names of the manifest are public (golden rule 12), so the manifest is the only list of
names: nothing here or in the pages names one. At the start of every build this extension

* imports every name of the manifest, and stops the build with an error naming each one that
  cannot be imported;
* writes one page per top-level name into ``api/`` (git-ignored): an ``autoclass`` with exactly
  the members the manifest lists under the class, an ``autofunction`` or an ``autodata``;
* defines the directive ``estel-api``, used in ``api.md``, which lists the top-level names by
  kind in ``autosummary`` tables whose toctrees hold those pages.

autodoc warns about a listed member it cannot find, and ``make docs`` and Read the Docs turn
warnings into errors.
"""

from __future__ import annotations

import importlib
import inspect
import json
from pathlib import Path
from typing import Any

from docutils import nodes
from docutils.statemachine import StringList
from sphinx.application import Sphinx
from sphinx.config import Config
from sphinx.errors import ExtensionError
from sphinx.util.docutils import SphinxDirective

MANIFEST = Path(__file__).resolve().parents[3] / "api" / "manifest.json"
PAGES = "api"  # the directory of the generated pages, in the source directory
# The kinds of top-level names, with the autodoc directive and the heading of each.
KINDS = {
    "class": ("autoclass", "Classes"),
    "function": ("autofunction", "Functions"),
    "constant": ("autodata", "Constants"),
}


def _public() -> dict[str, dict[str, Any]]:
    public: dict[str, dict[str, Any]] = json.loads(MANIFEST.read_text(encoding="utf-8"))["public"]
    return public


def _top_level(public: dict[str, dict[str, Any]]) -> dict[str, tuple[str, list[str]]]:
    """Each name whose parent is not in the manifest, with its kind and its members: the
    manifest names whose parent it is (the methods, properties and constants of a class)."""
    top: dict[str, tuple[str, list[str]]] = {}
    for name, entry in public.items():
        if name.rpartition(".")[0] in public:
            continue
        if entry["kind"] not in KINDS:
            raise ExtensionError(
                f"api/manifest.json: {name} is a top-level {entry['kind']!r}, a kind the API "
                f"reference does not document. Add it to KINDS in {Path(__file__).name}."
            )
        members = [m.rpartition(".")[2] for m in public if m.rpartition(".")[0] == name]
        top[name] = (entry["kind"], members)
    return top


def _resolve(name: str) -> Any:
    """The object ``name`` names: its longest importable module prefix, then attributes."""
    parts = name.split(".")
    for end in range(len(parts) - 1, 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:end]))
        except ModuleNotFoundError:
            continue
        for part in parts[end:]:
            obj = getattr(obj, part)
        return obj
    raise ModuleNotFoundError(f"no module of {name} can be imported")


def _check(public: dict[str, dict[str, Any]]) -> None:
    failures = []
    for name in public:
        try:
            _resolve(name)
        except Exception as error:  # every failure is reported, whatever its type
            failures.append(f"{name} ({type(error).__name__}: {error})")
    if failures:
        raise ExtensionError(
            "api/manifest.json lists names the documentation cannot import: "
            + "; ".join(failures)
            + ". The code and the manifest disagree: run `python tools/api_manifest.py --check` "
            "and fix the code, or propose the change of the manifest (it is protected)."
        )


def _page(name: str, kind: str, members: list[str]) -> str:
    lines = [name, "=" * len(name), "", f".. {KINDS[kind][0]}:: {name}"]
    if members:
        # Explicit members: undoc-members acts on these only (a class constant has no docstring).
        lines += [f"   :members: {', '.join(members)}", "   :undoc-members:"]
    return "\n".join(lines) + "\n"


def write_pages(app: Sphinx, config: Config) -> None:
    """Check the manifest and write its pages, before Sphinx looks for the source files.

    A page is rewritten only when its text changes, so that a rebuild reads it again only then,
    and the pages of names that left the manifest are removed."""
    public = _public()
    _check(public)
    pages = {
        f"{name}.rst": _page(name, kind, members)
        for name, (kind, members) in _top_level(public).items()
    }
    directory = Path(app.srcdir) / PAGES
    directory.mkdir(exist_ok=True)
    for path in directory.glob("*.rst"):
        if path.name not in pages:
            path.unlink()
    for filename, text in pages.items():
        path = directory / filename
        if not path.is_file() or path.read_text(encoding="utf-8") != text:
            path.write_text(text, encoding="utf-8")


class EstelApi(SphinxDirective):
    """The top-level names of the manifest, one ``autosummary`` table per kind."""

    def run(self) -> list[nodes.Node]:
        self.env.note_dependency(str(MANIFEST))
        top = _top_level(_public())
        lines: list[str] = []
        for kind, (_, heading) in KINDS.items():
            names = [name for name, (k, _) in top.items() if k == kind]
            if names:
                lines += [f".. rubric:: {heading}", "", ".. autosummary::"]
                lines += [f"   :toctree: {PAGES}", "   :nosignatures:", ""]
                lines += [f"   {name}" for name in names] + [""]
        container = nodes.container()
        self.state.nested_parse(StringList(lines, source=str(MANIFEST)), 0, container)
        return [*container.children]


def drop_type_docstring(
    app: Sphinx, what: str, name: str, obj: Any, options: Any, lines: list[str]
) -> None:
    """A module constant without a docstring of its own gets that of its type from autodoc
    (``int([x]) -> integer`` for ``STATISTICS_REFERENCE``): drop it."""
    if what == "data" and "\n".join(lines).strip() == inspect.cleandoc(type(obj).__doc__ or ""):
        lines.clear()


def setup(app: Sphinx) -> dict[str, Any]:
    app.connect("config-inited", write_pages)
    # Before numpydoc (priority 500), which rewrites the lines.
    app.connect("autodoc-process-docstring", drop_type_docstring, priority=100)
    app.add_directive("estel-api", EstelApi)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
