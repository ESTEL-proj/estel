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

It also finds the source of each documented name, for the ``[source]`` links of
``sphinx.ext.linkcode`` (``linkcode_resolve`` in ``conf.py``): ``source_lines``.
"""

from __future__ import annotations

import ast
import functools
import importlib
import importlib.util
import inspect
import json
from pathlib import Path
from types import ModuleType
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


def _assignment(owner: type | ModuleType, name: str) -> tuple[str, int, int]:
    """The statement that assigns ``name`` in the body of ``owner``, a class or a module: a
    constant or a class attribute, whose value ``inspect`` cannot locate. A name a module
    imports (``from ._reference import STATISTICS_REFERENCE``) is followed to the module that
    assigns it."""
    file = inspect.getsourcefile(owner)
    if file is None:
        raise LookupError(f"{owner.__name__} has no source file")
    body = ast.parse(Path(file).read_text(encoding="utf-8")).body
    if isinstance(owner, type):
        for part in owner.__qualname__.split("."):
            body = next(
                (n.body for n in body if isinstance(n, ast.ClassDef) and n.name == part), []
            )
    for node in body:
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        if any(isinstance(target, ast.Name) and target.id == name for target in targets):
            return file, node.lineno, node.end_lineno or node.lineno
        if isinstance(node, ast.ImportFrom) and isinstance(owner, ModuleType):
            for alias in node.names:
                if (alias.asname or alias.name) == name:
                    module = importlib.util.resolve_name(
                        "." * node.level + (node.module or ""), owner.__package__
                    )
                    if module != owner.__name__:  # not `from . import name`, a submodule
                        return _assignment(importlib.import_module(module), alias.name)
    raise LookupError(f"no statement of {file} assigns {name} in {owner.__name__}")


def source_lines(module: str, fullname: str) -> tuple[str, int, int]:
    """The path in the repository of the file that defines ``module.fullname``, with the first
    and the last line of its definition. The package is in ``src/`` in the repository, whether
    it is installed from there (editable) or not.

    A name re-exported from a private module is found where it is defined; a decorated function
    with its decorators (``inspect.unwrap``); a property at its getter, a classmethod at its
    function; a constant or a class attribute at the statement that assigns it, in the class of
    the method resolution order that holds it; a module as its whole file. Raises
    ``LookupError`` (or the error of ``inspect``) with the reason when no definition is found
    in the package."""
    *parents, name = fullname.split(".")
    owner: Any = importlib.import_module(module)
    for part in parents:
        owner = getattr(owner, part)
    if isinstance(owner, type):
        owner = next((c for c in owner.__mro__ if name in vars(c)), owner)
    obj = inspect.getattr_static(owner, name)
    if isinstance(obj, property):
        obj = obj.fget
    elif isinstance(obj, classmethod | staticmethod):
        obj = obj.__func__
    elif isinstance(obj, functools.cached_property):
        obj = obj.func
    if inspect.ismodule(obj) or inspect.isclass(obj) or inspect.isroutine(obj):
        obj = inspect.unwrap(obj) if inspect.isroutine(obj) else obj
        file = inspect.getsourcefile(obj)
        if file is None:
            raise LookupError(f"{obj!r} has no source file")
        lines, start = inspect.getsourcelines(obj)
        found = (file, max(start, 1), max(start, 1) + len(lines) - 1)
    else:
        found = _assignment(owner, name)
    package = Path(importlib.import_module(module.partition(".")[0]).__file__ or "")
    package = package.parent.resolve()
    defined_in = Path(found[0]).resolve()
    if not defined_in.is_relative_to(package):
        raise LookupError(f"it is defined in {defined_in}, outside the package {package}")
    return f"src/{package.name}/{defined_in.relative_to(package).as_posix()}", found[1], found[2]


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
