"""Check that the public API of estel is exactly ``api/manifest.json`` (Part I, §2.4).

A name is public when it is reachable without an underscore and outside a private module:

* a module is private when any component of its dotted path starts with ``_``
  (``estel._config``, ``estel.spectra._estimators``); ``__init__`` does not count;
* in a public module, every top-level function, class or constant whose name has no leading
  underscore is public, and so is every non-underscore method or property of a public class;
* in a package ``__init__.py``, every non-underscore name imported from estel is public
  (``from ._config import set_backend`` makes ``estel.set_backend`` public), and resolved to
  its definition so its signature is recorded;
* any name listed in a module's ``__all__`` is public.

The tool reads the source with ``ast`` and never imports estel, so it needs no backend.

Usage::

    python tools/api_manifest.py --check   # fail if the code and the manifest differ
    python tools/api_manifest.py --print   # print the manifest the code implies

The manifest is protected (CODEOWNER approval): ``--print`` only shows what to propose.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
PACKAGE = "estel"
MANIFEST = Path("api/manifest.json")
SCHEMA = 1

Entry = dict[str, str]
Manifest = dict[str, Entry]


# ------------------------------------------------------------------ reading the source


@dataclass
class Definition:
    kind: str  # "function", "class", "constant", "method", "property", "alias"
    signature: str | None = None
    members: dict[str, Definition] = field(default_factory=dict)


@dataclass
class Module:
    name: str
    is_package: bool
    definitions: dict[str, Definition] = field(default_factory=dict)
    imports: dict[str, tuple[str, str | None]] = field(default_factory=dict)  # alias -> (mod, name)
    all_names: list[str] | None = None


def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    text = f"({ast.unparse(node.args)})"
    if node.returns is not None:
        text += f" -> {ast.unparse(node.returns)}"
    return text


def _decorator_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    names = set()
    for dec in node.decorator_list:
        target = dec.func if isinstance(dec, ast.Call) else dec
        names.add(ast.unparse(target).split(".")[-1])
    return names


def _class_definition(node: ast.ClassDef) -> Definition:
    definition = Definition(kind="class", signature="()")
    for item in node.body:
        if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
            if item.name == "__init__":
                definition.signature = _signature(item)
            elif not item.name.startswith("_"):
                decorators = _decorator_names(item)
                if "property" in decorators or "cached_property" in decorators:
                    definition.members[item.name] = Definition(kind="property")
                elif decorators & {"setter", "deleter"}:
                    continue
                else:
                    definition.members[item.name] = Definition("method", _signature(item))
        elif isinstance(item, ast.Assign | ast.AnnAssign):
            for name in _assigned_names(item):
                if not name.startswith("_"):
                    definition.members[name] = Definition(kind="constant")
    return definition


def _assigned_names(node: ast.Assign | ast.AnnAssign) -> list[str]:
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    return [t.id for t in targets if isinstance(t, ast.Name)]


def _resolve_relative(module: str, is_package: bool, level: int, target: str | None) -> str:
    if level == 0:
        return target or ""
    parts = module.split(".")
    base = parts if is_package else parts[:-1]
    base = base[: len(base) - (level - 1)] if level > 1 else base
    return ".".join(base + ([target] if target else []))


def parse_module(name: str, path: Path) -> Module:
    tree = ast.parse(path.read_text(), filename=str(path))
    module = Module(name=name, is_package=path.name == "__init__.py")
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            module.definitions[node.name] = Definition("function", _signature(node))
        elif isinstance(node, ast.ClassDef):
            module.definitions[node.name] = _class_definition(node)
        elif isinstance(node, ast.Assign | ast.AnnAssign):
            for target in _assigned_names(node):
                if target == "__all__" and node.value is not None:
                    module.all_names = [
                        e.value
                        for e in getattr(node.value, "elts", [])
                        if isinstance(e, ast.Constant) and isinstance(e.value, str)
                    ]
                else:
                    module.definitions[target] = Definition(kind="constant")
        elif isinstance(node, ast.ImportFrom):
            source = _resolve_relative(name, module.is_package, node.level, node.module)
            for alias in node.names:
                module.imports[alias.asname or alias.name] = (source, alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                module.imports[alias.asname or alias.name.split(".")[0]] = (alias.name, None)
    return module


def read_package(src: Path) -> dict[str, Module]:
    """Parse every module of the package under ``src/<PACKAGE>``."""
    modules: dict[str, Module] = {}
    root = src / PACKAGE
    for path in sorted(root.rglob("*.py")):
        parts = list(path.relative_to(src).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        name = ".".join(parts)
        modules[name] = parse_module(name, path)
    return modules


def is_private_module(name: str) -> bool:
    return any(part.startswith("_") for part in name.split("."))


# ------------------------------------------------------------------ the public surface


def _resolve(
    modules: dict[str, Module], module: str, name: str, depth: int = 0
) -> Definition | None:
    """Follow imports from ``module.name`` to its definition inside the package."""
    if depth > 10 or module not in modules:
        return None
    mod = modules[module]
    if name in mod.definitions:
        return mod.definitions[name]
    if name in mod.imports:
        source, imported = mod.imports[name]
        if imported is None:
            return None
        if f"{source}.{imported}" in modules:  # ``from estel import data`` (a submodule)
            return None
        return _resolve(modules, source, imported, depth + 1)
    return None


def _entries(qualname: str, definition: Definition) -> Manifest:
    entry: Entry = {"kind": definition.kind}
    if definition.signature is not None:
        entry["signature"] = definition.signature
    out = {qualname: entry}
    for member, member_def in definition.members.items():
        out.update(_entries(f"{qualname}.{member}", member_def))
    return out


def public_api(src: Path) -> Manifest:
    """The manifest implied by the source tree under ``src``."""
    modules = read_package(src)
    api: Manifest = {}
    for name, mod in modules.items():
        if is_private_module(name):
            continue
        public: dict[str, Definition] = {
            n: d for n, d in mod.definitions.items() if not n.startswith("_")
        }
        exported = set(mod.all_names or [])
        if mod.is_package:
            exported |= {
                alias
                for alias, (source, _) in mod.imports.items()
                if source.split(".")[0] == PACKAGE and not alias.startswith("_")
            }
        for alias in sorted(exported - public.keys()):
            if alias in mod.imports:
                source, imported = mod.imports[alias]
                if imported is None or f"{source}.{imported}" in modules:
                    continue  # a module, not an API name
                target = _resolve(modules, source, imported)
                public[alias] = target or Definition(kind="alias", signature=None)
            elif alias in mod.definitions:
                public[alias] = mod.definitions[alias]
        for n, d in public.items():
            api.update(_entries(f"{name}.{n}", d))
    return dict(sorted(api.items()))


# ------------------------------------------------------------------ comparison


@dataclass
class Diff:
    added: list[str]
    removed: list[str]
    changed: list[str]

    @property
    def empty(self) -> bool:
        return not (self.added or self.removed or self.changed)


def compare(old: Manifest, new: Manifest) -> Diff:
    return Diff(
        added=sorted(new.keys() - old.keys()),
        removed=sorted(old.keys() - new.keys()),
        changed=sorted(k for k in old.keys() & new.keys() if old[k] != new[k]),
    )


def load_manifest_text(text: str, origin: str) -> Manifest:
    """Parse the manifest; an empty or missing file means an empty API."""
    if not text.strip():
        return {}
    data: Any = json.loads(text)
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        raise SystemExit(
            f"{origin}: expected a JSON object with \"schema\": {SCHEMA} and a \"public\" "
            "object. Restore the file from main, then add your entries to \"public\"."
        )
    public = data.get("public", {})
    if not isinstance(public, dict):
        raise SystemExit(f"{origin}: \"public\" must be an object mapping names to entries.")
    return public


def render_manifest(api: Manifest) -> str:
    return json.dumps({"schema": SCHEMA, "public": api}, indent=2) + "\n"


def check(root: Path) -> list[str]:
    path = root / MANIFEST
    if not path.exists():
        return [f"{MANIFEST} is missing. Restore it from main."]
    declared = load_manifest_text(path.read_text(), str(MANIFEST))
    actual = public_api(root / "src")
    diff = compare(declared, actual)
    errors = []
    for name in diff.added:
        errors.append(
            f"{name} is public but not in {MANIFEST}. Make it private (leading underscore, or "
            f"move it to a private module) or, if it is meant to be public, propose the entry "
            f"`\"{name}\": {json.dumps(actual[name])}` in {MANIFEST} (a minor change; the "
            "file needs CODEOWNER approval)."
        )
    for name in diff.removed:
        errors.append(
            f"{name} is in {MANIFEST} but not in the code. Removing a public name is a major "
            "change: restore it, or ask the maintainers in an issue."
        )
    for name in diff.changed:
        errors.append(
            f"{name} changed: {MANIFEST} has {json.dumps(declared[name])}, the code has "
            f"{json.dumps(actual[name])}. Changing a public signature is a major change: "
            "restore the signature, or ask the maintainers in an issue."
        )
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="compare the code with the manifest")
    mode.add_argument("--print", action="store_true", help="print the manifest the code implies")
    parser.add_argument("--root", type=Path, default=REPO, help="repository root")
    args = parser.parse_args(argv)

    if args.print:
        sys.stdout.write(render_manifest(public_api(args.root / "src")))
        return 0
    errors = check(args.root)
    for error in errors:
        print(f"api_manifest: {error}", file=sys.stderr)
    if not errors:
        print(f"api_manifest: OK, the public API matches {MANIFEST}.")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
