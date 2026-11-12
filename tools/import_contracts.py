"""Custom import-linter contract types for ESTEL (Part I, §2.4).

Two contract types, registered in ``pyproject.toml`` under ``[tool.importlinter]``:

``optional_forbidden``
    The built-in ``forbidden`` contract, except that internal source or forbidden modules
    that do not exist yet are skipped instead of being an error. ESTEL's layers are created
    phase by phase (Part II, §15), so the contracts must hold on the empty package of P0 and
    start biting as soon as a module appears.

``allowlist``
    A module (and its descendants) may import only the standard library and a listed set of
    third-party packages, and nothing from its own root package outside itself. Used for the
    oracle ``estel._spec`` ("NumPy only, nothing from estel").

Run with ``PYTHONPATH=tools lint-imports`` (``make lint-imports`` does this).
"""

from __future__ import annotations

import sys
from typing import Any

from grimp import ImportGraph
from importlinter.application import output
from importlinter.contracts.forbidden import ForbiddenContract
from importlinter.domain import fields
from importlinter.domain.contract import Contract, ContractCheck
from importlinter.domain.imports import ModuleExpression


def _root_packages(contract: Contract) -> set[str]:
    roots = contract.session_options.get("root_packages") or [
        contract.session_options.get("root_package")
    ]
    return {str(r) for r in roots if r}


def _is_internal(name: str, roots: set[str]) -> bool:
    return name.split(".")[0] in roots


class OptionalForbiddenContract(ForbiddenContract):
    """``forbidden``, skipping internal modules that do not exist yet."""

    type_name = "optional_forbidden"

    def check(self, graph: ImportGraph, verbose: bool) -> ContractCheck:
        roots = _root_packages(self)

        def present(expressions: set[ModuleExpression]) -> set[ModuleExpression]:
            kept = set()
            for expression in expressions:
                text = expression.expression
                if (
                    expression.has_wildcard_expression()
                    or not _is_internal(text, roots)
                    or text in graph.modules
                ):
                    kept.add(expression)
            return kept

        sources: set[ModuleExpression] = self.source_modules  # type: ignore[assignment]
        forbidden: set[ModuleExpression] = self.forbidden_modules  # type: ignore[assignment]
        self.source_modules = present(sources)  # type: ignore[assignment]
        self.forbidden_modules = present(forbidden)  # type: ignore[assignment]
        try:
            if not self.source_modules or not self.forbidden_modules:
                return ContractCheck(kept=True)
            return super().check(graph, verbose)
        finally:
            self.source_modules = sources  # type: ignore[assignment]
            self.forbidden_modules = forbidden  # type: ignore[assignment]


class AllowlistContract(Contract):
    """A module may import only the standard library and an allowlist of external packages.

    Options:
        - source_module:    the module checked, with its descendants. Skipped if absent.
        - allowed_external: third-party top-level packages it may import.
    """

    type_name = "allowlist"

    source_module = fields.StringField()
    allowed_external = fields.ListField(subfield=fields.StringField())

    def check(self, graph: ImportGraph, verbose: bool) -> ContractCheck:
        roots = _root_packages(self)
        source = str(self.source_module)
        allowed = {str(a) for a in self.allowed_external}  # type: ignore[attr-defined]
        if source not in graph.modules:
            return ContractCheck(kept=True)

        checked = {source, *graph.find_descendants(source)}
        violations: list[dict[str, Any]] = []
        for importer in sorted(checked):
            for imported in sorted(graph.find_modules_directly_imported_by(importer)):
                top = imported.split(".")[0]
                if _is_internal(imported, roots):
                    if imported in checked:
                        continue
                    reason = f"imports {imported}, from its own package"
                elif top in sys.stdlib_module_names or top in allowed:
                    continue
                else:
                    reason = f"imports {imported}, which is not in the allowlist {sorted(allowed)}"
                lines = [
                    d["line_number"]
                    for d in graph.get_import_details(importer=importer, imported=imported)
                ]
                violations.append({"importer": importer, "reason": reason, "lines": lines})

        return ContractCheck(kept=not violations, metadata={"violations": violations})

    def render_broken_contract(self, check: ContractCheck) -> None:
        for violation in check.metadata["violations"]:
            lines = ", ".join(f"l.{n}" for n in violation["lines"])
            output.print_error(f"{violation['importer']} {violation['reason']} ({lines}).")
            output.new_line()
        output.print(
            f"{self.source_module} is the oracle: it may use the standard library and "
            "NumPy only. Reimplement what you need inside it instead of importing it."
        )
