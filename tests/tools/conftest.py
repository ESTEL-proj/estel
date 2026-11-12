"""Helpers for the tests of the check tools: each builds a small fake repository."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TOOLS = REPO / "tools"


def write(root: Path, files: dict[str, str]) -> None:
    """Write ``files`` (relative path -> content) under ``root``."""
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)


def run_tool(tool: str, *args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    """Run ``tools/<tool>`` as CI does, as a script, and capture its output."""
    return subprocess.run(
        [sys.executable, str(TOOLS / tool), *args],
        capture_output=True,
        text=True,
        cwd=cwd,
        check=False,
    )
