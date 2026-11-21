"""Synthesis checkpoints (Part II, §9.2 and §12.9; maths.md §5.8; task T6.1).

A checkpoint is one ``.npz`` file, written without pickle: the running field u (in the
standardised units of the target, M-SYN-INIT), the loss after each iteration so far, the
optimizer state of ``Backend.lbfgs`` (its NumPy arrays under ``state.<name>``, its Python
scalars in the manifest), and a JSON manifest: the format, the settings of the run, which a
resumed run must share, the iterations and evaluations done, and the seed of the start.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..backend._protocol import LbfgsState

FORMAT = 1
"""The version of the checkpoint format."""
_STATE = "state."
_MANIFEST = "manifest"


@dataclass(frozen=True)
class Checkpoint:
    """What a checkpoint file holds."""

    field: NDArray[Any]
    """u, ``(n_batch, Nc, N1, N2)``, in the standardised units of the target."""
    losses: NDArray[Any]
    """The loss after each iteration so far."""
    state: LbfgsState
    """The optimizer state, to resume the run."""
    meta: dict[str, Any]
    """The settings of the run, ``n_iter``, ``n_evals`` and ``seed``."""


def path_for(directory: str | os.PathLike[str], n_iter: int) -> Path:
    """The file of the checkpoint after ``n_iter`` iterations."""
    return Path(directory) / f"checkpoint_{n_iter:06d}.npz"


def save(path: Path, checkpoint: Checkpoint) -> None:
    """Write ``checkpoint`` to ``path``, through a temporary file renamed at the end, so that an
    interrupted write never leaves a truncated checkpoint."""
    state = checkpoint.state
    scalars = {k: v for k, v in state.items() if not isinstance(v, np.ndarray)}
    manifest = {"format": FORMAT, "meta": checkpoint.meta, "state": scalars}
    arrays: dict[str, NDArray[Any]] = {
        "field": checkpoint.field,
        "losses": checkpoint.losses,
        **{_STATE + k: v for k, v in state.items() if isinstance(v, np.ndarray)},
        _MANIFEST: np.array(json.dumps(manifest, sort_keys=True)),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with open(partial, "wb") as stream:
        np.savez(stream, **arrays)  # type: ignore[arg-type]  # numpy's stubs: str keys only
    os.replace(partial, path)


def load(path: str | os.PathLike[str]) -> Checkpoint:
    """Read the checkpoint at ``path``."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"resume_from={str(path)!r}: no such file. Pass a checkpoint file written by a "
            "synthesis with checkpoint_dir, such as <checkpoint_dir>/checkpoint_000010.npz."
        )
    try:
        with np.load(path, allow_pickle=False) as stored:
            manifest = json.loads(str(stored[_MANIFEST]))
            arrays = {name: stored[name] for name in stored.files}
    except (OSError, ValueError, KeyError) as error:
        raise ValueError(
            f"resume_from={str(path)!r} is not a synthesis checkpoint ({error}). Pass a file "
            "written by a synthesis with checkpoint_dir."
        ) from error
    if manifest.get("format") != FORMAT:
        raise ValueError(
            f"{path} has checkpoint format {manifest.get('format')!r}; this version of ESTEL "
            f"reads format {FORMAT}. Resume it with the version of ESTEL that wrote it."
        )
    state: LbfgsState = {
        **manifest["state"],
        **{k[len(_STATE) :]: v for k, v in arrays.items() if k.startswith(_STATE)},
    }
    return Checkpoint(
        field=arrays["field"], losses=arrays["losses"], state=state, meta=manifest["meta"]
    )
