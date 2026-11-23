"""The benchmark cases and their seeded inputs (Part I, §6.3; task T2.11, issue #89).

NumPy and the standard library only: the worker that measures STL in the parity container
imports this module, and never ``estel``.

The inputs of a case are written once, by the process that runs the benchmark, and every
measured process reads the same file (:func:`write_inputs`, :func:`read_inputs`): ESTEL and STL,
natively and in the container, start from the same bytes.
"""

from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

CASES_FILE = Path(__file__).resolve().parent / "cases.toml"
DTYPES = ("float32", "float64")


@dataclass(frozen=True)
class Settings:
    """What every case shares: the dtype, the device, and how times are measured."""

    dtype: str
    device: str
    threads: int
    warmup: int
    repeats: int


@dataclass(frozen=True)
class Case:
    """One benchmark case: a field of one batch element, and the operator's parameters. A 2D
    field by default; with ``blocks``, a tensor product of those blocks (``"2D"``, ``"1D"``, in
    axis order; task T3.2), with one J, one L and one ``pbc`` per block, measured on ESTEL
    only."""

    name: str
    shape: tuple[int, ...]
    channels: int
    J: int | tuple[int, ...]
    L: int | tuple[int, ...]
    pbc: bool | tuple[bool, ...]
    seed: int
    target_seed: int
    mask_seed: int | None = None
    mask_holes: tuple[int, ...] = ()
    blocks: tuple[str, ...] = ()

    @property
    def tensor_prod(self) -> bool:
        """Whether the case is a tensor product, which STL does not run here (ESTEL only)."""
        return bool(self.blocks)


def _case(name: str, spec: dict[str, Any]) -> Case:
    """A :class:`Case` from its table, lists as tuples."""
    tupled = {
        key: tuple(value) if isinstance(value, list) else value for key, value in spec.items()
    }
    return Case(name=name, **cast(Any, {"mask_holes": (), **tupled}))


@dataclass(frozen=True)
class Inputs:
    """The inputs of a case, as the measured processes read them: the field and the second
    field, ``(1, Nc, N, M)`` in the dtype of the run, and the missing pixels (True = missing),
    ``(N, M)``, or None."""

    case: Case
    dtype: str
    field: NDArray[Any]
    target_field: NDArray[Any]
    mask: NDArray[np.bool_] | None
    digest: str


def load_cases(path: Path = CASES_FILE) -> tuple[Settings, dict[str, Case]]:
    """The settings and the cases of a cases file, ``bench/cases.toml`` by default."""
    with path.open("rb") as stream:
        content = tomllib.load(stream)
    try:
        settings = Settings(**content["settings"])
        cases = {name: _case(name, spec) for name, spec in content["cases"].items()}
    except (KeyError, TypeError) as error:
        raise ValueError(
            f"{path}: {error}. A cases file has a [settings] table (dtype, device, threads, "
            "warmup, repeats) and one [cases.<name>] table per case (shape, channels, J, L, pbc, "
            "seed, target_seed, and mask_seed with mask_holes for missing pixels): see "
            "bench/cases.toml."
        ) from error
    if settings.dtype not in DTYPES:
        raise ValueError(f"{path}: dtype={settings.dtype!r}; set it to one of {DTYPES}.")
    for case in cases.values():
        if case.tensor_prod and case.mask_holes:
            raise ValueError(
                f"{path}: case {case.name!r} is a tensor product, which has no missing values "
                "in ESTEL 1.0.0 (LF8): remove mask_seed and mask_holes."
            )
        if bool(case.mask_holes) != (case.mask_seed is not None):
            raise ValueError(
                f"{path}: case {case.name!r} needs both mask_seed and mask_holes, or neither."
            )
    return settings, cases


def input_mask(case: Case) -> NDArray[np.bool_] | None:
    """The missing pixels of a case (True = missing), or None: square holes of the sides
    ``mask_holes``, each placed by a seeded uniform draw, drawn again while it touches a hole
    already placed. The holes do not overlap, so the missing fraction is exact."""
    if case.mask_seed is None:
        return None
    rng = np.random.default_rng(case.mask_seed)
    missing = np.zeros(case.shape, dtype=bool)
    for side in case.mask_holes:
        if side > min(case.shape):
            raise ValueError(
                f"case {case.name!r}: a hole of side {side} does not fit in {case.shape}; "
                "use smaller mask_holes."
            )
        for _ in range(10_000):
            row = int(rng.integers(0, case.shape[0] - side + 1))
            col = int(rng.integers(0, case.shape[1] - side + 1))
            hole = (slice(row, row + side), slice(col, col + side))
            if not missing[hole].any():
                missing[hole] = True
                break
        else:
            raise ValueError(
                f"case {case.name!r}: no room left for a hole of side {side} in {case.shape}; "
                "use fewer or smaller mask_holes."
            )
    return missing


def input_field(case: Case, seed: int) -> NDArray[np.float64]:
    """A seeded field, ``(1, Nc, N, M)``, float64: the exponential of a Gaussian field with a
    power-law spectrum, so that it has structure at every scale and S3 and S4 are not noise.
    Each channel is correlated with the previous one, so that the cross statistics are not
    small. A tensor product has the same spectrum over all its axes."""
    if case.tensor_prod:
        return _tensor_prod_field(case, seed)
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((case.channels, *case.shape))
    k = np.hypot(np.fft.fftfreq(case.shape[0])[:, None], np.fft.rfftfreq(case.shape[1])[None, :])
    k[0, 0] = 1.0  # the mean is removed below
    red = np.fft.irfft2(np.fft.rfft2(noise) / k, s=case.shape)
    red = red - red.mean(axis=(-2, -1), keepdims=True)
    red = red / red.std(axis=(-2, -1), keepdims=True)
    channels = [red[0]]
    for c in range(1, case.channels):
        channels.append(0.6 * channels[-1] + 0.8 * red[c])
    field: NDArray[np.float64] = np.exp(0.5 * np.stack(channels))[None]
    return field


def _tensor_prod_field(case: Case, seed: int) -> NDArray[np.float64]:
    """:func:`input_field` on every axis of a tensor product: ``(1, Nc, *shape)``."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((case.channels, *case.shape))
    axes = tuple(range(1, 1 + len(case.shape)))
    frequencies = [np.fft.fftfreq(n) for n in case.shape[:-1]] + [np.fft.rfftfreq(case.shape[-1])]
    k = np.sqrt(sum(f**2 for f in np.meshgrid(*frequencies, indexing="ij")))
    k.flat[0] = 1.0  # the mean is removed below
    red = np.fft.irfftn(np.fft.rfftn(noise, axes=axes) / k, s=case.shape, axes=axes)
    red = red - red.mean(axis=axes, keepdims=True)
    red = red / red.std(axis=axes, keepdims=True)
    channels = [red[0]]
    for c in range(1, case.channels):
        channels.append(0.6 * channels[-1] + 0.8 * red[c])
    field: NDArray[np.float64] = np.exp(0.5 * np.stack(channels))[None]
    return field


def inputs_digest(description: str, arrays: dict[str, NDArray[Any]]) -> str:
    """The SHA-256 of the inputs of a case: results belong to the inputs with the same digest."""
    digest = hashlib.sha256(description.encode())
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(f"{name}:{array.dtype}:{array.shape}".encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def write_inputs(case: Case, dtype: str, path: Path) -> str:
    """Write the inputs of ``case`` in ``dtype`` to ``path`` (``.npz``), with the case itself,
    and return their digest."""
    arrays: dict[str, NDArray[Any]] = {
        "field": input_field(case, case.seed).astype(dtype),
        "target_field": input_field(case, case.target_seed).astype(dtype),
    }
    mask = input_mask(case)
    if mask is not None:
        arrays["mask"] = mask
    described = asdict(case)
    if not case.blocks:  # the description of planar cases, and so their digest, as before T3.2
        del described["blocks"]
    description = json.dumps({"case": described, "dtype": dtype}, sort_keys=True)
    members: dict[str, Any] = {"description": np.array(description), **arrays}
    np.savez(path, **members)
    return inputs_digest(description, arrays)


def read_inputs(path: Path) -> Inputs:
    """The inputs written by :func:`write_inputs`."""
    with np.load(path, allow_pickle=False) as stored:
        text = str(stored["description"])
        arrays = {key: stored[key] for key in stored.files if key != "description"}
        description = json.loads(text)
        spec = dict(description["case"])
        case = _case(spec.pop("name"), spec)
        return Inputs(
            case=case,
            dtype=description["dtype"],
            field=arrays["field"],
            target_field=arrays["target_field"],
            mask=arrays.get("mask"),
            digest=inputs_digest(text, arrays),
        )
