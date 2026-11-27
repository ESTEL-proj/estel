"""The benchmark cases and their seeded inputs (Part I, §6.3; task T2.11, issue #89).

NumPy and the standard library only: the worker that measures STL in the parity container
imports this module, and never ``estel``. The HEALPix cases, measured on ESTEL only, read the
pixel centres from ``estel`` when their inputs are written, natively (:func:`_healpix_centres`).

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
    only; with ``geometry = "healpix"``, a full sky of ``shape = [12 · 4^depth]`` pixels in the
    NESTED ordering (task T5.7), measured on ESTEL only, with missing pixels in discs of the
    angular radii ``mask_discs``, in degrees."""

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
    geometry: str = "2D"
    mask_discs: tuple[float, ...] = ()

    @property
    def tensor_prod(self) -> bool:
        """Whether the case is a tensor product, which STL does not run here (ESTEL only)."""
        return bool(self.blocks)

    @property
    def healpix(self) -> bool:
        """Whether the case is a HEALPix full sky, which STL does not run here (ESTEL only)."""
        return self.geometry == "healpix"

    @property
    def estel_only(self) -> bool:
        """Whether STL is not run on the case: tensor products and HEALPix (Part I, §6.3)."""
        return self.tensor_prod or self.healpix

    @property
    def depth(self) -> int:
        """log₂ ``nside`` of a HEALPix case, from its 12 · 4^depth pixels."""
        return (self.shape[0] // 12).bit_length() // 2


def _case(name: str, spec: dict[str, Any]) -> Case:
    """A :class:`Case` from its table, lists as tuples."""
    tupled = {
        key: tuple(value) if isinstance(value, list) else value for key, value in spec.items()
    }
    return Case(name=name, **cast(Any, {"mask_holes": (), "mask_discs": (), **tupled}))


# Fields of :class:`Case` added after the first cases, left out of the description, and so of
# the digest, of the cases that keep their default: their earlier results stay comparable.
_LATER_FIELDS = {"blocks": (), "geometry": "2D", "mask_discs": ()}


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
        _check_geometry(path, case)
        if case.tensor_prod and case.mask_holes:
            raise ValueError(
                f"{path}: case {case.name!r} is a tensor product, which has no missing values "
                "in ESTEL 1.0.0 (LF8): remove mask_seed and mask_holes."
            )
        holes = case.mask_discs if case.healpix else case.mask_holes
        if bool(holes) != (case.mask_seed is not None):
            name = "mask_discs" if case.healpix else "mask_holes"
            raise ValueError(
                f"{path}: case {case.name!r} needs both mask_seed and {name}, or neither."
            )
    return settings, cases


def _check_geometry(path: Path, case: Case) -> None:
    """A HEALPix case is a full sky, one channel axis and one pixel axis, with discs and not
    square holes; the other cases have no discs."""
    if case.geometry not in ("2D", "healpix"):
        raise ValueError(
            f"{path}: case {case.name!r} has geometry={case.geometry!r}; leave it out for 2D "
            'data, or set it to "healpix".'
        )
    if not case.healpix:
        if case.mask_discs:
            raise ValueError(
                f"{path}: case {case.name!r}: mask_discs are for HEALPix cases; use mask_holes."
            )
        return
    n_pix = case.shape[0] if len(case.shape) == 1 else 0
    if n_pix < 12 or 12 * 4**case.depth != n_pix:
        raise ValueError(
            f"{path}: case {case.name!r} is a HEALPix full sky: shape = [12 · 4^depth], such as "
            f"[49152] for nside 64; got {list(case.shape)}."
        )
    if case.pbc is not True or case.blocks or case.mask_holes:
        raise ValueError(
            f"{path}: case {case.name!r} is a HEALPix full sky: pbc = true, no blocks, and "
            "missing pixels given by mask_discs (angular radii in degrees), not mask_holes."
        )


def input_mask(case: Case) -> NDArray[np.bool_] | None:
    """The missing pixels of a case (True = missing), or None: square holes of the sides
    ``mask_holes``, each placed by a seeded uniform draw, drawn again while it touches a hole
    already placed. The holes do not overlap, so the missing fraction is exact."""
    if case.mask_seed is None:
        return None
    if case.healpix:
        return _healpix_mask(case)
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
    if case.healpix:
        return _healpix_field(case, seed)
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


# ---------------------------------------------------------------------- HEALPix (task T5.7)

# The field of a HEALPix case: a white noise averaged this many times over the pixels whose
# centres lie within this many pixel sizes, √(4π / Npix), of each pixel centre.
HEALPIX_AVERAGES = 3
HEALPIX_AVERAGE_RADIUS = 2.0


def _healpix_centres(case: Case) -> NDArray[np.float64]:
    """The unit vectors of the pixel centres of a HEALPix case, NESTED, ``(Npix, 3)``: from
    ``estel``, imported here only, natively, where the inputs are written (the container never
    runs a HEALPix case)."""
    from estel.wavelets._tables_healpix import centres_lonlat

    lon, lat = centres_lonlat(np.arange(case.shape[0], dtype=np.int64), case.depth)
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1)


def _healpix_mask(case: Case) -> NDArray[np.bool_]:
    """The missing pixels of a HEALPix case: discs of the angular radii ``mask_discs``, in
    degrees, centred on directions drawn uniformly on the sphere by a seeded draw, drawn again
    while a disc overlaps one already placed; a pixel is missing when its centre is in a disc."""
    assert case.mask_seed is not None
    rng = np.random.default_rng(case.mask_seed)
    centres = _healpix_centres(case)
    placed: list[tuple[NDArray[np.float64], float]] = []
    missing = np.zeros(case.shape[0], dtype=bool)
    for degrees in case.mask_discs:
        radius = np.radians(degrees)
        for _ in range(10_000):
            direction = rng.standard_normal(3)
            direction /= np.linalg.norm(direction)
            apart = all(
                np.arccos(np.clip(direction @ other, -1.0, 1.0)) > radius + other_radius
                for other, other_radius in placed
            )
            if apart:
                placed.append((direction, radius))
                missing |= centres @ direction >= np.cos(radius)
                break
        else:
            raise ValueError(
                f"case {case.name!r}: no room left for a disc of {degrees} degrees on the "
                "sphere; use fewer or smaller mask_discs."
            )
    return missing


def _healpix_field(case: Case, seed: int) -> NDArray[np.float64]:
    """:func:`input_field` on a HEALPix full sky, ``(1, Nc, Npix)``: a Gaussian white noise
    averaged :data:`HEALPIX_AVERAGES` times over the pixels within
    :data:`HEALPIX_AVERAGE_RADIUS` pixel sizes of each pixel, standardised, then exponentiated
    as on 2D data. No spherical-harmonic transform and no healpy (Part I, §1.5)."""
    from estel.wavelets._tables_healpix import _cone_pairs, centres_lonlat

    n_pix = case.shape[0]
    lon, lat = centres_lonlat(np.arange(n_pix, dtype=np.int64), case.depth)
    radius = HEALPIX_AVERAGE_RADIUS * np.sqrt(4 * np.pi / n_pix)
    rows, cols, _ = _cone_pairs(lon, lat, lon, lat, radius)
    counts = np.bincount(rows, minlength=n_pix)
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((case.channels, n_pix))
    red = []
    for channel in noise:
        for _ in range(HEALPIX_AVERAGES):
            channel = np.bincount(rows, weights=channel[cols], minlength=n_pix) / counts
        red.append((channel - channel.mean()) / channel.std())
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
    for name, default in _LATER_FIELDS.items():
        if described[name] == default:  # the description, and so the digest, of earlier cases
            del described[name]
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
