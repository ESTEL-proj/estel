"""The configurations of the differentiation tests, and the function they differentiate
(task T2.7, issue #19; Part I, §3.2.2).

Each case is ``estel.functional.scattering``, the pure function behind ``STOperator.apply``,
seen as a real function of the data array: it returns every coefficient that exists, the real
and the imaginary part of each complex one (``complex_to_real_pairs`` of the full grids, at the
positions the static layout marks present), and the moments removed by ``standardize``.
``to_vector`` would hide the imaginary parts of S1 and S2 (M-FLAT).

The cases are the smallest data that run the whole loop, J = 2. Together they cover 1D, 2D and
HEALPix (task T5.6: the full sky at ``nside`` 4, and a partial sky of 80 pixels at ``nside`` 8
with 2 missing pixels, the smallest where every mean keeps a valid pixel), ``pbc`` on
and off, missing values, two channels with their cross terms, ``standardize``, a batch, and the
three normalisations (M-NORM). With ``norm="self"`` the reference is part of
the function, so its derivative is part of what is checked; with ``norm="ref"`` the reference
comes from other data and is fixed.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

import estel
from estel import Data, Data1D, Data2D, DataHealpix, STOperator, STStatistics
from estel.backend._protocol import Backend

Function = Callable[[Any], Any]


@dataclass(frozen=True)
class Case:
    """One configuration: the data (batch, channels and spatial shape), and the operator. On
    HEALPix, ``pbc`` follows ``cell_ids`` (True on the full sky), and ``depth`` is that of the
    map."""

    name: str
    cls: type[Data2D] | type[Data1D] | type[DataHealpix]
    shape: tuple[int, ...]
    pbc: bool
    masked: bool = False
    options: dict[str, Any] = field(default_factory=dict)
    depth: int | None = None
    cell_ids: tuple[int, ...] | None = None


SMALLEST = Case("1d-pbc", Data1D, (1, 1, 8), pbc=True, options={"norm": "none"})
"""The case of the fast tier: one channel, 8 pixels, 24 coefficients."""

CASES = (
    SMALLEST,
    Case(
        "1d-open-mask-cross-self",
        Data1D,
        (1, 2, 16),
        pbc=False,
        masked=True,
        options={"norm": "self", "L": 2},
    ),
    Case(
        "1d-batch-standardize-ref",
        Data1D,
        (2, 1, 16),
        pbc=True,
        options={"norm": "ref", "standardize": True},
    ),
    Case("2d-pbc-cross-none", Data2D, (1, 2, 8, 8), pbc=True, options={"norm": "none", "L": 2}),
    Case(
        "2d-open-mask-standardize-self",
        Data2D,
        (1, 1, 16, 16),
        pbc=False,
        masked=True,
        options={"norm": "self", "standardize": True, "L": 2},
    ),
    Case(
        "2d-batch-mean-self",
        Data2D,
        (2, 1, 8, 8),
        pbc=True,
        options={"norm": "self", "ref_batch_mean": True},
    ),
    Case("2d-open-cross-ref", Data2D, (1, 2, 8, 8), pbc=False, options={"norm": "ref", "L": 2}),
    Case(
        "healpix-full-none", DataHealpix, (1, 1, 192), pbc=True, depth=2, options={"norm": "none"}
    ),
    Case(
        "healpix-partial-mask-self",
        DataHealpix,
        (1, 1, 80),
        pbc=False,
        masked=True,
        depth=3,
        cell_ids=tuple(range(80)),  # base face 0 and part of face 1, nside 8
        options={"norm": "self"},
    ),
)
J = 2


def field_of(case: Case, seed: int = 0) -> NDArray[np.float64]:
    values: NDArray[np.float64] = np.random.default_rng(seed).standard_normal(case.shape) + 0.2
    return values


def data_of(case: Case, seed: int = 0) -> Data:
    """Seeded data of the case. The missing pixels, two in a row along each axis, hold finite
    values, which are never read (OP-27): the derivatives with respect to them are zero."""
    mask = None
    if case.masked:
        mask = np.zeros(case.shape[2:], dtype=bool)
        mask[(slice(3, 5),) * case.cls.ndim] = True
    if issubclass(case.cls, DataHealpix):
        values = field_of(case, seed)
        return DataHealpix(values, mask=mask, depth=case.depth, cell_ids=case.cell_ids)
    return case.cls(field_of(case, seed), mask=mask, pbc=case.pbc)


def operator_of(case: Case, data: Data, **options: Any) -> STOperator:
    defaults: dict[str, Any] = {"compute_PS": False if data._geometry == "2d" else None}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # OP-3, OP-13: tested with the plan
        return STOperator(data, J, **{**defaults, **case.options, **options})


def coefficients(backend: Backend, stats: STStatistics) -> Any:
    """Every coefficient that exists, as one real vector: for each family the present
    positions of the full grid, complex values as (real, imaginary) pairs; then the mean and
    the standard deviation removed by ``standardize``, when there are some."""
    xp = backend.xp
    parts = []
    for grid in stats._grids:
        full = stats.full_grid(grid.name)
        present = grid.present
        if grid.is_complex:
            full = backend.complex_to_real_pairs(full)
            present = np.repeat(present[..., None], 2, axis=-1)
        flat = xp.reshape(full, (full.shape[0], -1))
        kept = xp.take(flat, xp.asarray(np.flatnonzero(present)), axis=1)
        parts.append(xp.reshape(kept, (-1,)))
    if stats._standardization is not None:
        parts.extend(xp.reshape(moment, (-1,)) for moment in stats._standardization)
    return xp.concat(parts)


def scattering_function(backend: Backend, case: Case) -> tuple[Data, Function]:
    """The data of the case and the function ``array -> coefficients`` to differentiate at
    ``data.array``. With ``norm="ref"``, the reference is that of other data (another seed)."""
    data = data_of(case)
    operator = operator_of(case, data)
    reference = None
    if operator.norm == "ref":
        storing = operator_of(case, data, norm="self")
        reference = estel.functional.scattering(data_of(case, 1), storing.plan, storing.filters)

    def function(array: Any) -> Any:
        stats = estel.functional.scattering(
            data.with_array(array), operator.plan, operator.filters, reference
        )
        return coefficients(backend, stats)

    return data, function


def direction_like(backend: Backend, array: Any, seed: int) -> Any:
    """A seeded direction with the shape of ``array``, of unit root mean square."""
    values = np.random.default_rng(seed).standard_normal(tuple(array.shape))
    return backend.asarray(values, "float64")


# ---------------------------------------------------------------------- the power spectrum (T4.1)


@dataclass(frozen=True)
class SpectrumCase:
    """One configuration of the 2D power spectrum (maths.md §1.9; tasks T4.1, T4.6), through
    ``STOperator`` with the spectrum and ``estel.functional.scattering``: J = 1 and L = 1 where
    the case is about the spectrum (decision 5 on #191), so that it dominates the cost."""

    name: str
    shape: tuple[int, ...]
    pbc: bool
    masked: bool = False
    J: int = 1
    L: int = 1
    options: dict[str, Any] = field(default_factory=dict)


SPECTRUM_SMALLEST = SpectrumCase("ps-fft-self", (1, 1, 16, 16), pbc=True)
"""The spectrum case of the fast tier: one map of 16 x 16 pixels, ``fft``, ``norm="self"``."""

SPECTRUM_CASES = (
    SPECTRUM_SMALLEST,
    SpectrumCase("ps-pbc-mask-cross-none", (1, 2, 16, 16), pbc=False, options={"norm": "none"}),
    SpectrumCase(
        "ps-nan-mask-batch-mean",
        (2, 1, 16, 16),
        pbc=True,
        masked=True,
        options={"ref_batch_mean": True},
    ),
    SpectrumCase(
        "ps-pbc-nan-mask-ref", (1, 1, 16, 16), pbc=False, masked=True, options={"norm": "ref"}
    ),
    SpectrumCase(
        "st-ps-open-mask-standardize-self",
        (1, 1, 16, 16),
        pbc=False,
        masked=True,
        J=J,
        L=2,
        options={"norm": "self", "standardize": True},
    ),
)


def spectrum_data_of(case: SpectrumCase, seed: int = 0) -> Data2D:
    mask = None
    if case.masked:
        mask = np.zeros(case.shape[2:], dtype=bool)
        mask[6:8, 5:7] = True
    values = np.random.default_rng(seed).standard_normal(case.shape) + 0.2
    return Data2D(values, mask=mask, pbc=case.pbc)


def spectrum_function(backend: Backend, case: SpectrumCase) -> tuple[Data, Function]:
    """The data of the case and the function ``array -> coefficients``: every coefficient that
    exists (``coefficients``), the spectrum included, through ``estel.functional.scattering``.
    With ``norm="ref"``, the reference is that of other data."""
    data = spectrum_data_of(case)

    def build(**options: Any) -> Any:
        merged = {**case.options, **options}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # OP-3, OP-13: tested with the plan
            return STOperator(data, case.J, case.L, **merged)

    operator = build()
    pure = estel.functional.scattering
    reference = None
    if operator.plan.norm == "ref":
        storing = build(norm="self")
        reference = pure(spectrum_data_of(case, 1), storing.plan, storing.filters)

    def function(array: Any) -> Any:
        stats = pure(data.with_array(array), operator.plan, operator.filters, reference)
        return coefficients(backend, stats)

    return data, function
