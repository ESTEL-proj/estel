"""The static layout of the full-grid statistics: named axes and the positions where a
coefficient exists (Part II, §12.7; maths.md M-CHAN, M-MEAN, M-FLAT; task T2.5).

In 1.0.0 the statistics are STL's full-grid tensors, NaN where no coefficient exists: the
structural NaN (j1 ≤ j2 ≤ j3, the channel pairs not computed, M-CHAN), the means with no valid
pixel (OP-3), the cross statistics of a channel without auto statistics after normalisation
(OP-30), and the harmonics of ``scale_ft`` built from missing scales (M-RED-SCALE). Every one of
them is known when the operator is built, from the plan alone. :class:`FamilyGrid` records them
as a static boolean array, ``present``, so that ``to_vector`` and the reductions never look for
NaN in the values (Part I, §3.1, rule 3), and so that gradients never meet them.

The axes have names, used by ``STStatistics.select`` (Part II, §12.7):

* ``mean``, ``var``: ``(c,)``;
* ``S1``, ``S2``: ``(c1, c2, j, l)``;
* ``S3``: ``(c1, c2, j2, j3, l2, l3)`` (maths.md M-S3, OP-15);
* ``S4``: ``(c1, c2, j1, j2, j3, l1, l2, l3)``.

``iso`` replaces the orientations of S1 and S2 by nothing, those of S3 by ``dl`` (l3 - l2 mod
L) and those of S4 by ``dl2``, ``dl3`` (l2 - l1, l3 - l1 mod L); ``angular_ft`` prefixes each
orientation axis it transforms with ``h`` (``hl2``, ``hdl``, …); ``scale_ft`` replaces the scales
of S3 by ``(k, a)`` and those of S4 by ``(k, a, b)`` (M-RED-ISO, M-RED-ANG, M-RED-SCALE). The
batch axis comes first in every array and is not named.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..scattering._plan import ScatteringPlan

FAMILIES = ("mean", "var", "S1", "S2", "S3", "S4")
"""The families of ESTEL 1.0.0, in STL's order (``to_flatten``); PS joins them in T4.1."""

REAL_PART_ONLY = ("S1", "S2", "PS")
"""The families whose imaginary part ``to_vector`` drops, as STL's ``to_flatten`` (M-FLAT,
OP-37)."""


@dataclass(frozen=True, eq=False)
class FamilyGrid:
    """The static layout of one family: its named axes, the original index of each position
    along them, and where its coefficients exist. Hashable, so that it can be static data of a
    pytree."""

    name: str
    """``"mean"``, ``"var"``, ``"S1"`` … ``"S4"``."""
    axes: tuple[str, ...]
    """The names of the axes after the batch axis."""
    coords: tuple[tuple[int, ...], ...]
    """For each axis, the original index of each of its positions (changed by ``select``)."""
    present: NDArray[np.bool_]
    """True where a coefficient exists, over the axes after the batch axis; read-only."""
    is_complex: bool
    """Whether the family is stored complex (S1 to S4) or real (mean, var)."""

    def __post_init__(self) -> None:
        if self.present.shape != tuple(len(c) for c in self.coords):
            raise ValueError(f"{self.name}: present {self.present.shape} does not match coords.")
        if len(self.axes) != len(self.coords):
            raise ValueError(f"{self.name}: {len(self.axes)} axes, {len(self.coords)} coords.")
        if self.present.flags.writeable:
            frozen = np.array(self.present, dtype=bool)
            frozen.flags.writeable = False
            object.__setattr__(self, "present", frozen)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.present.shape)

    def _key(self) -> tuple[Any, ...]:
        return (self.name, self.axes, self.coords, self.is_complex, self.present.tobytes())

    def __hash__(self) -> int:
        return hash(self._key())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FamilyGrid):
            return NotImplemented
        return self._key() == other._key()

    def replace(self, **changes: Any) -> FamilyGrid:
        fields = {
            "name": self.name,
            "axes": self.axes,
            "coords": self.coords,
            "present": self.present,
            "is_complex": self.is_complex,
        }
        return FamilyGrid(**{**fields, **changes})

    def axis(self, name: str) -> int:
        """The position of the axis ``name`` among :attr:`axes`."""
        return self.axes.index(name)

    def is_complete(self, names: Sequence[str], size: int) -> bool:
        """Whether every axis of ``names`` holds all of its ``size`` original positions, in
        order: what a reduction along them needs."""
        return all(self.coords[self.axis(n)] == tuple(range(size)) for n in names)


def _full(
    axes: tuple[str, ...], present: NDArray[np.bool_], is_complex: bool, name: str
) -> FamilyGrid:
    coords = tuple(tuple(range(n)) for n in present.shape)
    return FamilyGrid(name=name, axes=axes, coords=coords, present=present, is_complex=is_complex)


def base_grids(plan: ScatteringPlan) -> tuple[FamilyGrid, ...]:
    """The layout of the statistics without normalisation (maths.md M-MOM, M-S1 to M-S4,
    M-CHAN, OP-3): what the scattering loop computes, from the plan alone."""
    nc, J, L = plan.n_channels, plan.J, plan.L  # noqa: N806
    pairs = plan.pairs
    computed_12 = {(c, c) for c in pairs.auto} | set(pairs.cross)
    first = np.array([[(c1, c2) in computed_12 for c2 in range(nc)] for c1 in range(nc)])
    ordered = np.array([[(c1, c2) in pairs.ordered for c2 in range(nc)] for c1 in range(nc)])
    first = first.reshape(nc, nc)
    ordered = ordered.reshape(nc, nc)
    scales = np.array([j in plan.first_layer for j in range(J)], dtype=bool)
    s3 = np.array([[(j2, j3) in plan.s3 for j3 in range(J)] for j2 in range(J)], dtype=bool)
    s4 = np.array(
        [[[(j1, j2, j3) in plan.s4 for j3 in range(J)] for j2 in range(J)] for j1 in range(J)],
        dtype=bool,
    )
    moments = np.ones(nc, dtype=bool)
    s12 = np.broadcast_to(first[:, :, None, None] & scales[:, None], (nc, nc, J, L))
    s3_present = np.broadcast_to(
        ordered[:, :, None, None, None, None] & s3[:, :, None, None], (nc, nc, J, J, L, L)
    )
    s4_present = np.broadcast_to(
        ordered[:, :, None, None, None, None, None, None] & s4[:, :, :, None, None, None],
        (nc, nc, J, J, J, L, L, L),
    )
    return (
        _full(("c",), moments, False, "mean"),
        _full(("c",), moments, False, "var"),
        _full(("c1", "c2", "j", "l"), np.array(s12), True, "S1"),
        _full(("c1", "c2", "j", "l"), np.array(s12), True, "S2"),
        _full(("c1", "c2", "j2", "j3", "l2", "l3"), np.array(s3_present), True, "S3"),
        _full(("c1", "c2", "j1", "j2", "j3", "l1", "l2", "l3"), np.array(s4_present), True, "S4"),
    )


def auto_scales(plan: ScatteringPlan) -> NDArray[np.bool_]:
    """``(Nc, J)``: where the auto S2 exists, so where the normalisation scale sigma_c(j, l) of
    M-NORM is defined: the channels with auto statistics, at the scales with valid pixels."""
    present = np.array(
        [
            [c in plan.pairs.auto and j in plan.first_layer for j in range(plan.J)]
            for c in range(plan.n_channels)
        ],
        dtype=bool,
    ).reshape(plan.n_channels, plan.J)
    present.flags.writeable = False
    return present
