"""The normalisation of the statistics (maths.md M-NORM, OP-18, OP-30; Part II, §12.6
``_normalize``; task T2.5).

``norm="store_ref"`` normalises the statistics by themselves: the reference is computed from
them (:func:`store_reference`), is part of the computation, and gradients flow through it.
``norm="load_ref"`` normalises them by the reference of other statistics, a constant: gradients
do not flow into it (LF21, closed). With ``norm_batch_mean``, the ``store_ref`` reference is
first averaged over the batch.

The reference (:class:`_NormReference`, internal) holds STL's ``var_ref`` and
``S2_ref_sqrt_chan_diag``, sigma_c(j, l) = √S2[c, c, j, l], and where sigma exists: the channels
with auto statistics, at the scales with valid pixels. The divisors of M-NORM are products of
sigma; where a sigma does not exist the coefficient is NaN, as in STL (OP-30, OP-3). Absent
coefficients
and divisors are replaced by 0 and 1 before the division and by NaN after it, with static
masks, so that no NaN reaches the gradients.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from ..backend._protocol import Backend
from ._arrays import masked, nan_absent, zero_absent
from ._grid import FamilyGrid

if TYPE_CHECKING:
    from ..scattering._plan import ScatteringPlan


@dataclass(frozen=True, eq=False)
class _NormReference:
    """The normalisation reference of M-NORM: STL's ``var_ref`` and ``S2_ref_sqrt_chan_diag``.

    Internal (issue #17): the operator stores it with ``store_ref``, and ``load_ref`` reads it
    from the operator or from the statistics passed to ``apply``. From step 2 it also holds the
    first moments of the centred covariances (Part III, §20.11).
    """

    var: Any
    """``var_ref``, ``(Nr, Nc)``, real; Nr = 1 with ``norm_batch_mean``, else the batch size."""
    s2_sqrt: Any
    """sigma, STL's ``S2_ref_sqrt_chan_diag``: ``(Nr, Nc, J, L)``, complex, NaN where it does
    not exist."""
    present: NDArray[np.bool_]
    """``(Nc, J)``, static: where sigma exists (``_grid.auto_scales``)."""
    norm_batch_mean: bool
    """Whether the reference was averaged over the batch."""

    @property
    def n_batch(self) -> int:
        return int(self.var.shape[0])

    def static(self) -> tuple[Any, ...]:
        """The static part, hashable: for the pytree of ``STStatistics`` and for ``save``."""
        return (self.present.shape, self.present.tobytes(), self.norm_batch_mean)

    @classmethod
    def from_static(cls, static: tuple[Any, ...], var: Any, s2_sqrt: Any) -> _NormReference:
        shape, data, norm_batch_mean = static
        present = np.frombuffer(data, dtype=bool).reshape(shape).copy()
        present.flags.writeable = False
        return cls(var=var, s2_sqrt=s2_sqrt, present=present, norm_batch_mean=norm_batch_mean)


def store_reference(
    backend: Backend,
    values: dict[str, Any],
    present: NDArray[np.bool_],
    norm_batch_mean: bool,
) -> _NormReference:
    """The reference of ``store_ref``, from the statistics being normalised (M-NORM; STL
    ``to_norm(norm_type="self")``, ``_get_sqrt_chan_diag``). ``present`` is where sigma exists."""
    xp = backend.xp
    var, s2 = values["var"], values["S2"]
    n_channels = int(s2.shape[1])
    diagonal = xp.stack([s2[:, c, c] for c in range(n_channels)], axis=1)  # (Nb, Nc, J, L)
    where = present[:, :, None]
    diagonal = masked(backend, where, diagonal, xp.ones_like(diagonal))
    if norm_batch_mean:
        var = xp.mean(var, axis=0, keepdims=True)
        diagonal = xp.mean(diagonal, axis=0, keepdims=True)
    s2_sqrt = nan_absent(backend, where, xp.sqrt(diagonal))
    return _NormReference(
        var=var, s2_sqrt=s2_sqrt, present=present, norm_batch_mean=norm_batch_mean
    )


def as_constant(backend: Backend, reference: _NormReference) -> _NormReference:
    """The reference of ``load_ref``: a constant, through which no gradient flows (LF21)."""
    return _NormReference(
        var=backend.stop_gradient(reference.var),
        s2_sqrt=backend.stop_gradient(reference.s2_sqrt),
        present=reference.present,
        norm_batch_mean=reference.norm_batch_mean,
    )


def normalize(
    backend: Backend,
    values: dict[str, Any],
    grids: tuple[FamilyGrid, ...],
    reference: _NormReference,
) -> tuple[dict[str, Any], tuple[FamilyGrid, ...]]:
    """The statistics divided as in the table of M-NORM (STL ``to_norm``,
    ``_normalize_scatcov``), and their new layout: NaN where a divisor does not exist.

    ``mean`` is not normalised; ``var`` is divided by ``var_ref``; S1 by
    √(sigma_{c1} sigma_{c2}) and S2 by sigma_{c1} sigma_{c2} at (j, l); S3 by
    sigma_{c1}(j2, l2) sigma_{c2}(j3, l3); S4 by sigma_{c1}(j1, l1) sigma_{c2}(j2, l2). The
    reference broadcasts over the batch (Nr = 1) or matches it.
    """
    xp = backend.xp
    p = reference.present  # (Nc, J): where sigma exists
    sigma = masked(backend, p[:, :, None], reference.s2_sqrt, xp.ones_like(reference.s2_sqrt))
    nc, n_scales, n_orientations = (int(n) for n in sigma.shape[1:])

    def expand(x: Any, axes: tuple[int, ...]) -> Any:
        """sigma ``(Nr, Nc, J, L)`` with new axes of size 1 at ``axes`` (result positions)."""
        out = x
        for axis in axes:
            out = xp.expand_dims(out, axis=axis)
        return out

    s1_s2 = expand(sigma, (2,)) * expand(sigma, (1,))  # (Nr, Nc, Nc, J, L)
    s3_divisor = expand(sigma, (2, 4, 6)) * expand(sigma, (1, 3, 5))  # (.., J, J, L, L)
    s4_divisor = expand(sigma, (2, 4, 5, 7, 8)) * expand(sigma, (1, 3, 5, 6, 8))
    exists_12 = p[:, None, :] & p[None, :, :]  # (Nc, Nc, J)
    exists = {
        "S1": np.broadcast_to(exists_12[..., None], (nc, nc, n_scales, n_orientations)),
        "S2": np.broadcast_to(exists_12[..., None], (nc, nc, n_scales, n_orientations)),
        "S3": np.broadcast_to(
            (p[:, None, :, None] & p[None, :, None, :])[..., None, None],
            (nc, nc, n_scales, n_scales, n_orientations, n_orientations),
        ),
        "S4": np.broadcast_to(
            (p[:, None, :, None, None] & p[None, :, None, :, None])[..., None, None, None],
            (nc, nc, *(n_scales,) * 3, *(n_orientations,) * 3),
        ),
    }
    divisors = {"S1": xp.sqrt(s1_s2), "S2": s1_s2, "S3": s3_divisor, "S4": s4_divisor}

    out: dict[str, Any] = {**values, "var": values["var"] / reference.var}
    new_grids: list[FamilyGrid] = []
    for grid in grids:
        if grid.name not in divisors:
            new_grids.append(grid)
            continue
        present = grid.present & exists[grid.name]
        safe = zero_absent(backend, grid.present, values[grid.name])
        quotient = safe / divisors[grid.name]
        out = {**out, grid.name: nan_absent(backend, present, quotient)}
        new_grids.append(grid.replace(present=present))
    return out, tuple(new_grids)


def check_reference(
    reference: _NormReference, plan: ScatteringPlan, n_batch: int, dtype: Any
) -> None:
    """Raise unless ``reference`` can normalise the statistics of ``plan``, before they are
    computed (maths.md M-NORM).

    The reference must have the J, L and number of channels of the plan, and a normalisation
    scale sigma for every channel the plan normalises: each channel with auto statistics in the
    plan. A channel the plan meets only in cross pairs has no scale under ``store_ref`` either,
    and its coefficients stay NaN when the reference has none (OP-30). The size, the mask and
    the periodicity of the data the reference comes from are free: the syntheses normalise by
    the reference of a target of another size. A scale the reference lacks because its mask
    left it empty gives NaN, as in STL (OP-3). The batch is 1 or that of the data, as STL
    broadcasts; the dtype is that of the operator.
    """
    found_nc, found_j = (int(n) for n in reference.present.shape)
    found_l = int(reference.s2_sqrt.shape[-1])
    mismatches = [
        f"{name} = {found} (the operator has {expected})"
        for name, found, expected in (
            ("J", found_j, plan.J),
            ("L", found_l, plan.L),
            ("the number of channels", found_nc, plan.n_channels),
        )
        if found != expected
    ]
    if mismatches:
        raise ValueError(
            f"the reference has {', '.join(mismatches)}: normalise with statistics computed by an "
            "operator with the same J, L and channels; their size, mask and pbc may differ "
            "(maths.md M-NORM)."
        )
    with_scale: NDArray[np.bool_] = np.any(reference.present, axis=1)
    missing = [c for c in plan.pairs.auto if not with_scale[c]]
    if missing:
        raise ValueError(
            f"the reference has no normalisation scale for channel(s) {missing}, which this "
            "operator normalises (they have auto statistics in compute_cross_matrix): compute "
            "the reference with their diagonal entries set, so that it holds their auto "
            "statistics (maths.md M-NORM, M-CHAN, OP-30)."
        )
    if reference.n_batch not in (1, n_batch):
        raise ValueError(
            f"the reference was computed on {reference.n_batch} batch elements without "
            f"norm_batch_mean, and these data have {n_batch}: pass data with {reference.n_batch} "
            "batch element(s), or compute the reference with norm_batch_mean=True (maths.md "
            "M-NORM)."
        )
    if reference.var.dtype != dtype:
        raise TypeError(
            f"the reference is {reference.var.dtype}, but the operator computes in {dtype}: "
            "compute both with the same default dtype (estel.set_default_dtype)."
        )
