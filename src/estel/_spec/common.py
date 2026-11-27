"""The oracle of what every geometry shares: the means, the moments, the standardisation, the
channel pairs, the normalisation, the reductions and the flattening of ``docs/maths.md``.

A second implementation of the mathematics, written to be read against ``docs/maths.md`` rather
than to be fast (Part I, §4.2): float64, NumPy and the standard library only, nothing from the
rest of ``estel``. The loops over scales, orientations and channels are explicit, as the
formulas are written; only the sums over pixels are NumPy operations. The oracle of each
geometry calls these functions: ``cartesian`` (1D and 2D grids), ``healpix`` and
``tensor_prod``.

**Conventions.**

- ``ndim`` is the number of spatial axes, always the last ``ndim`` axes of the array; every
  function acts on them whatever the leading axes are.
- A mask is a boolean array of the spatial shape, True where a pixel is invalid (M-NOT), or
  ``None`` when no value is missing. Invalid pixels hold NaN (STL's ``replace_nan_value``,
  M-DOWN-MASK): no mean ever reads them, so a NaN in a result is a mean with no valid pixel
  (M-MEAN) or a coefficient that is not computed (M-CHAN).
- The statistics are STL's full grids (M-CHAN, §2.4): ``mean`` and ``var`` are ``(Nb, Nc)``,
  real; S1 and S2 ``(Nb, Nc, Nc, J, L)``, S3 ``(Nb, Nc, Nc, J, J, L, L)`` indexed
  ``[c1, c2, j2, j3, l2, l3]`` and S4 ``(Nb, Nc, Nc, J, J, J, L, L, L)`` indexed
  ``[c1, c2, j1, j2, j3, l1, l2, l3]``, complex, NaN where no coefficient exists.

Every public function lists the ``M-`` IDs it implements; ``tools/check_maths_ids.py`` checks
the mapping in both directions.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

Array = NDArray[Any]
Mask = NDArray[np.bool_] | None
Statistics = dict[str, Any]

S1_EPSILON = 1e-8
"""ε of the cross-channel S1 (M-S1)."""
FAMILIES = ("mean", "var", "S1", "S2", "S3", "S4")
"""The families, in the order of the flattening (M-FLAT)."""
SCATTERING = ("S1", "S2", "S3", "S4")
N_ORIENTATION_AXES = {"S1": 1, "S2": 1, "S3": 2, "S4": 3}
N_SCALE_AXES = {"S1": 1, "S2": 1, "S3": 2, "S4": 3}


# ---------------------------------------------------------------------- masks and means


def _union(a: Mask, b: Mask) -> Mask:
    """The union of the masks of the two arguments of a covariance (M-MASK, OP-35)."""
    if a is None or b is None:
        return None if a is None and b is None else (a if b is None else b)
    return a | b


def spec_valid_mean(a: Array, ndim: int, invalid: Mask = None, border: int = 0) -> Array:
    """The mean of ``a`` over its valid pixels, after the border crop: ``⟨a⟩_m``, over the last
    ``ndim`` axes. Implements M-NOT (the valid-pixel mean, M1) and M-MEAN, with the crop of
    M-BORDER.

    ``border`` pixels are removed at both ends of each axis; a border too large for the array
    (2β ≥ min N) is reduced to ⌊(min N - 1)/2⌋ (OP-13). The sum over the valid pixels is
    divided by their number; with no valid pixel the result is NaN (OP-3).
    """
    spatial = a.shape[a.ndim - ndim :]
    if border and 2 * border >= min(spatial):
        border = (min(spatial) - 1) // 2
    crop: tuple[Any, ...] = (..., *(slice(border, n - border) for n in spatial))
    kept = a[crop]
    valid = np.ones(kept.shape[kept.ndim - ndim :], dtype=bool)
    if invalid is not None:
        valid = ~invalid[crop]
    count = int(valid.sum())
    if count == 0:
        return np.full(kept.shape[: kept.ndim - ndim], np.nan, dtype=kept.dtype)
    axes = tuple(range(kept.ndim - ndim, kept.ndim))
    total: Array = np.where(valid, kept, 0.0).sum(axis=axes)
    return total / count


def spec_moments(x: Array, ndim: int, invalid: Mask = None) -> tuple[Array, Array]:
    """``mean`` and ``var`` of each map, ``(Nb, Nc)`` each. Implements M-MOM.

    ``mean = ⟨x⟩`` and ``var = ⟨x²⟩``: the second moment, not centred (OP-11).
    """
    return spec_valid_mean(x, ndim, invalid), spec_valid_mean(x * x, ndim, invalid)


def spec_standardize(x: Array, ndim: int, invalid: Mask = None) -> tuple[Array, Array, Array]:
    """The standardised field, with the mean μ and the standard deviation sigma removed from each
    map, ``(Nb, Nc)`` each. Implements M-STD.

    ``μ = ⟨x⟩``, ``sigma² = ⟨(x - μ)²⟩`` over the valid pixels, divided by their number, and
    ``x' = (x - μ) / sigma``.
    """
    expand = (..., *(None,) * ndim)
    mu = spec_valid_mean(x, ndim, invalid)
    centred = x - mu[expand]
    sigma = np.sqrt(spec_valid_mean(centred * centred, ndim, invalid))
    return centred / sigma[expand], mu, sigma


# ---------------------------------------------------------------------- channel pairs


def _channel_pairs(cross_matrix: Any, n_channels: int) -> NDArray[np.bool_]:
    """The boolean ``(Nc, Nc)`` matrix of the pairs c1 ≤ c2 to compute (M-CHAN): by default the
    upper triangle; the entries below the diagonal are ignored."""
    if cross_matrix is None:
        return np.triu(np.ones((n_channels, n_channels), dtype=bool))
    matrix = np.triu(np.asarray(cross_matrix, dtype=bool))
    if matrix.shape != (n_channels, n_channels):
        raise ValueError(f"cross_matrix must have shape ({n_channels}, {n_channels}) (M-CHAN).")
    in_no_pair = [c for c in range(n_channels) if not (matrix[c].any() or matrix[:, c].any())]
    if in_no_pair:
        raise ValueError(f"channel(s) {in_no_pair} are in no pair of cross_matrix (M-CHAN).")
    return matrix


# ---------------------------------------------------------------------- normalisation


def spec_reference(stats: Statistics, ref_batch_mean: bool = False) -> Statistics:
    """The normalisation reference of statistics computed without normalisation: ``var``,
    ``(Nr, Nc)``, and ``sigma``, ``(Nr, Nc, J, L)``, with ``sigma_c(j, l) = √S2[c, c, j, l]``.
    Implements M-NORM (the reference of ``norm="self"``).

    With ``ref_batch_mean`` the variance and the auto S2 are first averaged over the batch, and
    Nr = 1; otherwise Nr = Nb. A channel without auto statistics has sigma = NaN (OP-30).

    With the spectrum (a family ``PS`` in ``stats``), the reference also holds ``sigma_PS``,
    ``(Nr, Nc, n_b)``, with ``sigma_PS_c(b) = √PS[c, c, b]``, of the batch mean of the auto
    spectra with ``ref_batch_mean`` (M-PS-2D, "In the scattering operator"); a channel without
    auto spectrum has sigma_PS = NaN (OP-30).
    """
    var, s2 = stats["var"], stats["S2"]
    nc = s2.shape[1]
    diagonal = np.stack([s2[:, c, c] for c in range(nc)], axis=1)
    if ref_batch_mean:
        var = var.mean(axis=0, keepdims=True)
        diagonal = diagonal.mean(axis=0, keepdims=True)
    reference = {"var": var, "sigma": np.sqrt(diagonal)}
    if "PS" in stats:
        ps = stats["PS"]
        auto = np.stack([ps[:, c, c] for c in range(nc)], axis=1)
        if ref_batch_mean:
            auto = auto.mean(axis=0, keepdims=True)
        reference["sigma_PS"] = np.sqrt(auto)
    return reference


def spec_normalize(stats: Statistics, reference: Statistics) -> Statistics:
    """The statistics divided by the reference, as in the table of M-NORM. Implements M-NORM.

    ``mean`` is not normalised; ``var`` is divided by ``var_ref``; S1 by
    √(sigma_{c1}(j, l) sigma_{c2}(j, l)); S2 by sigma_{c1}(j, l) sigma_{c2}(j, l); S3 by
    sigma_{c1}(j2, l2) sigma_{c2}(j3, l3); S4 by sigma_{c1}(j1, l1) sigma_{c2}(j2, l2); PS, when
    the statistics hold it, by sigma_PS_{c1}(b) sigma_PS_{c2}(b) (M-PS-2D). A reference of one
    batch element applies to every batch element.
    """
    normalized: Statistics = {}
    if "PS" in stats:
        if "sigma_PS" not in reference:
            raise ValueError(
                "the statistics hold the spectrum but the reference has no sigma_PS (M-NORM, "
                "M-PS-2D); pass the reference of statistics computed with compute_PS=True."
            )
        sigma_ps = reference["sigma_PS"]
        ps = np.array(stats["PS"])
        if sigma_ps.shape[1:] != (ps.shape[1], ps.shape[3]):
            raise ValueError(
                f"the reference's sigma_PS has shape {sigma_ps.shape}, but the statistics have "
                f"{ps.shape[1]} channel(s) and {ps.shape[3]} bins (M-NORM, M-PS-2D); pass the "
                "reference of statistics with the same channels and the same bins (J, Jmin, "
                "n_bins)."
            )
        with np.errstate(invalid="ignore"):  # NaN where no pair or no sigma_PS exists
            for c1 in range(ps.shape[1]):
                for c2 in range(ps.shape[2]):
                    for b in range(ps.shape[3]):
                        divisor = sigma_ps[:, c1, b] * sigma_ps[:, c2, b]
                        ps[:, c1, c2, b] = ps[:, c1, c2, b] / divisor
        normalized["PS"] = ps
    sigma = reference["sigma"]
    nc, n_scales, n_orient = sigma.shape[1:]
    s1, s2, s3, s4 = (np.array(stats[k]) for k in SCATTERING)
    with np.errstate(invalid="ignore"):  # NaN where no coefficient or no sigma exists
        for c1 in range(nc):
            for c2 in range(nc):
                for j in range(n_scales):
                    for ell in range(n_orient):
                        divisor = sigma[:, c1, j, ell] * sigma[:, c2, j, ell]
                        s1[:, c1, c2, j, ell] = s1[:, c1, c2, j, ell] / np.sqrt(divisor)
                        s2[:, c1, c2, j, ell] = s2[:, c1, c2, j, ell] / divisor
                for j2, j3, l2, l3 in np.ndindex(n_scales, n_scales, n_orient, n_orient):
                    divisor = sigma[:, c1, j2, l2] * sigma[:, c2, j3, l3]
                    s3[:, c1, c2, j2, j3, l2, l3] = s3[:, c1, c2, j2, j3, l2, l3] / divisor
                for j1, j2, l1, l2 in np.ndindex(n_scales, n_scales, n_orient, n_orient):
                    divisor = sigma[:, c1, j1, l1] * sigma[:, c2, j2, l2]
                    # Every j3 and l3 at once: [b, j3, l3].
                    s4[:, c1, c2, j1, j2, :, l1, l2, :] = (
                        s4[:, c1, c2, j1, j2, :, l1, l2, :] / divisor[:, None, None]
                    )
    return {
        **stats,
        "var": stats["var"] / reference["var"],
        "S1": s1,
        "S2": s2,
        "S3": s3,
        "S4": s4,
        **normalized,
    }


# ---------------------------------------------------------------------- reductions


def spec_iso(stats: Statistics) -> Statistics:
    """The isotropic statistics. Implements M-RED-ISO.

    S1 and S2 are averaged over l. S3 becomes ``[.., δ]`` with
    ``(1/L) Σ_{l2} S3[.., l2, l2 + δ]``, and S4 ``[.., δ2, δ3]`` with
    ``(1/L) Σ_{l1} S4[.., l1, l1 + δ2, l1 + δ3]``, the orientations taken modulo L (OP-5).
    """
    s1, s2, s3, s4 = (stats[k] for k in SCATTERING)
    n_orient = s1.shape[-1]
    iso3 = np.zeros(s3.shape[:-1], dtype=s3.dtype)
    iso4 = np.zeros(s4.shape[:-1], dtype=s4.dtype)
    for delta in range(n_orient):
        for l2 in range(n_orient):
            iso3[..., delta] += s3[..., l2, (l2 + delta) % n_orient] / n_orient
    for d2 in range(n_orient):
        for d3 in range(n_orient):
            for l1 in range(n_orient):
                iso4[..., d2, d3] += (
                    s4[..., l1, (l1 + d2) % n_orient, (l1 + d3) % n_orient] / n_orient
                )
    return {**stats, "S1": s1.mean(axis=-1), "S2": s2.mean(axis=-1), "S3": iso3, "S4": iso4}


def spec_angular_fourier(
    stats: Statistics, harmonics: int | None = None, after_iso: bool = False
) -> Statistics:
    """The angular Fourier transform of S3 and S4. Implements M-RED-ANG.

    An orthonormal discrete Fourier transform, ``L^(-1/2) Σ_l x[l] exp(-2πi h l / L)``, along
    each orientation axis of S3 and S4 (their last 2 and 3 axes; 1 and 2 after ``iso``), keeping
    the harmonics h < ``harmonics``: all L by default (OP-17).
    """
    out = dict(stats)
    for family in ("S3", "S4"):
        x = stats[family]
        n_orient = x.shape[-1]
        kept = n_orient if harmonics is None else harmonics
        h, ell = np.arange(kept), np.arange(n_orient)
        basis = np.exp(-2j * np.pi * np.outer(h, ell) / n_orient) / math.sqrt(n_orient)  # [h, l]
        n_axes = N_ORIENTATION_AXES[family] - (1 if after_iso else 0)
        for axis in range(-n_axes, 0):
            # The sum over l along this axis; tensordot puts the new axis h last: move it back.
            x = np.moveaxis(np.tensordot(x, basis, axes=([axis], [1])), -1, axis)
        out[family] = x
    return out


def spec_scale_cosine(
    stats: Statistics, harmonics: int | None = None, dj: int | None = None
) -> Statistics:
    """The scale cosine transform of S3 and S4, after ``iso``. Implements M-RED-SCALE.

    The scales are reparametrised: S3 as ``(j1, a)`` reading the scales ``(j1, j1 + a)``, S4 as
    ``(j1, a, b)`` reading ``(j1, j1 + a, j1 + b)``, with 0 ≤ a ≤ b < ``dj`` (J by default). The
    missing entries are set to 0, the orthonormal DCT-II
    ``F[k, j] = alpha_k cos(kπ (j + 1/2) / J)`` is applied along j1, the missing entries are set
    back to NaN at their positions, and the harmonics k < ``harmonics`` (J by default) are kept.
    """
    out = dict(stats)
    for family in ("S3", "S4"):
        x = stats[family]  # (Nb, Nc, Nc, J, .., J, *orientations)
        n_scales = x.shape[3]
        width = n_scales if dj is None else dj
        kept = n_scales if harmonics is None else harmonics
        n_axes = N_SCALE_AXES[family]
        rest = x.shape[3 + n_axes :]
        reparam = np.full(
            (*x.shape[:3], n_scales, *(width,) * (n_axes - 1), *rest), np.nan, x.dtype
        )
        for j1 in range(n_scales):
            for b in range(min(width, n_scales - j1)):
                if family == "S3":
                    reparam[:, :, :, j1, b] = x[:, :, :, j1, j1 + b]
                else:
                    for a in range(b + 1):
                        reparam[:, :, :, j1, a, b] = x[:, :, :, j1, j1 + a, j1 + b]
        missing = np.isnan(reparam)
        filled = np.where(missing, 0.0, reparam)
        transformed = np.zeros_like(filled)
        for k in range(n_scales):
            alpha = math.sqrt((1.0 if k == 0 else 2.0) / n_scales)
            for j in range(n_scales):
                weight = alpha * math.cos(k * math.pi * (j + 0.5) / n_scales)
                transformed[:, :, :, k] += weight * filled[:, :, :, j]
        out[family] = np.where(missing, np.nan, transformed)[:, :, :, :kept]
    return out


# ---------------------------------------------------------------------- flattening


def spec_flatten(
    stats: Statistics,
    flatten_complex: bool = False,
    keep_batch_dim: bool = False,
    mean_along_batch: bool = False,
    mask_st: Any = None,
) -> Array:
    """The statistics as one vector. Implements M-FLAT.

    The families in the order ``mean``, ``var``, S1, S2, S3, S4, then PS when the statistics hold
    it (M-PS-2D), each in the row-major order of its array, without its NaN entries. With
    ``flatten_complex=False`` the vector is complex; with ``flatten_complex=True`` it is real:
    each complex coefficient gives its real and imaginary parts, except those of S1, S2 and PS,
    which give their real part only (OP-37), and
    ``mean`` and ``var`` give one value each. The batch axis is flattened with the rest, unless
    ``keep_batch_dim``, which gives ``(Nb, n)``; ``mean_along_batch`` first averages every
    family over the batch. ``mask_st`` is a boolean array over the vector: True keeps a
    coefficient.
    """
    rows = []
    for family in FAMILIES + (("PS",) if "PS" in stats else ()):
        x = np.asarray(stats[family])
        if mean_along_batch:
            x = x.mean(axis=0, keepdims=True)
        x = x.reshape(x.shape[0], -1)
        present = ~np.isnan(x)
        if keep_batch_dim:
            if not (present == present[0]).all():
                raise ValueError("keep_batch_dim: the NaN differ between batch elements (M-FLAT).")
            kept = x[:, present[0]]
        else:
            kept = x[present][None, :]
        if not flatten_complex:
            rows.append(kept.astype(np.complex128))
        elif family in ("mean", "var", "S1", "S2", "PS"):
            rows.append(kept.real.astype(np.float64))
        else:
            parts = np.stack([kept.real, kept.imag], axis=-1)
            rows.append(parts.reshape(kept.shape[0], -1))
    vector = np.concatenate(rows, axis=1)
    if mask_st is not None:
        keep = np.asarray(mask_st, dtype=bool)
        if keep.shape != (vector.shape[1],):
            raise ValueError(
                f"mask_st has shape {keep.shape}; its length must be the vector's, "
                f"{vector.shape[1]} (M-FLAT)."
            )
        vector = vector[:, keep]
    out: Array = vector if keep_batch_dim else vector.reshape(-1)
    return out
