"""Static masks and reweighting maps (maths.md M-DOWN-MASK, M-MASK; M-MASK-HPX for HEALPix).

Built once, in NumPy float64, when an operator is built, from the full-resolution mask of the
data; never traced, never differentiated (tag N of Part II, §12). The static part of the
wavelet operator (``estel.wavelets._operator``), which the ``STPlan`` holds, keeps the result
and casts it to the operator's dtype; nothing else reads this module (Part I, §2.3, §2.4).

This reproduces STL's ``_build_reweighting_maps_and_scattering_layer_masks`` (`master`
``8eaf769`` for 2D, `2D+1D` ``8f51143`` for 1D), for one padding mode: STL builds the maps of
both modes, ESTEL only those of the data's ``pbc``. One difference of representation: where STL
stores NaN in a reweighting map to mark an invalid pixel, ESTEL stores a boolean mask and a
reweighting of 0 there (maths.md OP-27). ``tests/parity/test_masks.py`` checks the equality.

Each downsampling step correlates with the smoothing kernel ``s``, the anti-aliasing filter of the
downsampling (maths.md M-SMOOTH-2D, M-SMOOTH-1D), passed in; the wavelet footprint E is a box
of the kernel size K on every spatial axis.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

BoolMap = NDArray[np.bool_]
RealMap = NDArray[np.float64]


@dataclass(frozen=True, eq=False)
class MaskMaps:
    """The static maps of one mask, for the levels of the scales given and one padding mode.

    Named as in maths.md M-NOT: ``d`` is a resolution level (the field downsampled ``d``
    times), ``j`` the scale of a first convolution, ``j2 ≤ j3`` the scales of the two
    convolutions of the second layer. The convolution at scale ``j`` happens at the level of
    that scale, ``scale_levels[j]`` (d = j in 1.0.0). Invalid pixels are True.
    """

    pbc: bool
    full_res: BoolMap
    """m⁽⁰⁾, the full-resolution mask."""
    smooth_invalid: Mapping[int, BoolMap]
    """For ``d`` in 1 … max(scale_levels): the invalid pixels of the field downsampled to
    level ``d`` (layer 0 of M-MASK)."""
    smooth_reweight: Mapping[int, RealMap]
    """For ``d`` in 1 … max(scale_levels): the reweighting 1 / (1 - f) of the last step down
    to level ``d`` of the field (M-DOWN-MASK), 0 at invalid pixels."""
    wav_invalid: Mapping[tuple[int, int], BoolMap]
    """For ``(d, j)``, ``d`` above the level of scale ``j``: the invalid pixels of first-layer
    data convolved at scale ``j``, then downsampled to level ``d``."""
    wav_reweight: Mapping[tuple[int, int], RealMap]
    """For ``(d, j)``, ``d`` above the level of scale ``j``: the reweighting of the last step
    down to level ``d`` of first-layer data convolved at scale ``j``, 0 at invalid pixels."""
    layer1: Mapping[int, BoolMap]
    """For each scale ``j``: the invalid pixels of data convolved once at scale ``j``, at the
    level of that scale (layer 1 of M-MASK)."""
    layer2: Mapping[tuple[int, int], BoolMap]
    """For ``(j3, j2)``, ``j2 ≤ j3``: the invalid pixels of data convolved at scale ``j2``,
    brought to the level of scale ``j3`` and convolved again at ``j3`` (layer 2 of M-MASK)."""


def _pad(a: RealMap, widths: Sequence[tuple[int, int]], pbc: bool) -> RealMap:
    """Pad every axis of ``a``: circular if ``pbc``, replicate otherwise (M-CONV-2D)."""
    if pbc:
        for (before, after), n in zip(widths, a.shape, strict=True):
            if max(before, after) > n:
                raise ValueError(
                    f"circular padding by {(before, after)} wraps more than once around an axis "
                    f"of length {n}: the data are too small for this number of levels. Use "
                    "fewer levels (a smaller J) or larger data."
                )
    return np.pad(a, list(widths), mode="wrap" if pbc else "edge")


def _correlate(a: RealMap, kernel: RealMap, stride: int) -> RealMap:
    """Valid correlation of ``a`` with ``kernel`` on every axis, at ``stride`` (M-CONV-2D)."""
    windows = np.lib.stride_tricks.sliding_window_view(a, kernel.shape)
    windows = windows[tuple(slice(None, None, stride) for _ in range(a.ndim))]
    axes = tuple(range(a.ndim, 2 * a.ndim))
    return np.tensordot(windows, kernel, axes=(axes, tuple(range(a.ndim))))


def downsample_step(a: RealMap, s: RealMap, pbc: bool) -> RealMap:
    """One downsampling step (M-DOWN): pad by ``(K-1)//2`` before and ``K//2`` after each axis,
    correlate with ``s``, keep one pixel in two. ``s`` has one axis per axis of ``a``."""
    if s.ndim != a.ndim:
        raise ValueError(
            f"the smoothing kernel has {s.ndim} axes but the map has {a.ndim}: pass the "
            "smoothing kernel of the geometry, smoothing_kernel(ndim) (maths.md M-SMOOTH-2D, "
            "M-SMOOTH-1D)."
        )
    odd = [n for n in a.shape if n % 2]
    if odd:
        raise ValueError(
            f"cannot downsample a map of shape {a.shape}: every axis must be even at each "
            "level, so the sizes must be divisible by 2^(J-1) (maths.md M-DATA-2D, OP-8)."
        )
    widths = [((k - 1) // 2, k // 2) for k in s.shape]
    return _correlate(_pad(a, widths, pbc), s, stride=2)


def propagate_mask_conv(mask: BoolMap, footprint: Sequence[int], pbc: bool) -> BoolMap:
    """The invalid outputs of a convolution (M-MASK): an output is invalid when the kernel
    footprint, a box of shape ``footprint`` centred on it, touches an invalid input.

    The mask is padded by ``⌊K/2⌋`` on each side like the data (circular if ``pbc``, replicate
    otherwise), then correlated with the box: (E ⋆ m) > 0.
    """
    box = np.ones(tuple(footprint), dtype=np.float64)
    widths = [(k // 2, k // 2) for k in footprint]
    return _correlate(_pad(mask.astype(np.float64), widths, pbc), box, stride=1) > 0.0


def _threshold(fraction: RealMap, threshold: float) -> tuple[BoolMap, RealMap]:
    """The invalid pixels (f > τ, as STL's ``f <= τ`` test fails) and the reweighting
    1 / (1 - f), 0 at invalid pixels (M-DOWN-MASK)."""
    invalid = ~(fraction <= threshold)
    reweight = np.where(invalid, 0.0, 1.0 / (1.0 - np.where(invalid, 0.0, fraction)))
    return invalid, reweight


def build_mask_maps(
    mask: BoolMap,
    scale_levels: Sequence[int],
    s: RealMap,
    kernel_size: int,
    threshold: float,
    pbc: bool,
) -> MaskMaps:
    """The static maps of ``mask`` for the scales of ``scale_levels`` (M-DOWN-MASK, M-MASK).

    ``mask`` is the full-resolution mask (True = missing), with one axis per spatial axis;
    ``scale_levels`` gives, for each scale ``j``, the level at which it is computed, the
    ``scale_levels`` of the wavelet plan (``(0, 1, …, J-1)`` in 1.0.0), and the maps cover the
    levels 0 … ``max(scale_levels)``; ``s`` the smoothing kernel of the geometry, summing to 1;
    ``kernel_size`` the wavelet support K; ``threshold`` the τ of the geometry
    (``Data2D._nan_threshold``, ``Data1D._nan_threshold``); ``pbc`` the padding mode.
    """
    m0 = np.asarray(mask, dtype=bool)
    s = np.asarray(s, dtype=np.float64)
    scale_levels = tuple(int(d) for d in scale_levels)
    if not scale_levels:
        raise ValueError(
            "scale_levels=(): there is at least one scale, so at least one level (J ≥ 1). Pass "
            "the scale_levels of the wavelet plan, the level of each scale: (0, 1, …, J-1) for "
            "J scales in 1.0.0."
        )
    if not np.isclose(s.sum(), 1.0):
        raise ValueError(
            f"the smoothing kernel sums to {s.sum()}, not 1: pass the smoothing kernel of the "
            "geometry, smoothing_kernel(ndim) (maths.md M-SMOOTH-2D, M-SMOOTH-1D)."
        )
    n_levels = max(scale_levels) + 1
    footprint = (kernel_size,) * m0.ndim

    # Layer 0: the field downsampled from level 0, the invalid pixels of level d - 1 feeding
    # level d (STL downsamples the thresholded mask, not the fraction).
    smooth = [(m0, np.ones(m0.shape))]
    for _ in range(1, n_levels):
        parent = smooth[-1][0].astype(np.float64)
        smooth.append(_threshold(downsample_step(parent, s, pbc), threshold))

    # Layer 1: data convolved at scale j, at the level of scale j.
    layer1 = {
        j: propagate_mask_conv(smooth[d][0], footprint, pbc) for j, d in enumerate(scale_levels)
    }

    # First-layer data convolved at j, downsampled step by step to every coarser level.
    wav: dict[tuple[int, int], tuple[BoolMap, RealMap]] = {}
    for j, level in enumerate(scale_levels):
        chain = [layer1[j]]
        for d in range(level + 1, n_levels):
            step = _threshold(downsample_step(chain[-1].astype(np.float64), s, pbc), threshold)
            chain.append(step[0])
            wav = {**wav, (d, j): step}

    # Layer 2: data convolved at j2, brought to the level of j3, convolved again at j3.
    layer2 = {
        (j3, j2): propagate_mask_conv(
            wav[(scale_levels[j3], j2)][0] if scale_levels[j2] < scale_levels[j3] else layer1[j2],
            footprint,
            pbc,
        )
        for j3 in range(len(scale_levels))
        for j2 in range(j3 + 1)
    }

    return MaskMaps(
        pbc=pbc,
        full_res=m0,
        smooth_invalid={d: smooth[d][0] for d in range(1, n_levels)},
        smooth_reweight={d: smooth[d][1] for d in range(1, n_levels)},
        wav_invalid={key: value[0] for key, value in wav.items()},
        wav_reweight={key: value[1] for key, value in wav.items()},
        layer1=layer1,
        layer2=layer2,
    )


# ---------------------------------------------------------------------- HEALPix (M-MASK-HPX)


HEALPIX_SUPPORT_THRESHOLD = 1e-12
"""Above it, the support of a pixel touches an invalid pixel (maths.md M-MASK-HPX); the same
threshold below which an interpolation weight counts as absent (M-CONV-HPX, OP-110)."""

HEALPIX_REWEIGHT_FLOOR = 1e-8
"""The floor of 1 - f in the reweighting maps of HEALPix (M-MASK-HPX)."""


def _as_real(mask: BoolMap) -> RealMap:
    """A mask as a 0/1 float64 map."""
    return mask.astype(np.float64)


def _threshold_floored(fraction: RealMap, threshold: float) -> tuple[BoolMap, RealMap]:
    """The invalid pixels (f > τ) and the reweighting 1 / (1 - f), with 1 - f not below
    :data:`HEALPIX_REWEIGHT_FLOOR`, 0 at invalid pixels (M-MASK-HPX)."""
    invalid = ~(fraction <= threshold)
    kept = np.maximum(1.0 - np.where(invalid, 0.0, fraction), HEALPIX_REWEIGHT_FLOOR)
    return invalid, np.where(invalid, 0.0, 1.0 / kept)


def build_healpix_mask_maps(
    mask: BoolMap,
    scale_levels: Sequence[int],
    support: Callable[[int, BoolMap], BoolMap],
    decimate: Callable[[int, RealMap], RealMap],
    threshold: float,
) -> MaskMaps:
    """The static maps of ``mask`` on HEALPix, for the scales of ``scale_levels`` (maths.md
    M-MASK-HPX, with M-MASK and M-DOWN-MASK), each map on the pixels of its level.

    ``support(d, invalid)``: the invalid outputs of a convolution at level ``d`` of data whose
    invalid pixels are ``invalid`` (the support test of M-MASK-HPX, above
    :data:`HEALPIX_SUPPORT_THRESHOLD` for at least one gauge); ``decimate(d, a)``: one step of
    M-DOWN-HPX of the map ``a`` from level ``d`` to ``d + 1``. The callers build these from the
    tables of each level. The maps are computed level by level, every convolution of a level
    together, so that the caller needs the stencils of one level at a time.

    The fields of :class:`MaskMaps` are those of the planar maps (``build_mask_maps``), with the
    reweighting 1 / (1 - f) floored as M-MASK-HPX says; ``pbc`` is True: no padding is involved.
    """
    m0 = np.asarray(mask, dtype=bool)
    scale_levels = tuple(int(d) for d in scale_levels)
    n_levels = max(scale_levels) + 1

    smooth: list[tuple[BoolMap, RealMap]] = [(m0, np.ones(m0.shape))]
    for d in range(1, n_levels):
        smooth.append(_threshold_floored(decimate(d - 1, _as_real(smooth[-1][0])), threshold))

    layer1: dict[int, BoolMap] = {}
    wav: dict[tuple[int, int], tuple[BoolMap, RealMap]] = {}
    layer2: dict[tuple[int, int], BoolMap] = {}
    for d in range(n_levels):
        # First-layer data of the scales of earlier levels, brought one step down to level d.
        wav = {
            **wav,
            **{
                (d, j): _threshold_floored(
                    decimate(d - 1, _as_real(layer1[j] if level == d - 1 else wav[(d - 1, j)][0])),
                    threshold,
                )
                for j, level in enumerate(scale_levels)
                if level < d
            },
        }
        # Every convolution at level d: layer 1 of its scales, then layer 2 of the scales j3
        # there, of data convolved at j2 ≤ j3 and brought to level d.
        layer1 = {
            **layer1,
            **{j: support(d, smooth[d][0]) for j, level in enumerate(scale_levels) if level == d},
        }
        layer2 = {
            **layer2,
            **{
                (j3, j2): support(d, layer1[j2] if scale_levels[j2] == d else wav[(d, j2)][0])
                for j3, level3 in enumerate(scale_levels)
                if level3 == d
                for j2 in range(j3 + 1)
            },
        }

    return MaskMaps(
        pbc=True,
        full_res=m0,
        smooth_invalid={d: smooth[d][0] for d in range(1, n_levels)},
        smooth_reweight={d: smooth[d][1] for d in range(1, n_levels)},
        wav_invalid={key: value[0] for key, value in wav.items()},
        wav_reweight={key: value[1] for key, value in wav.items()},
        layer1=layer1,
        layer2=layer2,
    )
