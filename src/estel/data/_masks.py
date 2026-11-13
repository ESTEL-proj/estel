"""Static masks and reweighting maps of planar data (maths.md M-DOWN-MASK, M-MASK).

Built once, in NumPy float64, when an operator is built, from the full-resolution mask of the
data; never traced, never differentiated (tag N of Part II, §12). The ``ScatteringPlan`` holds
the result and casts it to the operator's dtype (Part I, §2.3).

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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

BoolMap = NDArray[np.bool_]
FloatMap = NDArray[np.float64]


@dataclass(frozen=True, eq=False)
class MaskMaps:
    """The static maps of one mask, for ``n_levels`` resolution levels and one padding mode.

    Levels are named as in maths.md: ``d`` is a resolution level (the field downsampled ``d``
    times), ``j`` the scale of a first convolution, ``j2 ≤ j3`` the scales of the two
    convolutions of the second layer. Invalid pixels are True.
    """

    pbc: bool
    full_res: BoolMap
    """m⁽⁰⁾, the full-resolution mask."""
    smooth_invalid: Mapping[int, BoolMap]
    """For ``d`` in 1 … n_levels-1: the invalid pixels of the field downsampled to level
    ``d`` (layer 0 of M-MASK)."""
    smooth_reweight: Mapping[int, FloatMap]
    """For ``d`` in 1 … n_levels-1: the reweighting 1 / (1 - f) of the last step down to
    level ``d`` of the field (M-DOWN-MASK), 0 at invalid pixels."""
    wav_invalid: Mapping[tuple[int, int], BoolMap]
    """For ``(d, j)``, ``j < d``: the invalid pixels of first-layer data convolved at scale
    ``j``, then downsampled to level ``d``."""
    wav_reweight: Mapping[tuple[int, int], FloatMap]
    """For ``(d, j)``, ``j < d``: the reweighting of the last step down to level ``d`` of
    first-layer data convolved at scale ``j``, 0 at invalid pixels."""
    layer1: Mapping[int, BoolMap]
    """For ``j`` in 0 … n_levels-1: the invalid pixels of data convolved once at scale ``j``,
    at level ``j`` (layer 1 of M-MASK)."""
    layer2: Mapping[tuple[int, int], BoolMap]
    """For ``(j3, j2)``, ``j2 ≤ j3``: the invalid pixels of data convolved at ``j2``,
    downsampled to ``j3`` and convolved again at ``j3`` (layer 2 of M-MASK)."""


def _pad(a: FloatMap, widths: Sequence[tuple[int, int]], pbc: bool) -> FloatMap:
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


def _correlate(a: FloatMap, kernel: FloatMap, stride: int) -> FloatMap:
    """Valid correlation of ``a`` with ``kernel`` on every axis, at ``stride`` (M-CONV-2D)."""
    windows = np.lib.stride_tricks.sliding_window_view(a, kernel.shape)
    windows = windows[tuple(slice(None, None, stride) for _ in range(a.ndim))]
    axes = tuple(range(a.ndim, 2 * a.ndim))
    return np.tensordot(windows, kernel, axes=(axes, tuple(range(a.ndim))))


def downsample_step(a: FloatMap, s: FloatMap, pbc: bool) -> FloatMap:
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


def _threshold(fraction: FloatMap, threshold: float) -> tuple[BoolMap, FloatMap]:
    """The invalid pixels (f > τ, as STL's ``f <= τ`` test fails) and the reweighting
    1 / (1 - f), 0 at invalid pixels (M-DOWN-MASK)."""
    invalid = ~(fraction <= threshold)
    reweight = np.where(invalid, 0.0, 1.0 / (1.0 - np.where(invalid, 0.0, fraction)))
    return invalid, reweight


def build_mask_maps(
    mask: BoolMap,
    n_levels: int,
    s: FloatMap,
    kernel_size: int,
    threshold: float,
    pbc: bool,
) -> MaskMaps:
    """The static maps of ``mask`` for levels 0 … ``n_levels``-1 (M-DOWN-MASK, M-MASK).

    ``mask`` is the full-resolution mask (True = missing), with one axis per spatial axis;
    ``n_levels`` is J; ``s`` the smoothing kernel of the geometry, summing to 1;
    ``kernel_size`` the wavelet support K; ``threshold`` the τ of the geometry
    (``Data2D._nan_threshold``, ``Data1D._nan_threshold``); ``pbc`` the padding mode.
    """
    m0 = np.asarray(mask, dtype=bool)
    s = np.asarray(s, dtype=np.float64)
    if n_levels < 1:
        raise ValueError(f"n_levels={n_levels}: there is at least one level (J ≥ 1).")
    if not np.isclose(s.sum(), 1.0):
        raise ValueError(
            f"the smoothing kernel sums to {s.sum()}, not 1: pass the smoothing kernel of the "
            "geometry, smoothing_kernel(ndim) (maths.md M-SMOOTH-2D, M-SMOOTH-1D)."
        )
    footprint = (kernel_size,) * m0.ndim

    # Layer 0: the field downsampled from level 0, the invalid pixels of level d - 1 feeding
    # level d (STL downsamples the thresholded mask, not the fraction).
    smooth = [(m0, np.ones(m0.shape))]
    for _ in range(1, n_levels):
        parent = smooth[-1][0].astype(np.float64)
        smooth.append(_threshold(downsample_step(parent, s, pbc), threshold))

    # Layer 1: data convolved at scale j, at level j.
    layer1 = {j: propagate_mask_conv(smooth[j][0], footprint, pbc) for j in range(n_levels)}

    # First-layer data convolved at j, downsampled step by step to every level d > j.
    wav: dict[tuple[int, int], tuple[BoolMap, FloatMap]] = {}
    for j in range(n_levels - 1):
        chain = [layer1[j]]
        for _ in range(j + 1, n_levels):
            step = _threshold(downsample_step(chain[-1].astype(np.float64), s, pbc), threshold)
            chain.append(step[0])
            wav = {**wav, (j + len(chain) - 1, j): step}

    # Layer 2: data convolved at j2, brought to level j3, convolved again at j3.
    layer2 = {
        (j3, j2): propagate_mask_conv(wav[(j3, j2)][0] if j2 < j3 else layer1[j3], footprint, pbc)
        for j3 in range(n_levels)
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
