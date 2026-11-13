"""Parity case: the static masks and reweighting maps of STL `master` for 2D data (Part II,
§14.2; task T2.1, issue #13).

When the data have missing values, STL's 2D kernel operator (`WaveletOperator2Dkernel_torch`)
builds, once and for both padding modes, the static maps of maths.md M-DOWN-MASK and M-MASK
(`_build_reweighting_maps_and_scattering_layer_masks`):

* ``smooth``: for each level ``d`` in 1 … J-1, the reweighting map 1 / (1 - f) of the field
  downsampled from level 0, NaN where the fraction f of invalid input exceeds the threshold;
* ``wav``: for each level ``d`` and each scale ``j < d``, the same map for first-layer data
  convolved at scale ``j`` and downsampled to level ``d``;
* ``layer1``: for each level ``j``, the invalid pixels of data convolved once at scale ``j``;
* ``layer2``: for each level ``j3`` and each ``j2 ≤ j3``, the invalid pixels of data convolved
  at ``j2``, downsampled to ``j3`` and convolved again.

This case stores them as STL holds them, for ``pbc=False`` (padding mode ``replicate``) and
``pbc=True`` (``circular``), with array names ``<mode>_<family>_<levels>``: ``smooth_<d>``,
``wav_<d>_<j>``, ``layer1_<j>``, ``layer2_<j3>_<j2>``. The boolean masks are stored as float64
0/1, the reweighting maps as float64 with their NaN. It also stores the input mask
(``mask_full_res``) and the smoothing kernel STL uses (``smooth_kernel``, maths.md M-SMOOTH-2D).

The input is a 32 x 32 field whose missing pixels are a seeded scatter of isolated pixels plus a
rectangular hole, so that some downsampled pixels stay valid with a reweighting and others pass
the threshold τ = 0.33. L, K and the threshold are STL's defaults at `8eaf769`, written
explicitly; the maps do not depend on L, nor on the values of the field.

Runs in the parity container, with the interpreter of the `master` environment, through
`make parity-fixtures` (tools/stl_parity/make_fixtures.py). It imports STL, never `estel`.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

STL_ENV = "master"
PARAMETERS: dict[str, Any] = {
    "J": 4,
    "L": 4,
    "K": 5,
    "N": 32,
    "threshold": 0.33,
    "seed": 1,
    "scatter_fraction": 0.05,
    "hole": [[10, 17], [4, 13]],
}


def input_mask() -> NDArray[np.bool_]:
    """The missing pixels (True = missing): a seeded scatter plus a rectangular hole."""
    n = PARAMETERS["N"]
    rng = np.random.default_rng(PARAMETERS["seed"])
    scatter = rng.random((n, n)) < PARAMETERS["scatter_fraction"]
    (r0, r1), (c0, c1) = PARAMETERS["hole"]
    rows = (np.arange(n) >= r0) & (np.arange(n) < r1)
    cols = (np.arange(n) >= c0) & (np.arange(n) < c1)
    missing: NDArray[np.bool_] = scatter | (rows[:, None] & cols[None, :])
    return missing


def run() -> dict[str, NDArray[Any]]:
    """Build STL's operator on a field with missing values and return its static maps."""
    data_2d = importlib.import_module("STL_main.STL_2D_Kernel_Torch").STL_2D_Kernel_Torch
    n, n_levels = PARAMETERS["N"], PARAMETERS["J"]
    mask = input_mask()
    field = np.random.default_rng(PARAMETERS["seed"] + 1).standard_normal((n, n))
    data = data_2d(array=torch.from_numpy(np.where(mask, np.nan, field)), pbc=False)
    # The mask is detected from the NaN of the field, as in STL's default path.
    operator = data.get_wavelet_op(
        J=n_levels,
        L=PARAMETERS["L"],
        kernel_size=PARAMETERS["K"],
        downsample_nan_weight_threshold=PARAMETERS["threshold"],
    )

    def f64(x: torch.Tensor) -> NDArray[np.float64]:
        return np.asarray(x.detach().cpu().numpy(), dtype=np.float64)

    # STL has no public accessor for these maps.
    out: dict[str, NDArray[Any]] = {
        "mask_full_res": f64(operator.mask_full_res.array),
        "smooth_kernel": f64(operator._get_smooth_kernel(torch.device("cpu"), torch.float64)),
    }
    for mode in ("replicate", "circular"):
        smooth = operator._reweighting_maps_smooth[mode]
        wav = operator._reweighting_maps_wav[mode]
        layer1 = operator._layer1_mask[mode]
        layer2 = operator._layer2_mask[mode]
        for d in range(1, n_levels):
            out[f"{mode}_smooth_{d}"] = f64(smooth[d].array)
            for j in range(d):
                out[f"{mode}_wav_{d}_{j}"] = f64(wav[d][j].array)
        for j in range(n_levels):
            out[f"{mode}_layer1_{j}"] = f64(layer1[j].array)
        for j3 in range(n_levels):
            for j2 in range(j3 + 1):
                out[f"{mode}_layer2_{j3}_{j2}"] = f64(layer2[j3][j2].array)
    return out
