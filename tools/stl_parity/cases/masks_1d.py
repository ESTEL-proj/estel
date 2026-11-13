"""Parity case: the static masks and reweighting maps of STL `2D+1D` for 1D data (Part II,
§14.2; task T2.1, issue #13).

The 1D kernel operator of `2D+1D` (`WaveletOperator1Dkernel_torch`) inherits the construction
of the static maps of maths.md M-DOWN-MASK and M-MASK from the dimension-generic 2D class of
that branch, along one axis, with the 1D anti-aliasing kernel (maths.md M-H-1D) and the 1D
threshold τ = 0.5. The stored arrays are those of ``masks_2d.py`` (read its docstring for the
names): ``smooth_<d>``, ``wav_<d>_<j>``, ``layer1_<j>``, ``layer2_<j3>_<j2>`` for each padding
mode (``replicate`` for ``pbc=False``, ``circular`` for ``pbc=True``), plus ``mask_full_res`` and
``smooth_kernel``.

The input is a signal of 128 samples whose missing samples are a seeded scatter plus one gap,
so that some downsampled samples stay valid with a reweighting and others pass the threshold.
L, K and the threshold are STL's defaults at `8f51143`, written explicitly.

Runs in the parity container, with the interpreter of the `2d1d` environment, through
`make parity-fixtures` (tools/stl_parity/make_fixtures.py). It imports STL, never `estel`.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

STL_ENV = "2d1d"
PARAMETERS: dict[str, Any] = {
    "J": 4,
    "L": 1,
    "K": 5,
    "N": 128,
    "threshold": 0.5,
    "seed": 2,
    "scatter_fraction": 0.05,
    "gap": [40, 47],
}


def input_mask() -> NDArray[np.bool_]:
    """The missing samples (True = missing): a seeded scatter plus one gap."""
    n = PARAMETERS["N"]
    rng = np.random.default_rng(PARAMETERS["seed"])
    scatter = rng.random(n) < PARAMETERS["scatter_fraction"]
    g0, g1 = PARAMETERS["gap"]
    missing: NDArray[np.bool_] = scatter | ((np.arange(n) >= g0) & (np.arange(n) < g1))
    return missing


def run() -> dict[str, NDArray[Any]]:
    """Build STL's operator on a signal with missing values and return its static maps."""
    data_1d = importlib.import_module("STL_main.STL_1D_Kernel_Torch").STL_1D_Kernel_Torch
    n, n_levels = PARAMETERS["N"], PARAMETERS["J"]
    mask = input_mask()
    signal = np.random.default_rng(PARAMETERS["seed"] + 1).standard_normal(n)
    data = data_1d(array=torch.from_numpy(np.where(mask, np.nan, signal)), pbc=False)
    # The mask is detected from the NaN of the signal, as in STL's default path.
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
