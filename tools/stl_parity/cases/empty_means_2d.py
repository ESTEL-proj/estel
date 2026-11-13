"""Parity case: the NaN coefficients of STL `master`'s statistics where a mean has no valid pixel
(Part II, §14.2; task T2.3, issue #15; maths.md M-MEAN, OP-3).

With missing values, the masks of the scattering layers grow by the wavelet footprint at each
layer and the border crop grows with depth (M-MASK, M-BORDER), so some means of the coarsest
levels have no valid pixel left. STL then divides 0 by 0, and the coefficient is NaN; the other
coefficients are finite. This case runs STL's scattering operator (``ST_Operator.apply``, without
normalisation or spectrum) on a single map and stores where each family is NaN, as float64 0/1,
with names ``<config>_<family>_nan`` for ``mean``, ``var``, ``S1``, ``S2`` (shape ``(J, L)``),
``S3`` (``(J, J, L, L)``) and ``S4`` (``(J, J, J, L, L, L)``), the batch and channel axes
dropped. With one channel, the only other NaN are the structural ones, outside ``j2 ≤ j3`` for
S3 and ``j1 ≤ j2 ≤ j3`` for S4 (M-CHAN). It also stores the mask STL reads from the NaN of the
field (``mask_<name>``).

The configurations: a small rectangular hole at N = 64, J = 4, open (``pbc=False``), where some
S3 and S4 means have no valid pixel, and periodic, where every mean keeps valid pixels. The mask
is the ``small_hole`` mask of the ``plan_2d`` case. L and the threshold are STL's defaults at
`8eaf769`, written explicitly; the NaN positions do not depend on L, nor on the values of the
field.

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
NDIM = 2
PARAMETERS: dict[str, Any] = {
    "L": 4,
    "threshold": 0.33,
    "masks": {"small_hole": {"seed": 1, "scatter_fraction": 0.0, "hole": [[30, 33], [30, 34]]}},
    "configs": {
        "deep_open": {"N": 64, "J": 4, "pbc": False, "mask": "small_hole"},
        "deep_periodic": {"N": 64, "J": 4, "pbc": True, "mask": "small_hole"},
    },
}
FAMILIES = ("mean", "var", "S1", "S2", "S3", "S4")


def input_mask(n: int, spec: dict[str, Any]) -> NDArray[np.bool_]:
    """The missing pixels (True = missing): a seeded scatter plus a rectangular hole."""
    rng = np.random.default_rng(spec["seed"])
    scatter = rng.random((n, n)) < spec["scatter_fraction"]
    (r0, r1), (c0, c1) = spec["hole"]
    rows = (np.arange(n) >= r0) & (np.arange(n) < r1)
    cols = (np.arange(n) >= c0) & (np.arange(n) < c1)
    missing: NDArray[np.bool_] = scatter | (rows[:, None] & cols[None, :])
    return missing


def run() -> dict[str, NDArray[Any]]:
    """Run STL's scattering operator on each configuration and record its NaN coefficients."""
    data_cls = importlib.import_module("STL_main.STL_2D_Kernel_Torch").STL_2D_Kernel_Torch
    out: dict[str, NDArray[Any]] = {}
    for name, config in PARAMETERS["configs"].items():
        n, mask_name = config["N"], config["mask"]
        missing = input_mask(n, PARAMETERS["masks"][mask_name])
        field = np.random.default_rng(0).standard_normal((n,) * NDIM)
        data = data_cls(array=torch.from_numpy(np.where(missing, np.nan, field)), pbc=config["pbc"])
        # The mask is detected from the NaN of the field, as in STL's default path.
        operator = data.get_ST_op(
            J=config["J"],
            L=PARAMETERS["L"],
            norm="vanilla",
            compute_PS=False,
            downsample_nan_weight_threshold=PARAMETERS["threshold"],
        )
        stats = operator.apply(data)
        mask = operator.wavelet_op.mask_full_res.array.detach().cpu().numpy()
        out[f"mask_{mask_name}"] = mask.reshape(mask.shape[-NDIM:]).astype(np.float64)
        for family in FAMILIES:
            values = getattr(stats, family).detach().cpu().numpy()
            # mean, var: (Nb, Nc); S1 to S4: (Nb, Nc, Nc, ...), with Nb = Nc = 1.
            values = values[0, 0] if family in ("mean", "var") else values[0, 0, 0]
            out[f"{name}_{family}_nan"] = np.asarray(np.isnan(values), dtype=np.float64)
    return out
