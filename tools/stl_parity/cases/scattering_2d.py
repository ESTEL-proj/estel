"""Parity case: the vanilla scattering statistics of STL `master` on 2D data (Part II, §14.2;
task T2.4, issue #16; maths.md M-MOM, M-W1, M-S1 to M-S4, M-CHAN, OP-26, OP-27).

Runs STL's scattering operator (``ST_Operator.apply``) without normalisation and without
spectrum (``norm="vanilla"``, ``compute_PS=False``) on seeded fields, and stores, for each
configuration ``<config>``:

* ``<config>_field``: the input, ``(Nb, Nc, N, N)``, NaN at the missing pixels;
* ``<config>_mask``: the missing pixels, ``(N, N)``, float64 0/1 (absent without a mask);
* ``<config>_<family>`` for ``mean``, ``var`` (``(Nb, Nc)``, float64) and ``S1``, ``S2``,
  ``S3``, ``S4`` (STL's full grids, complex128, NaN where STL computes no coefficient);
* with ``standardize``, ``<config>_mean_pre_std`` and ``<config>_std_pre_std``, ``(Nb, Nc)``.

The configurations cover the cells of §14.4 that T2.4 owns: periodic and open borders (the open
one reduces a border too large for the coarsest level, OP-13), a mask of missing values shared by
every batch element and channel (OP-26), periodic and open, where STL fills the invalid pixels of
its downsamplings with NaN and ESTEL uses weights (OP-27), two channels with every pair, a
custom ``compute_cross_matrix`` where channel 1 has no auto statistics (OP-30), standardised
data (M-MOM), and the deep open configuration of ``empty_means_2d``, where some S3 and S4 means
have no valid pixel and STL's coefficients are NaN (M-MEAN, OP-3). The mask is passed to STL
explicitly (``mask_full_res``), since STL reads it from the NaN only for a single map. L and the
threshold are STL's defaults at `8eaf769`, written explicitly.

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
    "masks": {
        "scatter_hole": {"seed": 1, "scatter_fraction": 0.03, "hole": [[10, 17], [4, 13]]},
        "small_hole": {"seed": 1, "scatter_fraction": 0.0, "hole": [[30, 33], [30, 34]]},
    },
    "configs": {
        "periodic": {"N": 32, "J": 3, "Nb": 2, "Nc": 2, "pbc": True, "seed": 10},
        "open": {"N": 32, "J": 3, "Nb": 2, "Nc": 2, "pbc": False, "seed": 11},
        "masked_open": {
            "N": 32, "J": 3, "Nb": 2, "Nc": 2, "pbc": False, "seed": 12, "mask": "scatter_hole"
        },
        "masked_periodic": {
            "N": 32, "J": 3, "Nb": 1, "Nc": 2, "pbc": True, "seed": 13, "mask": "scatter_hole"
        },
        "standardized": {
            "N": 32, "J": 3, "Nb": 2, "Nc": 1, "pbc": False, "seed": 14, "mask": "scatter_hole",
            "standardize": True,
        },
        "cross_matrix": {
            "N": 32, "J": 3, "Nb": 1, "Nc": 3, "pbc": True, "seed": 15,
            "cross_matrix": [[True, False, True], [False, False, True], [False, False, True]],
        },
        "deep_open": {
            "N": 64, "J": 4, "Nb": 1, "Nc": 1, "pbc": False, "seed": 16, "mask": "small_hole"
        },
    },
}  # fmt: skip
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


def input_field(config: dict[str, Any]) -> NDArray[np.float64]:
    """Seeded Gaussian fields, ``(Nb, Nc, N, N)``, each channel correlated with the previous one
    so that the cross statistics are not small."""
    rng = np.random.default_rng(config["seed"])
    noise = rng.standard_normal((config["Nb"], config["Nc"]) + (config["N"],) * NDIM)
    channels = [noise[:, 0]]
    for c in range(1, config["Nc"]):
        channels.append(0.6 * channels[-1] + 0.8 * noise[:, c])
    field: NDArray[np.float64] = np.stack(channels, axis=1) + 0.3
    return field


def run() -> dict[str, NDArray[Any]]:
    """Run STL's scattering operator on each configuration and return its statistics."""
    data_cls = importlib.import_module("STL_main.STL_2D_Kernel_Torch").STL_2D_Kernel_Torch
    out: dict[str, NDArray[Any]] = {}
    for name, config in PARAMETERS["configs"].items():
        field = input_field(config)
        kwargs: dict[str, Any] = {}
        if "mask" in config:
            missing = input_mask(config["N"], PARAMETERS["masks"][config["mask"]])
            field = np.where(missing, np.nan, field)
            kwargs["mask_full_res"] = data_cls(array=torch.from_numpy(missing))
            out[f"{name}_mask"] = missing.astype(np.float64)
        out[f"{name}_field"] = field
        data = data_cls(array=torch.from_numpy(field.copy()), pbc=config["pbc"])
        operator = data.get_ST_op(
            J=config["J"],
            L=config.get("L", PARAMETERS["L"]),
            norm="vanilla",
            compute_PS=False,
            downsample_nan_weight_threshold=PARAMETERS["threshold"],
            **kwargs,
        )
        matrix = config.get("cross_matrix")
        stats = operator.apply(
            data,
            standardize=config.get("standardize", False),
            compute_cross_matrix=None if matrix is None else torch.tensor(matrix),
        )
        for family in FAMILIES:
            out[f"{name}_{family}"] = getattr(stats, family).detach().cpu().numpy()
        if config.get("standardize", False):
            for key in ("mean_pre_std", "std_pre_std"):
                out[f"{name}_{key}"] = getattr(stats, key).detach().cpu().numpy()
    return out
