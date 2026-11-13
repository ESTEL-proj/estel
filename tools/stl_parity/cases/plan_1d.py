"""Parity case: the border crops, masks and denominators of STL `2D+1D`'s means for 1D data
(Part II, §14.2; task T2.3, issue #15).

Every statistic of STL's scattering loop is a mean of data with a convolution history
(``conv_history``) at a resolution level (``dg``), computed by the wavelet operator's ``mean``,
``square_mean`` or ``cov``: crop a border whose width depends on the history (maths.md M-BORDER,
``_get_crop_border_size_fully_flexible``), reduced when too large for the array (OP-13,
``_crop``); pick the mask of the operands (M-MASK, ``_find_mask``, ``cov``); divide by the
number of valid pixels left (M-MEAN, ``maskmean``). The means of the loop (maths.md §1.6 to 1.7;
``ST_Operator.apply``), each at the level of its last convolution:

* ``moments``: ``mean`` and ``var`` of the field, history ``[]``, at level 0;
* ``S1S2_<j>``: S1 and S2, auto and cross, history ``[j]``, at level ``j``;
* ``S3_<j2>_<j3>``: ``cov`` of histories ``[j2, j3]`` and ``[j3]``, at level ``j3``;
* ``S4_<j1>_<j2>_<j3>``: ``cov`` of histories ``[j1, j3]`` and ``[j2, j3]``, at level ``j3``.

For each of them, this case calls STL's own ``mean``, ``square_mean`` and ``cov`` on data of the
right history and level, and records what STL's ``maskmean`` receives: the case wraps that
function for the duration of the call, without changing what it computes. For each
configuration ``<config>`` it stores, the means in the order above (that of STL's loops, with
S3 at ``(j2, j3)`` followed by the S4 at ``(j1, j2, j3)``, ``j1 ≤ j2``):

* ``<config>_border``: for each mean, the width STL's border method returns (the larger one for
  ``cov``), before the reduction of OP-13;
* ``<config>_count``: for each mean, the denominator of ``maskmean``, the number of pixels left
  after the crop and the mask (0 where STL divides by zero, OP-3);
* ``<config>_weights_<d>`` (only for the configurations marked ``"weights": True``, to keep the
  file small): for each mean at level ``d``, in the same order, 1 on the pixels that enter the
  mean and 0 elsewhere, at the full shape of the level.

S1 and S2 (through ``mean``, ``square_mean`` and ``cov``), like ``mean`` and ``var``, use the
same crop and mask; the case checks it and stores them once. It also stores the mask STL reads
from the NaN of the field for each mask of ``PARAMETERS["masks"]`` (``mask_<name>``), the input
of the configurations that name it.

STL's 1D class inherits these methods from its 2D class, written for any number of axes
(maths.md §2.3); they call the ``maskmean`` of ``STL_main.STL_2D_Kernel_Torch``, which is the
one wrapped.

The configurations: without mask, open (``pbc=False``, where the borders of the coarsest levels
are reduced, OP-13) and periodic, at N = 32, J = 4; with the mask of the ``masks_1d`` case (a
gap and a seeded scatter) at N = 128, J = 3, open and periodic, where every mean keeps valid
samples; and the same mask at J = 4, where some means of the open configuration have none. L,
K and the threshold are STL's defaults at `8f51143`, written explicitly; nothing here depends
on L, nor on the values of the signal.

Runs in the parity container, with the interpreter of the `2d1d` environment, through
`make parity-fixtures` (tools/stl_parity/make_fixtures.py). It imports STL, never `estel`.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

STL_ENV = "2d1d"
NDIM = 1
PARAMETERS: dict[str, Any] = {
    "L": 1,
    "K": 5,
    "threshold": 0.5,
    "masks": {"gap": {"seed": 2, "scatter_fraction": 0.05, "gap": [40, 47]}},
    "configs": {
        "open": {"N": 32, "J": 4, "pbc": False, "mask": None, "weights": False},
        "periodic": {"N": 32, "J": 4, "pbc": True, "mask": None, "weights": False},
        "masked_open": {"N": 128, "J": 3, "pbc": False, "mask": "gap", "weights": True},
        "masked_periodic": {"N": 128, "J": 3, "pbc": True, "mask": "gap", "weights": True},
        "deep_open": {"N": 128, "J": 4, "pbc": False, "mask": "gap", "weights": False},
        "deep_periodic": {"N": 128, "J": 4, "pbc": True, "mask": "gap", "weights": False},
    },
}


def input_mask(n: int, spec: dict[str, Any]) -> NDArray[np.bool_]:
    """The missing samples (True = missing): a seeded scatter plus a gap, as in masks_1d."""
    rng = np.random.default_rng(spec["seed"])
    scatter = rng.random(n) < spec["scatter_fraction"]
    g0, g1 = spec["gap"]
    missing: NDArray[np.bool_] = scatter | ((np.arange(n) >= g0) & (np.arange(n) < g1))
    return missing


def means(n_scales: int) -> dict[str, tuple[tuple[list[int], ...], int, tuple[str, ...]]]:
    """The means of STL's loop: name -> (operand histories, level, STL methods called)."""
    found: dict[str, tuple[tuple[list[int], ...], int, tuple[str, ...]]] = {
        "moments": (([],), 0, ("mean", "cov"))
    }
    for j3 in range(n_scales):
        found[f"S1S2_{j3}"] = (([j3],), j3, ("mean", "square_mean", "cov"))
        for j2 in range(j3 + 1):
            found[f"S3_{j2}_{j3}"] = (([j2, j3], [j3]), j3, ("cov",))
            for j1 in range(j2 + 1):
                found[f"S4_{j1}_{j2}_{j3}"] = (([j1, j3], [j2, j3]), j3, ("cov",))
    return found


def run() -> dict[str, NDArray[Any]]:
    """Call STL's means on data of every history and level, and record their crops."""
    # The 1D class inherits mean, square_mean and cov from the 2D module, which calls its own
    # maskmean.
    module: Any = importlib.import_module("STL_main.STL_2D_Kernel_Torch")
    data_cls = importlib.import_module("STL_main.STL_1D_Kernel_Torch").STL_1D_Kernel_Torch
    calls: list[tuple[tuple[int, ...], NDArray[np.bool_] | None]] = []
    original: Callable[..., Any] = module.maskmean

    def recording(x: torch.Tensor, *args: Any, mask: Any = None, **kwargs: Any) -> Any:
        spatial = tuple(int(n) for n in x.shape[-NDIM:])
        calls.append((spatial, None if mask is None else _spatial(mask)))
        return original(x, *args, mask=mask, **kwargs)

    out: dict[str, NDArray[Any]] = {}
    module.maskmean = recording
    try:
        for name, config in PARAMETERS["configs"].items():
            out.update(_config(data_cls, name, config, calls))
    finally:
        module.maskmean = original
    return out


def _spatial(mask: torch.Tensor) -> NDArray[np.bool_]:
    """The mask over the spatial axes alone; STL's masks may carry leading axes of size 1."""
    values = mask.detach().cpu().numpy().astype(bool)
    if int(np.prod(values.shape[:-NDIM])) != 1:
        raise RuntimeError(f"maskmean received a mask of shape {values.shape}")
    spatial: NDArray[np.bool_] = values.reshape(values.shape[-NDIM:])
    return spatial


def _config(
    data_cls: Any,
    name: str,
    config: dict[str, Any],
    calls: list[tuple[tuple[int, ...], NDArray[np.bool_] | None]],
) -> dict[str, NDArray[Any]]:
    n, n_scales, pbc = config["N"], config["J"], config["pbc"]
    full = (n,) * NDIM
    field = np.random.default_rng(0).standard_normal(full)
    out: dict[str, NDArray[Any]] = {}
    if config["mask"] is not None:
        field = np.where(input_mask(n, PARAMETERS["masks"][config["mask"]]), np.nan, field)
    data = data_cls(array=torch.from_numpy(field), pbc=pbc)
    # The mask is detected from the NaN of the field, as in STL's default path.
    operator = data.get_wavelet_op(
        J=n_scales,
        L=PARAMETERS["L"],
        kernel_size=PARAMETERS["K"],
        downsample_nan_weight_threshold=PARAMETERS["threshold"],
    )
    if config["mask"] is not None:
        out[f"mask_{config['mask']}"] = _spatial(operator.mask_full_res.array).astype(np.float64)

    def operand(history: list[int], level: int) -> Any:
        shape = tuple(size // 2**level for size in full)
        return data_cls(
            array=torch.ones((1, 1, *shape), dtype=torch.float64),
            pbc=pbc,
            dg=level,
            N0=full,
            conv_history=list(history),
        )

    borders, counts = [], []
    weights: dict[int, list[NDArray[np.float64]]] = {}
    for mean, (histories, level, methods) in means(n_scales).items():
        operands = [operand(h, level) for h in histories]
        border = max(
            operator._get_crop_border_size_method(data=d, wavelet_op=operator) for d in operands
        )
        seen = []
        for method in methods:
            calls.clear()
            if method == "cov":
                operator.cov(operands[0], operands[-1])
            else:
                getattr(operator, method)(operands[0])
            if len(calls) != 1:
                raise RuntimeError(f"{name} {mean} {method}: {len(calls)} calls to maskmean")
            seen.append(calls[0])
        level_shape = tuple(size // 2**level for size in full)
        cropped, mask = seen[0]
        for other_shape, other_mask in seen[1:]:
            same_mask = (mask is None and other_mask is None) or (
                mask is not None and other_mask is not None and np.array_equal(mask, other_mask)
            )
            if other_shape != cropped or not same_mask:
                raise RuntimeError(f"{name} {mean}: {methods} do not use the same crop and mask")
        reduced = (level_shape[0] - cropped[0]) // 2
        valid = np.ones(cropped) if mask is None else (~mask).astype(np.float64)
        borders.append(float(border))
        counts.append(float(valid.sum()))
        weights.setdefault(level, []).append(np.pad(valid, reduced))
    out[f"{name}_border"] = np.asarray(borders, dtype=np.float64)
    out[f"{name}_count"] = np.asarray(counts, dtype=np.float64)
    if config["weights"]:
        for level, maps in weights.items():
            out[f"{name}_weights_{level}"] = np.stack(maps)
    return out
