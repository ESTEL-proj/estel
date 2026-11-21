"""Parity case: the normalised statistics of STL `2D+1D` on tensor products, and their flat
vectors (Part II, §14.2, §14.4; task T3.4, issue #127, decisions of the maintainer, second
version; maths.md M-NORM-TP, M-NORM, M-FLAT, M-IDX-TP, M-LP).

On case C of maths.md §3 (a 16 × 16 periodic 2D block and an 8-sample open 1D block, STL's
default J = (2, 1), L = (4, 2), ``lowpass="ST"``), with two maps of two channels, runs STL's
tensor-product scattering operator (``ST_Operator_TensorProd.apply``) without spectrum
(``compute_PS=False``):

* with ``norm="store_ref"`` on a target field (``norm_batch_mean=False``, STL's default); the
  operator then holds the reference;
* with ``norm="load_ref"`` on a second field, normalised by the target's reference;
* with ``norm="store_ref"`` and ``norm_batch_mean=True`` on the target, with a new operator;
* ``to_flatten`` of the ``store_ref`` statistics in both forms, ``flatten_complex=False`` (a
  complex vector) and ``flatten_complex=True`` (a real vector).

It stores ``field`` and ``ref_field``, ``(Nb, Nc, 16, 16, 8)``, and for each run ``<run>``
(``store``, ``load``, ``store_batch_mean``):

* ``<run>_mean`` and ``<run>_var``, ``(Nb, Nc)``;
* S1 to S4, packed as in ``scattering_tp`` (``pack_grid``): ``<run>_<family>`` the finite
  entries in row-major order, ``<run>_<family>_shape`` and ``<run>_<family>_mask`` (the finite
  positions packed with ``np.packbits``);
* for ``store`` and ``store_batch_mean``, the reference the operator stored: ``<run>_var_ref``
  as it is, and ``<run>_s2_ref`` (``S2_ref_sqrt_chan_diag``, NaN where S2 is) packed;

and ``vector_complex`` and ``vector_real``, the two flat vectors.

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
    "case": "C",
    "blocks": ["2D", "1D"],
    "sizes": [[16, 16], [8]],
    "pbc": [True, False],
    "J": None,
    "lowpass": "ST",
    "Nb": 2,
    "Nc": 2,
    "seed": 400,
    "ref_seed": 401,
}
GRIDS = ("S1", "S2", "S3", "S4")


def pack_grid(key: str, grid: NDArray[Any]) -> dict[str, NDArray[Any]]:
    """A full grid as its finite entries in row-major order, its shape and its packed mask of
    finite positions (decisions on issue #127, second version, "Format")."""
    finite = np.isfinite(grid)
    return {
        key: np.ascontiguousarray(grid[finite]),
        f"{key}_shape": np.asarray(grid.shape, dtype=np.int64),
        f"{key}_mask": np.packbits(finite.ravel()),
    }


def input_field(shape: tuple[int, ...], nb: int, nc: int, seed: int) -> NDArray[np.float64]:
    """Seeded Gaussian fields, ``(Nb, Nc, *shape)``, each channel correlated with the previous
    one so that the cross statistics are not small (as in ``scattering_tp``)."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((nb, nc) + shape)
    channels = [noise[:, 0]]
    for c in range(1, nc):
        channels.append(0.6 * channels[-1] + 0.8 * noise[:, c])
    field: NDArray[np.float64] = np.stack(channels, axis=1) + 0.3
    return field


def tensor_data(field: NDArray[np.float64]) -> Any:
    """The STL tensor-product data of ``field``, with the blocks and periodicity of case C."""
    tensor_prod = importlib.import_module("STL_main.Tensor_Product_Kernel_Torch")
    blocks = {
        "2D": importlib.import_module("STL_main.STL_2D_Kernel_Torch").STL_2D_Kernel_Torch,
        "1D": importlib.import_module("STL_main.STL_1D_Kernel_Torch").STL_1D_Kernel_Torch,
    }
    return tensor_prod.STL_TensorProd_Kernel_Torch(
        array=torch.from_numpy(field.copy()),
        data_structure=[blocks[name] for name in PARAMETERS["blocks"]],
        pbc=list(PARAMETERS["pbc"]),
    )


def _numpy(x: Any) -> NDArray[Any]:
    array: NDArray[Any] = x.detach().cpu().numpy()
    return array


def _statistics(prefix: str, stats: Any) -> dict[str, NDArray[Any]]:
    out = {f"{prefix}_mean": _numpy(stats.mean), f"{prefix}_var": _numpy(stats.var)}
    for family in GRIDS:
        out.update(pack_grid(f"{prefix}_{family}", _numpy(getattr(stats, family))))
    return out


def _reference(prefix: str, operator: Any) -> dict[str, NDArray[Any]]:
    out = {f"{prefix}_var_ref": _numpy(operator.var_ref)}
    out.update(pack_grid(f"{prefix}_s2_ref", _numpy(operator.S2_ref_sqrt_chan_diag)))
    return out


def run() -> dict[str, NDArray[Any]]:
    """Run STL's normalisations and flattening on case C and return the results."""
    p = PARAMETERS
    shape = tuple(n for sizes in p["sizes"] for n in sizes)
    field = input_field(shape, p["Nb"], p["Nc"], p["seed"])
    ref_field = input_field(shape, p["Nb"], p["Nc"], p["ref_seed"])
    out: dict[str, NDArray[Any]] = {"field": field, "ref_field": ref_field}
    kwargs = {"lowpass": p["lowpass"], "compute_PS": False}  # J: STL's default per block

    target, other = tensor_data(field), tensor_data(ref_field)
    operator = target.get_ST_op(norm="store_ref", **kwargs)
    stored = operator.apply(target, norm="store_ref", norm_batch_mean=False)
    out.update(_statistics("store", stored))
    out.update(_reference("store", operator))
    out.update(_statistics("load", operator.apply(other, norm="load_ref")))

    batch_operator = target.get_ST_op(norm="store_ref", **kwargs)
    batch = batch_operator.apply(target, norm="store_ref", norm_batch_mean=True)
    out.update(_statistics("store_batch_mean", batch))
    out.update(_reference("store_batch_mean", batch_operator))

    out["vector_complex"] = _numpy(stored.to_flatten(flatten_complex=False))
    out["vector_real"] = _numpy(stored.to_flatten(flatten_complex=True))
    return out
