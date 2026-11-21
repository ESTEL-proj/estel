"""Parity case: the vanilla scattering statistics of STL `2D+1D` on tensor products (Part II,
§14.2, §14.4; task T3.4, issue #127, decisions of the maintainer, second version; maths.md
M-DATA-TP, M-W1-TP, M-LP, M-IDX-TP, M-S-TP, M-BORDER-TP, OP-30, OP-42, OP-43, OP-81, OP-82).

Runs STL's tensor-product scattering operator (``ST_Operator_TensorProd.apply``) without
normalisation and without spectrum (``norm="vanilla"``, ``compute_PS=False``) on seeded fields.
For each configuration ``<config>`` it stores ``<config>_field``, the input,
``(Nb, Nc, *N⁽⁰⁾, …, *N⁽ᴮ⁻¹⁾)``, and for each low-pass setting ``<mode>`` (``st`` for
``lowpass="ST"``, ``none`` for ``lowpass=None``):

* ``<config>_<mode>_mean`` and ``<config>_<mode>_var``, ``(Nb, Nc)``, float64;
* S1, S2 and, with ``"ST"`` only, S3 and S4, packed (``pack_grid``): ``<key>`` the finite
  entries of STL's full grid in row-major order (complex128), ``<key>_shape`` the shape of the
  grid (int64) and ``<key>_mask`` the finite positions, flattened in row-major order and packed
  with ``np.packbits`` (uint8). ``tests/parity/_compare.py`` (``full_grid``) rebuilds the grid,
  NaN elsewhere;
* with ``standardize``, ``<config>_st_mean_pre_std`` and ``<config>_st_std_pre_std``,
  ``(Nb, Nc)``.

With ``lowpass=None``, S3 and S4 are not stored: the script asserts that they are bit for bit
those of ``"ST"`` on the same field (maths.md M-LP, "Checked").

The configurations are those of the decisions on issue #127 (second version): case A of
maths.md §3 with two maps of two channels; cases B to G with one map of one channel; a
``[1D, 1D]`` field and tensor products of one 2D block and of one 1D block (OP-82), one map of
one channel; each with ``"ST"`` and ``None``; ``standardize=True`` on case A, one map of one
channel; a ``compute_cross_matrix`` where channel 1 has no auto statistics, on case A, one map
of three channels (OP-30). ``J`` set to ``None`` leaves J out of the call, so that STL takes its
default per block (cases C and D, M-DATA-TP); L is always STL's default per block (4 for a 2D
block, 2 for a 1D block). Tensor products take no mask (OP-41).

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
# The seven cases of maths.md §3, and the other geometries of the decisions on issue #127.
CASES: dict[str, dict[str, Any]] = {
    "A": {"blocks": ["2D", "1D"], "sizes": [[16, 16], [16]], "pbc": [True, True], "J": [2, 2]},
    "B": {"blocks": ["2D", "1D"], "sizes": [[16, 16], [16]], "pbc": [False, False], "J": [2, 2]},
    "C": {"blocks": ["2D", "1D"], "sizes": [[16, 16], [8]], "pbc": [True, False], "J": None},
    "D": {
        "blocks": ["2D", "1D", "1D"], "sizes": [[16, 16], [8], [16]],
        "pbc": [False, True, False], "J": None,
    },
    "E": {"blocks": ["1D", "2D"], "sizes": [[16], [16, 16]], "pbc": [False, True], "J": [2, 2]},
    "F": {"blocks": ["2D", "1D"], "sizes": [[16, 16], [12]], "pbc": [True, True], "J": [2, 3]},
    "G": {"blocks": ["2D", "1D"], "sizes": [[32, 16], [32]], "pbc": [False, False], "J": [3, 3]},
    "one_d_one_d": {
        "blocks": ["1D", "1D"], "sizes": [[16], [16]], "pbc": [True, False], "J": [2, 2],
    },
    "single_2d": {"blocks": ["2D"], "sizes": [[16, 16]], "pbc": [True], "J": [2]},
    "single_1d": {"blocks": ["1D"], "sizes": [[16]], "pbc": [False], "J": [2]},
}  # fmt: skip
BOTH = ["ST", None]
PARAMETERS: dict[str, Any] = {
    "cases": CASES,
    "configs": {
        "A": {"case": "A", "Nb": 2, "Nc": 2, "seed": 300, "lowpass": BOTH},
        "B": {"case": "B", "Nb": 1, "Nc": 1, "seed": 301, "lowpass": BOTH},
        "C": {"case": "C", "Nb": 1, "Nc": 1, "seed": 302, "lowpass": BOTH},
        "D": {"case": "D", "Nb": 1, "Nc": 1, "seed": 303, "lowpass": BOTH},
        "E": {"case": "E", "Nb": 1, "Nc": 1, "seed": 304, "lowpass": BOTH},
        "F": {"case": "F", "Nb": 1, "Nc": 1, "seed": 305, "lowpass": BOTH},
        "G": {"case": "G", "Nb": 1, "Nc": 1, "seed": 306, "lowpass": BOTH},
        "one_d_one_d": {"case": "one_d_one_d", "Nb": 1, "Nc": 1, "seed": 307, "lowpass": BOTH},
        "single_2d": {"case": "single_2d", "Nb": 1, "Nc": 1, "seed": 308, "lowpass": BOTH},
        "single_1d": {"case": "single_1d", "Nb": 1, "Nc": 1, "seed": 309, "lowpass": BOTH},
        "A_standardized": {
            "case": "A", "Nb": 1, "Nc": 1, "seed": 310, "lowpass": ["ST"], "standardize": True,
        },
        "A_cross_matrix": {
            "case": "A", "Nb": 1, "Nc": 3, "seed": 311, "lowpass": ["ST"],
            "cross_matrix": [[True, False, True], [False, False, True], [False, False, True]],
        },
    },
}  # fmt: skip
MODES = {"ST": "st", None: "none"}
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
    one so that the cross statistics are not small."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((nb, nc, *shape))
    channels = [noise[:, 0]]
    for c in range(1, nc):
        channels.append(0.6 * channels[-1] + 0.8 * noise[:, c])
    field: NDArray[np.float64] = np.stack(channels, axis=1) + 0.3
    return field


def case_field(case: dict[str, Any], nb: int, nc: int, seed: int) -> NDArray[np.float64]:
    """The seeded field of ``case``, the spatial axes of its blocks in block order."""
    return input_field(tuple(n for sizes in case["sizes"] for n in sizes), nb, nc, seed)


def tensor_data(case: dict[str, Any], field: NDArray[np.float64]) -> Any:
    """The STL tensor-product data of ``field``, with the blocks and periodicity of ``case``."""
    tensor_prod = importlib.import_module("STL_main.Tensor_Product_Kernel_Torch")
    blocks = {
        "2D": importlib.import_module("STL_main.STL_2D_Kernel_Torch").STL_2D_Kernel_Torch,
        "1D": importlib.import_module("STL_main.STL_1D_Kernel_Torch").STL_1D_Kernel_Torch,
    }
    return tensor_prod.STL_TensorProd_Kernel_Torch(
        array=torch.from_numpy(field.copy()),
        data_structure=[blocks[name] for name in case["blocks"]],
        pbc=list(case["pbc"]),
    )


def operator_kwargs(case: dict[str, Any]) -> dict[str, Any]:
    """J as a fresh list (STL completes a list holding None in place, OP-83), or nothing, so
    that STL takes its default J per block."""
    return {} if case["J"] is None else {"J": list(case["J"])}


def _numpy(x: Any) -> NDArray[Any]:
    array: NDArray[Any] = x.detach().cpu().numpy()
    return array


def run() -> dict[str, NDArray[Any]]:
    """Run STL's tensor-product scattering operator on each configuration."""
    out: dict[str, NDArray[Any]] = {}
    for name, config in PARAMETERS["configs"].items():
        case = CASES[config["case"]]
        field = case_field(case, config["Nb"], config["Nc"], config["seed"])
        out[f"{name}_field"] = field
        matrix = config.get("cross_matrix")
        computed: dict[Any, Any] = {}
        for lowpass in config["lowpass"]:
            data = tensor_data(case, field)
            operator = data.get_ST_op(
                lowpass=lowpass, norm="vanilla", compute_PS=False, **operator_kwargs(case)
            )
            computed[lowpass] = operator.apply(
                data,
                standardize=config.get("standardize", False),
                compute_cross_matrix=None if matrix is None else torch.tensor(matrix),
            )
        if None in computed:
            # maths.md M-LP, "Checked": with lowpass=None, S3 and S4 are bit for bit those of
            # "ST", so the fixture holds them once.
            for family in ("S3", "S4"):
                with_lp, without = (_numpy(getattr(computed[m], family)) for m in ("ST", None))
                assert with_lp.shape == without.shape, (name, family)
                assert np.array_equal(with_lp, without, equal_nan=True), (name, family)
        for lowpass, stats in computed.items():
            prefix = f"{name}_{MODES[lowpass]}"
            out[f"{prefix}_mean"] = _numpy(stats.mean)
            out[f"{prefix}_var"] = _numpy(stats.var)
            for family in GRIDS if lowpass == "ST" else ("S1", "S2"):
                out.update(pack_grid(f"{prefix}_{family}", _numpy(getattr(stats, family))))
            if config.get("standardize", False):
                for key in ("mean_pre_std", "std_pre_std"):
                    out[f"{prefix}_{key}"] = _numpy(getattr(stats, key))
    return out
