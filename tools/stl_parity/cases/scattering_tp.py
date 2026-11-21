"""Parity case: the vanilla scattering statistics of STL `2D+1D` on tensor products (Part II,
§14.2, §14.4; task T3.4, issue #127; maths.md M-DATA-TP, M-W1-TP, M-LP, M-IDX-TP, M-S-TP,
M-BORDER-TP, OP-30, OP-42, OP-43, OP-81, OP-82).

Runs STL's tensor-product scattering operator (``ST_Operator_TensorProd.apply``) without
normalisation and without spectrum (``norm="vanilla"``, ``compute_PS=False``) on seeded fields,
and stores, for each configuration ``<config>``:

* ``<config>_field``: the input, ``(Nb, Nc, *N⁽⁰⁾, …, *N⁽ᴮ⁻¹⁾)``;
* ``<config>_<family>`` for ``mean``, ``var`` (``(Nb, Nc)``, float64) and ``S1``, ``S2``,
  ``S3``, ``S4`` (STL's full grids, complex128, NaN where STL computes no coefficient);
* with ``standardize``, ``<config>_mean_pre_std`` and ``<config>_std_pre_std``, ``(Nb, Nc)``.

The configurations are those of the decisions of the maintainer on issue #127: the seven cases A
to G of maths.md §3 (blocks, sizes, periodicity, J; two maps of two channels unless said), each
with ``lowpass="ST"`` (``<case>_st``) and ``lowpass=None`` (``<case>_none``); a ``[1D, 1D]``
field; a tensor product of one 2D block and of one 1D block, with ``"ST"`` and ``None`` (OP-82);
``standardize=True`` on case A; and a ``compute_cross_matrix`` where channel 1 has no auto
statistics, on case A (OP-30). ``J`` set to ``None`` leaves J out of the call, so that STL takes
its default per block (cases C and D, M-DATA-TP); L is always STL's default per block (4 for a 2D
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
# The seven cases of maths.md §3, then the configurations of the decisions on issue #127.
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
    "G": {
        "blocks": ["2D", "1D"], "sizes": [[32, 16], [32]], "pbc": [False, False], "J": [3, 3],
        "Nb": 1, "Nc": 1,
    },
    "one_d_one_d": {
        "blocks": ["1D", "1D"], "sizes": [[16], [16]], "pbc": [True, False], "J": [2, 2],
    },
    "single_2d": {"blocks": ["2D"], "sizes": [[16, 16]], "pbc": [True], "J": [2]},
    "single_1d": {"blocks": ["1D"], "sizes": [[16]], "pbc": [False], "J": [2]},
}  # fmt: skip
PARAMETERS: dict[str, Any] = {
    "Nb": 2,
    "Nc": 2,
    "cases": CASES,
    "configs": {
        **{
            f"{case}_{mode}": {"case": case, "lowpass": lowpass, "seed": 300 + 2 * i + k}
            for i, case in enumerate(CASES)
            for k, (mode, lowpass) in enumerate((("st", "ST"), ("none", None)))
            if case != "one_d_one_d" or lowpass == "ST"
        },
        "A_standardized": {"case": "A", "lowpass": "ST", "seed": 330, "standardize": True},
        "A_cross_matrix": {
            "case": "A", "lowpass": "ST", "seed": 331, "Nc": 3,
            "cross_matrix": [[True, False, True], [False, False, True], [False, False, True]],
        },
    },
}  # fmt: skip
FAMILIES = ("mean", "var", "S1", "S2", "S3", "S4")


def input_field(case: dict[str, Any], nb: int, nc: int, seed: int) -> NDArray[np.float64]:
    """Seeded Gaussian fields, ``(Nb, Nc, *sizes)``, each channel correlated with the previous
    one so that the cross statistics are not small."""
    rng = np.random.default_rng(seed)
    shape = tuple(n for sizes in case["sizes"] for n in sizes)
    noise = rng.standard_normal((nb, nc) + shape)
    channels = [noise[:, 0]]
    for c in range(1, nc):
        channels.append(0.6 * channels[-1] + 0.8 * noise[:, c])
    field: NDArray[np.float64] = np.stack(channels, axis=1) + 0.3
    return field


def data_classes() -> tuple[Any, dict[str, Any]]:
    """STL's tensor-product data class and its block classes, by geometry name."""
    tensor_prod = importlib.import_module("STL_main.Tensor_Product_Kernel_Torch")
    blocks = {
        "2D": importlib.import_module("STL_main.STL_2D_Kernel_Torch").STL_2D_Kernel_Torch,
        "1D": importlib.import_module("STL_main.STL_1D_Kernel_Torch").STL_1D_Kernel_Torch,
    }
    return tensor_prod.STL_TensorProd_Kernel_Torch, blocks


def tensor_data(case: dict[str, Any], field: NDArray[np.float64]) -> Any:
    """The STL tensor-product data of ``field``, with the blocks and periodicity of ``case``."""
    data_cls, blocks = data_classes()
    return data_cls(
        array=torch.from_numpy(field.copy()),
        data_structure=[blocks[name] for name in case["blocks"]],
        pbc=list(case["pbc"]),
    )


def operator_kwargs(case: dict[str, Any]) -> dict[str, Any]:
    """J as a fresh list (STL completes a list holding None in place, OP-83), or nothing."""
    return {} if case["J"] is None else {"J": list(case["J"])}


def run() -> dict[str, NDArray[Any]]:
    """Run STL's tensor-product scattering operator on each configuration."""
    out: dict[str, NDArray[Any]] = {}
    for name, config in PARAMETERS["configs"].items():
        case = CASES[config["case"]]
        nb = config.get("Nb", case.get("Nb", PARAMETERS["Nb"]))
        nc = config.get("Nc", case.get("Nc", PARAMETERS["Nc"]))
        field = input_field(case, nb, nc, config["seed"])
        out[f"{name}_field"] = field
        data = tensor_data(case, field)
        operator = data.get_ST_op(
            lowpass=config["lowpass"], norm="vanilla", compute_PS=False, **operator_kwargs(case)
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
