"""Parity case: the gradient of the synthesis loss of STL `2D+1D` on a tensor product (Part II,
§14.3 "Gradients"; task T3.4, issue #127, decisions of the maintainer, second version; maths.md
M-SYN-LOSS, M-NORM-TENSORPROD, M-FLAT).

On case C of maths.md §3 (a 16 x 16 periodic 2D block and an 8-sample open 1D block, STL's
default J = (2, 1), L = (4, 2), ``lowpass="ST"``), with two maps of two channels, computes the
loss of maths.md M-SYN-LOSS with STL's own code and its gradient with respect to the field:

* the target statistics: ``apply`` on the target field with ``norm="store_ref"`` and
  ``norm_batch_mean=True`` (``mean_field=True``, STL's default), the weights 3.5 and 3.5² of S3
  and S4 (``reweight`` with STL's ``adhoc_weights``), then
  ``to_flatten(mean_along_batch=True, keepnans=False)``, as ``optimize_from_maps`` does;
* the running statistics: STL's ``ScatteringMatchModel`` on the running field (``init_map``),
  with the same operator, hence the target's reference (``norm="load_ref"``), without the
  Nyquist pre-filter, and with the same weights and flattening;
* the loss ``((Φ(u) - Φ(target)).abs() ** 2).sum()``, the ``loss_fn`` of ``optimize_from_maps``,
  and its gradient by ``backward``.

The target is used as it is: neither standardised nor pre-filtered (M-SYN-TARGET is not part of
this case).

It stores ``field`` (the running field) and ``target``, ``(Nb, Nc, 16, 16, 8)``, ``loss``,
``(1,)``, and ``gradient``, the gradient of the loss with respect to ``field``, of its shape.

Runs in the parity container, with the interpreter of the `2d1d` environment, through
`make parity-fixtures` (tools/stl_parity/make_fixtures.py). It imports STL, never `estel`.
"""

from __future__ import annotations

import contextlib
import importlib
import io
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
    "seed": 500,
    "target_seed": 501,
    "mean_field": True,
    "adhoc_weights": {"S3": 3.5, "S4": 3.5**2},
}


def input_field(shape: tuple[int, ...], nb: int, nc: int, seed: int) -> NDArray[np.float64]:
    """Seeded Gaussian fields, ``(Nb, Nc, *shape)``, each channel correlated with the previous
    one so that the cross statistics are not small (as in ``scattering_tensor_prod``)."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((nb, nc, *shape))
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


def run() -> dict[str, NDArray[Any]]:
    """Compute STL's synthesis loss on the running field and its gradient."""
    synthesis = importlib.import_module("STL_main.Synthesis")
    p = PARAMETERS
    shape = tuple(n for sizes in p["sizes"] for n in sizes)
    field = input_field(shape, p["Nb"], p["Nc"], p["seed"])
    target_field = input_field(shape, p["Nb"], p["Nc"], p["target_seed"])
    target = tensor_data(target_field)
    operator = target.get_ST_op(lowpass=p["lowpass"], compute_PS=False)

    with torch.no_grad():
        target_stats = operator.apply(
            target, norm="store_ref", norm_batch_mean=p["mean_field"], compute_PS=False
        )
        synthesis.reweight(target_stats, p["adhoc_weights"])
        target_vector = target_stats.to_flatten(
            mean_along_batch=p["mean_field"], keepnans=False
        ).detach()

    with contextlib.redirect_stdout(io.StringIO()):
        model = synthesis.ScatteringMatchModel(
            st_op=operator,
            DataClass=target.__class__,
            pbc=list(p["pbc"]),
            init_shape=field.shape,
            init_map=torch.from_numpy(field.copy()),
            device=torch.device("cpu"),
            dtype=torch.float64,
            has_fewer_convolutions=False,
            compute_cross_matrix=None,
            compute_PS=False,
            keep_batch_dim=False,
            mean_field=p["mean_field"],
            prefilter_Nyquist=False,
            adhoc_weights=p["adhoc_weights"],
        )
    loss = ((model() - target_vector).abs() ** 2).sum()  # loss_fn of optimize_from_maps
    loss.backward()
    gradient: NDArray[np.float64] = model.u.grad.detach().cpu().numpy()
    assert np.isfinite(gradient).all()
    return {
        "field": field,
        "target": target_field,
        "loss": np.asarray([float(loss.detach())], dtype=np.float64),
        "gradient": gradient,
    }
