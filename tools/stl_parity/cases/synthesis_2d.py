"""Parity case: the first iterations of a periodic synthesis of STL `master` (Part II, §14.2,
§14.3; task T6.1, issue #28, decisions 2 and 3 of 21 November 2026; maths.md section 5).

One periodic planar case, one channel, float64: a 64 x 64 target, J = 4, L = 4. STL runs its
synthesis from maps with its own settings (M-SYN-LBFGS: step 1.0, 50 correction pairs, strong
Wolfe line search, tolerances 1e-12 and 1e-15, ⌊5 max_iter / 4⌋ evaluations; seed 26, the
Nyquist pre-filter, the weights 3.5 and 3.5² of S3 and S4, ``mean_field=True``), without the
power spectrum, which ESTEL computes from T4.1 on: the mid-level ``optimize_from_maps`` with
``compute_PS=False`` (its default), given the two operators the user-level
``synthesize_from_maps`` builds (``get_ST_op`` with the synthesis J, the number of bins and
``has_fewer_convolutions``; ``replace_nan_value=None`` for the running one). The user level
itself always puts the spectrum in the loss of a periodic synthesis (M-SYN-OPS).

The loss after each iteration is read where PyTorch's L-BFGS gets it: the value its strong
Wolfe line search returns, at the point it accepts (``torch.optim.lbfgs._strong_wolfe``,
wrapped, not changed). STL's code is not changed.

It stores:

* ``target``: the target, ``(N, N)``: a lognormal field with a power-law spectrum, seeded;
* ``start``: the Gaussian start before the pre-filter, ``(1, 1, N, N)``, drawn as STL draws it
  (``torch.manual_seed(26)``, then ``torch.randn``; M-SYN-INIT);
* ``losses``: the loss after each iteration, ``(n_iter,)``;
* ``evaluations``: every value of the loss STL's closure computed, in order (STL's
  ``loss_history``; the first is the loss at the start);
* ``counts``: ``[n_iter, n_evals]``, the iterations and the evaluations of the loss;
* ``field``: the synthesised field STL returns, de-standardised, ``(N, N)``.

Runs in the parity container, with the interpreter of the `master` environment, through
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

STL_ENV = "master"
PARAMETERS: dict[str, Any] = {
    "N": 64,
    "J": 4,
    "L": 4,
    "target_seed": 61,
    "spectral_index": 3.0,
    "lognormal_scale": 0.8,
    "seed": 26,
    "max_iter": 20,
    "history_size": 50,
    "lr": 1.0,
    "mean_field": True,
    "compute_PS": False,
}


def target_field(n: int, seed: int, index: float, scale: float) -> NDArray[np.float64]:
    """A periodic lognormal field: exp(scale · g), g a unit Gaussian field of spectrum
    |k|^(-index), so that its third and fourth order statistics are not those of noise."""
    rng = np.random.default_rng(seed)
    k = np.fft.fftfreq(n) * n
    radius = np.sqrt(k[:, None] ** 2 + k[None, :] ** 2)
    amplitude = np.where(radius > 0, np.maximum(radius, 1.0) ** (-index / 2), 0.0)
    g = np.fft.ifft2(np.fft.fft2(rng.standard_normal((n, n))) * amplitude).real
    g = (g - g.mean()) / g.std()
    field: NDArray[np.float64] = np.exp(scale * g)
    return field


def _numpy(x: Any) -> NDArray[Any]:
    array: NDArray[Any] = np.asarray(x.detach().cpu().numpy(), dtype=np.float64)
    return array


def run() -> dict[str, NDArray[Any]]:
    """Run STL's synthesis for ``max_iter`` iterations and return its losses."""
    data_cls = importlib.import_module("STL_main.STL_2D_Kernel_Torch").STL_2D_Kernel_Torch
    synthesis = importlib.import_module("STL_main.Synthesis")
    p = PARAMETERS
    target = target_field(p["N"], p["target_seed"], p["spectral_index"], p["lognormal_scale"])
    data = data_cls(array=torch.from_numpy(target.copy()), pbc=True)

    # The operators of the user-level synthesize_from_maps (M-SYN-OPS), periodic case.
    running = data_cls(array=torch.from_numpy(target.copy()), pbc=True)
    n_bins = min(data.get_CS_op().n_bins, running.get_CS_op().n_bins)
    st_op_target = data.get_ST_op(J=p["J"], n_bins=n_bins, has_fewer_convolutions=False)
    st_op_running = running.get_ST_op(
        J=p["J"], n_bins=n_bins, has_fewer_convolutions=False, replace_nan_value=None
    )
    assert p["L"] == st_op_target.wavelet_op.L  # the default L (M-DEF)

    # The loss PyTorch's L-BFGS accepts at each iteration: what its line search returns.
    accepted: list[float] = []
    lbfgs = importlib.import_module("torch.optim.lbfgs")  # torch.optim does not export it
    strong_wolfe = lbfgs._strong_wolfe

    def recording(*args: Any, **kwargs: Any) -> Any:
        out = strong_wolfe(*args, **kwargs)
        accepted.append(float(out[0]))
        return out

    evaluations: list[float] = []
    optimize_lbfgs = synthesis.optimize_lbfgs

    def keeping_history(model: Any, loss_fn: Any, *args: Any, **kwargs: Any) -> Any:
        def loss(output: Any) -> Any:
            value = loss_fn(output)
            evaluations.append(float(value.detach()))
            return value

        return optimize_lbfgs(model, loss, *args, **kwargs)

    setattr(lbfgs, "_strong_wolfe", recording)  # noqa: B010 (a module of torch)
    setattr(synthesis, "optimize_lbfgs", keeping_history)  # noqa: B010
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            field = synthesis.optimize_from_maps(
                target=data,
                st_op_target=st_op_target,
                st_op_running=st_op_running,
                nbatch=1,
                pbc_running=True,
                compute_PS=p["compute_PS"],
                mean_field=p["mean_field"],
                lr=p["lr"],
                max_iter=p["max_iter"],
                history_size=p["history_size"],
                verbose=False,
                seed=p["seed"],
                prefilter_Nyquist=True,
            )
    finally:
        setattr(lbfgs, "_strong_wolfe", strong_wolfe)  # noqa: B010
        setattr(synthesis, "optimize_lbfgs", optimize_lbfgs)  # noqa: B010

    # The start, drawn again as STL drew it: nothing is drawn between the seeding and the
    # draw (M-SYN-INIT).
    torch.manual_seed(p["seed"])
    start = torch.randn((1, 1, p["N"], p["N"]), dtype=torch.float64)
    return {
        "target": target,
        "start": _numpy(start),
        "losses": np.asarray(accepted, dtype=np.float64),
        "evaluations": np.asarray(evaluations, dtype=np.float64),
        "counts": np.asarray([len(accepted), len(evaluations)], dtype=np.float64),
        "field": _numpy(field),
    }
