"""Parity case: the first iterations of a periodic synthesis of STL `master` with the power
spectrum in the loss (Part II, §14.2, §14.3; task T4.3, issue #150, for T4.5; maths.md section 5,
M-SYN-OPS, M-PS-2D).

The case of ``synthesis_2d`` (T6.1), with the spectrum: the same 64 x 64 periodic target, one
channel, float64, the same seed and settings. Where ``synthesis_2d`` calls the mid-level
``optimize_from_maps`` with ``compute_PS=False``, this case calls the user-level
``synthesize_from_maps`` itself, which builds the two operators (M-SYN-OPS: J the default of
M-DEF, here 4, passed explicitly to both, so the spectrum takes it as J_PS; the number of bins the
smaller of the two fields' defaults, here 9; ``replace_nan_value=None`` for the running one) and,
since neither field has NaN, puts the spectrum in the loss, with the ``fft`` estimator. Its
optimisation settings are its defaults (M-SYN-LBFGS: step 1.0, 50 correction pairs, strong Wolfe
line search, tolerances 1e-12 and 1e-15; seed 26, the Nyquist pre-filter, the weights 3.5 and
3.5² of S3 and S4, ``mean_field=True``), but ``max_iter``.

The loss after each iteration is read where PyTorch's L-BFGS gets it, as in ``synthesis_2d``:
the value its strong Wolfe line search returns, at the point it accepts
(``torch.optim.lbfgs._strong_wolfe``, wrapped, not changed). The operators are read by wrapping
``optimize_from_maps``, which the user level calls with them. STL's code is not changed.

It stores:

* ``target``: the target, ``(N, N)``: the lognormal field of ``synthesis_2d``;
* ``start``: the Gaussian start before the pre-filter, ``(1, 1, N, N)``, drawn as STL draws it
  (``torch.manual_seed(26)``, then ``torch.randn``; M-SYN-INIT);
* ``losses``: the loss after each iteration, ``(n_iter,)``;
* ``evaluations``: every value of the loss STL's closure computed, in order (STL's
  ``loss_history``; the first is the loss at the start);
* ``counts``: ``[n_iter, n_evals]``, the iterations and the evaluations of the loss;
* ``field``: the synthesised field STL returns, de-standardised, ``(N, N)``;
* ``settings``: ``[J, J_PS, n_b, compute_PS]`` of the operators STL built, ``(4,)``;
* ``var_ref``, ``s2_ref`` and ``ps_ref``: the reference the target operator stored and the
  running one received (``var_ref``, ``S2_ref_sqrt_chan_diag``, ``PS_ref_sqrt_chan_diag``),
  ``(1, 1)``, ``(1, 1, J, L)`` and ``(1, 1, n_b)``.

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
    "n_bins": 9,
    "target_seed": 61,
    "spectral_index": 3.0,
    "lognormal_scale": 0.8,
    "seed": 26,
    "max_iter": 20,
    "history_size": 50,
    "lr": 1.0,
    "mean_field": True,
    "compute_PS": True,
}


def target_field(n: int, seed: int, index: float, scale: float) -> NDArray[np.float64]:
    """The target of ``synthesis_2d``: exp(scale · g), g a unit Gaussian field of spectrum
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
    array: NDArray[Any] = np.asarray(x.detach().cpu().numpy())
    return array


def run() -> dict[str, NDArray[Any]]:
    """Run STL's user-level synthesis for ``max_iter`` iterations and return its losses."""
    data_cls = importlib.import_module("STL_main.STL_2D_Kernel_Torch").STL_2D_Kernel_Torch
    synthesis = importlib.import_module("STL_main.Synthesis")
    p = PARAMETERS
    target = target_field(p["N"], p["target_seed"], p["spectral_index"], p["lognormal_scale"])
    data = data_cls(array=torch.from_numpy(target.copy()), pbc=True)

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

    # The operators and settings the user level passes to the mid level.
    seen: dict[str, Any] = {}
    optimize_from_maps = synthesis.optimize_from_maps

    def keeping_operators(*args: Any, **kwargs: Any) -> Any:
        seen.update(kwargs)
        return optimize_from_maps(*args, **kwargs)

    setattr(lbfgs, "_strong_wolfe", recording)  # noqa: B010 (a module of torch)
    setattr(synthesis, "optimize_lbfgs", keeping_history)  # noqa: B010
    setattr(synthesis, "optimize_from_maps", keeping_operators)  # noqa: B010
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            field = synthesis.synthesize_from_maps(
                data,
                nbatch=1,
                pbc_running=True,
                max_iter=p["max_iter"],
                verbose=False,
            )
    finally:
        setattr(lbfgs, "_strong_wolfe", strong_wolfe)  # noqa: B010
        setattr(synthesis, "optimize_lbfgs", optimize_lbfgs)  # noqa: B010
        setattr(synthesis, "optimize_from_maps", optimize_from_maps)  # noqa: B010

    # The settings of M-SYN-OPS and M-SYN-LBFGS this case states, as STL used them.
    st_target, st_running = seen["st_op_target"], seen["st_op_running"]
    # STL's compute_PS here is a 0-d boolean tensor (not (target_has_nan or running_has_nan)).
    assert bool(seen["compute_PS"]) is p["compute_PS"]
    assert seen["mean_field"] is p["mean_field"]
    assert (seen["seed"], seen["lr"], seen["history_size"]) == (p["seed"], p["lr"], 50)
    assert seen["prefilter_Nyquist"] is True
    assert seen["adhoc_weights"] == {"S3": 3.5, "S4": 3.5**2}
    for operator in (st_target, st_running):
        assert (operator.J, operator.L, operator.n_bins) == (p["J"], p["L"], p["n_bins"])
        assert (operator.CS_op.J, operator.CS_op.spectrum_method) == (p["J"], "fft")
    assert st_running is not st_target
    assert st_running.PS_ref_sqrt_chan_diag is st_target.PS_ref_sqrt_chan_diag

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
        "field": np.asarray(_numpy(field), dtype=np.float64),
        "settings": np.asarray(
            [st_target.J, st_target.CS_op.J, st_target.n_bins, float(bool(seen["compute_PS"]))],
            dtype=np.float64,
        ),
        "var_ref": _numpy(st_target.var_ref),
        "s2_ref": _numpy(st_target.S2_ref_sqrt_chan_diag),
        "ps_ref": _numpy(st_target.PS_ref_sqrt_chan_diag),
    }
