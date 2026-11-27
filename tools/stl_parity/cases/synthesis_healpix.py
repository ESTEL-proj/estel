"""Parity case: the first iterations of a full-sky HEALPix synthesis of STL `HealpixUpdate`
(Part II, §14.2, §14.3; task T5.3, issue #164, decision 4; maths.md section 5, M-SYN-OPS,
OP-77).

One full-sky case at ``nside`` 16, one channel, float64, L = 4 gauges, with the settings of
``_healpix.py`` (the sphere, no geometry cache, the kernel in float64; OP-45, OP-46). STL runs
its synthesis from maps with its own settings (M-SYN-LBFGS: step 1.0, 50 correction pairs,
strong Wolfe line search, tolerances 1e-12 and 1e-15, ⌊5 max_iter / 4⌋ evaluations; seed 26,
the weights 3.5 and 3.5² of S3 and S4, ``mean_field=True``), without the band limit of the
target and of the start (``prefilter_Nyquist=False``) and without the angular spectrum
(``compute_PS=False``), which ESTEL 1.0.0 does not port (OP-77, LF24).

**The mid level, given the operators of the user level.** STL's user-level
``synthesize_from_maps`` builds its two operators without ``wavelet_op_kwargs``, so on HEALPix
it always runs on the ``"WGS84"`` ellipsoid with healpix-analyse's cache on. This case therefore
calls the mid-level ``optimize_from_maps``, as ``synthesis_2d`` does, with the two operators the
user level builds, rebuilt here with the settings of ``_healpix.py``:

* the running data: ``data_target.new_like(data_target.array, pbc=True)``, the target's grid
  (``running_mask`` and ``running_shape`` not given);
* J: for each, the default J of its wavelet operator, minus 1 if its data are not periodic
  (``get_wavelet_op().J - (not pbc)``), and the smaller of the two: here 3 = max(1, log₂ 16 -
  1), the full sky being periodic (M-DEF-HEALPIX);
* ``compute_PS=False`` and ``has_fewer_convolutions=False`` for both (no ``n_bins`` without the
  spectrum), ``replace_nan_value=None`` for the running one; L, the wavelet type and the kernel
  size are STL's defaults (M-SYN-OPS), L = 4 checked;
* the other settings of the user level: ``nbatch=1``, ``pbc_running=True``, ``max_iter``, ``lr``,
  ``history_size``, ``seed=26`` and the weights, with ``prefilter_Nyquist=False``.

The loss after each iteration is read where PyTorch's L-BFGS gets it: the value its strong
Wolfe line search returns, at the point it accepts (``torch.optim.lbfgs._strong_wolfe``,
wrapped, not changed). STL's code is not changed.

It stores:

* ``target``: the target, ``(Npix,)``, NESTED: exp(0.8 g), with g a Gaussian field of angular
  spectrum (l + 1)^(-``spectral_index``) for l ≥ 1, made from seeded harmonic coefficients by
  healpy's ``alm2map`` and standardised;
* ``start``: the Gaussian start STL used, ``(1, 1, Npix)``, not band-limited, read from the
  model before the first step. It is not the first draw after ``torch.manual_seed(26)``
  (M-SYN-INIT): between the seeding and ``torch.randn``, the target operator builds its J
  convolutions (one per level, when it first computes the target's statistics), and each
  ``HealPixConv`` initialises its weights with ``kaiming_uniform_`` from the global generator
  before ``set_kernel`` replaces them; the case checks this, bit for bit;
* ``losses``: the loss after each iteration, ``(n_iter,)``;
* ``evaluations``: every value of the loss STL's closure computed, in order (STL's
  ``loss_history``; the first is the loss at the start);
* ``counts``: ``[n_iter, n_evals]``, the iterations and the evaluations of the loss;
* ``field``: the synthesised field STL returns, de-standardised, ``(Npix,)``.

Runs in the parity container, with the interpreter of the `healpix` environment, through
`make parity-fixtures` (tools/stl_parity/make_fixtures.py). It imports STL, never `estel`.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import io
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

STL_ENV = "healpix"


# Repeated in each HEALPix case: make_fixtures.py imports a case by its path, so a case cannot
# import a sibling module, and this loader must exist before _healpix.py can be read.
def _helper() -> ModuleType:
    """The settings of the HEALPix cases, ``_healpix.py``, imported by its path."""
    path = Path(__file__).resolve().with_name("_healpix.py")
    spec = importlib.util.spec_from_file_location("stl_parity_cases_healpix", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HEALPIX = _helper()
PARAMETERS: dict[str, Any] = {
    "nside": 16,
    "J": 3,
    "L": 4,
    "target_seed": 62,
    "spectral_index": 2.0,
    "lognormal_scale": 0.8,
    "seed": 26,
    "max_iter": 20,
    "history_size": 50,
    "lr": 1.0,
    "mean_field": True,
    "compute_PS": False,
    "prefilter_Nyquist": False,
    "stl_settings": HEALPIX.SETTINGS,
}


def target_field(nside: int, seed: int, index: float, scale: float) -> NDArray[np.float64]:
    """A full-sky lognormal field, NESTED: exp(scale · g), g a unit Gaussian field of angular
    spectrum (l + 1)^(-index) for l ≥ 1 and 0 at l = 0, up to l = 3 nside - 1."""
    healpy = importlib.import_module("healpy")
    lmax = 3 * nside - 1
    ell, m = healpy.Alm.getlm(lmax)
    spectrum = np.where(ell > 0, (ell + 1.0) ** (-index), 0.0)
    rng = np.random.default_rng(seed)
    parts = rng.standard_normal((2, ell.size))
    # m = 0: real, of variance C_ell; m > 0: real and imaginary parts of variance C_ell / 2 each.
    alm = np.where(m == 0, parts[0], (parts[0] + 1j * parts[1]) / np.sqrt(2.0))
    ring = healpy.alm2map(alm * np.sqrt(spectrum), nside, lmax=lmax)
    g = np.asarray(healpy.reorder(ring, r2n=True), dtype=np.float64)
    g = (g - g.mean()) / g.std()
    field: NDArray[np.float64] = np.exp(scale * g)
    return field


def _numpy(x: Any) -> NDArray[Any]:
    array: NDArray[Any] = np.asarray(x.detach().cpu().numpy(), dtype=np.float64)
    return array


def run() -> dict[str, NDArray[Any]]:
    """Run STL's synthesis for ``max_iter`` iterations and return its losses."""
    data_cls = importlib.import_module("STL_main.STL_Healpix_Kernel_Torch").STL_Healpix_Kernel_Torch
    synthesis = importlib.import_module("STL_main.Synthesis")
    p = PARAMETERS
    target = target_field(p["nside"], p["target_seed"], p["spectral_index"], p["lognormal_scale"])
    n_pix = target.size

    # The loss PyTorch's L-BFGS accepts at each iteration: what its line search returns.
    accepted: list[float] = []
    lbfgs = importlib.import_module("torch.optim.lbfgs")  # torch.optim does not export it
    strong_wolfe = lbfgs._strong_wolfe

    def recording(*args: Any, **kwargs: Any) -> Any:
        out = strong_wolfe(*args, **kwargs)
        accepted.append(float(out[0]))
        return out

    evaluations: list[float] = []
    starts: list[torch.Tensor] = []
    optimize_lbfgs = synthesis.optimize_lbfgs

    def keeping_history(model: Any, loss_fn: Any, *args: Any, **kwargs: Any) -> Any:
        starts.append(model.u.detach().clone())  # the start, before the first step

        def loss(output: Any) -> Any:
            value = loss_fn(output)
            evaluations.append(float(value.detach()))
            return value

        return optimize_lbfgs(model, loss, *args, **kwargs)

    with HEALPIX.keep_kernel_float64():
        # The operators of the user-level synthesize_from_maps (M-SYN-OPS), full sky, with the
        # settings of _healpix.py.
        data = data_cls(array=torch.from_numpy(target.copy()))
        running = data.new_like(data.array, pbc=True)
        j_target = data.get_wavelet_op(**HEALPIX.WAVELET_OP_KWARGS).J - (not data.pbc)
        j_running = running.get_wavelet_op(**HEALPIX.WAVELET_OP_KWARGS).J - (not running.pbc)
        n_scales = min(j_target, j_running)
        if n_scales != p["J"]:
            raise ValueError(
                f"the user level would run with J = {n_scales}, not {p['J']}: set PARAMETERS['J'] "
                "to the J of synthesize_from_maps (M-SYN-OPS) and describe it in the docstring."
            )
        options: dict[str, Any] = {
            "compute_PS": p["compute_PS"],
            "has_fewer_convolutions": False,
            "wavelet_op_kwargs": HEALPIX.WAVELET_OP_KWARGS,
        }
        st_op_target = data.get_ST_op(J=n_scales, **options)
        st_op_running = running.get_ST_op(J=n_scales, replace_nan_value=None, **options)
        if not p["L"] == st_op_target.wavelet_op.L == st_op_running.wavelet_op.L:
            raise ValueError(
                f"STL's operators have L = {st_op_target.wavelet_op.L} and "
                f"{st_op_running.wavelet_op.L}, not {p['L']}: set PARAMETERS['L'] to STL's "
                "default L (M-DEF-HEALPIX)."
            )

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
                    prefilter_Nyquist=p["prefilter_Nyquist"],
                )
        finally:
            setattr(lbfgs, "_strong_wolfe", strong_wolfe)  # noqa: B010
            setattr(synthesis, "optimize_lbfgs", optimize_lbfgs)  # noqa: B010
        for operator in (st_op_target, st_op_running):
            HEALPIX.check_kernel_float64(operator.wavelet_op)

    # The start STL used is not the first draw after the seeding (M-SYN-INIT): the target
    # operator builds its convolutions when it first computes the target's statistics, after
    # the seeding, and each HealPixConv initialises its weights with kaiming_uniform_ from the
    # global generator before set_kernel replaces them. Checked here, bit for bit.
    torch.manual_seed(p["seed"])
    for _ in st_op_target.wavelet_op._conv_cache:  # one per level, in the order of the loop
        weight = torch.empty(p["L"], 1, 2, 25, dtype=torch.float64)
        torch.nn.init.kaiming_uniform_(weight.view(p["L"], 50), a=0.0, nonlinearity="relu")
    if not torch.equal(torch.randn((1, 1, n_pix), dtype=torch.float64), starts[0]):
        raise ValueError(
            "STL's start is not the draw that follows the seeding and the initialisation of "
            "the target's convolutions: check that the image holds the STL and healpix-analyse "
            "commits of Part II, §9.1 (make parity), then update the docstring of this case."
        )
    return {
        "target": target,
        "start": _numpy(starts[0]),
        "losses": np.asarray(accepted, dtype=np.float64),
        "evaluations": np.asarray(evaluations, dtype=np.float64),
        "counts": np.asarray([len(accepted), len(evaluations)], dtype=np.float64),
        "field": _numpy(field),
    }
