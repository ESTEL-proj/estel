"""A periodic synthesis follows STL's loss over its first iterations (Part II, §14.3, "Syntheses";
task T6.1, issue #28, decisions 2 and 3 of 21 November 2026; task T4.5, issue #152; task T5.8,
issue #169, decision 9; maths.md section 5).

Two fixtures (STL `master`; ``tools/stl_parity/cases/``) hold a 64 x 64 periodic target, J = 4,
L = 4, one channel, the Gaussian start STL drew, and STL's loss after each of its 20 iterations,
run with STL's settings (M-SYN-LBFGS): ``synthesis_2d`` without the power spectrum, from STL's
mid-level ``optimize_from_maps`` (T6.1), and ``synthesis_ps_2d`` with it, from the user-level
``synthesize_from_maps``, whose operators are those of M-SYN-OPS: J = 4, 9 bins and the
``fft`` estimator (T4.5). A third (STL ``HealpixUpdate``, T5.3) holds a full HEALPix sky at
``nside`` 16, J = 3, L = 4, one channel: ``synthesis_hpx``, from the mid-level
``optimize_from_maps`` without the pre-filter and without the spectrum (OP-77), on the sphere
and with the kernel in float64 (M-SYN-OPS, HEALPix), with the start STL used, which is not the
first draw after its seed (OP-99). ESTEL synthesises from the same target and the same start,
with the same settings, in float64, and must make as many iterations and evaluations of the
loss, with the loss after each iteration within a relative tolerance of STL's, and end on the
field STL ends on.

**Tolerance.** |ESTEL - STL| ≤ rtol x |STL| at every iteration, with rtol the float64 parity
tolerance of Part II, §14.3 (``PARITY_RTOL["float64"]``, 1e-10), without an absolute term: the
losses are positive and fall by three orders of magnitude, so an absolute term scaled on the
first would hide the last. Measured on T6.1 (PR description): the relative gap is at most
1.0e-15 over these 20 iterations, from 0 at the first; on 60 iterations of the same case it
grows to 6e-14. The margin is five orders of magnitude, for the rounding of other machines and
BLAS. The final field, de-standardised, is compared with ``assert_parity``: the same rtol, and
the absolute term of §14.3 for its values near zero; measured, its largest difference is 1.2e-14
for values up to 6.4, 9e-5 of what the tolerance allows. With the spectrum (T4.5, measured): the
relative gap of the loss is at most 3.5e-15, and the largest difference of the field 1.3e-13
for values up to 6.0. On HEALPix (T5.8, measured): the relative gap of the loss is at most
5.0e-13, at iteration 17, and the largest difference of the field 9.1e-13 for values up to 9.0.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

import estel
from estel import Data2D, DataHealpix, _config
from estel.backend._protocol import Backend
from estel.data._cartesian import nyquist_filter
from estel.scattering._plan import STPlanSingleDomain
from estel.synthesis._api import _maps_setup
from tests.parity._compare import PARITY_RTOL, assert_parity, load_fixture, read_manifest

RTOL = PARITY_RTOL["float64"]
"""Relative tolerance on the loss at each iteration: the float64 parity rtol of Part II, §14.3."""

pytestmark = pytest.mark.maths(
    "M-SYN-TARGET",
    "M-SYN-INIT",
    "M-SYN-NYQ",
    "M-SYN-LOSS",
    "M-SYN-LBFGS",
    "M-SYN-OPS",
)


@pytest.mark.slow
def test_synthesis_follows_stl(backend: Backend, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the spectrum (T6.1).

    Slow: 20 iterations of L-BFGS on 64 x 64 pixels with J = 4, about 25 evaluations of the loss
    and its gradient, take about a second; the comparison runs in the full tier (decision 2).
    """
    _follows_stl(backend, monkeypatch, "synthesis_2d", "#28")


@pytest.mark.slow
def test_synthesis_with_the_spectrum_follows_stl(
    backend: Backend, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the spectrum in the loss (T4.5; maths.md M-SYN-OPS).

    Slow: 20 iterations of L-BFGS on 64 x 64 pixels with J = 4 and the spectrum, 24 evaluations
    of the loss and its gradient, take 0.5 s of CPU in a warm process and 2 s in a cold one
    (measured); the comparison runs in the full tier, as that of T6.1 (decisions on #152).
    """
    _follows_stl(backend, monkeypatch, "synthesis_ps_2d", "#152")


@pytest.mark.slow
def test_healpix_synthesis_follows_stl(backend: Backend, monkeypatch: pytest.MonkeyPatch) -> None:
    """A full HEALPix sky, without the pre-filter and without the spectrum (T5.8, decision 9 on
    #169; maths.md M-SYN-OPS, OP-77, OP-99).

    Slow: 20 iterations of L-BFGS on 3072 pixels with J = 3, 25 evaluations of the loss and its
    gradient, take about 6 s (8.5 s of CPU); the comparison runs in the full tier, beside those
    of T6.1 and T4.5.
    """
    _follows_stl(backend, monkeypatch, "synthesis_hpx", "#169")


def _follows_stl(backend: Backend, monkeypatch: pytest.MonkeyPatch, case: str, issue: str) -> None:
    monkeypatch.setattr(_config, "_dtype", "float64")
    fixture = load_fixture(case)
    healpix = case.endswith("_hpx")
    # An array is a 2D field; a HEALPix target is given as a DataHealpix (decision 2 on #169).
    target = DataHealpix(fixture["target"]) if healpix else fixture["target"]
    p: dict[str, Any] = read_manifest()[f"{case}.npz"]["parameters"]
    settings: dict[str, Any] = {
        "init": fixture["start"],
        "mean_field": p["mean_field"],
        "compute_PS": p["compute_PS"],
        "max_iter": p["max_iter"],
        "history_size": p["history_size"],
        "lr": p["lr"],
        "st_kwargs": {"J": p["J"]},  # L is the default, 4, as in STL (M-SYN-OPS)
    }
    if "lbfgs" in getattr(backend, "not_provided", frozenset()):
        with pytest.raises(NotImplementedError, match="torch backend"):
            estel.synthesize_from_maps(target, **settings)
        return
    result = estel.synthesize_from_maps(target, **settings)
    expected = fixture["losses"]
    n_iter, n_evals = (int(n) for n in fixture["counts"])
    assert (result.n_iter, result.n_evals) == (n_iter, n_evals), (
        f"ESTEL made {result.n_iter} iterations and {result.n_evals} evaluations of the loss, "
        f"STL {n_iter} and {n_evals}: the settings of M-SYN-LBFGS differ."
    )
    gap = np.abs(result.losses - expected) / np.abs(expected)
    worst = int(np.argmax(gap))
    assert np.all(gap <= RTOL), (
        f"ESTEL's loss departs from STL's beyond rtol {RTOL:g}: worst at iteration {worst + 1}, "
        f"ESTEL {result.losses[worst]!r}, STL {expected[worst]!r}, relative gap "
        f"{gap[worst]:.3g}; gaps per iteration {np.array2string(gap, precision=2)}. Do not "
        f"regenerate the fixture or loosen the tolerance: report it on issue {issue}."
    )
    assert result.manifest["L"] == p["L"]
    # synthesis_hpx records prefilter_Nyquist=False (OP-77); the 2D fixtures ran with STL's
    # default, True, which their manifest entries do not record.
    assert result.manifest["prefilter_nyquist"] == p.get("prefilter_Nyquist", True)
    field = backend.to_numpy(result.field)
    if healpix:
        # A DataHealpix target keeps its channel axis, (1, Npix); STL, given an array, returns
        # (Npix,) (M-SYN-OUT, OP-75; decision 8 on #169).
        field = field[0]
    assert_parity(field, fixture["field"], "float64")


def test_operators_and_start_with_the_spectrum(
    backend: Backend, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The case of ``synthesis_ps_2d`` before L-BFGS (T4.5; M-SYN-OPS, M-SYN-LOSS): both
    operators have J = 4, 9 bins and ``fft``; the target's reference is the fixture's
    ``var_ref``, ``s2_ref`` and ``ps_ref``; and the loss at the pre-filtered start, the first
    evaluation of L-BFGS, is STL's. Without L-BFGS, on every backend."""
    monkeypatch.setattr(_config, "_dtype", "float64")
    fixture = load_fixture("synthesis_ps_2d")
    J, J_PS, n_bins, compute_ps = (int(n) for n in fixture["settings"])  # noqa: N806
    assert compute_ps == 1
    target = Data2D(backend.asarray(fixture["target"], "float64"), pbc=True)
    setup = _maps_setup(backend, target, (1, 1, *target.shape[2:]), True, True, {"J": J})
    for operator in (setup.target_op, setup.running_op):
        spectrum = cast(STPlanSingleDomain, operator.plan).spectrum
        assert spectrum is not None
        assert operator.J == J
        assert (spectrum.J, spectrum.n_bins, spectrum.Jmin, spectrum.method) == (
            J_PS,
            n_bins,
            0,
            "fft",
        )
    reference = setup.objective.reference
    assert_parity(backend.to_numpy(reference.var), fixture["var_ref"], "float64")
    assert_parity(backend.to_numpy(reference.s2_sqrt), fixture["s2_ref"], "float64")
    assert_parity(backend.to_numpy(reference.ps_sqrt), fixture["ps_ref"], "float64")
    start = nyquist_filter(backend, backend.asarray(fixture["start"], "float64"))
    first = float(backend.to_numpy(setup.objective(start)))
    expected = float(fixture["evaluations"][0])
    assert abs(first - expected) <= RTOL * abs(expected), (
        f"the loss at the start is {first!r}, STL's first evaluation {expected!r}: report it on "
        "issue #152."
    )
