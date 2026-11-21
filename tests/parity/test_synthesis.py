"""A periodic synthesis follows STL's loss over its first iterations (Part II, §14.3, "Syntheses";
task T6.1, issue #28, decisions 2 and 3 of 21 November 2026; maths.md section 5).

The fixture ``synthesis_2d`` (STL `master`; ``tools/stl_parity/cases/synthesis_2d.py``) holds a
64 x 64 periodic target, J = 4, L = 4, one channel, the Gaussian start STL drew, and STL's loss
after each of its 20 iterations, run with STL's settings (M-SYN-LBFGS) and without the power
spectrum. ESTEL synthesises from the same target and the same start, with the same settings, in
float64, and must make as many iterations and evaluations of the loss, with the loss after each
iteration within a relative tolerance of STL's.

**Tolerance.** |ESTEL - STL| ≤ ``RTOL`` x |STL| at every iteration, with ``RTOL`` = 1e-10, the
float64 parity tolerance of Part II, §14.3, without an absolute term: the losses are positive
and fall by three orders of magnitude, so an absolute term scaled on the first would hide the
last. Measured on T6.1 (PR description): the relative gap is at most 1.0e-15 over these 20
iterations, from 0 at the first; on 60 iterations of the same case it grows to 6e-14. The
margin is five orders of magnitude, for the rounding of other machines and BLAS.

Slow: 20 iterations of L-BFGS on 64 x 64 pixels with J = 4, about 25 evaluations of the loss and
its gradient, take about a second; the comparison runs in the full tier (decision 2).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import estel
from estel import _config
from estel.backend._protocol import Backend
from tests.parity._compare import load_fixture, read_manifest

RTOL = 1e-10
"""Relative tolerance on the loss at each iteration: the float64 parity rtol of Part II, §14.3."""

pytestmark = [
    pytest.mark.slow,
    pytest.mark.maths(
        "M-SYN-TARGET",
        "M-SYN-INIT",
        "M-SYN-NYQ",
        "M-SYN-LOSS",
        "M-SYN-LBFGS",
        "M-SYN-OPS",
    ),
]


def test_synthesis_follows_stl(backend: Backend, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_config, "_dtype", "float64")
    fixture = load_fixture("synthesis_2d")
    p: dict[str, Any] = read_manifest()["synthesis_2d.npz"]["parameters"]
    settings: dict[str, Any] = {
        "init": fixture["start"],
        "mean_field": p["mean_field"],
        "compute_PS": p["compute_PS"],
        "max_iter": p["max_iter"],
        "history_size": p["history_size"],
        "lr": p["lr"],
        "st_kwargs": {"J": p["J"], "L": p["L"]},
    }
    if "lbfgs" in getattr(backend, "not_provided", frozenset()):
        with pytest.raises(NotImplementedError, match="torch backend"):
            estel.synthesize_from_maps(fixture["target"], **settings)
        return
    result = estel.synthesize_from_maps(fixture["target"], **settings)
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
        "regenerate the fixture or loosen the tolerance: report it on issue #28."
    )
