"""``jax.test_util.check_grads`` of the scattering statistics, to second order in forward and
reverse mode (task T2.7, issue #19; Part I, §3.2.2).

``check_grads(order=2, modes=("fwd", "rev"))`` compares the JVP and the VJP of
``estel.functional.scattering`` (every coefficient that exists,
``tests/differentiation/_cases.py``), and those of their own derivatives, with central finite
differences along random directions, in float64. Collected only on JAX; ``test_torch.py`` is
the PyTorch counterpart.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import pytest
from jax.test_util import check_grads

from estel.backend._protocol import Backend
from tests.differentiation._cases import CASES, Case, scattering_function


@pytest.mark.slow
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_check_grads_to_second_order(case: Case, backend: Backend) -> None:
    """Slow: second-order derivatives through the whole loop, in both modes, on a backend that
    compiles each operation on first use (Part I, §5.4)."""
    data, function = scattering_function(backend, case)

    compiled = jax.jit(function)  # eagerly, JAX compiles every operation of every order

    def of_any_array(array: Any) -> Any:
        return compiled(jnp.asarray(array))  # check_grads evaluates its differences on NumPy

    check_grads(of_any_array, (data.array,), order=2, modes=("fwd", "rev"))  # type: ignore[no-untyped-call]
