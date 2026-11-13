"""What the conformance suite cannot say without naming PyTorch (task T1.2).

The backend-neutral checks are in ``test_conformance.py``; these pin the PyTorch choices of
Part II, §12.3: ``lbfgs`` is ``torch.optim.LBFGS(line_search_fn="strong_wolfe")``,
``deterministic`` is ``torch.use_deterministic_algorithms``, keys leave the global random state
alone, and nothing returned shares memory with an input.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from estel.backend._protocol import Backend, DType
from estel.backend.torch._primitives import TorchBackend


def test_satisfies_the_protocol() -> None:
    backend: Backend = TorchBackend()  # checked by mypy
    assert backend.name == "torch"


@pytest.mark.parametrize("max_iter", [1, 2, 3, 7, 25])
@pytest.mark.parametrize("seed", range(5))
def test_lbfgs_is_one_torch_step(max_iter: int, seed: int, dtype: DType) -> None:
    """Same iterates, bit for bit, and as many iterations as a single ``LBFGS.step``."""
    torch_dtype = {"float32": torch.float32, "float64": torch.float64}[dtype]
    generator = torch.Generator().manual_seed(seed)
    a = torch.randn(12, 6, generator=generator, dtype=torch_dtype)
    b = torch.randn(12, generator=generator, dtype=torch_dtype)

    def fun(x: torch.Tensor) -> torch.Tensor:
        return (torch.sin(a @ x - b) ** 2).sum() + 0.1 * (x**4).sum()

    settings: dict[str, Any] = {
        "history_size": 4,
        "tolerance_grad": 1e-9,
        "tolerance_change": 1e-11,
    }
    x = torch.zeros(6, dtype=torch_dtype, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [x], lr=1, max_iter=max_iter, line_search_fn="strong_wolfe", **settings
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = fun(x)
        loss.backward()  # type: ignore[no-untyped-call]
        return loss

    optimizer.step(closure)  # type: ignore[no-untyped-call]
    result = TorchBackend().lbfgs(
        fun, torch.zeros(6, dtype=torch_dtype), max_iter=max_iter, **settings
    )
    assert torch.equal(result.x, x.detach())
    assert result.losses.shape == (optimizer.state[x]["n_iter"],)


@pytest.mark.serial
def test_deterministic_sets_the_torch_flag() -> None:
    before = torch.are_deterministic_algorithms_enabled()
    backend = TorchBackend()
    try:
        backend.deterministic(True)
        assert torch.are_deterministic_algorithms_enabled()
        backend.deterministic(False)
        assert not torch.are_deterministic_algorithms_enabled()
    finally:
        torch.use_deterministic_algorithms(before)


def test_random_normal_leaves_the_global_state_alone(dtype: DType) -> None:
    backend = TorchBackend()
    state = torch.get_rng_state()
    backend.random_normal(backend.make_key(3), (100,), dtype)
    assert torch.equal(state, torch.get_rng_state())


def test_outputs_share_no_memory_with_inputs(dtype: DType) -> None:
    backend = TorchBackend()
    x = backend.asarray(np.arange(6.0).reshape(2, 3), dtype).requires_grad_(True)
    stopped = backend.stop_gradient(x)
    assert not stopped.requires_grad
    assert stopped.data_ptr() != x.data_ptr()
    padded = backend.pad(x, [(0, 0)], [1], "circular")
    assert padded.data_ptr() != x.data_ptr()
    pairs = backend.complex_to_real_pairs(backend.real_pairs_to_complex(x[:, :2]))
    assert pairs.data_ptr() != x.data_ptr()
    array = backend.to_numpy(x)
    np.testing.assert_array_equal(array, np.arange(6.0).reshape(2, 3).astype(dtype))


def test_asarray_uses_the_default_device(dtype: DType) -> None:
    x = TorchBackend().asarray(np.ones(3), dtype)
    assert x.device == torch.get_default_device()
    assert x.dtype == {"float32": torch.float32, "float64": torch.float64}[dtype]
