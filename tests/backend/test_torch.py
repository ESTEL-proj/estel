"""What the conformance suite cannot say without naming PyTorch (task T1.2).

The backend-neutral checks are in ``test_conformance.py``; these pin the PyTorch choices of
Part II, §12.3: ``lbfgs`` is ``torch.optim.LBFGS(line_search_fn="strong_wolfe")``,
``deterministic`` is ``torch.use_deterministic_algorithms``, keys leave the global random state
alone, and nothing returned shares memory with an input; ``register_pytree`` registers with
``torch.utils._pytree``, imported when it is called (task T2.9).
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
import pytest
import torch
import torch.utils._pytree as pytree

from estel.backend._protocol import Backend, DType
from estel.backend.torch._primitives import TorchBackend
from tests.backend.test_conformance import pair_class, register_pair


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


class TestRegisterPytree:
    """``register_pytree`` is ``torch.utils._pytree.register_pytree_node``, imported when it is
    called, with a fallback to the older name and a clear error without either (task T2.9)."""

    def test_torch_pytree_sees_the_children(self, dtype: DType) -> None:
        backend = TorchBackend()
        pair = register_pair(backend)
        a, b = backend.asarray(np.ones(3), dtype), backend.asarray(np.zeros(2), dtype)
        leaves, spec = pytree.tree_flatten(pair(a, b, "label"))
        assert len(leaves) == 2
        assert leaves[0] is a
        assert leaves[1] is b
        rebuilt = pytree.tree_unflatten(leaves, spec)
        assert type(rebuilt) is pair
        assert rebuilt.label == "label"
        doubled = pytree.tree_map(lambda x: 2 * x, pair(a, b, "label"))
        assert type(doubled) is pair
        assert torch.equal(doubled.a, 2 * a)

    def test_falls_back_to_the_older_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delattr(pytree, "register_pytree_node")
        backend = TorchBackend()
        pair = register_pair(backend)
        leaves, _ = pytree.tree_flatten(pair(torch.ones(2), torch.zeros(1), "label"))
        assert len(leaves) == 2

    @pytest.mark.parametrize("missing", ["module", "functions"])
    def test_without_the_private_module_says_what_to_do(
        self, missing: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        if missing == "module":
            monkeypatch.setitem(sys.modules, "torch.utils._pytree", None)
        else:
            monkeypatch.delattr(pytree, "register_pytree_node")
            monkeypatch.delattr(pytree, "_register_pytree_node")
        backend = TorchBackend()
        pair: Any = pair_class()
        with pytest.raises(RuntimeError, match="report this PyTorch version in an ESTEL issue"):
            backend.register_pytree(pair, lambda p: ((), None), lambda aux, ch: pair)
        x = backend.asarray(np.ones(2), "float32")  # the rest of the backend still works
        assert backend.to_numpy(backend.stop_gradient(x)).sum() == 2.0
