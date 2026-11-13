"""What the conformance suite cannot say without naming JAX (task T1.3, issue #10).

The backend-neutral checks are in ``test_conformance.py``; these pin the JAX choices: ``jit``
is ``jax.jit`` and every primitive traces under it (Part I, §3.1); 64-bit dtypes need JAX's
64-bit mode, which the backend never switches itself; ``deterministic`` sets
``--xla_gpu_deterministic_ops`` in ``XLA_FLAGS`` before JAX starts, and refuses a change it
could no longer make; keys work for every seed, whatever the 64-bit mode; ``register_pytree``
registers with ``jax.tree_util``, so that ``jax.vmap`` maps over the children (task T2.9).

Collected only when ``ESTEL_TEST_BACKEND=jax``, which switches the 64-bit mode on at start.
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Any

import jax
import numpy as np
import pytest

from estel.backend._protocol import Backend, DType
from estel.backend.jax._primitives import JaxBackend
from tests.backend.test_conformance import (
    LINEAR_CASES,
    Linear,
    assert_close,
    draw,
    register_pair,
)


def test_satisfies_the_protocol() -> None:
    backend: Backend = JaxBackend()  # checked by mypy
    assert backend.name == "jax"


def test_jit_is_jax_jit(dtype: DType) -> None:
    backend = JaxBackend()
    traces = []

    def f(a: Any) -> Any:
        traces.append(1)
        return backend.xp.sin(a)

    jitted = backend.jit(f)
    x = backend.asarray(np.ones(3), dtype)
    jitted(x)
    jitted(x)
    assert len(traces) == 1, "jit must compile once and reuse the trace"


@pytest.mark.parametrize("case", LINEAR_CASES, ids=lambda case: case.name)
def test_every_primitive_traces_under_jit(case: Linear, dtype: DType) -> None:
    """Part I, §3.1: no Python decision on array values, so jax.jit gives the eager values."""
    backend = JaxBackend()
    x = draw(case.shape, dtype, 1)
    jitted = backend.jit(lambda a: case.apply(backend, a, dtype))
    assert_close(backend, jitted(backend.asarray(x, dtype)), case.reference(x, dtype), dtype)


def test_jit_with_static_arguments_traces_a_composite(dtype: DType) -> None:
    """The §3.1.1 pattern: pad, correlate, masked mean, stack, scatter to the full grid."""
    backend = JaxBackend()
    xp = backend.xp
    x = backend.asarray(draw((2, 8, 8), dtype, 40), dtype)
    k = backend.asarray(draw((3, 3, 3), dtype, 41), dtype)
    w = backend.asarray((np.random.default_rng(42).random((8, 8)) > 0.3).astype(float), dtype)
    index = [backend.asarray(np.array([0, 1, 2]), "int64")]

    def statistics(a: Any, levels: int) -> Any:
        means = []
        for _ in range(levels):
            y = backend.conv_axes(backend.pad(a, [(1, 1), (1, 1)], [1, 2], "circular"), k, [1, 2])
            means.append(xp.sum(w * y, axis=(-2, -1)) / xp.sum(w))
        stacked = xp.stack(means, axis=-1)[..., 0, :]
        return backend.scatter_to_grid(stacked, index, (4,))

    eager = statistics(x, levels=3)
    traced = backend.jit(statistics, static_argnames=("levels",))(x, levels=3)
    assert_close(backend, traced, backend.to_numpy(eager).astype(np.float64), dtype)


class TestSixtyFourBitMode:
    """The backend never switches ``jax_enable_x64``; it says how to (issue #10)."""

    @pytest.mark.parametrize("wide", ["float64", "complex128", "int64"])
    def test_wide_dtypes_need_it(self, wide: DType) -> None:
        backend = JaxBackend()
        with jax.enable_x64(False), pytest.raises(TypeError, match="jax_enable_x64"):
            backend.asarray(np.ones(3, dtype=wide), wide)

    def test_random_float64_needs_it(self) -> None:
        backend = JaxBackend()
        with jax.enable_x64(False), pytest.raises(TypeError, match="JAX_ENABLE_X64=1"):
            backend.random_normal(backend.make_key(1), (3,), "float64")

    def test_float32_works_without_it(self) -> None:
        backend = JaxBackend()
        with jax.enable_x64(False):
            x = backend.asarray(np.ones((2, 3)), "float32")
            index = [backend.asarray(np.array([0, 2]), "int32")]
            grid = backend.scatter_to_grid(x[:, :2], index, (3,), fill=0.0)
            assert backend.to_numpy(grid).dtype == np.float32
            big = backend.random_normal(backend.make_key(2**63 - 1), (4,), "float32")
            assert backend.to_numpy(big).dtype == np.float32

    def test_the_backend_never_switches_it(self) -> None:
        with jax.enable_x64(False):
            backend = JaxBackend()
            backend.asarray(np.ones(3), "float32")
            assert not jax.config.read("jax_enable_x64")  # type: ignore[no-untyped-call]


def test_keys_use_all_64_bits_of_the_seed() -> None:
    backend = JaxBackend()
    low = backend.to_numpy(backend.random_normal(backend.make_key(5), (8,), "float32"))
    high = backend.to_numpy(backend.random_normal(backend.make_key(5 + 2**32), (8,), "float32"))
    assert not np.array_equal(low, high)


def test_lbfgs_names_its_step() -> None:
    backend = JaxBackend()
    assert backend.not_provided == frozenset({"lbfgs"})
    with pytest.raises(NotImplementedError, match="torch backend"):
        backend.lbfgs(
            lambda x: x,
            backend.asarray(np.zeros(2), "float32"),
            max_iter=1,
            history_size=1,
            tolerance_grad=0.0,
            tolerance_change=0.0,
        )


@pytest.mark.serial
class TestDeterministic:
    """``--xla_gpu_deterministic_ops`` in ``XLA_FLAGS``, read by XLA once, when JAX starts."""

    def _run(self, code: str, flags: str | None) -> subprocess.CompletedProcess[str]:
        env = {k: v for k, v in os.environ.items() if k != "XLA_FLAGS"}
        if flags is not None:
            env["XLA_FLAGS"] = flags
        prelude = (
            "import os\n"
            "from estel.backend.jax._primitives import JaxBackend\n"
            "backend = JaxBackend()\n"
        )
        return subprocess.run(
            [sys.executable, "-c", prelude + code],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_sets_and_removes_the_flag_before_jax_starts(self) -> None:
        result = self._run(
            "backend.deterministic(True)\n"
            "print(os.environ['XLA_FLAGS'])\n"
            "backend.deterministic(False)\n"
            "print(os.environ['XLA_FLAGS'])\n",
            "--xla_cpu_enable_fast_math=false",
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.splitlines() == [
            "--xla_cpu_enable_fast_math=false --xla_gpu_deterministic_ops=true",
            "--xla_cpu_enable_fast_math=false",
        ]

    def test_a_late_change_raises(self) -> None:
        result = self._run(
            "backend.asarray(__import__('numpy').ones(2), 'float32')\n"
            "backend.deterministic(False)\n"  # unchanged: accepted
            "backend.deterministic(True)\n",
            None,
        )
        assert result.returncode != 0
        assert "RuntimeError" in result.stderr
        assert "before the first JAX array is created" in result.stderr

    def test_an_unchanged_setting_is_accepted_late(self) -> None:
        result = self._run(
            "backend.asarray(__import__('numpy').ones(2), 'float32')\n"
            "backend.deterministic(True)\n",
            "--xla_gpu_deterministic_ops=true",
        )
        assert result.returncode == 0, result.stderr


@pytest.mark.serial
@pytest.mark.parametrize("missing", ["module", "function"])
def test_deterministic_without_the_private_jax_check(
    missing: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If JAX drops ``backends_are_initialized``, deterministic says so; the backend still works."""
    import jax._src.xla_bridge as xla_bridge

    if missing == "module":
        monkeypatch.setitem(sys.modules, "jax._src.xla_bridge", None)
    else:
        monkeypatch.delattr(xla_bridge, "backends_are_initialized")
    monkeypatch.delenv("XLA_FLAGS", raising=False)
    backend = JaxBackend()
    with pytest.raises(RuntimeError, match="could not check whether JAX has started"):
        backend.deterministic(True)
    assert "XLA_FLAGS" not in os.environ
    backend.deterministic(False)  # unchanged: needs no check
    assert backend.to_numpy(backend.asarray(np.ones(2), "float32")).sum() == 2.0


def test_asarray_takes_device_names(dtype: DType) -> None:
    backend = JaxBackend()
    x = backend.asarray(np.ones(3), dtype, device="cpu")
    assert x.devices() == {jax.devices("cpu")[0]}
    with pytest.raises(ValueError, match="not available to JAX"):
        backend.asarray(np.ones(3), dtype, device="cpu:99")


class TestRegisterPytree:
    """``register_pytree`` is ``jax.tree_util.register_pytree_node`` (task T2.9)."""

    def test_jax_tree_util_sees_the_children(self, dtype: DType) -> None:
        backend = JaxBackend()
        pair = register_pair(backend)
        a, b = backend.asarray(np.ones(3), dtype), backend.asarray(np.zeros(2), dtype)
        leaves, treedef = jax.tree_util.tree_flatten(pair(a, b, "label"))
        assert leaves == [a, b]
        rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
        assert type(rebuilt) is pair
        assert rebuilt.label == "label"

    def test_vmap_maps_over_the_children(self, dtype: DType) -> None:
        backend = JaxBackend()
        pair = register_pair(backend)
        a, b = draw((4, 3), dtype, 1), draw((4,), dtype, 2)
        out = jax.vmap(lambda p: pair(p.a * p.b, p.b, p.label))(
            pair(backend.asarray(a, dtype), backend.asarray(b, dtype), "label")
        )
        assert type(out) is pair
        assert out.label == "label"
        assert_close(backend, out.a, a * b[:, None], dtype)
