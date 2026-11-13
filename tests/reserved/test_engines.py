"""The reserved engine interface: registry, protocol and capabilities (Part II, §9.4; T1.4).

In 1.0.0 the only engine is ``"python"``, of differentiation order 2. These tests exercise the
trivial implementation so that the interface cannot rot before step 2 uses it.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from estel.engines import _registry
from estel.engines._protocol import (
    DEVICE_KINDS,
    DTYPES,
    GEOMETRIES,
    Engine,
    EngineCapabilities,
    EngineUnsupportedError,
    ForwardResult,
)


def test_python_is_the_only_engine_and_the_default() -> None:
    assert _registry.engine_names() == ("python",)
    assert _registry.DEFAULT_ENGINE == "python"
    assert _registry.resolve("python") is _registry.PYTHON


def test_python_engine_declares_order_two_and_everything_estel_defines() -> None:
    caps = _registry.resolve("python")
    assert caps.name == "python"
    assert caps.diff_order == 2
    assert caps.geometries == GEOMETRIES == {"1d", "2d", "healpix", "tensor_prod"}
    assert caps.dtypes == DTYPES == {"float32", "float64"}
    assert caps.devices == DEVICE_KINDS == {"cpu", "gpu"}
    for geometry in GEOMETRIES:
        for dtype in DTYPES:
            for device in DEVICE_KINDS:
                for order in (0, 1, 2):
                    caps.check(geometry=geometry, dtype=dtype, device=device, order=order)
    caps.check()


def test_capabilities_are_frozen_and_hashable() -> None:
    caps = _registry.resolve("python")
    assert hash(caps) == hash(dataclasses.replace(caps))
    with pytest.raises(dataclasses.FrozenInstanceError):
        caps.diff_order = 1  # type: ignore[misc]


@pytest.mark.parametrize("name", ["auto", "rust"])
def test_step_two_names_are_reserved_with_an_actionable_error(name: str) -> None:
    with pytest.raises(EngineUnsupportedError, match=rf"engine='{name}' is not available") as err:
        _registry.resolve(name)
    assert 'Use engine="python"' in str(err.value)


@pytest.mark.parametrize("name", ["Python", "numpy", "", None, 2])
def test_unknown_engine_names_say_which_names_are_accepted(name: object) -> None:
    with pytest.raises(EngineUnsupportedError, match="is not an engine of ESTEL") as err:
        _registry.resolve(name)
    assert "'python'" in str(err.value)


def test_engine_unsupported_error_is_a_value_error() -> None:
    assert issubclass(EngineUnsupportedError, ValueError)


@pytest.mark.parametrize(
    ("request_", "option"),
    [
        ({"geometry": "cubed_sphere"}, "geometry='cubed_sphere'"),
        ({"dtype": "float16"}, "dtype='float16'"),
        ({"device": "tpu"}, "device='tpu'"),
    ],
)
def test_an_unsupported_option_is_named(request_: dict[str, Any], option: str) -> None:
    with pytest.raises(EngineUnsupportedError, match=option):
        _registry.resolve("python").check(**request_)


def test_python_engine_refuses_order_three() -> None:
    with pytest.raises(EngineUnsupportedError, match="differentiates to order 2, and order 3"):
        _registry.resolve("python").check(order=3)


# An order-1 engine, as the Rust engine of step 2 will be: never registered, it exercises the
# protocol only.
_ORDER_ONE = EngineCapabilities(
    name="order_one",
    diff_order=1,
    geometries=frozenset({"healpix"}),
    dtypes=frozenset({"float64"}),
    devices=frozenset({"cpu"}),
)


class _IdentityEngine:
    """The smallest engine: statistics equal to the input, VJP the identity."""

    @property
    def capabilities(self) -> EngineCapabilities:
        return _ORDER_ONE

    def forward(self, x: Any, plan: Any, arrays: Any, ref: Any = None) -> ForwardResult:
        return ForwardResult(values=x, residual=(ref is not None,))

    def vjp(self, residual: Any, cotangent: Any) -> tuple[Any, Any]:
        (has_ref,) = residual
        return cotangent, (0.0 if has_ref else None)


def test_an_order_one_engine_points_to_the_python_engine() -> None:
    _ORDER_ONE.check(order=1)
    with pytest.raises(EngineUnsupportedError, match='Use engine="python"'):
        _ORDER_ONE.check(order=2)
    with pytest.raises(EngineUnsupportedError, match='Use engine="python"'):
        _ORDER_ONE.check(geometry="2d")


def test_forward_returns_values_and_a_residual_that_the_vjp_takes_back() -> None:
    engine: Engine = _IdentityEngine()
    values, residual = engine.forward(3.0, plan=None, arrays=None)
    assert values == 3.0
    assert engine.vjp(residual, 1.0) == (1.0, None)
    _, residual = engine.forward(3.0, plan=None, arrays=None, ref=2.0)
    assert engine.vjp(residual, 1.0) == (1.0, 0.0)


def test_order_one_engine_is_not_registered() -> None:
    with pytest.raises(EngineUnsupportedError):
        _registry.resolve(_ORDER_ONE.name)
