"""The engine protocol, reserved in 1.0.0 (Part I, §2.2.4 and §3.2.4; Part II, §9.4).

An engine runs the whole computation of the statistics: a forward pass that returns the
values and an opaque residual, and a VJP that takes the residual back. Nothing relies on state
hidden inside an engine. Every engine declares what it computes (:class:`EngineCapabilities`),
including its differentiation order, and a request outside its capabilities raises
:class:`EngineUnsupportedError` naming the option.

In 1.0.0 only the Python engine exists (L2 + L1 + L0, ``diff_order=2``); the Rust engine of
step 2 (Part III, §18) implements the same protocol. This module and ``_registry.py`` are
reserved: keep them minimal, and change them only through a major version or the step-2 work
that uses them (``src/estel/engines/AGENTS.md``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, NamedTuple, Protocol, TypeAlias

Statistics: TypeAlias = Any
"""What the forward pass returns: the statistics (``STStatistics`` from T2.5)."""

Plan: TypeAlias = Any
"""The static plan of an operator (``ScatteringPlan`` from T2.3)."""

Residual: TypeAlias = Any
"""Opaque data the forward pass hands to the VJP. Only the engine that made it reads it."""

Geometry = Literal["1d", "2d", "healpix", "tensor_prod"]
"""The geometries of ESTEL 1.0.0 (Part II, §9.3)."""

FloatDType = Literal["float32", "float64"]

DeviceKind = Literal["cpu", "gpu"]

GEOMETRIES: frozenset[Geometry] = frozenset({"1d", "2d", "healpix", "tensor_prod"})
DTYPES: frozenset[FloatDType] = frozenset({"float32", "float64"})
DEVICE_KINDS: frozenset[DeviceKind] = frozenset({"cpu", "gpu"})


class EngineUnsupportedError(ValueError):
    """An engine was asked for something it does not compute, or does not exist.

    The message names the engine and the unsupported option, and says which engine to use
    instead (golden rule 8).
    """


@dataclass(frozen=True)
class EngineCapabilities:
    """What an engine computes, declared when it is registered (Part I, §2.2.4).

    Frozen and hashable. Fields are added when the options they describe exist (for example
    the operator options of T2.4); removing one is a major change.
    """

    name: str
    """The engine's name, as accepted by ``engine=``."""
    diff_order: int
    """The highest order of differentiation the engine supports (Part I, §3.2.4)."""
    geometries: frozenset[Geometry]
    dtypes: frozenset[FloatDType]
    devices: frozenset[DeviceKind]

    def check(
        self,
        *,
        geometry: str | None = None,
        dtype: str | None = None,
        device: str | None = None,
        order: int | None = None,
    ) -> None:
        """Raise :class:`EngineUnsupportedError` if a request is outside these capabilities.

        Arguments left to ``None`` are not checked. ``order`` is the differentiation order a
        caller needs: 1 for gradients, 2 for JVPs of gradients, Hessian-vector products and
        double backward.
        """
        requested = (
            ("geometry", geometry, self.geometries),
            ("dtype", dtype, self.dtypes),
            ("device", device, self.devices),
        )
        for option, value, supported in requested:
            if value is not None and value not in supported:
                raise EngineUnsupportedError(
                    f"engine={self.name!r} does not support {option}={value!r}; it supports "
                    f"{option} in {sorted(supported)}.{self._fallback()}"
                )
        if order is not None and not 0 <= order <= self.diff_order:
            raise EngineUnsupportedError(
                f"engine={self.name!r} differentiates to order {self.diff_order}, and order "
                f"{order} was requested.{self._fallback()}"
            )

    def _fallback(self) -> str:
        if self.name == "python":
            return " No engine of ESTEL supports this request."
        return ' Use engine="python", which covers every option ESTEL defines, to order 2.'


class ForwardResult(NamedTuple):
    """What :meth:`Engine.forward` returns."""

    values: Statistics
    """The statistics."""
    residual: Residual
    """Opaque data for :meth:`Engine.vjp`."""


class Engine(Protocol):
    """An engine: the whole computation of the statistics, and its gradient.

    The Python engine's ``forward`` is ``estel.functional.scattering`` (T2.5; until then, the
    internal loop of T2.4, ``estel.scattering._loop.scattering``). An engine of order 1 is
    wrapped with ``jax.custom_vjp`` or a ``torch.autograd.Function`` marked
    ``once_differentiable``; an engine of order 2 also provides a JVP and a differentiable VJP
    (Part I, §3.2.4).
    """

    @property
    def capabilities(self) -> EngineCapabilities:
        """What this engine computes."""
        ...

    def forward(self, x: Any, plan: Plan, arrays: Any, ref: Any = None) -> ForwardResult:
        """Compute the statistics of ``x`` with ``plan``; ``ref`` is the normalisation
        reference, if any (``apply(data, ref)``).
        """
        ...

    def vjp(self, residual: Residual, cotangent: Statistics) -> tuple[Any, Any]:
        """The cotangents of ``(x, ref)`` for a cotangent of the statistics; the second is
        ``None`` when ``forward`` was called without ``ref``.
        """
        ...
