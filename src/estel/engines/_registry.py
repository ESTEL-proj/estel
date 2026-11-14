"""The engine registry, reserved in 1.0.0 (Part I, §2.2.4; Part II, §9.4).

Maps the values of ``engine=`` to the capabilities of each engine. In 1.0.0 the only engine is
``"python"``, of differentiation order 2; ``"auto"`` and ``"rust"`` are names of step 2
(Part III, §18) and raise :class:`EngineUnsupportedError`, as does any other name. The engine
objects themselves are connected when ``estel.functional.scattering`` becomes public (T2.5).
"""

from __future__ import annotations

from ._protocol import (
    DEVICE_KINDS,
    DTYPES,
    GEOMETRIES,
    EngineCapabilities,
    EngineUnsupportedError,
)

DEFAULT_ENGINE = "python"

PYTHON = EngineCapabilities(
    name="python",
    diff_order=2,
    geometries=GEOMETRIES,
    dtypes=DTYPES,
    devices=DEVICE_KINDS,
)
"""The Python engine (L2 + L1 + L0) covers everything ESTEL defines, to order 2."""

_ENGINES: dict[str, EngineCapabilities] = {PYTHON.name: PYTHON}

# Names reserved for step 2, with what to say when they are asked for in 1.x.
_RESERVED = {
    "auto": "it chooses between engines, and arrives with the Rust engine in step 2",
    "rust": "the Rust engine arrives in step 2 (Part III, §18)",
}


def engine_names() -> tuple[str, ...]:
    """The names ``engine=`` accepts, sorted."""
    return tuple(sorted(_ENGINES))


def resolve(name: object) -> EngineCapabilities:
    """The capabilities of the engine called ``name``, the value of an ``engine=`` argument.

    Raises :class:`EngineUnsupportedError` for any name not registered, saying which names
    are accepted.
    """
    if isinstance(name, str) and name in _ENGINES:
        return _ENGINES[name]
    accepted = ", ".join(repr(n) for n in engine_names())
    if isinstance(name, str) and name in _RESERVED:
        reason = _RESERVED[name]
        raise EngineUnsupportedError(
            f"engine={name!r} is not available in this version of ESTEL: {reason}. "
            f'Use engine="{DEFAULT_ENGINE}" (the default); accepted engines: {accepted}.'
        )
    raise EngineUnsupportedError(
        f"engine={name!r} is not an engine of ESTEL. Use one of {accepted}; "
        f'engine="{DEFAULT_ENGINE}" is the default.'
    )
