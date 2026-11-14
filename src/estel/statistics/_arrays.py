"""Small helpers of the statistics on backend arrays (task T2.5): static masks as backend
constants, NaN fills and dtype names. Written on the L0 protocol."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..backend import _registry
from ..backend._protocol import Backend, DType


def backend_of(x: Any) -> Backend:
    """The backend of the array ``x``."""
    name = _registry.backend_of(x)
    if name is None:
        raise TypeError(f"expected an array of a backend of ESTEL, got {type(x).__name__}.")
    return _registry.backend_named(name)


def device_of(x: Any) -> Any:
    """The device of ``x``, or ``None`` (the backend's default device) when the array does not
    say, as inside a JAX trace."""
    try:
        return getattr(x, "device", None)
    except Exception:  # a tracer may refuse to name its device
        return None


def is_complex(backend: Backend, x: Any) -> bool:
    return bool(backend.xp.isdtype(x.dtype, "complex floating"))


def dtype_name(x: Any) -> DType:
    """The dtype of a backend array as the protocol names it (``"float32"``, ``"complex128"``)."""
    name = str(x.dtype).split(".")[-1]
    if name not in ("float32", "float64", "complex64", "complex128"):
        raise TypeError(f"expected a floating array, got dtype {x.dtype}.")
    return name  # type: ignore[return-value]


def constant(backend: Backend, array: NDArray[Any], like: Any, dtype: DType | None = None) -> Any:
    """The static NumPy ``array`` as a constant of ``like``'s backend and device; booleans stay
    booleans unless ``dtype`` is given."""
    kind: DType = dtype if dtype is not None else "bool"
    return backend.asarray(np.asarray(array), kind, device_of(like))


def nan_like(backend: Backend, x: Any) -> Any:
    """An array like ``x``, NaN everywhere (``nan + nan·j`` for a complex ``x``, as STL)."""
    fill: complex | float = complex(np.nan, np.nan) if is_complex(backend, x) else float("nan")
    return backend.xp.full_like(x, fill)


def masked(backend: Backend, present: NDArray[np.bool_], x: Any, other: Any) -> Any:
    """``x`` where ``present`` (static, broadcast against the axes of ``x`` after the batch
    axis), ``other`` elsewhere."""
    mask = constant(backend, present[None], x)
    return backend.xp.where(mask, x, other)


def zero_absent(backend: Backend, present: NDArray[np.bool_], x: Any) -> Any:
    """``x`` with 0 where no coefficient exists: finite everywhere, so that nothing absent
    reaches a product or a gradient."""
    return masked(backend, present, x, backend.xp.zeros_like(x))


def nan_absent(backend: Backend, present: NDArray[np.bool_], x: Any) -> Any:
    """``x`` with NaN where no coefficient exists, STL's full grid."""
    return masked(backend, present, x, nan_like(backend, x))
