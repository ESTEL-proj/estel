"""Small helpers of the statistics on backend arrays (tasks T2.5, T2.15): static NumPy tables as
backend constants, backends, devices and dtype names. Written on the L0 protocol."""

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
