"""Global defaults: dtype and device (Part I, §3.1, rule 7; Part II, §12.2; issue #39).

The backend choice lives in ``estel.backend._registry`` (``set_backend``, ``get_backend``),
with the parts of these defaults that depend on a backend: the switch to 64-bit and the check
that a device exists. Operators read the defaults when they are built, through
:func:`default_dtype` and :func:`default_device`, which are internal.
"""

from __future__ import annotations

from typing import Literal, TypeAlias

from .backend import _registry

FloatDType: TypeAlias = Literal["float32", "float64"]

DTYPES: tuple[FloatDType, ...] = ("float32", "float64")
DEFAULT_DTYPE: FloatDType = "float32"

_dtype: FloatDType = DEFAULT_DTYPE


def set_default_dtype(dtype: Literal["float32", "float64"]) -> None:
    """Set the default floating dtype of operators and data: ``"float32"`` (the default) or
    ``"float64"``.

    ``"float64"`` enables float64 on the current backend and on every backend selected later:
    on JAX it switches ``jax_enable_x64`` on, and raises ``RuntimeError`` if JAX already ran in
    float32 in this process (call it at set-up, or set ``JAX_ENABLE_X64=1``). Going back to
    ``"float32"`` leaves JAX's 64-bit mode on; every computation still runs in float32. On any
    error, nothing changes.
    """
    global _dtype
    if dtype not in DTYPES:
        raise ValueError(
            f"set_default_dtype({dtype!r}): ESTEL computes in 'float32' (the default) or "
            "'float64'; pass one of them."
        )
    if dtype == "float64":
        _registry.enable_float64()
    _dtype = dtype


def set_default_device(device: str) -> None:
    """Set the default device of operators and data: ``"cpu"``, ``"cuda"`` or ``"cuda:N"``.

    Until it is called, the default is the one of STL: ``"cuda"`` if the current backend has a
    CUDA device, ``"cpu"`` otherwise. The device is checked against the current backend now,
    and again by ``set_backend``; one that does not exist raises ``ValueError`` listing the
    devices available. ESTEL keeps this default itself and never changes the backend's own
    (``torch.set_default_device``, JAX's default device).
    """
    _registry.set_device(device)


def default_dtype() -> FloatDType:
    """The default floating dtype, read when an operator is built."""
    return _dtype


def default_device() -> str:
    """The default device, read when an operator is built (see :func:`set_default_device`)."""
    return _registry.default_device()
