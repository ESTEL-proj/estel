"""The backend registry (Part I, §2.2.1 and §2.2.3; Part II, §12.2; issue #39).

Holds the global choices that depend on a backend: which backend is the default
(:func:`set_backend`, :func:`get_backend`), whether float64 was asked for, and the default
device. It also maps an array to its backend, so that mixing backends raises
:class:`BackendMismatchError`.

The public setters of dtype and device are in ``estel._config``, which keeps the default dtype
and checks arguments, then calls :func:`enable_float64` and :func:`set_device` here: switching
JAX's 64-bit mode on and checking that a device exists are backend matters, and only
``estel.backend`` may import a backend.

L1 and L2 reach the primitives through :func:`current_backend`; an operator records the
backend it was built with, and checks the arrays it receives with :func:`check_backend`.
"""

from __future__ import annotations

import importlib
import re
import sys
import warnings
from types import ModuleType

from ._protocol import Backend

BACKENDS = ("torch", "jax")
"""The names :func:`set_backend` accepts, the stable one first."""

DEFAULT_BACKEND = "torch"

# Name -> (module, class). A backend module also provides enable_float64() and
# available_devices(). Imported only when the backend is selected.
_MODULES = {
    "torch": ("estel.backend.torch._primitives", "TorchBackend"),
    "jax": ("estel.backend.jax._primitives", "JaxBackend"),
}

JAX_WARNING = (
    "The JAX backend of ESTEL is experimental: every L0 primitive is implemented and tested, "
    "but it is not covered by the guarantees of ESTEL 1.0.0, and syntheses (L-BFGS) need the "
    'torch backend until step 2. Call estel.set_backend("torch") for the stable backend.'
)

_DEVICE = re.compile(r"cpu|cuda(:\d+)?")

# How to turn an array of each backend into NumPy, for error messages.
_TO_NUMPY = {"torch": "x.detach().cpu().numpy()", "jax": "numpy.asarray(x)"}


class BackendMismatchError(TypeError):
    """Arrays of two backends were mixed (Part I, §2.2.3).

    Raised when an operator or a DataClass built with one backend receives an array of
    another: ESTEL never converts between backends silently. The message names both backends
    and says how to fix it.
    """


# ------------------------------------------------------------------ global state

_current: str = DEFAULT_BACKEND
_instances: dict[str, Backend] = {}
_float64: bool = False
"""``set_default_dtype("float64")`` was called: every backend selected gets 64-bit support."""
_device: str | None = None
"""The device passed to ``set_default_device``; ``None`` until then (STL's rule applies)."""


def _module(name: str) -> ModuleType:
    module_name, _ = _MODULES[name]
    try:
        return importlib.import_module(module_name)
    except ImportError as error:
        if name == "jax" and (error.name or "").split(".")[0] in ("jax", "jaxlib"):
            raise ImportError(
                'set_backend("jax") needs JAX, which is not installed. Install it with '
                "pip install 'estel[jax]', or keep the default torch backend."
            ) from error
        raise


def _instance(name: str) -> Backend:
    if name not in _instances:
        _, class_name = _MODULES[name]
        backend: Backend = getattr(_module(name), class_name)()
        _instances[name] = backend
    return _instances[name]


def set_backend(name: str) -> None:
    """Set the default backend: ``"torch"`` (the default) or, experimental, ``"jax"``.

    Operators record the backend when they are built; changing it later does not affect them.
    Selecting ``"jax"`` imports JAX (``pip install 'estel[jax]'``) and warns, at every call,
    that the JAX backend is experimental. If float64 was asked for, the backend gets it now;
    if a default device was set, it must exist on the backend. On any error, nothing changes.
    """
    global _current
    if not isinstance(name, str) or name not in _MODULES:
        accepted = ", ".join(repr(n) for n in BACKENDS)
        raise ValueError(
            f"set_backend({name!r}): ESTEL has no backend of that name. Use one of {accepted}; "
            f'"{DEFAULT_BACKEND}" is the default.'
        )
    module = _module(name)
    if _float64:
        module.enable_float64()
    if _device is not None:
        _check_device(name, module, _device, f"set_backend({name!r})")
    _instance(name)
    _current = name
    if name == "jax":
        warnings.warn(JAX_WARNING, UserWarning, stacklevel=2)


def get_backend() -> str:
    """The name of the default backend: ``"torch"`` or ``"jax"``."""
    return _current


def current_backend() -> Backend:
    """The primitives of the default backend, for L1 and L2. Internal: not in the manifest."""
    return _instance(_current)


# ------------------------------------------------------------------ dtype and device


def enable_float64() -> None:
    """Give the current backend, and every backend selected later, float64 support.

    Called by ``estel.set_default_dtype("float64")``. On JAX, it switches the process-wide
    ``jax_enable_x64`` on, and raises ``RuntimeError`` if JAX already started in 32-bit mode;
    the flag is then left unset, so nothing changes.
    """
    global _float64
    _module(_current).enable_float64()
    _float64 = True


def set_device(device: str) -> None:
    """Set the default device, checked against the current backend. Raises ``ValueError``
    for a malformed name or a device the backend does not have. Called by
    ``estel.set_default_device``."""
    global _device
    caller = f"set_default_device({device!r})"
    if not isinstance(device, str) or not _DEVICE.fullmatch(device):
        raise ValueError(
            f"{caller}: a device is named 'cpu', 'cuda' or 'cuda:N' (N the index of a CUDA "
            "device, from 0)."
        )
    _check_device(_current, _module(_current), device, caller)
    _device = device


def default_device() -> str:
    """The default device: the one set with ``set_default_device``, else, as in STL,
    ``"cuda"`` if the current backend has a CUDA device and ``"cpu"`` otherwise.

    On JAX, the first answer starts JAX's backends, after which float64 can no longer be
    switched on: set the dtype first.
    """
    if _device is not None:
        return _device
    devices = _module(_current).available_devices()
    return "cuda" if any(d.startswith("cuda") for d in devices) else "cpu"


def _check_device(name: str, module: ModuleType, device: str, caller: str) -> None:
    if device == "cpu":  # every backend has one; asking JAX for its devices would start it
        return
    available: tuple[str, ...] = module.available_devices()
    has_cuda = any(d.startswith("cuda") for d in available)
    if device in available or (device == "cuda" and has_cuda):
        return
    raise ValueError(
        f"{caller}: device {device!r} is not available on the {name} backend, whose devices "
        f"are {list(available)}. Pass one of them, for example "
        f"estel.set_default_device('cpu')."
    )


# ------------------------------------------------------------------ mixing backends


def backend_of(x: object) -> str | None:
    """The backend ``x`` belongs to: ``"torch"`` for a ``torch.Tensor``, ``"jax"`` for a
    ``jax.Array`` (tracers included), ``None`` for anything else, NumPy arrays included.

    Never imports a backend: an array of a backend that is not imported cannot exist.
    """
    torch = sys.modules.get("torch")
    if torch is not None and isinstance(x, torch.Tensor):
        return "torch"
    jax = sys.modules.get("jax")
    if jax is not None and isinstance(x, jax.Array):
        return "jax"
    return None


def check_backend(expected: str, *arrays: object, owner: str = "this operator") -> None:
    """Raise :class:`BackendMismatchError` if an array of ``arrays`` belongs to a backend
    other than ``expected``, the backend ``owner`` was built with. Arrays of no backend
    (NumPy, Python numbers) pass: they are converted at the boundary.
    """
    for position, x in enumerate(arrays):
        found = backend_of(x)
        if found is None or found == expected:
            continue
        which = "the array" if len(arrays) == 1 else f"array {position}"
        raise BackendMismatchError(
            f"{which} passed to {owner} is a {found} array, but {owner} was built with the "
            f"{expected} backend, and ESTEL never converts between backends silently. Either "
            f"build {owner} after estel.set_backend({found!r}), or convert the array to NumPy "
            f"first (for a {found} array x: {_TO_NUMPY[found]}) and pass that."
        )
