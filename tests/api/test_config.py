"""The backend registry and the global configuration (task T1.5, issue #39).

Public: ``set_backend``, ``get_backend``, ``set_default_dtype``, ``set_default_device`` and
``BackendMismatchError`` (Part II, §12.2). ``"jax"`` is selectable as an experimental backend,
with a warning; ``set_default_dtype("float64")`` switches JAX's 64-bit mode on, and refuses
once JAX ran in float32 (Part I, §3.1, rule 7); mixing backends raises ``BackendMismatchError``
(Part I, §2.2.3).

The tests in this process restore the registry after each test, and never switch JAX's 64-bit
mode: what does is tested in fresh processes, in the ``serial`` pass.
"""

from __future__ import annotations

import os
import subprocess
import sys
import warnings
from collections.abc import Callable
from types import ModuleType
from typing import Any

import numpy as np
import pytest

import estel
from estel import _config
from estel.backend import _registry
from estel.backend._registry import BackendMismatchError, backend_of, check_backend

PUBLIC = (
    "set_backend",
    "get_backend",
    "set_default_dtype",
    "set_default_device",
    "BackendMismatchError",
)


@pytest.fixture(autouse=True)
def _restore_the_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test starts from the run's backend and the defaults, and leaves them so."""
    monkeypatch.setattr(_registry, "_current", _registry._current)
    monkeypatch.setattr(_registry, "_instances", dict(_registry._instances))
    monkeypatch.setattr(_registry, "_float64", False)
    monkeypatch.setattr(_registry, "_device", None)
    monkeypatch.setattr(_config, "_dtype", _config.DEFAULT_DTYPE)


def _other(name: str) -> str:
    return next(n for n in _registry.BACKENDS if n != name)


def _module(name: str) -> ModuleType:
    return _registry._module(name)


def _fake_devices(monkeypatch: pytest.MonkeyPatch, name: str, *devices: str) -> None:
    monkeypatch.setattr(_module(name), "available_devices", lambda: devices)


def _select(name: str) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        estel.set_backend(name)


# ------------------------------------------------------------------ public names


@pytest.mark.parametrize("name", PUBLIC)
def test_public_names_are_exported(name: str) -> None:
    assert name in estel.__all__
    assert getattr(estel, name) is getattr(_registry if hasattr(_registry, name) else _config, name)


def test_backend_mismatch_error_is_a_type_error() -> None:
    assert issubclass(estel.BackendMismatchError, TypeError)


# ------------------------------------------------------------------ set_backend, get_backend


def test_torch_is_selected_without_a_warning() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        estel.set_backend("torch")
    assert estel.get_backend() == "torch"
    assert _registry.current_backend().name == "torch"


def test_jax_is_selectable_with_a_warning_at_every_call() -> None:
    for _ in range(2):
        with pytest.warns(UserWarning, match="experimental") as record:
            estel.set_backend("jax")
        assert len(record) == 1
        assert record[0].filename == __file__, "the warning points at the caller"
        assert "not covered by the guarantees of ESTEL 1.0.0" in str(record[0].message)
    assert estel.get_backend() == "jax"
    assert _registry.current_backend().name == "jax"


def test_get_backend_returns_the_name() -> None:
    for name in _registry.BACKENDS:
        _select(name)
        assert estel.get_backend() == name
        assert isinstance(estel.get_backend(), str)


def test_the_backend_object_is_made_once() -> None:
    assert _registry.current_backend() is _registry.current_backend()


@pytest.mark.parametrize("name", ["numpy", "Torch", "rust", "", None, 3])
def test_other_names_are_refused(name: Any) -> None:
    before = estel.get_backend()
    with pytest.raises(ValueError, match=r"Use one of 'torch', 'jax'"):
        estel.set_backend(name)
    assert estel.get_backend() == before


# ------------------------------------------------------------------ set_default_dtype


def test_float32_is_the_default_dtype() -> None:
    assert _config.DEFAULT_DTYPE == "float32"
    assert _config.default_dtype() == "float32"


def test_set_default_dtype(dtype: str) -> None:
    estel.set_default_dtype(dtype)  # type: ignore[arg-type]
    assert _config.default_dtype() == dtype
    estel.set_default_dtype("float32")
    assert _config.default_dtype() == "float32"


@pytest.mark.parametrize("value", ["float16", "complex64", "Float64", "float", np.float64, None])
def test_other_dtypes_are_refused(value: Any) -> None:
    with pytest.raises(ValueError, match="'float32' \\(the default\\) or 'float64'"):
        estel.set_default_dtype(value)
    assert _config.default_dtype() == "float32"
    assert not _registry._float64


def test_float64_reaches_the_current_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    current = estel.get_backend()
    monkeypatch.setattr(_module(current), "enable_float64", lambda: calls.append(current))
    estel.set_default_dtype("float64")
    assert calls == [current]


def test_float64_reaches_a_backend_selected_later(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    for name in _registry.BACKENDS:
        monkeypatch.setattr(_module(name), "enable_float64", lambda n=name: calls.append(n))
    current = estel.get_backend()
    estel.set_default_dtype("float64")
    _select(_other(current))
    assert calls == [current, _other(current)]


def _refuse(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    def refuse() -> None:
        raise RuntimeError("JAX already started in 32-bit mode")

    monkeypatch.setattr(_module(name), "enable_float64", refuse)


def test_a_refused_float64_changes_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    _refuse(monkeypatch, estel.get_backend())
    with pytest.raises(RuntimeError, match="32-bit"):
        estel.set_default_dtype("float64")
    assert _config.default_dtype() == "float32"
    assert not _registry._float64


def test_a_backend_refusing_float64_is_not_selected(monkeypatch: pytest.MonkeyPatch) -> None:
    current = estel.get_backend()
    monkeypatch.setattr(_module(current), "enable_float64", lambda: None)
    _refuse(monkeypatch, _other(current))
    estel.set_default_dtype("float64")
    with pytest.raises(RuntimeError, match="32-bit"):
        _select(_other(current))
    assert estel.get_backend() == current


# ------------------------------------------------------------------ set_default_device


def test_the_starting_device_follows_stl(monkeypatch: pytest.MonkeyPatch) -> None:
    """STL: CUDA if available, else CPU."""
    current = estel.get_backend()
    has_cuda = any(d.startswith("cuda") for d in _module(current).available_devices())
    assert _config.default_device() == ("cuda" if has_cuda else "cpu")
    _fake_devices(monkeypatch, current, "cpu")
    assert _config.default_device() == "cpu"
    _fake_devices(monkeypatch, current, "cpu", "cuda:0")
    assert _config.default_device() == "cuda"


def test_cpu_is_always_available() -> None:
    estel.set_default_device("cpu")
    assert _config.default_device() == "cpu"


def test_cuda_devices_are_checked(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_devices(monkeypatch, estel.get_backend(), "cpu", "cuda:0", "cuda:1")
    for device in ("cuda", "cuda:0", "cuda:1"):
        estel.set_default_device(device)
        assert _config.default_device() == device
    with pytest.raises(ValueError, match=r"'cuda:2' is not available .* \['cpu', 'cuda:0'"):
        estel.set_default_device("cuda:2")
    assert _config.default_device() == "cuda:1"


def test_cuda_without_a_gpu_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_devices(monkeypatch, estel.get_backend(), "cpu")
    for device in ("cuda", "cuda:0"):
        with pytest.raises(ValueError, match="set_default_device\\('cpu'\\)"):
            estel.set_default_device(device)
    assert _registry._device is None


def test_a_real_missing_gpu_is_refused() -> None:
    with pytest.raises(ValueError, match="is not available on the"):
        estel.set_default_device("cuda:99")


@pytest.mark.parametrize("device", ["gpu", "gpu:0", "CPU", "cuda:", "cuda:x", "mps", "", 0, None])
def test_malformed_devices_are_refused(device: Any) -> None:
    with pytest.raises(ValueError, match="'cpu', 'cuda' or 'cuda:N'"):
        estel.set_default_device(device)
    assert _registry._device is None


def test_set_backend_checks_the_device(monkeypatch: pytest.MonkeyPatch) -> None:
    current = estel.get_backend()
    _fake_devices(monkeypatch, current, "cpu", "cuda:0")
    _fake_devices(monkeypatch, _other(current), "cpu")
    estel.set_default_device("cuda:0")
    with pytest.raises(ValueError, match=f"not available on the {_other(current)} backend"):
        _select(_other(current))
    assert estel.get_backend() == current
    estel.set_default_device("cpu")
    _select(_other(current))
    assert estel.get_backend() == _other(current)


# ------------------------------------------------------------------ mixing backends


def _torch_array() -> Any:
    import torch

    return torch.zeros(2)


def _jax_array() -> Any:
    import jax.numpy as jnp

    return jnp.zeros(2)


MAKERS: dict[str, Callable[[], Any]] = {"torch": _torch_array, "jax": _jax_array}


def test_backend_of() -> None:
    assert backend_of(_torch_array()) == "torch"
    assert backend_of(_jax_array()) == "jax"
    for other in (np.zeros(2), 1.0, [1.0], None, "x"):
        assert backend_of(other) is None


def test_backend_of_a_jax_tracer() -> None:
    import jax

    seen: list[str | None] = []

    def f(a: Any) -> Any:
        seen.append(backend_of(a))
        return a

    jax.jit(f)(_jax_array())
    assert seen == ["jax"]


@pytest.mark.parametrize("expected", sorted(MAKERS))
def test_arrays_of_the_same_backend_pass(expected: str) -> None:
    check_backend(expected, MAKERS[expected](), np.zeros(2), 1.0, owner="STOperator")


@pytest.mark.parametrize("expected", sorted(MAKERS))
def test_mixing_backends_raises(expected: str) -> None:
    found = _other(expected)
    with pytest.raises(BackendMismatchError) as caught:
        check_backend(expected, np.zeros(2), MAKERS[found](), owner="STOperator")
    message = str(caught.value)
    assert f"array 1 passed to STOperator is a {found} array" in message
    assert f"built with the {expected} backend" in message
    assert f"estel.set_backend({found!r})" in message
    assert _registry._TO_NUMPY[found] in message
    with pytest.raises(TypeError, match="never converts between backends silently"):
        check_backend(expected, MAKERS[found]())


# ------------------------------------------------------------------ fresh processes


def _run(code: str, **env: str) -> subprocess.CompletedProcess[str]:
    base = {k: v for k, v in os.environ.items() if k != "JAX_ENABLE_X64"}
    return subprocess.run(
        [sys.executable, "-c", "import warnings\nwarnings.simplefilter('ignore')\n" + code],
        env={**base, **env},
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.serial
class TestFreshProcess:
    """Defaults, imports and JAX's 64-bit mode, which only a fresh process shows."""

    def test_defaults_and_no_backend_import(self) -> None:
        result = _run(
            "import sys, estel\n"
            "from estel import _config\n"
            "from estel.backend._registry import backend_of\n"
            "backend_of(1.0)\n"
            "print(estel.get_backend(), _config.default_dtype())\n"
            "print(sorted(m for m in ('torch', 'jax') if m in sys.modules))\n"
            "import torch\n"
            "print(_config.default_device() == ('cuda' if torch.cuda.is_available() else 'cpu'))\n"
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.splitlines() == ["torch float32", "[]", "True"]

    @pytest.mark.parametrize("order", ["dtype first", "backend first"])
    def test_float64_switches_jax_x64_on(self, order: str) -> None:
        steps = ["estel.set_default_dtype('float64')", "estel.set_backend('jax')"]
        if order == "backend first":
            steps.reverse()
        result = _run(
            "import estel, jax, numpy as np\n"
            + "\n".join(steps)
            + "\nfrom estel.backend._registry import current_backend\n"
            "x = current_backend().asarray(np.ones(2), 'float64')\n"
            "print(jax.config.read('jax_enable_x64'), x.dtype)\n"
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.split() == ["True", "float64"]

    def test_float64_after_jax_ran_in_float32_raises(self) -> None:
        result = _run(
            "import estel, jax, jax.numpy as jnp\n"
            "from estel import _config\n"
            "estel.set_backend('jax')\n"
            "jnp.zeros(2)\n"
            "try:\n"
            "    estel.set_default_dtype('float64')\n"
            "except RuntimeError as error:\n"
            "    print(error)\n"
            "print(_config.default_dtype(), jax.config.read('jax_enable_x64'))\n"
        )
        assert result.returncode == 0, result.stderr
        message, state = result.stdout.splitlines()
        assert "before the first JAX array is created" in message
        assert "JAX_ENABLE_X64=1" in message
        assert state == "float32 False"

    def test_jax_started_in_float32_is_not_selected_for_float64(self) -> None:
        result = _run(
            "import estel, jax.numpy as jnp\n"
            "estel.set_default_dtype('float64')\n"  # torch: nothing to switch
            "jnp.zeros(2)\n"
            "try:\n"
            "    estel.set_backend('jax')\n"
            "except RuntimeError as error:\n"
            "    print('RuntimeError')\n"
            "print(estel.get_backend())\n"
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.split() == ["RuntimeError", "torch"]

    def test_x64_already_on_is_accepted_late(self) -> None:
        result = _run(
            "import estel, jax.numpy as jnp\n"
            "estel.set_backend('jax')\n"
            "jnp.zeros(2)\n"
            "estel.set_default_dtype('float64')\n"
            "print('ok')\n",
            JAX_ENABLE_X64="1",
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.split() == ["ok"]

    def test_jax_not_installed(self) -> None:
        result = _run(
            "import sys\n"
            "sys.modules['jax'] = None\n"
            "import estel\n"
            "try:\n"
            "    estel.set_backend('jax')\n"
            "except ImportError as error:\n"
            "    print(error)\n"
            "print(estel.get_backend())\n"
        )
        assert result.returncode == 0, result.stderr
        message, backend = result.stdout.splitlines()
        assert "pip install 'estel[jax]'" in message
        assert backend == "torch"


# ------------------------------------------------------------------ pytrees (task T2.9)


class _RecordingBackend:
    """Stands for a backend: records the classes registered with it, or fails as told."""

    def __init__(self, error: Exception | None = None) -> None:
        self.registered: list[type] = []
        self.error = error

    def register_pytree(self, cls: type, flatten: Any, unflatten: Any) -> None:
        if self.error is not None:
            raise self.error
        self.registered.append(cls)


@pytest.fixture
def _fresh_pytrees(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_registry, "_pytrees", [])
    monkeypatch.setattr(_registry, "_pytrees_done", {})


def _flatten(obj: Any) -> tuple[tuple[()], None]:
    return (), None


def _unflatten(aux: Any, children: Any) -> None:
    return None


@pytest.mark.usefixtures("_fresh_pytrees")
def test_register_pytree_reaches_loaded_and_later_backends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded, later = _RecordingBackend(), _RecordingBackend()
    monkeypatch.setattr(_registry, "_instances", {"torch": loaded})
    _registry.register_pytree(int, _flatten, _unflatten)
    assert loaded.registered == [int]
    fake = ModuleType("fake_backend")
    fake.JaxBackend = lambda: later  # type: ignore[attr-defined]
    monkeypatch.setattr(_registry, "_module", lambda name: fake)
    loaded_later: object = _registry._instance("jax")
    assert loaded_later is later
    assert later.registered == [int]


@pytest.mark.usefixtures("_fresh_pytrees")
def test_register_pytree_registers_once_per_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Registration with a library is process-wide: a new instance of a backend skips it."""
    first, second = _RecordingBackend(), _RecordingBackend()
    monkeypatch.setattr(_registry, "_instances", {"torch": first})
    _registry.register_pytree(int, _flatten, _unflatten)
    fake = ModuleType("fake_backend")
    fake.TorchBackend = lambda: second  # type: ignore[attr-defined]
    monkeypatch.setattr(_registry, "_module", lambda name: fake)
    monkeypatch.setattr(_registry, "_instances", {})
    _registry._instance("torch")
    assert (first.registered, second.registered) == ([int], [])


@pytest.mark.usefixtures("_fresh_pytrees")
def test_register_pytree_warns_when_torch_cannot(monkeypatch: pytest.MonkeyPatch) -> None:
    broken = _RecordingBackend(RuntimeError("no torch pytrees: install a supported PyTorch"))
    monkeypatch.setattr(_registry, "_instances", {"torch": broken})
    with pytest.warns(RuntimeWarning, match="install a supported PyTorch"):
        _registry.register_pytree(int, _flatten, _unflatten)


@pytest.mark.usefixtures("_fresh_pytrees")
def test_register_pytree_raises_when_jax_cannot(monkeypatch: pytest.MonkeyPatch) -> None:
    broken = _RecordingBackend(RuntimeError("jax failed"))
    monkeypatch.setattr(_registry, "_instances", {"jax": broken})
    with pytest.raises(RuntimeError, match="jax failed"):
        _registry.register_pytree(int, _flatten, _unflatten)
