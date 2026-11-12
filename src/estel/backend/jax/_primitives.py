"""JAX implementation of the L0 primitives: the experimental backend (Part I, §2.2.5).

Every method follows the contract written in ``estel.backend._protocol``; read it there. This
module adds only what the contract leaves to the backend: how each primitive is computed with
JAX. JAX arrays are immutable, so every primitive is out of place by construction; every one
is differentiable to second order, in reverse and forward mode, and traceable by ``jax.jit``.

Two things differ from PyTorch, both decided by the maintainers (issue #10):

* **64-bit mode.** JAX stores float64, complex128 and int64 in 32 bits unless its process-wide
  ``jax_enable_x64`` flag is on. The backend never switches it: asking for a 64-bit dtype
  while it is off raises ``TypeError`` with the way to switch it on.
* **No L-BFGS before step 2.** :meth:`JaxBackend.lbfgs` raises ``NotImplementedError``; the
  backend declares it in ``JaxBackend.not_provided``, and the conformance suite checks that
  exact error. ``optax.lbfgs`` arrives in step 2 (LF1).
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from types import ModuleType
from typing import Any, ClassVar

import array_api_compat
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax._src import xla_bridge

from .._protocol import DType, LbfgsResult, PadMode, Primals

_DTYPES: dict[str, Any] = {
    "float32": jnp.float32,
    "float64": jnp.float64,
    "complex64": jnp.complex64,
    "complex128": jnp.complex128,
    "int32": jnp.int32,
    "int64": jnp.int64,
    "bool": jnp.bool_,
}
_WIDE = ("float64", "complex128", "int64")
_REAL_FLOATING = (np.dtype("float32"), np.dtype("float64"))
_COMPLEX = (np.dtype("complex64"), np.dtype("complex128"))
_INDEX = (np.dtype("int32"), np.dtype("int64"))

# NumPy dtype kinds, from narrowest to widest: asarray may widen the kind, never narrow it.
_KIND_RANK = {"b": 0, "u": 1, "i": 1, "f": 2, "c": 3}
_PAD_MODES = ("circular", "replicate", "zeros")

_DETERMINISTIC_FLAG = "--xla_gpu_deterministic_ops"

LBFGS_NOT_PROVIDED = (
    "L-BFGS on the JAX backend arrives in step 2 (LF1), with optax.lbfgs. Run syntheses with "
    "the torch backend for now."
)


@dataclass(frozen=True)
class JaxKey:
    """A random key of the JAX backend: the seed itself, immutable.

    The JAX key is built from it at each draw, so that seeds up to ``2**63`` work whether or
    not 64-bit mode is on.
    """

    seed: int


# ------------------------------------------------------------------ argument checks


def _array(x: Any, what: str) -> jax.Array:
    if not isinstance(x, jax.Array):
        raise TypeError(
            f"{what} must be a jax.Array, got {type(x).__name__}. Convert NumPy arrays with "
            "backend.asarray(np_array, dtype) first."
        )
    return x


def _real(value: Any, what: str) -> jax.Array:
    x = _array(value, what)
    if x.dtype in _COMPLEX:
        raise TypeError(
            f"{what} is {x.dtype}, but this primitive takes real floating arrays. Split complex "
            "values into real and imaginary parts in L1 and make one call for each "
            "(Part I, §3.1, rule 6), or convert with complex_to_real_pairs."
        )
    if x.dtype not in _REAL_FLOATING:
        raise TypeError(f"{what} is {x.dtype}: pass a float32 or float64 array.")
    return x


def _floating(value: Any, what: str) -> jax.Array:
    x = _array(value, what)
    if x.dtype not in _REAL_FLOATING + _COMPLEX:
        raise TypeError(
            f"{what} is {x.dtype}: pass a floating array (float32, float64, complex64 or "
            "complex128)."
        )
    return x


def _index(value: Any, what: str) -> jax.Array:
    x = _array(value, what)
    if x.dtype not in _INDEX:
        raise TypeError(f"{what} is {x.dtype}: index arrays must be int32 or int64.")
    return x


def _same_dtype(*named: tuple[str, jax.Array]) -> None:
    dtypes = {x.dtype for _, x in named}
    if len(dtypes) > 1:
        detail = ", ".join(f"{name} is {x.dtype}" for name, x in named)
        raise TypeError(
            f"{detail}: floating inputs of one call must share one dtype, and no primitive "
            "converts them silently. Cast them to the operator's dtype when the plan is built."
        )


def _axis(axis: int, ndim: int) -> int:
    if not isinstance(axis, int) or isinstance(axis, bool):
        raise TypeError(f"axis {axis!r} must be a Python int.")
    if not -ndim <= axis < ndim:
        raise ValueError(f"axis {axis} is out of range for an array with {ndim} dimensions.")
    return axis % ndim


def _axes(axes: Sequence[int], ndim: int) -> list[int]:
    normalised = [_axis(a, ndim) for a in axes]
    if len(set(normalised)) != len(normalised):
        raise ValueError(f"axes {list(axes)} repeat an axis: name each axis once.")
    return normalised


def _x64_enabled() -> bool:
    return bool(jax.config.read("jax_enable_x64"))  # type: ignore[no-untyped-call]


def _require_x64(dtype: str) -> None:
    if dtype in _WIDE and not _x64_enabled():
        raise TypeError(
            f"dtype {dtype} needs JAX's 64-bit mode, which is off: JAX would store the array "
            "in 32 bits. Switch it on at start-up, before creating any JAX array, with "
            "jax.config.update('jax_enable_x64', True) or the environment variable "
            "JAX_ENABLE_X64=1. The backend never switches it itself, since it changes the "
            "default dtypes of all JAX code in the process."
        )


def _device(device: Any) -> Any:
    """A JAX device from a device or a name such as ``"cpu"`` or ``"cuda:0"``; ``None`` stays."""
    if not isinstance(device, str):
        return device
    platform, _, number = device.partition(":")
    platform = {"cuda": "gpu"}.get(platform, platform)
    try:
        return jax.devices(platform)[int(number or 0)]
    except (RuntimeError, IndexError, ValueError) as error:
        raise ValueError(
            f"device {device!r} is not available to JAX; the available devices are "
            f"{jax.devices()}. Pass one of them, or None for the default device."
        ) from error


def _flag_value(flags: Sequence[str]) -> bool:
    """Whether ``XLA_FLAGS`` asks for deterministic GPU operations (the last setting wins)."""
    value = False
    for flag in flags:
        name, _, setting = flag.partition("=")
        if name == _DETERMINISTIC_FLAG:
            value = setting.lower() in ("", "true", "1")
    return value


# ------------------------------------------------------------------ the backend


class JaxBackend:
    """The L0 primitives on JAX. See ``estel.backend._protocol.Backend`` for every contract."""

    not_provided: ClassVar[frozenset[str]] = frozenset({"lbfgs"})
    """Primitives of the protocol this backend does not provide yet; they raise
    ``NotImplementedError``. Only ``lbfgs``, until step 2 (LF1)."""

    @property
    def name(self) -> str:
        return "jax"

    # ------------------------------------------------------------ namespace and conversion

    @property
    def xp(self) -> ModuleType:
        return array_api_compat.array_namespace(jnp.empty(0))

    def asarray(self, np_array: Any, dtype: DType, device: Any = None) -> jax.Array:
        if not isinstance(np_array, np.ndarray):
            raise TypeError(
                f"asarray takes a numpy.ndarray, got {type(np_array).__name__}. Arrays of a "
                "backend are converted at the DataClass boundary, not through asarray."
            )
        if dtype not in _DTYPES:
            raise ValueError(f"dtype {dtype!r} is not one of {sorted(_DTYPES)}.")
        target = np.dtype(dtype)
        source_rank = _KIND_RANK.get(np_array.dtype.kind)
        if source_rank is None:
            raise TypeError(
                f"asarray cannot convert a NumPy array of dtype {np_array.dtype}: pass a "
                "boolean, integer, floating or complex array."
            )
        if source_rank > _KIND_RANK[target.kind]:
            raise TypeError(
                f"asarray would cast {np_array.dtype} to {dtype}, which drops information "
                "silently (complex to real, floating to integer, anything to bool). Convert "
                "explicitly in NumPy first, if that is really intended."
            )
        _require_x64(dtype)
        copy = np.array(np_array, dtype=target, copy=True, order="C")
        result: jax.Array = jax.device_put(copy, _device(device))
        return result

    def to_numpy(self, x: jax.Array) -> Any:
        x = _array(x, "x")
        return np.array(x, copy=True)

    # ------------------------------------------------------------ filtering

    def pad(
        self, x: jax.Array, widths: Sequence[tuple[int, int]], axes: Sequence[int], mode: PadMode
    ) -> jax.Array:
        x = _floating(x, "x")
        if mode not in _PAD_MODES:
            raise ValueError(f"pad mode {mode!r} is not one of {_PAD_MODES}.")
        if len(widths) != len(axes):
            raise ValueError(
                f"pad got {len(widths)} widths for {len(axes)} axes: give one (before, after) "
                "pair per axis."
            )
        norm_axes = _axes(axes, x.ndim)
        for axis, (before, after) in zip(norm_axes, widths, strict=True):
            n = x.shape[axis]
            if before < 0 or after < 0:
                raise ValueError(f"pad widths ({before}, {after}) must be non-negative.")
            if mode != "zeros" and n == 0:
                raise ValueError(f"{mode} padding of axis {axis} needs a non-empty axis.")
            if mode == "circular" and max(before, after) > n:
                raise ValueError(
                    f"circular padding by ({before}, {after}) of axis {axis} of length {n} would "
                    "wrap more than once. Reject this configuration when the plan is built "
                    "(fewer scales, or a larger input)."
                )
        y = x
        for axis, (before, after) in zip(norm_axes, widths, strict=True):
            n = y.shape[axis]
            if mode == "zeros":
                full = [(0, 0)] * y.ndim
                full[axis] = (before, after)
                y = jnp.pad(y, full, mode="constant")
            else:
                positions = np.arange(-before, n + after)
                if mode == "circular":
                    positions = np.mod(positions, n)
                else:
                    positions = np.clip(positions, 0, n - 1)
                y = jnp.take(y, jnp.asarray(positions, dtype=jnp.int32), axis=axis)
        return y

    def conv_axes(
        self, x: jax.Array, k: jax.Array, axes: Sequence[int], stride: int | Sequence[int] = 1
    ) -> jax.Array:
        x = _real(x, "x")
        k = _real(k, "k")
        _same_dtype(("x", x), ("k", k))
        norm_axes = _axes(axes, x.ndim)
        d = len(norm_axes)
        if d == 0:
            raise ValueError("conv_axes needs at least one axis to filter along.")
        first = norm_axes[0]
        if norm_axes != list(range(first, first + d)):
            raise ValueError(
                f"conv_axes filters along contiguous, increasing axes; got {list(axes)}. Order "
                "the block's axes, or filter each block separately."
            )
        if k.ndim < d:
            raise ValueError(
                f"k has {k.ndim} dimensions, fewer than the {d} filtered axes: its shape is "
                "(*O, K_1, ..., K_d)."
            )
        strides = [stride] * d if isinstance(stride, int) else list(stride)
        if len(strides) != d:
            raise ValueError(f"give one stride, or one per filtered axis ({d}); got {strides}.")
        if any(s < 1 for s in strides):
            raise ValueError(f"strides must be at least 1; got {strides}.")
        sizes = list(x.shape[first : first + d])
        support = list(k.shape[k.ndim - d :])
        if any(n < kk for n, kk in zip(sizes, support, strict=True)):
            raise ValueError(
                f"the filtered axes have lengths {sizes}, smaller than the kernel support "
                f"{support}: pad x first, or use fewer scales."
            )
        out_axes = tuple(k.shape[: k.ndim - d])
        out = [(n - kk) // s + 1 for n, kk, s in zip(sizes, support, strides, strict=True)]
        pre = tuple(x.shape[:first])
        post = tuple(x.shape[first + d :])
        # Batch every other axis: (pre, N, post) -> (pre, post, N) -> (B, 1, N).
        moved = jnp.moveaxis(
            x, tuple(range(first + d, x.ndim)), tuple(range(first, first + len(post)))
        )
        batched = moved.reshape(-1, 1, *sizes)
        kernels = k.reshape(-1, 1, *support)
        # lax.conv is a cross-correlation, like torch's: the kernel is not flipped (OP-1).
        y = lax.conv_general_dilated(
            batched, kernels, strides, "VALID", precision=lax.Precision.HIGHEST
        )
        y = y.reshape(*pre, *post, *out_axes, *out)
        n_lead = len(pre) + len(post)
        return jnp.moveaxis(
            y,
            tuple(range(len(pre), n_lead)),
            tuple(range(y.ndim - len(post), y.ndim)),
        )

    # ------------------------------------------------------------ sparse operations

    def gather_weighted(
        self, x: jax.Array, axis: int, index: jax.Array, weight: jax.Array
    ) -> jax.Array:
        x = _real(x, "x")
        weight = _real(weight, "weight")
        index = _index(index, "index")
        _same_dtype(("x", x), ("weight", weight))
        if index.ndim != 3 or index.shape != weight.shape:
            raise ValueError(
                f"index {tuple(index.shape)} and weight {tuple(weight.shape)} must have one "
                "shape (P, Q, M): a padded CSR stencil."
            )
        ax = _axis(axis, x.ndim)
        p, q, m = index.shape
        post = x.shape[ax + 1 :]
        taken = jnp.take(x, index.reshape(-1), axis=ax)
        taken = taken.reshape(*x.shape[:ax], p, q, m, *post)
        w = weight.reshape(p, q, m, *([1] * len(post)))
        return jnp.sum(taken * w, axis=ax + 2)

    def segment_sum(
        self, x: jax.Array, axis: int, segment_ids: jax.Array, num_segments: int
    ) -> jax.Array:
        x = _floating(x, "x")
        segment_ids = _index(segment_ids, "segment_ids")
        if num_segments < 0:
            raise ValueError(f"num_segments must be non-negative; got {num_segments}.")
        ax = _axis(axis, x.ndim)
        if tuple(segment_ids.shape) != (x.shape[ax],):
            raise ValueError(
                f"segment_ids has shape {tuple(segment_ids.shape)}; it needs one id per entry "
                f"of axis {ax}, shape ({x.shape[ax]},)."
            )
        moved = jnp.moveaxis(x, ax, 0)
        zeros = jnp.zeros((num_segments, *moved.shape[1:]), dtype=x.dtype)
        return jnp.moveaxis(zeros.at[segment_ids].add(moved), 0, ax)

    def _flat_index(self, index: Sequence[Any], shape: Sequence[int], n: int | None) -> Any:
        coords = [_index(c, f"index[{i}]") for i, c in enumerate(index)]
        if len(coords) != len(shape):
            raise ValueError(
                f"index has {len(coords)} coordinate arrays for a grid of {len(shape)} axes: "
                "give one per grid axis."
            )
        lengths = {tuple(c.shape) for c in coords}
        if len(lengths) > 1 or any(len(c.shape) != 1 for c in coords):
            raise ValueError(
                f"the coordinate arrays have shapes {sorted(lengths)}; they must all be (n,)."
            )
        if n is not None and coords and tuple(coords[0].shape) != (n,):
            raise ValueError(
                f"the coordinate arrays have length {coords[0].shape[0]}, but values has {n} "
                "entries on its last axis."
            )
        # Row-major ravel, as numpy.ravel_multi_index; int64 when 64-bit mode is on.
        flat_dtype = jnp.int64 if _x64_enabled() else jnp.int32
        flat = coords[0].astype(flat_dtype)
        for c, size in zip(coords[1:], shape[1:], strict=True):
            flat = flat * size + c.astype(flat_dtype)
        return flat

    def scatter_to_grid(
        self,
        values: jax.Array,
        index: Sequence[Any],
        shape: Sequence[int],
        fill: float = float("nan"),
    ) -> jax.Array:
        values = _floating(values, "values")
        if values.ndim == 0:
            raise ValueError("values must have shape (*B, n); got a 0-d array.")
        shape = [int(s) for s in shape]
        n = values.shape[-1]
        if len(shape) == 0:
            raise ValueError("shape must name at least one grid axis.")
        flat = self._flat_index(index, shape, n)
        batch = values.shape[:-1]
        grid = jnp.full((*batch, math.prod(shape)), fill, dtype=values.dtype)
        return grid.at[..., flat].set(values).reshape(*batch, *shape)

    def gather_from_grid(self, grid: jax.Array, index: Sequence[Any]) -> jax.Array:
        grid = _floating(grid, "grid")
        d = len(index)
        if d == 0 or d > grid.ndim:
            raise ValueError(
                f"index has {d} coordinate arrays for a grid of {grid.ndim} dimensions: give "
                "one per grid axis, at least one."
            )
        shape = grid.shape[grid.ndim - d :]
        flat = self._flat_index(index, shape, None)
        batch = grid.shape[: grid.ndim - d]
        return jnp.take(grid.reshape(*batch, -1), flat, axis=len(batch))

    # ------------------------------------------------------------ complex values

    def complex_to_real_pairs(self, z: jax.Array) -> jax.Array:
        z = _array(z, "z")
        if z.dtype not in _COMPLEX:
            raise TypeError(f"z is {z.dtype}: complex_to_real_pairs takes complex64 or complex128.")
        return jnp.stack((jnp.real(z), jnp.imag(z)), axis=-1)

    def real_pairs_to_complex(self, r: jax.Array) -> jax.Array:
        r = _real(r, "r")
        if r.ndim == 0 or r.shape[-1] != 2:
            raise ValueError(
                f"r has shape {tuple(r.shape)}; its last axis must hold (real, imaginary) pairs, "
                "length 2."
            )
        return lax.complex(r[..., 0], r[..., 1])

    # ------------------------------------------------------------ differentiation

    def stop_gradient(self, x: jax.Array) -> jax.Array:
        return lax.stop_gradient(_array(x, "x"))

    def value_and_grad(self, fn: Callable[..., Any]) -> Callable[..., tuple[Any, Any]]:
        scalar_fn = _real_scalar(fn)

        def value_and_grad_fn(x: Any, *args: Any) -> tuple[Any, Any]:
            _real(x, "x")
            value, grad = jax.value_and_grad(scalar_fn)(x, *args)
            return value, grad

        return value_and_grad_fn

    def vjp(self, fn: Callable[..., Any], *primals: Any) -> tuple[Any, Callable[[Any], Primals]]:
        _real_primals(primals, "primals")
        out, vjp_fn = jax.vjp(_real_output(fn), *primals)
        return out, vjp_fn

    def jvp(self, fn: Callable[..., Any], primals: Primals, tangents: Primals) -> tuple[Any, Any]:
        primals, tangents = tuple(primals), tuple(tangents)
        if len(primals) != len(tangents):
            raise ValueError(
                f"jvp got {len(primals)} primals and {len(tangents)} tangents: give one tangent "
                "per primal, with its shape."
            )
        _real_primals(primals, "primals")
        _real_primals(tangents, "tangents")
        out, tangent_out = jax.jvp(_real_output(fn), primals, tangents)
        return out, tangent_out

    def hvp(self, fn: Callable[[Any], Any], x: jax.Array, v: Any) -> Any:
        _real(x, "x")
        _real(v, "v")
        # Forward over reverse: the JVP of the gradient.
        return jax.jvp(jax.grad(_real_scalar(fn)), (x,), (v,))[1]

    def jit(
        self, fn: Callable[..., Any], static_argnames: Sequence[str] = ()
    ) -> Callable[..., Any]:
        return jax.jit(fn, static_argnames=tuple(static_argnames))

    # ------------------------------------------------------------ randomness

    def make_key(self, seed: int) -> JaxKey:
        if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**63:
            raise ValueError(f"seed must be an int in [0, 2**63); got {seed!r}.")
        return JaxKey(seed)

    def random_normal(self, key: Any, shape: Sequence[int], dtype: DType) -> jax.Array:
        if not isinstance(key, JaxKey):
            raise TypeError(
                f"key is a {type(key).__name__}: make it with make_key(seed) on this backend."
            )
        if dtype not in ("float32", "float64"):
            raise TypeError(f"random_normal draws float32 or float64 samples; got {dtype!r}.")
        _require_x64(dtype)
        # The default (threefry) key holds two 32-bit words: the high and low halves of seed.
        data = np.array([key.seed >> 32, key.seed & 0xFFFFFFFF], dtype=np.uint32)
        jax_key = jax.random.wrap_key_data(jnp.asarray(data), impl="threefry2x32")
        return jax.random.normal(jax_key, tuple(shape), dtype=_DTYPES[dtype])

    # ------------------------------------------------------------ optimization

    def lbfgs(
        self,
        fun: Callable[[Any], Any],
        x0: Any,
        *,
        max_iter: int,
        history_size: int,
        tolerance_grad: float,
        tolerance_change: float,
        state: Any = None,
    ) -> LbfgsResult:
        raise NotImplementedError(LBFGS_NOT_PROVIDED)

    def deterministic(self, enabled: bool) -> None:
        """Set ``--xla_gpu_deterministic_ops`` in ``XLA_FLAGS``, before JAX starts.

        XLA reads ``XLA_FLAGS`` once, when JAX initialises its backends (at the first array).
        A call that would change the setting after that raises ``RuntimeError``, since it
        would have no effect; a call that leaves it unchanged is accepted. On CPU, XLA is
        deterministic anyway.
        """
        flags = os.environ.get("XLA_FLAGS", "").split()
        if _flag_value(flags) == bool(enabled):
            return
        if xla_bridge.backends_are_initialized():
            raise RuntimeError(
                f"deterministic({bool(enabled)}) came after JAX started its backends, and XLA "
                "reads XLA_FLAGS only once, so it would have no effect. Call it at set-up, "
                "before the first JAX array is created, or set "
                f"XLA_FLAGS={_DETERMINISTIC_FLAG}={str(bool(enabled)).lower()} in the "
                "environment before starting Python."
            )
        others = [f for f in flags if f.partition("=")[0] != _DETERMINISTIC_FLAG]
        if enabled:
            others.append(f"{_DETERMINISTIC_FLAG}=true")
        if others:
            os.environ["XLA_FLAGS"] = " ".join(others)
        else:
            os.environ.pop("XLA_FLAGS", None)


# ------------------------------------------------------------------ differentiation helpers


def _real_primals(arrays: Sequence[Any], what: str) -> None:
    named = [(f"{what}[{i}]", _real(a, f"{what}[{i}]")) for i, a in enumerate(arrays)]
    if named:
        _same_dtype(*named)


def _real_output(fn: Callable[..., Any]) -> Callable[..., Any]:
    def checked(*args: Any) -> Any:
        return _real(fn(*args), "the output of fn")

    return checked


def _real_scalar(fn: Callable[..., Any]) -> Callable[..., Any]:
    def checked(*args: Any) -> Any:
        value = _real(fn(*args), "the value of fn")
        if value.ndim != 0:
            raise ValueError(
                f"fn returns shape {tuple(value.shape)}; it must return a 0-d real array. Sum "
                "or reduce the output, or use vjp for a non-scalar function."
            )
        return value

    return checked
