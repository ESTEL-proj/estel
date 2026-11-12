"""PyTorch implementation of the L0 primitives (Part II, §12.3).

Every method follows the contract written in ``estel.backend._protocol``; read it there. This
module adds only what the contract leaves to the backend: how each primitive is computed with
PyTorch. Every primitive is out of place and differentiable to second order, in reverse mode
(autograd) and in forward mode (``torch.func.jvp``).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from types import ModuleType
from typing import Any

import array_api_compat
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

from .._protocol import DType, LbfgsResult, PadMode, Primals
from . import _optim

_DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float64": torch.float64,
    "complex64": torch.complex64,
    "complex128": torch.complex128,
    "int32": torch.int32,
    "int64": torch.int64,
    "bool": torch.bool,
}
_REAL_FLOATING = (torch.float32, torch.float64)
_COMPLEX = (torch.complex64, torch.complex128)
_INDEX = (torch.int32, torch.int64)
_TO_COMPLEX = {torch.float32: torch.complex64, torch.float64: torch.complex128}
_TO_REAL = {torch.complex64: torch.float32, torch.complex128: torch.float64}

# NumPy dtype kinds, from narrowest to widest: asarray may widen the kind, never narrow it.
_KIND_RANK = {"b": 0, "u": 1, "i": 1, "f": 2, "c": 3}
_PAD_MODES = ("circular", "replicate", "zeros")
_CONV = {1: F.conv1d, 2: F.conv2d, 3: F.conv3d}


@dataclass(frozen=True)
class TorchKey:
    """A random key of the PyTorch backend: the seed itself, immutable."""

    seed: int


# ------------------------------------------------------------------ argument checks


def _tensor(x: Any, what: str) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        raise TypeError(
            f"{what} must be a torch.Tensor, got {type(x).__name__}. Convert NumPy arrays with "
            "backend.asarray(np_array, dtype) first."
        )
    return x


def _real(value: Any, what: str) -> torch.Tensor:
    x = _tensor(value, what)
    if x.dtype in _COMPLEX:
        raise TypeError(
            f"{what} is {x.dtype}, but this primitive takes real floating arrays. Split complex "
            "values into real and imaginary parts in L1 and make one call for each "
            "(Part I, §3.1, rule 6), or convert with complex_to_real_pairs."
        )
    if x.dtype not in _REAL_FLOATING:
        raise TypeError(f"{what} is {x.dtype}: pass a float32 or float64 array.")
    return x


def _floating(value: Any, what: str) -> torch.Tensor:
    x = _tensor(value, what)
    if x.dtype not in _REAL_FLOATING + _COMPLEX:
        raise TypeError(
            f"{what} is {x.dtype}: pass a floating array (float32, float64, complex64 or "
            "complex128)."
        )
    return x


def _index(value: Any, what: str) -> torch.Tensor:
    x = _tensor(value, what)
    if x.dtype not in _INDEX:
        raise TypeError(f"{what} is {x.dtype}: index arrays must be int32 or int64.")
    return x


def _same_dtype(*named: tuple[str, torch.Tensor]) -> None:
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


# ------------------------------------------------------------------ the backend


class TorchBackend:
    """The L0 primitives on PyTorch. See ``estel.backend._protocol.Backend`` for every contract."""

    @property
    def name(self) -> str:
        return "torch"

    # ------------------------------------------------------------ namespace and conversion

    @property
    def xp(self) -> ModuleType:
        return array_api_compat.array_namespace(torch.empty(0))

    def asarray(self, np_array: Any, dtype: DType, device: Any = None) -> torch.Tensor:
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
        copy = np.array(np_array, dtype=target, copy=True, order="C")
        target_device = torch.get_default_device() if device is None else device
        return torch.from_numpy(copy).to(device=target_device)

    def to_numpy(self, x: torch.Tensor) -> Any:
        x = _tensor(x, "x")
        return np.array(x.detach().resolve_conj().resolve_neg().cpu().numpy(), copy=True)

    # ------------------------------------------------------------ filtering

    def pad(
        self, x: torch.Tensor, widths: Sequence[tuple[int, int]], axes: Sequence[int], mode: PadMode
    ) -> torch.Tensor:
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
                shape_b = (*y.shape[:axis], before, *y.shape[axis + 1 :])
                shape_a = (*y.shape[:axis], after, *y.shape[axis + 1 :])
                zeros_b = torch.zeros(shape_b, dtype=y.dtype, device=y.device)
                zeros_a = torch.zeros(shape_a, dtype=y.dtype, device=y.device)
                y = torch.cat((zeros_b, y, zeros_a), dim=axis)
            else:
                positions = np.arange(-before, n + after)
                if mode == "circular":
                    positions = np.mod(positions, n)
                else:
                    positions = np.clip(positions, 0, n - 1)
                index = torch.from_numpy(positions).to(device=y.device)
                y = torch.index_select(y, axis, index)
        return y

    def conv_axes(
        self, x: torch.Tensor, k: torch.Tensor, axes: Sequence[int], stride: int | Sequence[int] = 1
    ) -> torch.Tensor:
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
        out_axes = k.shape[: k.ndim - d]
        out = [(n - kk) // s + 1 for n, kk, s in zip(sizes, support, strides, strict=True)]
        pre = x.shape[:first]
        post = x.shape[first + d :]
        if d in _CONV:
            # Batch every other axis: (pre, N, post) -> (pre, post, N) -> (B, 1, N).
            moved = torch.movedim(
                x, tuple(range(first + d, x.ndim)), tuple(range(first, first + len(post)))
            )
            batched = moved.reshape(-1, 1, *sizes)
            kernels = k.reshape(-1, 1, *support)
            y = _CONV[d](batched, kernels, stride=strides)
            y = y.reshape(*pre, *post, *out_axes, *out)
            n_lead = len(pre) + len(post)
            return torch.movedim(
                y,
                tuple(range(len(pre), n_lead)),
                tuple(range(y.ndim - len(post), y.ndim)),
            )
        # More than three axes: windows by unfold, then a contraction with the kernels.
        windows = x
        for axis, kk, s in zip(norm_axes, support, strides, strict=True):
            windows = windows.unfold(axis, kk, s)
        y = torch.tensordot(
            windows,
            k,
            dims=(list(range(windows.ndim - d, windows.ndim)), list(range(k.ndim - d, k.ndim))),
        )
        # y is (pre, M, post, O): bring O just before the first filtered axis.
        return torch.movedim(
            y,
            tuple(range(y.ndim - len(out_axes), y.ndim)),
            tuple(range(first, first + len(out_axes))),
        )

    # ------------------------------------------------------------ sparse operations

    def gather_weighted(
        self, x: torch.Tensor, axis: int, index: torch.Tensor, weight: torch.Tensor
    ) -> torch.Tensor:
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
        taken = torch.index_select(x, ax, index.reshape(-1))
        taken = taken.reshape(*x.shape[:ax], p, q, m, *post)
        w = weight.reshape(p, q, m, *([1] * len(post)))
        return (taken * w).sum(dim=ax + 2)

    def segment_sum(
        self, x: torch.Tensor, axis: int, segment_ids: torch.Tensor, num_segments: int
    ) -> torch.Tensor:
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
        shape = (*x.shape[:ax], num_segments, *x.shape[ax + 1 :])
        zeros = torch.zeros(shape, dtype=x.dtype, device=x.device)
        return zeros.index_add(ax, segment_ids, x)

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
        # Row-major ravel, as numpy.ravel_multi_index.
        flat = coords[0].to(torch.int64)
        for c, size in zip(coords[1:], shape[1:], strict=True):
            flat = flat * size + c.to(torch.int64)
        return flat

    def scatter_to_grid(
        self,
        values: torch.Tensor,
        index: Sequence[Any],
        shape: Sequence[int],
        fill: float = float("nan"),
    ) -> torch.Tensor:
        values = _floating(values, "values")
        if values.ndim == 0:
            raise ValueError("values must have shape (*B, n); got a 0-d array.")
        shape = [int(s) for s in shape]
        n = values.shape[-1]
        if len(shape) == 0:
            raise ValueError("shape must name at least one grid axis.")
        flat = self._flat_index(index, shape, n).to(values.device)
        batch = values.shape[:-1]
        grid = torch.full(
            (*batch, math.prod(shape)), fill, dtype=values.dtype, device=values.device
        )
        return grid.index_copy(values.ndim - 1, flat, values).reshape(*batch, *shape)

    def gather_from_grid(self, grid: torch.Tensor, index: Sequence[Any]) -> torch.Tensor:
        grid = _floating(grid, "grid")
        d = len(index)
        if d == 0 or d > grid.ndim:
            raise ValueError(
                f"index has {d} coordinate arrays for a grid of {grid.ndim} dimensions: give "
                "one per grid axis, at least one."
            )
        shape = grid.shape[grid.ndim - d :]
        flat = self._flat_index(index, shape, None).to(grid.device)
        batch = grid.shape[: grid.ndim - d]
        return torch.index_select(grid.reshape(*batch, -1), len(batch), flat)

    # ------------------------------------------------------------ complex values

    def complex_to_real_pairs(self, z: torch.Tensor) -> torch.Tensor:
        z = _tensor(z, "z")
        if z.dtype not in _COMPLEX:
            raise TypeError(f"z is {z.dtype}: complex_to_real_pairs takes complex64 or complex128.")
        return torch.stack((z.real, z.imag), dim=-1)

    def real_pairs_to_complex(self, r: torch.Tensor) -> torch.Tensor:
        r = _real(r, "r")
        if r.ndim == 0 or r.shape[-1] != 2:
            raise ValueError(
                f"r has shape {tuple(r.shape)}; its last axis must hold (real, imaginary) pairs, "
                "length 2."
            )
        return torch.complex(r[..., 0], r[..., 1])

    # ------------------------------------------------------------ differentiation

    def stop_gradient(self, x: torch.Tensor) -> torch.Tensor:
        return _tensor(x, "x").detach().clone()

    def value_and_grad(self, fn: Callable[..., Any]) -> Callable[..., tuple[Any, Any]]:
        scalar_fn = _real_scalar(fn)

        def value_and_grad_fn(x: Any, *args: Any) -> tuple[Any, Any]:
            _real(x, "x")
            grad, value = torch.func.grad_and_value(scalar_fn)(x, *args)
            return value, grad

        return value_and_grad_fn

    def vjp(self, fn: Callable[..., Any], *primals: Any) -> tuple[Any, Callable[[Any], Primals]]:
        _real_primals(primals, "primals")
        out, vjp_fn = torch.func.vjp(_real_output(fn), *primals)[:2]
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
        out, tangent_out = torch.func.jvp(_real_output(fn), primals, tangents)[:2]
        return out, tangent_out

    def hvp(self, fn: Callable[[Any], Any], x: torch.Tensor, v: Any) -> Any:
        _real(x, "x")
        _real(v, "v")
        vjp_fn = torch.func.vjp(torch.func.grad(_real_scalar(fn)), x)[1]
        return vjp_fn(v)[0]

    def jit(
        self, fn: Callable[..., Any], static_argnames: Sequence[str] = ()
    ) -> Callable[..., Any]:
        return fn

    # ------------------------------------------------------------ randomness

    def make_key(self, seed: int) -> TorchKey:
        if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**63:
            raise ValueError(f"seed must be an int in [0, 2**63); got {seed!r}.")
        return TorchKey(seed)

    def random_normal(self, key: Any, shape: Sequence[int], dtype: DType) -> torch.Tensor:
        if not isinstance(key, TorchKey):
            raise TypeError(
                f"key is a {type(key).__name__}: make it with make_key(seed) on this backend."
            )
        if dtype not in ("float32", "float64"):
            raise TypeError(f"random_normal draws float32 or float64 samples; got {dtype!r}.")
        device = torch.get_default_device()
        generator = torch.Generator(device=device)
        generator.manual_seed(key.seed)
        return torch.randn(tuple(shape), generator=generator, dtype=_DTYPES[dtype], device=device)

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
        x0 = _real(x0, "x0")
        return _optim.lbfgs(
            _real_scalar(fun),
            x0,
            max_iter=max_iter,
            history_size=history_size,
            tolerance_grad=tolerance_grad,
            tolerance_change=tolerance_change,
            state=state,
        )

    def deterministic(self, enabled: bool) -> None:
        torch.use_deterministic_algorithms(bool(enabled))


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
