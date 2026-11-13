"""The mother DataClass (Part II, §12.4; Part I, §2.3; maths.md M-DATA-2D, M-DATA-1D).

A DataClass is one backend array plus static metadata: the periodicity ``pbc`` and the mask of
missing values. It is immutable: operations return new objects (:meth:`DataClass.with_array`),
never modify one. Subclasses (``Data1D``, ``Data2D`` in ``_planar.py``) set the number of spatial
axes and the geometry constants, and override only the geometry hooks (tag G of §12.4).

Validation happens once, here, when the object is built (review rule B1): the array is converted
to the backend, promoted to ``(Nb, Nc, *spatial)``, and its NaN become the mask. The mask is a
NumPy boolean array over the spatial axes (True = missing), shared by every batch element and
every channel (maths.md OP-26). It is static: the operator turns it into weights and
reweighting maps once, when it is built (``estel.data._masks``), and the computation never looks
for missing values again (Part I, §3.1, rule 3).

The values of the array at masked pixels are kept as given (NaN, if the mask came from them).
They are never read: the statistics use the mask as weights (M-DOWN-MASK, M-MEAN; OP-27).

Every DataClass is a pytree whose only leaf is the array (Part I, §2.3 and §3.1, rule 9; task
T2.9): each subclass is registered when it is defined, on every backend, so that ``jax.jit``
and ``jax.vmap`` accept a DataClass and return one.
"""

from __future__ import annotations

from typing import Any, ClassVar, NamedTuple

import numpy as np
from numpy.typing import NDArray

from .._config import default_device, default_dtype
from ..backend import _registry

_FLOAT_DTYPES = ("float32", "float64")


class DataClass:
    """Data plus static metadata: the mother class of every geometry.

    Build a geometry instead: ``Data1D`` or ``Data2D``. The mother class holds the shared
    logic; a subclass sets ``ndim``, the number of spatial axes, and overrides only the
    geometry hooks.

    Parameters
    ----------
    array:
        The data, real-valued: a NumPy array (or anything ``numpy.asarray`` accepts), converted
        to the current backend with the default dtype and device; or an array of the current
        backend, float32 or float64, kept as it is (so that gradients reach it). The accepted
        shapes are those of the geometry (``Data2D``: ``(N1, N2)``, ``(Nc, N1, N2)`` or
        ``(Nb, Nc, N1, N2)``); the array is promoted to ``(Nb, Nc, *spatial)``.
    mask:
        Optional boolean array of the spatial shape, True where a value is missing, shared by
        every batch element and every channel. If it is not given, it is read from the NaN of
        ``array``, which must then hold a single map (``Nb = Nc = 1``).
    pbc:
        Periodic boundary conditions, ``True`` or ``False``: required, with no default.
    """

    ndim: ClassVar[int]
    """The number of spatial axes of the geometry."""

    _nan_threshold: ClassVar[float]
    """τ of maths.md M-DOWN-MASK: a downsampled pixel is invalid when its fraction of invalid
    input exceeds it."""

    __slots__ = ("_array", "_backend", "_mask", "_pbc")

    _array: Any
    _backend: str
    _mask: NDArray[np.bool_] | None
    _pbc: bool

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Register every geometry as a pytree on every backend (task T2.9): registration is
        per exact class, so each subclass needs its own."""
        super().__init_subclass__(**kwargs)
        _registry.register_pytree(cls, cls._tree_flatten, cls._tree_unflatten)

    def __init__(self, array: Any, mask: Any = None, *, pbc: bool) -> None:
        cls = type(self)
        if not hasattr(cls, "ndim"):
            raise TypeError(
                f"{cls.__name__} is the mother class and has no geometry: build a geometry "
                "instead, estel.Data1D or estel.Data2D."
            )
        if not isinstance(pbc, bool | np.bool_):
            raise TypeError(
                f"{cls.__name__}(pbc={pbc!r}): pbc must be True (periodic) or False, written "
                "explicitly; it has no default (maths.md OP-4)."
            )
        backend_name = _registry.get_backend()
        values = _to_backend(array, backend_name, cls)
        values = _promote(values, cls)
        backend = _registry.current_backend()
        nan = np.asarray(backend.to_numpy(backend.xp.isnan(values)), dtype=bool)
        full_mask = _validate_mask(mask, nan, values.shape, backend, cls)
        self._set(values, full_mask, bool(pbc), backend_name)

    @classmethod
    def _accepted_shapes(cls) -> str:
        """The shapes the constructor accepts, for error messages."""
        spatial = ", ".join(f"N{i + 1}" for i in range(cls.ndim)) if cls.ndim > 1 else "N"
        return f"({spatial}), (Nc, {spatial}) or (Nb, Nc, {spatial})"

    # ------------------------------------------------------------------ construction

    def _set(self, array: Any, mask: NDArray[np.bool_] | None, pbc: bool, backend: str) -> None:
        object.__setattr__(self, "_array", array)
        object.__setattr__(self, "_mask", mask)
        object.__setattr__(self, "_pbc", pbc)
        object.__setattr__(self, "_backend", backend)

    @classmethod
    def _from_parts(
        cls, array: Any, mask: NDArray[np.bool_] | None, pbc: bool, backend: str
    ) -> DataClass:
        """A new object from parts already validated: no conversion, no check of values, so
        that it works inside traced code."""
        new = cls.__new__(cls)
        new._set(array, mask, pbc, backend)
        return new

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(
            f"{type(self).__name__} is immutable: build a new object instead, for example "
            "data.with_array(new_array) to replace the array."
        )

    def __delattr__(self, name: str) -> None:
        raise AttributeError(f"{type(self).__name__} is immutable: its attributes stay set.")

    # ------------------------------------------------------------------ accessors

    @property
    def array(self) -> Any:
        """The data, an array of the backend of shape ``(Nb, Nc, *spatial)``."""
        return self._array

    @property
    def mask(self) -> NDArray[np.bool_] | None:
        """The missing values, a read-only NumPy boolean array of the spatial shape (True =
        missing), or ``None`` when no value is missing."""
        return self._mask

    @property
    def pbc(self) -> bool:
        """Whether the boundary conditions are periodic."""
        return self._pbc

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape of the array, ``(Nb, Nc, *spatial)``."""
        return tuple(int(n) for n in self._array.shape)

    @property
    def n_batch(self) -> int:
        """Nb, the number of batch elements."""
        return self.shape[0]

    @property
    def n_channels(self) -> int:
        """Nc, the number of channels."""
        return self.shape[1]

    @property
    def backend(self) -> str:
        """The name of the backend of the array: ``"torch"`` or ``"jax"``."""
        return self._backend

    # ------------------------------------------------------------------ functional updates

    def with_array(self, new_array: Any) -> DataClass:
        """A new object with ``new_array`` in place of the array, and the same metadata.

        This replaces STL's in-place updates (``inplace=True``). ``new_array`` must be an array
        of the same backend with the shape of :attr:`shape`; it is used as it is, without a
        copy and without looking at its values, so this works inside traced and
        differentiated code.
        """
        _registry.check_backend(self._backend, new_array, owner=f"this {type(self).__name__}")
        if _registry.backend_of(new_array) is None:
            raise TypeError(
                f"with_array takes an array of the {self._backend} backend, got "
                f"{type(new_array).__name__}. To start from NumPy, build a new "
                f"{type(self).__name__}(array, mask=data.mask, pbc=data.pbc) instead."
            )
        new_shape = tuple(int(n) for n in new_array.shape)
        if new_shape != self.shape:
            raise ValueError(
                f"with_array: the new array has shape {new_shape}, but this "
                f"{type(self).__name__} has shape {self.shape}. Pass an array of the same "
                "shape, or build a new object."
            )
        return self._from_parts(new_array, self._mask, self._pbc, self._backend)

    def copy(self, empty: bool = False) -> DataClass:
        """A new object with a copy of the array, and the same metadata.

        With ``empty=True`` the array of the copy is filled with zeros (STL sets it to
        ``None``; an ESTEL object always holds an array of its shape).
        """
        xp = _registry.backend_named(self._backend).xp
        array = xp.zeros_like(self._array) if empty else xp.asarray(self._array, copy=True)
        return self._from_parts(array, self._mask, self._pbc, self._backend)

    def __getitem__(self, key: Any) -> DataClass:
        """Select batch elements and channels: ``data[b]``, ``data[b, c]``, with ``b`` and
        ``c`` ints or slices.

        The result keeps the axes ``(Nb, Nc)``: an int selects one element and keeps its axis
        (``data[0].shape == (1, Nc, *spatial)``). The spatial axes cannot be indexed, since the
        mask belongs to the whole spatial grid.
        """
        keys = key if isinstance(key, tuple) else (key,)
        if len(keys) > 2:
            raise IndexError(
                f"{type(self).__name__}[...] selects batch elements and channels only: use "
                "data[b] or data[b, c]. Slice data.array to work on spatial pixels."
            )
        slices = tuple(
            _axis_slice(k, n, name)
            for k, n, name in zip(keys, self.shape, ("batch", "channel"), strict=False)
        )
        array = self._array[slices]
        if 0 in tuple(int(n) for n in array.shape):
            raise IndexError(
                f"{type(self).__name__}[{key!r}] selects no element: the data have "
                f"{self.n_batch} batch element(s) and {self.n_channels} channel(s)."
            )
        return self._from_parts(array, self._mask, self._pbc, self._backend)

    # ------------------------------------------------------------------ pytree

    def _tree_flatten(self) -> tuple[tuple[Any], tuple[Any, ...]]:
        """The array is the only leaf; everything else is static (Part I, §2.3).

        The static part must be hashable and compare by value, since JAX caches compiled
        functions by it: the mask goes in as a :class:`_StaticMask`.
        """
        mask = None if self._mask is None else _StaticMask(self._mask)
        return (self._array,), (mask, self._pbc, self._backend)

    @classmethod
    def _tree_unflatten(cls, aux: tuple[Any, ...], children: tuple[Any, ...]) -> DataClass:
        """Rebuild from :meth:`_tree_flatten`, without looking at the leaf: the backend passes
        tracers, batched arrays and placeholders."""
        mask, pbc, backend = aux
        return cls._from_parts(children[0], None if mask is None else mask.mask, pbc, backend)

    # ------------------------------------------------------------------ geometry hooks

    def _spatial_axes(self) -> tuple[int, ...]:
        """The axes the geometry acts on: the last ``ndim`` axes of the array."""
        return tuple(range(2, 2 + self.ndim))

    def _conv(self, x: Any, kernels: Any, level: int) -> Any:
        """Filter ``x`` with the wavelet ``kernels`` at resolution ``level`` (maths.md M-W1).

        ``x`` is an array of this object's backend, ``(Nb, Nc, *spatial)`` at that level, real
        or complex; ``kernels`` the complex bank of the geometry, cast to the backend and to the
        complex dtype matching ``x``. The result is complex, with the orientation axis inserted
        before the spatial axes: ``(Nb, Nc, L, *spatial)``. Each geometry implements it
        (planar: ``pad`` + ``conv_axes``, M-CONV-2D).
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement the geometry hook _conv: every geometry "
            "overrides it (Part II, §12.4)."
        )

    def _downsample(self, x: Any, weights: DownsampleWeights, level: int) -> Any:
        """Downsample ``x`` by one step, from resolution ``level`` to ``level + 1`` (maths.md
        M-DOWN, M-DOWN-MASK).

        ``x`` is an array of this object's backend, ``(Nb, Nc, *spatial)`` at ``level``, real
        or complex; ``weights`` the static weights of the step (:class:`DownsampleWeights`).
        The result has the shape of the coarser level and the dtype of ``x``. Each geometry
        implements it (planar: a normalised strided correlation with the smoothing kernel).
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement the geometry hook _downsample: every "
            "geometry overrides it (Part II, §12.4)."
        )

    def _crop_weights(self, level: int, border: int) -> NDArray[np.float64]:
        """The border crop of maths.md M-BORDER at resolution ``level``, as weights.

        A NumPy float64 array of the spatial shape at ``level``: 1 where a pixel enters the
        means, 0 where the crop removes it, ``border`` pixels on each side of every spatial
        axis. The plan computes ``border`` from the convolutions applied (M-BORDER) and
        multiplies these weights by the valid pixels of the mask, once, when an operator is
        built (Part I, §3.1.1). Each geometry implements it (planar: STL's ``_crop``, with its
        reduction of a border too large for the array, OP-13; HEALPix: no crop).
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement the geometry hook _crop_weights: every "
            "geometry overrides it (Part II, §12.4)."
        )

    def __repr__(self) -> str:
        n_missing = 0 if self._mask is None else int(self._mask.sum())
        return (
            f"{type(self).__name__}(shape={self.shape}, pbc={self._pbc}, "
            f"missing={n_missing}, backend={self._backend!r})"
        )


# ---------------------------------------------------------------------- geometry hook inputs


class DownsampleWeights(NamedTuple):
    """The static weights of one downsampling step (:meth:`DataClass._downsample`), as arrays
    of the data's backend, built from the plan.

    ``wavelets`` and ``data`` are independent layers, so the kernel is passed in rather than
    imported: the operator casts ``smoothing_kernel(ndim)`` once and passes it at every step.
    """

    kernel: Any
    """The smoothing kernel of the geometry, real, with the real dtype of the data
    (planar: the smoothing kernel of maths.md M-SMOOTH-2D, M-SMOOTH-1D)."""
    valid: Any = None
    """Without missing values, ``None``. Otherwise the valid input pixels, a boolean array of
    the spatial shape at the input level: invalid pixels are set to 0 before filtering, never
    read (M-DOWN-MASK, OP-27)."""
    reweight: Any = None
    """Without missing values, ``None``. Otherwise the reweighting 1 / (1 - f) of the output
    pixels, of the spatial shape at the output level, 0 at invalid outputs (M-DOWN-MASK;
    ``MaskMaps.smooth_reweight`` and ``wav_reweight`` of ``_masks.py``)."""


# ---------------------------------------------------------------------- pytree metadata


class _StaticMask:
    """The mask of missing values as pytree metadata: hashable, and equal to another
    ``_StaticMask`` exactly when the two masks have the same shape and values.

    JAX hashes and compares the static part of a pytree to reuse compiled functions, and a
    NumPy array can be neither hashed nor compared to a single bool. Two DataClasses with equal
    masks then share one compiled function; a different mask compiles a new one, since the
    mask is static.

    The hash is computed once, here, and the wrapped mask is read-only, so that the hash can
    never go stale. A DataClass mask is already a read-only array that owns its data, and is
    kept as it is; any other mask (writable, or a view whose base may be written) is copied
    first.
    """

    __slots__ = ("_hash", "mask")

    def __init__(self, mask: NDArray[np.bool_]) -> None:
        if mask.flags.writeable or not mask.flags.owndata:
            mask = np.array(mask, dtype=bool, copy=True)
            mask.flags.writeable = False
        self.mask = mask
        self._hash = hash((mask.shape, np.packbits(mask).tobytes()))

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _StaticMask):
            return NotImplemented
        return self.mask is other.mask or (
            self.mask.shape == other.mask.shape and bool(np.array_equal(self.mask, other.mask))
        )

    def __repr__(self) -> str:
        return f"_StaticMask(shape={self.mask.shape}, missing={int(self.mask.sum())})"


# ---------------------------------------------------------------------- validation


def _to_backend(array: Any, backend_name: str, cls: type[DataClass]) -> Any:
    """``array`` as an array of the current backend, with a real floating dtype."""
    if array is None:
        raise TypeError(f"{cls.__name__} needs an array, got None.")
    _registry.check_backend(backend_name, array, owner=f"{cls.__name__}")
    backend = _registry.current_backend()
    if _registry.backend_of(array) is not None:
        xp = backend.xp
        dtype = next((name for name in _FLOAT_DTYPES if array.dtype == getattr(xp, name)), None)
        if dtype is None:
            raise TypeError(
                f"{cls.__name__}: the array has dtype {array.dtype}; ESTEL data are real, "
                "float32 or float64. Convert it first, for example to the default dtype, or "
                "pass a NumPy array."
            )
        return array
    values = np.asarray(array)
    if values.dtype.kind == "c":
        raise TypeError(
            f"{cls.__name__}: the array is complex ({values.dtype}); ESTEL data are real "
            "(maths.md M-DATA-2D). Pass the real and imaginary parts as two channels."
        )
    if values.dtype.kind not in "biuf":
        raise TypeError(
            f"{cls.__name__}: cannot read an array of dtype {values.dtype}; pass real numbers."
        )
    return backend.asarray(values.astype(np.float64), default_dtype(), default_device())


def _promote(array: Any, cls: type[DataClass]) -> Any:
    """Promote ``array`` to ``(Nb, Nc, *spatial)`` (maths.md M-DATA-2D, M-DATA-1D, OP-29)."""
    shape = tuple(int(n) for n in array.shape)
    accepted = cls._accepted_shapes()
    if not cls.ndim <= len(shape) <= cls.ndim + 2:
        raise ValueError(
            f"{cls.__name__}: the array has shape {shape}; accepted shapes are {accepted}."
        )
    if 0 in shape:
        raise ValueError(f"{cls.__name__}: the array has shape {shape}, with an empty axis.")
    full = (1,) * (cls.ndim + 2 - len(shape)) + shape
    if full == shape:
        return array
    return _registry.current_backend().xp.reshape(array, full)


def _validate_mask(
    mask: Any,
    nan: NDArray[np.bool_],
    shape: tuple[int, ...],
    backend: Any,
    cls: type[DataClass],
) -> NDArray[np.bool_] | None:
    """The mask of missing values, from ``mask`` or from the NaN of the array (OP-26)."""
    spatial = tuple(int(n) for n in shape[2:])
    if mask is None:
        if not nan.any():
            return None
        if shape[0] * shape[1] > 1:
            raise ValueError(
                f"{cls.__name__}: the array has NaN and {shape[0]} batch element(s) x "
                f"{shape[1]} channel(s). The mask of missing values is one map shared by every "
                "batch element and channel (maths.md OP-26), so it is read from the NaN only "
                f"for a single map. Pass it explicitly: mask=<boolean array of shape {spatial}>."
            )
        found = nan.reshape(spatial)
    else:
        if _registry.backend_of(mask) is not None:
            mask = backend.to_numpy(mask)
        found = np.asarray(mask)
        if found.dtype != np.bool_:
            raise TypeError(
                f"{cls.__name__}: the mask has dtype {found.dtype}; pass a boolean array, True "
                "where a value is missing."
            )
        if found.shape != spatial:
            raise ValueError(
                f"{cls.__name__}: the mask has shape {found.shape}, but the spatial shape of "
                f"the array is {spatial}. The mask is one map shared by every batch element "
                "and channel (maths.md OP-26)."
            )
        stray = nan & ~found
        if stray.any():
            raise ValueError(
                f"{cls.__name__}: the array has {int(stray.sum())} NaN outside the mask, which "
                "would make every statistic NaN. Add those pixels to the mask, or replace the "
                "NaN with values."
            )
    if found.all():
        raise ValueError(
            f"{cls.__name__}: every pixel is missing, so no statistic can be computed "
            "(maths.md OP-3). Pass data with at least one valid pixel."
        )
    frozen = np.array(found, dtype=bool, copy=True)
    frozen.setflags(write=False)
    return frozen


def _axis_slice(key: Any, n: int, name: str) -> slice:
    """``key`` as a slice of an axis of length ``n``: an int selects one element and keeps
    the axis."""
    if isinstance(key, slice):
        return key
    if isinstance(key, int | np.integer) and not isinstance(key, bool):
        index = int(key)
        if not -n <= index < n:
            raise IndexError(f"{name} index {index} is out of range for {n} {name} element(s).")
        index %= n
        return slice(index, index + 1)
    raise IndexError(
        f"{name} index {key!r}: select batch elements and channels with ints or slices."
    )
