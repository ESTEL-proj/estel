"""The mother class ``Data`` (Part II, §12.4; Part I, §2.3; maths.md M-DATA-2D, M-DATA-1D).

A ``Data`` object is one backend array plus static metadata: the periodicity ``pbc`` and the mask of
missing values. It is immutable: operations return new objects (:meth:`Data.with_array`),
never modify one. Subclasses (``Data1D``, ``Data2D`` in ``_cartesian.py``, under
``DataCartesian``) set the number of spatial axes and the geometry constants, and override only
the geometry hooks (tag G of §12.4);
``DataHealpix`` (``_healpix.py``) has one pixel axis on the sphere; ``DataTensorProd``
(``_tensor_prod.py``) composes the planar ones, one geometry per block.

Validation happens once, here, when the object is built (review rule B1): the array is converted
to the backend, promoted to ``(Nb, Nc, *spatial)``, and its NaN become the mask. The mask is a
NumPy boolean array over the spatial axes (True = missing), shared by every batch element and
every channel (maths.md OP-26). It is static: the operator turns it into weights and
reweighting maps once, when it is built (``estel.data._masks``), and the computation never looks
for missing values again (Part I, §3.1, rule 3).

The values of the array at masked pixels are kept as given (NaN, if the mask came from them).
They are never read: the statistics use the mask as weights (M-DOWN-MASK, M-MEAN; OP-27).

Every ``Data`` object is a pytree whose only leaf is the array (Part I, §2.3 and §3.1, rule 9; task
T2.9): each subclass is registered when it is defined, on every backend, so that ``jax.jit``
and ``jax.vmap`` accept a ``Data`` object and return one.
"""

from __future__ import annotations

from typing import Any, ClassVar, NamedTuple, cast

import numpy as np
from numpy.typing import NDArray

from .._config import default_device, default_dtype
from ..backend import _registry

_FLOAT_DTYPES = ("float32", "float64")


class Data:
    """Data plus static metadata: the mother class of every geometry.

    Build a geometry instead: ``Data1D``, ``Data2D`` or ``DataHealpix``. The mother class holds
    the shared logic; a subclass sets ``ndim``, the number of spatial axes (a class constant of a
    single geometry), and overrides only the geometry hooks.

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

    @property
    def ndim(self) -> int:
        """The number of spatial axes of the array: a class constant of a single geometry
        (``Data1D.ndim``, ``Data2D.ndim``, ``DataHealpix.ndim``), those of every block for a
        tensor product (``DataTensorProd.ndim``). It counts axes and does not name the geometry:
        ``DataHealpix.ndim`` is 1, its pixels lying on one axis, though the sphere is a 2D
        surface. The geometry is the tag the class declares, ``_geometry``."""
        raise NotImplementedError

    _nan_threshold: ClassVar[float]
    """τ of maths.md M-DOWN-MASK: a downsampled pixel is invalid when its fraction of invalid
    input exceeds it."""

    _geometry: ClassVar[str]
    """The name of the geometry, a capability the class declares: ``"1d"``, ``"2d"``, the
    geometries of the engine capabilities (``estel.engines._protocol.GEOMETRIES``). It also
    names the geometry in saved files (``estel.io``), and :func:`geometry_class` maps it back to
    the class."""

    __slots__ = ("_array", "_backend", "_mask", "_pbc")

    _array: Any
    _backend: str
    _mask: NDArray[np.bool_] | None
    _pbc: bool | tuple[bool, ...]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Register every geometry as a pytree on every backend (task T2.9): registration is
        per exact class, so each subclass needs its own."""
        super().__init_subclass__(**kwargs)
        _registry.register_pytree(cls, cls._tree_flatten, cls._tree_unflatten)
        if "_geometry" in cls.__dict__:
            _GEOMETRIES.update({cls._geometry: cls})

    def __init__(self, array: Any, mask: Any = None, *, pbc: bool) -> None:
        cls = type(self)
        if not hasattr(cls, "_geometry"):
            raise TypeError(
                f"{cls.__name__} is the mother class and has no geometry: build a geometry "
                "instead, estel.Data1D or estel.Data2D, or estel.DataHealpix."
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
        """The shapes the constructor accepts, for error messages: each geometry gives its own."""
        raise NotImplementedError(
            f"{cls.__name__} does not implement _accepted_shapes: every single-domain "
            "geometry overrides it (Part I, §2.3.1; Part II, §12.4)."
        )

    # ------------------------------------------------------------------ construction

    def _set(
        self,
        array: Any,
        mask: NDArray[np.bool_] | None,
        pbc: bool | tuple[bool, ...],
        backend: str,
        meta: tuple[Any, ...] = (),
    ) -> None:
        object.__setattr__(self, "_array", array)
        object.__setattr__(self, "_mask", mask)
        object.__setattr__(self, "_pbc", pbc)
        object.__setattr__(self, "_backend", backend)
        self._set_meta(meta)

    @classmethod
    def _from_parts(
        cls,
        array: Any,
        mask: NDArray[np.bool_] | None,
        pbc: bool | tuple[bool, ...],
        backend: str,
        meta: tuple[Any, ...] = (),
    ) -> Data:
        """A new object from parts already validated: no conversion, no check of values, so
        that it works inside traced code. ``meta`` is the static metadata of the geometry
        beyond the mask and ``pbc``, :meth:`_meta` of an object of the same class: ``()`` for
        planar data, ``(depth, cell_ids)`` for ``DataHealpix``."""
        new = cls.__new__(cls)
        new._set(array, mask, pbc, backend, meta)
        return new

    def _meta(self) -> tuple[Any, ...]:
        """The static metadata of the geometry beyond the mask and ``pbc``, hashable: what
        :meth:`_from_parts` needs, with them, to rebuild an object of this class (``()`` for
        1D and 2D data, ``(depth, cell_ids)`` for ``DataHealpix``); set by :meth:`_set_meta`."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement _meta: every single-domain "
            "geometry overrides it (Part I, §2.3.1; Part II, §12.4)."
        )

    @classmethod
    def _meta_content(cls, meta: tuple[Any, ...]) -> tuple[dict[str, Any], dict[str, NDArray[Any]]]:
        """The metadata ``meta`` of :meth:`_meta` for a file (``estel.io``): JSON parameters,
        and arrays by name, which :meth:`_example_from_content` reads back."""
        raise NotImplementedError(
            f"{cls.__name__} does not implement _meta_content: every single-domain "
            "geometry overrides it (Part I, §2.3.1; Part II, §12.4)."
        )

    @classmethod
    def _example_from_content(
        cls,
        n_channels: int,
        spatial_shape: tuple[int, ...],
        pbc: bool,
        content: dict[str, Any],
        arrays: dict[str, NDArray[Any]],
    ) -> Data:
        """An example of the data a file describes (``STStatistics.load``), zeros of one map of
        ``n_channels`` channels, built by the constructor, which checks it: the spatial shape,
        ``pbc``, the metadata of :meth:`_meta_content`, its JSON ``content``, and the plan's
        ``arrays`` by name, the mask (``"mask"``, when values are missing) and the arrays of
        :meth:`_meta_content`."""
        raise NotImplementedError(
            f"{cls.__name__} does not implement _example_from_content: every single-domain "
            "geometry overrides it (Part I, §2.3.1; Part II, §12.4)."
        )

    def _set_meta(self, meta: tuple[Any, ...]) -> None:
        """Set the metadata of :meth:`_meta` on a new object (from :meth:`_set`)."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement _set_meta: every single-domain "
            "geometry overrides it (Part I, §2.3.1; Part II, §12.4)."
        )

    def _like(self, array: Any) -> Data:
        """A new object of this geometry with ``array`` and the static metadata of this one: no
        conversion, no check of values (the functional updates and the slicing use it)."""
        return self._from_parts(array, self._mask, self._pbc, self._backend, self._meta())

    def _rebuild_advice(self) -> str:
        """The constructor call that builds an object like this one from a NumPy ``array``,
        for the messages."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement _rebuild_advice: every single-domain "
            "geometry overrides it (Part I, §2.3.1; Part II, §12.4)."
        )

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
    def pbc(self) -> bool | tuple[bool, ...]:
        """Whether the boundary conditions are periodic: one boolean for a single geometry, one
        per block for a tensor product (``DataTensorProd.pbc``)."""
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

    def with_array(self, new_array: Any) -> Data:
        """A new object with ``new_array`` in place of the array, and the same metadata.

        Data objects are never updated in place. ``new_array`` must be an array of the same
        backend with the shape of :attr:`shape`; it is used as it is, without a copy and
        without looking at its values, so this works inside traced and differentiated code.

        For a ``DataTensorProd``, the new array must hold no NaN: tensor products have no
        missing values in ESTEL 1.0.0 (LF8), and their constructor, which refuses a NaN, is the
        only place where values are looked at; a NaN given here makes every statistic NaN
        (maths.md OP-80).
        """
        _registry.check_backend(self._backend, new_array, owner=f"this {type(self).__name__}")
        if _registry.backend_of(new_array) is None:
            raise TypeError(
                f"with_array takes an array of the {self._backend} backend, got "
                f"{type(new_array).__name__}. To start from NumPy, convert the array to the "
                f"{self._backend} backend first, in the dtype of data.array, or build a new "
                f"{self._rebuild_advice()} instead."
            )
        new_shape = tuple(int(n) for n in new_array.shape)
        if new_shape != self.shape:
            raise ValueError(
                f"with_array: the new array has shape {new_shape}, but this "
                f"{type(self).__name__} has shape {self.shape}. Pass an array of the same "
                "shape, or build a new object."
            )
        return self._like(new_array)

    def copy(self, empty: bool = False) -> Data:
        """A new object with a copy of the array, and the same metadata.

        With ``empty=True`` the array of the copy is filled with zeros: a data object always
        holds an array of its shape.
        """
        xp = _registry.backend_named(self._backend).xp
        array = xp.zeros_like(self._array) if empty else xp.asarray(self._array, copy=True)
        return self._like(array)

    def __getitem__(self, key: Any) -> Data:
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
        return self._like(array)

    # ------------------------------------------------------------------ pytree

    def _tree_flatten(self) -> tuple[tuple[Any], tuple[Any, ...]]:
        """The array is the only leaf; everything else is static (Part I, §2.3).

        The static part must be hashable and compare by value, since JAX caches compiled
        functions by it: the mask goes in as a :class:`_StaticMask`, and the metadata of the
        geometry (:meth:`_meta`) is hashable by contract.
        """
        mask = None if self._mask is None else _StaticMask(self._mask)
        return (self._array,), (mask, self._pbc, self._backend, self._meta())

    @classmethod
    def _tree_unflatten(cls, aux: tuple[Any, ...], children: tuple[Any, ...]) -> Data:
        """Rebuild from :meth:`_tree_flatten`, without looking at the leaf: the backend passes
        tracers, batched arrays and placeholders."""
        mask, pbc, backend, meta = aux
        return cls._from_parts(children[0], None if mask is None else mask.mask, pbc, backend, meta)

    # ------------------------------------------------------------------ geometry hooks

    # Only ``WaveletOperatorSingleDomain`` (``estel.wavelets._operator``) calls the hooks of
    # filtering and downsampling, _conv, _downsample and _crop_weights (Part I, §2.3, §2.4).

    def _conv(self, x: Any, tables: Any, level: int) -> Any:
        """Filter ``x`` with the wavelets at resolution ``level`` (maths.md M-W1), with the
        static ``tables`` of that level, given by ``WaveletOperatorSingleDomain``.

        ``x`` is an array of this object's backend, ``(Nb, Nc, *spatial)`` at that level, real
        or complex; ``tables`` are, for planar data, the complex bank of the geometry, cast to
        the backend and to the complex dtype matching ``x``. The result is complex, with the
        orientation axis of the convolution right after the batch axis:
        ``(Nb, L, Nc, *spatial)``, for the first and the second layer alike. The maps of one
        orientation are then contiguous, and those of the second layer form the matrix of the
        Gram form without a copy (Part III, §19.12, point 2; maths_step2.md M-GRAM). The hook
        builds the result in that layout directly. Each geometry implements it
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
        or complex; ``weights`` the static weights of the step (:class:`DownsampleWeights`),
        given by ``WaveletOperatorSingleDomain``.
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
        axis. The wavelet operator computes ``border`` from the convolutions applied (M-BORDER)
        and multiplies these weights by the valid pixels of the mask, once, when an operator
        is built (Part I, §3.1.1). Each geometry implements it (planar: STL's ``_crop``, with its
        reduction of a border too large for the array, OP-13; HEALPix: no crop).
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement the geometry hook _crop_weights: every "
            "geometry overrides it (Part II, §12.4)."
        )

    def _nyquist_filter(self, x: Any) -> Any:
        """The pre-filter of a synthesis (maths.md M-SYN-NYQ; Part II, §12.4), applied to the
        target and to the start: ``x``, an array of this object's backend, ``(Nb, Nc,
        *spatial)``, band-limited as the geometry does it, with the shape and the dtype of
        ``x``. Only the syntheses call it (``estel.synthesis``), as STL's ``apply_nyquist_filter``
        and ``apply_bandlimit`` go through the data type. Each geometry the syntheses take
        implements it (2D: the filter of M-SYN-NYQ; HEALPix: none in ESTEL 1.0.0, OP-77)."""
        raise NotImplementedError(
            f"{type(self).__name__} has no pre-filter of syntheses: the syntheses of ESTEL 1.0.0 "
            "are of 2D data and of full-sky HEALPix fields (maths.md section 5). Synthesise "
            "estel.Data2D or estel.DataHealpix maps instead."
        )

    # The same three operations along given axes of an array, for a geometry that is a block of
    # a tensor product (``DataTensorProd``, maths.md M-NOT-TP): the tensor product dispatches its
    # hooks to them, along the axes of each block, without moving any axis (Part II, §12.4).

    @classmethod
    def _conv_along(
        cls, x: Any, tables: Any, *, axes: tuple[int, ...], pbc: bool, backend: str, owner: str
    ) -> Any:
        """:meth:`_conv` of this geometry along ``axes`` of ``x``: the orientation axis of the
        result right after the batch axis, ``(Nb, L, Nc, ...)``."""
        raise NotImplementedError(
            f"{owner}: {cls.__name__} cannot be a block of a tensor product; the blocks of 1.0.0 "
            "are Data1D and Data2D (maths.md M-DATA-TP)."
        )

    @classmethod
    def _downsample_along(
        cls,
        x: Any,
        weights: DownsampleWeights,
        *,
        axes: tuple[int, ...],
        pbc: bool,
        backend: str,
        owner: str,
    ) -> Any:
        """:meth:`_downsample` of this geometry along ``axes`` of ``x``."""
        raise NotImplementedError(
            f"{owner}: {cls.__name__} cannot be a block of a tensor product; the blocks of 1.0.0 "
            "are Data1D and Data2D (maths.md M-DATA-TP)."
        )

    @classmethod
    def _crop_weights_along(
        cls, shape: tuple[int, ...], level: int, border: int, *, owner: str
    ) -> NDArray[np.float64]:
        """:meth:`_crop_weights` of this geometry on a grid of full-resolution ``shape``."""
        raise NotImplementedError(
            f"{owner}: {cls.__name__} cannot be a block of a tensor product; the blocks of 1.0.0 "
            "are Data1D and Data2D (maths.md M-DATA-TP)."
        )


# ---------------------------------------------------------------------- geometry hook inputs


_GEOMETRIES: dict[str, type[Data]] = {}
"""The geometry classes, by the name they declare (``Data._geometry``)."""


def geometry_class(name: str) -> type[Data]:
    """The Data subclass that declares the geometry ``name`` (``"1d"``, ``"2d"``)."""
    if name not in _GEOMETRIES:
        known = ", ".join(repr(n) for n in sorted(_GEOMETRIES))
        raise ValueError(
            f"no Data subclass declares the geometry {name!r}; known geometries: {known}."
        )
    return _GEOMETRIES[name]


class DownsampleWeights(NamedTuple):
    """The static weights of one downsampling step (:meth:`Data._downsample`), as arrays
    of the data's backend, built from the plan.

    ``data`` does not import ``wavelets``, so the kernel is passed in rather than imported:
    the wavelet operator casts ``smoothing_kernel(ndim)`` once and passes it at every step.
    """

    kernel: Any
    """The static kernel of the step, with the real dtype of the data: planar, the smoothing
    kernel of maths.md M-SMOOTH-2D, M-SMOOTH-1D; HEALPix, the decimation tables of the step
    (``DownTablesHealpix``, M-DOWN-HPX)."""
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
    NumPy array can be neither hashed nor compared to a single bool. Two Data objects with equal
    masks then share one compiled function; a different mask compiles a new one, since the
    mask is static.

    The hash is computed once, here, and the wrapped mask is read-only, so that the hash can
    never go stale. The mask of a ``Data`` object is already a read-only array that owns its
    data, and is kept as it is; any other mask (writable, or a view whose base may be written)
    is copied first.
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


def _to_backend(array: Any, backend_name: str, cls: type[Data]) -> Any:
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


def _promote(array: Any, cls: type[Data]) -> Any:
    """Promote ``array`` to ``(Nb, Nc, *spatial)`` (maths.md M-DATA-2D, M-DATA-1D, OP-29)."""
    ndim = cast(int, cls.ndim)  # the class constant of a single geometry
    return _promote_spatial(array, ndim, cls.__name__, cls._accepted_shapes())


def _promote_spatial(array: Any, ndim: int, owner: str, accepted: str) -> Any:
    """Promote ``array``, with ``ndim`` spatial axes last, to ``(Nb, Nc, *spatial)``: one
    leading axis is the channels, none gives ``(1, 1, *spatial)`` (maths.md M-DATA-2D,
    M-DATA-1D, M-DATA-TP, OP-29)."""
    shape = tuple(int(n) for n in array.shape)
    if not ndim <= len(shape) <= ndim + 2:
        raise ValueError(f"{owner}: the array has shape {shape}; accepted shapes are {accepted}.")
    if 0 in shape:
        raise ValueError(f"{owner}: the array has shape {shape}, with an empty axis.")
    full = (1,) * (ndim + 2 - len(shape)) + shape
    if full == shape:
        return array
    return _registry.current_backend().xp.reshape(array, full)


def _validate_mask(
    mask: Any,
    nan: NDArray[np.bool_],
    shape: tuple[int, ...],
    backend: Any,
    cls: type[Data],
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
