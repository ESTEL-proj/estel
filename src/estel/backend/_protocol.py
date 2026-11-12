"""The L0 contract: every primitive a backend provides (Part I, §2.2.1; Part II, §12.3).

This file is the contract of the primitive layer. Each backend (``torch/``, and ``jax/`` as an
experimental backend from P1) implements :class:`Backend`, and the conformance suite of
``tests/backend/`` checks every implementation against NumPy float64. L1 and L2 reach a backend
only through this protocol, never by importing ``torch`` or ``jax``.

Every primitive documents, in this order:

* **Shapes:** input and output shapes;
* **Dtypes:** accepted dtypes and the dtype of the result;
* **Values:** what is computed; where values matter for the statistics, the entry of
  ``docs/maths.md`` it implements (``M-...``);
* **Gradients:** what is differentiable, and to which order;
* **Edge cases:** what is rejected, with which error, and what is not checked.

General rules, which every primitive follows unless its docstring says otherwise:

* **Pure and out of place.** No primitive modifies its inputs; outputs never share mutable
  state with them. Nothing here keeps state between calls, except :meth:`Backend.deterministic`,
  which sets a process-wide flag.
* **No silent dtype change.** Floating inputs of one call share one dtype (float32 or float64,
  or the matching complex64 or complex128); a mismatch raises ``TypeError``. The result has
  that dtype unless the docstring says otherwise. No primitive promotes float32 to float64.
* **No decision on array values.** Primitives branch on shapes, dtypes and static arguments
  only, never on the values of an array (Part I, §3.1, rule 3), so they can be traced by
  ``jax.jit``. Index arrays are therefore not range-checked here: the NumPy code that builds
  the static plan validates them once, at construction.
* **Axes** are Python ints; negative values count from the end, as in NumPy.
* **Errors** say what to do next (golden rule 8).
* **Differentiation.** Primitives are differentiable with respect to their floating array
  inputs, to second order (Part I, §3.2), unless the docstring says otherwise. Integer and
  boolean arrays, shapes and static arguments carry no gradient.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from types import ModuleType
from typing import Any, Literal, NamedTuple, Protocol, TypeAlias

Array: TypeAlias = Any
"""An array of the backend: ``torch.Tensor`` in step 1, ``jax.Array`` in the JAX backend."""

Key: TypeAlias = Any
"""An opaque, immutable random key, made by :meth:`Backend.make_key`."""

Device: TypeAlias = Any
"""A device of the backend, or its name (``"cpu"``, ``"cuda:0"``); ``None`` is the default."""

DType = Literal["float32", "float64", "complex64", "complex128", "int32", "int64", "bool"]
"""Dtypes are named by strings, the same on every backend."""

PadMode = Literal["circular", "replicate", "zeros"]

Primals: TypeAlias = tuple[Array, ...]


class LbfgsResult(NamedTuple):
    """What :meth:`Backend.lbfgs` returns."""

    x: Array
    """The last iterate, with the shape and dtype of ``x0``."""
    state: Any
    """Opaque optimizer state; passing it back to ``lbfgs`` continues the same run."""
    losses: Array
    """The loss after each iteration, shape ``(n_iter,)``, dtype of ``x0``."""


class Backend(Protocol):
    """The primitives of one backend. Implemented once per backend, nowhere else."""

    @property
    def name(self) -> str:
        """The backend's name, as accepted by ``estel.set_backend``: ``"torch"`` or ``"jax"``."""
        ...

    # ------------------------------------------------------------ namespace and conversion

    @property
    def xp(self) -> ModuleType:
        """The Array API namespace of the backend, from ``array_api_compat.array_namespace``.

        Shapes:
            As in the Array API standard (elementwise operations, reductions, ``reshape``,
            ``concat``, ``stack``, ``where``, ``take``, and the ``fft`` extension).
        Dtypes:
            As in the Array API standard; float32, float64, complex64 and complex128 are all
            supported.
        Values:
            As in the Array API standard. The values that matter for the statistics use it as
            follows: masked means are ``sum(w * a) / sum(w)`` with weights ``w`` in {0, 1}
            (M-MEAN, M-NOT), never boolean indexing; the spectra use the orthonormal
            ``fft.rfft2`` (``norm="ortho"``, M-PS-2D); the angular transform uses the
            orthonormal ``fft.fft`` (M-RED-ANG).
        Gradients:
            Every function of the namespace used by L1 and L2 is differentiable to second
            order on real and complex inputs, except at the points where the function itself
            is not (``sqrt`` and ``abs`` at 0: use the safe modulus of
            ``docs/agents/differentiation.md``).
        Edge cases:
            The namespace also exposes in-place and ``out=`` forms; L1 and L2 must not use
            them (AST check, Part I, §2.4). Missing extensions (padding modes, convolution
            along axes, weighted gathers, segment sums, grids, keys, differentiation,
            optimizers) are the other primitives of this protocol.
        """
        ...

    def asarray(self, np_array: Any, dtype: DType, device: Device = None) -> Array:
        """Move a NumPy array (plans, kernels, masks, data) to the backend.

        Shapes:
            Same shape as ``np_array``; 0-d arrays stay 0-d.
        Dtypes:
            The result has exactly ``dtype``, which is required: nothing is inferred. Casting
            towards a lower precision of the same kind (float64 to float32) rounds to nearest,
            as ``numpy.ndarray.astype``. Casting towards a wider kind (bool to int to float to
            complex) is allowed.
        Values:
            ``np_array.astype(dtype)``, on ``device`` (the backend's default device if
            ``None``). The result never shares memory with ``np_array``.
        Gradients:
            None: the result is a constant (a leaf).
        Edge cases:
            ``np_array`` must be a ``numpy.ndarray``; anything else raises ``TypeError``
            (convert backend arrays at the DataClass boundary instead). Casting towards a
            narrower kind (complex to float, float to int, anything to bool) raises
            ``TypeError``, since it would drop information silently. On the JAX backend,
            float64, complex128 and int64 need JAX's process-wide 64-bit mode
            (``jax_enable_x64``), which the backend never switches itself; without it they
            raise ``TypeError``, as does ``random_normal`` in float64.
        """
        ...

    def to_numpy(self, x: Array) -> Any:
        """Copy a backend array to a NumPy array on the host.

        Shapes:
            Same shape as ``x``.
        Dtypes:
            The NumPy dtype matching ``x`` (float32 stays float32).
        Values:
            The values of ``x``, detached from any computation graph and copied to host
            memory; the result never shares memory with ``x``.
        Gradients:
            None. For boundaries only (results, I/O, plots): never call it inside the
            computation of the statistics, which must stay traceable.
        Edge cases:
            None.
        """
        ...

    # ------------------------------------------------------------ filtering

    def pad(
        self, x: Array, widths: Sequence[tuple[int, int]], axes: Sequence[int], mode: PadMode
    ) -> Array:
        """Pad ``x`` along ``axes`` (absent from the Array API).

        Shapes:
            ``widths[i] = (before, after)`` for axis ``axes[i]``; along that axis the length N
            becomes ``before + N + after``. Other axes are unchanged.
        Dtypes:
            Real or complex floating; the result has the dtype of ``x``.
        Values:
            With ``i`` the index in the padded axis and ``N`` its original length:

            * ``"circular"``: ``y[i] = x[(i - before) mod N]``;
            * ``"replicate"``: ``y[i] = x[clip(i - before, 0, N - 1)]`` (edge values repeated);
            * ``"zeros"``: ``y[i] = x[i - before]`` inside, 0 outside.

            The scattering transform pads by ``⌊K/2⌋`` on each side, circular if ``pbc=True``
            and replicate otherwise (M-CONV-2D), and by 2 before and 3 after for the
            downsampling kernel H (M-DOWN). Complex arrays are padded like their real and
            imaginary parts.
        Gradients:
            Linear in ``x``: the gradient sums the cotangents of all copies of each input
            entry.
        Edge cases:
            ``len(widths) != len(axes)``, repeated axes or negative widths raise
            ``ValueError``. For ``"circular"`` each width must be at most N (a single wrap,
            as in STL, whose padding refuses larger widths); a larger one raises
            ``ValueError``: reject the configuration when the plan is built. ``"circular"`` and
            ``"replicate"`` on an axis of length 0 raise ``ValueError``. Zero widths are
            allowed and return a copy of ``x``.
        """
        ...

    def conv_axes(
        self, x: Array, k: Array, axes: Sequence[int], stride: int | Sequence[int] = 1
    ) -> Array:
        """Valid real correlation of ``x`` with the kernels ``k`` along ``axes``.

        Applied directly along a block's axes, without moving them, and batched over every
        other axis of ``x``.

        Shapes:
            ``axes`` are d contiguous, increasing axes of ``x`` with lengths
            ``N_1 … N_d``. ``k`` has shape ``(*O, K_1, …, K_d)``: zero or more leading
            kernel axes ``O`` (for example the L orientations), then the kernel support.
            ``stride`` is one int for every axis or one per axis. The result has shape
            ``x.shape[:axes[0]] + O + (M_1, …, M_d) + x.shape[axes[-1] + 1:]``, with
            ``M_i = (N_i - K_i) // s_i + 1``: the kernel axes are inserted just before the
            first filtered axis.
        Dtypes:
            ``x`` and ``k`` are real and share one floating dtype.
        Values:
            A cross-correlation without padding and without flipping the kernel::

                y[..., o, n, ...] = Σ_m k[o, m] · x[..., s·n + m, ...]

            with ``n``, ``m`` and ``s·n`` multi-indices over the d axes. Padding is a
            separate call to :meth:`pad`. Padding by ``⌊K/2⌋`` then correlating gives the
            filtering of M-CONV-2D (M-W1); padding by (2, 3) then correlating with H at
            stride 2 gives one downsampling step (M-DOWN). This is a correlation, not a
            convolution (OP-1): it reproduces STL.
        Gradients:
            Bilinear in ``x`` and ``k``; differentiable with respect to both, to any order.
        Edge cases:
            Complex ``x`` or ``k`` raises ``TypeError``: a complex correlation is two real
            calls, made in L1 (M-CONV-2D; Part I, §3.1, rule 6). ``k.ndim < d``,
            non-contiguous or repeated axes, a stride below 1, or ``N_i < K_i`` raise
            ``ValueError``.
        """
        ...

    # ------------------------------------------------------------ sparse operations

    def gather_weighted(self, x: Array, axis: int, index: Array, weight: Array) -> Array:
        """Apply a sparse matrix, stored in padded CSR form, along ``axis``.

        Used for the HEALPix stencils and decimation of step 1 (Part II, §9.3).

        Shapes:
            ``index`` and ``weight`` have the same shape ``(P, Q, M)``: for each output
            pixel ``p`` and stencil entry ``q``, ``M`` (input position, weight) pairs. The
            result replaces ``axis`` of ``x`` by the two axes ``(P, Q)``.
        Dtypes:
            ``x`` and ``weight`` are real and share one floating dtype; ``index`` is int32 or
            int64.
        Values:
            ``y[..., p, q, ...] = Σ_m weight[p, q, m] · x[..., index[p, q, m], ...]``.
            Rows with fewer than ``M`` entries are padded with weight 0 and any in-range
            index.
        Gradients:
            Bilinear in ``x`` and ``weight``; differentiable with respect to both, to any
            order. None with respect to ``index``.
        Edge cases:
            Shape or dtype mismatches raise ``ValueError`` or ``TypeError``. Index values are
            not checked (they are plan data, validated when the plan is built). A padding
            entry still reads ``x`` at its index, so a NaN there gives ``0 · NaN = NaN``:
            missing data must be weights, not NaN (M-DOWN-MASK, OP-27).
        """
        ...

    def segment_sum(self, x: Array, axis: int, segment_ids: Array, num_segments: int) -> Array:
        """Sum the entries of ``x`` along ``axis`` by segment.

        Used for bin averages (radial bins of the spectra), HEALPix parent binning and
        reductions.

        Shapes:
            ``segment_ids`` has shape ``(x.shape[axis],)``. The result replaces ``axis`` by
            an axis of length ``num_segments``.
        Dtypes:
            ``x`` is real or complex floating; ``segment_ids`` is int32 or int64.
        Values:
            ``y[..., s, ...] = Σ_{i: segment_ids[i] = s} x[..., i, ...]``; an empty segment
            gives 0. The order of summation is unspecified. Assigning each FFT mode to its
            radial index and summing is the first step of the radial spectrum (M-PS-2D).
        Gradients:
            Linear in ``x``: the gradient of entry ``i`` is the cotangent of its segment.
        Edge cases:
            Ids are not checked; they must lie in ``[0, num_segments)``, which the plan
            ensures. ``num_segments < 0`` raises ``ValueError``.
        """
        ...

    def scatter_to_grid(
        self,
        values: Array,
        index: Sequence[Array],
        shape: Sequence[int],
        fill: float = float("nan"),
    ) -> Array:
        """Build a new array on the complete grid of indices, out of place.

        Assembles the full-grid statistics once, at the end of the computation, from the
        values computed at the static positions of the plan (Part I, §3.1.1).

        Shapes:
            ``index`` holds d integer arrays of shape ``(n,)``, one coordinate per grid axis;
            ``shape`` has length d. ``values`` has shape ``(*B, n)``; the result has shape
            ``(*B, *shape)``.
        Dtypes:
            Real or complex floating; the result has the dtype of ``values``. ``fill`` is
            converted to that dtype; for a complex dtype it becomes ``fill + 0j``.
        Values:
            ``y[..., index[0][i], …, index[d-1][i]] = values[..., i]``, and ``fill`` (NaN by
            default) at every other position. Coefficients that STL leaves as NaN, such as
            the (c₂, c₁) entries of S1 and S2, are positions absent from ``index`` (M-CHAN).
        Gradients:
            Linear in ``values``; nothing flows from the ``fill`` positions.
        Edge cases:
            Positions must be distinct and inside ``shape``; this is not checked here (plan
            data). Mismatched lengths raise ``ValueError``.
        """
        ...

    def gather_from_grid(self, grid: Array, index: Sequence[Array]) -> Array:
        """Read back the coefficients at the static positions ``index``: the inverse of
        :meth:`scatter_to_grid`.

        Shapes:
            ``index`` holds d integer arrays of shape ``(n,)``; ``grid`` has shape
            ``(*B, *shape)`` with ``len(shape) = d``. The result has shape ``(*B, n)``.
        Dtypes:
            Real or complex floating; the result has the dtype of ``grid``.
        Values:
            ``y[..., i] = grid[..., index[0][i], …, index[d-1][i]]``, so that
            ``gather_from_grid(scatter_to_grid(v, index, shape), index) == v``.
        Gradients:
            Linear in ``grid``: zero cotangent at the positions not read.
        Edge cases:
            Index values are not checked (plan data). Mismatched lengths raise
            ``ValueError``.
        """
        ...

    # ------------------------------------------------------------ complex values

    def complex_to_real_pairs(self, z: Array) -> Array:
        """Store complex values as trailing (real, imaginary) pairs.

        Shapes:
            ``(...)`` to ``(..., 2)``.
        Dtypes:
            complex64 to float32, complex128 to float64.
        Values:
            ``r[..., 0] = Re z``, ``r[..., 1] = Im z``.
        Gradients:
            Linear; differentiable to any order.
        Edge cases:
            A real ``z`` raises ``TypeError``.
        """
        ...

    def real_pairs_to_complex(self, r: Array) -> Array:
        """The inverse of :meth:`complex_to_real_pairs`.

        Shapes:
            ``(..., 2)`` to ``(...)``.
        Dtypes:
            float32 to complex64, float64 to complex128.
        Values:
            ``z = r[..., 0] + 1j · r[..., 1]``.
        Gradients:
            Linear; differentiable to any order.
        Edge cases:
            A complex ``r`` raises ``TypeError``; a last axis of length other than 2 raises
            ``ValueError``.
        """
        ...

    # ------------------------------------------------------------ differentiation

    def stop_gradient(self, x: Array) -> Array:
        """Block derivatives through ``x``.

        Shapes:
            Same as ``x``.
        Dtypes:
            Same as ``x``.
        Values:
            The values of ``x``.
        Gradients:
            All derivatives of the result with respect to ``x`` are zero, at every order
            (``x.detach()`` in PyTorch, ``jax.lax.stop_gradient`` in JAX).
        Edge cases:
            None.
        """
        ...

    def value_and_grad(self, fn: Callable[..., Array]) -> Callable[..., tuple[Array, Array]]:
        """Value and gradient of a real scalar function, with respect to its first argument.

        Shapes:
            ``fn(x, *args)`` returns a 0-d array. The returned function ``g(x, *args)``
            returns ``(fn(x, *args), ∂fn/∂x)``, the gradient with the shape of ``x``.
        Dtypes:
            ``x`` and the value are real floating; the gradient has the dtype of ``x``.
        Values:
            The derivative of ``fn`` at ``x``, by reverse-mode differentiation; the
            conformance suite checks it against finite differences in float64.
        Gradients:
            ``g`` is itself differentiable once more (second order, through :meth:`hvp`).
        Edge cases:
            A complex ``x`` or a complex or non-scalar value raises ``TypeError`` or
            ``ValueError``: PyTorch and JAX use different conventions for complex
            derivatives, so convert complex outputs with :meth:`complex_to_real_pairs`
            first.
        """
        ...

    def vjp(
        self, fn: Callable[..., Array], *primals: Array
    ) -> tuple[Array, Callable[[Array], Primals]]:
        """Vector-Jacobian product (reverse mode).

        Shapes:
            Returns ``(out, vjp_fn)`` with ``out = fn(*primals)``; ``vjp_fn(cotangent)``,
            with a cotangent of the shape of ``out``, returns one array per primal, with its
            shape.
        Dtypes:
            Primals, ``out`` and cotangents are real floating, of one dtype.
        Values:
            ``vjp_fn(c)[i] = cᵀ · ∂out/∂primals[i]``.
        Gradients:
            ``vjp_fn`` is differentiable once more (second order, Part I, §3.2).
        Edge cases:
            Complex primals or outputs raise ``TypeError``: convert them with
            :meth:`complex_to_real_pairs` first (the conventions of PyTorch and JAX for
            complex cotangents differ).
        """
        ...

    def jvp(
        self, fn: Callable[..., Array], primals: Primals, tangents: Primals
    ) -> tuple[Array, Array]:
        """Jacobian-vector product (forward mode).

        Shapes:
            ``tangents`` has one array per primal, with its shape. Returns
            ``(fn(*primals), tangent_out)``, ``tangent_out`` with the shape of the output.
        Dtypes:
            Primals, tangents and outputs are real floating, of one dtype.
        Values:
            ``tangent_out = Σ_i ∂out/∂primals[i] · tangents[i]``.
        Gradients:
            Differentiable once more (second order).
        Edge cases:
            As :meth:`vjp`; a length mismatch between primals and tangents raises
            ``ValueError``.
        """
        ...

    def hvp(self, fn: Callable[[Array], Array], x: Array, v: Array) -> Array:
        """Hessian-vector product of a real scalar function.

        Shapes:
            ``fn(x)`` is 0-d; ``v`` and the result have the shape of ``x``.
        Dtypes:
            Real floating, the dtype of ``x``.
        Values:
            ``(∂²fn/∂x²) · v``, used for implicit differentiation (Part I, §3.2.1).
        Gradients:
            Not required beyond this.
        Edge cases:
            As :meth:`value_and_grad`.
        """
        ...

    def jit(
        self, fn: Callable[..., Any], static_argnames: Sequence[str] = ()
    ) -> Callable[..., Any]:
        """Compile ``fn`` where the backend can.

        Shapes:
            Unchanged.
        Dtypes:
            Unchanged.
        Values:
            Identical to ``fn`` up to the backend's rounding for fused operations. The
            identity in PyTorch (returns ``fn``); ``jax.jit`` in the JAX backend.
        Gradients:
            Unchanged.
        Edge cases:
            Arguments named in ``static_argnames`` must be hashable (the plan is).
        """
        ...

    # ------------------------------------------------------------ randomness

    def make_key(self, seed: int) -> Key:
        """Make a random key from an integer seed (Part I, §3.1, rule 5).

        Shapes:
            Not applicable: the key is opaque.
        Dtypes:
            Not applicable.
        Values:
            The same seed gives the same key. The key is immutable and independent of any
            global random state.
        Gradients:
            None.
        Edge cases:
            ``seed`` must be an int in ``[0, 2**63)``; anything else raises ``ValueError``.
            There is no key-splitting primitive: make several independent draws from several
            seeds.
        """
        ...

    def random_normal(self, key: Key, shape: Sequence[int], dtype: DType) -> Array:
        """Independent standard normal samples.

        Shapes:
            ``shape``.
        Dtypes:
            ``dtype``, float32 or float64.
        Values:
            Samples of N(0, 1). The same key, shape, dtype and device give the same values;
            the key is not consumed, and the global random state is neither read nor
            changed. Different backends give different values for the same seed: synthesis
            references are kept per backend (Part III, §17).
        Gradients:
            None: the result is a constant.
        Edge cases:
            A complex or integer ``dtype`` raises ``TypeError``.
        """
        ...

    # ------------------------------------------------------------ optimization

    def lbfgs(
        self,
        fun: Callable[[Array], Array],
        x0: Array,
        *,
        max_iter: int,
        history_size: int,
        tolerance_grad: float,
        tolerance_change: float,
        state: Any = None,
    ) -> LbfgsResult:
        """Minimise ``fun`` with the backend's built-in L-BFGS.

        ``torch.optim.LBFGS(line_search_fn="strong_wolfe")`` in PyTorch; ``optax.lbfgs`` in
        step 2 (LF1), until which the experimental JAX backend raises ``NotImplementedError``;
        the Rust L-BFGS when the Rust engine is selected (Part III, §18.2).

        Shapes:
            ``fun(x)`` is 0-d, ``x`` with the shape of ``x0``. See :class:`LbfgsResult`.
        Dtypes:
            ``x0`` and the loss are real floating, of one dtype.
        Values:
            At most ``max_iter`` iterations, with a strong Wolfe line search, keeping
            ``history_size`` correction pairs, stopping early when the largest gradient
            component falls below ``tolerance_grad`` or the change of loss or step below
            ``tolerance_change``. The settings have no default here: their values are those
            of STL's ``optimize_lbfgs``, fixed by the synthesis section of the maths tab
            (task T6.0), and ``run_lbfgs`` passes them. ``state`` from a previous result
            resumes that run with its history (synthesis checkpoints, Part II, §12.9);
            ``None`` starts a new one. Trajectories differ between backends.
        Gradients:
            ``fun`` is differentiated by the backend; the result itself is not
            differentiable.
        Edge cases:
            ``x0`` is not modified. ``max_iter < 1`` or ``history_size < 1`` raise
            ``ValueError``; a ``state`` from another backend raises ``TypeError``.
        """
        ...

    def deterministic(self, enabled: bool) -> None:
        """Switch deterministic algorithms on or off, for the whole process.

        Shapes:
            Not applicable.
        Dtypes:
            Not applicable.
        Values:
            When on, repeated runs on the same device and software give bitwise-identical
            results, and an operation without a deterministic implementation raises an error.
            ``torch.use_deterministic_algorithms`` in PyTorch;
            ``--xla_gpu_deterministic_ops=true`` in ``XLA_FLAGS`` in the JAX backend. On in
            tests.
        Gradients:
            Not applicable.
        Edge cases:
            Process-wide state: call it at set-up, never inside the computation. XLA reads
            ``XLA_FLAGS`` only when JAX starts, so on the JAX backend a call that would change
            the setting after the first JAX array is created raises ``RuntimeError``.
        """
        ...
