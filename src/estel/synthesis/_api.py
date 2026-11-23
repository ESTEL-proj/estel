"""The synthesis API: ``synthesize_from_maps``, ``synthesize_from_stats`` and ``SynthesisResult``
(Part II, §12.9; maths.md section 5; task T6.1, issue #28).

A synthesis looks for fields u whose statistics match those of a target, with STL's settings:
the Nyquist pre-filter, the standardisation of the target, the Gaussian start, the weighted
loss on the complex form of the flattened statistics and PyTorch's L-BFGS (M-SYN-TARGET to
M-SYN-OPS). What ESTEL adds is checkpoints (maths.md §5.8). In 1.0.0 the fields are periodic,
planar 2D and without missing values (LF47), and syntheses run on the PyTorch backend only
(LF1); the 2D power spectrum joins the loss with T4.1.

The driver is not traced code: it reads the losses and the counts of the optimizer to write
checkpoints and to stop. The loss itself is pure (``estel.synthesis._loss``).
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from .._config import default_dtype
from .._reference import STATISTICS_REFERENCE, SYNTHESIS_REFERENCE
from ..backend import _registry
from ..backend._protocol import Backend, LbfgsState
from ..data._base import Data, geometry_class
from ..scattering._operator import STOperator
from ..scattering._plan import PlanarPlan
from ..statistics._arrays import device_of
from ..statistics._statistics import _NO_REDUCTION, STStatistics
from ..wavelets._operator import default_scales
from . import _checkpoints
from ._init import gaussian_start, nyquist_filter, standardize_target, unstandardize
from ._loss import WEIGHTS, Objective, flatten

# The settings of STL's optimize_lbfgs that the synthesis functions do not take (M-SYN-LBFGS).
TOLERANCE_GRAD = 1e-12
TOLERANCE_CHANGE = 1e-15
# The options of the operators a synthesis takes from st_kwargs: J only, as STL's user level
# (maths.md M-SYN-OPS); L and the rest are the defaults, and the synthesis sets the others.
ST_KWARGS = ("J",)
# The settings a resumed run must share with the run that wrote its checkpoint.
_SAME_RUN = (
    "entry",
    "backend",
    "dtype",
    "running_shape",
    "mean_field",
    "J",
    "L",
    "lr",
    "history_size",
    "target_sha256",
)


class SynthesisResult:
    """The result of ``synthesize_from_maps`` or ``synthesize_from_stats`` (Part II, §12.9).

    Read :attr:`field`, the synthesised fields; :attr:`losses`, the loss after each iteration;
    :attr:`n_iter` and :attr:`n_evals`, the iterations and the evaluations of the loss, which
    differ (maths.md M-SYN-LBFGS, OP-74); :attr:`checkpoints`, the files this call wrote; and
    :attr:`manifest`, the settings of the run.
    """

    __slots__ = ("_checkpoints", "_converged", "_field", "_losses", "_manifest", "_n_evals")

    _field: Any
    _losses: NDArray[Any]
    _n_evals: int
    _converged: bool
    _checkpoints: tuple[str, ...]
    _manifest: dict[str, Any]

    def __init__(self) -> None:
        raise TypeError(
            "A SynthesisResult is returned by estel.synthesize_from_maps or "
            "estel.synthesize_from_stats, not built."
        )

    @classmethod
    def _create(cls, **parts: Any) -> SynthesisResult:
        new = cls.__new__(cls)
        for name, value in parts.items():
            object.__setattr__(new, f"_{name}", value)
        return new

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("A SynthesisResult is immutable.")

    @property
    def field(self) -> Any:
        """The synthesised fields, an array of the backend, de-standardised (M-SYN-OUT).

        ``(n_batch, Nc, N1, N2)``, without the batch axis if ``n_batch`` is 1, and without the
        channel axis from maps if the target was a 2D array, from statistics if Nc is 1 (OP-75).
        """
        return self._field

    @property
    def losses(self) -> NDArray[Any]:
        """The loss after each iteration of the run, a NumPy array of length :attr:`n_iter`,
        those before a checkpoint included when the run was resumed from it."""
        return self._losses.copy()

    @property
    def n_iter(self) -> int:
        """The iterations of L-BFGS done in the run."""
        return int(self._losses.shape[0])

    @property
    def n_evals(self) -> int:
        """The evaluations of the loss counted by L-BFGS in the run: those of its line
        searches and the first, at the start (OP-74). The cap ⌊5 x ``max_iter`` / 4⌋ is
        checked between iterations only, as in PyTorch and STL, so a run can exceed it by the
        evaluations of its last iteration."""
        return self._n_evals

    @property
    def converged(self) -> bool:
        """Whether the run stopped on a tolerance of L-BFGS (M-SYN-LBFGS) rather than on
        ``max_iter`` or on the evaluation budget."""
        return self._converged

    @property
    def checkpoints(self) -> tuple[str, ...]:
        """The checkpoint files this call wrote, in order; the last one resumes the run."""
        return self._checkpoints

    @property
    def manifest(self) -> dict[str, Any]:
        """The settings of the run: the entry, the backend, dtype and device, the seed, the
        shapes, the operator's J and L, the settings of L-BFGS, the weights, the versions of
        ESTEL and of its references, and the checkpoint the run was resumed from."""
        return dict(self._manifest)

    def __repr__(self) -> str:
        shape = tuple(int(n) for n in self._field.shape)
        return (
            f"SynthesisResult(field shape={shape}, n_iter={self.n_iter}, "
            f"n_evals={self._n_evals}, final loss={self._final_loss()}, "
            f"converged={self._converged})"
        )

    def _final_loss(self) -> float | None:
        return float(self._losses[-1]) if self._losses.shape[0] else None


# ---------------------------------------------------------------------- the two entries


def synthesize_from_maps(
    target: Any,
    *,
    n_batch: int = 1,
    running_shape: tuple[int, int] | None = None,
    init: Any = "gaussian",
    mean_field: bool = True,
    compute_PS: bool | None = None,  # noqa: N803 (STL's name)
    max_iter: int = 100,
    history_size: int = 50,
    lr: float = 1.0,
    seed: int = 0,
    st_kwargs: Mapping[str, Any] | None = None,
    checkpoint_every: int | None = None,
    checkpoint_dir: str | os.PathLike[str] | None = None,
    resume_from: str | os.PathLike[str] | None = None,
) -> SynthesisResult:
    """Synthesise ``n_batch`` fields whose statistics match those of the target maps (maths.md
    section 5; STL's ``synthesize_from_maps``).

    Parameters
    ----------
    target:
        The target maps: an ``estel.Data2D``, periodic and without missing values, or an array
        of shape ``(N1, N2)``, ``(Nc, N1, N2)`` or ``(Nb, Nc, N1, N2)``, taken as periodic.
        Non-periodic and masked targets come later (LF47).
    n_batch:
        The number of fields synthesised. With ``mean_field=False`` it must be the target's Nb.
    running_shape:
        The spatial shape ``(N1, N2)`` of the synthesised fields; by default the target's.
    init:
        ``"gaussian"`` (the default): a Gaussian white noise drawn with ``seed``; or an array
        that broadcasts to ``(n_batch, Nc, N1, N2)``, in the standardised units of the target.
        Either is then pre-filtered (M-SYN-INIT, M-SYN-NYQ).
    mean_field:
        Compare the statistics averaged over the batch (True, the default), so that
        ``n_batch`` may differ from the target's Nb; or map by map (M-SYN-TARGET, M-SYN-LOSS).
    compute_PS:
        Whether the 2D power spectrum enters the loss: True by default in STL, for periodic
        fields (M-SYN-OPS). The spectra come in task T4.1: pass ``compute_PS=False``.
    max_iter:
        The iterations of L-BFGS of the whole run, those before ``resume_from`` included; the
        evaluations of the loss are capped at ⌊5 x ``max_iter`` / 4⌋ (M-SYN-LBFGS, OP-74).
    history_size:
        The correction pairs L-BFGS keeps: 50, STL's (decision of the maintainer on #28).
    lr:
        The step size of L-BFGS: 1.0, STL's.
    seed:
        The seed of the Gaussian start (Part I, §3.1, rule 5). ESTEL has no global seed; STL's
        default seed is 26, and its draws differ from ESTEL's (OP-55).
    st_kwargs:
        Options of the scattering operators: ``J`` only, as in STL, by default the smaller of
        the default J of the target and of the synthesised fields; L and the channel pairs are
        the defaults (M-SYN-OPS). Any other key raises an error.
    checkpoint_every, checkpoint_dir:
        Write a checkpoint to ``checkpoint_dir`` every ``checkpoint_every`` iterations, and at
        the end of the run: ``checkpoint_000010.npz`` after 10 iterations (maths.md §5.8).
        Give both or neither.
    resume_from:
        A checkpoint file: the run continues from it with the same trajectory, on the same
        device, up to ``max_iter`` iterations in all. The target and the settings must be those
        of the run that wrote it. With ``max_iter`` reached, the checkpoint is the result.

    Returns
    -------
    SynthesisResult
        The fields, de-standardised with the target's mean and standard deviation, the losses,
        the counts, the checkpoints written and the manifest of the run (M-SYN-OUT).
    """
    _refuse_tensor_product(target, "synthesize_from_maps", "target")
    data, ndim = _target_data(target)
    backend = _check_backend(data.backend)
    _check_periodic(data, "target")
    _check_loss_compute_ps(compute_PS)
    options = _st_options(st_kwargs)
    n_batch = _positive_int("n_batch", n_batch)
    _check_flag("mean_field", mean_field)
    if not mean_field and n_batch != data.n_batch:
        raise ValueError(
            f"mean_field=False compares the statistics map by map, so n_batch ({n_batch}) must "
            f"be the number of target maps ({data.n_batch}) (M-SYN-LOSS). Pass n_batch="
            f"{data.n_batch}, or mean_field=True."
        )
    shape = _running_shape(running_shape, data)
    dtype = default_dtype()

    # The target (M-SYN-TARGET): pre-filtered, standardised, then its statistics with
    # norm="self", reweighted and flattened.
    filtered = nyquist_filter(backend, data.array)
    standardized = standardize_target(backend, filtered, mean_field)
    target_data = data.with_array(standardized.x)
    running = _running_data(data, (n_batch, data.n_channels, *shape), backend)
    J = options.get("J")  # noqa: N806 (maths.md notation)
    if J is None:
        J = min(default_scales(target_data), default_scales(running))  # noqa: N806
    target_op = STOperator(target_data, J, norm="self", ref_batch_mean=mean_field, compute_PS=False)
    target_stats = target_op.apply(target_data)
    target_vector = backend.stop_gradient(flatten(target_stats, mean_field, False))
    reference = target_stats._reference
    assert reference is not None  # norm="self"
    running_op = STOperator(running, J, norm="ref", compute_PS=False)
    objective = Objective(
        template=running,
        plan=running_op.plan,
        filters=running_op.filters,
        reference=reference,
        target=target_vector,
        mean_field=mean_field,
        keep_batch_dim=False,
        divisor=1,
    )
    settings = {
        **_settings(
            "maps", backend, dtype, running, running_op, mean_field, max_iter, history_size, lr
        ),
        "target_sha256": _sha256(backend, target_vector),
    }
    u, losses, n_evals, converged, written, start = _optimize(
        backend,
        objective,
        running,
        init,
        seed,
        settings,
        checkpoint_every,
        checkpoint_dir,
        resume_from,
    )
    field = unstandardize(u, standardized.mean, standardized.std)
    field = _drop_axes(field, n_batch, drop_channel=ndim == 2)
    return _result(field, losses, n_evals, converged, written, settings, start, resume_from)


def synthesize_from_stats(
    stats: STStatistics,
    *,
    n_batch: int,
    running_shape: tuple[int, int] | None = None,
    init: Any = None,
    mean_field: bool = True,
    max_iter: int = 100,
    history_size: int = 50,
    lr: float = 1.0,
    seed: int = 0,
    checkpoint_every: int | None = None,
    checkpoint_dir: str | os.PathLike[str] | None = None,
    resume_from: str | os.PathLike[str] | None = None,
) -> SynthesisResult:
    """Synthesise ``n_batch`` periodic fields whose statistics match ``stats`` (maths.md
    section 5; STL's ``synthesize_from_stats``).

    ``stats`` are 2D statistics computed with ``norm="self"``, neither reduced nor selected;
    their normalisation, ``ref_batch_mean`` and standardisation are those the caller chose
    (M-SYN-TARGET). The synthesised fields are periodic and without missing values, whatever
    the data of ``stats`` (LF47). Every option of the running operator, J, L and the channel
    pairs, comes from ``stats`` (OP-76). ``stats`` are not modified (OP-70).

    With ``mean_field=True`` (the default) the loss compares the batch means and is divided by
    the Nb of ``stats`` (OP-71); with ``mean_field=False``, ``n_batch`` must be that Nb. If
    ``stats`` were computed with ``standardize=True``, the fields are de-standardised with the
    removed means and standard deviations, averaged over the batch with ``mean_field``
    (M-SYN-OUT, OP-72); otherwise they are left in the units of the statistics.

    ``init`` is ``None`` or ``"gaussian"`` for a Gaussian start drawn with ``seed``, or an
    array that broadcasts to ``(n_batch, Nc, N1, N2)``. The other parameters are those of
    :func:`synthesize_from_maps`. The power spectrum joins with T4.1.
    """
    _refuse_tensor_product(stats, "synthesize_from_stats", "stats")
    if not isinstance(stats, STStatistics):
        raise TypeError(
            f"synthesize_from_stats takes STStatistics, got {type(stats).__name__}; to start "
            "from maps, use estel.synthesize_from_maps."
        )
    if stats._plan.geometry._geometry == "tensor_prod":
        raise NotImplementedError(
            "synthesize_from_stats: these are the statistics of a tensor product; syntheses on "
            "tensor products are not part of ESTEL 1.0.0, which synthesises 2D planar data "
            "(maths.md section 5). Pass the statistics of estel.Data2D maps, computed with "
            "estel.STOperator."
        )
    plan = cast(PlanarPlan, stats._plan)  # a single geometry: a tensor product raised above
    backend = _check_backend(stats._backend)
    if plan.ndim != 2:
        raise ValueError(
            f"synthesize_from_stats: these statistics are of {plan.ndim}D data; syntheses of "
            "ESTEL 1.0.0 are of 2D planar data (maths.md section 5)."
        )
    if plan.norm != "self" or stats._reference is None:
        raise ValueError(
            f'synthesize_from_stats: these statistics were computed with norm="{plan.norm}"; '
            'a synthesis needs the reference of norm="self" (M-SYN-TARGET). Compute them '
            'with STOperator(..., norm="self").'
        )
    complete = {g.name for g in stats._grids} == {"mean", "var", "S1", "S2", "S3", "S4"}
    if stats._reductions != _NO_REDUCTION or not complete:
        raise ValueError(
            "synthesize_from_stats: these statistics are reduced or selected; a synthesis "
            "compares the full statistics of M-SYN-LOSS. Pass the statistics as "
            "STOperator.apply returns them."
        )
    n_batch = _positive_int("n_batch", n_batch)
    _check_flag("mean_field", mean_field)
    n_target = int(stats.full_grid("mean").shape[0])
    if not mean_field and n_batch != n_target:
        raise ValueError(
            f"mean_field=False compares the statistics map by map, so n_batch ({n_batch}) must "
            f"be the Nb of the statistics ({n_target}) (M-SYN-LOSS). Pass n_batch={n_target}, "
            "or mean_field=True."
        )
    if stats._reference.n_batch not in (1, n_batch):
        raise ValueError(
            f"these statistics were normalised by a reference of each of their {n_target} maps "
            f"(ref_batch_mean=False), which cannot normalise n_batch={n_batch} fields (M-NORM). "
            f"Pass n_batch={n_target}, or compute the statistics with ref_batch_mean=True."
        )
    shape = _running_shape(running_shape, None, plan.spatial_shape)
    dtype = default_dtype()
    template = plan.geometry._from_parts(
        backend.xp.zeros(
            (n_batch, plan.n_channels, *shape),
            dtype=_xp_dtype(backend, dtype),
            device=device_of(stats.full_grid("mean")),
        ),
        None,
        True,
        backend.name,
    )
    running_op = STOperator(
        template,
        plan.J,
        plan.L,
        norm="ref",
        cross_matrix=plan.cross_matrix,
        compute_PS=False,
    )
    target_vector = backend.stop_gradient(flatten(stats, mean_field, True))
    objective = Objective(
        template=template,
        plan=running_op.plan,
        filters=running_op.filters,
        reference=stats._reference,
        target=target_vector,
        mean_field=mean_field,
        keep_batch_dim=True,
        divisor=n_target if mean_field else 1,
    )
    settings = {
        **_settings(
            "stats", backend, dtype, template, running_op, mean_field, max_iter, history_size, lr
        ),
        "target_sha256": _sha256(backend, target_vector),
    }
    u, losses, n_evals, converged, written, start = _optimize(
        backend,
        objective,
        template,
        "gaussian" if init is None else init,
        seed,
        settings,
        checkpoint_every,
        checkpoint_dir,
        resume_from,
    )
    if stats._standardization is not None:
        mean, std = stats._standardization
        if mean_field:
            mean, std = backend.xp.mean(mean, axis=0), backend.xp.mean(std, axis=0)
        u = unstandardize(u, mean, std)
    field = _drop_axes(u, n_batch, drop_channel=plan.n_channels == 1)
    return _result(field, losses, n_evals, converged, written, settings, start, resume_from)


# ---------------------------------------------------------------------- the driver


def _optimize(
    backend: Backend,
    objective: Objective,
    running: Data,
    init: Any,
    seed: int,
    settings: dict[str, Any],
    every: int | None,
    directory: str | os.PathLike[str] | None,
    resume_from: str | os.PathLike[str] | None,
) -> tuple[Any, NDArray[Any], int, bool, tuple[str, ...], dict[str, Any]]:
    """Run L-BFGS (M-SYN-LBFGS) in chunks between checkpoints, each resuming from the state
    of the last, with what is left of the iterations and of the evaluation budget: the
    iterates are those of a single call. Returns u, the losses, the evaluations, whether the
    run converged, the files written and the start (seed or checkpoint)."""
    max_iter = _positive_int("max_iter", settings["max_iter"])
    every = _checkpoint_settings(every, directory)
    budget = max_iter * 5 // 4
    losses: list[NDArray[Any]] = []
    if resume_from is not None:
        checkpoint = _checkpoints.load(resume_from)
        _check_same_run(checkpoint.meta, settings, resume_from)
        x = backend.asarray(checkpoint.field, settings["dtype"], device_of(running.array))
        state: LbfgsState | None = checkpoint.state
        losses.append(np.asarray(checkpoint.losses))
        done, evals = int(checkpoint.meta["n_iter"]), int(checkpoint.meta["n_evals"])
        converged = bool(checkpoint.meta["converged"])
        seed = int(checkpoint.meta["seed"])
    else:
        x = nyquist_filter(backend, _start(backend, init, seed, running, settings["dtype"]))
        state, done, evals, converged = None, 0, 0, False
    written: list[str] = []
    while not converged and done < max_iter and evals < budget:
        chunk = max_iter - done if every is None else min(every - done % every, max_iter - done)
        result = backend.lbfgs(
            objective,
            x,
            lr=settings["lr"],
            max_iter=chunk,
            max_eval=budget - evals,
            history_size=settings["history_size"],
            tolerance_grad=TOLERANCE_GRAD,
            tolerance_change=TOLERANCE_CHANGE,
            state=state,
        )
        n = int(result.losses.shape[0])
        x, state = result.x, result.state
        losses.append(np.asarray(backend.to_numpy(result.losses)))
        done, evals = done + n, evals + result.n_evals
        converged = result.converged
        over = converged or done >= max_iter or evals >= budget
        if every is not None and directory is not None and n and (done % every == 0 or over):
            path = _checkpoints.path_for(directory, done)
            meta = {**settings, "n_iter": done, "n_evals": evals, "converged": converged}
            _checkpoints.save(
                path,
                _checkpoints.Checkpoint(
                    field=np.asarray(backend.to_numpy(x)),
                    losses=np.concatenate(losses),
                    state=state,
                    meta={**meta, "seed": seed},
                ),
            )
            written.append(str(path))
    all_losses = np.concatenate(losses) if losses else np.zeros((0,))
    return x, all_losses, evals, converged, tuple(written), {"seed": seed}


def _start(backend: Backend, init: Any, seed: int, running: Data, dtype: Any) -> Any:
    """The start before the pre-filter (M-SYN-INIT): a Gaussian noise drawn with ``seed``, or
    ``init`` broadcast to the running shape, on the device of the running data, where the
    operators compute."""
    shape = running.shape
    device = device_of(running.array)
    if isinstance(init, str):
        if init != "gaussian":
            raise ValueError(
                f'init={init!r}: pass "gaussian" for a Gaussian start, or an array of the '
                f"fields to start from, which broadcasts to {shape}."
            )
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise TypeError(f"seed={seed!r}: pass an int in [0, 2**63).")
        noise = gaussian_start(backend, seed, shape, dtype)
        if device is not None and device_of(noise) != device:
            # random_normal draws on the backend's own default device.
            noise = backend.asarray(backend.to_numpy(noise), dtype, device)
        return noise
    name = _registry.backend_of(init)
    values = np.asarray(init if name is None else _registry.backend_named(name).to_numpy(init))
    if values.dtype.kind not in "biuf":
        raise TypeError(f"init: pass real numbers, got an array of dtype {values.dtype}.")
    try:
        values = np.broadcast_to(values.astype(np.float64), shape)
    except ValueError as error:
        raise ValueError(
            f"init has shape {values.shape}, which does not broadcast to the running fields "
            f"{shape} (n_batch, Nc, N1, N2)."
        ) from error
    if not np.isfinite(values).all():
        raise ValueError(
            "init has NaN or infinite values; syntheses with missing values come later "
            "(LF47). Pass finite values."
        )
    return backend.asarray(values, dtype, device)


# ---------------------------------------------------------------------- checks and helpers


def _refuse_tensor_product(value: object, function: str, name: str) -> None:
    """Syntheses on tensor products are not part of ESTEL 1.0.0: refused before any other
    check, since a ``DataTensorProd`` of one 2D block has ``ndim == 2`` and its ``pbc`` is a
    tuple, which the checks of planar data would let through. Read from the geometry the class
    declares, as L2 knows the mother class only (Part I, §2.4)."""
    if isinstance(value, Data) and value._geometry == "tensor_prod":
        advice = (
            "Pass estel.Data2D maps, or a single 2D field as an array."
            if function == "synthesize_from_maps"
            else "Pass the statistics of estel.Data2D maps, computed with estel.STOperator."
        )
        raise NotImplementedError(
            f"{function}: the {name} is a DataTensorProd; syntheses on tensor products are not "
            f"part of ESTEL 1.0.0, which synthesises 2D planar data (maths.md section 5). {advice}"
        )


def _target_data(target: Any) -> tuple[Data, int]:
    """The target as a ``Data`` object, and the number of axes it was given with: an array is
    taken as periodic 2D data (M-SYN-TARGET, M-DATA-2D)."""
    if isinstance(target, Data):
        if target.ndim != 2:
            raise ValueError(
                f"synthesize_from_maps: the target is {type(target).__name__}; syntheses of "
                "ESTEL 1.0.0 are of 2D planar data, estel.Data2D (maths.md section 5)."
            )
        return target, 4
    if isinstance(target, STStatistics):
        raise TypeError(
            "synthesize_from_maps takes maps; to start from statistics, use "
            "estel.synthesize_from_stats."
        )
    data = geometry_class("2d")(target, pbc=True)
    return data, len(np.shape(target)) if _registry.backend_of(target) is None else target.ndim


def _check_backend(name: str) -> Backend:
    backend = _registry.backend_named(name)
    if "lbfgs" in getattr(backend, "not_provided", frozenset()):
        raise NotImplementedError(
            f"Syntheses need the L-BFGS of the backend, which the {name} backend provides from "
            "step 2 (LF1). Run syntheses on the torch backend: estel.set_backend('torch')."
        )
    return backend


def _check_periodic(data: Data, what: str) -> None:
    if not data.pbc or data.mask is not None:
        raise ValueError(
            f"the {what} is {'not periodic' if not data.pbc else 'masked'}: syntheses of ESTEL "
            "1.0.0 are periodic, without missing values; non-periodic and masked syntheses "
            "come later (LF47). Pass periodic data (pbc=True) without NaN."
        )


def _check_loss_compute_ps(compute_ps: object) -> None:
    if compute_ps is not None and not isinstance(compute_ps, bool | np.bool_):
        raise TypeError(f"compute_PS={compute_ps!r}: pass True, False or None (the default).")
    if compute_ps is None or compute_ps:
        raise NotImplementedError(
            "compute_PS: the 2D power spectrum enters the loss of periodic syntheses by default "
            "(maths.md M-SYN-OPS), and the spectra come in task T4.1. Pass compute_PS=False."
        )


def _check_flag(name: str, value: object) -> None:
    if not isinstance(value, bool | np.bool_):
        raise TypeError(f"{name}={value!r}: pass True or False.")


def _positive_int(name: str, value: object) -> int:
    if not isinstance(value, int | np.integer) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name}={value!r}: pass an int of at least 1.")
    return int(value)


def _st_options(st_kwargs: Mapping[str, Any] | None) -> dict[str, Any]:
    options = dict(st_kwargs or {})
    unknown = sorted(set(options) - set(ST_KWARGS))
    if unknown:
        raise ValueError(
            f"st_kwargs: {unknown} are not accepted; a synthesis takes {list(ST_KWARGS)} only, "
            "as STL's: L and the channel pairs are the defaults, and the synthesis sets the "
            "normalisation, the standardisation and the spectrum itself (maths.md M-SYN-OPS). "
            "Pass st_kwargs={'J': ...}, or nothing."
        )
    return options


def _running_shape(
    running_shape: Any, data: Data | None, default: tuple[int, ...] | None = None
) -> tuple[int, int]:
    if running_shape is None:
        spatial = data.shape[2:] if data is not None else default
        assert spatial is not None
        return int(spatial[0]), int(spatial[1])
    shape = tuple(running_shape)
    if len(shape) != 2 or not all(
        isinstance(n, int | np.integer) and not isinstance(n, bool) and n > 0 for n in shape
    ):
        raise ValueError(f"running_shape={running_shape!r}: pass (N1, N2), two positive ints.")
    return int(shape[0]), int(shape[1])


def _xp_dtype(backend: Backend, dtype: str) -> Any:
    return getattr(backend.xp, dtype)


def _running_data(target: Data, shape: tuple[int, ...], backend: Backend) -> Data:
    """Periodic data of the running fields' shape, of the target's geometry, without mask."""
    zeros = backend.xp.zeros(shape, dtype=target.array.dtype, device=target.array.device)
    return type(target)._from_parts(zeros, None, True, backend.name)


def _checkpoint_settings(every: object, directory: object) -> int | None:
    if (every is None) != (directory is None):
        raise ValueError(
            "checkpoint_every and checkpoint_dir go together: pass both, to write a checkpoint "
            "every checkpoint_every iterations to checkpoint_dir, or neither."
        )
    return None if every is None else _positive_int("checkpoint_every", every)


def _settings(
    entry: str,
    backend: Backend,
    dtype: str,
    running: Data,
    operator: STOperator,
    mean_field: bool,
    max_iter: int,
    history_size: int,
    lr: float,
) -> dict[str, Any]:
    if not isinstance(lr, int | float) or isinstance(lr, bool) or not lr > 0:
        raise ValueError(f"lr={lr!r}: pass a positive step size (STL's: 1.0).")
    return {
        "entry": entry,
        "backend": backend.name,
        "dtype": dtype,
        "running_shape": list(running.shape),
        "mean_field": bool(mean_field),
        "J": operator.J,
        "L": operator.L,
        "lr": float(lr),
        "max_iter": max_iter,
        "history_size": _positive_int("history_size", history_size),
        "tolerance_grad": TOLERANCE_GRAD,
        "tolerance_change": TOLERANCE_CHANGE,
        "weights": dict(WEIGHTS),
        "prefilter_nyquist": True,
    }


def _sha256(backend: Backend, vector: Any) -> str:
    return hashlib.sha256(np.ascontiguousarray(backend.to_numpy(vector)).tobytes()).hexdigest()


def _check_same_run(
    meta: dict[str, Any], settings: dict[str, Any], path: str | os.PathLike[str]
) -> None:
    differ = [
        f"{key} {meta.get(key)!r} (now {settings[key]!r})"
        for key in _SAME_RUN
        if meta.get(key) != settings[key]
    ]
    if differ:
        raise ValueError(
            f"resume_from={str(path)!r} was written by another run: {'; '.join(differ)}. "
            "Resume with the target and the settings of that run; only max_iter, "
            "checkpoint_every and checkpoint_dir may change."
        )


def _drop_axes(field: Any, n_batch: int, drop_channel: bool) -> Any:
    """STL's axes of the result (M-SYN-OUT, OP-75)."""
    if drop_channel:
        field = field[:, 0]
    if n_batch == 1:
        field = field[0]
    return field


def _result(
    field: Any,
    losses: NDArray[Any],
    n_evals: int,
    converged: bool,
    written: tuple[str, ...],
    settings: dict[str, Any],
    start: dict[str, Any],
    resume_from: str | os.PathLike[str] | None,
) -> SynthesisResult:
    manifest = {
        **{k: v for k, v in settings.items() if k != "target_sha256"},
        **start,
        "device": str(device_of(field)),
        "resumed_from": None if resume_from is None else str(Path(resume_from)),
        "estel": importlib.metadata.version("estel"),
        "statistics_reference": STATISTICS_REFERENCE,
        "synthesis_reference": SYNTHESIS_REFERENCE,
    }
    return SynthesisResult._create(
        field=field,
        losses=losses,
        n_evals=n_evals,
        converged=converged,
        checkpoints=written,
        manifest=manifest,
    )
