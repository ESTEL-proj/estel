"""L-BFGS on PyTorch: ``torch.optim.LBFGS`` with a strong Wolfe line search (Part II, §12.3;
maths.md M-SYN-LBFGS).

The driver runs ``LBFGS.step`` one iteration at a time, so that it can record the loss after
each iteration and hand back a state that resumes the same run (synthesis checkpoints,
Part II, §12.9). The optimizer keeps its history between steps, so the iterates are those of
a single ``step`` with ``max_iter`` iterations and ``max_eval`` evaluations. The stopping rules
are torch's own, checked between iterations:

* the largest gradient component at the current iterate is at most ``tolerance_grad``;
* the last step, ``max |t · d|``, is at most ``tolerance_change``;
* the last change of loss is below ``tolerance_change``;
* the evaluations of the loss reach ``max_eval``.

A run stopped by one of the first three is converged: its state says so (``"converged"``), and
a call that resumes from it returns at once, without iterating, where a single step would have
stopped.

The state handed back is the optimizer's history as NumPy arrays and Python scalars
(``estel.backend._protocol.LbfgsState``), so that a caller can save it; a resumed run rebuilds
the history on the device and in the dtype of its starting point, value for value.

This is the optimizer's driver, not traced computation: it may read values to stop.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from .._protocol import LbfgsResult, LbfgsState

BACKEND = "torch"
# The tensors of torch's history, and the lists of tensors (one entry per correction pair).
_TENSORS = ("d", "prev_flat_grad")
_PAIRS = ("old_dirs", "old_stps", "ro")
# Entries that are a tensor or a Python number, depending on the iteration.
_NUMBERS = ("t", "H_diag", "prev_loss")
_COUNTS = ("n_iter", "func_evals")


class _ConvergedError(Exception):
    """Raised by the closure, before any update, when the loss stopped changing."""


def lbfgs(
    fun: Callable[[torch.Tensor], torch.Tensor],
    x0: torch.Tensor,
    *,
    lr: float,
    max_iter: int,
    max_eval: int | None,
    history_size: int,
    tolerance_grad: float,
    tolerance_change: float,
    state: LbfgsState | None = None,
) -> LbfgsResult:
    # torch's evaluation budget for one step of max_iter iterations (its default max_eval).
    budget = max_iter * 5 // 4 if max_eval is None else max_eval
    if max_iter < 1 or history_size < 1 or budget < 1:
        raise ValueError(
            f"max_iter ({max_iter}), history_size ({history_size}) and max_eval ({budget}) must "
            "be at least 1."
        )
    if state is not None and (not isinstance(state, dict) or state.get("backend") != BACKEND):
        found = state.get("backend") if isinstance(state, dict) else type(state).__name__
        raise TypeError(
            f"state comes from {found!r}, not from a PyTorch L-BFGS run. Pass the state of a "
            "result of this backend's lbfgs, or None to start a new run."
        )
    x = x0.detach().clone().requires_grad_(True)
    if state is not None and state.get("converged"):
        # The run stopped on one of its tolerances: a single step would have stopped there.
        empty = torch.zeros((0,), dtype=x0.dtype, device=x0.device)
        return LbfgsResult(x=x.detach().clone(), state=dict(state), losses=empty, n_evals=0)
    optimizer = torch.optim.LBFGS(
        [x],
        lr=lr,
        max_iter=1,
        history_size=history_size,
        tolerance_grad=tolerance_grad,
        tolerance_change=tolerance_change,
        line_search_fn="strong_wolfe",
    )
    if state is not None:
        optimizer.state[x] = _history_from_numpy(state, x0)
    history = optimizer.state[x]

    # Loss at each iterate: values[0] at x0, values[i] after iteration i.
    values: list[torch.Tensor] = []

    def closure_for_step() -> Callable[[], torch.Tensor]:
        first_call = True

        def closure() -> torch.Tensor:
            nonlocal first_call
            optimizer.zero_grad()
            loss = fun(x)
            loss.backward()  # type: ignore[no-untyped-call]
            if first_call:
                first_call = False
                values.append(loss.detach().clone())
                # torch keeps the loss before the last iteration as "prev_loss", also in a
                # resumed state, so a resumed run stops where an uninterrupted one would.
                previous = history.get("prev_loss")
                if previous is not None and abs(float(loss.detach()) - previous) < tolerance_change:
                    raise _ConvergedError
            return loss

        return closure

    # The line search of each iteration may use what is left of the budget, as in a single
    # step. The evaluation that opens each of our steps re-reads the current iterate and is not
    # counted, except the first of a new run, which is torch's evaluation at x0.
    used = 0
    converged = True  # False when the run stops on max_iter or on the budget
    for iteration in range(max_iter):
        iterations_before = history.get("n_iter", 0)
        evals_before = history.get("func_evals", 0)
        opening = 1 if iteration == 0 and state is None else 0
        optimizer.param_groups[0]["max_eval"] = budget - (used + opening) + 1
        try:
            optimizer.step(closure_for_step())  # type: ignore[no-untyped-call]
        except _ConvergedError:
            break
        if history["n_iter"] == iterations_before:
            break  # the gradient at the current iterate is below tolerance_grad
        line_search_evals = history["func_evals"] - evals_before - 1
        used += opening + line_search_evals
        if line_search_evals == 0:
            values.append(_loss(fun, x))
            break  # not a descent direction: torch stops without a line search
        step = history["d"].mul(history["t"]).abs().max()
        if float(step) <= tolerance_change:
            values.append(_loss(fun, x))
            break
        if used >= budget:
            values.append(_loss(fun, x))
            converged = False
            break
    else:
        values.append(_loss(fun, x))
        converged = False
    if state is None and not used and values:
        used = 1  # stopped at x0: torch's single step evaluated the loss there once

    losses = values[1:]
    loss_array = (
        torch.stack(losses) if losses else torch.zeros((0,), dtype=x0.dtype, device=x0.device)
    )
    return LbfgsResult(
        x=x.detach().clone(),
        state={**_history_to_numpy(history), "converged": converged},
        losses=loss_array.to(x0.dtype),
        n_evals=used,
    )


def _loss(fun: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return fun(x).detach().clone()


# ---------------------------------------------------------------------- the state as NumPy


def _number_to_numpy(value: Any) -> Any:
    """A tensor as a 0-d NumPy array of its dtype; a Python number as it is."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().copy()
    return value


def _history_to_numpy(history: dict[str, Any]) -> LbfgsState:
    """torch's history of the parameter as a flat dict of NumPy arrays and Python scalars.

    ``al`` is left out: torch writes each of its entries before reading it, at every
    iteration, and recreates it when it is absent.
    """
    out: LbfgsState = {"backend": BACKEND}
    for name in _COUNTS:
        if name in history:
            out[name] = int(history[name])
    for name in _TENSORS:
        if history.get(name) is not None:
            out[name] = history[name].detach().cpu().numpy().copy()
    for name in _NUMBERS:
        if history.get(name) is not None:
            out[name] = _number_to_numpy(history[name])
    if "old_dirs" in history:
        for name in _PAIRS:
            pairs = history[name]
            out[f"n_{name}"] = len(pairs)
            if pairs:
                out[name] = torch.stack([p.detach() for p in pairs]).cpu().numpy().copy()
    return out


def _history_from_numpy(state: LbfgsState, x0: torch.Tensor) -> dict[str, Any]:
    """The inverse of :func:`_history_to_numpy`, on the device of ``x0``: arrays become
    tensors of their own dtype, Python numbers stay numbers."""

    def tensor(array: Any) -> torch.Tensor:
        return torch.from_numpy(np.array(array, copy=True)).to(device=x0.device)

    history: dict[str, Any] = {}
    for name in _COUNTS:
        if name in state:
            history[name] = int(state[name])
    for name in _TENSORS:
        if name in state:
            history[name] = tensor(state[name])
    for name in _NUMBERS:
        if name in state:
            value = state[name]
            history[name] = tensor(value) if isinstance(value, np.ndarray) else value
    for name in _PAIRS:
        if f"n_{name}" in state:
            count = int(state[f"n_{name}"])
            history[name] = [tensor(state[name][i]) for i in range(count)] if count else []
    return history
