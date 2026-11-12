"""L-BFGS on PyTorch: ``torch.optim.LBFGS`` with a strong Wolfe line search (Part II, §12.3).

The driver runs ``LBFGS.step`` one iteration at a time, so that it can record the loss after
each iteration and hand back a state that resumes the same run (synthesis checkpoints,
Part II, §12.9). The optimizer keeps its history between steps, so the iterates are those of
a single ``step`` with ``max_iter`` iterations. The stopping rules are torch's own, checked
between iterations:

* the largest gradient component at the current iterate is at most ``tolerance_grad``;
* the last step, ``max |t · d|``, is at most ``tolerance_change``;
* the last change of loss is below ``tolerance_change``.

This is the optimizer's driver, not traced computation: it may read values to stop.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from .._protocol import LbfgsResult


@dataclass(frozen=True)
class TorchLbfgsState:
    """The history of a PyTorch L-BFGS run, opaque to callers; pass it back to resume."""

    optimizer_state: dict[str, Any]


class _ConvergedError(Exception):
    """Raised by the closure, before any update, when the loss stopped changing."""


def lbfgs(
    fun: Callable[[torch.Tensor], torch.Tensor],
    x0: torch.Tensor,
    *,
    max_iter: int,
    history_size: int,
    tolerance_grad: float,
    tolerance_change: float,
    state: Any = None,
) -> LbfgsResult:
    if max_iter < 1 or history_size < 1:
        raise ValueError(
            f"max_iter ({max_iter}) and history_size ({history_size}) must be at least 1."
        )
    if state is not None and not isinstance(state, TorchLbfgsState):
        raise TypeError(
            f"state is a {type(state).__name__}, not the state of a PyTorch L-BFGS run. Pass "
            "the state of a result of this backend's lbfgs, or None to start a new run."
        )
    x = x0.detach().clone().requires_grad_(True)
    optimizer = torch.optim.LBFGS(
        [x],
        lr=1,
        max_iter=1,
        history_size=history_size,
        tolerance_grad=tolerance_grad,
        tolerance_change=tolerance_change,
        line_search_fn="strong_wolfe",
    )
    if state is not None:
        optimizer.state[x] = copy.deepcopy(state.optimizer_state)
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

    # torch's evaluation budget for one step of max_iter iterations (its default max_eval).
    # The line search of each iteration may use what is left of it, as in a single step; the
    # evaluation that opens each of our steps re-reads the current iterate and is not counted.
    budget = max_iter * 5 // 4
    used = 0
    for iteration in range(max_iter):
        iterations_before = history.get("n_iter", 0)
        evals_before = history.get("func_evals", 0)
        opening = 1 if iteration == 0 else 0
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
        if float(step) <= tolerance_change or used >= budget:
            values.append(_loss(fun, x))
            break
    else:
        values.append(_loss(fun, x))

    losses = values[1:]
    loss_array = (
        torch.stack(losses) if losses else torch.zeros((0,), dtype=x0.dtype, device=x0.device)
    )
    return LbfgsResult(
        x=x.detach().clone(),
        state=TorchLbfgsState(copy.deepcopy(dict(history))),
        losses=loss_array.to(x0.dtype),
    )


def _loss(fun: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return fun(x).detach().clone()
