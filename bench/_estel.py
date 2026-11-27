"""A benchmark case in ESTEL (task T2.11, issue #89): the forward pass and the gradient of the
loss, on the backend the process selected."""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

import estel
from bench._cases import Inputs
from estel.backend._registry import current_backend

FAMILIES = ("mean", "var", "S1", "S2", "S3", "S4")


def configure(backend: str, dtype: str, device: str, threads: int) -> dict[str, str]:
    """Select the backend, the dtype, the device and the number of threads of this process, and
    return the versions of the libraries that compute."""
    estel.set_backend(backend)
    estel.set_default_dtype(dtype)  # type: ignore[arg-type]
    estel.set_default_device(device)
    if backend == "torch":
        import torch

        torch.set_num_threads(threads)
        return {"torch": torch.__version__}
    import jax

    # JAX's CPU thread pool has no setting of its own: the measured times on JAX are not yet at a
    # fixed number of threads. Measuring JAX (compile time, jit, threads) is T2.13's.
    return {"jax": jax.__version__}


BLOCKS = {"2D": estel.Data2D, "1D": estel.Data1D}


def _data(inputs: Inputs, field: NDArray[Any]) -> estel.Data:
    """The data of a case: ``Data2D``, the ``DataTensorProd`` of its blocks (task T3.2), or a
    ``DataHealpix`` full sky (task T5.7)."""
    case = inputs.case
    if case.tensor_prod:
        blocks = [BLOCKS[name] for name in case.blocks]
        return estel.DataTensorProd(field, blocks, pbc=list(cast(Any, case.pbc)))
    if case.healpix:
        return estel.DataHealpix(field, mask=inputs.mask)
    return estel.Data2D(field, mask=inputs.mask, pbc=bool(case.pbc))


class Run:
    """The operator, the data and the loss of a case, built once; the two measured calls."""

    def __init__(self, inputs: Inputs, built: Callable[[], None] | None = None) -> None:
        """``built``, if given, is called once the operator is built (the peak memory of the
        build); the construction of the operator is timed in ``build_seconds``."""
        case = inputs.case
        self._backend = current_backend()
        self.data = _data(inputs, inputs.field)
        with warnings.catch_warnings():
            # OP-3 and OP-13: a non-periodic case at a large J has coarse means without a valid
            # pixel, which STL returns as NaN and ESTEL leaves out; the check against STL
            # compares their positions. OP-34 for a tensor product.
            warnings.simplefilter("ignore", UserWarning)
            self.operator: Any
            J, L = cast(Any, case.J), cast(Any, case.L)  # noqa: N806 (maths.md notation)
            start = time.perf_counter()
            if case.tensor_prod:
                self.operator = estel.STOperatorTensorProd(
                    self.data, list(J), list(L), standardize=False
                )
            else:
                self.operator = estel.STOperator(
                    self.data, J, L, standardize=False, compute_PS=False
                )
            self.build_seconds = time.perf_counter() - start
        if built is not None:
            built()
        target_data = _data(inputs, inputs.target_field)
        # The first call of the operator: on JAX it compiles the loop (task T2.13).
        start = time.perf_counter()
        first = self._done(self.operator.apply(target_data, norm="none"))
        self.first_apply_seconds = time.perf_counter() - start
        target = self._backend.stop_gradient(first.to_vector())
        xp = self._backend.xp
        operator, data = self.operator, self.data

        def loss(array: Any) -> Any:
            vector = operator.apply(data.with_array(array), norm="none").to_vector()
            return xp.sum((vector - target) ** 2)

        self._value_and_grad = self._backend.value_and_grad(loss)

    def forward(self) -> Any:
        """``STOperator.apply`` on the field: the statistics."""
        return self._done(self.operator.apply(self.data, norm="none"))

    def gradient(self) -> Any:
        """The loss ``||to_vector(S(x)) - t||²`` and its gradient with respect to the field."""
        return self._done(self._value_and_grad(self.data.array))

    def _done(self, result: Any) -> Any:
        if self._backend.name == "jax":
            import jax

            # JAX may return before the computation ends.
            jax.block_until_ready(result)  # type: ignore[no-untyped-call]
        return result

    def arrays(self, statistics: Any, loss_and_gradient: Any) -> dict[str, NDArray[Any]]:
        """The results of the two calls as NumPy arrays: each family in STL's full-grid shape,
        the flat vector, the loss and its gradient."""
        to_numpy = self._backend.to_numpy
        out = {family: to_numpy(statistics.full_grid(family)) for family in FAMILIES}
        out["vector"] = to_numpy(statistics.to_vector())
        loss, gradient = loss_and_gradient
        out["loss"] = np.asarray(to_numpy(loss))
        out["gradient"] = to_numpy(gradient)
        return out
