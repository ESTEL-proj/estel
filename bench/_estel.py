"""A benchmark case in ESTEL (task T2.11, issue #89): the forward pass and the gradient of the
loss, on the backend the process selected."""

from __future__ import annotations

import warnings
from typing import Any

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


class Run:
    """The operator, the data and the loss of a case, built once; the two measured calls."""

    def __init__(self, inputs: Inputs) -> None:
        case = inputs.case
        self._backend = current_backend()
        self.data = estel.Data2D(inputs.field, mask=inputs.mask, pbc=case.pbc)
        with warnings.catch_warnings():
            # OP-3 and OP-13: a non-periodic case at a large J has coarse means without a valid
            # pixel, which STL returns as NaN and ESTEL leaves out; the check against STL
            # compares their positions.
            warnings.simplefilter("ignore", UserWarning)
            self.operator = estel.STOperator(
                self.data, case.J, case.L, norm="none", compute_PS=False
            )
        target_data = estel.Data2D(inputs.target_field, mask=inputs.mask, pbc=case.pbc)
        target = self._backend.stop_gradient(self.operator.apply(target_data).to_vector())
        xp = self._backend.xp
        operator, data = self.operator, self.data

        def loss(array: Any) -> Any:
            vector = operator.apply(data.with_array(array)).to_vector()
            return xp.sum((vector - target) ** 2)

        self._value_and_grad = self._backend.value_and_grad(loss)

    def forward(self) -> Any:
        """``STOperator.apply`` on the field: the statistics."""
        return self._done(self.operator.apply(self.data))

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
