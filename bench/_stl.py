"""A benchmark case in STL (task T2.11, issue #89): the same case and the same loss as
``bench/_estel.py``, through STL's own functions.

Runs in the parity container, with the interpreter of the `master` environment
(tools/stl_parity/README.md). It imports STL, never ``estel``.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import subprocess
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from bench._cases import Inputs

FAMILIES = ("mean", "var", "S1", "S2", "S3", "S4")
# STL's threshold on the fraction of missing pixels, as the parity cases pass it
# (tools/stl_parity/cases/scattering_2d.py); ESTEL's Data2D has the same constant.
NAN_THRESHOLD = 0.33
STL_CHECKOUT = "/opt/stl/master"


def configure(backend: str, dtype: str, device: str, threads: int) -> dict[str, str]:
    """Set the dtype and the number of threads of this process, and return the versions of the
    libraries that compute. STL follows the dtype of its input, and builds some arrays at
    torch's default dtype (tools/stl_parity/README.md, *Precision*), so both are set."""
    if backend != "torch" or device != "cpu":
        raise ValueError(
            f"STL is measured on PyTorch and on CPU, got backend={backend!r}, device={device!r}: "
            "run it with --backend torch and device = 'cpu' in the cases file."
        )
    torch.set_default_dtype(getattr(torch, dtype))
    torch.set_num_threads(threads)
    # The checkout belongs to root and the process runs as the caller: tell git it is safe.
    command = [
        "git",
        "-c",
        f"safe.directory={STL_CHECKOUT}",
        "-C",
        STL_CHECKOUT,
        "rev-parse",
        "HEAD",
    ]
    commit = subprocess.run(command, capture_output=True, text=True).stdout.strip()
    return {
        "torch": torch.__version__,
        "STL": importlib.metadata.version("STL"),
        "STL commit": commit or "unknown",
    }


class Run:
    """The operator, the data and the loss of a case, built once; the two measured calls."""

    def __init__(self, inputs: Inputs) -> None:
        case = inputs.case
        self._data_cls = importlib.import_module("STL_main.STL_2D_Kernel_Torch").STL_2D_Kernel_Torch
        self._pbc = case.pbc
        self._field = torch.from_numpy(inputs.field)
        options: dict[str, Any] = {}
        if inputs.mask is not None:
            # As in STL's syntheses: the mask belongs to the operator, and the field keeps
            # finite values at the missing pixels, which are not read.
            options["mask_full_res"] = self._data_cls(array=torch.from_numpy(inputs.mask))
        self.data = self._data_cls(array=self._field, pbc=case.pbc)
        self.operator = self.data.get_ST_op(
            J=case.J,
            L=case.L,
            norm="vanilla",
            compute_PS=False,
            downsample_nan_weight_threshold=NAN_THRESHOLD,
            # As STL's syntheses pass it: the coarse pixels declared invalid keep their value
            # instead of NaN, so the gradient stays finite; the statistics do not change, since
            # the masked means never read these pixels (maths.md OP-60, OP-27).
            replace_nan_value=None,
            **options,
        )
        target_data = self._data_cls(array=torch.from_numpy(inputs.target_field), pbc=case.pbc)
        self._target = self._vector(self.operator.apply(target_data)).detach()

    @staticmethod
    def _vector(statistics: Any) -> Any:
        # ESTEL's to_vector is STL's to_flatten(flatten_complex=True) (maths.md M-FLAT).
        return statistics.to_flatten(flatten_complex=True)

    def forward(self) -> Any:
        """``ST_Operator.apply`` on the field: the statistics."""
        return self.operator.apply(self.data)

    def gradient(self) -> Any:
        """The loss ``||to_flatten(S(x)) - t||²`` and its gradient with respect to the field."""
        field = self._field.clone().requires_grad_()
        statistics = self.operator.apply(self._data_cls(array=field, pbc=self._pbc))
        loss = ((self._vector(statistics) - self._target) ** 2).sum()
        loss.backward()
        return loss.detach(), field.grad

    def arrays(self, statistics: Any, loss_and_gradient: Any) -> dict[str, NDArray[Any]]:
        """The results of the two calls as NumPy arrays: each family, the flat vector, the loss
        and its gradient."""
        out = {family: getattr(statistics, family).detach().numpy() for family in FAMILIES}
        out["vector"] = self._vector(statistics).detach().numpy()
        loss, gradient = loss_and_gradient
        out["loss"] = np.asarray(loss.numpy())
        out["gradient"] = gradient.numpy()
        return out
