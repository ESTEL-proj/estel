"""Parity case: the 2D bump-steerable kernels of STL `master` (Part II, §14.2; task T0.8).

STL's 2D kernel operator (`WaveletOperator2Dkernel_torch`, built by
`STL_2D_Kernel_Torch.get_wavelet_op`) convolves in pixel space with L oriented complex kernels of
K x K pixels. It builds them once: the bump-steerable bank of the Fourier-space class at its first
scale, on a 256 x 256 grid, brought to pixel space by an inverse FFT, cropped to K x K around the
centre, and made zero-mean. This case stores them as STL holds them, shape (1, L, K, K), in
complex128.

L and K are STL's defaults at `8eaf769` (L=4, K=5), written explicitly. J is the default of
`get_wavelet_op` for a 32 x 32 input; the kernels do not depend on it, nor on the values of the
input, which STL needs only to build the operator.

ESTEL's kernels are compared with this fixture by task T2.2 (issue #14).

Runs in the parity container, with the interpreter of the `master` environment, through
`make parity-fixtures` (tools/stl_parity/make_fixtures.py). It imports STL, never `estel`.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

STL_ENV = "master"
PARAMETERS: dict[str, int] = {"J": 3, "L": 4, "K": 5, "N": 32, "seed": 0}


def run() -> dict[str, NDArray[Any]]:
    """Build STL's operator on a seeded input and return its kernels."""
    data_2d = importlib.import_module("STL_main.STL_2D_Kernel_Torch").STL_2D_Kernel_Torch
    n = PARAMETERS["N"]
    field = np.random.default_rng(PARAMETERS["seed"]).standard_normal((n, n))
    data = data_2d(array=torch.from_numpy(field), pbc=True)
    operator = data.get_wavelet_op(
        J=PARAMETERS["J"], L=PARAMETERS["L"], kernel_size=PARAMETERS["K"]
    )
    kernel: torch.Tensor = operator._wav_kernel  # (1, L, K, K); STL has no public accessor
    return {"kernel": kernel.detach().cpu().numpy()}
