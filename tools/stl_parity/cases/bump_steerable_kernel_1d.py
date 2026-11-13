"""Parity case: the 1D bump-steerable kernels of STL `2D+1D` (Part II, §14.2; task T2.2, issue #14).

STL's 1D kernel operator (`WaveletOperator1Dkernel_torch`, built by
`STL_1D_Kernel_Torch.get_wavelet_op`) convolves with L complex kernels of K taps. It builds them
once: the 1D bump of its `bump_steerable_bank` at the first scale, on a grid of 256 points,
brought to pixel space by an inverse FFT, cropped to K taps around the centre, and made
zero-mean (maths.md M-WAV-1D). With L = 2 the second kernel is the conjugate of the first. This
case stores them as STL holds them, shape (1, L, K), in complex128, for L = 1 (``kernel_L1``)
and L = 2 (``kernel_L2``).

K is STL's default at `8f51143` (K=5), written explicitly; L = 1 is the default and L = 2 the
value of the 1D blocks of tensor products. J is the default of `get_wavelet_op` for an input of
128 samples; the kernels do not depend on it, nor on the values of the input, which STL needs
only to build the operator.

Runs in the parity container, with the interpreter of the `2d1d` environment, through
`make parity-fixtures` (tools/stl_parity/make_fixtures.py). It imports STL, never `estel`.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

STL_ENV = "2d1d"
PARAMETERS: dict[str, Any] = {"J": 5, "L": [1, 2], "K": 5, "N": 128, "seed": 0}


def run() -> dict[str, NDArray[Any]]:
    """Build STL's operator on a seeded input, for each L, and return its kernels."""
    data_1d = importlib.import_module("STL_main.STL_1D_Kernel_Torch").STL_1D_Kernel_Torch
    n = PARAMETERS["N"]
    signal = np.random.default_rng(PARAMETERS["seed"]).standard_normal(n)
    data = data_1d(array=torch.from_numpy(signal), pbc=True)
    out: dict[str, NDArray[Any]] = {}
    for n_orientations in PARAMETERS["L"]:
        operator = data.get_wavelet_op(
            J=PARAMETERS["J"], L=n_orientations, kernel_size=PARAMETERS["K"]
        )
        kernel: torch.Tensor = operator._wav_kernel  # (1, L, K); STL has no public accessor
        out[f"kernel_L{n_orientations}"] = kernel.detach().cpu().numpy()
    return out
