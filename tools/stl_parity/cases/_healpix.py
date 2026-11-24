"""The settings of STL `HealpixUpdate` shared by the HEALPix parity cases (Part II, §14.2; task
T5.3, issue #164, decision 1; maths.md OP-45, OP-46).

Not a case: `make_fixtures.py` skips the files of `cases/` whose name starts with ``_``. The
HEALPix cases import it by its path, since a case is itself imported by its path and not as a
module of a package.

Every HEALPix case runs STL with three settings, on top of ``torch.set_default_dtype(
torch.float64)``, which ``make_fixtures.py`` sets for every case:

* **The sphere.** STL computes the pixel centres of healpix-analyse on the ``"WGS84"`` ellipsoid
  by default; ESTEL computes on the sphere only (OP-45). The cases pass
  ``ellipsoid="sphere"`` to STL's wavelet operator, through ``ST_Operator(...,
  wavelet_op_kwargs=WAVELET_OP_KWARGS)`` or ``get_wavelet_op(**WAVELET_OP_KWARGS)``.
* **No geometry cache.** healpix-analyse caches the interpolation weights of ``HealPixConv`` on
  disk in float32, and an operator built after the first one of a geometry reads them back in
  float32 (OP-46). The cases pass ``conv_cache_dir=None``, which disables the cache: every
  operator computes its weights in float64.
* **The kernel in float64.** ``HealPixConv.set_kernel`` casts the kernel through float32
  (``np.asarray(W, dtype=np.float32)``) whatever the dtype of the operator, which moves it by
  5e-8 of its largest modulus (OP-46). :func:`keep_kernel_float64` replaces, at run time and
  for the duration of a ``with`` block, ``HealPixConv.set_kernel`` with a function that calls
  the original one (its checks of the shape, its bias, its ``requires_grad``) and then copies
  the kernel again into the weights of the operator, from a float64 copy of the array given,
  so that the weights hold the float64 kernel. Nothing else changes: the code of STL and of
  healpix-analyse is unchanged on disk, and the original method is restored when the block
  exits. ``HealPixConv`` is the class STL imports (``STL_main.STL_Healpix_Kernel_Torch``
  imports it from ``healpix_analyse.convol``), so the replacement reaches every convolution
  STL builds, the wavelet convolutions and the convolutions with a kernel of ones that build
  the masks (whose ones are exact in float32 anyway). :func:`check_kernel_float64` checks
  afterwards that every convolution of an operator has float64 weights, and that those of its
  wavelet convolutions are the float64 kernel, bit for bit, for each gauge.

The manifest records these settings under ``"stl_settings"`` in the parameters of each case
(:data:`SETTINGS`).

Runs in the parity container, with the interpreter of the `healpix` environment. It imports STL
and healpix-analyse, never `estel`.
"""

from __future__ import annotations

import contextlib
import importlib
from collections.abc import Iterator
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

WAVELET_OP_KWARGS: dict[str, Any] = {"ellipsoid": "sphere", "conv_cache_dir": None}
SETTINGS: dict[str, Any] = {
    "torch_default_dtype": "float64",
    "wavelet_op_kwargs": {"ellipsoid": "sphere", "conv_cache_dir": None},
    "kernel": (
        "float64: HealPixConv.set_kernel replaced at run time by a function that calls it, "
        "then copies the float64 kernel into its weights (tools/stl_parity/cases/_healpix.py, "
        "keep_kernel_float64); STL and healpix-analyse unchanged on disk"
    ),
}


@contextlib.contextmanager
def keep_kernel_float64() -> Iterator[None]:
    """Within the block, ``HealPixConv.set_kernel`` keeps the kernel in float64 (see above)."""
    conv_cls = importlib.import_module("healpix_analyse.convol").HealPixConv
    original = conv_cls.set_kernel

    def set_kernel(self: Any, W: Any, bias: Any = None, requires_grad: bool = False) -> Any:  # noqa: N803 (healpix-analyse's name)
        out = original(self, W, bias=bias, requires_grad=requires_grad)
        kernel = np.asarray(W, dtype=np.float64)
        if kernel.ndim == 3:  # one kernel for every gauge, as the original broadcasts it
            kernel = np.broadcast_to(kernel[None], (self.G, *kernel.shape)).copy()
        with torch.no_grad():
            self.weight.copy_(torch.as_tensor(kernel, dtype=self.dtype, device=self.device))
        return out

    conv_cls.set_kernel = set_kernel
    try:
        yield
    finally:
        conv_cls.set_kernel = original


def check_kernel_float64(wavelet_op: Any) -> None:
    """Raise unless every wavelet convolution that ``wavelet_op`` built holds its float64
    kernel bit for bit, in float64, for each gauge."""
    expected = wavelet_op._wav_weights.detach()
    for key, conv in wavelet_op._conv_cache.items():
        weight = conv.weight.detach()
        if weight.dtype != torch.float64:
            raise TypeError(f"HealPixConv {key} has weights in {weight.dtype}, not float64")
        if weight.shape[-2] == expected.shape[-2] and not all(
            torch.equal(weight[g], expected) for g in range(weight.shape[0])
        ):
            raise ValueError(f"HealPixConv {key} does not hold the float64 wavelet kernel")


def pixel_vectors(nside: int) -> NDArray[np.float64]:
    """The unit vectors of the centres of the NESTED pixels of ``nside``, ``(3, 12 nside²)``,
    on the sphere (healpy)."""
    healpy = importlib.import_module("healpy")
    vectors: NDArray[np.float64] = np.asarray(
        healpy.pix2vec(nside, np.arange(12 * nside**2), nest=True), dtype=np.float64
    )
    return vectors


def disc(
    nside: int, colatitude_deg: float, longitude_deg: float, radius_deg: float
) -> NDArray[np.int64]:
    """The NESTED indices, ascending, of the pixels of ``nside`` whose centre lies within
    ``radius_deg`` of the point (``colatitude_deg``, ``longitude_deg``), as int64."""
    theta, phi = np.radians(colatitude_deg), np.radians(longitude_deg)
    centre = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
    inside = centre @ pixel_vectors(nside) >= np.cos(np.radians(radius_deg))
    cell_ids: NDArray[np.int64] = np.nonzero(inside)[0].astype(np.int64)
    return cell_ids
