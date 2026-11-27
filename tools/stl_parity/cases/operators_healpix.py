"""Parity case: the convolution and the decimation of STL `HealpixUpdate` on HEALPix data, as
operators (Part II, §14.2; task T5.3, issue #164, decision 3; maths.md M-CONV-HEALPIX,
M-WAV-HEALPIX, M-DOWN-HEALPIX, OP-92).

STL's HEALPix wavelet operator (``WaveletOperatorHealpixKernel_torch``) convolves with
healpix-analyse's ``HealPixConv`` (the wavelet kernel, L = 4 gauges) and decimates with its
``HealPixDown``. ESTEL builds its own tables (T5.1), which equal healpix-analyse's as linear
operators, not entry by entry (OP-92): so this case stores the operators by what they do, through
STL's own entry points, ``apply(data, 0)`` and ``downsample(data, 1)``, with the settings of
``_healpix.py`` (the sphere, no cache of tables, the kernel in float64; OP-45, OP-46).

It stores:

* ``kernel``: STL's wavelet kernel, ``(1, 1, 25)`` complex, the K x K = 5 x 5 stencil flattened
  as in M-CONV-HEALPIX; the convolution holds its real and imaginary parts in float64, checked bit
  for bit (``_healpix.check_kernel_float64``);
* at ``nside`` 4, full sky, the dense matrices: ``conv_4``, ``(L, Npix, Npix)`` complex, with
  ``conv_4[g, k, n]`` the response at pixel k and gauge g to the unit map at pixel n, and
  ``down_4``, ``(Npix / 4, Npix)`` real, the decimation to ``nside`` 2, with
  ``down_4_cell_ids`` the coarse pixels (all of them, ascending);
* for ``<domain>_<nside>``, ``<domain>`` ``full`` (the full sky) or ``disc`` (the pixels whose
  centre is within ``disc_radius_deg`` of the centre of the disc, a partial sky that crosses the
  equator), at ``nside`` 8 and 16: ``<domain>_<nside>_cell_ids``, the NESTED pixels,
  ascending; ``<domain>_<nside>_maps``, 3 seeded unit Gaussian maps, ``(3, Npix)``;
  ``<domain>_<nside>_conv``, their convolutions, ``(3, L, Npix)`` complex;
  ``<domain>_<nside>_down``, their decimations, ``(3, Npix_coarse)``, on the coarse pixels
  ``<domain>_<nside>_down_cell_ids``.

The pixel indices are stored as float64, exact below 2⁵³. With one channel and one map per
batch entry, ``(3, Npix)`` is STL's ``(Nb, Npix)``.

Runs in the parity container, with the interpreter of the `healpix` environment, through
`make parity-fixtures` (tools/stl_parity/make_fixtures.py). It imports STL, never `estel`.
"""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

STL_ENV = "healpix"


# Repeated in each HEALPix case: make_fixtures.py imports a case by its path, so a case cannot
# import a sibling module, and this loader must exist before _healpix.py can be read.
def _helper() -> ModuleType:
    """The settings of the HEALPix cases, ``_healpix.py``, imported by its path."""
    path = Path(__file__).resolve().with_name("_healpix.py")
    spec = importlib.util.spec_from_file_location("stl_parity_cases_healpix", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HEALPIX = _helper()
PARAMETERS: dict[str, Any] = {
    "L": 4,
    "K": 5,
    "dense_nside": 4,
    "nsides": [8, 16],
    "n_maps": 3,
    "seeds": {"full_8": 80, "disc_8": 81, "full_16": 160, "disc_16": 161},
    "disc_centre_deg": {"colatitude": 80.0, "longitude": 30.0},
    "disc_radius_deg": 30.0,
    "stl_settings": HEALPIX.SETTINGS,
}


def _f64(x: torch.Tensor) -> NDArray[Any]:
    array: NDArray[Any] = x.detach().cpu().numpy()
    if array.dtype not in (np.float64, np.complex128):
        array = array.astype(np.float64)
    return array


def _responses(
    data_cls: Any, nside: int, cell_ids: NDArray[np.int64] | None, maps: NDArray[np.float64]
) -> tuple[Any, Any, Any]:
    """STL's wavelet operator on the grid of ``maps``, its convolutions at scale 0 and its
    decimations by one level."""
    kwargs: dict[str, Any] = {} if cell_ids is None else {"nside": nside, "cell_ids": cell_ids}
    data = data_cls(array=torch.from_numpy(maps.copy()), **kwargs)
    operator = data.get_wavelet_op(
        J=2, L=PARAMETERS["L"], kernel_size=PARAMETERS["K"], **HEALPIX.WAVELET_OP_KWARGS
    )
    conv = operator.apply(data, 0)
    down = operator.downsample(data.copy(), 1)
    HEALPIX.check_kernel_float64(operator)
    return operator, conv, down


def run() -> dict[str, NDArray[Any]]:
    """Run STL's convolution and decimation and return them as matrices and responses."""
    data_cls = importlib.import_module("STL_main.STL_Healpix_Kernel_Torch").STL_Healpix_Kernel_Torch
    p = PARAMETERS
    out: dict[str, NDArray[Any]] = {}
    with HEALPIX.keep_kernel_float64():
        # The dense matrices: the responses to the unit maps, one per batch entry.
        n_pix = 12 * p["dense_nside"] ** 2
        operator, conv, down = _responses(data_cls, p["dense_nside"], None, np.eye(n_pix))
        out["kernel"] = _f64(operator.kernel)
        out["conv_4"] = _f64(conv.array.permute(1, 2, 0))  # (unit n, g, k) -> (g, k, n)
        out["down_4"] = _f64(down.array.T)  # (unit n, coarse k) -> (k, n)
        out["down_4_cell_ids"] = _f64(down.cell_ids)

        centre = p["disc_centre_deg"]
        for nside in p["nsides"]:
            for domain in ("full", "disc"):
                name = f"{domain}_{nside}"
                cell_ids = (
                    None
                    if domain == "full"
                    else HEALPIX.disc(
                        nside, centre["colatitude"], centre["longitude"], p["disc_radius_deg"]
                    )
                )
                n = 12 * nside**2 if cell_ids is None else cell_ids.size
                rng = np.random.default_rng(p["seeds"][name])
                maps = rng.standard_normal((p["n_maps"], n))
                _, conv, down = _responses(data_cls, nside, cell_ids, maps)
                out[f"{name}_cell_ids"] = _f64(conv.cell_ids)
                out[f"{name}_maps"] = maps
                out[f"{name}_conv"] = _f64(conv.array)
                out[f"{name}_down"] = _f64(down.array)
                out[f"{name}_down_cell_ids"] = _f64(down.cell_ids)
    return out
