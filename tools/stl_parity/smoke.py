"""Smoke check of the parity container (Part II, §14.2; task T0.6): STL runs in float64.

Runs inside the container, with the interpreter of one STL environment:

    /opt/venv/<env>/bin/python /opt/stl_parity/smoke.py <env>      # env: master, 2d1d, healpix

For each geometry that the environment serves (Part II, §14.2, "Which STL for what"), it builds a
small float64 input, runs STL's scattering operator (and, on `master`, the four 2D cross-spectrum
estimators), and checks that every output is float64 or complex128 and holds finite values. It
checks that STL runs, not what it computes: the parity cases compare values.

Like make_fixtures.py, it sets torch's default dtype to float64 before STL runs.

This file is not part of ESTEL. It imports STL, never `estel`.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import sys
import traceback
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import torch
import torch.backends.nnpack

COMMITS = {
    "master": "8eaf7691d55f3c3441cde9b95f1f80eb17aa8014",
    "2d1d": "8f51143c5f8ebc259328830f246894d7031acae5",
    "healpix": "90eb1a954e71abb7af621b10e70b0456e8a86b4e",
}
HEALPIX_ANALYSE_COMMIT = "aac5e2673e53209e2a8188715c033663d0dcfd5b"
HEALPIX_ANALYSE_VERSION = "2026.10.2"
FLOAT64_DTYPES = (torch.float64, torch.complex128)
STATISTICS_FIELDS = ("mean", "var", "S1", "S2", "S3", "S4", "PS")
J = 3


def _stl(module: str, name: str) -> Any:
    return getattr(importlib.import_module(f"STL_main.{module}"), name)


def _field(shape: tuple[int, ...], seed: int) -> torch.Tensor:
    return torch.from_numpy(np.random.default_rng(seed).standard_normal(shape))


def _with_nans(x: torch.Tensor) -> torch.Tensor:
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask[..., 2:5, 3:6] = True
    return torch.where(mask, torch.full_like(x, torch.nan), x)


def _tensors(label: str, output: Any) -> Iterator[tuple[str, torch.Tensor]]:
    if isinstance(output, torch.Tensor):
        yield label, output
        return
    for name in STATISTICS_FIELDS:
        value = getattr(output, name, None)
        if isinstance(value, torch.Tensor):
            yield f"{label}.{name}", value


def _check(label: str, output: Any) -> list[str]:
    found = list(_tensors(label, output))
    if not found:
        return [f"{label}: STL returned no tensor ({type(output).__name__})"]
    errors = []
    for name, tensor in found:
        if tensor.dtype not in FLOAT64_DTYPES:
            errors.append(f"{name}: dtype {tensor.dtype}, expected float64 or complex128")
        elif not bool(torch.isfinite(tensor).any()):
            errors.append(f"{name}: no finite value")
        else:
            print(f"  ok {name} {tuple(tensor.shape)} {tensor.dtype}")
    return errors


def _scattering(data: Any, j: int | list[int] = J) -> Any:
    return data.get_ST_op(J=j).apply(data)


def _planar_2d() -> Any:
    data_2d = _stl("STL_2D_Kernel_Torch", "STL_2D_Kernel_Torch")
    return _scattering(data_2d(array=_field((32, 32), seed=0), pbc=True))


def _planar_1d() -> Any:
    data_1d = _stl("STL_1D_Kernel_Torch", "STL_1D_Kernel_Torch")
    return _scattering(data_1d(array=_field((64,), seed=1), pbc=True))


def _tensor_product() -> Any:
    tensor_prod = _stl("Tensor_Product_Kernel_Torch", "STL_TensorProd_Kernel_Torch")
    data_2d = _stl("STL_2D_Kernel_Torch", "STL_2D_Kernel_Torch")
    data_1d = _stl("STL_1D_Kernel_Torch", "STL_1D_Kernel_Torch")
    data = tensor_prod(
        array=_field((32, 32, 32), seed=2),
        data_structure=[data_2d, data_1d],
        pbc=[False, False],
    )
    # In a tensor product, STL takes J and the other wavelet arguments as lists, one entry per
    # block (Wavelet_Operator_TensorProd_Kernel_Torch at 8f51143); L defaults per block.
    return _scattering(data, j=[J, J])


def _healpix() -> Any:
    data_healpix = _stl("STL_Healpix_Kernel_Torch", "STL_Healpix_Kernel_Torch")
    nside = 16
    return _scattering(data_healpix(array=_field((12 * nside**2,), seed=3)))


def _cross_spectrum(method: str) -> Callable[[], Any]:
    pbc = method in ("fft", "nan_mask")
    nans = method in ("nan_mask", "pbc_nan_mask")

    def run() -> Any:
        data_2d = _stl("STL_2D_Kernel_Torch", "STL_2D_Kernel_Torch")
        x = _field((32, 32), seed=4)
        data = data_2d(array=_with_nans(x) if nans else x, pbc=pbc)
        return data.get_CS_op(spectrum_method=method).apply(data)

    return run


CASES: dict[str, dict[str, Callable[[], Any]]] = {
    "master": {
        "2D statistics": _planar_2d,
        **{
            f"2D cross spectrum, {method}": _cross_spectrum(method)
            for method in ("fft", "pbc_mask", "nan_mask", "pbc_nan_mask")
        },
    },
    "2d1d": {
        "1D statistics": _planar_1d,
        "2D statistics": _planar_2d,
        "2D+1D statistics": _tensor_product,
    },
    "healpix": {"HEALPix statistics": _healpix},
}


def _environment(env: str) -> list[str]:
    stl = importlib.import_module("STL_main")
    backend = importlib.import_module("STL_main.torch_backend")
    print(f"  STL {stl.__version__} at {COMMITS[env][:7]}, from {stl.__file__}")
    print(f"  torch {torch.__version__}, numpy {np.__version__}, python {sys.version.split()[0]}")
    print(f"  torch default dtype {torch.get_default_dtype()}")
    errors = _healpix_analyse() if env == "healpix" else []
    default_dtype = backend._DEFAULT_DTYPE
    if default_dtype is not torch.float64:
        errors.append(
            f"STL's default dtype is {default_dtype} on {backend._DEFAULT_DEVICE}, "
            "expected torch.float64: run the container on CPU."
        )
    return errors


def _healpix_analyse() -> list[str]:
    # healpix-analyse has no __version__: read the installed distribution. Its commit is the one
    # uv records for a Git install (direct_url.json, PEP 610).
    dist = importlib.metadata.distribution("healpix-analyse")
    direct_url = json.loads(dist.read_text("direct_url.json") or "{}")
    commit = direct_url.get("vcs_info", {}).get("commit_id", "unknown")
    print(f"  healpix-analyse {dist.version} at {commit[:7]}, from {direct_url.get('url', '?')}")
    errors = []
    if commit != HEALPIX_ANALYSE_COMMIT:
        errors.append(
            f"healpix-analyse is installed at commit {commit}, expected {HEALPIX_ANALYSE_COMMIT} "
            "(Part II, §9.1): check its line in requirements/healpix.txt and rebuild the image."
        )
    if dist.version != HEALPIX_ANALYSE_VERSION:
        errors.append(
            f"healpix-analyse reports version {dist.version}, expected {HEALPIX_ANALYSE_VERSION} "
            f"(tag v{HEALPIX_ANALYSE_VERSION} of commit {HEALPIX_ANALYSE_COMMIT[:7]}). "
            "If the commit check passed, the code is right but its build (setuptools-scm) did "
            "not see the tag: check how the Dockerfile installs healpix-analyse."
        )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Check that STL runs in float64.")
    parser.add_argument("env", choices=sorted(CASES), help="STL environment of the image")
    env = parser.parse_args().env
    # NNPACK fails to initialise under linux/amd64 emulation (Apple Silicon) and PyTorch's C++
    # side then prints "Unsupported hardware" warnings, which Python's warnings filters cannot
    # catch. Disabling it skips the initialisation. NNPACK only handles float32, so the float64
    # results are unchanged.
    torch.backends.nnpack.set_flags(False)  # type: ignore[no-untyped-call]  # unannotated in torch
    # Maintainer decision on #57: STL builds some arrays at torch's default dtype (its 2D kernels
    # call torch.fft.fftfreq without a dtype), so every STL run of the container sets it to
    # float64 first, and STL's own, unchanged code then computes them in float64.
    torch.set_default_dtype(torch.float64)
    errors = _environment(env)
    for label, case in CASES[env].items():
        print(f"- {label}")
        try:
            errors += _check(label, case())
        except Exception as exc:
            print(f"  STL raised {type(exc).__name__} in '{label}':", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            errors.append(f"{label}: STL raised {type(exc).__name__}: {exc}")
    for error in errors:
        print(f"FAIL {error}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
