"""Write the STL parity fixtures (Part II, §14.2; task T0.8).

Runs inside the parity container, with the interpreter of one STL environment, for the cases of
`cases/` that name that environment:

    /opt/venv/<env>/bin/python make_fixtures.py <env> --out <dir> --image-id <id> [case ...]

`make parity-fixtures` runs it in each environment (run_fixtures.sh). For each case it writes
`<dir>/<case>.npz` and its entry of `<dir>/MANIFEST.json`: the case script, the STL branch and
the commit of its checkout, the container image ID, the case's parameters, torch's default dtype,
each array's dtype and shape, and the SHA-256 of the file. The `.npz` files are written byte for
byte the same when the arrays are the same, so regenerating an unchanged case leaves its file and
checksum unchanged.

Every run sets torch's default dtype to float64 before STL runs (maintainer decision on #57): STL
builds some arrays at the default dtype, such as its 2D kernels, which call torch.fft.fftfreq
without a dtype. With float64, STL's own, unchanged code computes them in float64.

A case is a module of `cases/` that defines `STL_ENV` (`master`, `2d1d` or `healpix`, the
environments of the image), `PARAMETERS` (a JSON-serialisable dict) and `run()`, which returns a
dict of float64 or complex128 NumPy arrays.

This file is not part of ESTEL. It imports STL, never `estel`.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import traceback
import zipfile
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch.backends.nnpack
from numpy.typing import NDArray

CASES = Path(__file__).resolve().parent / "cases"
MANIFEST = "MANIFEST.json"
# The STL-Dev branch of each environment of the image (Dockerfile; Part II, §9.1).
BRANCHES = {"master": "master", "2d1d": "2D+1D", "healpix": "HealpixUpdate"}
FLOAT64_DTYPES = (np.dtype(np.float64), np.dtype(np.complex128))
# A fixed timestamp for every member of the archive, so that the bytes depend on the arrays only.
ZIP_DATE = (1980, 1, 1, 0, 0, 0)


def load_case(path: Path) -> ModuleType:
    """Import the case script at ``path`` and check that it defines what a case needs."""
    spec = importlib.util.spec_from_file_location(f"stl_parity_case_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import the case script {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    missing = [name for name in ("STL_ENV", "PARAMETERS", "run") if not hasattr(module, name)]
    if missing:
        raise AttributeError(
            f"{path.name} does not define {', '.join(missing)}: a case defines STL_ENV, "
            "PARAMETERS and run() (see the docstring of make_fixtures.py)."
        )
    if module.STL_ENV not in BRANCHES:
        raise ValueError(
            f"{path.name}: STL_ENV is {module.STL_ENV!r}; set it to one of {sorted(BRANCHES)}."
        )
    return module


def check_arrays(case: str, arrays: Any) -> dict[str, NDArray[Any]]:
    """Return the arrays of a case, or raise if they are not float64 or complex128 arrays."""
    if not isinstance(arrays, dict) or not arrays:
        raise TypeError(f"{case}: run() must return a non-empty dict of NumPy arrays.")
    for name, array in arrays.items():
        if not isinstance(array, np.ndarray) or array.dtype not in FLOAT64_DTYPES:
            kind = array.dtype if isinstance(array, np.ndarray) else type(array).__name__
            raise TypeError(
                f"{case}: array {name!r} is {kind}; parity runs in float64 (§14.3), so return "
                "float64 or complex128 NumPy arrays."
            )
    return arrays


def write_npz(path: Path, arrays: dict[str, NDArray[Any]]) -> None:
    """Write ``arrays`` as an uncompressed ``.npz`` whose bytes depend on the arrays only."""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(arrays):
            info = zipfile.ZipInfo(f"{name}.npy", date_time=ZIP_DATE)
            with archive.open(info, "w") as stream:
                np.lib.format.write_array(
                    stream, np.ascontiguousarray(arrays[name]), allow_pickle=False
                )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stl_commit(env: str) -> str:
    """The commit of the STL checkout installed in ``env``, read from Git."""
    result = subprocess.run(
        ["git", "-c", "safe.directory=*", "-C", f"/opt/stl/{env}", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def entry(case_path: Path, module: ModuleType, image_id: str, file: Path) -> dict[str, Any]:
    with np.load(file, allow_pickle=False) as stored:
        arrays = {
            name: {"dtype": str(stored[name].dtype), "shape": list(stored[name].shape)}
            for name in stored.files
        }
    return {
        "case": f"tools/stl_parity/cases/{case_path.name}",
        "stl_branch": BRANCHES[module.STL_ENV],
        "stl_commit": stl_commit(module.STL_ENV),
        "image_id": image_id,
        "parameters": module.PARAMETERS,
        # Read after the case ran, so a case that changed it is recorded as it ran.
        "torch_default_dtype": str(torch.get_default_dtype()).removeprefix("torch."),
        "arrays": arrays,
        "sha256": sha256(file),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Write the STL parity fixtures of one env.")
    parser.add_argument("env", choices=sorted(BRANCHES), help="STL environment of the image")
    parser.add_argument("--out", type=Path, required=True, help="tests/parity/fixtures, mounted")
    parser.add_argument("--image-id", required=True, help="ID of the image this runs in")
    parser.add_argument("cases", nargs="*", help="case names (default: every case of the env)")
    args = parser.parse_intermixed_args()
    # See smoke.py: NNPACK fails to initialise under linux/amd64 emulation; float64 is unaffected.
    torch.backends.nnpack.set_flags(False)  # type: ignore[no-untyped-call]  # unannotated in torch
    # Maintainer decision on #57: STL builds some arrays at torch's default dtype (its 2D kernels
    # call torch.fft.fftfreq without a dtype), so every STL run of the container sets it to
    # float64 first, and STL's own, unchanged code then computes them in float64.
    torch.set_default_dtype(torch.float64)

    paths = sorted(CASES.glob("[!_]*.py"))
    unknown = sorted(set(args.cases) - {path.stem for path in paths})
    if unknown:
        print(f"FAIL no case named {', '.join(unknown)} in {CASES}", file=sys.stderr)
        return 1
    manifest_path = args.out / MANIFEST
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    errors = []
    for path in paths:
        if args.cases and path.stem not in args.cases:
            continue
        try:
            module = load_case(path)
            if args.env != module.STL_ENV:
                continue
            print(f"- {path.stem} (STL {BRANCHES[args.env]})")
            arrays = check_arrays(path.stem, module.run())
            file = args.out / f"{path.stem}.npz"
            write_npz(file, arrays)
            manifest[file.name] = entry(path, module, args.image_id, file)
            for name, info in manifest[file.name]["arrays"].items():
                print(f"  wrote {file.name}:{name} {tuple(info['shape'])} {info['dtype']}")
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            errors.append(f"{path.stem}: {type(exc).__name__}: {exc}")
    manifest_path.write_text(json.dumps(dict(sorted(manifest.items())), indent=2) + "\n")
    for error in errors:
        print(f"FAIL {error}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
