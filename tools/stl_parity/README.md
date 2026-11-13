# STL parity harness

The harness that compares ESTEL with STL during step 1 (Part II, §14 of
[`docs/transfer/part2.md`](../../docs/transfer/part2.md#14-parity-protocol-with-stl)). It lives in the
repository but is not shipped and not imported by ESTEL, and it is retired after 1.0.0.

It holds the container (task T0.6) and the parity fixtures (task T0.8). The parity cases come
with the phases that need them (P2 onwards, §14.4): `cases/` holds the example of T0.8 (the 2D
kernels, `bump_steerable_kernel_2d.py` on `master`), the masks and reweighting maps of T2.1
(`masks_2d.py` on `master`, `masks_1d.py` on `2D+1D`) and the 1D kernels of T2.2
(`bump_steerable_kernel_1d.py` on `2D+1D`).

## The container

One image, built from [`Dockerfile`](Dockerfile), with one Python environment per frozen STL commit
(§9.1), each used for what §14.2 assigns to it:

| Environment | STL-Dev branch | Commit | Used for |
| --- | --- | --- | --- |
| `/opt/venv/master` | `master` | `8eaf769` (v1.6.0) | every 2D result: statistics and spectra |
| `/opt/venv/2d1d` | `2D+1D` | `8f51143` | 1D and tensor-product statistics |
| `/opt/venv/healpix` | `HealpixUpdate` | `90eb1a9` | HEALPix statistics |

The three commits install the same package (`STL_main`), so they cannot share an environment. Each
checkout is at `/opt/stl/<environment>`.

**Pins.** STL-Dev declares no dependency versions, so the image pins them all:

- STL at the full commit hashes; the build fails if a checkout does not land on its hash.
- healpix-analyse at `aac5e26` (its tag `v2026.10.2`), from its Git repository, in the `healpix`
  environment only. It is a test-only dependency (§9.3).
- PyTorch 2.15.0, CPU build, in all three environments: the version of ESTEL's `uv.lock`, so that
  ESTEL and STL run on the same PyTorch.
- Python 3.11 (`python:3.11-slim-bookworm`, linux/amd64), uv 0.13.8.
- Every other dependency in [`requirements/`](requirements): `<env>.in` lists what each environment
  needs, `<env>.txt` pins it; `build.txt` pins STL's build backend.

The pins in `requirements/*.txt` fix what the build installs, but not the base image or the
packages it downloads, which can disappear. The image that produced the parity report is therefore
archived with it (§14.2), and is what reruns the comparison afterwards.

**Precision.** STL runs in float64 on CPU by default (`STL_main.torch_backend`), which is what
parity needs (§14.3). The container has no GPU. STL still builds some arrays at torch's default
dtype, float32 unless changed: its 2D kernels, for one, call `torch.fft.fftfreq` without a dtype.
So every STL run of the container, the smoke check and the fixtures alike, first calls
`torch.set_default_dtype(torch.float64)` (Part II, §14.2; maths.md OP-31), and STL's own, unchanged
code then builds them in float64. `MANIFEST.json` records the setting for each fixture.

## Commands

Run from the repository root. Docker by default; `CONTAINER=podman` works too.

```bash
make parity-image
```

```bash
make parity
```

`make parity-image` builds the image `estel-stl-parity`. `make parity` runs it: for now, the smoke
check [`smoke.py`](smoke.py) in each environment, which runs STL on small float64 inputs (statistics
in every geometry, and the four 2D cross-spectrum estimators on `master`) and fails unless every
output is float64 or complex128 with finite values. In the `healpix` environment it also fails
unless healpix-analyse is installed at commit `aac5e26` with version `2026.10.2`. It prints the
traceback of any case that raises.

To archive the image with the parity report:

```bash
docker save estel-stl-parity | gzip > estel-stl-parity.tar.gz
```

## Parity fixtures

STL's results enter the tests as fixtures (§14.2): `.npz` files in
[`tests/parity/fixtures/`](../../tests/parity/fixtures), produced once in the container and
compared with ESTEL on every PR, in float64, without Docker.

- **A case** is a script of [`cases/`](cases) that names its STL environment (`STL_ENV`: `master`,
  `2d1d` or `healpix`), its parameters and seeded inputs (`PARAMETERS`), and returns its results
  as float64 or complex128 NumPy arrays (`run()`). It imports STL, never `estel`. The example,
  [`cases/bump_steerable_kernel_2d.py`](cases/bump_steerable_kernel_2d.py), stores the 2D
  bump-steerable kernels of STL `master`.
- **`make parity-fixtures`** runs every case in its environment
  ([`make_fixtures.py`](make_fixtures.py), through [`run_fixtures.sh`](run_fixtures.sh)) and writes
  `tests/parity/fixtures/<case>.npz`. `make parity-fixtures CASES=<case>` runs one case. The cases
  are mounted into the container, not copied, so a new case needs no new image; build the image
  first with `make parity-image`.
- **`MANIFEST.json`** records for each file the case script, the STL branch and the commit of its
  checkout, the container image ID (`docker image inspect --format '{{.Id}}'`: a locally built
  image has no registry digest), the parameters, torch's default dtype (`float64`, see
  *Precision*), each array's dtype and shape, and the file's SHA-256. The files are written byte for byte the same when STL's results are the same, so
  regenerating an unchanged case changes nothing.
- **The tests** of [`tests/parity/`](../../tests/parity) check every fixture against the manifest,
  and compare ESTEL with it through `load_fixture` and `assert_parity`
  ([`tests/parity/_compare.py`](../../tests/parity/_compare.py)): same shape, NaN at the same
  positions, and `|ESTEL − STL| ≤ atol + rtol·|STL|` element by element, with rtol 1e-10 in
  float64 and 1e-5 in float32, and atol = 0.01 × rtol × the largest finite |STL| of the array
  (§14.3), so that values zero in theory pass on rounding.
- **`tests/parity/fixtures/` is protected.** Only a maintainer runs `make parity-fixtures` and
  commits its output. An agent that needs a new case writes the case and its test, and asks for
  the fixture in its PR.

## Changing the pins

The pins change only with the sources of §9.1. To recompile an environment after editing its `.in`
file, from `tools/stl_parity/requirements/`:

```bash
uv pip compile master.in -o master.txt --no-header --python-version 3.11 --python-platform x86_64-manylinux_2_28 --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match --exclude-newer 2026-11-12
```

and the same for `2d1d`, `healpix` and `build`.
