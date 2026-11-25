# Benchmark of step 1

A few clear cases at a realistic size, run **outside CI**, that judge each optimisation of step 1
(T2.12 to T2.15) before and after (Part I, §3.1, rule 8 and §6.3 of
[`docs/transfer/part1.md`](../docs/transfer/part1.md#63-benchmarks); Part II, §15.2, task T2.11).
A reorganisation belongs to step 1 only if every value stays within the parity tolerance with STL
and this benchmark shows its gain.

## Cases

[`cases.toml`](cases.toml) holds them. All are on CPU, in float32, one batch element, without
power spectrum, `norm="none"`; the planar ones are 2D:

| Case | Field | J, L | Borders | Missing pixels |
| --- | --- | --- | --- | --- |
| `planar-periodic` | 512 × 512, one channel | 7, 4 | periodic | none |
| `planar-masked` | 512 × 512, one channel | 7, 4 | non-periodic | 10.1 %, five square holes of sides 112 to 24 |
| `planar-cross` | 256 × 256, two channels with their cross terms | 6, 4 | periodic | none |
| `planar-open` | 512 × 512, one channel | 7, 4 | non-periodic | none |
| `tensor-prod-2d1d-64` | 2D+1D, 64 × 64 × 32, one channel (ESTEL only) | (4, 3), (4, 2) | periodic | none |
| `healpix-masked-64` | HEALPix full sky, `nside` 64, 49,152 pixels, one channel (ESTEL only) | 5, 4 | none (full sky, `pbc = true`) | 10.0 %, five discs of radii 21° to 10° |

**Tensor products** (task T3.2, issue #23). A case with `blocks` (for example
`blocks = ["2D", "1D"]`, `shape = [N, N, T]`) is a tensor product, with one `J`, `L` and `pbc` per
block, run through `STOperatorTensorProd`. It is measured on **ESTEL only**: STL is not run on it,
neither in the container nor as a reference, so its check against STL is replaced by the parity
tests of `tests/parity/test_tensor_prod.py`, and its results say so (`check`). It is compared
with `main` (and, with `--compare`, B with A) only when that commit has `STOperatorTensorProd`;
`--compare` with a commit before T3.2 stops with an error naming the case. The first case,
`tensor-prod-2d1d` (32 × 32 × 16, J = (3, 2)), was replaced by `tensor-prod-2d1d-64` with task
T3.5 (issue #128), once the slow path of `conv_axes` along a 1D block was gone; its results stay
in `results/`.

**HEALPix** (task T5.7, issue #168). A case with `geometry = "healpix"` is a full sky of
`shape = [12 · 4^depth]` pixels, with `pbc = true`, which the harness requires and which means a full
sky (`DataHealpix.pbc`; a partial sky is not a case), in the NESTED ordering, run through `STOperator` on
`DataHealpix`, with its missing pixels in discs of the angular radii `mask_discs` (in degrees),
centred on directions drawn by a seeded draw, without overlap. Like a tensor product it is
measured on **ESTEL only**: its check against STL is replaced by the parity tests of HEALPix
(task T5.6, `tests/parity/test_healpix_statistics.py`), its results say so (`check`), and STL's
HEALPix times come with the benchmark of step 2 (LF62). It is compared with `main`, and with
`--compare`, only when that commit has `STOperator` on HEALPix data (T5.6). Its masks make the
operator build its tables for the support test as well as for the filters, so that path is
measured.

The fields are seeded: the exponential of a Gaussian field with a power-law spectrum
([`_cases.py`](_cases.py)), over every axis for a tensor product. On HEALPix, with no
spherical-harmonic transform and no healpy (test-only, Part I, §1.5): a Gaussian white noise
averaged three times over the pixels whose centres lie within two pixel sizes of each pixel
centre, standardised, then exponentiated; the pixel centres come from `estel`, imported only when
the inputs are written, natively. The process that runs the benchmark writes the inputs of a case once,
and every measured process reads the same file.

Cases are added, not edited: changing one makes the earlier results incomparable.

## What is measured

Each case runs in a process of its own ([`worker.py`](worker.py)), which measures:

- **build**: the construction of the operator, timed once, apart (`build_seconds`), and the peak
  resident memory of the process once it is built (`build_peak_rss_bytes`): on HEALPix the build
  of the tables can set the peak of the whole process (task T5.7);
- **forward**: `STOperator.apply(data)`, the operator being built before;
- **forward + gradient**: the loss L = ‖`to_vector`(S(x)) − t‖² and its gradient with respect to
  the field, with t the statistics of a second seeded field;
- **peak memory**: the peak resident memory of the process (`ru_maxrss`) after both, the backward
  pass included. It counts the interpreter and the imported libraries;
- **baseline memory**: the resident memory of the process once the library and the inputs are
  loaded, just before the operator is built. The table shows the peak minus the baseline, the
  memory of the computation itself, which compares between STL and ESTEL.

Times are wall times: one warm-up run, then the median of 5 runs; every run is kept in the
results. The number of threads is fixed (`threads` in `cases.toml`, 4) and recorded.

STL runs the same cases and the same loss through its own functions ([`_stl.py`](_stl.py)):
`ST_Operator.apply` with `norm="vanilla"`, `to_flatten(flatten_complex=True)`, `backward`. It
computes in the dtype of its input, float32 here, and stores S1 to S4 in complex128.

## Environments

- **`container`**: the parity container ([`tools/stl_parity/`](../tools/stl_parity/README.md),
  STL `master`) with ESTEL installed in the same Python environment ([`Dockerfile`](Dockerfile)):
  STL and ESTEL on the same interpreter, NumPy and PyTorch, so that their times compare.
  `make bench` rebuilds this image from the working tree before each run.
- **`native`**: ESTEL on this machine, in the environment of `uv.lock`. Each case is measured
  twice here, one after the other: the ESTEL of `main` (`estel (main)` in the table), then the
  working tree. `main` is `origin/main` (`--main REF` names another commit): its `src/` is
  exported once to `bench/results/.work/`, with no checkout, and the same worker imports it. Run
  `git fetch origin` first.

The image is `linux/amd64`. On an arm64 host (Apple silicon) the container is emulated: its times
compare STL with ESTEL, not with the native run, and the results say so (`emulated`).

## Correctness

Every ESTEL run is compared with STL's statistics on the same inputs (Part I, §6.3): each family
in STL's full-grid shape and the flat vector, same shape, NaN at the same positions, values within
the float32 tolerances of Part II, §14.3 (`assert_parity` of
[`tests/parity/_compare.py`](../tests/parity/_compare.py)). `make bench` fails otherwise, and
writes no result: no gain can come from a wrong computation.

The gradient of the loss is compared with STL's too. §14.3 sets no float32 tolerance for it, so
the largest difference, relative to STL's largest gradient, is recorded with each run
(`gradient_relative_error`) and not enforced. STL's operator is built with
`replace_nan_value=None`, as STL's syntheses build it: with its default, NaN, the coarse pixels
declared invalid hold NaN values, and its gradient on `planar-masked` is NaN at every valid pixel
although its statistics are finite (`docs/maths.md`, OP-60). With `None` its statistics are the
same bit for bit and its gradient is finite, so the gradient of every case is compared. Should
STL's gradient still have non-finite values, nothing is compared there and the pixels are counted
(`stl_gradient_not_finite`, 0 in every case today).

STL's results stay in `bench/results/.work/` (ignored by git), so that a later native run is
checked without the container. They are used only for inputs with the same digest.

The native run is compared with `main` too, on the same inputs (Part II, §15.2, T2.12 to T2.15):
the statistics within the same float32 tolerances, and the gradient of the loss within
max |g − g_main| ≤ 1e-4 × max |g_main|, every value finite. `make bench` fails otherwise: an
optimisation of step 1 changes no value beyond the parity tolerance (Part I, §3.1, rule 8), and
until the planar syntheses exist (T6.1) this comparison is what validates its gradients. The
measured differences are printed, each relative to the largest value of `main`, and recorded
with the run (`check_main`).

## Commands

From the repository root. The container needs Docker (`CONTAINER=podman` works too) and the
parity image:

```bash
make parity-image
```

```bash
make bench
```

Before and after an optimisation, run it twice in a row on AC power, so that `main` and the
working tree alternate (main, branch, main, branch), and commit the results. On this kind of
machine the times vary by about 15 % from one run to the next: a gain in time is one above about
20 %.

One case, or one environment:

```bash
make bench BENCH_CASE=planar-periodic
```

```bash
make bench BENCH_ENV=native
```

Other options go through `BENCH_ARGS` (`uv run python -m bench --help`): `--backend jax` runs
ESTEL on JAX, natively. On JAX the computation is compiled at its first call: every run records
the time of the very first call of the operator (`first_apply_seconds`, made on the second field
when the run is built), which holds the compilation of the forward pass, and the time of the
first call of each measure (`forward_first_seconds`, `gradient_first_seconds`), the second of
which holds the compilation of the gradient, apart from the timed calls. JAX has no setting for the number of
threads of its CPU computations: its times are not at the `threads` of the cases file.

## Comparing two commits

An optimisation is judged against the commit before it (task T2.13, issue #91):

```bash
make bench BENCH_ARGS="--compare origin/main HEAD"
```

measures the two commits A and B natively, each exported like `main` above, in `--rounds` rounds
(5): a round measures A then B, the next one B then A, every measure in a fresh process (warm-up,
then the timed runs, their median). Per round and case it takes the ratios B / A of the time of
the build, of the forward time, of the time with the gradient, of the memory of the computation
(peak minus baseline), of the peak memory once the operator is built and of the peak memory, and
it prints their median and their range over the rounds, with the measures whose ratio is below 1
in every round and those above 1.05 in some round (the times of the forward pass and of the
gradient, and the memory of the computation: the build is printed, not read by this criterion,
since on 2D cases it takes milliseconds and its ratios are noise). Every run is checked
against STL's statistics, and B against the statistics and the gradient of A, as above. The
results go to `results/<date>-<machine>-compare.json`. It needs STL's statistics of a container
run on the same inputs, and no container itself.

Run it once with A = B to see the noise floor of the comparison on the machine: a ratio inside
that range shows nothing.

`make bench` prints a table and writes `results/<date>-<machine>.json`: the description of each
environment (CPU model and cores, memory, OS, Python and library versions, image), the ESTEL
commit, the settings, the cases, and every run. `<machine>` comes from the CPU model
(`--machine` names it otherwise); a result is never overwritten, a second one on the same day
takes the suffix `-2`. Commit the results of a run made on a clean tree, with the pull request of
the optimisation they judge, before and after.

On macOS the results record the power source and Low Power Mode (`power`), and `make bench` warns
when the machine is on battery or in Low Power Mode: such times are slower by a factor of about
two and do not compare with the others.

If a case does not fit in memory, or a process runs for more than 45 minutes, `make bench` stops:
report what was measured on the task's issue, so that the maintainer resizes the case.

## Tests

`tests/bench/` runs the harness on tiny cases (32 × 32, J = 2; a tensor product; a HEALPix
sky at `nside` 4) in the fast tier, so that it cannot rot. The real cases never run in CI, and there is no GitHub workflow for the benchmark.
`bench/thresholds.toml` (protected) does not exist yet: the thresholds are set with the benchmark
of step 2a (LF62, `docs/transfer/part3.md`).
