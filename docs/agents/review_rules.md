# Review rules for ESTEL

<!-- AGENTS: DO NOT EDIT. Changing this file needs CODEOWNER approval. -->

The AI review applies these rules to every pull request. It comments and never blocks; every rule that can be
checked mechanically is also a CI check (Part I, §2.4 and §5.3 of `docs/transfer/part1.md`).

These rules are written for ESTEL. Their structure and many of their ideas come from the GRID4EARTH review
rubric for healpix-geo and xdggs (github.com/GRID4EARTH/review-rules, commit `d38b775`); the IDs are kept so
that the two can be compared. Rules that only made sense for healpix-geo are dropped; rules X1 to X10 are
specific to ESTEL.

Each finding names its rule ID and a severity:

- **must-fix**: a defect, or a breach of an X rule; the PR should not merge as is;
- **should-fix**: a real problem that can be fixed in a named follow-up (rule E2);
- **suggestion**: a matter of taste or polish.

## A. API design and scope

- **A1. Keep the public namespace minimal.** Prefer one function that accepts several input shapes over
  several functions. Do not add public API without a known user. Anything public is in `api/manifest.json`.
- **A3. Match the shape of the existing API.** A new function follows the conventions of its neighbours:
  argument order and names, return types, module placement, shared parameter descriptions copied verbatim.
- **A4. Terminology.** Follow the names of `docs/maths.md` and of the transfer specification. For HEALPix,
  use the healpix-geo vocabulary: `depth` (not `nside`), NESTED ordering, `cell_ids`.
- **A6. One way to say it.** No redundant parameters, no aliases, no option that can be set both at
  construction and at call time (Part I, §2.3).
- **A7. Backwards compatibility follows the version rules.** Within a major version the public API does not
  break (Part I, §7); experimental, reserved interfaces may change through a major version.
- **A8. Respect the library boundary.** ESTEL computes scattering statistics, spectra and simple syntheses.
  Applications (component separation, inverse problems) live downstream. Point to a separate package when
  something is out of scope.
- **A9. Reuse before reimplementing.** If NumPy, the Array API, PyTorch or JAX already provide it, use it.
  Do not add a direct dependency for something an existing one already gives (Part I, §1.6).
- **A10. Options are keyword-only.** Data comes first and positionally; everything else comes after `*`.
- **A11. One dtype contract.** float32 by default, float64 on request, complex results as complex; dtypes are
  fixed once, at construction, and never silently changed (Part I, §3.1, rule 7).

## B. Implementation

- **B1. Validate once, at one layer.** Inputs are validated when a DataClass or an operator is built, not
  again in every function. Validation errors say what to do next (X8).
- **B3. No optimisation without measurement.** Caching, fused kernels, avoided allocations or reorganised
  loops need a benchmark (Part I, §6.3). Suggest the simple version first.
- **B4. No debug code.** No `print`, no `breakpoint`, no assertion used for debugging, no commented-out code
  "for later". Errors a user can trigger are exceptions with an actionable message.
- **B5. Dependency hygiene.** Heavy or optional packages are imported lazily, with a clear `ImportError`
  message. Every new dependency is justified in the PR; transitive dependencies are not listed.
- **B6. Remove repetition with the language's idiom.** A repeated construction becomes a helper or a default;
  a repeated helper becomes a method of the mother class.
- **B7. Change the function, don't wrap it.** When all callers are in this repository, modify the function
  instead of adding a wrapper around it.
- **B8. Correct numeric conventions.** Normalisations, index ranges and conventions follow `docs/maths.md`
  exactly, including STL's conventions reproduced in step 1 (its open points OP-n).
- **B9. Typing conventions.** Use modern annotations; do not annotate `cls`; use `Self` for the return type of
  class methods.
- **B10. Right for the reason it claims.** A result that is correct only by coincidence (a no-op
  transformation, a filter that never filters, a default that hides a mismatch) is a defect, even when the
  output is right. Name the condition under which it would break.

## C. Tests

- **C1. Group tests per function in a class** (`TestFunctionName`) once there are more than two.
- **C2. Test the unit, not the dispatcher.** Call the lowest-level public entry point; keep high-level tests
  few and small.
- **C3. Dummy data in tests signals an API problem.** An object built only to satisfy a signature suggests
  the API should change.
- **C4. Expected values.** Explicit expected values for small cases; the `_spec` oracle for full cases.
- **C5. No references to removed code.** Tests and comments do not mention superseded implementations.
- **C7. Tests assert properties of ESTEL.** A test that only checks a reference implementation (STL, healpy,
  NumPy) adds nothing, unless it says why it is there.
- **C8. Parametrize.** `pytest.mark.parametrize` with readable ids; no loops inside test bodies; fixtures only
  where they do not hide the inputs.
- **C9. Expected errors are parametrised** next to the valid cases, with `pytest.raises(..., match=...)`.
- **C10. Tolerances.** Default tolerances, or values from `tests/tolerances.toml` with a comment citing it;
  variables named `actual` and `expected`.

## D. Documentation

- **D1. numpydoc, checked by a linter.** Do not spend review effort on docstring format; review content.
- **D2. Reference pages list, guides explain.** Explanations go to the user guide; examples go to docstrings.
- **D3. Precise terminology.** Use the terms of `docs/maths.md` and of the glossary of
  `docs/transfer/appendices.md`.
- **D4. Claims match what is implemented.** Do not describe more than the code does.
- **D5. Delete rather than defer.** No placeholder pages, outdated paragraphs or filler sentences.
- **D6. Structure follows the reader.** The recommended entry points come first; advanced APIs later.
- **D7. Links and credits.** Cross-reference with the documentation tool's syntax; credit ported code and
  upstream work accurately (STL, healpix-analyse, STL-Dev-MPI, `NOTICE`).
- **D9. Examples use their results.** A call whose return value is discarded reads like an in-place
  operation: assign and use it, or remove it.
- **D10. Stay within the library's competence.** Keep advice outside scattering statistics to one sentence.
- **D11. English, and no typos** in prose, headings and messages.
- **D12. Every public function has a docstring,** listing the `M-` IDs of `docs/maths.md` it implements.

## E. Pull-request hygiene

- **E1. One topic per PR, branched from `main`.** Mixed PRs are split.
- **E2. Name the follow-up explicitly.** Anything out of scope is called out as a separate PR or issue, with
  what it would contain.
- **E3. Merge pragmatically.** When only small issues remain, say so and merge with named follow-ups;
  unrelated CI failures are stated as unrelated.
- **E5. Workflow hygiene.** GitHub Actions pinned to a commit SHA with a version comment; checkouts with
  `persist-credentials: false`; minimal `permissions` per job.

## Rust (`rust/`, step 2)

These apply in `rust/`, whose local `AGENTS.md` repeats them:

- **A5. Type safety.** Options crossing the Python/Rust boundary are enums parsed once; internals stay
  private to the crate.
- **B2. One parallelism mechanism.** Use the crate's shared parallel-execution helper; custom parallelism
  needs profiling evidence.
- **C6. One `#[test]` per behaviour.**
- **D8. Rust doc comments describe the crate**, referring to items by crate path, not by source file.
- **E4. Package layout.** Mirrored scalar, vectorised and binding layers with identical file names; `mod.rs`
  files only list modules.

## X. ESTEL-specific rules

- **X1. Reference values.** No change to `tests/reference/` or `tests/tolerances.toml` without a bump of
  `STATISTICS_REFERENCE` and CODEOWNER approval (Part I, §5.3 and §7.2).
- **X2. Backend purity.** No `torch`, `jax` or `optax` import outside `src/estel/backend/`; no `estel_rust`
  import outside `src/estel/engines/`.
- **X3. Functional style.** No in-place updates in L1/L2, except in the NumPy plan builders; no control flow
  on array values; static shapes (Part I, §3.1).
- **X4. Second order.** The Python engine stays differentiable twice on PyTorch and JAX: safe modulus, masks as
  weights, no `custom_vjp` (Part I, §3.2).
- **X5. Storage discipline.** No full-grid, NaN-filled tensor inside the computation; full-grid views are for
  output only.
- **X6. Mathematics, oracle and tests in sync.** Every `M-` ID of `docs/maths.md` has a `_spec` function and a
  test, and the reverse.
- **X7. Version impact.** The declared bump matches the change (Part I, §7.2).
- **X8. Actionable errors.** Every error message says what to do next.
- **X9. Step discipline.** Before 1.0.0, no capability that STL lacks, except synthesis checkpoints; STL is
  reproduced as it is, and where it fails ESTEL raises a clear error (Part II, §9.2).
- **X10. Public or internal.** New names are either in `api/manifest.json` or private: in a private module, or
  with a leading underscore (Part I, §2.4).
