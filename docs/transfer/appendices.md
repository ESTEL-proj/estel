# Appendices

[Back to the overview](overview.md)

## Contents of this tab

- [Appendix A — Glossary](#appendix-a--glossary)
- [Appendix B — References](#appendix-b--references)
- [Appendix C — Version notes](#appendix-c--version-notes)

## Appendix A — Glossary

| Term | Meaning |
| --- | --- |
| Scattering-transform statistics | Numbers that summarise a field from its responses to wavelets of several scales and orientations, and from the interactions between those responses |
| S1, S2, S3, S4 | The families of statistics: mean modulus and energy of the wavelet responses (S1, S2), and correlations between scales (S3, S4) |
| Wavelet, low-pass filter (φ) | A wavelet selects a band of scales and an orientation; the low-pass filter keeps only the largest scales |
| Box term | The statistics of the low-passed field, with its mean removed |
| Tensor product, block | A field defined on several domains at once (space × time, space × depth); each domain is a block, with its own wavelets |
| pbc | Periodic boundary conditions: whether a domain wraps around |
| NaN mask | The set of missing pixels, handled as weights |
| Synthesis | Generating a new field whose statistics match a target |
| Backend | The numerical library that runs the computation: PyTorch, JAX |
| Primitive (L0) | One of the twenty-odd low-level functions every backend must provide |
| Array API standard | A common interface for array libraries in Python, which ESTEL uses through `array-api-compat` |
| Static plan | Everything computed once, in NumPy, when an operator is built |
| Pytree | A JAX notion: a structure whose arrays JAX can trace through, the rest being fixed metadata |
| `jit` | Just-in-time compilation of a whole computation (JAX) |
| VJP, JVP | Vector–Jacobian product (reverse mode, gradients) and Jacobian–vector product (forward mode) |
| Hessian-vector product (HVP) | A second-order derivative applied to a vector, without forming the Hessian |
| Implicit differentiation | Differentiating the solution of an optimisation problem with respect to its parameters, through the implicit function theorem |
| L-BFGS | A quasi-Newton optimiser, used for syntheses |
| Gram matrix | The matrix of all pairwise inner products of a set of vectors; a covariance matrix is one |
| Full-grid view | The statistics laid out on the complete grid of indices, with NaN where no coefficient exists. Built by scatter\_to\_grid; gather\_from\_grid reads the existing coefficients back |
| Compact layout | Storage of the statistics as one flat buffer holding only the existing coefficients |
| Oracle (`_spec`) | A deliberately simple implementation of the mathematics, independent of the library, that produces reference values |
| Reference values | Stored outputs of fixed inputs, which the library must reproduce within tolerance |
| Parity | Equality of results between ESTEL and STL on the same inputs |
| Engine | An implementation of the whole computation behind the ESTEL API: the Python engine (built on PyTorch or JAX) or the Rust engine |
| MPI | Message Passing Interface: a standard for computing across many processes or machines |
| ABI | Application binary interface: the binary contract between ESTEL and a compiled engine |
| Wheel | A built Python package, possibly containing compiled code |
| CODEOWNERS | A GitHub file naming who must approve changes to given paths |
| Major, minor, patch | The three numbers of a version (major.minor.patch), with the meanings of Part I, §7.1 |
| Reference counters | STATISTICS\_REFERENCE and SYNTHESIS\_REFERENCE: two integers, never reset, bumped whenever statistics or reference synthesis trajectories change; the same number means the same values (Part I, §7.1) |
| Public / internal | Public names are those of api/manifest.json, frozen within a major version; everything else lives in a private module or starts with an underscore, and may change (Part I, §2.4) |

## Appendix B — References

**Sources of the transfer**

- STL-Dev: [github.com/STL-Proj/STL-Dev](https://github.com/STL-Proj/STL-Dev), branches [`master`](https://github.com/STL-Proj/STL-Dev/tree/master), [`2D+1D`](https://github.com/STL-Proj/STL-Dev/tree/2D%2B1D) and [`HealpixUpdate`](https://github.com/STL-Proj/STL-Dev/tree/HealpixUpdate).
- STL-Dev-MPI, commit `6f0f870`, private repository (access on request to its author).
- STOC deliverable 1, *Development of the code* (block approach), 30 October 2026.
- [ESTEL: proposal for integrating the Rust engines of STL-Dev-MPI](https://claude.ai/artifact/JoNB1adBaYoMW3PgK5Xd6k), 5 November 2026.
- GRID4EARTH review rules: [github.com/GRID4EARTH/review-rules](https://github.com/GRID4EARTH/review-rules).
- healpix-analyse: [pypi.org/project/healpix-analyse](https://pypi.org/project/healpix-analyse/).

**Tools and standards**

- Python Array API standard: [data-apis.org/array-api](https://data-apis.org/array-api/latest/).
- array-api-compat: [github.com/data-apis/array-api-compat](https://github.com/data-apis/array-api-compat).
- optax L-BFGS: [optax.readthedocs.io](https://optax.readthedocs.io/en/latest/api/generated/optax.lbfgs.html).
- import-linter: [import-linter.readthedocs.io](https://import-linter.readthedocs.io/).
- PEP 541, package index name retention: [peps.python.org/pep-0541](https://peps.python.org/pep-0541/).

## Appendix C — Version notes

Times in UTC; "\~" marks an approximate time.

| Version | When | By | Changes |
| --- | --- | --- | --- |
| 0.1 | 4 Nov, \~18:40 | Claude, for Erwan Allys | First draft: structure proposal and critical review, 18 sections |
| 0.1 | 5 Nov, \~12:40 | Claude | Table of contents added |
| 0.1 | 5 Nov, 14:17–15:57 | Pablo, Erwan Allys | 15 comments on draft 0.1 |
| 0.2 | 6 Nov, \~09:00–09:45 | Claude, for Erwan Allys | Full rewrite: preamble, Parts I–IV, step 1 / step 2 split, two-step diagram, note on the Rust engines integrated, comments addressed |
| 0.2 | 6 Nov, 09:49–09:50 | Erwan Allys | Comments resolved |
| 0.2.1 | 6 Nov, \~10:00–10:15 | Claude | Split into six tabs; detailed contents in each part tab |
| 0.2.2 | 6 Nov, \~10:20–10:30 | Erwan Allys, Claude | Overview contents: one section per line; "recursive acronym" removed |
| 0.2.3 | 6 Nov, \~10:50 | Claude, for Erwan Allys | Layers clarified: an engine layer with two engines (Python, native); PyTorch and JAX as backends of the Python engine; layer diagram redrawn; "L3" dropped; this appendix added |
| 0.2.4 | 6 Nov, \~11:00 | Claude, for Erwan Allys | Register of later features renumbered FL1–FL22 (was L1–L22, easily confused with the layers L0–L2); remaining layer mentions in Parts III and IV aligned on "Python engine" and "native engine" |
| 0.2.5 | 6 Nov, \~11:10 | Claude, for Erwan Allys | New Part I, §3.1.1, "The scattering loop without in-place updates"; "NaN-padded view" renamed "full-grid view" (full\_grid()), primitive scatter\_dense renamed scatter\_to\_grid, with its inverse gather\_from\_grid |
| 0.2.6 | 6 Nov, \~11:35 | Claude, for Erwan Allys | Register of later features grouped in four categories (structure, new features, optimisation, mathematical choices) with a "Changes reference values" column, and renumbered LF1–LF22 in category order (the FL numbers of 0.2.4 are retired); package tree (§11) shows where step-2 components will live |
| 0.2.7 | 6 Nov, \~11:40–11:50 | Claude, for Erwan Allys | Register sub-sections numbered 16.1–16.4 and added to the Part III contents; contents added to the Appendices tab; ESTEL logo added to the Overview; Part I §2.4: a tensor-product object holds one array, all blocks on the same backend; Part I §3.2.3: note on why a single where is not enough |
| 0.2.8 | 12 Nov | Claude, for Erwan Allys | New tab "maths.md — Mathematics of step 1": first draft of the 2D part, extracted from the STL code (2D+1D 8f51143, master 8eaf769), with 27 open points OP-1 to OP-27; this tab is the reference version of maths.md during the review. Organised in five sections (planar 2D, 1D, tensor products, HEALPix, syntheses); only 2D is written so far. Part II §13.2: correspondence table from the gaps G1–G18 to the open points |
| 0.3 | 12 Nov | Claude, for Erwan Allys, from the answers of Erwan Allys to the review points (C1–C23, S1–S18, A1–A12) | Review round applied throughout (Part IV, §22.4): "native" renamed "rust" and the Rust engine moved into the ESTEL repository; healpix-analyse ported (test-only, pinned at aac5e26); STL sources frozen; 1.0.0 spectra limited to 2D (master); public/internal rule; class names in CapWords; static masks; versioning with two reference counters; CODEOWNERS and end-of-step review; new invariance tests; synthesis tests after 1.0.0, checkpoints in step 1; diagnostics, mask\_st, \_normalize; Part II §13.2 regrouped into four groups with G3 explicit; work breakdown of step 1 (Part II §15.2); register extended to LF31 with a Source column; Part IV §24 (deferred work, points to discuss); open questions R11–R13. Part I §1.5 (Licensing) had been added beforehand in another editing session; its rename to "rust" is the only change made to it |
| 0.3.1 | 12 Nov | Claude, for Erwan Allys | Licence: Apache-2.0, as written by Pablo in Part I §1.5 (R5, R11 answered); CODEOWNERS handles jmdelouis and pablo-richard (R12); minimum versions confirmed (R13); work breakdown: the maths tab sections are written at the start of the phase that needs them (T2.0, T3.0, T5.0, T6.0) |
| 0.3.2 | 12 Nov | Claude, for Erwan Allys | Philosophy of step 1 stated (Part II §9.2); errors where STL fails (OP-3, OP-8); Morlet excluded (OP-24, Part II §10); masked parity cases from P2 (OP-26, OP-27); OP-5 traced as LF32; R1 and R2 deferred to step 2 |
