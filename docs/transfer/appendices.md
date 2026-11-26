# Appendices

[Back to the overview](overview.md)

## Contents of this tab

- [Appendix A — Glossary](#appendix-a--glossary)
- [Appendix B — References](#appendix-b--references)

## Appendix A — Glossary

| Term | Meaning |
| --- | --- |
| Scattering-transform statistics | Numbers that summarise a field from its responses to wavelets of several scales and orientations, and from the interactions between those responses |
| S1, S2, S3, S4 | The families of statistics: mean modulus and energy of the wavelet responses (S1, S2), and correlations between scales (S3, S4) |
| Wavelet, low-pass filter (φ) | A wavelet selects a band of scales and an orientation; the low-pass filter keeps only the largest scales |
| Smoothing kernel | s, the filter of one downsampling step (M-SMOOTH-2D, M-SMOOTH-1D) |
| Low-pass filter | φ\_J, the effective filter of scale J (M-LP); realised in STL and 1.0.0 as the cascade of the smoothing kernels; never used for s |
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
| Geometry | The kind of domain a field lives on, a node of the tree of Part I, §2.3.1: single-domain (Cartesian, 1D or 2D, and HEALPix) or tensor product (a product of single-domain blocks) |
| Single-domain | Data on one domain, Cartesian or HEALPix: not a tensor product. A tensor product of one block is still a tensor product; the blocks of a tensor product are single-domain |
| Cartesian | The regular grids of 1D and 2D data; "1D" and "2D" always mean Cartesian. "Cartesian product" (of index sets) and "Cartesian direction" keep their usual meaning |

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
