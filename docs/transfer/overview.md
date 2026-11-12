# ESTEL: structure proposal and critical review

Nov 6, 2026 · @Erwan Allys

**ESTEL — Estel is a Scattering Transform Exquisite Library**. The successor of STL.

&#91;image: ESTEL — Estel is a Scattering Transform Exquisite Library\]

Draft 0.3 · 12 November 2026 · supersedes draft 0.2 of 6 November 2026 (version notes: Appendices, C). It integrates the decisions taken since then (Part IV, §22), the review round of 12 November 2026 (Part IV, §22.4), and the note *ESTEL: proposal for integrating the Rust engines of STL-Dev-MPI* (decisions Q10–Q23).

## Contents

Each part opens in its own tab, with a detailed contents at its top.

- **Preamble** — below, in this tab.
- Part I — General rules
  - 1\. Principles
  - 2\. Architecture
  - 3\. Numerical coding rules
  - 4\. Mathematics
  - 5\. Tests, reference values, tolerances
  - 6\. CI, GPU access, benchmarks
  - 7\. Versioning and merge policy
  - 8\. Context files for agents
- Part II — Step 1: transfer from STL
  - 9\. Sources and scope of the transfer
  - 10\. What is not transferred
  - 11\. Package tree
  - 12\. Function inventory per module
  - 13\. Mathematics extracted from STL, and open points
  - 14\. Parity protocol with STL
  - 15\. Roadmap to 1.0.0
- Part III — Step 2: later developments
  - 16\. Register of later features
  - 17\. JAX backend
  - 18\. The Rust engine, from STL-Dev-MPI
  - 19\. Compact coefficient layout
  - 20\. New mathematics
- Part IV — Review and decisions
  - 21\. Critical review: failure modes
  - 22\. Decision log
  - 23\. Open questions
  - 24\. Deferred work and points to discuss later
- maths.md — Mathematics of step 1 (draft; its sections have their own numbering)
  - 1\. Planar 2D
  - 2\. 1D (not started)
  - 3\. Tensor products (not started)
  - 4\. HEALPix (not started)
  - 5\. Syntheses (not started)
- Appendices
  - A. Glossary
  - B. References
  - C. Version notes

## Preamble

ESTEL is a Python library for **scattering-transform statistics**. These statistics summarise a field — an image, a time series, a map of the sky, a space–time cube of ocean temperature — by a few thousand numbers. Those numbers capture its textures, filaments and intermittent bursts, far beyond what a power spectrum sees. The same statistics also let one **synthesise** new fields that share them: realistic surrogates to test a method, to fill gaps in data, or to build generative models. ESTEL is the successor of STL and is written from scratch.

**Why a new library.** STL proved the science, but its code grew by accretion. The geometry of the data and the computing backend are fused in the same classes, the same logic is copied in several places, and only three automated tests exist. ESTEL will be developed largely by AI coding agents, without human review of each change. That only works if the structure is fixed in advance, if every rule can be checked by a machine, and if the numbers the library produces are protected.

**What this document is.** This is the **transfer document**. It records what ESTEL takes from STL and from complementary sources, how it takes it, and the rules ESTEL follows from its first commit. The sources are three branches of STL-Dev, the Rust engines of STL-Dev-MPI, the STOC deliverable on tensor products, and the GRID4EARTH review rules. Once the transfer is complete, this document is used to write a second one, the **ESTEL reference document**. That document describes ESTEL on its own terms and no longer refers to STL. It becomes the context that agents and users read, and this transfer document is then archived.

**Two steps.**

- **Step 1 — transfer (ESTEL 1.0.0).** Import only what STL already does, and reproduce its results exactly, in a framework where ESTEL and STL can be compared number by number. Nothing new enters the mathematics.
- **Step 2 — extension (ESTEL 1.x and beyond).** Add new capabilities: a stable JAX backend (experimental from step 1), the Rust engine, with GPU and MPI support, a compact storage of the coefficients, and new mathematics. Each one first lands as an opt-in feature, and becomes a default only through the versioning rules.

&#91;embedded content: ESTEL development · from the STL sources to the reference document\]

The structure is designed **now** for step 2, even though step 1 implements only the results of STL. Every interface that step 2 needs is reserved in 1.0.0: the backend layer, the engine interface, the arguments for distributed data, and an access to the statistics that does not depend on how they are stored. Step 2 can then add code without breaking any user's program.

**The multi-backend strategy.** Running one library on several computing backends — PyTorch and an experimental JAX backend today, a stable JAX backend and a Rust engine tomorrow — is the most critical design point of ESTEL. It rests on three mechanisms:

1. **One thin primitive layer.** Only about twenty low-level functions touch a backend: convolution along chosen axes, padding, weighted gathers, sums by segment, random numbers, differentiation helpers. Everything above them — geometry, scattering loop, statistics, synthesis — is written once, in plain Python. Supporting a new backend means implementing these functions and passing their conformance tests.
2. **Static plans.** Everything that depends only on sizes and parameters is computed once, in NumPy, when an operator is built: wavelet kernels, border masks, index tables, stencils on the sphere. The backend then only runs arithmetic. This is what lets JAX compile the computation, and what keeps the code identical across backends.
3. **Two interchangeable engines.** The whole computation is run by an *engine*, chosen with `engine=`. The **Python engine** runs the code described above, on PyTorch or JAX (chosen with `set_backend()`); it is the readable reference and covers everything. The **Rust engine** is the Rust code of STL-Dev-MPI, with GPU and MPI support; it runs the whole computation by itself, for the configurations it declares. Both sit behind the same public interface and must reproduce the same reference values.

Step 1 implements PyTorch, and JAX as an experimental backend from P1, and follows from its first line of code the rules that make JAX and engines possible (Part I, §3.1).

**Safeguards.** Five devices keep an agent-maintained library trustworthy:

- a frozen public interface, checked in CI against a machine-readable manifest;
- reference values produced by an independent, deliberately simple implementation of the mathematics;
- a version number tied to those reference values, so that the numbers never change within a major version;
- protected files that only humans may change;
- the same context files for every coding agent.

**How to read this document.** Each part has its own tab. Part I sets the rules that apply to ESTEL at every stage. Part II describes step 1: what is transferred from STL, how, and how equality with STL is proven. Part III documents step 2 now, so that step 1 reserves the right interfaces. Part IV gathers the critical review, the decision log and the open questions. The appendices explain the technical terms for non-specialists and list the references.
