# Storage of data and statistics

Summary of [Part I, §2.3 — Object model](../transfer/part1.md#23-object-model) and [Part III, §19 — Compact coefficient layout](../transfer/part3.md#19-compact-coefficient-layout). The transfer document is authoritative.

## Data

- A `Data` object holds one array plus immutable geometry metadata (pbc, cell ids, block structure); it is a pytree whose only leaf is the array. Operations return new objects (`with_array`); nothing is modified in place, and there is no `conv_history`.
- **Masks** of missing data and their reweighting maps are computed once, at construction, and stored in the static plan; they reach the computation as constant weight arrays.

## Statistics: the compact layout (Part III, §19; task T2.15)

Principle: **compute on structured arrays, store in a flat compact buffer, never mix the two.** The conversion is a static gather whose indices are built in NumPy with the plan.

- **Two representations.** Compute: per level, structured arrays, inside `scattering()` only, no NaN (masks are weights). Storage: one real buffer `(B, n_floats)` in `STStatistics`, its only array leaf with those of the reference and the standardisation.
- **What is stored.** Every coefficient STL computes, and only those: no structural NaN, no mean without a valid pixel (M-MEAN, OP-3). Real entries take one float (mean, var, the auto S1 and S2, the diagonal of S4); complex entries take two, (re, im). STL's conjugate duplicates are kept, without packing (Part III, §19.8, option a); the canonical set is LF20.
- **`Layout` and `Segment`** (`src/estel/statistics/_layout.py`, internal): frozen, hashable descriptions, usable as JAX static arguments. One segment per family and level, with its named index table (the position of each entry on the family's full grid), its complex flags and its offset. Within the buffer the families follow the order mean, var, S1 … S4, then the levels, then the order of the full grid.
- **The conversion** from the loop's sums to the buffer is written once, in shared Python on generic primitives (Part III, §19.9): `scattering._plan.loop_sums` gives the order of the loop's sums, `_layout.loop_gather` the static gather, which the loop applies once, at its end.
- **Normalisation, reductions and selection** (`iso`, `angular_fourier`, `scale_cosine`, `select`) are static linear maps on the buffer (Part III, §19.6): each is built in NumPy from the layout, as a copy, a gather or a sparse real matrix in padded CSR form, applied with `take` or `gather_weighted`, and returns a new layout. They are differentiable to second order and backend-neutral.
- **Full-grid views** (`stats.full_grid("S4")`) keep STL's shapes and dtypes, NaN where no coefficient exists, the imaginary part 0 for the real entries. They are built on demand with `scatter_to_grid`, for humans, plots and comparisons with STL, and are never called inside `scattering()` (review rule X5; a test asserts it). Code outside `STStatistics` uses the accessors (`full_grid`, `select`, `to_vector`, `reduce`), never the buffer directly.
- **Files.** `estel.io` writes the buffer and the layout (schema version 3); files of version 2 are refused.
- **Later.** The segments of tensor products, with their multi-index levels, come with their loop (T3.2); packing the conjugate duplicates comes with the canonical set (LF20).
