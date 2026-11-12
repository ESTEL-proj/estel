# Storage of data and statistics

Summary of [Part I, §2.3 — Object model](../transfer/part1.md#23-object-model) and [Part III, §19 — Compact coefficient layout](../transfer/part3.md#19-compact-coefficient-layout). The transfer document is authoritative.

## In 1.0.0 (step 1)

- **Data.** A DataClass holds one array plus immutable geometry metadata (pbc, cell ids, block structure, level index); it is a pytree whose only leaf is the array. Operations return new objects (`with_array`); nothing is modified in place, and there is no `conv_history`.
- **Masks** of missing data and their reweighting maps are computed once, at construction, and stored in the static plan; they reach the computation as constant weight arrays.
- **Statistics** keep STL's full-grid storage (NaN where no coefficient exists), but only **behind accessors that do not depend on it** (Part II, §12.7). Code outside `STStatistics` must use the accessors, never index the full grid directly, so that the compact layout can replace it without changing a signature or a value (LF14).
- **No full-grid tensor inside the computation** (review rule X5): the loop computes per-level values, stacks them, and assembles the full-grid view once, at the end, with `scatter_to_grid`.

## After 1.0.0: the compact layout (Part III, §19)

Principle: **compute on structured arrays, store in a flat compact buffer, never mix the two.** The conversion is a static gather whose indices are built in NumPy with the plan.

- **Why.** STL's full grid is mostly NaN: only `j1 ≤ j2 ≤ j3` is filled, the order is componentwise in tensor products, low-pass entries have one orientation, and real families are stored complex. Savings range from 3.9× (2D) to 26× (2D+1D+1D).
- **Two representations.** Compute: per level `(B, n_slots, *spatial)`, inside `scattering()` only, no NaN (masks are weights). Storage: one real buffer `(B, n_floats)` in `STStatistics`, complex entries as (re, im) pairs, absent entries not stored.
- **`Layout` and `Segment`**: frozen, hashable descriptions (family, level, named index table, complex flags, offset) usable as JAX static arguments.
- **Full-grid views** (`stats.family("S4").full_grid()`) are for humans, plots and comparison with 1.0.x; they are never called inside `scattering()`.
- **Reductions** (normalisation, `iso`, `angular_ft`, `scale_ft`) become static linear maps applied per segment.
- **Open points:** the canonical set (keep STL's duplicates or not, LF20) and where the conversion is implemented (R1). The level-wise Gram reorganisation of the loop is adopted only after a benchmark and a human validation.
