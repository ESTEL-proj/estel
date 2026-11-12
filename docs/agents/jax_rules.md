# Functional, JAX-ready style

Summary of [Part I, §3.1 — Functional, JAX-ready style](../transfer/part1.md#31-functional-jax-ready-style). The transfer document is authoritative.

These rules apply to L1 and L2 from the first commit of step 1, although JAX arrives in step 2. From phase P1, tier 0 traces L2 with a minimal JAX L0 to prove the code stays traceable.

## Rules

1. **Functional code only.** Build per-level results in Python lists, then `concat` or `stack`. No preallocation filled by index, no `out=`, no `inplace=`. (Plan-building NumPy code, run once at construction, is exempt.)
2. **Static shapes.** Index sets, crop widths, stencils, bins and mean denominators live in the `ScatteringPlan`, built in NumPy and passed as a static argument.
3. **No decisions on array values in traced code.** No `if x.isnan().any()`, `.item()`, `float(x)`, `print(x)`. NaN detection and estimator selection happen at construction; their masks and maps are static plan data.
4. **Large constants are arguments**, not closures: kernels, masks and stencils are passed in `arrays`.
5. **Explicit randomness.** Every random function takes a key or a seed (`make_key(seed)`).
6. **Complex convolutions are two real convolutions**; kernels are stored as real pairs.
7. **float32 by default, float64 on request.** No hard-coded float64 constant, no silent dtype change; plan arrays are cast to the operator's dtype at construction; tests run in both dtypes.
8. **Short Python loops.** JAX unrolls them. Step 1 keeps STL's loop structure (loops over scales and levels, never over coefficients); reorganising it is step 2 work, after a benchmark and human validation.
9. **Data and statistics are pytrees** (arrays as leaves, metadata static).

## The scattering loop without in-place updates ([§3.1.1](../transfer/part1.md#311-the-scattering-loop-without-in-place-updates))

```python
# STL: preallocate, then write in place  — do not do this
S4 = torch.full((Nb, Nc, Nc, J, J, J, L, L, L), nan)
for j3 in range(J): ...
    S4[:, :, :, j1, j2, j3] = cov(U[j1, j3], U[j2, j3])

# ESTEL: compute, collect, assemble
values = [cov_valid(u[j1, j3], u[j2, j3], plan.weights[j3])
          for (j1, j2, j3) in plan.s4_triples]     # static list, built in NumPy
s4 = xp.stack(values, axis=-1)
x_coarse = geometry.downsample(x, level)           # new object; x untouched
```

- Every step returns a value; data objects are never modified (`with_array`).
- The full-grid view is assembled once, at the end, with `scatter_to_grid`; `gather_from_grid` reads it back.
- **Missing data are weights:** means are `sum(w * x) / sum(w)` with `w` in {0, 1}, never `x[mask]` or `nanmean`; the denominators come from the plan.

See also [differentiation.md](differentiation.md): the same rules keep second-order derivatives correct in PyTorch.
