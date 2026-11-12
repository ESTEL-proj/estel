# Differentiation contract

Summary of [Part I, §3.2 — Differentiation contract](../transfer/part1.md#32-differentiation-contract). The transfer document is authoritative.

## Why second order ([§3.2.1](../transfer/part1.md#321-why-second-order))

Synthesis needs only gradients. Downstream implicit differentiation (dθ\*/dφ = −[∂²L/∂θ²]⁻¹ ∂²L/∂θ∂φ) needs Hessian-vector products and mixed second derivatives, so ESTEL guarantees order two for its statistics. This changes no formula; it only forbids some shortcuts.

## Guarantees ([§3.2.2](../transfer/part1.md#322-guarantees))

| Guarantee | PyTorch (step 1) | JAX (experimental from P1) | Test |
| --- | --- | --- | --- |
| VJP of `scattering`, `cross_spectrum`, `loss` | autograd | `jax.vjp` | `gradcheck`, `check_grads(order=1)` |
| JVP | `torch.func.jvp` | `jax.jvp` | `check_grads(modes=("fwd",))` |
| Second order | double backward, `torch.func` | `jax.jvp(jax.grad(...))` | `gradgradcheck`, `check_grads(order=2, modes=("fwd", "rev"))` |
| Implicit differentiation, end to end | | | small argmin problem vs finite differences |

All differentiation tests run in float64 on small inputs.

## Implementation rules ([§3.2.3](../transfer/part1.md#323-implementation-rules))

1. **No `custom_vjp` in the Python engine** (it removes JAX forward mode). If a custom rule is needed, use `custom_jvp`.
2. **No in-place operations** in L1 and L2: they break double backward.
3. **Safe modulus, double `where`:** `m = |z|² > 0; r = sqrt(where(m, |z|², 1)); |z| = where(m, r, 0)`. A single `where` fixes the value but not the gradient (0 × NaN = NaN from the discarded branch). Same care for STL's regularised division `z / sqrt(|z| + 1e-8)`.
4. **Masks are weights**, never boolean indexing or `nanmean`.
5. **The reference is an explicit argument:** in `apply(data, ref)`, gradients flow into `ref` unless the caller stops them.

## Engines ([§3.2.4](../transfer/part1.md#324-engines-and-differentiation-order))

Every engine declares `diff_order`. The Python engine has order 2 on PyTorch and JAX. An order-1 engine (the Rust engine of step 2) raises an error pointing to `engine="python"` when asked for a JVP, a Hessian-vector product or a double backward.
