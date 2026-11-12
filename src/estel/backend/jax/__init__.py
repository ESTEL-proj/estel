"""The experimental JAX backend of the L0 contract (from P1; Part I, §2.2.5; Part II, §12.3).

The implementation lives in a private module: ``_primitives.JaxBackend`` implements
:class:`estel.backend._protocol.Backend`. Nothing here is public; the backend is reached
through the protocol. It is opt-in (``pip install estel[jax]``), not covered by the 1.0.0
guarantees, and has no L-BFGS before step 2 (LF1).
"""
