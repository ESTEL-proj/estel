"""The PyTorch backend of the L0 contract (step 1; Part II, §12.3).

The implementation lives in private modules: ``_primitives.TorchBackend`` implements
:class:`estel.backend._protocol.Backend`, and ``_optim`` holds its L-BFGS driver. Nothing here
is public; the backend is reached through the protocol.
"""
