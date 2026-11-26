# Install

ESTEL needs Python 3.11 or later, NumPy 2.0 or later and PyTorch 2.6 or later. Until its first
release, install it from its repository:

```bash
pip install "estel @ git+https://github.com/ESTEL-proj/estel"
```

pip installs PyTorch from PyPI if it is missing. For a CPU-only build or a build for a given CUDA
version, install PyTorch first, as [its instructions](https://pytorch.org/get-started/locally/)
say; pip then keeps it.

The experimental JAX backend needs JAX, which the extra `jax` installs:

```bash
pip install "estel[jax] @ git+https://github.com/ESTEL-proj/estel"
```

## From a clone

The development environment is managed with [uv](https://docs.astral.sh/uv/). In a clone of the
repository, `uv sync` creates it in `.venv`, with the tools of the tests; on Linux it takes the
CPU-only build of PyTorch.

```bash
git clone https://github.com/ESTEL-proj/estel
cd estel
uv sync
```
