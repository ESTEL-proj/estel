# User guide

The guides are being written. Until they come, the [quickstart](../quickstart.md) shows the main
calls, and the [API reference](../api.md) documents every option. They will cover:

- **Data types**: 2D and 1D grids, tensor products of them, and the HEALPix sphere; batches and
  channels.
- **Normalisation and reductions**: the normalisations `self`, `ref` and `none`; isotropy, the
  angular Fourier transform and the scale cosine transform.
- **Masks**: missing values, and how they enter each statistic.
- **Syntheses**: from target maps or from statistics, and their checkpoints.
- **Backends**: PyTorch and the experimental JAX backend, dtypes and devices.
