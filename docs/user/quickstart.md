---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Quickstart

This page computes the statistics of a small 2D field, then synthesises a field with the same
statistics. Its code runs at every build of the documentation, and the outputs shown are its
outputs.

## Data

ESTEL computes on `Data` objects, which hold the array with its geometry, its boundary conditions
and its mask. Here, a periodic 32 × 32 field of Gaussian white noise:

```{code-cell} ipython3
import numpy as np

import estel

rng = np.random.default_rng(0)
field = rng.standard_normal((32, 32))
data = estel.Data2D(field, pbc=True)
data.shape
```

`Data2D` promotes the 2D array to the shape `(Nb, Nc, N1, N2)`: one batch element, one channel.
`pbc`, the periodic boundary conditions, has no default. The array is converted to the current
backend, PyTorch by default.

## Statistics

An `STOperator` is built from an example of the data: its geometry, shape, boundary conditions,
mask and number of channels fix the operator, and its values are not read. `J` is the number of
scales and `L` the number of orientations; `apply` returns the statistics, an `STStatistics`.

```{code-cell} ipython3
operator = estel.STOperator(data, J=3, L=4)
stats = operator.apply(data)
{family: tuple(stats.full_grid(family).shape) for family in ["mean", "var", "S1", "S2", "S3", "S4", "PS"]}
```

The statistics come in families: `mean` and `var`, the scattering statistics `S1` to `S4`, and
`PS`, the power spectrum, which the operator computes on 2D data by default. `full_grid` returns
one family as an array of the backend, with the batch axis first; after it come the channel
axes, `(c,)` for mean and var and `(c1, c2)` for the others, then the scales and orientations:
`(j, l)` for S1 and S2, `(j2, j3, l2, l3)` for S3, `(j1, j2, j3, l1, l2, l3)` for S4. The grid
holds NaN where no coefficient exists.

By default (`norm="self"`) the operator normalises the statistics by a reference computed from
the statistics themselves, and keeps it: here, S2 is divided by itself, and is 1.

```{code-cell} ipython3
stats.full_grid("S2")[0, 0, 0].real
```

`norm="none"` gives the statistics without normalisation, and `norm="ref"` normalises them by the
reference of other statistics ([M-NORM](maths.md#m-norm)).

## Reductions and vectors

`reduce` returns new statistics. With `iso=True`, S3 and S4 are averaged over the first
orientation at fixed orientation differences, so S3 has one orientation axis, `dl`
([M-RED-ISO](maths.md#m-red-iso--isotropy)):

```{code-cell} ipython3
tuple(stats.reduce(iso=True).full_grid("S3").shape)
```

`to_vector` flattens the statistics into one real vector, the form a loss takes
([M-FLAT](maths.md#m-flat--flattening)):

```{code-cell} ipython3
tuple(stats.to_vector().shape)
```

## Synthesis

`synthesize_from_maps` starts from a Gaussian white noise and runs L-BFGS on a loss that compares
the statistics of the field with those of the target maps. A short run, of 20 iterations:

```{code-cell} ipython3
result = estel.synthesize_from_maps(field, max_iter=20, st_kwargs={"J": 3})
result.n_iter, float(result.losses[0]), float(result.losses[-1])
```

`result.field` is the synthesised field. Its S2, without normalisation, against that of the
target:

```{code-cell} ipython3
raw = estel.STOperator(data, J=3, L=4, norm="none")
synthesis = estel.Data2D(result.field, pbc=True)
raw.apply(data).full_grid("S2")[0, 0, 0].real, raw.apply(synthesis).full_grid("S2")[0, 0, 0].real
```

## Next

The [API reference](api.md) documents every public name, and the [mathematics](maths.md) page
gives the definition of every statistic.
