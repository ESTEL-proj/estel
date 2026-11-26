# ESTEL

ESTEL computes scattering-transform statistics of fields: their mean and variance, the
scattering statistics S1 to S4 and, on 2D grids, the power spectrum. It works on 2D and 1D grids,
on tensor products of them and on the HEALPix sphere, with optional masks and several channels,
and it synthesises fields whose statistics match those of target maps. It runs on PyTorch, and on
an experimental JAX backend; the statistics are differentiable with respect to the data.

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} Install
:link: install
:link-type: doc

Install ESTEL and PyTorch.
:::

:::{grid-item-card} Quickstart
:link: quickstart
:link-type: doc

Statistics of a 2D field, and a synthesis.
:::

:::{grid-item-card} API reference
:link: api
:link-type: doc

Every public name.
:::

:::{grid-item-card} Mathematics
:link: maths
:link-type: doc

The definitions behind every statistic.
:::

:::{grid-item-card} Contributing
:link: contributing
:link-type: doc

How changes to ESTEL are made.
:::
::::

```{toctree}
:hidden:

install
quickstart
api
maths
contributing
```
