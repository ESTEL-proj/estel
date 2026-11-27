"""The checks shared by the banks and the tables of every geometry (Part I, §2.3.1; decision on
#197).

NumPy only, run once when an operator is built (tag N of Part II, §12). A bare module: what it
holds serves every single-domain geometry, Cartesian and HEALPix, and the blocks of tensor
products. ``_operator.py``, ``_banks_cartesian.py``, ``_banks_healpix.py`` and the HEALPix
modules import from it, and it imports nothing from ``estel``, so that the banks import it
without a cycle.

* :func:`_check_n_orientations`: L, an integer ≥ 1, as STL accepts (Part II, §12.5);
* :func:`_check_zero_sum`: every band-pass kernel sums to zero within rounding (Part III,
  §20.11, rule 3).
"""

from __future__ import annotations

from numbers import Integral

import numpy as np
from numpy.typing import NDArray

_ZERO_SUM_RTOL = 1e-12
"""A band-pass kernel passes the zero-sum check when |sum| <= this times max|kernel|, in float64
(Part III, §20.11, rule 3)."""


def _check_n_orientations(value: object) -> int:
    """L as an int: an integer ≥ 1, as STL accepts (Part II, §12.5)."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"L={value!r}: the number of orientations is an integer ≥ 1, such as L=4.")
    n = int(value)
    if n < 1:
        raise ValueError(
            f"L={n}: the number of orientations is at least 1; the default is 4 in 2D and 1 in 1D."
        )
    return n


def _check_zero_sum(kernels: NDArray[np.complexfloating], ndim: int, name: str) -> None:
    """Raise unless every band-pass kernel sums to zero within rounding.

    ``kernels`` holds one kernel per index of its leading axis, over its last ``ndim`` axes. Each
    must satisfy |sum| <= 1e-12 max|kernel|, computed in float64 (Part III, §20.11, rule 3: the
    centring rule of step 2 relies on this guarantee of the filter bank).
    """
    axes = tuple(range(-ndim, 0))
    values = np.asarray(kernels, dtype=np.complex128)
    sums = np.abs(values.sum(axis=axes)).reshape(-1)
    peaks = np.abs(values).max(axis=axes).reshape(-1)
    for index, (total, peak) in enumerate(zip(sums, peaks, strict=True)):
        if not total <= _ZERO_SUM_RTOL * peak:
            raise ValueError(
                f"{name}: kernel {index} is not zero-sum: |sum| = {total:.3e} exceeds "
                f"{_ZERO_SUM_RTOL:g} x max|kernel| = {_ZERO_SUM_RTOL * peak:.3e}. Every "
                "band-pass kernel must sum to zero (Part III, §20.11, rule 3); subtract its mean "
                "in float64 after the crop."
            )
