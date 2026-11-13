"""Planar data: ``Data1D`` and ``Data2D`` (Part II, §12.4; maths.md M-DATA-1D, M-DATA-2D).

The two classes share everything but the number of spatial axes and the threshold τ of the
masked downsampling (maths.md M-DOWN-MASK, OP-12): STL's 1D class inherits from its 2D class
and differs only there (STL `2D+1D`, `STL_1D_Kernel_Torch`).
"""

from __future__ import annotations

from typing import Any, ClassVar

from ._base import DataClass


class Data2D(DataClass):
    """2D planar data, shape ``(Nb, Nc, N1, N2)``.

    ``Data2D(array, mask=None, *, pbc)``: see :class:`estel.DataClass` for the parameters.
    A 2D array is promoted to ``(1, 1, N1, N2)``, a 3D array to ``(1, Nc, N1, N2)``
    (maths.md M-DATA-2D).
    """

    ndim: ClassVar[int] = 2
    _nan_threshold: ClassVar[float] = 0.33

    def __init__(self, array: Any, mask: Any = None, *, pbc: bool) -> None:
        super().__init__(array, mask, pbc=pbc)


class Data1D(DataClass):
    """1D planar data, shape ``(Nb, Nc, N)``.

    ``Data1D(array, mask=None, *, pbc)``: see :class:`estel.DataClass` for the parameters.
    A 1D array is promoted to ``(1, 1, N)``; a 2D array to ``(1, Nc, N)``, its first axis read
    as channels, not as a batch, as in STL (maths.md M-DATA-1D, OP-29).
    """

    ndim: ClassVar[int] = 1
    _nan_threshold: ClassVar[float] = 0.5

    def __init__(self, array: Any, mask: Any = None, *, pbc: bool) -> None:
        super().__init__(array, mask, pbc=pbc)
