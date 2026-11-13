"""ESTEL: scattering-transform statistics and simple syntheses."""

from ._config import set_default_device, set_default_dtype
from ._reference import STATISTICS_REFERENCE, SYNTHESIS_REFERENCE
from .backend._registry import BackendMismatchError, get_backend, set_backend
from .data._base import DataClass
from .data.planar import Data1D, Data2D

__all__ = [
    "STATISTICS_REFERENCE",
    "SYNTHESIS_REFERENCE",
    "BackendMismatchError",
    "Data1D",
    "Data2D",
    "DataClass",
    "get_backend",
    "set_backend",
    "set_default_device",
    "set_default_dtype",
]
