"""ESTEL: scattering-transform statistics and simple syntheses."""

from . import functional
from ._config import set_default_device, set_default_dtype
from ._reference import STATISTICS_REFERENCE, SYNTHESIS_REFERENCE
from .backend._registry import BackendMismatchError, get_backend, set_backend
from .data._base import Data
from .data._planar import Data1D, Data2D
from .scattering._filters import STFilters
from .scattering._operator import STOperator
from .scattering._plan import STPlan
from .statistics._statistics import STStatistics

__all__ = [
    "STATISTICS_REFERENCE",
    "SYNTHESIS_REFERENCE",
    "BackendMismatchError",
    "Data",
    "Data1D",
    "Data2D",
    "STFilters",
    "STOperator",
    "STPlan",
    "STStatistics",
    "functional",
    "get_backend",
    "set_backend",
    "set_default_device",
    "set_default_dtype",
]
