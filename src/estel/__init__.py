"""ESTEL: scattering-transform statistics and simple syntheses."""

from . import _reference, functional
from ._config import set_default_device, set_default_dtype
from .backend._registry import get_backend, set_backend
from .data._base import Data
from .data._cartesian import Data1D, Data2D
from .data._healpix import DataHealpix
from .data._tensor_prod import DataTensorProd
from .scattering._filters import STFilters
from .scattering._operator import STOperator, STOperatorTensorProd
from .scattering._plan import STPlan
from .statistics._statistics import STStatistics
from .synthesis._api import SynthesisResult, synthesize_from_maps, synthesize_from_stats

# The counters are set in _reference.py, which tools/version_impact.py reads; they are bound here
# so that their docstrings, which a module constant can only carry where it is assigned, are on
# the API reference (D12; issue #190).
STATISTICS_REFERENCE: int = _reference.STATISTICS_REFERENCE
"""The reference counter of the statistics, an integer never reset: it increases by one
whenever any statistic value changes, so that within one value the statistics are unchanged up
to the tolerances of ESTEL's tests (Part I, §7.1). It becomes 1 at the 1.0.0 freeze."""

SYNTHESIS_REFERENCE: int = _reference.SYNTHESIS_REFERENCE
"""The reference counter of the syntheses, an integer never reset: it increases by one whenever
the reference synthesis trajectories change, as ``STATISTICS_REFERENCE`` does for the statistics
(Part I, §7.2). It becomes 1 at the 1.0.0 freeze."""

__all__ = [
    "STATISTICS_REFERENCE",
    "SYNTHESIS_REFERENCE",
    "Data",
    "Data1D",
    "Data2D",
    "DataHealpix",
    "DataTensorProd",
    "STFilters",
    "STOperator",
    "STOperatorTensorProd",
    "STPlan",
    "STStatistics",
    "SynthesisResult",
    "functional",
    "get_backend",
    "set_backend",
    "set_default_device",
    "set_default_dtype",
    "synthesize_from_maps",
    "synthesize_from_stats",
]
