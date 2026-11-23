"""The loss a synthesis minimises (maths.md M-SYN-LOSS; Part II, §12.9; task T6.1).

:class:`Objective` holds what is fixed during a run: the running data's metadata, the plan
and filters of the running operator, the target's reference and its flattened statistics.
Called on the running field u, it computes the statistics of u with ``norm="ref"`` and the
target's reference, weights S3 and S4, flattens them in the complex form of M-FLAT
(``STStatistics._to_complex_vector``) and returns the pure loss of ``estel.functional``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..data._base import Data
from ..functional._loss import loss
from ..functional._scattering import compute
from ..scattering._filters import PlanarFilters
from ..scattering._plan import PlanarPlan
from ..statistics._normalization import _NormReference
from ..statistics._statistics import STStatistics

WEIGHTS: dict[str, float] = {"S3": 3.5, "S4": 3.5**2}
"""STL's ``adhoc_weights``, the default weights of the loss (M-SYN-LOSS, OP-51)."""


def flatten(stats: STStatistics, mean_field: bool, keep_batch_dim: bool) -> Any:
    """The weighted statistics in the complex form of M-FLAT, as the loss compares them:
    averaged over the batch with ``mean_field`` (M-SYN-LOSS, "Batch")."""
    return stats._to_complex_vector(
        keep_batch_dim=keep_batch_dim, mean_along_batch=mean_field, weights=WEIGHTS
    )


@dataclass(frozen=True, eq=False)
class Objective:
    """u ↦ Σᵢ |w Φᵢ(u) - w Φᵢ(target)|², divided by ``divisor`` (M-SYN-LOSS)."""

    template: Data
    """The running data: ``u`` replaces its array (``Data.with_array``)."""
    plan: PlanarPlan
    """The plan of the running operator, with ``norm="ref"``."""
    filters: PlanarFilters
    reference: _NormReference
    """The target's reference (``var_ref``, ``S2_ref_sqrt_chan_diag``), a constant."""
    target: Any
    """The target's flattened, weighted statistics, a constant."""
    mean_field: bool
    keep_batch_dim: bool
    """STL's ``keep_batch_dim``: False from maps, True from statistics."""
    divisor: int
    """1 from maps; from statistics with ``mean_field``, the target's Nb (OP-71)."""

    def __call__(self, u: Any) -> Any:
        stats = compute(self.template.with_array(u), self.plan, self.filters, self.reference)
        value = loss(flatten(stats, self.mean_field, self.keep_batch_dim), self.target)
        return value if self.divisor == 1 else value / self.divisor
