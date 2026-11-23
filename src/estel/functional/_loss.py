"""The pure loss of the syntheses (Part II, §11 and §12.9; maths.md M-SYN-LOSS; task T6.1).

Internal (tag I of §12.9): the synthesis driver (``estel.synthesis``) calls it on the complex
vectors of M-FLAT (``STStatistics._to_complex_vector``), weighted beforehand.
"""

from __future__ import annotations

from typing import Any

from ..statistics._arrays import backend_object_of


def loss(values: Any, target_values: Any) -> Any:
    """Σᵢ |values_i - target_values_i|², a real 0-d array (maths.md M-SYN-LOSS).

    ``values`` and ``target_values`` are complex arrays of one shape and dtype, the flattened
    statistics of the running field and of the target. The squared modulus is taken as
    (Re z)² + (Im z)², the squared Euclidean norm of the complete real vector: its value is
    STL's ``(s - t).abs() ** 2`` summed, and its gradient is 0 where z = 0, as PyTorch's
    gradient of STL's form.
    """
    backend = backend_object_of(values)
    xp = backend.xp
    pairs = backend.complex_to_real_pairs(values - target_values)
    return xp.sum(pairs * pairs)
