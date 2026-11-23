"""Scattering: the static plan, the filters, the loop and the operator (Part II, §12.6).

``_plan.py`` holds the ``STPlan``, the static description of a scattering transform,
built once in NumPy when an operator is built (tag N; Part I, §2.3, §3.1.1); ``_filters.py`` the
``STFilters``, its numerical arrays on the backend; ``_loop.py`` the scattering loop, which
computes the statistics without normalisation; ``_tensor_prod.py`` and ``_loop_tp.py`` the static
description and the loop of tensor products (task T3.2); and ``_operator.py`` the ``STOperator``
and the ``STOperatorTensorProd``. The plan,
the filters and the operator are exported by ``estel`` (task T2.5).
"""
