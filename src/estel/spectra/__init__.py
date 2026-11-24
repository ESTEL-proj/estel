"""Spectra: the 2D power spectrum and its operator (Part II, §12.8; maths.md §1.9; task T4.1).

``_bins.py`` holds the log-Gaussian bins, their normalisation, the borders of the non-periodic
estimators and the Bessel function J₀ they need, built once in NumPy with the plan;
``_estimators.py`` the four estimators of STL v1.6.0 and the automatic choice, on the L0
primitives; ``_plan.py`` the plan and the filters of ``SpectrumOperator``; and ``_operator.py``
the ``SpectrumOperator``, exported by ``estel``. ``estel.functional.spectrum`` is the pure function
behind its ``apply``. ``STOperator`` computes its spectrum with the same plan of the spectrum and
the same estimators.
"""
