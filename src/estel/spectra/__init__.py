"""Spectra: the bricks of the 2D power spectrum (Part II, §12.8; maths.md §1.9; tasks T4.1, T4.6).

``_bins.py`` holds the log-Gaussian bins, their normalisation, the borders of the non-periodic
estimators and the Bessel function J₀ they need, built once in NumPy with the plan;
``_estimators.py`` the four estimators of STL v1.6.0 and the automatic choice, on the L0
primitives. ``STOperator`` computes its spectrum with them (``compute_PS``, ``J_PS``,
``Jmin_PS``, ``n_bins``, ``spectrum_method``; task T4.6), from the description of the spectrum
in its plan (``scattering._plan.SpectrumPlan``).
"""
