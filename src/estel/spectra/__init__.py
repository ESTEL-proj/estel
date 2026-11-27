"""Spectra: the bricks of the 2D power spectrum (Part II, §12.8; maths.md §1.9; tasks T4.1, T4.6).

``_bins.py`` holds the log-Gaussian bins, their normalisation, the borders of the non-periodic
estimators and the Bessel function J₀ they need, built once in NumPy with the plan;
``_estimators.py`` the description of the spectrum in the plan of ``STOperator``
(``SpectrumEstimatorPlan2D``, ``build_spectrum_plan_2d``), its filters, and the four estimators
of STL v1.6.0 and the automatic choice, on the L0 primitives. ``STOperator`` computes its
spectrum with them (``compute_PS``, ``n_bins``, ``Jmin``, ``spectrum_method``; task T4.6), over
the band of its J. The bricks named ``2D`` compute on 2D grids only, the others hold nothing of
2D (Part I, §2.3.1, §2.4.1; task T7.9). Nothing here imports from ``estel.scattering``.
"""
