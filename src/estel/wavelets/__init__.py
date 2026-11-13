"""Wavelets: the static banks of filters and the anti-aliasing kernel (Part II, §12.5).

The banks are built in NumPy float64 when an operator is built (tag N): ``_banks.py`` holds the
planar bump-steerable kernels (maths.md M-WAV-2D, M-WAV-1D) and the anti-aliasing kernel H
(M-H, M-H-1D).
"""
