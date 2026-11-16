"""Wavelets: the wavelet operator, the static banks of filters and the smoothing kernel
(Part II, §12.5).

``_operator.py`` holds the internal ``WaveletOperator``, the only way to the wavelets and to the
geometry of the pyramid (Part I, §2.3): the rest of ESTEL goes through it, and imports neither
the banks nor the mask builders of ``estel.data``.

The banks are built in NumPy float64 when an operator is built (tag N): ``_banks.py`` holds the
planar bump-steerable kernels (maths.md M-WAV-2D, M-WAV-1D) and the smoothing kernel, the
anti-aliasing filter applied before each downsampling (M-SMOOTH-2D, M-SMOOTH-1D).
"""
