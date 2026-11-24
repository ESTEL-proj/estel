"""Wavelets: the wavelet operator, the static banks of filters and the smoothing kernel
(Part II, §12.5).

``_operator.py`` holds the internal ``WaveletOperator``, the only way to the wavelets and to the
geometry of the pyramid (Part I, §2.3): the rest of ESTEL goes through it, and imports neither
the banks nor the mask builders of ``estel.data``.

The banks are built in NumPy float64 when an operator is built (tag N): ``_banks.py`` holds the
planar bump-steerable kernels (maths.md M-WAV-2D, M-WAV-1D) and the smoothing kernel, the
anti-aliasing filter applied before each downsampling (M-SMOOTH-2D, M-SMOOTH-1D);
``_healpix_bank.py`` the HEALPix kernel (M-WAV-HPX), and ``_healpix_geometry.py`` the HEALPix
tables of the convolution and of the decimation, ported from STL's ``TorchGeometry``
(M-CONV-HPX, M-DOWN-HPX). ``_healpix_operator.py`` holds the static part and the arrays of the
wavelet operator of HEALPix data, built on these tables (M-MASK-HPX, M-DEF-HPX).
"""
