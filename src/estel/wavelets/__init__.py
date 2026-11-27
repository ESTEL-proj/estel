"""Wavelets: the wavelet operator, the static banks of filters and the smoothing kernel
(Part II, §12.5).

``_operator.py`` holds the internal ``WaveletOperator``, the only way to the wavelets and to the
geometry of the pyramid (Part I, §2.3): the rest of ESTEL goes through it, and imports neither the
banks nor the mask builders of ``estel.data``.

The modules of one geometry are named after it (Part I, §2.4.1, rule 3); ``_operator.py`` is
bare, and holds what every geometry shares. The banks are built in NumPy float64 when an
operator is built (tag N): ``_banks_cartesian.py`` holds the bump-steerable kernels of 1D and 2D
data (maths.md M-WAV-2D, M-WAV-1D) and the smoothing kernel, the anti-aliasing filter applied
before each downsampling (M-SMOOTH-2D, M-SMOOTH-1D); ``_banks_healpix.py`` the HEALPix kernel
(M-WAV-HPX), ``_tables_healpix.py`` the HEALPix tables of the convolution and of the
decimation, ported from STL's ``TorchGeometry`` (M-CONV-HPX, M-DOWN-HPX), and
``_pixels_healpix.py`` the pixel functions they use. ``_operator_healpix.py`` holds the static
part and the arrays of the wavelet operator of HEALPix data, built on these tables (M-MASK-HPX,
M-DEF-HPX), and ``_operator_tensor_prod.py`` the wavelet operator of a tensor product, composed
from those of its blocks. ``_checks.py`` holds the checks that the banks and tables of every
geometry share.
"""
