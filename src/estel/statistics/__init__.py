"""Statistics: ``STStatistics``, the compact layout, the normalisation and the reductions
(Part II, §12.7; Part III, §19; tasks T2.5, T2.15, T3.7).

``_statistics.py`` holds ``STStatistics``, exported by ``estel``; ``_layout.py`` the compact
layout of the statistics (``Layout``, ``Segment``: one real buffer, every coefficient STL
computes and only those, described by chunks) and the static maps between buffers, which act
chunk by chunk; ``_grid.py`` the full-grid view of each family (named axes, its chunks);
``_normalization.py`` the normalisation of maths.md M-NORM; ``_reductions.py`` the reductions
``iso``, ``angular_fourier`` and ``scale_cosine`` (M-RED-ISO, M-RED-ANG, M-RED-SCALE), which
``STStatistics.reduce`` applies. In ESTEL "reduction" means these three only.
"""
