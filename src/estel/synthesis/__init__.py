"""Syntheses: fields whose statistics match a target (Part II, §12.9; maths.md section 5;
task T6.1).

``_api.py`` holds ``synthesize_from_maps``, ``synthesize_from_stats`` and ``SynthesisResult``,
exported by ``estel``, and the driver of L-BFGS between checkpoints; ``_init.py`` the Nyquist
pre-filter, the standardisation of the target and the Gaussian start; ``_loss.py`` the loss on
the running field; ``_checkpoints.py`` the checkpoint files. In 1.0.0 the fields are periodic,
planar 2D and without missing values (LF47), on the PyTorch backend (LF1).
"""
