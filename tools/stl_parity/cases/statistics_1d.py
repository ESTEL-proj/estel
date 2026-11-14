"""Parity case: the normalised and reduced statistics of STL `2D+1D` on 1D data (Part II, §14.2;
task T2.5, issue #17; maths.md M-NORM, M-RED-ISO, M-RED-ANG, M-RED-SCALE, M-FLAT, OP-3, OP-30,
OP-37).

For each configuration ``<config>``, runs STL's scattering operator (``ST_Operator.apply``)
without spectrum (``compute_PS=False``):

* with ``norm="store_ref"`` on a target field, and the configuration's ``norm_batch_mean``; the
  operator then holds the reference;
* with ``norm="load_ref"`` on a second field with the same mask, normalised by the target's
  reference;
* then, on copies of the ``store_ref`` statistics, the reductions of ``ST_Statistics``, in STL's
  order: ``to_iso``; ``to_angular_ft`` without ``iso``; ``to_iso`` then ``to_angular_ft``;
  ``to_iso`` then ``to_scale_ft``; ``to_iso``, ``to_angular_ft`` then ``to_scale_ft``, with the
  configuration's harmonics; and ``to_flatten(flatten_complex=True)`` of the normalised and of
  the fully reduced statistics.

It stores, for each configuration:

* ``<config>_field`` and ``<config>_ref_field``: the target and the second field,
  ``(Nb, Nc, N)``, NaN at the missing samples; ``<config>_mask`` (absent without a mask);
* ``<config>_store_<family>`` and ``<config>_load_<family>`` for ``mean``, ``var``, ``S1`` to
  ``S4``; ``<config>_var_ref`` and ``<config>_s2_ref``: the reference the operator stored
  (``var_ref``, ``S2_ref_sqrt_chan_diag``);
* ``<config>_<reduction>_<family>`` for the reductions ``iso`` (S1 to S4), ``angular`` (S3, S4),
  ``iso_angular`` (S3, S4), ``iso_scale`` (S3, S4) and ``iso_angular_scale`` (S3, S4);
* ``<config>_vector`` and ``<config>_vector_reduced``: the flat real vectors.

The configurations cover the cells of §14.4 that T2.5 owns: periodic and open data, a mask of
missing values, two channels with every pair, L = 2 (the analytic wavelet and its conjugate),
``norm_batch_mean`` False (STL's default) and True, a channel without auto statistics whose cross
statistics become NaN after normalisation (OP-30), and the deep open configuration of
``empty_means_1d``, where some means have no valid sample (OP-3). The fields are built as in
``scattering_1d``, with other seeds.

Runs in the parity container, with the interpreter of the `2D+1D` environment, through
`make parity-fixtures` (tools/stl_parity/make_fixtures.py). It imports STL, never `estel`.
"""

from __future__ import annotations

import copy
import importlib
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

STL_ENV = "2d1d"
NDIM = 1
PARAMETERS: dict[str, Any] = {
    "L": 1,
    "threshold": 0.5,
    "masks": {"gap": {"seed": 2, "scatter_fraction": 0.05, "gap": [40, 47]}},
    "configs": {
        "periodic": {
            "N": 128, "J": 4, "Nb": 2, "Nc": 2, "pbc": True, "seed": 50, "ref_seed": 60,
            "norm_batch_mean": False, "harmonics_angle": 1, "harmonics_scale": 3, "dj": 3,
        },
        "conjugate_batch_mean": {
            "N": 128, "J": 4, "Nb": 2, "Nc": 2, "pbc": False, "seed": 51, "ref_seed": 61,
            "L": 2, "norm_batch_mean": True, "harmonics_angle": 1, "harmonics_scale": 4, "dj": 4,
        },
        "masked_open": {
            "N": 128, "J": 3, "Nb": 2, "Nc": 2, "pbc": False, "seed": 52, "ref_seed": 62,
            "mask": "gap", "norm_batch_mean": False, "harmonics_angle": 1, "harmonics_scale": 2,
            "dj": 3,
        },
        "cross_matrix": {
            "N": 128, "J": 4, "Nb": 1, "Nc": 3, "pbc": True, "seed": 53, "ref_seed": 63,
            "cross_matrix": [[True, False, True], [False, False, True], [False, False, True]],
            "L": 2, "norm_batch_mean": False, "harmonics_angle": 2, "harmonics_scale": 2,
            "dj": 2,
        },
        "deep_open": {
            "N": 128, "J": 4, "Nb": 1, "Nc": 1, "pbc": False, "seed": 54, "ref_seed": 64,
            "mask": "gap", "norm_batch_mean": False, "harmonics_angle": 1, "harmonics_scale": 4,
            "dj": 3,
        },
    },
}  # fmt: skip
FAMILIES = ("mean", "var", "S1", "S2", "S3", "S4")


def input_mask(n: int, spec: dict[str, Any]) -> NDArray[np.bool_]:
    """The missing samples (True = missing): a seeded scatter plus a gap, as in masks_1d."""
    rng = np.random.default_rng(spec["seed"])
    scatter = rng.random(n) < spec["scatter_fraction"]
    g0, g1 = spec["gap"]
    missing: NDArray[np.bool_] = scatter | ((np.arange(n) >= g0) & (np.arange(n) < g1))
    return missing


def input_field(config: dict[str, Any], seed: int) -> NDArray[np.float64]:
    """Seeded Gaussian signals, ``(Nb, Nc, N)``, each channel correlated with the previous one
    so that the cross statistics are not small."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((config["Nb"], config["Nc"]) + (config["N"],) * NDIM)
    channels = [noise[:, 0]]
    for c in range(1, config["Nc"]):
        channels.append(0.6 * channels[-1] + 0.8 * noise[:, c])
    field: NDArray[np.float64] = np.stack(channels, axis=1) + 0.3
    return field


def _numpy(x: Any) -> NDArray[Any]:
    array: NDArray[Any] = x.detach().cpu().numpy()
    return array


def run() -> dict[str, NDArray[Any]]:
    """Run STL's normalisations and reductions on each configuration and return the results."""
    data_cls = importlib.import_module("STL_main.STL_1D_Kernel_Torch").STL_1D_Kernel_Torch
    out: dict[str, NDArray[Any]] = {}
    for name, config in PARAMETERS["configs"].items():
        fields = {"field": input_field(config, config["seed"])}
        fields["ref_field"] = input_field(config, config["ref_seed"])
        kwargs: dict[str, Any] = {}
        if "mask" in config:
            missing = input_mask(config["N"], PARAMETERS["masks"][config["mask"]])
            fields = {key: np.where(missing, np.nan, f) for key, f in fields.items()}
            kwargs["mask_full_res"] = data_cls(array=torch.from_numpy(missing))
            out[f"{name}_mask"] = missing.astype(np.float64)
        out.update({f"{name}_{key}": f for key, f in fields.items()})
        target = data_cls(array=torch.from_numpy(fields["field"].copy()), pbc=config["pbc"])
        other = data_cls(array=torch.from_numpy(fields["ref_field"].copy()), pbc=config["pbc"])
        operator = target.get_ST_op(
            J=config["J"],
            L=config.get("L", PARAMETERS["L"]),
            norm="store_ref",
            compute_PS=False,
            downsample_nan_weight_threshold=PARAMETERS["threshold"],
            **kwargs,
        )
        matrix = config.get("cross_matrix")
        apply_kwargs: dict[str, Any] = {
            "norm_batch_mean": config["norm_batch_mean"],
            "compute_cross_matrix": None if matrix is None else torch.tensor(matrix),
        }
        stored = operator.apply(target, norm="store_ref", **apply_kwargs)
        out[f"{name}_var_ref"] = _numpy(operator.var_ref)
        out[f"{name}_s2_ref"] = _numpy(operator.S2_ref_sqrt_chan_diag)
        loaded = operator.apply(other, norm="load_ref", **apply_kwargs)
        for family in FAMILIES:
            out[f"{name}_store_{family}"] = _numpy(getattr(stored, family))
            out[f"{name}_load_{family}"] = _numpy(getattr(loaded, family))

        h_angle, h_scale, dj = config["harmonics_angle"], config["harmonics_scale"], config["dj"]
        iso = copy.deepcopy(stored).to_iso()
        angular = copy.deepcopy(stored).to_angular_ft(h_angle)
        iso_angular = copy.deepcopy(stored).to_iso().to_angular_ft(h_angle)
        iso_scale = copy.deepcopy(stored).to_iso().to_scale_ft(h_scale, dj)
        full = copy.deepcopy(stored).to_iso().to_angular_ft(h_angle)
        full = full.to_scale_ft(h_scale, dj, h_angle)
        for family in ("S1", "S2", "S3", "S4"):
            out[f"{name}_iso_{family}"] = _numpy(getattr(iso, family))
        for key, reduced in (
            ("angular", angular),
            ("iso_angular", iso_angular),
            ("iso_scale", iso_scale),
            ("iso_angular_scale", full),
        ):
            for family in ("S3", "S4"):
                out[f"{name}_{key}_{family}"] = _numpy(getattr(reduced, family))
        out[f"{name}_vector"] = _numpy(copy.deepcopy(stored).to_flatten(flatten_complex=True))
        out[f"{name}_vector_reduced"] = _numpy(full.to_flatten(flatten_complex=True))
    return out
