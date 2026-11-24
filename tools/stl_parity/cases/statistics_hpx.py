"""Parity case: the statistics of STL `HealpixUpdate` on HEALPix data, normalised and reduced,
with the masks of the scattering layers (Part II, §14.2, §14.4; task T5.3, issue #164, decision
2; maths.md section 1, M-DATA-HPX, M-MASK-HPX).

For each configuration ``<config>``, runs STL's scattering operator (``ST_Operator.apply``)
without spectrum (``compute_PS=False``), L = 4 gauges, with the settings of ``_healpix.py`` (the
sphere, no geometry cache, the kernel in float64; OP-45, OP-46) and with
``replace_nan_value=None``, as ``statistics_2d`` does:

* with ``norm="vanilla"`` on a target field (ESTEL's ``"none"``);
* with ``norm="store_ref"`` on the same field (ESTEL's ``"self"``), and the configuration's
  ``norm_batch_mean`` (ESTEL's ``ref_batch_mean``); the operator then holds the reference;
* with ``norm="load_ref"`` on a second field on the same grid (ESTEL's ``"ref"``), normalised by
  the target's reference;
* then, on copies of the ``store_ref`` statistics, the reductions of ``ST_Statistics``, in STL's
  order: ``to_iso``; ``to_angular_ft`` without ``iso``; ``to_iso`` then ``to_angular_ft``;
  ``to_iso`` then ``to_scale_ft``; ``to_iso``, ``to_angular_ft`` then ``to_scale_ft``, with the
  configuration's harmonics; and ``to_flatten(flatten_complex=True)`` of the normalised and of
  the fully reduced statistics.

**``replace_nan_value=None``** (decided by the maintainer on #163, the open point T5.2 records).
With STL's default, ``replace_nan_value=NaN``, the invalid pixels of a downsampled map are set to
NaN, and a valid pixel whose interpolation reads an invalid pixel with a weight of rounding size (a
zero weight of OP-92, below the 1e-12 of STL's support test) gets 0 x NaN = NaN in S1 and S2, and
through the normalisation NaN in S3, S4 and the flat vectors (``nan_16``: level 1, pixel 188 reads
pixel 167 with a weight of 8.6e-16 at gauge 2). ESTEL uses masks as weights and gives a finite
value. With ``None``, the setting STL's own syntheses pass (as for OP-60), the invalid pixels keep
the zero the masked decimation gives them, and the masks alone decide which pixels the means read.
The manifest records it under ``stl_settings``. The setting acts on the decimation only: at level 0,
STL convolves the field with its NaN, and the same effect remains there. In ``disc_nan_16``, the
valid cell 60 reads the missing cell 39 with a weight of 1.87e-15 at gauge 2, so S1 and S2 are NaN
at ``[0, 0, 0, 0, 2]``, and through ``s2_ref`` S3, S4 and the vectors where they normalise by it
(reported on #164; not decided).

It stores, for each configuration:

* ``<config>_field`` and ``<config>_ref_field``: the target and the second field,
  ``(Nb, Nc, Npix)``, NaN at the missing pixels; ``<config>_mask`` (with missing pixels only),
  1 at the missing pixels; ``<config>_cell_ids``, the NESTED pixels of the grid, ascending;
* ``<config>_vanilla_<family>``, ``<config>_store_<family>`` and ``<config>_load_<family>`` for
  ``mean``, ``var``, ``S1`` to ``S4``; ``<config>_var_ref`` and ``<config>_s2_ref``: the
  reference the operator stored (``var_ref``, ``S2_ref_sqrt_chan_diag``);
* ``<config>_<reduction>_<family>`` for the reductions ``iso`` (S1 to S4), ``angular`` (S3, S4),
  ``iso_angular`` (S3, S4), ``iso_scale`` (S3, S4) and ``iso_angular_scale`` (S3, S4);
* ``<config>_vector`` and ``<config>_vector_reduced``: the flat real vectors;
* ``<config>_cell_ids_<d>``, the pixels of level d = 1 … J-1, as the decimation of the grid
  gives them (M-DOWN-HPX: the NESTED parents of the available fine pixels, ascending);
* with missing pixels, the static maps of M-MASK-HPX as STL holds them, on the pixels
  ``<config>_cell_ids_<d>`` of their level, the boolean masks as float64 0/1 (1 = invalid):
  ``<config>_smooth_<d>`` (layer 0, d = 0 … J-1, d = 0 being the input mask) and
  ``<config>_reweight_smooth_<d>`` (d ≥ 1); ``<config>_layer1_<j>`` (first-layer data
  convolved at scale j, at level j); ``<config>_wav_<d>_<j>`` and
  ``<config>_reweight_wav_<d>_<j>`` (the same, decimated to level d > j);
  ``<config>_layer2_<j3>_<j2>`` (convolved at j2, decimated to level j3, convolved again at
  j3, j2 ≤ j3). A reweighting map is 1/(1 - f), and 0 on the invalid pixels.

The configurations cover the cells of §14.4 for HEALPix: the full sky at ``nside`` 8 (J = 2)
and 16 (J = 3); two channels with every pair (cross terms); a disc at ``nside`` 16, a partial
sky given by ``cell_ids``; a mask of missing values on the full sky at ``nside`` 16 (J = 3),
given as ``mask_full_res``, with a batch of two and ``norm_batch_mean=True``; and a disc with a
hole of missing values, its mask inferred by STL from the NaN of a single map (M-DATA-HPX); the
hole is 2 pixels, so its masks of the field and of the first layer at levels 1 and 2
(``smooth_1``, ``smooth_2``, ``layer1_1``, ``layer1_2``) are empty, and ``layer2_2_0`` is
invalid everywhere (the support at level 2 covers its 30 pixels). J is
STL's default, max(1, ⌊log₂ ``nside``⌋ - 1), passed explicitly. The fields are seeded Gaussian
white noise, each channel correlated with the previous one, as in ``statistics_2d``. The pixel
indices are stored as float64, exact below 2⁵³.

Runs in the parity container, with the interpreter of the `healpix` environment, through
`make parity-fixtures` (tools/stl_parity/make_fixtures.py). It imports STL, never `estel`.
"""

from __future__ import annotations

import copy
import importlib
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

STL_ENV = "healpix"


# Repeated in each HEALPix case: make_fixtures.py imports a case by its path, so a case cannot
# import a sibling module, and this loader must exist before _healpix.py can be read.
def _helper() -> ModuleType:
    """The settings of the HEALPix cases, ``_healpix.py``, imported by its path."""
    path = Path(__file__).resolve().with_name("_healpix.py")
    spec = importlib.util.spec_from_file_location("stl_parity_cases_healpix", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HEALPIX = _helper()
PARAMETERS: dict[str, Any] = {
    "L": 4,
    "threshold": 0.33,
    "domains": {
        "disc": {"colatitude_deg": 80.0, "longitude_deg": 30.0, "radius_deg": 26.0},
        "large_disc": {"colatitude_deg": 80.0, "longitude_deg": 30.0, "radius_deg": 40.0},
    },
    "masks": {
        # A disc of missing pixels plus a seeded scatter of isolated ones, on the full sky.
        "disc_scatter": {
            "colatitude_deg": 60.0, "longitude_deg": 200.0, "radius_deg": 20.0,
            "seed": 1, "scatter_fraction": 0.003,
        },
        # A hole of missing pixels inside the disc.
        "hole": {
            "colatitude_deg": 70.0, "longitude_deg": 40.0, "radius_deg": 3.0,
            "seed": 2, "scatter_fraction": 0.0,
        },
    },
    "configs": {
        "full_8": {
            "nside": 8, "J": 2, "Nb": 2, "Nc": 1, "seed": 30, "ref_seed": 40,
            "norm_batch_mean": False, "harmonics_angle": 2, "harmonics_scale": 2, "dj": 2,
        },
        "full_16_cross": {
            "nside": 16, "J": 3, "Nb": 1, "Nc": 2, "seed": 31, "ref_seed": 41,
            "norm_batch_mean": False, "harmonics_angle": 3, "harmonics_scale": 3, "dj": 3,
        },
        "disc_16": {
            "nside": 16, "J": 3, "Nb": 1, "Nc": 1, "seed": 32, "ref_seed": 42,
            "domain": "disc", "norm_batch_mean": False, "harmonics_angle": 4,
            "harmonics_scale": 2, "dj": 3,
        },
        "nan_16": {
            "nside": 16, "J": 3, "Nb": 2, "Nc": 1, "seed": 33, "ref_seed": 43,
            "mask": "disc_scatter", "norm_batch_mean": True, "harmonics_angle": 2,
            "harmonics_scale": 3, "dj": 2,
        },
        "disc_nan_16": {
            "nside": 16, "J": 3, "Nb": 1, "Nc": 1, "seed": 34, "ref_seed": 44,
            "domain": "large_disc", "mask": "hole", "mask_from_nan": True,
            "norm_batch_mean": False, "harmonics_angle": 3, "harmonics_scale": 3, "dj": 3,
        },
    },
    "stl_settings": HEALPIX.SETTINGS | {"replace_nan_value": None},
}  # fmt: skip
FAMILIES = ("mean", "var", "S1", "S2", "S3", "S4")


def grid(config: dict[str, Any]) -> NDArray[np.int64] | None:
    """The NESTED pixels of the configuration's grid, or None for the full sky."""
    if "domain" not in config:
        return None
    spec = PARAMETERS["domains"][config["domain"]]
    return HEALPIX.disc(  # type: ignore[no-any-return]
        config["nside"], spec["colatitude_deg"], spec["longitude_deg"], spec["radius_deg"]
    )


def input_mask(
    nside: int, cell_ids: NDArray[np.int64] | None, spec: dict[str, Any]
) -> NDArray[np.bool_]:
    """The missing pixels of the grid (True = missing): a disc plus a seeded scatter."""
    hole = HEALPIX.disc(nside, spec["colatitude_deg"], spec["longitude_deg"], spec["radius_deg"])
    ids = np.arange(12 * nside**2) if cell_ids is None else cell_ids
    rng = np.random.default_rng(spec["seed"])
    scatter = rng.random(ids.size) < spec["scatter_fraction"]
    missing: NDArray[np.bool_] = np.isin(ids, hole) | scatter
    return missing


def input_field(config: dict[str, Any], n_pix: int, seed: int) -> NDArray[np.float64]:
    """Seeded Gaussian fields, ``(Nb, Nc, Npix)``, each channel correlated with the previous one
    so that the cross statistics are not small."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((config["Nb"], config["Nc"], n_pix))
    channels = [noise[:, 0]]
    for c in range(1, config["Nc"]):
        channels.append(0.6 * channels[-1] + 0.8 * noise[:, c])
    field: NDArray[np.float64] = np.stack(channels, axis=1) + 0.3
    return field


def _numpy(x: Any) -> NDArray[Any]:
    array: NDArray[Any] = x.detach().cpu().numpy()
    if array.dtype not in (np.float64, np.complex128):
        array = array.astype(np.float64)
    return array


def _masks(name: str, wavelet_op: Any) -> dict[str, NDArray[Any]]:
    """The static maps of M-MASK-HPX that STL's wavelet operator holds (no public accessor)."""
    out: dict[str, NDArray[Any]] = {}
    for d, mask in wavelet_op._mask_smooth.items():
        out[f"{name}_smooth_{d}"] = _numpy(mask)
    for d, weights in wavelet_op._reweight_smooth.items():
        out[f"{name}_reweight_smooth_{d}"] = _numpy(weights)
    for j, mask in wavelet_op._layer1_mask.items():
        out[f"{name}_layer1_{j}"] = _numpy(mask)
    for (d, j), mask in wavelet_op._mask_wav.items():
        out[f"{name}_wav_{d}_{j}"] = _numpy(mask)
        out[f"{name}_reweight_wav_{d}_{j}"] = _numpy(wavelet_op._reweight_wav[(d, j)])
    for (j3, j2), mask in wavelet_op._layer2_mask.items():
        out[f"{name}_layer2_{j3}_{j2}"] = _numpy(mask)
    return out


def _level_cell_ids(name: str, wavelet_op: Any, cell_ids: Any) -> dict[str, NDArray[Any]]:
    """The pixels of each level d ≥ 1, as STL's decimation gives them."""
    out: dict[str, NDArray[Any]] = {}
    ids = cell_ids
    for d in range(1, wavelet_op.J):
        ids = torch.as_tensor(np.asarray(wavelet_op._get_down(d - 1, ids).cell_ids_out))
        out[f"{name}_cell_ids_{d}"] = _numpy(ids)
    if wavelet_op._cell_ids_at is not None:  # the levels of the masks are the same pixels
        for d in range(1, wavelet_op.J):
            if not np.array_equal(_numpy(wavelet_op._cell_ids_at[d]), out[f"{name}_cell_ids_{d}"]):
                raise ValueError(f"{name}: the masks of level {d} are not on its pixels")
    return out


def run() -> dict[str, NDArray[Any]]:
    """Run STL's statistics, normalisations and reductions on each configuration."""
    data_cls = importlib.import_module("STL_main.STL_Healpix_Kernel_Torch").STL_Healpix_Kernel_Torch
    out: dict[str, NDArray[Any]] = {}
    with HEALPIX.keep_kernel_float64():
        for name, config in PARAMETERS["configs"].items():
            out.update(_configuration(data_cls, name, config))
    return out


def _configuration(data_cls: Any, name: str, config: dict[str, Any]) -> dict[str, NDArray[Any]]:
    nside, cell_ids = config["nside"], grid(config)
    n_pix = 12 * nside**2 if cell_ids is None else cell_ids.size
    fields = {"field": input_field(config, n_pix, config["seed"])}
    fields["ref_field"] = input_field(config, n_pix, config["ref_seed"])
    geometry: dict[str, Any] = {} if cell_ids is None else {"nside": nside, "cell_ids": cell_ids}
    kwargs: dict[str, Any] = {}
    out: dict[str, NDArray[Any]] = {}
    example = None
    if "mask" in config:
        missing = input_mask(nside, cell_ids, PARAMETERS["masks"][config["mask"]])
        fields = {key: np.where(missing, np.nan, f) for key, f in fields.items()}
        out[f"{name}_mask"] = missing.astype(np.float64)
        if config.get("mask_from_nan"):
            # STL infers the mask from the NaN of a single map (M-DATA-HPX).
            example = data_cls(array=torch.from_numpy(fields["field"][0, 0].copy()), **geometry)
        else:
            kwargs["mask_full_res"] = data_cls(
                array=torch.from_numpy(missing), **({"nside": nside} | geometry)
            )
    out.update({f"{name}_{key}": f for key, f in fields.items()})
    target = data_cls(array=torch.from_numpy(fields["field"].copy()), **geometry)
    other = data_cls(array=torch.from_numpy(fields["ref_field"].copy()), **geometry)
    operator = (target if example is None else example).get_ST_op(
        J=config["J"],
        L=PARAMETERS["L"],
        norm="store_ref",
        compute_PS=False,
        replace_nan_value=PARAMETERS["stl_settings"]["replace_nan_value"],  # see the docstring
        downsample_nan_weight_threshold=PARAMETERS["threshold"],
        wavelet_op_kwargs=HEALPIX.WAVELET_OP_KWARGS,
        **kwargs,
    )
    wavelet_op = operator.wavelet_op
    if ("mask" in config) != (wavelet_op.mask_full_res is not None):
        raise ValueError(f"{name}: STL's operator has no mask where the configuration has one")
    out[f"{name}_cell_ids"] = _numpy(target.cell_ids)
    out.update(_level_cell_ids(name, wavelet_op, target.cell_ids))
    if "mask" in config:
        out.update(_masks(name, wavelet_op))

    apply_kwargs: dict[str, Any] = {"norm_batch_mean": config["norm_batch_mean"]}
    vanilla = operator.apply(target.copy(), norm="vanilla", **apply_kwargs)
    stored = operator.apply(target, norm="store_ref", **apply_kwargs)
    out[f"{name}_var_ref"] = _numpy(operator.var_ref)
    out[f"{name}_s2_ref"] = _numpy(operator.S2_ref_sqrt_chan_diag)
    loaded = operator.apply(other, norm="load_ref", **apply_kwargs)
    HEALPIX.check_kernel_float64(wavelet_op)
    for family in FAMILIES:
        out[f"{name}_vanilla_{family}"] = _numpy(getattr(vanilla, family))
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
