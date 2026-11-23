"""Parity case: the scattering statistics of STL `master` with the power spectrum, on 2D data
(Part II, §14.2, §14.4; task T4.3, issue #150; maths.md M-PS-2D, M-ORDER, M-NORM, M-FLAT,
OP-30, OP-37).

For each configuration ``<config>``, runs STL's scattering operator (``ST_Operator.apply``) with
the spectrum (``compute_PS=True``, STL's default), as ``statistics_2d`` does without it:

* with ``norm="store_ref"`` on a target field, and the configuration's ``norm_batch_mean``; the
  operator then holds the reference, ``PS_ref_sqrt_chan_diag`` included;
* with ``norm="load_ref"`` on a second field with the same mask, normalised by the target's
  reference;
* then ``to_flatten`` of the ``store_ref`` statistics, in its complex form
  (``flatten_complex=False``, STL's default and the form of the synthesis loss), in its real form
  (``flatten_complex=True``), and in its complex form averaged over the batch
  (``mean_along_batch=True``, as STL's syntheses with ``mean_field=True``); and the complex form of
  the ``load_ref`` statistics.

The operator is built as a user builds it: ``get_ST_op`` with the configuration's J where it
gives one, and, except in ``spectrum_options``, no option of the spectrum, so that STL chooses the
estimator from ``pbc`` and the NaN of the example field, with its default bins.
``spectrum_options`` passes the three options of the spectrum that ``ST_Operator`` forwards and
that ESTEL keeps (#148): ``Jmin=1``, ``n_bins=5`` and ``spectrum_method="pbc_mask"`` on periodic
data, an estimator forced against the automatic choice (``fft``). Given J, STL passes it to the
spectrum as J_PS (M-PS-2D, "Range"); without it, the spectrum takes its own default, equal to the
scattering one.

It stores, for each configuration:

* ``<config>_field`` and ``<config>_ref_field``: the target and the second field,
  ``(Nb, Nc, N, N)``, NaN at the missing pixels; ``<config>_mask`` (absent without a mask);
* ``<config>_ps_method``: the estimator of the spectrum, as its index in ``METHODS``, ``(1,)``;
  ``<config>_ps_bins``: [J, J_PS, J_min, n_b], ``(4,)``;
* ``<config>_raw_PS``: the spectrum without normalisation, ``(Nb, Nc, Nc, n_b)``, complex128,
  computed by the operator's own spectrum operator with the same channel pairs;
* ``<config>_store_<family>`` and ``<config>_load_<family>`` for ``mean``, ``var``, ``S1`` to
  ``S4`` and ``PS``, normalised, in STL's full-grid shapes (``PS``: ``(Nb, Nc, Nc, n_b)``,
  complex128), so that the normalised statistics and the flat vectors can be rebuilt from this
  fixture alone;
* ``<config>_var_ref``, ``<config>_s2_ref`` and ``<config>_ps_ref``: the reference the operator
  stored (``var_ref``, ``S2_ref_sqrt_chan_diag``, ``PS_ref_sqrt_chan_diag``; the last
  ``(Nb or 1, Nc, n_b)``);
* ``<config>_vector`` (complex), ``<config>_vector_real`` (real),
  ``<config>_vector_batch_mean`` (complex) and ``<config>_load_vector`` (complex): the flat
  vectors, PS last (M-FLAT). The spectrum changes no other family (M-ORDER), but the fields are
  not those of ``statistics_2d``: S1 to S4 here are this case's own.

The configurations cover the cells of §14.4 that T4.3 owns for the statistics: one and two
channels, ``norm_batch_mean`` False (STL's default) and True, the default J and an explicit J
smaller than the default (J = 2 on 32 x 32, where the default is 3: J_PS = 2 and 4 bins instead of
6), periodic and open data, with and without a mask of missing values (so each of the four
estimators), the options of the spectrum forwarded by the operator, and a channel without auto
statistics, whose spectrum reference is NaN, so that its cross spectrum becomes NaN after
normalisation (OP-30). The fields are built as in ``statistics_2d``, with other seeds.

Runs in the parity container, with the interpreter of the `master` environment, through
`make parity-fixtures` (tools/stl_parity/make_fixtures.py). It imports STL, never `estel`.
"""

from __future__ import annotations

import contextlib
import copy
import importlib
import io
import warnings
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

STL_ENV = "master"
NDIM = 2
METHODS = ("fft", "pbc_mask", "nan_mask", "pbc_nan_mask")
PARAMETERS: dict[str, Any] = {
    "L": 4,
    "threshold": 0.33,
    "methods": list(METHODS),
    "masks": {
        "scatter_hole": {"seed": 1, "scatter_fraction": 0.03, "hole": [[10, 17], [4, 13]]},
    },
    "configs": {
        "periodic": {
            "N": 32, "J": None, "Nb": 2, "Nc": 2, "pbc": True, "seed": 80, "ref_seed": 90,
            "norm_batch_mean": False, "auto": "fft",
        },
        "periodic_J2": {
            "N": 32, "J": 2, "Nb": 2, "Nc": 1, "pbc": True, "seed": 81, "ref_seed": 91,
            "norm_batch_mean": True, "auto": "fft",
        },
        "open": {
            "N": 32, "J": None, "Nb": 1, "Nc": 1, "pbc": False, "seed": 82, "ref_seed": 92,
            "norm_batch_mean": False, "auto": "pbc_mask",
        },
        "open_masked": {
            "N": 32, "J": None, "Nb": 2, "Nc": 2, "pbc": False, "seed": 83, "ref_seed": 93,
            "mask": "scatter_hole", "norm_batch_mean": True, "auto": "pbc_nan_mask",
        },
        "periodic_masked": {
            "N": 32, "J": None, "Nb": 1, "Nc": 2, "pbc": True, "seed": 84, "ref_seed": 94,
            "mask": "scatter_hole", "norm_batch_mean": False, "auto": "nan_mask",
        },
        "cross_matrix": {
            "N": 32, "J": None, "Nb": 1, "Nc": 2, "pbc": True, "seed": 85, "ref_seed": 95,
            "cross_matrix": [[False, True], [False, True]], "norm_batch_mean": False,
            "auto": "fft",
        },
        "spectrum_options": {
            "N": 32, "J": None, "Nb": 2, "Nc": 2, "pbc": True, "seed": 86, "ref_seed": 96,
            "norm_batch_mean": False, "Jmin": 1, "n_bins": 5, "spectrum_method": "pbc_mask",
            "auto": "fft",
        },
    },
}  # fmt: skip
FAMILIES = ("mean", "var", "S1", "S2", "S3", "S4", "PS")


def input_mask(n: int, spec: dict[str, Any]) -> NDArray[np.bool_]:
    """The missing pixels (True = missing): a seeded scatter plus a rectangular hole."""
    rng = np.random.default_rng(spec["seed"])
    scatter = rng.random((n, n)) < spec["scatter_fraction"]
    (r0, r1), (c0, c1) = spec["hole"]
    rows = (np.arange(n) >= r0) & (np.arange(n) < r1)
    cols = (np.arange(n) >= c0) & (np.arange(n) < c1)
    missing: NDArray[np.bool_] = scatter | (rows[:, None] & cols[None, :])
    return missing


def input_field(config: dict[str, Any], seed: int) -> NDArray[np.float64]:
    """Seeded Gaussian fields, ``(Nb, Nc, N, N)``, each channel correlated with the previous one
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
    """Run STL's statistics with the spectrum on each configuration and return the results."""
    data_cls = importlib.import_module("STL_main.STL_2D_Kernel_Torch").STL_2D_Kernel_Torch
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
        options = {k: config[k] for k in ("Jmin", "n_bins", "spectrum_method") if k in config}
        operator = target.get_ST_op(
            J=config["J"],
            L=PARAMETERS["L"],
            norm="store_ref",
            compute_PS=True,
            downsample_nan_weight_threshold=PARAMETERS["threshold"],
            **options,
            **kwargs,
        )
        method = config.get("spectrum_method", config["auto"])
        assert operator.CS_op.spectrum_method == method, (
            f"{name}: STL's spectrum uses {operator.CS_op.spectrum_method}, not {method}"
        )
        auto = target.get_CS_op().spectrum_method  # what STL would choose without the option
        assert auto == config["auto"], f"{name}: STL chose {auto}, not {config['auto']}"
        assert config["J"] in (None, operator.CS_op.J)
        assert (operator.CS_op.Jmin, operator.n_bins) == (
            config.get("Jmin", 0),
            config.get("n_bins", operator.n_bins),
        )
        out[f"{name}_ps_method"] = np.asarray([METHODS.index(method)], dtype=np.float64)
        out[f"{name}_ps_bins"] = np.asarray(
            [operator.J, operator.CS_op.J, operator.CS_op.Jmin, operator.n_bins],
            dtype=np.float64,
        )
        matrix = config.get("cross_matrix")
        pairs = None if matrix is None else torch.tensor(matrix)
        apply_kwargs: dict[str, Any] = {
            "norm_batch_mean": config["norm_batch_mean"],
            "compute_cross_matrix": pairs,
        }
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
            if name == "cross_matrix":  # STL warns on purpose here: no auto statistics (OP-30)
                warnings.simplefilter("ignore", UserWarning)
            stored = operator.apply(target, norm="store_ref", **apply_kwargs)
            loaded = operator.apply(other, norm="load_ref", **apply_kwargs)
        out[f"{name}_var_ref"] = _numpy(operator.var_ref)
        out[f"{name}_s2_ref"] = _numpy(operator.S2_ref_sqrt_chan_diag)
        out[f"{name}_ps_ref"] = _numpy(operator.PS_ref_sqrt_chan_diag)
        all_pairs = torch.triu(torch.ones((config["Nc"], config["Nc"]), dtype=torch.bool))
        raw_target = data_cls(array=torch.from_numpy(fields["field"].copy()), pbc=config["pbc"])
        out[f"{name}_raw_PS"] = _numpy(
            operator.CS_op.apply(
                raw_target,
                compute_cross_spectrum_matrix=all_pairs if pairs is None else pairs,
            )
        )
        for family in FAMILIES:
            out[f"{name}_store_{family}"] = _numpy(getattr(stored, family))
            out[f"{name}_load_{family}"] = _numpy(getattr(loaded, family))

        out[f"{name}_vector"] = _numpy(copy.deepcopy(stored).to_flatten())
        out[f"{name}_vector_real"] = _numpy(copy.deepcopy(stored).to_flatten(flatten_complex=True))
        out[f"{name}_vector_batch_mean"] = _numpy(
            copy.deepcopy(stored).to_flatten(mean_along_batch=True)
        )
        out[f"{name}_load_vector"] = _numpy(copy.deepcopy(loaded).to_flatten())
    return out
