"""Parity case: STL `master`'s 2D power spectrum operator (Part II, §14.2, §14.4; task T4.3,
issue #150; maths.md M-PS-2D, OP-6, OP-7, OP-19, OP-20, OP-21).

For each configuration ``<config>``, builds STL's spectrum operator
(``STL_2D_Kernel_Torch.get_CS_op``) on the configuration's field with STL's defaults (log-Gaussian
bins, ``n_bins=None``, ``Jmin=0``, the hard border mask, ``flexible_crop``), but ``J``, ``Jmin``
and ``n_bins`` where the configuration gives them (the options kept on #148), then:

* lets STL choose the estimator from ``pbc`` and the NaN of the field, checks that it chose the
  configuration's ``auto`` estimator, and applies it with the default channel pairs
  (``compute_cross_spectrum_matrix=None``: the auto spectra only);
* builds one operator for each estimator that accepts the field (all four without NaN,
  ``nan_mask`` and ``pbc_nan_mask`` with NaN), ``get_CS_op(spectrum_method=<estimator>)`` with
  the same options, and applies it with every channel pair c₁ ≤ c₂, auto and cross spectra
  (``compute_cross_spectrum_matrix`` the upper triangle).

It also checks STL's automatic choice with ``pbc=None``, which counts as periodic (``fft``
without NaN, ``nan_mask`` with NaN; maths.md M-PS-2D, OP-4), and stores nothing for it: ESTEL
requires an explicit boolean.

It stores, for each configuration:

* ``<config>_field``: the field, ``(Nb, Nc, N1, N2)``, NaN at the missing pixels;
* ``<config>_auto_method``: the estimator STL chose, as its index in ``METHODS``, ``(1,)``;
* ``<config>_auto``: the spectrum of that estimator with the default pairs, ``(Nb, Nc, Nc, n_b)``,
  complex128 as STL returns it (real values, NaN where no pair is computed);
* ``<config>_method_<estimator>``: the spectrum of each estimator that accepts the field, every
  pair c₁ ≤ c₂ (NaN for c₁ > c₂), ``(Nb, Nc, Nc, n_b)``, complex128;
* the bins and what the estimators read from them, as STL holds them on the operator:
  ``<config>_bin_scalars`` = [J_PS, J_min, n_b, k_min, k_max, N_max]; ``<config>_radial_k``,
  ``(N_max + 1,)``; ``<config>_n_modes_radial`` (N_k), ``(N_max + 1,)``;
  ``<config>_radial_bin_of_k``, the radial index of each rFFT mode, -1 beyond N_max,
  ``(N1, N2 // 2 + 1)``; ``<config>_weight_of_k``, its weight (2, 1 on the zero and Nyquist
  columns, 0 beyond N_max), ``(N1, N2 // 2 + 1)``; ``<config>_bin_centers`` (c_b), ``(n_b,)``;
  ``<config>_bin_windows`` (W_b(k)), ``(n_b, N_max + 1)``; ``<config>_bin_norm`` (Z_b),
  ``(n_b,)``; ``<config>_crop_borders``, the border width of each bin in pixels, ``(n_b,)``.

The configurations cover the cells of §14.4 that T4.3 owns for the operator: each estimator,
the automatic choice in each of its four cases, auto and cross spectra of two channels, two
rectangular images (OP-6), the long side on either axis, and the default bins with the default J
(6 bins, ``round(3 log₂(k_max / k_min))``) and with an explicit J = 2 (4 bins, the floor of
``max(4, ·)``), and ``Jmin = 1`` with an explicit ``n_bins = 5`` (``open_Jmin1``, on open
data, so that the border widths of the non-periodic estimators follow the bins of J_min; k_max
is then 0.5/√2/2). Every size is even (OP-20) and divisible by 2^(J-1). ``nan_per_map`` has missing
pixels that differ between batch elements and channels, so that a pixel is observed only if it is
valid in every map, and each map is filled with its own mean (OP-7).

The fields are power-law Gaussian fields (spectrum |k|^(-spectral_index)) cropped from a field
twice as large, so not periodic, plus an offset: a flat spectrum would not tell the bins apart.
Each channel is correlated with the previous one so that the cross spectra are not small.

Runs in the parity container, with the interpreter of the `master` environment, through
`make parity-fixtures` (tools/stl_parity/make_fixtures.py). It imports STL, never `estel`.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

STL_ENV = "master"
METHODS = ("fft", "pbc_mask", "nan_mask", "pbc_nan_mask")
PARAMETERS: dict[str, Any] = {
    "methods": list(METHODS),
    "spectral_index": 2.0,
    "offset": 0.3,
    "masks": {
        "scatter_hole": {"seed": 1, "scatter_fraction": 0.03, "hole": [[10, 17], [4, 13]]},
        "per_map": {"seed": 2, "scatter_fraction": 0.05, "hole": [[20, 26], [18, 22]]},
    },
    "configs": {
        "periodic": {"shape": [32, 32], "Nb": 2, "Nc": 2, "pbc": True, "seed": 70, "auto": "fft"},
        "open": {"shape": [32, 32], "Nb": 2, "Nc": 2, "pbc": False, "seed": 71, "auto": "pbc_mask"},
        "periodic_nan": {
            "shape": [32, 32], "Nb": 2, "Nc": 2, "pbc": True, "seed": 72, "mask": "scatter_hole",
            "auto": "nan_mask",
        },
        "open_nan": {
            "shape": [32, 32], "Nb": 2, "Nc": 2, "pbc": False, "seed": 73, "mask": "scatter_hole",
            "auto": "pbc_nan_mask",
        },
        "nan_per_map": {
            "shape": [32, 32], "Nb": 2, "Nc": 2, "pbc": True, "seed": 74, "mask": "per_map",
            "mask_per_map": True, "auto": "nan_mask",
        },
        "rect_wide": {"shape": [32, 48], "Nb": 1, "Nc": 2, "pbc": True, "seed": 75, "auto": "fft"},
        "rect_tall": {
            "shape": [48, 32], "Nb": 1, "Nc": 2, "pbc": False, "seed": 76, "auto": "pbc_mask",
        },
        "periodic_J2": {
            "shape": [32, 32], "Nb": 1, "Nc": 2, "pbc": True, "seed": 77, "J": 2, "auto": "fft",
        },
        "open_Jmin1": {
            "shape": [32, 32], "Nb": 2, "Nc": 2, "pbc": False, "seed": 78, "Jmin": 1,
            "n_bins": 5, "auto": "pbc_mask",
        },
    },
}  # fmt: skip


def input_mask(shape: tuple[int, ...], spec: dict[str, Any], seed: int) -> NDArray[np.bool_]:
    """The missing pixels (True = missing): a seeded scatter plus a rectangular hole."""
    rng = np.random.default_rng(seed)
    scatter = rng.random(shape) < spec["scatter_fraction"]
    (r0, r1), (c0, c1) = spec["hole"]
    rows = (np.arange(shape[0]) >= r0) & (np.arange(shape[0]) < r1)
    cols = (np.arange(shape[1]) >= c0) & (np.arange(shape[1]) < c1)
    missing: NDArray[np.bool_] = scatter | (rows[:, None] & cols[None, :])
    return missing


def power_law_noise(
    rng: np.random.Generator, count: int, shape: tuple[int, int], index: float
) -> NDArray[np.float64]:
    """``count`` unit Gaussian fields of spectrum |k|^(-index), ``(count, *shape)``, each cropped
    from a periodic field twice as large, so not periodic."""
    big = (2 * shape[0], 2 * shape[1])
    radius = np.sqrt(np.fft.fftfreq(big[0])[:, None] ** 2 + np.fft.fftfreq(big[1])[None, :] ** 2)
    amplitude = np.where(radius > 0, np.maximum(radius, 1 / max(big)) ** (-index / 2), 0.0)
    white = rng.standard_normal((count, *big))
    fields = np.fft.ifft2(np.fft.fft2(white) * amplitude).real[:, : shape[0], : shape[1]]
    out: NDArray[np.float64] = fields / fields.std(axis=(-2, -1), keepdims=True)
    return out


def input_field(config: dict[str, Any]) -> NDArray[np.float64]:
    """Seeded fields ``(Nb, Nc, N1, N2)``, each channel correlated with the previous one."""
    rng = np.random.default_rng(config["seed"])
    nb, nc, shape = config["Nb"], config["Nc"], tuple(config["shape"])
    noise = power_law_noise(rng, nb * nc, shape, PARAMETERS["spectral_index"])
    noise = noise.reshape(nb, nc, *shape)
    channels = [noise[:, 0]]
    for c in range(1, nc):
        channels.append(0.6 * channels[-1] + 0.8 * noise[:, c])
    field: NDArray[np.float64] = np.stack(channels, axis=1) + PARAMETERS["offset"]
    if "mask" in config:
        spec = PARAMETERS["masks"][config["mask"]]
        if config.get("mask_per_map"):
            missing = np.stack(
                [
                    np.stack(
                        [input_mask(shape, spec, spec["seed"] + nc * b + c) for c in range(nc)]
                    )
                    for b in range(nb)
                ]
            )
        else:
            missing = input_mask(shape, spec, spec["seed"])
        field = np.where(missing, np.nan, field)
    return field


def _numpy(x: Any) -> NDArray[Any]:
    array: NDArray[Any] = x.detach().cpu().numpy()
    return array


def _float(x: Any) -> NDArray[np.float64]:
    return np.asarray(_numpy(x), dtype=np.float64)


def check_unset_pbc(data_cls: Any) -> None:
    """With ``pbc=None``, STL takes the data as periodic (M-PS-2D, OP-4)."""
    field = input_field(PARAMETERS["configs"]["periodic"])
    for array, expected in ((field, "fft"), (np.where(field > 1.5, np.nan, field), "nan_mask")):
        data = data_cls(array=torch.from_numpy(array.copy()), pbc=None)
        method = data.get_CS_op().spectrum_method
        assert method == expected, f"pbc=None: STL chose {method}, not {expected}"


def run() -> dict[str, NDArray[Any]]:
    """Run STL's spectrum operator on each configuration and return the spectra and the bins."""
    data_cls = importlib.import_module("STL_main.STL_2D_Kernel_Torch").STL_2D_Kernel_Torch
    check_unset_pbc(data_cls)
    out: dict[str, NDArray[Any]] = {}
    for name, config in PARAMETERS["configs"].items():
        field = input_field(config)
        out[f"{name}_field"] = field
        data = data_cls(array=torch.from_numpy(field.copy()), pbc=config["pbc"])
        options = {key: config[key] for key in ("J", "Jmin", "n_bins") if key in config}
        operator = data.get_CS_op(**options)
        assert operator.spectrum_method == config["auto"], (
            f"{name}: STL chose {operator.spectrum_method}, not {config['auto']}"
        )
        out[f"{name}_auto_method"] = np.asarray([METHODS.index(config["auto"])], dtype=np.float64)
        out[f"{name}_auto"] = _numpy(operator.apply(data))

        pairs = torch.triu(torch.ones((config["Nc"], config["Nc"]), dtype=torch.bool))
        has_nan = bool(np.isnan(field).any())
        for method in METHODS:
            if has_nan and method in ("fft", "pbc_mask"):
                continue
            # One operator per estimator, chosen at construction, as ESTEL chooses it.
            forced = data.get_CS_op(spectrum_method=method, **options)
            assert forced.spectrum_method == method
            spectrum = forced.apply(data, compute_cross_spectrum_matrix=pairs)
            out[f"{name}_method_{method}"] = _numpy(spectrum)

        out[f"{name}_bin_scalars"] = np.asarray(
            [
                operator.J,
                operator.Jmin,
                operator.n_bins,
                operator.k_min,
                operator.k_max,
                operator.N_max,
            ],
            dtype=np.float64,
        )
        out[f"{name}_radial_k"] = _float(operator.radial_k)
        out[f"{name}_n_modes_radial"] = _float(operator.n_modes_radial)
        out[f"{name}_radial_bin_of_k"] = _float(operator.radial_bin_of_k)
        out[f"{name}_weight_of_k"] = _float(operator._weight_of_k)
        out[f"{name}_bin_centers"] = _float(operator.bin_centers)
        out[f"{name}_bin_windows"] = _float(operator.bin_windows)
        out[f"{name}_bin_norm"] = _float(operator.bin_norm)
        out[f"{name}_crop_borders"] = _float(operator.crop_borders)
    return out
