"""Prove that a change leaves every value of ESTEL as on ``main`` (phase P7; task T7.2, #193).

The proof runs a fixed set of cases on two trees, ``origin/main`` (checked out by the tool in a
temporary worktree, removed at the end) and the working tree, each case in a fresh subprocess,
on CPU with one thread (PyTorch and XLA thread counts set to 1, PyTorch's deterministic mode),
on both backends, in float32 and float64. It compares:

* every output, **bitwise**: dtype, shape, NaN positions and bits; values, and the gradient of
  a loss on ``to_vector`` with respect to the data;
* a canonical, recursive dump of every plan's ``_key`` and ``_content``: class objects and
  qualified names mapped through the table of renamed names (``tools/renamed_names.toml``), no
  ``hash()`` across processes;
* files: statistics saved by ``main`` loaded on the branch, equal bitwise, and
  ``from_statistics`` rebuilding the same plan;
* every function and method body of ``src/estel/``, by AST, modulo the table, whitespace and
  docstrings: the bodies that differ, and the definitions present on one side only, are
  **listed for review**; they are not a failure.

The cases use the public API. The internal names they need (the oracle, the plans' ``_key``
and ``_content``) are written with the names of ``main`` and resolved through the table of the
tree that runs them, so that the same script runs on both trees.

Exit status (the verdict):

* 0: every output bitwise equal, every dump equal;
* 1: some outputs differ, all within the parity tolerance of Part II, §14.3 (rtol 1e-10 in
  float64, 1e-5 in float32, atol = 0.1 x rtol x max|expected|): the task's pull request locates
  the cause and states it;
* 2: a difference beyond the tolerance, a dump that differs, an output on one side only, or a
  case that fails on one side only: the task stops, and the maintainer decides;
* 3: the proof could not run (a git or environment error), or a case raised on both trees
  and is not in ``EXPECTED_FAILURES``: that case proved nothing.

Usage::

    python tools/prove_unchanged.py                     # origin/main against the working tree
    python tools/prove_unchanged.py --base <ref>        # another base
    python tools/prove_unchanged.py --case 2d-periodic  # some cases only (repeat --case)
    make prove                                          # the same, through uv

It takes minutes; it is not part of the test suite. Its own tests (tests/tools/) check that it
sees a difference, on crafted dumps.
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import dataclasses
import enum
import hashlib
import importlib
import math
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
import types
import warnings
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
TABLE = Path("tools/renamed_names.toml")

BACKENDS = ("torch", "jax")
DTYPES = ("float32", "float64")
# Part II, §14.3: rtol by dtype; atol = 0.1 x rtol x max|expected|.
RTOL = {"float64": 1e-10, "float32": 1e-5}
EQUAL, WITHIN, BEYOND, ERROR = 0, 1, 2, 3
VERDICTS = {
    EQUAL: "every output bitwise equal",
    WITHIN: "differences within the parity tolerance (Part II, §14.3): locate and state the cause",
    BEYOND: "differences beyond the parity tolerance: stop, the maintainer decides",
    ERROR: "a case did not run on both trees: the proof is incomplete",
}

# One thread, on CPU, the same hash seed: the environment of every case.
ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
    "JAX_PLATFORMS": "cpu",
    "CUDA_VISIBLE_DEVICES": "",
    "PYTHONHASHSEED": "0",
}


# ================================================================== names through the table


@dataclass(frozen=True)
class Row:
    old: str
    new: str
    kind: str
    task: str
    done: bool
    note: str = ""


def read_table(tree: Path) -> list[Row]:
    """The rows of the table of ``tree`` (none if it has no table, as ``main`` before T7.2)."""
    path = tree / TABLE
    if not path.exists():
        return []
    return [Row(**row) for row in tomllib.loads(path.read_text()).get("row", [])]


def _split(qualified: str) -> tuple[str, str]:
    module, _, qualname = qualified.partition(":")
    return module, qualname


class Names:
    """The names of one tree and those of ``main``, through the rows done of its table.

    ``main`` names are *canonical*: what the cases are written with, and what the dumps hold.

    A ``split`` row of a definition (``module:Class``) records the classes its instances now
    have, beside the ``rename`` row of the name itself (T7.5: the instances of ``WaveletPlan``,
    renamed ``WaveletPlanSingleDomain`` and made abstract, are ``WaveletPlanCartesian``). One old
    name then has several new ones: :meth:`canonical` maps each of them back, the class itself
    only, so that a dump that names the split class equals ``main``'s; its methods keep their
    names, and are compared as bodies of their own.
    """

    def __init__(self, rows: list[Row]) -> None:
        self.modules: dict[str, list[str]] = {}  # old module -> new modules
        self.definitions: dict[str, str] = {}  # old module:qualname -> new module:qualname
        self.split_back: dict[str, str] = {}  # new module:Class of a split row -> old one
        self.reintroduced: set[str] = set()
        for row in rows:
            if not row.done or "#" in row.old or row.old.startswith("M-"):
                continue
            if row.kind == "reintroduce":
                self.reintroduced.add(row.old)
                continue
            if ":" in row.old:
                module, qualname = _split(row.old)
                news = [
                    new if ":" in new else f"{module}:{self._requalify(qualname, new)}"
                    for new in row.new.replace(",", " ").split()
                ]
                if row.kind == "split":
                    self.split_back.update(dict.fromkeys(news, row.old))
                else:
                    self.definitions[row.old] = news[0]
            elif "/" not in row.old:
                self.modules[row.old] = [m for m in row.new.replace(",", " ").split() if m]
        self.module_back = {new: old for old, news in self.modules.items() for new in news}
        self.definition_back = {new: old for old, new in self.definitions.items()}

    @staticmethod
    def _requalify(qualname: str, new: str) -> str:
        prefix = qualname.rsplit(".", 1)[0] + "." if "." in qualname else ""
        return new if new.startswith(prefix) and prefix else prefix + new

    def new_modules(self, module: str) -> list[str]:
        return self.modules.get(module, [module])

    def resolve(self, qualified: str) -> Any:
        """The object a name of ``main`` (``module:qualname``) names on this tree."""
        target = self.definitions.get(qualified, qualified)
        module, qualname = _split(target)
        candidates = [f"{m}:{qualname}" for m in self.new_modules(module)]
        for candidate in candidates:
            module_name, path = _split(candidate)
            try:
                obj: Any = importlib.import_module(module_name)
                for part in path.split("."):
                    obj = getattr(obj, part)
                return obj
            except (ImportError, AttributeError):
                continue
        raise LookupError(
            f"{qualified}: not found on this tree, also through the table ({candidates}); "
            "a task that renames it records it in tools/renamed_names.toml"
        )

    def canonical(self, module: str, qualname: str) -> str:
        """The name of ``main`` of a definition of this tree (``module:qualname``)."""
        exact = self.definition_back.get(f"{module}:{qualname}") or self.split_back.get(
            f"{module}:{qualname}"
        )
        if exact:
            return exact
        old_module = self.module_back.get(module, module)
        parts = qualname.split(".")
        for cut in range(len(parts), 0, -1):
            head = ".".join(parts[:cut])
            for mod in {module, old_module}:
                back = self.definition_back.get(f"{mod}:{head}")
                if back:
                    tail = ".".join(parts[cut:])
                    return back + ("." + tail if tail else "")
        name = f"{old_module}:{qualname}"
        if name in self.reintroduced:
            return name + " (reintroduced)"
        return name

    def bare_back(self) -> dict[str, str]:
        """New bare name -> old bare name, for the bodies compared by AST."""
        mapping: dict[str, str] = {}
        for old, new in self.definitions.items():
            mapping[new.rsplit(":", 1)[-1].rsplit(".", 1)[-1]] = old.rsplit(".", 1)[-1].split(":")[
                -1
            ]
        for old, news in self.modules.items():
            for new in news:
                mapping[new.rsplit(".", 1)[-1]] = old.rsplit(".", 1)[-1]
        return mapping


# ================================================================== the canonical dump


def _array_of(value: Any) -> Any:
    """A NumPy array of a backend array, or None."""
    import numpy as np

    module = type(value).__module__.split(".")[0]
    if module == "torch":
        return value.detach().cpu().resolve_conj().numpy()
    if module in ("jax", "jaxlib"):
        return np.asarray(value)
    return None


def canonical_dump(value: Any, names: Names, _seen: frozenset[int] = frozenset()) -> Any:
    """A structure of tuples, strings and numbers that two processes can compare with ``==``.

    Floats by their hex, arrays by dtype, shape and SHA-256 of their bytes, classes and functions
    by their name of ``main``; never ``hash()``.
    """
    import numpy as np

    if value is None or isinstance(value, bool | int | str | bytes):
        return value
    if isinstance(value, float):
        return ("float", value.hex())
    if isinstance(value, complex):
        return ("complex", value.real.hex(), value.imag.hex())
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest = hashlib.sha256(array.tobytes()).hexdigest()
        if array.dtype == object:
            digest = repr(canonical_dump(array.tolist(), names, _seen))
        return ("array", array.dtype.str, array.shape, digest)
    if isinstance(value, slice):
        return (
            "slice",
            tuple(canonical_dump(v, names, _seen) for v in (value.start, value.stop, value.step)),
        )
    if isinstance(value, range):
        return ("range", value.start, value.stop, value.step)
    if isinstance(value, np.dtype):
        return ("dtype", value.str, value.descr)  # the byte order is part of .str
    if value is Ellipsis:
        return ("ellipsis",)
    if isinstance(value, np.generic):
        return ("scalar", value.dtype.str, value.tobytes().hex())
    if isinstance(value, type):
        return ("class", names.canonical(value.__module__, value.__qualname__))
    if isinstance(value, types.FunctionType | types.BuiltinFunctionType | types.MethodType):
        module = getattr(value, "__module__", None) or "?"
        return ("function", names.canonical(module, value.__qualname__))
    if isinstance(value, enum.Enum):
        return ("enum", canonical_dump(type(value), names), value.name)
    converted = _array_of(value)
    if converted is not None:
        return canonical_dump(converted, names, _seen)
    if id(value) in _seen:
        return ("cycle",)
    seen = _seen | {id(value)}
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        cls = canonical_dump(type(value), names)
        return ("namedtuple", cls, tuple(canonical_dump(v, names, seen) for v in value))
    if isinstance(value, list | tuple):
        return (type(value).__name__, tuple(canonical_dump(v, names, seen) for v in value))
    if isinstance(value, dict):
        items = [
            (canonical_dump(k, names, seen), canonical_dump(v, names, seen))
            for k, v in value.items()
        ]
        return ("dict", tuple(sorted(items, key=repr)))
    if isinstance(value, set | frozenset):
        return ("set", tuple(sorted((canonical_dump(v, names, seen) for v in value), key=repr)))
    if isinstance(value, types.MappingProxyType):
        return canonical_dump(dict(value), names, seen)
    cls = canonical_dump(type(value), names)
    if dataclasses.is_dataclass(value):
        fields = tuple(
            (f.name, canonical_dump(getattr(value, f.name), names, seen))
            for f in dataclasses.fields(value)
        )
        return ("dataclass", cls, fields)
    attributes: dict[str, Any] = {}
    has_slots = False
    for klass in type(value).__mro__:
        slots = getattr(klass, "__slots__", ())
        has_slots = has_slots or bool(slots)
        for slot in (slots,) if isinstance(slots, str) else slots:
            if hasattr(value, slot):
                attributes[slot] = getattr(value, slot)
    if not has_slots and not hasattr(value, "__dict__"):
        raise TypeError(
            f"canonical_dump: no canonical form for {type(value).__module__}."
            f"{type(value).__qualname__} ({value!r:.80}): it has no fields to compare. Add its "
            "type to canonical_dump (tools/prove_unchanged.py), with what makes two values equal."
        )
    attributes.update(getattr(value, "__dict__", {}))
    return (
        "object",
        cls,
        tuple(sorted((k, canonical_dump(v, names, seen)) for k, v in attributes.items())),
    )


# ================================================================== the cases (in the worker)


@dataclass
class Context:
    names: Names
    backend: str
    dtype: str
    files: Path
    is_base: bool
    outputs: dict[str, Any] = field(default_factory=dict)

    def record(self, name: str, value: Any) -> None:
        import numpy as np

        if isinstance(value, dict) and all(isinstance(k, str) for k in value):
            for key, item in value.items():
                self.record(f"{name}/{key}", item)
            return
        array = _array_of(value)
        if array is None and isinstance(value, np.ndarray):
            array = value
        if array is not None:
            self.outputs[name] = np.array(array, copy=True)
        else:
            self.outputs[name] = ("dump", canonical_dump(value, self.names))


def _field(shape: tuple[int, ...], seed: int = 0) -> Any:
    import numpy as np

    values = np.random.default_rng(seed).standard_normal(shape) + 0.2
    return values


FAMILIES = ("mean", "var", "S1", "S2", "S3", "S4", "PS")


def _record_statistics(ctx: Context, name: str, stats: Any) -> None:
    ctx.record(f"{name}/to_vector", stats.to_vector())
    for family in FAMILIES:
        try:
            grid = stats.full_grid(family)
        except Exception as error:  # absent family: recorded by its type
            ctx.record(f"{name}/{family}", ("raises", type(error).__name__))
        else:
            ctx.record(f"{name}/{family}", grid)
    try:
        centers = stats.bin_centers
    except Exception as error:  # no power spectrum: recorded by its type
        ctx.record(f"{name}/bin_centers", ("raises", type(error).__name__))
    else:
        ctx.record(f"{name}/bin_centers", centers)


def _record_plan(ctx: Context, name: str, plan: Any) -> None:
    key = _attribute(ctx.names, plan, "_key")
    content = _attribute(ctx.names, plan, "_content")()
    ctx.record(f"{name}/plan._key", key)
    ctx.record(f"{name}/plan._content", content)


def _attribute(names: Names, obj: Any, attribute: str) -> Any:
    """``obj.<attribute>``, the attribute named as on ``main``, through the table."""
    for klass in type(obj).__mro__:
        old = names.canonical(klass.__module__, f"{klass.__qualname__}.{attribute}")
        new = names.definitions.get(old)
        if new:
            return getattr(obj, new.rsplit(".", 1)[-1])
    return getattr(obj, attribute)


def _loss(ctx: Context, stats: Any) -> Any:
    vector = stats.to_vector()
    if ctx.backend == "torch":
        import torch

        squares = vector.abs() ** 2 if torch.is_complex(vector) else vector**2
        return torch.nansum(squares)
    import jax.numpy as jnp

    return jnp.nansum(jnp.abs(vector) ** 2)


def _apply(ctx: Context, name: str, operator: Any, data: Any, ref: Any = None, **kw: Any) -> Any:
    """Apply ``operator``, record the statistics, then the gradient of a loss on to_vector."""
    stats = operator.apply(data, ref, **kw)
    _record_statistics(ctx, name, stats)
    if ctx.backend == "torch":
        array = data.array.detach().clone().requires_grad_(True)
        loss = _loss(ctx, operator.apply(data.with_array(array), ref, **kw))
        loss.backward()
        ctx.record(f"{name}/gradient", array.grad)
    else:
        import jax

        def of_array(array: Any) -> Any:
            return _loss(ctx, operator.apply(data.with_array(array), ref, **kw))

        ctx.record(f"{name}/gradient", jax.grad(of_array)(data.array))
    return stats


def _operator(*args: Any, **kwargs: Any) -> Any:
    import estel

    return estel.STOperator(*args, **kwargs)


def case_1d(ctx: Context) -> None:
    from estel import Data1D

    data = Data1D(_field((2, 1, 32)), pbc=True)
    operator = _operator(data, 3)
    _record_plan(ctx, "1d", operator.plan)
    _apply(ctx, "1d", operator, data)


def case_periodic_2d(ctx: Context) -> None:
    from estel import Data2D

    data = Data2D(_field((1, 1, 16, 16)), pbc=True)
    operator = _operator(data, 2, 4, compute_PS=False)
    _record_plan(ctx, "2d", operator.plan)
    _apply(ctx, "2d", operator, data)


def _mask_2d(size: int) -> Any:
    import numpy as np

    mask = np.zeros((size, size), dtype=bool)
    mask[5:8, 9:11] = True
    return mask


def case_mask_2d(ctx: Context) -> None:
    from estel import Data2D

    data = Data2D(_field((1, 1, 32, 32)), mask=_mask_2d(32), pbc=False)
    operator = _operator(data, 3, 4, compute_PS=False)
    _record_plan(ctx, "2d-mask", operator.plan)
    _apply(ctx, "2d-mask", operator, data)


def _sky(depth: int, n: int | None = None) -> Any:
    return _field((1, 1, n if n is not None else 12 * 4**depth))


def case_full_healpix(ctx: Context) -> None:
    from estel import DataHealpix

    data = DataHealpix(_sky(2))
    operator = _operator(data, 2)
    _record_plan(ctx, "healpix", operator.plan)
    _apply(ctx, "healpix", operator, data)


def case_partial_healpix(ctx: Context) -> None:
    import numpy as np

    from estel import DataHealpix

    data = DataHealpix(_sky(3, 80), depth=3, cell_ids=np.arange(80))
    operator = _operator(data, 2)
    _record_plan(ctx, "healpix-partial", operator.plan)
    _apply(ctx, "healpix-partial", operator, data)


def case_mask_healpix(ctx: Context) -> None:
    import numpy as np

    from estel import DataHealpix

    mask = np.zeros(192, dtype=bool)
    mask[[5, 6, 70]] = True
    data = DataHealpix(_sky(2), mask=mask)
    operator = _operator(data, 2)
    _record_plan(ctx, "healpix-mask", operator.plan)
    _apply(ctx, "healpix-mask", operator, data)


def _tensor_prod(ctx: Context, lowpass: str | None) -> None:
    from estel import Data1D, Data2D, DataTensorProd, STOperatorTensorProd

    data = DataTensorProd(_field((1, 1, 16, 16, 8)), [Data2D, Data1D], pbc=[True, True])
    operator = STOperatorTensorProd(data, [2, 2], lowpass=lowpass)
    name = f"tensor-prod-{lowpass}"
    _record_plan(ctx, name, operator.plan)
    _apply(ctx, name, operator, data)


def case_st_tensor_prod(ctx: Context) -> None:
    _tensor_prod(ctx, "ST")


def case_none_tensor_prod(ctx: Context) -> None:
    _tensor_prod(ctx, None)


def case_spectrum_auto(ctx: Context) -> None:
    from estel import Data2D

    data = Data2D(_field((1, 1, 32, 32)), pbc=True)
    operator = _operator(data, 3, compute_PS=True)
    _record_plan(ctx, "spectrum-auto", operator.plan)
    _apply(ctx, "spectrum-auto", operator, data)


def case_spectrum_masked(ctx: Context) -> None:
    from estel import Data2D

    data = Data2D(_field((1, 1, 32, 32)), mask=_mask_2d(32), pbc=False)
    operator = _operator(data, 3, compute_PS=True, spectrum_method="nan_mask")
    _record_plan(ctx, "spectrum-masked", operator.plan)
    _apply(ctx, "spectrum-masked", operator, data)


def case_cross(ctx: Context) -> None:
    from estel import Data2D

    data = Data2D(_field((1, 2, 16, 16)), pbc=True)
    operator = _operator(data, 2, compute_PS=True)
    _record_plan(ctx, "cross", operator.plan)
    _apply(ctx, "cross", operator, data)


def case_norms(ctx: Context) -> None:
    from estel import Data2D

    data = Data2D(_field((2, 1, 16, 16)), pbc=True)
    other = Data2D(_field((2, 1, 16, 16), seed=1), pbc=True)
    operator = _operator(data, 2)
    _record_plan(ctx, "norms", operator.plan)
    ref = _apply(ctx, "norm-self", operator, other, norm="self")
    _apply(ctx, "norm-ref", operator, data, ref, norm="ref")
    _apply(ctx, "norm-none", operator, data, norm="none")
    _apply(ctx, "norm-self-batch-mean", operator, data, norm="self", ref_batch_mean=True)


def case_reduce_select(ctx: Context) -> None:
    from estel import Data2D

    data = Data2D(_field((1, 2, 16, 16)), pbc=True)
    stats = _operator(data, 2, compute_PS=True).apply(data)
    reductions: tuple[tuple[str, Callable[[], Any]], ...] = (
        ("iso", lambda: stats.reduce(iso=True)),
        ("angular-fourier", lambda: stats.reduce(angular_fourier=True)),
        ("iso-scale-cosine", lambda: stats.reduce(iso=True, scale_cosine=True)),
        ("select-S2", lambda: stats.select("S2")),
        ("select-S1-S3", lambda: stats.select(["S1", "S3"])),
    )
    for name, reduced in reductions:
        _record_statistics(ctx, name, reduced())


def _synthesis(ctx: Context, name: str, target: Any, **settings: Any) -> None:
    import estel

    result = estel.synthesize_from_maps(target, max_iter=3, seed=3, **settings)
    ctx.record(
        f"{name}/field", result.field.array if hasattr(result.field, "array") else result.field
    )
    ctx.record(f"{name}/losses", list(result.losses))
    ctx.record(f"{name}/n_iter", result.n_iter)


def case_synthesis_2d(ctx: Context) -> None:
    import numpy as np

    target = np.random.default_rng(0).standard_normal((1, 1, 16, 16)) * 2.0 + 1.0
    _synthesis(ctx, "synthesis-2d", target, compute_PS=True, st_kwargs={"J": 2})


def case_synthesis_healpix(ctx: Context) -> None:
    import numpy as np

    from estel import DataHealpix

    sky = np.random.default_rng(0).standard_normal((1, 1, 192)) * 2.0 + 1.0
    _synthesis(ctx, "synthesis-healpix", DataHealpix(sky), st_kwargs={"J": 2})


def case_files(ctx: Context) -> None:
    """Statistics saved by the base tree, loaded on both trees: their outputs and the plan of
    ``from_statistics``."""
    import numpy as np

    import estel
    from estel import Data1D, Data2D, DataHealpix, DataTensorProd, STStatistics

    directory = ctx.files / f"{ctx.backend}-{ctx.dtype}"
    makers: dict[str, tuple[Any, Callable[[], Any]]] = {
        "2d-mask-spectrum": (
            estel.STOperator,
            lambda: _operator(
                d := Data2D(_field((1, 2, 32, 32)), mask=_mask_2d(32), pbc=False),
                3,
                compute_PS=True,
            ).apply(d),
        ),
        "healpix-partial": (
            estel.STOperator,
            lambda: _operator(
                d := DataHealpix(_sky(3, 80), depth=3, cell_ids=np.arange(80)), 2
            ).apply(d, norm="none"),
        ),
        "tensor-prod": (
            estel.STOperatorTensorProd,
            lambda: estel.STOperatorTensorProd(
                d := DataTensorProd(_field((1, 1, 16, 16, 8)), [Data2D, Data1D], pbc=[True, False]),
                [2, 2],
            ).apply(d),
        ),
    }
    for name, (operator_class, make) in makers.items():
        path = directory / f"{name}.npz"
        if ctx.is_base:
            directory.mkdir(parents=True, exist_ok=True)
            make().save(path)
        loaded = STStatistics.load(path)
        _record_statistics(ctx, f"files/{name}", loaded)
        _record_plan(ctx, f"files/{name}", operator_class.from_statistics(loaded).plan)


def case_oracle(ctx: Context) -> None:
    """The oracle's outputs on the inputs of the cases (NumPy, float64; one run per tree)."""
    import numpy as np

    names = ctx.names
    spec_apply = names.resolve("estel._spec.planar:spec_apply")
    spec_apply_hpx = names.resolve("estel._spec.healpix:spec_apply_hpx")
    spec_tensor_prod = names.resolve("estel._spec.tensor_prod:spec_tensor_prod")
    ctx.record("oracle/2d", spec_apply(_field((1, 1, 16, 16)), 2, True, J=2, L=4, norm="self"))
    ctx.record(
        "oracle/2d-mask",
        spec_apply(_field((1, 1, 32, 32)), 2, False, mask=_mask_2d(32), J=3, L=4, norm="self"),
    )
    ctx.record("oracle/1d", spec_apply(_field((2, 1, 32)), 1, True, J=3, norm="self"))
    ctx.record("oracle/healpix", spec_apply_hpx(_sky(2), nside=4, J=2))
    ctx.record(
        "oracle/healpix-partial",
        spec_apply_hpx(_sky(3, 80), nside=8, cell_ids=np.arange(80), J=2),
    )
    hole = np.zeros(192, dtype=bool)
    hole[[5, 6, 70]] = True
    ctx.record("oracle/healpix-mask", spec_apply_hpx(_sky(2), nside=4, mask=hole, J=2))
    for lowpass in ("ST", None):
        ctx.record(
            f"oracle/tensor-prod-{lowpass}",
            spec_tensor_prod(
                _field((1, 1, 16, 16, 8)),
                [2, 1],
                [True, True],
                [2, 2],
                lowpass=lowpass,
                norm="self",
            ),
        )
    spec_spectra = names.resolve("estel._spec.spectra:spec_spectra")
    ctx.record("oracle/spectrum-auto", spec_spectra(_field((1, 1, 32, 32)), True, J=3))
    masked = _field((1, 1, 32, 32))
    masked[..., _mask_2d(32)] = np.nan
    ctx.record("oracle/spectrum-masked", spec_spectra(masked, False, J=3, method="nan_mask"))


CASES: dict[str, Callable[[Context], None]] = {
    "1d": case_1d,
    "2d-periodic": case_periodic_2d,
    "2d-mask": case_mask_2d,
    "healpix-full": case_full_healpix,
    "healpix-partial": case_partial_healpix,
    "healpix-mask": case_mask_healpix,
    "tensor-prod-st": case_st_tensor_prod,
    "tensor-prod-none": case_none_tensor_prod,
    "spectrum-auto": case_spectrum_auto,
    "spectrum-masked": case_spectrum_masked,
    "cross": case_cross,
    "norms": case_norms,
    "reduce-select": case_reduce_select,
    "synthesis-2d": case_synthesis_2d,
    "synthesis-healpix": case_synthesis_healpix,
    "files": case_files,
    "oracle": case_oracle,
}
# The cases expected to raise, on both trees, with why: any other case that raises on both trees
# proved nothing, and the verdict says so (status 3).
EXPECTED_FAILURES: dict[tuple[str, str], str] = {
    ("synthesis-2d", "jax"): "syntheses need L-BFGS, which only PyTorch provides in step 1 (LF1)",
    (
        "synthesis-healpix",
        "jax",
    ): "syntheses need L-BFGS, which only PyTorch provides in step 1 (LF1)",
}
# The oracle is NumPy only, in float64: one run per tree.
NUMPY_ONLY = frozenset({"oracle"})


def run_case(name: str, tree: Path, backend: str, dtype: str, files: Path, is_base: bool) -> Any:
    """In the worker: run one case on ``tree`` and return its outputs. A case that raises
    records the type of its error (its message is not a value)."""
    warnings.simplefilter("ignore")
    try:
        import torch
    except ImportError:  # the JAX-only environment of CI
        pass
    else:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
    import estel

    if name not in NUMPY_ONLY:
        estel.set_backend(backend)
        estel.set_default_dtype(dtype)  # type: ignore[arg-type]
    ctx = Context(Names(read_table(tree)), backend, dtype, files, is_base)
    try:
        CASES[name](ctx)
    except Exception as error:
        ctx.outputs["error"] = ("dump", ("raises", type(error).__name__))
    return {"estel": str(Path(estel.__file__).resolve()), "outputs": ctx.outputs}


# ================================================================== comparing outputs


@dataclass
class Difference:
    where: str  # case/backend/dtype/output
    status: int
    detail: str


def compare_arrays(where: str, base: Any, branch: Any, dtype: str) -> Difference | None:
    """None if bitwise equal; otherwise how they differ, against the parity tolerance."""
    import numpy as np

    if base.dtype != branch.dtype or base.shape != branch.shape:
        return Difference(
            where,
            BEYOND,
            f"dtype or shape {base.dtype}{base.shape} -> {branch.dtype}{branch.shape}",
        )
    if np.ascontiguousarray(base).tobytes() == np.ascontiguousarray(branch).tobytes():
        return None
    if base.dtype.kind not in "fc":
        return Difference(where, BEYOND, "integer or boolean values differ")
    nan_base, nan_branch = np.isnan(base), np.isnan(branch)
    if not np.array_equal(nan_base, nan_branch):
        return Difference(where, BEYOND, "NaN positions differ")
    finite = ~nan_base
    a, b = base[finite], branch[finite]
    if a.size == 0:
        return Difference(where, WITHIN, "NaN payloads differ (the bits of a NaN)")
    absolute = np.abs(b - a)
    largest = float(absolute.max())
    scale = float(np.abs(a).max())
    relative = largest / scale if scale else math.inf
    rtol = RTOL.get(dtype, RTOL["float32"] if base.dtype in (np.float32, np.complex64) else 1e-10)
    atol = 0.1 * rtol * scale
    within = bool(np.all(absolute <= atol + rtol * np.abs(a)))
    return Difference(
        where,
        WITHIN if within else BEYOND,
        f"largest absolute difference {largest:.3e}, relative to max|main| {relative:.3e} "
        f"(rtol {rtol:g}, atol {atol:.3e})",
    )


def compare_outputs(
    where: str, base: dict[str, Any], branch: dict[str, Any], dtype: str
) -> list[Difference]:
    import numpy as np

    differences = []
    for name in sorted(set(base) | set(branch)):
        here = f"{where}/{name}"
        if name not in branch or name not in base:
            side = "main" if name in base else "the branch"
            differences.append(Difference(here, BEYOND, f"an output of {side} only"))
            continue
        a, b = base[name], branch[name]
        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            difference = compare_arrays(here, a, b, dtype)
            if difference:
                differences.append(difference)
        elif not (isinstance(a, np.ndarray) or isinstance(b, np.ndarray)) and a == b:
            continue
        else:
            differences.append(Difference(here, BEYOND, f"the dumps differ: {_first_change(a, b)}"))
    return differences


def _first_change(a: Any, b: Any, path: str = "") -> str:
    """Where two dumps first differ, for the report."""
    if isinstance(a, tuple) and isinstance(b, tuple) and len(a) == len(b):
        for index, (x, y) in enumerate(zip(a, b, strict=True)):
            if x != y:
                return _first_change(x, y, f"{path}[{index}]")
    text_a, text_b = repr(a), repr(b)
    return f"at {path or 'the top'}: {text_a[:160]} -> {text_b[:160]}"


# ================================================================== bodies by AST


class _Canonical(ast.NodeTransformer):
    """Names of the branch mapped back to those of ``main``; docstrings removed."""

    def __init__(self, bare: dict[str, str], modules: dict[str, str]) -> None:
        self.bare, self.modules = bare, modules

    def _name(self, name: str) -> str:
        return self.bare.get(name, name)

    def _module(self, module: str) -> str:
        if module in self.modules:
            return self.modules[module]
        return ".".join(self._name(part) for part in module.split("."))

    def visit_Name(self, node: ast.Name) -> ast.AST:
        node.id = self._name(node.id)
        return node

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        self.generic_visit(node)
        node.attr = self._name(node.attr)
        return node

    def visit_alias(self, node: ast.alias) -> ast.AST:
        node.name = self._module(node.name)
        if node.asname:
            node.asname = self._name(node.asname)
        return node

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.AST:
        self.generic_visit(node)
        if node.module:
            node.module = self._module(node.module)
        return node

    def visit_keyword(self, node: ast.keyword) -> ast.AST:
        self.generic_visit(node)
        if node.arg:
            node.arg = self._name(node.arg)
        return node

    def visit_arg(self, node: ast.arg) -> ast.AST:
        self.generic_visit(node)
        node.arg = self._name(node.arg)
        return node

    def _strip(self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> ast.AST:
        body = node.body
        first = body[0] if body else None
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            node.body = body[1:] or [ast.Pass()]
        node.name = self._name(node.name)
        self.generic_visit(node)
        return node

    visit_FunctionDef = _strip  # noqa: N815
    visit_AsyncFunctionDef = _strip  # noqa: N815
    visit_ClassDef = _strip  # noqa: N815


def function_bodies(src: Path, names: Names, kept: frozenset[str] = frozenset()) -> dict[str, str]:
    """Every function and method of ``src/estel/``, by its name of ``main``: its AST dump.

    ``kept``: names defined on ``main`` (:func:`defined_names`), never mapped back. A row
    whose new name is also an old one (a hook ``_planar_conv`` moved to the method ``_conv``
    of ``DataCartesian``) would otherwise map every call of the existing ``_conv`` back.
    """
    bare = {new: old for new, old in names.bare_back().items() if new not in kept}
    modules = dict(names.module_back)
    bodies: dict[str, str] = {}
    for path in sorted((src / "estel").rglob("*.py")):
        rel = path.relative_to(src).with_suffix("")
        parts = list(rel.parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        module = ".".join(parts)
        tree = ast.parse(path.read_text())
        for qualname, node in _functions(tree.body, ""):
            canonical = names.canonical(module, qualname)
            clean = _Canonical(bare, modules).visit(node)
            bodies[canonical] = ast.dump(clean, include_attributes=False)
    return bodies


def defined_names(bodies: dict[str, str]) -> frozenset[str]:
    """Every module, class and function name of :func:`function_bodies`."""
    found: set[str] = set()
    for name in bodies:
        module, _, qualname = name.partition(":")
        found |= set(module.split(".")) | set(qualname.split("."))
    return frozenset(found)


def _functions(body: list[ast.stmt], prefix: str) -> Iterator[tuple[str, ast.AST]]:
    for node in body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            yield f"{prefix}{node.name}", node
        elif isinstance(node, ast.ClassDef):
            yield from _functions(node.body, f"{prefix}{node.name}.")


def compare_bodies(base: dict[str, str], branch: dict[str, str]) -> dict[str, list[str]]:
    """The bodies that differ, and the definitions on one side only: listed for review."""
    return {
        "differ": sorted(n for n in set(base) & set(branch) if base[n] != branch[n]),
        "main only": sorted(set(base) - set(branch)),
        "branch only": sorted(set(branch) - set(base)),
    }


# ================================================================== the two trees


@dataclass(frozen=True)
class Job:
    case: str
    backend: str
    dtype: str

    @property
    def label(self) -> str:
        return f"{self.case}/{self.backend}/{self.dtype}"


def jobs(cases: list[str]) -> list[Job]:
    found = []
    for case in cases:
        if case in NUMPY_ONLY:
            found.append(Job(case, "numpy", "float64"))
            continue
        found += [Job(case, b, d) for b in BACKENDS for d in DTYPES]
    return found


def _worker(job: Job, tree: Path, files: Path, is_base: bool, out: Path) -> tuple[Job, Any]:
    env = {**os.environ, **ENVIRONMENT, "PYTHONPATH": str(tree / "src")}
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        job.case,
        job.backend,
        job.dtype,
        "--tree",
        str(tree),
        "--files",
        str(files),
        "--out",
        str(out),
    ] + (["--is-base"] if is_base else [])
    result = subprocess.run(command, env=env, capture_output=True, text=True, check=False)
    if result.returncode != 0 or not out.exists():
        return job, {"crash": (result.stderr or result.stdout)[-2000:]}
    with out.open("rb") as handle:
        return job, pickle.load(handle)


def run_tree(
    tree: Path, cases: list[Job], files: Path, scratch: Path, is_base: bool, workers: int
) -> dict[Job, Any]:
    results: dict[Job, Any] = {}
    side = "main" if is_base else "branch"
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                _worker,
                job,
                tree,
                files,
                is_base,
                scratch / f"{side}-{job.label.replace('/', '-')}.pkl",
            )
            for job in cases
        ]
        for future in concurrent.futures.as_completed(futures):
            job, result = future.result()
            results[job] = result
    return results


def compare_runs(base: dict[Job, Any], branch: dict[Job, Any]) -> list[Difference]:
    differences = []
    for job in sorted(base, key=lambda j: j.label):
        a, b = base[job], branch.get(job, {"crash": "not run"})
        if "crash" in a or "crash" in b:
            both = "crash" in a and "crash" in b
            side = "both trees" if both else "main" if "crash" in a else "the branch"
            text = str(a.get("crash") or b.get("crash") or "").strip().splitlines()
            status = ERROR if both else BEYOND
            differences.append(
                Difference(job.label, status, f"the worker failed on {side}: {text[-1:]}")
            )
            continue
        differences += compare_outputs(job.label, a["outputs"], b["outputs"], job.dtype)
        raised = a["outputs"].get("error")
        if raised is not None and raised == b["outputs"].get("error"):
            reason = EXPECTED_FAILURES.get((job.case, job.backend))
            detail = f"raised {raised[1][1]} on both trees"
            if reason:
                differences.append(Difference(job.label, EQUAL, f"{detail}, as expected: {reason}"))
            else:
                differences.append(
                    Difference(
                        job.label, ERROR, f"{detail}: the case did not run, so it proves nothing"
                    )
                )
    return differences


def imported_from(where: str, tree: Path) -> bool:
    """Whether the ``estel`` a worker imported (its ``__file__``) is that of ``tree``."""
    return Path(where).resolve().is_relative_to((tree / "src").resolve())


def _git(*args: str, cwd: Path) -> str:
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def prove(base_ref: str, cases: list[str], workers: int) -> int:
    start = time.monotonic()
    scratch = Path(tempfile.mkdtemp(prefix="estel-prove-"))
    base_tree = scratch / "main"
    try:
        commit = _git("rev-parse", base_ref, cwd=REPO)
        head = _git("rev-parse", "HEAD", cwd=REPO)
        dirty = bool(_git("status", "--porcelain", "--untracked-files=no", cwd=REPO))
        _git("worktree", "add", "--detach", "--quiet", str(base_tree), commit, cwd=REPO)
    except subprocess.CalledProcessError as error:
        print(f"prove_unchanged: git failed: {error.stderr.strip()}", file=sys.stderr)
        shutil.rmtree(scratch, ignore_errors=True)
        return ERROR
    try:
        print(
            f"prove_unchanged: {base_ref} ({commit[:10]}) against the working tree "
            f"({head[:10]}{', with changes' if dirty else ''}); {len(jobs(cases))} runs per tree, "
            f"{workers} at a time"
        )
        files = scratch / "files"
        planned = jobs(cases)
        base_runs = run_tree(base_tree, planned, files, scratch, True, workers)
        branch_runs = run_tree(REPO, planned, files, scratch, False, workers)
        for side, runs in (("main", base_runs), ("branch", branch_runs)):
            expected = base_tree if side == "main" else REPO
            for job, result in runs.items():
                where = result.get("estel", "")
                if where and not imported_from(where, expected):
                    print(f"prove_unchanged: {job.label} on {side} imported estel from {where}")
                    return ERROR
        differences = compare_runs(base_runs, branch_runs)
        main_bodies = function_bodies(base_tree / "src", Names(read_table(base_tree)))
        branch_bodies = function_bodies(
            REPO / "src", Names(read_table(REPO)), defined_names(main_bodies)
        )
        bodies = compare_bodies(main_bodies, branch_bodies)
    finally:
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(base_tree)], cwd=REPO, check=False
        )
        shutil.rmtree(scratch, ignore_errors=True)
    return report(base_runs, differences, bodies, time.monotonic() - start)


def report(
    runs: dict[Job, Any],
    differences: list[Difference],
    bodies: dict[str, list[str]],
    seconds: float,
) -> int:
    n_outputs = sum(len(r.get("outputs", {})) for r in runs.values())
    status = max((d.status for d in differences), default=EQUAL)
    for difference in differences:
        level = {EQUAL: "expected", WITHIN: "within", BEYOND: "BEYOND", ERROR: "ERROR"}
        print(f"  {level[difference.status]} {difference.where}: {difference.detail}")
    n_differ = sum(d.status != EQUAL for d in differences)
    print("prove_unchanged: bodies compared by AST, modulo the table (listed for review):")
    for kind, listed in bodies.items():
        print(f"  {kind}: {len(listed)}")
        for name in listed:
            print(f"    {name}")
    print(
        f"prove_unchanged: {len(runs)} runs per tree, {n_outputs} outputs; "
        f"{n_differ} differ. Verdict {status}: {VERDICTS[status]}. "
        f"Wall time {seconds:.0f} s."
    )
    return status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", default="origin/main", help="the reference tree (git ref)")
    parser.add_argument("--case", action="append", choices=sorted(CASES), help="a case to run")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument(
        "--worker", nargs=3, metavar=("CASE", "BACKEND", "DTYPE"), help=argparse.SUPPRESS
    )
    parser.add_argument("--tree", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--files", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--out", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--is-base", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        case, backend, dtype = args.worker
        result = run_case(case, args.tree, backend, dtype, args.files, args.is_base)
        with args.out.open("wb") as handle:
            pickle.dump(result, handle)
        return 0
    return prove(args.base, args.case or list(CASES), args.workers)


if __name__ == "__main__":
    sys.exit(main())
