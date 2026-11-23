"""The versioned schema of the files ESTEL writes (Part II, §9.4, §12.7; task T2.5).

A file is one NumPy ``.npz`` archive, written at the exact path given: named NumPy arrays, plus
one member, ``__meta__``, a JSON document. The JSON document records:

* ``format``: what the file holds (``"estel.statistics"`` for ``STStatistics.save``);
* ``schema_version``: :data:`SCHEMA_VERSION`, increased whenever the content of a format
  changes, so that a reader knows what it reads;
* ``statistics_reference`` and ``synthesis_reference``: the reference counters of the writer
  (``estel.STATISTICS_REFERENCE``, ``estel.SYNTHESIS_REFERENCE``; Part I, §7.1): statistics
  written under another counter may differ from today's beyond the tolerances;
* ``estel_version``: the version of the writer, for error messages;
* ``content``: the description of the arrays, owned by the format.

The reader refuses another format, a schema version it does not know, and pickled arrays; it
warns when the statistics reference of the file differs from the current one.
"""

from __future__ import annotations

import json
import os
import warnings
import zipfile
from collections.abc import Mapping
from importlib import metadata
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .._reference import STATISTICS_REFERENCE, SYNTHESIS_REFERENCE

SCHEMA_VERSION = 5
"""The version of the schema written by this version of ESTEL. Version 2 renamed the values of
``norm`` (``"none"``, ``"self"``, ``"ref"``), ``ref_batch_mean`` and the reductions
(``angular_fourier``, ``scale_cosine``); version 3 stores the statistics in the compact layout
(Part III, §19; task T2.15): the buffer, its layout, and the compact scales of the
normalisation reference; version 4 stores the layout by chunks (task T3.7): one row per channel
pair and scale combination, the buffer unchanged; version 5 adds the statistics of tensor
products (task T3.6): their blocks, J, L and low-pass flags, the orientation shape of each kind
of chunk, and the blocks of the reference, the content of planar statistics unchanged. Files of
earlier versions are not read."""

META = "__meta__"
"""The name of the member holding the JSON document."""

STATISTICS_FORMAT = "estel.statistics"
"""The format of ``STStatistics.save``."""


def _estel_version() -> str:
    try:
        return metadata.version("estel")
    except metadata.PackageNotFoundError:
        return "unknown"


def write(
    path: str | os.PathLike[str],
    file_format: str,
    arrays: Mapping[str, NDArray[Any]],
    content: Mapping[str, Any],
) -> None:
    """Write ``arrays`` and the JSON ``content`` to ``path``, with the schema header."""
    if META in arrays:
        raise ValueError(f"{META!r} is reserved for the JSON document of the schema.")
    meta = {
        "format": file_format,
        "schema_version": SCHEMA_VERSION,
        "statistics_reference": STATISTICS_REFERENCE,
        "synthesis_reference": SYNTHESIS_REFERENCE,
        "estel_version": _estel_version(),
        "content": content,
    }
    members: dict[str, Any] = {META: np.array(json.dumps(meta, sort_keys=True)), **arrays}
    with open(path, "wb") as handle:
        np.savez(handle, **members)


def read(
    path: str | os.PathLike[str], file_format: str
) -> tuple[dict[str, NDArray[Any]], dict[str, Any]]:
    """The arrays and the JSON ``content`` of the file at ``path``, after checking its header.

    Raises ``ValueError`` if the file is not an ESTEL file of ``file_format``, or has a schema
    version this version of ESTEL does not read. Warns when it was written under another
    ``STATISTICS_REFERENCE``.
    """
    try:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise ValueError(
            f"{os.fspath(path)} is not an ESTEL file ({error}): pass a file written by "
            "STStatistics.save."
        ) from error
    if META not in arrays:
        raise ValueError(
            f"{os.fspath(path)} has no {META} member, so it was not written by ESTEL: pass a file "
            "written by STStatistics.save."
        )
    meta: dict[str, Any] = json.loads(str(arrays.pop(META)[()]))
    found = meta.get("format")
    if found != file_format:
        raise ValueError(
            f"{os.fspath(path)} holds {found!r}, not {file_format!r}: pass a file of that format."
        )
    version = meta.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"{os.fspath(path)} has schema version {version!r}, written by ESTEL "
            f"{meta.get('estel_version', 'unknown')}; this version of ESTEL reads version "
            f"{SCHEMA_VERSION}. Read it with the version of ESTEL that wrote it."
        )
    reference = meta.get("statistics_reference")
    if reference != STATISTICS_REFERENCE:
        warnings.warn(
            f"{os.fspath(path)} was written with STATISTICS_REFERENCE = {reference}, and this "
            f"version of ESTEL has {STATISTICS_REFERENCE}: its statistics may differ from those "
            "computed today beyond the tolerances. Recompute them if you compare the two.",
            UserWarning,
            stacklevel=3,
        )
    content: dict[str, Any] = meta["content"]
    return arrays, content
