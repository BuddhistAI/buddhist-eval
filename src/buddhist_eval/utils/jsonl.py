"""Load the arena's JSONL files: Inputs, Ground Truth and Predictions.

Every file holds one JSON object per line, keyed by a string ``id``:

- Inputs: ``{"id": ..., "input": <any>, "media": "<path>"}`` (``media`` optional)
- Ground Truth: ``{"id": ..., "reference": <any>}``
- Predictions: ``{"id": ..., "prediction": <any>}``

The loaders check only this row structure. Whether a value has the shape a
Scorer expects is checked by ``Scorer.validate_ground_truth`` and
``Scorer.validate_predictions``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import IO, Any, Union

from buddhist_eval.errors import MalformedJsonlError

Source = Union[str, "os.PathLike[str]", IO[str], IO[bytes]]
"""A file path, or an open text or binary stream (read to the end)."""

_ID = "id"
_MEDIA = "media"


def load_inputs(source: Source) -> dict[str, Any]:
    """Read an Inputs JSONL file into a mapping from id to Input value.

    Args:
        source: Path or open stream of ``{"id", "input", "media"?}`` rows.

    Returns:
        Mapping from id to the decoded ``input`` value, in file order.

    Raises:
        MalformedJsonlError: If a line is not a JSON object, an ``id`` is
            missing, blank, not a string or repeated, ``input`` is missing,
            ``media`` is not a non-empty string, or a row has other keys.
    """
    rows = _load_rows(source, "input", optional=(_MEDIA,))
    return {item_id: row["input"] for item_id, row in rows.items()}


def load_media_paths(source: Source) -> dict[str, str]:
    """Read the ``media`` paths of an Inputs JSONL file.

    Args:
        source: Path or open stream of Inputs rows. A stream is read to the
            end, so pass a fresh one rather than one ``load_inputs`` consumed.

    Returns:
        Mapping from id to the path inside the Media Archive, for rows that
        have one.

    Raises:
        MalformedJsonlError: For the same reasons as ``load_inputs``.
    """
    rows = _load_rows(source, "input", optional=(_MEDIA,))
    return {item_id: row[_MEDIA] for item_id, row in rows.items() if _MEDIA in row}


def load_ground_truth(source: Source) -> dict[str, Any]:
    """Read a Ground Truth JSONL file into a mapping from id to Reference.

    Args:
        source: Path or open stream of ``{"id", "reference"}`` rows.

    Returns:
        Mapping from id to the decoded ``reference`` value, in file order.

    Raises:
        MalformedJsonlError: If a line is not a JSON object, an ``id`` is
            missing, blank, not a string or repeated, ``reference`` is
            missing, or a row has other keys.
    """
    rows = _load_rows(source, "reference")
    return {item_id: row["reference"] for item_id, row in rows.items()}


def load_predictions(source: Source) -> dict[str, Any]:
    """Read a Predictions JSONL file into a mapping from id to Prediction.

    Args:
        source: Path or open stream of ``{"id", "prediction"}`` rows.

    Returns:
        Mapping from id to the decoded ``prediction`` value, in file order.

    Raises:
        MalformedJsonlError: If a line is not a JSON object, an ``id`` is
            missing, blank, not a string or repeated, ``prediction`` is
            missing, or a row has other keys.
    """
    rows = _load_rows(source, "prediction")
    return {item_id: row["prediction"] for item_id, row in rows.items()}


def _load_rows(
    source: Source, value_key: str, optional: tuple[str, ...] = ()
) -> dict[str, dict[str, Any]]:
    name, text = _read_text(source)
    allowed = {_ID, value_key, *optional}
    rows: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(text.split("\n"), start=1):
        if not line.strip():
            continue
        where = f"{name} line {line_number}"
        row = _parse_row(line, where)
        _check_keys(row, allowed, value_key, where)
        item_id = _check_id(row.get(_ID), where)
        if item_id in rows:
            raise MalformedJsonlError(f"{where}: duplicate id {item_id!r}")
        if _MEDIA in row and _MEDIA in optional:
            _check_media(row[_MEDIA], where)
        rows[item_id] = row
    if not rows:
        raise MalformedJsonlError(f"{name} has no rows")
    return rows


def _read_text(source: Source) -> tuple[str, str]:
    if isinstance(source, (str, os.PathLike)):
        path = Path(source)
        try:
            data: str | bytes = path.read_bytes()
        except OSError as exc:
            raise MalformedJsonlError(f"cannot read {path}: {exc}") from exc
        name = str(path)
    else:
        data = source.read()
        name = getattr(source, "name", None) or "<stream>"
        name = str(name)
    if isinstance(data, bytes):
        try:
            data = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise MalformedJsonlError(f"{name} is not UTF-8: {exc}") from exc
    return name, data.removeprefix("﻿")


def _parse_row(line: str, where: str) -> dict[str, Any]:
    try:
        row = json.loads(line)
    except json.JSONDecodeError as exc:
        raise MalformedJsonlError(f"{where}: invalid JSON ({exc.msg})") from exc
    if not isinstance(row, dict):
        raise MalformedJsonlError(
            f"{where}: expected a JSON object, got {type(row).__name__}"
        )
    return row


def _check_keys(
    row: dict[str, Any], allowed: set[str], value_key: str, where: str
) -> None:
    for key in (_ID, value_key):
        if key not in row:
            raise MalformedJsonlError(f"{where}: missing {key!r}")
    unknown = sorted(set(row) - allowed)
    if unknown:
        raise MalformedJsonlError(
            f"{where}: unexpected key(s) {unknown}; allowed {sorted(allowed)}"
        )


def _check_id(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MalformedJsonlError(
            f"{where}: 'id' must be a non-empty string, got {value!r}"
        )
    return value


def _check_media(value: Any, where: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise MalformedJsonlError(
            f"{where}: 'media' must be a non-empty string, got {value!r}"
        )
