"""Load EvalAI ground-truth and submission JSON files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, NoReturn

from buddhist_eval.errors import MalformedEvalAIFileError


def load_evalai_json(path: Path) -> dict[str, str]:
    """Read an EvalAI JSON file into a filename-to-text mapping.

    Accepts a list of ``{filename, label}`` or ``{filename, prediction}``
    objects, or a single such object treated as a one-item list.

    Args:
        path: Local path to the JSON file.

    Returns:
        Mapping from ``filename`` to the corresponding Reference or
        Prediction text.

    Raises:
        MalformedEvalAIFileError: If the file is invalid JSON, an entry is
            missing ``filename`` or a text field, or a ``filename`` is duplicated.
    """
    data = _read_json(path)
    entries = _coerce_entries(data, path)
    return _parse_entries(entries, path)


def _malformed(path: Path) -> MalformedEvalAIFileError:
    return MalformedEvalAIFileError(f"Malformed EvalAI file: {path}")


def _raise_malformed(path: Path, cause: BaseException | None = None) -> NoReturn:
    if cause is None:
        raise _malformed(path)
    raise _malformed(path) from cause


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        _raise_malformed(path, exc)


def _coerce_entries(data: Any, path: Path) -> list[Any]:
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return data
    _raise_malformed(path)


def _entry_text(entry: dict[str, Any], path: Path) -> str:
    if "label" in entry:
        text = entry["label"]
    elif "prediction" in entry:
        text = entry["prediction"]
    else:
        _raise_malformed(path)
    if not isinstance(text, str):
        _raise_malformed(path)
    return text


def _parse_entry(entry: Any, path: Path) -> tuple[str, str]:
    if not isinstance(entry, dict):
        _raise_malformed(path)
    if "filename" not in entry:
        _raise_malformed(path)
    filename = entry["filename"]
    if not isinstance(filename, str):
        _raise_malformed(path)
    return filename, _entry_text(entry, path)


def _parse_entries(entries: list[Any], path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for entry in entries:
        filename, text = _parse_entry(entry, path)
        if filename in result:
            _raise_malformed(path)
        result[filename] = text
    return result
