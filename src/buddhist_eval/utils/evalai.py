"""Load EvalAI ground-truth and submission JSON files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from buddhist_eval.errors import MalformedEvalAIFileError


def load_evalai_json(path: Path) -> dict[str, str]:
    """Read an EvalAI JSON file into a filename-to-text mapping.

    Accepts a list of ``{filename, label}`` or ``{filename, prediction}``
    objects, or a single such object treated as a one-item list.

    Args:
        path: Local path to the JSON file.

    Returns:
        Mapping from ``filename`` to the corresponding label or prediction text.

    Raises:
        MalformedEvalAIFileError: If the file is invalid JSON, an entry is
            missing ``filename`` or a text field, or a ``filename`` is duplicated.
    """
    try:
        raw = path.read_text(encoding="utf-8")
        data: Any = json.loads(raw)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise MalformedEvalAIFileError(f"Malformed EvalAI file: {path}") from exc

    if isinstance(data, dict):
        entries = [data]
    elif isinstance(data, list):
        entries = data
    else:
        raise MalformedEvalAIFileError(f"Malformed EvalAI file: {path}")

    result: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise MalformedEvalAIFileError(f"Malformed EvalAI file: {path}")
        if "filename" not in entry:
            raise MalformedEvalAIFileError(f"Malformed EvalAI file: {path}")
        filename = entry["filename"]
        if not isinstance(filename, str):
            raise MalformedEvalAIFileError(f"Malformed EvalAI file: {path}")
        if "label" in entry:
            text = entry["label"]
        elif "prediction" in entry:
            text = entry["prediction"]
        else:
            raise MalformedEvalAIFileError(f"Malformed EvalAI file: {path}")
        if not isinstance(text, str):
            raise MalformedEvalAIFileError(f"Malformed EvalAI file: {path}")
        if filename in result:
            raise MalformedEvalAIFileError(f"Malformed EvalAI file: {path}")
        result[filename] = text
    return result
