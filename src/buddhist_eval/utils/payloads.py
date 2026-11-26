"""Parse decoded JSON prediction and reference values into scorer input types."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, TypeVar

from buddhist_eval.core.spans import Document, Span
from buddhist_eval.core.toc import Heading, PredictedHeading, TocPassage
from buddhist_eval.errors import BuddhistEvalError

T = TypeVar("T")
PayloadError = type[BuddhistEvalError]
Parser = Callable[[Any, str, PayloadError], T]


def parse_mapping(
    values: Mapping[str, Any], parse: Parser[T], side: str, error: PayloadError
) -> dict[str, T]:
    """Parse every value of an id-keyed mapping.

    Args:
        values: Mapping from item id to a decoded JSON value.
        parse: Parser applied to each value.
        side: ``"prediction"`` or ``"reference"``, used in error messages.
        error: Exception class raised for a malformed value.

    Returns:
        Mapping from item id to the parsed value.
    """
    return {
        item_id: parse(value, f"{side} {item_id!r}", error)
        for item_id, value in values.items()
    }


def parse_text(value: Any, where: str, error: PayloadError) -> str:
    """Return ``value`` if it is a string.

    Raises:
        BuddhistEvalError: ``error`` if ``value`` is not a string.
    """
    if not isinstance(value, str):
        raise error(f"{where} must be a string, got {type(value).__name__}")
    return value


def parse_spans(value: Any, where: str, error: PayloadError) -> tuple[Span, ...]:
    """Parse a list of ``{start, end, tag}`` objects into Spans.

    Raises:
        BuddhistEvalError: ``error`` if the list or any span is malformed.
    """
    items = _require_list(value, where, error)
    return tuple(
        _parse_span(item, f"{where}[{i}]", error) for i, item in enumerate(items)
    )


def parse_document(value: Any, where: str, error: PayloadError) -> Document:
    """Parse a ``{text, spans}`` object into a Document.

    Raises:
        BuddhistEvalError: ``error`` if the object or any span is malformed.
    """
    obj = _require_object(value, where, error, ("text", "spans"))
    return Document(
        text=parse_text(obj["text"], f"{where}.text", error),
        spans=parse_spans(obj["spans"], f"{where}.spans", error),
    )


def parse_predicted_headings(
    value: Any, where: str, error: PayloadError
) -> tuple[PredictedHeading, ...]:
    """Parse a list of ``{depth, anchor}`` objects into Predicted Headings.

    Raises:
        BuddhistEvalError: ``error`` if the list or any heading is malformed.
    """
    items = _require_list(value, where, error)
    headings = []
    for i, item in enumerate(items):
        obj = _require_object(item, f"{where}[{i}]", error, ("depth", "anchor"))
        headings.append(
            PredictedHeading(
                depth=_require_int(obj["depth"], f"{where}[{i}].depth", error),
                anchor=parse_text(obj["anchor"], f"{where}[{i}].anchor", error),
            )
        )
    return tuple(headings)


def parse_toc_passage(value: Any, where: str, error: PayloadError) -> TocPassage:
    """Parse a ``{text, headings: [{depth, start}]}`` object into a TocPassage.

    Raises:
        BuddhistEvalError: ``error`` if the object or any heading is malformed.
    """
    obj = _require_object(value, where, error, ("text", "headings"))
    items = _require_list(obj["headings"], f"{where}.headings", error)
    headings = []
    for i, item in enumerate(items):
        at = f"{where}.headings[{i}]"
        heading = _require_object(item, at, error, ("depth", "start"))
        headings.append(
            Heading(
                depth=_require_int(heading["depth"], f"{at}.depth", error),
                start=_require_int(heading["start"], f"{at}.start", error),
            )
        )
    return TocPassage(
        text=parse_text(obj["text"], f"{where}.text", error), headings=tuple(headings)
    )


def _parse_span(item: Any, where: str, error: PayloadError) -> Span:
    obj = _require_object(item, where, error, ("start", "end", "tag"))
    return Span(
        start=_require_int(obj["start"], f"{where}.start", error),
        end=_require_int(obj["end"], f"{where}.end", error),
        tag=parse_text(obj["tag"], f"{where}.tag", error),
    )


def _require_object(
    value: Any, where: str, error: PayloadError, keys: tuple[str, ...]
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise error(f"{where} must be an object, got {type(value).__name__}")
    missing = [key for key in keys if key not in value]
    if missing:
        raise error(f"{where} is missing {', '.join(missing)}")
    return value


def _require_list(value: Any, where: str, error: PayloadError) -> list[Any]:
    if not isinstance(value, list):
        raise error(f"{where} must be a list, got {type(value).__name__}")
    return value


def _require_int(value: Any, where: str, error: PayloadError) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise error(f"{where} must be an integer, got {value!r}")
    return value
