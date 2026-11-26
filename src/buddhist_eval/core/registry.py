"""Registry of every Scorer, looked up by name with one ``score`` signature.

Each Scorer wraps a batch metric. Predictions and references are mappings
from item id to decoded JSON values; ``Scorer.score`` returns one float.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from buddhist_eval.core.error_rates import (
    ErrorRateBatchScore,
    score_cer_batch,
    score_wer_batch,
)
from buddhist_eval.core.mcq import score_mcq
from buddhist_eval.core.scorer import Parameter, Scorer, SortDirection
from buddhist_eval.core.spans import Document, score_spans_batch
from buddhist_eval.core.toc import score_toc_batch
from buddhist_eval.errors import (
    InvalidPredictionError,
    InvalidReferenceError,
    UnknownScorerError,
)
from buddhist_eval.utils.payloads import (
    PayloadError,
    parse_document,
    parse_mapping,
    parse_predicted_headings,
    parse_spans,
    parse_text,
    parse_toc_passage,
)

_TEXT_SCHEMA: dict[str, Any] = {"type": "string"}

_SPAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "start": {"type": "integer", "minimum": 0},
        "end": {
            "type": "integer",
            "minimum": 0,
            "description": "Inclusive end offset.",
        },
        "tag": {"type": "string"},
    },
    "required": ["start", "end", "tag"],
}

_SPANS_SCHEMA: dict[str, Any] = {"type": "array", "items": _SPAN_SCHEMA}

_DOCUMENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"text": {"type": "string"}, "spans": _SPANS_SCHEMA},
    "required": ["text", "spans"],
}

_PREDICTED_HEADINGS_SCHEMA: dict[str, Any] = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "depth": {"type": "integer", "minimum": 1},
            "anchor": {
                "type": "string",
                "description": "Verbatim text the heading starts with.",
            },
        },
        "required": ["depth", "anchor"],
    },
}

_TOC_PASSAGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "headings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "depth": {"type": "integer", "minimum": 1},
                    "start": {"type": "integer", "minimum": 0},
                },
                "required": ["depth", "start"],
            },
        },
    },
    "required": ["text", "headings"],
}

_LANGUAGE = Parameter(
    name="language",
    kind=str,
    default="bo",
    description='ISO 639-1 code; "bo" converts both texts to Wylie before scoring.',
)

_AGGREGATE = Parameter(
    name="aggregate",
    kind=str,
    default="mean",
    choices=("mean", "corpus"),
    description='"mean" averages per-item rates (EvalAI); "corpus" is total edits over total Reference length.',
)

_IOU_THRESHOLD = Parameter(
    name="iou_threshold",
    kind=float,
    default=0.5,
    exclusive_minimum=0.0,
    maximum=1.0,
    description="Minimum IoU for a Predicted Span to match a Gold Span.",
)

_TOLERANCE = Parameter(
    name="tolerance",
    kind=int,
    default=3,
    minimum=0,
    description="Maximum character distance between a located Anchor and a Gold Heading start.",
)


def _texts(values: Mapping[str, Any], side: str, error: PayloadError) -> dict[str, str]:
    return parse_mapping(values, parse_text, side, error)


def _pick_rate(batch: ErrorRateBatchScore, aggregate: str) -> float:
    return batch.mean if aggregate == "mean" else batch.corpus


def _score_cer(
    predictions: Mapping[str, Any],
    references: Mapping[str, Any],
    params: Mapping[str, Any],
) -> float:
    batch = score_cer_batch(
        _texts(references, "reference", InvalidReferenceError),
        _texts(predictions, "prediction", InvalidPredictionError),
        language=params["language"],
    )
    return _pick_rate(batch, params["aggregate"])


def _score_wer(
    predictions: Mapping[str, Any],
    references: Mapping[str, Any],
    params: Mapping[str, Any],
) -> float:
    batch = score_wer_batch(
        _texts(references, "reference", InvalidReferenceError),
        _texts(predictions, "prediction", InvalidPredictionError),
        language=params["language"],
    )
    return _pick_rate(batch, params["aggregate"])


def _score_span_f1(
    predictions: Mapping[str, Any],
    references: Mapping[str, Any],
    params: Mapping[str, Any],
) -> float:
    golds = parse_mapping(
        references, parse_document, "reference", InvalidReferenceError
    )
    spans = parse_mapping(
        predictions, parse_spans, "prediction", InvalidPredictionError
    )
    predicted = {
        item_id: Document(text=golds[item_id].text, spans=item_spans)
        for item_id, item_spans in spans.items()
        if item_id in golds
    }
    return score_spans_batch(golds, predicted, iou_threshold=params["iou_threshold"]).f1


def _score_toc(
    predictions: Mapping[str, Any],
    references: Mapping[str, Any],
    params: Mapping[str, Any],
) -> float:
    passages = parse_mapping(
        references, parse_toc_passage, "reference", InvalidReferenceError
    )
    predicted = parse_mapping(
        predictions, parse_predicted_headings, "prediction", InvalidPredictionError
    )
    return score_toc_batch(passages, predicted, tolerance=params["tolerance"]).mean


def _score_mcq(
    predictions: Mapping[str, Any],
    references: Mapping[str, Any],
    params: Mapping[str, Any],
) -> float:
    answer_key = _texts(references, "reference", InvalidReferenceError)
    return score_mcq(
        answer_key, _texts(predictions, "prediction", InvalidPredictionError)
    ).accuracy


_SCORERS: tuple[Scorer, ...] = (
    Scorer(
        name="cer",
        label="Character Error Rate (CER)",
        direction=SortDirection.LOWER_IS_BETTER,
        parameters=(_LANGUAGE, _AGGREGATE),
        prediction_schema=_TEXT_SCHEMA,
        reference_schema=_TEXT_SCHEMA,
        score_function=_score_cer,
    ),
    Scorer(
        name="wer",
        label="Word Error Rate (WER)",
        direction=SortDirection.LOWER_IS_BETTER,
        parameters=(_LANGUAGE, _AGGREGATE),
        prediction_schema=_TEXT_SCHEMA,
        reference_schema=_TEXT_SCHEMA,
        score_function=_score_wer,
    ),
    Scorer(
        name="span_f1",
        label="Span IoU F1",
        direction=SortDirection.HIGHER_IS_BETTER,
        parameters=(_IOU_THRESHOLD,),
        prediction_schema=_SPANS_SCHEMA,
        reference_schema=_DOCUMENT_SCHEMA,
        score_function=_score_span_f1,
    ),
    Scorer(
        name="toc",
        label="TOC heading score",
        direction=SortDirection.HIGHER_IS_BETTER,
        parameters=(_TOLERANCE,),
        prediction_schema=_PREDICTED_HEADINGS_SCHEMA,
        reference_schema=_TOC_PASSAGE_SCHEMA,
        score_function=_score_toc,
    ),
    Scorer(
        name="mcq_accuracy",
        label="Multiple-choice accuracy",
        direction=SortDirection.HIGHER_IS_BETTER,
        parameters=(),
        prediction_schema=_TEXT_SCHEMA,
        reference_schema=_TEXT_SCHEMA,
        score_function=_score_mcq,
    ),
)

_BY_NAME: dict[str, Scorer] = {scorer.name: scorer for scorer in _SCORERS}


def list_scorers() -> tuple[Scorer, ...]:
    """Return every registered Scorer in registration order.

    Returns:
        A tuple of Scorers.
    """
    return _SCORERS


def get_scorer(name: str) -> Scorer:
    """Look up a registered Scorer by name.

    Args:
        name: The Scorer name stored on a challenge, for example ``"cer"``.

    Returns:
        The Scorer registered under ``name``.

    Raises:
        UnknownScorerError: If no Scorer has that name.
    """
    try:
        return _BY_NAME[name]
    except KeyError:
        raise UnknownScorerError(
            f"no scorer named {name!r} (known: {sorted(_BY_NAME)})"
        ) from None
