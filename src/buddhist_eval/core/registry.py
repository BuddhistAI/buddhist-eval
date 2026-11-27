"""Registry of every Scorer, looked up by name with one ``score`` signature.

Each Scorer wraps a batch metric from ``buddhist_eval.core`` without changing
its numbers. Scorer names are immutable: a changed computation is registered
under a new name (see docs/adr/0002-scorer-names-are-immutable.md).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from buddhist_eval.core.error_rates import score_cer_batch, score_wer_batch
from buddhist_eval.core.mcq import score_mcq
from buddhist_eval.core.scorer import (
    Direction,
    DisplayFormat,
    DisplayStyle,
    Parameter,
    Scorer,
    SummaryEntry,
)
from buddhist_eval.core.spans import Document, score_spans_batch
from buddhist_eval.core.toc import TocPassage, score_toc_batch
from buddhist_eval.errors import UnknownScorerError
from buddhist_eval.utils.payloads import (
    parse_gold_headings,
    parse_predicted_headings,
    parse_spans,
    parse_text,
)

_TEXT_SCHEMA: dict[str, Any] = {"type": "string"}

_SPAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "start": {"type": "integer", "minimum": 0},
        "end": {
            "type": "integer",
            "minimum": 0,
            "description": "Inclusive end offset in the Input passage text.",
        },
        "tag": {"type": "string"},
    },
    "required": ["start", "end", "tag"],
}

_SPANS_SCHEMA: dict[str, Any] = {"type": "array", "items": _SPAN_SCHEMA}

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

_GOLD_HEADINGS_SCHEMA: dict[str, Any] = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "depth": {"type": "integer", "minimum": 1},
            "start": {
                "type": "integer",
                "minimum": 0,
                "description": "Character offset in the Input passage text.",
            },
        },
        "required": ["depth", "start"],
    },
}

_PASSAGE_INPUT_SCHEMA: dict[str, Any] = {
    "type": "string",
    "description": "Passage text; offsets in references and predictions index into it.",
}

_IGNORED_INPUT_SCHEMA: dict[str, Any] = {
    "description": "Not read by this Scorer; any JSON value.",
}

_LANGUAGE = Parameter(
    name="language",
    kind=str,
    default="bo",
    description='ISO 639-1 code; "bo" converts both texts to Wylie before scoring.',
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

_PERCENT_2 = DisplayFormat(DisplayStyle.PERCENT, 2)
_DECIMAL_3 = DisplayFormat(DisplayStyle.DECIMAL, 3)
# Accuracy reads naturally as a share of questions, so it is shown as a percent.
_PERCENT_1 = DisplayFormat(DisplayStyle.PERCENT, 1)
_COUNT = DisplayFormat(DisplayStyle.DECIMAL, 0)

# Public Summaries declare only top-level aggregate numbers from each metric's
# details. Per-item results (items, skipped/missing/extra ids) stay private.
# span_f1's per_tag is an aggregate, but its tags differ per Challenge, so it
# cannot be declared as fixed keys. toc's details hold no aggregate besides
# the Headline Metric (mean), so its summary is empty.
_CER_SUMMARY = (SummaryEntry("corpus", "Corpus CER", _PERCENT_2),)
_WER_SUMMARY = (SummaryEntry("corpus", "Corpus WER", _PERCENT_2),)
_SPAN_F1_SUMMARY = (
    SummaryEntry("precision", "Precision", _DECIMAL_3),
    SummaryEntry("recall", "Recall", _DECIMAL_3),
    SummaryEntry("mean_iou", "Mean IoU of matched spans", _DECIMAL_3),
)
_MCQ_SUMMARY = (
    SummaryEntry("n_correct", "Correct answers", _COUNT),
    SummaryEntry("n_total", "Questions", _COUNT),
)

Values = Mapping[str, Any]


def _compute_cer(
    inputs: Values, references: Values, predictions: Values, params: Values
) -> tuple[float, Any]:
    batch = score_cer_batch(
        dict(references), dict(predictions), language=params["language"]
    )
    return batch.mean, batch


def _compute_wer(
    inputs: Values, references: Values, predictions: Values, params: Values
) -> tuple[float, Any]:
    batch = score_wer_batch(
        dict(references), dict(predictions), language=params["language"]
    )
    return batch.mean, batch


def _compute_span_f1(
    inputs: Values, references: Values, predictions: Values, params: Values
) -> tuple[float, Any]:
    golds = {
        item_id: Document(text=inputs[item_id], spans=spans)
        for item_id, spans in references.items()
    }
    predicted = {
        item_id: Document(text=inputs.get(item_id, ""), spans=spans)
        for item_id, spans in predictions.items()
    }
    batch = score_spans_batch(golds, predicted, iou_threshold=params["iou_threshold"])
    return batch.f1, batch


def _compute_toc(
    inputs: Values, references: Values, predictions: Values, params: Values
) -> tuple[float, Any]:
    passages = {
        item_id: TocPassage(text=inputs[item_id], headings=headings)
        for item_id, headings in references.items()
    }
    batch = score_toc_batch(passages, dict(predictions), tolerance=params["tolerance"])
    return batch.mean, batch


def _compute_mcq(
    inputs: Values, references: Values, predictions: Values, params: Values
) -> tuple[float, Any]:
    result = score_mcq(dict(references), dict(predictions))
    return result.accuracy, result


_SCORERS: tuple[Scorer, ...] = (
    Scorer(
        name="cer",
        label="Character Error Rate (CER)",
        direction=Direction.LOWER_IS_BETTER,
        parameters=(_LANGUAGE,),
        input_schema=_IGNORED_INPUT_SCHEMA,
        reference_schema=_TEXT_SCHEMA,
        prediction_schema=_TEXT_SCHEMA,
        sample_prediction="",
        parse_input=None,
        parse_reference=parse_text,
        parse_prediction=parse_text,
        compute=_compute_cer,
        display_format=_PERCENT_2,
        public_summary=_CER_SUMMARY,
    ),
    Scorer(
        name="wer",
        label="Word Error Rate (WER)",
        direction=Direction.LOWER_IS_BETTER,
        parameters=(_LANGUAGE,),
        input_schema=_IGNORED_INPUT_SCHEMA,
        reference_schema=_TEXT_SCHEMA,
        prediction_schema=_TEXT_SCHEMA,
        sample_prediction="",
        parse_input=None,
        parse_reference=parse_text,
        parse_prediction=parse_text,
        compute=_compute_wer,
        display_format=_PERCENT_2,
        public_summary=_WER_SUMMARY,
    ),
    Scorer(
        name="span_f1",
        label="Span IoU F1",
        direction=Direction.HIGHER_IS_BETTER,
        parameters=(_IOU_THRESHOLD,),
        input_schema=_PASSAGE_INPUT_SCHEMA,
        reference_schema=_SPANS_SCHEMA,
        prediction_schema=_SPANS_SCHEMA,
        sample_prediction=[],
        parse_input=parse_text,
        parse_reference=parse_spans,
        parse_prediction=parse_spans,
        compute=_compute_span_f1,
        display_format=_DECIMAL_3,
        public_summary=_SPAN_F1_SUMMARY,
    ),
    Scorer(
        name="toc",
        label="TOC heading score",
        direction=Direction.HIGHER_IS_BETTER,
        parameters=(_TOLERANCE,),
        input_schema=_PASSAGE_INPUT_SCHEMA,
        reference_schema=_GOLD_HEADINGS_SCHEMA,
        prediction_schema=_PREDICTED_HEADINGS_SCHEMA,
        sample_prediction=[],
        parse_input=parse_text,
        parse_reference=parse_gold_headings,
        parse_prediction=parse_predicted_headings,
        compute=_compute_toc,
        display_format=_DECIMAL_3,
        public_summary=(),
    ),
    Scorer(
        name="mcq_accuracy",
        label="Multiple-choice accuracy",
        direction=Direction.HIGHER_IS_BETTER,
        parameters=(),
        input_schema=_IGNORED_INPUT_SCHEMA,
        reference_schema=_TEXT_SCHEMA,
        prediction_schema=_TEXT_SCHEMA,
        sample_prediction="",
        parse_input=None,
        parse_reference=parse_text,
        parse_prediction=parse_text,
        compute=_compute_mcq,
        display_format=_PERCENT_1,
        public_summary=_MCQ_SUMMARY,
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
