"""Behaviour and parity tests for score_spans_batch (public API only)."""

from dataclasses import is_dataclass

import pytest

from buddhist_eval import (
    Document,
    EmptyInputError,
    InvalidSpanError,
    MisalignedTextsError,
    Span,
    SpanBatchScore,
    TagScore,
    score_spans,
    score_spans_batch,
)

# Multi-document parity fixture: pooled counts = sum of per-doc match_iou,
# then prf (as iou50_metrics). Expected values computed offline.
_PARITY_TEXT_A = "x" * 55
_PARITY_GOLD_A = (
    Span(0, 4, "A"),
    Span(10, 19, "B"),
    Span(30, 39, "A"),
)
_PARITY_PRED_A = (
    Span(0, 4, "A"),
    Span(10, 14, "B"),
    Span(25, 34, "A"),
    Span(50, 54, "C"),
)
_PARITY_TEXT_B = "0123456789"
_PARITY_GOLD_B = (Span(0, 4, "A"), Span(5, 9, "B"))
_PARITY_PRED_B = (Span(0, 4, "A"),)
# doc A: n_match=2, IoUs 1.0+0.5; doc B: n_match=1, IoU 1.0
# pooled: n_gold=5, n_pred=5, n_match=3
_PARITY_BATCH_EXPECTED = {
    "n_gold": 5,
    "n_pred": 5,
    "n_match": 3,
    "precision": 0.6,
    "recall": 0.6,
    "f1": 0.6,
    "mean_iou": 2.5 / 3,
}


def test_empty_batch_raises_empty_input_error() -> None:
    """An empty golds mapping raises EmptyInputError."""
    with pytest.raises(EmptyInputError):
        score_spans_batch({}, {})


def test_span_batch_score_is_frozen_dataclass() -> None:
    """SpanBatchScore is a frozen dataclass."""
    assert is_dataclass(SpanBatchScore)
    assert SpanBatchScore.__dataclass_params__.frozen  # type: ignore[attr-defined]


def test_missing_predicted_document_counts_gold_spans_with_no_matches() -> None:
    """A gold id with no predicted document adds its Gold Spans to n_gold only."""
    text = "0123456789"
    golds = {
        "d1": Document(text=text, spans=(Span(0, 4, "Q"), Span(5, 9, "Q"))),
    }
    predictions: dict[str, Document] = {}

    result = score_spans_batch(golds, predictions)

    assert isinstance(result, SpanBatchScore)
    assert result.n_gold == 2
    assert result.n_pred == 0
    assert result.n_match == 0
    assert result.precision == 0.0
    assert result.recall == 0.0
    assert result.f1 == 0.0
    assert result.mean_iou == 0.0
    assert result.extra_ids == ()


def test_extra_predicted_ids_are_ignored_and_listed() -> None:
    """Predicted document ids with no gold are ignored and listed in extra_ids."""
    text = "0123456789"
    golds = {
        "d1": Document(text=text, spans=(Span(0, 4, "Q"),)),
    }
    predictions = {
        "d1": Document(text=text, spans=(Span(0, 4, "Q"),)),
        "extra": Document(text=text, spans=(Span(0, 9, "Q"),)),
    }

    result = score_spans_batch(golds, predictions)

    assert result.extra_ids == ("extra",)
    assert result.n_gold == 1
    assert result.n_pred == 1
    assert result.n_match == 1
    assert result.f1 == 1.0


def test_pooled_f1_differs_from_average_of_per_document_f1() -> None:
    """Pooled F1 is not the mean of per-document F1 on an uneven fixture."""
    text = "0123456789"
    golds = {
        "perfect": Document(text=text, spans=(Span(0, 4, "Q"),)),
        "missed": Document(text=text, spans=(Span(0, 4, "Q"),)),
    }
    predictions = {
        "perfect": Document(text=text, spans=(Span(0, 4, "Q"),)),
        # "missed" absent -> recall penalty on pooled counts
    }

    result = score_spans_batch(golds, predictions)
    per_doc_f1 = (
        score_spans(golds["perfect"], predictions["perfect"]).f1
        + 0.0  # missed document: no matches, recall 0
    ) / 2

    assert result.n_gold == 2
    assert result.n_pred == 1
    assert result.n_match == 1
    assert result.f1 == pytest.approx(2 / 3)
    assert per_doc_f1 == pytest.approx(0.5)
    assert result.f1 != pytest.approx(per_doc_f1)


def test_misaligned_texts_raise() -> None:
    """Differing gold and predicted texts raise MisalignedTextsError."""
    golds = {"d1": Document(text="abc", spans=())}
    predictions = {"d1": Document(text="abd", spans=())}

    with pytest.raises(MisalignedTextsError):
        score_spans_batch(golds, predictions)


@pytest.mark.parametrize(
    "span",
    [
        Span(-1, 2, "Q"),
        Span(3, 1, "Q"),
        Span(0, 10, "Q"),
    ],
    ids=["negative_start", "inverted", "past_end"],
)
def test_invalid_span_raises(span: Span) -> None:
    """Inverted, negative, or out-of-range Spans raise InvalidSpanError."""
    text = "0123456789"
    golds = {"d1": Document(text=text, spans=(span,))}
    predictions = {"d1": Document(text=text, spans=())}

    with pytest.raises(InvalidSpanError):
        score_spans_batch(golds, predictions)


def test_invalid_span_on_missing_predicted_doc_raises() -> None:
    """Gold Spans are validated even when the predicted document is missing."""
    text = "0123456789"
    golds = {"d1": Document(text=text, spans=(Span(0, 10, "Q"),))}

    with pytest.raises(InvalidSpanError):
        score_spans_batch(golds, {})


@pytest.mark.parametrize("threshold", [0.0, -0.1, 1.1, 50])
def test_out_of_range_threshold_raises(threshold: float) -> None:
    """iou_threshold outside (0, 1] raises InvalidSpanError."""
    text = "abc"
    golds = {"d1": Document(text=text, spans=())}
    predictions = {"d1": Document(text=text, spans=())}

    with pytest.raises(InvalidSpanError):
        score_spans_batch(golds, predictions, iou_threshold=threshold)


def test_parity_pooled_match_iou_and_prf_across_documents() -> None:
    """Pooled scores equal summing match_iou counts then prf across documents."""
    golds = {
        "a": Document(text=_PARITY_TEXT_A, spans=_PARITY_GOLD_A),
        "b": Document(text=_PARITY_TEXT_B, spans=_PARITY_GOLD_B),
    }
    predictions = {
        "a": Document(text=_PARITY_TEXT_A, spans=_PARITY_PRED_A),
        "b": Document(text=_PARITY_TEXT_B, spans=_PARITY_PRED_B),
    }

    result = score_spans_batch(golds, predictions)

    assert result.n_gold == _PARITY_BATCH_EXPECTED["n_gold"]
    assert result.n_pred == _PARITY_BATCH_EXPECTED["n_pred"]
    assert result.n_match == _PARITY_BATCH_EXPECTED["n_match"]
    assert result.precision == _PARITY_BATCH_EXPECTED["precision"]
    assert result.recall == _PARITY_BATCH_EXPECTED["recall"]
    assert result.f1 == _PARITY_BATCH_EXPECTED["f1"]
    assert result.mean_iou == _PARITY_BATCH_EXPECTED["mean_iou"]
    assert result.extra_ids == ()
    # Pooled per-Tag: A gold=3 pred=3 match=2; B gold=2 pred=1 match=1; C gold=0 pred=1 match=0
    assert set(result.per_tag) == {"A", "B", "C"}
    assert isinstance(result.per_tag["A"], TagScore)
    assert result.per_tag["A"].n_gold == 3
    assert result.per_tag["A"].n_pred == 3
    assert result.per_tag["A"].n_match == 2
    assert result.per_tag["A"].f1 == pytest.approx(2 / 3)
    assert result.per_tag["B"].n_gold == 2
    assert result.per_tag["B"].n_pred == 1
    assert result.per_tag["B"].n_match == 1
    assert result.per_tag["B"].f1 == pytest.approx(2 / 3)
    assert result.per_tag["C"].n_gold == 0
    assert result.per_tag["C"].n_pred == 1
    assert result.per_tag["C"].n_match == 0
