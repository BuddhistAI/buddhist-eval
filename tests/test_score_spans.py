"""Behaviour and parity tests for score_spans (public API only)."""

from dataclasses import is_dataclass

import pytest

from buddhist_eval import (
    BuddhistEvalError,
    Document,
    InvalidSpanError,
    MisalignedTextsError,
    Span,
    SpanScore,
    TagScore,
    score_spans,
)

# Fixed expected values from quotation-detection match_iou + prf on the spans
# below (same-Tag filter, inclusive IoU, greedy sort). Computed once offline.
_PARITY_TEXT = "x" * 55
_PARITY_GOLD = (
    Span(0, 4, "A"),
    Span(10, 19, "B"),
    Span(30, 39, "A"),
)
_PARITY_PRED = (
    Span(0, 4, "A"),
    Span(10, 14, "B"),
    Span(25, 34, "A"),
    Span(50, 54, "C"),
)
_PARITY_EXPECTED = {
    "n_match": 2,
    "precision": 0.5,
    "recall": 2 / 3,
    "f1": 0.5714285714285715,
    "mean_iou": 0.75,
}


def test_public_span_types_are_frozen_dataclasses() -> None:
    """Document, Span, SpanScore and TagScore are frozen dataclasses."""
    for cls in (Document, Span, SpanScore, TagScore):
        assert is_dataclass(cls)
        assert cls.__dataclass_params__.frozen  # type: ignore[attr-defined]



def test_score_spans_no_spans_on_either_side_gives_all_zeros() -> None:
    """No Gold or Predicted Spans yields precision, recall and F1 of 0.0."""
    text = "abcdefghij"
    gold = Document(text=text, spans=())
    predicted = Document(text=text, spans=())

    result = score_spans(gold, predicted)

    assert isinstance(result, SpanScore)
    assert result.precision == 0.0
    assert result.recall == 0.0
    assert result.f1 == 0.0
    assert result.n_gold == 0
    assert result.n_pred == 0
    assert result.n_match == 0
    assert result.mean_iou == 0.0
    assert result.iou_threshold == 0.5
    assert result.per_tag == {}


def test_inclusive_end_iou_treats_zero_four_as_five_chars() -> None:
    """Inclusive ends: (0, 4) covers five characters and exact overlap is IoU 1.0."""
    text = "0123456789"
    gold = Document(text=text, spans=(Span(0, 4, "Q"),))
    predicted = Document(text=text, spans=(Span(0, 4, "Q"),))

    result = score_spans(gold, predicted)

    assert result.n_match == 1
    assert result.mean_iou == 1.0
    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.f1 == 1.0


def test_tag_mismatch_never_matches() -> None:
    """Same offsets with different Tags do not form a Match."""
    text = "0123456789"
    gold = Document(text=text, spans=(Span(0, 4, "A"),))
    predicted = Document(text=text, spans=(Span(0, 4, "B"),))

    result = score_spans(gold, predicted)

    assert result.n_match == 0
    assert result.precision == 0.0
    assert result.recall == 0.0
    assert result.f1 == 0.0


def test_one_to_one_when_two_preds_compete_for_one_gold() -> None:
    """Each Gold Span is used in at most one Match under contention."""
    text = "0123456789ABCDEF"
    gold = Document(text=text, spans=(Span(0, 9, "Q"),))
    predicted = Document(
        text=text,
        spans=(Span(0, 9, "Q"), Span(0, 4, "Q")),
    )

    result = score_spans(gold, predicted)

    assert result.n_match == 1
    assert result.n_gold == 1
    assert result.n_pred == 2
    assert result.precision == 0.5
    assert result.recall == 1.0


def test_tie_break_prefers_higher_gold_then_pred_index() -> None:
    """Equal IoU ties break by descending (gold index, predicted index)."""
    # p0 has IoU 1/3 with both golds; p1 has IoU 0.3 with gold0 only.
    # Descending sort accepts (g1, p0) then (g0, p1) -> 2 Matches.
    # Preferring the lower gold index would accept (g0, p0) and leave only 1.
    text = "0123456789ABCDEFGHIJ"
    gold = Document(
        text=text,
        spans=(Span(0, 9, "Q"), Span(10, 19, "Q")),
    )
    predicted = Document(
        text=text,
        spans=(Span(5, 14, "Q"), Span(0, 2, "Q")),
    )

    result = score_spans(gold, predicted, iou_threshold=0.3)

    assert result.n_match == 2
    assert result.n_gold == 2
    assert result.n_pred == 2


def test_per_tag_holds_score_for_each_tag_seen() -> None:
    """per_tag includes every Tag present on either side."""
    text = "0123456789ABCDEFGHIJ"
    gold = Document(
        text=text,
        spans=(Span(0, 4, "A"), Span(10, 14, "B")),
    )
    predicted = Document(
        text=text,
        spans=(Span(0, 4, "A"), Span(10, 14, "C")),
    )

    result = score_spans(gold, predicted)

    assert set(result.per_tag) == {"A", "B", "C"}
    assert isinstance(result.per_tag["A"], TagScore)
    assert result.per_tag["A"].n_match == 1
    assert result.per_tag["A"].precision == 1.0
    assert result.per_tag["B"].n_gold == 1
    assert result.per_tag["B"].n_pred == 0
    assert result.per_tag["B"].recall == 0.0
    assert result.per_tag["C"].n_gold == 0
    assert result.per_tag["C"].n_pred == 1
    assert result.per_tag["C"].precision == 0.0


def test_mean_iou_is_zero_when_there_are_no_matches() -> None:
    """Mean IoU is 0.0 when no Matches are accepted."""
    text = "0123456789"
    gold = Document(text=text, spans=(Span(0, 2, "Q"),))
    predicted = Document(text=text, spans=(Span(7, 9, "Q"),))

    result = score_spans(gold, predicted)

    assert result.n_match == 0
    assert result.mean_iou == 0.0


def test_misaligned_texts_raise() -> None:
    """Differing gold and predicted texts raise MisalignedTextsError."""
    gold = Document(text="abc", spans=())
    predicted = Document(text="abd", spans=())

    with pytest.raises(MisalignedTextsError):
        score_spans(gold, predicted)


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
    gold = Document(text=text, spans=(span,))
    predicted = Document(text=text, spans=())

    with pytest.raises(InvalidSpanError):
        score_spans(gold, predicted)


@pytest.mark.parametrize("threshold", [0.0, -0.1, 1.1, 50])
def test_out_of_range_threshold_raises(threshold: float) -> None:
    """iou_threshold outside (0, 1] raises BuddhistEvalError (not InvalidSpanError)."""
    text = "abc"
    gold = Document(text=text, spans=())
    predicted = Document(text=text, spans=())

    with pytest.raises(BuddhistEvalError) as exc_info:
        score_spans(gold, predicted, iou_threshold=threshold)
    assert type(exc_info.value) is BuddhistEvalError


def test_parity_with_match_iou_and_prf_fixtures() -> None:
    """precision, recall, F1 and n_match match stored match_iou+prf fixtures."""
    gold = Document(text=_PARITY_TEXT, spans=_PARITY_GOLD)
    predicted = Document(text=_PARITY_TEXT, spans=_PARITY_PRED)

    result = score_spans(gold, predicted)

    assert result.n_match == _PARITY_EXPECTED["n_match"]
    assert result.precision == _PARITY_EXPECTED["precision"]
    assert result.recall == _PARITY_EXPECTED["recall"]
    assert result.f1 == _PARITY_EXPECTED["f1"]
    assert result.mean_iou == _PARITY_EXPECTED["mean_iou"]
    assert result.n_gold == 3
    assert result.n_pred == 4
