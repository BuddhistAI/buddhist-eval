"""Tests for score_toc and score_toc_batch (outline heading scoring)."""

from dataclasses import is_dataclass

import pytest

from buddhist_eval import (
    BuddhistEvalError,
    EmptyInputError,
    Heading,
    InvalidHeadingError,
    PredictedHeading,
    TocBatchScore,
    TocPassage,
    TocScore,
    score_toc,
    score_toc_batch,
)

TEXT = "abcdefghij0123456789"
PASSAGE = TocPassage(text=TEXT, headings=(Heading(1, 0), Heading(2, 10)))


def _pred(*pairs: tuple[int, str]) -> tuple[PredictedHeading, ...]:
    return tuple(PredictedHeading(depth, anchor) for depth, anchor in pairs)


def test_result_types_are_frozen_dataclasses() -> None:
    """TocScore and TocBatchScore are frozen dataclasses."""
    for cls in (TocScore, TocBatchScore):
        assert is_dataclass(cls)
        assert cls.__dataclass_params__.frozen  # type: ignore[attr-defined]


def test_gold_as_prediction_scores_one() -> None:
    """Every heading found at the right depth scores 1.0."""
    result = score_toc(PASSAGE, _pred((1, "abc"), (2, "012")))

    assert result.score == 1.0
    assert result.position_f1 == 1.0
    assert result.depth_f1 == 1.0
    assert (result.n_gold, result.n_pred, result.n_located) == (2, 2, 2)
    assert (result.n_position_match, result.n_depth_match) == (2, 2)


def test_right_positions_wrong_depths_score_position_weight_only() -> None:
    """Position F1 1.0 with depth F1 0.0 gives 0.3."""
    result = score_toc(PASSAGE, _pred((2, "abc"), (1, "012")))

    assert result.position_f1 == 1.0
    assert result.depth_f1 == 0.0
    assert result.score == pytest.approx(0.3)


def test_missed_heading_lowers_recall() -> None:
    """One of two headings found: F1 = 2/3 on both components."""
    result = score_toc(PASSAGE, _pred((1, "abc")))

    assert result.position_f1 == pytest.approx(2 / 3)
    assert result.depth_f1 == pytest.approx(2 / 3)
    assert result.score == pytest.approx(2 / 3)


def test_unlocated_anchor_counts_as_prediction_but_never_matches() -> None:
    """An anchor not found verbatim adds to n_pred, lowering precision."""
    result = score_toc(PASSAGE, _pred((1, "abc"), (2, "not in text")))

    assert result.n_pred == 2
    assert result.n_located == 1
    assert result.n_position_match == 1
    assert result.score == pytest.approx(0.5)


def test_anchor_within_tolerance_matches_and_beyond_does_not() -> None:
    """A located start 2 chars off matches; 4 chars off does not (tolerance 3)."""
    passage = TocPassage(text=TEXT, headings=(Heading(1, 0),))

    assert score_toc(passage, _pred((1, "cde"))).score == 1.0
    assert score_toc(passage, _pred((1, "efg"))).score == 0.0
    assert score_toc(passage, _pred((1, "efg")), tolerance=4).score == 1.0


def test_anchors_are_located_after_the_previous_one() -> None:
    """A repeated anchor resolves to its next occurrence, not the first."""
    passage = TocPassage(text="ab---ab---", headings=(Heading(1, 0), Heading(1, 5)))

    result = score_toc(passage, _pred((1, "ab"), (1, "ab")))

    assert result.n_position_match == 2
    assert result.score == 1.0


def test_anchor_whitespace_is_stripped_and_blank_anchor_is_unlocated() -> None:
    """Anchors are stripped before search; an empty anchor is never located."""
    result = score_toc(PASSAGE, _pred((1, "  abc "), (2, "   ")))

    assert result.n_located == 1
    assert result.n_position_match == 1


def test_each_gold_heading_matches_at_most_once() -> None:
    """Two predictions near one Gold Heading yield one match."""
    passage = TocPassage(text=TEXT, headings=(Heading(1, 1),))

    result = score_toc(passage, _pred((1, "abc"), (1, "bcd")))

    assert result.n_position_match == 1
    assert result.position_f1 == pytest.approx(2 / 3)


@pytest.mark.parametrize(
    ("n_pred", "expected"), [(0, 1.0), (1, 0.75), (2, 0.5), (5, 0.0)]
)
def test_passage_without_gold_headings_penalises_each_prediction(
    n_pred: int, expected: float
) -> None:
    """No Gold Headings: 1.0 minus 0.25 per Predicted Heading, floored at 0."""
    passage = TocPassage(text=TEXT, headings=())
    predicted = _pred(*[(1, "abc")] * n_pred)

    assert score_toc(passage, predicted).score == pytest.approx(expected)


@pytest.mark.parametrize(
    "heading",
    [Heading(0, 0), Heading(1, -1), Heading(1, len(TEXT))],
    ids=["depth_zero", "negative_start", "start_past_end"],
)
def test_invalid_gold_heading_raises(heading: Heading) -> None:
    """A Gold Heading with depth < 1 or a start outside the text raises."""
    with pytest.raises(InvalidHeadingError):
        score_toc(TocPassage(text=TEXT, headings=(heading,)), ())


def test_negative_tolerance_raises() -> None:
    """A negative tolerance raises BuddhistEvalError."""
    with pytest.raises(BuddhistEvalError):
        score_toc(PASSAGE, (), tolerance=-1)


def test_batch_mean_weights_passages_equally() -> None:
    """The batch mean is the plain average of per-passage scores."""
    passages = {"p1": PASSAGE, "p2": PASSAGE}
    predictions = {
        "p1": _pred((1, "abc"), (2, "012")),
        "p2": _pred((2, "abc"), (1, "012")),
        "extra": _pred((1, "abc")),
    }

    result = score_toc_batch(passages, predictions)

    assert result.items["p1"].score == 1.0
    assert result.items["p2"].score == pytest.approx(0.3)
    assert result.mean == pytest.approx(0.65)
    assert result.extra_ids == ("extra",)


def test_batch_missing_prediction_is_scored_as_empty() -> None:
    """A passage id with no prediction scores like an empty prediction."""
    passages = {"p1": PASSAGE, "none": TocPassage(text=TEXT, headings=())}

    result = score_toc_batch(passages, {})

    assert result.items["p1"].score == 0.0
    assert result.items["none"].score == 1.0
    assert result.mean == pytest.approx(0.5)


def test_empty_batch_raises() -> None:
    """An empty passages mapping raises EmptyInputError."""
    with pytest.raises(EmptyInputError):
        score_toc_batch({}, {})
