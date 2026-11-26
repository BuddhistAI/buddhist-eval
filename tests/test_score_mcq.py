"""Tests for score_mcq (multiple-choice accuracy)."""

from dataclasses import is_dataclass

import pytest

from buddhist_eval import EmptyInputError, McqScore, score_mcq


def test_result_type_is_frozen_dataclass() -> None:
    """McqScore is a frozen dataclass."""
    assert is_dataclass(McqScore)
    assert McqScore.__dataclass_params__.frozen  # type: ignore[attr-defined]


def test_89_of_100_correct_is_89_percent() -> None:
    """Accuracy is correct answers over total questions."""
    answer_key = {f"q{i}": "A" for i in range(100)}
    predictions = {f"q{i}": "A" if i < 89 else "B" for i in range(100)}

    result = score_mcq(answer_key, predictions)

    assert result.accuracy == pytest.approx(0.89)
    assert (result.n_correct, result.n_total) == (89, 100)
    assert result.items["q0"] is True
    assert result.items["q99"] is False


def test_comparison_ignores_case_and_surrounding_whitespace() -> None:
    """' b ' matches 'B'."""
    result = score_mcq({"q1": "B"}, {"q1": " b "})

    assert result.accuracy == 1.0


def test_missing_prediction_counts_as_wrong() -> None:
    """A question with no prediction is wrong and listed in missing_ids."""
    result = score_mcq({"q1": "A", "q2": "C"}, {"q1": "A"})

    assert result.accuracy == 0.5
    assert result.items["q2"] is False
    assert result.missing_ids == ("q2",)


def test_extra_predictions_are_ignored_and_listed() -> None:
    """Predictions with no question do not affect accuracy."""
    result = score_mcq({"q1": "A"}, {"q1": "A", "q9": "D"})

    assert result.accuracy == 1.0
    assert result.n_total == 1
    assert result.extra_ids == ("q9",)


def test_empty_answer_key_raises() -> None:
    """An empty Answer Key raises EmptyInputError."""
    with pytest.raises(EmptyInputError):
        score_mcq({}, {"q1": "A"})


def test_blank_answer_in_key_raises() -> None:
    """A blank correct answer raises EmptyInputError."""
    with pytest.raises(EmptyInputError):
        score_mcq({"q1": "  "}, {"q1": ""})
