"""Behaviour and parity tests for single-pair CER and WER."""

import pytest

from buddhist_eval import EmptyInputError, ErrorRateScore, score_cer, score_wer

# Fixed Tibetan pairs; expected rates/counts computed once offline via
# pyewts Wylie + jiwer (same path as Hugging Face evaluate cer/wer).
_IDENTICAL_BO_REF = "བོད་ཡིག"
_IDENTICAL_BO_PRED = "བོད་ཡིག"
_NEAR_BO_REF = "བོད་སྐད་"
_NEAR_BO_PRED = "བོད་སྐད།"
_DELETE_BO_REF = "སངས་རྒྱས་ཆོས་"
_DELETE_BO_PRED = "སངས་རྒྱས་"
_SUB_BO_REF = "ཁྱེད་རང་བདེ་པོ་ཡིན་པས།"
_SUB_BO_PRED = "ཁྱོད་རང་བདེ་པོ་ཡིན།"


def test_score_cer_empty_reference_raises_empty_input_error() -> None:
    """An empty Reference after stripping raises EmptyInputError."""
    with pytest.raises(EmptyInputError):
        score_cer("", "anything", language="en")


def test_score_cer_whitespace_only_reference_raises_empty_input_error() -> None:
    """A whitespace-only Reference is empty after stripping and raises."""
    with pytest.raises(EmptyInputError):
        score_cer("   \n\t  ", "anything", language="en")


def test_score_cer_empty_prediction_is_all_deletions() -> None:
    """An empty Prediction scores CER 1.0 with every Reference character deleted."""
    result = score_cer("abcd", "", language="en")
    assert result == ErrorRateScore(
        rate=1.0,
        substitutions=0,
        deletions=4,
        insertions=0,
        hits=0,
        reference_length=4,
    )


def test_score_wer_empty_reference_raises_empty_input_error() -> None:
    """An empty Reference after stripping raises EmptyInputError for WER."""
    with pytest.raises(EmptyInputError):
        score_wer("", "anything", language="en")


def test_score_wer_empty_prediction_is_all_deletions() -> None:
    """An empty Prediction scores WER 1.0 with every Reference word deleted."""
    result = score_wer("one two three", "", language="en")
    assert result == ErrorRateScore(
        rate=1.0,
        substitutions=0,
        deletions=3,
        insertions=0,
        hits=0,
        reference_length=3,
    )


def test_non_tibetan_language_passes_through_unchanged() -> None:
    """A non-Tibetan language scores the stripped text without Wylie conversion."""
    result = score_cer("café", "café", language="fr")
    assert result.rate == 0.0
    assert result.hits == 4
    assert result.reference_length == 4


def test_single_wrong_syllable_gives_expected_wer_edit_counts() -> None:
    """One substituted syllable yields one substitution and the expected WER."""
    # Wylie of བོད་ཡིག is "bod yig"; wrong second syllable -> one substitution.
    result = score_wer("བོད་ཡིག", "བོད་སྐད", language="bo")
    assert result == ErrorRateScore(
        rate=0.5,
        substitutions=1,
        deletions=0,
        insertions=0,
        hits=1,
        reference_length=2,
    )


def test_score_cer_strips_leading_and_trailing_whitespace() -> None:
    """Leading and trailing whitespace are stripped before scoring."""
    result = score_cer("  abcd  ", "  abcd  ", language="en")
    assert result.rate == 0.0
    assert result.reference_length == 4


@pytest.mark.parametrize(
    ("reference", "prediction", "expected"),
    [
        (
            _IDENTICAL_BO_REF,
            _IDENTICAL_BO_PRED,
            ErrorRateScore(
                rate=0.0,
                substitutions=0,
                deletions=0,
                insertions=0,
                hits=7,
                reference_length=7,
            ),
        ),
        (
            _NEAR_BO_REF,
            _NEAR_BO_PRED,
            ErrorRateScore(
                rate=0.125,
                substitutions=0,
                deletions=0,
                insertions=1,
                hits=8,
                reference_length=8,
            ),
        ),
        (
            _DELETE_BO_REF,
            _DELETE_BO_PRED,
            ErrorRateScore(
                rate=0.3125,
                substitutions=0,
                deletions=5,
                insertions=0,
                hits=11,
                reference_length=16,
            ),
        ),
        (
            _SUB_BO_REF,
            _SUB_BO_PRED,
            ErrorRateScore(
                rate=0.19230769230769232,
                substitutions=1,
                deletions=4,
                insertions=0,
                hits=21,
                reference_length=26,
            ),
        ),
    ],
)
def test_score_cer_parity_with_evaluate_on_wylie(
    reference: str, prediction: str, expected: ErrorRateScore
) -> None:
    """Tibetan CER matches evaluate cer on pyewts Wylie (fixture values)."""
    assert score_cer(reference, prediction, language="bo") == expected


@pytest.mark.parametrize(
    ("reference", "prediction", "expected"),
    [
        (
            _IDENTICAL_BO_REF,
            _IDENTICAL_BO_PRED,
            ErrorRateScore(
                rate=0.0,
                substitutions=0,
                deletions=0,
                insertions=0,
                hits=2,
                reference_length=2,
            ),
        ),
        (
            _NEAR_BO_REF,
            _NEAR_BO_PRED,
            ErrorRateScore(
                rate=0.5,
                substitutions=1,
                deletions=0,
                insertions=0,
                hits=1,
                reference_length=2,
            ),
        ),
        (
            _DELETE_BO_REF,
            _DELETE_BO_PRED,
            ErrorRateScore(
                rate=1.0 / 3.0,
                substitutions=0,
                deletions=1,
                insertions=0,
                hits=2,
                reference_length=3,
            ),
        ),
        (
            _SUB_BO_REF,
            _SUB_BO_PRED,
            ErrorRateScore(
                rate=0.5,
                substitutions=2,
                deletions=1,
                insertions=0,
                hits=3,
                reference_length=6,
            ),
        ),
    ],
)
def test_score_wer_parity_with_evaluate_on_wylie(
    reference: str, prediction: str, expected: ErrorRateScore
) -> None:
    """Tibetan WER matches evaluate wer on pyewts Wylie (fixture values)."""
    assert score_wer(reference, prediction, language="bo") == expected
