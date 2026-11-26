"""Behaviour and parity tests for batch CER and WER."""

import pytest

from buddhist_eval import (
    EmptyInputError,
    ErrorRateBatchScore,
    ErrorRateScore,
    score_cer_batch,
    score_wer_batch,
)

# Fixed Tibetan batch; expected means match EvalAI (per-item evaluate CER/WER
# on pyewts Wylie, then averaged). Values stored as fixtures — no evaluate dep.
_BO_REFS = {
    "identical": "བོད་ཡིག",
    "near": "བོད་སྐད་",
    "delete": "སངས་རྒྱས་ཆོས་",
    "sub": "ཁྱེད་རང་བདེ་པོ་ཡིན་པས།",
}
_BO_PREDS = {
    "identical": "བོད་ཡིག",
    "near": "བོད་སྐད།",
    "delete": "སངས་རྒྱས་",
    "sub": "ཁྱོད་རང་བདེ་པོ་ཡིན།",
}
_BO_CER_MEAN = 0.15745192307692307
_BO_WER_MEAN = 0.3333333333333333


def test_score_cer_batch_mean_and_corpus_on_uniform_english_batch() -> None:
    """Mean and corpus CER agree when every scored item has the same length."""
    references = {"a": "abcd", "b": "efgh"}
    predictions = {"a": "abcd", "b": "efgh"}
    result = score_cer_batch(references, predictions, language="en")
    assert isinstance(result, ErrorRateBatchScore)
    assert result.mean == 0.0
    assert result.corpus == 0.0
    assert result.skipped_ids == ()
    assert result.extra_ids == ()
    assert set(result.items) == {"a", "b"}
    assert result.items["a"] == ErrorRateScore(
        rate=0.0,
        substitutions=0,
        deletions=0,
        insertions=0,
        hits=4,
        reference_length=4,
    )


def test_score_cer_batch_corpus_differs_from_mean_when_lengths_differ() -> None:
    """Corpus CER weights by Reference length; mean does not."""
    # Item a: 1 edit / 2 chars -> rate 0.5
    # Item b: 0 edits / 4 chars -> rate 0.0
    # mean = 0.25; corpus = 1/6
    references = {"a": "ab", "b": "abcd"}
    predictions = {"a": "ax", "b": "abcd"}
    result = score_cer_batch(references, predictions, language="en")
    assert result.mean == 0.25
    assert result.corpus == pytest.approx(1 / 6)
    assert result.items["a"].rate == 0.5
    assert result.items["b"].rate == 0.0


def test_score_cer_batch_missing_prediction_scores_as_empty() -> None:
    """A Reference id with no Prediction is scored as an empty Prediction (rate 1.0)."""
    references = {"a": "abcd"}
    predictions: dict[str, str] = {}
    result = score_cer_batch(references, predictions, language="en")
    assert result.mean == 1.0
    assert result.corpus == 1.0
    assert result.items["a"] == ErrorRateScore(
        rate=1.0,
        substitutions=0,
        deletions=4,
        insertions=0,
        hits=0,
        reference_length=4,
    )


def test_score_cer_batch_empty_reference_is_skipped() -> None:
    """An empty Reference is left out of mean/corpus and listed in skipped_ids."""
    references = {"keep": "ab", "skip": "", "also_skip": "   "}
    predictions = {"keep": "ab", "skip": "x", "also_skip": "y"}
    result = score_cer_batch(references, predictions, language="en")
    assert result.mean == 0.0
    assert result.corpus == 0.0
    assert set(result.skipped_ids) == {"skip", "also_skip"}
    assert set(result.items) == {"keep"}


def test_score_cer_batch_extra_prediction_ids_are_listed() -> None:
    """Prediction ids with no Reference are ignored and listed in extra_ids."""
    references = {"a": "ab"}
    predictions = {"a": "ab", "extra": "zz", "also_extra": "yy"}
    result = score_cer_batch(references, predictions, language="en")
    assert result.mean == 0.0
    assert set(result.extra_ids) == {"extra", "also_extra"}
    assert set(result.items) == {"a"}


def test_score_cer_batch_empty_batch_raises_empty_input_error() -> None:
    """An empty References mapping raises EmptyInputError."""
    with pytest.raises(EmptyInputError):
        score_cer_batch({}, {}, language="en")


def test_score_cer_batch_all_skipped_raises_empty_input_error() -> None:
    """A batch where every Reference is empty raises EmptyInputError."""
    with pytest.raises(EmptyInputError):
        score_cer_batch({"a": "", "b": "  "}, {"a": "x", "b": "y"}, language="en")


def test_score_cer_batch_skips_tibetan_reference_empty_after_wylie() -> None:
    """A tsek-only Tibetan Reference is skipped so corpus never divides by zero."""
    references = {"keep": "བོད་ཡིག", "empty_wylie": "་"}
    predictions = {"keep": "བོད་ཡིག", "empty_wylie": "བོད"}
    result = score_cer_batch(references, predictions, language="bo")
    assert result.mean == 0.0
    assert result.corpus == 0.0
    assert result.skipped_ids == ("empty_wylie",)
    assert set(result.items) == {"keep"}


def test_score_cer_batch_all_empty_after_wylie_raises_empty_input_error() -> None:
    """A batch where every Reference is empty after Wylie raises EmptyInputError."""
    with pytest.raises(EmptyInputError):
        score_cer_batch({"a": "་", "b": "་་"}, {"a": "བོད", "b": "ཡིག"}, language="bo")


def test_score_wer_batch_missing_prediction_scores_as_empty() -> None:
    """A missing Prediction yields WER 1.0 with all Reference words deleted."""
    references = {"a": "one two three"}
    predictions: dict[str, str] = {}
    result = score_wer_batch(references, predictions, language="en")
    assert result.mean == 1.0
    assert result.items["a"] == ErrorRateScore(
        rate=1.0,
        substitutions=0,
        deletions=3,
        insertions=0,
        hits=0,
        reference_length=3,
    )


def test_score_wer_batch_empty_reference_is_skipped() -> None:
    """An empty Reference is left out of mean/corpus and listed in skipped_ids."""
    references = {"keep": "one two", "skip": "", "also_skip": "   "}
    predictions = {"keep": "one two", "skip": "x", "also_skip": "y"}
    result = score_wer_batch(references, predictions, language="en")
    assert result.mean == 0.0
    assert result.corpus == 0.0
    assert set(result.skipped_ids) == {"skip", "also_skip"}
    assert set(result.items) == {"keep"}


def test_score_wer_batch_extra_prediction_ids_are_listed() -> None:
    """Prediction ids with no Reference are ignored and listed in extra_ids."""
    references = {"a": "one two"}
    predictions = {"a": "one two", "extra": "zz", "also_extra": "yy"}
    result = score_wer_batch(references, predictions, language="en")
    assert result.mean == 0.0
    assert set(result.extra_ids) == {"extra", "also_extra"}
    assert set(result.items) == {"a"}


def test_score_wer_batch_empty_batch_raises_empty_input_error() -> None:
    """An empty References mapping raises EmptyInputError."""
    with pytest.raises(EmptyInputError):
        score_wer_batch({}, {}, language="en")


def test_score_cer_batch_parity_mean_on_tibetan_fixture() -> None:
    """Batch CER mean equals EvalAI method (per-item evaluate CER on Wylie, averaged)."""
    result = score_cer_batch(_BO_REFS, _BO_PREDS, language="bo")
    assert result.mean == _BO_CER_MEAN
    assert result.skipped_ids == ()
    assert result.extra_ids == ()
    assert set(result.items) == set(_BO_REFS)


def test_score_wer_batch_parity_mean_on_tibetan_fixture() -> None:
    """Batch WER mean equals EvalAI method (per-item evaluate WER on Wylie, averaged)."""
    result = score_wer_batch(_BO_REFS, _BO_PREDS, language="bo")
    assert result.mean == _BO_WER_MEAN
    assert result.skipped_ids == ()
    assert result.extra_ids == ()
    assert set(result.items) == set(_BO_REFS)
