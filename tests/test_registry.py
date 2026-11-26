"""Tests for the scorer registry (get_scorer, list_scorers, Scorer.score)."""

import json

import pytest

from buddhist_eval import (
    BuddhistEvalError,
    Heading,
    InvalidParameterError,
    InvalidPredictionError,
    InvalidReferenceError,
    PredictedHeading,
    SortDirection,
    TocPassage,
    UnknownScorerError,
    get_scorer,
    list_scorers,
    score_cer_batch,
    score_toc_batch,
)


def test_registry_lists_every_scorer_once() -> None:
    """Every metric is registered under a unique name."""
    names = [scorer.name for scorer in list_scorers()]

    assert names == ["cer", "wer", "span_f1", "toc", "mcq_accuracy"]


@pytest.mark.parametrize(
    ("name", "direction"),
    [
        ("cer", SortDirection.LOWER_IS_BETTER),
        ("wer", SortDirection.LOWER_IS_BETTER),
        ("span_f1", SortDirection.HIGHER_IS_BETTER),
        ("toc", SortDirection.HIGHER_IS_BETTER),
        ("mcq_accuracy", SortDirection.HIGHER_IS_BETTER),
    ],
)
def test_direction_per_scorer(name: str, direction: SortDirection) -> None:
    """Error rates sort ascending; F1, TOC and accuracy sort descending."""
    assert get_scorer(name).direction is direction


def test_direction_values_match_challenge_json() -> None:
    """Direction values are the strings stored in challenge.json."""
    assert get_scorer("cer").direction.value == "lower_is_better"
    assert get_scorer("mcq_accuracy").direction.value == "higher_is_better"


def test_unknown_scorer_raises() -> None:
    """An unregistered name raises UnknownScorerError, a BuddhistEvalError."""
    with pytest.raises(UnknownScorerError):
        get_scorer("bleu")
    assert issubclass(UnknownScorerError, BuddhistEvalError)


def test_describe_is_json_serialisable() -> None:
    """describe() gives a listing endpoint everything it needs as plain JSON."""
    described = json.loads(json.dumps([s.describe() for s in list_scorers()]))

    cer = described[0]
    assert cer["name"] == "cer"
    assert cer["label"] == "Character Error Rate (CER)"
    assert cer["direction"] == "lower_is_better"
    assert set(cer["parameter_schema"]["properties"]) == {"language", "aggregate"}
    assert cer["prediction_schema"] == {"type": "string"}


def test_parameter_schema_lists_defaults_and_bounds() -> None:
    """Parameter schemas carry type, default and bounds."""
    schema = get_scorer("span_f1").parameter_schema

    iou = schema["properties"]["iou_threshold"]
    assert schema["additionalProperties"] is False
    assert (iou["type"], iou["default"]) == ("number", 0.5)
    assert (iou["exclusiveMinimum"], iou["maximum"]) == (0.0, 1.0)
    assert get_scorer("cer").parameter_schema["properties"]["aggregate"]["enum"] == [
        "mean",
        "corpus",
    ]


def test_cer_matches_batch_mean_and_corpus() -> None:
    """CER returns the batch mean by default and the corpus rate on request."""
    references = {"a": "abcd", "b": "ef gh"}
    predictions = {"a": "abxd"}
    batch = score_cer_batch(references, predictions, language="en")
    cer = get_scorer("cer")

    assert cer.score(predictions, references, language="en") == batch.mean
    assert (
        cer.score(predictions, references, language="en", aggregate="corpus")
        == batch.corpus
    )


def test_wer_defaults_to_tibetan() -> None:
    """With no params, text is scored as Tibetan on Wylie syllables."""
    assert get_scorer("wer").score({"x": "བོད་སྐད"}, {"x": "བོད་ཡིག"}) == 0.5


def test_span_f1_scores_json_spans_against_gold_documents() -> None:
    """Predictions are span lists; the gold document supplies the text."""
    references = {
        "d1": {"text": "0123456789", "spans": [{"start": 0, "end": 4, "tag": "quote"}]}
    }
    predictions = {
        "d1": [
            {"start": 0, "end": 4, "tag": "quote"},
            {"start": 6, "end": 8, "tag": "quote"},
        ]
    }

    f1 = get_scorer("span_f1").score(predictions, references, iou_threshold=1)

    assert f1 == pytest.approx(2 / 3)


def test_toc_matches_batch_mean() -> None:
    """TOC returns score_toc_batch's mean."""
    references = {
        "p": {
            "text": "abcdefghij0123456789",
            "headings": [{"depth": 1, "start": 0}, {"depth": 2, "start": 10}],
        }
    }
    predictions = {"p": [{"depth": 1, "anchor": "abc"}, {"depth": 1, "anchor": "012"}]}
    passage = TocPassage(
        text="abcdefghij0123456789", headings=(Heading(1, 0), Heading(2, 10))
    )
    expected = score_toc_batch(
        {"p": passage}, {"p": (PredictedHeading(1, "abc"), PredictedHeading(1, "012"))}
    )

    assert get_scorer("toc").score(predictions, references) == expected.mean


def test_mcq_accuracy() -> None:
    """MCQ returns accuracy over the Answer Key."""
    score = get_scorer("mcq_accuracy").score(
        {"q1": "a", "q2": "D"}, {"q1": "A", "q2": "C"}
    )

    assert score == 0.5


def test_unknown_parameter_raises() -> None:
    """A parameter the scorer does not declare is rejected."""
    with pytest.raises(InvalidParameterError):
        get_scorer("mcq_accuracy").score({"q1": "A"}, {"q1": "A"}, case_sensitive=True)


@pytest.mark.parametrize(
    ("name", "params"),
    [
        ("cer", {"aggregate": "median"}),
        ("cer", {"language": 7}),
        ("span_f1", {"iou_threshold": 0}),
        ("span_f1", {"iou_threshold": 1.5}),
        ("span_f1", {"iou_threshold": True}),
        ("toc", {"tolerance": -1}),
        ("toc", {"tolerance": 2.5}),
    ],
)
def test_invalid_parameter_values_raise(name: str, params: dict[str, object]) -> None:
    """Wrong types, values outside choices and out-of-range values are rejected."""
    with pytest.raises(InvalidParameterError):
        get_scorer(name).validate_params(params)


def test_validate_params_fills_defaults() -> None:
    """validate_params returns every parameter, defaulted where omitted."""
    assert get_scorer("cer").validate_params({}) == {
        "language": "bo",
        "aggregate": "mean",
    }


def test_malformed_prediction_raises_invalid_prediction() -> None:
    """A prediction that breaks the prediction schema raises InvalidPredictionError."""
    references = {"d1": {"text": "0123", "spans": []}}

    with pytest.raises(InvalidPredictionError, match="d1"):
        get_scorer("span_f1").score({"d1": [{"start": 0, "tag": "q"}]}, references)
    with pytest.raises(InvalidPredictionError):
        get_scorer("cer").score({"a": 3}, {"a": "abc"}, language="en")


def test_malformed_reference_raises_invalid_reference() -> None:
    """A reference that breaks the reference schema raises InvalidReferenceError."""
    with pytest.raises(InvalidReferenceError):
        get_scorer("toc").score({}, {"p": {"text": "abc"}})
