"""Tests for the scorer registry: list_scorers, get_scorer and Scorer.score."""

import json

import pytest

from buddhist_eval import (
    BuddhistEvalError,
    Direction,
    Document,
    Heading,
    InvalidInputError,
    InvalidParameterError,
    InvalidPredictionError,
    InvalidReferenceError,
    InvalidSpanError,
    PredictedHeading,
    ScoreResult,
    Span,
    TocPassage,
    UnknownScorerError,
    get_scorer,
    list_scorers,
    score_cer_batch,
    score_mcq,
    score_spans_batch,
    score_toc_batch,
    score_wer_batch,
)

PASSAGE = "abcdefghij0123456789"


def test_registry_lists_every_scorer_once() -> None:
    """Every metric is registered under a unique name, in a stable order."""
    names = [scorer.name for scorer in list_scorers()]

    assert names == ["cer", "wer", "span_f1", "toc", "mcq_accuracy"]


@pytest.mark.parametrize(
    ("name", "direction"),
    [
        ("cer", Direction.LOWER_IS_BETTER),
        ("wer", Direction.LOWER_IS_BETTER),
        ("span_f1", Direction.HIGHER_IS_BETTER),
        ("toc", Direction.HIGHER_IS_BETTER),
        ("mcq_accuracy", Direction.HIGHER_IS_BETTER),
    ],
)
def test_direction_per_scorer(name: str, direction: Direction) -> None:
    """Error rates rank ascending; F1, TOC and accuracy rank descending."""
    assert get_scorer(name).direction is direction


def test_direction_values_match_challenge_json() -> None:
    """Direction values are the strings stored in challenge.json."""
    assert get_scorer("cer").direction.value == "lower_is_better"
    assert get_scorer("mcq_accuracy").direction.value == "higher_is_better"


def test_unknown_scorer_raises() -> None:
    """An unregistered name raises UnknownScorerError, a BuddhistEvalError."""
    with pytest.raises(UnknownScorerError, match="bleu"):
        get_scorer("bleu")
    assert issubclass(UnknownScorerError, BuddhistEvalError)


def test_describe_is_json_serialisable() -> None:
    """describe() gives a listing endpoint everything it needs as plain JSON."""
    described = json.loads(json.dumps([s.describe() for s in list_scorers()]))

    cer = described[0]
    assert cer["name"] == "cer"
    assert cer["label"] == "Character Error Rate (CER)"
    assert cer["direction"] == "lower_is_better"
    assert cer["parameter_schema"]["properties"]["language"]["default"] == "bo"
    assert cer["reference_schema"] == {"type": "string"}
    assert cer["prediction_schema"] == {"type": "string"}
    assert cer["sample_prediction"] == ""
    assert described[2]["input_schema"]["type"] == "string"


def test_describe_returns_copies() -> None:
    """Mutating a description does not change the registered Scorer."""
    described = get_scorer("span_f1").describe()
    described["sample_prediction"].append("x")
    described["reference_schema"]["items"]["required"].append("x")

    fresh = get_scorer("span_f1").describe()
    assert fresh["sample_prediction"] == []
    assert fresh["reference_schema"]["items"]["required"] == ["start", "end", "tag"]


@pytest.mark.parametrize(
    ("name", "defaults"),
    [
        ("cer", {"language": "bo"}),
        ("wer", {"language": "bo"}),
        ("span_f1", {"iou_threshold": 0.5}),
        ("toc", {"tolerance": 3}),
        ("mcq_accuracy", {}),
    ],
)
def test_parameter_defaults(name: str, defaults: dict[str, object]) -> None:
    """Each Scorer declares its parameters with defaults; omitted ones are filled."""
    scorer = get_scorer(name)

    assert scorer.validate_params({}) == defaults
    assert {
        key: value["default"]
        for key, value in scorer.parameter_schema["properties"].items()
    } == defaults


def test_parameter_schema_lists_bounds() -> None:
    """Parameter schemas carry type, default and bounds, and forbid extras."""
    schema = get_scorer("span_f1").parameter_schema

    iou = schema["properties"]["iou_threshold"]
    assert schema["additionalProperties"] is False
    assert (iou["type"], iou["default"]) == ("number", 0.5)
    assert (iou["exclusiveMinimum"], iou["maximum"]) == (0.0, 1.0)


def test_cer_headline_is_batch_mean_with_corpus_in_details() -> None:
    """CER ranks by the batch mean; corpus and per-item rates are details."""
    references = {"a": "abcd", "b": "ef gh"}
    predictions = {"a": "abxd"}
    batch = score_cer_batch(references, predictions, language="en")

    result = get_scorer("cer").score({}, references, predictions, language="en")

    assert isinstance(result, ScoreResult)
    assert result.headline_metric == batch.mean
    assert result.details["corpus"] == batch.corpus
    assert result.details["items"]["a"]["rate"] == batch.items["a"].rate
    assert result.details["items"]["b"]["deletions"] == 5


def test_wer_headline_is_batch_mean_and_defaults_to_tibetan() -> None:
    """With no params, WER scores Tibetan on Wylie syllables."""
    references = {"x": "བོད་ཡིག", "y": "ཆོས་ཀྱི་འཁོར་ལོ"}
    predictions = {"x": "བོད་སྐད", "y": "ཆོས་ཀྱི་འཁོར་ལོ"}
    batch = score_wer_batch(references, predictions, language="bo")

    result = get_scorer("wer").score({}, references, predictions)

    assert result.headline_metric == batch.mean == 0.25
    assert result.details["corpus"] == batch.corpus


def test_cer_ignores_inputs() -> None:
    """CER does not read Inputs, so any value or none at all is accepted."""
    cer = get_scorer("cer")

    assert not cer.uses_inputs
    with_inputs = cer.score(
        {"a": {"image": 1}}, {"a": "ab"}, {"a": "ab"}, language="en"
    )
    without = cer.score({}, {"a": "ab"}, {"a": "ab"}, language="en")
    assert with_inputs == without


def test_span_f1_takes_passage_text_from_inputs() -> None:
    """References and Predictions are span lists; the Input is the passage text."""
    inputs = {"d1": "0123456789", "d2": "abcdef"}
    references = {
        "d1": [{"start": 0, "end": 4, "tag": "quote"}],
        "d2": [{"start": 1, "end": 3, "tag": "name"}],
    }
    predictions = {
        "d1": [
            {"start": 0, "end": 4, "tag": "quote"},
            {"start": 6, "end": 8, "tag": "quote"},
        ],
        "d2": [{"start": 1, "end": 2, "tag": "name"}],
    }
    expected = score_spans_batch(
        {
            "d1": Document("0123456789", (Span(0, 4, "quote"),)),
            "d2": Document("abcdef", (Span(1, 3, "name"),)),
        },
        {
            "d1": Document("0123456789", (Span(0, 4, "quote"), Span(6, 8, "quote"))),
            "d2": Document("abcdef", (Span(1, 2, "name"),)),
        },
        iou_threshold=0.5,
    )

    result = get_scorer("span_f1").score(inputs, references, predictions)

    assert result.headline_metric == expected.f1
    assert result.details["per_tag"]["name"]["n_match"] == 1
    assert result.details["n_pred"] == 3


def test_span_f1_reports_extra_prediction_ids() -> None:
    """A prediction for an id with no Reference is listed, not scored."""
    result = get_scorer("span_f1").score(
        {"d1": "0123"},
        {"d1": [{"start": 0, "end": 1, "tag": "q"}]},
        {"d1": [{"start": 0, "end": 1, "tag": "q"}], "zz": []},
    )

    assert result.headline_metric == 1.0
    assert result.details["extra_ids"] == ["zz"]


def test_span_outside_passage_raises() -> None:
    """A span past the end of the Input text is rejected by the metric."""
    with pytest.raises(InvalidSpanError):
        get_scorer("span_f1").score(
            {"d1": "0123"}, {"d1": [{"start": 0, "end": 9, "tag": "q"}]}, {}
        )


def test_toc_takes_passage_text_from_inputs() -> None:
    """TOC References are Gold Headings; the Input is the passage text."""
    references = {"p": [{"depth": 1, "start": 0}, {"depth": 2, "start": 10}]}
    predictions = {"p": [{"depth": 1, "anchor": "abc"}, {"depth": 1, "anchor": "012"}]}
    expected = score_toc_batch(
        {"p": TocPassage(text=PASSAGE, headings=(Heading(1, 0), Heading(2, 10)))},
        {"p": (PredictedHeading(1, "abc"), PredictedHeading(1, "012"))},
    )

    result = get_scorer("toc").score({"p": PASSAGE}, references, predictions)

    assert result.headline_metric == expected.mean
    assert result.details["items"]["p"]["n_depth_match"] == 1


def test_toc_tolerance_is_passed_through() -> None:
    """tolerance changes which located anchors match a Gold Heading."""
    references = {"p": [{"depth": 1, "start": 5}]}
    predictions = {"p": [{"depth": 1, "anchor": "a"}]}
    toc = get_scorer("toc")

    strict = toc.score({"p": PASSAGE}, references, predictions, tolerance=0)
    loose = toc.score({"p": PASSAGE}, references, predictions, tolerance=5)

    assert strict.headline_metric < loose.headline_metric == 1.0


def test_mcq_headline_is_accuracy() -> None:
    """MCQ ranks by accuracy over the Answer Key and ignores Inputs."""
    references = {"q1": "A", "q2": "C", "q3": "B"}
    predictions = {"q1": "a", "q2": "D", "q9": "A"}
    expected = score_mcq(references, predictions)

    result = get_scorer("mcq_accuracy").score({}, references, predictions)

    assert result.headline_metric == expected.accuracy == 1 / 3
    assert result.details["items"] == {"q1": True, "q2": False, "q3": False}
    assert result.details["missing_ids"] == ["q3"]
    assert result.details["extra_ids"] == ["q9"]


@pytest.mark.parametrize(
    ("name", "inputs", "references", "predictions"),
    [
        ("cer", {}, {"a": "abc"}, {"a": "abd"}),
        ("wer", {}, {"a": "བོད་ཡིག"}, {"a": "བོད་སྐད"}),
        (
            "span_f1",
            {"d": "0123"},
            {"d": [{"start": 0, "end": 1, "tag": "q"}]},
            {"d": [{"start": 0, "end": 1, "tag": "q"}]},
        ),
        (
            "toc",
            {"p": PASSAGE},
            {"p": [{"depth": 1, "start": 0}]},
            {"p": [{"depth": 1, "anchor": "abc"}]},
        ),
        ("mcq_accuracy", {}, {"q": "A"}, {"q": "B"}),
    ],
)
def test_result_round_trips_through_json(
    name: str,
    inputs: dict[str, object],
    references: dict[str, object],
    predictions: dict[str, object],
) -> None:
    """Every Scorer's result serialises to JSON unchanged."""
    result = get_scorer(name).score(inputs, references, predictions)
    payload = result.to_dict()

    assert json.loads(json.dumps(payload)) == payload
    assert isinstance(payload["headline_metric"], float)


def test_unknown_parameter_raises() -> None:
    """A parameter the Scorer does not declare is rejected."""
    with pytest.raises(InvalidParameterError):
        get_scorer("mcq_accuracy").score(
            {}, {"q1": "A"}, {"q1": "A"}, case_sensitive=True
        )


def test_aggregate_is_not_a_parameter() -> None:
    """CER and WER always rank by the mean; the corpus rate is only a detail."""
    with pytest.raises(InvalidParameterError):
        get_scorer("cer").validate_params({"aggregate": "corpus"})


@pytest.mark.parametrize(
    ("name", "params"),
    [
        ("cer", {"language": 7}),
        ("span_f1", {"iou_threshold": 0}),
        ("span_f1", {"iou_threshold": 1.5}),
        ("span_f1", {"iou_threshold": True}),
        ("toc", {"tolerance": -1}),
        ("toc", {"tolerance": 2.5}),
    ],
)
def test_invalid_parameter_values_raise(name: str, params: dict[str, object]) -> None:
    """Wrong types and out-of-range values are rejected."""
    with pytest.raises(InvalidParameterError):
        get_scorer(name).validate_params(params)


def test_int_is_accepted_for_float_parameter() -> None:
    """An integer iou_threshold is widened to float."""
    assert get_scorer("span_f1").validate_params({"iou_threshold": 1}) == {
        "iou_threshold": 1.0
    }


def test_malformed_prediction_raises_invalid_prediction() -> None:
    """A prediction that breaks the prediction schema raises InvalidPredictionError."""
    with pytest.raises(InvalidPredictionError, match="d1"):
        get_scorer("span_f1").score(
            {"d1": "0123"}, {"d1": []}, {"d1": [{"start": 0, "tag": "q"}]}
        )
    with pytest.raises(InvalidPredictionError):
        get_scorer("cer").score({}, {"a": "abc"}, {"a": 3}, language="en")


def test_malformed_reference_raises_invalid_reference() -> None:
    """A reference that breaks the reference schema raises InvalidReferenceError."""
    with pytest.raises(InvalidReferenceError):
        get_scorer("toc").score({"p": PASSAGE}, {"p": [{"depth": 1}]}, {})


def test_malformed_input_raises_invalid_input() -> None:
    """A Scorer that reads Inputs rejects a non-string passage."""
    with pytest.raises(InvalidInputError, match="d1"):
        get_scorer("span_f1").score({"d1": {"text": "0123"}}, {"d1": []}, {})


def test_reference_without_input_raises_invalid_input() -> None:
    """Span F1 and TOC need an Input for every Reference id."""
    with pytest.raises(InvalidInputError, match="p2"):
        get_scorer("toc").score({"p1": PASSAGE}, {"p1": [], "p2": []}, {})
