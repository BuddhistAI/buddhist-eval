"""Tests for each Scorer's display format and Public Summary."""

import pytest

from buddhist_eval import get_scorer, list_scorers

PASSAGE = "abcdefghij0123456789"

# One real batch per Scorer, with several Input ids, a missing and an extra
# Prediction, so every per-item field in the details is populated.
REAL_BATCHES: dict[str, tuple[dict, dict, dict]] = {
    "cer": ({}, {"a": "abcd", "b": "ef gh"}, {"a": "abxd", "z": "q"}),
    "wer": ({}, {"a": "བོད་ཡིག", "b": "ཆོས"}, {"a": "བོད་སྐད", "z": "ཆོས"}),
    "span_f1": (
        {"d1": "0123456789", "d2": "abcdef"},
        {
            "d1": [{"start": 0, "end": 4, "tag": "quote"}],
            "d2": [{"start": 1, "end": 3, "tag": "name"}],
        },
        {"d1": [{"start": 0, "end": 3, "tag": "quote"}], "zz": []},
    ),
    "toc": (
        {"p": PASSAGE, "q": PASSAGE},
        {"p": [{"depth": 1, "start": 0}], "q": [{"depth": 2, "start": 10}]},
        {"p": [{"depth": 1, "anchor": "abc"}], "zz": []},
    ),
    "mcq_accuracy": ({}, {"q1": "A", "q2": "C"}, {"q1": "a", "q9": "B"}),
}


@pytest.mark.parametrize(
    ("name", "style", "decimals"),
    [
        ("cer", "percent", 2),
        ("wer", "percent", 2),
        ("span_f1", "decimal", 3),
        ("toc", "decimal", 3),
        ("mcq_accuracy", "percent", 1),
    ],
)
def test_display_format_per_scorer(name: str, style: str, decimals: int) -> None:
    """Each Scorer says how its Headline Metric is shown: style and decimals."""
    described = get_scorer(name).describe()

    assert described["display_format"] == {"style": style, "decimals": decimals}


PERCENT_2 = {"style": "percent", "decimals": 2}
DECIMAL_3 = {"style": "decimal", "decimals": 3}
COUNT = {"style": "decimal", "decimals": 0}


@pytest.mark.parametrize(
    ("name", "declared"),
    [
        ("cer", [{"key": "corpus", "label": "Corpus CER", "format": PERCENT_2}]),
        ("wer", [{"key": "corpus", "label": "Corpus WER", "format": PERCENT_2}]),
        (
            "span_f1",
            [
                {"key": "precision", "label": "Precision", "format": DECIMAL_3},
                {"key": "recall", "label": "Recall", "format": DECIMAL_3},
                {
                    "key": "mean_iou",
                    "label": "Mean IoU of matched spans",
                    "format": DECIMAL_3,
                },
            ],
        ),
        ("toc", []),
        (
            "mcq_accuracy",
            [
                {"key": "n_correct", "label": "Correct answers", "format": COUNT},
                {"key": "n_total", "label": "Questions", "format": COUNT},
            ],
        ),
    ],
)
def test_public_summary_declaration_per_scorer(
    name: str, declared: list[dict[str, object]]
) -> None:
    """Each Scorer lists, in order, the aggregate numbers it lets the public see."""
    assert get_scorer(name).describe()["public_summary"] == declared


def test_summarise_cer_returns_the_corpus_rate() -> None:
    """Summarising a CER result keeps the corpus rate and drops per-item scores."""
    cer = get_scorer("cer")
    # "a": 1 substitution over 4 characters; "b": 5 deletions over 5.
    result = cer.score({}, {"a": "abcd", "b": "ef gh"}, {"a": "abxd"}, language="en")

    summary = cer.summarise(result.details)

    assert summary == [
        {
            "key": "corpus",
            "label": "Corpus CER",
            "value": pytest.approx(6 / 9),
            "format": PERCENT_2,
        }
    ]


def test_summarise_omits_a_declared_key_missing_from_details() -> None:
    """Details stored before a key existed summarise without it, not with an error."""
    summary = get_scorer("mcq_accuracy").summarise({"accuracy": 0.5, "n_total": 4})

    assert summary == [
        {"key": "n_total", "label": "Questions", "value": 4, "format": COUNT}
    ]


def test_every_scorer_has_a_real_batch_here() -> None:
    """The privacy check below covers every registered Scorer."""
    assert sorted(REAL_BATCHES) == sorted(s.name for s in list_scorers())


@pytest.mark.parametrize("name", sorted(REAL_BATCHES))
def test_summarising_a_real_result_never_exposes_per_item_scores(name: str) -> None:
    """Every summarised value is one number; per-item lists and maps never appear."""
    scorer = get_scorer(name)
    inputs, references, predictions = REAL_BATCHES[name]
    result = scorer.score(inputs, references, predictions)

    summary = scorer.summarise(result.details)

    assert [e["key"] for e in summary] == [e.key for e in scorer.public_summary]
    for entry in summary:
        assert type(entry["value"]) in (int, float), entry
        assert set(entry) == {"key", "label", "value", "format"}
