"""Tests for each Scorer's display format and Public Summary."""

import pytest

from buddhist_eval import get_scorer


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
