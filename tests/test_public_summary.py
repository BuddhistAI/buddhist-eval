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
