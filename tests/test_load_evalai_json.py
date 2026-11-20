"""Behaviour tests for the EvalAI JSON loader."""

from pathlib import Path

import pytest

from buddhist_eval import MalformedEvalAIFileError, load_evalai_json, score_cer_batch

FIXTURES = Path(__file__).parent / "fixtures" / "evalai"


def test_load_evalai_json_reads_label_entries(tmp_path: Path) -> None:
    """Ground-truth files map each filename to its label text."""
    path = tmp_path / "ground_truth.json"
    path.write_text(
        '[{"filename": "a.txt", "label": "hello"},'
        ' {"filename": "b.txt", "label": "world"}]',
        encoding="utf-8",
    )
    assert load_evalai_json(path) == {"a.txt": "hello", "b.txt": "world"}


def test_load_evalai_json_reads_prediction_entries(tmp_path: Path) -> None:
    """Submission files map each filename to its prediction text."""
    path = tmp_path / "submission.json"
    path.write_text(
        '[{"filename": "a.txt", "prediction": "hello"},'
        ' {"filename": "b.txt", "prediction": "world"}]',
        encoding="utf-8",
    )
    assert load_evalai_json(path) == {"a.txt": "hello", "b.txt": "world"}


def test_load_evalai_json_accepts_single_object_as_one_item_list(
    tmp_path: Path,
) -> None:
    """A top-level object is treated as a one-item list, as EvalAI does."""
    path = tmp_path / "single.json"
    path.write_text(
        '{"filename": "only.txt", "label": "solo"}',
        encoding="utf-8",
    )
    assert load_evalai_json(path) == {"only.txt": "solo"}


def test_load_evalai_json_rejects_invalid_json(tmp_path: Path) -> None:
    """Invalid JSON raises MalformedEvalAIFileError naming the file."""
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(MalformedEvalAIFileError, match=str(path)):
        load_evalai_json(path)


def test_load_evalai_json_rejects_entry_missing_filename(tmp_path: Path) -> None:
    """An entry without filename raises MalformedEvalAIFileError naming the file."""
    path = tmp_path / "no_filename.json"
    path.write_text('[{"label": "hello"}]', encoding="utf-8")
    with pytest.raises(MalformedEvalAIFileError, match=str(path)):
        load_evalai_json(path)


def test_load_evalai_json_rejects_entry_missing_text_field(tmp_path: Path) -> None:
    """An entry without label or prediction raises MalformedEvalAIFileError."""
    path = tmp_path / "no_text.json"
    path.write_text('[{"filename": "a.txt"}]', encoding="utf-8")
    with pytest.raises(MalformedEvalAIFileError, match=str(path)):
        load_evalai_json(path)


def test_load_evalai_json_rejects_duplicate_filename(tmp_path: Path) -> None:
    """A duplicated filename raises MalformedEvalAIFileError naming the file."""
    path = tmp_path / "dupes.json"
    path.write_text(
        '[{"filename": "a.txt", "label": "one"},'
        ' {"filename": "a.txt", "label": "two"}]',
        encoding="utf-8",
    )
    with pytest.raises(MalformedEvalAIFileError, match=str(path)):
        load_evalai_json(path)


def test_load_evalai_json_and_score_cer_batch_end_to_end() -> None:
    """Fixture ground-truth and submission files score via score_cer_batch."""
    references = load_evalai_json(FIXTURES / "ground_truth.json")
    predictions = load_evalai_json(FIXTURES / "submission.json")
    result = score_cer_batch(references, predictions, language="en")
    # page1: "abcd" vs "abxd" -> 1/4; page2: "efgh" vs "efgh" -> 0/4; mean = 0.125
    assert result.mean == 0.125
    assert result.corpus == 0.125
    assert set(result.items) == {"page1.txt", "page2.txt"}
