"""Tests for the JSONL loaders and validation against a Scorer's schemas."""

import io
import json
from pathlib import Path

import pytest

from buddhist_eval import (
    BuddhistEvalError,
    IdMismatchError,
    InvalidInputError,
    InvalidParameterError,
    InvalidPredictionError,
    InvalidReferenceError,
    MalformedJsonlError,
    get_scorer,
    load_ground_truth,
    load_inputs,
    load_media_paths,
    load_predictions,
)


def write_jsonl(path: Path, rows: list[object]) -> Path:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def test_load_three_files_and_score(tmp_path: Path) -> None:
    """The three arena files load into id-keyed mappings that score directly."""
    inputs = load_inputs(
        write_jsonl(
            tmp_path / "inputs.jsonl",
            [
                {"id": "p1", "input": "", "media": "images/p1.png"},
                {"id": "p2", "input": ""},
            ],
        )
    )
    references = load_ground_truth(
        write_jsonl(
            tmp_path / "gt.jsonl",
            [{"id": "p1", "reference": "བོད་ཡིག"}, {"id": "p2", "reference": "ཆོས"}],
        )
    )
    predictions = load_predictions(
        write_jsonl(
            tmp_path / "pred.jsonl",
            [{"id": "p2", "prediction": "ཆོས"}, {"id": "p1", "prediction": "བོད་ཡིག"}],
        )
    )

    cer = get_scorer("cer")
    cer.validate_ground_truth(inputs, references)
    cer.validate_predictions(inputs, predictions)

    assert inputs == {"p1": "", "p2": ""}
    assert cer.score(inputs, references, predictions).headline_metric == 0.0


def test_load_inputs_keeps_structured_input_values(tmp_path: Path) -> None:
    """An Input may be any JSON value; the loader returns it decoded."""
    path = write_jsonl(
        tmp_path / "inputs.jsonl",
        [{"id": "q1", "input": {"question": "?", "choices": ["A", "B"]}}],
    )

    assert load_inputs(path) == {"q1": {"question": "?", "choices": ["A", "B"]}}


def test_load_media_paths(tmp_path: Path) -> None:
    """Media paths are returned only for rows that have one."""
    path = write_jsonl(
        tmp_path / "inputs.jsonl",
        [{"id": "a", "input": "", "media": "a.wav"}, {"id": "b", "input": ""}],
    )

    assert load_media_paths(path) == {"a": "a.wav"}


def test_loaders_accept_binary_and_text_streams() -> None:
    """Streams (for example a file downloaded from storage) load like paths."""
    data = '﻿{"id": "a", "prediction": "x"}\r\n\n{"id": "b", "prediction": "y"}'

    assert load_predictions(io.BytesIO(data.encode("utf-8"))) == {"a": "x", "b": "y"}
    assert load_predictions(io.StringIO(data)) == {"a": "x", "b": "y"}


def test_loader_keeps_line_separator_characters_inside_strings() -> None:
    """Only newlines split rows; U+2028 inside a JSON string is kept."""
    line = json.dumps({"id": "a", "reference": "x y"}, ensure_ascii=False)

    assert load_ground_truth(io.StringIO(line)) == {"a": "x y"}


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("", "no rows"),
        ("{not json}\n", "line 1: invalid JSON"),
        ('["a"]\n', "line 1: expected a JSON object"),
        (
            '{"id": "a", "prediction": "x"}\n{"prediction": "y"}\n',
            "line 2: missing 'id'",
        ),
        ('{"id": "a"}\n', "missing 'prediction'"),
        ('{"id": 3, "prediction": "x"}\n', "'id' must be a non-empty string"),
        ('{"id": " ", "prediction": "x"}\n', "'id' must be a non-empty string"),
        (
            '{"id": "a", "prediction": "x"}\n{"id": "a", "prediction": "y"}\n',
            "line 2: duplicate id 'a'",
        ),
        ('{"id": "a", "predicton": "x"}\n', "missing 'prediction'"),
        ('{"id": "a", "prediction": "x", "score": 1}\n', "unexpected key"),
    ],
)
def test_malformed_predictions_raise(content: str, message: str) -> None:
    """Every structural problem names the line and raises MalformedJsonlError."""
    with pytest.raises(MalformedJsonlError, match=message):
        load_predictions(io.StringIO(content))
    assert issubclass(MalformedJsonlError, BuddhistEvalError)


def test_ground_truth_rejects_media_key() -> None:
    """Only Inputs rows may carry ``media``."""
    with pytest.raises(MalformedJsonlError, match="unexpected key"):
        load_ground_truth(
            io.StringIO('{"id": "a", "reference": "x", "media": "a.png"}')
        )


def test_inputs_reject_blank_media() -> None:
    """A ``media`` value must be a non-empty string."""
    with pytest.raises(MalformedJsonlError, match="'media'"):
        load_inputs(io.StringIO('{"id": "a", "input": "", "media": ""}'))


def test_missing_file_raises_malformed(tmp_path: Path) -> None:
    """An unreadable path raises MalformedJsonlError naming the file."""
    with pytest.raises(MalformedJsonlError, match="missing.jsonl"):
        load_inputs(tmp_path / "missing.jsonl")


def test_non_utf8_raises_malformed() -> None:
    """Bytes that are not UTF-8 raise MalformedJsonlError."""
    with pytest.raises(MalformedJsonlError, match="UTF-8"):
        load_inputs(io.BytesIO(b'{"id": "a", "input": "\xff"}'))


def test_ground_truth_ids_must_equal_input_ids() -> None:
    """Creation fails unless Inputs and Ground Truth hold the same ids."""
    with pytest.raises(IdMismatchError) as caught:
        get_scorer("cer").validate_ground_truth(
            {"a": "", "b": ""}, {"a": "x", "c": "y"}
        )

    assert caught.value.missing_ids == ("b",)
    assert caught.value.extra_ids == ("c",)


def test_ground_truth_reference_must_match_schema() -> None:
    """A Reference of the wrong shape fails Ground Truth validation."""
    with pytest.raises(InvalidReferenceError, match="'a'"):
        get_scorer("mcq_accuracy").validate_ground_truth({"a": ""}, {"a": ["A"]})


def test_ground_truth_input_must_match_schema() -> None:
    """Span F1 needs each Input to be the passage text."""
    with pytest.raises(InvalidInputError):
        get_scorer("span_f1").validate_ground_truth({"d": 5}, {"d": []})


def test_ground_truth_must_be_scorable() -> None:
    """Gold Spans outside the passage, or a blank answer, fail validation."""
    with pytest.raises(InvalidReferenceError, match="cannot be scored"):
        get_scorer("span_f1").validate_ground_truth(
            {"d": "abc"}, {"d": [{"start": 0, "end": 5, "tag": "q"}]}
        )
    with pytest.raises(InvalidReferenceError, match="cannot be scored"):
        get_scorer("mcq_accuracy").validate_ground_truth({"q": ""}, {"q": " "})
    with pytest.raises(InvalidReferenceError, match="cannot be scored"):
        get_scorer("toc").validate_ground_truth(
            {"p": "abc"}, {"p": [{"depth": 0, "start": 0}]}
        )


def test_ground_truth_validation_checks_params() -> None:
    """Invalid parameters fail Ground Truth validation."""
    with pytest.raises(InvalidParameterError):
        get_scorer("toc").validate_ground_truth({"p": "abc"}, {"p": []}, tolerance=-1)


def test_valid_ground_truth_passes() -> None:
    """Well-formed Inputs and Ground Truth validate without error."""
    get_scorer("toc").validate_ground_truth(
        {"p": "abcdef"}, {"p": [{"depth": 1, "start": 0}]}, tolerance=2
    )


def test_predictions_need_one_row_per_input_id() -> None:
    """A Predictions File with missing or extra ids is rejected."""
    with pytest.raises(IdMismatchError, match="missing 1 id"):
        get_scorer("cer").validate_predictions({"a": "", "b": ""}, {"a": "x"})
    with pytest.raises(IdMismatchError, match="not in Inputs"):
        get_scorer("cer").validate_predictions({"a": ""}, {"a": "x", "z": "y"})


def test_prediction_must_match_schema() -> None:
    """A Prediction of the wrong shape is rejected before scoring."""
    with pytest.raises(InvalidPredictionError, match="'p'"):
        get_scorer("toc").validate_predictions(
            {"p": "abc"}, {"p": [{"depth": 1, "start": 0}]}
        )


@pytest.mark.parametrize("name", ["cer", "wer", "span_f1", "toc", "mcq_accuracy"])
def test_sample_prediction_passes_validation(name: str) -> None:
    """Each Scorer's placeholder Prediction is valid for sample_submission.jsonl."""
    scorer = get_scorer(name)

    scorer.validate_predictions({"x": ""}, {"x": scorer.sample_prediction})


def test_id_mismatch_message_is_truncated() -> None:
    """Long id lists are shortened in the message but kept in full on the error."""
    inputs = {f"id{i:02d}": "" for i in range(20)}

    with pytest.raises(IdMismatchError, match=r"\.\.\. 15 more") as caught:
        get_scorer("cer").validate_predictions(inputs, {})
    assert len(caught.value.missing_ids) == 20
