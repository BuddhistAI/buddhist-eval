"""CER and WER scoring for a single Reference/Prediction pair."""

from dataclasses import dataclass

import jiwer

from buddhist_eval.errors import EmptyInputError
from buddhist_eval.utils.text import prepare_text


@dataclass(frozen=True)
class ErrorRateScore:
    """Unrounded error-rate result with edit counts and Reference length."""

    rate: float
    substitutions: int
    deletions: int
    insertions: int
    hits: int
    reference_length: int


def score_cer(reference: str, prediction: str, language: str) -> ErrorRateScore:
    """Score Character Error Rate for one Reference/Prediction pair.

    Args:
        reference: Ground-truth text for one item.
        prediction: Model text for the same item.
        language: ISO 639-1 language code; ``"bo"`` triggers Wylie conversion.

    Returns:
        An ``ErrorRateScore`` with character-level edits and Reference length.

    Raises:
        EmptyInputError: If the Reference is empty after stripping.
    """
    prepared_reference, prepared_prediction = _prepare_pair(
        reference, prediction, language
    )
    output = jiwer.process_characters(prepared_reference, prepared_prediction)
    return ErrorRateScore(
        rate=output.cer,
        substitutions=output.substitutions,
        deletions=output.deletions,
        insertions=output.insertions,
        hits=output.hits,
        reference_length=output.hits + output.substitutions + output.deletions,
    )


def score_wer(reference: str, prediction: str, language: str) -> ErrorRateScore:
    """Score Word Error Rate for one Reference/Prediction pair.

    Args:
        reference: Ground-truth text for one item.
        prediction: Model text for the same item.
        language: ISO 639-1 language code; ``"bo"`` triggers Wylie conversion.

    Returns:
        An ``ErrorRateScore`` with word-level edits and Reference length.

    Raises:
        EmptyInputError: If the Reference is empty after stripping.
    """
    prepared_reference, prepared_prediction = _prepare_pair(
        reference, prediction, language
    )
    output = jiwer.process_words(prepared_reference, prepared_prediction)
    return ErrorRateScore(
        rate=output.wer,
        substitutions=output.substitutions,
        deletions=output.deletions,
        insertions=output.insertions,
        hits=output.hits,
        reference_length=output.hits + output.substitutions + output.deletions,
    )


def _prepare_pair(
    reference: str, prediction: str, language: str
) -> tuple[str, str]:
    """Strip and optionally Wylie-convert both texts; reject empty References.

    Emptiness is decided after stripping, before Wylie conversion.
    """
    if not reference.strip():
        raise EmptyInputError("Reference is empty after stripping")
    return prepare_text(reference, language), prepare_text(prediction, language)
