"""CER and WER scoring for single pairs and batches."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

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


@dataclass(frozen=True)
class ErrorRateBatchScore:
    """Unrounded batch error-rate result with per-item scores and bookkeeping."""

    mean: float
    corpus: float
    items: Mapping[str, ErrorRateScore]
    skipped_ids: tuple[str, ...]
    extra_ids: tuple[str, ...]


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
    return _score_single_pair(
        reference, prediction, language, jiwer.process_characters, "cer"
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
    return _score_single_pair(
        reference, prediction, language, jiwer.process_words, "wer"
    )


def score_cer_batch(
    references: Mapping[str, str],
    predictions: Mapping[str, str],
    language: str,
) -> ErrorRateBatchScore:
    """Score Character Error Rate over a batch of Reference/Prediction pairs.

    Args:
        references: Mapping from item id to Reference text.
        predictions: Mapping from item id to Prediction text.
        language: ISO 639-1 language code; ``"bo"`` triggers Wylie conversion.

    Returns:
        An ``ErrorRateBatchScore`` with mean, corpus, per-item scores, and
        skipped/extra id lists.

    Raises:
        EmptyInputError: If the batch is empty or every item is skipped.
    """
    return _score_batch(references, predictions, language, score_cer)


def score_wer_batch(
    references: Mapping[str, str],
    predictions: Mapping[str, str],
    language: str,
) -> ErrorRateBatchScore:
    """Score Word Error Rate over a batch of Reference/Prediction pairs.

    Args:
        references: Mapping from item id to Reference text.
        predictions: Mapping from item id to Prediction text.
        language: ISO 639-1 language code; ``"bo"`` triggers Wylie conversion.

    Returns:
        An ``ErrorRateBatchScore`` with mean, corpus, per-item scores, and
        skipped/extra id lists.

    Raises:
        EmptyInputError: If the batch is empty or every item is skipped.
    """
    return _score_batch(references, predictions, language, score_wer)


def _prepare_pair(
    reference: str, prediction: str, language: str
) -> tuple[str, str]:
    """Strip and optionally Wylie-convert both texts; reject empty References.

    Emptiness is decided after stripping, before Wylie conversion.
    """
    if not reference.strip():
        raise EmptyInputError("Reference is empty after stripping")
    return prepare_text(reference, language), prepare_text(prediction, language)


def _build_error_rate_score(rate: float, output: Any) -> ErrorRateScore:
    """Build an ErrorRateScore from a jiwer process_* output."""
    return ErrorRateScore(
        rate=rate,
        substitutions=output.substitutions,
        deletions=output.deletions,
        insertions=output.insertions,
        hits=output.hits,
        reference_length=output.hits + output.substitutions + output.deletions,
    )


def _score_single_pair(
    reference: str,
    prediction: str,
    language: str,
    process: Callable[[str, str], Any],
    rate_attr: str,
) -> ErrorRateScore:
    """Prepare texts, run a jiwer process_* call, and build the score."""
    prepared_reference, prepared_prediction = _prepare_pair(
        reference, prediction, language
    )
    output = process(prepared_reference, prepared_prediction)
    return _build_error_rate_score(getattr(output, rate_attr), output)


def _batch_id_lists(
    references: Mapping[str, str],
    predictions: Mapping[str, str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return skipped empty-Reference ids and extra Prediction ids."""
    skipped_ids = tuple(
        item_id for item_id, text in references.items() if not text.strip()
    )
    extra_ids = tuple(
        item_id for item_id in predictions if item_id not in references
    )
    return skipped_ids, extra_ids


def _score_batch_items(
    references: Mapping[str, str],
    predictions: Mapping[str, str],
    language: str,
    skipped_ids: tuple[str, ...],
    score_one: Callable[[str, str, str], ErrorRateScore],
) -> dict[str, ErrorRateScore]:
    """Score every non-skipped Reference against its Prediction."""
    skipped_set = set(skipped_ids)
    items: dict[str, ErrorRateScore] = {}
    for item_id, reference in references.items():
        if item_id in skipped_set:
            continue
        prediction = predictions.get(item_id, "")
        items[item_id] = score_one(reference, prediction, language)
    return items


def _aggregate_batch_rates(
    items: Mapping[str, ErrorRateScore],
) -> tuple[float, float]:
    """Compute unrounded mean and corpus rates from per-item scores."""
    rates = [score.rate for score in items.values()]
    mean = sum(rates) / len(rates)
    total_edits = sum(
        score.substitutions + score.deletions + score.insertions
        for score in items.values()
    )
    total_reference_length = sum(
        score.reference_length for score in items.values()
    )
    corpus = total_edits / total_reference_length
    return mean, corpus


def _score_batch(
    references: Mapping[str, str],
    predictions: Mapping[str, str],
    language: str,
    score_one: Callable[[str, str, str], ErrorRateScore],
) -> ErrorRateBatchScore:
    """Aggregate per-item scores into mean and corpus rates with id bookkeeping."""
    if not references:
        raise EmptyInputError("Batch has no References")

    skipped_ids, extra_ids = _batch_id_lists(references, predictions)
    items = _score_batch_items(
        references, predictions, language, skipped_ids, score_one
    )
    if not items:
        raise EmptyInputError("Batch has no scorable items after skipping")

    mean, corpus = _aggregate_batch_rates(items)
    return ErrorRateBatchScore(
        mean=mean,
        corpus=corpus,
        items=items,
        skipped_ids=skipped_ids,
        extra_ids=extra_ids,
    )
