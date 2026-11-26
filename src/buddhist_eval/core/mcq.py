"""Multiple-choice question accuracy against an exact Answer Key."""

from __future__ import annotations

from dataclasses import dataclass

from buddhist_eval.errors import EmptyInputError


@dataclass(frozen=True)
class McqScore:
    """Multiple-choice accuracy for a batch of questions."""

    accuracy: float
    n_correct: int
    n_total: int
    items: dict[str, bool]
    missing_ids: tuple[str, ...]
    extra_ids: tuple[str, ...]


def score_mcq(answer_key: dict[str, str], predictions: dict[str, str]) -> McqScore:
    """Score predicted choices against an Answer Key by exact match.

    Both sides are stripped and casefolded before comparison, so ``" b"``
    matches ``"B"``. A question with no prediction counts as wrong.

    Args:
        answer_key: Mapping from question id to the correct choice.
        predictions: Mapping from question id to the model's choice.

    Returns:
        An McqScore. ``accuracy`` is ``n_correct / n_total`` over every
        question in the Answer Key; ``items`` records correctness per id,
        ``missing_ids`` lists questions with no prediction and ``extra_ids``
        lists predictions with no question.

    Raises:
        EmptyInputError: If ``answer_key`` is empty or holds a blank answer.
    """
    if not answer_key:
        raise EmptyInputError("answer key is empty")
    items = {
        question_id: _is_correct(question_id, answer, predictions)
        for question_id, answer in answer_key.items()
    }
    n_correct = sum(items.values())
    return McqScore(
        accuracy=n_correct / len(items),
        n_correct=n_correct,
        n_total=len(items),
        items=items,
        missing_ids=tuple(sorted(set(answer_key) - set(predictions))),
        extra_ids=tuple(sorted(set(predictions) - set(answer_key))),
    )


def _is_correct(question_id: str, answer: str, predictions: dict[str, str]) -> bool:
    expected = _normalise(answer)
    if not expected:
        raise EmptyInputError(f"answer key has a blank answer for {question_id!r}")
    predicted = predictions.get(question_id)
    return predicted is not None and _normalise(predicted) == expected


def _normalise(choice: str) -> str:
    return choice.strip().casefold()
