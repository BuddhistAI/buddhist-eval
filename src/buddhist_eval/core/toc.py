"""Table-of-contents (outline heading) scoring for single passages and batches.

Follows the buddhist-benchmarks ``toc-extraction`` metric used by
toc-extraction-tibetan: anchors are located verbatim in the passage, matched
one-to-one to Gold Headings by start offset, and scored as
``0.3 * position F1 + 0.7 * position-and-depth F1``.
"""

from __future__ import annotations

from dataclasses import dataclass

from buddhist_eval.errors import BuddhistEvalError, EmptyInputError, InvalidHeadingError

POSITION_WEIGHT = 0.3
DEPTH_WEIGHT = 0.7
SPURIOUS_PENALTY = 0.25


@dataclass(frozen=True)
class Heading:
    """A ground-truth outline heading: its depth and start offset in the passage."""

    depth: int
    start: int


@dataclass(frozen=True)
class PredictedHeading:
    """A model's outline heading: its depth and the verbatim text it starts with."""

    depth: int
    anchor: str


@dataclass(frozen=True)
class TocPassage:
    """A passage text together with its Gold Headings in reading order."""

    text: str
    headings: tuple[Heading, ...]


@dataclass(frozen=True)
class TocScore:
    """Outline heading scoring result for one passage."""

    score: float
    position_f1: float
    depth_f1: float
    n_gold: int
    n_pred: int
    n_located: int
    n_position_match: int
    n_depth_match: int


@dataclass(frozen=True)
class TocBatchScore:
    """Outline heading scoring result for a batch of passages."""

    mean: float
    items: dict[str, TocScore]
    extra_ids: tuple[str, ...]


def score_toc(
    passage: TocPassage,
    predicted: tuple[PredictedHeading, ...],
    tolerance: int = 3,
) -> TocScore:
    """Score Predicted Headings against a passage's Gold Headings.

    Each anchor is searched verbatim in the text, starting after the previous
    located anchor. A located heading matches the closest unused Gold Heading
    whose start is within ``tolerance`` characters. Unlocated anchors count as
    predictions but never match.

    Args:
        passage: The passage text and its Gold Headings.
        predicted: Predicted Headings in reading order.
        tolerance: Maximum character distance between a located anchor and a
            Gold Heading start; must be 0 or more.

    Returns:
        A TocScore. ``score`` is ``0.3 * position_f1 + 0.7 * depth_f1``. A
        passage with no Gold Headings scores ``1.0`` minus ``0.25`` per
        Predicted Heading, floored at ``0.0``.

    Raises:
        BuddhistEvalError: If ``tolerance`` is negative.
        InvalidHeadingError: If a Gold Heading has depth below 1 or a start
            outside the text.
    """
    _validate_tolerance(tolerance)
    _validate_headings(passage)
    return _score_passage(passage, predicted, tolerance)


def score_toc_batch(
    passages: dict[str, TocPassage],
    predictions: dict[str, tuple[PredictedHeading, ...]],
    tolerance: int = 3,
) -> TocBatchScore:
    """Score Predicted Headings across a batch of passages.

    A passage id with no prediction is scored as an empty prediction.

    Args:
        passages: Mapping from passage id to TocPassage.
        predictions: Mapping from passage id to Predicted Headings.
        tolerance: Maximum character distance for a match; must be 0 or more.

    Returns:
        A TocBatchScore whose ``mean`` gives every passage equal weight, plus
        per-passage ``items`` and ``extra_ids`` (predictions with no passage).

    Raises:
        EmptyInputError: If ``passages`` is empty.
        BuddhistEvalError: If ``tolerance`` is negative.
        InvalidHeadingError: If a Gold Heading is invalid.
    """
    if not passages:
        raise EmptyInputError("passages batch is empty")
    _validate_tolerance(tolerance)
    items: dict[str, TocScore] = {}
    for passage_id, passage in passages.items():
        _validate_headings(passage)
        items[passage_id] = _score_passage(
            passage, predictions.get(passage_id, ()), tolerance
        )
    return TocBatchScore(
        mean=sum(s.score for s in items.values()) / len(items),
        items=items,
        extra_ids=tuple(sorted(set(predictions) - set(passages))),
    )


def _score_passage(
    passage: TocPassage,
    predicted: tuple[PredictedHeading, ...],
    tolerance: int,
) -> TocScore:
    """Locate, match and score one passage that has already been validated."""
    gold = passage.headings
    located = _locate_anchors(passage.text, predicted)
    matches = _match_starts([start for start, _ in located], gold, tolerance)
    n_depth = sum(1 for p, g in matches if located[p][1].depth == gold[g].depth)
    n_gold, n_pred = len(gold), len(predicted)
    position_f1 = _f1(len(matches), n_pred, n_gold)
    depth_f1 = _f1(n_depth, n_pred, n_gold)
    return TocScore(
        score=_combine(position_f1, depth_f1, n_gold, n_pred),
        position_f1=position_f1,
        depth_f1=depth_f1,
        n_gold=n_gold,
        n_pred=n_pred,
        n_located=len(located),
        n_position_match=len(matches),
        n_depth_match=n_depth,
    )


def _locate_anchors(
    text: str, predicted: tuple[PredictedHeading, ...]
) -> list[tuple[int, PredictedHeading]]:
    """Find each stripped anchor after the previous located one; drop misses."""
    located: list[tuple[int, PredictedHeading]] = []
    cursor = 0
    for heading in predicted:
        anchor = heading.anchor.strip()
        position = text.find(anchor, cursor) if anchor else -1
        if position >= 0:
            located.append((position, heading))
            cursor = position + 1
    return located


def _match_starts(
    pred_starts: list[int], gold: tuple[Heading, ...], tolerance: int
) -> list[tuple[int, int]]:
    """Greedy in prediction order: each takes the nearest unused gold in range."""
    used: set[int] = set()
    matches: list[tuple[int, int]] = []
    for p_idx, start in enumerate(pred_starts):
        best: int | None = None
        for g_idx, heading in enumerate(gold):
            distance = abs(heading.start - start)
            if g_idx in used or distance > tolerance:
                continue
            if best is None or distance < abs(gold[best].start - start):
                best = g_idx
        if best is not None:
            used.add(best)
            matches.append((p_idx, best))
    return matches


def _f1(n_match: int, n_pred: int, n_gold: int) -> float:
    if n_pred == 0 and n_gold == 0:
        return 1.0
    if n_match == 0:
        return 0.0
    precision, recall = n_match / n_pred, n_match / n_gold
    return 2 * precision * recall / (precision + recall)


def _combine(position_f1: float, depth_f1: float, n_gold: int, n_pred: int) -> float:
    if n_gold == 0:
        return max(0.0, 1.0 - SPURIOUS_PENALTY * n_pred)
    return POSITION_WEIGHT * position_f1 + DEPTH_WEIGHT * depth_f1


def _validate_tolerance(tolerance: int) -> None:
    if tolerance < 0:
        raise BuddhistEvalError(f"tolerance must be 0 or more, got {tolerance!r}")


def _validate_headings(passage: TocPassage) -> None:
    for heading in passage.headings:
        if heading.depth < 1 or not 0 <= heading.start < len(passage.text):
            raise InvalidHeadingError(
                f"invalid heading depth={heading.depth} start={heading.start} "
                f"for text length {len(passage.text)}"
            )
