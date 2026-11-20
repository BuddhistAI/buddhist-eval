"""Span IoU-F1 scoring for single documents and batches."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from buddhist_eval.errors import (
    EmptyInputError,
    InvalidSpanError,
    MisalignedTextsError,
)


@dataclass(frozen=True)
class Span:
    """A labelled, contiguous character range with an inclusive end."""

    start: int
    end: int
    tag: str


@dataclass(frozen=True)
class Document:
    """A text together with its Spans."""

    text: str
    spans: tuple[Span, ...]


@dataclass(frozen=True)
class TagScore:
    """Precision, recall and F1 for one Tag."""

    precision: float
    recall: float
    f1: float
    n_gold: int
    n_pred: int
    n_match: int


@dataclass(frozen=True)
class SpanScore:
    """IoU-matched span scoring result for one document pair."""

    iou_threshold: float
    precision: float
    recall: float
    f1: float
    n_gold: int
    n_pred: int
    n_match: int
    mean_iou: float
    per_tag: dict[str, TagScore]


@dataclass(frozen=True)
class SpanBatchScore:
    """Pooled IoU-matched span scoring result for a document batch."""

    iou_threshold: float
    precision: float
    recall: float
    f1: float
    n_gold: int
    n_pred: int
    n_match: int
    mean_iou: float
    per_tag: dict[str, TagScore]
    extra_ids: tuple[str, ...]


def score_spans(
    gold: Document,
    predicted: Document,
    iou_threshold: float = 0.5,
) -> SpanScore:
    """Score Predicted Spans against Gold Spans with IoU-matched PRF.

    Args:
        gold: Document holding ground-truth Spans.
        predicted: Document holding model Spans.
        iou_threshold: Minimum IoU for a Match; must be in (0, 1].

    Returns:
        A SpanScore with overall and per-Tag precision, recall and F1.

    Raises:
        MisalignedTextsError: If gold and predicted texts differ.
        InvalidSpanError: If a Span is inverted, negative, or past the text,
            or if ``iou_threshold`` is outside (0, 1].
    """
    _validate_threshold(iou_threshold)
    _validate_documents(gold, predicted)

    matches = _match_iou(gold.spans, predicted.spans, iou_threshold)
    n_gold = len(gold.spans)
    n_pred = len(predicted.spans)
    n_match = len(matches)
    precision, recall, f1 = _prf(n_match, n_pred, n_gold)
    mean_iou = (
        sum(iou for _, _, iou in matches) / n_match if n_match else 0.0
    )
    per_tag = _per_tag_scores(gold.spans, predicted.spans, matches)

    return SpanScore(
        iou_threshold=iou_threshold,
        precision=precision,
        recall=recall,
        f1=f1,
        n_gold=n_gold,
        n_pred=n_pred,
        n_match=n_match,
        mean_iou=mean_iou,
        per_tag=per_tag,
    )


def score_spans_batch(
    golds: dict[str, Document],
    predictions: dict[str, Document],
    iou_threshold: float = 0.5,
) -> SpanBatchScore:
    """Score Predicted Spans against Gold Spans across a batch of documents.

    Match, gold and predicted counts are pooled across documents before
    precision, recall and F1 are computed.

    Args:
        golds: Mapping from document id to gold Document.
        predictions: Mapping from document id to predicted Document.
        iou_threshold: Minimum IoU for a Match; must be in (0, 1].

    Returns:
        A SpanBatchScore with pooled overall and per-Tag metrics plus
        ``extra_ids``.

    Raises:
        EmptyInputError: If ``golds`` is empty.
        MisalignedTextsError: If a paired document's texts differ.
        InvalidSpanError: If a Span is invalid or ``iou_threshold`` is
            outside (0, 1].
    """
    if not golds:
        raise EmptyInputError("golds batch is empty")
    _validate_threshold(iou_threshold)

    extra_ids = tuple(sorted(set(predictions) - set(golds)))
    all_gold_spans: list[Span] = []
    all_pred_spans: list[Span] = []
    # Matches stored as (global_gold_idx, global_pred_idx, iou) for pooling.
    all_matches: list[tuple[int, int, float]] = []
    gold_offset = 0
    pred_offset = 0
    iou_sum = 0.0

    for doc_id, gold in golds.items():
        predicted = predictions.get(doc_id)
        if predicted is None:
            _validate_spans(gold)
            all_gold_spans.extend(gold.spans)
            gold_offset += len(gold.spans)
            continue

        _validate_documents(gold, predicted)
        matches = _match_iou(gold.spans, predicted.spans, iou_threshold)
        for g_idx, p_idx, iou in matches:
            all_matches.append((gold_offset + g_idx, pred_offset + p_idx, iou))
            iou_sum += iou
        all_gold_spans.extend(gold.spans)
        all_pred_spans.extend(predicted.spans)
        gold_offset += len(gold.spans)
        pred_offset += len(predicted.spans)

    n_gold = len(all_gold_spans)
    n_pred = len(all_pred_spans)
    n_match = len(all_matches)
    precision, recall, f1 = _prf(n_match, n_pred, n_gold)
    mean_iou = iou_sum / n_match if n_match else 0.0
    per_tag = _per_tag_scores(
        tuple(all_gold_spans),
        tuple(all_pred_spans),
        all_matches,
    )

    return SpanBatchScore(
        iou_threshold=iou_threshold,
        precision=precision,
        recall=recall,
        f1=f1,
        n_gold=n_gold,
        n_pred=n_pred,
        n_match=n_match,
        mean_iou=mean_iou,
        per_tag=per_tag,
        extra_ids=extra_ids,
    )


def _validate_threshold(iou_threshold: float) -> None:
    if not (0.0 < iou_threshold <= 1.0):
        raise InvalidSpanError(
            f"iou_threshold must be in (0, 1], got {iou_threshold!r}"
        )


def _validate_documents(gold: Document, predicted: Document) -> None:
    if gold.text != predicted.text:
        raise MisalignedTextsError("gold and predicted document texts differ")
    _validate_spans(gold)
    _validate_spans(predicted)


def _validate_spans(document: Document) -> None:
    last = len(document.text) - 1
    for span in document.spans:
        if span.start < 0 or span.end < span.start or span.end > last:
            raise InvalidSpanError(
                f"invalid span start={span.start} end={span.end} "
                f"for text length {len(document.text)}"
            )


def _inclusive_iou(a: Span, b: Span) -> float:
    lo = max(a.start, b.start)
    hi = min(a.end, b.end)
    if hi < lo:
        return 0.0
    inter = hi - lo + 1
    union = (a.end - a.start + 1) + (b.end - b.start + 1) - inter
    return inter / union if union else 0.0


def _match_iou(
    gold: tuple[Span, ...],
    pred: tuple[Span, ...],
    threshold: float,
) -> list[tuple[int, int, float]]:
    """Greedy one-to-one matching by descending (IoU, gold index, pred index)."""
    candidates: list[tuple[float, int, int]] = []
    for p_idx, p in enumerate(pred):
        for g_idx, g in enumerate(gold):
            if g.tag != p.tag:
                continue
            value = _inclusive_iou(g, p)
            if value >= threshold:
                candidates.append((value, g_idx, p_idx))
    candidates.sort(reverse=True)
    used_g: set[int] = set()
    used_p: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for value, g_idx, p_idx in candidates:
        if g_idx in used_g or p_idx in used_p:
            continue
        used_g.add(g_idx)
        used_p.add(p_idx)
        matches.append((g_idx, p_idx, value))
    return matches


def _prf(n_match: int, n_pred: int, n_gold: int) -> tuple[float, float, float]:
    precision = n_match / n_pred if n_pred else 0.0
    recall = n_match / n_gold if n_gold else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return precision, recall, f1


def _per_tag_scores(
    gold: tuple[Span, ...],
    pred: tuple[Span, ...],
    matches: list[tuple[int, int, float]],
) -> dict[str, TagScore]:
    tags = {span.tag for span in gold} | {span.tag for span in pred}
    gold_by_tag: dict[str, list[int]] = defaultdict(list)
    pred_by_tag: dict[str, list[int]] = defaultdict(list)
    for idx, span in enumerate(gold):
        gold_by_tag[span.tag].append(idx)
    for idx, span in enumerate(pred):
        pred_by_tag[span.tag].append(idx)

    n_match_by_tag: dict[str, int] = defaultdict(int)
    for g_idx, _p_idx, _iou in matches:
        n_match_by_tag[gold[g_idx].tag] += 1

    per_tag: dict[str, TagScore] = {}
    for tag in sorted(tags):
        n_gold = len(gold_by_tag[tag])
        n_pred = len(pred_by_tag[tag])
        n_match = n_match_by_tag[tag]
        precision, recall, f1 = _prf(n_match, n_pred, n_gold)
        per_tag[tag] = TagScore(
            precision=precision,
            recall=recall,
            f1=f1,
            n_gold=n_gold,
            n_pred=n_pred,
            n_match=n_match,
        )
    return per_tag
