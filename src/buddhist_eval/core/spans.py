"""Span IoU-F1 scoring for single documents and batches."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from buddhist_eval.errors import (
    BuddhistEvalError,
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


@dataclass(frozen=True)
class _SpanMatch:
    """One gold/predicted Span pairing with its IoU."""

    gold_index: int
    pred_index: int
    iou: float


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
        BuddhistEvalError: If ``iou_threshold`` is outside (0, 1].
        InvalidSpanError: If a Span is inverted, negative, or past the text.
    """
    _validate_threshold(iou_threshold)
    _validate_documents(gold, predicted)
    matches = _match_iou(gold.spans, predicted.spans, iou_threshold)
    return _build_span_score(iou_threshold, gold.spans, predicted.spans, matches)


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
        BuddhistEvalError: If ``iou_threshold`` is outside (0, 1].
        InvalidSpanError: If a Span is invalid.
    """
    if not golds:
        raise EmptyInputError("golds batch is empty")
    _validate_threshold(iou_threshold)

    extra_ids = tuple(sorted(set(predictions) - set(golds)))
    gold_spans, pred_spans, matches = _pool_batch_matches(
        golds, predictions, iou_threshold
    )
    base = _build_span_score(iou_threshold, gold_spans, pred_spans, matches)
    return SpanBatchScore(
        iou_threshold=base.iou_threshold,
        precision=base.precision,
        recall=base.recall,
        f1=base.f1,
        n_gold=base.n_gold,
        n_pred=base.n_pred,
        n_match=base.n_match,
        mean_iou=base.mean_iou,
        per_tag=base.per_tag,
        extra_ids=extra_ids,
    )


def _build_span_score(
    iou_threshold: float,
    gold_spans: tuple[Span, ...],
    pred_spans: tuple[Span, ...],
    matches: list[_SpanMatch],
) -> SpanScore:
    """Build overall and per-Tag SpanScore fields from matched Spans."""
    n_gold = len(gold_spans)
    n_pred = len(pred_spans)
    n_match = len(matches)
    precision, recall, f1 = _precision_recall_f1(n_match, n_pred, n_gold)
    mean_iou = sum(m.iou for m in matches) / n_match if n_match else 0.0
    return SpanScore(
        iou_threshold=iou_threshold,
        precision=precision,
        recall=recall,
        f1=f1,
        n_gold=n_gold,
        n_pred=n_pred,
        n_match=n_match,
        mean_iou=mean_iou,
        per_tag=_per_tag_scores(gold_spans, pred_spans, matches),
    )


def _pool_batch_matches(
    golds: dict[str, Document],
    predictions: dict[str, Document],
    iou_threshold: float,
) -> tuple[tuple[Span, ...], tuple[Span, ...], list[_SpanMatch]]:
    """Validate documents and pool Spans/Matches with global indices."""
    all_gold: list[Span] = []
    all_pred: list[Span] = []
    all_matches: list[_SpanMatch] = []
    gold_offset = 0
    pred_offset = 0

    for doc_id, gold in golds.items():
        predicted = predictions.get(doc_id)
        if predicted is None:
            _validate_spans(gold)
            all_gold.extend(gold.spans)
            gold_offset += len(gold.spans)
            continue

        _validate_documents(gold, predicted)
        for match in _match_iou(gold.spans, predicted.spans, iou_threshold):
            all_matches.append(
                _SpanMatch(
                    gold_index=gold_offset + match.gold_index,
                    pred_index=pred_offset + match.pred_index,
                    iou=match.iou,
                )
            )
        all_gold.extend(gold.spans)
        all_pred.extend(predicted.spans)
        gold_offset += len(gold.spans)
        pred_offset += len(predicted.spans)

    return tuple(all_gold), tuple(all_pred), all_matches


def _validate_threshold(iou_threshold: float) -> None:
    if not (0.0 < iou_threshold <= 1.0):
        raise BuddhistEvalError(
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
) -> list[_SpanMatch]:
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
    matches: list[_SpanMatch] = []
    for value, g_idx, p_idx in candidates:
        if g_idx in used_g or p_idx in used_p:
            continue
        used_g.add(g_idx)
        used_p.add(p_idx)
        matches.append(_SpanMatch(gold_index=g_idx, pred_index=p_idx, iou=value))
    return matches


def _precision_recall_f1(
    n_match: int, n_pred: int, n_gold: int
) -> tuple[float, float, float]:
    precision = n_match / n_pred if n_pred else 0.0
    recall = n_match / n_gold if n_gold else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return precision, recall, f1


def _index_spans_by_tag(spans: tuple[Span, ...]) -> dict[str, list[int]]:
    by_tag: dict[str, list[int]] = defaultdict(list)
    for idx, span in enumerate(spans):
        by_tag[span.tag].append(idx)
    return by_tag


def _match_counts_by_tag(
    gold: tuple[Span, ...], matches: list[_SpanMatch]
) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for match in matches:
        counts[gold[match.gold_index].tag] += 1
    return counts


def _per_tag_scores(
    gold: tuple[Span, ...],
    pred: tuple[Span, ...],
    matches: list[_SpanMatch],
) -> dict[str, TagScore]:
    tags = {span.tag for span in gold} | {span.tag for span in pred}
    gold_by_tag = _index_spans_by_tag(gold)
    pred_by_tag = _index_spans_by_tag(pred)
    n_match_by_tag = _match_counts_by_tag(gold, matches)

    per_tag: dict[str, TagScore] = {}
    for tag in sorted(tags):
        n_gold = len(gold_by_tag[tag])
        n_pred = len(pred_by_tag[tag])
        n_match = n_match_by_tag[tag]
        precision, recall, f1 = _precision_recall_f1(n_match, n_pred, n_gold)
        per_tag[tag] = TagScore(
            precision=precision,
            recall=recall,
            f1=f1,
            n_gold=n_gold,
            n_pred=n_pred,
            n_match=n_match,
        )
    return per_tag
