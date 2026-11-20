# 05: Score Spans for one document

**What to build:** An annotation researcher can call `score_spans(gold, predicted, iou_threshold=0.5)` with two documents, each holding the text and its Spans. A Span holds `start`, `end` (inclusive) and `tag`. The call returns a `SpanScore` holding:

- precision, recall and F1
- `n_gold`, `n_pred` and `n_match`
- the mean IoU of the Matches
- a per-Tag breakdown

Matching reproduces `match_iou` and `prf` from the quotation-detection repo exactly, so the numbers agree with `eval_quotation_v3.py`. See the RFC (Implementation Decisions › Span scoring module).

**Blocked by:** 01 (package foundation).

**Status:** ready-for-agent

- [ ] `score_spans`, the document and Span types, `SpanScore` and `TagScore` are exported from `buddhist_eval`. All are frozen dataclasses.
- [ ] IoU is computed on inclusive ranges, so `(0, 4)` covers five characters.
- [ ] Matching only pairs a Predicted Span with a Gold Span of the same Tag, and only when their IoU is at least `iou_threshold`. Candidates are sorted in descending order of (IoU, gold index, predicted index) and accepted greedily, so each Span is used at most once.
- [ ] Precision, recall and F1 are each 0.0 when their denominator is 0, so no spans on either side gives all zeros, with `n_gold == n_pred == 0`.
- [ ] `per_tag` holds a `TagScore` for each Tag seen on either side.
- [ ] Mean IoU is the average IoU of accepted Matches, or 0.0 when there are none.
- [ ] Errors are raised when the gold and predicted texts differ, a Span has `end < start`, a negative `start`, or an `end` past the last character, or `iou_threshold` is outside (0, 1].
- [ ] Parity test: on fixed span sets, precision, recall, F1 and `n_match` equal `match_iou` plus `prf` from the reference repo. The expected values are stored as fixtures.
- [ ] Behaviour tests: Tag mismatch never matches, one-to-one matching holds when two Predicted Spans compete for one Gold Span, ties are broken in the order above, and each validation error is raised.
