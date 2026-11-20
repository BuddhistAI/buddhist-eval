# 06: Score Spans over a batch of documents

**What to build:** An annotation researcher can call `score_spans_batch(golds, predictions, iou_threshold=0.5)` with two dictionaries from document id to document, and get one corpus-level result. The Match, gold and predicted counts are pooled across documents before precision, recall and F1 are computed, as `iou50_metrics` does in the quotation-detection repo. Skipped documents lower recall, and extra predicted documents are reported. See the RFC (Implementation Decisions › Span scoring module).

**Blocked by:** 05 (single-document span scoring).

**Status:** ready-for-agent

- [ ] `score_spans_batch` is exported from `buddhist_eval`.
- [ ] The batch result holds the pooled precision, recall and F1, `n_gold`, `n_pred` and `n_match`, the mean IoU over all Matches, a pooled per-Tag breakdown, and `extra_ids`.
- [ ] A gold document id with no predicted document adds its Gold Spans to `n_gold` with no Matches.
- [ ] Predicted document ids with no gold are ignored and listed in `extra_ids`.
- [ ] Per-document validation (identical texts, Spans inside the text) and threshold validation behave as in `score_spans`.
- [ ] An empty batch raises the package's empty-input error.
- [ ] Parity test: on a fixed multi-document fixture, the pooled scores equal summing `match_iou` counts across documents and applying `prf`, as `iou50_metrics` does.
- [ ] Behaviour tests: pooled F1 differs from the average of per-document F1 on an uneven fixture, and the missing-document, extra-document and empty-batch cases behave as above.
