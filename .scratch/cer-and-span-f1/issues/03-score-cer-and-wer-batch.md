# 03: Score CER and WER over a batch

**What to build:** A leaderboard maintainer can call `score_cer_batch(references, predictions, language)` or `score_wer_batch(...)` with two dictionaries from item id to text, and get an `ErrorRateBatchScore` back. It holds:

- the mean rate, which equals today's leaderboard figure
- the corpus rate: total edits divided by total Reference length
- the per-item `ErrorRateScore`s
- `skipped_ids`: items whose Reference is empty
- `extra_ids`: Predictions with no Reference

Models are penalised for missing or empty Predictions instead of having them skipped. See the RFC (Implementation Decisions › Text scoring module).

**Blocked by:** 02 (single-pair CER and WER).

**Status:** resolved

- [x] `score_cer_batch` and `score_wer_batch` are exported from `buddhist_eval`.
- [x] `ErrorRateBatchScore` is a frozen dataclass with `mean`, `corpus`, `items`, `skipped_ids` and `extra_ids`. Scores are unrounded.
- [x] A Reference id with no Prediction is scored as an empty Prediction, with a rate of 1.0.
- [x] A Reference that is empty after stripping is left out of `mean` and `corpus`, and its id is listed in `skipped_ids`.
- [x] Prediction ids with no Reference are ignored and listed in `extra_ids`.
- [x] An empty batch, or one where every item is skipped, raises the package's empty-input error.
- [x] Parity test: on a fixed Tibetan batch with no empty or missing items, `mean` equals the EvalAI backend's method (per-item `evaluate` CER and WER on Wylie, averaged). The expected values are stored as fixtures.
- [x] Behaviour tests: `corpus` differs from `mean` when items have different lengths and equals the hand-computed value, and each of the missing, extra, empty-Reference and empty-batch cases behaves as above.

## Comments

- Merged via `impl/03-score-cer-wer-batch` at c105b49.
