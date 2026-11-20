# RFC 0001: CER, WER and Span IoU-F1 scoring

Status: ready-for-agent

## Problem Statement

Teams building Tibetan OCR, speech-to-text and span-annotation models (for example quotation detection) each score their models with their own copy of the evaluation code. CER and WER live inside the EvalAI leaderboard backend, tied to S3 downloads and the database. Span IoU-F1 lives inside the quotation-detection training repo, tied to BIOE token sequences and model logits. Nobody can score a Prediction against a Reference, or a set of Predicted Spans against Gold Spans, without pulling in one of those systems. And when the copies drift, scores from different teams stop being comparable.

The existing code also hides failures. The leaderboard skips items whose Prediction is empty, so a model that outputs nothing for a page looks better than one that outputs something wrong. It also turns any exception into a CER of 1.0, so a bug looks the same as a bad model.

## Solution

`buddhist-eval` is a small, dependency-light Python library. Its public functions take in-memory Python data, not files or URLs, and return typed result objects:

- `score_cer` and `score_cer_batch` score the CER of one Reference/Prediction pair or of a batch.
- `score_wer` and `score_wer_batch` do the same for WER.
- `score_spans` and `score_spans_batch` score Predicted Spans against Gold Spans with IoU-matched precision, recall and F1 at a chosen IoU Threshold.

Every CER and WER function takes a required `language`. For Tibetan (`"bo"`), text is converted to Wylie before scoring, exactly as the leaderboard does (ADR-0001). Span matching reproduces the quotation-detection algorithm exactly: same Tag only, greedy one-to-one by highest IoU, inclusive ends. So scores from this library match the numbers teams already report.

## User Stories

1. As an OCR researcher, I want to score one Prediction against its Reference with a single function call, so that I can check a model output without setting up a pipeline.
2. As an OCR researcher, I want CER and WER as separate functions, so that I only compute and depend on the metric I need.
3. As an OCR researcher, I want to pass `language="bo"` and have Tibetan handled correctly, so that I don't have to know about Wylie conversion.
4. As a speech-to-text researcher, I want the same CER and WER functions to work on transcripts, so that OCR and speech-to-text use one scoring standard.
5. As a researcher working in a language other than Tibetan, I want to pass that language's ISO 639-1 code and have the text scored unchanged, so that the library is still usable outside Tibetan.
6. As a leaderboard maintainer, I want the mean CER and mean WER of a batch to equal what the current EvalAI backend reports for the same data, so that switching to this library does not change published rankings.
7. As a leaderboard maintainer, I want the corpus CER and corpus WER (total edits divided by total Reference length) alongside the mean, so that long pages carry proportionally more weight when I need them to.
8. As a researcher, I want a per-item breakdown in a batch result, so that I can find the pages my model does worst on.
9. As a researcher, I want substitution, deletion and insertion counts on every CER and WER result, so that I can tell whether my model drops text or hallucinates text.
10. As a leaderboard maintainer, I want an empty Prediction to count as a CER and WER of 1.0, so that a model cannot improve its score by outputting nothing.
11. As a leaderboard maintainer, I want items whose Reference is empty to be skipped and listed in the result, so that bad ground truth is visible instead of breaking the run.
12. As a leaderboard maintainer, I want a Reference with no matching Prediction to count as an empty Prediction, so that incomplete submissions are penalised, not ignored.
13. As a leaderboard maintainer, I want Predictions with no matching Reference to be ignored and listed in the result, so that I can tell submitters they sent extra items.
14. As a service developer, I want a specific error raised on invalid input (for example an empty batch), not a silent worst-case score, so that I can tell bugs apart from bad models.
15. As a service developer, I want unrounded scores, so that I decide the display precision myself.
16. As a service developer, I want a loader for the existing EvalAI JSON format (a list of `filename`/`label` and `filename`/`prediction` objects), so that I can score current leaderboard files without writing parsing code.
17. As an annotation researcher, I want to score a document's Predicted Spans against its Gold Spans with one function call, so that I can evaluate span models outside the training repo.
18. As an annotation researcher, I want to choose the IoU Threshold, defaulting to 0.5, so that I can be strict or lenient about boundaries.
19. As an annotation researcher, I want an error when the IoU Threshold is outside (0, 1], so that a typo like `50` instead of `0.5` cannot silently score zero.
20. As an annotation researcher, I want a Predicted Span to only Match a Gold Span with the same Tag, so that a correctly placed span with the wrong Tag is not rewarded.
21. As an annotation researcher, I want each Gold Span and each Predicted Span to be used in at most one Match, so that one long prediction cannot be credited for several Gold Spans.
22. As an annotation researcher, I want span results to match `eval_quotation_v3.py` exactly on the same data, so that numbers in past reports stay comparable.
23. As an annotation researcher, I want precision, recall and F1 broken down per Tag, so that I can see which classes my model handles badly.
24. As an annotation researcher, I want the mean IoU of matched spans, so that I can tell whether my model finds spans but places their edges poorly.
25. As an annotation researcher, I want `n_gold`, `n_pred` and `n_match` on every result, so that I can spot over-proposal and the case where neither side has any spans.
26. As an annotation researcher, I want to score many documents at once with counts pooled across documents, so that I get one corpus-level F1.
27. As an annotation researcher, I want a missing predicted document to count all of its Gold Spans as missed, so that skipped documents lower recall.
28. As an annotation researcher, I want extra predicted documents with no gold to be ignored and listed in the result, so that I can catch id mismatches.
29. As an annotation researcher, I want an error if the gold and predicted documents have different texts, so that offset drift never produces a meaningless score.
30. As an annotation researcher, I want an error if a Span runs past the end of its text or ends before it starts, so that broken annotations are caught early.
31. As a newcomer to the codebase, I want the domain terms defined in one glossary, so that "Reference", "Prediction", "Gold Span" and "Tag" mean one thing everywhere.

## Implementation Decisions

### Package

- The template package `project_name` is renamed to `buddhist_eval`. It is distributed as `buddhist-eval` at version `0.1.0`. `pyproject.toml` is the single source of truth for metadata.
- Python 3.10 or later is required.
- The layout is a src-layout with `core/` for the metric logic and `utils/` for helpers (text preparation, file loading). `__init__.py` only re-exports the public API and the version.
- Runtime dependencies are `jiwer` (whose edit-distance work runs on `rapidfuzz`) and `pyewts`. Hugging Face `evaluate` is not a dependency. Its `cer` and `wer` metrics are wrappers around `jiwer`, and it adds heavy dependencies plus a network fetch at runtime, so calling `jiwer` directly gives the same numbers more cheaply.
- No HTTP service and no CLI in this RFC. The public Python functions are the API.

### Text scoring module (CER and WER)

- There are four public functions: `score_cer(reference, prediction, language)`, `score_cer_batch(references, predictions, language)`, `score_wer(reference, prediction, language)` and `score_wer_batch(references, predictions, language)`. CER and WER are fully independent: computing one never computes the other.
- `language` is a required ISO 639-1 code.
- Text preparation is the same for both metrics and both modes:
  1. Strip leading and trailing whitespace.
  2. If `language == "bo"`, convert to Wylie with `pyewts` (ADR-0001). In Wylie each tsek becomes a space, so WER counts syllables.
  3. Any other language code leaves the text unchanged.
- Each single-pair function returns an `ErrorRateScore` holding the rate, the substitution, deletion, insertion and hit counts, and the Reference length in the metric's units (characters for CER, words for WER).
- Each batch function takes two mappings from item id to text, one for References and one for Predictions, and returns an `ErrorRateBatchScore` holding:
  - `mean`: the average of each scored item's rate, which equals the leaderboard figure.
  - `corpus`: total edits across items divided by total Reference length.
  - `items`: the `ErrorRateScore` for each id.
  - `skipped_ids`: ids whose Reference is empty after stripping.
  - `extra_ids`: Prediction ids with no Reference.
- `ErrorRateScore` and `ErrorRateBatchScore` are shared by CER and WER. The function name says which metric was computed.
- An empty Prediction is scored, not skipped: every Reference unit counts as a deletion, giving a rate of 1.0. A Reference id with no Prediction is scored the same way.
- An empty Reference in a batch is skipped and its id recorded. In a single-pair call there is nothing to divide by, so it raises an error.
- An empty batch, or a batch where every item is skipped, raises an error. It does not return 1.0.
- Scores are not rounded.

### Span scoring module (IoU-F1)

- There are two public functions: `score_spans(gold, predicted, iou_threshold=0.5)` and `score_spans_batch(golds, predictions, iou_threshold=0.5)`.
- Input is a document object holding the text and its Spans. A Span holds `start`, `end` and `tag`, and its `end` is **inclusive**, so `(0, 4)` covers five characters. This is the opposite of Python slicing and deliberately matches the quotation-detection data.
- Validation raises an error if the gold and predicted documents' texts differ, or if any Span has `end < start`, a negative `start`, or `end` past the last character of the text.
- IoU is computed over inclusive ranges: intersection length divided by union length.
- Matching reproduces `match_iou` from the quotation-detection repo:
  1. Consider only pairs of Gold Span and Predicted Span with the same Tag.
  2. Keep pairs whose IoU is at or above the IoU Threshold.
  3. Sort them in descending order of (IoU, gold index, predicted index).
  4. Walk the list greedily, accepting a pair only if neither span has been used yet.
- Precision, recall and F1 reproduce `prf` from the same repo: each ratio is 0.0 when its denominator is 0. So when there are no Gold Spans and no Predicted Spans, all three are 0.0. `n_gold` and `n_pred` make that case visible.
- Each call returns a `SpanScore` holding:
  - the IoU Threshold
  - precision, recall and F1
  - `n_gold`, `n_pred` and `n_match`
  - the mean IoU of the Matches, which is 0.0 when there are none
  - `per_tag`: a `TagScore` for each Tag, with the same counts and ratios for that Tag
- `iou_threshold` is a single float in (0, 1]. Anything else raises an error.
- The batch function takes two mappings from document id to document. It pools the Match, gold and predicted counts across documents before computing the ratios. A missing predicted document adds its Gold Spans to `n_gold` with no Matches. Extra predicted ids are ignored and listed on the batch result.

### Errors

- Every failure raises a specific exception from one package-level hierarchy: one base error, with subclasses for empty input, misaligned texts and invalid Spans. Nothing is caught and turned into a score.

### Loading files

- A helper in `utils` reads the EvalAI JSON format, a list of `{filename, label}` or `{filename, prediction}` objects, into the id-to-text mappings the batch functions take.
- No span loader yet. The quotation gold file has no document text, and the span API needs it.

## Testing Decisions

- Tests exercise only the six public functions and the EvalAI loader. No test imports from `core/` or `utils/` internals, so the implementation can change freely underneath.
- A good test states one behaviour in its name and checks only returned values or raised exception types, never internal call patterns.
- **Parity tests** run small fixed Tibetan fixtures. They assert that:
  - `score_cer_batch(...).mean` and `score_wer_batch(...).mean` equal the figures from the EvalAI backend's method (`evaluate` `cer`/`wer` on `pyewts` Wylie, averaged per item) on data with no empty items;
  - `score_spans` and `score_spans_batch` equal `match_iou` plus `prf` from `eval_quotation_v3.py`'s helpers.

  The expected numbers are computed once with the reference code and stored as fixtures, so the reference repos are not test dependencies.
- **Behaviour tests**, one for each decided edge case:
  - an empty Prediction, and a missing Prediction id
  - an empty Reference, both in a batch and as a single pair
  - an empty batch, and extra ids
  - a non-Tibetan language passing through unchanged
  - Tag mismatch, and one-to-one matching under contention
  - tie-breaking order, and inclusive-end IoU
  - no spans on either side, and an out-of-range threshold
  - a missing predicted document, and extra predicted documents
  - mismatched texts, and out-of-range or inverted Spans
- There's no prior art: the repo has only the template's example test, which is deleted along with the example module. pytest is already a dev dependency.

## Out of Scope

- An HTTP service or REST endpoint, and a CLI.
- Downloading from S3 or URLs, and writing results to a database. These stay in the EvalAI backend, which will call this library.
- Text normalization beyond stripping whitespace and Wylie conversion: Unicode normalization, punctuation, tsek variants and shad handling.
- Grapheme-cluster CER and code-point CER for Tibetan (ADR-0001 explains why Wylie is used instead).
- Aligning span offsets between two different texts, for example when OCR output is tagged.
- BIOE and BIO token sequences, Viterbi decoding, and model logits.
- MUC COR/PAR/INC/MIS/SPU, boundary MAE, coverage histograms, per-book reports, and "excluding the largest book" scores.
- A span file loader.
- Scoring several IoU Thresholds in one call. Callers who want a sweep call `score_spans` once per threshold.

## Further Notes

- Domain terms used here are defined in `GLOSSARY.md`. The Wylie decision is recorded in `docs/adr/0001-tibetan-scored-on-wylie.md`.
- References: `WeBuddhist/openpecha_evalai_backend` (`Evaluation/evaluation.py`) for CER and WER; `WeBuddhist/Tibetan-quotation-detection` (`script/eval_quotation_v3.py` and `src/layer_detection/v2_metrics.py`) for span matching and precision, recall and F1.
- Mean CER will differ from the current leaderboard on submissions that contain empty Predictions or missing items, because those are now penalised instead of skipped. That change is intended (user stories 10 and 12) and should be announced when the leaderboard switches over.
- An empty Reference in a single-pair call raises an error, while in a batch it is skipped and listed. The difference is deliberate: a batch should not fail because of one bad ground-truth item.
