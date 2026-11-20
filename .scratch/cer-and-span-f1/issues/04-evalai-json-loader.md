# 04: Load EvalAI JSON files for batch scoring

**What to build:** A service developer can point a loader at the existing EvalAI ground-truth and submission files and pass the results straight to `score_cer_batch` and `score_wer_batch`. The ground-truth file is a JSON list of `{filename, label}` objects, and the submission file a list of `{filename, prediction}` objects. Scoring a current leaderboard submission then takes a few lines with no parsing code. See the RFC (Implementation Decisions › Loading files).

**Blocked by:** 03 (batch CER and WER).

**Status:** resolved

- [x] A public loader, exported from `buddhist_eval`, reads a local EvalAI JSON file, given as a `pathlib.Path`, into a mapping from `filename` to text. It reads `label` for ground truth and `prediction` for submissions.
- [x] A single top-level object, not a list, is accepted as a one-item list, as the EvalAI backend does.
- [x] Malformed input raises a specific package error that names the file. Malformed means invalid JSON, an entry missing `filename` or the text field, or a duplicate `filename`.
- [x] No network or S3 access.
- [x] End-to-end test: small ground-truth and submission fixture files are loaded and passed to `score_cer_batch`, and the result matches the expected fixture score.
- [x] Behaviour tests cover each malformed-input case.

## Comments

- Merged via `impl/04-evalai-json-loader` at 4546720.
