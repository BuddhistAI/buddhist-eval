<h1 align="center">
  <br>
  <a href="https://openpecha.org"><img src="https://avatars.githubusercontent.com/u/82142807?s=400&u=19e108a15566f3a1449bafb03b8dd706a72aebcd&v=4" alt="OpenPecha" width="150"></a>
  <br>
  buddhist-eval
</h1>

<p align="center">
  CER, WER, and span IoU F1 for Tibetan OCR, speech-to-text, and span annotation.
</p>

<p align="center">
  <a href="#what-it-does">What it does</a> •
  <a href="#who-this-is-for">Who this is for</a> •
  <a href="#install">Install</a> •
  <a href="#character-and-word-error-rate">CER and WER</a> •
  <a href="#span-iou-f1">Span IoU F1</a> •
  <a href="#toc-heading-score">TOC</a> •
  <a href="#multiple-choice-accuracy">MCQ</a> •
  <a href="#load-evalai-json">EvalAI JSON</a> •
  <a href="#errors">Errors</a> •
  <a href="#how-to-get-help">Help</a>
</p>

`buddhist-eval` scores a model's output against human ground truth. It is a small Python library: functions take in-memory data and return typed results. There is no CLI and no HTTP service.

Scores stay comparable with the numbers teams already publish. Tibetan text is converted to Wylie before CER and WER, the same way the OpenPecha EvalAI leaderboard does. Span matching follows the quotation-detection evaluator: same tag only, inclusive character offsets, greedy one-to-one matching by IoU.

| Term | Meaning |
| --- | --- |
| Reference | Ground-truth text a human produced for one item. |
| Prediction | Text a model produced for that same item. |
| Gold Span | A labelled character range from the ground truth. |
| Predicted Span | A labelled character range from the model. |
| Tag | The label on a span. A predicted span matches a gold span only when the tags are the same. |
| Language | ISO 639-1 code. `"bo"` converts both texts to Wylie before scoring. |

Domain terms are defined in [GLOSSARY.md](GLOSSARY.md). The Wylie decision is recorded in [docs/adr/0001-tibetan-scored-on-wylie.md](docs/adr/0001-tibetan-scored-on-wylie.md).

## Owner

- [OpenPecha](https://github.com/OpenPecha) (`dev@openpecha.org`)

## What it does

- `score_cer` and `score_wer` score one Reference against one Prediction.
- `score_cer_batch` and `score_wer_batch` score a mapping of item ids and return a per-item mean plus a corpus rate.
- `score_spans` and `score_spans_batch` score Predicted Spans against Gold Spans and return precision, recall, and F1.
- `score_toc` and `score_toc_batch` score extracted outline headings by position and depth.
- `score_mcq` scores multiple-choice answers against an Answer Key and returns accuracy.
- `load_evalai_json` reads the existing EvalAI file shape into the mapping the batch scorers take.
- `list_scorers` and `get_scorer` expose every metric as a named Scorer with one `score(inputs, references, predictions, **params)` call that returns a Headline Metric plus JSON details, for the challenge backend.
- `load_inputs`, `load_ground_truth` and `load_predictions` read the arena's JSONL files.

CER and WER are independent. Calling one never computes the other. Scores are unrounded.

## Who this is for

- OCR and speech-to-text researchers who want one CER or WER call, including Tibetan.
- Leaderboard maintainers who need the EvalAI mean, plus a length-weighted corpus rate and a per-item breakdown.
- Annotation researchers who want IoU-matched span F1 outside the quotation-detection training repo.

## Install

Python 3.10 or later. Runtime dependencies are `jiwer` and `pyewts`.

```bash
pip install .
```

For tests and pre-commit:

```bash
pip install ".[dev]"
PYTHONPATH=src pytest
```

## Character and word error rate

Both metrics use the same preparation and the same ratio. The unit changes.

```text
rate = (substitutions + deletions + insertions) / reference_length
reference_length = hits + substitutions + deletions
```

CER counts characters. WER counts whitespace-separated words. For Tibetan, Wylie turns each tsek into a space, so a WER word is a syllable.

### How CER and WER are calculated

```mermaid
flowchart TD
  pair["Reference and Prediction plus an ISO 639-1 language code"]
  pair --> strip["Strip leading and trailing whitespace on both texts"]
  strip --> lang{"language is bo?"}
  lang -->|yes| wylie["Convert both texts to Wylie with pyewts"]
  lang -->|no| ready["Keep the stripped text"]
  wylie --> empty{"Reference empty after preparation?"}
  ready --> empty
  empty -->|yes| mode{"Single pair or batch item?"}
  mode -->|single pair| raiseEmpty["Raise EmptyInputError"]
  mode -->|batch item| skip["Skip the item and record its id"]
  empty -->|no| unit{"Which function?"}
  unit -->|score_cer| chars["Align characters with jiwer"]
  unit -->|score_wer| words["Split on whitespace and align words with jiwer"]
  chars --> counts["Count substitutions deletions insertions and hits"]
  words --> counts
  counts --> rate["rate equals S plus D plus I over reference length"]
  rate --> result["ErrorRateScore with rate edit counts and reference length"]
```

An empty Prediction is scored. Every Reference unit is a deletion, so the rate is `1.0`. A Reference id with no Prediction is scored the same way. A Prediction id with no Reference is ignored and listed in `extra_ids`.

### Single pair

```python
from buddhist_eval import score_cer, score_wer

cer = score_cer("བོད་ཡིག", "བོད་སྐད", language="bo")
wer = score_wer("བོད་ཡིག", "བོད་སྐད", language="bo")
```

`བོད་ཡིག` is Wylie `bod yig` and `བོད་སྐད` is `bod skad`. WER is one substitution out of two syllables, so `wer.rate` is `0.5`. `language="en"`, `"fr"`, or any code other than `"bo"` scores the stripped text with no Wylie conversion.

`ErrorRateScore` fields:

| Field | Meaning |
| --- | --- |
| `rate` | CER or WER, unrounded. |
| `substitutions`, `deletions`, `insertions`, `hits` | Edit counts from the alignment. |
| `reference_length` | Characters for CER, words for WER. |

### Batch

```python
from buddhist_eval import score_cer_batch, score_wer_batch

references = {"page-1": "abcd", "page-2": "ef gh"}
predictions = {"page-1": "abxd", "page-3": "extra"}

batch = score_cer_batch(references, predictions, language="en")
# batch.mean    average of each scored item's rate
# batch.corpus  total edits / total reference length
# batch.items   ErrorRateScore per scored id
# batch.skipped_ids  empty References
# batch.extra_ids    predictions with no Reference ("page-3")
```

`mean` gives every scored item equal weight and matches the EvalAI leaderboard average. `corpus` weights long items by their Reference length: total substitutions, deletions, and insertions divided by total Reference length. `score_wer_batch` returns the same shape.

An empty batch, or a batch whose every Reference is empty after preparation, raises `EmptyInputError`. A lone tsek (`"་"`) is empty for Tibetan because Wylie turns it into whitespace.

## Span IoU F1

A `Span` is `start`, `end`, and `tag`. `end` is inclusive, so `(0, 4)` covers five characters. A `Document` is the text plus its spans. Gold and predicted documents must carry the same text, because offsets are character indexes into that string.

A Match is a one-to-one pairing of one Gold Span and one Predicted Span with the same Tag whose intersection-over-union is at least the IoU threshold. The default threshold is `0.5`. It must be in `(0, 1]`.

```text
intersection = overlap of the two inclusive ranges
union        = gold length + predicted length - intersection
IoU          = intersection / union

precision = n_match / n_pred    (0 when n_pred is 0)
recall    = n_match / n_gold    (0 when n_gold is 0)
F1        = 2 * precision * recall / (precision + recall)   (0 when both are 0)
```

### How IoU F1 is calculated

```mermaid
flowchart TD
  docs["Gold Document and Predicted Document"]
  docs --> threshold{"IoU threshold inside 0 exclusive to 1 inclusive?"}
  threshold -->|no| badThreshold["Raise BuddhistEvalError"]
  threshold -->|yes| sameText{"Gold text equals predicted text?"}
  sameText -->|no| misaligned["Raise MisalignedTextsError"]
  sameText -->|yes| valid{"Every span starts at 0 or later and ends inside the text?"}
  valid -->|no| badSpan["Raise InvalidSpanError"]
  valid -->|yes| candidates["Pair each Gold Span with each Predicted Span that shares its Tag"]
  candidates --> iou["Inclusive IoU equals intersection length over union length"]
  iou --> keep["Keep pairs whose IoU meets the threshold"]
  keep --> sort["Sort descending by IoU then gold index then predicted index"]
  sort --> greedy["Accept a pair only when neither span is already used"]
  greedy --> ratios["Precision recall and F1 from match gold and predicted counts"]
  ratios --> extras["Mean IoU of the Matches and the same ratios per Tag"]
  extras --> spanScore["SpanScore"]
```

When a document has no Gold Spans and no Predicted Spans, precision, recall, and F1 are `0.0`. `n_gold` and `n_pred` show that case. `mean_iou` is `0.0` when there are no Matches.

### Single document

```python
from buddhist_eval import Document, Span, score_spans

text = "0123456789"
gold = Document(text=text, spans=(Span(0, 4, "quote"),))
predicted = Document(text=text, spans=(Span(0, 4, "quote"),))

result = score_spans(gold, predicted)          # threshold 0.5
# result.f1 == 1.0, result.mean_iou == 1.0

strict = score_spans(gold, predicted, iou_threshold=0.8)
```

`SpanScore` fields: `iou_threshold`, `precision`, `recall`, `f1`, `n_gold`, `n_pred`, `n_match`, `mean_iou`, and `per_tag`. Each `TagScore` repeats precision, recall, F1, and the three counts for one Tag. Tags are every label that appears on either side.

### Batch

```python
from buddhist_eval import score_spans_batch

golds = {"doc-1": gold}
predictions = {"doc-1": predicted, "doc-9": predicted}

batch = score_spans_batch(golds, predictions, iou_threshold=0.5)
```

Counts are pooled across documents before precision, recall, and F1 are computed, so the result is one corpus-level score. A gold id with no predicted document adds its Gold Spans to `n_gold` and contributes no Matches, which lowers recall. Predicted ids with no gold are ignored and listed in `extra_ids`.

## TOC heading score

Scores outline headings (ས་བཅད, sa-bcad) extracted from a passage. This is the metric of the `toc-extraction/bo` task in [buddhist-benchmarks](https://github.com/BuddhistAI/buddhist-benchmarks), as implemented by `evaluate_headings` in [toc-extraction-tibetan](https://github.com/BuddhistAI/toc-extraction-tibetan/blob/main/benchmark/gepa_toc.py). Scores are identical to that implementation.

A `Heading` is a Gold Heading: `depth` (1 is the shallowest level in the passage) and `start`, its character offset in the text. A `PredictedHeading` is `depth` and `anchor`, the verbatim text the heading starts with. A `TocPassage` is the text plus its Gold Headings in reading order. Titles are not scored.

1. Each anchor is stripped and searched verbatim in the text, starting one character after the previous located anchor. An anchor that is not found still counts as a prediction but cannot match.
2. In prediction order, each located heading matches the nearest unused Gold Heading whose `start` is within `tolerance` characters (default `3`). Each Gold Heading matches at most once.
3. Two F1 scores are computed over all predictions and all Gold Headings: position F1 counts every match, and depth F1 counts only matches with the same depth.

```text
score = 0.3 * position_f1 + 0.7 * depth_f1
F1    = 1.0 when there are no predictions and no Gold Headings
score = max(0, 1 - 0.25 * n_pred) when the passage has no Gold Headings
```

```python
from buddhist_eval import Heading, PredictedHeading, TocPassage, score_toc

passage = TocPassage(text="abcdefghij0123456789", headings=(Heading(1, 0), Heading(2, 10)))
result = score_toc(passage, (PredictedHeading(1, "abc"), PredictedHeading(1, "012")))
# result.position_f1 == 1.0, result.depth_f1 == 0.5, result.score is about 0.65
```

`TocScore` fields: `score`, `position_f1`, `depth_f1`, `n_gold`, `n_pred`, `n_located`, `n_position_match`, and `n_depth_match`.

`score_toc_batch(passages, predictions)` takes mappings from passage id to `TocPassage` and to a tuple of `PredictedHeading`. It returns `TocBatchScore` with `mean` (every passage weighted equally, as in the benchmark), per-passage `items`, and `extra_ids`. A passage with no prediction is scored as an empty prediction.

## Multiple-choice accuracy

Scores a multiple-choice benchmark against an Answer Key: a mapping from question id to the correct choice. Each predicted choice is compared to the key by exact match after stripping whitespace and ignoring case, so `" b "` matches `"B"`. The model's choice must already be extracted; free-text answers such as `"The answer is B"` are not parsed.

```text
accuracy = n_correct / n_total
n_total  = every question in the Answer Key
```

```python
from buddhist_eval import score_mcq

answer_key = {"q1": "A", "q2": "C", "q3": "B"}
predictions = {"q1": "A", "q2": "D", "q9": "A"}

result = score_mcq(answer_key, predictions)
# result.accuracy == 1 / 3   (multiply by 100 for a percentage)
# result.items       {"q1": True, "q2": False, "q3": False}
# result.missing_ids ("q3",)  counted as wrong
# result.extra_ids   ("q9",)  ignored
```

With 89 correct out of 100 questions, `result.accuracy` is `0.89`, or 89%. `McqScore` fields: `accuracy`, `n_correct`, `n_total`, `items`, `missing_ids`, and `extra_ids`. An empty Answer Key, or a blank answer in it, raises `EmptyInputError`.

## Load EvalAI JSON

The loader accepts a list of `{filename, label}` or `{filename, prediction}` objects, or one such object. `label` is the Reference field and `prediction` is the Prediction field.

```python
from pathlib import Path

from buddhist_eval import load_evalai_json, score_cer_batch

references = load_evalai_json(Path("ground_truth.json"))
predictions = load_evalai_json(Path("submission.json"))
batch = score_cer_batch(references, predictions, language="bo")
```

Duplicate filenames, a missing `filename`, a missing text field, or invalid JSON raise `MalformedEvalAIFileError`. There is no span-file loader: the span API needs the document text, and the quotation gold files do not include it.

## Scorer registry

The challenge backend never calls the metric functions directly. It lists the registered Scorers to fill the scorer dropdown, stores a Scorer's name, params and Direction on the challenge, validates the uploaded files against the Scorer, and scores each submission with `get_scorer(name).score(...)`.

Each `Scorer` has:

| Field | Meaning |
| --- | --- |
| `name` | Registry key stored on the challenge, for example `"cer"`. Never reused for a different computation. |
| `label` | Display name for the dropdown. |
| `direction` | `Direction.LOWER_IS_BETTER` or `Direction.HIGHER_IS_BETTER`; `.value` is the `challenge.json` string. |
| `parameter_schema` | JSON Schema for the params object. Every parameter is optional and has a default. |
| `input_schema` | JSON Schema for one Input value. CER, WER and MCQ accept any value and ignore it. |
| `reference_schema` | JSON Schema for one Reference value. |
| `prediction_schema` | JSON Schema for one Prediction value. |
| `sample_prediction` | A placeholder Prediction of the right shape, for `sample_submission.jsonl`. |
| `score(inputs, references, predictions, **params)` | Returns a `ScoreResult`. |
| `validate_ground_truth(inputs, references, **params)` | Checks a challenge before it is saved. |
| `validate_predictions(inputs, predictions)` | Checks a Predictions File before it is scored. |
| `describe()` | All of the above as plain JSON, for a listing endpoint. |

`inputs`, `references` and `predictions` each map item id to a decoded JSON value. `score` returns a `ScoreResult` with:

- `headline_metric`: the unrounded float a leaderboard ranks by, sorted by `direction`.
- `details`: the metric's full result object as plain JSON (dicts, lists, numbers, strings). `result.to_dict()` gives both, ready to store.

| Name | Direction | Params (default) | Input | Reference | Prediction | Headline Metric | Details |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `cer` | lower | `language` (`"bo"`) | ignored | string | string | `score_cer_batch` `mean` | `corpus`, per-item `items`, `skipped_ids`, `extra_ids` |
| `wer` | lower | `language` (`"bo"`) | ignored | string | string | `score_wer_batch` `mean` | same as `cer` |
| `span_f1` | higher | `iou_threshold` (`0.5`) | passage text | `[{start, end, tag}]` | `[{start, end, tag}]` | `score_spans_batch` `f1` | precision, recall, counts, `per_tag`, `extra_ids` |
| `toc` | higher | `tolerance` (`3`) | passage text | `[{depth, start}]` | `[{depth, anchor}]` | `score_toc_batch` `mean` | per-passage `items`, `extra_ids` |
| `mcq_accuracy` | higher | none | ignored | string (the correct choice) | string | `score_mcq` `accuracy` | `n_correct`, `n_total`, `items`, `missing_ids`, `extra_ids` |

For CER and WER the Headline Metric is the per-item `mean`, which matches the EvalAI leaderboard; the length-weighted `corpus` rate is a detail. Span and heading offsets index into the Input's passage text. Each Scorer returns exactly the numbers of the batch function it wraps.

### JSONL files

All arena data is JSONL, one object per line, keyed by a string `id`:

| File | Row | Loader |
| --- | --- | --- |
| Inputs | `{"id": ..., "input": <any>, "media": "<path>"}` (`media` optional) | `load_inputs` (id to `input`), `load_media_paths` (id to `media`) |
| Ground Truth | `{"id": ..., "reference": <value>}` | `load_ground_truth` |
| Predictions | `{"id": ..., "prediction": <value>}` | `load_predictions` |

Each loader takes a path or an open text or binary stream and returns a dict in file order. Blank lines and a UTF-8 byte-order mark are ignored. Invalid JSON, a row that is not an object, a missing or blank `id`, a duplicate `id`, a missing value key or any other key raises `MalformedJsonlError` naming the line. The loaders check only row structure; the Scorer checks value shapes.

```python
from buddhist_eval import (
    get_scorer,
    list_scorers,
    load_ground_truth,
    load_inputs,
    load_predictions,
)

dropdown = [scorer.describe() for scorer in list_scorers()]  # JSON-ready

# Creating a challenge
cer = get_scorer("cer")
params = cer.validate_params({"language": "bo"})  # fills defaults
inputs = load_inputs("inputs.jsonl")
references = load_ground_truth("ground_truth.jsonl")
cer.validate_ground_truth(inputs, references, **params)
sample = [{"id": i, "prediction": cer.sample_prediction} for i in inputs]

# Scoring a submission
predictions = load_predictions("predictions.jsonl")
cer.validate_predictions(inputs, predictions)
result = cer.score(inputs, references, predictions, **params)
result.headline_metric  # rank by this, using cer.direction
result.details["corpus"]  # every other number
```

`validate_ground_truth` requires the Inputs and the Ground Truth to hold exactly the same ids, every value to match its schema, and the Ground Truth to be scorable (for example every Gold Span inside its passage, no blank answer). `validate_predictions` requires exactly one Prediction per Input id, each matching `prediction_schema`.

An unregistered name raises `UnknownScorerError`. An unknown or invalid parameter raises `InvalidParameterError`. Differing id sets raise `IdMismatchError`, whose `missing_ids` and `extra_ids` list them in full. A value that does not match its schema raises `InvalidInputError`, `InvalidReferenceError` or `InvalidPredictionError`, so the backend can tell a bad submission from bad challenge data.

### Scorer names are immutable

A released Scorer name always computes the same thing. A release that changes an existing Scorer's numbers is a breaking change; the new computation is registered under a new name such as `cer_v2`. Challenges record the package version for audit only. See [docs/adr/0002-scorer-names-are-immutable.md](docs/adr/0002-scorer-names-are-immutable.md).

## Errors

Every failure raises `BuddhistEvalError` or a subclass. Invalid input is never turned into a score.

| Exception | When |
| --- | --- |
| `EmptyInputError` | Empty Reference on a single pair, an empty batch, a batch with nothing left to score, or an empty or blank Answer Key entry. |
| `MisalignedTextsError` | Gold and predicted document texts differ. |
| `InvalidSpanError` | A span is negative, ends before it starts, or runs past the text. |
| `InvalidHeadingError` | A Gold Heading has a depth below 1 or a start outside its text. |
| `MalformedEvalAIFileError` | The EvalAI JSON cannot be read into a filename-to-text mapping. |
| `MalformedJsonlError` | An Inputs, Ground Truth or Predictions JSONL file cannot be read into id-keyed rows. |
| `UnknownScorerError` | `get_scorer` is given a name that is not registered. |
| `InvalidParameterError` | A Scorer parameter is unknown, has the wrong type, is not an allowed choice, or is out of range. |
| `IdMismatchError` | Inputs and Ground Truth, or Inputs and Predictions, do not hold exactly the same ids. |
| `InvalidInputError` | An Input value does not match `input_schema`, or a Scorer that reads Inputs has no Input for a Reference id. |
| `InvalidPredictionError` | A Prediction value does not match `prediction_schema`. |
| `InvalidReferenceError` | A Reference value does not match `reference_schema`, or the Ground Truth cannot be scored. |
| `BuddhistEvalError` | `iou_threshold` is outside `(0, 1]`, or `tolerance` is negative. |

## Development

```bash
pip install ".[dev]"
PYTHONPATH=src pytest
PYTHONPATH=src pytest --cov buddhist_eval
```

Public entry points live in `buddhist_eval`. Metric logic is in `src/buddhist_eval/core/`. The Scorer type and registry are in `core/scorer.py` and `core/registry.py`. Text preparation and the EvalAI and JSONL loaders are in `src/buddhist_eval/utils/`. Tests call only the public functions.

## Additional documentation

- [GLOSSARY.md](GLOSSARY.md) — Reference, Prediction, Input, CER, WER, Span, Tag, Match, IoU Threshold, Scorer, Headline Metric, Direction.
- [docs/adr/0001-tibetan-scored-on-wylie.md](docs/adr/0001-tibetan-scored-on-wylie.md) — why Tibetan is scored on Wylie.
- [docs/adr/0002-scorer-names-are-immutable.md](docs/adr/0002-scorer-names-are-immutable.md) — why a changed computation gets a new Scorer name.
- [docs/help.md](docs/help.md) — where to ask questions.

## How to get help

- File an issue on the [bug tracker](https://github.com/BuddhistAI/buddhist-eval/issues).
- Email openpecha[at]gmail.com.
- Join the [OpenPecha Discord](https://discord.com/invite/7GFpPFSTeA).

## Terms of use

`buddhist-eval` is licensed under the [MIT License](LICENSE).
