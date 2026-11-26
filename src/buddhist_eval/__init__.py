"""buddhist-eval: CER, WER, Span IoU-F1, TOC and MCQ scoring for Buddhist text pipelines."""

from buddhist_eval.core.error_rates import (
    ErrorRateBatchScore,
    ErrorRateScore,
    score_cer,
    score_cer_batch,
    score_wer,
    score_wer_batch,
)
from buddhist_eval.core.mcq import McqScore, score_mcq
from buddhist_eval.core.registry import get_scorer, list_scorers
from buddhist_eval.core.scorer import (
    Direction,
    DisplayFormat,
    DisplayStyle,
    Parameter,
    Scorer,
    ScoreResult,
)
from buddhist_eval.core.spans import (
    Document,
    Span,
    SpanBatchScore,
    SpanScore,
    TagScore,
    score_spans,
    score_spans_batch,
)
from buddhist_eval.core.toc import (
    Heading,
    PredictedHeading,
    TocBatchScore,
    TocPassage,
    TocScore,
    score_toc,
    score_toc_batch,
)
from buddhist_eval.errors import (
    BuddhistEvalError,
    EmptyInputError,
    IdMismatchError,
    InvalidHeadingError,
    InvalidInputError,
    InvalidParameterError,
    InvalidPredictionError,
    InvalidReferenceError,
    InvalidSpanError,
    MalformedEvalAIFileError,
    MalformedJsonlError,
    MisalignedTextsError,
    UnknownScorerError,
)
from buddhist_eval.utils.evalai import load_evalai_json
from buddhist_eval.utils.jsonl import (
    load_ground_truth,
    load_inputs,
    load_media_paths,
    load_predictions,
)

__version__ = "0.2.0"

__all__ = [
    "BuddhistEvalError",
    "Direction",
    "DisplayFormat",
    "DisplayStyle",
    "Document",
    "EmptyInputError",
    "ErrorRateBatchScore",
    "ErrorRateScore",
    "Heading",
    "IdMismatchError",
    "InvalidHeadingError",
    "InvalidInputError",
    "InvalidParameterError",
    "InvalidPredictionError",
    "InvalidReferenceError",
    "InvalidSpanError",
    "MalformedEvalAIFileError",
    "MalformedJsonlError",
    "McqScore",
    "MisalignedTextsError",
    "Parameter",
    "PredictedHeading",
    "ScoreResult",
    "Scorer",
    "Span",
    "SpanBatchScore",
    "SpanScore",
    "TagScore",
    "TocBatchScore",
    "TocPassage",
    "TocScore",
    "UnknownScorerError",
    "get_scorer",
    "list_scorers",
    "load_evalai_json",
    "load_ground_truth",
    "load_inputs",
    "load_media_paths",
    "load_predictions",
    "score_cer",
    "score_cer_batch",
    "score_mcq",
    "score_spans",
    "score_spans_batch",
    "score_toc",
    "score_toc_batch",
    "score_wer",
    "score_wer_batch",
    "__version__",
]
