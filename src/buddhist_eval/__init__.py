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
    InvalidHeadingError,
    InvalidSpanError,
    MalformedEvalAIFileError,
    MisalignedTextsError,
)
from buddhist_eval.utils.evalai import load_evalai_json

__version__ = "0.1.0"

__all__ = [
    "BuddhistEvalError",
    "Document",
    "EmptyInputError",
    "ErrorRateBatchScore",
    "ErrorRateScore",
    "Heading",
    "InvalidHeadingError",
    "InvalidSpanError",
    "MalformedEvalAIFileError",
    "McqScore",
    "MisalignedTextsError",
    "PredictedHeading",
    "Span",
    "SpanBatchScore",
    "SpanScore",
    "TagScore",
    "TocBatchScore",
    "TocPassage",
    "TocScore",
    "load_evalai_json",
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
