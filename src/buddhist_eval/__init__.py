"""buddhist-eval: CER, WER and Span IoU-F1 scoring for Buddhist text pipelines."""

from buddhist_eval.core.spans import (
    Document,
    Span,
    SpanScore,
    TagScore,
    score_spans,
)
from buddhist_eval.errors import (
    BuddhistEvalError,
    EmptyInputError,
    InvalidSpanError,
    MisalignedTextsError,
)

__version__ = "0.1.0"

__all__ = [
    "BuddhistEvalError",
    "Document",
    "EmptyInputError",
    "InvalidSpanError",
    "MisalignedTextsError",
    "Span",
    "SpanScore",
    "TagScore",
    "score_spans",
    "__version__",
]
