"""buddhist-eval: CER, WER and Span IoU-F1 scoring for Buddhist text pipelines."""

from buddhist_eval.core.error_rates import (
    ErrorRateBatchScore,
    ErrorRateScore,
    score_cer,
    score_cer_batch,
    score_wer,
    score_wer_batch,
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
    "EmptyInputError",
    "ErrorRateBatchScore",
    "ErrorRateScore",
    "InvalidSpanError",
    "MisalignedTextsError",
    "score_cer",
    "score_cer_batch",
    "score_wer",
    "score_wer_batch",
    "__version__",
]
