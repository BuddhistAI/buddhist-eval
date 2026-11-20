"""buddhist-eval: CER, WER and Span IoU-F1 scoring for Buddhist text pipelines."""

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
    "InvalidSpanError",
    "MisalignedTextsError",
    "__version__",
]
