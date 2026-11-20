"""Package-level error hierarchy for buddhist-eval."""


class BuddhistEvalError(Exception):
    """Base error for all buddhist-eval failures."""


class EmptyInputError(BuddhistEvalError):
    """Raised when required input is empty (for example an empty Reference)."""


class MisalignedTextsError(BuddhistEvalError):
    """Raised when gold and predicted document texts differ."""


class InvalidSpanError(BuddhistEvalError):
    """Raised when a Span is out of range or inverted."""
