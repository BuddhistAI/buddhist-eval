"""Package-level error hierarchy for buddhist-eval."""


class BuddhistEvalError(Exception):
    """Base error for all buddhist-eval failures."""


class EmptyInputError(BuddhistEvalError):
    """Raised when required input is empty (for example an empty Reference)."""


class MisalignedTextsError(BuddhistEvalError):
    """Raised when gold and predicted document texts differ."""


class InvalidSpanError(BuddhistEvalError):
    """Raised when a Span is out of range or inverted."""


class InvalidHeadingError(BuddhistEvalError):
    """Raised when a Gold Heading has a depth below 1 or a start outside its text."""


class MalformedEvalAIFileError(BuddhistEvalError):
    """Raised when an EvalAI JSON file is invalid or inconsistently structured."""
