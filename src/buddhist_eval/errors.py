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


class UnknownScorerError(BuddhistEvalError):
    """Raised when no Scorer is registered under the requested name."""


class InvalidParameterError(BuddhistEvalError):
    """Raised when a Scorer parameter is unknown, mistyped or out of range."""


class InvalidPredictionError(BuddhistEvalError):
    """Raised when a Prediction value does not match the Scorer's prediction schema."""


class InvalidReferenceError(BuddhistEvalError):
    """Raised when a Reference value does not match the Scorer's reference schema."""


class InvalidInputError(BuddhistEvalError):
    """Raised when an Input value does not match the Scorer's input schema, or is missing."""


class MalformedJsonlError(BuddhistEvalError):
    """Raised when an Inputs, Ground Truth or Predictions JSONL file cannot be read."""


class IdMismatchError(BuddhistEvalError):
    """Raised when two id-keyed files do not hold exactly the same ids.

    Attributes:
        missing_ids: Expected ids that are absent, sorted.
        extra_ids: Ids present that were not expected, sorted.
    """

    def __init__(
        self, message: str, missing_ids: tuple[str, ...], extra_ids: tuple[str, ...]
    ) -> None:
        super().__init__(message)
        self.missing_ids = missing_ids
        self.extra_ids = extra_ids
