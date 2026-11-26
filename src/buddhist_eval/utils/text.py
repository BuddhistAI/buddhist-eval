"""Text preparation helpers for CER and WER scoring."""

import pyewts

_EWTS = pyewts.pyewts()


def prepare_text(text: str, language: str) -> str:
    """Strip whitespace and, for Tibetan, convert to Wylie.

    Args:
        text: Raw Reference or Prediction text.
        language: ISO 639-1 language code; ``"bo"`` selects Wylie conversion.

    Returns:
        Prepared text ready for edit-distance scoring.
    """
    stripped = text.strip()
    if language == "bo":
        return _EWTS.toWylie(stripped)
    return stripped
