"""Smoke tests for the public package foundation."""

import buddhist_eval
from buddhist_eval import (
    BuddhistEvalError,
    EmptyInputError,
    InvalidSpanError,
    MisalignedTextsError,
)


def test_package_version_is_0_1_0() -> None:
    """buddhist_eval exposes version 0.1.0."""
    assert buddhist_eval.__version__ == "0.1.0"


def test_error_classes_are_exported_and_share_base() -> None:
    """Empty-input, misaligned-text and invalid-span errors share BuddhistEvalError."""
    assert issubclass(EmptyInputError, BuddhistEvalError)
    assert issubclass(MisalignedTextsError, BuddhistEvalError)
    assert issubclass(InvalidSpanError, BuddhistEvalError)
    assert issubclass(BuddhistEvalError, Exception)
