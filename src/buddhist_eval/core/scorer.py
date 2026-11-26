"""Scorer, Parameter and ScoreResult types behind the scorer registry."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from typing import Any

from buddhist_eval.errors import (
    BuddhistEvalError,
    IdMismatchError,
    InvalidInputError,
    InvalidParameterError,
    InvalidPredictionError,
    InvalidReferenceError,
)
from buddhist_eval.utils.payloads import Parser, parse_mapping

Compute = Callable[
    [Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]],
    tuple[float, Any],
]
"""``(inputs, references, predictions, params) -> (headline_metric, result)``."""

_JSON_TYPES: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}

_MAX_IDS_IN_MESSAGE = 5


class Direction(str, Enum):
    """Whether a lower or a higher Headline Metric ranks first."""

    LOWER_IS_BETTER = "lower_is_better"
    HIGHER_IS_BETTER = "higher_is_better"


class DisplayStyle(str, Enum):
    """How a number is shown: as a percentage (0.0123 -> 1.23%) or as a decimal."""

    PERCENT = "percent"
    DECIMAL = "decimal"


@dataclass(frozen=True)
class DisplayFormat:
    """How to show a number: a style and how many decimals to keep.

    ``decimals`` counts digits after the point in the shown text, so CER
    0.01234 as ``DisplayFormat(PERCENT, 2)`` reads "1.23%".
    """

    style: DisplayStyle
    decimals: int

    def to_dict(self) -> dict[str, Any]:
        """Return ``{"style": ..., "decimals": ...}`` as plain JSON values."""
        return {"style": self.style.value, "decimals": self.decimals}


@dataclass(frozen=True)
class ScoreResult:
    """What ``Scorer.score`` returns.

    Attributes:
        headline_metric: The unrounded number a leaderboard ranks by.
        details: The metric's full result object as plain JSON values
            (dicts, lists, strings, numbers, booleans).
    """

    headline_metric: float
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return ``{"headline_metric": ..., "details": ...}`` for storage as JSON."""
        return {"headline_metric": self.headline_metric, "details": self.details}


@dataclass(frozen=True)
class Parameter:
    """One optional, defaulted Scorer parameter."""

    name: str
    kind: type
    default: Any
    description: str
    choices: tuple[Any, ...] = ()
    minimum: float | None = None
    exclusive_minimum: float | None = None
    maximum: float | None = None

    def json_schema(self) -> dict[str, Any]:
        """Return the JSON Schema for this parameter's value.

        Returns:
            A JSON Schema object with type, default, description and any
            enum or bounds.
        """
        schema: dict[str, Any] = {
            "type": _JSON_TYPES[self.kind],
            "default": self.default,
            "description": self.description,
        }
        if self.choices:
            schema["enum"] = list(self.choices)
        bounds = {
            "minimum": self.minimum,
            "exclusiveMinimum": self.exclusive_minimum,
            "maximum": self.maximum,
        }
        schema.update(
            {key: value for key, value in bounds.items() if value is not None}
        )
        return schema

    def validate(self, value: Any) -> Any:
        """Check a supplied value against this parameter's type, choices and bounds.

        Args:
            value: The supplied value. An int is accepted for a float parameter.

        Returns:
            The value, with an int widened to float for a float parameter.

        Raises:
            InvalidParameterError: If the value has the wrong type, is not one
                of ``choices``, or is outside the bounds.
        """
        value = self._check_type(value)
        if self.choices and value not in self.choices:
            self._reject(value, f"one of {list(self.choices)}")
        if self.minimum is not None and value < self.minimum:
            self._reject(value, f">= {self.minimum}")
        if self.exclusive_minimum is not None and value <= self.exclusive_minimum:
            self._reject(value, f"> {self.exclusive_minimum}")
        if self.maximum is not None and value > self.maximum:
            self._reject(value, f"<= {self.maximum}")
        return value

    def _check_type(self, value: Any) -> Any:
        if isinstance(value, bool) != (self.kind is bool):
            self._reject(value, f"of type {_JSON_TYPES[self.kind]}")
        if self.kind is float and isinstance(value, int):
            return float(value)
        if not isinstance(value, self.kind):
            self._reject(value, f"of type {_JSON_TYPES[self.kind]}")
        return value

    def _reject(self, value: Any, expected: str) -> None:
        raise InvalidParameterError(
            f"parameter {self.name!r} must be {expected}, got {value!r}"
        )


@dataclass(frozen=True)
class Scorer:
    """A registered metric: identity, Direction, schemas and one ``score`` call.

    ``input_schema``, ``reference_schema`` and ``prediction_schema`` are JSON
    Schemas for one value in the ``inputs``, ``references`` and ``predictions``
    mappings, each keyed by item id. A Scorer whose ``parse_input`` is
    ``None`` ignores ``inputs``.
    """

    name: str
    label: str
    direction: Direction
    parameters: tuple[Parameter, ...]
    input_schema: Mapping[str, Any]
    reference_schema: Mapping[str, Any]
    prediction_schema: Mapping[str, Any]
    sample_prediction: Any
    parse_input: Parser[Any] | None = field(repr=False, compare=False)
    parse_reference: Parser[Any] = field(repr=False, compare=False)
    parse_prediction: Parser[Any] = field(repr=False, compare=False)
    compute: Compute = field(repr=False, compare=False)
    display_format: DisplayFormat

    @property
    def uses_inputs(self) -> bool:
        """Whether ``score`` reads ``inputs`` (span F1 and TOC take passage text)."""
        return self.parse_input is not None

    @property
    def parameter_schema(self) -> dict[str, Any]:
        """JSON Schema for the ``params`` object; every parameter is optional."""
        return {
            "type": "object",
            "properties": {p.name: p.json_schema() for p in self.parameters},
            "additionalProperties": False,
        }

    def validate_params(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """Check supplied parameters and fill in defaults.

        Args:
            params: Parameter values by name; omitted names take their default.

        Returns:
            Every parameter by name, with its supplied or default value.

        Raises:
            InvalidParameterError: If a name is unknown or a value is invalid.
        """
        known = {p.name: p for p in self.parameters}
        unknown = sorted(set(params) - set(known))
        if unknown:
            raise InvalidParameterError(
                f"scorer {self.name!r} has no parameter(s) {unknown}"
            )
        return {
            name: p.validate(params[name]) if name in params else p.default
            for name, p in known.items()
        }

    def score(
        self,
        inputs: Mapping[str, Any],
        references: Mapping[str, Any],
        predictions: Mapping[str, Any],
        **params: Any,
    ) -> ScoreResult:
        """Score Predictions against References and return the Headline Metric.

        Missing and extra prediction ids are handled as the underlying metric
        handles them; use ``validate_predictions`` first to reject them.

        Args:
            inputs: Mapping from item id to an Input value matching
                ``input_schema``. Ignored unless ``uses_inputs``.
            references: Mapping from item id to a value matching ``reference_schema``.
            predictions: Mapping from item id to a value matching ``prediction_schema``.
            **params: Values for this Scorer's parameters.

        Returns:
            A ScoreResult with the unrounded Headline Metric (sort it by
            ``direction``) and the metric's full result as JSON values.

        Raises:
            InvalidParameterError: If a parameter is unknown or invalid.
            InvalidInputError: If an Input has the wrong shape, or a Reference
                has no Input (only when ``uses_inputs``).
            InvalidReferenceError: If a Reference has the wrong shape.
            InvalidPredictionError: If a Prediction has the wrong shape.
            BuddhistEvalError: Any error the underlying metric raises.
        """
        values = self.validate_params(params)
        parsed_inputs = self._parse_inputs(inputs, references)
        parsed_references = parse_mapping(
            references, self.parse_reference, "reference", InvalidReferenceError
        )
        parsed_predictions = parse_mapping(
            predictions, self.parse_prediction, "prediction", InvalidPredictionError
        )
        headline, result = self.compute(
            parsed_inputs, parsed_references, parsed_predictions, values
        )
        return ScoreResult(headline_metric=headline, details=_to_json(result))

    def validate_ground_truth(
        self,
        inputs: Mapping[str, Any],
        references: Mapping[str, Any],
        **params: Any,
    ) -> None:
        """Check a Challenge's Inputs, Ground Truth and parameters before saving.

        The Inputs and the Ground Truth must hold exactly the same ids, every
        value must match its schema, and the Ground Truth must be scorable
        (for example every Gold Span lies inside its passage text).

        Args:
            inputs: Mapping from item id to Input value.
            references: Mapping from item id to Reference value.
            **params: Values for this Scorer's parameters.

        Raises:
            InvalidParameterError: If a parameter is unknown or invalid.
            IdMismatchError: If the two id sets differ.
            InvalidInputError: If an Input has the wrong shape.
            InvalidReferenceError: If a Reference has the wrong shape or
                cannot be scored.
        """
        self.validate_params(params)
        _require_same_ids(inputs, references, "Inputs", "Ground Truth")
        try:
            self.score(inputs, references, {}, **params)
        except (InvalidInputError, InvalidReferenceError):
            raise
        except BuddhistEvalError as exc:
            raise InvalidReferenceError(
                f"Ground Truth cannot be scored: {exc}"
            ) from exc

    def validate_predictions(
        self, inputs: Mapping[str, Any], predictions: Mapping[str, Any]
    ) -> None:
        """Check a Predictions File: one Prediction per Input id, each well shaped.

        Args:
            inputs: Mapping from item id to Input value; only its ids are used.
            predictions: Mapping from item id to Prediction value.

        Raises:
            IdMismatchError: If the prediction ids differ from the Input ids.
            InvalidPredictionError: If a Prediction has the wrong shape.
        """
        _require_same_ids(inputs, predictions, "Inputs", "Predictions")
        parse_mapping(
            predictions, self.parse_prediction, "prediction", InvalidPredictionError
        )

    def describe(self) -> dict[str, Any]:
        """Return a JSON-serialisable description for listing endpoints.

        Returns:
            The name, label, direction value, the Headline Metric's display
            format, the four JSON Schemas and the placeholder Prediction for
            ``sample_submission.jsonl``.
        """
        return {
            "name": self.name,
            "label": self.label,
            "direction": self.direction.value,
            "display_format": self.display_format.to_dict(),
            "parameter_schema": self.parameter_schema,
            "input_schema": deepcopy(dict(self.input_schema)),
            "reference_schema": deepcopy(dict(self.reference_schema)),
            "prediction_schema": deepcopy(dict(self.prediction_schema)),
            "sample_prediction": deepcopy(self.sample_prediction),
        }

    def _parse_inputs(
        self, inputs: Mapping[str, Any], references: Mapping[str, Any]
    ) -> dict[str, Any]:
        if self.parse_input is None:
            return {}
        missing = sorted(set(references) - set(inputs))
        if missing:
            raise InvalidInputError(f"no Input for reference id(s) {_preview(missing)}")
        return parse_mapping(inputs, self.parse_input, "input", InvalidInputError)


def _require_same_ids(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
    expected_name: str,
    actual_name: str,
) -> None:
    missing = tuple(sorted(set(expected) - set(actual)))
    extra = tuple(sorted(set(actual) - set(expected)))
    if not (missing or extra):
        return
    problems = []
    if missing:
        problems.append(f"missing {len(missing)} id(s) {_preview(missing)}")
    if extra:
        problems.append(
            f"has {len(extra)} id(s) not in {expected_name} {_preview(extra)}"
        )
    raise IdMismatchError(
        f"{actual_name} {'; '.join(problems)}", missing_ids=missing, extra_ids=extra
    )


def _preview(ids: list[str] | tuple[str, ...]) -> str:
    shown = ", ".join(repr(i) for i in ids[:_MAX_IDS_IN_MESSAGE])
    more = len(ids) - _MAX_IDS_IN_MESSAGE
    return f"[{shown}{f', ... {more} more' if more > 0 else ''}]"


def _to_json(value: Any) -> Any:
    """Convert a metric result into plain JSON values."""
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: _to_json(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _to_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json(item) for item in value]
    return value
