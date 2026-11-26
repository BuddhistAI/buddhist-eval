"""Scorer and Parameter types behind the scorer registry."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from buddhist_eval.errors import InvalidParameterError

ScoreFunction = Callable[
    [Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]], float
]

_JSON_TYPES: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}


class SortDirection(str, Enum):
    """Which way a leaderboard sorts a Scorer's value."""

    LOWER_IS_BETTER = "lower_is_better"
    HIGHER_IS_BETTER = "higher_is_better"


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
    """A registered metric: identity, sort direction, input schemas and ``score``.

    ``prediction_schema`` and ``reference_schema`` are JSON Schemas for one
    value in the ``predictions`` and ``references`` mappings, keyed by item id.
    """

    name: str
    label: str
    direction: SortDirection
    parameters: tuple[Parameter, ...]
    prediction_schema: Mapping[str, Any]
    reference_schema: Mapping[str, Any]
    score_function: ScoreFunction = field(repr=False, compare=False)

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
        predictions: Mapping[str, Any],
        references: Mapping[str, Any],
        **params: Any,
    ) -> float:
        """Score predictions against references and return the leaderboard value.

        Args:
            predictions: Mapping from item id to a value matching ``prediction_schema``.
            references: Mapping from item id to a value matching ``reference_schema``.
            **params: Values for this Scorer's parameters.

        Returns:
            The unrounded score; sort it using ``direction``.

        Raises:
            InvalidParameterError: If a parameter is unknown or invalid.
            InvalidPredictionError: If a prediction value has the wrong shape.
            InvalidReferenceError: If a reference value has the wrong shape.
            BuddhistEvalError: Any error the underlying metric raises.
        """
        return self.score_function(
            predictions, references, self.validate_params(params)
        )

    def describe(self) -> dict[str, Any]:
        """Return a JSON-serialisable description for listing endpoints.

        Returns:
            The name, label, direction value and the three JSON Schemas.
        """
        return {
            "name": self.name,
            "label": self.label,
            "direction": self.direction.value,
            "parameter_schema": self.parameter_schema,
            "prediction_schema": dict(self.prediction_schema),
            "reference_schema": dict(self.reference_schema),
        }
