"""Structured warning diagnostics for public interface objects."""

from __future__ import annotations

from collections import Counter
import warnings
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias, cast

from oipd.warnings import (
    DataQualityWarning,
    ModelRiskWarning,
    NumericalWarning,
    OIPDWarning,
    WorkflowWarning,
)

WarningCategory: TypeAlias = Literal[
    "data_quality",
    "model_risk",
    "numerical",
    "workflow",
]
WarningSeverity: TypeAlias = Literal["info", "warning", "severe"]

VALID_WARNING_CATEGORIES: frozenset[str] = frozenset(
    {"data_quality", "model_risk", "numerical", "workflow"}
)
VALID_WARNING_SEVERITIES: frozenset[str] = frozenset({"info", "warning", "severe"})
SEVERITY_RANK: Mapping[WarningSeverity, int] = {
    "info": 0,
    "warning": 1,
    "severe": 2,
}


def _freeze_detail_value(value: Any) -> Any:
    """Return an immutable defensive copy of a diagnostic detail value.

    Detail values must be small JSON-like audit payloads: scalars, mappings,
    or nested list/tuple/set containers of scalars. Arrays, DataFrames, model
    objects, and other live mutable objects should be summarized before storing.

    Args:
        value: Detail value supplied by a caller.

    Returns:
        Any: Frozen copy suitable for storing on an immutable warning event.

    Raises:
        TypeError: If ``value`` is not a supported JSON-like detail value.
    """
    if isinstance(value, _FrozenDetails):
        return value
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return _FrozenDetails(value)
    if isinstance(value, tuple):
        return tuple(_freeze_detail_value(item) for item in value)
    if isinstance(value, list):
        return tuple(_freeze_detail_value(item) for item in value)
    if isinstance(value, frozenset):
        return frozenset(_freeze_detail_value(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze_detail_value(item) for item in value)
    raise TypeError(
        "warning event details must contain only small JSON-like audit values; "
        "summarize arrays, DataFrames, or model objects before storing."
    )


class _FrozenDetails(Mapping[str, Any]):
    """Read-only mapping used for warning event details.

    Detail values must be small JSON-like audit payloads. Arrays, DataFrames,
    and other live mutable objects should be summarized before storing.

    Args:
        data: Detail mapping to freeze into the event.
    """

    def __init__(self, data: Mapping[str, Any] | None = None) -> None:
        """Initialize a read-only detail mapping.

        Args:
            data: Detail mapping to copy and freeze.

        Raises:
            TypeError: If any detail key is not a string or if any detail value
                is not a supported JSON-like audit value.
        """
        source = data or {}
        frozen_data: dict[str, Any] = {}
        for key, value in source.items():
            if not isinstance(key, str):
                raise TypeError("detail keys must be strings.")
            frozen_data[key] = _freeze_detail_value(value)
        self._data = frozen_data

    def __getitem__(self, key: str) -> Any:
        """Return a detail value by key.

        Args:
            key: Detail key to retrieve.

        Returns:
            Any: Frozen detail value.
        """
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        """Return an iterator over detail keys.

        Returns:
            Iterator[str]: Iterator over stored detail keys.
        """
        return iter(self._data)

    def __len__(self) -> int:
        """Return the number of stored detail keys.

        Returns:
            int: Count of detail keys.
        """
        return len(self._data)

    def __repr__(self) -> str:
        """Return a developer-readable representation.

        Returns:
            str: Representation of the underlying frozen details.
        """
        return repr(self._data)

    def __eq__(self, other: object) -> bool:
        """Compare frozen details to another mapping.

        Args:
            other: Object to compare against.

        Returns:
            bool: ``True`` when ``other`` contains the same key-value pairs.
        """
        if not isinstance(other, Mapping):
            return False
        return dict(self.items()) == dict(other.items())

    def __deepcopy__(self, memo: dict[int, Any]) -> "_FrozenDetails":
        """Return self because frozen details are already detached.

        Args:
            memo: Deepcopy memo dictionary.

        Returns:
            _FrozenDetails: This immutable detail mapping.
        """
        return self


def _validate_warning_category(category: object) -> WarningCategory:
    """Validate and normalize a warning category value.

    Args:
        category: Candidate warning category.

    Returns:
        WarningCategory: Valid warning category.

    Raises:
        TypeError: If ``category`` is not a string.
        ValueError: If ``category`` is not supported.
    """
    if not isinstance(category, str):
        raise TypeError("category must be a string.")
    if category not in VALID_WARNING_CATEGORIES:
        raise ValueError(f"Unsupported warning category: {category!r}.")
    return cast(WarningCategory, category)


def _validate_warning_severity(severity: object) -> WarningSeverity:
    """Validate and normalize a warning severity value.

    Args:
        severity: Candidate warning severity.

    Returns:
        WarningSeverity: Valid warning severity.

    Raises:
        TypeError: If ``severity`` is not a string.
        ValueError: If ``severity`` is not supported.
    """
    if not isinstance(severity, str):
        raise TypeError("severity must be a string.")
    if severity not in VALID_WARNING_SEVERITIES:
        raise ValueError(f"Unsupported warning severity: {severity!r}.")
    return cast(WarningSeverity, severity)


def _validate_non_empty_string(value: object, field_name: str) -> str:
    """Validate and trim a required string field.

    Args:
        value: Candidate string value.
        field_name: Name of the field for error messages.

    Returns:
        str: Trimmed non-empty string.

    Raises:
        TypeError: If ``value`` is not a string.
        ValueError: If ``value`` is empty after trimming whitespace.
    """
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")
    trimmed_value = value.strip()
    if not trimmed_value:
        raise ValueError(f"{field_name} must be a non-empty string.")
    return trimmed_value


def warning_class_for_category(category: str) -> type[OIPDWarning]:
    """Return the public warning class for a diagnostic category.

    Args:
        category: Diagnostic category such as ``"data_quality"``.

    Returns:
        type[OIPDWarning]: Public broad warning class for that category.

    Raises:
        ValueError: If ``category`` is not supported.
    """
    validated_category = _validate_warning_category(category)
    warning_classes: Mapping[WarningCategory, type[OIPDWarning]] = {
        "data_quality": DataQualityWarning,
        "model_risk": ModelRiskWarning,
        "numerical": NumericalWarning,
        "workflow": WorkflowWarning,
    }
    return warning_classes[validated_category]


def _emit_warning_summaries(events: Sequence["WarningEvent"], *, owner: str) -> None:
    """Emit one concise Python warning per broad warning category.

    Args:
        events: Warning events recorded for the public operation.
        owner: Interface operation name shown in the warning message.
    """
    events_by_category: dict[str, list[WarningEvent]] = {}
    for event in events:
        events_by_category.setdefault(event.category, []).append(event)

    category_labels = {
        "data_quality": "data-quality",
        "model_risk": "model-risk",
        "numerical": "numerical",
        "workflow": "workflow",
    }
    for category, category_events in events_by_category.items():
        event_count = len(category_events)
        warning_class = warning_class_for_category(category)
        label = category_labels[category]
        warnings.warn(
            f"{owner} recorded {event_count} {label} warning event"
            f"{'' if event_count == 1 else 's'}; inspect "
            ".warning_diagnostics.events for details.",
            warning_class,
            stacklevel=3,
        )


@dataclass(frozen=True)
class WarningEvent:
    """Single structured warning-worthy event.

    Attributes:
        category: Broad warning category used for warning-class mapping.
        event_type: Specific event identity within the broad category.
        severity: Severity label ordered as ``info < warning < severe``.
        scope: Operation or object scope that owns replacement semantics.
        message: Human-readable event summary.
        details: Read-only structured event details. Values must be small
            JSON-like audit payloads; arrays, DataFrames, and model objects
            should be summarized before storing.
    """

    category: WarningCategory
    event_type: str
    severity: WarningSeverity
    scope: str
    message: str
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate event fields and defensively freeze details.

        Raises:
            ValueError: If category, severity, event type, scope, or message is invalid.
            TypeError: If required string fields are not strings, if ``details``
                is not a mapping, or if details contain unsupported live objects.
        """
        object.__setattr__(
            self,
            "category",
            _validate_warning_category(self.category),
        )
        object.__setattr__(
            self,
            "severity",
            _validate_warning_severity(self.severity),
        )
        if not isinstance(self.details, Mapping):
            raise TypeError("details must be a mapping.")
        object.__setattr__(
            self,
            "event_type",
            _validate_non_empty_string(self.event_type, "event_type"),
        )
        object.__setattr__(
            self,
            "scope",
            _validate_non_empty_string(self.scope, "scope"),
        )
        object.__setattr__(
            self,
            "message",
            _validate_non_empty_string(self.message, "message"),
        )
        object.__setattr__(self, "details", _FrozenDetails(self.details))


@dataclass(frozen=True)
class WarningDiagnosticsSummary:
    """Aggregate counts over current warning diagnostic events.

    Attributes:
        total_events: Total number of current warning events.
        by_category: Event counts by broad warning category.
        by_event_type: Event counts by specific event type.
        by_severity: Event counts by severity label.
        worst_severity: Highest current severity, or ``None`` when empty.
    """

    total_events: int = 0
    by_category: Mapping[str, int] = field(default_factory=dict)
    by_event_type: Mapping[str, int] = field(default_factory=dict)
    by_severity: Mapping[str, int] = field(default_factory=dict)
    worst_severity: WarningSeverity | None = None

    def __post_init__(self) -> None:
        """Validate summary fields and freeze count mappings.

        Raises:
            ValueError: If ``worst_severity`` is not a supported severity.
        """
        if self.worst_severity is not None:
            _validate_warning_severity(self.worst_severity)
        object.__setattr__(self, "total_events", int(self.total_events))
        object.__setattr__(self, "by_category", _FrozenDetails(self.by_category))
        object.__setattr__(self, "by_event_type", _FrozenDetails(self.by_event_type))
        object.__setattr__(self, "by_severity", _FrozenDetails(self.by_severity))


class WarningDiagnostics:
    """Read-only current-state warning diagnostics for an interface object.

    Users should inspect ``events`` and ``summary``. Event mutation is reserved
    for interface internals via private writer helpers.

    Args:
        events: Optional initial warning events.
    """

    def __init__(self, events: Iterable[WarningEvent] | None = None) -> None:
        """Initialize a warning diagnostics container.

        Args:
            events: Optional initial warning events.
        """
        self._events: list[WarningEvent] = []
        if events is not None:
            self._events = self._validate_events(events)

    @property
    def events(self) -> list[WarningEvent]:
        """Return a copy of current warning events.

        Returns:
            list[WarningEvent]: Copy of the current event list.
        """
        return list(self._events)

    @property
    def summary(self) -> WarningDiagnosticsSummary:
        """Return aggregate counts over current warning events.

        Returns:
            WarningDiagnosticsSummary: Current diagnostic summary.
        """
        by_category = Counter(event.category for event in self._events)
        by_event_type = Counter(event.event_type for event in self._events)
        by_severity = Counter(event.severity for event in self._events)
        worst_severity = self._worst_severity(self._events)
        return WarningDiagnosticsSummary(
            total_events=len(self._events),
            by_category=dict(by_category),
            by_event_type=dict(by_event_type),
            by_severity=dict(by_severity),
            worst_severity=worst_severity,
        )

    def _replace_scope_events(
        self,
        scope: str,
        events: Iterable[WarningEvent],
    ) -> None:
        """Atomically replace all current events for one scope.

        The caller should build the full replacement batch first, then call this
        method once. That prevents stale same-scope events from lingering when a
        rerun produces fewer events than the prior run.

        Args:
            scope: Scope whose current events should be replaced.
            events: Complete replacement batch for ``scope``.

        Raises:
            ValueError: If ``scope`` is empty or any event belongs to another scope.
            TypeError: If ``events`` contains a non-``WarningEvent`` item.
        """
        normalized_scope = _validate_non_empty_string(scope, "scope")
        replacement_events = self._validate_events(events)
        mismatched_events = [
            event for event in replacement_events if event.scope != normalized_scope
        ]
        if mismatched_events:
            raise ValueError("All replacement events must match the target scope.")

        preserved_events = [
            event for event in self._events if event.scope != normalized_scope
        ]
        self._events = [*preserved_events, *replacement_events]

    @staticmethod
    def _validate_events(events: Iterable[WarningEvent]) -> list[WarningEvent]:
        """Validate an iterable of warning events.

        Args:
            events: Candidate warning events.

        Returns:
            list[WarningEvent]: Validated event list.

        Raises:
            TypeError: If any item is not a ``WarningEvent``.
        """
        validated_events = list(events)
        for event in validated_events:
            if not isinstance(event, WarningEvent):
                raise TypeError("events must contain only WarningEvent instances.")
        return validated_events

    @staticmethod
    def _worst_severity(
        events: Iterable[WarningEvent],
    ) -> WarningSeverity | None:
        """Return the highest severity in an event collection.

        Args:
            events: Warning events to summarize.

        Returns:
            WarningSeverity | None: Highest severity, or ``None`` for no events.
        """
        worst_event = max(
            events,
            key=lambda event: SEVERITY_RANK[event.severity],
            default=None,
        )
        if worst_event is None:
            return None
        return worst_event.severity

    def __deepcopy__(self, memo: dict[int, Any]) -> "WarningDiagnostics":
        """Return a detached copy of the diagnostics container.

        Args:
            memo: Deepcopy memo dictionary.

        Returns:
            WarningDiagnostics: Copied diagnostics with copied event list.
        """
        return WarningDiagnostics(self._events)


__all__ = [
    "WarningCategory",
    "WarningDiagnostics",
    "WarningDiagnosticsSummary",
    "WarningEvent",
    "WarningSeverity",
    "warning_class_for_category",
]
