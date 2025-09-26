from __future__ import annotations

"""Centralized error types for the OIPD core package."""


class OIPDError(Exception):
    """Base exception for OIPD package."""

    pass


class InvalidInputError(OIPDError):
    """Exception raised for invalid input parameters."""

    pass


class CalculationError(OIPDError):
    """Exception raised when calculations fail."""

    pass


__all__ = ["OIPDError", "InvalidInputError", "CalculationError"]

