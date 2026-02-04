"""Logging helpers for the OIPD package."""

from __future__ import annotations

import logging
from typing import Iterable, Optional

ROOT_LOGGER_NAME = "oipd"
_NULL_HANDLER = logging.NullHandler()


def get_logger(name: str = ROOT_LOGGER_NAME) -> logging.Logger:
    """Return a logger initialised with a null handler by default.

    Args:
        name: Fully qualified logger name.

    Returns:
        A configured :class:`logging.Logger` instance.
    """

    logger = logging.getLogger(name)
    if not any(isinstance(handler, logging.NullHandler) for handler in logger.handlers):
        logger.addHandler(_NULL_HANDLER)
    return logger


def configure_logging(
    level: int = logging.INFO,
    *,
    handlers: Optional[Iterable[logging.Handler]] = None,
    format_string: str | None = None,
) -> None:
    """Attach handlers to the OIPD root logger.

    Args:
        level: Logging level to apply to the root OIPD logger.
        handlers: Optional iterable of handlers to attach.
        format_string: Optional log message format applied to newly created
            handlers.
    """

    logger = get_logger(ROOT_LOGGER_NAME)
    logger.setLevel(level)

    if handlers:
        for handler in handlers:
            if format_string:
                handler.setFormatter(logging.Formatter(format_string))
            logger.addHandler(handler)


__all__ = ["ROOT_LOGGER_NAME", "configure_logging", "get_logger"]
