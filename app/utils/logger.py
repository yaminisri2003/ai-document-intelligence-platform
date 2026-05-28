# logger.py
# This file creates a reusable logger for the entire project.
# Every other module imports get_logger() from here
# and creates its own named logger.
#
# Why named loggers?
# When you have 10 different modules all logging at once,
# named loggers tell you WHICH module produced each message.
# Without names, all log messages look identical and debugging
# becomes extremely difficult.

import logging
import sys
from app.utils.config import settings


def get_logger(name: str) -> logging.Logger:
    """
    Create and return a named logger for a module.

    Usage in any other file:
        from app.utils.logger import get_logger
        logger = get_logger(__name__)
        logger.info("This module is working correctly")

    Args:
        name: The module name. Always pass __name__ here.
              Python automatically fills __name__ with
              the current file's module path.
              e.g. "app.utils.config" or "app.rag.ingestion"

    Returns:
        A configured Logger object ready to use
    """

    # Get or create a logger with this specific name
    logger = logging.getLogger(name)

    # Only add a handler if this logger does not have one yet.
    # Without this check, every call to get_logger() would add
    # another handler and you would see duplicate log messages.
    if not logger.handlers:

        # StreamHandler sends log output to the terminal
        handler = logging.StreamHandler(sys.stdout)

        # Formatter defines exactly how each log line looks
        # %(asctime)s   = timestamp e.g. 2025-01-15 14:23:05
        # %(levelname)s = INFO, WARNING, ERROR etc
        # %(name)s      = which module this came from
        # %(message)s   = the actual message you logged
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # Set the minimum level of messages to show
    # getattr converts the string "INFO" to logging.INFO
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    logger.setLevel(log_level)

    return logger