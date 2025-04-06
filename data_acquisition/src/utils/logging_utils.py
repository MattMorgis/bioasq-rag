"""Logging utilities for the BioASQ RAG application."""

import logging
import sys
from pathlib import Path
from typing import List, Optional, Union


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = "bioasq-rag.log",
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
) -> None:
    """
    Configure logging to output to both console and file.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file, or None to disable file logging
        log_format: Format string for log messages
    """
    # Convert string log level to numeric value
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    # Create handlers list
    handlers: List[Union[logging.StreamHandler, logging.FileHandler]] = [
        logging.StreamHandler(sys.stdout)
    ]

    # Add file handler if log_file is specified
    if log_file:
        # Create log directory if it doesn't exist
        log_path = Path(log_file)
        if log_path.parent != Path("."):
            log_path.parent.mkdir(parents=True, exist_ok=True)

        handlers.append(logging.FileHandler(log_file))

    # Configure logging
    logging.basicConfig(
        level=numeric_level,
        format=log_format,
        handlers=handlers,
    )

    # Log confirmation
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured with level {log_level}")
    if log_file:
        logger.info(f"Logging to file: {log_file}")
