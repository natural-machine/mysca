"""Logger configuration for the `mysca` package.

Provides `configure_logging()` for entrypoints to call once at startup. Every
module inside `mysca` obtains its logger via `logging.getLogger(__name__)`;
these propagate to the package-root `"mysca"` logger which owns the handlers.

Entrypoints emit to stderr AND (optionally) a logfile in the run's output
directory. tqdm progress bars bypass logging entirely (they write directly to
sys.stderr with carriage returns) and therefore never reach the logfile.
"""

import logging
import os
import sys


PACKAGE_LOGGER_NAME = "mysca"

_STREAM_FORMAT = "%(levelname)s %(name)s: %(message)s"
_FILE_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def verbosity_to_level(verbosity: int) -> int:
    """Map the CLI `--verbosity` integer to a stdlib logging level.

    0 -> WARNING, 1 -> INFO, 2 or more -> DEBUG.
    """
    if verbosity <= 0:
        return logging.WARNING
    if verbosity == 1:
        return logging.INFO
    return logging.DEBUG


def configure_logging(
    verbosity: int = 1,
    logfile: str | os.PathLike | None = None,
    *,
    capture_warnings: bool = True,
) -> logging.Logger:
    """Configure the `mysca` package logger. Safe to call more than once."""
    logger = logging.getLogger(PACKAGE_LOGGER_NAME)

    for h in list(logger.handlers):
        if not isinstance(h, logging.NullHandler):
            logger.removeHandler(h)
            h.close()

    level = verbosity_to_level(verbosity)
    logger.setLevel(level)
    logger.propagate = False

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(logging.Formatter(_STREAM_FORMAT))
    logger.addHandler(stream_handler)

    if logfile is not None:
        os.makedirs(os.path.dirname(os.path.abspath(logfile)), exist_ok=True)
        file_handler = logging.FileHandler(logfile, mode="w")
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(_FILE_FORMAT))
        logger.addHandler(file_handler)

    logging.captureWarnings(capture_warnings)

    return logger
