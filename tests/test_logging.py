"""Tests for the `mysca.logging_config` module and package logger setup."""

import logging
import os
import subprocess
import sys

import pytest

from mysca.logging_config import (
    PACKAGE_LOGGER_NAME,
    configure_logging,
    verbosity_to_level,
)
from tests.conftest import TMPDIR, remove_dir


@pytest.fixture
def reset_package_logger():
    """Put the `mysca` logger into a clean state before each test.

    Saves current state, clears handlers, re-enables propagation so pytest's
    caplog fixture can capture records, then restores state after.
    """
    logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    saved_handlers = list(logger.handlers)
    saved_level = logger.level
    saved_propagate = logger.propagate

    for h in list(logger.handlers):
        if not isinstance(h, logging.NullHandler):
            logger.removeHandler(h)
            h.close()
    logger.setLevel(logging.NOTSET)
    logger.propagate = True

    yield logger

    for h in list(logger.handlers):
        logger.removeHandler(h)
        if not isinstance(h, logging.NullHandler):
            h.close()
    for h in saved_handlers:
        logger.addHandler(h)
    logger.setLevel(saved_level)
    logger.propagate = saved_propagate
    logging.captureWarnings(False)


class TestVerbosityMapping:

    @pytest.mark.parametrize("verbosity,expected", [
        (0, logging.WARNING),
        (1, logging.INFO),
        (2, logging.DEBUG),
        (3, logging.DEBUG),
        (-1, logging.WARNING),
    ])
    def test_verbosity_to_level(self, verbosity, expected):
        assert verbosity_to_level(verbosity) == expected


class TestConfigureLogging:

    def test_returns_package_logger(self, reset_package_logger):
        logger = configure_logging(verbosity=1)
        assert logger.name == PACKAGE_LOGGER_NAME

    def test_level_from_verbosity(self, reset_package_logger):
        logger = configure_logging(verbosity=2)
        assert logger.level == logging.DEBUG

    def test_stream_handler_attached_by_default(self, reset_package_logger):
        logger = configure_logging(verbosity=1)
        stream_handlers = [
            h for h in logger.handlers
            if isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.FileHandler)
        ]
        assert len(stream_handlers) == 1

    def test_no_file_handler_when_logfile_none(self, reset_package_logger):
        logger = configure_logging(verbosity=1, logfile=None)
        file_handlers = [
            h for h in logger.handlers if isinstance(h, logging.FileHandler)
        ]
        assert file_handlers == []

    def test_file_handler_attached_when_logfile_given(
        self, reset_package_logger
    ):
        outdir = os.path.join(TMPDIR, "test_logging_filehandler")
        os.makedirs(outdir, exist_ok=True)
        logfile = os.path.join(outdir, "run.log")
        try:
            logger = configure_logging(verbosity=1, logfile=logfile)
            file_handlers = [
                h for h in logger.handlers
                if isinstance(h, logging.FileHandler)
            ]
            assert len(file_handlers) == 1
            logger.info("hello")
            for h in file_handlers:
                h.flush()
            with open(logfile) as f:
                content = f.read()
            assert "hello" in content
        finally:
            for h in list(logger.handlers):
                if isinstance(h, logging.FileHandler):
                    logger.removeHandler(h)
                    h.close()
            remove_dir(outdir)

    def test_idempotent(self, reset_package_logger):
        logger = configure_logging(verbosity=1)
        configure_logging(verbosity=1)
        configure_logging(verbosity=1)
        stream_handlers = [
            h for h in logger.handlers
            if isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.FileHandler)
        ]
        assert len(stream_handlers) == 1

    def test_reconfigure_replaces_file_handler(self, reset_package_logger):
        outdir = os.path.join(TMPDIR, "test_logging_reconfigure")
        os.makedirs(outdir, exist_ok=True)
        logfile_a = os.path.join(outdir, "a.log")
        logfile_b = os.path.join(outdir, "b.log")
        try:
            logger = configure_logging(verbosity=1, logfile=logfile_a)
            configure_logging(verbosity=1, logfile=logfile_b)
            file_handlers = [
                h for h in logger.handlers
                if isinstance(h, logging.FileHandler)
            ]
            assert len(file_handlers) == 1
            assert os.path.basename(file_handlers[0].baseFilename) == "b.log"
        finally:
            for h in list(logger.handlers):
                if isinstance(h, logging.FileHandler):
                    logger.removeHandler(h)
                    h.close()
            remove_dir(outdir)

    def test_captures_warnings(self, reset_package_logger, caplog):
        import warnings
        configure_logging(verbosity=1, capture_warnings=True)
        with caplog.at_level(logging.WARNING, logger="py.warnings"):
            warnings.warn("soft fail", UserWarning)
        assert any("soft fail" in rec.message for rec in caplog.records)


class TestLoadMsaSoftWarnings:
    """The two io.py canonical-symbol soft warnings from the plan."""

    def test_warning_when_sequences_removed(self, reset_package_logger, caplog):
        from mysca.io import load_msa
        from mysca.mappings import SymMap

        mapping = SymMap("ACDEF", "-", exclude_syms="X")
        with caplog.at_level(logging.WARNING, logger="mysca.io"):
            load_msa(
                "tests/_data/msas/msa02.faa",
                format="fasta",
                mapping=mapping,
            )
        assert any(
            "Removed" in rec.message and "excluded symbols" in rec.message
            and rec.levelno == logging.WARNING
            for rec in caplog.records
        )

    def test_no_warning_when_nothing_removed(self, reset_package_logger, caplog):
        from mysca.io import load_msa
        from mysca.mappings import SymMap

        mapping = SymMap("ACDEF", "-", exclude_syms="X")
        with caplog.at_level(logging.WARNING, logger="mysca.io"):
            load_msa(
                "tests/_data/msas/msa01.faa",
                format="fasta",
                mapping=mapping,
            )
        assert not any(
            "Removed" in rec.message and rec.levelno == logging.WARNING
            for rec in caplog.records
        )

    def test_warning_on_autodetected_noncanonical(
        self, reset_package_logger, caplog,
    ):
        from mysca.io import load_msa

        with caplog.at_level(logging.WARNING, logger="mysca.io"):
            load_msa(
                "tests/_data/msas/msa02.faa",
                format="fasta",
                mapping=None,
            )
        assert any(
            "non-canonical symbols" in rec.message
            and rec.levelno == logging.WARNING
            for rec in caplog.records
        )


class TestLibraryHygiene:

    def test_import_mysca_is_silent(self):
        """Bare `import mysca` must not write to stderr."""
        result = subprocess.run(
            [sys.executable, "-c", "import mysca"],
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": os.path.abspath("src")},
        )
        assert result.returncode == 0
        assert result.stderr == ""

    def test_null_handler_present_on_import(self):
        """The package logger always has a NullHandler to prevent warnings."""
        import importlib
        import mysca
        importlib.reload(mysca)
        logger = logging.getLogger(PACKAGE_LOGGER_NAME)
        null_handlers = [
            h for h in logger.handlers if isinstance(h, logging.NullHandler)
        ]
        assert len(null_handlers) >= 1
