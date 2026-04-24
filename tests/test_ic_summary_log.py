"""Unit tests for log_top_ic_summary.

Verifies the human-readable top-N IC summary: significance marker, the
eigenvalue column, and position mappings (processed → unprocessed →
reference). Uses a local-handler capture pattern because the package
logger is configured non-propagating in entrypoint tests.
"""

import logging

import numpy as np
import pytest
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment

from mysca.run_sca import log_top_ic_summary


def _msa(seq_pairs):
    return MultipleSeqAlignment(
        [SeqRecord(Seq(s), id=i) for i, s in seq_pairs]
    )


@pytest.fixture
def capture_run_sca_logs():
    """Collect records from the mysca.run_sca logger.

    Avoids pytest's caplog because the mysca package logger is
    configured with propagate=False in earlier tests.
    """
    target = logging.getLogger("mysca.run_sca")
    records = []

    class _Collector(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = _Collector(level=logging.DEBUG)
    old_level = target.level
    target.addHandler(handler)
    target.setLevel(logging.DEBUG)
    try:
        yield records
    finally:
        target.removeHandler(handler)
        target.setLevel(old_level)


def _processed_line(msgs):
    return next(m for m in msgs if "processed:" in m and "unprocessed" not in m)


def _unprocessed_line(msgs):
    return next(m for m in msgs if "unprocessed:" in m)


def _reference_pos_line(msgs):
    return next(m for m in msgs if "reference pos:" in m)


def _reference_res_line(msgs):
    return next(m for m in msgs if "reference res:" in m)


def test_log_top_ic_summary_basic_marker_and_positions(capture_run_sca_logs):
    """With kstar=2, ICs 0-1 are `*`, ICs 2+ are `-`."""
    groups = [
        np.array([0, 1]),
        np.array([2]),
        np.array([1, 3]),
    ]
    evals_sca = np.array([1.5, 0.9, 0.3, 0.1])
    retained_positions = np.array([10, 12, 14, 16])
    msa = _msa([
        ("ref",   "ACDEF"),
        ("other", "ACDEF"),
    ])

    log_top_ic_summary(
        groups, kstar=2, evals_sca=evals_sca,
        retained_positions=retained_positions,
        msa_obj_orig=msa, reference_id=None,
        n_logged_comps=10,
    )
    msgs = [r.getMessage() for r in capture_run_sca_logs]
    header = next(m for m in msgs if m.startswith("Top "))
    assert "3/3 ICs" in header
    # Without a reference, header should not echo one.
    assert "reference=" not in header

    ic_lines = [m for m in msgs if m.startswith("IC ")]
    assert ic_lines[0].startswith("IC 0: * ")
    assert "λ_0=1.5" in ic_lines[0]
    assert "(2 positions)" in ic_lines[0]
    assert ic_lines[1].startswith("IC 1: * ")
    assert "(1 positions)" in ic_lines[1]
    assert ic_lines[2].startswith("IC 2: - ")  # below kstar

    # Three ICs × two indented lines (processed, unprocessed) = 6 lines.
    processed_lines = [m for m in msgs if "processed:" in m and "unprocessed" not in m]
    unprocessed_lines = [m for m in msgs if "unprocessed:" in m]
    assert len(processed_lines) == 3
    assert len(unprocessed_lines) == 3

    # IC 0 processed=[0, 1]; unprocessed = retained_positions[[0,1]] = [10,12].
    assert processed_lines[0].endswith("[0, 1]")
    assert unprocessed_lines[0].endswith("[10, 12]")
    assert unprocessed_lines[1].endswith("[14]")
    assert unprocessed_lines[2].endswith("[12, 16]")

    # No reference requested → no reference lines.
    assert not any("reference" in m for m in msgs if m.startswith("    "))


def test_log_top_ic_summary_reference_mapping(capture_run_sca_logs):
    """Reference residue indices and letters respect gap columns."""
    # MSA columns:    0=A  1=-  2=C  3=D  4=E
    # Reference seq has A,-,C,D,E → raw indices 0,-1,1,2,3
    msa = _msa([
        ("ref",   "A-CDE"),
        ("other", "AACDE"),
    ])
    groups = [np.array([0, 1, 2])]  # processed positions
    evals_sca = np.array([1.0])
    retained_positions = np.array([0, 1, 2])  # unprocessed columns 0,1,2

    log_top_ic_summary(
        groups, kstar=1, evals_sca=evals_sca,
        retained_positions=retained_positions,
        msa_obj_orig=msa, reference_id="ref",
        n_logged_comps=5,
    )
    msgs = [r.getMessage() for r in capture_run_sca_logs]
    header = next(m for m in msgs if m.startswith("Top "))
    assert "reference=ref" in header

    # Column 0 → raw idx 0, residue A
    # Column 1 is a gap → '-' / '-'
    # Column 2 → raw idx 1, residue C
    assert _reference_pos_line(msgs).endswith("[0, -, 1]")
    assert _reference_res_line(msgs).endswith("[A, -, C]")


def test_log_top_ic_summary_with_dropped_cols_and_reference_gap(
        capture_run_sca_logs,
):
    """Combined transform: processed → unprocessed (drops cols) → reference raw
    (with a gap at a retained unprocessed column).

    Fixture: original MSA of length 6, columns 1 and 4 dropped by
    preprocessing → retained_positions=[0,2,3,5]. Reference sequence
    "AB-DEF" has raw-residue indices 0,1,-,2,3,4 at original columns
    0..5 (gap at column 2). The processed MSA positions [0,1,2,3]
    therefore map to:
        proc 0 → orig 0 → ref raw 0   ('A')
        proc 1 → orig 2 → ref gap     '-'
        proc 2 → orig 3 → ref raw 2   ('D', 3rd non-gap residue)
        proc 3 → orig 5 → ref raw 4   ('F', 5th non-gap residue)
    """
    msa = _msa([
        ("ref",   "AB-DEF"),
        ("other", "ABCDEF"),
    ])
    groups = [np.array([0, 1, 2, 3])]
    evals_sca = np.array([1.0])
    retained_positions = np.array([0, 2, 3, 5])

    log_top_ic_summary(
        groups, kstar=1, evals_sca=evals_sca,
        retained_positions=retained_positions,
        msa_obj_orig=msa, reference_id="ref",
        n_logged_comps=5,
    )
    msgs = [r.getMessage() for r in capture_run_sca_logs]
    assert _unprocessed_line(msgs).endswith("[0, 2, 3, 5]")
    assert _reference_pos_line(msgs).endswith("[0, -, 2, 4]")
    assert _reference_res_line(msgs).endswith("[A, -, D, F]")


def test_log_top_ic_summary_respects_n_logged_comps(capture_run_sca_logs):
    groups = [np.array([i]) for i in range(5)]
    evals_sca = np.arange(5, dtype=float)[::-1]
    retained_positions = np.arange(5)
    msa = _msa([("x", "AAAAA")])

    log_top_ic_summary(
        groups, kstar=5, evals_sca=evals_sca,
        retained_positions=retained_positions,
        msa_obj_orig=msa, reference_id=None,
        n_logged_comps=2,
    )
    msgs = [r.getMessage() for r in capture_run_sca_logs]
    ic_lines = [m for m in msgs if m.startswith("IC ")]
    assert len(ic_lines) == 2  # capped at n_logged_comps


def test_log_top_ic_summary_noop_when_n_logged_comps_zero(capture_run_sca_logs):
    log_top_ic_summary(
        [np.array([0])], kstar=1, evals_sca=np.array([1.0]),
        retained_positions=np.array([0]),
        msa_obj_orig=_msa([("x", "A")]), reference_id=None,
        n_logged_comps=0,
    )
    assert capture_run_sca_logs == []
