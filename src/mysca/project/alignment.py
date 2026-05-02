"""Out-of-sample sequence alignment for `mysca.project`.

Dispatch registry mapping a method name to a callable that aligns new
sequences onto an existing MSA's column structure (preserving column
count so that `retained_positions` indexing stays valid downstream).

Public surface:

- ``ALIGNERS`` — ``{name: callable}``.
- ``register_aligner(name)`` — decorator to register a new aligner.
- ``align_to_msa(...)`` — top-level dispatch.

Each aligner callable has signature
``(new_fasta_path, msa_obj_loaded, workdir) -> dict`` and must write the
aligned output for *only the new sequences* to
``workdir/aligned_new.fasta``. The length of each aligned sequence
must equal the length of ``msa_obj_loaded`` (column-preserving). Return
value is a dict of diagnostics (``n_in``, ``n_out``, ``elapsed_s``).
"""

import logging
import os
import subprocess
import tempfile
import time
from typing import Callable, Iterable

from Bio import AlignIO, SeqIO
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from mysca.prealign import _resolve_bin

logger = logging.getLogger("mysca.project.alignment")


ALIGNED_NEW_FNAME = "aligned_new.fasta"


def register_aligner(name: str) -> Callable[[Callable], Callable]:
    def _decorator(fn):
        ALIGNERS[name] = fn
        return fn
    return _decorator


def _write_msa_to_fasta(msa_obj, path):
    AlignIO.write(msa_obj, path, "fasta")


def _mafft_add(
    new_fasta_path: str,
    msa_obj_loaded: MultipleSeqAlignment,
    workdir: str,
    *,
    bin_path: str | None = None,
    threads: int = 1,
    extra_args: Iterable[str] = (),
) -> dict:
    """Align new sequences onto `msa_obj_loaded` via ``mafft --add --keeplength``.

    ``--keeplength`` is load-bearing: it forbids MAFFT from inserting new
    columns, so `msa_obj_loaded`'s column indexing (and therefore the
    preprocessing `retained_positions` array) stays valid for the newly
    aligned rows.

    Writes the aligned new sequences (reference rows excluded) to
    ``workdir/aligned_new.fasta``.
    """
    mafft = _resolve_bin("mafft", override=bin_path)
    os.makedirs(workdir, exist_ok=True)

    ref_fpath = os.path.join(workdir, "ref.fasta")
    combined_fpath = os.path.join(workdir, "mafft_add_out.fasta")
    out_fpath = os.path.join(workdir, ALIGNED_NEW_FNAME)

    _write_msa_to_fasta(msa_obj_loaded, ref_fpath)

    with open(new_fasta_path) as f:
        new_ids = [rec.id for rec in SeqIO.parse(f, "fasta")]
    n_in = len(new_ids)
    if n_in == 0:
        raise ValueError(f"No sequences found in {new_fasta_path}")

    argv = [
        mafft, "--add", new_fasta_path, "--keeplength",
        "--thread", str(threads),
    ]
    argv.extend(extra_args)
    argv.append(ref_fpath)

    logger.info(
        "Aligning %d new sequences with mafft --add --keeplength "
        "against a %d-sequence reference",
        n_in, len(msa_obj_loaded),
    )
    logger.info("Running: %s > %s", " ".join(argv), combined_fpath)

    t0 = time.perf_counter()
    with open(combined_fpath, "w") as fout:
        try:
            subprocess.run(
                argv, check=True, stdout=fout, stderr=subprocess.PIPE,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            logger.warning("mafft --add failed (rc=%s)", e.returncode)
            if e.stderr:
                logger.warning("stderr:\n%s", e.stderr)
            raise
    elapsed = time.perf_counter() - t0

    n_ref = len(msa_obj_loaded)
    new_recs = []
    with open(combined_fpath) as f:
        for i, rec in enumerate(SeqIO.parse(f, "fasta")):
            if i < n_ref:
                continue
            new_recs.append(rec)
    if len(new_recs) != n_in:
        raise RuntimeError(
            f"mafft --add produced {len(new_recs)} new aligned rows; "
            f"expected {n_in} (one per input sequence)."
        )
    ref_len = msa_obj_loaded.get_alignment_length()
    bad = [r.id for r in new_recs if len(r.seq) != ref_len]
    if bad:
        raise RuntimeError(
            f"mafft --add --keeplength produced aligned rows whose length "
            f"({{length}}) differs from the reference ({ref_len}); affected "
            f"IDs: {bad[:5]}{'...' if len(bad) > 5 else ''}"
        )

    with open(out_fpath, "w") as fout:
        SeqIO.write(new_recs, fout, "fasta")

    logger.info(
        "mafft --add done: %d new sequences aligned in %.2fs",
        len(new_recs), elapsed,
    )
    return {
        "n_in": n_in,
        "n_out": len(new_recs),
        "elapsed_s": elapsed,
        "aligned_new_fpath": out_fpath,
    }


def _write_stockholm_with_full_rf(msa_obj, path):
    """Write an MSA to Stockholm with an ``#=GC RF`` line marking every
    column as a match column ('x'). Consumed by ``hmmbuild --hand`` so
    the resulting HMM has one match state per original MSA column.

    IDs are written as-is; Stockholm permits anything except whitespace.
    """
    L = msa_obj.get_alignment_length()
    max_id = max((len(rec.id) for rec in msa_obj), default=10)
    col = max(max_id, 10) + 2
    with open(path, "w") as f:
        f.write("# STOCKHOLM 1.0\n\n")
        for rec in msa_obj:
            f.write(f"{rec.id:<{col}}{str(rec.seq)}\n")
        f.write(f"{'#=GC RF':<{col}}{'x' * L}\n")
        f.write("//\n")


def _hmmalign(
    new_fasta_path: str,
    msa_obj_loaded: MultipleSeqAlignment,
    workdir: str,
    *,
    bin_path: str | None = None,
    hmmbuild_bin: str | None = None,
    threads: int = 1,
    extra_args: Iterable[str] = (),
) -> dict:
    """Align new sequences onto ``msa_obj_loaded`` via HMMER.

    Pipeline: build a profile HMM from the reference MSA with every
    column forced to a match state (``hmmbuild --hand``), then align
    the new sequences to that HMM (``hmmalign --outformat afa``) and
    strip lowercase insert-state columns so each aligned row has
    exactly ``L_orig`` columns.

    ``bin_path`` overrides the ``hmmalign`` binary; ``hmmbuild_bin``
    overrides the ``hmmbuild`` binary. ``threads`` is accepted for API
    parity with mafft; HMMER's alignment step is single-threaded so
    it's unused here.
    """
    del threads  # parity with mafft_add signature; hmmalign is single-threaded
    hmmbuild = _resolve_bin("hmmbuild", override=hmmbuild_bin)
    hmmalign = _resolve_bin("hmmalign", override=bin_path)
    os.makedirs(workdir, exist_ok=True)

    ref_sto = os.path.join(workdir, "ref.sto")
    hmm_path = os.path.join(workdir, "ref.hmm")
    afa_path = os.path.join(workdir, "hmmalign_out.afa")
    out_fpath = os.path.join(workdir, ALIGNED_NEW_FNAME)

    with open(new_fasta_path) as f:
        n_in = sum(1 for _ in SeqIO.parse(f, "fasta"))
    if n_in == 0:
        raise ValueError(f"No sequences found in {new_fasta_path}")

    _write_stockholm_with_full_rf(msa_obj_loaded, ref_sto)
    L_orig = msa_obj_loaded.get_alignment_length()

    logger.info(
        "hmmbuild --hand on %d reference sequences (%d columns).",
        len(msa_obj_loaded), L_orig,
    )
    t0 = time.perf_counter()
    hmmbuild_argv = [hmmbuild, "--hand", "--amino", hmm_path, ref_sto]
    try:
        subprocess.run(
            hmmbuild_argv, check=True, capture_output=True, text=True,
        )
    except subprocess.CalledProcessError as e:
        logger.warning("hmmbuild failed (rc=%s)", e.returncode)
        if e.stderr:
            logger.warning("stderr:\n%s", e.stderr)
        if e.stdout:
            logger.warning("stdout:\n%s", e.stdout)
        raise

    logger.info("hmmalign --outformat afa on %d new sequences.", n_in)
    hmmalign_argv = [hmmalign, "--outformat", "afa", hmm_path, new_fasta_path]
    hmmalign_argv.extend(extra_args)
    with open(afa_path, "w") as fout:
        try:
            subprocess.run(
                hmmalign_argv, check=True, stdout=fout,
                stderr=subprocess.PIPE, text=True,
            )
        except subprocess.CalledProcessError as e:
            logger.warning("hmmalign failed (rc=%s)", e.returncode)
            if e.stderr:
                logger.warning("stderr:\n%s", e.stderr)
            raise
    elapsed = time.perf_counter() - t0

    # hmmalign emits an A2M-style aFASTA: uppercase = match column,
    # lowercase = insert column, '-' = match column with a delete, '.'
    # = insert column with a gap. We keep only match columns so the
    # output lines up 1:1 with msa_obj_loaded's columns.
    new_recs_out = []
    for rec in SeqIO.parse(afa_path, "fasta"):
        s = str(rec.seq)
        match_only = "".join(c for c in s if c.isupper() or c == "-")
        if len(match_only) != L_orig:
            raise RuntimeError(
                f"hmmalign produced a {len(match_only)}-column aligned row "
                f"for {rec.id!r}; expected {L_orig}. HMM match-state count "
                "likely disagrees with the reference MSA."
            )
        new_recs_out.append(
            SeqRecord(Seq(match_only), id=rec.id, description=""),
        )
    if len(new_recs_out) != n_in:
        raise RuntimeError(
            f"hmmalign produced {len(new_recs_out)} aligned rows; "
            f"expected {n_in}"
        )

    with open(out_fpath, "w") as fout:
        SeqIO.write(new_recs_out, fout, "fasta")

    logger.info(
        "hmmalign done: %d sequences aligned in %.2fs", n_in, elapsed,
    )
    return {
        "n_in": n_in,
        "n_out": len(new_recs_out),
        "elapsed_s": elapsed,
        "aligned_new_fpath": out_fpath,
    }


ALIGNERS: dict[str, Callable] = {
    "mafft_add": _mafft_add,
    "hmmalign": _hmmalign,
}


def align_to_msa(
    new_fasta_path: str,
    msa_obj_loaded: MultipleSeqAlignment,
    workdir: str,
    *,
    method: str = "mafft_add",
    **method_kwargs,
) -> dict:
    """Align new FASTA sequences onto `msa_obj_loaded`'s column structure.

    Returns a dict with ``aligned_new_fpath`` (path to the aligned-only-
    new-sequences FASTA) plus any backend-specific diagnostics.
    """
    if method not in ALIGNERS:
        raise ValueError(
            f"Unknown aligner {method!r}. Known: {sorted(ALIGNERS)}"
        )
    os.makedirs(workdir, exist_ok=True)
    return ALIGNERS[method](
        new_fasta_path, msa_obj_loaded, workdir, **method_kwargs,
    )
