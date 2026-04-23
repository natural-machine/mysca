"""Out-of-sample sequence alignment for `mysca.project`.

Dispatch registry mapping a method name to a callable that aligns new
sequences onto an existing MSA's column structure (preserving column
count so that `retained_positions` indexing stays valid downstream).

Public surface:

- ``ALIGNERS`` — ``{name: callable}``.
- ``register_aligner(name)`` — decorator to register a new aligner.
- ``align_to_msa(...)`` — top-level dispatch.

Each aligner callable has signature
``(new_fasta_path, msa_obj_orig, workdir) -> dict`` and must write the
aligned output for *only the new sequences* to
``workdir/aligned_new.fasta``. The length of each aligned sequence
must equal the length of ``msa_obj_orig`` (column-preserving). Return
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
    msa_obj_orig: MultipleSeqAlignment,
    workdir: str,
    *,
    bin_path: str | None = None,
    threads: int = 1,
    extra_args: Iterable[str] = (),
) -> dict:
    """Align new sequences onto `msa_obj_orig` via ``mafft --add --keeplength``.

    ``--keeplength`` is load-bearing: it forbids MAFFT from inserting new
    columns, so `msa_obj_orig`'s column indexing (and therefore the
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

    _write_msa_to_fasta(msa_obj_orig, ref_fpath)

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
        n_in, len(msa_obj_orig),
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

    n_ref = len(msa_obj_orig)
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
    ref_len = msa_obj_orig.get_alignment_length()
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


def _hmmalign(*args, **kwargs):
    raise NotImplementedError(
        "hmmalign backend is registered but not yet implemented. "
        "Default to 'mafft_add' or contribute an hmmalign wrapper."
    )


ALIGNERS: dict[str, Callable] = {
    "mafft_add": _mafft_add,
    "hmmalign": _hmmalign,
}


def align_to_msa(
    new_fasta_path: str,
    msa_obj_orig: MultipleSeqAlignment,
    workdir: str,
    *,
    method: str = "mafft_add",
    **method_kwargs,
) -> dict:
    """Align new FASTA sequences onto `msa_obj_orig`'s column structure.

    Returns a dict with ``aligned_new_fpath`` (path to the aligned-only-
    new-sequences FASTA) plus any backend-specific diagnostics.
    """
    if method not in ALIGNERS:
        raise ValueError(
            f"Unknown aligner {method!r}. Known: {sorted(ALIGNERS)}"
        )
    os.makedirs(workdir, exist_ok=True)
    return ALIGNERS[method](
        new_fasta_path, msa_obj_orig, workdir, **method_kwargs,
    )
