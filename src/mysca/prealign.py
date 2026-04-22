"""Pre-alignment stage: cluster and align raw (unaligned) FASTA.

Provides thin wrappers around external tools (mmseqs2, MAFFT) that shell out
via subprocess. The public surface is `run_cluster` and `run_align`, dispatched
through the `CLUSTERERS` and `ALIGNERS` registries so additional tools can be
added without changing the CLI.

Binaries must be resolvable via `shutil.which` (i.e. present on PATH) or via
an explicit `bin_path` override. `_resolve_bin` raises FileNotFoundError
immediately if a required binary is missing.
"""

import logging
import os
import shutil
import subprocess
import tempfile
import time
from typing import Iterable

from Bio import AlignIO, SeqIO

logger = logging.getLogger("mysca.prealign")

SUPPORTED_ALIGNMENT_FORMATS = ("fasta", "stockholm")


def _resolve_bin(binary_name: str, override: str | None = None) -> str:
    """Resolve an external binary to an absolute path, or raise.

    If `override` is given, it is used directly (either an absolute path or a
    name looked up on PATH). Otherwise `binary_name` is looked up on PATH.
    The resolved path is logged at INFO.
    """
    candidate = override or binary_name
    resolved = shutil.which(candidate)
    if resolved is None:
        raise FileNotFoundError(
            f"Required binary {candidate!r} not found on PATH. "
            f"Make it available on PATH (e.g. via "
            f"`conda install -c bioconda {binary_name}`) or pass an explicit "
            f"path via the corresponding --*_bin CLI flag."
        )
    logger.info("Resolved %s binary to %s", binary_name, resolved)
    return resolved


def _run_cmd(argv: list[str], *, cwd: str | None = None) -> subprocess.CompletedProcess:
    """Run a subprocess, logging argv and propagating failures."""
    logger.info("Running: %s", " ".join(argv))
    try:
        return subprocess.run(
            argv, cwd=cwd, check=True, capture_output=True, text=True,
        )
    except subprocess.CalledProcessError as e:
        logger.warning("Command failed (rc=%s): %s", e.returncode, " ".join(argv))
        if e.stdout:
            logger.warning("stdout:\n%s", e.stdout)
        if e.stderr:
            logger.warning("stderr:\n%s", e.stderr)
        raise


def _count_fasta(fpath: str) -> int:
    with open(fpath) as f:
        return sum(1 for rec in SeqIO.parse(f, "fasta"))


def _cluster_mmseqs2(
    in_fasta: str,
    out_fasta: str,
    *,
    min_seq_id: float,
    coverage: float,
    coverage_mode: int,
    threads: int,
    bin_path: str | None,
    tmpdir: str | None,
) -> dict:
    mmseqs = _resolve_bin("mmseqs", override=bin_path)
    n_in = _count_fasta(in_fasta)
    logger.info(
        "Clustering %d sequences with mmseqs2 easy-cluster "
        "(min_seq_id=%s, coverage=%s, cov_mode=%s, threads=%d)",
        n_in, min_seq_id, coverage, coverage_mode, threads,
    )

    if tmpdir is None:
        tmp_ctx = tempfile.TemporaryDirectory()
        tmp_root = tmp_ctx.name
    else:
        os.makedirs(tmpdir, exist_ok=True)
        tmp_ctx = None
        tmp_root = tmpdir

    try:
        prefix = os.path.join(tmp_root, "cluster")
        mmseqs_tmp = os.path.join(tmp_root, "mmseqs_tmp")
        os.makedirs(mmseqs_tmp, exist_ok=True)
        argv = [
            mmseqs, "easy-cluster",
            in_fasta, prefix, mmseqs_tmp,
            "--min-seq-id", str(min_seq_id),
            "-c", str(coverage),
            "--cov-mode", str(coverage_mode),
            "--threads", str(threads),
        ]
        t0 = time.perf_counter()
        _run_cmd(argv)
        elapsed = time.perf_counter() - t0

        rep_fasta = f"{prefix}_rep_seq.fasta"
        if not os.path.isfile(rep_fasta):
            raise RuntimeError(
                f"mmseqs2 did not produce expected output {rep_fasta}"
            )
        os.makedirs(os.path.dirname(os.path.abspath(out_fasta)), exist_ok=True)
        shutil.copyfile(rep_fasta, out_fasta)
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()

    n_out = _count_fasta(out_fasta)
    logger.info(
        "Clustering done: %d -> %d representative sequences in %.2fs",
        n_in, n_out, elapsed,
    )
    return {"n_in": n_in, "n_out": n_out, "elapsed_s": elapsed}


CLUSTERERS = {
    "mmseqs2": _cluster_mmseqs2,
}


def run_cluster(
    in_fasta: str,
    out_fasta: str,
    *,
    method: str = "mmseqs2",
    min_seq_id: float = 0.9,
    coverage: float = 0.8,
    coverage_mode: int = 1,
    threads: int = 1,
    bin_path: str | None = None,
    tmpdir: str | None = None,
) -> dict:
    """Cluster raw FASTA sequences, writing representatives to `out_fasta`."""
    if method not in CLUSTERERS:
        raise ValueError(
            f"Unknown cluster method {method!r}. "
            f"Known: {sorted(CLUSTERERS)}"
        )
    return CLUSTERERS[method](
        in_fasta, out_fasta,
        min_seq_id=min_seq_id,
        coverage=coverage,
        coverage_mode=coverage_mode,
        threads=threads,
        bin_path=bin_path,
        tmpdir=tmpdir,
    )


def _align_mafft(
    in_fasta: str,
    out_path: str,
    *,
    threads: int,
    bin_path: str | None,
    extra_args: Iterable[str],
    output_format: str,
) -> dict:
    mafft = _resolve_bin("mafft", override=bin_path)
    n_in = _count_fasta(in_fasta)
    logger.info(
        "Aligning %d sequences with MAFFT --auto (threads=%d, output_format=%s)",
        n_in, threads, output_format,
    )

    argv = [mafft, "--auto", "--thread", str(threads)]
    argv.extend(extra_args)
    argv.append(in_fasta)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    # MAFFT writes FASTA natively. If the caller wants Stockholm, run MAFFT
    # into a temp file and convert with Bio.AlignIO.
    if output_format == "fasta":
        fasta_sink = out_path
        tmp_fasta = None
    else:
        tmp_fd, tmp_fasta = tempfile.mkstemp(suffix=".fasta")
        os.close(tmp_fd)
        fasta_sink = tmp_fasta

    logger.info("Running: %s > %s", " ".join(argv), fasta_sink)
    t0 = time.perf_counter()
    try:
        with open(fasta_sink, "w") as fout:
            try:
                subprocess.run(
                    argv, check=True, stdout=fout, stderr=subprocess.PIPE, text=True,
                )
            except subprocess.CalledProcessError as e:
                logger.warning("MAFFT failed (rc=%s)", e.returncode)
                if e.stderr:
                    logger.warning("stderr:\n%s", e.stderr)
                raise

        if output_format != "fasta":
            AlignIO.convert(fasta_sink, "fasta", out_path, output_format)
    finally:
        if tmp_fasta is not None and os.path.isfile(tmp_fasta):
            os.unlink(tmp_fasta)
    elapsed = time.perf_counter() - t0

    n_out = _count_aligned(out_path, output_format)
    logger.info(
        "Alignment done: %d sequences written to %s in %.2fs",
        n_out, out_path, elapsed,
    )
    return {"n_in": n_in, "n_out": n_out, "elapsed_s": elapsed}


def _count_aligned(fpath: str, fmt: str) -> int:
    with open(fpath) as f:
        return sum(1 for _ in AlignIO.read(f, fmt))


ALIGNERS = {
    "mafft": _align_mafft,
}


def run_align(
    in_fasta: str,
    out_path: str,
    *,
    method: str = "mafft",
    threads: int = 1,
    bin_path: str | None = None,
    extra_args: Iterable[str] = (),
    output_format: str = "fasta",
) -> dict:
    """Align FASTA sequences, writing an aligned MSA to `out_path`.

    `output_format` controls the written alignment format: "fasta" (default)
    or "stockholm".
    """
    if method not in ALIGNERS:
        raise ValueError(
            f"Unknown alignment method {method!r}. "
            f"Known: {sorted(ALIGNERS)}"
        )
    if output_format not in SUPPORTED_ALIGNMENT_FORMATS:
        raise ValueError(
            f"Unknown output_format {output_format!r}. "
            f"Supported: {list(SUPPORTED_ALIGNMENT_FORMATS)}"
        )
    return ALIGNERS[method](
        in_fasta, out_path,
        threads=threads,
        bin_path=bin_path,
        extra_args=tuple(extra_args),
        output_format=output_format,
    )
