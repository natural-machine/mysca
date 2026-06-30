"""Build a guide tree over a family subsample with FAMSA.

The phylogeny analyses (:mod:`mysca.examine.pagel`, :mod:`mysca.examine.splits`)
need a tree relating the sampled sequences. We use FAMSA's guide-tree export
(``-gt_export``) rather than a full alignment, which is fast and keeps the
sca-examine pipeline self-contained. A user who already has a tree they trust
can bypass this entirely by passing it to ``sca-examine --tree``.

Each subsampled sequence is given a synthetic, delimiter-free label
(``s0``, ``s1``, ...). The accession extractors in :mod:`mysca.examine.newick`
act as the identity on these labels, so the tree<->projection join is exact and
decoupled from any UniProt naming convention.
"""

import logging
import os
import shutil
import subprocess

import numpy as np

logger = logging.getLogger("mysca.examine.guide_tree")

GT_METHODS = ("sl", "upgma", "nj")


def resolve_famsa(famsa_bin=None):
    """Resolve the FAMSA binary path, or raise with an install hint."""
    if famsa_bin and os.path.exists(famsa_bin):
        return famsa_bin
    found = shutil.which("famsa") or shutil.which("FAMSA")
    if found:
        return found
    raise FileNotFoundError(
        "FAMSA binary not found on PATH. Install FAMSA "
        "(https://github.com/refresh-bio/FAMSA), pass --famsa_bin "
        "/path/to/famsa, or supply your own guide tree with --tree."
    )


def write_subsample_fasta(path, labels, msa_int_sub, int2char):
    """Write ungapped sequences (FAMSA strips gaps anyway) under the synthetic
    delimiter-free ``labels``."""
    gap_idx = int(np.where(int2char == "-")[0][0])
    with open(path, "w") as fh:
        for lab, row in zip(labels, msa_int_sub):
            seq = "".join(int2char[v] for v in row if v != gap_idx)
            fh.write(f">{lab}\n{seq}\n")


def build_guide_tree(fasta_path, tree_path, *, famsa_bin=None,
                     gt_method="nj", threads=0):
    """Run FAMSA to export only a Newick guide tree (no full alignment).
    Returns the Newick text."""
    famsa = resolve_famsa(famsa_bin)
    cmd = [famsa, "-gt", gt_method, "-gt_export",
           "-t", str(threads), fasta_path, tree_path]
    logger.info("  $ %s", " ".join(cmd))
    subprocess.run(cmd, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not os.path.exists(tree_path) or os.path.getsize(tree_path) == 0:
        raise RuntimeError(f"FAMSA produced no guide tree at {tree_path}")
    with open(tree_path) as fh:
        return fh.read()
