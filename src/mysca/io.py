"""Input/Output functions

"""

import logging

import numpy as np
from numpy.typing import NDArray
from Bio import AlignIO
from Bio.PDB import PDBParser
from Bio.PDB.Structure import Structure
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.AlignIO import MultipleSeqAlignment

from mysca.constants import AA_STD20
from mysca.mappings import SymMap, DEFAULT_MAP

logger = logging.getLogger(__name__)


def _strip_trailing_stops(seq_str: str) -> tuple[str, int, bool]:
    """Replace trailing stop-codon characters (``*``) with gaps (``-``).

    "Trailing" means: at the end of the residue content, ignoring gaps.
    A sequence like ``ACDE*-`` has a trailing ``*``; so does
    ``ACDE-*--`` (the ``*`` comes after ``E``, separated only by gaps).
    A ``*`` that has any non-gap residue to its right is interior.

    Returns:
        cleaned (str): the sequence with trailing ``*`` replaced by ``-``.
        n_trailing_replaced (int): how many trailing ``*`` were replaced.
        has_internal_stop (bool): True iff any ``*`` remains after the
            trailing strip — interpreted as a premature stop / frameshift
            and a strong signal to drop the sequence.
    """
    seq_list = list(seq_str)
    n_trailing = 0
    i = len(seq_list) - 1
    while i >= 0:
        if seq_list[i] == "*":
            seq_list[i] = "-"
            n_trailing += 1
            i -= 1
        elif seq_list[i] == "-":
            i -= 1
        else:
            break
    cleaned = "".join(seq_list)
    return cleaned, n_trailing, ("*" in cleaned)


def load_msa(
        fpath,
        format: str = "fasta",
        mapping: SymMap = None,
        verbosity: int = 2,
) -> tuple[MultipleSeqAlignment, NDArray[np.int_], list, SymMap, int, int]:
    """Load an MSA fasta file and return the MSA object, matrix, and IDs.

    Performs three filter steps on the input records, in order:

    1. **Trailing stop codon (``*``) strip.** Trailing ``*`` is a normal
       CDS-translation artifact; replace with gap. Counted but not
       reported separately (info-level log only — sequence content
       changes, no rows dropped).
    2. **Internal stop codon drop.** Any ``*`` remaining after step 1
       indicates a premature stop / frameshift; drop the sequence.
    3. **Excluded-symbol drop.** Any sequence still containing a
       symbol marked excluded by ``mapping`` (default: anything not in
       the canonical AA alphabet + gap) is dropped.

    Args:
        fpath (str): Path to input MSA file.
        format (str): Format of the input file. Default "fasta".
        mapping (SeqMap): SeqMap object defining the mapping from AAs to ints,
            or None. If None, determine the mapping from the given MSA.
        verbosity (int): verbosity level. Default 1.

    Returns:
        MultipleSeqAlignment: MSA object (post-filter).
        NDArray[int]: Matrix representation of the MSA.
        list[str]: Retained sequence IDs.
        SymMap: Mapping from amino acids to index used to compute MSA matrix.
        int: Number of input sequences dropped at the excluded-symbol step.
        int: Number of input sequences dropped because they contained an
            internal stop codon (post-trailing-strip).
    """
    msa_obj = AlignIO.read(fpath, format)

    # Convert all symbols to uppercase
    for entry in msa_obj:
        entry.seq = Seq(str(entry.seq).upper())

    # Stop-codon handling. Trailing * (relative to non-gap content) is a
    # normal CDS-translation artifact: replace with gap so the sequence
    # is preserved and column alignment is maintained. Internal * (after
    # trailing strip) almost always means premature stop / frameshift —
    # drop the sequence and count separately so it shows as its own
    # filter_history stage. We never let * through to the alphabet check.
    n_trailing_stripped = 0
    internal_stop_mask = np.zeros(len(msa_obj), dtype=bool)
    for k, entry in enumerate(msa_obj):
        cleaned, n_replaced, has_internal = _strip_trailing_stops(
            str(entry.seq)
        )
        if n_replaced > 0:
            entry.seq = Seq(cleaned)
            n_trailing_stripped += 1
        internal_stop_mask[k] = has_internal
    if n_trailing_stripped > 0:
        logger.info(
            "Stripped trailing '*' (replaced with gap) in %d sequences.",
            n_trailing_stripped,
        )
    n_internal_stop = int(internal_stop_mask.sum())
    if n_internal_stop > 0:
        logger.warning(
            "Dropping %d sequences with internal '*' "
            "(premature stop / frameshift).",
            n_internal_stop,
        )
        kept = [
            msa_obj[int(i)]
            for i in np.arange(len(msa_obj))[~internal_stop_mask]
        ]
        msa_obj = MultipleSeqAlignment(kept)

    GAPSYM = "-"
    if mapping is None:
        seq_join = ""
        for entry in msa_obj:
            seq = str(entry.seq)
            seq_join += seq
        aa_syms = np.sort(np.unique([c for c in seq_join if c != GAPSYM]))
        aa_syms = "".join(aa_syms)
        non_canon = sorted(set(aa_syms) - set(AA_STD20))
        if non_canon:
            logger.warning(
                "Auto-detected alphabet contains non-canonical symbols: %s "
                "— no filtering applied.", non_canon
            )
        mapping = SymMap(aa_syms, gapsym=GAPSYM, exclude_syms="")

    # Keep records in the MSA not containing excluded symbols.
    exc_recs_screen = np.array([
        any(mapping.is_excluded(sym) for sym in str(record.seq))
        for record in msa_obj
    ], dtype=bool)

    keep_records = [
        msa_obj[int(i)] for i in np.arange(len(msa_obj))[~exc_recs_screen]
    ]

    assert exc_recs_screen.sum() + len(keep_records) == len(msa_obj)

    msa_obj = MultipleSeqAlignment(keep_records)

    n_removed = int(exc_recs_screen.sum())
    if n_removed > 0:
        logger.warning(
            "Removed %d sequences containing excluded symbols.", n_removed
        )
    else:
        logger.debug("No sequences removed during excluded-symbol filtering.")

    # Construct the MSA matrix.
    msa_matrix = np.array([
        [mapping[aa] for aa in record.seq] for record in msa_obj
    ])

    # Retrieve MSA sequence IDs.
    msa_ids = [record.id for record in msa_obj]
    return msa_obj, msa_matrix, msa_ids, mapping, n_removed, n_internal_stop


def load_pdb_structure(
        fpath: str, 
        id: str,
        quiet: bool = True
) -> Structure:
    """Load a PDB structure from a pdb file.
    
    Args:
        fpath (str): path to pdb file.
        id (str): the id for the returned structure.
    
    Returns:
        (Structure) Protein structure.
    """
    parser = PDBParser(QUIET=quiet)
    structure = parser.get_structure(id, fpath)
    return structure


def get_residue_sequence_from_pdb_structure(
        struct: Structure, 
) -> list[str]:
    """Return list of residues from a PDB protein structure.

    Assumes a single chain.

    Args:
        struct (Structure): Protein structure object

    Returns:
        list[str]: List of residues.
    """
    # Assuming single chain
    chain = next(struct.get_chains())
    # Map sequence indices to PDB residue numbers
    residues = [r for r in chain.get_residues() if r.id[0] == " "]
    return residues


def msa_from_aligned_seqs(
        seqs_aligned: list[str],
        ids: list[str] = None,
) -> MultipleSeqAlignment:
    """Get an MSA from a list of aligned sequences.

    Args:
        seqs_aligned (list[str]): Aligned sequences, all of the same length.
        ids (list[str], optional): List of IDs. Defaults to None.

    Returns:
        MultipleSeqAlignment: MSA object.
    """
    if ids is None:
        ids = [f"sequence{i}" for i in range(len(seqs_aligned))]
    records = [
        SeqRecord(Seq(s), id=ids[i]) for i, s in enumerate(seqs_aligned)
    ]
    msa_obj = MultipleSeqAlignment(records)
    return msa_obj
