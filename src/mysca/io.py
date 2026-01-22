"""Input/Output functions

"""

import numpy as np
from numpy.typing import NDArray
from Bio import AlignIO
from Bio.PDB import PDBParser
from Bio.PDB.Structure import Structure
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.AlignIO import MultipleSeqAlignment

from mysca.mappings import SymMap, DEFAULT_MAP


def load_msa(
        fpath, 
        format: str = "fasta", 
        mapping: SymMap = None,
        verbosity: int = 2,
) -> tuple[MultipleSeqAlignment, NDArray[np.int_], list, SymMap]:
    """Load an MSA fasta file and return the MSA object, matrix, and IDs.
    
    Filters out any sequences that contain excluded characters in the given 
    mapping.

    Args:
        fpath (str): Path to input MSA file.
        format (str): Format of the input file. Default "fasta".
        mapping (SeqMap): SeqMap object defining the mapping from AAs to ints,
            or None. If None, determine the mapping from the given MSA.
        verbosity (int): verbosity level. Default 1.
    
    Returns:
        MultipleSeqAlignment: MSA object.
        NDArray[int]: Matrix representation of the MSA, as defined by the 
            given mapping.
        list[str]: Sequence IDs as defined in the input fasta file, that are 
            retained in the MSA.
        SymMap: Mapping from amino acids to index used to compute MSA matrix. 
    """
    msa_obj = AlignIO.read(fpath, format)
    
    # Convert all symbols to uppercase
    for entry in msa_obj:
        entry.seq = Seq(str(entry.seq).upper())

    GAPSYM = "-"
    if mapping is None:
        seq_join = ""
        for entry in msa_obj:
            seq = str(entry.seq)
            seq_join += seq
        aa_syms = np.sort(np.unique([c for c in seq_join if c != GAPSYM]))
        aa_syms = "".join(aa_syms)
        mapping = SymMap(aa_syms, gapsym=GAPSYM, exclude_syms="")

    # Keep records in the MSA not containing excluded symbols.
    exc_recs_screen = np.array([
        any([sym in str(record.seq) for sym in mapping.exclude_syms]) 
        for record in msa_obj
    ], dtype=bool)

    keep_records = [
        msa_obj[int(i)] for i in np.arange(len(msa_obj))[~exc_recs_screen]
    ]

    assert exc_recs_screen.sum() + len(keep_records) == len(msa_obj)

    msa_obj = MultipleSeqAlignment(keep_records)

    if verbosity > 1:
        print(f"Removed {exc_recs_screen.sum()} seqs with excluded syms.")

    # Construct the MSA matrix.
    msa_matrix = np.array([
        [mapping[aa] for aa in record.seq] for record in msa_obj 
        if np.all([excsym not in record.seq for excsym in mapping.exclude_syms])
    ])

    # Retrieve MSA sequence IDs.
    msa_ids = [record.id for record in msa_obj]
    return msa_obj, msa_matrix, msa_ids, mapping


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
