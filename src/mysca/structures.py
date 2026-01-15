"""Protein structures

"""

from Bio.PDB import PPBuilder
from Bio.PDB.Structure import Structure


def struct2seq(structure: Structure) -> str:
    """Convert a protein structure into its amino acid sequence.
    
    Args:
        structure (Structure): Protein structure.

    Returns:
        (str) amino acid sequence of the protein structure.
    """
    ppb = PPBuilder()
    for pp in ppb.build_peptides(structure):
        protein_seq = pp.get_sequence()  # returns a Bio.Seq.Seq object
        break
    return str(protein_seq)
