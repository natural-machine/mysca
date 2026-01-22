"""Convenience tools

"""

from Bio import SeqIO
from Bio import AlignIO


def convert_msa(infile, outfile, itype, otype):
    "Convert an MSA file from one format into another."
    ftypes = ["stockholm", "fasta"]
    assert itype in ftypes, f"Input type should be one of {ftypes}"
    assert otype in ftypes, f"Output type should be one of {ftypes}"
    alignment = AlignIO.read(infile, itype)
    AlignIO.write(alignment, outfile, otype)
    return


def remove_sequences_with_X(input_fasta, output_fasta):
    """Read a FASTA file and write only sequences that do NOT contain 'X'."""
    with open(input_fasta) as infile, open(output_fasta, "w") as outfile:
        kept_records = (
            record for record in SeqIO.parse(infile, "fasta")
            if "X" not in str(record.seq).upper()
        )
        count = SeqIO.write(kept_records, outfile, "fasta")

    print(f"Wrote {count} sequences (removed those containing 'X').")
        
    
