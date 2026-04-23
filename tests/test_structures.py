"""Protein structure tests

"""

import pytest
from contextlib import nullcontext as does_not_raise
from tests.conftest import DATDIR, TMPDIR, remove_dir

from mysca.io import load_pdb_structure
from mysca.structure import struct2seq
from Bio import SeqIO


#####################
##  Configuration  ##
#####################


        
###############################################################################
###############################   BEGIN TESTS   ###############################
###############################################################################

@pytest.mark.parametrize('fa_fpath, pdb_fpath', [
    [f"{DATDIR}/seqs/Soil3.scaffold_414071996_c1_8.fasta",
     f"{DATDIR}/structs/Soil3.scaffold_414071996_c1_8.pdb"],
])
def test_struct2seq(fa_fpath, pdb_fpath):
    seqs = []
    for record in SeqIO.parse(fa_fpath, "fasta"):
        seqs.append(record.seq)
    assert len(seqs) == 1, \
        f"Test file {fa_fpath} should contain only 1 sequence. Got {len(seqs)}."
    
    seq_exp = seqs[0]
    struct = load_pdb_structure(pdb_fpath, "structure")
    seq = struct2seq(struct)
    msg = f"\tGot sequence: {seq[0:10]}...{seq[-10:]}\n" + \
          f"\t    Expected: {seq_exp[0:10]}...{seq_exp[-10:]}"
    assert seq == seq_exp, msg
