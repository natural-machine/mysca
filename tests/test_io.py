"""IO tests

"""

import pytest
import numpy as np
from contextlib import nullcontext as does_not_raise
from tests.conftest import DATDIR, TMPDIR, remove_dir

from Bio import AlignIO
from mysca.io import load_msa
from mysca.mappings import SymMap


#####################
##  Configuration  ##
#####################

SYMMAP1 = SymMap("ACDEF", '-')
SYMMAP1_NO_EXCLUDE = SymMap("ACDEF", '-', exclude_syms="")
SYMMAP1_EXC_X = SymMap("ACDEF", '-', "X")
        
###############################################################################
###############################   BEGIN TESTS   ###############################
###############################################################################

@pytest.mark.parametrize(
        "fa_fpath, msa_shape_exp, symmap, expect_context", [
    [f"{DATDIR}/msas/msa01.faa", (5, 10), SYMMAP1, does_not_raise()],
    [f"{DATDIR}/msas/msa02.faa", (2, 10), SYMMAP1, does_not_raise()],
    [f"{DATDIR}/msas/msa02.faa", None, SYMMAP1_NO_EXCLUDE, pytest.raises(KeyError)],
    [f"{DATDIR}/msas/msa02.faa", (2, 10), SYMMAP1_EXC_X, does_not_raise()],
    [f"{DATDIR}/msas/msa03.faa", (5, 10), SYMMAP1_EXC_X, does_not_raise()],
])
def test_load_msa(fa_fpath, msa_shape_exp, symmap, expect_context):
    with expect_context:
        msa_obj, msa, msa_ids, _, _, _ = load_msa(
            fa_fpath, format="fasta",
            mapping=symmap
        )

        errors = []
        if not isinstance(msa_obj, AlignIO.MultipleSeqAlignment):
            msg = "Expected type AlignIO.MultipleSeqAlignment for msa_obj."
            msg += f" Got {type(msa_obj)}."
            errors.append(msg)
        if not isinstance(msa, np.ndarray):
            msg = "Expected type np.ndarray for msa."
            msg += f" Got {type(msa)}."
            errors.append(msg)
        if not isinstance(msa_ids, list):
            msg = "Expected type list for msa_ids."
            msg += f" Got {type(msa_ids)}."
            errors.append(msg)
        if msa.shape != msa_shape_exp:
            msg = f"Expected msa shape {msa_shape_exp}. Got {msa.shape}."
            errors.append(msg)
        if len(msa_obj) != msa_shape_exp[0]:
            msg = f"Expected msa_obj length {msa_shape_exp[0]}. Got {len(msa_obj)}."
            errors.append(msg)
        assert not errors, "Errors occurred:\n{}".format("\n".join(errors))
    
    
