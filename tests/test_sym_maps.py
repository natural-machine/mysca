"""SymMap tests

"""

import pytest
from contextlib import nullcontext as does_not_raise
from tests.conftest import DATDIR, TMPDIR, remove_dir

from mysca.mappings import SymMap, DEFAULT_MAP


#####################
##  Configuration  ##
#####################


        
###############################################################################
###############################   BEGIN TESTS   ###############################
###############################################################################


def test_default_mapping():
    exp_num_aas = 20
    exp_num_syms = 21
    exp_gapsym = "-"
    exp_gapint = 0
    
    mapping = DEFAULT_MAP
    errors = []
    if not (exp_num_aas == len(mapping.aa_list)):
        msg = f"Wrong number of aas in default map."
        errors.append(msg)
    if not (exp_num_syms == len(mapping.sym_list)):
        msg = f"Wrong number of syms in default map."
        errors.append(msg)
    if not (exp_gapsym == mapping.gapsym):
        msg = f"Wrong gapsym in default map."
        errors.append(msg)
    if not (exp_gapint == mapping.gapint):
        msg = f"Wrong gapint in default map."
        errors.append(msg)
    assert not errors, "Errors occurred:\n{}".format("\n".join(errors))
    

@pytest.fixture(params=[
    {
        "args": ["ABCDE", "-", ""],
        "exp_gapsym": "-",
        "exp_gapint": 0,
        "exp_num_exclusions": 0,
    },
    {
        "args": ["ABCDE", "-", "X"],
        "exp_gapsym": "-",
        "exp_gapint": 0,
        "exp_num_exclusions": 1,
    },
])
def mapping_and_expecteds(request) -> tuple[SymMap, dict]:
    mapping = SymMap(*request.param["args"])
    return mapping, request.param


class TestMapping:
    
    def test_gapint(self, mapping_and_expecteds):
        mapping, expecteds = mapping_and_expecteds
        exp_gapsym = expecteds["exp_gapsym"]
        exp_gapint = expecteds["exp_gapint"]

        errors = []
        if not (mapping.gapint == exp_gapint):
            msg = f"Wrong gapint from direct attribute. "
            msg += f"Expected {exp_gapint}. Got {mapping.gapint}."
            errors.append(msg)
        if not (mapping.gapsym == exp_gapsym):
            msg = f"Wrong gapsym from direct attribute. "
            msg += f"Expected {exp_gapsym}. Got {mapping.gapsym}."
            errors.append(msg)
        if not (mapping[mapping.gapsym] == exp_gapint):
            msg = "Wrong gapint from gapsym value. "
            msg += f"Expected {exp_gapint}. Got {mapping[mapping.gapsym]}."
            errors.append(msg)
        assert not errors, "Errors occurred:\n{}".format("\n".join(errors))
        
    def test_exclusions(self, mapping_and_expecteds):
        mapping, expecteds = mapping_and_expecteds
        exp_num_exclusions = expecteds["exp_num_exclusions"]
        num_exclusions = len(mapping.exclude_syms)
        msg = "Incorrect number of excluded syms."
        msg += f"Expected {exp_num_exclusions}. Got {num_exclusions}."
        assert num_exclusions == exp_num_exclusions


@pytest.mark.parametrize("gap_value,exp_sym_list,exp_gapint", [
    (0, list("-ABCDE"), 0),
    (1, list("A-BCDE"), 1),
    (3, list("ABC-DE"), 3),
    (5, list("ABCDE-"), 5),
])
def test_gap_value_parameter(gap_value, exp_sym_list, exp_gapint):
    mapping = SymMap("ABCDE", "-", gap_value=gap_value)
    assert mapping.sym_list == exp_sym_list
    assert mapping.gapint == exp_gapint
    assert mapping.aa_list == list("ABCDE")
    for aa in "ABCDE":
        assert mapping[aa] != mapping.gapint


@pytest.mark.parametrize("gap_value", [-1, 6, 100])
def test_gap_value_out_of_range_raises(gap_value):
    with pytest.raises(ValueError):
        SymMap("ABCDE", "-", gap_value=gap_value)
