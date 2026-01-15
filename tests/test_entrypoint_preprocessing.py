"""Preprocessing entrypoint tests

"""

import pytest
import os
import numpy as np
import scipy.sparse as sp
import json
from contextlib import nullcontext as does_not_raise

from tests.conftest import DATDIR, TMPDIR, remove_dir

from mysca.run_preprocessing import parse_args, main


#####################
##  Configuration  ##
#####################
 
def get_args(fpath):
    with open(fpath, 'r') as f:
        argstring = f.readline()
        arglist = argstring.split(" ")
        return arglist
        
###############################################################################
###############################   BEGIN TESTS   ###############################
###############################################################################

@pytest.mark.parametrize(
        "argstring_fpath, sym2int_exp, results_exp, msa_binary3d_shape_exp", [
    [f"{DATDIR}/entrypoint_tests/preprocessing/argstrings/argstring4a.txt",
     {"A":0,"B":1,"C":2,"D":3,"E":4,"F":5,"G":6,"H":7,"-":8},
     {
        "retained_sequences": np.arange(20),
        "retained_positions": np.concatenate(  # remove positions 10, 16, 17
            [np.arange(10), np.arange(11, 16), np.arange(18, 22)]),
        "sequence_weights": np.array([
            0.25, 0.1, 0.25, 0.25, 0.25, 0.1, 0.2, 0.2, 0.1, 
            0.1, 1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.1, 0.2, 0.1
        ]),
        "msa": np.array([
            [6,6,6,6,6,6,6,6,6,6,0,1,1,2,2,3,3,5,6],
            [6,6,6,6,6,6,6,6,6,6,0,1,1,2,2,3,3,4,4],
            [6,6,6,6,6,6,6,6,6,6,0,1,1,2,2,3,3,5,6],
            [6,6,6,6,6,6,6,6,6,6,0,1,1,2,2,3,3,5,6],
            [6,6,6,6,6,6,6,6,6,6,0,1,1,2,2,3,3,5,6],
            [6,6,6,6,6,6,6,6,6,6,0,1,1,2,2,3,3,4,4],
            [6,6,6,6,6,6,6,6,6,6,2,1,1,2,2,3,3,6,5],
            [6,6,6,6,6,6,6,6,6,6,2,1,1,2,2,3,3,6,5],
            [6,6,6,6,6,6,6,6,6,6,0,1,1,2,2,3,3,4,4],
            [6,6,6,6,6,6,6,6,6,6,0,1,1,2,2,3,3,4,4],
            [6,6,6,6,6,6,6,6,6,6,3,1,1,2,2,3,3,6,5],
            [6,6,6,6,6,6,6,6,6,6,0,1,1,2,2,3,3,4,4],
            [6,6,6,6,6,6,6,6,6,6,0,1,1,2,2,3,3,4,4],
            [6,6,6,6,6,6,6,6,6,6,0,1,1,2,2,3,3,4,4],
            [6,6,6,6,6,6,6,6,6,6,0,1,1,2,2,3,3,4,4],
            [6,6,6,6,6,6,6,6,6,6,2,1,1,2,2,3,3,6,5],
            [6,6,6,6,6,6,6,6,6,6,2,1,1,2,2,3,3,6,5],
            [6,6,6,6,6,6,6,6,6,6,0,1,1,2,2,3,3,4,4],
            [6,6,6,6,6,6,6,6,6,6,2,1,1,2,2,3,3,6,5],
            [6,6,6,6,6,6,6,6,6,6,0,1,1,2,2,3,3,4,4],
        ]),
     },
     (20,19,8),
    ],
    [f"{DATDIR}/entrypoint_tests/preprocessing/argstrings/argstring5a.txt",
     {"A":0,"B":1,"C":2,"D":3,"E":4,"F":5,"G":6,"H":7,"-":8},
     {
        "retained_sequences": np.array([0,1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,19,20]),
        "retained_positions": np.array([0,1,2,3,4,5,6,7,8,9,11,12,13,14,15,18,19,20,21]), 
        "sequence_weights": np.array([
            0.25, 0.1, 0.25, 0.25, 0.25, 0.1, 0.2, 0.2, 0.1, 
            0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.1, 0.2, 0.1, 1
        ]),
        "msa": np.array([
            [6,6,6,6,6,6,6,6,6,6,0,1,1,2,2,3,3,5,6],
            [6,6,6,6,6,6,6,6,6,6,0,1,1,2,2,3,3,4,4],
            [6,6,6,6,6,6,6,6,6,6,0,1,1,2,2,3,3,5,6],
            [6,6,6,6,6,6,6,6,6,6,0,1,1,2,2,3,3,5,6],
            [6,6,6,6,6,6,6,6,6,6,0,1,1,2,2,3,3,5,6],
            [6,6,6,6,6,6,6,6,6,6,0,1,1,2,2,3,3,4,4],
            [6,6,6,6,6,6,6,6,6,6,2,1,1,2,2,3,3,6,5],
            [6,6,6,6,6,6,6,6,6,6,2,1,1,2,2,3,3,6,5],
            [6,6,6,6,6,6,6,6,6,6,0,1,1,2,2,3,3,4,4],
            [6,6,6,6,6,6,6,6,6,6,0,1,1,2,2,3,3,4,4],
            [6,6,6,6,6,6,6,6,6,6,0,1,1,2,2,3,3,4,4],
            [6,6,6,6,6,6,6,6,6,6,0,1,1,2,2,3,3,4,4],
            [6,6,6,6,6,6,6,6,6,6,0,1,1,2,2,3,3,4,4],
            [6,6,6,6,6,6,6,6,6,6,0,1,1,2,2,3,3,4,4],
            [6,6,6,6,6,6,6,6,6,6,2,1,1,2,2,3,3,6,5],
            [6,6,6,6,6,6,6,6,6,6,2,1,1,2,2,3,3,6,5],
            [6,6,6,6,6,6,6,6,6,6,0,1,1,2,2,3,3,4,4],
            [6,6,6,6,6,6,6,6,6,6,2,1,1,2,2,3,3,6,5],
            [6,6,6,6,6,6,6,6,6,6,0,1,1,2,2,3,3,4,4],
            [6,6,6,6,6,6,6,6,6,6,3,1,1,2,2,3,3,6,5],
        ])
     },
     (20,19,8),
    ],
    [f"{DATDIR}/entrypoint_tests/preprocessing/argstrings/argstring6a.txt",
     {"A":0,"B":1,"C":2,"D":3,"-":4},
     {
        "retained_sequences": np.arange(20),
        "retained_positions": np.arange(5), 
        "sequence_weights": np.array([
            0.2, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.2, 0.2, 0.2, 
            0.2, 0.2, 0.2, 0.2, 0.2, 0.5, 1/3, 1/3, 1/3
        ]),
        "msa": np.array([
            [0, 1, 2, 3, 3],
            [1, 1, 2, 3, 3],
            [2, 1, 2, 3, 3],
            [3, 1, 2, 3, 3],
            [0, 3, 3, 3, 3],
            [1, 3, 3, 3, 3],
            [2, 3, 3, 3, 3],
            [3, 3, 3, 3, 3],
            [0, 2, 0, 3, 3],
            [0, 2, 1, 3, 3],
            [1, 2, 0, 3, 3],
            [1, 2, 1, 3, 3],
            [2, 2, 0, 3, 3],
            [2, 2, 1, 3, 3],
            [3, 2, 0, 3, 3],
            [3, 2, 1, 3, 3],
            [0, 1, 2, 0, 1],
            [0, 1, 2, 0, 2],
            [0, 1, 2, 1, 2],
            [0, 1, 2, 1, 3],
        ])
     },
     (20,5,4),
    ],
    [f"{DATDIR}/entrypoint_tests/preprocessing/argstrings/argstring7a.txt",
     {"A":0,"C":1,"D":2,"E":3,"-":4},
     {
        "retained_sequences": np.arange(20),
        "retained_positions": np.arange(5), 
        "sequence_weights": np.array([
            0.2, 0.25, 0.25, 0.25, 1, 1/3, 1/3, 1/3, 1, 0.5, 
            1, 1, 0.5, 1, 1/3, 1/3, 1, 0.5, 1/3, 1/3
        ]),
        "msa": np.array([
            [0, 3, 1, 2, 2],
            [3, 3, 1, 2, 2],
            [1, 3, 1, 2, 2],
            [2, 3, 1, 2, 2],
            [0, 2, 2, 4, 4],
            [3, 2, 2, 2, 2],
            [1, 2, 2, 2, 2],
            [2, 2, 2, 2, 2],
            [0, 1, 4, 4, 2],
            [0, 1, 3, 2, 2],
            [3, 1, 0, 2, 3],
            [3, 4, 3, 2, 2],
            [1, 1, 0, 2, 2],
            [1, 4, 3, 2, 4],
            [2, 1, 0, 2, 2],
            [2, 1, 3, 2, 2],
            [4, 3, 1, 0, 3],
            [0, 3, 1, 0, 1],
            [0, 3, 1, 3, 1],
            [0, 3, 1, 3, 2],
        ])
     },
     (20,5,4),
    ],
    [f"{DATDIR}/entrypoint_tests/preprocessing/argstrings/argstring8a.txt",
     {"A":0,"B":1,"C":2,"D":3,"-":4},
     {
        "retained_sequences": np.array(
            [0,1,2,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], dtype=int
        ),
        "retained_positions": np.arange(5), 
        "sequence_weights": np.array([
            1/3, 0.5, 0.5, 1/3, 0.5, 0.5, 0.5, 1/3, 0.5, 0.5, 1, 
            1/3, 0.5, 0.5, 0.5, 1, 1, 1/3, 1/3, 1
        ]),
        "msa": np.array([
            [0, 1, 2, 3, 3],
            [1, 0, 2, 3, 3],
            [2, 3, 4, 3, 3],
            [0, 1, 3, 2, 2],
            [1, 0, 3, 2, 2],
            [2, 3, 3, 2, 2],
            [3, 2, 3, 2, 2],
            [0, 1, 1, 3, 3],
            [1, 0, 1, 3, 3],
            [2, 3, 1, 3, 3],
            [3, 2, 1, 3, 3],
            [0, 1, 0, 2, 2],
            [1, 0, 0, 2, 2],
            [2, 3, 0, 2, 2],
            [3, 2, 0, 2, 2],
            [0, 1, 0, 0, 0],
            [0, 1, 1, 1, 1],
            [0, 1, 2, 2, 2],
            [0, 1, 3, 3, 3],
            [3, 0, 2, 3, 1],
        ])
     },
     (20,5,4),
    ],
])
def test_main(
        argstring_fpath, sym2int_exp, results_exp, msa_binary3d_shape_exp
):
    argstring = get_args(argstring_fpath)
    args = parse_args(argstring)
    args.msa_fpath = f"{DATDIR}/{args.msa_fpath}"
    args.outdir = f"{TMPDIR}/{args.outdir}"
    # Run the entrypoint
    main(args)
    # Open the preprocessing results archive
    fpath = os.path.join(args.outdir, "preprocessing_results.npz")
    if not os.path.exists(fpath):
        raise FileNotFoundError(fpath)
    results = np.load(fpath)
    errors = []
    keys_to_check = [
        "msa",
        "retained_sequences",
        "retained_positions",
        "retained_sequence_ids",
        "sequence_weights",
        "fi0_pretruncation",
    ]
    for k in keys_to_check:
        if k not in results:
            msg = f"key {k} not found in results!"
            errors.append(msg)
    assert not errors, "Errors occurred:\n{}".format("\n".join(errors))
    # Check values
    errors = []
    for k, v_exp in results_exp.items():
        v = results[k]
        if not np.all(np.equal(v, v_exp)):
            msg = f"Mismatch in result {k}"
            msg += f"Expected:\n{v_exp}\nGot:\n{v}"
            errors.append(msg)
    assert not errors, "Errors occurred:\n{}".format("\n".join(errors))

    # Check sym2int
    sym2int_fpath = os.path.join(args.outdir, "sym2int.json")
    with open(sym2int_fpath, "rb") as f:
        sym2int = json.load(f)
    errors = []
    if len(sym2int) != len(sym2int_exp):
        msg = f"Mismatch in sym2int. Expected {sym2int_exp}. Got {sym2int}."
        errors.append(msg)
        for k in sym2int_exp:
            if k not in sym2int:
                errors.append(f"Missing key {k} in sym2int")
            elif sym2int[k] != sym2int_exp[k]:
                msg = "Mismatch for key {}. Expected {}. Got {}.".format(
                    k, sym2int_exp[k], sym2int[k]
                )
                errors.append(msg)
    assert not errors, "Errors occurred:\n{}".format("\n".join(errors))
    
    # Check binary msa
    retained_sequences = results["retained_sequences"]
    retained_positions = results["retained_positions"]
    msa_fpath = os.path.join(args.outdir, "msa_binary2d_sp.npz")
    msa_binary2d_sp = sp.load_npz(msa_fpath)
    msa_binary3d = msa_binary2d_sp.toarray().reshape(
        [len(retained_sequences), len(retained_positions), -1]
    )
    assert msa_binary3d.shape == msa_binary3d_shape_exp

    # Remove the output directory
    remove_dir(args.outdir)
