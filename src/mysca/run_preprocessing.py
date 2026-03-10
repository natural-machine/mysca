"""SCA preprocessing pipeline

Runs the preprocessing steps of SCA, given an input MSA in fasta format.
Creates and populates an output directory with the preprocessed results.
See references:
    [1] SI to Rivoire et al., 2016

-------------------------------------------------------------------------------
COMMAND LINE ARGUMENTS:

Command line arguments include the SCA parameters as detailed in [1]. These are
specified as
    --gap_truncation_thresh : 
    --sequence_gap_thresh : 
    --reference : 
    --reference_similarity_thresh : 
    --sequence_similarity_thresh : 
    --position_gap_thresh : 

-------------------------------------------------------------------------------
OUTPUTS:

The core results are stored in a numpy archive file `preprocessed_results.npz`.
This file contains the following keys, mapped to numpy arrays:

preprocessed_results.npz
    msa : preprocessed MSA. Integer 2d array of shape (M x L)
    retained_sequences : Indices of retained sequences in the original MSA.
    retained_positions : Indices of retained positions in the original MSA.
    retained_sequence_ids : IDs of retained sequences in the original MSA.
    sequence_weights : Sampling weights for the retained sequences.
    fi0_pretruncation : Gap frequency per position, prior to truncation.

msa_binary2d_sp.npz:
    MSA in a 2-dimensional sparse binary format, of shape (M x DL) with D=20 
    for 20 amino acids. The MSA is saved in a compressed numpy format.
    
sym2int.json
    Mapping from sequence characters (i.e. amino acid symbols) to their integer 
    representation.

preprocessing_args.json
    Mapping from command line SCA parameters to their values.

msa_orig.fasta-aln
    MSA in fasta format, before truncation steps and preprocessing.

-------------------------------------------------------------------------------
EXAMPLE USAGE:

sca-preprocess -i </path/to/msa.fasta> -o </path/to/outdir> \
    --gap_truncation_thresh 0.4 \
    --sequence_gap_thresh 0.2 \
    --reference <reference-fasta-id> \
    --reference_similarity_thresh 0.2 \
    --sequence_similarity_thresh 0.8 \
    --position_gap_thresh 0.2

"""

import argparse
import os, sys
import numpy as np
import tqdm as tqdm
import json
from scipy import sparse
from Bio import AlignIO

from mysca.io import load_msa
from mysca.mappings import SymMap
from mysca.constants import AA_STD20
from mysca.preprocess import preprocess_msa

# Quick reference for saved files
OUTPUT_RESULTS_FNAME = "preprocessing_results.npz"
OUTPUT_SYMMAP_FNAME = "sym2int.json"
OUTPUT_ARGS_FNAME = "preprocessing_args.json"
OUTPUT_MSAORIG_FNAME = "msa_orig.fasta-aln"


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--msa_fpath", type=str, required=True,
                        help="Filepath of input MSA in fasta format.")
    parser.add_argument("-o", "--outdir", type=str, required=True, 
                        help="Output directory.")
    parser.add_argument("-v", "--verbosity", type=int, default=1)
    parser.add_argument("--pbar", action="store_true")
    parser.add_argument("--plot", action="store_true")
    
    parser.add_argument("--syms", type=str, default="default")
    parser.add_argument("--gapsym", type=str, default="-")
    
    sca_params = parser.add_argument_group("SCA parameters")
    sca_params.add_argument(
        "--gap_truncation_thresh", type=float, default=0.4,
        help="SCA parameter gap_truncation_thresh"
    )
    sca_params.add_argument(
        "--sequence_gap_thresh", type=float, default=0.2,
        help="SCA parameter sequence_gap_thresh γ_{seq}"
    )
    sca_params.add_argument(
        "--reference", type=str, default=None, 
        help="SCA optional reference entry in MSA"
    )
    sca_params.add_argument(
        "--reference_similarity_thresh", type=float, default=0.2,
        help="SCA parameter reference_similarity_thresh Δ"
    )
    sca_params.add_argument(
        "--sequence_similarity_thresh", type=float, default=0.8,
        help="SCA parameter sequence_similarity_thresh δ"
    )
    sca_params.add_argument(
        "--position_gap_thresh", type=float, default=0.2,
        help="SCA parameter position_gap_thresh γ_{pos}"
    )
    return parser.parse_args(args)


def main(args):

    # Process command line args
    msa_fpath = args.msa_fpath
    outdir = args.outdir
    verbosity = args.verbosity
    pbar = args.pbar
    do_plot = args.plot
    
    syms = args.syms
    gapsym = args.gapsym

    gap_truncation_thresh = args.gap_truncation_thresh
    sequence_gap_thresh = args.sequence_gap_thresh
    reference_id = args.reference
    reference_similarity_thresh = args.reference_similarity_thresh
    sequence_similarity_thresh = args.sequence_similarity_thresh
    position_gap_thresh = args.position_gap_thresh
    
    # Housekeeping
    if reference_id is None or reference_id.lower() == "none":
        if verbosity:
            print("No reference entry specified.")
        reference_id = None
    
    os.makedirs(outdir, exist_ok=True)

    if do_plot:
        imgdir = os.path.join(outdir, "images")
        os.makedirs(imgdir, exist_ok=True)

    if syms.lower() in ["default"]:
        sym_map = SymMap(aa_syms=AA_STD20, gapsym=gapsym)
    elif syms.lower() in ["none"]:
        sym_map = None
    else:
        sym_map = SymMap(aa_syms=syms, gapsym=gapsym)

    # Load MSA
    if verbosity:
        print(f"Loading MSA from: {msa_fpath}")
    msa_obj_orig, msa_orig, seqids_orig, sym_map = load_msa(
        msa_fpath, format="fasta", 
        mapping=sym_map,
        verbosity=1
    )
    num_seq_orig, num_pos_orig = msa_orig.shape
    
    if verbosity:
        print(f"Loaded MSA. shape: {msa_orig.shape} (sequences x positions)")
        print(f"Symbols: {sym_map.aa_list}")

    # Run preprocessing script
    msa, preprocessing_results = preprocess_msa(
        msa_orig, seqids_orig,
        mapping=sym_map,
        gap_truncation_thresh=gap_truncation_thresh,
        sequence_gap_thresh=sequence_gap_thresh,
        reference_id=reference_id,
        reference_similarity_thresh=reference_similarity_thresh,
        sequence_similarity_thresh=sequence_similarity_thresh,
        position_gap_thresh=position_gap_thresh,
        use_pbar=pbar,
        verbosity=verbosity,
    )

    msa_binary3d = preprocessing_results["msa_binary3d"]
    retained_sequences = preprocessing_results["retained_sequences"]
    retained_positions = preprocessing_results["retained_positions"]
    seqids = preprocessing_results["retained_sequence_ids"]
    weights = preprocessing_results["sequence_weights"]
    fi0_pretrunc = preprocessing_results["fi0_pretruncation"]
    preprocessing_args = preprocessing_results["args"]

    # Save a single, compressed version of the numpy result files
    np.savez(
        os.path.join(outdir, OUTPUT_RESULTS_FNAME),
        msa=msa,
        retained_sequences=retained_sequences,
        retained_positions=retained_positions,
        retained_sequence_ids=seqids,
        sequence_weights=weights,
        fi0_pretruncation=fi0_pretrunc,
    )

    # Save the command line arguments for reproducibility
    with open(os.path.join(outdir, OUTPUT_ARGS_FNAME), "w") as f:
        json.dump(preprocessing_args, f)

    # Save symbolic mapping (amino acid to integer)
    with open(os.path.join(outdir, OUTPUT_SYMMAP_FNAME), "w") as f:
        json.dump(sym_map.sym2int, f)

    AlignIO.write(
        msa_obj_orig, 
        os.path.join(outdir, OUTPUT_MSAORIG_FNAME), 
        format="fasta"
    )

    # Save 3d MSA (M x L x 20) in sparse 2d format (M x 20L)
    sparse.save_npz(
        f"{outdir}/msa_binary2d_sp.npz", 
        sparse.csr_matrix(msa_binary3d.reshape([msa_binary3d.shape[0], -1]))
    )

    if verbosity:
        print(f"Output saved to {outdir}")
        print(f"Preprocessing complete!")

    # # Save processed results in the output directory as individual files
    # np.save(f"{outdir}/retained_sequences.npy", retained_sequences)
    # np.save(f"{outdir}/retained_positions.npy", retained_positions)
    # np.save(f"{outdir}/retained_sequence_ids.npy", seqids)
    # np.save(f"{outdir}/fi0_pretrunc.npy", fi0_pretrunc)
    # np.save(f"{outdir}/sequence_weights.npy", weights)
    # np.save(f"{outdir}/msa.npy", msa)

    # Save miscellaneous info
    # np.savetxt(f"{outdir}/position_gap_thresh.txt", [position_gap_thresh])
    # np.savetxt(f"{outdir}/npos_original.txt", [num_pos_orig], fmt="%d")
    
    
if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
