"""Project PDB structure(s) onto an existing SCA result.

Two input modes (mutually exclusive):

- ``-s <pdb_path>``: a single PDB file. Use ``--chain`` to select the
  chain (defaults to the first chain).
- ``--seq_map <tsv>``: a TSV from MSA sequence ID to PDB file (with
  optional chain), one row per sequence to project. Format documented
  in ``mysca.structure.mapping.SequencePdbMap.from_tsv``.

-------------------------------------------------------------------------------
COMMAND LINE ARGUMENTS:

    -s --structure : Path to a single PDB file (mutually exclusive with
        --seq_map).
    --chain : Chain ID within --structure. Optional; defaults to the
        first chain.
    --seq_map : Path to a TSV mapping MSA seq IDs to PDB paths (mutually
        exclusive with --structure).
    --seq_id : Header to use when projecting --structure's sequence
        (triggers in-sample short-circuit if this ID is already in the
        reference MSA). Ignored when --seq_map is used (the TSV's keys
        are used instead).
    --preprocessing : sca-preprocess output directory.
    --scacore : sca-core output directory.
    -o --outdir : Output directory.
    --aligner : Out-of-sample alignment method (default 'mafft_add').
    --align_bin : Explicit path to the alignment binary.
    --align_threads : Threads for the alignment tool.

-------------------------------------------------------------------------------
OUTPUTS:

structure_projection.json
    List of per-structure dicts with the fields from
    ``PdbProjection.to_dict()``: structure_id, chain_id,
    sequence_projection (full raw-residue-coordinate projection), and
    ic_pdb_residues (per-IC list of PDB residue numbers).

per_structure/<structure_id>_ic_residues.tsv
    One row per (ic_index, raw_residue_idx, pdb_residue, processed_col,
    v_ica_loading) for readable inspection.

structure_args.json
    Mapping from CLI argument to value.

structure.log
    Run log.

-------------------------------------------------------------------------------
EXAMPLE USAGE:

    sca-structure -s my_protein.pdb --chain A \\
        --preprocessing preprocess_out \\
        --scacore scacore_out \\
        -o structure_out

    sca-structure --seq_map seq_to_pdb.tsv \\
        --preprocessing preprocess_out \\
        --scacore scacore_out \\
        -o structure_out

"""

import argparse
import json
import logging
import os
import sys

from mysca.logging_config import configure_logging
from mysca.project import ALIGNERS
from mysca.run_project import _safe_filename_component
from mysca.structure import (
    PDBStructure,
    SequencePdbMap,
    project_pdb,
)


STRUCTURE_LOG_FNAME = "structure.log"
STRUCTURE_RESULTS_FNAME = "structure_projection.json"
STRUCTURE_ARGS_FNAME = "structure_args.json"
PER_STRUCTURE_DIRNAME = "per_structure"

logger = logging.getLogger("mysca.run_structure")


def parse_args(args):
    parser = argparse.ArgumentParser(
        description=(
            "Project PDB structure(s) onto an existing SCA result. "
            "Pick EXACTLY ONE of --structure or --seq_map."
        ),
    )
    parser.add_argument(
        "-s", "--structure", type=str, default=None, metavar="PDB",
        help="Path to a single PDB file.",
    )
    parser.add_argument(
        "--chain", type=str, default=None,
        help="Chain ID within --structure. Default: first chain.",
    )
    parser.add_argument(
        "--seq_map", type=str, default=None, metavar="TSV",
        help="Path to a TSV of seq_id<TAB>pdb_path[<TAB>chain].",
    )
    parser.add_argument(
        "--seq_id", type=str, default=None,
        help="Header to use for --structure's sequence when running "
        "the project step. When this matches an ID in the reference "
        "MSA, the in-sample short-circuit kicks in. Ignored when "
        "--seq_map is used.",
    )
    parser.add_argument(
        "--preprocessing", type=str, required=True, metavar="DIR",
        help="sca-preprocess output directory.",
    )
    parser.add_argument(
        "--scacore", type=str, required=True, metavar="DIR",
        help="sca-core output directory.",
    )
    parser.add_argument(
        "-o", "--outdir", type=str, required=True,
        help="Output directory.",
    )
    parser.add_argument(
        "--aligner", type=str, default="mafft_add",
        choices=sorted(ALIGNERS),
        help="Out-of-sample alignment method.",
    )
    parser.add_argument(
        "--align_bin", type=str, default=None,
        help="Explicit path to the alignment binary.",
    )
    parser.add_argument(
        "--align_threads", type=int, default=1,
        help="Threads for the alignment tool.",
    )
    parser.add_argument("-v", "--verbosity", type=int, default=1,
                        help="Verbosity level (0=warnings only).")

    parsed = parser.parse_args(args)
    if bool(parsed.structure) == bool(parsed.seq_map):
        parser.error(
            "Exactly one of --structure or --seq_map is required."
        )
    return parsed


def _write_per_structure_tsv(outdir, proj):
    per_dir = os.path.join(outdir, PER_STRUCTURE_DIRNAME)
    os.makedirs(per_dir, exist_ok=True)
    path = os.path.join(
        per_dir,
        f"{_safe_filename_component(proj.structure_id)}_ic_residues.tsv",
    )
    seq_proj = proj.sequence_projection
    with open(path, "w") as f:
        f.write(
            "ic_index\traw_residue_idx\tpdb_residue\t"
            "processed_col\tv_ica_loading\n"
        )
        for ic_idx, (members, loadings, cols, pdb_resids) in enumerate(zip(
            seq_proj.ic_memberships,
            seq_proj.ic_loadings,
            seq_proj.ic_processed_cols,
            proj.ic_pdb_residues,
        )):
            for resi, loading, col, pdb_resi in zip(
                members, loadings, cols, pdb_resids
            ):
                f.write(
                    f"{ic_idx}\t{int(resi)}\t{int(pdb_resi)}\t"
                    f"{int(col)}\t{float(loading)}\n"
                )


def main(args):
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    configure_logging(
        verbosity=args.verbosity,
        logfile=os.path.join(outdir, STRUCTURE_LOG_FNAME),
    )

    with open(os.path.join(outdir, STRUCTURE_ARGS_FNAME), "w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

    aligner_kwargs = {
        "bin_path": args.align_bin,
        "threads": args.align_threads,
    }

    projections = []
    if args.structure:
        pdb = PDBStructure.from_file(args.structure, chain=args.chain)
        logger.info(
            "Projecting PDB %s chain %s (%d residues).",
            pdb.structure_id, pdb.chain_id, len(pdb),
        )
        proj = project_pdb(
            pdb,
            sca_result_dir=args.scacore,
            preproc_result_dir=args.preprocessing,
            seq_id=args.seq_id,
            aligner=args.aligner,
            aligner_kwargs=aligner_kwargs,
            workdir=os.path.join(outdir, "_align_workdir"),
        )
        projections.append(proj)
        _write_per_structure_tsv(outdir, proj)
    else:
        seq_map = SequencePdbMap.from_tsv(args.seq_map)
        logger.info(
            "Projecting %d PDB structure(s) from %s.",
            len(seq_map), args.seq_map,
        )
        for seq_id, entry in seq_map.items():
            pdb = PDBStructure.from_file(
                entry.pdb_path, chain=entry.chain, structure_id=seq_id,
            )
            proj = project_pdb(
                pdb,
                sca_result_dir=args.scacore,
                preproc_result_dir=args.preprocessing,
                seq_id=seq_id,
                aligner=args.aligner,
                aligner_kwargs=aligner_kwargs,
                workdir=os.path.join(outdir, "_align_workdir"),
            )
            projections.append(proj)
            _write_per_structure_tsv(outdir, proj)

    with open(os.path.join(outdir, STRUCTURE_RESULTS_FNAME), "w") as f:
        json.dump(
            [p.to_dict() for p in projections], f, indent=2,
        )

    logger.info(
        "sca-structure done. %d structure(s) projected. Output at %s",
        len(projections), outdir,
    )


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
