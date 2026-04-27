"""Project PDB structure(s) onto an existing SCA result.

Three input modes (mutually exclusive; pick exactly one):

- ``-s <pdb_path>``: a single PDB file. Use ``--chain`` to select the
  chain (defaults to the first chain).
- ``--seq_map <tsv>``: a TSV from MSA sequence ID to PDB file (with
  optional chain), one row per sequence to project. Format documented
  in ``mysca.structure.mapping.SequencePdbMap.from_tsv``.
- ``--uniprot_ids <UID> [<UID> ...]``: one or more UniProt accessions.
  Each is resolved to its top-ranked PDB via EBI's SIFTS
  ``best_structures`` endpoint (cached under ``--cache_dir``); the
  resolved file must already exist in ``--pdb_dir`` UNLESS ``--fetch``
  is set, in which case missing PDBs are downloaded into ``--pdb_dir``
  on demand. ``--pdb_dir`` defaults to ``./.pdb_cache/`` when fetching.

-------------------------------------------------------------------------------
COMMAND LINE ARGUMENTS:

    -s --structure : Path to a single PDB file.
    --chain : Chain ID within --structure. Optional; defaults to the
        first chain.
    --seq_map : Path to a TSV mapping MSA seq IDs to PDB paths.
    --uniprot_ids : one or more UniProt accessions (space-separated).
    --pdb_dir : directory of pre-downloaded PDB files (required with
        --uniprot_ids unless --fetch is set, in which case it defaults
        to ``./.pdb_cache/``).
    --cache_dir : SIFTS JSON cache directory (default
        ``./.sifts_cache``; only consulted with --uniprot_ids).
    --fetch : opt-in download of missing PDB files from --pdb_source
        into --pdb_dir on demand. Off by default. Only valid with
        --uniprot_ids.
    --pdb_source : {"rcsb","pdbe"} source for --fetch. Default rcsb.
    --pdb_form : {"asym","assembly1","assembly2"} structure form for
        --fetch. Default asym (asymmetric unit; what most users get
        when hand-downloading from RCSB).
    --force_refetch : bypass the on-disk PDB cache and re-download.
        Off by default.
    --seq_id : Header to use when projecting --structure's sequence
        (triggers in-sample short-circuit if this ID is already in the
        reference MSA). Ignored in --seq_map and --uniprot_ids modes
        (seq IDs come from the map/UniProt list itself).
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

    sca-structure --uniprot_ids P06241 P12931 \\
        --pdb_dir ./pdbs \\
        --cache_dir ./.sifts_cache \\
        --preprocessing preprocess_out \\
        --scacore scacore_out \\
        -o structure_out

    # Auto-fetch missing PDBs into the default cache dir:
    sca-structure --uniprot_ids P06241 P12931 \\
        --fetch \\
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
from mysca.structure.fetcher import DEFAULT_PDB_CACHE_DIR


STRUCTURE_LOG_FNAME = "structure.log"
STRUCTURE_RESULTS_FNAME = "structure_projection.json"
STRUCTURE_ARGS_FNAME = "structure_args.json"
PER_STRUCTURE_DIRNAME = "per_structure"

logger = logging.getLogger("mysca.run_structure")


def parse_args(args):
    parser = argparse.ArgumentParser(
        description=(
            "Project PDB structure(s) onto an existing SCA result. "
            "Pick EXACTLY ONE of --structure, --seq_map, or --uniprot_ids."
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
        "--uniprot_ids", type=str, nargs="+", default=None, metavar="UID",
        help="UniProt accessions to resolve via SIFTS best_structures. "
        "For each ID, the top-ranked PDB is resolved and looked up "
        "inside --pdb_dir. Requires --pdb_dir; honors --cache_dir.",
    )
    parser.add_argument(
        "--pdb_dir", type=str, default=None, metavar="DIR",
        help="Directory containing pre-downloaded PDB files. Required "
        "when --uniprot_ids is used unless --fetch is also set, in "
        "which case it defaults to ./.pdb_cache/ and missing PDBs "
        "are downloaded into it.",
    )
    parser.add_argument(
        "--cache_dir", type=str, default=None, metavar="DIR",
        help="Local directory to cache SIFTS JSON responses. Default: "
        "./.sifts_cache under the current working directory. Only "
        "consulted when --uniprot_ids is used.",
    )
    parser.add_argument(
        "--fetch", action="store_true",
        help="Opt-in: download missing PDB files from --pdb_source "
        "into --pdb_dir on demand. Off by default. Only valid with "
        "--uniprot_ids.",
    )
    parser.add_argument(
        "--pdb_source", type=str, default="rcsb",
        choices=["rcsb", "pdbe"],
        help="Source for --fetch. Default: rcsb. Only valid with "
        "--uniprot_ids.",
    )
    parser.add_argument(
        "--pdb_form", type=str, default="asym",
        choices=["asym", "assembly1", "assembly2"],
        help="Which form to fetch. Default: asym (asymmetric unit, "
        "single .pdb). assembly1/assembly2 fetch the .pdb1/.pdb2 "
        "biological assembly. Only valid with --uniprot_ids.",
    )
    parser.add_argument(
        "--force_refetch", action="store_true",
        help="Bypass the on-disk PDB cache and re-download. Off by "
        "default. Only valid with --uniprot_ids + --fetch.",
    )
    parser.add_argument(
        "--seq_id", type=str, default=None,
        help="Header to use for --structure's sequence when running "
        "the project step. When this matches an ID in the reference "
        "MSA, the in-sample short-circuit kicks in. Ignored when "
        "--seq_map or --uniprot_ids is used.",
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
    chosen = sum(
        x is not None and x != []
        for x in (parsed.structure, parsed.seq_map, parsed.uniprot_ids)
    )
    if chosen != 1:
        parser.error(
            "Exactly one of --structure, --seq_map, or --uniprot_ids "
            "is required."
        )
    if parsed.uniprot_ids is not None:
        if parsed.pdb_dir is None and not parsed.fetch:
            parser.error(
                "--uniprot_ids requires --pdb_dir (the directory of "
                "pre-downloaded PDB files). SIFTS resolves UniProt IDs "
                "to PDB entries but does not fetch the structure files. "
                "Pass --fetch to download missing PDBs into --pdb_dir "
                f"(defaults to ./{DEFAULT_PDB_CACHE_DIR}/)."
            )
        if parsed.pdb_dir is None:  # implies --fetch
            parsed.pdb_dir = DEFAULT_PDB_CACHE_DIR
    fetch_flags_used = (
        parsed.fetch
        or parsed.pdb_source != "rcsb"
        or parsed.pdb_form != "asym"
        or parsed.force_refetch
    )
    if fetch_flags_used and parsed.uniprot_ids is None:
        parser.error(
            "--fetch / --pdb_source / --pdb_form / --force_refetch "
            "only apply with --uniprot_ids."
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
            seq_proj.ic_residues,
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
        with open(os.path.join(outdir, STRUCTURE_RESULTS_FNAME), "w") as f:
            json.dump([p.to_dict() for p in projections], f, indent=2)
        logger.info(
            "sca-structure done. %d structure(s) projected. Output at %s",
            len(projections), outdir,
        )
        return

    if args.uniprot_ids is not None:
        logger.info(
            "Resolving %d UniProt accession(s) via SIFTS (cache_dir=%s, "
            "fetch=%s).",
            len(args.uniprot_ids),
            args.cache_dir or ".sifts_cache",
            args.fetch,
        )
        seq_map = SequencePdbMap.from_sifts_for_uniprot_ids(
            args.uniprot_ids,
            pdb_dir=args.pdb_dir,
            cache_dir=args.cache_dir,
            fetch=args.fetch,
            pdb_source=args.pdb_source,
            pdb_form=args.pdb_form,
            force_refetch=args.force_refetch,
        )
        logger.info(
            "SIFTS resolved %d/%d UniProt IDs into usable PDB entries.",
            len(seq_map), len(args.uniprot_ids),
        )
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
