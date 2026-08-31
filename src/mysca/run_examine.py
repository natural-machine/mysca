"""Characterize the independent components (ICs) of a finished SCA run.

Given an ``sca-core`` + ``sca-preprocess`` result for one protein family,
``sca-examine`` characterizes each significant IC along three axes and then
groups the ICs into sectors, following Andrews, "How to characterize
independent components after performing SCA":

  1. Conservation  - Pearson correlation of the IC's position loadings with
     per-position conservation (relative entropy). ``r > --cons_r`` marks the
     IC conservation-associated ("core") and protects it from filtering.
  2. Phylogeny     - Pagel's lambda phylogenetic-signal test on the sequence
     projections (Uᵖ) over a guide tree, calibrated against a random-projection
     null (``--null_projections``); plus the best clade split by Mann-Whitney U
     + Cliff's delta. ``lrt_z > --phylo_z`` -> phylogeny pseudo-sector;
     ``|delta| > --subclade_delta`` -> subclade pseudo-sector.
  3. Structure     - PAE-masked minimum heavy-atom distances on a supplied or
     AlphaFold-fetched structure: per-IC contiguity vs a size-matched null, and
     cross-IC neighbour distances. ICs that sit closer to each other than to
     themselves are merged into a sector; a one-directional reach makes the
     reaching IC a co-sector.

Each leg is optional and degrades gracefully: with no structure source only the
sequence-based legs run; with ``--no_phylogeny`` only conservation (+ structure)
run. The SCA result folder is never modified.

-------------------------------------------------------------------------------
COMMAND LINE ARGUMENTS:

    --scacore : sca-core output directory (SCAResults.load target;
        contains sca_results/, ic_positions/, scarun_results.npz).
    --preprocessing : sca-preprocess output directory (contains
        preprocessing_results.npz, sym2int.json, msa_orig.fasta-aln).
    -o --outdir : Output directory (created; all outputs go here).
    --family : Label used for output feature names. Default: the
        basename of the directory containing --scacore (e.g. PF00001).

  Phylogeny leg:
    --no_phylogeny : Skip the Pagel's-lambda + clade-split analysis.
    --tree : Path to a user-supplied Newick guide tree. Leaves are
        joined to sequences by UniProt accession. Skips FAMSA.
    --projections : Path to a user-supplied per-sequence projection TSV
        (columns seq_id, aligned_sequence, Up_0..Up_k). Bypasses
        subsampling/projection entirely; requires --tree.
    --n_subsample : Sequences to subsample for tree building / phylogeny
        (default 1000; the processed MSA is already redundancy-reduced).
    --gt_method : FAMSA guide-tree method {sl, upgma, nj} (default nj).
    --famsa_bin : Path to the FAMSA binary (auto-detected if omitted).
    --threads : FAMSA threads (0 = half the logical cores).
    --min_side_size : Min valid leaves on each side of a clade split
        (default 10).
    --null_projections : Random alignment projections used to calibrate
        each IC's Pagel lambda / LRT against generic phylogenetic
        autocorrelation (default 200; 0 disables, and lrt_z is then NaN).
    --no_transform : Skip the van der Waerden transform for Pagel's lambda.

  Structure leg (pick at most one source):
    -s --structure : Path to a user-supplied PDB file. Use --ref_seq_id
        to name the family sequence it corresponds to (enables the
        in-sample mapping short-circuit); otherwise the structure is
        aligned out-of-sample (needs --aligner's binary).
    --pae : Optional predicted-aligned-error JSON for --structure. When
        absent, no PAE masking is applied (all residue pairs are used).
    --chain : Chain ID within --structure (default: first chain).
    --fetch_alphafold : Download an AlphaFold model + PAE for a family
        member (random, or --ref_seq_id) instead of supplying --structure.
    --ref_seq_id : Family sequence id. With --structure, the sequence the
        structure represents; with --fetch_alphafold, the member to fetch
        (default: a random family sequence).
    --max_tries : AlphaFold fetch attempts before giving up (default 5).
    --pae_max_err : PAE mask threshold in Angstrom; residue pairs with
        PAE >= this are dropped (default 5.0, per the reference document).
    --cross_thresh : Mean cross-neighbour distance (A) below which an IC
        pair is reported as touching (default 5.0).
    --n_null : Resamples per IC for the size-matched contiguity null
        (default 1000).
    --aligner : Out-of-sample alignment method for mapping --structure
        when --ref_seq_id is not in-sample (default mafft_add).
    --align_bin : Explicit path to the alignment binary.
    --align_threads : Threads for the alignment tool.

  Sector-grouping thresholds:
    --no_sectors : Skip grouping ICs into sectors.
    --cons_r : conservation correlation above which an IC is kept as
        core/conservation-associated (default 0.5).
    --subclade_delta : abs Cliff's delta above which an IC is a subclade
        pseudo-sector (default 0.95).
    --phylo_z : Pagel LRT z-score above which an IC is a phylogeny
        pseudo-sector (default 7.0).

    --seed : RNG seed (default 0).
    -v --verbosity : Verbosity level (0 = warnings only).

-------------------------------------------------------------------------------
OUTPUTS (under --outdir):

per_ic_characterization.tsv
    One row per significant IC, merging the conservation table, the
    Pagel's-lambda + clade-split statistics, and the within-IC structural
    neighbour distance.
sector_features.tsv (unless --no_sectors)
    One row per feature: sector / co-sector / pseudo-sector, with member
    components and labels.
ic_cross_neighbor_distance.tsv (structure leg only)
    K x K mean cross-neighbour distance matrix.
subsample_projections.tsv (phylogeny leg, normal path)
    label / seq_id / Up_* for the subsample used to build the tree.
guide_tree.nwk (phylogeny leg, FAMSA path)
    The guide tree FAMSA built over the subsample.
structure_source.json (--fetch_alphafold only)
    Metadata for the downloaded AlphaFold model + PAE.
examine_args.json
    Mapping from CLI argument to value.
examine.log
    Run log.

-------------------------------------------------------------------------------
EXAMPLE USAGE:

    # Sequence-only characterization (conservation + phylogeny):
    sca-examine --scacore PF00001/scacore \\
        --preprocessing PF00001/preprocessing -o examine_out

    # Add structure via automated AlphaFold lookup:
    sca-examine --scacore PF00001/scacore \\
        --preprocessing PF00001/preprocessing \\
        --fetch_alphafold -o examine_out

    # Use your own structure + PAE, naming the family sequence it represents:
    sca-examine --scacore PF00001/scacore \\
        --preprocessing PF00001/preprocessing \\
        --structure my_model.pdb --pae my_pae.json \\
        --ref_seq_id 'MYSEQ_HUMAN/1-100' -o examine_out

    # Bring your own guide tree instead of building one with FAMSA:
    sca-examine --scacore PF00001/scacore \\
        --preprocessing PF00001/preprocessing \\
        --tree my_tree.nwk -o examine_out

"""

import argparse
import json
import logging
import os
import sys

import numpy as np
import pandas as pd

from mysca.logging_config import configure_logging
from mysca.examine import (
    ExamineResults,
    cross_ic_distances,
    fetch_alphafold_structure,
    group_ics,
    ic_residues_on_structure,
    load_pae_json,
    min_atom_distance_matrix,
    n_significant_ics,
    parse_newick,
    per_ic_contiguity,
    per_ic_table,
    project_subsample,
    residue_coords,
    run_pagel,
    run_splits,
    subsample_indices,
    touching_pairs,
)
from mysca.examine.guide_tree import (
    GT_METHODS,
    build_guide_tree,
    write_subsample_fasta,
)
from mysca.examine.newick import extract_accession_from_seq_id
from mysca.examine.structure import align_pae_to_residues
from mysca.examine.examine import int2char_from_sym2int
from mysca.project import ALIGNERS

EXAMINE_LOG_FNAME = "examine.log"
EXAMINE_ARGS_FNAME = "examine_args.json"

logger = logging.getLogger("mysca.run_examine")


def parse_args(args):
    p = argparse.ArgumentParser(
        description=(
            "Characterize the independent components of an SCA run "
            "(conservation, phylogeny, structure) and group them into sectors."
        ),
    )
    p.add_argument("--scacore", required=True, metavar="DIR",
                   help="sca-core output directory.")
    p.add_argument("--preprocessing", required=True, metavar="DIR",
                   help="sca-preprocess output directory.")
    p.add_argument("-o", "--outdir", required=True, help="Output directory.")
    p.add_argument("--family", default=None,
                   help="Label for output feature names "
                        "(default: basename of --scacore's parent dir).")

    # Phylogeny leg
    p.add_argument("--no_phylogeny", action="store_true",
                   help="Skip the Pagel's-lambda + clade-split analysis.")
    p.add_argument("--tree", default=None, metavar="NEWICK",
                   help="User-supplied Newick guide tree (skips FAMSA); "
                        "leaves joined to sequences by accession.")
    p.add_argument("--projections", default=None, metavar="TSV",
                   help="User-supplied per-sequence projection TSV "
                        "(seq_id, aligned_sequence, Up_*); requires --tree.")
    p.add_argument("--n_subsample", type=int, default=1000,
                   help="Sequences to subsample for the tree (default 1000).")
    p.add_argument("--gt_method", default="nj", choices=list(GT_METHODS),
                   help="FAMSA guide-tree method (default nj).")
    p.add_argument("--famsa_bin", default=None,
                   help="Path to the FAMSA binary (auto-detected if omitted).")
    p.add_argument("--threads", type=int, default=0,
                   help="FAMSA threads (0 = half the logical cores).")
    p.add_argument("--min_side_size", type=int, default=10,
                   help="Min valid leaves on each side of a clade split.")
    p.add_argument("--null_projections", type=int, default=200, metavar="N",
                   help="Random alignment projections to calibrate Pagel "
                        "lambda/LRT (default 200; 0 disables, lrt_z NaN).")
    p.add_argument("--no_transform", action="store_true",
                   help="Skip the van der Waerden transform for Pagel's lambda.")

    # Structure leg
    p.add_argument("-s", "--structure", default=None, metavar="PDB",
                   help="User-supplied PDB file.")
    p.add_argument("--pae", default=None, metavar="JSON",
                   help="Predicted-aligned-error JSON for --structure.")
    p.add_argument("--chain", default=None,
                   help="Chain ID within --structure (default: first chain).")
    p.add_argument("--fetch_alphafold", action="store_true",
                   help="Download an AlphaFold model + PAE for a family member.")
    p.add_argument("--ref_seq_id", default=None,
                   help="Family sequence the structure represents / to fetch.")
    p.add_argument("--max_tries", type=int, default=5,
                   help="AlphaFold fetch attempts before giving up.")
    p.add_argument("--pae_max_err", type=float, default=5.0,
                   help="PAE mask: drop residue pairs with PAE >= this (A).")
    p.add_argument("--cross_thresh", type=float, default=5.0,
                   help="Mean cross-neighbour distance (A) flagging touching ICs.")
    p.add_argument("--n_null", type=int, default=1000,
                   help="Resamples per IC for the contiguity null.")
    p.add_argument("--aligner", default="mafft_add", choices=sorted(ALIGNERS),
                   help="Out-of-sample alignment method for --structure mapping.")
    p.add_argument("--align_bin", default=None,
                   help="Explicit path to the alignment binary.")
    p.add_argument("--align_threads", type=int, default=1,
                   help="Threads for the alignment tool.")

    # Sector thresholds
    p.add_argument("--no_sectors", action="store_true",
                   help="Skip grouping ICs into sectors.")
    p.add_argument("--cons_r", type=float, default=0.5,
                   help="cons_corr_r above this -> conservation-associated.")
    p.add_argument("--subclade_delta", type=float, default=0.95,
                   help="abs Cliff's delta above this -> subclade pseudo-sector.")
    p.add_argument("--phylo_z", type=float, default=7.0,
                   help="Pagel LRT z above this -> phylogeny pseudo-sector.")

    p.add_argument("--seed", type=int, default=0, help="RNG seed.")
    p.add_argument("-v", "--verbosity", type=int, default=1,
                   help="Verbosity level (0 = warnings only).")

    parsed = p.parse_args(args)
    if parsed.structure is not None and parsed.fetch_alphafold:
        p.error("Pass at most one structure source: --structure OR "
                "--fetch_alphafold, not both.")
    if parsed.pae is not None and parsed.structure is None:
        p.error("--pae only applies with --structure.")
    if parsed.chain is not None and parsed.structure is None:
        p.error("--chain only applies with --structure.")
    if parsed.projections is not None and parsed.tree is None:
        p.error("--projections requires --tree (no sequences to build one).")
    if parsed.projections is not None and parsed.no_phylogeny:
        p.error("--projections is meaningless with --no_phylogeny.")
    return parsed


# ---------------------------------------------------------------------------
# Phylogeny inputs (subsample + project, or user projections; tree or FAMSA)
# ---------------------------------------------------------------------------

def _phylogeny_inputs(args, sca, prep, kstar, comp_names, axis_cols, out_dir, rng):
    """Return (acc_to_row, aligned_seqs, tree_text, subsample_df).

    Three shapes:
      * --projections + --tree : read everything from the TSV; tree from file.
      * --tree (no projections): subsample + project, key by accession, tree
        from file.
      * neither                : subsample + project, synthetic delimiter-free
        labels, build a FAMSA guide tree.
    """
    if args.projections is not None:
        df = pd.read_csv(args.projections, sep="\t")
        up_cols = sorted([c for c in df.columns if c.startswith("Up_")],
                         key=lambda c: int(c.split("_")[1]))
        if len(up_cols) < kstar:
            raise ValueError(
                f"--projections has {len(up_cols)} Up_* columns; need >= "
                f"{kstar} (the significant ICs).")
        if "seq_id" not in df.columns:
            raise ValueError("--projections is missing a 'seq_id' column.")
        keys = df["seq_id"].map(extract_accession_from_seq_id).tolist()
        up = df[up_cols[:kstar]].to_numpy(dtype=np.float64)
        acc_to_row = {k: up[i] for i, k in enumerate(keys)}
        aligned_seqs = None
        if "aligned_sequence" in df.columns:
            aligned_seqs = {k: s for k, s in zip(keys, df["aligned_sequence"])}
        tree_text = open(args.tree).read()
        sub_df = pd.DataFrame({"label": keys, "seq_id": df["seq_id"]})
        for k in range(kstar):
            sub_df[axis_cols[k]] = up[:, k]
        return acc_to_row, aligned_seqs, tree_text, sub_df

    # subsample + project from the processed MSA
    with open(os.path.join(args.preprocessing, "sym2int.json")) as fh:
        sym2int = json.load(fh)
    int2char = int2char_from_sym2int(sym2int)
    gap_idx = int(np.where(int2char == "-")[0][0])

    msa = np.asarray(prep.msa)
    seq_ids = np.asarray(prep.retained_sequence_ids).astype(str)
    nonempty = (msa != gap_idx).any(axis=1)
    keep = np.where(nonempty)[0]
    sel = keep[subsample_indices(len(keep), args.n_subsample, rng)]
    msa_sub = msa[sel]
    ids_sub = seq_ids[sel]
    logger.info("Subsampled %d sequences (of %d non-empty; seed=%d).",
                len(sel), len(keep), args.seed)

    up = project_subsample(sca, msa_sub)[:, :kstar]
    aligned_seqs_full = {}      # key -> gapped processed sequence

    if args.tree is not None:
        labels = [extract_accession_from_seq_id(s) for s in ids_sub]
        tree_text = open(args.tree).read()
    else:
        labels = [f"s{i}" for i in range(len(sel))]
        fasta_path = os.path.join(out_dir, "subsample.fasta")
        tree_path = os.path.join(out_dir, "guide_tree.nwk")
        write_subsample_fasta(fasta_path, labels, msa_sub, int2char)
        logger.info("Building FAMSA guide tree (%s)...", args.gt_method)
        tree_text = build_guide_tree(
            fasta_path, tree_path, famsa_bin=args.famsa_bin,
            gt_method=args.gt_method, threads=args.threads)

    acc_to_row = {lab: up[i] for i, lab in enumerate(labels)}
    for lab, row in zip(labels, msa_sub):
        aligned_seqs_full[lab] = "".join(int2char[v] for v in row)

    sub_df = pd.DataFrame({"label": labels, "seq_id": ids_sub})
    for k in range(kstar):
        sub_df[axis_cols[k]] = up[:, k]
    return acc_to_row, aligned_seqs_full, tree_text, sub_df


# ---------------------------------------------------------------------------
# Structure source resolution
# ---------------------------------------------------------------------------

def _resolve_structure(args, prep, out_dir, rng):
    """Return (pdb_path, pae_array_or_None, ref_seq_id) for the chosen source,
    or None when no structure was requested."""
    if args.fetch_alphafold:
        seq_ids = np.asarray(prep.retained_sequence_ids).astype(str)
        logger.info("Fetching an AlphaFold structure...")
        meta = fetch_alphafold_structure(
            seq_ids, out_dir, rng=rng, seq_id=args.ref_seq_id,
            max_tries=args.max_tries)
        pae = load_pae_json(meta["pae_path"])
        return meta["pdb_path"], pae, meta["seq_id"]
    if args.structure is not None:
        pae = load_pae_json(args.pae) if args.pae is not None else None
        return args.structure, pae, args.ref_seq_id
    return None


def _structure_leg(args, sca, prep, kstar, out_dir, rng):
    """Run the structural-contiguity leg. Returns (within_by_ic, cross,
    touching, structure_meta) or (None, None, None, None) when skipped."""
    resolved = _resolve_structure(args, prep, out_dir, rng)
    if resolved is None:
        return None, None, None, None
    pdb_path, pae, ref_seq_id = resolved

    # Lazy imports: structure deps (Bio.PDB) only loaded when needed.
    from mysca.structure import PDBStructure, project_pdb

    pdb = PDBStructure.from_file(pdb_path, chain=args.chain)
    logger.info("Structure %s chain %s (%d residues); ref_seq_id=%s",
                pdb.structure_id, pdb.chain_id, len(pdb), ref_seq_id)

    coords = residue_coords(pdb)
    dist_mat = min_atom_distance_matrix(coords)
    ali_err = align_pae_to_residues(pae, pdb.residue_ids) if pae is not None else None
    if ali_err is None:
        logger.warning("No usable PAE; running structure analysis without "
                       "PAE masking (all residue pairs used).")

    proj = project_pdb(
        pdb, sca_result_dir=args.scacore, preproc_result_dir=args.preprocessing,
        seq_id=ref_seq_id, aligner=args.aligner,
        workdir=os.path.join(out_dir, "_align_workdir"),
        aligner_kwargs={"bin_path": args.align_bin, "threads": args.align_threads},
    )
    ic_residues = ic_residues_on_structure(proj, pdb, kstar)
    sizes = [len(r) for r in ic_residues]
    if sum(sizes) == 0:
        logger.warning("No IC residues mapped onto the structure; skipping "
                       "the structure leg.")
        return None, None, None, {"pdb_path": os.path.abspath(pdb_path),
                                   "ref_seq_id": ref_seq_id}
    logger.info("IC residues on structure: %s", sizes)

    nbr_df = per_ic_contiguity(ic_residues, dist_mat, ali_err,
                               max_err=args.pae_max_err, n_null=args.n_null,
                               seed=args.seed)
    cross = cross_ic_distances(ic_residues, dist_mat, ali_err,
                               max_err=args.pae_max_err)
    touching = touching_pairs(cross, args.cross_thresh)
    # IC index k (matrix order) -> within-IC mean neighbour distance.
    within_by_ic = {k: float(d) for k, d in enumerate(nbr_df["mean_nbr_dist"])}
    meta = {"pdb_path": os.path.abspath(pdb_path), "ref_seq_id": ref_seq_id}
    return within_by_ic, cross, touching, meta


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main(args):
    out_dir = args.outdir
    os.makedirs(out_dir, exist_ok=True)
    configure_logging(verbosity=args.verbosity,
                      logfile=os.path.join(out_dir, EXAMINE_LOG_FNAME))
    with open(os.path.join(out_dir, EXAMINE_ARGS_FNAME), "w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

    rng = np.random.default_rng(args.seed)

    from mysca.results import PreprocessingResults, SCAResults
    sca = SCAResults.load(args.scacore)
    prep = PreprocessingResults.load(args.preprocessing)
    family = args.family or os.path.basename(
        os.path.dirname(os.path.abspath(args.scacore)))

    kstar = n_significant_ics(sca)
    axis_cols = [f"Up_{k}" for k in range(kstar)]
    comp_names = [f"{family}_{c}" for c in axis_cols]
    n_seqs = int(prep.msa.shape[0])
    align_length = int(prep.msa.shape[1])
    logger.info("%s: %d significant IC(s) (kstar=%d of %d); n_seqs=%d L=%d",
                family, kstar, kstar,
                len(sca.ic_positions) if sca.ic_positions is not None else kstar,
                n_seqs, align_length)

    per_ic = per_ic_table(sca, kstar, comp_names,
                          n_seqs=n_seqs, align_length=align_length)

    tree_newick = subsample_df = None
    if not args.no_phylogeny:
        acc_to_row, aligned_seqs, tree_newick, subsample_df = _phylogeny_inputs(
            args, sca, prep, kstar, comp_names, axis_cols, out_dir, rng)
        n_null = args.null_projections
        if n_null > 0 and aligned_seqs is None:
            logger.warning("No aligned_sequence available; disabling the "
                           "random-projection null (lrt_z will be NaN).")
            n_null = 0
        logger.info("Pagel's lambda per IC:")
        pagel_df = run_pagel(parse_newick(tree_newick), acc_to_row, comp_names,
                             transform=not args.no_transform,
                             aligned_seqs=aligned_seqs, n_null=n_null,
                             seed=args.seed)
        per_ic = per_ic.merge(pagel_df, on="component", how="left")
        logger.info("Best clade split per IC:")
        splits_df = run_splits(parse_newick(tree_newick), acc_to_row, comp_names,
                               min_side_size=args.min_side_size, seed=args.seed)
        if not splits_df.empty:
            per_ic = per_ic.merge(
                splits_df[["component", "abs_cliffs_delta", "clade_size",
                           "q_bh"]].rename(columns={"q_bh": "split_q_bh"}),
                on="component", how="left")

    cross = structure_meta = None
    if args.structure is not None or args.fetch_alphafold:
        within_by_ic, cross, touching, structure_meta = _structure_leg(
            args, sca, prep, kstar, out_dir, rng)
        if within_by_ic is not None:
            per_ic["within_dist"] = per_ic["ic_idx"].map(within_by_ic)
            if touching:
                logger.info("Touching IC pairs (< %.1f A): %s", args.cross_thresh,
                            ", ".join(f"IC{i}-IC{j}({kind})"
                                      for i, j, _a, _b, kind in touching))

    sectors = None
    if not args.no_sectors:
        sectors = group_ics(per_ic, cross, family=family, cons_r=args.cons_r,
                            subclade_delta=args.subclade_delta,
                            phylo_z=args.phylo_z)

    results = ExamineResults(
        family, per_ic, sectors=sectors, cross=cross,
        subsample_projections=subsample_df, tree_newick=tree_newick,
        structure_meta=structure_meta)
    written = results.save(out_dir)

    _report(family, per_ic, sectors)
    logger.info("sca-examine done. Wrote %d file(s) to %s", len(written), out_dir)


def _report(family, per_ic, sectors):
    show = [c for c in ["component", "lambda", "ic_n_residues",
                        "ic_frac_residues", "cons_corr_r", "cons_corr_p_adj",
                        "lamP", "lrt_z", "abs_cliffs_delta", "within_dist"]
            if c in per_ic.columns]
    print("\n" + "=" * 78)
    print(f"{family}: per-IC characterization")
    print("=" * 78)
    with pd.option_context("display.float_format", "{:.4g}".format,
                           "display.width", 200, "display.max_columns", None):
        print(per_ic[show].to_string(index=False))
        if sectors is not None and not sectors.empty:
            print("\nSector features:")
            print(sectors[["feature_name", "feature_type", "n_ics",
                           "n_positions", "labels", "associated_sector",
                           "member_components"]].to_string(index=False))


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
