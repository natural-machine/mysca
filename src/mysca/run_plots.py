"""Replay plotting CLI (``sca-plots``).

Regenerate diagnostic plots from persisted results without rerunning the
pipeline. Each stage is opt-in via its own flag; at least one must be given.

-------------------------------------------------------------------------------
EXAMPLE USAGE:

    sca-plots --prealign prealign_out
    sca-plots --preprocessing preprocess_out --scacore scacore_out
    sca-plots --scacore scacore_out --imgdir shared_images/

-------------------------------------------------------------------------------
COVERED PLOTS:

Per-stage, plots are written into ``{stage_dir}/images/`` by default, or into
``--imgdir`` if given. Each stage only runs when its flag is passed.

    --prealign DIR       plot_prealign_filter_history
    --preprocessing DIR  plot_filter_history, plot_filter_distributions,
                         plot_sequence_similarity
    --scacore DIR        plot_conservation (+ _top and _positional when
                         --preprocessing is also given so retained_positions
                         and the original MSA length are available),
                         plot_sca_matrix, plot_covariance_matrix (when
                         Cij_raw is present — persisted by sca-core runs
                         on or after commit HEAD), plot_sca_spectrum,
                         plot_sca_spectrum_vs_null, plot_dendrogram,
                         plot_t_distributions, plot_data_2d/3d (EV + IC
                         sweeps), plot_sca_matrix_sector_subset,
                         plot_seq_projection_2d (requires --preprocessing
                         for msa_binary3d; optionally colored by
                         --seq_proj_color_by COLUMN of
                         sequence_metadata.tsv).
"""

import argparse
import json
import logging
import os
import sys

from mysca.logging_config import configure_logging
from mysca.results import PreprocessingResults, SCAResults
from mysca.pl import (
    plot_conservation,
    plot_conservation_positional,
    plot_conservation_top,
    plot_covariance_matrix,
    plot_data_2d,
    plot_data_3d,
    plot_dendrogram,
    plot_filter_distributions,
    plot_filter_history,
    plot_prealign_filter_history,
    plot_sca_matrix,
    plot_sca_matrix_sector_subset,
    plot_sca_spectrum,
    plot_sca_spectrum_vs_null,
    plot_sequence_similarity,
    plot_seq_projection_2d,
    plot_t_distributions,
    resolve_color_values,
)


PREALIGN_FILTER_HISTORY_FNAME = "filter_history.json"

logger = logging.getLogger("mysca.run_plots")


# Kept in sync with run_sca.make_plots sweeps. plot_data_2d/3d silently
# skip axes beyond data.shape[1], so no extra guarding is needed here.
_AXES_2D = [
    ((0, 1), "all"),
    ((1, 2), "all"),
    ((2, 3), "all"),
    ((3, 4), "all"),
    ((4, 5), "all"),
    ((5, 6), "all"),
    ((0, 1), [0, 1, 2]),
    ((1, 2), [0, 1, 2]),
]

_AXES_3D = [
    ((0, 1, 2), "all"),
    ((1, 2, 3), "all"),
    ((0, 1, 2), [0, 1, 2]),
    ((1, 2, 3), [0, 1, 2]),
]


def parse_args(args):
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate SCA diagnostic plots from persisted results. "
            "At least one of --prealign/--preprocessing/--scacore is required."
        ),
    )
    parser.add_argument(
        "--prealign", type=str, default=None, metavar="DIR",
        help="Prealign output directory (contains filter_history.json).",
    )
    parser.add_argument(
        "--preprocessing", type=str, default=None, metavar="DIR",
        help="Preprocessing output directory "
        "(contains preprocessing_results.npz, filter_history.json, "
        "msa_binary2d_sp.npz).",
    )
    parser.add_argument(
        "--scacore", type=str, default=None, metavar="DIR",
        help="SCA core output directory (contains scarun_results.npz, "
        "sca_eigendecomp.npz, sca_results/).",
    )
    parser.add_argument(
        "--imgdir", type=str, default=None, metavar="DIR",
        help="Output directory for all plots. Default: write into each "
        "stage's own 'images/' subdirectory.",
    )
    parser.add_argument(
        "--seq_proj_color_by", type=str, default=None, metavar="COLUMN",
        help="Color the seq_proj_ic*.png plot by this column of "
        "sequence_metadata.tsv (loaded from --scacore). Numeric "
        "columns get a colorbar; categorical columns get a legend.",
    )
    parser.add_argument("-v", "--verbosity", type=int, default=1)

    parsed = parser.parse_args(args)
    if not any([parsed.prealign, parsed.preprocessing, parsed.scacore]):
        parser.error(
            "At least one of --prealign/--preprocessing/--scacore is required."
        )
    return parsed


def _resolve_imgdir(stage_dir, override):
    target = override if override is not None else os.path.join(stage_dir, "images")
    os.makedirs(target, exist_ok=True)
    return target


def _replay_prealign(stage_dir, imgdir_override):
    if not os.path.isdir(stage_dir):
        raise FileNotFoundError(f"Prealign directory not found: {stage_dir}")
    fh_path = os.path.join(stage_dir, PREALIGN_FILTER_HISTORY_FNAME)
    if not os.path.isfile(fh_path):
        raise FileNotFoundError(
            f"Prealign filter history not found at {fh_path}. "
            "Rerun sca-prealign; filter_history.json is always persisted."
        )
    with open(fh_path) as f:
        filter_history = json.load(f)
    imgdir = _resolve_imgdir(stage_dir, imgdir_override)
    logger.info("Replaying prealign plots into %s", imgdir)
    plot_prealign_filter_history(filter_history, imgdir)


def _replay_preprocessing(stage_dir, imgdir_override):
    if not os.path.isdir(stage_dir):
        raise FileNotFoundError(f"Preprocessing directory not found: {stage_dir}")
    prep = PreprocessingResults.load(stage_dir)
    imgdir = _resolve_imgdir(stage_dir, imgdir_override)
    logger.info("Replaying preprocessing plots into %s", imgdir)

    if prep.filter_history is not None:
        plot_filter_history(prep.filter_history, imgdir)
        plot_filter_distributions(prep.filter_history, imgdir)
    else:
        logger.warning(
            "No filter_history.json in %s; skipping filter-history / "
            "filter-distribution plots.", stage_dir,
        )

    if prep.msa_binary3d is not None:
        plot_sequence_similarity(prep.msa_binary3d, imgdir)
    else:
        logger.warning(
            "No msa_binary2d_sp.npz in %s; skipping sequence similarity plot.",
            stage_dir,
        )


def _replay_scacore(
        stage_dir, imgdir_override, *,
        preproc_dir=None,
        seq_proj_color_by=None,
):
    if not os.path.isdir(stage_dir):
        raise FileNotFoundError(f"SCA core directory not found: {stage_dir}")
    sca = SCAResults.load(stage_dir)
    imgdir = _resolve_imgdir(stage_dir, imgdir_override)
    logger.info("Replaying scacore plots into %s", imgdir)

    kstar = sca.kstar if sca.kstar is not None else 1

    # Conservation plots — positional variants need retained_positions +
    # the original MSA length, which come from the upstream preprocessing
    # directory. Plain conservation.png only needs sca.conservation.
    if sca.conservation is not None:
        plot_conservation(sca.conservation, imgdir)
        prep = _maybe_load_preprocessing(preproc_dir, stage_dir)
        if prep is not None and prep.retained_positions is not None \
                and prep.msa_obj_loaded is not None:
            num_pos_loaded = prep.msa_obj_loaded.get_alignment_length()
            plot_conservation_top(
                prep.retained_positions, sca.conservation, num_pos_loaded,
                imgdir,
            )
            plot_conservation_positional(
                prep.retained_positions, sca.conservation, num_pos_loaded,
                imgdir,
            )
        else:
            logger.warning(
                "Preprocessing dir not resolved; skipping top_conservation.png "
                "and positional_conservation.png (they need retained_positions "
                "+ original MSA length)."
            )
    else:
        logger.warning(
            "No scarun_results.npz (conservation) in %s; skipping "
            "conservation plots.", stage_dir,
        )

    if sca.sca_matrix is not None:
        plot_sca_matrix(sca.sca_matrix, imgdir)
        plot_dendrogram(sca.sca_matrix, imgdir, nclusters=kstar)
    else:
        logger.warning(
            "No sca_matrix in %s; skipping sca_matrix and dendrogram.",
            stage_dir,
        )

    if sca.evals_sca is not None and sca.evals_shuff is not None \
            and len(sca.evals_shuff) > 0:
        plot_sca_spectrum(sca.evals_sca, sca.evals_shuff, imgdir)
        if sca.cutoff is not None:
            plot_sca_spectrum_vs_null(
                sca.evals_sca, sca.evals_shuff, sca.cutoff,
                len(sca.evals_shuff), imgdir,
            )
        else:
            logger.warning(
                "No cutoff scalar in %s; skipping sca_matrix_spectrum_vs_null.",
                stage_dir,
            )
    else:
        logger.warning(
            "Missing evals_sca or evals_shuff in %s; skipping spectrum plots.",
            stage_dir,
        )

    if sca.Cij_raw is not None:
        plot_covariance_matrix(sca.Cij_raw, imgdir)
    else:
        logger.info(
            "No Cij_raw in %s; skipping covariance_matrix.png "
            "(the core run predates Cij_raw persistence).", stage_dir,
        )

    if sca.v_ica is not None and sca.t_dists_info is not None:
        plot_t_distributions(
            sca.v_ica, sca.t_dists_info, imgdir, max_plots=kstar,
        )
    else:
        logger.warning(
            "Missing v_ica_normalized or t_dists_info in %s; "
            "skipping t-distribution plot.", stage_dir,
        )

    if sca.ic_positions is None:
        logger.warning(
            "No IC positions on disk in %s; skipping EV/IC scatter + "
            "sector-subset plots.",
            stage_dir,
        )
        return

    if sca.significant_evecs_sca is not None:
        _replay_data_sweeps(
            "ev", sca.significant_evecs_sca, sca.ic_positions, imgdir,
        )
    else:
        logger.warning(
            "No significant_evecs_sca in %s; skipping EV scatter plots.",
            stage_dir,
        )

    if sca.v_ica is not None:
        _replay_data_sweeps("ic", sca.v_ica, sca.ic_positions, imgdir)
    else:
        logger.warning(
            "No v_ica_normalized in %s; skipping IC scatter plots.", stage_dir,
        )

    if sca.sca_matrix_sector_subset is not None:
        from mysca.constants import SECTOR_COLORS
        plot_sca_matrix_sector_subset(
            sca.sca_matrix_sector_subset, sca.ic_positions,
            SECTOR_COLORS, imgdir,
        )
    else:
        logger.warning(
            "No sca_matrix_sector_subset in %s; skipping sector-subset plot.",
            stage_dir,
        )

    prep = _maybe_load_preprocessing(preproc_dir, stage_dir)
    if prep is not None and prep.msa_binary3d is not None:
        try:
            up_seq = sca.project_sequences(prep.msa_binary3d)
        except RuntimeError as e:
            logger.warning(
                "Cannot compute sequence projection in %s: %s", stage_dir, e,
            )
        else:
            color_values = None
            color_label = None
            if seq_proj_color_by is not None:
                if sca.sequence_metadata is None:
                    logger.warning(
                        "--seq_proj_color_by=%r ignored: %s has no "
                        "sequence_metadata.tsv.",
                        seq_proj_color_by, stage_dir,
                    )
                elif seq_proj_color_by not in sca.sequence_metadata.columns:
                    logger.warning(
                        "--seq_proj_color_by=%r ignored: column missing. "
                        "Available: %s",
                        seq_proj_color_by,
                        list(sca.sequence_metadata.columns),
                    )
                else:
                    color_values = resolve_color_values(
                        sca.sequence_metadata,
                        list(prep.retained_sequence_ids),
                        seq_proj_color_by,
                    )
                    color_label = seq_proj_color_by
            plot_seq_projection_2d(
                up_seq, (0, 1), imgdir,
                color_values=color_values, color_label=color_label,
            )
    else:
        logger.warning(
            "Preprocessing dir not resolved or msa_binary3d missing; "
            "skipping seq_proj_ic*.png (needs PreprocessingResults.msa_binary3d).",
        )


def _maybe_load_preprocessing(preproc_dir, stage_dir):
    if preproc_dir is None:
        return None
    if not os.path.isdir(preproc_dir):
        logger.warning(
            "Preprocessing dir %s not found; skipping plots that depend on it.",
            preproc_dir,
        )
        return None
    return PreprocessingResults.load(preproc_dir)


def _replay_data_sweeps(ic_or_ev, data, groups, imgdir):
    for axidxs, group_idxs in _AXES_2D:
        plot_data_2d(ic_or_ev, axidxs, group_idxs, groups, data, imgdir)
    for axidxs, group_idxs in _AXES_3D:
        plot_data_3d(ic_or_ev, axidxs, group_idxs, groups, data, imgdir)


def main(args):
    configure_logging(verbosity=args.verbosity, logfile=None)

    if args.prealign:
        _replay_prealign(args.prealign, args.imgdir)
    if args.preprocessing:
        _replay_preprocessing(args.preprocessing, args.imgdir)
    if args.scacore:
        _replay_scacore(
            args.scacore, args.imgdir, preproc_dir=args.preprocessing,
            seq_proj_color_by=args.seq_proj_color_by,
        )

    logger.info("sca-plots done.")


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
