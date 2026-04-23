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
    --scacore DIR        plot_dendrogram, plot_t_distributions,
                         plot_data_2d/3d (EV + IC sweeps)

The inline matplotlib figures in ``run_sca.py::make_plots`` (conservation,
SCA-matrix imshow, spectrum vs null, sector-subset) are not replayed by this
CLI — they are tied to the ``make_plots`` refactor and will be pulled into
``mysca.pl`` in a future change.
"""

import argparse
import json
import logging
import os
import sys

from mysca.logging_config import configure_logging
from mysca.results import PreprocessingResults, SCAResults
from mysca.pl import (
    plot_data_2d,
    plot_data_3d,
    plot_dendrogram,
    plot_filter_distributions,
    plot_filter_history,
    plot_prealign_filter_history,
    plot_sequence_similarity,
    plot_t_distributions,
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


def _replay_scacore(stage_dir, imgdir_override):
    if not os.path.isdir(stage_dir):
        raise FileNotFoundError(f"SCA core directory not found: {stage_dir}")
    sca = SCAResults.load(stage_dir)
    imgdir = _resolve_imgdir(stage_dir, imgdir_override)
    logger.info("Replaying scacore plots into %s", imgdir)

    kstar = sca.kstar if sca.kstar is not None else 1

    if sca.sca_matrix is not None:
        plot_dendrogram(sca.sca_matrix, imgdir, nclusters=kstar)
    else:
        logger.warning(
            "No scarun_results.npz in %s; skipping dendrogram.", stage_dir,
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

    if sca.groups is None:
        logger.warning(
            "No sector groups on disk in %s; skipping EV/IC scatter plots.",
            stage_dir,
        )
        return

    if sca.significant_evecs_sca is not None:
        _replay_data_sweeps("ev", sca.significant_evecs_sca, sca.groups, imgdir)
    else:
        logger.warning(
            "No significant_evecs_sca in %s; skipping EV scatter plots.",
            stage_dir,
        )

    if sca.v_ica is not None:
        _replay_data_sweeps("ic", sca.v_ica, sca.groups, imgdir)
    else:
        logger.warning(
            "No v_ica_normalized in %s; skipping IC scatter plots.", stage_dir,
        )


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
        _replay_scacore(args.scacore, args.imgdir)

    logger.info("sca-plots done.")


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
