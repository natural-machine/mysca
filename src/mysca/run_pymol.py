"""Visualize SCA sectors on 3D protein structures using PyMOL.

Consumes ``structure_projection.json`` from ``sca-structure`` (which
carries per-structure ``ic_pdb_residues`` in authoritative PDB
residue-number coordinates, plus the originating ``pdb_path``) and
renders each IC group on the structure.

Protein-specific annotations (cofactors, iron-sulfur clusters,
ligands, etc.) are supplied by a user Python file via
``--features_py PATH --features NAME,NAME,...``. The file is imported
at runtime; each requested name must resolve to a callable with
signature::

    def feature_fn(struct, cmd, *, color=None, context=None) -> None

``struct`` is the PyMOL object name; ``cmd`` is PyMOL's ``cmd`` module
(injected so users don't need ``from pymol import cmd``); ``context``
carries a dict with keys ``projection``, ``scaffold``, ``group_idx``,
``outdir`` so feature functions can read ``chain_id`` /
``ic_pdb_residues`` / ``pdb_path`` off the projection.

-------------------------------------------------------------------------------
COMMAND LINE ARGUMENTS:

    --structure DIR : sca-structure output directory (reads
        structure_projection.json).
    --structure_id ID : optional selector when the json has >1
        entry. Default: render every entry.
    -o --outdir : Output directory for rendered PNGs / GIFs.
    -r --reference : another structure_id from the same json to
        align against via cmd.align.
    --groups G [G ...] : IC group indices to render. Default: all
        groups present in the projection.
    --multisector : render all groups on a single frame per structure
        instead of one frame per group.
    --features_py PATH : path to a Python file with user-supplied
        feature functions.
    --features NAME[,NAME,...] : names in --features_py to invoke per
        render pass. Requires --features_py.
    --views : save four rotated side views per frame.
    --animate : save a rotating GIF per rendered frame (one per IC
        group in the default mode; one covering all groups under
        ``--multisector``).
    --nframes N : animation frame count (default 24).
    --duration SEC : animation duration in seconds (default 2.4).

-------------------------------------------------------------------------------
EXAMPLE USAGE:

    # Single structure:
    sca-pymol --structure out/structure --groups 0 1 2 -o out/pymol

    # Batch, pick one structure, align to another:
    sca-pymol --structure out/structure --structure_id NarG_1Q16 \\
        -r NarG_3IR6 --multisector --views -o out/pymol

    # With user features:
    sca-pymol --structure out/structure --structure_id NarG_1Q16 \\
        --features_py scripts/narg_features.py \\
        --features show_molybdenum,show_sf4_cluster \\
        -o out/pymol
"""

import argparse
import importlib.util
import json
import logging
import os
import sys
from typing import Callable, Iterable

import numpy as np

from mysca.constants import SECTOR_COLORS
from mysca.logging_config import configure_logging

# pymol, imageio, and PIL are deferred imports (see _require_cmd /
# _load_animation_deps) so argparse validation and features-loader
# tests can run in environments without pymol-open-source installed.

PYMOL_LOG_FNAME = "pymol.log"
STRUCTURE_JSON_FNAME = "structure_projection.json"

logger = logging.getLogger("mysca.run_pymol")

DEFAULT_STRUCT_COLOR = "gray70"
DEFAULT_STRUCT_STYLE = "sticks"
DEFAULT_STRUCT_ALPHA = 0.5
DEFAULT_SECTOR_COLORS = SECTOR_COLORS
DEFAULT_SECTOR_STYLE = "spheres"
DEFAULT_BG_COLOR = "white"


def parse_args(args):
    parser = argparse.ArgumentParser(
        description=(
            "Render SCA sectors on PDB structures via PyMOL. Consumes "
            "sca-structure output; protein-specific annotations are "
            "supplied by a user --features_py Python file."
        ),
    )
    parser.add_argument(
        "--structure", type=str, required=True, metavar="DIR",
        help="sca-structure output directory (must contain "
        f"{STRUCTURE_JSON_FNAME}).",
    )
    parser.add_argument(
        "--structure_id", type=str, default=None, metavar="ID",
        help="Specific structure_id to render when the json has more "
        "than one entry. Default: render every entry.",
    )
    parser.add_argument(
        "-o", "--outdir", type=str, required=True,
        help="Output directory for rendered images.",
    )
    parser.add_argument(
        "-r", "--reference", type=str, default=None,
        help="Align the target structure to this structure_id from "
        "the same sca-structure batch.",
    )
    parser.add_argument(
        "--groups", type=int, nargs="*", default=None,
        help="IC group indices to render (0-based). Default: all "
        "groups present in the projection.",
    )
    parser.add_argument(
        "--multisector", action="store_true",
        help="Render all selected groups on a single frame per "
        "structure instead of one frame per group.",
    )
    parser.add_argument(
        "--features_py", type=str, default=None, metavar="PATH",
        help="Path to a user Python file supplying protein-specific "
        "annotation functions.",
    )
    parser.add_argument(
        "--features", type=str, default=None, metavar="NAMES",
        help="Comma-separated names in --features_py to invoke per "
        "render pass. Requires --features_py.",
    )
    parser.add_argument(
        "--views", action="store_true",
        help="Save four rotated side views per frame.",
    )
    parser.add_argument(
        "--animate", action="store_true",
        help="Save a rotating GIF per rendered frame (one per IC group "
        "in the default mode; one covering all groups under "
        "--multisector).",
    )
    parser.add_argument(
        "--nframes", type=int, default=None,
        help="Animation frame count (used with --animate). Default 24.",
    )
    parser.add_argument(
        "--duration", type=float, default=None,
        help="Animation duration in seconds (used with --animate). "
        "Default 2.4.",
    )
    parser.add_argument(
        "-v", "--verbosity", type=int, default=1,
        help="Verbosity level (0=warnings only).",
    )

    parsed = parser.parse_args(args)
    if parsed.features and not parsed.features_py:
        parser.error("--features requires --features_py.")
    return parsed


def _load_features_module(path: str, names: Iterable[str]) -> list[Callable]:
    """Import a user Python file and resolve each requested callable.

    Raises ``FileNotFoundError`` for a missing file, ``ValueError``
    for a missing attribute, and ``TypeError`` for a non-callable
    attribute. The user module is loaded with a private module name
    (``"mysca_user_features"``) so it does not collide with any real
    package.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Features file not found: {path}")
    spec = importlib.util.spec_from_file_location(
        "mysca_user_features", path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not build import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    resolved: list[Callable] = []
    for name in names:
        if not hasattr(module, name):
            raise ValueError(
                f"Features file {path!r} has no attribute {name!r}"
            )
        fn = getattr(module, name)
        if not callable(fn):
            raise TypeError(
                f"Features file {path!r} attribute {name!r} is not "
                f"callable: {type(fn).__name__}"
            )
        resolved.append(fn)
    return resolved


def _split_features_names(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [name.strip() for name in raw.split(",") if name.strip()]


def _load_projections(structure_dir: str) -> list[dict]:
    json_path = os.path.join(structure_dir, STRUCTURE_JSON_FNAME)
    if not os.path.isfile(json_path):
        raise FileNotFoundError(
            f"{STRUCTURE_JSON_FNAME} not found in {structure_dir}"
        )
    with open(json_path) as f:
        return json.load(f)


def _require_cmd():
    """Import pymol lazily so argparse / features-loader tests can run
    in environments without pymol-open-source installed."""
    try:
        from pymol import cmd  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "sca-pymol requires the optional pymol-open-source "
            "dependency. Install via `conda install -c conda-forge "
            "pymol-open-source` or equivalent."
        ) from exc
    return cmd


def _load_animation_deps():
    """Lazy import for the animation path (imageio + PIL)."""
    import imageio
    from PIL import Image
    return imageio, Image


def main(args):
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    configure_logging(
        verbosity=args.verbosity,
        logfile=os.path.join(outdir, PYMOL_LOG_FNAME),
    )

    projections = _load_projections(args.structure)
    proj_by_id = {p["structure_id"]: p for p in projections}
    if not proj_by_id:
        raise ValueError(
            f"{STRUCTURE_JSON_FNAME} in {args.structure} has no entries."
        )

    if args.structure_id is not None:
        if args.structure_id not in proj_by_id:
            raise KeyError(
                f"structure_id {args.structure_id!r} not found in "
                f"{args.structure}; available: {sorted(proj_by_id)}"
            )
        target_ids = [args.structure_id]
    else:
        target_ids = list(proj_by_id)

    if args.reference is not None and args.reference not in proj_by_id:
        raise KeyError(
            f"reference structure_id {args.reference!r} not found in "
            f"{args.structure}; available: {sorted(proj_by_id)}"
        )

    feature_fns: list[Callable] = []
    if args.features_py is not None:
        names = _split_features_names(args.features)
        feature_fns = _load_features_module(args.features_py, names)
        logger.info(
            "Loaded %d feature function(s) from %s",
            len(feature_fns), args.features_py,
        )

    sector_colors = [_hex2color(x) for x in DEFAULT_SECTOR_COLORS]
    cmd = _require_cmd()

    for sid in target_ids:
        projection = proj_by_id[sid]
        ic_pdb_residues = projection.get("ic_pdb_residues") or []
        available_groups = list(range(len(ic_pdb_residues)))
        if args.groups is None:
            group_idxs = available_groups
        else:
            group_idxs = [g for g in args.groups if g in available_groups]
            missing = set(args.groups) - set(available_groups)
            if missing:
                logger.warning(
                    "Structure %s has no residues for groups %s; skipping.",
                    sid, sorted(missing),
                )

        _render_one_structure(
            cmd=cmd,
            projection=projection,
            reference_projection=(
                proj_by_id[args.reference] if args.reference else None
            ),
            group_idxs=group_idxs,
            multisector=args.multisector,
            sector_colors=sector_colors,
            feature_fns=feature_fns,
            views=args.views,
            animate=args.animate,
            nframes=args.nframes,
            duration=args.duration,
            outdir=outdir,
        )

    logger.info("Done!")


def _render_one_structure(
        *,
        cmd,
        projection: dict,
        reference_projection: dict | None,
        group_idxs: list[int],
        multisector: bool,
        sector_colors: list[str],
        feature_fns: list[Callable],
        views: bool,
        animate: bool,
        nframes: int | None,
        duration: float | None,
        outdir: str,
):
    struct_color = DEFAULT_STRUCT_COLOR
    struct_style = DEFAULT_STRUCT_STYLE
    struct_alpha = DEFAULT_STRUCT_ALPHA
    sector_style = DEFAULT_SECTOR_STYLE

    scaffold = projection["structure_id"]
    pdb_path = projection.get("pdb_path")
    if not pdb_path:
        raise ValueError(
            f"Projection for {scaffold!r} has no pdb_path; rerun "
            "sca-structure so the path is recorded in "
            f"{STRUCTURE_JSON_FNAME}."
        )
    if not os.path.isfile(pdb_path):
        raise FileNotFoundError(
            f"Projection for {scaffold!r} points at a PDB that does "
            f"not exist: {pdb_path}"
        )

    # Fresh PyMOL session per structure so selections / objects don't
    # leak between structures in seq_map batches.
    cmd.delete("all")
    cmd.load(pdb_path, "struct")

    ref_scaffold = None
    if reference_projection is not None:
        ref_scaffold = reference_projection["structure_id"]
        ref_pdb = reference_projection.get("pdb_path")
        if ref_pdb and os.path.isfile(ref_pdb):
            cmd.load(ref_pdb, "ref_struct")
        else:
            logger.warning(
                "Reference %s has no valid pdb_path; skipping alignment.",
                ref_scaffold,
            )
            ref_scaffold = None

    cmd.hide("everything", "struct")
    if ref_scaffold:
        cmd.hide("everything", "ref_struct")
    cmd.bg_color(DEFAULT_BG_COLOR)
    cmd.show(struct_style, "struct")
    cmd.color(struct_color, "struct")
    cmd.set(
        {"sticks": "stick_transparency"}.get(struct_style, DEFAULT_STRUCT_STYLE),
        1 - struct_alpha,
        "struct",
    )
    cmd.zoom(complete=1)

    if multisector:
        logger.info("Plotting %s with all sectors...", scaffold)
        _plot_with_multiple_sectors(
            cmd=cmd,
            scaffold=scaffold,
            projection=projection,
            group_idxs=group_idxs,
            struct_color=struct_color,
            sector_colors=sector_colors,
            sector_style=sector_style,
            ref_scaffold=ref_scaffold,
            feature_fns=feature_fns,
            views=views,
            animate=animate,
            nframes=nframes,
            duration=duration,
            outdir=outdir,
        )
    else:
        logger.info("Plotting %s by sector...", scaffold)
        _plot_by_sectors(
            cmd=cmd,
            scaffold=scaffold,
            projection=projection,
            group_idxs=group_idxs,
            struct_color=struct_color,
            sector_colors=sector_colors,
            sector_style=sector_style,
            ref_scaffold=ref_scaffold,
            feature_fns=feature_fns,
            views=views,
            animate=animate,
            nframes=nframes,
            duration=duration,
            outdir=outdir,
        )


def _selection_from_residues(residues):
    residues = [int(r) for r in residues]
    if not residues:
        return None
    return "resi " + "+".join(str(r) for r in residues)


def _run_feature_fns(cmd, feature_fns, scaffold, projection, group_idx, outdir):
    if not feature_fns:
        return
    context = {
        "projection": projection,
        "scaffold": scaffold,
        "group_idx": group_idx,
        "outdir": outdir,
    }
    for fn in feature_fns:
        fn(scaffold, cmd, color=None, context=context)


def _apply_group_coloring(
        cmd, selection_name, pdb_resids, scores, sector_color, sector_style,
):
    """Make a PyMOL selection for one IC group, color + show it, and
    apply per-residue alpha from the optional loadings.

    Returns the selection name on success, ``None`` when the group has
    no residues.
    """
    selection_string = _selection_from_residues(pdb_resids)
    if selection_string is None:
        return None
    cmd.select(selection_name, selection_string)
    cmd.show(sector_style, selection_name)
    cmd.color(sector_color, selection_name)

    if not scores or not pdb_resids:
        return selection_name

    MIN_ALPHA = 0.5
    svals = np.square(np.asarray(scores, dtype=float))
    s0, s1 = float(svals.min()), float(svals.max())
    if s1 == s0:
        alphas = np.full_like(svals, 1.0)
    else:
        alphas = (1.0 - MIN_ALPHA) / (s1 - s0) * (svals - s1) + 1.0
    logger.debug(
        "Applying alphas [%.4g, %.4g] to %s",
        alphas.min(), alphas.max(), selection_name,
    )
    transparency_attr = {
        "spheres": "sphere_transparency",
    }.get(sector_style, DEFAULT_SECTOR_STYLE)
    for resi, alpha in zip(pdb_resids, alphas):
        cmd.set(
            transparency_attr,
            1 - float(alpha),
            f"{selection_name} and resi {int(resi)}",
        )
    return selection_name


def _align_and_focus(cmd, ref_scaffold):
    if ref_scaffold:
        cmd.align("struct", "ref_struct")
    cmd.center()
    cmd.zoom(complete=1)


def _write_views(cmd, outdir, basename, ref_scaffold):
    """Save four rotated side views (90° steps around Y) under
    ``<outdir>/views/<basename>_view{0..3}.png``."""
    viewsdir = os.path.join(outdir, "views")
    os.makedirs(viewsdir, exist_ok=True)
    for ri in range(4):
        cmd.png(os.path.join(viewsdir, f"{basename}_view{ri}.png"), dpi=300)
        cmd.rotate("y", 90, "struct")
        if ref_scaffold:
            cmd.rotate("y", 90, "ref_struct")


def _write_animation(cmd, outdir, basename, nframes, duration):
    imageio, Image = _load_animation_deps()
    import tqdm as _tqdm
    RAY_FIRST = 1
    seconds_per_frame = duration / nframes
    framesdir = os.path.join(outdir, "frames", f"{basename}_frames")
    os.makedirs(framesdir, exist_ok=True)
    for i in _tqdm.trange(nframes, leave=False):
        cmd.turn("y", 360 / nframes)
        cmd.png(
            os.path.join(framesdir, f"frame_{i:03d}.png"),
            dpi=300, ray=RAY_FIRST,
        )
    frames = []
    for i in range(nframes):
        im = Image.open(
            os.path.join(framesdir, f"frame_{i:03d}.png"),
        ).convert("RGBA")
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.getchannel("A"))
        frames.append(np.array(bg))
    outfile = os.path.join(outdir, f"{basename}.gif")
    imageio.mimsave(
        outfile, frames, duration=seconds_per_frame, loop=0, disposal=2,
    )


def _render_frame(
        *,
        cmd,
        scaffold,
        projection,
        group_idxs,
        sector_colors,
        sector_style,
        struct_color,
        ref_scaffold,
        feature_fns,
        views,
        animate,
        nframes,
        duration,
        basename,
        group_idx_for_features,
        outdir,
):
    """Render one frame with the given set of IC groups lit up.

    Collects per-group selections, aligns + focuses, invokes feature
    functions, writes the main PNG, optionally writes rotated views and
    a rotating-GIF animation, and cleans up every selection it created.
    Per-group rendering loops over this once per ``gidx`` (passing the
    ``gidx`` through to feature fns); multisector rendering calls it
    once with all ``group_idxs`` and ``group_idx_for_features=None``.
    """
    ic_pdb_residues = projection["ic_pdb_residues"]
    ic_loadings = projection["sequence_projection"]["ic_loadings"]

    os.makedirs(outdir, exist_ok=True)

    created: list[str] = []
    for gidx in group_idxs:
        sector_color = sector_colors[gidx % len(sector_colors)]
        pdb_resids = ic_pdb_residues[gidx]
        scores = ic_loadings[gidx] if gidx < len(ic_loadings) else []
        sel_name = f"group_selection{gidx}"
        selection = _apply_group_coloring(
            cmd, sel_name, pdb_resids, scores,
            sector_color, sector_style,
        )
        if selection is None:
            logger.info(
                "Structure %s group %d has no residues; skipping.",
                scaffold, gidx,
            )
        else:
            created.append(selection)

    _align_and_focus(cmd, ref_scaffold)
    _run_feature_fns(
        cmd, feature_fns, scaffold, projection,
        group_idx_for_features, outdir,
    )

    cmd.png(f"{outdir}/{basename}.png", dpi=300)

    if views:
        _write_views(cmd, outdir, basename, ref_scaffold)

    if animate:
        _write_animation(cmd, outdir, basename, nframes, duration)

    for sel_name in created:
        cmd.hide(sector_style, sel_name)
        cmd.color(struct_color, sel_name)
        cmd.delete(sel_name)


def _plot_by_sectors(
        *,
        cmd,
        scaffold,
        projection,
        group_idxs,
        struct_color,
        sector_colors,
        sector_style,
        ref_scaffold,
        feature_fns,
        views,
        animate,
        nframes,
        duration,
        outdir,
):
    if nframes is None:
        nframes = 24
    if duration is None:
        duration = 2.4

    for gidx in group_idxs:
        _render_frame(
            cmd=cmd,
            scaffold=scaffold,
            projection=projection,
            group_idxs=[gidx],
            sector_colors=sector_colors,
            sector_style=sector_style,
            struct_color=struct_color,
            ref_scaffold=ref_scaffold,
            feature_fns=feature_fns,
            views=views,
            animate=animate,
            nframes=nframes,
            duration=duration,
            basename=f"{scaffold}_group{gidx}",
            group_idx_for_features=gidx,
            outdir=outdir,
        )


def _plot_with_multiple_sectors(
        *,
        cmd,
        scaffold,
        projection,
        group_idxs,
        struct_color,
        sector_colors,
        sector_style,
        ref_scaffold,
        feature_fns,
        views,
        animate,
        nframes,
        duration,
        outdir,
):
    if nframes is None:
        nframes = 24
    if duration is None:
        duration = 2.4

    tag = ",".join(str(i) for i in group_idxs)
    _render_frame(
        cmd=cmd,
        scaffold=scaffold,
        projection=projection,
        group_idxs=group_idxs,
        sector_colors=sector_colors,
        sector_style=sector_style,
        struct_color=struct_color,
        ref_scaffold=ref_scaffold,
        feature_fns=feature_fns,
        views=views,
        animate=animate,
        nframes=nframes,
        duration=duration,
        basename=f"{scaffold}_groups_{tag}",
        group_idx_for_features=None,
        outdir=outdir,
    )


def _hex2color(x):
    return "0x" + x[1:]


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
