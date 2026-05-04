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

``struct`` is the PyMOL object name of the loaded scaffold and is
always the literal string ``"struct"`` (see ``SCAFFOLD_OBJECT_NAME``);
the structure_id (e.g. ``"1Q16"``) is in ``context["scaffold"]``.
``cmd`` is PyMOL's ``cmd`` module (injected so users don't need
``from pymol import cmd``); ``context`` carries a dict with keys
``projection``, ``scaffold``, ``group_idx``, ``outdir``, and
``select`` (a noisy ``cmd.select`` wrapper that logs a WARNING on
zero-match selections — preferred over calling ``cmd.select``
directly).

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
    --struct_style {sticks,cartoon,ribbon,lines,surface} : PyMOL
        representation for the scaffold structure. Default 'sticks'.
    --views : save four rotated side views per frame.
    --animate : save a rotating GIF per rendered frame (one per IC
        group in the default mode; one covering all groups under
        ``--multisector``).
    --nframes N : animation frame count (default 24).
    --duration SEC : animation duration in seconds (default 2.4).
    --spin_axis {x,y,z} : rotation axis for animation (default y).
    --spin_degrees N : total rotation in degrees over --nframes
        (default 360 — full loop).
    --ray {none,first,all} : ray-tracing policy for animation frames.
        'all' (default) rays every frame, 'first' rays only frame 0,
        'none' disables ray-tracing.
    --dpi N : DPI for all rendered PNGs (default 300).
    --format {gif,mp4,both} : animation output format (default gif).
        'mp4' / 'both' require the optional ``imageio-ffmpeg``
        dependency.
    --mode {spin,reveal} : animation mode. 'spin' (default) rotates;
        'reveal' is a still-camera narrative animation. See the
        "Reveal mode" section of `docs/cli_reference.md#sca-pymol`
        for stage scheduling details and examples.
    --reveal_schedule {cumulative,sequential,custom} : how groups
        appear across stages in --mode reveal. Default cumulative.
    --reveal_custom STAGE [STAGE ...] : custom stage schedule when
        --reveal_schedule custom. Each STAGE is a comma-separated
        list of IC group indices visible in that stage.

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

# PyMOL object names. Hoisted from previously-scattered string literals
# so the features-plugin contract ("struct" is always literally the
# loaded object name") is enforced at one place.
SCAFFOLD_OBJECT_NAME = "struct"
REF_SCAFFOLD_OBJECT_NAME = "ref_struct"

logger = logging.getLogger("mysca.run_pymol")

DEFAULT_STRUCT_COLOR = "gray70"
DEFAULT_STRUCT_STYLE = "sticks"
DEFAULT_STRUCT_ALPHA = 0.5
DEFAULT_SECTOR_COLORS = SECTOR_COLORS
DEFAULT_SECTOR_STYLE = "spheres"
DEFAULT_BG_COLOR = "white"

STRUCT_STYLE_CHOICES = ("sticks", "cartoon", "ribbon", "lines", "surface")
# PyMOL transparency settings keyed by representation name.
STRUCT_TRANSPARENCY_PROP = {
    "sticks": "stick_transparency",
    "cartoon": "cartoon_transparency",
    "ribbon": "ribbon_transparency",
    "lines": "line_transparency",
    "surface": "transparency",
}


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
        "--struct_style", type=str, default=DEFAULT_STRUCT_STYLE,
        choices=list(STRUCT_STYLE_CHOICES),
        help="PyMOL representation for the scaffold structure. Default "
        f"{DEFAULT_STRUCT_STYLE!r}. Use 'cartoon' to see secondary "
        "structure; 'sticks' shows every atom.",
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
        "--spin_axis", type=str, default="y", choices=["x", "y", "z"],
        help="Axis to rotate around when --animate is passed. Default y.",
    )
    parser.add_argument(
        "--spin_degrees", type=float, default=360.0,
        help="Total rotation in degrees over --nframes. Default 360 "
        "(full loop). Set to e.g. 180 for a half-spin, 90 for a "
        "quarter-turn.",
    )
    parser.add_argument(
        "--ray", type=str, default="all", choices=["none", "first", "all"],
        help="Ray-tracing policy for animation frames. 'all' "
        "(default) ray-traces every frame (best quality, slowest); "
        "'first' rays only frame 0 and uses viewport for the rest; "
        "'none' disables ray-tracing entirely (fastest).",
    )
    parser.add_argument(
        "--dpi", type=int, default=300,
        help="DPI for all rendered PNGs (stills, views, and animation "
        "frames). Default 300.",
    )
    parser.add_argument(
        "--format", type=str, default="gif",
        choices=["gif", "mp4", "both"],
        help="Animation output format. 'gif' (default), 'mp4' "
        "(smaller / higher quality, requires the optional "
        "imageio-ffmpeg dependency), or 'both' (writes .gif AND "
        ".mp4 from the same frame series).",
    )
    parser.add_argument(
        "--mode", type=str, default="spin",
        choices=["spin", "reveal"],
        help="Animation mode. 'spin' (default) rotates the structure "
        "with all selected groups lit. 'reveal' is a still-camera "
        "narrative animation that walks through stages of which "
        "groups are visible (controlled by --reveal_schedule).",
    )
    parser.add_argument(
        "--reveal_schedule", type=str, default="cumulative",
        choices=["cumulative", "sequential", "custom"],
        help="Stage schedule for --mode reveal. 'cumulative' (default) "
        "stacks groups one at a time on top of each other. "
        "'sequential' shows one group at a time, swapping groups "
        "out as the next appears. 'custom' takes an explicit "
        "list of stages from --reveal_custom.",
    )
    parser.add_argument(
        "--reveal_custom", type=str, nargs="*", default=None,
        metavar="STAGE",
        help="Custom reveal schedule when --reveal_schedule custom. "
        "Each STAGE is a comma-separated list of IC group indices "
        "visible in that stage; stages are space-separated. "
        "Example: --reveal_custom \"1\" \"1,2\" \"1,3\" \"2,3\".",
    )
    parser.add_argument(
        "-v", "--verbosity", type=int, default=1,
        help="Verbosity level (0=warnings only).",
    )

    parsed = parser.parse_args(args)
    if parsed.features and not parsed.features_py:
        parser.error("--features requires --features_py.")
    if parsed.reveal_custom and parsed.reveal_schedule != "custom":
        parser.error(
            "--reveal_custom only applies with --reveal_schedule custom."
        )
    if parsed.reveal_schedule == "custom" and not parsed.reveal_custom:
        parser.error(
            "--reveal_schedule custom requires --reveal_custom STAGE [...]."
        )
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


def _require_ffmpeg():
    """Verify imageio-ffmpeg is importable for MP4 output."""
    try:
        import imageio_ffmpeg  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "--format mp4/both requires the optional imageio-ffmpeg "
            "dependency. Install via `pip install imageio-ffmpeg` "
            "(ships a bundled ffmpeg binary)."
        ) from exc


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

    custom_reveal = (
        _parse_reveal_custom(args.reveal_custom)
        if args.reveal_custom else None
    )

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

        reveal_schedule = None
        if args.mode == "reveal":
            reveal_schedule = _resolve_reveal_schedule(
                args.reveal_schedule, group_idxs, custom_reveal,
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
            spin_axis=args.spin_axis,
            spin_degrees=args.spin_degrees,
            ray=args.ray,
            dpi=args.dpi,
            format=args.format,
            mode=args.mode,
            reveal_schedule=reveal_schedule,
            struct_style=args.struct_style,
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
        spin_axis: str = "y",
        spin_degrees: float = 360.0,
        ray: str = "all",
        dpi: int = 300,
        format: str = "gif",
        mode: str = "spin",
        reveal_schedule: list[list[int]] | None = None,
        struct_style: str = DEFAULT_STRUCT_STYLE,
):
    struct_color = DEFAULT_STRUCT_COLOR
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
    cmd.load(pdb_path, SCAFFOLD_OBJECT_NAME)

    ref_scaffold = None
    if reference_projection is not None:
        ref_scaffold = reference_projection["structure_id"]
        ref_pdb = reference_projection.get("pdb_path")
        if ref_pdb and os.path.isfile(ref_pdb):
            cmd.load(ref_pdb, REF_SCAFFOLD_OBJECT_NAME)
        else:
            logger.warning(
                "Reference %s has no valid pdb_path; skipping alignment.",
                ref_scaffold,
            )
            ref_scaffold = None

    cmd.hide("everything", SCAFFOLD_OBJECT_NAME)
    if ref_scaffold:
        cmd.hide("everything", REF_SCAFFOLD_OBJECT_NAME)
    cmd.bg_color(DEFAULT_BG_COLOR)
    cmd.show(struct_style, SCAFFOLD_OBJECT_NAME)
    cmd.color(struct_color, SCAFFOLD_OBJECT_NAME)
    cmd.set(
        STRUCT_TRANSPARENCY_PROP[struct_style],
        1 - struct_alpha,
        SCAFFOLD_OBJECT_NAME,
    )
    cmd.zoom(complete=1)

    if mode == "reveal" and not multisector:
        # Reveal schedules are global across all groups; per-group
        # iteration would feed each call a schedule referencing groups
        # not in its own group_idxs. Force the single-render path.
        logger.info(
            "Plotting %s in reveal mode (forcing multisector path).",
            scaffold,
        )
        multisector = True

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
            spin_axis=spin_axis,
            spin_degrees=spin_degrees,
            ray=ray,
            dpi=dpi,
            format=format,
            mode=mode,
            reveal_schedule=reveal_schedule,
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
            spin_axis=spin_axis,
            spin_degrees=spin_degrees,
            ray=ray,
            dpi=dpi,
            format=format,
            mode=mode,
            reveal_schedule=reveal_schedule,
        )


def _selection_from_residues(residues):
    residues = [int(r) for r in residues]
    if not residues:
        return None
    return "resi " + "+".join(str(r) for r in residues)


def _make_safe_select(cmd, scaffold):
    """Return a wrapper around ``cmd.select`` that logs a WARNING when a
    selection matches zero atoms.

    Plumbed into the features context as ``context["select"]`` so user
    feature functions can opt into noisy failure for the most common
    brittle pattern (PDB form mismatch — segi-path selectors that match
    the biological-assembly form but not the asymmetric-unit form, or
    vice versa).
    """
    def select(name, selection, *args, **kwargs):
        n = cmd.select(name, selection, *args, **kwargs)
        if n == 0:
            logger.warning(
                "Feature selection %r matched 0 atoms on %s "
                "(selection=%r). PDB form mismatch?",
                name, scaffold, selection,
            )
        return n
    return select


def _run_feature_fns(cmd, feature_fns, scaffold, projection, group_idx, outdir):
    if not feature_fns:
        return
    context = {
        "projection": projection,
        "scaffold": scaffold,
        "group_idx": group_idx,
        "outdir": outdir,
        "select": _make_safe_select(cmd, scaffold),
    }
    for fn in feature_fns:
        fn(SCAFFOLD_OBJECT_NAME, cmd, color=None, context=context)


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
        cmd.align(SCAFFOLD_OBJECT_NAME, REF_SCAFFOLD_OBJECT_NAME)
    cmd.center()
    cmd.zoom(complete=1)


def _write_views(cmd, outdir, basename, ref_scaffold, *, dpi: int = 300):
    """Save four rotated side views (90° steps around Y) under
    ``<outdir>/views/<basename>_view{0..3}.png``."""
    viewsdir = os.path.join(outdir, "views")
    os.makedirs(viewsdir, exist_ok=True)
    for ri in range(4):
        cmd.png(os.path.join(viewsdir, f"{basename}_view{ri}.png"), dpi=dpi)
        cmd.rotate("y", 90, SCAFFOLD_OBJECT_NAME)
        if ref_scaffold:
            cmd.rotate("y", 90, REF_SCAFFOLD_OBJECT_NAME)


def _ray_sequence(mode: str, nframes: int) -> list[int]:
    if mode == "none":
        return [0] * nframes
    if mode == "first":
        return [1] + [0] * (nframes - 1)
    return [1] * nframes  # "all"


def _parse_reveal_custom(tokens: list[str]) -> list[list[int]]:
    """Parse the --reveal_custom CLI tokens into a stage schedule.

    Each token is one stage; comma-separated integers within the token
    are the IC group indices visible in that stage.
    """
    schedule: list[list[int]] = []
    for tok in tokens:
        parts = [p.strip() for p in tok.split(",")]
        if not all(parts) or not parts:
            raise ValueError(f"Empty reveal stage: {tok!r}")
        try:
            stage = [int(p) for p in parts]
        except ValueError:
            raise ValueError(
                f"Non-integer group index in reveal stage {tok!r}"
            )
        schedule.append(stage)
    return schedule


def _resolve_reveal_schedule(
        sub_mode: str,
        group_idxs: list[int],
        custom: list[list[int]] | None,
) -> list[list[int]]:
    """Build the per-stage list-of-IC-groups schedule from the user's
    sub-mode choice + the resolved group_idxs from --groups."""
    if sub_mode == "cumulative":
        return [list(group_idxs[: i + 1]) for i in range(len(group_idxs))]
    if sub_mode == "sequential":
        return [[g] for g in group_idxs]
    if sub_mode == "custom":
        if not custom:
            raise ValueError(
                "--reveal_schedule custom requires --reveal_custom"
            )
        valid = set(group_idxs)
        for stage in custom:
            invalid = [g for g in stage if g not in valid]
            if invalid:
                raise ValueError(
                    f"--reveal_custom stage {stage!r} references "
                    f"groups {invalid!r} not in --groups "
                    f"({sorted(valid)})"
                )
        return [list(stage) for stage in custom]
    raise ValueError(f"Unknown reveal sub-mode: {sub_mode!r}")


def _frames_per_stage(nframes: int, n_stages: int) -> list[int]:
    """Allocate ``nframes`` evenly across ``n_stages``; remainder is
    pinned to the last stage so the final state has a slightly longer
    hold (and the totals always sum to nframes exactly)."""
    base = nframes // n_stages
    counts = [base] * n_stages
    counts[-1] += nframes - sum(counts)
    return counts


def _compose_and_save(outdir, basename, nframes, duration, format):
    """Composite the per-frame PNGs in ``frames/<basename>_frames/`` over
    a white background and save as GIF / MP4 / both, depending on
    ``format``. Shared by spin and reveal animation paths."""
    imageio, Image = _load_animation_deps()
    # imageio's Pillow GIF writer interprets `duration` as MILLISECONDS
    # per frame; passing 0.x seconds rounds to 0 and the GIF plays at
    # the minimum-frame-time fallback (~10ms). Convert to int ms here.
    ms_per_frame = max(1, int(round(1000 * duration / nframes)))
    fps = nframes / duration
    framesdir = os.path.join(outdir, "frames", f"{basename}_frames")
    frames = []
    for i in range(nframes):
        im = Image.open(
            os.path.join(framesdir, f"frame_{i:03d}.png"),
        ).convert("RGBA")
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.getchannel("A"))
        frames.append(np.array(bg))
    if format in ("gif", "both"):
        imageio.mimsave(
            os.path.join(outdir, f"{basename}.gif"),
            frames, duration=ms_per_frame, loop=0, disposal=2,
        )
    if format in ("mp4", "both"):
        _require_ffmpeg()
        # macro_block_size=1 sidesteps ffmpeg's requirement that video
        # dimensions be multiples of 16 (which our raw PNGs rarely are).
        imageio.mimsave(
            os.path.join(outdir, f"{basename}.mp4"),
            frames, fps=fps, macro_block_size=1,
        )


def _write_animation(
        cmd, outdir, basename, nframes, duration,
        *,
        spin_axis: str = "y",
        spin_degrees: float = 360.0,
        ray: str = "all",
        dpi: int = 300,
        format: str = "gif",
):
    _load_animation_deps()  # fail-fast if imageio/PIL missing
    import tqdm as _tqdm
    framesdir = os.path.join(outdir, "frames", f"{basename}_frames")
    os.makedirs(framesdir, exist_ok=True)
    per_turn = spin_degrees / nframes
    ray_per_frame = _ray_sequence(ray, nframes)
    for i in _tqdm.trange(nframes, leave=False):
        cmd.turn(spin_axis, per_turn)
        cmd.png(
            os.path.join(framesdir, f"frame_{i:03d}.png"),
            dpi=dpi, ray=ray_per_frame[i],
        )
    _compose_and_save(outdir, basename, nframes, duration, format)


def _write_reveal_animation(
        cmd, outdir, basename,
        *,
        schedule: list[list[int]],
        projection: dict,
        sector_colors: list[str],
        sector_style: str,
        struct_color: str,
        nframes: int,
        duration: float,
        dpi: int = 300,
        ray: str = "all",
        format: str = "gif",
):
    """Sequential-reveal animation: still camera, per-stage selection
    mutation. Each stage in ``schedule`` is a list of IC group indices
    visible in that stage; frames are evenly distributed across stages
    (remainder pinned to the last stage)."""
    _load_animation_deps()  # fail-fast if imageio/PIL missing
    import tqdm as _tqdm
    framesdir = os.path.join(outdir, "frames", f"{basename}_frames")
    os.makedirs(framesdir, exist_ok=True)

    ic_pdb_residues = projection["ic_pdb_residues"]
    ic_loadings = projection["sequence_projection"]["ic_loadings"]
    counts = _frames_per_stage(nframes, len(schedule))
    ray_seq = _ray_sequence(ray, nframes)

    # Track which selections are alive in PyMOL — never call cmd.delete
    # on one that wasn't built (PyMOL raises CmdException). Cumulative
    # schedules can keep groups across stages; sequential schedules
    # rebuild every stage.
    alive: set[int] = set()
    frame_idx = 0
    pbar = _tqdm.tqdm(total=nframes, leave=False)
    try:
        for stage_idx, stage_groups in enumerate(schedule):
            stage_set = set(stage_groups)
            for gidx in list(alive - stage_set):
                sel = f"reveal_sel{gidx}"
                cmd.hide(sector_style, sel)
                cmd.color(struct_color, sel)
                cmd.delete(sel)
                alive.discard(gidx)
            for gidx in stage_groups:
                if gidx in alive:
                    continue
                sector_color = sector_colors[gidx % len(sector_colors)]
                pdb_resids = ic_pdb_residues[gidx]
                scores = (
                    ic_loadings[gidx] if gidx < len(ic_loadings) else []
                )
                if _apply_group_coloring(
                    cmd, f"reveal_sel{gidx}",
                    pdb_resids, scores, sector_color, sector_style,
                ) is not None:
                    alive.add(gidx)
            for _ in range(counts[stage_idx]):
                cmd.png(
                    os.path.join(framesdir, f"frame_{frame_idx:03d}.png"),
                    dpi=dpi, ray=ray_seq[frame_idx],
                )
                frame_idx += 1
                pbar.update(1)
    finally:
        pbar.close()
        for gidx in list(alive):
            sel = f"reveal_sel{gidx}"
            cmd.hide(sector_style, sel)
            cmd.color(struct_color, sel)
            cmd.delete(sel)
            alive.discard(gidx)

    _compose_and_save(outdir, basename, nframes, duration, format)


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
        spin_axis: str = "y",
        spin_degrees: float = 360.0,
        ray: str = "all",
        dpi: int = 300,
        format: str = "gif",
        mode: str = "spin",
        reveal_schedule: list[list[int]] | None = None,
):
    """Render one frame with the given set of IC groups lit up.

    Collects per-group selections, aligns + focuses, invokes feature
    functions, writes the main PNG, optionally writes rotated views and
    a rotating-GIF animation, and cleans up every selection it created.
    Per-group rendering loops over this once per ``gidx`` (passing the
    ``gidx`` through to feature fns); multisector rendering calls it
    once with all ``group_idxs`` and ``group_idx_for_features=None``.

    When ``mode='reveal'``, the animation step uses
    :func:`_write_reveal_animation` (still camera, per-stage selection
    mutation) instead of :func:`_write_animation` (rotating spin), and
    the still selections are torn down before the reveal so they don't
    leak into the animation frames.
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

    cmd.png(f"{outdir}/{basename}.png", dpi=dpi)

    if views:
        _write_views(cmd, outdir, basename, ref_scaffold, dpi=dpi)

    if animate:
        if mode == "reveal":
            if reveal_schedule is None:
                raise ValueError(
                    "_render_frame mode='reveal' requires reveal_schedule"
                )
            # Tear down still selections so the reveal starts on an
            # empty canvas; _write_reveal_animation manages its own
            # reveal_sel<gidx> namespace.
            for sel_name in created:
                cmd.hide(sector_style, sel_name)
                cmd.color(struct_color, sel_name)
                cmd.delete(sel_name)
            created.clear()
            _write_reveal_animation(
                cmd, outdir, basename,
                schedule=reveal_schedule,
                projection=projection,
                sector_colors=sector_colors,
                sector_style=sector_style,
                struct_color=struct_color,
                nframes=nframes, duration=duration,
                dpi=dpi, ray=ray, format=format,
            )
        else:
            _write_animation(
                cmd, outdir, basename, nframes, duration,
                spin_axis=spin_axis, spin_degrees=spin_degrees,
                ray=ray, dpi=dpi, format=format,
            )

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
        spin_axis: str = "y",
        spin_degrees: float = 360.0,
        ray: str = "all",
        dpi: int = 300,
        format: str = "gif",
        mode: str = "spin",
        reveal_schedule: list[list[int]] | None = None,
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
            spin_axis=spin_axis,
            spin_degrees=spin_degrees,
            ray=ray,
            dpi=dpi,
            format=format,
            mode=mode,
            reveal_schedule=reveal_schedule,
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
        spin_axis: str = "y",
        spin_degrees: float = 360.0,
        ray: str = "all",
        dpi: int = 300,
        format: str = "gif",
        mode: str = "spin",
        reveal_schedule: list[list[int]] | None = None,
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
        spin_axis=spin_axis,
        spin_degrees=spin_degrees,
        ray=ray,
        dpi=dpi,
        format=format,
        mode=mode,
        reveal_schedule=reveal_schedule,
    )


def _hex2color(x):
    return "0x" + x[1:]


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
