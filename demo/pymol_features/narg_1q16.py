"""Example sca-pymol features file for 1Q16 NarG alpha subunit.

Each function must match the sca-pymol features contract:

    def feature_fn(struct, cmd, *, color=None, context=None) -> None

- ``struct``: PyMOL object name of the loaded structure (always
  ``"struct"`` for the primary scaffold in the current implementation).
- ``cmd``: PyMOL's ``cmd`` module, injected by sca-pymol so this file
  does not need its own ``from pymol import cmd``.
- ``color``: optional per-feature color (unused in this commit; plumbed
  for a future ``--features_color`` flag).
- ``context``: dict with at least ``projection`` (the loaded per-
  structure dict from ``structure_projection.json``), ``scaffold``
  (same as ``struct``), ``group_idx`` (IC index for the current render
  pass, or ``None`` under --multisector), and ``outdir`` (where the
  PNG will be written).

Invoke via::

    sca-pymol --structure <dir> \\
        --features_py demo/pymol_features/narg_1q16.py \\
        --features show_molybdenum,show_sf4_cluster,show_mgd \\
        -o <outdir>
"""


def show_molybdenum(struct, cmd, *, color=None, context=None):
    """Render the active-site molybdenum of NarG as a sphere."""
    cmd.select("mo", f"{struct}/F/A/6MO`1302/MO")
    cmd.show("everything", "mo")
    if isinstance(color, str):
        cmd.color(color, "mo")


def show_sf4_cluster(struct, cmd, *, color=None, context=None):
    """Render the proximal [4Fe-4S] cluster as spheres."""
    cmd.select("sf4", f"{struct}/G/A/SF4`1401/*")
    cmd.show("everything", "sf4")
    if isinstance(color, str):
        cmd.color(color, "sf4")


def show_mgd(struct, cmd, *, color=None, context=None):
    """Render the molybdopterin guanine dinucleotide cofactors."""
    cmd.select(
        "cofactor",
        f"{struct}/D/A/MD1`1300/* {struct}/E/A/MD1`1301/*",
    )
    cmd.show("sticks", "cofactor")
    if isinstance(color, str):
        cmd.color(color, "cofactor")
