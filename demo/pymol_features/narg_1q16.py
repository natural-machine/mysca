"""Example sca-pymol features file for 1Q16 NarG alpha subunit.

Each function must match the sca-pymol features contract:

    def feature_fn(struct, cmd, *, color=None, context=None) -> None

- ``struct``: PyMOL object name of the loaded structure. Always the
  literal ``"struct"`` (matches ``SCAFFOLD_OBJECT_NAME`` in
  ``mysca.run_pymol``).
- ``cmd``: PyMOL's ``cmd`` module, injected by sca-pymol so this file
  does not need its own ``from pymol import cmd``.
- ``color``: optional per-feature color (unused in this commit; plumbed
  for a future ``--features_color`` flag).
- ``context``: dict with at least ``projection`` (the loaded per-
  structure dict from ``structure_projection.json``), ``scaffold``
  (the structure_id, e.g. ``"1Q16"``), ``group_idx`` (IC index for
  the current render pass, or ``None`` under --multisector), ``outdir``
  (where the PNG will be written), and ``select`` (a noisy
  ``cmd.select`` wrapper that logs a WARNING on zero-match
  selections).

The selectors below use attribute-based syntax (``resn``, ``resi``,
``name``) so they match regardless of which 1Q16 form is loaded —
RCSB asymmetric-unit, RCSB biological-assembly, and PDBe-updated
mmCIF all expose the cofactors under different chain/segi layouts but
share the same residue names + numbers. See
``docs/cli_reference.md`` (Features plugin > Authoring guidance) for
the broader rationale.

Invoke via::

    sca-pymol --structure <dir> \\
        --features_py demo/pymol_features/narg_1q16.py \\
        --features show_molybdenum,show_sf4_cluster,show_mgd \\
        -o <outdir>
"""


def show_molybdenum(struct, cmd, *, color=None, context=None):
    """Render the active-site molybdenum of NarG as a sphere."""
    select = context["select"] if context else cmd.select
    select("mo", "resn 6MO and resi 1302 and name MO")
    cmd.show("everything", "mo")
    if isinstance(color, str):
        cmd.color(color, "mo")


def show_sf4_cluster(struct, cmd, *, color=None, context=None):
    """Render the proximal [4Fe-4S] cluster as spheres."""
    select = context["select"] if context else cmd.select
    select("sf4", "resn SF4 and resi 1401")
    cmd.show("everything", "sf4")
    if isinstance(color, str):
        cmd.color(color, "sf4")


def show_mgd(struct, cmd, *, color=None, context=None):
    """Render the molybdopterin guanine dinucleotide cofactors."""
    select = context["select"] if context else cmd.select
    select("cofactor", "resn MD1 and resi 1300+1301")
    cmd.show("sticks", "cofactor")
    if isinstance(color, str):
        cmd.color(color, "cofactor")
