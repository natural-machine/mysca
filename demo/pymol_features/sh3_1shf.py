"""Example sca-pymol features file for 1SHF (human Fyn SH3 domain).

Unlike NarG (1Q16), the SH3 PDBs in the demo (1SHF, 2ABL) are *apo* — no
small-molecule cofactor is bound. The closest functional-relevance
features for SH3 are the **canonical proline-rich-peptide binding
surface**: a hydrophobic pocket composed of conserved aromatic
residues, plus three loops (RT, n-Src, distal) that flank it. SCA
sectors for SH3 typically map onto these elements; the features below
let you visualize them alongside the IC-colored sector residues.

Each function matches the sca-pymol features contract:

    def feature_fn(struct, cmd, *, color=None, context=None) -> None

See ``demo/pymol_features/narg_1q16.py`` (and
``docs/cli_reference.md``: Features plugin > Authoring guidance) for
the broader rationale on selector form and the ``context`` dict.

Residue numbering below is **1SHF chain A** (human Fyn, residues
84–141 in UniProt P06241). For 2ABL (Abl SH3 fragment) see the
sibling file ``sh3_2abl.py``.

Invoke via::

    sca-pymol --structure <dir> --structure_id 1SHF \\
        --features_py demo/pymol_features/sh3_1shf.py \\
        --features show_pxxp_pocket,show_rt_loop,show_n_src_loop \\
        -o <outdir>
"""


def show_pxxp_pocket(struct, cmd, *, color=None, context=None):
    """Render the canonical hydrophobic pocket that grips a bound
    proline-rich peptide. Conserved aromatic + Pro residues on the
    β-barrel surface (Fyn numbering: Y91, F103, W119, P134, Y137)."""
    select = context["select"] if context else cmd.select
    select("pxxp_pocket", "chain A and resi 91+103+119+134+137")
    cmd.show("sticks", "pxxp_pocket")
    if isinstance(color, str):
        cmd.color(color, "pxxp_pocket")


def show_rt_loop(struct, cmd, *, color=None, context=None):
    """Render the RT loop (between β1 and β2; Fyn residues 92–101).
    Most variable SH3 loop — modulates ligand specificity."""
    select = context["select"] if context else cmd.select
    select("rt_loop", "chain A and resi 92-101")
    cmd.show("cartoon", "rt_loop")
    cmd.set("cartoon_loop_radius", 0.4, "rt_loop")
    if isinstance(color, str):
        cmd.color(color, "rt_loop")


def show_n_src_loop(struct, cmd, *, color=None, context=None):
    """Render the n-Src loop (between β3 and β4; Fyn residues 113–117).
    Together with the RT loop, frames the peptide-binding groove."""
    select = context["select"] if context else cmd.select
    select("n_src_loop", "chain A and resi 113-117")
    cmd.show("cartoon", "n_src_loop")
    cmd.set("cartoon_loop_radius", 0.4, "n_src_loop")
    if isinstance(color, str):
        cmd.color(color, "n_src_loop")


def show_distal_loop(struct, cmd, *, color=None, context=None):
    """Render the distal loop (between β4 and 3₁₀; Fyn residues
    124–128). The third loop bordering the peptide-binding face."""
    select = context["select"] if context else cmd.select
    select("distal_loop", "chain A and resi 124-128")
    cmd.show("cartoon", "distal_loop")
    cmd.set("cartoon_loop_radius", 0.4, "distal_loop")
    if isinstance(color, str):
        cmd.color(color, "distal_loop")


def show_specificity_tyr(struct, cmd, *, color=None, context=None):
    """Render the conserved 'specificity tyrosine' (Fyn Y137).
    Distinguishes class I vs class II PXXP-peptide orientations and
    tends to anchor sector residues in SCA decompositions."""
    select = context["select"] if context else cmd.select
    select("spec_tyr", "chain A and resi 137 and resn TYR")
    cmd.show("sticks", "spec_tyr")
    if isinstance(color, str):
        cmd.color(color, "spec_tyr")
