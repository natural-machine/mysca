"""Example sca-pymol features file for 2ABL (human Abl SH3-SH2 fragment).

Like 1SHF, 2ABL is *apo* — no small-molecule cofactor is bound (only
crystallographic waters). The features below highlight the canonical
proline-rich-peptide binding surface of the SH3 domain (the SH2
domain is also present in this PDB but not annotated here).

See the sibling file ``sh3_1shf.py`` for the rationale; the only
substantive difference between the two is the residue numbering.
2ABL chain A residue numbering corresponds to UniProt P00519
(human Abl1) with the SH3 domain spanning ~residues 64–117.

Invoke via::

    sca-pymol --structure <dir> --structure_id 2ABL \\
        --features_py demo/pymol_features/sh3_2abl.py \\
        --features show_pxxp_pocket,show_rt_loop,show_n_src_loop \\
        -o <outdir>
"""


def show_pxxp_pocket(struct, cmd, *, color=None, context=None):
    """Render the canonical hydrophobic pocket that grips a bound
    proline-rich peptide. Conserved aromatic + Pro residues on the
    β-barrel surface (Abl numbering: Y89, F91, W118, P131, Y134)."""
    select = context["select"] if context else cmd.select
    select("pxxp_pocket", "chain A and resi 89+91+118+131+134")
    cmd.show("sticks", "pxxp_pocket")
    if isinstance(color, str):
        cmd.color(color, "pxxp_pocket")


def show_rt_loop(struct, cmd, *, color=None, context=None):
    """Render the RT loop (between β1 and β2; Abl residues 90–100).
    Most variable SH3 loop — modulates ligand specificity."""
    select = context["select"] if context else cmd.select
    select("rt_loop", "chain A and resi 90-100")
    cmd.show("cartoon", "rt_loop")
    cmd.set("cartoon_loop_radius", 0.4, "rt_loop")
    if isinstance(color, str):
        cmd.color(color, "rt_loop")


def show_n_src_loop(struct, cmd, *, color=None, context=None):
    """Render the n-Src loop (between β3 and β4; Abl residues 112–117).
    Together with the RT loop, frames the peptide-binding groove."""
    select = context["select"] if context else cmd.select
    select("n_src_loop", "chain A and resi 112-117")
    cmd.show("cartoon", "n_src_loop")
    cmd.set("cartoon_loop_radius", 0.4, "n_src_loop")
    if isinstance(color, str):
        cmd.color(color, "n_src_loop")


def show_distal_loop(struct, cmd, *, color=None, context=None):
    """Render the distal loop (between β4 and 3₁₀; Abl residues
    123–127). The third loop bordering the peptide-binding face."""
    select = context["select"] if context else cmd.select
    select("distal_loop", "chain A and resi 123-127")
    cmd.show("cartoon", "distal_loop")
    cmd.set("cartoon_loop_radius", 0.4, "distal_loop")
    if isinstance(color, str):
        cmd.color(color, "distal_loop")


def show_specificity_tyr(struct, cmd, *, color=None, context=None):
    """Render the conserved 'specificity tyrosine' (Abl Y134).
    Distinguishes class I vs class II PXXP-peptide orientations and
    tends to anchor sector residues in SCA decompositions."""
    select = context["select"] if context else cmd.select
    select("spec_tyr", "chain A and resi 134 and resn TYR")
    cmd.show("sticks", "spec_tyr")
    if isinstance(color, str):
        cmd.color(color, "spec_tyr")
