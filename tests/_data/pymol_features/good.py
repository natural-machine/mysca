"""Test fixture: minimal user features file.

Two no-op callables + one non-callable attribute, used by
tests/test_entrypoint_pymol.py to exercise the features loader's
happy path and its non-callable rejection path.
"""


def feature_a(struct, cmd, *, color=None, context=None):
    cmd.select("feature_a_sel", f"{struct}/*")
    cmd.show("spheres", "feature_a_sel")


def feature_b(struct, cmd, *, color=None, context=None):
    cmd.select("feature_b_sel", f"{struct}/resi 1")
    cmd.color("red", "feature_b_sel")


NOT_CALLABLE = 42
