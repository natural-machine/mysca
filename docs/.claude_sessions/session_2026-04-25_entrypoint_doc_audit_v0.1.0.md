# 2026-04-25 — entrypoint documentation audit + 0.1.0 version bump

## Summary

A focused audit pass across all seven CLI entrypoints (`sca-prealign`,
`sca-preprocess`, `sca-core`, `sca-project`, `sca-structure`,
`sca-pymol`, `sca-plots`) to confirm that each entrypoint's argparse
surface, top-of-file docstring, [docs/cli_reference.md](../cli_reference.md),
and the [README.md](../../README.md) are mutually consistent after the
recent dense run of feature work.

Found one stale-doc bug (`hmmalign` described as not implemented),
several README gaps (missing `clustalo`, missing pip extras, missing
inputs/outputs summaries per entrypoint), and otherwise green.
Followed up with a 0.1.0 version bump to flag the breaking on-disk
rename arc that landed in the prior session
([statsectors → ic_residues_per_seq](session_2026-04-25_pymol_animation_glossary_rename.md)).

## Pre-session state

```sh
git checkout a8a6b28
```

Branch `addison-dev`, 7 commits ahead of `origin/addison-dev`. Tip
commit is the doc note for the previous session.

## Audit pass

For each of the seven entrypoints, cross-checked four surfaces:

1. `argparse` definitions in `src/mysca/run_*.py`.
2. The module-level docstring (`COMMAND LINE ARGUMENTS:` /
   `OUTPUTS:` blocks).
3. The matching section in [docs/cli_reference.md](../cli_reference.md).
4. The relevant Usage subsection in [README.md](../../README.md).

### Clean (no changes needed)

- **`sca-prealign`** — argparse, docstring, cli_reference all
  consistent after the `--align_args` refactor and clustalo addition.
- **`sca-preprocess`** — output table in cli_reference matches what
  `PreprocessingResults.save()` writes; argparse help strings agree.
- **`sca-core`** — outputs table reflects the post-rename layout
  (`ic_residues_per_seq.npz` / `ic_loadings_per_seq.npz` /
  `ic_positions/`); `--sectors_for` help wording agrees with the
  Phase C tightening (`82d2007`).
- **`sca-structure`** — three input modes documented end-to-end
  (`-s` / `--seq_map` / `--uniprot_ids`), `--cache_dir` default
  consistent across surfaces.
- **`sca-pymol`** — argparse and docstring describe the full
  animation surface added this past arc (`--mode {spin,reveal}`,
  `--reveal_schedule`, `--reveal_custom`, `--format {gif,mp4,both}`,
  `--spin_axis`/`--spin_degrees`/`--ray`/`--dpi`).
- **`sca-plots`** — flags, replay matrix, and the `Cij_raw`
  back-compat note all agree across surfaces.

### Stale doc fixed

- **`sca-project`** — module docstring + the help text on
  `--aligner` / `--align_bin` / `--align_threads` claimed
  `hmmalign` was *"reserved as a name but not yet implemented"*.
  But [src/mysca/project/alignment.py](../../src/mysca/project/alignment.py)
  has had a fully working `_hmmalign` for several sessions, and
  cli_reference describes it accurately. Same stale claim was also
  in [src/mysca/project/__init__.py](../../src/mysca/project/__init__.py).
  Both fixed.

### Why the inconsistency wasn't caught earlier

The hmmalign-status doc string was correct when the aligner was a
TODO; the implementation landed in a session that didn't touch the
docstring. No automated check correlates argparse `choices` against
the help-text descriptions, so this kind of "the choice exists but
the help still calls it unimplemented" drift slips through.

## README gaps closed

[README.md](../../README.md) was missing three things:

1. **`clustalo` in the optional-binaries matrix.** Added (the
   aligner was integrated in `1504830` but the README never picked
   it up).
2. **Pip extras `[pymol]` and `[mp4]`.** New "Optional Python
   extras" section explaining each extra and when to use it. The
   `[mp4]` extra is currently the canonical way to enable
   `sca-pymol --format mp4`.
3. **Inputs/Outputs per entrypoint.** Each Usage subsection now
   carries a one-paragraph **Inputs:** + **Outputs:** summary so
   readers don't have to bounce to cli_reference for the basics.
   Full per-flag tables and exhaustive output lists continue to
   live in cli_reference.

## Verification

```sh
env/bin/python -m pytest tests/
```

End of audit: **1056 passed**, 5 skipped (binary-gated as
before — same skip set as session-end of `2026-04-25` arc).

## 0.1.0 version bump

Bumped `mysca.__version__` from `0.0.2` → `0.1.0`.

The minor bump (rather than 0.0.3) reflects that the prior arc
introduced **breaking on-disk changes**: `statsectors_msa.npz` /
`statsectors_seq.npz` were replaced by `ic_residues_per_seq.npz`
+ `ic_loadings_per_seq.npz`; the `sca_results/msa_sectors/` per-IC
score files were removed; `SCAResults.groups` /
`SCAResults.statsectors_msa` etc. were renamed in-memory. Old
`sca-core` output directories cannot be loaded by this version
without re-running `sca-core` against the same preprocessing
output. That's a meaningful enough migration to warrant signaling
in semver, even at the 0.x level.

No deprecation shim was added because the on-disk rename was the
explicit point of the change; users on 0.0.2 outputs simply rerun
the SCA-core step (preprocessing outputs are unchanged).

## Notes for next session

- **Test coverage gap**: there's no test that the help string of
  any flag stays consistent with the underlying behavior — the
  hmmalign drift bug went undetected for several sessions. Could
  be worth a parser-level test that diffs the rendered `--help`
  against a checked-in golden, or just a less-fragile contract
  test that walks each entry in `ALIGNERS` and asserts the help
  text doesn't say "not yet implemented" / "reserved" / etc.
  Low-priority but trivial.
- **README ↔ cli_reference duplication**: with the new
  Inputs/Outputs blocks, both files now describe outputs at
  different granularity. The README is intentionally lossy
  (one-paragraph) and cli_reference is exhaustive; if either
  drifts again, the README block is the cheaper one to refresh.
- **`environment.yml` was already correct**: clustalo was already
  in the commented-optional list, and the `[mp4]` pip extra was
  already noted in a comment block. Only the README lagged.
