# 2026-05-05 — Audit-plan rollout: steps 1–6 + 8 (dropped 7), plus §8b.1 follow-up

## Summary

Implementation of the audit plan at
[.claude/plans/i-want-to-audit-calm-harp.md](../../.claude/plans/i-want-to-audit-calm-harp.md).
Worked through §9 rollout steps 1 through 6, walked back one logging
change in §8b.1, dropped step 7 on user direction, then implemented
step 8. Step 9 (structure-based IC-group connectivity / new
`sca-evaluate` entrypoint) and §7 Phase 1 item 2 (header-pattern
parser) remain documented in the plan file but unimplemented.

Test suite: **1112 → 1131 passing** across the rollout. The
single GPU-flake (`test_compute_fijab_gpu_precision_match`) on this
machine's GB10 (CUDA cap 12.1, above PyTorch's declared 12.0 support)
is environmental and unrelated; passes in isolation.

## Commits (oldest → newest)

| SHA | Subject | Notes |
|---|---|---|
| `8f2e0df` | Docs pass: README quickstart + three-surface argparse/docstring fix-ups | §9 steps 1+2 |
| `d185246` | Defensive guards: missing reference, empty filter result, missing eigendecomp | §9 step 3 (B1, B5, B8) + 4 new tests |
| `128e4b4` | Logging upgrades: fi0 warnings, regularization params, ICA retries, filter totals | §9 step 4 |
| `56a113c` | Per-IC projection quality metrics on SequenceProjection | §9 step 5: `gap_fraction_per_ic`, `informative_positions_per_ic` |
| `9317798` | Walk back fi0 ≈ 0 WARNING to a DEBUG breadcrumb | §8b.1 follow-up to `128e4b4` |
| `063a34e` | Preserve FASTA header descriptions through preprocessing | §9 step 6 / Deliverable A |
| `21084f6` | sca-project: plumb --seq_metadata + sequence_metadata.tsv | §9 step 6 / Deliverable B1 |
| `25f5024` | sca-structure: pass --seq_metadata through to underlying sca-project | §9 step 6 / Deliverable B2 |
| `faebb4e` | sca-project: emit projection plot with metadata coloring | §9 step 8 |
| `f671a3a` | preprocess: defer dense one-hot to post-filter to fix OOM on large MSAs | Out-of-band fix by user; documented in its own session note. |

## Notable design decisions

### §9 step 5 (per-IC projection quality metrics)
- `gap_fraction_per_ic[i] = 1 - len(ic_processed_cols[i]) / len(sca.ic_positions[i])` —
  denominator is the IC's full **training-time** support, not the
  per-projection filtered support, so a fully-gapped projection
  doesn't blow up to 0/0.
- `informative_positions_per_ic[i]` uses the strict
  `xmsa[m, j, :].any()` semantic (positions that *actually* contribute
  non-zero mass to the Uᵖ math), not just the non-gap count — so
  non-canonical residues are correctly counted as non-informative.
- Metric arithmetic factored into the testable
  `mysca.project.projection._per_ic_quality_metrics` helper; one unit
  test on a hand-built xmsa, one end-to-end smoke test on the
  synthetic fixture.
- Surfaced in `seq_projections.tsv` as per-IC columns
  (`gap_frac_ic_*`, `n_inform_ic_*`); not added to sca-core's
  in-sample TSV by design (sca-core's in-sample sequences come
  directly from the preprocessed MSA, so the metrics would be a
  permutation of `prep.msa`'s already-known gap pattern).

### §8b.1 (fi0 walk-back)
The audit's original §B2 framing — "bump fi0 ≈ 0 to WARNING" — was
incorrect. `fi0` is the weighted *gap frequency* per column, so
`fi0 ≈ 0` is the normal state of any well-conserved column on a
clean MSA. Commit `128e4b4` shipped the WARNING based on that framing
and immediately spammed every healthy SH3 run with multi-line
warnings on 9 fully-occupied positions per bootstrap iteration.

The downstream `Di` term (the only consumer of `fi0` in `run_sca`) is
already guarded with `np.where(fi0 > 0, np.log(fi0/q0hat), 0)` under
`np.errstate('ignore')` — `fi0=0` contributes exactly 0. No NaN risk.

`9317798` reverted to DEBUG-level with a count-only message, dropped
the `# TODO: handle this` comment (the case is correctly handled), and
added an inline comment at the detection site documenting why this is
intentionally a quiet diagnostic. Plan file §8b.1 captures the
rationale and explicitly marks the original framing withdrawn.

### §9 step 6 / Deliverable A (FASTA descriptions)
- `load_msa` return tuple grew from 6 → 7 elements. **Backward-incompatible**
  for any external code unpacking positionally. All 11 in-tree
  positional callers updated.
- `preprocess_msa` gained an optional `seq_descriptions=None` kwarg; threading
  follows the same idiom as `seqids` at every filter site that rebuilds the
  ID list. None-in-None-out contract.
- `PreprocessingResults.retained_sequence_descriptions` is persisted only when
  present, and the loader uses `data.files`-guarded access so legacy NPZ
  bundles still load with the field defaulting to `None`.

### §9 step 6 / Deliverable B (—seq_metadata in projection)
- Metadata stays out of `projection.json` / `structure_projection.json`;
  TSV is the right home for tabular metadata. Mirrors sca-core.
- `sca-structure --seq_metadata` is a transparent pass-through:
  forwarded to every per-PDB `project_pdb()` call AND copied verbatim
  to `<outdir>/sequence_metadata.tsv` once at the entrypoint level.
  One TSV covers an entire batch.

### §9 step 7 (NCBI taxid resolver)
Dropped on user direction. Users can supply taxonomy via the
`--seq_metadata` sidecar TSV that step 6 plumbed through; an in-tree
NCBI taxid resolver isn't worth the new module + sqlite cache +
network-dependency surface unless a concrete need surfaces.

### §9 step 8 (sca-project plotting + generalized coloring)
After a brief detour through a sca-plots-only design, settled on
**Option A**: sca-project owns its own plotting (mirrors sca-core's
`--plot`/`--no-plot` pattern). Sca-plots is left untouched.
- New flags: `--plot`/`--no-plot`, `--seq_proj_axes` (custom argparse
  type for `i,j` tokens), `--seq_proj_color_by`.
- Output: `<outdir>/images/seq_proj_ic{i}v{j}[_by_<col>].png` per axis
  pair.
- The `--seq_proj_color_by` mechanism is metadata-column-agnostic —
  numeric → colorbar, categorical → legend. The "taxonomy-colored"
  framing in the audit was always about the wiring, not about
  taxonomy specifically.
- `_resolve_color_values` extracted from `run_sca.py` to a new shared
  module `mysca.pl._coloring` so all three callers (run_sca,
  run_plots, run_project) import the same function.

## Process feedback the user gave (saved to memory)

- **Don't reference rollout-plan step numbers in code or test
  comments.** Section dividers are fine, but the labels inside should
  describe what the code does, not the plan step that produced them.
  (Saved at
  `/home/ahowe/.claude/projects/-home-ahowe-Projects-mysca/memory/feedback_no_rollout_labels_in_code.md`.)
  All offending labels in commits `56a113c`, `063a34e`, `21084f6`
  were rewritten in `25f5024`.

## Out of scope / deferred

- **§9 step 9** — structure-based IC-group connectivity (new
  `sca-evaluate` entrypoint, contact-graph metrics, permutation null,
  three aggregation modes). Designed in plan file §7b; deferred per
  user direction.
- **§7 Phase 1 item 2** — optional header-pattern parser
  (`mysca.metadata.headers` for UniProt / NCBI / GenBank header
  patterns). Marked **PENDING USER REVIEW** in plan file §9c. The
  foundation (Deliverable A: descriptions are now preserved) is in
  place to add it whenever the call comes.
- **§9 step 10 / larger advances** — taxonomy-aware sequence
  weighting, per-clade analysis, supervised PLS/LDA on top of ICs,
  GPU/streaming projection. Each remains a self-contained design
  proposal in the plan file.

## Test suite trajectory

| After commit | Passing | New tests |
|---|---|---|
| Pre-rollout | 1112 | — |
| `8f2e0df` | 1112 | — (doc-only) |
| `d185246` | 1116 | 4 (B1, B5a, B5b, B8) |
| `128e4b4` | 1116 | — (logging-only) |
| `56a113c` | 1118 | 2 (per-IC quality metrics: helper unit + in-sample full coverage) |
| `9317798` | 1118 | — (single-line revert) |
| `063a34e` | 1122 | 4 (descriptor preservation: io, preprocess, npz round-trip, legacy bundle) |
| `21084f6` | 1124 | 2 (sca-project metadata round-trip + missing-column error) |
| `25f5024` | 1126 | 2 (sca-structure metadata pass-through + missing-column error) |
| `faebb4e` | 1131 | 5 (sca-project plotting: default, --no-plot, axes, color-by, missing-metadata) |
