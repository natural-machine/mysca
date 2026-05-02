# 2026-05-02 — Sequence-space projection (Uᵖ): library + DataFrame + metadata + GPU precision

## Summary

Built out the canonical Rivoire et al. (2016) sequence-position
mapping (Eqs. 14, 15) as a first-class capability in `mysca`, then
layered metadata, DataFrame export, GPU `--precision`, SH3 PyMOL
features, and a stop-codon / `_loaded` rename audit on top.

Result: three commits on `addison-dev`:

- **`2686ac8`** — Add sequence-space projection (Uᵖ), metadata,
  DataFrame export, GPU `--precision` (14 files, +1108/-42).
- **`ccd6ab4`** — Add SH3 PyMOL features files (apo PXXP binding
  surface) (3 new files, +197).
- **`04aa385`** — Stop-codon handling, `_loaded` rename, `Up_*`
  DataFrame columns, filter-history audit (24 files, +491/-171).

Test suite: **1112 passing, 1 skipped** (+22 new tests this session).

## Reading this session against the paper

Rivoire et al. (2016) (the PDF was placed in the repo root mid-session
for reference) gives the canonical math in Box 1, Eqs. 4 setup and
14–15:

```
P̃ᵢᵃ  = φᵢᵃ fᵢᵃ / (Σ_b (φᵢᵇ fᵢᵇ)²)^(1/2)        # per-position normalization
xₛᵢ   = Σ_a P̃ᵢᵃ xₛᵢᵃ                           # M × L × D → M × L
Ũ     = x V̌ Λ̃^(-1/2)                          # Eq. 14
Uᵖ    = W Ũ                                    # Eq. 15
```

`figure_scripts/gen_fig5.py` (placed by the user mid-session) used a
*global* normalization `Pia = phi_ia * fia / np.sqrt(np.sum(...))`
without the `axis=-1` restriction. That diverges from the paper.
Confirmed via reading; the new `SCAResults.project_sequences` uses
the per-position form. The legacy figure scripts were patched in
place to match (see "Cleanup" below) but otherwise left unmodified
since they're user-placed and out of scope.

## Major workstream — sequence-space projection (Uᵖ)

### 1. `SCAResults.project_sequences(xmsa)` — the load-bearing addition

[`src/mysca/results.py:638–700`](../../src/mysca/results.py#L638-L700)

- Takes a one-hot tensor of shape `(M, L_proc, D)`.
- Returns Uᵖ, shape `(M, n_components)`.
- **Zero new persisted fields.** Every operand (`phi_ia`, `fia`,
  `evecs_sca`, `evals_sca`, `w_ica`) is already on `SCAResults`.
- Bug fix folded in: when `--n_components > kstar` (e.g.
  `--n_components all`), `w_ica` is sized to `n_components`, not
  `kstar`. The method picks `evecs_sca[:, :w_ica.shape[0]]` and
  `evals_sca[:w_ica.shape[0]]`, which is correct in both cases. The
  obvious-but-wrong implementation using `significant_evecs_sca`
  (only `kstar` columns) caused
  `test_n_components_all_runs_ica_on_all_eigenvectors` to fail on
  first attempt — kept as a regression sentinel.

### 2. Plot wired into sca-core and sca-plots replay

- `plot_seq_projection_2d(up, axidxs, outdir, *, color_values, color_label)`
  in [`src/mysca/pl/plotting.py`](../../src/mysca/pl/plotting.py).
- Default render: scatter of `Uᵖ[:,0]` vs `Uᵖ[:,1]` →
  `seq_proj_ic0v1.png`.
- Supports optional `color_values` (numeric → colorbar; categorical
  → discrete legend with NA in grey).
- `sca-core` computes `up_seq = results.project_sequences(prep.msa_binary3d)`
  inline before `make_plots`.
- `sca-plots` `_replay_scacore` loads `prep.msa_binary3d` (from
  `--preprocessing`) and recomputes Uᵖ at replay time.

### 3. Out-of-sample Uᵖ in `sca-project`

- `SequenceProjection.up_score` (length `n_components`).
- `ProjectionResult.up_scores` (`M × n_components` numpy array).
- `_aligned_to_xmsa()` helper in
  [`src/mysca/project/projection.py`](../../src/mysca/project/projection.py)
  builds the one-hot tensor uniformly from aligned-sequence strings
  for both in-sample and out-of-sample paths. Best-effort: warns and
  skips silently when `SCAResults` lacks the eigendecomposition
  fields.

### 4. DataFrame exports

- `SCAResults.to_dataframe(prep)` — in-sample table (`seq_id`,
  `aligned_sequence`, `Up_0..Up_{k-1}`, plus any merged metadata
  columns).
- `ProjectionResult.to_dataframe()` — out-of-sample table (`seq_id`,
  `aligned_sequence`, `raw_sequence`, `in_sample`, `Up_0..Up_{k-1}`).
- `--save_dataframe` flag on both `sca-core` and `sca-project` writes
  `seq_projections.tsv`.
- **Late-session rename**: lowercase `up_*` → capital `Up_*` (matches
  the Uᵖ notation; user request).

### 5. Per-sequence metadata: `--seq_metadata <tsv>` + colored plot

- `SCAResults.sequence_metadata` (pandas DataFrame field).
- Persisted as `sequence_metadata.tsv`; round-trips through
  `SCAResults.save/load`.
- `to_dataframe(prep)` left-joins on `seq_id`.
- `--seq_proj_color_by COLUMN` on `sca-core` and `sca-plots` colors
  the seq_proj plot by a metadata column.
- Demoed end-to-end: synthesized a fake taxonomy TSV for the SH3
  demo's 7861 retained sequences and rendered
  `seq_proj_ic0v1_by_kingdom.png` with a categorical legend.

### 6. GPU `--precision {fp64, fp32, fp16}`

[`src/mysca/_acceleration.py:resolve_torch_dtype`](../../src/mysca/_acceleration.py)

- Threads through `_compute_fijab_gpu` and
  `_compute_eigvalsh_bootstrap_gpu` only — CPU paths ignore it.
- fp16 auto-promotes the eigendecomposition to fp32 inside the
  bootstrap kernel (eigvalsh is unstable in fp16; unsupported on most
  backends). Still useful for the cheap fijab/Cij_corr ops.
- Output always cast back to fp64 for downstream consistency.
- Verified: SH3 demo run with `--accelerator gpu --precision fp32`
  completes cleanly and persists `precision: "fp32"` in
  `scarun_args.json`.

## Side workstream — SH3 PyMOL features

User asked whether the SH3 demo PDBs (1SHF Fyn, 2ABL Abl) carry a
bound cofactor analogous to NarG (1Q16 → molybdenum / [4Fe-4S] /
MGD). **Honest finding: no.** 1SHF has zero HET records; 2ABL has
only crystallographic waters. Both are *apo* SH3 domains.

The closest functional-relevance analog is the canonical PXXP
peptide-binding surface — what SH3 sectors typically map onto. New
files (committed in `ccd6ab4`):

- `demo/pymol_features/sh3_1shf.py` — Fyn numbering: pocket
  Y91/F103/W119/P134/Y137; RT loop 92–101; n-Src loop 113–117;
  distal loop 124–128; specificity Tyr Y137.
- `demo/pymol_features/sh3_2abl.py` — Abl numbering: pocket
  Y89/F91/W118/P131/Y134; RT loop 90–100; n-Src loop 112–117;
  distal loop 123–127; specificity Tyr Y134.
- `demo/SH3/scripts/step7c_pymol_features.sh` — invokes `sca-pymol`
  with the right `--features_py` per `--structure_id` (`Fyn_1SHF`
  or `Abl1_2ABL`).

If the user later wants a *real* cofactor demo, swap in or add an
SH3+peptide co-crystal (1FYN, 1ABO, 1AVZ, 1SEM all viable) and add
a `show_bound_peptide` feature.

## Side workstream — `_orig` → `_loaded` rename + filter-history audit + stop codons

Driven by a user audit question: *"do `retained_sequences` indices
ever drift from the post-load MSA to the raw FASTA?"* Audit
verdict: **no**. `retained_sequences[i]` is always a 0-based index
into `msa_obj_loaded`, which is the only MSA persisted to disk
(`msa_orig.fasta-aln`). The pre-load FASTA records exist briefly
inside `load_msa` and are unreachable downstream. New
`excluded_symbols` / `internal_stop_codon` counts are scalar
metadata, never participate in indexing.

### Rename

`msa_obj_orig` / `msa_orig` / `seqids_orig` / `msa_ids_orig` /
`num_seq_orig` / `num_pos_orig` → `_loaded` everywhere
(139 occurrences across 19 files; perl one-liner with `\b` word
boundaries and a negative lookahead `(?!\.fasta-aln)` to preserve
the disk filename).

The disk filename `msa_orig.fasta-aln` was kept (renaming would
break already-saved preprocessing dirs). Constant
`PREPROCESSING_MSAORIG_FNAME` also kept since it documents that
literal filename. The variable name now correctly conveys
"post-load post-exclusion MSA, not the literal raw input file."

### filter_history stages for pre-preprocessing drops

Previously the "initial" bar showed the post-load count, so any
sequences dropped by `load_msa` were invisible. Now:

- `preprocess_msa` accepts `n_excluded_pre_load` and
  `n_internal_stop_pre_load` kwargs.
- "initial" bar reflects pre-drop input count.
- `internal_stop_codon` and `excluded_symbols` stages inserted in
  load-msa-execution order (internal-stop drop happens before the
  alphabet check), each carrying its `n_filtered`.
- Stages have `stat_values=None` — `plot_filter_distributions`
  iterates only stages with stat_values, so no degenerate
  histograms get rendered.

### Stop-codon handling in `load_msa`

User asked whether to drop stop codons explicitly. Decision matrix:
- Trailing `*` (relative to non-gap content) is a normal CDS-translation
  artifact. Strip it (replace with `-`), preserve the sequence,
  preserve column alignment. INFO log on count.
- Internal `*` (after trailing strip) almost always means premature
  stop / frameshift. Drop the sequence, count separately. WARNING
  log on count.
- `*` is **never** allowed in `SymMap.aa_list` —
  `SymMap.__init__` raises with a message pointing at `load_msa`'s
  handling. Stop codons don't participate in `phi_ia` or any
  frequency calc.

`_strip_trailing_stops(seq_str)` helper in
[`src/mysca/io.py`](../../src/mysca/io.py) handles all the corner
cases (multiple trailing `*`, gap-interleaved trailing `*`, internal
`*`, clean) — covered by a dedicated unit test.

`load_msa` now returns a 6-tuple:
`(msa_obj, msa_matrix, msa_ids, mapping, n_excluded, n_internal_stop)`.

## Cleanup — figure_scripts

The user-placed `figure_scripts/` directory (still untracked) was
patched in two places — `gen_fig5.py` and `gen_fig6.py` — to use
per-position P̃ normalization (paper Eq. 4 setup) instead of the
legacy global form. Header notes added pointing at the new mysca
canonical path. The remaining figure scripts (gen_fig1–4, gen_fig7,
gen_fign) were not touched. The directory remains untracked per
user instruction.

## Test surface added (+22 tests, all passing)

- `tests/test_results.py` — `project_sequences` known-value match,
  paper-vs-global normalization differentiation, save/load roundtrip,
  missing-fields error, shape validation, `to_dataframe`,
  metadata round-trip, metadata merge, plot color paths
  (categorical + numeric).
- `tests/test_project.py` — in-sample `up_score` matches direct
  `SCAResults.project_sequences`; `to_dataframe` schema.
- `tests/test_core.py` — `resolve_torch_dtype` choices; fp32-vs-fp64
  GPU kernel agreement (skips on no-GPU).
- `tests/test_preprocess.py` — `_strip_trailing_stops` unit cases,
  end-to-end `load_msa` stop-codon test, `SymMap` `*` rejection,
  `internal_stop_codon` filter_history stage, behavior when
  no pre-load drops occur.

## Doc surfaces touched (CLAUDE.md three-surface rule)

For every new CLI flag (`--save_dataframe`, `--seq_metadata`,
`--seq_proj_color_by`, `--precision`):

1. argparse `help=` string in the relevant `run_*.py`.
2. Module-level docstring in the same file (COMMAND LINE ARGUMENTS
   + OUTPUTS sections).
3. `docs/cli_reference.md` (per-flag table row + outputs section).

`README.md` also updated for the user-facing surface: new sca-core
key options, sca-project outputs, an out-of-sample `--save_dataframe`
example, and Python API snippets for `project_sequences` /
`to_dataframe` / `up_score`.

## Open items

- 11 commits ahead of `origin/addison-dev`. Push at user's
  discretion.
- `figure_scripts/`, the Rivoire PDF, and `docs/.claude_prompts/` all
  remain untracked per user instruction (`.claude_prompts/` is now
  also gitignored).
- The SH3 demo could later gain an actual ligand-aware structure
  (1FYN / 1ABO / 1AVZ / 1SEM) if a non-apo cofactor demo is desired.
- The fp16 path is preview-only by design; fp32 is the recommended
  fast alternative to fp64. Could later add a `--precision fp32`
  numerical-fidelity benchmark to the perf workstream.
