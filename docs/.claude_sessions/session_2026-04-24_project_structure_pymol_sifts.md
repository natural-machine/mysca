# 2026-04-23 → 2026-04-24 — mysca.project, mysca.structure, sca-pymol rewrite, SIFTS

## Summary

Two-day arc that executes the two plan docs from the prior session
([plan_mysca_project_and_structure.md](plan_mysca_project_and_structure.md)
and [plan_sca_pymol_rewrite.md](plan_sca_pymol_rewrite.md)) plus a
pile of follow-ups surfaced by actually running the SH3 demo
end-to-end.

24 commits, test count 911 → 995 (+84). The `mysca` CLI surface grew
from 5 to 7 entrypoints; the library gained two new subpackages
(`project`, `structure`); the demo grew from 3 steps to 9 (a `step0`
prealign plus `step1`–`step8`), covering every CLI. One deliberate
feature gate was opened mid-arc (HMMER aligner; SIFTS lookup both
library and CLI).

## Pre-session state

```sh
git checkout cd4e8db
```

Branch `addison-dev`. Last session ended with a plan doc for
mysca.project + mysca.structure (session
[2026-04-22_replay_cli_docs_mapping_info.md](session_2026-04-22_replay_cli_docs_mapping_info.md)).
All execution below is linear on top of cd4e8db.

## Deliverables — major features

### `mysca.project` subpackage + `sca-project` CLI (`d9a2404`)

Out-of-sample primary-sequence projection onto an existing SCA
result. Aligner dispatch table with `mafft_add` (default) and
`hmmalign` (initially stubbed, filled in later). In-sample
short-circuit: records whose ID is already in the reference MSA
bypass alignment and reuse the MSA row directly. New result
containers `SequenceProjection` + `ProjectionResult` with
`FIELD_DESCRIPTIONS` + `info()` following the pattern from
`results.py`. CLI writes `projection.json` + `per_sequence/<id>.tsv`.

### `mysca.structure` subpackage + `sca-structure` CLI (`1372d0e`)

PDB/tertiary integration, composing over `sca-project`. New
`PDBStructure` (Biopython-backed, lenient loader that handles
malformed fixtures via PPBuilder), `SequencePdbMap` (TSV primary;
SIFTS placeholder raising `NotImplementedError` initially), and
`PdbProjection` (IC-group memberships in PDB residue-number
coordinates via `residue_ids`). Retired the plural
`src/mysca/structures.py`; migrated `struct2seq` into
`structure.pdb`.

### `make_plots` refactor into `mysca.pl` (`86f64de`)

Extracted the 7 inline matplotlib blocks in `run_sca.py::make_plots`
into individual `mysca.pl` functions matching the existing
save/return contract. `make_plots` is now a ~40-line delegate.
`sca-plots --scacore` replay coverage expanded from 7 functions to
all but one (`plot_covariance_matrix` still deferred — fixed
later by `3acc927`). Removed unused matplotlib imports from
`run_sca`.

### HMMER alignment backend (`912e708`)

Filled in the `hmmalign` dispatch entry that had been stubbed in
`mysca.project.alignment`. Pipeline: Stockholm with full
`#=GC RF` line → `hmmbuild --hand --amino` → `hmmalign --outformat
afa` → strip lowercase insert-column residues to preserve
`L_orig` column count. Installed HMMER 3.4 via conda to validate.
Existing out-of-sample roundtrip test parametrized over both
aligners.

### `sca-pymol` rewrite on `sca-structure` output (`5efa1a1`, `6074fcf`, `59ac8c2`)

Dropped the old input path entirely (`-s/--scaffold`, `--pdb_dir`,
`--modes <statsectors_seq.npz>`, `--show_molybdenum`, JSON
`--features`). New input is `--structure DIR` reading
`structure_projection.json`; PDB residue numbers come from
`projection["ic_pdb_residues"]` directly (no more `1 + raw_index`
fudge, so non-1-indexed PDBs like 1SHF residues 84–142 render
correctly). Protein-specific annotations moved to a user Python
file loaded via `--features_py PATH --features NAME,NAME,...`;
signature `fn(struct, cmd, *, color=None, context=None)`. Shipped
`demo/pymol_features/narg_1q16.py` as a ported reference example.

Commit `59ac8c2` preceded the rewrite to add `pdb_path` to
`PdbProjection.to_dict()` — necessary so `sca-pymol` can reload
the file without a reintroduced `--pdb_dir` flag.

### SIFTS best-structures lookup: library (`ee902e0`) + CLI (`0858090`)

`mysca.structure.sifts.fetch_best_structures_for_uniprot` wraps
EBI PDBe's `mappings/best_structures/{id}` endpoint with local
JSON caching (default `./.sifts_cache/`). No new Python dep —
uses stdlib `urllib.request`; tests patch `urlopen` so nothing
touches the network.

`ee902e0` wired SIFTS into `SequencePdbMap.from_sifts_for_uniprot_ids`
as a library API. `0858090` exposed it as a third input mode on
`sca-structure` with `--uniprot_ids UID [UID ...]`, `--pdb_dir DIR`,
`--cache_dir DIR`. Case-insensitive PDB filename lookup (RCSB
downloads are typically uppercase; SIFTS returns lowercase IDs).

### Cij_raw persistence (`3acc927`)

Closed a small replay gap: the `L x L` reduced covariance matrix
now rides along in `scarun_results.npz`, so
`sca-plots --scacore` can regenerate `covariance_matrix.png`
without rerunning core. Old saves without `Cij_raw` load
gracefully with `None`.

### `run_pymol` render helpers refactor (`ffed823`)

Extracted 4 helpers (`_apply_group_coloring`, `_align_and_focus`,
`_write_views`, `_write_animation`) so `_plot_by_sectors` and
`_plot_with_multiple_sectors` stop duplicating ~60% of their
bodies. Fixed a latent divide-by-zero in the alpha math when all
sector scores are equal. 6 new pymol-free unit tests using
`MagicMock()` as the `cmd` stand-in.

## Deliverables — correctness + testing

### Synthetic projection-test fixture (`88f4fab`, `4d96ec3`, `f248db6`)

Hand-crafted MSA + query sequences + PDB residue offsets that
exhaustively test the raw/aligned/processed/PDB coordinate-system
transforms under non-trivial preprocessing column drops and varied
input lengths. MSA design: `L_orig=10`, `L_proc=7` with three
heavily-gapped columns dropped; five queries covering every length
regime (shorter than L_proc → equal to L_proc → between → equal to
L_orig → strictly longer). Every prediction in the plan doc held
empirically under both aligners. `f248db6` added the defensive
`.`-gap handling + artificial-IC-assignment tests (inject
hand-crafted groups into the sca_dir and verify per-sequence
membership).

### IC summary log reshape + `get_rawseq_indices_of_msa` hardening (`3bf1660`)

`log_top_ic_summary` now emits a multi-line aligned format with
the reference ID echoed and reference *residue letters* (not just
indices). `get_rawseq_indices_of_msa` default widened from `"-"`
to `"-."` so the helper's library API handles Stockholm's insert-
column gap char consistently with the parallel defensive change
we had already made in `mysca.project`.

### Demo-surfaced correctness fixes (three bugs)

1. **`raw_sequence` inconsistency** (`693415b`): out-of-sample
   records were storing `raw_sequence` from the input FASTA, but
   `residue_by_processed_col` / `ic_memberships` count residues in
   the *aligned* (post-drop) sequence. Indices didn't dereference
   into `raw_sequence` correctly when aligners dropped residues.
   TDD: wrote failing test first, confirmed, then fixed in the
   out-of-sample branch to match the in-sample path.

2. **`input_residue_indices` recovery** (`df217af`): 1SHF chain A
   has 59 residues. mafft `--add --keeplength` drops 2 to fit the
   62-column SH3 MSA; `project_groups_to_pdb` refused to map
   because `raw_sequence` (57) ≠ `pdb.sequence` (59). Fix: recover
   the input indices that survived via a subsequence match (column-
   preserving aligners drop in place, don't substitute), store on
   `SequenceProjection.input_residue_indices`, and compose
   `raw_idx → input_idx → pdb.residue_ids[input_idx]`.

3. **`seq_id` filename sanitization** (`48b688c`): Pfam headers
   like `VAV_HUMAN/788-834` were being passed through as filename
   components, so the `/` became a subdirectory boundary and
   caused `FileNotFoundError` on the per-sequence TSV write.
   Introduced `_safe_filename_component` replacing `/ \ | : * ? " < > \s`
   with `_`; wired into both `run_project.py` and `run_structure.py`.

## Deliverables — documentation, demo, wrap

### Demo end-to-end coverage (`2b3d85e`, `76cb8ad`, `2ed9abe`, `696f4eb`, `67ec578`)

Renumbered to a canonical 0–8 step scheme with both `from_msa` and
`from_raw` paths covered where applicable:

| Step | from_msa | from_raw |
|---|---|---|
| 0 prealign | — | ✓ |
| 1 preprocess | ✓ | ✓ |
| 2 scacore | ✓ | ✓ |
| 3 project (mafft) | ✓ | ✓ |
| 4 project (hmmer) | ✓ | — |
| 5 structure (direct PDB) | ✓ | — |
| 6 plots replay | ✓ | ✓ |
| 7 pymol (skipped without pymol-open-source) | ✓ | — |
| 8 structure via SIFTS | ✓ | — |

Shipped fixtures:
[1SHF.pdb](../../demo/SH3/data/pdbs/1SHF.pdb) (human Fyn SH3,
residues 84–142 — a concrete example of non-1-indexed PDB
numbering),
[1SHF.fasta](../../demo/SH3/data/pdbs/1SHF.fasta) (derived primary
sequence),
[demo/SH3/data/sifts_cache/P06241.json](../../demo/SH3/data/sifts_cache/P06241.json)
(pre-populated SIFTS cache entry mapping Fyn's UniProt ID to
1SHF — lets the SIFTS demo step run fully offline).

`step7_pymol.sh` guards on `python -c "import pymol"` and exits 0
with an install hint if pymol-open-source isn't available; it
still runs end-to-end to a PNG when the dep is installed (the
live smoke test at `tests/test_entrypoint_pymol.py::test_sca_pymol_cli_smoke`
verifies this — skipped on envs without pymol, passing on envs
with it).

### README + CLI audit + docstrings (`d243c53`, `f929236`)

`d243c53` filled in missing `help=` text on
`sca-prealign --pbar/-v`, `sca-core --pbar/-v/--seed`,
`sca-preprocess --block_size`, `sca-pymol --nframes/--duration/-v`.
Rewrote `run_sca.py`'s top docstring (was listing ~5 of 20 args).

`f929236` rewrote the README: all seven CLIs listed, stale
`--weight_method` choice list fixed (v3/v4/v5/gpu → sparse/gpu),
PyMOL section rewritten against the new surface, added sections
for sca-project / sca-structure / sca-plots, bioconda optional
deps documented.

### SIFTS cache default moved to cwd (`786e067`)

Followed by the user: SIFTS cache default moved from
`~/.mysca/sifts_cache` to `./.sifts_cache` so per-machine state
stays local to the project / notebook / demo it was generated
from. `.sifts_cache/` added to `.gitignore`. `data/sifts_cache/`
(no leading dot) is deliberately NOT gitignored so the shipped
demo cache file can live in the repo.

### Pymol smoke test fix (`e505b23`)

The live pymol smoke test in `tests/test_entrypoint_pymol.py` had
a stale import that only surfaced once the user installed
pymol-open-source. Rewrote the test to inline the msa07-driven
prep+sca setup (matching `tests/test_structure.py`'s pattern).

### Demo bug fixes from user runs (`fb1da6e`)

First-pass `./run_demo_SH3.sh` run revealed the
`step3_project.sh` awk was doing full-line equality against the
FASTA header (`$0 == ">"+reference`), which fails as soon as the
header carries any text after the ID. Fixed to match against
`$1` (first whitespace-delimited token) — consistent with
Biopython's `rec.id` convention.

## Planning docs retained inline

- [plan_synthetic_projection_fixture.md](plan_synthetic_projection_fixture.md)
  (written `88f4fab`; executed `4d96ec3` + `f248db6`).
- [plan_sca_pymol_rewrite.md](plan_sca_pymol_rewrite.md) (from the
  prior session; executed `5efa1a1` + `6074fcf`).
- `~/.claude/plans/let-s-go-i-want-validated-peacock.md` (plan mode
  scratch doc for the sca-pymol rewrite — not shipped in the repo).

## Verification

```sh
env/bin/python -m pytest tests/
```

End-of-session: **995 passed**. 0 skipped — the live pymol smoke
test now runs (user installed pymol-open-source mid-session).
Baseline at start of session (last session ended at `cd4e8db`):
**911 passed** + pymol-gated skips. Net: +84 tests, no regressions.

The SH3 demo runs end-to-end through all 9 steps on a machine with
the optional deps (mafft, hmmer, pymol-open-source) installed.
Without pymol it still runs fully; step7 emits a skip hint.

## Notes for next session

- **`1SHF.fasta` shipped** alongside `1SHF.pdb` for the HMMER demo
  step; could be regenerated via `PDBStructure` at demo-run time
  instead of shipped, but the file is tiny and keeps the demo
  offline-friendly.
- **SH3 demo from_raw path** still stops at `step3_project_raw.sh`.
  Extending it to cover structure + plots + pymol + sifts would
  duplicate the from_msa coverage; probably not worth it unless a
  feature only shows up on the raw path.
- **sca-pymol rendering** still has the two sector-drawing call
  sites (`_plot_by_sectors` and `_plot_with_multiple_sectors`) with
  some residual duplication. Shared helpers extracted in `ffed823`
  covered the biggest chunks; what remains is mostly the
  frame-level iteration shape, which differs meaningfully between
  the two paths.
- **No known correctness issues** in project / structure / pymol /
  sifts at session end. The three demo-surfaced bugs from this
  arc (`693415b`, `df217af`, `48b688c`) all have regression tests.
