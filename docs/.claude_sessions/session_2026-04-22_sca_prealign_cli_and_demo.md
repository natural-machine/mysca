# 2026-04-22 — `sca-prealign` CLI and raw-sequence SH3 demo path

## Summary

Added a new CLI, `sca-prealign`, that turns raw (unaligned) FASTA input into the aligned FASTA expected by `sca-preprocess`. It supports an optional clustering step (mmseqs2, off by default) followed by a required alignment step (MAFFT `--auto` by default). Both stages dispatch through small registries so additional tools can be plugged in later.

External binaries (`mafft`, and `mmseqs` when `--cluster mmseqs2`) must be resolvable on `PATH`; the CLI resolves every needed binary up front and raises `FileNotFoundError` immediately if any required tool is missing. No binaries were added to `environment.yml` — the install is optional, documented in the README and the demo README.

The SH3 demo was extended to exercise both pipelines from a shared top-level driver. A Pfam PF00018 seed Stockholm file was added and converted (once, offline) into an unaligned FASTA checked in as `demo/SH3/data/seqs/PF00018_raw.fasta`; the conversion itself is documented in prose in `demo/README.md` rather than shipped as code.

## Pre-session state

Session started on branch `main` at:

```
99cf0c2 Bump version to 0.0.2
```

To reach pre-session state:

```sh
git checkout 99cf0c2
```

## Design decisions

Locked in via AskUserQuestion before implementation:

1. **CLI shape** — new standalone `sca-prealign`, chained into `sca-preprocess`. Not added as an `--unaligned` flag on `sca-preprocess`, to keep each CLI focused and to match the existing three-CLI pattern.
2. **External binary install** — conda / bioconda locally into `./env`; at runtime the binaries are located on `PATH`, with a clear error if missing. Mirrors how PyMOL is handled. `environment.yml` intentionally not modified.
3. **Default aligner** — MAFFT `--auto`.
4. **Clustering default** — off. Users opt in with `--cluster mmseqs2`.

Plan file: [for-sca-preprocessing-i-mighty-kitten.md](../../../.claude/plans/for-sca-preprocessing-i-mighty-kitten.md) (outside the repo).

## Changes made

### New package modules

- **[src/mysca/prealign.py](../../src/mysca/prealign.py)** — library layer. `run_cluster` and `run_align` are the public entrypoints, dispatching through `CLUSTERERS = {"mmseqs2": _cluster_mmseqs2}` and `ALIGNERS = {"mafft": _align_mafft}`. `_resolve_bin(binary, override=None)` uses `shutil.which`, logs the resolved absolute path at INFO, and raises `FileNotFoundError` naming the missing binary if not found. `_run_cmd` wraps `subprocess.run(check=True, capture_output=True, text=True)` and logs stdout/stderr at WARNING on failure. `_cluster_mmseqs2` runs `mmseqs easy-cluster` inside a `tempfile.TemporaryDirectory()` and copies `<prefix>_rep_seq.fasta` to the requested output path. `_align_mafft` runs `mafft --auto --thread <n> <in> > <out>`.

- **[src/mysca/run_prealign.py](../../src/mysca/run_prealign.py)** — CLI entrypoint, argparse structure modeled on `run_preprocessing.py`. Resolves every binary it will need up front (mafft always; mmseqs only when clustering) so missing-tool failures happen before any work. Writes `aligned.fasta` (primary output), optional `clustered.fasta`, `prealign_args.json`, and `prealign.log`.

### Wiring

- **[src/mysca/__main__.py](../../src/mysca/__main__.py)** — added `run_prealign()` dispatcher alongside the existing three.
- **[pyproject.toml](../../pyproject.toml)** — added `sca-prealign = "mysca.__main__:run_prealign"` under `[project.scripts]` and an empty `prealign = []` optional-dependencies group as a documentary hook.

### Docs

- **[README.md](../../README.md)** — added a "Preparing raw sequences (optional)" subsection between Setup and Preprocessing, documenting `sca-prealign`, the external-binary requirement, and the `--*_bin` override flags.
- **[docs/cli_reference.md](../cli_reference.md)** — added a full `sca-prealign` section with argument tables.

### Tests

- **[tests/test_entrypoint_prealign.py](../../tests/test_entrypoint_prealign.py)** — 4 tests, gated with `pytest.mark.skipif(shutil.which("mafft") is None)` / `shutil.which("mmseqs") is None` so CI without the binaries skips cleanly:
  - `test_align_only` — align-only path, asserts uniform record length and record count preservation.
  - `test_cluster_then_align` — cluster-then-align, asserts `0 < n_clust ≤ n_in` and `n_aligned == n_clust`.
  - `test_end_to_end_with_preprocess` — prealign → feed `aligned.fasta` into `run_preprocessing.main(...)` → assert `preprocessing_results.npz` exists.
  - `test_missing_binary_fails_fast` — `--align_bin /nonexistent/mafft` must raise `FileNotFoundError` immediately (and does not require the real binary to be absent).

Full pytest suite: **374 passed** (4 new + 370 pre-existing).

### Demo

Existing `demo/SH3/scripts/step1_preprocessing.sh` and `step2_scacore.sh` retained; their `outdir` was retargeted to `out/from_msa/`. Three new scripts added for the raw-sequence path:

- `step1_prealign_raw.sh` — `sca-prealign -i data/seqs/PF00018_raw.fasta -o out/from_raw/prealign`.
- `step2_preprocessing_raw.sh` — `sca-preprocess -i out/from_raw/prealign/aligned.fasta -o out/from_raw/preprocessing`. No `--reference` flag, since Pfam accessions don't match the original `4837_jgi||3708||Equilibrative` reference of the other MSA.
- `step3_scacore_raw.sh` — `sca-core -i out/from_raw/preprocessing -o out/from_raw/scacore`.

`demo/run_demo_SH3.sh` gained `set -e` and now invokes both pipelines in sequence. Output partitions into `SH3/out/from_msa/{preprocessing,scacore}` and `SH3/out/from_raw/{prealign,preprocessing,scacore}`. The `SH3/out/` tree is already gitignored.

**[demo/README.md](../../demo/README.md)** — new README explaining the two paths and documenting the Stockholm→FASTA conversion narrative in prose: the PF00018 seed was retrieved from Pfam in Stockholm format, `-` and `.` gap characters were removed, and the result was rewritten as FASTA with Pfam accessions preserved as record IDs. The Stockholm file is kept alongside the resulting FASTA for provenance. The conversion itself is not a checked-in script.

### Raw FASTA generation (performed once, offline)

The conversion was executed ad-hoc via a Python one-liner invoking `Bio.AlignIO.read(..., "stockholm")` followed by `SeqRecord(Seq(str(r.seq).replace("-", "").replace(".", "").upper()), id=r.id)` for each record, written via `Bio.SeqIO.write(..., "fasta")`. Result: 55 sequences, lengths 46–51. The script is intentionally not committed; future regeneration should be guided by the README prose, not a packaged helper.

## Local dev environment change

Before implementation, installed the external binaries into the project conda env once:

```sh
conda install -p ./env -c bioconda -c conda-forge -y mafft mmseqs2
```

This does not affect `environment.yml` — the install is documented in READMEs only, matching the PyMOL-extra pattern.

## Verification

1. `sca-prealign --help` registered correctly after `pip install -e '.[dev]'` reinstall.
2. Smoke run: `sca-prealign -i tests/_data/seqs/seqs07.fasta -o /tmp/prealign_smoke` → 23 records, all aligned to length 8. Verified with `awk '/^>/{next} {print length($0)}' | sort -u`.
3. Cluster+align smoke: `--cluster mmseqs2 --cluster_min_seq_id 0.5` → 23 → 22 reps → aligned.
4. End-to-end chain into `sca-preprocess` produces `preprocessing_results.npz`.
5. Full demo: `demo/run_demo_SH3.sh` populates both `out/from_msa/{preprocessing,scacore}` and `out/from_raw/{prealign,preprocessing,scacore}` without errors.
6. `pytest tests/` → 374 passed, 120 warnings (same warning set as before — SCA plot singular-transform warnings unchanged).

## Commits made during this session

- `f975fb0` — "Add sca-prealign CLI for clustering and aligning raw FASTA input" (src/mysca/prealign.py, run_prealign.py, __main__.py, pyproject.toml, README.md, docs/cli_reference.md, tests/test_entrypoint_prealign.py). 7 files, +599 / -5.
- `e454601` — "Extend SH3 demo to exercise sca-prealign from raw sequences" (demo/README.md, demo/SH3/data/msas/PF00018.alignment.seed, demo/SH3/data/seqs/PF00018_raw.fasta, three new raw-path scripts, updated step1/step2 scripts and run_demo_SH3.sh). 9 files, +351 / -2.

## Lingering items (not implemented)

- Only `mafft` is registered as an aligner today; `muscle` / `clustalo` would fit directly into the `ALIGNERS` registry without CLI churn when a reason to add them arises.
- Only `mmseqs2` is registered as a clusterer; `cd-hit` or similar can be added analogously.
- No JSON-schema / args-dump consumed programmatically from `prealign_args.json`; it exists for reproducibility and human inspection.
- The raw-path demo skips the `--reference` filter. If a canonical SH3 reference accession (e.g. `VAV_HUMAN/788-834` or similar) becomes the standard, the step2 script can pass `--reference` back in.
- No unit tests for `prealign.py` at the `run_cluster` / `run_align` function level independent of the CLI — only entrypoint-level tests. Fine for now since the CLI is the only caller, but worth splitting if the library gets reused elsewhere.
