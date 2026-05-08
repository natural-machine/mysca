"""Tests for PreprocessingResults and SCAResults container classes.

Tests round-trip save/load for both classes and verifies the factory methods
produce the expected attributes.
"""

import os
import json
import numpy as np
import pytest
from scipy import sparse

from tests.conftest import TMPDIR, remove_dir
from mysca.results import PreprocessingResults, SCAResults


OUTDIR_PREP = os.path.join(TMPDIR, "test_results_prep")
OUTDIR_SCA = os.path.join(TMPDIR, "test_results_sca")


class TestPreprocessingResults:

    def setup_method(self):
        os.makedirs(TMPDIR, exist_ok=True)

    def teardown_method(self):
        if os.path.isdir(OUTDIR_PREP):
            remove_dir(OUTDIR_PREP)

    def _make_sample_results(self):
        """Create a PreprocessingResults with small synthetic data."""
        rng = np.random.default_rng(42)
        M, L, D = 10, 8, 20
        msa = rng.integers(0, D + 1, size=(M, L))
        msa_binary3d = np.eye(D + 1, dtype=bool)[msa][:, :, :-1]
        retained_sequences = np.array([0, 1, 3, 4, 5, 7, 8, 10, 11, 12])
        retained_positions = np.array([1, 2, 3, 5, 6, 8, 9, 10])
        retained_sequence_ids = np.array([
            f"seq_{i}" for i in retained_sequences
        ])
        sequence_weights = rng.random(M)
        fi0_pretruncation = rng.random(15)
        args = {
            "gap_truncation_thresh": 0.4,
            "sequence_gap_thresh": 0.2,
            "sequence_similarity_thresh": 0.8,
        }
        sym_map = {chr(65 + i): i for i in range(D)}
        sym_map["-"] = D

        return PreprocessingResults(
            msa=msa,
            msa_binary3d=msa_binary3d,
            retained_sequences=retained_sequences,
            retained_positions=retained_positions,
            retained_sequence_ids=retained_sequence_ids,
            sequence_weights=sequence_weights,
            fi0_pretruncation=fi0_pretruncation,
            args=args,
            sym_map=sym_map,
            msa_obj_loaded=None,
        )

    def test_attributes(self):
        results = self._make_sample_results()
        assert results.n_sequences == 10
        assert results.n_positions == 8
        assert results.msa.shape == (10, 8)
        assert results.msa_binary3d.shape == (10, 8, 20)
        assert len(results.sequence_weights) == 10
        assert results.args["gap_truncation_thresh"] == 0.4

    def test_save_creates_expected_files(self):
        results = self._make_sample_results()
        results.save(OUTDIR_PREP)
        assert os.path.isfile(
            os.path.join(OUTDIR_PREP, "preprocessing_results.npz")
        )
        assert os.path.isfile(
            os.path.join(OUTDIR_PREP, "preprocessing_args.json")
        )
        assert os.path.isfile(
            os.path.join(OUTDIR_PREP, "sym2int.json")
        )
        assert os.path.isfile(
            os.path.join(OUTDIR_PREP, "msa_binary2d_sp.npz")
        )

    def test_round_trip(self):
        """Save then load and verify all fields match."""
        original = self._make_sample_results()
        original.save(OUTDIR_PREP)
        loaded = PreprocessingResults.load(OUTDIR_PREP)

        np.testing.assert_array_equal(loaded.msa, original.msa)
        np.testing.assert_array_equal(
            loaded.msa_binary3d, original.msa_binary3d
        )
        np.testing.assert_array_equal(
            loaded.retained_sequences, original.retained_sequences
        )
        np.testing.assert_array_equal(
            loaded.retained_positions, original.retained_positions
        )
        np.testing.assert_array_equal(
            loaded.retained_sequence_ids, original.retained_sequence_ids
        )
        np.testing.assert_allclose(
            loaded.sequence_weights, original.sequence_weights
        )
        np.testing.assert_allclose(
            loaded.fi0_pretruncation, original.fi0_pretruncation
        )
        assert loaded.args == original.args
        # load() reconstructs a SymMap; compare its sym2int against the
        # original raw dict.
        assert loaded.sym_map.sym2int == original.sym_map
        assert loaded.n_sequences == original.n_sequences
        assert loaded.n_positions == original.n_positions

    def test_round_trip_filter_history(self):
        """filter_history survives save/load with numpy arrays intact.

        stat_values is typically an NDArray in-process; it has to round-trip
        to a list in JSON and back to an array so replaying the filter
        plots sees the same shape the original run produced.
        """
        original = self._make_sample_results()
        original.filter_history = [
            {
                "stage": "initial",
                "label": "initial",
                "n_sequences": 10,
                "n_positions": 8,
                "n_filtered": 0,
                "axis": None,
                "stat_name": None,
                "stat_values": None,
                "threshold": None,
                "threshold_symbol": None,
                "filter_direction": None,
            },
            {
                "stage": "position_gap",
                "label": "position gap (τ)",
                "n_sequences": 10,
                "n_positions": 7,
                "n_filtered": 1,
                "axis": "positions",
                "stat_name": "gap frequency per position",
                "stat_values": np.array([0.0, 0.1, 0.5, 0.2]),
                "threshold": 0.4,
                "threshold_symbol": "τ",
                "filter_direction": "above",
            },
        ]
        original.save(OUTDIR_PREP)
        loaded = PreprocessingResults.load(OUTDIR_PREP)

        assert loaded.filter_history is not None
        assert len(loaded.filter_history) == len(original.filter_history)
        for orig_entry, loaded_entry in zip(
            original.filter_history, loaded.filter_history
        ):
            orig_sv = orig_entry["stat_values"]
            loaded_sv = loaded_entry["stat_values"]
            if orig_sv is None:
                assert loaded_sv is None
            else:
                assert isinstance(loaded_sv, np.ndarray)
                np.testing.assert_array_equal(loaded_sv, orig_sv)
            for key in (
                "stage", "label", "n_sequences", "n_positions", "n_filtered",
                "axis", "stat_name", "threshold", "threshold_symbol",
                "filter_direction",
            ):
                assert loaded_entry[key] == orig_entry[key]

    def test_round_trip_without_filter_history(self):
        """A saved run with no filter_history loads back as None."""
        original = self._make_sample_results()
        assert original.filter_history is None
        original.save(OUTDIR_PREP)
        loaded = PreprocessingResults.load(OUTDIR_PREP)
        assert loaded.filter_history is None

    def test_from_preprocess_output(self):
        """Test the factory that unpacks preprocess_msa() output format."""
        rng = np.random.default_rng(0)
        M, L, D = 5, 4, 20
        msa = rng.integers(0, D + 1, size=(M, L))
        results_dict = {
            "msa_binary3d": np.eye(D + 1, dtype=bool)[msa][:, :, :-1],
            "retained_sequences": np.arange(M),
            "retained_positions": np.arange(L),
            "retained_sequence_ids": np.array([f"s{i}" for i in range(M)]),
            "sequence_weights": np.ones(M),
            "fi0_pretruncation": rng.random(6),
            "args": {"param": "value"},
        }
        result = PreprocessingResults.from_preprocess_output(
            msa, results_dict, sym_map={"A": 0}
        )
        assert result.n_sequences == M
        assert result.n_positions == L
        np.testing.assert_array_equal(result.msa, msa)
        assert result.sym_map == {"A": 0}

    def test_on_disk_format_readable_without_mysca(self):
        """Verify npz and json files are directly loadable with numpy/json."""
        original = self._make_sample_results()
        original.save(OUTDIR_PREP)

        # Load npz directly
        data = np.load(
            os.path.join(OUTDIR_PREP, "preprocessing_results.npz"),
            allow_pickle=True,
        )
        expected_keys = {
            "msa", "retained_sequences", "retained_positions",
            "retained_sequence_ids", "sequence_weights", "fi0_pretruncation",
        }
        assert set(data.keys()) == expected_keys
        np.testing.assert_array_equal(data["msa"], original.msa)

        # Load json directly
        with open(os.path.join(OUTDIR_PREP, "sym2int.json")) as f:
            sym2int = json.load(f)
        assert sym2int == original.sym_map

        # Load sparse binary MSA directly
        sp = sparse.load_npz(
            os.path.join(OUTDIR_PREP, "msa_binary2d_sp.npz")
        )
        assert sp.shape == (
            original.n_sequences,
            original.n_positions * original.msa_binary3d.shape[-1],
        )


class TestSCAResults:

    def setup_method(self):
        os.makedirs(TMPDIR, exist_ok=True)

    def teardown_method(self):
        if os.path.isdir(OUTDIR_SCA):
            remove_dir(OUTDIR_SCA)

    def _make_sample_results(self, include_sectors=True):
        """Create an SCAResults with small synthetic data."""
        rng = np.random.default_rng(99)
        L, D = 12, 20
        K = 3

        Dia = rng.random((L, D))
        conservation = rng.random(L)
        sca_matrix = rng.random((L, L))
        sca_matrix = sca_matrix + sca_matrix.T  # symmetric
        Cij_raw = rng.random((L, L))
        Cij_raw = Cij_raw + Cij_raw.T  # symmetric
        phi_ia = rng.random((L, D))
        fi0 = rng.random(L)
        fia = rng.random((L, D))

        evals_sca = np.sort(rng.random(L))[::-1]
        evecs_sca = rng.random((L, L))
        sig_evals = evals_sca[:K]
        sig_evecs = evecs_sca[:, :K]

        results = SCAResults(
            Dia=Dia,
            conservation=conservation,
            sca_matrix=sca_matrix,
            phi_ia=phi_ia,
            fi0=fi0,
            fia=fia,
            Cij_raw=Cij_raw,
            evals_sca=evals_sca,
            evecs_sca=evecs_sca,
            significant_evals_sca=sig_evals,
            significant_evecs_sca=sig_evecs,
            kstar=K,
            kstar_identified=K,
            cutoff=0.5,
            evals_shuff=rng.random((10, L)),
            v_ica=rng.random((L, K)),
            w_ica=rng.random((K, K)),
            t_dists_info=[
                {"df": 5.0, "loc": 0.0, "scale": 1.0, "cutoff": 1.5},
                {"df": 4.0, "loc": 0.1, "scale": 0.9, "cutoff": 1.4},
                {"df": 6.0, "loc": -0.1, "scale": 1.1, "cutoff": 1.6},
            ],
            args={"regularization": 0.03, "n_boot": 10, "seed": 42},
        )

        if include_sectors:
            results.ic_positions = [
                np.array([0, 3, 5]),
                np.array([1, 7]),
                np.array([2, 9, 11]),
            ]
            results.group_scores = [
                rng.random(3),
                rng.random(2),
                rng.random(3),
            ]
            results.sca_matrix_sector_subset = rng.random((8, 8))
            results.ic_residues_per_seq = {
                "ic_0_seq1": np.array([0, 3, 5]),
                "ic_1_seq1": np.array([1, 7]),
            }
            results.ic_loadings_per_seq = {
                "ic_0_seq1": rng.random(3),
                "ic_1_seq1": rng.random(2),
            }

        return results

    def test_attributes(self):
        results = self._make_sample_results()
        assert results.n_ic_positions == 3
        assert results.n_positions == 12
        assert results.kstar == 3
        assert results.conservation.shape == (12,)
        assert results.sca_matrix.shape == (12, 12)

    def test_from_core_output(self):
        """Test factory that maps core.run_sca() dict keys."""
        L, D = 8, 20
        rng = np.random.default_rng(0)
        core_dict = {
            "fi0": rng.random(L),
            "fia": rng.random((L, D)),
            "fijab": rng.random((L, L, D, D)),
            "Dia": rng.random((L, D)),
            "Di": rng.random(L),
            "Cijab_raw": rng.random((L, L, D, D)),
            "Cij_raw": rng.random((L, L)),
            "phi_ia": rng.random((L, D)),
            "Cijab_corr": rng.random((L, L, D, D)),
            "Cij_corr": rng.random((L, L)),
        }
        results = SCAResults.from_core_output(core_dict, args={"reg": 0.03})
        # Verify key name mapping
        np.testing.assert_array_equal(results.conservation, core_dict["Di"])
        np.testing.assert_array_equal(
            results.sca_matrix, core_dict["Cij_corr"]
        )
        np.testing.assert_array_equal(results.Dia, core_dict["Dia"])
        np.testing.assert_array_equal(results.Cij_raw, core_dict["Cij_raw"])
        assert results.args == {"reg": 0.03}
        # Eigen/ICA/sector fields should be None
        assert results.evals_sca is None
        assert results.v_ica is None
        assert results.ic_positions is None

    def test_save_creates_expected_files(self):
        results = self._make_sample_results()
        results.save(OUTDIR_SCA)

        assert os.path.isfile(
            os.path.join(OUTDIR_SCA, "scarun_results.npz")
        )
        assert os.path.isfile(
            os.path.join(OUTDIR_SCA, "scarun_args.json")
        )
        assert os.path.isfile(
            os.path.join(OUTDIR_SCA, "sca_eigendecomp.npz")
        )
        assert os.path.isfile(
            os.path.join(OUTDIR_SCA, "ic_residues_per_seq.npz")
        )
        assert os.path.isfile(
            os.path.join(OUTDIR_SCA, "ic_loadings_per_seq.npz")
        )
        # Legacy file names should not be written.
        assert not os.path.isfile(
            os.path.join(OUTDIR_SCA, "statsectors_msa.npz")
        )
        assert not os.path.isfile(
            os.path.join(OUTDIR_SCA, "statsectors_seq.npz")
        )

        scadir = os.path.join(OUTDIR_SCA, "sca_results")
        assert os.path.isfile(os.path.join(scadir, "kstar.txt"))
        assert os.path.isfile(os.path.join(scadir, "kstar_identified.txt"))
        assert os.path.isfile(os.path.join(scadir, "eigenvalue_cutoff.txt"))
        assert os.path.isfile(os.path.join(scadir, "v_ica_normalized.npy"))
        assert os.path.isfile(os.path.join(scadir, "w_ica.npy"))
        assert os.path.isfile(os.path.join(scadir, "t_dists_info.json"))
        assert os.path.isfile(os.path.join(scadir, "evals_shuff.npy"))
        assert os.path.isfile(
            os.path.join(scadir, "sca_matrix_sector_subset.npy")
        )
        # IC positions live in the top-level ic_positions/ directory.
        ic_pos_dir = os.path.join(OUTDIR_SCA, "ic_positions")
        assert os.path.isfile(
            os.path.join(ic_pos_dir, "ic_0_msaproc.npy")
        )
        # Without retained_positions on save(), the msaorig sibling
        # is not written; covered separately below.
        assert not os.path.isfile(
            os.path.join(ic_pos_dir, "ic_0_msaorig.npy")
        )

    def test_save_writes_msaorig_when_retained_positions_supplied(self):
        """Passing retained_positions to save() writes the original-MSA
        coord sibling alongside the processed-MSA file."""
        results = self._make_sample_results()
        # 12 processed-MSA cols (matches L_proc in _make_sample_results);
        # map them onto a hypothetical L_orig=20 by scattering.
        retained_positions = np.array(
            [0, 2, 3, 5, 6, 8, 10, 12, 14, 15, 17, 19], dtype=int,
        )
        results.save(OUTDIR_SCA, retained_positions=retained_positions)
        ic_pos_dir = os.path.join(OUTDIR_SCA, "ic_positions")
        proc = np.load(os.path.join(ic_pos_dir, "ic_0_msaproc.npy"))
        orig = np.load(os.path.join(ic_pos_dir, "ic_0_msaorig.npy"))
        # ic_0 = [0, 3, 5] (processed), maps to retained_positions[[0,3,5]].
        np.testing.assert_array_equal(proc, np.array([0, 3, 5]))
        np.testing.assert_array_equal(
            orig, retained_positions[np.array([0, 3, 5])],
        )

    def test_round_trip_core_fields(self):
        """Save then load and verify core SCA fields match."""
        original = self._make_sample_results()
        original.save(OUTDIR_SCA)
        loaded = SCAResults.load(OUTDIR_SCA)

        np.testing.assert_allclose(loaded.Dia, original.Dia)
        np.testing.assert_allclose(
            loaded.conservation, original.conservation
        )
        np.testing.assert_allclose(loaded.sca_matrix, original.sca_matrix)
        np.testing.assert_allclose(loaded.phi_ia, original.phi_ia)
        np.testing.assert_allclose(loaded.fi0, original.fi0)
        np.testing.assert_allclose(loaded.fia, original.fia)
        # Cij_raw rides along with the rest of scarun_results.npz so
        # sca-plots can render covariance_matrix.png on replay.
        np.testing.assert_allclose(loaded.Cij_raw, original.Cij_raw)
        assert loaded.args == original.args

    def test_round_trip_without_Cij_raw_is_graceful(self):
        """A saved result without Cij_raw (legacy or sca-core run that
        went through the LOAD_DATA path) loads back with Cij_raw=None —
        no exception, no field key error."""
        original = self._make_sample_results()
        original.Cij_raw = None
        original.save(OUTDIR_SCA)
        loaded = SCAResults.load(OUTDIR_SCA)
        assert loaded.Cij_raw is None
        # Other fields still load correctly.
        np.testing.assert_allclose(loaded.sca_matrix, original.sca_matrix)

    def test_round_trip_eigen_fields(self):
        """Save then load and verify eigendecomposition fields."""
        original = self._make_sample_results()
        original.save(OUTDIR_SCA)
        loaded = SCAResults.load(OUTDIR_SCA)

        np.testing.assert_allclose(loaded.evals_sca, original.evals_sca)
        np.testing.assert_allclose(loaded.evecs_sca, original.evecs_sca)
        np.testing.assert_allclose(
            loaded.significant_evals_sca, original.significant_evals_sca
        )
        np.testing.assert_allclose(
            loaded.significant_evecs_sca, original.significant_evecs_sca
        )

    def test_round_trip_bootstrap_fields(self):
        """Save then load and verify bootstrap/significance fields."""
        original = self._make_sample_results()
        original.save(OUTDIR_SCA)
        loaded = SCAResults.load(OUTDIR_SCA)

        assert loaded.kstar == original.kstar
        assert loaded.kstar_identified == original.kstar_identified
        assert loaded.cutoff == pytest.approx(original.cutoff)
        np.testing.assert_allclose(loaded.evals_shuff, original.evals_shuff)

    def test_round_trip_ica_fields(self):
        """Save then load and verify ICA fields."""
        original = self._make_sample_results()
        original.save(OUTDIR_SCA)
        loaded = SCAResults.load(OUTDIR_SCA)

        np.testing.assert_allclose(loaded.v_ica, original.v_ica)
        np.testing.assert_allclose(loaded.w_ica, original.w_ica)
        assert loaded.t_dists_info == original.t_dists_info

    def test_round_trip_sector_fields(self):
        """Save then load and verify sector/group fields."""
        original = self._make_sample_results()
        original.save(OUTDIR_SCA)
        loaded = SCAResults.load(OUTDIR_SCA)

        assert loaded.n_ic_positions == original.n_ic_positions
        for i in range(original.n_ic_positions):
            np.testing.assert_array_equal(
                loaded.ic_positions[i], original.ic_positions[i]
            )
            np.testing.assert_allclose(
                loaded.group_scores[i], original.group_scores[i]
            )

    def test_round_trip_per_seq_dicts(self):
        """Save then load and verify ic_residues_per_seq /
        ic_loadings_per_seq npz dicts."""
        original = self._make_sample_results()
        original.save(OUTDIR_SCA)
        loaded = SCAResults.load(OUTDIR_SCA)

        assert set(loaded.ic_residues_per_seq.keys()) == set(
            original.ic_residues_per_seq.keys()
        )
        for k in original.ic_residues_per_seq:
            np.testing.assert_array_equal(
                loaded.ic_residues_per_seq[k],
                original.ic_residues_per_seq[k],
            )

        assert set(loaded.ic_loadings_per_seq.keys()) == set(
            original.ic_loadings_per_seq.keys()
        )
        for k in original.ic_loadings_per_seq:
            np.testing.assert_allclose(
                loaded.ic_loadings_per_seq[k],
                original.ic_loadings_per_seq[k],
            )

    def test_round_trip_component_coverage_per_seq(self):
        """Save then load and verify component_coverage_per_seq is
        round-tripped key-for-key."""
        rng = np.random.default_rng(7)
        original = self._make_sample_results()
        original.component_coverage_per_seq = {
            "seq_a": rng.random(3),
            "seq_b": np.array([0.5, np.nan, 1.0]),
            "seq_c": np.array([1.0, 1.0, 1.0]),
        }
        original.save(OUTDIR_SCA)
        loaded = SCAResults.load(OUTDIR_SCA)

        assert set(loaded.component_coverage_per_seq.keys()) == set(
            original.component_coverage_per_seq.keys()
        )
        for k, v in original.component_coverage_per_seq.items():
            np.testing.assert_array_equal(
                loaded.component_coverage_per_seq[k], v,
            )

    def test_save_all_includes_large_arrays(self):
        """Verify save_all=True includes Cijab_raw and fijab."""
        rng = np.random.default_rng(0)
        L, D = 5, 4
        results = SCAResults(
            Dia=rng.random((L, D)),
            conservation=rng.random(L),
            sca_matrix=rng.random((L, L)),
            phi_ia=rng.random((L, D)),
            fi0=rng.random(L),
            fia=rng.random((L, D)),
            Cijab_raw=rng.random((L, L, D, D)),
            fijab=rng.random((L, L, D, D)),
        )
        results.save(OUTDIR_SCA, save_all=True)
        data = np.load(os.path.join(OUTDIR_SCA, "scarun_results.npz"))
        assert "Cijab_raw" in data
        assert "fijab" in data

    def test_save_default_excludes_large_arrays(self):
        """Verify save_all=False (default) omits Cijab_raw and fijab."""
        rng = np.random.default_rng(0)
        L, D = 5, 4
        results = SCAResults(
            Dia=rng.random((L, D)),
            conservation=rng.random(L),
            sca_matrix=rng.random((L, L)),
            phi_ia=rng.random((L, D)),
            fi0=rng.random(L),
            fia=rng.random((L, D)),
            Cijab_raw=rng.random((L, L, D, D)),
            fijab=rng.random((L, L, D, D)),
        )
        results.save(OUTDIR_SCA, save_all=False)
        data = np.load(os.path.join(OUTDIR_SCA, "scarun_results.npz"))
        assert "Cijab_raw" not in data
        assert "fijab" not in data

    def test_load_partial_results(self):
        """Loading a directory with only core results gives None for others."""
        rng = np.random.default_rng(0)
        L, D = 5, 4
        results = SCAResults(
            Dia=rng.random((L, D)),
            conservation=rng.random(L),
            sca_matrix=rng.random((L, L)),
            phi_ia=rng.random((L, D)),
            fi0=rng.random(L),
            fia=rng.random((L, D)),
        )
        results.save(OUTDIR_SCA)
        loaded = SCAResults.load(OUTDIR_SCA)

        # Core fields present
        assert loaded.conservation is not None
        assert loaded.sca_matrix is not None
        # Optional fields absent
        assert loaded.evals_sca is None
        assert loaded.v_ica is None
        assert loaded.ic_positions is None
        assert loaded.kstar is None
        assert loaded.evals_shuff is None

    def test_on_disk_format_readable_without_mysca(self):
        """Verify npz and json files are directly loadable."""
        original = self._make_sample_results()
        original.save(OUTDIR_SCA)

        # scarun_results.npz
        data = np.load(os.path.join(OUTDIR_SCA, "scarun_results.npz"))
        expected_keys = {
            "Dia", "conservation", "sca_matrix", "phi_ia", "fi0", "fia"
        }
        assert expected_keys.issubset(set(data.keys()))
        np.testing.assert_allclose(data["conservation"], original.conservation)

        # sca_eigendecomp.npz
        eigen = np.load(os.path.join(OUTDIR_SCA, "sca_eigendecomp.npz"))
        assert "evals_sca" in eigen
        assert "evecs_sca" in eigen

        # scarun_args.json
        with open(os.path.join(OUTDIR_SCA, "scarun_args.json")) as f:
            args = json.load(f)
        assert args == original.args

        # kstar.txt
        kstar = int(np.loadtxt(
            os.path.join(OUTDIR_SCA, "sca_results", "kstar.txt")
        ))
        assert kstar == original.kstar

    def _make_projection_fixture(self, L=4, D=3, K=2, M=5, seed=0):
        """Construct a minimal SCAResults sized for project_sequences tests
        plus a synthetic one-hot xmsa (M, L, D)."""
        rng = np.random.default_rng(seed)
        phi_ia = rng.uniform(0.1, 1.0, size=(L, D))
        fia = rng.uniform(0.1, 1.0, size=(L, D))
        evecs_sca = rng.standard_normal((L, L))
        evals_sca = np.sort(rng.uniform(0.5, 2.0, size=L))[::-1]
        w_ica = rng.standard_normal((K, K))

        sca = SCAResults(
            phi_ia=phi_ia, fia=fia,
            evecs_sca=evecs_sca,
            evals_sca=evals_sca,
            significant_evecs_sca=evecs_sca[:, :K],
            significant_evals_sca=evals_sca[:K],
            w_ica=w_ica,
            n_components=K,
        )

        msa_int = rng.integers(0, D, size=(M, L))
        xmsa = np.eye(D, dtype=bool)[msa_int]
        return sca, xmsa

    def test_project_sequences_known_value(self):
        """Hand-derive Uᵖ for a small fixture and verify the method matches."""
        sca, xmsa = self._make_projection_fixture()

        pf = sca.phi_ia * sca.fia
        denom = np.sqrt(np.sum(pf * pf, axis=-1, keepdims=True))
        pia = pf / denom
        xsi = np.einsum("ia,mia->mi", pia, xmsa.astype(np.float64))
        n_comp = sca.w_ica.shape[0]
        evecs = sca.evecs_sca[:, :n_comp]
        evals = sca.evals_sca[:n_comp]
        utilde = (xsi @ evecs) / np.sqrt(evals)
        expected = utilde @ sca.w_ica.T

        got = sca.project_sequences(xmsa)
        assert got.shape == expected.shape
        np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)

    def test_project_sequences_paper_normalization(self):
        """Per-position P̃ normalization differs from a global-scalar one
        whenever positions have different ||phi*fia||₂. Construct such a
        case and verify project_sequences uses the per-position form."""
        L, D, K = 3, 2, 1
        phi_ia = np.array([[1.0, 0.0], [2.0, 0.0], [4.0, 0.0]])
        fia = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
        # Full eigendecomposition; project_sequences uses the first
        # w_ica.shape[0] columns/values, here just position 0.
        evecs_sca = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        evals_sca = np.array([1.0, 1.0, 1.0])
        w_ica = np.array([[1.0]])
        sca = SCAResults(
            phi_ia=phi_ia, fia=fia,
            evecs_sca=evecs_sca,
            evals_sca=evals_sca,
            significant_evecs_sca=evecs_sca[:, :K],
            significant_evals_sca=evals_sca[:K],
            w_ica=w_ica,
            n_components=K,
        )
        # Single sequence: amino acid 0 at every position
        xmsa = np.zeros((1, L, D), dtype=bool)
        xmsa[0, :, 0] = True

        # Per-position: pia[0,0] = 1 (since phi*f at position 0 is [1, 0])
        # so xsi[0,0] = 1, utilde[0] = 1 / sqrt(1) = 1, Up = 1.
        per_position_expected = np.array([[1.0]])
        # Global: pia[0,0] = 1 / sqrt(1+4+16) = 1/sqrt(21), xsi[0,0] = 1/sqrt(21),
        # utilde[0] = 1/sqrt(21), Up = 1/sqrt(21) ≈ 0.2182.
        global_alt = 1.0 / np.sqrt(21.0)

        got = sca.project_sequences(xmsa)
        np.testing.assert_allclose(got, per_position_expected, atol=1e-12)
        assert not np.isclose(got[0, 0], global_alt)

    def test_project_sequences_round_trip(self):
        """Save + load preserves project_sequences output bit-for-bit."""
        sca = self._make_sample_results(include_sectors=False)
        # Reuse the larger sample fixture's dimensions so save() writes
        # all six core fields cleanly.
        L, D = sca.fia.shape
        K = sca.significant_evals_sca.shape[0]
        rng = np.random.default_rng(7)
        msa_int = rng.integers(0, D, size=(8, L))
        xmsa = np.eye(D, dtype=bool)[msa_int]
        before = sca.project_sequences(xmsa)

        sca.save(OUTDIR_SCA)
        loaded = SCAResults.load(OUTDIR_SCA)
        after = loaded.project_sequences(xmsa)

        np.testing.assert_allclose(after, before, rtol=0, atol=1e-12)

    def test_project_sequences_missing_fields(self):
        """Clear error when a required operand is None."""
        sca, xmsa = self._make_projection_fixture()
        sca.w_ica = None
        with pytest.raises(RuntimeError, match="w_ica"):
            sca.project_sequences(xmsa)

    def test_project_sequences_rejects_wrong_shape(self):
        sca, xmsa = self._make_projection_fixture()
        with pytest.raises(ValueError, match="expected"):
            sca.project_sequences(xmsa[:, :, :-1])  # truncate D

    def test_sequence_metadata_round_trip(self):
        """save() writes sequence_metadata.tsv; load() restores it."""
        pd = pytest.importorskip("pandas")
        sca = self._make_sample_results(include_sectors=False)
        sca.sequence_metadata = pd.DataFrame({
            "seq_id": ["seq_0", "seq_1", "seq_2"],
            "kingdom": ["Bacteria", "Archaea", "Eukaryota"],
            "taxid": [1, 2, 3],
        })
        sca.save(OUTDIR_SCA)
        assert os.path.isfile(
            os.path.join(OUTDIR_SCA, "sequence_metadata.tsv")
        )
        loaded = SCAResults.load(OUTDIR_SCA)
        assert loaded.sequence_metadata is not None
        pd.testing.assert_frame_equal(
            loaded.sequence_metadata.reset_index(drop=True),
            sca.sequence_metadata.reset_index(drop=True),
        )

    def test_plot_seq_projection_2d_categorical_coloring(self, tmp_path):
        """plot_seq_projection_2d accepts categorical color_values and
        writes a PNG without raising."""
        from mysca.pl import plot_seq_projection_2d
        rng = np.random.default_rng(31)
        up = rng.standard_normal((20, 3))
        kingdoms = np.array(
            ["Bacteria"] * 8 + ["Archaea"] * 6 + ["Eukaryota"] * 6,
            dtype=object,
        )
        ax = plot_seq_projection_2d(
            up, (0, 1), str(tmp_path),
            color_values=kingdoms, color_label="kingdom", save=True,
        )
        assert ax is not None
        png_path = tmp_path / "seq_proj_ic0v1_by_kingdom.png"
        assert png_path.exists()

    def test_plot_seq_projection_2d_numeric_coloring(self, tmp_path):
        """plot_seq_projection_2d accepts numeric color_values and adds
        a colorbar."""
        from mysca.pl import plot_seq_projection_2d
        rng = np.random.default_rng(32)
        up = rng.standard_normal((15, 3))
        scores = rng.uniform(0, 1, size=15)
        ax = plot_seq_projection_2d(
            up, (0, 1), str(tmp_path),
            color_values=scores, color_label="score", save=True,
        )
        assert ax is not None
        assert (tmp_path / "seq_proj_ic0v1_by_score.png").exists()

    def test_to_dataframe_merges_metadata(self):
        """to_dataframe joins sequence_metadata onto seq_id when present."""
        pd = pytest.importorskip("pandas")
        from Bio.Seq import Seq
        from Bio.SeqRecord import SeqRecord
        from Bio.Align import MultipleSeqAlignment

        sca = self._make_sample_results(include_sectors=False)
        L, D = sca.fia.shape
        rng = np.random.default_rng(13)
        M = 3
        msa_int = rng.integers(0, D, size=(M, L))
        msa_binary3d = np.eye(D, dtype=bool)[msa_int]
        aa_list = list("ACDEFGHIKLMNPQRSTVWY")
        records = [
            SeqRecord(
                Seq("".join(aa_list[c] for c in msa_int[m])),
                id=f"seq_{m}",
            )
            for m in range(M)
        ]
        prep = PreprocessingResults(
            msa=msa_int, msa_binary3d=msa_binary3d,
            retained_sequences=np.arange(M),
            retained_positions=np.arange(L),
            retained_sequence_ids=np.array([f"seq_{i}" for i in range(M)]),
            sequence_weights=np.ones(M),
            fi0_pretruncation=np.zeros(L),
            args={},
            sym_map={c: i for i, c in enumerate(aa_list)} | {"-": D},
            msa_obj_loaded=MultipleSeqAlignment(records),
            filter_history=None,
        )
        sca.sequence_metadata = pd.DataFrame({
            "seq_id": ["seq_0", "seq_1"],
            "phylum": ["Firmicutes", "Proteobacteria"],
        })
        df = sca.to_dataframe(prep)
        assert "phylum" in df.columns
        # seq_2 has no metadata entry → NaN
        assert df.loc[df["seq_id"] == "seq_2", "phylum"].isna().all()
        assert df.loc[df["seq_id"] == "seq_0", "phylum"].iloc[0] == \
            "Firmicutes"

    def test_to_dataframe_columns_and_rows(self):
        """SCAResults.to_dataframe(prep) yields one row per retained
        sequence with seq_id, aligned_sequence, and up_* columns."""
        pd = pytest.importorskip("pandas")
        from Bio.Seq import Seq
        from Bio.SeqRecord import SeqRecord
        from Bio.Align import MultipleSeqAlignment

        sca = self._make_sample_results(include_sectors=False)
        L, D = sca.fia.shape
        K = sca.significant_evals_sca.shape[0]
        rng = np.random.default_rng(11)
        # Build a synthetic preprocessing fixture matching sca's L_proc.
        M = 4
        msa_int = rng.integers(0, D, size=(M, L))
        msa_binary3d = np.eye(D, dtype=bool)[msa_int]
        retained_sequences = np.arange(M)
        retained_sequence_ids = np.array([f"seq_{i}" for i in range(M)])
        # Aligned sequences (length L, no actual gaps because the test
        # MSA has no truncation; the original-MSA-coordinate sequence
        # equals the processed sequence here).
        aa_list = list("ACDEFGHIKLMNPQRSTVWY")  # 20-AA canonical
        records = [
            SeqRecord(
                Seq("".join(aa_list[c] for c in msa_int[m])),
                id=f"seq_{m}",
            )
            for m in range(M)
        ]
        msa_obj_loaded = MultipleSeqAlignment(records)
        prep = PreprocessingResults(
            msa=msa_int,
            msa_binary3d=msa_binary3d,
            retained_sequences=retained_sequences,
            retained_positions=np.arange(L),
            retained_sequence_ids=retained_sequence_ids,
            sequence_weights=np.ones(M),
            fi0_pretruncation=np.zeros(L),
            args={},
            sym_map={c: i for i, c in enumerate(aa_list)} | {"-": D},
            msa_obj_loaded=msa_obj_loaded,
            filter_history=None,
        )

        df = sca.to_dataframe(prep)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == M
        assert set(["seq_id", "aligned_sequence"]).issubset(df.columns)
        for k in range(K):
            assert f"Up_{k}" in df.columns
        assert df["seq_id"].tolist() == list(retained_sequence_ids)


class TestFieldDescriptions:
    """FIELD_DESCRIPTIONS / info() contract for both result classes."""

    def test_preprocessing_field_descriptions_cover_all_init_args(self):
        """Every attribute set by PreprocessingResults.__init__ must have
        a description entry so the info() output is never silently missing
        a field."""
        sample = PreprocessingResults(
            msa=np.zeros((2, 3), dtype=int),
            msa_binary3d=None,
            retained_sequences=np.arange(2),
            retained_positions=np.arange(3),
            retained_sequence_ids=np.array(["a", "b"]),
            sequence_weights=np.ones(2),
            fi0_pretruncation=np.zeros(3),
            args={},
        )
        init_attrs = set(vars(sample).keys())
        described = set(PreprocessingResults.FIELD_DESCRIPTIONS.keys())
        missing = init_attrs - described
        assert not missing, f"FIELD_DESCRIPTIONS missing: {sorted(missing)}"
        extras = described - init_attrs
        assert not extras, (
            f"FIELD_DESCRIPTIONS has entries with no matching attribute: "
            f"{sorted(extras)}"
        )

    def test_sca_field_descriptions_cover_all_init_args(self):
        sample = SCAResults()
        init_attrs = set(vars(sample).keys())
        described = set(SCAResults.FIELD_DESCRIPTIONS.keys())
        missing = init_attrs - described
        assert not missing, f"FIELD_DESCRIPTIONS missing: {sorted(missing)}"
        extras = described - init_attrs
        assert not extras, (
            f"FIELD_DESCRIPTIONS has entries with no matching attribute: "
            f"{sorted(extras)}"
        )

    def test_preprocessing_info_marks_populated_vs_none(self):
        r = PreprocessingResults(
            msa=np.zeros((2, 3), dtype=int),
            msa_binary3d=None,
            retained_sequences=np.arange(2),
            retained_positions=np.arange(3),
            retained_sequence_ids=np.array(["a", "b"]),
            sequence_weights=np.ones(2),
            fi0_pretruncation=np.zeros(3),
            args={"threshold": 0.4},
        )
        text = r.info()
        assert "PreprocessingResults" in text
        # Populated ndarray fields show a shape/dtype description.
        assert "retained_positions" in text
        assert "ndarray(3,)" in text
        # None fields render as "(none)".
        lines = {
            ln.split()[0]: ln
            for ln in text.splitlines()
            if ln and not ln.startswith(("-", " ", "P", "f"))
        }
        # msa_binary3d was left as None.
        none_field_lines = [
            ln for ln in text.splitlines() if "(none)" in ln
        ]
        assert any("msa_binary3d" in ln for ln in none_field_lines)

    def test_sca_info_shows_scalars_and_none(self):
        r = SCAResults(kstar=3, conservation=np.arange(5, dtype=float))
        text = r.info()
        assert "SCAResults" in text
        assert "kstar" in text and "int=3" in text
        assert "conservation" in text and "ndarray(5,)" in text
        assert "(none)" in text  # many fields are left unset

    def test_field_descriptions_are_class_level_constants(self):
        """FIELD_DESCRIPTIONS should live on the class, not on instances."""
        assert "FIELD_DESCRIPTIONS" in vars(PreprocessingResults)
        assert "FIELD_DESCRIPTIONS" in vars(SCAResults)
        r = SCAResults()
        assert "FIELD_DESCRIPTIONS" not in vars(r)
