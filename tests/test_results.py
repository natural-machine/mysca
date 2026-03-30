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
            msa_obj_orig=None,
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
        assert loaded.sym_map == original.sym_map
        assert loaded.n_sequences == original.n_sequences
        assert loaded.n_positions == original.n_positions

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
            results.groups = [
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
            results.statsectors_msa = {
                "group_0_seq1": np.array([0, 3, 5]),
                "group_1_seq1": np.array([1, 7]),
            }
            results.statsectors_seq = {
                "sector_0_pdbpos_seq1": np.array([0, 3, 5]),
                "sector_0_scores_seq1": rng.random(3),
            }

        return results

    def test_attributes(self):
        results = self._make_sample_results()
        assert results.n_sectors == 3
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
        assert results.groups is None

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
            os.path.join(OUTDIR_SCA, "statsectors_msa.npz")
        )
        assert os.path.isfile(
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
        assert os.path.isfile(
            os.path.join(scadir, "msa_sectors", "sector_0_msapos.npy")
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
        assert loaded.args == original.args

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

        assert loaded.n_sectors == original.n_sectors
        for i in range(original.n_sectors):
            np.testing.assert_array_equal(
                loaded.groups[i], original.groups[i]
            )
            np.testing.assert_allclose(
                loaded.group_scores[i], original.group_scores[i]
            )

    def test_round_trip_statsectors(self):
        """Save then load and verify statsectors npz dicts."""
        original = self._make_sample_results()
        original.save(OUTDIR_SCA)
        loaded = SCAResults.load(OUTDIR_SCA)

        assert set(loaded.statsectors_msa.keys()) == set(
            original.statsectors_msa.keys()
        )
        for k in original.statsectors_msa:
            np.testing.assert_array_equal(
                loaded.statsectors_msa[k], original.statsectors_msa[k]
            )

        assert set(loaded.statsectors_seq.keys()) == set(
            original.statsectors_seq.keys()
        )
        for k in original.statsectors_seq:
            np.testing.assert_allclose(
                loaded.statsectors_seq[k], original.statsectors_seq[k]
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
        assert loaded.groups is None
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
