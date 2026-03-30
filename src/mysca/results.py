"""Result container classes for SCA preprocessing and core analysis.

These classes provide attribute-based access to SCA results and encapsulate
save/load logic. The on-disk format uses standard numpy (.npz, .npy) and
JSON files, so results can be loaded without mysca installed.

On-disk formats are documented in each class's docstring.
"""

import os
import json
import numpy as np
from numpy.typing import NDArray
from scipy import sparse


# File name constants — shared with run_preprocessing.py and run_sca.py
PREPROCESSING_RESULTS_FNAME = "preprocessing_results.npz"
PREPROCESSING_SYMMAP_FNAME = "sym2int.json"
PREPROCESSING_ARGS_FNAME = "preprocessing_args.json"
PREPROCESSING_MSAORIG_FNAME = "msa_orig.fasta-aln"
PREPROCESSING_BINARY2D_FNAME = "msa_binary2d_sp.npz"

SCARUN_RESULTS_FNAME = "scarun_results.npz"
SCARUN_ARGS_FNAME = "scarun_args.json"
SCARUN_EIGENDECOMP_FNAME = "sca_eigendecomp.npz"
STATSECTORS_MSA_FNAME = "statsectors_msa.npz"
STATSECTORS_SEQ_FNAME = "statsectors_seq.npz"
EVALS_SHUFF_FNAME = "evals_shuff.npy"


class PreprocessingResults:
    """Container for SCA preprocessing results.

    Provides named attribute access to preprocessing outputs and handles
    persistence to/from a directory of npz/json files.

    On-disk format (usable without mysca):
        preprocessing_results.npz
            msa                   : int array (M x L), processed MSA
            retained_sequences    : int array, indices into original MSA rows
            retained_positions    : int array, indices into original MSA columns
            retained_sequence_ids : str array, IDs of retained sequences
            sequence_weights      : float array (M,)
            fi0_pretruncation     : float array, gap freq before truncation
        preprocessing_args.json   : dict of preprocessing parameters
        sym2int.json              : dict mapping symbols to integers
        msa_binary2d_sp.npz       : sparse CSR (M x 20L), one-hot MSA
        msa_orig.fasta-aln        : original MSA in FASTA format (optional)
    """

    def __init__(
        self,
        msa,
        msa_binary3d,
        retained_sequences,
        retained_positions,
        retained_sequence_ids,
        sequence_weights,
        fi0_pretruncation,
        args,
        sym_map=None,
        msa_obj_orig=None,
    ):
        self.msa = msa
        self.msa_binary3d = msa_binary3d
        self.retained_sequences = retained_sequences
        self.retained_positions = retained_positions
        self.retained_sequence_ids = retained_sequence_ids
        self.sequence_weights = sequence_weights
        self.fi0_pretruncation = fi0_pretruncation
        self.args = args
        self.sym_map = sym_map
        self.msa_obj_orig = msa_obj_orig

    @property
    def n_sequences(self):
        return self.msa.shape[0]

    @property
    def n_positions(self):
        return self.msa.shape[1]

    @classmethod
    def from_preprocess_output(cls, msa, results_dict, sym_map=None,
                               msa_obj_orig=None):
        """Construct from the (msa, results_dict) returned by preprocess_msa().
        """
        return cls(
            msa=msa,
            msa_binary3d=results_dict["msa_binary3d"],
            retained_sequences=results_dict["retained_sequences"],
            retained_positions=results_dict["retained_positions"],
            retained_sequence_ids=results_dict["retained_sequence_ids"],
            sequence_weights=results_dict["sequence_weights"],
            fi0_pretruncation=results_dict["fi0_pretruncation"],
            args=results_dict["args"],
            sym_map=sym_map,
            msa_obj_orig=msa_obj_orig,
        )

    def save(self, outdir):
        """Save all results to the given directory."""
        os.makedirs(outdir, exist_ok=True)

        np.savez(
            os.path.join(outdir, PREPROCESSING_RESULTS_FNAME),
            msa=self.msa,
            retained_sequences=self.retained_sequences,
            retained_positions=self.retained_positions,
            retained_sequence_ids=self.retained_sequence_ids,
            sequence_weights=self.sequence_weights,
            fi0_pretruncation=self.fi0_pretruncation,
        )

        with open(os.path.join(outdir, PREPROCESSING_ARGS_FNAME), "w") as f:
            json.dump(self.args, f)

        if self.sym_map is not None:
            sym2int = (self.sym_map.sym2int
                       if hasattr(self.sym_map, "sym2int")
                       else self.sym_map)
            with open(os.path.join(outdir, PREPROCESSING_SYMMAP_FNAME), "w") as f:
                json.dump(sym2int, f)

        if self.msa_binary3d is not None:
            sparse.save_npz(
                os.path.join(outdir, PREPROCESSING_BINARY2D_FNAME),
                sparse.csr_matrix(
                    self.msa_binary3d.reshape(
                        [self.msa_binary3d.shape[0], -1]
                    )
                ),
            )

        if self.msa_obj_orig is not None:
            from Bio import AlignIO
            AlignIO.write(
                self.msa_obj_orig,
                os.path.join(outdir, PREPROCESSING_MSAORIG_FNAME),
                format="fasta",
            )

    @classmethod
    def load(cls, dirpath):
        """Load results from a directory previously created by save()."""
        data = np.load(
            os.path.join(dirpath, PREPROCESSING_RESULTS_FNAME),
            allow_pickle=True,
        )
        msa = data["msa"]
        retained_sequences = data["retained_sequences"]
        retained_positions = data["retained_positions"]
        retained_sequence_ids = data["retained_sequence_ids"]
        sequence_weights = data["sequence_weights"]
        fi0_pretruncation = data["fi0_pretruncation"]

        # Load args
        args_path = os.path.join(dirpath, PREPROCESSING_ARGS_FNAME)
        args = None
        if os.path.isfile(args_path):
            with open(args_path, "r") as f:
                args = json.load(f)

        # Load sym_map as plain dict
        sym_map = None
        symmap_path = os.path.join(dirpath, PREPROCESSING_SYMMAP_FNAME)
        if os.path.isfile(symmap_path):
            with open(symmap_path, "r") as f:
                sym_map = json.load(f)

        # Load sparse binary MSA -> dense 3d
        msa_binary3d = None
        binary_path = os.path.join(dirpath, PREPROCESSING_BINARY2D_FNAME)
        if os.path.isfile(binary_path):
            msa_binary3d = sparse.load_npz(binary_path).toarray().reshape(
                [len(retained_sequences), len(retained_positions), -1]
            )

        # Try to load original MSA (requires Biopython)
        msa_obj_orig = None
        orig_path = os.path.join(dirpath, PREPROCESSING_MSAORIG_FNAME)
        if os.path.isfile(orig_path):
            try:
                from Bio import AlignIO
                msa_obj_orig = AlignIO.read(orig_path, "fasta")
            except ImportError:
                pass

        return cls(
            msa=msa,
            msa_binary3d=msa_binary3d,
            retained_sequences=retained_sequences,
            retained_positions=retained_positions,
            retained_sequence_ids=retained_sequence_ids,
            sequence_weights=sequence_weights,
            fi0_pretruncation=fi0_pretruncation,
            args=args,
            sym_map=sym_map,
            msa_obj_orig=msa_obj_orig,
        )


class SCAResults:
    """Container for SCA core + analysis results.

    On-disk format (usable without mysca):
        scarun_results.npz
            Dia          : float array (L x D), conservation per position+aa
            conservation : float array (L,), aggregated Di
            sca_matrix   : float array (L x L), weighted SCA covariance Cij
            phi_ia       : float array (L x D)
            fi0          : float array (L,)
            fia          : float array (L x D)
            (optional) Cijab_raw, fijab : large pairwise arrays
        sca_eigendecomp.npz
            evals_sca, evecs_sca, significant_evals_sca, significant_evecs_sca
        scarun_args.json              : dict of SCA parameters
        statsectors_msa.npz           : sector positions in MSA coordinates
        statsectors_seq.npz           : sector positions in sequence coordinates
        sca_results/
            kstar.txt                 : number of significant eigenvalues used
            kstar_identified.txt      : number identified by bootstrap
            eigenvalue_cutoff.txt     : bootstrap cutoff value
            v_ica_normalized.npy      : normalized ICA components
            w_ica.npy                 : ICA unmixing matrix
            t_dists_info.json         : t-distribution fit parameters
            evals_shuff.npy           : bootstrap eigenvalues
            sca_matrix_sector_subset.npy
            msa_sectors/sector_*      : per-sector position and score arrays
    """

    def __init__(
        self,
        # Core SCA
        Dia=None,
        conservation=None,
        sca_matrix=None,
        phi_ia=None,
        fi0=None,
        fia=None,
        # Optional large arrays
        Cijab_raw=None,
        fijab=None,
        Cij_raw=None,
        # Eigendecomposition
        evals_sca=None,
        evecs_sca=None,
        significant_evals_sca=None,
        significant_evecs_sca=None,
        # Bootstrap / significance
        kstar=None,
        kstar_identified=None,
        cutoff=None,
        evals_shuff=None,
        # ICA
        v_ica=None,
        w_ica=None,
        # Sectors
        groups=None,
        group_scores=None,
        t_dists_info=None,
        statsectors_msa=None,
        statsectors_seq=None,
        sca_matrix_sector_subset=None,
        # Args
        args=None,
    ):
        self.Dia = Dia
        self.conservation = conservation
        self.sca_matrix = sca_matrix
        self.phi_ia = phi_ia
        self.fi0 = fi0
        self.fia = fia
        self.Cijab_raw = Cijab_raw
        self.fijab = fijab
        self.Cij_raw = Cij_raw
        self.evals_sca = evals_sca
        self.evecs_sca = evecs_sca
        self.significant_evals_sca = significant_evals_sca
        self.significant_evecs_sca = significant_evecs_sca
        self.kstar = kstar
        self.kstar_identified = kstar_identified
        self.cutoff = cutoff
        self.evals_shuff = evals_shuff
        self.v_ica = v_ica
        self.w_ica = w_ica
        self.groups = groups
        self.group_scores = group_scores
        self.t_dists_info = t_dists_info
        self.statsectors_msa = statsectors_msa
        self.statsectors_seq = statsectors_seq
        self.sca_matrix_sector_subset = sca_matrix_sector_subset
        self.args = args

    @property
    def n_sectors(self):
        if self.groups is None:
            return None
        return len(self.groups)

    @property
    def n_positions(self):
        if self.conservation is not None:
            return len(self.conservation)
        return None

    @classmethod
    def from_core_output(cls, sca_results_dict, args=None):
        """Construct from the dict returned by core.run_sca().

        Maps core key names to the canonical attribute names used on disk:
            Di -> conservation, Cij_corr -> sca_matrix, etc.
        """
        return cls(
            Dia=sca_results_dict.get("Dia"),
            conservation=sca_results_dict.get("Di"),
            sca_matrix=sca_results_dict.get("Cij_corr"),
            phi_ia=sca_results_dict.get("phi_ia"),
            fi0=sca_results_dict.get("fi0"),
            fia=sca_results_dict.get("fia"),
            Cijab_raw=sca_results_dict.get("Cijab_raw"),
            fijab=sca_results_dict.get("fijab"),
            Cij_raw=sca_results_dict.get("Cij_raw"),
            args=args,
        )

    def save(self, outdir, save_all=False):
        """Save results to the given directory.

        Args:
            outdir: Output directory path.
            save_all: If True, include large arrays (Cijab_raw, fijab).
        """
        scadir = os.path.join(outdir, "sca_results")
        os.makedirs(outdir, exist_ok=True)
        os.makedirs(scadir, exist_ok=True)

        # Core SCA results
        if self.conservation is not None:
            tosave = {
                "Dia": self.Dia,
                "conservation": self.conservation,
                "sca_matrix": self.sca_matrix,
                "phi_ia": self.phi_ia,
                "fi0": self.fi0,
                "fia": self.fia,
            }
            if save_all:
                if self.Cijab_raw is not None:
                    tosave["Cijab_raw"] = self.Cijab_raw
                if self.fijab is not None:
                    tosave["fijab"] = self.fijab
            np.savez_compressed(
                os.path.join(outdir, SCARUN_RESULTS_FNAME),
                **tosave,
            )

        # Args
        if self.args is not None:
            with open(os.path.join(outdir, SCARUN_ARGS_FNAME), "w") as f:
                json.dump(self.args, f)

        # Eigendecomposition
        if self.evals_sca is not None:
            np.savez_compressed(
                os.path.join(outdir, SCARUN_EIGENDECOMP_FNAME),
                evals_sca=self.evals_sca,
                evecs_sca=self.evecs_sca,
                significant_evals_sca=self.significant_evals_sca,
                significant_evecs_sca=self.significant_evecs_sca,
            )

        # Bootstrap / significance scalars and arrays
        if self.kstar_identified is not None:
            np.savetxt(
                os.path.join(scadir, "kstar_identified.txt"),
                [self.kstar_identified], fmt="%d",
            )
        if self.kstar is not None:
            np.savetxt(
                os.path.join(scadir, "kstar.txt"),
                [self.kstar], fmt="%d",
            )
        if self.cutoff is not None:
            np.savetxt(
                os.path.join(scadir, "eigenvalue_cutoff.txt"),
                [self.cutoff],
            )
        if self.evals_sca is not None:
            np.save(os.path.join(scadir, "all_evals_sca.npy"), self.evals_sca)
            np.save(os.path.join(scadir, "all_evecs_sca.npy"), self.evecs_sca)
        if self.significant_evals_sca is not None:
            np.save(
                os.path.join(scadir, "significant_evals_sca.npy"),
                self.significant_evals_sca,
            )
            np.save(
                os.path.join(scadir, "significant_evecs_sca.npy"),
                self.significant_evecs_sca,
            )
        if self.evals_shuff is not None and len(self.evals_shuff) > 0:
            np.save(os.path.join(scadir, EVALS_SHUFF_FNAME), self.evals_shuff)

        # ICA
        if self.v_ica is not None:
            np.save(os.path.join(scadir, "v_ica_normalized.npy"), self.v_ica)
        if self.w_ica is not None:
            np.save(os.path.join(scadir, "w_ica.npy"), self.w_ica)

        # t-distribution info
        if self.t_dists_info is not None:
            with open(os.path.join(scadir, "t_dists_info.json"), "w") as f:
                json.dump(self.t_dists_info, f)

        # Sector subset of SCA matrix
        if self.sca_matrix_sector_subset is not None:
            np.save(
                os.path.join(scadir, "sca_matrix_sector_subset.npy"),
                self.sca_matrix_sector_subset,
            )

        # Groups (sectors) in MSA coordinates
        if self.groups is not None:
            sector_dir = os.path.join(scadir, "msa_sectors")
            groups_dir = os.path.join(outdir, "groups")
            os.makedirs(sector_dir, exist_ok=True)
            os.makedirs(groups_dir, exist_ok=True)
            for i, group in enumerate(self.groups):
                np.save(
                    os.path.join(groups_dir, f"group_{i}_msapos.npy"), group
                )
                np.save(
                    os.path.join(sector_dir, f"sector_{i}_msapos.npy"), group
                )
                if self.group_scores is not None:
                    np.save(
                        os.path.join(sector_dir, f"sector_{i}_scores.npy"),
                        self.group_scores[i],
                    )
            # Combined mapping
            group_idxs_all = np.concatenate(self.groups, axis=0)
            msapos_to_groupidx = np.vstack([
                group_idxs_all,
                np.concatenate(
                    [len(g) * [i] for i, g in enumerate(self.groups)], axis=0
                ),
            ])
            np.save(
                os.path.join(scadir, "msapos_to_groupidx.npy"),
                msapos_to_groupidx,
            )
            # All important positions
            all_imp = np.unique(group_idxs_all)
            np.save(
                os.path.join(scadir, "all_important_positions.npy"), all_imp
            )

        # Statistical sectors in MSA and sequence coordinates
        if self.statsectors_msa is not None:
            np.savez_compressed(
                os.path.join(outdir, STATSECTORS_MSA_FNAME),
                **self.statsectors_msa,
            )
        if self.statsectors_seq is not None:
            np.savez_compressed(
                os.path.join(outdir, STATSECTORS_SEQ_FNAME),
                **self.statsectors_seq,
            )

    @classmethod
    def load(cls, dirpath):
        """Load results from a directory previously created by save()."""
        scadir = os.path.join(dirpath, "sca_results")

        # Core SCA results
        Dia = conservation = sca_matrix = phi_ia = fi0 = fia = None
        Cijab_raw = fijab = None
        results_path = os.path.join(dirpath, SCARUN_RESULTS_FNAME)
        if os.path.isfile(results_path):
            data = np.load(results_path)
            Dia = data.get("Dia")
            conservation = data.get("conservation")
            sca_matrix = data.get("sca_matrix")
            phi_ia = data.get("phi_ia")
            fi0 = data.get("fi0")
            fia = data.get("fia")
            Cijab_raw = data.get("Cijab_raw")
            fijab = data.get("fijab")

        # Args
        args = None
        args_path = os.path.join(dirpath, SCARUN_ARGS_FNAME)
        if os.path.isfile(args_path):
            with open(args_path, "r") as f:
                args = json.load(f)

        # Eigendecomposition
        evals_sca = evecs_sca = None
        significant_evals_sca = significant_evecs_sca = None
        eigen_path = os.path.join(dirpath, SCARUN_EIGENDECOMP_FNAME)
        if os.path.isfile(eigen_path):
            data = np.load(eigen_path)
            evals_sca = data.get("evals_sca")
            evecs_sca = data.get("evecs_sca")
            significant_evals_sca = data.get("significant_evals_sca")
            significant_evecs_sca = data.get("significant_evecs_sca")

        # Bootstrap / significance
        kstar = _load_scalar_txt(os.path.join(scadir, "kstar.txt"), int)
        kstar_identified = _load_scalar_txt(
            os.path.join(scadir, "kstar_identified.txt"), int
        )
        cutoff = _load_scalar_txt(
            os.path.join(scadir, "eigenvalue_cutoff.txt"), float
        )
        evals_shuff = _load_npy_or_none(
            os.path.join(scadir, EVALS_SHUFF_FNAME)
        )

        # ICA
        v_ica = _load_npy_or_none(
            os.path.join(scadir, "v_ica_normalized.npy")
        )
        w_ica = _load_npy_or_none(os.path.join(scadir, "w_ica.npy"))

        # t-distribution info
        t_dists_info = None
        tdist_path = os.path.join(scadir, "t_dists_info.json")
        if os.path.isfile(tdist_path):
            with open(tdist_path, "r") as f:
                t_dists_info = json.load(f)

        # Sector subset
        sca_matrix_sector_subset = _load_npy_or_none(
            os.path.join(scadir, "sca_matrix_sector_subset.npy")
        )

        # Groups from msa_sectors directory
        groups = None
        group_scores = None
        sector_dir = os.path.join(scadir, "msa_sectors")
        if os.path.isdir(sector_dir):
            groups = []
            group_scores = []
            i = 0
            while True:
                gpath = os.path.join(sector_dir, f"sector_{i}_msapos.npy")
                if not os.path.isfile(gpath):
                    break
                groups.append(np.load(gpath))
                spath = os.path.join(sector_dir, f"sector_{i}_scores.npy")
                if os.path.isfile(spath):
                    group_scores.append(np.load(spath))
                i += 1
            if not groups:
                groups = None
                group_scores = None
            elif not group_scores:
                group_scores = None

        # Statistical sectors
        statsectors_msa = None
        msa_path = os.path.join(dirpath, STATSECTORS_MSA_FNAME)
        if os.path.isfile(msa_path):
            statsectors_msa = dict(np.load(msa_path, allow_pickle=True))

        statsectors_seq = None
        seq_path = os.path.join(dirpath, STATSECTORS_SEQ_FNAME)
        if os.path.isfile(seq_path):
            statsectors_seq = dict(np.load(seq_path, allow_pickle=True))

        return cls(
            Dia=Dia,
            conservation=conservation,
            sca_matrix=sca_matrix,
            phi_ia=phi_ia,
            fi0=fi0,
            fia=fia,
            Cijab_raw=Cijab_raw,
            fijab=fijab,
            evals_sca=evals_sca,
            evecs_sca=evecs_sca,
            significant_evals_sca=significant_evals_sca,
            significant_evecs_sca=significant_evecs_sca,
            kstar=kstar,
            kstar_identified=kstar_identified,
            cutoff=cutoff,
            evals_shuff=evals_shuff,
            v_ica=v_ica,
            w_ica=w_ica,
            groups=groups,
            group_scores=group_scores,
            t_dists_info=t_dists_info,
            statsectors_msa=statsectors_msa,
            statsectors_seq=statsectors_seq,
            sca_matrix_sector_subset=sca_matrix_sector_subset,
            args=args,
        )


def _load_npy_or_none(path):
    """Load a .npy file if it exists, otherwise return None."""
    if os.path.isfile(path):
        return np.load(path)
    return None


def _load_scalar_txt(path, dtype):
    """Load a single scalar from a text file, or return None."""
    if os.path.isfile(path):
        return dtype(np.loadtxt(path))
    return None
