"""SCA Preprocessing

"""

import numpy as np
from numpy.typing import NDArray
from collections import Counter
import tqdm
import scipy.sparse as sp
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import torch
torch.set_float32_matmul_precision("high")

from mysca.mappings import SymMap, DEFAULT_MAP
from mysca.helpers import iterblocks


def preprocess_msa(
        msa: NDArray[np.int_], 
        seqids: list[str], 
        mapping: SymMap = DEFAULT_MAP, 
        *, 
        gap_truncation_thresh: float = 0.4,
        sequence_gap_thresh: float = 0.2, 
        reference_id: str = None,
        reference_similarity_thresh: float = 0.2,
        sequence_similarity_thresh: float = 0.8,
        position_gap_thresh: float = 0.2, 
        use_pbar: bool = False,
        verbosity: int = 1, 
        weight_computation_version: str = "v5",
        block_size: int = 1024
):
    """Run preprocessing steps on a given MSA matrix.

    Ref [1] Rivoire et al. 2016. https://doi.org/10.1371/journal.pcbi.1004817

    Args:
        (NDArray[np.int_]) msa: MSA object.
        (list[str]) seqids: IDs of sequences in the MSA.
        (SymMap) mapping: SymMap mapping symbols to integer values.
        (float) gap_truncation_thresh: Freq of gaps τ above which a position 
            (i.e. column) is removed for excessive gaps. Default 0.4.
        (float) sequence_gap_thresh: Freq of gaps γ_seq above which a sequence 
            (i.e. row) is removed. Default 0.2.
        (str) reference_id: ID of the reference sequence, or None.
        (float) reference_similarity_thresh: Identity threshold Δ below which
            sequences are excluded for not being close enough to the reference.
            Default 0.2.
        (float) sequence_similarity_thresh: Identity threshold δ above which 
            sequences are clustered together for weighting purposes. Default 0.8.
        (float) position_gap_thresh: Freq of gaps γ_pos above which a position 
            (i.e. column) is removed after weighting. Default 0.2.
        (bool) use_pbar: show progress bar for scans over sequences.
        (int) verbosity: verbosity level. Default 1.
        (str) weight_computation_version: String to identify which approach to use
            for computing sequence weights.
        (int) block_size: block size to use for computing sequence weights.
    
    Returns:
        (MultSeqAlignment) processed MSA.
        (dict[str, *]) dictionary mapping following keys to results:
            msa_binary3d: (NDArray[bool]) boolean MSA matrix after processing.
            retained_sequences: (NDArray[int]) retained sequences.
            retained_positions: (NDArray[int]) retained positions.
            retained_sequences_ids: (list[str]) retained sequence IDs.
            sequence_weights: (NDArray[float]) sequence weights.
            fi0_pretruncation: (NDArray[float]) gap frequency fi0.
            reference_results: (dict): reference similarity results. If a 
                reference ID is specified, contains keys reference_id, ref_idx, 
                and ref_similarity.
    """

    args = {
        "gap_truncation_thresh": gap_truncation_thresh,
        "sequence_gap_thresh": sequence_gap_thresh,
        "reference_id": reference_id,
        "reference_similarity_thresh": reference_similarity_thresh,
        "sequence_similarity_thresh": sequence_similarity_thresh,
        "position_gap_thresh": position_gap_thresh,
    }
    
    if verbosity:
        print("Preprocessing with parameters:")
        print(f"  gap_truncation_thresh τ={gap_truncation_thresh}")
        print(f"  sequence_gap_thresh γ_seq={sequence_gap_thresh}")
        print(f"  reference_id: {reference_id}")
        print(f"  reference_similarity_thresh Δ={reference_similarity_thresh}")
        print(f"  sequence_similarity_thresh δ={sequence_similarity_thresh}")
        print(f"  position_gap_thresh γ_pos={position_gap_thresh}")
    
    msa_orig = msa
    msa = msa_orig.copy()
    seqids_orig = seqids
    seqids = seqids_orig.copy()
    num_seqs, num_pos = msa_orig.shape

    if not isinstance(msa_orig, np.ndarray):
        raise RuntimeError(
            f"Input MSA should be an NDArray. Got {type(msa_orig)}"
        )
    if not isinstance(msa_orig[0,0], np.int_):
        raise RuntimeError(
            f"Input MSA should be an NDArray of ints. Got {type(msa_orig[0,0])}"
        )

    NUM_SYMS = len(mapping)
    NUM_AAS = NUM_SYMS - 1
    GAP = mapping.gapint

    # Track which rows and columns will be kept
    retained_sequences = np.arange(num_seqs)
    retained_positions = np.arange(num_pos)

    # Constuct the boolean MSA matrix
    xmsa = np.eye(NUM_SYMS, dtype=bool)[msa][:,:,:-1]
    xmsa = xmsa.astype(np.int16)

    #~~~ Remove columns (i.e. positions) with too many gaps
    if verbosity:
        print("Removing positions with too many gaps...")
    gapfreqs = np.sum(msa == GAP, axis=0) / msa.shape[0]
    screen = gapfreqs < gap_truncation_thresh
    msa = msa[:,screen]  # keep columns with gap freq < gap_truncation_thresh
    xmsa = xmsa[:,screen,:]
    retained_positions = retained_positions[screen]
    if verbosity:
        print(f"Filtered {np.sum(~screen)} positions at threshold τ={gap_truncation_thresh}.")
        print(f"  MSA shape: {msa.shape} (sequences x positions)")
    assert len(retained_positions) == msa.shape[1], "Mismatch"

    #~~~ Remove rows (i.e. sequences) with too many gaps
    if verbosity:
        print("Removing sequences with too many gaps...")
    gapfreqs = np.sum(msa == GAP, axis=1) / msa.shape[1]
    screen = gapfreqs < sequence_gap_thresh
    msa = msa[screen,:]  # keep rows with gap freq < sequence_gap_thresh
    xmsa = xmsa[screen,:,:]
    retained_sequences = retained_sequences[screen]
    seqids = np.array([seqids_orig[i] for i in retained_sequences])
    if verbosity:
        print(f"Filtered {np.sum(~screen)} sequences at threshold γ_seq={sequence_gap_thresh}.")
        print(f"  MSA shape: {msa.shape} (sequences x positions)")
    assert len(retained_sequences) == msa.shape[0], "Mismatch"

    #~~~ Compare with reference, if specified
    if reference_id:
        ref_idx = np.where(seqids == reference_id)[0][0]
        if verbosity:
            print(f"Found reference seq {reference_id} at position {ref_idx}.")
        refrow = msa[ref_idx,:]
        ref_similarity = np.sum(msa == refrow, axis=1) / msa.shape[1]
        ref_results = {}
        ref_results["reference_id"] = reference_id
        ref_results["ref_idx"] = ref_idx
        ref_results["ref_similarity"] = ref_similarity
        
        # Remove rows too dissimilar from the reference
        if verbosity:
            print("Removing sequences too dissimilar from reference...")
        screen = ref_similarity >= reference_similarity_thresh
        msa = msa[screen,:]  # keep rows with similarity >= reference_similarity_thresh
        xmsa = xmsa[screen,:,:]
        retained_sequences = retained_sequences[screen]
        seqids = np.array([seqids_orig[i] for i in retained_sequences])
        if verbosity:
            print(f"Filtered {np.sum(~screen)} sequences at threshold Δ={reference_similarity_thresh}.")
            print(f"  MSA shape: {msa.shape} (sequences x positions)")
        assert len(retained_sequences) == msa.shape[0], "Mismatch"
    else:
        ref_results = {}

    #~~~ Compute sequence weights
    if verbosity:
        print(f"Computing weights with version: {weight_computation_version}")
        print(f"Computing sequence weights (round 1)...")

    ws = compute_weights(
        version=weight_computation_version,
        msa=msa, 
        xmsa=xmsa, 
        seqsim_thresh=sequence_similarity_thresh,
        block_size=block_size,
        use_pbar=use_pbar,
        gap=GAP,
        num_aas=NUM_AAS
    )

    #~~~ Remove positions with too many (weighted) gaps
    if verbosity:
        print("Removing positions with too many (weighted) gaps...")
    fi0 = np.sum(ws[:,None] * (msa == GAP), axis=0) / ws.sum()
    screen = fi0 < position_gap_thresh
    msa = msa[:,screen]
    xmsa = xmsa[:,screen,:]
    retained_positions = retained_positions[screen]
    if verbosity:
        print(f"Filtered {np.sum(~screen)} positions at threshold γ_pos={position_gap_thresh}.")
        print(f"  MSA shape: {msa.shape} (sequences x positions)")
    assert len(retained_positions) == msa.shape[1], "Mismatch"

    #~~~ Re-compute sequence weights
    if verbosity:
        print("Computing sequence weights (round 2)...")

    ws = compute_weights(
        version=weight_computation_version,
        msa=msa, 
        xmsa=xmsa, 
        seqsim_thresh=sequence_similarity_thresh,
        block_size=block_size,
        use_pbar=use_pbar,
        gap=GAP,
        num_aas=NUM_AAS
    )
    
    if verbosity:
        print(f"Effective sample size (sum of weights): {ws.sum()}")

    preprocessing_results = {
        "msa_binary3d": xmsa.astype(int),
        "retained_sequences": retained_sequences,
        "retained_positions": retained_positions,
        "retained_sequence_ids": seqids,
        "sequence_weights": ws,
        "fi0_pretruncation": fi0, 
        "reference_results": ref_results,
        "args": args,
    }

    return msa, preprocessing_results


def compute_background_freqs(msa_obj, gapstr="-"):
    background_freqs = {}
    for entry in msa_obj:
        seq = str(entry.seq)
        counts = Counter(seq)
        for k, v in counts.items():
            if k not in background_freqs:
                background_freqs[k] = v
            else:
                background_freqs[k] += v
    if gapstr in background_freqs:
        background_freqs.pop(gapstr)
    total = np.sum(background_freqs[k] for k in background_freqs.keys())
    for k in background_freqs:
        background_freqs[k] /= total
    return background_freqs


def compute_weights(version="v1", **kwargs):
    if version == "v1":
        return _compute_weights_v1(**kwargs)
    if version == "v2":
        return _compute_weights_v2(**kwargs)
    if version == "v3":
        return _compute_weights_v3(**kwargs)
    elif version == "v4":
        return _compute_weights_v4(**kwargs)
    elif version == "v5":
        return _compute_weights_v5(**kwargs)
    elif version == "v6":
        return _compute_weights_v6(**kwargs)
    elif version == "gpu":
        return _compute_weights_torch(**kwargs)
    else:
        raise RuntimeError(f"Weight computation {version} not found")


def _compute_weights_v1(**kwargs):
    raise RuntimeError("BUGGY VERSION OF WEIGHT COMPUTATION")
    msa = kwargs["msa"]
    use_pbar = kwargs["use_pbar"]
    seqsim_thresh = kwargs["seqsim_thresh"]
    nseqs = msa.shape[0]
    npos = msa.shape[1]
    ws = np.nan * np.ones(nseqs)
    for i, s in tqdm.tqdm(enumerate(msa), total=nseqs, disable=not use_pbar):
        similarities = np.sum(s == msa, axis=1) / npos
        screen = similarities >= seqsim_thresh
        ws[i] = 1 / screen.sum()
    return ws


def _compute_weights_v2(**kwargs):
    raise RuntimeError("BUGGY VERSION OF WEIGHT COMPUTATION")
    xmsa = kwargs["xmsa"]
    block_size = kwargs["block_size"]
    use_pbar = kwargs["use_pbar"]
    seqsim_thresh = kwargs["seqsim_thresh"]
    assert isinstance(xmsa[0,0,0], np.uint16), \
        f"Expected xmsa to have np.uint16 data. Got {xmsa.dtype}"
    nseqs = xmsa.shape[0]
    npos = xmsa.shape[1]
    nalph = xmsa.shape[2]
    xmsa = xmsa.reshape([nseqs, -1])
    ws = np.nan * np.ones(nseqs)
    for idx1_start, idx1_stop, block1 in iterblocks(xmsa, block_size, use_pbar=use_pbar):
        block_sims = (xmsa @ block1.T / npos).T
        assert block_sims.shape == (len(block1), nseqs), \
            f"Expected {(len(block1), nseqs)}. Got {block_sims.shape}"
        block_screen = block_sims >= seqsim_thresh
        ws[idx1_start:idx1_stop] = 1 / block_screen.sum(axis=1)
    return ws


def _compute_weights_v3(**kwargs):
    msa = kwargs["msa"]
    block_size = kwargs["block_size"]
    use_pbar = kwargs["use_pbar"]
    seqsim_thresh = kwargs["seqsim_thresh"]
    gap = kwargs["gap"]
    assert isinstance(msa[0,0], (np.int_)), \
        f"Expected msa to have int data. Got {msa.dtype}"
    nseqs = msa.shape[0]
    npos = msa.shape[1]
    ws = np.nan * np.ones(nseqs)
    for idx1_start, idx1_stop, block1 in iterblocks(msa, block_size, use_pbar=use_pbar):
        # Compute pairwise similarity between sequences in block and all sequences in msa
        block_sims = (
            (block1[:,None,:] == msa[None,:,:]) #& (block1[:,None,:] != gap) & (msa[None,:,:] != gap)
        ).sum(axis=2) / npos
        rows = np.arange(len(block1))
        cols = idx1_start + rows
        block_sims[rows, cols] = 1.0
        assert block_sims.shape == (len(block1), nseqs), f"Expected {(len(block1), nseqs)}. Got {block_sims.shape}"
        block_screen = block_sims >= seqsim_thresh
        ws[idx1_start:idx1_stop] = 1 / block_screen.sum(axis=1)
    return ws


def _compute_weights_v4(**kwargs):
    """
    Adapted from: 
        https://github.com/ranganathanlab/pySCA/blob/master/pysca/scaTools.py
    """
    msa = kwargs["msa"]
    block_size = kwargs["block_size"]
    use_pbar = kwargs["use_pbar"]
    seqsim_thresh = kwargs["seqsim_thresh"]
    gap = kwargs["gap"]
    num_aas = kwargs["num_aas"]
    assert isinstance(msa[0,0], (np.int_)), \
        f"Expected msa to have int data. Got {msa.dtype}"
    nseqs = msa.shape[0]
    npos = msa.shape[1]
    ws = np.nan * np.ones(nseqs)
    msa_sparse = get_onehotmsa_sparse(msa, num_aas, gap)
    neigh = np.zeros(nseqs, dtype=np.uint32)

    # blockwise similarity counting
    # CRITICAL: Keep sparse operations to avoid memory explosion for large MSAs
    # For 300k seqs, even block_size=1024 creates 1024×300k dense = 1.2GB per block
    # Solution: Use sparse operations and only extract row sums we need
    for i0 in tqdm.trange(0, nseqs, block_size, disable=not use_pbar):
        i1 = min(i0 + block_size, nseqs)
        # Compute sparse dot product (result is sparse)
        counts_sparse = msa_sparse[i0:i1] @ msa_sparse.T  # Sparse matrix: block_size × nseqs
        # Convert to dense only for the threshold comparison (much smaller)
        # But we can be smarter: only compute what we need
        sim_sparse = counts_sparse / float(npos)

        # For each row, count how many values > max_seqid
        # Use sparse matrix operations to avoid full dense conversion
        for local_idx, global_idx in enumerate(tqdm.trange(i0, i1, disable=not use_pbar, leave=False)):
            row_data = sim_sparse[local_idx, :].toarray().flatten()
            neigh[global_idx] = (row_data >= seqsim_thresh).sum()

    # Avoid division by zero (shouldn't happen because self-similarity is 1.0 > max_seqid)
    # But handle edge cases where neigh might be 0 (very dissimilar sequences)
    neigh_float = neigh.astype(float)
    # Replace 0 with 1 to avoid infinity (gives those sequences weight 1.0)
    neigh_float[neigh_float == 0] = 1.0
    ws = 1.0 / neigh_float
    return ws


def _compute_weights_v5(**kwargs):
    """
    Adapted from: 
        https://github.com/ranganathanlab/pySCA/blob/master/pysca/scaTools.py

    With faster sparse operations directly on data.
    """
    msa = kwargs["msa"]
    block_size = kwargs["block_size"]
    use_pbar = kwargs["use_pbar"]
    seqsim_thresh = kwargs["seqsim_thresh"]
    gap = kwargs["gap"]
    num_aas = kwargs["num_aas"]
    assert isinstance(msa[0,0], (np.int_)), \
        f"Expected msa to have int data. Got {msa.dtype}"
    nseqs = msa.shape[0]
    npos = msa.shape[1]
    ws = np.nan * np.ones(nseqs)
    msa_sparse = get_onehotmsa_sparse(msa, num_aas, gap)
    neigh = np.zeros(nseqs, dtype=np.uint32)

    thresh = seqsim_thresh * npos

    for i0 in tqdm.trange(0, nseqs, block_size, disable=not use_pbar):
        i1 = min(i0 + block_size, nseqs)
        counts_sparse = (msa_sparse[i0:i1] @ msa_sparse.T).tocsr()
        data = counts_sparse.data
        indptr = counts_sparse.indptr
        # Iterate rows but operate directly on sparse storage
        for r in range(i1 - i0):
            start, end = indptr[r], indptr[r + 1]
            neigh[i0 + r] = np.count_nonzero(data[start:end] >= thresh)
        # ----------------------------------------
        # --- Alternative to the inner loop:
        # mask = counts_sparse.data >= thresh
        # row_counts = np.add.reduceat(mask, counts_sparse.indptr[:-1])
        # neigh[i0:i1] = row_counts
        # ----------------------------------------

    neigh_float = neigh.astype(float)
    neigh_float[neigh_float == 0] = 1.0
    ws = 1.0 / neigh_float

    return ws


def _compute_weights_v6(**kwargs):
    msa = kwargs["msa"]
    block_size = kwargs["block_size"]
    use_pbar = kwargs["use_pbar"]
    seqsim_thresh = kwargs["seqsim_thresh"]
    gap = kwargs["gap"]
    num_aas = kwargs["num_aas"]

    assert isinstance(msa[0,0], (np.int_)), \
        f"Expected msa to have int data. Got {msa.dtype}"

    nseqs, npos = msa.shape
    thresh = seqsim_thresh * npos

    msa_sparse = get_onehotmsa_sparse(msa, num_aas, gap)

    neigh = np.zeros(nseqs, dtype=np.uint32)

    # JAX compiled row counting
    @jax.jit
    def count_rows(data, indptr):
        mask = data >= thresh
        mask = mask.astype(jnp.int32)

        # compute row sums from CSR structure
        row_counts = jnp.add.reduceat(mask, indptr[:-1])
        return row_counts

    for i0 in tqdm.trange(0, nseqs, block_size, disable=not use_pbar):
        i1 = min(i0 + block_size, nseqs)

        counts_sparse = (msa_sparse[i0:i1] @ msa_sparse.T).tocsr()

        data = jnp.array(counts_sparse.data)
        indptr = jnp.array(counts_sparse.indptr)

        row_counts = count_rows(data, indptr)

        neigh[i0:i1] = np.array(row_counts)

    neigh_float = neigh.astype(float)
    neigh_float[neigh_float == 0] = 1.0
    ws = 1.0 / neigh_float

    return ws

def _detect_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


def _compute_weights_torch(**kwargs):

    msa = kwargs["msa"]
    block_size = kwargs.get("block_size", 512)
    use_pbar = kwargs["use_pbar"]
    seqsim_thresh = kwargs["seqsim_thresh"]
    gap = kwargs["gap"]

    assert isinstance(msa[0, 0], np.integer), \
        f"Expected msa to have int data. Got {msa.dtype}"

    nseqs, npos = msa.shape
    thresh = seqsim_thresh * npos

    device = _detect_device()
    if device == "cpu":
        print("No device found. Reverting to version v5!")
        return _compute_weights_v5(**kwargs)

    # move msa to torch
    msa_t = torch.as_tensor(msa, dtype=torch.int16, device=device)
    neigh = torch.zeros(nseqs, dtype=torch.int32, device=device)
    outer = range(0, nseqs, block_size)
    if use_pbar:
        outer = tqdm.trange(0, nseqs, block_size)

    for i0 in outer:
        i1 = min(i0 + block_size, nseqs)
        block_i = msa_t[i0:i1]
        counts = torch.zeros(i1 - i0, dtype=torch.int32, device=device)
        for j0 in range(0, nseqs, block_size):
            j1 = min(j0 + block_size, nseqs)
            block_j = msa_t[j0:j1]
            # broadcast comparison
            matches = (
                block_i[:, None, :] == block_j[None, :, :]
                # & (block_i[:, None, :] != gap)
                # & (block_j[None, :, :] != gap)
            ).sum(dim=2)
            counts += (matches >= thresh).sum(dim=1)
        neigh[i0:i1] = counts[:]

    neigh_cpu = neigh.cpu().numpy().astype(float)
    del neigh
    neigh_cpu[neigh_cpu == 0] = 1.0

    ws = 1.0 / neigh_cpu
    return ws


def get_onehotmsa_sparse(msa, num_aa, gap):
    """
    Convert a numeric alignment (Nseq x Npos) to a sparse binary one-hot matrix,
    including the gap as its own symbol in the one-hot encoding.

    Parameters
    ----------
    msa : np.ndarray, shape (Nseq, Npos)
        Numeric alignment where amino acids are 0..num_aa-1 and num_aa means "other"/gap.
    num_aa : int
        Number of amino-acid states (not counting the GAP state).
    gap: int
        Integer value representing gaps.

    Returns
    -------
    Abin : scipy.sparse.csr_matrix, shape (Nseq, (num_aa + 1) * Npos)
        One-hot encoding (sparse), with gaps encoded as their own symbol.
    """
    msa = np.asarray(msa)
    if msa.ndim != 2:
        raise ValueError("get_onehotmsa_sparse expects a 2D array (Nseq x Npos).")
    if gap != num_aa:
        msg = "get_onehotmsa_sparse expects gap == num_aa."
        msg += f"Got gap={gap} when num_aas={num_aa}"
        raise ValueError(msg)
    num_symbols = num_aa + 1  # include gap
    nseqs, npos = msa.shape
    a = msa.astype(np.int8, copy=False).ravel()
    rows = np.repeat(np.arange(nseqs, dtype=np.uint16), npos)
    pos = np.tile(np.arange(npos, dtype=np.uint16), nseqs)
    cols = pos * num_symbols + a
    data = np.ones(cols.shape[0], dtype=np.int16)
    onehotmsa = sp.csr_matrix(
        (data, (rows, cols)),
        shape=(nseqs, num_symbols * npos)
    )
    return onehotmsa


def get_onehotmsa_sparse_nogap(msa, num_aa, gap):
    """
    Convert a numeric alignment (Nseq x Npos) to a sparse binary one-hot matrix,
    where gaps are treated as missing, and not as a one-hot encoded symbol.

    Parameters
    ----------
    msa : np.ndarray, shape (Nseq, Npos)
        Numeric alignment where amino acids are 0..num_aa-1 and num_aa means "other"/gap.
    num_aa : int
        Number of amino-acid states (not counting the GAP state).
    gap: int
        Integer value representing gaps.

    Returns
    -------
    Abin : scipy.sparse.csr_matrix, shape (Nseq, num_aa * Npos)
        One-hot encoding (sparse).
    """
    msa = np.asarray(msa)
    if msa.ndim != 2:
        raise ValueError("get_onehotmsa_sparse expects a 2D array (Nseq x Npos).")
    if gap != num_aa:
        msg = "get_onehotmsa_sparse expects gap == num_aa."
        msg += f"Got gap={gap} when num_aas={num_aa}"
        raise ValueError(msg)
    nseqs, npos = msa.shape
    a = msa.astype(np.int8, copy=False).ravel()
    rows = np.repeat(np.arange(nseqs, dtype=np.uint16), npos)
    pos = np.tile(np.arange(npos, dtype=np.uint16), nseqs)
    mask = a < num_aa
    cols = pos[mask] * num_aa + a[mask]
    data = np.ones(cols.shape[0], dtype=np.int16)

    onehotmsa = sp.csr_matrix(
        (data, (rows[mask], cols)), 
        shape=(nseqs, num_aa * npos)
    )
    return onehotmsa
