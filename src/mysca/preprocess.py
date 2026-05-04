"""SCA Preprocessing

"""

import logging

import numpy as np
from numpy.typing import NDArray
from collections import Counter
import tqdm
import scipy.sparse as sp
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from mysca.mappings import SymMap, DEFAULT_MAP
from mysca.helpers import iterblocks

logger = logging.getLogger(__name__)


def _check_msa_nonempty_after_filter(msa, stage):
    """Raise a clear ValueError if a filter stage has zeroed the MSA.

    `stage` is the just-appended filter_history entry; we use its
    `label`, `threshold_symbol`, and `threshold` for the message.
    """
    n_seqs, n_pos = msa.shape
    if n_seqs > 0 and n_pos > 0:
        return
    axis = "sequences" if n_seqs == 0 else "positions"
    raise ValueError(
        f"Preprocessing filter '{stage['label']}' "
        f"({stage['threshold_symbol']}={stage['threshold']}) left "
        f"zero {axis}; MSA shape is now {msa.shape}. "
        f"Loosen this threshold or revisit upstream filtering."
    )


def onehot_without_gap(
        msa: NDArray[np.int_],
        num_syms: int,
        gapint: int,
) -> NDArray[np.bool_]:
    """Dense one-hot encoding of an integer MSA with the gap column removed.

    The resulting axis-2 ordering matches ``mapping.aa_list`` regardless of
    where the gap sits in ``sym_list``.
    """
    onehot = np.eye(num_syms, dtype=bool)[msa]
    return np.delete(onehot, gapint, axis=-1)


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
        weight_computation_version: str = "sparse",
        block_size: int = 1024,
        n_excluded_pre_load: int = 0,
        n_internal_stop_pre_load: int = 0,
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
        (int) n_excluded_pre_load: count of input sequences dropped *before*
            this call by ``mysca.io.load_msa`` because they contained excluded
            symbols (non-canonical AAs by default). When > 0, the
            ``"initial"`` filter_history stage's ``n_sequences`` is set to the
            pre-exclusion count and an ``"excluded_symbols"`` stage is
            inserted right after it. Defaults to 0 (no excluded-symbols
            stage recorded).
        (int) n_internal_stop_pre_load: count of input sequences dropped
            *before* this call by ``mysca.io.load_msa`` because they
            contained an internal stop codon (post-trailing-strip). When
            > 0, an ``"internal_stop_codon"`` filter_history stage is
            inserted between ``"initial"`` and ``"excluded_symbols"``
            (matching the order of operations in ``load_msa``).

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
            filter_history: (list[dict]) per-stage record of dataset size and
                the pre-filter statistic distribution with its threshold, used
                for diagnostic plotting. Each entry has keys: stage, label,
                n_sequences, n_positions, n_filtered, axis, stat_name,
                stat_values, threshold, threshold_symbol, filter_direction.
    """

    args = {
        "gap_truncation_thresh": gap_truncation_thresh,
        "sequence_gap_thresh": sequence_gap_thresh,
        "reference_id": reference_id,
        "reference_similarity_thresh": reference_similarity_thresh,
        "sequence_similarity_thresh": sequence_similarity_thresh,
        "position_gap_thresh": position_gap_thresh,
    }
    
    logger.info("Preprocessing with parameters:")
    logger.info("  gap_truncation_thresh τ=%s", gap_truncation_thresh)
    logger.info("  sequence_gap_thresh γ_seq=%s", sequence_gap_thresh)
    logger.info("  reference_id: %s", reference_id)
    logger.info(
        "  reference_similarity_thresh Δ=%s", reference_similarity_thresh
    )
    logger.info(
        "  sequence_similarity_thresh δ=%s", sequence_similarity_thresh
    )
    logger.info("  position_gap_thresh γ_pos=%s", position_gap_thresh)
    
    msa_loaded = msa
    msa = msa_loaded.copy()
    seqids_loaded = seqids
    seqids = seqids_loaded.copy()
    num_seqs, num_pos = msa_loaded.shape

    if not isinstance(msa_loaded, np.ndarray):
        raise RuntimeError(
            f"Input MSA should be an NDArray. Got {type(msa_loaded)}"
        )
    if not isinstance(msa_loaded[0,0], np.int_):
        raise RuntimeError(
            f"Input MSA should be an NDArray of ints. Got {type(msa_loaded[0,0])}"
        )

    NUM_SYMS = len(mapping)
    NUM_AAS = NUM_SYMS - 1
    GAP = mapping.gapint

    # Track which rows and columns will be kept
    retained_sequences = np.arange(num_seqs)
    retained_positions = np.arange(num_pos)

    # Record dataset size and the pre-filter statistic at each stage.
    # When load_msa dropped sequences before this call (due to internal
    # stop codons or non-canonical / excluded symbols), the "initial"
    # bar represents the raw input size and sibling stages show the
    # drops in load_msa's actual order: internal_stop_codon first, then
    # excluded_symbols. Both stages carry no stat_values — by design;
    # the count alone is the useful signal, and
    # plot_filter_distributions iterates only stages with stat_values.
    n_excluded_pre_load = int(n_excluded_pre_load)
    n_internal_stop_pre_load = int(n_internal_stop_pre_load)
    n_initial = num_seqs + n_excluded_pre_load + n_internal_stop_pre_load
    filter_history = [{
        "stage": "initial",
        "label": "initial",
        "n_sequences": n_initial,
        "n_positions": num_pos,
        "n_filtered": 0,
        "axis": None,
        "stat_name": None,
        "stat_values": None,
        "threshold": None,
        "threshold_symbol": None,
        "filter_direction": None,
    }]
    n_running = n_initial
    if n_internal_stop_pre_load > 0:
        n_running -= n_internal_stop_pre_load
        filter_history.append({
            "stage": "internal_stop_codon",
            "label": "internal stop codon",
            "n_sequences": n_running,
            "n_positions": num_pos,
            "n_filtered": n_internal_stop_pre_load,
            "axis": "sequences",
            "stat_name": None,
            "stat_values": None,
            "threshold": None,
            "threshold_symbol": None,
            "filter_direction": None,
        })
    if n_excluded_pre_load > 0:
        n_running -= n_excluded_pre_load
        filter_history.append({
            "stage": "excluded_symbols",
            "label": "excluded symbols",
            "n_sequences": n_running,
            "n_positions": num_pos,
            "n_filtered": n_excluded_pre_load,
            "axis": "sequences",
            "stat_name": None,
            "stat_values": None,
            "threshold": None,
            "threshold_symbol": None,
            "filter_direction": None,
        })

    # Constuct the boolean MSA matrix
    xmsa = onehot_without_gap(msa, NUM_SYMS, GAP)
    xmsa = xmsa.astype(np.int16)

    #~~~ Remove columns (i.e. positions) with too many gaps
    logger.info("Removing positions with too many gaps...")
    gapfreqs = np.sum(msa == GAP, axis=0) / msa.shape[0]
    screen = gapfreqs < gap_truncation_thresh
    msa = msa[:,screen]  # keep columns with gap freq < gap_truncation_thresh
    xmsa = xmsa[:,screen,:]
    retained_positions = retained_positions[screen]
    filter_history.append({
        "stage": "position_gap",
        "label": "position gap (τ)",
        "n_sequences": msa.shape[0],
        "n_positions": msa.shape[1],
        "n_filtered": int(np.sum(~screen)),
        "axis": "positions",
        "stat_name": "gap frequency per position",
        "stat_values": gapfreqs.copy(),
        "threshold": gap_truncation_thresh,
        "threshold_symbol": "τ",
        "filter_direction": "above",
    })
    logger.info(
        "Filtered %d positions with gap frequency ≥ τ (%s).",
        int(np.sum(~screen)), gap_truncation_thresh,
    )
    logger.info("  MSA shape: %s (sequences x positions)", msa.shape)
    assert len(retained_positions) == msa.shape[1], "Mismatch"
    _check_msa_nonempty_after_filter(msa, filter_history[-1])

    #~~~ Remove rows (i.e. sequences) with too many gaps
    logger.info("Removing sequences with too many gaps...")
    gapfreqs = np.sum(msa == GAP, axis=1) / msa.shape[1]
    screen = gapfreqs < sequence_gap_thresh
    msa = msa[screen,:]  # keep rows with gap freq < sequence_gap_thresh
    xmsa = xmsa[screen,:,:]
    retained_sequences = retained_sequences[screen]
    seqids = np.array([seqids_loaded[i] for i in retained_sequences])
    filter_history.append({
        "stage": "sequence_gap",
        "label": "sequence gap (γ_seq)",
        "n_sequences": msa.shape[0],
        "n_positions": msa.shape[1],
        "n_filtered": int(np.sum(~screen)),
        "axis": "sequences",
        "stat_name": "gap frequency per sequence",
        "stat_values": gapfreqs.copy(),
        "threshold": sequence_gap_thresh,
        "threshold_symbol": "γ_seq",
        "filter_direction": "above",
    })
    logger.info(
        "Filtered %d sequences with gap frequency ≥ γ_seq (%s).",
        int(np.sum(~screen)), sequence_gap_thresh,
    )
    logger.info("  MSA shape: %s (sequences x positions)", msa.shape)
    assert len(retained_sequences) == msa.shape[0], "Mismatch"
    _check_msa_nonempty_after_filter(msa, filter_history[-1])

    #~~~ Compare with reference, if specified
    if reference_id:
        ref_matches = np.where(seqids == reference_id)[0]
        if ref_matches.size == 0:
            in_loaded = reference_id in seqids_loaded
            stage_hint = (
                "It was present in the input MSA but was dropped by an "
                "earlier preprocessing filter "
                "(internal_stop_codon, excluded_symbols, or sequence_gap)."
                if in_loaded
                else "It is not present in the input MSA."
            )
            raise ValueError(
                f"Reference sequence ID {reference_id!r} not found in the "
                f"MSA after early-stage filtering. {stage_hint} "
                f"Pick a reference present in the post-filter MSA, or "
                f"loosen --sequence_gap_thresh / --syms accordingly."
            )
        ref_idx = ref_matches[0]
        logger.info(
            "Found reference seq %s at position %d.", reference_id, ref_idx
        )
        refrow = msa[ref_idx,:]
        ref_similarity = np.sum(msa == refrow, axis=1) / msa.shape[1]
        ref_results = {}
        ref_results["reference_id"] = reference_id
        ref_results["ref_idx"] = ref_idx
        ref_results["ref_similarity"] = ref_similarity

        # Remove rows too dissimilar from the reference
        logger.info("Removing sequences too dissimilar from reference...")
        screen = ref_similarity >= reference_similarity_thresh
        msa = msa[screen,:]  # keep rows with similarity >= reference_similarity_thresh
        xmsa = xmsa[screen,:,:]
        retained_sequences = retained_sequences[screen]
        seqids = np.array([seqids_loaded[i] for i in retained_sequences])
        filter_history.append({
            "stage": "reference_similarity",
            "label": "reference similarity (Δ)",
            "n_sequences": msa.shape[0],
            "n_positions": msa.shape[1],
            "n_filtered": int(np.sum(~screen)),
            "axis": "sequences",
            "stat_name": "similarity to reference",
            "stat_values": ref_similarity.copy(),
            "threshold": reference_similarity_thresh,
            "threshold_symbol": "Δ",
            "filter_direction": "below",
        })
        logger.info(
            "Filtered %d sequences with similarity to reference < Δ (%s).",
            int(np.sum(~screen)), reference_similarity_thresh,
        )
        logger.info("  MSA shape: %s (sequences x positions)", msa.shape)
        assert len(retained_sequences) == msa.shape[0], "Mismatch"
        _check_msa_nonempty_after_filter(msa, filter_history[-1])
    else:
        ref_results = {}

    #~~~ Compute sequence weights
    logger.info(
        "Computing weights with version: %s", weight_computation_version
    )
    logger.info("Computing sequence weights (round 1)...")

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
    logger.info("Removing positions with too many (weighted) gaps...")
    fi0 = np.sum(ws[:,None] * (msa == GAP), axis=0) / ws.sum()
    screen = fi0 < position_gap_thresh
    msa = msa[:,screen]
    xmsa = xmsa[:,screen,:]
    retained_positions = retained_positions[screen]
    filter_history.append({
        "stage": "position_weighted_gap",
        "label": "weighted position gap (γ_pos)",
        "n_sequences": msa.shape[0],
        "n_positions": msa.shape[1],
        "n_filtered": int(np.sum(~screen)),
        "axis": "positions",
        "stat_name": "weighted gap frequency per position",
        "stat_values": fi0.copy(),
        "threshold": position_gap_thresh,
        "threshold_symbol": "γ_pos",
        "filter_direction": "above",
    })
    logger.info(
        "Filtered %d positions with weighted gap frequency ≥ γ_pos (%s).",
        int(np.sum(~screen)), position_gap_thresh,
    )
    logger.info("  MSA shape: %s (sequences x positions)", msa.shape)
    assert len(retained_positions) == msa.shape[1], "Mismatch"
    _check_msa_nonempty_after_filter(msa, filter_history[-1])

    #~~~ Re-compute sequence weights
    logger.info("Computing sequence weights (round 2)...")

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
    
    logger.info("Effective sample size (sum of weights): %s", ws.sum())

    preprocessing_results = {
        "msa_binary3d": xmsa.astype(int),
        "retained_sequences": retained_sequences,
        "retained_positions": retained_positions,
        "retained_sequence_ids": seqids,
        "sequence_weights": ws,
        "fi0_pretruncation": fi0,
        "reference_results": ref_results,
        "args": args,
        "filter_history": filter_history,
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


def compute_weights(version="sparse", **kwargs):
    """Dispatch to a sequence-weight implementation by version string.

    Production methods (exposed via the sca-preprocess CLI):
      - ``"sparse"``: CPU sparse-CSR dot-product (default).
      - ``"gpu"``: torch GPU (CUDA/MPS/XPU); falls back to ``"sparse"`` if
        no accelerator is detected.

    Non-production methods (kept for benchmarking and correctness tests;
    leading underscore in the version string signals "not intended for
    routine use"):
      - ``"_v3"``: naive O(N²) pairwise comparison, blockwise.
      - ``"_v4"``: sparse dot-product with a dense threshold per row.
      - ``"_v6"``: JAX-compiled CSR row counting.
    """
    if version == "sparse":
        return _compute_weights_sparse(**kwargs)
    elif version == "gpu":
        return _compute_weights_gpu(**kwargs)
    elif version == "_v3":
        return _compute_weights_v3(**kwargs)
    elif version == "_v4":
        return _compute_weights_v4(**kwargs)
    elif version == "_v6":
        return _compute_weights_v6(**kwargs)
    else:
        raise RuntimeError(f"Weight computation {version} not found")


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


def _compute_weights_sparse(**kwargs):
    """CPU sparse-CSR sequence-similarity weighting.

    Adapted from: https://github.com/ranganathanlab/pySCA/blob/master/pysca/scaTools.py
    with direct iteration over the sparse CSR data buffer to avoid
    materializing dense similarity blocks.
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

from mysca._acceleration import detect_device as _detect_device


def _compute_weights_gpu(**kwargs):
    """GPU-accelerated sequence-similarity weighting via torch.

    Uses the first available torch device (CUDA / MPS / XPU). Falls back
    to the CPU sparse implementation when no accelerator is detected.
    """
    import torch
    torch.set_float32_matmul_precision("high")

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
        logger.warning(
            "No GPU device found; falling back to CPU sparse weights."
        )
        return _compute_weights_sparse(**kwargs)

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
        Numeric alignment whose entries are integers in [0, num_aa], with
        exactly one integer (``gap``) reserved for gaps and the remaining
        ``num_aa`` integers denoting amino-acid states.
    num_aa : int
        Number of amino-acid states (not counting the GAP state).
    gap: int
        Integer value representing gaps. May occur at any position in
        [0, num_aa].

    Returns
    -------
    Abin : scipy.sparse.csr_matrix, shape (Nseq, (num_aa + 1) * Npos)
        One-hot encoding (sparse), with gaps encoded as their own symbol.
        Each (position, symbol) pair maps to a unique column.
    """
    msa = np.asarray(msa)
    if msa.ndim != 2:
        raise ValueError("get_onehotmsa_sparse expects a 2D array (Nseq x Npos).")
    if not (0 <= gap <= num_aa):
        raise ValueError(
            f"gap must be in [0, num_aa={num_aa}]; got gap={gap}"
        )
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
        Numeric alignment whose entries are integers in [0, num_aa], with
        exactly one integer (``gap``) reserved for gaps and the remaining
        ``num_aa`` integers denoting amino-acid states.
    num_aa : int
        Number of amino-acid states (not counting the GAP state).
    gap: int
        Integer value representing gaps. May occur at any position in
        [0, num_aa].

    Returns
    -------
    Abin : scipy.sparse.csr_matrix, shape (Nseq, num_aa * Npos)
        One-hot encoding (sparse). The ``num_aa`` columns per position follow
        the order of the amino-acid symbols with the gap integer excised (i.e.
        input value ``a`` maps to AA-index ``a`` if ``a < gap`` and
        ``a - 1`` if ``a > gap``).
    """
    msa = np.asarray(msa)
    if msa.ndim != 2:
        raise ValueError("get_onehotmsa_sparse expects a 2D array (Nseq x Npos).")
    if not (0 <= gap <= num_aa):
        raise ValueError(
            f"gap must be in [0, num_aa={num_aa}]; got gap={gap}"
        )
    nseqs, npos = msa.shape
    a = msa.astype(np.int8, copy=False).ravel()
    rows = np.repeat(np.arange(nseqs, dtype=np.uint16), npos)
    pos = np.tile(np.arange(npos, dtype=np.uint16), nseqs)
    mask = a != gap
    a_masked = a[mask]
    aa_idx = a_masked - (a_masked > gap).astype(a_masked.dtype)
    cols = pos[mask] * num_aa + aa_idx
    data = np.ones(cols.shape[0], dtype=np.int16)

    onehotmsa = sp.csr_matrix(
        (data, (rows[mask], cols)),
        shape=(nseqs, num_aa * npos)
    )
    return onehotmsa
