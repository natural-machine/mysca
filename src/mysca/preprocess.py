"""SCA Preprocessing

"""

import numpy as np
from numpy.typing import NDArray
from collections import Counter
import tqdm

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
        print("Computing sequence weights (round 1)...")
    # ws = np.nan * np.ones(msa.shape[0])
    # for i, s in tqdm.tqdm(enumerate(msa), total=len(msa), disable=not use_pbar):
    #     similarities = np.sum(s == msa, axis=1) / msa.shape[1]
    #     screen = similarities >= sequence_similarity_thresh
    #     ws[i] = 1 / screen.sum()
    
    weight_comp_version = "v3"
    block_size = 100

    if weight_comp_version == "v1":
        ws = compute_weights(
            version="v1", 
            msa=msa, use_pbar=use_pbar, 
            seqsim_thresh=sequence_similarity_thresh
        )
    elif weight_comp_version == "v2":
        ws = compute_weights(
            version="v2",
            xmsa=xmsa, use_pbar=use_pbar, 
            seqsim_thresh=sequence_similarity_thresh,
            block_size=block_size
        )
    elif weight_comp_version == "v3":
        ws = compute_weights(
            version="v3",
            msa=msa, use_pbar=use_pbar, 
            seqsim_thresh=sequence_similarity_thresh,
            block_size=block_size
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
    # ws = np.nan * np.ones(msa.shape[0])
    # for i, s in tqdm.tqdm(enumerate(msa), total=len(msa), disable=not use_pbar):
    #     similarities = np.sum(s == msa, axis=1) / msa.shape[1]
    #     screen = similarities >= sequence_similarity_thresh
    #     ws[i] = 1 / screen.sum()
    if weight_comp_version == "v1":
        ws = compute_weights(
            version="v1", 
            msa=msa, use_pbar=use_pbar, seqsim_thresh=sequence_similarity_thresh
        )
    elif weight_comp_version == "v2":
        ws = compute_weights(
            version="v2",
            xmsa=xmsa, use_pbar=use_pbar, 
            seqsim_thresh=sequence_similarity_thresh,
            block_size=block_size
        )
    elif weight_comp_version == "v3":
        ws = compute_weights(
            version="v3",
            msa=msa, use_pbar=use_pbar, 
            seqsim_thresh=sequence_similarity_thresh,
            block_size=block_size
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
    else:
        raise RuntimeError(f"Weight computation {version} not found")


def _compute_weights_v1(msa, seqsim_thresh, use_pbar):
    nseqs = msa.shape[0]
    npos = msa.shape[1]
    ws = np.nan * np.ones(nseqs)
    for i, s in tqdm.tqdm(enumerate(msa), total=nseqs, disable=not use_pbar):
        similarities = np.sum(s == msa, axis=1) / npos
        screen = similarities >= seqsim_thresh
        ws[i] = 1 / screen.sum()
    return ws


def _compute_weights_v2(xmsa, seqsim_thresh, use_pbar, block_size=1000):
    assert isinstance(xmsa[0,0,0], np.uint16), \
        f"Expected xmsa to have np.uint16 data. Got {xmsa.dtype}"
    nseqs = xmsa.shape[0]
    npos = xmsa.shape[1]
    nalph = xmsa.shape[2]
    xmsa = xmsa.reshape([nseqs, -1])
    ws = np.nan * np.ones(nseqs)
    for idx1_start, idx1_stop, block1 in iterblocks(xmsa, block_size, use_pbar=use_pbar):
        block_sims = (xmsa @ block1.T / npos).T
        assert block_sims.shape == (len(block1), nseqs), f"Expected {(len(block1), nseqs)}. Got {block_sims.shape}"
        block_screen = block_sims >= seqsim_thresh
        ws[idx1_start:idx1_stop] = 1 / block_screen.sum(axis=1)
    return ws


def _compute_weights_v3(msa, seqsim_thresh, use_pbar, block_size=1000):
    assert isinstance(msa[0,0], (np.int_)), \
        f"Expected msa to have int data. Got {msa.dtype}"
    nseqs = msa.shape[0]
    npos = msa.shape[1]
    ws = np.nan * np.ones(nseqs)
    for idx1_start, idx1_stop, block1 in iterblocks(msa, block_size, use_pbar=use_pbar):
        # Compute pairwise similarity between sequences in block and all sequences in msa
        block_sims = (block1[:, None, :] == msa[None, :, :]).sum(axis=2) / npos
        assert block_sims.shape == (len(block1), nseqs), f"Expected {(len(block1), nseqs)}. Got {block_sims.shape}"
        block_screen = block_sims >= seqsim_thresh
        ws[idx1_start:idx1_stop] = 1 / block_screen.sum(axis=1)
    return ws
