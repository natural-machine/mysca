"""Helper functions

"""

import math
import tqdm
import numpy as np
from numpy.typing import NDArray
from Bio.Align import MultipleSeqAlignment


def get_top_k_conserved_retained_positions(
        retained_positions: NDArray[np.int_], 
        Di: NDArray[np.float64], 
        k: int,
) -> tuple[NDArray[np.int_], NDArray[np.float64]]:
    """Positional indices and Di values among the top-k most conserved.

    Args:
        retained_positions (NDArray): _description_
        Di (NDArray): _description_
        k (_type_): _description_

    Returns:
        NDArray[int]: Top conserved positions in the MSA with frequencies Di.
        NDArray[float]: Top conserved values.
    """
    if isinstance(retained_positions, list):
        retained_positions = np.array(retained_positions, dtype=int)
    if isinstance(Di, list):
        Di = np.array(Di)
    top_conserved_idxs = np.flip(np.argsort(Di))[:k]
    top_conserved_positions = retained_positions[top_conserved_idxs]
    top_conserved_values = Di[top_conserved_idxs]
    return top_conserved_positions, top_conserved_values


def map_msa_positions_to_sequence(
        msa_sequence: str,
        raw_sequence: str = None,
        gapstr: str = "-",
) -> tuple[dict[int,int], dict[int,int]]:
    """Map non-gap positions in an MSA sequence to an original sequence.

    From conserved MSA indices, determine the corresponding residue index in the 
    original fasta file for the protein. That is, map MSA index to AA index per 
    protein.

    Args:
        (str) msa_sequence: The aligned MSA sequence, possibly containing gaps.
        (str) raw_sequence: The original, raw sequence. Optional. If given, 
            checks that the msa_sequence when stripped of gaps, matches.
    Returns:
        (dict[int,int]) msa2seq: Map from aligned sequence position to original.
        (dict[int,int]) seq2msa: The corresponding inverse map.
    """
    if raw_sequence is None:
        raw_sequence = msa_sequence.replace(gapstr, "")
    else:
        if msa_sequence.replace(gapstr, "") != raw_sequence:
            msg = "Aligned sequence does not match original!\n"
            msg += f" Aligned: {msa_sequence}\n"
            msg += f"Original: {raw_sequence}\n"
            raise RuntimeError(msg)
    
    aa_screen = np.array([c != gapstr for c in msa_sequence], dtype=bool)
    msa_idxs = np.arange(len(msa_sequence))[aa_screen]
    msa2seq = {int(i): int(j) for i, j in zip(msa_idxs, range(len(raw_sequence)))}
    seq2msa = {int(i): int(j) for i, j in zip(range(len(raw_sequence)), msa_idxs)}    
    return msa2seq, seq2msa


def get_rawseq_indices_of_msa(
        msa_obj: MultipleSeqAlignment,
        gapstr: str = "-",
) -> NDArray[np.int_]:
    """Get indices of non-gap positions with respect to the raw sequence.

    Args:
        msa_obj (MultipleSeqAlignment): MSA object.
        gapstr (str, optional): Gap string character. Defaults to "-".

    Returns:
        NDArray[np.int_]: Screen of the msa with -1 at gaps and positional 
            indices at amino acids. Same shape as the MultipleSeqAlignment.
    """
    nseqs = len(msa_obj)
    npos = len(msa_obj[0].seq)
    rawseq_idxs = -1 * np.ones([nseqs, npos], dtype=int)
    for i, entry in enumerate(msa_obj):
        aln_seq = entry.seq
        aa_screen = np.array([c != gapstr for c in aln_seq], dtype=bool)
        rawseq_idxs[i,aa_screen] = np.arange(aa_screen.sum())
    return rawseq_idxs


def get_conserved_rawseq_positions(
        msa_obj: MultipleSeqAlignment,
        retained_sequences: NDArray[np.int_],
        conserved_msa_positions: NDArray[np.int_],
):
    """Positions in raw sequences corresponding to conserved locations of an MSA.

    Args:
        msa_obj (MultipleSeqAlignment): MSA object.
        retained_sequences (NDArray[np.int_]): Indices of retained sequences.
        conserved_msa_positions (NDArray[np.int_]): Conserved MSA positions.

    Returns:
        (NDArray) Array C of shape (n,k) where n is the number of retained 
            sequences and k is the number of conserved MSA positions. If 
            retained sequence i has a gap at MSA position j, then C[i,j]=-1. 
            Otherwise, C[i,j] gives the positional index that MSA position j 
            corresponds to in retained sequence i.
    """
    rawseq_idxs = get_rawseq_indices_of_msa(msa_obj)
    rawseq_idxs = rawseq_idxs[retained_sequences,:]
    conserved_rawseq_idxs = rawseq_idxs[:,conserved_msa_positions]

    return conserved_rawseq_idxs


def get_rawseq_positions_in_groups(
        rawseq_idxs: NDArray[np.int_],
        groups: list[list[int]],
) -> list[list[list[int]]]:
    """Positions in raw sequences that correspond to groups.
    
    Args:
        rawseq_idxs (NDArray[np.int_]): Array of shape (n, m) where n is the 
            number of retained sequences and m is the number of retained 
            positions in the MSA.
        groups (list[list[int]]): Group definitions, with groups[i] listing
            the positions in the trimmed MSA (consisting of only retained
            sequences and positions) that are part of group i.

    Returns:
        list[list[list[int]]]: A nested list providing for each sequence and 
            group, the raw sequence positions that are part of the group.
    """
    nseqs, npos = rawseq_idxs.shape
    positions = [[None for _ in range(len(groups))] for _ in range(nseqs)]
    for gidx, group in enumerate(groups):
        group_idxs = rawseq_idxs[:, group]
        for seqidx in range(nseqs):
            positions[seqidx][gidx] = list(
                [int(i) for i in group_idxs[seqidx] if i >= 0]
            )

    return positions


def get_rawseq_scores_in_groups(
        rawseq_idxs: NDArray[np.int_],
        groups: list[NDArray[np.int_]],
        group_scores: list[NDArray[np.float64]],
) -> list[list[list[float]]]:
    """Scores in raw sequences that correspond to groups.
    
    Args:
        rawseq_idxs (NDArray[np.int_]): Array of shape (n, m) where n is the 
            number of retained sequences and m is the number of retained 
            positions in the MSA.
        groups (list[list[int]]): Group definitions, with groups[i] listing
            the positions in the trimmed MSA (consisting of only retained
            sequences and positions) that are part of group i.
        group_scores (list[NDArray[np.float64]]): Group scores, with 
            group_scores[i] listing the scores of the positions in the trimmed 
            MSA.

    Returns:
        list[list[list[float]]]: A nested list providing for each sequence and 
            group, the scores (projection onto the group component V_j) of the 
            raw sequence positions that are part of the group.
    """
    nseqs, npos = rawseq_idxs.shape
    scores = [[None for _ in range(len(groups))] for _ in range(nseqs)]
    for gidx, (group, score_vect) in enumerate(zip(groups, group_scores)):
        group_idxs = rawseq_idxs[:, group]
        for seqidx in range(nseqs):
            scores[seqidx][gidx] = list(
                [s for i, s in zip(group_idxs[seqidx], score_vect) if i >= 0]
            )
    return scores


def get_group_rawseq_positions_by_entry(
        msa_obj: MultipleSeqAlignment,
        retained_sequences: NDArray[np.int_],
        groups: list[list[int]],
        group_rawseq_positions: list[list[list[int]]],
) -> dict[str,list[NDArray[np.int_]]]:
    """Get indices

    Args:
        msa_obj (MultipleSeqAlignment): MSA object
        retained_sequences (NDArray[np.int_]): Indices of retained sequences.
        groups (list[list[int]]): Group definitions, with groups[i] listing
            the positions in the trimmed MSA (consisting of only retained
            sequences and positions) that are part of group i.
        group_rawseq_positions (list[list[list[int]]]): A nested list providing 
            for each sequence and group, the raw sequence positions that are 
            part of the group.

    Returns:
        dict[str,list[NDArray[np.int_]]]: Dictionary mapping entry ID to list of
            groups. Each group is an array of indices, the residue position in 
            the structure, with index starting at 0.
    """
    group_rawseq_positions_by_entry = {}
    for i, seqidx in enumerate(retained_sequences):
        entry = msa_obj[int(seqidx)]
        id = entry.id
        group_rawseq_positions_by_entry[id] = []
        for groupidx in range(len(groups)):
            group_rawseq_positions_by_entry[id].append(
                np.array(group_rawseq_positions[i][groupidx])
            )
    return group_rawseq_positions_by_entry


def get_group_rawseq_scores_by_entry(
        msa_obj: MultipleSeqAlignment,
        retained_sequences: NDArray[np.int_],
        groups: list[list[int]],
        group_rawseq_scores: list[list[list[float]]],
) -> dict[str,list[NDArray[np.float64]]]:
    """Get V scores

    Args:
        msa_obj (MultipleSeqAlignment): MSA object
        retained_sequences (NDArray[np.int_]): Indices of retained sequences.
        groups (list[list[int]]): Group definitions, with groups[i] listing
            the positions in the trimmed MSA (consisting of only retained
            sequences and positions) that are part of group i.
        group_rawseq_scores (list[list[list[float]]]): A nested list providing 
            for each sequence and group, the raw sequence scores corresponding 
            to positions that are part of the group.

    Returns:
        dict[str,list[NDArray[np.float64]]]: Dictionary mapping entry ID to list 
            of group scores. Each vector of group scores is an array of scores, 
            the score corresponding to the residue position in the structure, 
            with index starting at 0.
    """
    group_rawseq_scores_by_entry = {}
    for i, seqidx in enumerate(retained_sequences):
        entry = msa_obj[int(seqidx)]
        id = entry.id
        group_rawseq_scores_by_entry[id] = []
        for groupidx in range(len(groups)):
            group_rawseq_scores_by_entry[id].append(
                np.array(group_rawseq_scores[i][groupidx])
            )
    return group_rawseq_scores_by_entry


def iterblocks(x, block_size, use_pbar=False):
    """
    Iterate over the first axis of an array in contiguous blocks.

    This generator yields slices of `x` along its first dimension, along with
    the corresponding start and stop indices. It is useful for blockwise
    computation on large arrays where operating on the full array at once
    would be memory- or compute-intensive.

    Parameters
    ----------
    x : array-like
        Input array to iterate over. Blocks are taken along the first axis.
    block_size : int
        Number of rows to include in each block.

    Yields
    ------
    start : int
        Starting index (inclusive) of the block along the first axis.
    stop : int
        Ending index (exclusive) of the block along the first axis.
    block : array-like
        View of `x[start:stop]`.

    Notes
    -----
    The final block may contain fewer than `block_size` rows if the size of
    the first dimension is not divisible by `block_size`.

    Examples
    --------
    >>> for start, stop, block in iterblocks(x, 1000):
    ...     result[start:stop] = process(block)

    This pattern is useful for blockwise matrix operations, such as computing
    similarities between a full matrix and subsets of its rows.
    """
    n = x.shape[0]
    starts = range(0, n, block_size)
    if use_pbar:
        starts = tqdm.tqdm(starts, total=math.ceil(n / block_size))
    for start in starts:
        stop = min(start + block_size, n)
        yield start, stop, x[start:stop]


def iterblockpairs(x, block_size):
    n = x.shape[0]
    for i_start in range(0, n, block_size):
        i_stop = min(i_start + block_size, n)
        block_i = x[i_start:i_stop]
        for j_start in range(0, n, block_size):
            j_stop = min(j_start + block_size, n)
            block_j = x[j_start:j_stop]
            yield (i_start, i_stop, block_i), (j_start, j_stop, block_j)
