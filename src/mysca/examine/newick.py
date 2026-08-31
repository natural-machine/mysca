"""Newick guide-tree parsing and the shared tree primitives used by the
phylogenetic-signal (:mod:`mysca.examine.pagel`) and clade-split
(:mod:`mysca.examine.splits`) analyses.

The parser is iterative (no recursion) so it scales to the 10k+ tip guide
trees that ``sca-examine`` builds from a family subsample. Everything is kept
in flat numpy arrays plus a per-node ``children`` list.

Leaves are joined to per-sequence IC projections by an *accession* key. The
extractor is deliberately permissive: it handles the UniProt entry-name
conventions seen in Pfam/InterPro alignments (``ACC_ORGANISM``, ``n_ACC_...``,
``ACC|...``) and falls back to the identity on labels with no delimiter, so a
caller that supplies its own delimiter-free leaf labels gets an exact join.
"""

import numpy as np
from scipy import stats

_NEWICK_DELIMS = set("(),:;")


def tokenize_newick(text):
    """Yield Newick tokens: single delimiters ``( ) , : ;`` or runs of
    name/number characters. All whitespace (including the newlines some tools
    emit between a label and its branch length) is skipped."""
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c.isspace():
            i += 1
            continue
        if c in _NEWICK_DELIMS:
            yield c
            i += 1
            continue
        j = i
        while j < n and text[j] not in _NEWICK_DELIMS and not text[j].isspace():
            j += 1
        yield text[i:j]
        i = j


def parse_newick(text):
    """Parse a Newick string into flat arrays. No recursion.

    Returns a dict with keys:
        parent       int32[n_nodes]   -1 at root
        children     list[list[int]]
        branch_len   float64[n_nodes] NaN where unspecified
        name         object[n_nodes]  '' for unnamed internal nodes
        is_leaf      bool[n_nodes]
        leaf_ids     int32[n_leaves]  node ids of leaves
        root         int
        n_nodes      int
    """
    parent = []
    children = []
    branch_len = []
    name = []

    def new_node():
        nid = len(parent)
        parent.append(-1)
        children.append([])
        branch_len.append(float("nan"))
        name.append("")
        return nid

    stack = []
    current = None
    expecting_child = False

    tokens = tokenize_newick(text)
    for tok in tokens:
        if tok == "(":
            nid = new_node()
            if stack:
                p = stack[-1]
                parent[nid] = p
                children[p].append(nid)
            stack.append(nid)
            current = nid
            expecting_child = True
        elif tok == ",":
            expecting_child = True
            current = None
        elif tok == ")":
            current = stack.pop()
            expecting_child = False
        elif tok == ":":
            length_tok = next(tokens)
            branch_len[current] = float(length_tok)
        elif tok == ";":
            break
        else:
            if expecting_child:
                nid = new_node()
                p = stack[-1]
                parent[nid] = p
                children[p].append(nid)
                name[nid] = tok
                current = nid
                expecting_child = False
            elif current is not None:
                name[current] = tok

    n_nodes = len(parent)
    parent_arr = np.asarray(parent, dtype=np.int32)
    branch_len_arr = np.asarray(branch_len, dtype=np.float64)
    name_arr = np.asarray(name, dtype=object)
    is_leaf = np.array([len(c) == 0 for c in children], dtype=bool)
    leaf_ids = np.where(is_leaf)[0].astype(np.int32)

    root_candidates = np.where(parent_arr == -1)[0]
    if len(root_candidates) != 1:
        raise ValueError(f"Expected 1 root, found {len(root_candidates)}")
    root = int(root_candidates[0])

    return {
        "parent": parent_arr,
        "children": children,
        "branch_len": branch_len_arr,
        "name": name_arr,
        "is_leaf": is_leaf,
        "leaf_ids": leaf_ids,
        "root": root,
        "n_nodes": n_nodes,
    }


def iterative_postorder(tree):
    """Children-before-parents ordering of nodes reachable from the root."""
    root = tree["root"]
    children = tree["children"]
    n = tree["n_nodes"]
    order = np.empty(n, dtype=np.int32)
    out_idx = 0
    stack = [(root, False)]
    while stack:
        v, visited = stack.pop()
        if visited:
            order[out_idx] = v
            out_idx += 1
        else:
            stack.append((v, True))
            for c in children[v]:
                stack.append((c, False))
    return order[:out_idx]


def reroot_at_leaf(tree, new_root_id):
    """In-place reroot. Flips parent pointers along the path from
    ``new_root_id`` to the old root and rotates branch lengths accordingly.
    The old root becomes a unifurcation (one child); split enumeration skips
    nodes with #children == 1, so we leave it in place."""
    parent = tree["parent"]
    children = tree["children"]
    branch_len = tree["branch_len"]

    path = []
    v = new_root_id
    while v != -1:
        path.append(int(v))
        v = int(parent[v])

    old_lens = branch_len.copy()
    for i in range(len(path) - 1):
        child = path[i]            # becomes the parent in the new tree
        old_par = path[i + 1]      # becomes the child in the new tree
        children[old_par].remove(child)
        children[child].append(old_par)
        parent[old_par] = child
        branch_len[old_par] = old_lens[child]

    parent[new_root_id] = -1
    branch_len[new_root_id] = float("nan")
    tree["root"] = int(new_root_id)
    return tree


def extract_accession_from_leaf(leaf_name):
    """Tree label -> accession key for the projection join.

    ``'1_A0A010Z1T0_unreviewed_..._taxID_927661'`` -> ``'A0A010Z1T0'``;
    ``'A0A6J1W0I0_9SAUR/25-273'`` -> ``'A0A6J1W0I0'``. A delimiter-free label
    (e.g. the synthetic ``s0``, ``s1`` labels :mod:`mysca.examine.guide_tree`
    writes) maps to itself, giving an exact tree<->projection join.
    """
    token = str(leaf_name).split("/", 1)[0]
    parts = token.split("_", 2)
    if len(parts) >= 2 and parts[0].isdigit():
        return parts[1]              # leading-index form: n_ACC_...
    return token.split("_", 1)[0]    # ACC_ORGANISM (or bare ACC)


def extract_accession_from_seq_id(seq_id):
    """Projection-table ``seq_id`` -> accession key.

    Handles both the pipe-delimited (``'A0A010Z1T0|unreviewed|...'``) and the
    UniProt entry-name (``'A0A6J1W0I0_9SAUR/25-273'``) conventions, plus bare
    delimiter-free labels. UniProt accessions carry no ``_``, so splitting on
    the first ``_`` only ever strips an organism mnemonic."""
    token = str(seq_id).split("/", 1)[0]
    token = token.split("|", 1)[0]
    return token.split("_", 1)[0]


def prune_to_accessions(tree, keep_acc):
    """Drop every leaf whose accession is not in ``keep_acc`` and every
    internal node that loses all descendants. Unifurcations are left in place
    (they contribute no contrast and preserve root-to-tip distances). Mutates
    and returns ``tree`` with refreshed ``children`` / ``is_leaf`` /
    ``leaf_ids``."""
    children = tree["children"]
    name = tree["name"]
    is_leaf = tree["is_leaf"]
    n = tree["n_nodes"]
    root = tree["root"]

    post = iterative_postorder(tree)
    keep = np.zeros(n, dtype=bool)
    for node in post:  # children before parents
        if is_leaf[node]:
            keep[node] = extract_accession_from_leaf(name[node]) in keep_acc
        else:
            keep[node] = any(keep[c] for c in children[node])

    if not keep[root]:
        raise ValueError("Pruning removed the root: no kept leaves under it.")

    new_children = [[] for _ in range(n)]
    for node in range(n):
        if keep[node] and not is_leaf[node]:
            new_children[node] = [c for c in children[node] if keep[c]]

    tree["children"] = new_children
    tree["is_leaf"] = np.array(
        [keep[i] and len(new_children[i]) == 0 for i in range(n)], dtype=bool
    )
    tree["leaf_ids"] = np.where(tree["is_leaf"])[0].astype(np.int32)
    return tree


def descendants_of(tree, node_id):
    """Leaf node ids descended from ``node_id`` (or ``[node_id]`` if a leaf)."""
    out = []
    stack = [node_id]
    children = tree["children"]
    while stack:
        v = stack.pop()
        if len(children[v]) == 0:
            out.append(v)
        else:
            stack.extend(children[v])
    return out


def sample_accessions(tree, leaf_node_ids, k, rng):
    """Up to ``k`` representative accessions sampled from ``leaf_node_ids``."""
    names = tree["name"]
    if len(leaf_node_ids) <= k:
        chosen = list(leaf_node_ids)
    else:
        chosen = rng.choice(leaf_node_ids, size=k, replace=False).tolist()
    return [extract_accession_from_leaf(names[n]) for n in chosen]


def bh_fdr(p_flat):
    """Benjamini-Hochberg FDR; NaN p-values are passed through as NaN."""
    p = np.asarray(p_flat, dtype=np.float64)
    q = np.full_like(p, np.nan)
    mask = np.isfinite(p)
    if mask.sum():
        q[mask] = stats.false_discovery_control(p[mask])
    return q
