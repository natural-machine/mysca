"""Group SCA independent components into sectors, co-sectors and pseudo-sectors.

Implements the strategy proposed in Andrews, "How to characterize independent
components after performing SCA" (the document this package operationalises):

  1. Conservation: ``cons_corr_r > cons_r`` -> mark the IC ``core``
     (conservation-associated) and protect it from the masking in steps 2-3.
  2. Subclade: ``abs_cliffs_delta > subclade_delta`` -> ``pseudo-sector``
     (subclade origin); masked out of the merge graph.
  3. Phylogeny: ``lrt_z > phylo_z`` -> ``pseudo-sector`` (phylogeny origin);
     masked.
  4. Among the surviving ICs, build a directed reachability graph: x reaches y
     iff y's residues sit closer to x than x's own internal spread
     (cross-neighbour distance < within-IC distance).
  5. Reciprocal reach (x<->y) unions the two ICs; transitive closure gives
     sector clusters.
  6. A one-directional reach makes the reaching IC a ``co-sector`` of the
     sector it points to. Every other surviving IC is its own ``sector``.

Default thresholds match the document's Conclusions (cons_r=0.5,
subclade_delta=0.95, phylo_z=7). All are caller-overridable so the strategy can
be retuned per family or per dataset.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("mysca.examine.sectors")


# Document Conclusions defaults.
DEFAULT_CONS_R = 0.5
DEFAULT_SUBCLADE_DELTA = 0.95
DEFAULT_PHYLO_Z = 7.0


class _UnionFind:
    """Plain union-find for the reciprocal-merge transitive closure (step 5)."""

    def __init__(self, items):
        self.parent = {x: x for x in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]   # path-halving
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def group_ics(per_ic, cross, *, family="family",
              cons_r=DEFAULT_CONS_R, subclade_delta=DEFAULT_SUBCLADE_DELTA,
              phylo_z=DEFAULT_PHYLO_Z):
    """Categorise and merge a family's ICs into a feature table.

    Parameters
    ----------
    per_ic : pandas.DataFrame
        One row per significant IC, ordered by IC index, with (at least) the
        columns ``component`` and ``ic_idx``. Optional columns drive the rules:
        ``cons_corr_r``, ``abs_cliffs_delta``, ``lrt_z``, ``within_dist``
        (intra-IC mean neighbour distance), ``ic_n_residues``. Missing columns
        simply disable the corresponding rule.
    cross : np.ndarray or None
        (K, K) cross-neighbour matrix from
        :func:`mysca.examine.structure.cross_ic_distances`, indexed by
        ``ic_idx``. ``cross[i, j]`` = mean nearest distance to IC i over IC j's
        residues. None disables the structural merge (every surviving IC stays
        a standalone sector).
    family : str
        Prefix for generated feature names.
    cons_r, subclade_delta, phylo_z : float
        Thresholds for steps 1-3.

    Returns
    -------
    pandas.DataFrame
        One row per feature: ``family, feature_name, feature_type, n_ics,
        n_positions, labels, associated_sector, member_components``. Feature
        types are ``sector`` / ``co-sector`` / ``pseudo-sector``.
    """
    def col(name):
        return (per_ic[name] if name in per_ic.columns
                else pd.Series(np.nan, index=per_ic.index))

    cons = col("cons_corr_r")
    cliffs = col("abs_cliffs_delta")
    lrtz = col("lrt_z")
    nres = col("ic_n_residues")
    within = col("within_dist")

    # --- per-IC categorisation (steps 1-3) -----------------------------------
    ics = {}
    for i, row in per_ic.reset_index(drop=True).iterrows():
        comp = str(row["component"])
        labels = []
        conservation = bool(pd.notna(cons.iloc[i]) and cons.iloc[i] > cons_r)
        if conservation:
            labels.append("conservation-associated")
        masked, ptype = False, None
        if not conservation:                              # conservation preempts
            if pd.notna(cliffs.iloc[i]) and cliffs.iloc[i] > subclade_delta:
                masked, ptype = True, "subclade origin"
            elif pd.notna(lrtz.iloc[i]) and lrtz.iloc[i] > phylo_z:
                masked, ptype = True, "phylogeny origin"
        if masked:
            labels.append(ptype)
        ics[comp] = {
            "component": comp,
            "idx": int(row["ic_idx"]),
            "n_positions": int(nres.iloc[i]) if pd.notna(nres.iloc[i]) else None,
            "within_dist": float(within.iloc[i]) if pd.notna(within.iloc[i]) else np.nan,
            "conservation": conservation,
            "masked": masked,
            "pseudo_type": ptype,
            "labels": labels,
        }

    unmasked = [c for c, v in ics.items() if not v["masked"]]

    # --- directed reachability among unmasked ICs ----------------------------
    # edge x -> y  iff  cross[idx_y, idx_x] < within_dist[x]  ("IC_y sits closer
    # to IC_x's residues than IC_x's own internal spread").
    def cross_dist(x, y):
        ix, iy = ics[x]["idx"], ics[y]["idx"]
        if cross is None:
            return np.nan
        if not (0 <= iy < cross.shape[0] and 0 <= ix < cross.shape[1]):
            return np.nan
        return cross[iy, ix]

    def reaches(x, y):
        d, w = cross_dist(x, y), ics[x]["within_dist"]
        return np.isfinite(d) and np.isfinite(w) and d < w

    edges = {x: set() for x in unmasked}
    for x in unmasked:
        for y in unmasked:
            if x != y and reaches(x, y):
                edges[x].add(y)

    # --- step 5: reciprocal merge -> clusters --------------------------------
    uf = _UnionFind(unmasked)
    for x in unmasked:
        for y in edges[x]:
            if x < y and x in edges[y]:           # reciprocal (x<-->y) -> union
                uf.union(x, y)
    clusters = {}
    for c in unmasked:
        clusters.setdefault(uf.find(c), []).append(c)
    cluster_of = {c: uf.find(c) for c in unmasked}

    # --- step 6: classify each unmasked IC -----------------------------------
    is_cosector = {}
    for c in unmasked:
        if len(clusters[cluster_of[c]]) >= 2:
            is_cosector[c] = False                # part of a merged sector
        else:
            is_cosector[c] = len(edges[c]) > 0    # reaches something one-way

    sector_clusters = [members for root, members in clusters.items()
                       if len(members) >= 2 or not is_cosector[members[0]]]

    def cluster_key(members):
        idxs = [ics[m]["idx"] for m in members]
        return (min(idxs) if idxs else 10**9, min(members))
    sector_clusters.sort(key=cluster_key)

    rows = []
    sector_name_of_member = {}
    for n, members in enumerate(sector_clusters, start=1):
        members = sorted(members, key=lambda m: (ics[m]["idx"], m))
        fname = f"{family}_sector_{n}"
        for m in members:
            sector_name_of_member[m] = fname
        labels = sorted({l for m in members for l in ics[m]["labels"]})
        npos = [ics[m]["n_positions"] for m in members
                if ics[m]["n_positions"] is not None]
        rows.append(_feature_row(
            family, fname, "sector", members, ics, labels, npos))

    # --- step 6b: resolve each co-sector to the sector it attaches to --------
    # Follow the strongest outgoing edge (smallest cross_dist), walking the
    # chain until it lands on a sector member. An unresolvable chain demotes the
    # IC to its own standalone sector so nothing is silently dropped.
    cosectors = [c for c in unmasked if is_cosector[c]]

    def best_target(x):
        cands = [(cross_dist(x, y), y) for y in edges[x]
                 if np.isfinite(cross_dist(x, y))]
        return min(cands)[1] if cands else None

    def resolve_sector(x):
        seen, cur = set(), x
        while cur is not None and cur not in seen:
            seen.add(cur)
            tgt = best_target(cur)
            if tgt is None:
                return None
            if tgt in sector_name_of_member:
                return sector_name_of_member[tgt]
            cur = tgt
        return None

    co_n = 0
    for c in sorted(cosectors, key=lambda m: (ics[m]["idx"], m)):
        assoc = resolve_sector(c)
        if assoc is None:
            n_sector = sum(1 for r in rows if r["feature_type"] == "sector") + 1
            fname = f"{family}_sector_{n_sector}"
            sector_name_of_member[c] = fname
            npos = [ics[c]["n_positions"]] if ics[c]["n_positions"] is not None else []
            rows.append(_feature_row(
                family, fname, "sector", [c], ics, sorted(set(ics[c]["labels"])), npos))
            continue
        co_n += 1
        npos = [ics[c]["n_positions"]] if ics[c]["n_positions"] is not None else []
        rows.append(_feature_row(
            family, f"{family}_cosector_{co_n}", "co-sector", [c], ics,
            sorted(set(ics[c]["labels"])), npos, associated_sector=assoc))

    # --- pseudo-sectors (one feature per masked IC) --------------------------
    ps_n = 0
    for c, v in ics.items():
        if not v["masked"]:
            continue
        ps_n += 1
        npos = [v["n_positions"]] if v["n_positions"] is not None else []
        rows.append(_feature_row(
            family, f"{family}_pseudosector_{ps_n}", "pseudo-sector", [c], ics,
            sorted(set(v["labels"])), npos))

    cols = ["family", "feature_name", "feature_type", "n_ics", "n_positions",
            "labels", "associated_sector", "member_components"]
    return pd.DataFrame(rows, columns=cols)


def _feature_row(family, fname, ftype, members, ics, labels, npos,
                 associated_sector=""):
    members = list(members)
    return {
        "family": family,
        "feature_name": fname,
        "feature_type": ftype,
        "n_ics": len(members),
        "n_positions": int(sum(npos)) if npos else np.nan,
        "labels": ";".join(labels),
        "associated_sector": associated_sector,
        "member_components": ",".join(
            sorted(members, key=lambda m: (ics[m]["idx"], m))),
    }
