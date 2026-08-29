#!/usr/bin/env python3
"""Persistent, self-sizing discovered atlas.

Two things moe-discovered-atlas.py did NOT do, both requested directly:

1. The co-activation graph was rebuilt from scratch, from a fixed trace set,
   every time it ran - a snapshot, not something that accumulates. This
   persists the graph itself (node index + edge weights) to --state and
   ADDS each run's --traces on top of whatever is already there, the same
   session.history-style accumulation moe-incremental-atlas.py already does
   for its SGNS vectors, but for the graph spectral factoring actually
   reads. Re-running with more data does not restart; it keeps going.

2. The number of discovered axes (--dims elsewhere) was a number we picked.
   This picks it FROM THE DATA instead, every run, via the standard spectral
   eigengap heuristic (von Luxburg, "A Tutorial on Spectral Clustering",
   2007, section 8.4): compute the top eigenvalues of the graph's normalised
   adjacency, and cut where consecutive eigenvalue magnitudes drop off
   sharply - everything before that point is structure, everything after is
   the noise floor. As more data accumulates the graph gets less noisy and
   the natural cut point can move, so the axis count is free to grow (or
   shrink) run over run instead of being fixed at whatever we guessed
   up front. That is what "the atlas decides how many topics it wants, and
   never stops" means concretely: re-run this with new traces periodically
   and both the graph and its own dimensionality keep evolving.

Output matches the existing atlas schema (x, y per cell from components 1-2,
plus a `dims` vector holding the rest) so it is a drop-in for
--expert-atlas-file exactly like moe-discovered-atlas.py's output.
"""
import json, sys, os, argparse
from collections import defaultdict
import numpy as np

K_SELECTED = 8


def load_traces(paths, keep=K_SELECTED):
    """One sequence per file, kept separate (not concatenated) so fold_in can
    reset its cross-layer batch grouping at each file boundary - otherwise
    the last token of one trace file would get spuriously linked to the
    first token of the next.

    Each line is `layer token_id expert:weight ...` - a trace file logs one
    batch of tokens at a time, one layer's worth of lines for the whole
    batch before moving to the next layer, so `token_id` re-identifies the
    same token across those layer-major blocks."""
    out = []
    for p in paths:
        seq = []
        for line in open(p):
            f = line.split()
            seq.append((int(f[0]), int(f[1]), [int(x.split(':')[0]) for x in f[2:2 + keep]]))
        out.append(seq)
    return out


def fold_coact(paths, nodes, edges):
    """Fold co-activation edges emitted by the running server itself
    (GGML_CUDA_MOE_CACHE_COACT_FILE) straight into the persisted graph.

    The cache already accumulates exactly this graph while it serves - one
    edge per pair of experts the router selected together, keyed by
    (layer, expert), which is the same node identity fold_in() builds from
    trace files. So there is nothing to re-derive here: the server writes
    `layer_a expert_a layer_b expert_b weight` and this adds those weights
    on top of whatever the state already holds, exactly as fold_in does.

    The point of this path is that the graph then reflects REAL SERVED
    TRAFFIC rather than a separate offline llama-moe-trace run over prompts
    somebody invented - an atlas built from what a deployment actually
    routes, which an offline probe cannot produce by construction.
    """
    def nid(l, e):
        k = (int(l), int(e))
        i = nodes.get(k)
        if i is None:
            i = len(nodes)
            nodes[k] = i
        return i
    folded = 0
    for p in paths:
        with open(p) as f:
            first = f.readline()
            if not first.startswith('moe-cache-coact'):
                print(f"  {p}: not a coact file (bad header) - skipped", file=sys.stderr)
                continue
            for line in f:
                fs = line.split()
                if len(fs) != 5:
                    continue
                la, ea, lb, eb, w = fs
                a_i, b_i = nid(la, ea), nid(lb, eb)
                if a_i == b_i:
                    continue
                edges[(min(a_i, b_i), max(a_i, b_i))] += float(w)
                folded += 1
    return folded


def load_state(path):
    """Persisted (node index, edge weights) - empty if this is the first run."""
    if not path or not os.path.exists(path):
        return {}, defaultdict(float)
    z = np.load(path, allow_pickle=True)
    nodes = {tuple(int(x) for x in k): int(v) for k, v in zip(z['keys'], z['vals'])}
    edges = defaultdict(float)
    for a, b, w in zip(z['edge_a'], z['edge_b'], z['edge_w']):
        edges[(int(a), int(b))] = float(w)
    return nodes, edges


def save_state(path, nodes, edges):
    keys = np.array(list(nodes.keys()))
    vals = np.array(list(nodes.values()))
    ea = np.fromiter((e[0] for e in edges), dtype=np.int64, count=len(edges))
    eb = np.fromiter((e[1] for e in edges), dtype=np.int64, count=len(edges))
    ew = np.fromiter(edges.values(), dtype=np.float64, count=len(edges))
    np.savez(path, keys=keys, vals=vals, edge_a=ea, edge_b=eb, edge_w=ew)


def fold_in(seq, nodes, edges):
    """Add one run's worth of co-activation observations onto the persisted
    graph in place - ADDS to existing edge weights, never overwrites, which
    is what makes repeated runs accumulate instead of restart.

    Within-layer edges alone leave every layer as its own disconnected
    component - a normalized adjacency's eigenvalue 1 has multiplicity equal
    to the connected-component count, so with N layers the top N eigenvalues
    are all trivially 1.0 and the eigengap heuristic sees pure degeneracy,
    not structure (caught empirically: a 2-topic smoke test returned 31
    components all magnitude 1.0). moe-discovered-atlas.py's build_graph
    fixed connectivity with a cross-layer edge (layer L's top pick -> layer
    L+1's top pick), but only ever to the immediately next layer, for every
    token, unconditionally - that makes each layer a tight within-layer
    clique joined to its neighbours by one narrow, always-present bottleneck
    edge. Structurally that's a chain of clusters, and the leading
    eigenvectors of a chain-of-clusters graph are low-frequency waves along
    the chain: "which cluster (layer) am I in" dominates the embedding
    regardless of edge weight, which is exactly the layer-ordered "spokes"
    artifact seen in the atlas view.

    Fix: link a token's top pick at EVERY layer it touched to its top pick
    at every OTHER layer it touched (all pairs, not just adjacent), so two
    layers can bridge directly whenever they tend to route the same token
    to a related expert - the graph stops being forced through a rigid
    layer-sequential backbone and can bridge on topic/content similarity
    instead. Requires grouping lines by token_id within each file's
    layer-major batches (see load_traces): a batch boundary is detected
    whenever the layer number goes DOWN instead of up.
    """
    def nid(l, e):
        k = (l, e)
        i = nodes.get(k)
        if i is None:
            i = len(nodes)
            nodes[k] = i
        return i

    def flush_batch(batch):
        for layer_map in batch.values():
            top1 = list(layer_map.values())
            for i in range(len(top1)):
                for j in range(i + 1, len(top1)):
                    a, b = top1[i], top1[j]
                    if a != b:
                        edges[(min(a, b), max(a, b))] += 1.0

    for stream in seq:
        batch = {}
        prev_layer = None
        for il, tid, picks in stream:
            idx = [nid(il, e) for e in picks]
            for i in range(len(idx)):
                for j in range(i + 1, len(idx)):
                    a, b = idx[i], idx[j]
                    if a != b:
                        edges[(min(a, b), max(a, b))] += 1.0
            if prev_layer is not None and il < prev_layer:
                flush_batch(batch)
                batch = {}
            batch.setdefault(tid, {})[il] = idx[0]
            prev_layer = il
        flush_batch(batch)


def spectral(nodes, edges, max_dims, iters=12, seed=0):
    """Top eigenvectors of the normalised adjacency, randomised subspace
    iteration (same method moe-discovered-atlas.py uses) - sparse matvecs
    only, so this scales to graphs far larger than would fit a dense
    factorisation. Computes up to max_dims+1 components (component 0 is the
    degree vector, dropped by the caller) so eigengap_k below has a real
    spectrum to look at rather than just whatever we already decided to keep.
    """
    n = len(nodes)
    r = np.fromiter((e[0] for e in edges), dtype=np.int64, count=len(edges))
    c = np.fromiter((e[1] for e in edges), dtype=np.int64, count=len(edges))
    w = np.fromiter(edges.values(), dtype=np.float64, count=len(edges))
    deg = np.zeros(n)
    np.add.at(deg, r, w)
    np.add.at(deg, c, w)
    dinv = np.where(deg > 0, 1.0 / np.sqrt(np.maximum(deg, 1e-12)), 0.0)
    wn = w * dinv[r] * dinv[c]

    def matvec(X):
        Y = np.zeros_like(X)
        np.add.at(Y, r, wn[:, None] * X[c])
        np.add.at(Y, c, wn[:, None] * X[r])
        return Y

    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((n, max_dims + 4)))
    for _ in range(iters):
        Q, _ = np.linalg.qr(matvec(Q))
    B = Q.T @ matvec(Q)
    vals, vecs = np.linalg.eigh((B + B.T) / 2)
    order = np.argsort(-np.abs(vals))
    return (Q @ vecs[:, order])[:, :max_dims + 1], vals[order][:max_dims + 1], deg


def eigengap_k(vals, min_dims, max_dims):
    """Component 0 is the degree vector, excluded here (same reasoning as
    moe-discovered-atlas.py: it encodes raw usage frequency, which the cache's
    own heat counters already track, not topic structure). Pick k = the
    point where consecutive |eigenvalue| magnitudes among components 1..
    max_dims drop off the most - the standard eigengap heuristic. Returns
    (k, the gap sequence actually examined) so the caller can report why.
    """
    mags = np.abs(vals[1:1 + max_dims])
    if len(mags) < min_dims + 1:
        return min(min_dims, len(mags)), np.array([])
    gaps = mags[:-1] - mags[1:]
    lo = min_dims - 1
    hi = max(lo + 1, len(gaps))
    i_star = lo + int(np.argmax(gaps[lo:hi]))
    k = i_star + 1
    return max(min_dims, min(k, max_dims)), gaps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--traces', nargs='*', default=[],
                     help='new trace files to fold into the persisted graph this run (optional - '
                          'omit to just re-decide dims on the existing state)')
    ap.add_argument('--coact', nargs='*', default=[],
                     help='co-activation edge files written by the running server itself '
                          '(GGML_CUDA_MOE_CACHE_COACT_FILE) - real served traffic, no offline '
                          'trace run needed')
    ap.add_argument('--state', default='traces/atlas-evolve-state.npz',
                     help='persisted graph - accumulates across runs, like moe-incremental-atlas.py')
    ap.add_argument('--out', default='/mnt/nvme/models/ornith/expert-atlas-discovered-evolve.json')
    ap.add_argument('--min-dims', type=int, default=4)
    ap.add_argument('--max-dims', type=int, default=64,
                     help='computational ceiling on how many components to even consider - '
                          'the eigengap heuristic still picks the real count within this bound')
    a = ap.parse_args()

    nodes, edges = load_state(a.state)
    prior_nodes, prior_edges = len(nodes), len(edges)
    if prior_nodes:
        print(f"resumed state: {prior_nodes:,} nodes, {prior_edges:,} edges")
    else:
        print("no prior state - starting fresh")

    if a.coact:
        n_edges = fold_coact(a.coact, nodes, edges)
        print(f"folded in {n_edges:,} co-activation edges from {len(a.coact)} live-traffic file(s)")
        print(f"graph now: {len(nodes):,} nodes ({len(nodes) - prior_nodes:+,}), "
              f"{len(edges):,} edges ({len(edges) - prior_edges:+,})")
        prior_nodes, prior_edges = len(nodes), len(edges)

    if a.traces:
        seq = load_traces(a.traces)
        fold_in(seq, nodes, edges)
        n_decisions = sum(len(stream) for stream in seq)
        print(f"folded in {n_decisions:,} decisions from {len(a.traces)} trace file(s)")
        print(f"graph now: {len(nodes):,} nodes ({len(nodes) - prior_nodes:+,}), "
              f"{len(edges):,} edges ({len(edges) - prior_edges:+,})")
    elif not prior_nodes and not a.coact:
        print("no --traces/--coact given and no prior state - nothing to build", file=sys.stderr)
        sys.exit(1)
    else:
        print("no new --traces - re-deciding dimensionality on the existing graph only")

    save_state(a.state, nodes, edges)

    emb, vals, deg = spectral(nodes, edges, a.max_dims)
    k, gaps = eigengap_k(vals, a.min_dims, a.max_dims)
    print(f"\ntop eigenvalue magnitudes (first {min(12, len(vals))}): "
          f"{np.round(np.abs(vals[:12]), 4)}")
    if len(gaps):
        print(f"eigengap heuristic: chose k={k} axes "
              f"(largest gap {gaps.max():.4f} at that cut, min_dims={a.min_dims}, max_dims={a.max_dims})")
    else:
        print(f"eigengap heuristic: too few components for a real gap search, k={k}")

    # Position each expert the same way the old fixed 9-topic atlas did -
    # a weighted combination of topic anchor positions - generalised to
    # however many axes k the eigengap heuristic picked THIS run (not a
    # fixed 9), and with the anchors themselves data-driven instead of
    # preset. Anchors start on an equal-angle circle (same mechanism the old
    # atlas used for its fixed categories, just sized to k) then get pulled
    # by reciprocal averaging: each topic's anchor moves to the weighted
    # centroid of the experts that load on it, so topics whose experts
    # overlap drift toward each other and topics with nothing in common
    # drift apart - unlike the old preset categories, which sat at their
    # starting angle forever regardless of what the data said.
    if k >= 2:
        w = emb[:, 1:k] ** 2  # non-negative affinity per (expert, axis) - direction doesn't matter for a spatial pull, magnitude does
        row_sums = w.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        w_norm = w / row_sums

        n_axes = k - 1
        angles = 2.0 * np.pi * np.arange(n_axes) / n_axes
        topic_xy = np.stack([np.cos(angles), np.sin(angles)], axis=1)

        for _ in range(2):
            xy_raw = w_norm @ topic_xy
            col_sums = w.sum(axis=0)
            col_sums[col_sums == 0] = 1.0
            topic_xy = (w.T @ xy_raw) / col_sums[:, None]
            scale = np.abs(topic_xy).max()
            if scale > 0:
                topic_xy = topic_xy / scale

        xy_raw = w_norm @ topic_xy
    else:
        xy_raw = np.zeros((len(nodes), 2))

    # Strip out whatever's left correlated with layer position: the Grid
    # panel already shows layer directly, so the atlas plotting layer again
    # (even as a residual trend after the cross-layer edge fix above) is
    # redundant at best. Per-layer mean-centering removes any systematic
    # layer effect - linear or not - leaving only the within-layer spread
    # that actually reflects topic content.
    layer_of = np.empty(len(nodes), dtype=np.int64)
    for (l, _e), i in nodes.items():
        layer_of[i] = l
    for lyr in np.unique(layer_of):
        m = layer_of == lyr
        xy_raw[m, 0] -= xy_raw[m, 0].mean()
        xy_raw[m, 1] -= xy_raw[m, 1].mean()

    def rank_normalize(v):
        order = np.argsort(v)
        rank = np.empty_like(order, dtype=np.float64)
        rank[order] = np.arange(len(v))
        return 2.0 * (rank + 0.5) / len(v) - 1.0

    # Same square->disc remap as moe-discovered-atlas.py's fix (d162224c4) -
    # rank-normalize x/y independently, then Shirley-Chiu concentric map so
    # corner points land inside the circle the UI actually draws.
    sq = np.stack([rank_normalize(xy_raw[:, 0]), rank_normalize(xy_raw[:, 1])], axis=1)
    sx, sy = sq[:, 0], sq[:, 1]
    r = np.where(np.abs(sx) > np.abs(sy), sx, sy)
    theta = np.where(
        np.abs(sx) > np.abs(sy),
        (np.pi / 4) * np.divide(sy, sx, out=np.zeros_like(sy), where=sx != 0),
        (np.pi / 2) - (np.pi / 4) * np.divide(sx, sy, out=np.zeros_like(sx), where=sy != 0),
    )
    xy = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)
    xy[(sx == 0) & (sy == 0)] = 0.0

    cells = []
    for (l, e), i in nodes.items():
        cells.append({
            "layer": l, "expert": e,
            "n": int(deg[i]),
            "spec": round(float(np.abs(xy[i]).max()), 4),
            "x": round(float(xy[i, 0]), 4),
            "y": round(float(xy[i, 1]), 4),
            "dims": [round(float(v), 4) for v in emb[i, 1:k]],
        })
    cells.sort(key=lambda c: (c['layer'], c['expert']))
    json.dump({"categories": [f"d{i}" for i in range(1, k + 1)],
               "n_layer": max(c['layer'] for c in cells) + 1,
               "n_probes": 0,
               "discovered": True,
               "evolving": True,
               "cells": cells}, open(a.out, 'w'))
    print(f"\nwrote {len(cells)} cells, {k} discovered axes, to {a.out}")
    print(f"state saved to {a.state} - next run with new --traces will resume and re-decide k")


if __name__ == '__main__':
    main()
