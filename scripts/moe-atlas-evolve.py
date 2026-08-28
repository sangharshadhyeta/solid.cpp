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
    reset its cross-layer `prev` pointer at each file boundary - otherwise
    the last decision of one trace file would get a spurious cross-layer
    edge to the first decision of the next."""
    out = []
    for p in paths:
        seq = []
        for line in open(p):
            f = line.split()
            seq.append((int(f[0]), [int(x.split(':')[0]) for x in f[2:2 + keep]]))
        out.append(seq)
    return out


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
    avoided this with a cross-layer edge (layer L's top pick -> layer L+1's
    top pick, step 4b) tying the whole graph into one connected space - same
    fix here, restructured to fold in incrementally.
    """
    def nid(l, e):
        k = (l, e)
        i = nodes.get(k)
        if i is None:
            i = len(nodes)
            nodes[k] = i
        return i
    for stream in seq:
        prev = None
        for il, picks in stream:
            idx = [nid(il, e) for e in picks]
            for i in range(len(idx)):
                for j in range(i + 1, len(idx)):
                    a, b = idx[i], idx[j]
                    if a != b:
                        edges[(min(a, b), max(a, b))] += 1.0
            if prev is not None and prev[0] != il:
                a, b = nid(prev[0], prev[1]), idx[0]
                if a != b:
                    edges[(min(a, b), max(a, b))] += 1.0
            prev = (il, picks[0])


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

    if a.traces:
        seq = load_traces(a.traces)
        fold_in(seq, nodes, edges)
        n_decisions = sum(len(stream) for stream in seq)
        print(f"folded in {n_decisions:,} decisions from {len(a.traces)} trace file(s)")
        print(f"graph now: {len(nodes):,} nodes ({len(nodes) - prior_nodes:+,}), "
              f"{len(edges):,} edges ({len(edges) - prior_edges:+,})")
    elif not prior_nodes:
        print("no --traces given and no prior state - nothing to build", file=sys.stderr)
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

    xy_raw = emb[:, 1:3] if k >= 2 else np.zeros((len(nodes), 2))

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
