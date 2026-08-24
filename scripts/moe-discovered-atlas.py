#!/usr/bin/env python3
"""Track 1.6 - build an atlas whose axes are DISCOVERED, not assigned.

The existing atlas positions each expert against 9 categories hardcoded in
k_probes[] ("code", "math", "law", ...). Those are human guesses about what a
model ought to specialise in, measured offline with 27 probes. This builds the
same kind of map from what the router actually did, with no category list at
all: factor the co-activation graph and use its components as the axes.

Why co-activation and not req_dir: updating an atlas from req_dir is
self-training, because req_dir is itself inferred from the atlas - a slightly
wrong estimate reinforces itself (docs/plan.md, Track 1.6). "These two experts
fired in the same routing decision" is a direct observation with no atlas in
the loop, so there is no circularity to break.

The graph is GLOBAL over all (layer, expert) nodes rather than one embedding
per layer. Per-layer factorings each come out in their own arbitrary rotation,
so coordinates would not be comparable between layers - and req_dir averages
positions across layers, so it needs one shared space. Cross-layer edges
(layer L's top pick -> layer L+1's top pick, step 4b) are what tie the layers
into that shared space.

Output matches the existing atlas schema (x, y per cell) so it is a drop-in
for everything that already reads one, plus a `dims` vector holding more
components - the discovered analogue of `cats{}`.
"""
import json, sys, argparse
from collections import defaultdict
import numpy as np

TOPICS = ['code', 'math', 'history', 'medicine']
K = 8


def load_traces(prefix='traces/deep-', topics=TOPICS, keep=K):
    """Returns per-topic lists of (layer, [expert ids]) in decode order."""
    out = {}
    for t in topics:
        seq = []
        for line in open(f"{prefix}{t}.txt"):
            p = line.split()
            seq.append((int(p[0]), [int(x.split(':')[0]) for x in p[2:2 + keep]]))
        out[t] = seq
    return out


def build_graph(streams, n_expert=256):
    """Weighted undirected graph over (layer, expert).

    within-layer : experts selected in the SAME routing decision (step 4a)
    cross-layer  : consecutive layers' top picks for the same token (step 4b)
    """
    edges = defaultdict(float)
    nodes = {}

    def nid(l, e):
        k = (l, e)
        if k not in nodes:
            nodes[k] = len(nodes)
        return nodes[k]

    for seq in streams.values():
        prev = None
        for il, picks in seq:
            for i in range(len(picks)):
                a = nid(il, picks[i])
                for j in range(i + 1, len(picks)):
                    b = nid(il, picks[j])
                    if a != b:
                        edges[(min(a, b), max(a, b))] += 1.0
            if prev is not None and prev[0] != il:
                a, b = nid(prev[0], prev[1]), nid(il, picks[0])
                if a != b:
                    edges[(min(a, b), max(a, b))] += 1.0
            prev = (il, picks[0])
    return nodes, edges


def embed(nodes, edges, dims=8, iters=12, seed=0):
    """Top eigenvectors of the normalised adjacency, by randomised subspace
    iteration - keeps everything to sparse matvecs so a 7680-node graph never
    needs a dense 7680x7680 factorisation."""
    n = len(nodes)
    r = np.fromiter((e[0] for e in edges), dtype=np.int64, count=len(edges))
    c = np.fromiter((e[1] for e in edges), dtype=np.int64, count=len(edges))
    w = np.fromiter(edges.values(), dtype=np.float64, count=len(edges))

    deg = np.zeros(n)
    np.add.at(deg, r, w)
    np.add.at(deg, c, w)
    dinv = np.where(deg > 0, 1.0 / np.sqrt(np.maximum(deg, 1e-12)), 0.0)
    wn = w * dinv[r] * dinv[c]          # D^-1/2 A D^-1/2

    def matvec(X):
        Y = np.zeros_like(X)
        np.add.at(Y, r, wn[:, None] * X[c])
        np.add.at(Y, c, wn[:, None] * X[r])
        return Y

    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((n, dims + 4)))
    for _ in range(iters):
        Q, _ = np.linalg.qr(matvec(Q))
    B = Q.T @ matvec(Q)
    vals, vecs = np.linalg.eigh((B + B.T) / 2)
    order = np.argsort(-np.abs(vals))
    return (Q @ vecs[:, order])[:, :dims], vals[order][:dims], deg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/mnt/nvme/models/ornith/expert-atlas-discovered.json')
    ap.add_argument('--dims', type=int, default=8)
    a = ap.parse_args()

    streams = load_traces()
    nodes, edges = build_graph(streams)
    print(f"graph: {len(nodes)} nodes, {len(edges)} edges", file=sys.stderr)
    emb, vals, deg = embed(nodes, edges, dims=a.dims)
    print(f"top eigenvalues: {np.round(vals, 4)}", file=sys.stderr)

    # Component 0 of a normalised adjacency is the degree vector - it encodes
    # "how often was this expert used at all", which the heat counters already
    # track. The informative axes start at 1.
    xy = emb[:, 1:3]
    scale = np.abs(xy).max() or 1.0
    xy = xy / scale

    cells = []
    for (l, e), i in nodes.items():
        cells.append({
            "layer": l, "expert": e,
            "n": int(deg[i]),
            "spec": round(float(np.abs(xy[i]).max()), 4),
            "x": round(float(xy[i, 0]), 4),
            "y": round(float(xy[i, 1]), 4),
            "dims": [round(float(v), 4) for v in emb[i, 1:a.dims]],
        })
    cells.sort(key=lambda c: (c['layer'], c['expert']))
    json.dump({"categories": [f"d{i}" for i in range(1, a.dims)],
               "n_layer": max(c['layer'] for c in cells) + 1,
               "n_probes": 0,
               "discovered": True,
               "cells": cells}, open(a.out, 'w'))
    print(f"wrote {len(cells)} cells to {a.out}", file=sys.stderr)


if __name__ == '__main__':
    main()
