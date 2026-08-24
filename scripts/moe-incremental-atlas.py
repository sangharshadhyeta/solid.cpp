#!/usr/bin/env python3
"""Incremental discovered atlas: co-firing pulls experts together, over time.

The spectral version factors a snapshot in one step. This is the mechanism
actually intended: every routing decision nudges the experts that fired
together closer and pushes unrelated ones apart, the layout accumulates across
runs, and the map settles - or fails to - on its own.

Settling is NOT annealed in. The learning rate is CONSTANT. If displacement
falls over time it is because the objective is being satisfied, not because the
step size was turned down; an annealed rate would manufacture the appearance of
settling regardless of whether real structure existed. That is the whole point
of the test, so the knob stays fixed.

Objective is skip-gram with negative sampling over co-firing pairs: a pair that
fired in the same routing decision is a positive, a random same-layer pair is a
negative. Negatives are drawn within the layer because cross-layer pairs are
trivially separable and would flatter the model.

State persists to an .npz holding the vectors, the node index, and the total
step count. Re-running RESUMES from that file and keeps training, the same way
session.history accumulates across restarts rather than starting cold.
"""
import argparse, json, os, sys, time
from collections import defaultdict
import numpy as np

K_SELECTED = 8


def load_pooled(paths, keep=K_SELECTED):
    seq = []
    for p in paths:
        for line in open(p):
            f = line.split()
            seq.append((int(f[0]), [int(x.split(':')[0]) for x in f[2:2 + keep]]))
    return seq


def pairs_from(seq, nodes, grow=True):
    """Positive co-firing pairs as node-index arrays, plus per-layer node lists."""
    A, B = [], []
    by_layer = defaultdict(set)
    for il, picks in seq:
        idx = []
        for e in picks:
            k = (il, e)
            i = nodes.get(k)
            if i is None:
                if not grow:
                    continue
                i = len(nodes); nodes[k] = i
            idx.append(i); by_layer[il].add(i)
        for x in range(len(idx)):
            for y in range(x + 1, len(idx)):
                if idx[x] != idx[y]:
                    A.append(idx[x]); B.append(idx[y])
    return (np.array(A, dtype=np.int64), np.array(B, dtype=np.int64),
            {l: np.array(sorted(v), dtype=np.int64) for l, v in by_layer.items()})


def sgd_epoch(emb, A, B, layer_of, layer_members, lr, negs, rng, batch=1_000_000):
    """One pass. Constant lr, by design - see the module docstring."""
    n = len(A)
    for s in range(0, n, batch):
        a = A[s:s + batch]; b = B[s:s + batch]
        # positive: pull together
        # Accumulate this batch's updates, then divide each node's total by how
        # many times it appeared. Without that, np.add.at applies every update a
        # node collected at once, so a frequently selected expert takes a step
        # proportional to its frequency - which does not diverge once vectors
        # are normalised, it OSCILLATES, flipping to the antipode each pass
        # (median move 2.0 on a unit sphere is a 180 degree flip). Averaging
        # makes the step independent of frequency. The learning rate itself is
        # still constant, so settling remains something the objective has to
        # earn.
        d = np.zeros_like(emb)
        cnt = np.zeros(len(emb))
        ua, ub = emb[a], emb[b]
        sig = 1.0 / (1.0 + np.exp(np.clip(np.einsum('ij,ij->i', ua, ub), -30, 30)))
        g = (lr * sig)[:, None]
        np.add.at(d, a, g * ub); np.add.at(d, b, g * ua)
        np.add.at(cnt, a, 1.0);  np.add.at(cnt, b, 1.0)
        # negative: push apart, sampled within the same layer
        for _ in range(negs):
            ls = layer_of[a]
            c = np.empty_like(a)
            for l in np.unique(ls):
                m = ls == l
                mem = layer_members[l]
                c[m] = mem[rng.integers(0, len(mem), size=int(m.sum()))]
            ua, uc = emb[a], emb[c]
            sig = 1.0 / (1.0 + np.exp(np.clip(-np.einsum('ij,ij->i', ua, uc), -30, 30)))
            g = (-lr * sig)[:, None]
            np.add.at(d, a, g * uc); np.add.at(d, c, g * ua)
            np.add.at(cnt, a, 1.0);  np.add.at(cnt, c, 1.0)
        emb += d / np.maximum(cnt, 1.0)[:, None]
        # Project back onto the unit sphere. Needed because np.add.at applies
        # every update a node collected in this batch at once, so a frequently
        # selected expert takes a step proportional to its frequency and the
        # vectors diverge within one epoch. Normalising bounds the magnitude
        # WITHOUT touching the learning rate - settling still has to emerge
        # from the objective, and now shows up as shrinking angular movement
        # rather than shrinking vector length. It also matches what the atlas
        # already is: a position map, where only direction carries meaning.
        nrm = np.linalg.norm(emb, axis=1, keepdims=True)
        np.divide(emb, np.maximum(nrm, 1e-12), out=emb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--traces', nargs='+', required=True)
    ap.add_argument('--state', default='traces/incremental-atlas.npz')
    ap.add_argument('--dims', type=int, default=32)
    ap.add_argument('--lr', type=float, default=0.01)
    ap.add_argument('--negs', type=int, default=1)
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--fresh', action='store_true', help='ignore any saved state')
    a = ap.parse_args()

    seq = load_pooled(a.traces)
    cut = len(seq) // 2
    train, test = seq[:cut], seq[cut:]

    nodes, emb, steps = {}, None, 0
    if os.path.exists(a.state) and not a.fresh:
        z = np.load(a.state, allow_pickle=True)
        nodes = {tuple(k): int(v) for k, v in zip(z['keys'], z['vals'])}
        emb = z['emb']; steps = int(z['steps'])
        print(f"RESUMED from {a.state}: {len(nodes):,} nodes, {steps:,} prior steps")

    A, B, members = pairs_from(train, nodes)
    rng = np.random.default_rng(0)
    if emb is None:
        emb = rng.standard_normal((len(nodes), a.dims)) * 0.01
        print(f"fresh start: {len(nodes):,} nodes, {a.dims} dims")
    elif len(nodes) > len(emb):
        emb = np.vstack([emb, rng.standard_normal((len(nodes) - len(emb), emb.shape[1])) * 0.01])
        print(f"grew to {len(nodes):,} nodes")

    layer_of = np.zeros(len(nodes), dtype=np.int64)
    for (l, e), i in nodes.items():
        layer_of[i] = l

    print(f"positive pairs/epoch: {len(A):,}   lr={a.lr} (CONSTANT), negs={a.negs}")
    for ep in range(a.epochs):
        prev = emb.copy()
        t0 = time.time()
        sgd_epoch(emb, A, B, layer_of, members, a.lr, a.negs, rng)
        steps += len(A)
        # On the unit sphere, chord distance is a direct read of how far each
        # expert actually moved this pass.
        disp = np.linalg.norm(emb - prev, axis=1)
        cos = np.clip(np.einsum('ij,ij->i', emb, prev), -1, 1)
        print(f"  epoch {ep+1}: {time.time()-t0:5.1f}s  mean move {disp.mean():.5f}  "
              f"median {np.median(disp):.5f}  mean angle {np.degrees(np.arccos(cos)).mean():6.2f} deg")

    keys = np.array(list(nodes.keys())); vals = np.array(list(nodes.values()))
    np.savez(a.state, emb=emb, keys=keys, vals=vals, steps=steps)
    print(f"saved {a.state}  (total steps {steps:,})")


if __name__ == '__main__':
    main()
