# Plan: work stemming from the VRAM-item table

Context: a table auditing what lives in VRAM (ideal case: model fits entirely;
our case: larger-than-VRAM MoE model, this fork's actual placement/cache
stack) surfaced several concrete proposals for the "our case" column. This
file tracks the resulting build plan, one track per proposal, so it survives
across sessions instead of living only in chat history.

## Track 1 — Atlas-driven cache warming

Stems from the "MoE expert weights" row: use the already-existing Expert
Atlas (per-expert topic-affinity data, measured offline by
`llama-expert-atlas`, feeding the Brain/Atlas UI) as an extra promotion
signal for the VRAM expert cache, on top of plain LFRU heat.

1. **[done]** Step 0 — live request-direction tracking. `ggml_moe_cache.set_atlas`
   registers each expert's measured `(x, y, spec)`; moe-cache maintains a
   live, decaying centroid (`req_dir_x/y`) of where the current request is
   trending in that space, updated on every real routing decision (hit or
   miss). Purely observational — nothing reads it yet. Verified end to end
   on a live server (60 tensors registered, centroid genuinely tracks real
   traffic across a 200-token generation, no crash). Commit `5fc3f803e`.
2. **[done]** Expose `req_dir_x/y` via `/experts` (`stats.req_dir`, omitted
   until at least one atlas-covered expert has been selected). Verified live:
   absent before generation, real coordinates after. Commit `5b739f7dc`.
   Frontend rendering (an actual marker drawn on the canvas) is not part of
   this - only the data is available so far.
3. **[done]** Warming action — `moe_cache_atlas_warm`, rate-limited to once
   every 64 `plan()` calls, ranks a tensor's atlas-covered candidates by
   cosine similarity to `req_dir` (discounted by specialization confidence),
   admits top-K into free slots only - **never evicts**. Off by default
   (`GGML_CUDA_MOE_CACHE_ATLAS_WARM=1` to enable), not yet A/B'd for a real
   effect, same discipline as everything else predictive here. Caught and
   fixed a real bug during first live test: atlas expert indices weren't
   bounds-checked against the tensor's actual expert count, causing an
   out-of-bounds read (confirmed segfault, not hypothetical) when tested
   with a mismatched model/atlas pair - fixed with the same `expert <
   n_expert` guard every other caller already had. Verified clean afterward
   across two real generations (800 tokens total), no crash, hit rate
   climbing to 94%.
4. **[done]** Pairwise co-activation ("fire together")
   signal. Split in two once actually scoped:
   - **4a, done**: within-layer co-activation - which experts get selected
     *together* in the same real routing decision (`device.co_activation`,
     an unordered-pair count keyed by tensor + both expert indices). Capped
     to `n_ids <= 32` so a large prefill batch's O(n²) pass over its own ids
     never runs - verified no slowdown on a ~2000-token prefill (1.8s, no
     crash) and no crash across normal decode traffic either. Observational
     only, same as req_dir - nothing reads it yet.
   - **4b, done**: the token/batch-boundary prerequisite turned out to
     already exist - `moe_cache_session_enter` is called exactly once per
     real `ggml_backend_sched_compute_splits` invocation (one graph compute
     = one forward pass/batch) via an existing RAII scope in
     `ggml-backend.cpp`, so resetting `last_top_expert_valid` there needed
     no new API surface at all. Cross-layer edges tracked using `ids[0]`
     (rank-ordered router output, index 0 = top pick) as each layer's
     representative expert, into `device.co_activation_cross_layer`.
     Imperfect for a multi-sequence batched compute call (mixes different
     sequences into one shared "last layer" state for that call) - same
     granularity limit `req_dir_x/y` already has as a single per-device
     value, not pretending otherwise. Verified live: 3 concurrent requests
     (the case most likely to expose issues in session-wide state), a real
     ~2000-token prefill, no crash in either. This is the more direct answer
     to "why can't the atlas be dynamic" than nudging static positions from
     a live-classified topic would be.
5. **[built, then REVERTED]** 3D topic-space visualization (`ec1354bc1`,
   reverted in `de08f87b2`). Built as a real 3D sphere with topic (not layer)
   on the third axis, Fibonacci-sphere anchors, orbit/zoom. Reverted at the
   user's request: too taxing on a machine that is usually also serving the
   model. The cost I had underweighted - the full atlas ships in the
   `/experts` response on **every 1.5s poll**, so adding the per-category
   `cats{}` vector inflated that payload from 1.2MB to 2.0MB per poll per
   open tab, on top of the per-frame draw cost. The active atlas file was
   reverted to the pre-`cats` v1 for the same reason.
   Kept (data-side, no per-frame or per-poll cost): the generator still
   emits `cats{}` (opt-in by regenerating), the co-activation export API,
   and the server-side tensor->layer translation. Those are what Track 1.6
   would build on. If revisited, WebGL would suit an interactive connectome
   better than the 2D-canvas projection used here.

**Same-layer artifact in cross-layer edges — FIXED** (`9f68d2f64`). One
logical layer dispatches several expert tensors (gate_up_exps then
down_exps) off a single router decision, so consecutive `plan()` calls were
frequently the same layer and got recorded as self-edges - measured at 55 of
64 exported edges on Ornith and 45 of 64 on Gemma, i.e. ~86% of the signal
was artifact. Fixed by parsing the logical layer from the tensor name
`begin()` already receives (llama.cpp names every per-layer tensor
`blk.<layer>.<what>.weight`, so no new API parameter was needed) and only
recording an edge when the layer actually changes; an unparseable name
yields -1 and records nothing rather than guessing. Verified live on
Ornith: **64 of 64 exported cross-layer edges are now genuine transitions,
0 self-edges** (was 9 of 64 genuine) - a 7x increase in usable signal.

**Step 5 prerequisites (done)**: storage bug fixed (`moe_cache_edge`
replaces the original opaque-hash counters for both `co_activation` and
`co_activation_cross_layer`); export API added
(`ggml_moe_cache.get_co_activation`, top-K by count, real tensor identity);
server-side tensor->layer translation done (`atlas_tensor_layer` map,
reusing the mapping already built for `set_atlas` registration).

**Step 3's real-world verdict, tested twice, two different data sources**:
- *Mismatched data* (Ornith's atlas on Gemma - arbitrary topic labels, no
  real relationship to what Gemma's experts do): flat-to-noise on both
  tok/s and hit rate, both the conservative and looser "speculative" eviction
  variants, across 2 interleaved rounds. See the git log for the full numbers.
- *Matched data* (Gemma's own atlas, `/mnt/nvme/models/moe-test/expert-atlas.json`,
  n_layer=30, on Gemma): a materially different picture. The conservative
  variant (`GGML_CUDA_MOE_CACHE_ATLAS_WARM=1`, defaults `EVICT_MIN=0.6
  EVICT_CAP=1`) beat baseline tok/s in **both** interleaved rounds (58.44 vs
  58.28, 58.49 vs 58.33 - consistently, not a coin flip like the mismatched
  test), a small but real +0.29% average. The looser "speculative" variant
  (`EVICT_MIN=0.0 EVICT_CAP=2`) won hit rate consistently (~93% vs ~91%
  baseline in both rounds) but showed real instability on tok/s - a 54.8
  tok/s outlier in round 1 (a genuine ~6% drop, not noise), suggesting more
  frequent eviction sometimes costs more than the hit-rate gain returns.
  Only 2 rounds so far - real, consistent direction, not yet enough samples
  to call fully proven. Needs a longer run (more rounds, larger/richer atlas)
  before treating this as validated - in progress.
- **Update, 4-round re-test on the same matched data**: the 2-round signal
  did not hold up. Conservative: tok/s 58.32 avg, exactly tied with
  baseline's 58.32 avg, full range overlap (58.07-58.50 vs 57.97-58.53).
  Speculative (n=3, one round lost to an unrelated build-vs-test race, not
  the mechanism's fault): hit rate 93.47%, now *below* baseline's 94.86%,
  reversed from the earlier apparent win. Textbook regression-to-the-mean -
  a small early sample looked like a trend by chance. **Corrected verdict:
  no measurable benefit from atlas warming on real matched data, at this
  sample size. Not proven harmful either - everything sits within noise of
  baseline.** Stays off by default. Documented plainly rather than kept as
  the earlier, more optimistic read.
- **Final, n=8**: baseline 58.10 tok/s / 94.83% hit rate, conservative 58.29
  / 94.19%, speculative 58.00 / 94.05%, spec_agree 58.34 / 94.30%. All four
  configs land within a ~0.34 tok/s / ~0.78pp band - confirms flat. Tested
  on a steady single-topic workload, where LFRU alone is already near-
  optimal - see the design-flaw note below for why this doesn't settle
  whether the *corrected* design (below) would do better on a topic-switch
  workload, which is what it's actually meant for.

**Real design flaw found, not yet fixed**: `moe_cache_atlas_warm` competes
for free slots and evicts probation candidates to admit *new* experts - the
same operation router-lookahead prefetch already does, just with a weaker
prediction signal (cosine similarity to a decaying centroid vs. the real
router's own one-layer-ahead logits, 59.3% precision at depth 1). That's
structural redundancy with an existing, better mechanism, not a distinct
purpose - explains why the A/B tests kept coming back flat regardless of
tuning. The right purpose for topic-affinity data is different: not "what to
fetch" (already router-lookahead's job) but **"what to keep"** - of experts
already resident and cooling toward eviction, protect the ones topically
aligned with where the request is heading, even if their raw LFRU heat says
they're not urgent. That's a genuinely different signal neither LFRU nor
router-lookahead has access to. The correct integration point is
`moe_cache_weighted_heat`/`moe_cache_cost_tier_weight` (already built earlier
today for the NVMe-vs-RAM eviction cost tier) - topic alignment should be
another factor in eviction-candidate scoring there, not a second competing
admission path. Real redesign needed, not a tuning fix - not started.

Separately, all A/B testing so far used a single fixed-topic prompt
generated start to finish - exactly the regime where LFRU's reactive signal
is already near-optimal (nothing changes, nothing to predict) and where
atlas warming (in either the current or the corrected design) has no room to
show a difference. Built a topic-switch test (code-heavy prompt, then an
immediate hard pivot to medicine, measuring the post-switch request
specifically) and ran it against the OLD (pre-redesign, admission-competing)
mechanism before removing it from source. **Real result, not noise**: hit
rate right after the switch was higher with warming on in **4 of 4 rounds**,
no exceptions (89.31% avg vs 86.56% off, +2.75pp), at a small tok/s cost
(58.22 vs 58.40, -0.3%). This is a genuinely different, much stronger signal
than anything the steady-state tests showed - confirms the workload
diagnosis was right, and that the underlying concept has real value even in
the flawed (redundant-with-prefetch) design. Re-testing the corrected
eviction-weighting design against this same topic-switch workload next, to
see whether it preserves or improves this result.

**Resolution - three-way comparison, same topic-switch workload, same
methodology throughout:**

| Variant | Rounds won (of 4) | Hit-rate delta vs. off | tok/s cost |
|---|---|---|---|
| Admission only (`moe_cache_atlas_warm`, original) | **4/4** | **+2.75pp** | -0.3% |
| Eviction-weighting only (the "corrected" redesign) | 3/4 | +0.63pp | -0.15% |
| Both combined | 1/4 | **-0.95pp (net negative)** | -0.84% |

Two real reversals in sequence, both reported honestly rather than kept as
the more optimistic earlier read: (1) the theoretically-cleaner eviction-
weighting redesign measurably underperformed the "redundant" admission
design it was meant to replace - right after a topic switch, most newly-
relevant experts genuinely aren't resident yet, so "protect what's already
there" has nothing to work with until something else brings them in first;
(2) combining both (on the theory that they're complementary, not
competing) measured *worse* than either alone, not better - best guess is
real interference, not addition: the eviction-weighting factor makes
topically-aligned candidates resist eviction everywhere, including when
`moe_cache_atlas_warm`'s own admission calls need to evict something to
make room for a *different* atlas-suggested candidate, so admission ends up
fighting itself.

**Final state**: reverted to admission-only (`moe_cache_atlas_warm`,
unchanged from the original design) - the clear, empirically-best performer
across all three variants tested. `moe_cache_atlas_align_weight` (the
eviction-weighting factor) was removed from the codebase entirely, not left
disabled - it measured worse standalone and worse combined, with no
condition under which it won. Still gated behind
`GGML_CUDA_MOE_CACHE_ATLAS_WARM`, off by default; the topic-switch result
(+2.75pp hit rate, 4/4 rounds, small tok/s cost) is real but has not yet
been re-validated with more rounds the way the steady-state tests were -
worth doing before considering a default flip.

## Step 6 — measured: is the co-activation signal actually predictive?

Built opt-in scoring (`GGML_CUDA_MOE_CACHE_MEASURE_PRED=1`, commit
`95456179c`) rather than building a policy on an unmeasured signal - this
session had already produced three mechanisms that were built, A/B'd, and
found flat or reverted.

**Result (Ornith-1.5-35B, 22000 scored predictions, 9475 tracked edges):**

| predictor | depth-1 precision | cost per layer |
|---|---|---|
| router-lookahead (existing) | **59.3%** | real `n_embd x n_expert` matmul |
| cross-layer co-activation | **33.8%** | hash lookup |

Stable as the table grew (33.8 / 33.7 / 33.0 / 33.8 over 16k-22k samples),
so converged, not undertrained. Worse than router-lookahead at depth 1, and
worse even than its depth-3 precision (41.1%).

Structural reason: the router predicts from the **actual hidden state of
the current token**; the co-activation table is a static frequency prior
with no token-specific information, and routing is data-dependent. A prior
cannot compete with a computation that sees the input.

**Consequence: step 7b (co-activation as a cheaper/deeper prefetch
predictor) is DROPPED, not built.** Measuring cost ~15 lines; building it
would have cost far more and ended flat.

**Step 7a - MEASURED, and it is strong** (commit `ea822e946`). Different
question from step 6: "given one expert in a routing decision, is its most
frequent partner also selected in that SAME decision?"

  within-layer partner precision@1:  **67.6%** (54080/80000 scored decisions)
  chance baseline:                     2.8%  (from real n_ids / n_expert)
  -> **24x over chance**, stable and rising slightly as the table matures
  (cross-layer prediction sat at 37.2% in the same run, for contrast)

This differs from step 6 in kind, not degree. Step 6 was *prediction*,
competing against a router that sees the token's actual hidden state and
therefore wins. 7a measures co-occurrence *structure*, which the router's
per-token computation does not help the cache exploit at all - the cache
still admits **one expert at a time**, so when A is admitted and B co-fires
67.6% of the time, that is a miss taken structurally, not for want of a
prediction.

**This is the first result in Track 1 that justifies building on it.**
Next: group-aware admission - admit/evict correlated experts as a unit
rather than individually - then A/B it like everything else. Not built yet.
Worth noting the honest risk before building: a 67.6% partner hit rate does
not automatically mean group admission wins, because admitting B early also
costs a slot that something else could have used. The A/B decides it, not
the precision number.

## Track 1.6 — Discovered topics from co-activation — ATTEMPTED, inconclusive (2026-08-25)

Built (`scripts/moe-discovered-atlas.py`): factor the co-activation graph, use
its components as axes, no category list at all. Global graph over all
(layer, expert) nodes rather than one embedding per layer — per-layer
factorings each come out in their own arbitrary rotation, and `req_dir`
averages positions across layers, so it needs one shared space. Randomised
subspace iteration over sparse matvecs, so 8,661 nodes never need a dense
factorisation. Output matches the existing atlas schema, so it is a drop-in.

Built on co-activation, NOT `req_dir`, which is the whole point — the
circularity argument below still holds. Verified in code that substitution
does not break it: `co_activation` is keyed on `ids[a]`/`ids[b]`, the
router's *requested* experts, so a substituted pick never feeds back.

**Held-out result** (graph from the first half of each trace, scored on the
second): the discovered axes carry essentially no topic information.

| space | Fisher | topic-acc (chance 25%) |
|---|---|---|
| human 9 categories (probe-measured) | 0.4136 | **71.6%** |
| discovered (raw) | 0.0005 | 25.5% |
| discovered (layer-centred) | 0.0030 | 31.0% |

**Diagnosis — the axes encode LAYER, not topic.** Fisher ratio for layer
identity is **2.5420** against 0.0005 for topic. A co-activation graph is a
set of per-layer cliques (experts only co-fire within a layer), so that is by
far its dominant structure and spectral factoring spends every component on
it. Not a cross-layer sampling artifact either: weighting all-pairs
cross-layer edges up to 64x moved layer-Fisher only 2.475 → 2.279, with topic
accuracy pinned at chance throughout.

**Why this is NOT yet a verdict.** Two objections, both fair:

1. *The data was far too thin.* 28,800 decisions over 4 topics gives a
   co-activation graph at 22.9% fill with **median edge weight 2, and 46% of
   pairs seen exactly once**. Half the graph is noise. Projection: median
   weight 20 needs ~10x the data (~1,800 tokens/topic), median 50 needs ~25x.
2. *One-shot spectral is not the mechanism described.* The design intent was
   experts that co-fire *moving nearer* and the map *settling over a long
   run* — incremental attraction dynamics, accumulated and persisted like
   `session.history`. Eigendecomposition solves a fixed objective on a fixed
   snapshot in one step; it has no notion of settling and no memory between
   runs. It is a different algorithm, not a fast approximation of that one.

What does survive, because it is about graph topology rather than counts: the
layer blocks are intrinsic and any method here must handle them explicitly.
The folds, if they exist, live *within* each layer with cross-layer alignment
stitching corresponding regions — not as one undifferentiated cloud, which is
what the global embedding assumed.

### Label-free rebuild on 31.6x the data — the discovered atlas still loses

Rebuilt on 909,480 pooled decisions from 12 topics (2,000 tokens each), with
the topic labels discarded entirely, and scored by held-out link prediction
(`scripts/moe-link-prediction.py`): given two experts in a layer, will they
fire in the SAME routing decision later? Positives are held-out pairs that
were **never seen in training** — predicting a pair already observed is
memorisation, not prediction. Negatives are sampled within the same layer, so
layer structure confers no advantage either way.

The data objection is now answered: median edge weight 4 (was 2), mean 16.0,
seen-once down to 21.8% (was 46%).

| method | AUC |
|---|---|
| popularity control (`log f_a + log f_b`) | 0.5437 |
| **probe atlas (9 categories)** | **0.7116** |
| discovered co-activation (16 dims) | 0.5706 |
| discovered co-activation (32 dims) | 0.5715 |
| discovered co-activation (64 dims) | 0.5761 |

The discovered embedding **does** beat the popularity control, so it has
learned something real rather than reflecting expert frequency — but it
saturates around 0.576 and under-parameterisation is not the explanation.

The uncomfortable part, stated plainly: **the probe atlas predicts co-firing
better than an embedding built from co-firing itself**, and it does so having
never seen this traffic. Topic affinity appears to be a better organising
principle for "which experts work together" than the co-firing graph's own
spectral structure. That is an argument FOR keeping the 9 categories, arrived
at by a test designed to be fair to the alternative.

**Still genuinely open**: the algorithm. Everything tested so far is one-shot
spectral factorisation of a snapshot. The design intent — co-firing experts
*moving nearer*, the map *settling over a long run*, persisted across restarts
like `session.history` — is incremental attraction dynamics and has not been
built. Spectral factoring is not a fast approximation of it. That remains the
untested hypothesis, and it is the one worth building next.

**Next**: rebuild on ~34x the data (12 topics x 2,000 tokens), pooled and
UNLABELED, and score by held-out link prediction
(`scripts/moe-link-prediction.py`) rather than by agreement with the 9
categories — scoring "did it recover our labels" is rigged toward the probes.
The popularity control in that script is the one that matters: frequently-used
experts co-fire often merely because they are frequent, so any embedding that
cannot beat popularity has not earned its complexity.

## Track 1.6 — original note (circularity argument, still stands)

The atlas's 9 categories are human guesses hardcoded in `k_probes[]`, and a
*dynamic* atlas built on them has a trap: live traffic carries no category
label, so the only available labeller is `req_dir` — itself inferred from
the static atlas. Updating the atlas from that is self-training, and the
known failure mode is confirmation drift (a slightly wrong estimate
reinforces itself).

Co-activation (steps 4a/4b, already collected) sidesteps both problems at
once. "These two experts fired in the same routing decision" is a direct
observation, not an inferred label, so there is no circularity — and
embedding/clustering that graph yields dimensions that are *discovered*
rather than assigned, including structure no human would have thought to
name. One mechanism answers both "can the atlas be dynamic" and "must it
use preset topics", with no damping leash required.

Sketch: build an expert-by-expert co-activation matrix, factor or cluster it
(spectral / NMF / graph embedding), use the resulting components as axes.
The `cats{}` vector now emitted by `llama-expert-atlas` makes the two
directly comparable — discovered axes can be scored against the labelled
ones to see how much of the human category set the model actually
reproduces on its own. Not started; needs its own design pass.

If instead staying with labelled probes, the cheap intermediate is simply
more categories: `cats{}` is schema-stable under a changing category set,
and the sphere anchors are computed for any N, so growing `k_probes[]` and
regenerating needs no code change on either side. Untested — no measurement
has been taken of whether more categories actually improves anything.

## Track 1.5 — CPU-GPU split gap (from FreeToken, never carried over)

Real gap identified while re-testing step 3, not hypothetical: the atlas
warming ranking pass (`moe_cache_atlas_warm`'s cosine-similarity scan over a
tensor's whole atlas-covered candidate set) runs **inline, synchronously, on
the critical GPU-dispatch path** (`moe_cache_plan`, which must complete
before the GPU op for that layer can be issued) - not offloaded to the
existing background fill-worker thread the way the actual expert-weight
*copies* already are. FreeToken's reference architecture keeps this kind of
CPU-side prediction/ranking work off the hot dispatch path, running on spare
CPU cycles while the GPU is busy with the previous layer's compute, instead
of blocking the thread that has to issue the next GPU op. We didn't carry
that split over when building Track 1 - everything CPU-side (the ranking
loop, not just the fill worker's actual copies) has been synchronous so far.

**Hypothesis, not yet measured**: moving the ranking computation itself onto
the background worker (leaving only "read the precomputed result and admit a
slot" on the critical path, the same shape moe_cache_plan already uses for
real demand fills queued to the worker) could reduce the per-call latency
atlas warming currently adds every 64 plan() calls, independent of whether
the warming decision itself is good. Not started - needs its own scoped
design, likely reusing the existing `device.queue`/worker-thread plumbing
rather than inventing new infrastructure.

**BUILT (2026-08-24), NOT YET MEASURED.** The split now exists, off by
default behind `GGML_CUDA_MOE_CACHE_ATLAS_WARM_ASYNC=1` (and only reachable
at all with `GGML_CUDA_MOE_CACHE_ATLAS_WARM=1`, which is itself off by
default). It reuses the existing fill-worker thread and session lock; no new
thread, no new synchronization primitive.

What changed:

- `moe_cache_atlas_warm` was split into its two real halves.
  `moe_cache_atlas_rank` is the expensive one - a cosine scan over the
  tensor's whole atlas-covered candidate set, a `sqrt` and a few flops each,
  up to `n_expert` of them - and is now *pure*: it touches no cache state and
  takes no lock. `moe_cache_atlas_admit` is the cheap one - a few hash
  lookups, at most one eviction, a queue push - and is the only half that
  needs `session.mu`.
- Async path: `moe_cache_plan` now does a bounds check, a hash lookup and a
  `push_back` (`moe_cache_atlas_warm_enqueue`), and nothing else. The fill
  worker picks the request up in its existing wait loop, **releases the
  session lock for the ranking scan**, retakes it, re-validates the pool, and
  admits. Ranking therefore overlaps whatever the GPU is doing, which is the
  FreeToken split this port never carried over.
- The warm queue is deliberately **one deep**. A warming hint describes where
  the request is heading *now*; if one is still pending, a second is already
  stale by the time it would run, and queueing it would only make the worker
  do the same scan twice. Dropping is correct, not a shortcut.
- `atlas_by_tensor` rows are now `shared_ptr<const ...>` and `set_atlas`
  publishes a replacement row instead of mutating in place. Without that, a
  re-registration could free the vector the worker is mid-scan on. The
  ranking pass also captures `req_dir_x/y` **by value** at enqueue time, so
  it ranks against where the request was heading when the hint was raised,
  not wherever the centroid drifted to by the time the worker got there.
- The sync path (default) still runs both halves inline under the lock, so
  the old behavior is what ships and the two are directly A/B-able.

**One real behavior change, not hidden:** the old scan skipped
already-resident experts inline, which needs `pool.map` and therefore the
lock. The lock-free pass cannot, so it keeps a deeper list
(`MOE_CACHE_ATLAS_RANK_K=4`) and the admission half - which holds the lock
and already re-checked residency anyway - walks down it admitting the same
top-2 non-resident candidates in the same score order
(`MOE_CACHE_ATLAS_ADMIT_K=2`). The two differ only when 3+ of the top 4 are
already resident. Small, but it means the +2.75pp topic-switch number above
was measured with the *old* scan and would need re-validating before any
default flip.

**Measurement hook, not yet run**: `GGML_CUDA_MOE_CACHE_ATLAS_WARM_TIMING=1`
accumulates the time the warming block costs *on the dispatch path
specifically* and prints a running average every 256 warm calls. That is the
quantity this track is about, and tok/s is the wrong instrument for it - the
block fires once every 64 `plan()` calls, so even a large per-call saving is
a rounding error at the throughput level. Measuring the thing directly, for
the same reason steps 6 and 7a were measured before anything was built on
them.

### MEASURED (2026-08-24) - the hypothesis holds, and the caveat matters more

Ornith-1.5-35B Q4_K_M, `--moe-cache auto -c 65536 -ngl 99`, its own matched
atlas, `GGML_CUDA_MOE_CACHE_ATLAS_WARM=1`, scratch server on 8098 (the live
8099 deployment was stopped for the test and restored to its exact
last-known-good command immediately afterward). Two interleaved rounds per
arm, four 700-token generations each (alternating a code-heavy and a
medical prompt), timing read from
`GGML_CUDA_MOE_CACHE_ATLAS_WARM_TIMING=1`.

| | dispatch-path cost | hit rate | prefetches | evictions |
|---|---|---|---|---|
| sync (old, default) | **4.9 us** / 4.9 us | 77.60% / 77.53% | 3591 / 3584 | 53015 / 53268 |
| async (Track 1.5) | **0.6 us** / 0.5 us | 77.67% / 77.71% | 3592 / 3623 | 52808 / 52535 |
| | **-88%** | +0.13pp | +0.6% | -0.9% |

Both arms coherent, zero `CUDA error` / `ggml_abort` / copy-failure-disable
markers across all four servers. The per-arm averages were flat from the
first 2048-call print to the last (4.8-4.9 vs 0.5-0.6), so this is converged,
not a warmup artifact - the cleanest signal since the D2D measurement, and in
the same style: near-zero within-arm variance, no overlap between arms.

**What it means, stated narrowly.** The ranking scan really was ~4.3 us of
synchronous CPU work sitting on the GPU-dispatch path, and moving it to the
worker removes essentially all of it - what remains (0.5-0.6 us) is the hash
lookup and `push_back` the enqueue still has to do. The cache does the *same
work*: prefetch counts land within 1% and hit rate within 0.2pp, i.e.
deferring the ranking by one worker hop did not make it rank against a
meaningfully staler direction, which was the main correctness worry.

**What it does NOT mean.** This is not a throughput win and should not be
reported as one. tok/s was 58.6-62.0 in both arms with full overlap
(59.72/61.93/58.61/60.97 sync vs 59.11/62.00/59.25/61.32 async) - exactly as
expected, because 4.3 us saved once every 64 `plan()` calls is ~0.07 us per
call against a ~17000 us token, i.e. four orders of magnitude below anything
tok/s can resolve. The honest summary is: **the latency this track set out to
remove is real and is now gone, and it was never large enough to matter for
throughput at the current 1-in-64 rate limit.** That is a completed
architectural fix, not a performance result.

Where it *would* start to matter: any future change that raises the warming
rate (a lower rate limit, per-layer warming, a larger candidate set, or a
richer atlas with more categories per expert - all of which scale the scan,
none of which now scale the dispatch path). The split is what makes those
affordable to try; it is not itself a win to bank.

Still off by default. Nothing here re-validates the +2.75pp topic-switch
result, which was measured with the old residency-filtering scan (see the
behavior-change note above) - that remains the open question before any
default flip, and this measurement deliberately did not touch it.

## Track 2 — Unified tiered placement — RESCOPED (mostly already exists)

Original framing was "remove the hard `-ncmoe` boundary, give every MoE layer
one global LFRU budget" - loader change, budget-math change, cache-sizing
change. Investigating the code before building showed most of that is
already true:

1. **Pools are keyed by `(expert_size, wtype)`, not by layer**
   (`moe_cache_find_pool`). A single global budget spanning all layers
   already exists - every layer sharing a tensor shape shares a pool.
   Nothing to build.
2. **Offloading every MoE layer is already expressible** as
   `-ncmoe <n_layer>`; `common_moe_build_cpu_overrides(n)` simply emits an
   override per layer below `n`. No loader change needed.
3. What genuinely remains is a **default-policy question**, not an
   architecture one: `common_maybe_raise_moe_for_ctx` picks the *minimum*
   offload that makes the requested context fit. Its own comment already
   flags this as "a sufficiency question, not a throughput one", with a
   measured counter-example (Nemotron 3.5: fit search stopped at 46 layers
   -> 1239 slots -> 57.1% hit rate; offloading all 53 -> 2307 slots ->
   68.3%). `--moe-calibrate` already finds the throughput optimum
   empirically, but is opt-in.

**So the remaining work is an experiment, not a rewrite**: compare
(a) auto placement (minimum that fits) against (b) `-ncmoe <n_layer>`
(maximum offload, larger cache budget) on tok/s and hit rate, on this
hardware and model. If (b) wins consistently, the change is to the default
placement policy - a small, well-contained edit - not to the loader or the
cache. Testable with zero new code.

Caveat worth stating before running it: more offload means more experts
served from the cache, which is only a win while the cache can hold enough
of the working set. Past some point it trades resident weights for cache
capacity that cannot cover the extra demand, and the constrained-regime
result (33% hit rate at a 512 MiB budget) shows what the far end of that
looks like. The optimum is empirical, which is exactly why
`--moe-calibrate` exists.

**Status (2026-08-24): NOT RUN.** The rescope (`2b721d844`) is the last thing
that happened here - it established that items 1 and 2 need no code, which is
what makes this track look further along than it is. The one thing actually
left, the (a)-vs-(b) placement comparison, has not been measured: no commit,
no numbers, nothing in `docs/index.html`. It needs no code at all, only the
GPU - and the same 8099 occupancy blocking the Track 1.5 A/B blocks it too.
Track 2 is therefore *descoped*, not *done*.

## Track 3 — "True MoE" emulation

Newest addition, **not yet scoped**. Needs a clarifying conversation before
a real plan exists, because two very different projects both fit the name:

- MoE-izing the *dense* attention layers themselves (a genuine "mixture of
  attention experts" architecture change) — this is the only way the earlier
  "attention weights, only for the agent using them" table proposal becomes
  literally buildable, since standard dense attention has no per-token
  selection mechanism to hang caching off of.
- Using the *existing* MoE routing plus Atlas/co-activation data (Track 1) to
  treat clusters of frequently-co-activated experts as emergent "sub-models"
  within the current architecture — no architecture change, a different lens
  on data we're already collecting.

## Degenerate output under VRAM exhaustion — ROOT-CAUSED AND FIXED (2026-08-24)

**Root cause found and fixed** (`b321b6259`).
`moe_cache_prefill_advance()` published a prefill slot
(`inflight_host[slot] = next_host_base`) **unconditionally**, even when the
copies into that slot had failed. `prefill_wait()` uses `inflight_host` to
decide the buffer holds live data, so it then handed
`ggml_cuda_mul_mat_id` a buffer still containing the previous tenant's
bytes - read as expert weights. Every copy *was* error-checked; the return
values were discarded. Under VRAM pressure that is exactly how a cache
turns into a source of fluent-looking nonsense with a clean log.

Fixed by making `copy_split` report whether every copy landed, refusing to
publish a slot whose copy failed (so `wait()` returns NULL and the caller
falls back to its normal H2D path), and disabling the cache entirely after
N consecutive copy failures with a loud one-time log. Also removed the
`cudaErrorMemoryAllocation` exemption from `moe_cache_cuda_ok`'s fatal
branch - it was meant for retryable allocation sites but also covered OOM
on copy paths, where nothing retries and the buffer keeps stale bytes.

Historical detail retained below, including two wrong turns.



A live server returned **pure `////////` for every token**. Root cause is
almost certainly **VRAM exhaustion**, not any of this session's changes -
the user reports seeing it before the atlas work existed at all.

**Trigger correlates with VRAM pressure from a co-resident GPU process**,
in every observed case:
- first sighting: server started immediately after `llama-expert-atlas`
  had been holding the GPU
- second sighting: a 64k server running while diagnostic servers were
  launched alongside it
- a deliberate contention test: free VRAM already at 152 MiB, dropping to
  43 MiB once a second instance loaded - the request then hung outright
- every **single clean instance** test came back coherent, including
  `--moe-cache auto` *with* the atlas file (3/3 runs)

**Only reproducible through the chat-templated prompt**, not a raw one:
`<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n<think>\n`.
A plain `/completion` prompt stays coherent even in a failing instance,
which is why an early pass at this wrongly concluded "not reproducible" -
it was testing the wrong endpoint. A later pass wrongly blamed the atlas
registration; a clean single instance with the atlas is fine.

Ruled out: Ornith itself, the chat template, the atlas file (v1 and v2),
context size alone, and moe-cache alone. Each is fine in isolation.

**The real defect is the failure mode, not the OOM**: moe-cache sizes its
slab from free VRAM at init, and when a co-resident process takes VRAM
afterward, it degrades into silently producing wrong weights - garbage
tokens - instead of raising a clean allocation error. The corrupted
instance logged no CUDA error, no fill failure, no dispatch failure, and
also hung on shutdown (ignored SIGTERM ~12 min, needed SIGKILL).

**Not fixed.** The worthwhile fix is defensive: detect allocation
failure/VRAM pressure after init and fail loudly (or fall back to
`--moe-cache off`) rather than emitting garbage. Practical workaround
today: run one model instance per GPU and check `nvidia-smi` for
competing processes before blaming the model.

## MTP speculative decoding: works, but NOT on Ornith (2026-08-24)

Prompted by the Codacus video's claim that the expert cache is worth ~5%
alone and that its real value is the interaction with speculative decoding.
Tested that premise on our stack; it does not transfer, and turned up a
separate real finding.

**Ornith-1.5-35B (`qwen35moe`, 41 blocks, 256 experts,
`nextn_predict_layers=1`) with its built-in MTP head is a large
regression**, reproduced across two rounds:

| config | tok/s | draft acceptance |
|---|---|---|
| cache off, MTP off | 47.28 / 46.91 | - |
| cache auto, MTP off | 65.33 / 65.87 | - |
| cache off, MTP on | 24.54 | 0.203 |
| cache auto, MTP on | 22.21 | 0.187 |

Control on the same build, Gemma-4-26B with its SEPARATE MTP draft model
(`mtp-gemma-4-26B-A4B-it-Q8_0.gguf`): acceptance **0.718 / 0.738**. So the
`--spec-type draft-mtp` path is fine - MTP wires up correctly (zero
"unused tensor blk.*nextn" lines once the flag is passed, versus 4
without). **Ornith's built-in head simply produces bad drafts**: ~0.19
acceptance at mean draft length 1.6 means paying a draft pass plus a
verify pass and discarding 4 of every 5 tokens, which is exactly the ~2x
slowdown observed. Most plausible cause: Ornith-1.5 is a finetune whose
`nextn` head was not retrained alongside the modified backbone, so it
predicts for a model that no longer exists. Not fixable in llama.cpp - a
property of the checkpoint.

**Actionable**: do not use `--spec-type draft-mtp` with Ornith. Nothing in
the output warns you; the failure is silent and expensive. A diagnostic
that fires when acceptance stays very low would be worth adding.

**The Codacus premise does not transfer.** They report cache alone at
+5% (42 -> 44 tok/s). On our stack the cache alone is worth **+38%**
(47.28 -> 65.33 on Ornith), so we are not leaving the cache x spec
interaction unexploited - we already get most of that value from the
cache by itself. That also explains why every cache-POLICY tweak measured
flat today: the cache is already doing its job here, and the remaining
headroom inside it is small.

**Not established**: whether Gemma+MTP is a throughput win on this rig.
Round 1 showed +43%, round 2 showed +1.5%, at essentially identical
acceptance - so the acceptance gap is solid but the throughput claim needs
more rounds before it means anything.

## Silent corruption fix — VALIDATED (2026-08-24, overnight)

The fix in `b321b6259` is now proven against a deterministic reproduction,
not just assumed. Getting there corrected two of my own wrong conclusions.

**Wrong conclusion 1 (mine): "the prefill double-buffer is inert in server
mode."** It is not. There are TWO separate prefetch mechanisms and I was
enabling the wrong one for hours:
- `GGML_SCHED_PREFETCH_EXPERTS` - the ggml-backend scheduler's prefetch
  slots (the +12.7% pp2048 measurement)
- `GGML_CUDA_MOE_PREFILL_BUFFER` - moe-cache's own prefill double-buffer
  (`register_successor` / `advance` / `wait` / `copy_split`), which is
  where the corruption fix lives

With the correct flag the path is healthy: **122 HIT / 40 MISS**.

**Wrong conclusion 2 (mine): "fault injection proves the fix is
incomplete."** That run showed `////////`, but it was never exercising the
injected path (wrong flag), so it was the original intermittent bug
appearing spontaneously - not evidence against the fix.

**Validation, Gemma-4, `GGML_CUDA_MOE_PREFILL_BUFFER=1`, fault injection
every 3rd copy:**

| | no injection | injection |
|---|---|---|
| prefill HIT | 122 | 19 |
| prefill MISS | 40 | 143 |
| output | OK | **OK 3/3** |
| loud disable | - | **fired** |

Hits collapse (failed copies are no longer published), callers fall back,
output stays correct where this path previously produced `////`, and the
safety valve logs once: `CUDA0 DISABLED after 8 copy failures (likely VRAM
exhaustion) - falling back to uncached offload`.

**Still true and still worth fixing separately**: only `gemma4.cpp` calls
`build_moe_prefill_prefetch`, so the moe-cache prefill double-buffer (and
its LFRU **D2D** path) is unavailable to every other architecture,
including Ornith (`qwen35moe`). That is a real coverage gap, not a bug in
the mechanism.

## Prefill double-buffer coverage gap — attempted, REVERTED (2026-08-24)

Only `gemma4.cpp` calls `build_moe_prefill_prefetch`, so the moe-cache
prefill double-buffer and the LFRU **device-to-device** path riding on it
(a cache-resident expert copied D2D instead of over PCIe) are unavailable
to every other architecture, including Ornith (`qwen35moe`). Measured
before any change: **0 prefill hits on qwen35moe** (162 consulted, 162
miss) vs **122 hits / 40 misses on gemma4**.

Tried closing it by adding the same registration to `qwen35moe.cpp`.
**Registration works** - hits went 0 -> 14 (of 48 consulted) - **but the
model then crashes**:

```
prefill wait-for-consumed failed: operation not permitted on an event
last recorded in a capturing stream
CUDA error: operation failed due to a previous error during capture
-> ggml_abort
```

`moe_cache_prefill_advance` does a host-side `cudaEventSynchronize` on the
slot's `consumed` event before overwriting the buffer. That call is illegal
while a CUDA graph capture is active on the stream that recorded the event.
Gemma's capture boundaries evidently avoid this; qwen35moe's do not.

**Reverted** - a crashing change is worse than a missing optimization, and
this is not something to leave in the tree unattended. Verified clean after
revert (correct output, zero crash markers).

**So the coverage gap is not an oversight - it is gated on a real
capture-safety problem.** Closing it properly means making the
wait-for-consumed step capture-safe, e.g. skipping the host sync when a
capture is active and relying on stream ordering instead, or restructuring
so the overwrite cannot race the consumer. That is a careful, standalone
piece of work with a genuine correctness hazard (the wait exists to stop a
prefetch two calls later from overwriting a buffer still being read), and
should not be attempted casually. This is the natural next task for anyone
picking Track 1 back up, and it is the prerequisite for the D2D-instead-of-
PCIe idea on any non-Gemma model.

## moe-cache prefill double-buffer (the D2D path) — MEASURED, costs 5%

Never benchmarked before tonight: the documented +12.7% belongs to the
SEPARATE `GGML_SCHED_PREFETCH_EXPERTS` mechanism, not to this one. Gemma-4
is the only architecture that registers prefill successors, so it is the
only place this could be measured at all.

`llama-bench -p 2048 -ncmoe 15 --moe-cache auto`, 3 rounds:

| `GGML_CUDA_MOE_PREFILL_BUFFER` | pp2048 | values |
|---|---|---|
| off | **953.00** | 954.8, 951.8, 952.4 |
| on (D2D active) | 905.35 | 906.3, 905.0, 904.8 |
| | **-5.00%** | 3/3 rounds |

Near-zero variance within each arm - the cleanest signal measured in this
whole effort, and unambiguous.

**So the D2D path does not pay, even where it works.** D2D genuinely avoids
PCIe for cache-resident experts, but the double-buffer machinery wrapping
it - per-layer slab bookkeeping, host-side event syncs, `advance`/`wait`
lock traffic on every dispatch - costs more than the PCIe traffic it saves
at this scale.

**Direct consequence: do NOT do the CUDA-graph capture-safety work** needed
to extend this to `qwen35moe`. That effort would only make a measured
regression available to more architectures. The coverage gap documented
above is therefore not worth closing as things stand; it would need the
per-dispatch overhead reduced first, and only then re-measured.

## Resident-constrained routing ("use what is in VRAM") — 2026-08-25

Design premise: stop racing to fill VRAM in time, and instead run whichever
experts are already resident. Measured end to end this session.

### The premise is correct, and here is the measurement that proves it

Removing the admission gate entirely (`ADMIT_AFTER=1` + `THROTTLE=1`) buys
**+10.79pp hit rate and costs -7.6% throughput** (52.43 vs 56.75 tok/s,
evictions 110,169 vs 23,714). The racing *succeeds* at filling VRAM and loses
anyway, because fill traffic is the binding constraint. `ADMIT_AFTER=1` alone
is ~neutral (+0.6 tok/s), because `readmit_after` defaults to **8** and
`max(admit_after, readmit_after)` is the gate that actually binds for anything
previously evicted.

Corollary: **hit rate is purchasable and therefore not a valid objective on
its own.** 79.25% is available for one env var and a slower server.

### Prefetching is capped, not merely weak

Atlas warming gains **+0.31pp** hit rate in simulation. A *topic oracle* with
perfect foreknowledge of the segment's experts gains **+0.30pp**. The atlas is
already at the ceiling; the ceiling is the problem. Cause: every topic touches
~6,000 of 7,680 experts, with top-1000 covering only 52-61% of selections —
there is no compact topic hot set to preload. This single result explains
every refuted Track 1 A/B at once.

Atlas warming on/off on the live server, 2 reps each: **+0.27 tok/s, +0.71pp
hit, +509 fills out of ~23,000** — within run-to-run spread, i.e. neutral to
marginally positive. It is rate-limited to once per 64 `plan()` calls and
admission-gated, so it never generates enough fill traffic to compete with
demand. It is not free-loading.

### Where the headroom actually is: eviction

Belady's optimal beats LRU by **+16.86pp** at the real 18.5% cache ratio
(81.85% vs 64.98%). Cross-check: the live server measures 68.7% hit on
topic-diverse traffic against the LRU simulation's 64.98%, so the production
policy is behaving near-LRU here and that headroom is real.

Five signals tested as eviction policies (`scripts/moe-eviction-sim.py`), all
scored on retained gate mass, sampled eviction K=32:

| policy | retained | hit |
|---|---|---|
| **lru** | **75.72%** | 58.63% |
| random | 72.51% | 53.31% |
| coact | 72.74% | 58.12% |
| atlas | 71.06% | 51.59% |
| router score | 68.90% | 46.27% |

**LRU wins on retained mass at every fill rate**, and atlas and router score
lose to *random*. Both rank experts context-free, with no knowledge of what
was just used, so evicting by them tears up the working set recency is
holding. Belady is recency with perfect foresight — so the 17pp lives in
predicting **reuse distance**, and none of the available signals approximates
it. Co-activation is the sole exception: it beats LRU on **hit rate** at fill
rates above 1 (64.88% vs 63.37% at fill 4), reaching a higher hit rate with
*fewer* fills.

### Fill cost — and a retraction

Both simulators originally scored hits and gate mass with **no notion that
admitting an expert costs anything**, and consequently recommended "admit
everything" — the worst arm on real hardware. `scripts/moe_cost_model.py`
fits tok/s to three measured arms rather than inventing constants:

```
tok/s = -42.290 + 153.014*hit - 140.862*fills_per_pick
  1pp of hit rate  = +1.5301 tok/s
  1 fill/1000 picks = -0.1409 tok/s
  break-even: 1pp of hit must not cost more than 10.9 extra fills/1000 picks
```

Distinct from the cache's existing `MOE_CACHE_COST_TIER_NVME/RAM` weight (5.8
vs 1.0), which ranks *which* expert is dearer to refetch, not the cost of
filling at all. Applying it inverts the earlier conclusions: fill_rate 1 beats
fill_rate 8, and the corrected model reproduces the ordering it previously got
backwards ("admit everything" now ranks last, 10.34 vs 30.17). Absolute
predictions are NOT trustworthy — the sims run at 52-450 fills/1000 picks
against a calibration range of 40-188. Ranking only.

### Substitution — BUILT, measured, off by default

`GGML_CUDA_MOE_CACHE_SUBSTITUTE=1`. On a miss, run the best resident expert of
the same tensor instead of the CPU fallback. Mechanically only needs
`slot_indices[]` to point at a different slot; the gate weight applied
afterwards is still the one the router computed for the expert it asked for.

Stand-in chosen by **co-activation** — how often the candidate fired in the
same routing decision as the expert it replaces. Candidates enumerated from
the resident bitmask (one hash for the tensor, then bit tests over ~4 words),
which is why that structure exists: 64 hash lookups per decision on the
dispatch path is what Track 1.5 exists to avoid.

Admission bookkeeping deliberately still follows what the *router wanted*, not
what was executed, or resident experts become the only ones ever used and
therefore the only ones ever kept. A stand-in earns heat but never promotion
to protected.

| | tok/s | hit | substitutions |
|---|---|---|---|
| OFF | 56.82 | 67.62% | 0 |
| ON | **59.88** | 65.51% | 200,637 (99.5% of misses) |

**+5.4% throughput.** Quality, paired per-chunk over 120 chunks (the unpaired
±1.5% bars cannot resolve this; pairing tightens it 11x):

```
mean   +0.181% perplexity
95% CI [-0.106%, +0.468%]     t = 1.23, NOT significant
chunks where ON was worse: 59/120 (49%)
```

So the cost is **bounded at +0.47% perplexity at 95% confidence** for +5.4%
throughput. Caveats: one corpus (this repo's docs), and perplexity is a weak
proxy — it will not catch degradation in long-form reasoning at a 99.5%
substitution rate. Keep off by default until run against a reasoning
benchmark.

Measuring this needed `-b 4 -ub 4`: co-activation is only recorded when
`n_ids <= 32`, so any larger batch leaves the substitution table empty and
both arms return byte-identical PPL. Cost two wasted runs to rediscover.
It also means **substitution is inherently decode-only**.

### Instrumentation added

- **Rank-resolved hit/miss** (`rank_hits`/`rank_misses` on `/experts`): misses
  concentrate in the router's low-confidence tail — rank 0 misses 20.0% of the
  time, rank 7 misses 44.0%, monotonic, with only **8.0% of all misses landing
  on the router's top pick**. That is what makes substitution cheap.
- **Signal 3** (`MOE_TRACE_SCORES=1` in `llama-moe-trace`): the router's score
  per pick, not just its ranking. Rank 0 carries 26.48% of gate mass, rank 7
  carries 6.82% (3.88x spread); ranks 4-7 together carry 32.17%.
- **Resident bitmask** (`GGML_CUDA_MOE_CACHE_MASK_AUDIT=1` proves it): one bit
  per expert per tensor, maintained only through
  `moe_cache_map_insert`/`erase`. Validated live with the pool saturated and
  13,639 evictions of churn: 8 audits, 0 disagreements.

### Hit rate is workload-dependent — quote it with its traffic

A single repeated prompt measures **100.00%** (0 misses in 601,440 lookups).
Four topic-diverse prompts measure **68.7%**. Same cache, same size. Topic
diversity produces misses, not cache pressure — so any hit-rate figure quoted
without its workload's topic spread is meaningless.

## Anti-co-firing and the incremental atlas — both tested, both fail (2026-08-25)

### Anti-co-firing exists, and is stronger than attraction

Measured over 70,800 frequent same-layer pairs (expected co-fires >= 5), on
909,480 pooled decisions:

```
NEVER co-fire despite that expectation : 6.68%
PMI mean -1.123   median -0.784
share negative (anti-associated)       : 71.5%
share strongly negative (PMI < -1)     : 45.0%
share strongly positive (PMI > +1)     :  8.2%
```

Strong repulsion (45%) is 5x more prevalent than strong attraction (8.2%).
Every mechanism built so far uses only the attraction half.

*Correction*: a first pass computed expected co-fires as `pa*pb*N*(K-1)`. The
extra `(K-1)` inflates the expectation 7x and biases every PMI down by
log(7)=1.95, which produced a spurious "98.6% anti-associated". Correct
expectation under independence is `N*pa*pb`.

### As an eviction signal it is the worst arm yet

1421 slots, window 64, all scores accumulated online, 151,580 decisions:

| fill | policy | retained | hit | fills/1k | pred tok/s |
|---|---|---|---|---|---|
| 1 | **lru** | **89.69%** | 77.45% | 92.4 | **63.20** |
| 1 | random | 86.37% | 71.09% | 106.9 | 51.43 |
| 1 | coact | 82.70% | 72.98% | 88.2 | 56.95 |
| 1 | anti | 73.76% | 59.11% | 108.2 | 32.91 |
| 1 | anti+lru | 74.71% | 60.26% | 106.9 | 34.85 |

Worse than random, and it generates MORE fills. The reason is structural and
general: anti-co-firing measures **simultaneity** ("not in the same decision"),
while eviction needs **temporal proximity** ("not needed soon"). An expert can
be strongly anti-associated with the current picks and fire on the very next
token.

**That is now six signals beaten by plain LRU** — atlas, router score,
co-activation, frequency, admission demand, anti-co-firing. Every
co-activation-derived signal describes within-decision relationships in one
polarity or the other; eviction is an across-time question. The single place
such a signal has ever won is SUBSTITUTION, a placement decision, where it
delivered +5.4%.

This supports separating the two roles explicitly: co-firing structure is a
hard rule governing PLACEMENT (who stands in for whom, who sits near whom),
and recency/heat is the soft rule governing RETENTION probability. Applying a
placement signal to a retention decision has now failed six times.

### Incremental atlas with AdaGrad, and the SILO WARNING (2026-08-25)

Per-node AdaGrad replaces uniform averaging: each node's step scales by its own
accumulated gradient history, so frequent experts damp down as their history
grows while rare ones keep moving — "keep the history of affinity rather than
overwrite it", applied per node. Base learning rate still constant.

Dynamics are now correct: 89.9 deg -> 7.7 deg over 14 epochs, still decreasing,
no oscillation, no collapse (pairwise cosine sd 0.177), and no edge-sticking
(mean radius 0.398 in the 2D projection, 5.1% in the outer 20%). Persistence
verified: resuming adds to the saved map (152.8M -> 178.3M steps) and settling
continues smoothly rather than restarting.

Held-out link prediction is still **AUC 0.4823 — chance** (0.4991 under uniform
averaging). AdaGrad fixed the dynamics, not the prediction.

**But the isolated test does not predict integrated behaviour.** Loaded onto
8099 against the probe atlas, with prompts absent from all 12 trace topics and
every earlier A/B, 2 reps each:

| atlas | tok/s | hit | fills |
|---|---|---|---|
| probe (9 categories) | 58.45 | 71.20% | 21,714 |
| incremental (AUC 0.48) | 58.43 | 71.28% | 20,875 |

**Identical.** A map scoring at chance performs exactly as well in the running
system as one scoring 0.7116. Cause: atlas warming is rate-limited to once per
64 `plan()` calls and admission-gated, contributing ~2% of fills, so the system
barely consumes the atlas — consistent with the warming on/off result
(+0.27 tok/s, within noise).

Consequence for how this is evaluated: **the probe atlas's link-prediction
advantage is real as a statement about the maps, but it never reaches the
system.** No atlas comparison run through 8099 can currently distinguish a good
map from a bad one, because the consuming mechanism has almost no leverage.
Any future claim that one atlas beats another must say which of the two it
means. The incremental map did produce 3.9% fewer fills, consistently across
both reps, which is the direction fill cost rewards.

Untested suspicion: training negatives are drawn uniformly from same-layer
experts while the graph is 22.9% dense, so ~a quarter of the "negatives" are
real co-firing pairs. That would poison SGNS and could explain chance-level AUC
despite clean convergence.

### Incremental atlas — first attempt, settles but learns nothing

`scripts/moe-incremental-atlas.py`: SGNS over co-firing pairs, constant
learning rate by design (annealing would manufacture the appearance of
settling), state persisted to .npz and resumed on the next run so it
accumulates like `session.history`.

Two failure modes found and fixed in order: (1) unnormalised vectors diverge
to 1e70 within 4 epochs; (2) with normalisation they OSCILLATE, flipping to
the antipode every pass (median move 2.0 on a unit sphere), because `np.add.at`
applies every update a node collected in a batch at once, so a frequently
selected expert takes a step proportional to its frequency. Averaging each
node's update by its occurrence count fixes both.

It then settles cleanly — 0.25 deg/epoch, stable from epoch 2, and does NOT
collapse (pairwise cosine mean -0.0013, sd 0.1767, range -0.69..+0.65, so
negative sampling supplies real repulsion). But held-out link prediction gives
**AUC 0.4991 — pure chance**. It settled into something with no predictive
content: the per-node averaging that cured the oscillation shrank each step
enough to underfit. That is an implementation failure, not a refutation of the
incremental idea, and it remains the one untested variant.

Reference points on the same test: probe atlas 0.7116, spectral discovered
0.5761, popularity control 0.5437.

*Note*: with 5x the data LRU's absolute retained mass rose from 75.72% to
89.69%. Rankings held throughout, but earlier absolute simulation figures are
pessimistic.

## CRITICAL: moe-cache corrupts output — PRE-EXISTING, still open (2026-08-25)

**Deterministic reproducer**: `--moe-cache auto`, any context size, prompt
"Work through this step by step: a train leaves A at 60mph, another leaves B
200 miles away at 40mph one hour later. Explain your full reasoning." at
max_tokens=600. Output is `////` from character 0.

**Once triggered the server is permanently poisoned** — a subsequent "Name
three primary colours" also returns `////`. Only a restart clears it.

**Pre-existing, not introduced this session.** Reproduces identically on
`5de4ff917` (fingerprint b10601), the build from before this session's work.

What has been ruled out:
- **Not atlas warming.** An earlier isolation blamed warming, but that
  experiment varied max_tokens between arms (500 clean / 600 degenerate) and
  was worthless. With length held fixed, WARM=0 and WARM=1 both corrupt.
- **Not generation length alone** — 800 tokens is clean on a server that has
  not yet seen the trigger prompt.
- **Not any optional subsystem**: HOST_MB=0, HOSTREG_MB=0, COLD_AFTER_S=0,
  GGML_SCHED_PREFETCH_EXPERTS=0, ATLAS_WARM_ASYNC=0/1 all still corrupt.
- **Not offload depth** — corrupts at 27, 28 and 30 CPU MoE layers alike.
- **Not eviction** — `evictions: 0` at the moment of corruption.
- **Not corrupted slot contents.** A sampled verifier
  (`GGML_CUDA_MOE_CACHE_VERIFY_SLOTS=N`) comparing each hit's slot against the
  host weights it mirrors, across head/middle/tail windows, reports **zero
  mismatches** through a full corrupting generation. The cache holds the RIGHT
  DATA. `fill_failures` is also 0.

So it is an **indexing/routing bug, not a data bug**: correct expert weights
reaching the wrong row, or the wrong slot being selected for a row. The row
bookkeeping in `ggml_compute_forward_mul_mat_id` is the place to look; its six
parallel arrays are correctly sized (all MOE_CACHE_MAX_TOPK=4096, matching the
`n_ids * ids->ne[1] <= 4096` gate), and the collect-failure fallback's apparent
missing `matrix_row_counts` increment is NOT a bug — it recomputes each row
immediately after writing the mapping, so reusing the scratch index is safe.

**INVESTIGATION STATUS (2026-08-25, unresolved).** Three separate "found it"
claims were made and all three were WRONG, each because more than one variable
moved between arms:

1. *Atlas warming* - arms differed in max_tokens (500 clean / 600 degenerate).
2. *moe-cache itself* - `GGML_CUDA_MOE_CACHE=0` looked clean, but that arm also
   had `-c 65536`; with `-c 2048 -ncmoe 40` and no cache it degenerates too.
3. *VRAM exhaustion / OOM* - `GGML_CUDA_MOE_CACHE_RESERVE_MB=2560` leaves
   2544 MiB free and it still degenerates 3/3. Not memory pressure.

Contradictory observations that no single variable explains:

| config | result |
|---|---|
| `cache=0, -c 65536` (train prompt included) | clean |
| `cache=0, -c 2048 -ncmoe 40` | DEGENERATE |
| `cache=auto, -c 65536` | DEGENERATE |
| `cache=auto, -c 4096`, 800-token generation | clean |
| `cache=auto, -c 4096`, then train prompt | DEGENERATE |

What IS established:
- Deterministic trigger prompt (the train word-problem), and once triggered the
  server stays poisoned for unrelated later requests until restart.
- Pre-existing: reproduces on `5de4ff917` from before this session.
- Slot CONTENTS are correct (`GGML_CUDA_MOE_CACHE_VERIFY_SLOTS`, 0 mismatches).
- Slot->expert MAPPING is correct (`GGML_CUDA_MOE_CACHE_VERIFY_ROWS`, 0
  mismatches).
- Row accounting BALANCES (`MOE_CACHE_ROW_AUDIT`, 0 mismatches) - no dropped rows.
- Persists with `GGML_CUDA_MOE_CACHE_FAIL=dispatch`, i.e. with all expert math
  on the CPU.
- No `cudaMalloc failed` / OOM lines in any log.
- The cache expands to fill VRAM down to the ~128 MiB reserve floor regardless
  of `-ngl`, which is worth fixing on its own merits but is NOT this bug.

**CONTROLLED MATRIX RUN (36 fresh servers, one per sample so a poisoned server
never contaminates the next cell, max_tokens/temperature/-ngl pinned):**

| cache | ctx | prompt | degenerate |
|---|---|---|---|
| auto | 2048 / 4096 / 65536 | train | **3/3 each** |
| auto | any | halting | 0/9 |
| off | any | either | 0/18 |

`by cache: auto=9/18, off=0/18` · `by prompt: train=9/18, halting=0/18` ·
`by context: 3/12 at every size - NO effect`

So it IS moe-cache, interacting with a specific prompt, deterministic in both
directions. The earlier retraction of that diagnosis was wrong: the arm that
seemed to exonerate the cache (`cache=0 -c 2048 -ncmoe 40`) was broken by a
hand-set `-ncmoe 40` on a model whose expert map has only 30 rows - a
misconfiguration introduced by the investigator, not the cache.

**Greedy (temperature=0) divergence: character 0.** Cache-on and cache-off
diverge on the very first generated token, so the damage is already present
before any decode step.

**Sharp bisect on batch size:**

```
GGML_CUDA_MOE_CACHE_MAX_BATCH=1  -> clean
                              2  -> DEGENERATE
                              4/8/16/64 -> DEGENERATE
```

`max_batch=1` admits only single-token nodes. Everything >= 2 corrupts. This is
the tightest handle on the bug so far.

**Hypothesis tested and REJECTED**: that the multi-activation path is at fault.
`ggml_cuda_moe_cache_mmv`'s parameter mapping carries the comment "ne12 = ne13
= 1 (single token)", suggesting it is only correct when all hits share one
activation row. Refusing nodes with `!shared_activation` did NOT fix the
corruption, so `shared_activation` is true in the failing case and
multi-activation is not the mechanism. Guard reverted rather than left in as a
false fix.

**Hypothesis SIX: CUDA-graph capture collision - REJECTED (2026-08-25).**
The cache's fill worker issues cudaMemcpyAsync on its own stream while a graph
capture may be live on the main stream, and plan.md already records this codebase
hitting exactly that ("operation not permitted on an event last recorded in a
capturing stream ... Gemma's capture boundaries evidently avoid this;
qwen35moe's do not"). Ornith is qwen35moe-class. It also uniquely explained the
shape of the evidence: a timing collision corrupts nothing that a state check
would see, which is why every verifier comes back clean. Measured:

```
graphs ON  (default), cache on : DEGEN DEGEN DEGEN
graphs OFF, cache on           : DEGEN DEGEN DEGEN
```

```
graphs OFF, cache off (CONTROL) : DEGEN DEGEN DEGEN
```

**THE CONTROL ALSO DEGENERATED, so this test proves nothing** - not the
hypothesis, not its rejection. Cache-off was clean 18/18 in the controlled
matrix earlier the same day and fails 3/3 here, so the machine or build state
has drifted between the two and the graph arms cannot be interpreted.

The hypothesis is therefore UNTESTED, not rejected. Re-running it requires
first re-establishing that cache-off is clean on the current build - i.e.
re-validating the control before trusting any arm.

That drift is itself the most important open fact: **the reproducer's
dependence on cache state is no longer stable**, which undermines the
controlled matrix's central finding (auto 9/18 vs off 0/18). Everything
downstream of that matrix needs re-confirmation.

**Five hypotheses have been tested and rejected; the sixth is untested (broken control)** (atlas warming, moe-cache
as a whole, OOM/VRAM, scratch-capacity overrun, multi-activation mapping). The
bug is real, deterministic, and still unexplained.

**Current stopgap**: `GGML_CUDA_MOE_CACHE_MAX_BATCH=1` keeps the cache ON and
active (hit rate 0.70, 5650 slots) and is clean 4/4 including the reproducer,
at 44 t/s - the same as cache-off, so the multi-token path was carrying the
throughput benefit. Better than disabling the cache outright.

**Next step**: instrument inside the `max_batch >= 2` path specifically -
compare a multi-token node's GPU result against a CPU recomputation of the same
rows, which turns "output is wrong" into "row R of node N differs", the fact
that will finally name it.

**Superseded note - the original next step was a controlled matrix** - cache x context x prompt, one
variable at a time, repeated - not another hypothesis. Related prior art: a
dangling commit (`c45ae5b309`, 2026-08-13) documents a real OOM-vs-throughput
interaction with `cudaMalloc failed: out of memory` on the KV cache, and the
user recalls an uncommitted "loud failure" guard for this class of problem
that has not been located (dangling candidates are either already merged - the
64->4096 ceiling - or superseded).

**No configuration is currently known-safe.** `GGML_CUDA_MOE_CACHE=0`, which costs ~21%
throughput (44 t/s vs ~56).

### Consequence for this session's server-measured numbers

Degenerate output generates FASTER than clean (68-69 t/s vs 55-56), so any A/B
arm that silently corrupted would read as a throughput win. Every server
measurement taken before this was found lacked a degeneration check.

Re-measured with every sample degeneration-checked (0/4 degenerate in all
arms), 2 reps:

| arm | tok/s |
|---|---|
| substitution OFF | 57.81, 57.67 |
| substitution ON | 60.55, 60.34 |

**+4.7%, confirmed real** (previously reported as +5.4% unguarded). The
perplexity bound is unaffected — those runs used `llama-perplexity`, where
corruption would have produced absurd PPL rather than 6.31.

Still to re-measure with the guard: the admission A/B (-7.6% for removing the
gate), the atlas warming A/B (+0.27 tok/s), and the fill-cost model calibrated
from those arms.

## Fold thesis — VALIDATED: real 3-way structure beyond pairs (2026-08-25)

Groups of experts fire together as sets, not merely as correlated pairs. Tested
on 113,685 pooled decisions, held-out split, triples with count >= 5, scored
against the Kirkwood superposition (what the three pairwise marginals predict):

```
mean log-lift  +1.050    (2.9x more often than pairwise predicts)
median         +1.017
share above pairwise prediction:  96.7%
strongly above (>+0.5):           83.1%
```

**96.7% of triples exceed what pairwise structure explains.** Every mechanism
built so far - co-activation substitution, the atlas, all the eviction arms -
reads only the pairwise shadow of this.

Order limits: recurrence falls with order (2-way 65.3% of train groups recur
held-out, 3-way 30.4%, 4-way 14.4%) and seen-once rises (30.4% / 54.7% /
70.1%). Triples are the usable order; 4-way is mostly noise at this sample size.

Note the selection process constrains exactly 8 of 256 per decision, which
induces *competition* between experts and would bias higher-order correlation
NEGATIVE - so a +1.05 positive lift is not a selection artifact.

**Actionable**: substitution currently picks a stand-in by pairwise
co-activation. Selecting by triple co-occurrence with the other experts in the
same decision is the direct test of whether the higher-order structure converts
into a better mechanism, and it reuses data already collected.

## Commit audit — behaviour changes shipped in the days before the bug (2026-08-25)

The corruption reproduces on `5de4ff917` (08-24 09:33), so it predates this
session. Auditing what landed on the MoE execution path in the days before,
most commits are instrumentation, measurement or reverts. The ones that changed
DEFAULT behaviour for every user:

- **`107389751` (08-23 20:04) "default SPEC_EVICT_MODE to agree, not off"** -
  ships SPECULATIVE EVICTION on by default. Its own message argues "Safe even
  though it's now the default". Evicting a slot on a *prediction* is precisely
  the shape of a bug that corrupts output while every state check passes, and
  it landed before the earliest commit the corruption is confirmed on.
  **Leading suspect.** Disable with
  `GGML_CUDA_MOE_CACHE_SPEC_EVICT_MODE=off`.
- `ef2c30b43` (08-23 19:56) - fixes agree-mode at depth 3+, same subsystem.
- `fce3e9fa5` (08-23 19:30) - introduces cross-depth-agreement spec eviction.
- `30d0f8447` (08-23 20:25) - mincore-verified NVMe cost tier (read-only, low risk).
- `b321b6259` (08-24 04:27) - prefill copy-failure guard (a FIX for a prior
  silent-corruption bug in the same area).

Test queued: control (cache off) FIRST - the previous graph test was void
because its control degenerated - then spec_evict default vs off.

## DESIGN ARGUMENT — read this before proposing new cache work (2026-08-25)

The reasoning behind the current direction, recorded because several sessions
have re-derived (and mis-derived) it. The measurements backing each claim are
in the sections below.

### 1. Hit rate is the wrong metric for this design

Hit rate is what a FETCH-based cache lives or dies by: a miss means a stall, so
minimising misses is the whole game. Under SUBSTITUTION a miss costs no time at
all - it costs a slightly different expert. The right measure is **retained gate
mass**: how much of the weight the router intended actually got executed.

The two diverge violently at low residency. From 18.5% down to 1.2%:

- hit rate collapses 96.9% -> 38.7% (looks fatal)
- retained gate mass degrades 99.1% -> 80.4% (entirely workable)

An earlier assessment in this project extrapolated from hit rate and concluded
the GLM-scale vision was out of reach. That conclusion was WRONG, and wrong
specifically because it used the fetch-design metric on a substitution design.

Hit rate is also *purchasable*, which disqualifies it as an objective on its
own: removing the admission gate buys +10.79pp of hit rate and costs 7.6% of
throughput. Any optimisation target has to price fills.

### 2. Why substitutes stay good as the cache shrinks (structural, not luck)

Stand-in quality is FLAT at ~230% of the replaced pick's router score from 5%
residency down to 0.6%. The substitute is consistently rated HIGHER than what
it replaced. That is not an artifact:

- Misses concentrate on the router's weak picks - rank 7 misses 44% of the time
  against rank 0's 20%, monotonic across all 8 ranks.
- So the expert that goes missing is usually one the router rated poorly.
- Whatever IS resident is, by construction, the frequently-selected hot core -
  experts the router rates highly.
- Shrinking the cache concentrates it further onto that core, which keeps the
  relationship intact rather than eroding it.

This is why the design should be expected to scale to models where residency is
1% rather than 18%.

### 3. The binding constraint at scale is COVERAGE, not accuracy

The number that actually degrades with residency is the **no-candidate tail** -
picks with neither a hit nor any resident stand-in, which fall back to CPU:

```
18.5% residency ->  0.68%
 5.0%           ->  2.56%
 1.2%           -> 15.62%
 0.6%           -> 22.86%
```

At GLM-scale residency roughly one pick in six has nothing usable resident.
That, not stand-in quality, is what would set throughput. It is a different
problem from everything worked on so far: it asks "is SOMETHING usable always
reachable?" rather than "is the RIGHT expert present?".

### 4. Consequence — an eviction objective nobody has tried

Six signals have been tested as eviction policies and all lost to plain LRU
(atlas, router score, co-activation, frequency, admission demand,
anti-co-firing). But every one of them was scored on **hit rate or retained
mass** - i.e. on getting the right expert resident.

Nobody has asked a policy to maximise **coverage**: keep at least one viable
stand-in reachable for every likely pick, accepting a worse hit rate to do it.
That is a group property rather than a per-expert one, so it is also the most
plausible place for the validated 3-way fold structure (96.7% of triples exceed
pairwise prediction) to finally earn its keep - a policy that keeps one member
of each active group resident, rather than the k hottest individuals.

### 5. What is settled and what is open

**Settled - stop re-testing these:**
- Prefetching cannot pay. Capped at +0.30pp even with a perfect topic oracle,
  because every topic touches ~6,000 of 7,680 experts.
- Placement/"which experts to keep" is effectively solved. LRU wins, and Belady
  bounds ANY policy's remaining headroom at ~17pp on hit rate.
- The atlas is neutral-to-marginal as a policy input and cannot currently be
  evaluated through the server at all (warming touches ~2% of fills).
- Filling harder loses. Removing admission control costs 7.6% throughput.

**Open, in priority order:**
1. The corruption bug - blocks every live measurement.
2. Perplexity at LOW residency. The +0.47% bound is an 18.5% number;
   substitution actually bites at 1.2%. This does NOT need the bug fixed -
   `llama-perplexity` with a small `GGML_CUDA_MOE_CACHE_BUDGET_MB` measures it.
   It is the single result that could still sink the design.
3. Coverage-oriented eviction (section 4).
4. Scale validation: 4,713 units here, low cells only 28-117 slots. Whether a
   750B model's hot core stays this compact is untested.

## Embedding implementation fixed — and why probes beat co-occurrence (2026-08-25)

The learned embedding scored AUC 0.4900 on held-out link prediction while a
trivial common-neighbours count scored 0.6530. That was an implementation
failure, now corrected by dropping SGNS and representing each expert directly
by its row of co-firing counts:

| representation | AUC |
|---|---|
| co-occurrence rows, cosine (**the fix**) | **0.6345** |
| + truncated SVD to 16 / 32 / 64 dims | 0.6072 / 0.6126 / 0.6212 |
| + PPMI weighting | 0.5996 |
| common neighbours (trivial baseline) | 0.6530 |
| previous SGNS embedding | 0.4900 |
| **probe atlas (9 categories)** | **0.7116** |

Two findings beyond the fix:

**Dimensionality reduction actively hurts.** SVD to 64 dims loses ground and
PPMI - the textbook weighting for co-occurrence - is worse still. The signal is
in the full high-dimensional pattern; there is no low-dimensional manifold to
recover. This is why the spectral discovered atlas was never going to work.

**Yet 9 probe dimensions beat 256 co-occurrence dimensions.** So the problem is
not dimensionality - it is what the dimensions measure. Probes measure a
FUNCTIONAL property (what causes this expert to activate); co-occurrence
measures a CORRELATIONAL one (what happens to fire alongside it). For
predicting FUTURE co-firing the functional property generalises and the
correlational one memorises.

That now explains every discovered-atlas result in this document: spectral
(0.5761), incremental SGNS (0.4823-0.4900), and raw co-occurrence rows (0.6345)
have all lost to the probe atlas (0.7116). Four independent methods, one
verdict - and the verdict is about the SIGNAL, not the method. Keeping the
hand-written probes is the better-supported position.

## Low-residency perplexity — partial result (2026-08-25)

Substitution's quality cost measured with `llama-perplexity`, 60 chunks per
arm, paired per-chunk against a substitution-OFF baseline, residency varied via
`GGML_CUDA_MOE_CACHE_RESERVE_MB` (`BUDGET_MB` is ignored - verified: identical
11,797 MiB peak at 3413 and 256):

| arm | delta PPL | 95% CI | significant | worse chunks |
|---|---|---|---|---|
| full residency | -0.217% | [-0.597%, +0.165%] | no | 28/60 |
| RESERVE 1500 | -0.035% | [-0.462%, +0.395%] | no | 31/60 |
| **RESERVE 2500** | **+0.998%** | **[+0.148%, +1.855%]** | **YES** | 36/60 |
| RESERVE 3300 | -0.069% | [-0.433%, +0.296%] | no | 29/60 |

**At full residency substitution is free** - CI spans zero, 28/60 chunks worse,
a coin flip. Independently confirms the earlier 120-chunk result (+0.181%, CI
[-0.106%, +0.468%]) on a different corpus slice.

**At one reduced-residency point it costs ~1%** - the first statistically
significant quality cost measured for substitution anywhere in this project,
and still small.

**The curve is NOT established.** The ladder was not clean: RESERVE 3300 shows
ZERO moe-cache log lines, i.e. the cache never allocated at all, so there were
no substitutions and its null result is meaningless as a low-residency point.
That leaves two live rungs and no way to tell their residency apart, because
`llama-perplexity` exposes no `slots_used`.

Methodological note: RESERVE_MB was used as a residency proxy without verifying
it produced a graded ladder. It does not - it goes full -> reduced -> off, with
no readback. **To finish this measurement the cache needs to report residency
from a non-server binary** (a stderr summary at exit would do), or the server
route, which is blocked by the corruption bug.

## The six eviction signals RETESTED — two of the verdicts were wrong (2026-08-25)

The six signals were each tested as a WHOLE policy at 18.5% residency. The
coverage result proved both choices can hide a real signal (redundancy: -18pp
as a whole policy, +4.2pp as a tie-break, and only at low residency). Retested
in the tie-break shape (LRU narrows a 32-sample to its M=8 stalest, the signal
breaks that tie) at 1.2% residency, scored on retained gate mass:

| signal | hit | no candidate | retained | vs pure LRU |
|---|---|---|---|---|
| (pure LRU) | 39.70% | 17.69% | 78.30% | - |
| **coverage** | 36.89% | 14.48% | 82.48% | **+4.18pp** |
| **atlas** | 39.39% | 16.61% | 79.46% | **+1.17pp** |
| **router score** | 36.20% | 17.78% | 79.01% | **+0.71pp** |
| co-activation | 39.79% | 19.40% | 76.24% | -2.06pp |
| frequency | 38.25% | 20.04% | 75.58% | -2.72pp |

**The atlas and router score verdicts were wrong.** Both were reported as
failures - the atlas as losing even to random - and both are positive in the
correct shape and regime. Co-activation and frequency remain negative, so those
two verdicts stand.

The methodological lesson, now demonstrated three times (anti-co-firing,
coverage, and these two): **a signal's value depends on the shape it is used in
and the regime it is measured at.** Testing a signal as a whole policy at high
residency is close to a worst case for it - recency dominates when the cache is
large, and any signal given full control fights it. Future signal tests must
report shape and residency alongside the result.

Untested and now the obvious next step: **combinations** - coverage as primary
tie-break with atlas or router score as a secondary, since the three positive
signals may be capturing different structure.

## Combined eviction signals — all three, softly (2026-08-25)

Once retested in the tie-break shape, three signals are positive (coverage
+4.18pp, atlas +1.20pp, router score +0.70pp). Combined at 1.2% residency, LRU
narrowing a 32-sample to its M=8 stalest:

| combination | hit | no candidate | retained | vs LRU |
|---|---|---|---|---|
| coverage | 36.89% | 14.48% | 82.48% | +4.18pp |
| coverage+atlas | 38.54% | 13.57% | 83.31% | +5.01pp |
| coverage+score | 36.80% | 14.42% | 83.17% | +4.87pp |
| atlas+score | 38.08% | 14.78% | 82.64% | +4.34pp |
| **all three (rank-sum)** | 37.56% | **13.35%** | **84.23%** | **+5.93pp** |

Every pair beats its members and the triple beats every pair, so the three are
capturing different structure rather than restating one signal.

### A hierarchy was tried and LOST

Reasoning said coverage should be a hard VETO (never evict the sole resident
representative of a neighbourhood), then atlas for relevance, then score for
value. Measured:

| policy | retained | vs LRU |
|---|---|---|
| rank-sum of all three | 84.23% | +5.93pp |
| coverage veto -> LRU | 78.31% | +0.02pp |
| coverage veto -> atlas -> score | 78.82% | +0.52pp |
| coverage veto -> score -> atlas | 79.30% | +1.01pp |

**The veto is worth nothing.** It only excludes redundancy == 0, and at 1.2%
residency almost nothing has zero redundancy, so it rarely fires and the policy
degenerates to LRU.

**Redundancy's value is in its GRADIENT, not its extreme.** "How redundant"
carries signal across its whole range; "is it the very last one" is a threshold
that almost never trips. Promoting a graded signal to a hard rule destroys the
part doing the work - the same failure mode as testing signals as whole
policies, one level down.

Untested refinement: WEIGHTED rank-sum, with coverage weighted above atlas and
score to reflect their individual strengths, which keeps every gradient while
respecting that they are not equally informative.

## Gemma UD-quant substitution test — INVALID, baseline broken (2026-08-25)

Question: does substitution cost more on an Unsloth dynamic quant, whose bit
budget is spent on attention and keeps MoE experts thinnest? That matters for
the 2-bit GLM plan, where quantisation and substitution would degrade the same
component.

Ran gemma-4-26B-A4B-it-UD-Q4_K_M, same corpus and settings as Ornith:

```
off  PPL = 141.10 +/- 7.06
on   PPL = 139.65 +/- 6.98
paired -1.026%  CI [-4.434%, +2.502%]  not significant, 25/60 chunks worse
```

**The baseline is broken.** PPL 141 against Ornith's 4.34-7.04 on the identical
corpus is not a quantisation difference - the model is producing near-garbage
regardless of substitution, so on/off against it means nothing. The +-4.4% CI
also swamps the ~1% effects being hunted.

No warnings in the log and the config loaded normally (n_ctx=512, batch 4).
Most likely cause: gemma-4 uses interleaved local/global attention with a
sliding window, and n_ctx=512 is too small for it on raw technical prose.
Ornith tolerates that context size; gemma evidently does not.

**To answer the question properly** the gemma arm needs a config where its
baseline PPL is sane (larger n_ctx, and possibly a corpus closer to its
training distribution), verified BEFORE comparing substitution on/off. Until
then the UD-quant interaction with substitution is unmeasured, and the 2-bit
GLM plan rests on an untested assumption.

## How to combine eviction signals — four forms tried (2026-08-25)

Criticism that prompted this: sum vs product was being chosen by result rather
than by what the quantities mean. Correct, and worth resolving properly.

The eviction decision should minimise expected future loss:

```
Loss(evict X) = P(X requested soon) x value(X) x P(no good substitute)
```

which is a product of probabilities, i.e. a SUM OF LOG-PROBABILITIES. So
addition and multiplication are the same model in different spaces; what was
genuinely arbitrary was combining raw RANKS, which are neither. Also noted:
recency is itself an estimator of P(requested soon), so treating LRU as a
separate narrowing stage rather than a factor was a modelling error.

Results, all at 1.2% residency:

| form | vs LRU |
|---|---|
| hierarchy (coverage veto -> relevance -> value) | +0.02 to +1.01pp |
| product of raw quantities | +4.72 to +4.84pp |
| calibrated log-additive expected loss | **-9.51pp** |
| **weighted rank-sum (heuristic)** | **+6.18pp** |

**Why the principled forms lost - both times it was the inputs, not the form.**
The product let whichever factor had the widest dynamic range dominate (score
EMA spans orders of magnitude, cosine is bounded in [-1,1], redundancy is a
small integer), silently discarding two of three signals. The log-additive
model was calibrated with a 2000-decision reuse window, which turned out to be
so long that P(reused) sits between 0.86 and 0.99 - almost everything returns,
so the term carries nearly no information and the value term takes over.
Recency's predictive content is not WHETHER an expert returns but HOW SOON.

**Why rank-sum keeps winning**: ranks are invariant to miscalibration. Every
principled form needs its inputs correctly scaled or calibrated, and that
requirement was violated both times. Rank aggregation needs only each signal's
ORDERING to be informative - a much weaker condition, and the one these signals
actually meet.

That is not a claim that rank-sum is theoretically superior. It is the right
tool for three signals of unknown calibration, which is the situation here. A
log-additive model remains the better target IF the factors are calibrated
properly - and the failure above shows the calibration, especially the reuse
window, is where the work is.

## Signal weighting and orthogonality — the eviction recipe (2026-08-25)

A flat 3-way rank-sum lets coverage be outvoted 2:1 by atlas and score, which
both measure an INDIVIDUAL expert's worth, while coverage is the only one
measuring STRUCTURE. Weighting coverage equal to their sum is better:

| weighting (cov, atl, scr) | no candidate | retained | vs LRU |
|---|---|---|---|
| 1, 1, 1 (flat sum) | 13.35% | 84.26% | +5.96pp |
| **1, 0.5, 0.5** (coverage = atlas+score) | **13.06%** | **84.47%** | **+6.18pp** |
| 3, 1, 1 | 13.33% | 84.02% | +5.73pp |
| 4, 1, 1 | 13.73% | 83.51% | +5.21pp |
| 1, 0.5, 0 (drop score) | 13.50% | 83.53% | +5.23pp |
| 1, 0, 0.5 (drop atlas) | 13.47% | 84.07% | +5.77pp |

Over-weighting coverage past parity HURTS, so this is not "coverage matters
most" - the balance between structure and individual worth is what matters.

### The three signals are near-ORTHOGONAL

Spearman rank correlation measured on the actual candidate pools they rank
(n=64,890 pools):

```
cov~atl   -0.0611      cov~scr   +0.1139      atl~scr   -0.0210
```

Coverage and atlas are essentially independent, and no pair exceeds |0.11|.
They measure three different things: **structure** (will the neighbourhood
survive an eviction), **direction** (where is the request heading), and
**realised value** (what has this expert been worth lately).

That orthogonality explains the combination results - every pair beats its
members and the triple beats every pair - and it is why summing recovers more
than any hierarchy. A correction to an earlier inference in this session:
the ablation gap (dropping atlas costs -0.41pp, dropping score -0.95pp) was
read as atlas being redundant with coverage. It is not. At that scale those are
interaction effects, not evidence of overlap.

**Current best recipe**: LRU narrows a 32-candidate sample to its M=8 stalest,
then evict by rank-sum with weights coverage 1.0, atlas 0.5, score 0.5.
+6.18pp retained gate mass at 1.2% residency, no-candidate 17.69% -> 13.06%.
Simulation only; needs a second trace set and more seeds before it justifies
C++ work.

## Coverage-oriented eviction — FIRST policy to beat LRU (2026-08-25)

Predicted by DESIGN ARGUMENT section 4: six signals had failed as eviction
policies, but every one was scored on getting the RIGHT expert resident. This
asks instead for COVERAGE - keep at least one viable stand-in reachable.

Signal: **redundancy** - how many of an expert's co-firing partners are still
resident. Evict the most redundant; protect the sole resident representative of
a neighbourhood.

**As a whole policy it is catastrophic**: no-candidate rises 12-18pp and hit
rate collapses (24% vs LRU's 71% at 5% residency). Redundancy correlates with
being well-connected, which is the hot core, so "evict the most redundant"
evicts precisely the experts that both get hit and serve as stand-ins.

**As a tie-break inside LRU it wins.** LRU narrows a 32-candidate sample to its
M stalest; redundancy picks the most redundant of those. M=1 is pure LRU:

| resident | M | hit | no candidate | retained mass |
|---|---|---|---|---|
| 2.5% | 1 | 54.51% | 8.05% | 89.33% |
| 2.5% | 8 | 51.29% | **6.17%** | **92.01%** (+2.68pp) |
| 1.2% | 1 | 39.70% | 17.69% | 78.29% |
| 1.2% | 8 | 36.90% | **14.44%** | **82.52%** (+4.23pp) |

It LOSES hit rate (-2.8pp) and WINS retained gate mass (+4.23pp) by cutting the
no-candidate tail. That is the trade section 1 of the design argument says is
worth taking, and it is invisible to any hit-rate-scored experiment - which is
why six previous signals missed it.

Two properties that matter:
- **The gain grows as residency falls** (+2.68pp at 2.5%, +4.23pp at 1.2%) -
  the direction needed for the GLM-scale case at ~1.5% residency.
- **Monotone in M** up to 8, so the optimum has not been found; M>8 untested.

Same structural lesson as anti-co-firing: a signal that is destructive as a
whole policy can still be useful as a tie-break inside recency. The difference
is that this one survives the tie-break test and anti-co-firing did not.

Not yet implemented in the cache - this is a simulation result on
`traces/big-*.txt`, and it needs the per-expert resident-partner count
maintained cheaply on the eviction path before it could ship.

## Cache-size sweep — stand-in quality holds as residency collapses (2026-08-25)

The question behind the GLM-scale vision: extract a small working set of a huge
model into VRAM and run the rest from host RAM. Simulated with the SHIPPED
logic (LRU eviction, demand admission, rank-ordered stand-in selection), 113,685
decisions, 4,713 distinct (layer, expert) units:

| resident | slots | hit | no candidate | stand-in vs missed | retained mass |
|---|---|---|---|---|---|
| 18.5% | 871 | 96.90% | 0.68% | 277.1% | 99.11% |
| 10.0% | 471 | 87.21% | 0.79% | 252.7% | 98.93% |
| 5.0% | 235 | 70.31% | 2.56% | 232.0% | 96.33% |
| 2.5% | 117 | 53.66% | 7.62% | 225.1% | 89.55% |
| **1.2%** | 56 | 38.67% | 15.62% | **227.6%** | **80.37%** |
| 0.6% | 28 | 26.80% | 22.86% | 231.5% | 73.11% |

**Stand-in quality does not degrade.** 277% -> ~230% and then flat from 5% down
to 0.6%: even with a 0.6% working set the substitute is still rated HIGHER by
the router than the pick it replaced. Structural reason: misses concentrate on
the router's weak picks (rank 7 misses 44% against rank 0's 20%), so whatever
is resident tends to outrank what was lost, and shrinking the cache only
concentrates it further onto the hot core.

**This is why hit rate is the wrong metric for this design.** Hit rate collapses
96.9% -> 26.8%, but retained gate mass degrades gracefully 99% -> 73%. At the
1.2% residency corresponding to an "8B working set" of a ~750B model, a
fetch-based design faces a 61% miss rate and is bandwidth-dead, while
substitution executes **80% of the router's intended weight** with only 15.6%
of picks falling back to CPU.

Caveats:
- Retained gate mass is a PROXY. The +0.47% perplexity bound was measured at
  18.5% residency; nobody has measured quality at 1.2%, which is where
  substitution actually bites.
- The no-candidate tail grows fast (0.7% -> 15.6% -> 22.9%). At very low
  residency the cost shifts from PCIe bandwidth to CPU compute.
- Scale is NOT proven: 4,713 units here, and the low cells are 28-117 slots. A
  750B model is far wider and whether its hot core stays this compact is
  untested.

## Substitution stand-in selection — router score beats co-activation by 78%

The substitution mechanism picks a stand-in by pairwise co-activation, but that
selector was never validated - only substitution on/off was measured. Tested
offline on 151,584 decisions, scoring each method by the ROUTER'S OWN score for
the stand-in it chose (a good substitute is one the router also rated highly
for this token):

| method | mean stand-in score | vs missed expert | vs random |
|---|---|---|---|
| random resident | 0.01387 | 87.9% | - |
| pairwise co-activation | 0.02132 | 130.9% | **+53.7%** |
| triple co-occurrence | 0.02175 | 133.9% | +56.9% |
| **router score among resident** | **0.03792** | **238.2%** | **+173.5%** |

Three conclusions:

1. **The pairwise selector is real work**, +53.7% over random. It was doing
   something, contrary to doubt.
2. **Triples add only +2.0% over pairs** for this decision. The 3-way structure
   is genuinely there (96.7% of triples exceed pairwise prediction) but does
   not convert into better stand-in selection.
3. **Router score is the right selector** - 78% better than the current
   pairwise implementation. Co-activation is a *proxy* for "which expert suits
   this input"; the router computed that answer exactly, and selecting by
   co-activation discards it.

Stand-ins scoring ABOVE 100% of the missed expert is expected, not anomalous:
misses concentrate on the router's weak picks (rank 7 misses 44% of the time
against rank 0's 20%), so the best resident alternative is frequently an expert
the router rated higher than the one that was lost.

**Implementation**: `ffn_moe_probs` is computed before the argsort that selects
top-k, so the full score vector exists at decision time. `plan()` currently
receives only the chosen ids. Plumbing the score vector through is the change,
after which the stand-in is `argmax(probs masked by residency)` - the
constrained-routing formulation, with no prediction and no new data.

## Anti-co-firing as a probabilistic modifier — tested properly, still fails

Earlier it was tested standalone and deterministic, which was the wrong shape.
Re-tested layered on the existing logic: LRU narrows a 32-candidate sample to
its M stalest, and anti-affinity only breaks that tie. M=1 is pure LRU.

| fill | M | retained | hit | pred tok/s |
|---|---|---|---|---|
| 1 | **1** | **89.62%** | 77.50% | **63.32** |
| 1 | 2 | 89.41% | 77.36% | 63.16 |
| 1 | 4 | 88.82% | 76.70% | 62.11 |
| 1 | 8 | 87.26% | 74.30% | 57.94 |
| 1 | 32 | 73.86% | 59.28% | 33.20 |

Monotonically worse as anti-affinity gets more say. Even the mildest blend
(M=2) is neutral-to-slightly-negative. Recency dominates even when the
alternative signal only has to break a tie.

## Pending work queue (2026-08-25)

**Blocking**
1. The corruption bug above. Until it is understood, no server-measured number
   is trustworthy and the cache cannot be relied on.

**Invalidated, need re-running with the degeneration guard**
2. Admission A/B (-7.6% for removing the gate) - the empirical basis for
   "racing to fill is the wrong move".
3. Atlas warming A/B (+0.27 tok/s, "neutral").
4. Fill-cost model - calibrated from those arms, so every simulation ranking
   that used it is provisional.

**Confirmed, ready to progress**
5. Substitution: +4.7% guarded, quality bounded at +0.47% perplexity. Needs a
   reasoning-benchmark check before shipping; off by default.

**Requested, not yet done**
6. Anti-co-firing as a PROBABILISTIC modifier layered on the existing eviction
   logic (it was only ever tested standalone and deterministic, which is the
   wrong shape).
7. The fold thesis - higher-order 3-way/4-way structure. The test was killed
   mid-run to free RAM and never repeated.
8. Fix the embedding implementation: a trivial common-neighbours baseline
   scores AUC 0.653, the learned embedding 0.490. It should at least match the
   baseline.
9. Integrate a mechanism that actually CONSUMES the atlas. Warming touches ~2%
   of fills, which is why a chance-level map and a 0.71 map perform identically
   on 8099 - no integrated atlas comparison means anything until this exists.

**Open research**
10. Coverage-oriented eviction - maximise "some viable stand-in is always
    resident" rather than hit rate. Never tried; the most plausible home for the
    validated 3-way fold structure. See DESIGN ARGUMENT section 4.
11. Perplexity at low residency (1-2%), which is where substitution bites. Does
    NOT need the corruption bug fixed. The result most likely to sink the design.
12. Reuse-distance prediction - the 17pp Belady gap. Six signals have failed;
    none of them addresses the actual quantity.
11. Persist co-activation across runs like `session.history`, for cold start.
12. Track 1.5 - merged, value never independently verified.
13. The cache filling VRAM to a 128 MiB floor regardless of `-ngl`; the reserve
    should scale with the device, and the budget should yield under pressure.

## Open, no proposal yet

From the original table, marked "to be discussed" and not designed further:
embedding + output head weights, compute/activation scratch buffers, CUDA
graph capture state.

## Already closed out (adjacent work, same session)

- Readahead re-test: re-measured properly with an interleaved harness under
  today's full stack. Real result — now consistently *worse* on cold-start
  (~+21%, not the originally documented improvement), roughly neutral on
  warm throughput. Not shipped as a default change; documented honestly.
- `GGML_SCHED_PREFETCH_EXPERTS` VRAM wiring: real measured cost (820 MiB)
  reserved additively in `fit.cpp`'s margin. Commit `3417a616b`. Feature
  itself still ships off by default — this only makes it safe to opt into.
