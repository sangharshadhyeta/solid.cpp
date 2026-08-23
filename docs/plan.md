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
4. **[step 4a done, 4b blocked]** Pairwise co-activation ("fire together")
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
5. **[done]** 3D topic-space visualization (commit `ec1354bc1`) — third axis is
   TOPIC, not layer: `llama-expert-atlas` now keeps the per-category vector
   (`cats{}`) instead of discarding it, and the Brain view projects it onto
   Fibonacci-sphere anchors (near-equidistant, no arbitrary array-order
   adjacency). Drag to orbit, scroll to zoom. Kept cheap: cold experts draw
   as 1px rects and skip the depth sort; drag redraws coalesce onto rAF.
   Originally built as a pathway view with layer depth on z, plotting
   per-token trajectories through the layer stack — that was the wrong axis
   and was corrected: layer is a single scalar the layer × expert grid
   already shows well, while topic affinity is genuinely N-dimensional and
   is where the information was actually being lost. Co-activation edges are
   still drawn, now reading as "these topic regions fire together" rather
   than as trajectories through depth.

**Known gap found while smoke-testing the export API**: cross-layer edges
(4b) don't distinguish "the next `plan()` call is a different tensor of the
*same* logical layer" (gate_up_exps -> down_exps, which share one router
decision and very often the same selected expert) from "the next call is
genuinely the following layer." A live sample showed exactly this: `layer
6 expert 6 -> layer 6 expert 6`, a same-layer artifact, not a real
cross-layer hop. Roughly half of tracked edges are probably this kind of
artifact, diluting the genuine signal. Not fixed yet - would need
`moe_cache_plan` to know it's looking at a same-logical-layer tensor pair
(e.g. via the shape/pool_index already available) before folding it into
`last_top_expert`, rather than treating every call as a layer boundary.

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

## Track 1.6 — Discovered topics from co-activation (not started)

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

## Track 2 — Unified tiered placement

Stems from "MoE expert weights — resident also tiered": today's hard
`-ncmoe` boundary means GPU-resident layers get every expert unconditionally
loaded, no cache, no eviction; CPU-offloaded layers get the opposite
treatment entirely. Proposal: replace the boundary with one uniform, global
LFRU budget spanning every MoE layer.

1. **[not started]** Loader change — every MoE layer loads CPU-resident, no
   exceptions; removes `common_moe_build_cpu_overrides`'s current
   below-cutoff-only behavior.
2. **[not started]** Budget/placement redesign — `common_maybe_raise_moe_for_ctx`
   stops deciding *which* layers are resident; it only sizes the total VRAM
   budget handed to the cache. Real re-derivation of the placement/fit math,
   not a flag flip.
3. **[not started]** Measure the real, already-flagged risk before shipping
   anything beyond a prototype: early layers are touched by nearly every
   token structurally, not because of hot/cold dynamics — under a unified
   cache they compete for eviction instead of having guaranteed residency,
   a real cold-start regression risk. Build behind a flag, A/B the cold-start
   cost specifically against the current fixed-cutoff scheme.

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

## UNRESOLVED — degenerate output incident (2026-08-24)

A live `llama-server` on Ornith-1.5-35B (`--moe-cache auto -c 65536 -ngl 99
--expert-atlas-file`, atlas v2) returned **pure `////////` for every token**
on a raw `/completion` request. Real corruption, not a rendering artifact:
`stop_type: limit`, 40 tokens, all slashes.

**Could not be reproduced.** Every variable isolated, all coherent:

| cache | ctx | atlas | result |
|---|---|---|---|
| off | 4096 | none | coherent |
| auto | 4096 | none | coherent |
| off | 65536 | none | coherent |
| auto | 65536 | none | coherent |
| auto | 65536 | v1 | coherent (×2) |
| auto | 65536 | v2 | coherent |
| auto | 65536 | v2 | coherent replaying the exact prime-then-query sequence |

The failing instance logged **no** CUDA error, fill failure, dispatch
failure, or assert. The one genuinely anomalous thing about it: **the same
process also hung on shutdown** — stopped listening, then ignored SIGTERM
for ~12 minutes and needed `SIGKILL`. Degenerate output *and* a wedged
shutdown in one process, with clean logs, reads as a single transient fault
in that instance rather than a deterministic config bug.

Not fixed, not explained, and explicitly **not** claimed fixed — it may
recur. If it does, the things worth capturing before killing the process:
`/experts` stats (fill/dispatch failure counters), whether shutdown also
hangs, and `nvidia-smi -q` for ECC/Xid errors. A GPU-level transient fault
would explain both symptoms at once and would not be a moe-cache bug at all;
that hypothesis is untested.

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
