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
5. **[not started]** Pathway/connectome visualization — add Z = layer depth
   to the Atlas view; plot real per-token trajectories through the resulting
   3D volume using step 4's co-activation data. Recurring pathways should
   show up as literal bundles of parallel lines through the volume (the
   "brain folds toward what's used together" intuition that motivated this,
   and the direct visual payoff of steps 3-4).

**Step 5 prerequisites (in progress)**: storage bug fixed (`moe_cache_edge`
replaces the original opaque-hash counters for both `co_activation` and
`co_activation_cross_layer`); export API added
(`ggml_moe_cache.get_co_activation`, top-K by count, real tensor identity);
server-side tensor->layer translation in progress (`atlas_tensor_layer` map,
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
