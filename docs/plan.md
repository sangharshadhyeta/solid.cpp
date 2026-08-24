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
