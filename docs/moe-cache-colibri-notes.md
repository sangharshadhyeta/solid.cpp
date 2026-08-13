# Ideas from JustVugg/colibri worth evaluating for moe-cache

Source: https://github.com/JustVugg/colibri (Apache-2.0, C engine + React web UI,
disk/RAM/VRAM tiered MoE inference). Not a port target as a whole — most of its
scope (disk tier, io_uring, NUMA, Metal/Vulkan backends, its own model loaders)
doesn't apply here. These are specific, small, self-contained techniques that do.

Status: research notes only, nothing implemented yet. Each needs its own A/B
against the ported moe-cache before merging — Colibri's own README is explicit
that even their validated techniques are "measurable policies, not promises."

## 1. LFRU eviction + hysteresis + decay (runtime, dynamic)

File: `c/tier.h` (~60 lines).

Our ported `ggml/src/ggml-cuda/moe-cache.cu` uses plain LRU (doubly-linked list,
evict from `lru_head` — see `moe_cache_lru_remove`/`moe_cache_lru_push_back`,
around line 452-660). Colibri instead scores each candidate with
`(heat << 8) | recency`, so frequency is primary and recency only breaks ties
(`tier_pick_lfru`), and gates every eviction behind a 25%-plus-4 hysteresis
margin so two similarly-hot experts can't ping-pong. `tier_decay()` halves heat
counters periodically so the hot set can still adapt to a workload change.

Motivation: matches the RFC's own measured routing skew (top 10% of experts
get ~80% of hits) better than plain recency. Worth A/B'ing against our LRU on
real hit-rate/decode-speed numbers before considering it.

## 2. Persistent cross-session usage history (semi-dynamic — persists across runs)

File: `c/route_trace.h`, `.coli_usage` file format.

leloch's final rework explicitly *removed* hot-set persistence to cut
maintenance burden (commit a01eb2646). Colibri kept it and the implementation
is a careful reference for doing it safely if we ever revisit that trade-off:
- versioned text format, two header records (`-1 <n_layers> <n_experts>`,
  `-2 <version> <engine_id>`) so old parsers still work on new files
- engine-identity hash in the header so one model/engine's history can't
  silently corrupt another's placement (refuses by default, explicit
  `PIN=<path>` override required)
- atomic write via `<path>.tmp` + `rename()`
- its own decay (`COLI_USAGE_DECAY`) separate from the in-memory tier decay,
  because a cumulative histogram gets less responsive as it grows (measured:
  at 18.2M selections a typical turn only moves the ranking 0.2%)

## 3. Live per-expert tier/heat heatmap ("Brain" view) — dynamic, requires a running server

File: `web/src/Brain.tsx`.

Wire format is the interesting part: `/experts` returns one **hex byte per
expert** (top 2 bits = tier, bottom 6 bits = heat), plus a separate bitset of
experts that fired since the last poll. ~39KB of text for 19,456 experts, not
JSON objects. Rendering is plain Canvas 2D (no WebGL/three.js — confirmed
absent from `package.json`): `fillRect` per cell, color = tier, brightness =
heat, and the "flash on fire" effect is a per-expert `Float32Array` pulse set
to 1.0 on a hit and decayed `*= 0.94` per animation frame. Polls every 1.5s,
not a websocket.

Only works while the server is actively running — needs a live endpoint, not
a static report. Portable version: `moe-cache.cu` already tracks
hits/misses/evictions per device (`MOE_CACHE_LOG`); extending that to
per-(layer,expert) tier+heat state, packed the same way, plus a small canvas
component in `tools/ui`, would be a scoped debugging/demo view — much smaller
than adopting Colibri's engine.

## 4. Static pre-flight resource planner — NOT dynamic, no running server needed

File: `c/resource_plan.py`, `build_plan()`.

Scans a model's tensor headers directly (no weights loaded) to get exact
per-expert-layer byte sizes, discovers hardware (RAM via /proc/meminfo or
platform equivalents, VRAM via nvidia-smi/rocm-smi, physical core count
correcting for SMT/Apple P+E cores, NUMA socket count), and computes a
**projected cache hit rate** before the model even runs — `resident_expert /
total_expert` bytes given the real VRAM+RAM budget. Correctly avoids
double-counting budget on unified-memory chips (Grace Blackwell, Jetson)
where VRAM and RAM share one physical pool.

The genuinely novel bit: it ties that projection to auto-tuning *other*
features, e.g. disabling speculative decoding when the projected regime is
compute-bound or disk-bound with low hit rate, because their own measurements
show speculation actively loses there (-42% one case, -32% another).

Portable version for us: a `llama-plan`-ish pre-flight tool — scan a GGUF's
expert tensor sizes (metadata only, fast), discover VRAM/RAM, project a hit
rate for our `--moe-cache` budget before the user runs anything. Complements
the runtime cache rather than replacing it.

## Reality check on Colibri's own numbers

Independent review (wavect.io, GLM-5.2 on consumer hardware) tempers the
README's framing considerably: cold inference on a 24GB-RAM laptop is
0.05-0.1 tok/s (one token every 10-20s) with a 3-4% expert hit rate; the
5.8-6.8 tok/s "success" case needed six RTX 5090s and 251GB RAM with all
experts VRAM-resident (60%+ hit rate) — not actually consumer hardware.
Also: single-generation only (no concurrent requests/batching), no
multimodal input. Doesn't diminish the individual techniques above (each is
small, self-contained, independently verifiable in the source) but the
engine's own headline claims are rosier than its real-world numbers -
calibrate expectations accordingly, especially for candidate #4's hit-rate
projections, which assume a use case with a large enough cache budget to
actually work.

## Load-bearing caveat: cache value depends on routing skew, not just implementation

From `ggml-org/llama.cpp` PR #26563 ("expert hot store", a third independent
implementation of this idea, active Aug 2026) comment thread, user
blakemartz, 2026-08-08: tested the same hot-cache approach on
DeepSeek-V4-Flash-0731 (256 experts, top-6+1, CPU-offloaded, 2x RTX 4090).
DeepSeek's aux-loss-free load balancing keeps routing near-uniform — after a
1.4K-token warm pass, 209-229 of 256 experts showed warm per layer. At ~11%
VRAM:model capacity ratio, a dynamic hot-cache can't beat ~11% hit rate on a
uniformly-routed model. Measured: dynamic cache 12.77 tok/s vs 13.45 tok/s
for a dead-simple static "pin N complete layers to GPU" baseline at the same
VRAM — static won because complete layers hit 100% by construction with zero
lookup/swap overhead, while the dynamic cache pays real bookkeeping cost for
a hit rate that doesn't beat blind capacity.

Their conclusion, and ours: the 1.7-2.1x wins reported for Qwen3.6-35B-A3B
(this PR's headline number) likely depend specifically on Qwen's routing
being skewed. Balanced-routing models (increasingly common via aux-loss-free
training) may see static placement beat a dynamic cache outright.

Action for our own testing: once Gemma-4-26B-A4B finishes downloading, don't
just measure decode speedup — measure the hit-rate *distribution* across
experts (how concentrated vs uniform). A speedup that tracks raw VRAM:model
capacity ratio means we got a favorable model, not that the approach
generalizes. Encouragingly, a separate data point in the same thread
(Tropfchen, Gemma4-26A4-APEX, cold 18 tok/s -> warm 30 tok/s) suggests Gemma
4's routing does have exploitable skew, unlike DeepSeek's.

## GLM-5.2 specifically: skew appears real (independent field data)

From the same RFC #24528 thread, user SharkWipf (RTX 6000 + Threadripper,
independent of leloch/noonghunna): measured 30-50% decode speedup on GLM 5.2
from a hot-cache approach, and directly observed the shape: "the top ~1000
most frequently hit experts per conversation thread seem to be the most
important ones... after that it seems to fall off rapidly." noonghunna
initially suspected this could be the same under-warmed-pool artifact that
tripped their own early DeepSeek measurement, but after cross-checking
conceded "your falloff is probably real" for the GLM-5.2 case specifically
(with the residual caveat that nobody fully ruled out a VRAM-ceiling
explanation over a routing-saturation one). Net: GLM-5.2's routing looks
skewed like Qwen's, not uniform like DeepSeek's aux-loss-free routing -
consistent with Gemma-4-A4B's own cold/warm gap noted above. Reasonable
prior going into our own test: expect a real win, not a capacity-ratio-only
one, but confirm by looking at the hit-rate-vs-pool-size *shape*, not just
the final speedup number.

## Benchmarking traps to avoid (learned the hard way by others, same thread)

Several testers burned real runs on these; worth checking off before trusting
our own numbers once Gemma-4-26B-A4B is downloaded:

- **`--moe-cache` defaults to `auto`, which is ON.** Omitting the flag is
  NOT a control arm - it measures cache-vs-cache. The zero arm needs
  `--moe-cache off` (or `0`) explicit.
- **The requested budget isn't the granted budget.** VRAM-capped silently;
  read the actual `[moe-cache] ... pool[0]: ... slots=N total=M MiB` log
  line, not the flag you passed.
- **Never set `GGML_OP_OFFLOAD_MIN_BATCH` low on a cache-enabled config.**
  It offloads expert `MUL_MAT_ID` to GPU directly, so no CPU-resident expert
  op remains for the cache to intercept - the cache silently never
  allocates at all. One tester measured a 4.2x *loss* (22.34 -> 5.31 tok/s)
  from this exact mistake. Leave it at the default (32).
- **A batch-size gap exists between the cache's max batch (default 8,
  `GGML_CUDA_MOE_CACHE_MAX_BATCH`) and the offload threshold (default 32,
  `GGML_OP_OFFLOAD_MIN_BATCH`).** Batches of 9-31 are served by neither
  mechanism. Matters for `--parallel N` or larger speculative-decode drafts.
- **Tune admission (`GGML_CUDA_MOE_CACHE_ADMIT_AFTER`) by wall-clock time,
  not hit rate.** A bigger pool with a *higher* hit rate measured ~11%
  slower for one tester - hit rate is a diagnostic, not the objective.
- **Prefill is essentially untouched by the cache itself** (within 0.3-0.6%
  in a clean single-variable test) - large reported prefill regressions
  elsewhere turned out to be `-ub` size tradeoffs (smaller ubatch frees
  compute-buffer VRAM for the pool, at a measured -12.5% prefill / +10%
  decode tradeoff) or mmap/disk-streaming effects conflated with the cache.
- **Composes well with an external speculative-decode drafter, but only if
  the drafter is CPU-resident.** One tester measured 24.4 tok/s (cache
  alone) + 33.6 tok/s (GPU-resident DSpark drafter alone) failing to
  compose at all when combined on GPU (cache sees ~0 hits - a device/session
  binding bug under the shared-draft-device reordering), but 46.8-48.2 tok/s
  when the same drafter ran on CPU (`-devd none`) instead. Relevant for our
  "expert-cache first, DSpark after" sequencing decision.

## 5. Live skew detection driving adaptive strategy (our own idea, not from Colibri)

Checked: Colibri does not do this anywhere (`tier.h`, `route_trace.h`,
`resource_plan.py` all grepped for gini/entropy/skew/concentration - none
found). It only reacts to raw heat counts, which implicitly reflects skew
but never quantifies or acts on it. So this is a novel design, synthesized
from what we've found, not an extraction from Colibri's source.

The idea: leloch's earlier (pre-rework) design had a "baseline-sampled
bail-out" - measured the cache against the pure-CPU path and disabled
itself if not winning - removed in the final rework for maintenance burden
(commit a01eb2646). The RFC thread independently confirmed *why* a static
signal like that matters: dynamic caching only beats static layer-pinning
on skewed routing (GLM-5.2, Qwen) and can lose to it on uniform routing
(DeepSeek's aux-loss-free routing). A live skew measurement (e.g. a
running Gini coefficient or top-K-share over the per-expert hit-count
window, computed at negligible cost since we're already counting hits)
could drive that decision continuously instead of a one-shot bail-out:
lean into aggressive admission when concentration is high, fall back
toward simpler/static-like behavior (or warn and suggest `-ot` instead)
when it's not. Also matches SharkWipf's observation that skew may be
partly *workload*-dependent (per conversation thread), not purely a fixed
model property, which a one-shot startup measurement can't capture but a
live one could.

Not scoped or estimated yet - flagging as a real candidate, not a plan.

## Already checked, no action needed

**NaN-safe router argmax** (`route_trace.h`'s `rt_router_pick`): if router
logits are ever NaN, naive argmax leaves `best = -1`, later used as an array
index / file offset — silent corruption, not a crash. Checked our
`ggml/src/ggml-cuda/topk-moe.cu:136-146` — already sanitizes NaN to
`-FLT_MAX` before the iterative argmax, with a comment referencing
https://github.com/ggml-org/llama.cpp/issues/19659. Nothing to port.
