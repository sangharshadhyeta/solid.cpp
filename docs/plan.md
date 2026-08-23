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
2. **[in progress]** Expose `req_dir_x/y` (and per-expert positions) via
   `/experts` so the Brain/Atlas UI can render it as a live marker over the
   existing static topic map.
3. **[not started]** Warming action — rank each layer's experts by cosine
   similarity between their Atlas position and `req_dir`, feed the top-K into
   the existing `host_promote_queue`/prefetch machinery. **Prefetch-only,
   never evict**, until proven — same discipline every other predictive
   mechanism in this cache (router-lookahead, speculative eviction,
   cross-depth agreement) was held to before earning eviction rights.
4. **[not started]** Pairwise co-activation ("fire together") signal — track
   live `(expert_i, expert_j)` co-selection frequency across real traffic,
   independent of topic label. This is the more direct answer to "why can't
   the atlas be dynamic" than nudging static positions from a live-classified
   topic would be, and it's a genuinely different, complementary signal to
   step 3, not a replacement.
5. **[not started]** Pathway/connectome visualization — add Z = layer depth
   to the Atlas view; plot real per-token trajectories through the resulting
   3D volume using step 4's co-activation data. Recurring pathways should
   show up as literal bundles of parallel lines through the volume (the
   "brain folds toward what's used together" intuition that motivated this,
   and the direct visual payoff of steps 3-4).

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
