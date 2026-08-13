# START HERE - handoff summary for a fresh session (e.g. on the H200 box)

If you're picking this up cold, with none of the conversation that produced
this document: read this section first, then use the rest of the file as
reference. Everything below is dated and sourced - treat anything not
re-verified on the new hardware as a hypothesis carried over from a
different GPU (RTX 3060, 12GB), not a guarantee.

## What this is

A working port of a community MoE expert-cache feature into llama.cpp,
validated end-to-end with a real measured speedup. Goal: run large MoE
models (target: GLM-5.2, 744B) fast on hardware where the model doesn't fit
in VRAM, by keeping hot experts cached in VRAM while cold experts run on
CPU - instead of either the whole model running slow on CPU, or needing
enough VRAM for the whole thing.

## Where the code is

Branch `moe-cache-port` in this repo, based on `ggml-org/llama.cpp` master
at commit `1f368f354`, 22 commits ahead. Remotes already configured:
`origin` = `ggml-org/llama.cpp`, `leloch` = `leloch/llama.cpp` (source of
the ported feature, branch `moe-cache-pr`). If this is a fresh checkout
elsewhere, `git log --oneline` on this branch is the actual build history -
trust it over this doc for anything code-related, since docs can drift and
git can't.

Key commits, in order: our own mmap host-pinning fix, then 4 ported commits
from leloch/moe-cache-pr (the core feature + its rework), then a fixup
commit for rebase drift (2 small bugs from porting across 797 commits of
upstream drift, both documented in the commit message), then this
documentation trail.

## How to build (what worked on the sandbox RTX 3060 box)

CUDA 13.3 toolkit, gcc-toolset-13 (needed specifically because the stock
system gcc/binutils couldn't assemble AVX-VNNI instructions the CPU backend
generates with `-march=native` - unrelated to CUDA, just a build-env
gotcha). Configure with `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=<your
GPU's compute capability>` (was 86 for the 3060; H200 is Hopper, compute
capability 90 - use `-DCMAKE_CUDA_ARCHITECTURES=90` or `native`). Verify
with `./build/bin/llama-cli --list-devices` before doing anything else.

## The two gotchas that will bite you immediately - set these explicitly

Do not trust the defaults. Confirmed by direct testing, not assumption:

1. `--moe-cache auto` (and `on`) **never engages on a single-GPU system**,
   full stop, regardless of free VRAM - there's a hard `automatic &&
   devices < 2` check in the code. **Always use an explicit numeric budget**:
   `--moe-cache <MiB>`.
2. `GGML_CUDA_MOE_CACHE_MAX_BATCH` defaults to **1**, not 8 (8 is only the
   max allowed value). Default 1 rejects every call except exact
   single-token batches - even ordinary prefill gets silently rejected, no
   error. **Set `GGML_CUDA_MOE_CACHE_MAX_BATCH=8` explicitly.**
3. `GGML_CUDA_MOE_CACHE_RESERVE_MB` defaults to **3072 (3GB)**, sized for
   24GB+ cards. Check actual free VRAM after model load (`nvidia-smi` while
   the model is loaded, or read the `[moe-cache] CUDA%d prepare_budget`
   figures if you re-add diagnostics) before assuming the default is fine -
   on a 12GB card it left only ~47MB, blocking the cache entirely. **On the
   H200 (141GB) this is much less likely to bite, but verify rather than
   assume** - a 26B model with 5-10 concurrent user KV caches could still
   eat into headroom differently than our single-user sandbox test did.

**How to verify it's actually working, don't trust silence**: run with `-v`
and grep for `[moe-cache] CUDA%d pool[` (confirms a pool was actually
allocated) and `[moe-cache] CUDA%d hits=` (confirms real cache traffic). No
error is printed when the cache silently fails to engage - absence of these
two log lines, not an error message, is the failure signal. If you don't
see them, one of the three gotchas above is still active.

## What's actually validated vs. still a hypothesis

**Validated by direct measurement** (RTX 3060, Gemma-4-26B-A4B, see full
section below): the port compiles, runs, produces correct output, and
delivers a real +54% decode speedup (27.0 -> 41.6 tok/s) with 62.1% cache
hit rate once the three gotchas above are fixed.

**Still a hypothesis, not yet measured**: everything about the GLM-5.2/H200
target specifically - the 40+ tok/s target, the capacity math (~439GB
model, ~429GB experts fitting in 512GB RAM), the MTP interaction, the
concurrency behavior at 5-10 users. All reasoned from evidence (GLM-5.2's
confirmed routing skew, other people's benchmarks on different hardware)
but none of it has been run for real yet. That's the next step.

## Recommended first moves on the H200 box

1. Build (see above), confirm `--list-devices` sees the H200.
2. Download `unsloth/GLM-5.2-GGUF` (UD-Q4_K_M, 11 shards, ~439GB total) -
   this will take a while regardless of connection speed given the size;
   start it early and work on other setup while it downloads.
3. Reproduce the RTX 3060 validation methodology on GLM-5.2 before trusting
   any tok/s number: explicit `--moe-cache <budget>`,
   `GGML_CUDA_MOE_CACHE_MAX_BATCH=8`, a `RESERVE_MB` sized to actual free
   VRAM after load (check with `-v`, don't assume), a clean `--moe-cache
   off` baseline run for comparison, and `-v` to confirm real
   `pool[`/`hits=` log lines before trusting any number.
4. Measure hit-rate *distribution*, not just the aggregate hit rate - see
   the "load-bearing caveat" section below on why routing skew (not just
   the mechanism) determines whether this actually helps GLM-5.2 the way it
   helped Gemma-4-A4B.
5. Only after that's confirmed working: layer in native MTP (already merged
   upstream, verified present in the unsloth GGUF - see below), then work
   through the medium-term list (concurrency tuning, prefix caching) as
   real usage surfaces real bottlenecks, not before.

---

# VALIDATED 2026-08-13: moe-cache works, real 54% speedup measured on RTX 3060

First real end-to-end test on the downloaded Gemma-4-26B-A4B model. Took real
debugging to get there - two undocumented defaults silently blocked
engagement on a VRAM-constrained card, found only by adding temporary
diagnostic prints to the actual source (reverted after, see git history for
the fix commit if these ever get upstreamed):

1. **`GGML_CUDA_MOE_CACHE_MAX_BATCH` defaults to 1, not 8** (8 is the max
   *allowed* value, not the default - our own earlier notes had this wrong,
   corrected now). Default 1 means the cache rejects every call except exact
   single-token batches - even a 2-token prefill gets silently rejected, no
   error, no log. Confirmed empirically: baseline and "cache on" runs were
   bit-identical in speed until this was set to 8 explicitly.
2. **`GGML_CUDA_MOE_CACHE_RESERVE_MB` defaults to 3072 (3GB)**, sized for
   GPUs with real headroom (RTX 3090 24GB, A100, H100 - the RFC thread's
   test rigs). On our 12GB RTX 3060, `-ngl 99` for a 26B/262144-vocab model
   leaves only ~3.05GB free after loading - the 3GB reserve ate nearly all
   of it (47MB left, against a ~136MB minimum pool requirement). Lowered to
   256MB to unblock. **This will matter for the H200 deployment too** if
   VRAM headroom after loading GLM-5.2's dense weights + KV cache for 5-10
   slots is proportionally tight - worth checking early there, not assuming
   3GB reserve is fine just because the card is bigger.

**Clean A/B result** (same prompt, same n_predict=100, `-ngl 99 -ncmoe 99`):

| config | decode tok/s | hit rate |
|---|---:|---|
| `--moe-cache off` (baseline) | 27.0 | - |
| `--moe-cache 2048` + tuned MAX_BATCH=8, RESERVE_MB=256 | **41.6** | **62.1%** (32421/52200) |

**+54% decode speedup, real and measured**, not projected. Hit rate climbed
from 44.2% on a shorter 20-token run to 62.1% on this 100-token run - cache
warm-up matters, consistent with the RFC thread's own "under-warmed pool
reads as saturated" warning. This is now hard evidence for what was
previously projection based on other people's hardware/models - directly
validates the core premise (Gemma-4-A4B's routing skew is real and
exploitable) and confirms the port itself works correctly end-to-end
(loads, runs, generates coherent output, no crashes, real speedup).

**Action for the GLM-5.2/H200 deployment**: don't assume defaults work.
Explicitly set `--moe-cache <budget>` (never `auto`/`on` alone - see the
single-GPU dormancy finding above), `GGML_CUDA_MOE_CACHE_MAX_BATCH=8`, and
size `GGML_CUDA_MOE_CACHE_RESERVE_MB` based on actual free VRAM after model
load, not the 3GB default. Verify eligibility empirically (real pool[]/hits=
log lines with `-v`), don't just assume config was accepted.

**Superseded 2026-08-13 (same day, later): these three defaults were fixed
in core code, not just documented as gotchas.** User pushback: "they removed
it so we won't do is not the correct argument... our system should at least
be as smart as we are doing by self." Changes in `moe-cache.cu`:
`max_batch` default 1→8; `reserve_mb` default 3072→0 (sentinel meaning
"compute live": `min(1024, max(128, free_mb/20))` MiB, i.e. 5% of free VRAM
at the moment of session creation, clamped 128-1024 MiB); the single-GPU
`automatic && devices < 2` dormancy gate relaxed to just `devices == 0` at
all three call sites (session creation, budget check, eligibility check).
Net effect: `--moe-cache auto` with zero env vars now does on a 12GB
single-GPU card what previously required hand-tuning two env vars per
piece of hardware. See the sweep results below for the A/B proof this
actually closes the gap to manual tuning.

# Autonomous calibration sweep, 2026-08-13 (RTX 3060, Gemma-4-26B-A4B)

Two unsupervised sweeps run via `llama-bench`/`llama-server` while stepping
away, testing whether the live-computed `auto` defaults above actually hold
up across configs, and how much headroom is left in placement/concurrency.
Scripts and raw logs: `/root/.claude/jobs/a804561e/tmp/moe_cache_sweep.sh`
(v1) and `moe_cache_sweep_v2.sh` (v2) - not committed to the repo (job
scratch dir), summarized here instead.

## v1 (126s wall clock - budgeted 55 min, workload was just fast on this HW)

**Phase 1 - `--moe-cache` mode/budget sweep, fixed safe placement (`-ncmoe
99`, all experts CPU-resident, GPU handles cache-hit rows only), `-p 128 -n
64 -r 3`:**

| `--moe-cache` | decode tok/s |
|---|---:|
| `off` | 27.71 ± 0.05 |
| `auto` (live-computed) | **48.90 ± 6.64** |
| `512` | 30.39 ± 1.08 |
| `1024` | 37.59 ± 1.40 |
| `2048` | 43.60 ± 1.72 |
| `4096` | 50.34 ± 8.29 |
| `8192` | 50.21 ± 7.70 |

**Headline finding: `auto` (48.90) lands within ~3% of the best
hand-picked value (4096: 50.34), both within the run-to-run noise band (±7-8
tok/s at this batch size).** The live-reserve fix isn't just "unblocked" -
it's landing at essentially the same place a human sweeping budgets by hand
would land, with zero tuning. This is the direct evidence for the
self-correcting behavior that was asked for: no env vars, no per-card
tuning, and it's not leaving the manually-tunable gains on the table.

**Phase 2 - `-ncmoe` placement sweep, `--moe-cache auto`, `-p 128 -n 64 -r
2`, pushing past the previously-assumed-safe boundary of 30:**

| `-ncmoe` | decode tok/s | result |
|---:|---:|---|
| 30 | 46.63 ± 8.60 | ok (previous "safe" baseline) |
| 29 | 47.55 ± 9.19 | ok |
| 28 | 47.71 ± 8.68 | ok |
| 26 | 49.72 ± 8.78 | ok |
| 24 | 50.77 ± 7.91 | ok, **no OOM** |

Monotonically climbing as more experts move to GPU, and the sweep stopped
at 24 only because that was the last value scripted, not because it hit a
wall. **This means the previously-documented "safe" placement (30) was
conservative - there was real untested headroom below it.** v2 (below)
pushes further to find the actual boundary.

**Phase 3 - concurrent-request emulation** (`llama-server --parallel 8`,
`--moe-cache auto`, mixed short/long/agentic-style prompts, `n_predict=60`):

| concurrency | aggregate tok/s |
|---:|---:|
| 1 | 37.56 |
| 2 | 70.22 |
| 4 | 80.88 |
| 8 | 103.07 |

Near-linear 1→2 (1.87x), diminishing but still positive returns through 8.
Healthy scaling shape for the batching to build on.

**Phase 4 - variance check** (5x repeat, `auto`, `-ncmoe 99`): 52.79 ± 7.36
tok/s. Consistent with phase 1's `auto` number within noise; ±14% run-to-run
variance at this prompt/gen size is real and should be accounted for when
comparing configs that differ by less than ~10-15%, not treated as
measurement error to explain away.

## v2 (403s wall clock - budgeted 50 min, again just fast on this HW)

Purpose: v1 left three things unexplored that the user's original ask
("emulate users and requests to find optimal solutions... various kinds of
configs and loads") called for and 126s wasn't enough time to cover: the
real OOM boundary, whether `MAX_BATCH` still matters now that it defaults
to 8, and concurrency at the actual "5-10 users" deployment target with
longer, more agentic-shaped generations.

**Phase A - joint `-ncmoe` x `--moe-cache` grid (`-p 256 -n 128 -r 3`),
pushing past v1's un-OOM'd floor of 24:**

| `-ncmoe` | `auto` | `4096` | `8192` |
|---:|---:|---:|---:|
| 24 | 57.30 ± 5.59 | 56.87 ± 4.96 | 57.04 ± 5.87 |
| 20 | 59.60 ± 4.74 | 59.54 ± 5.34 | 59.92 ± 5.19 |
| **16** | **60.99 ± 3.41** | **61.06 ± 3.36** | **61.35 ± 3.10** |
| 12 | 53.51 ± 0.94 | 53.67 ± 0.85 | 53.51 ± 0.94 |
| 8 | **OOM** (model load fails) | OOM | OOM |

**Two real findings here, both correcting earlier assumptions:**

1. **The optimum is `-ncmoe 16`, not 24 or 30.** Throughput is
   *non-monotonic* in placement aggressiveness: 24→20→16 climbs (57→60→61),
   but 16→12 drops sharply (61→53.5) before failing outright at 8. The
   previously-documented "safe" value of 30 (46.63 tok/s, from v1 phase 2)
   left ~31% of decode throughput on the table. The likely mechanism: below
   ~16, there's not enough free VRAM left for the cache pool + KV cache +
   compute buffers to operate without contention, even though the model
   itself still loads down to 12 - the failure mode isn't a hard wall until
   8, it's a soft throughput cliff starting around 12. **True OOM boundary
   on this card/model: between `-ncmoe 8` (fails) and `-ncmoe 12` (loads,
   but already past the throughput cliff).**
2. **Cache budget stops mattering once placement is right.** At every
   placement tested, `auto`/`4096`/`8192` are statistically indistinguishable
   (differences are inside the ± noise band). This means the earlier finding
   ("`auto` ≈ best hand-tuned budget") wasn't a coincidence of the specific
   `-ncmoe 99` placement it was measured at - it holds across placements too.
   Once `-ncmoe` is dialed in, hand-tuning `--moe-cache` past `auto` buys
   nothing further on this hardware.

**Phase B - `GGML_CUDA_MOE_CACHE_MAX_BATCH` sweep** (2/4/8/16/32, `-ncmoe
99`, `auto`): 53.40-53.78 tok/s across the entire range, all within ±6.8
noise. **Confirms the earlier fix (default 1→8) was the fix that mattered;
tuning past 8 buys nothing.** No further action needed here.

**Phase C - concurrency, wider bracket + longer generations** (`--parallel
16 -c 16384`, `auto`, `n_predict=200`, mixed agentic-style prompts):

| concurrency | aggregate tok/s | per-user tok/s |
|---:|---:|---:|
| 1 | 46.88 | 46.88 |
| 2 | 74.44 | 37.22 |
| 4 | 88.60 | 22.15 |
| **5** | **99.02** | 19.80 |
| 8 | 96.42 | 12.05 |
| 10 | 58.57 | 5.85 |
| 12 | 57.94 | 4.82 |
| 16 | 63.93 | 3.99 |

**Unexpected and important: aggregate throughput peaks at concurrency=5,
then collapses at 10-12 before a partial recovery at 16.** This is not a
smooth diminishing-returns curve - there's a real cliff between 8 and 10.
Not yet root-caused (candidates: KV-cache VRAM exhaustion at high
`--parallel` x `-c` on a 12GB card with `-ncmoe 99`, or slot-scheduling
contention in the server) - flagged as an open question, not a settled
conclusion. **Directly actionable for the H200 deployment**: don't assume
higher `--parallel` is free, or that a bigger card removes this cliff -
this exact sweep methodology (concurrency emulation against the real
target count) needs to be re-run on the actual deployment hardware/model,
since the cliff's location is plausibly VRAM-budget-relative, not a fixed
concurrency number. On this card, for a "5-10 users" target, **5 concurrent
requests is the actual sweet spot, not 8 or 16** - matching the top end
of the target range, not the middle.

**Phase D - variance confirmation** (5x repeat, `-ncmoe 99`, `auto`,
`-p 256 -n 128`): 56.06 ± 5.20 tok/s tg128, 244.38 ± 8.86 pp256. Consistent
with phases A/B within noise. (Note: the script's auto-picked "best config"
for this phase mis-selected `-ncmoe 24` instead of the actual winner `16`,
due to a sort-key bug in the shell one-liner - harmless since phase A's raw
numbers above are already the authoritative source, but worth knowing the
phase D number isn't the true best-config confirmation it was meant to be.)

## Updated recommendation after v1+v2

For this card/model: `-ncmoe 16 --moe-cache auto`, no env var tuning needed
(`MAX_BATCH` default of 8 and live `RESERVE_MB` computation are both
already at their empirical optimum). Expected ~61 tok/s decode, up from the
27.71 tok/s `off` baseline (**+120%**, larger than the +54% first measured
because that used a suboptimal `-ncmoe 99` placement, not because the cache
mechanism itself improved). For concurrent serving, provision for the
concurrency sweet spot found empirically (5 here) rather than assuming
higher `--parallel` monotonically helps - re-run phase C's methodology on
whatever hardware is actually deployed to, since the cliff is suspected to
be VRAM-relative rather than a portable fixed number.

# Concurrency-cliff root-cause investigation, 2026-08-13 (same day, follow-up)

Dug into *why* the concurrency cliff (peak ~99 tok/s at concurrency=5,
collapse to ~57 tok/s at concurrency=10-12) happens specifically between 8
and 10. Three hypotheses, each tested with a real controlled change and
re-measurement, not just theorized:

## Hypothesis 1: `GGML_CUDA_MOE_CACHE_MAX_BATCH` gate - CONFIRMED a real bug, but not this bug

Traced the code path: `moe_cache_begin()` in `moe-cache.cu` rejects (falls
back to full CPU compute, no GPU cache acceleration at all for that call)
any batch where `n_tokens > config.max_batch`. `n_tokens` here is
`ids->ne[1]`, the actual token count in the current ubatch - during
concurrent decode with `--parallel N`, this scales with how many sequences
are decoding a token in the same step, up to `N`. The env var clamp was
hardcoded to `[1, 8]` - not just defaulting to 8, *incapable* of being set
higher - while the default was also 8. With `--parallel 16`, any decode
step batching more than 8 concurrent tokens together silently loses GPU
cache acceleration entirely for that step.

**Verified this ceiling is artificial, not structural**: the GPU scratch
buffers (`moe_cache_scratch_requirements`'s `constexpr max_rows = 64`) and
the CPU-side stack arrays (`ggml-cpu.c`'s `MOE_CACHE_MAX_TOPK = 64`) are
already sized for up to 64 rows regardless of `max_batch` - raising the
ceiling costs nothing structurally. (64 itself is *also* just a
human-picked constant, not derived from anything live - the difference
from the old 8 is only that 64 is wired into fixed-size memory layouts, so
raising it further would need an actual code change, not a config tweak.
Worth being honest about that distinction rather than treating 64 as
principled.)

**Fixed**: `max_batch`'s default is now derived live from the real
`n_seq_max` of the context (`llama_context` calls a new
`ggml_moe_cache.set_max_batch_hint()` vtable entry with `cparams.n_seq_max`
before `ggml_backend_sched_new()`, i.e. before the moe-cache session reads
its config), floored at 8 (preserves current single/few-user behavior and
ordinary prefill batching), ceilinged at the real 64 structural limit. The
env var override range was widened from `[1,8]` to `[1,64]` to match.
Verified end-to-end: with `--parallel 16`, the `[moe-cache] enabled` log
line now reads `max-batch=16` where it previously silently used 8 no
matter what `--parallel` was set to (this required moving
`ggml-backend-moe-cache.h` from `ggml/src/` to `ggml/include/` since
`llama-context.cpp` needed to call the new vtable entry and the private
`ggml/src` include path doesn't propagate to `libllama`).

**Re-tested the concurrency sweep with the fix active** (confirmed
`max-batch=16` in the log): the cliff was still there, nearly identical in
shape (concurrency=10 landed at 57.4 tok/s vs. 58.6 before the fix). *The
fix is real and worth keeping - `MAX_BATCH=8` was a genuine silent
cache-disabling bug for any concurrent deployment - but it was not the
cause of this specific cliff.*

## Hypothesis 2: cache-capacity thrashing under diverse concurrent workloads - RULED OUT

Enabled `GGML_CUDA_MOE_CACHE_STATS=100` for periodic hit-rate logging and
watched it through the concurrency=10 run. Hit rate held rock steady at
~82% (424680/517480 through 630399/766400 across the run, drifting only
±0.4%), pool fully utilized (2851/2851 slots) with healthy, steady eviction
churn, **zero** `dispatch-fail` or `collect-fail` throughout. The cache
mechanism itself performs identically before, during, and after the cliff
- ruling out thrashing/eviction-storm explanations entirely.

## Hypothesis 3: CPU thread-pool saturation for the miss-row (~18%) CPU fallback - RULED OUT

`n_threads` defaulted to 6 on this box's 6 physical / 12 logical
(hyperthreaded) cores (`common_cpu_get_num_math()` uses physical core
count). Reasoned this was a plausible bottleneck: cache misses still need
CPU computation regardless of GPU cache health, and more concurrent
sequences mean more total miss-row work funneling through a fixed 6-thread
pool - a classic queueing cliff as utilization crosses saturation. Note
this default is *not* an unmotivated guess like the old `max_batch=8` -
physical-core-only is a legitimate heuristic to avoid SMT contention within
one tight single-stream compute loop. But it's a heuristic tuned for one
stream monopolizing the CPU, not tested against many independent
concurrent streams, so it was worth checking.

Re-ran the identical concurrency test with `-t 12 -tb 12` (all logical
cores). Result: **not better, slightly worse** (concurrency=8: 80.1 vs 87.4
tok/s; concurrency=10: 47.8 vs 57.4 tok/s), plausibly because the compute
threadpool now saturates all cores and starves the server's own
request-handling/sampling/JSON threads. Ruled out as the cause - more
threads didn't fix it and moved the wrong direction.

## Conclusion: not root-caused yet, points outside moe-cache

| config | concurrency=8 | concurrency=10 | concurrency=12 |
|---|---:|---:|---:|
| `max_batch=8` (original bug) | 96.4 | 58.6 | 57.9 |
| `max_batch=16` (fixed) | 87.4 | 57.4 | 57.7 |
| `max_batch=16`, `-t 12` | 80.1 | 47.8 | 58.9 |

The cliff's shape (rise to a peak around 5-8, hard drop at 10, plateau
57-59 through 12) is essentially identical across all three configurations
- strong evidence the bottleneck is **not** in moe-cache at all (its own
health metrics stayed clean throughout every run), but in something
structural to `llama-server`'s own slot scheduling / continuous-batching
logic. That's a different investigation from the moe-cache port itself and
wasn't pursued further this session - flagged here as a genuine open
question, not swept under a wrong explanation. **Action for the H200
deployment**: re-run this same three-hypothesis-style controlled
investigation there rather than assuming any of these three (all ruled out
here) explain a cliff if one appears - the real cause is still unknown and
may be config- or version-specific to `llama-server`'s scheduler, not
something moe-cache-specific work will fix.

# Live self-tuning design, 2026-08-13 - what's done vs. what's next

Prompted by direct pushback during the sweep-result review: computing
defaults once from a snapshot isn't the same as being genuinely
self-correcting, and env-var gotchas someone has to know about isn't
"smart," it's just documentation debt. Two things now compute live at
*every* process start (not cached, not a one-time calibration artifact):

1. **`reserve_mb`** (fixed earlier session) - live 5%-of-free-VRAM at
   session creation, clamped 128-1024 MiB.
2. **`max_batch`** (fixed above) - live from `n_seq_max` at every launch.

**What's still static and shouldn't be**: `-ncmoe` placement. Our own
sweep proved memory-fit and throughput-optimal are different things
(`ncmoe=12` loads fine, `ncmoe=16` is 12%+ faster) - a real OOM-vs-throughput
tradeoff exists here that doesn't exist for the two levers above (their
scratch/reserve costs don't scale with the setting, so there was no
tradeoff to balance - raising them was a free win). Planned fix, agreed
scope for this session:

**Layer 1 (built and validated this session)**: `common_maybe_autoplace_moe_cpu()`
in `common/common.cpp`, called from `common_init_result::common_init_result()`
right after the existing `--fit` system has had its chance and right before
the real model load. Only steps in when nothing has already decided
placement (`-ncmoe`/`-ot`/`-cmoe`, or `--fit` itself, leave real patterns in
`params.tensor_buft_overrides` - checked directly, any real pattern anywhere
in that vector means skip) and the exact config as given would fail to fit -
predicted via the same no-alloc probe `--fit-moe-cache`/`common_fit_params`
already use (`common_get_device_memory_data`), used as a gate *before* the
real allocation happens, not a catch-after-crash retry. When it doesn't fit
and the model has MoE tensors, binary-searches the minimal CPU-offloaded
layer count that does fit (O(log n_layer) no-alloc probes, not O(n_layer) -
monotonic in placement aggressiveness so binary search is valid), then
writes the winning overrides into both `params.tensor_buft_overrides` (for
consistency/future reuse) and the `mparams` about to be used for the real
load.

**Why this needed to run after `--fit`, not skip when `--fit` is on (as
first implemented)**: `--fit` runs by default for `llama-server`. Initial
version bailed out whenever `params.fit_params` was true, assuming the
general system already covered this. It doesn't - `--fit`'s only levers are
`-c` and `-ngl` (whole-model layer count), and it aborts as soon as *either*
is user-pinned, with no MoE-specific fallback at all. Confirmed directly:
launching with `-ngl 99 -c 16384 --parallel 16` and no `-ncmoe` (both
explicitly pinned, nothing unusual) produced
`common_fit_params: failed to fit params to free device memory: n_gpu_layers
already set by user to 99, abort` - and `common_init_result` **silently
discards that failure status** and attempts the real load anyway, which
then hard-crashed exactly as our very first mid-session OOM did (`-ncmoe 16
-c 16384 --parallel 16`: `cudaMalloc failed: out of memory` on the KV cache,
because moving experts onto the GPU left no room for it). Moved Layer 1 to
run as an explicit fallback after `--fit`, operating on `--fit`'s own
(possibly still-failing) `mparams`/`cparams` rather than rebuilding fresh
ones - this is the gap `--fit` doesn't cover, not a duplicate of it.

**Validated end-to-end, four real scenarios, real hardware:**

| scenario | model | config | result |
|---|---|---|---|
| previously-crashing OOM | Gemma-4-26B-A4B (15.77 GiB) | `-ngl 99 -c 16384 --parallel 16`, no `-ncmoe` | auto-placed first 19 layers to CPU (6 binary-search probes), server started, served real completions at 37.2 tok/s |
| same model, smaller context | same | `-ngl 99 -c 2048 --parallel 1`, no `-ncmoe` | auto-placed first **12** layers (correctly less aggressive - smaller KV cache footprint leaves more room for experts on GPU; confirms this is live per-config, not a cached/fixed answer) |
| explicit override respected | same | `-ngl 99 -ncmoe 30 -c 16384 --parallel 16` | zero auto-placement log output, user's exact value used untouched |
| dense model, already fits | Qwen2.5-0.5B-Instruct | `-ngl 99 -c 16384 --parallel 16`, no `-ncmoe` | zero auto-placement log output (fits as given - `--fit` itself found no reduction needed); concurrent load test (1/4/8/16 requests) all succeeded, clean scaling 326→1656 agg tok/s, no cliff |

Case 2 is the important one for the "live, not one-shot" property asked
for earlier: the same model, same hardware, produces a *different* correct
answer (12 vs. 19 layers) purely because the context/concurrency config
changed - there's no cached/stale value involved, every launch computes
fresh.

**Layer 2 (deferred, scoped but not built)**: the throughput-optimal
refinement on top of Layer 1's safe floor - a short empirical calibration
(2-3 candidate placements, few seconds) run once per (GPU, model, context
config) combination and cached, instead of re-measured on every restart.
Also now includes `n_threads`/`n_threads_batch` as a swept dimension (see
the concurrency-cliff investigation above - no safe-analytic-floor exists
for that lever, only a real curve to measure). Explicitly deferred this
session in favor of shipping and validating Layer 1 first.

**Third item for Layer 2, not Layer 1**: `n_threads`/`n_threads_batch`.
Unlike `-ncmoe`, this has no safe-analytic-floor property to give it a
Layer 1 - our own test proved the "obviously safe" direction (more threads
= more parallelism) actually made concurrent throughput *worse* (see the
root-cause investigation above), so there's nothing to compute ahead of
time, only a real curve to measure. Belongs in the same empirical
calibration pass as placement, swept as its own dimension (the sweep
script should test `-t` values, not just `-ncmoe` × `--moe-cache`, exactly
per the standing ask that every check like this becomes part of the
repeatable script rather than a one-off manual investigation).

# Build list summary (consolidated 2026-08-13) - see sections below for detail/sourcing

## Already built, VALIDATED with real measurement
- moe-cache core port (leloch's design). Load-bearing feature. **Measured
  +54% decode speedup (27.0 -> 41.6 tok/s), 62.1% hit rate on our RTX 3060 +
  Gemma-4-26B-A4B** - see validation section above. RFC: 1.7-2.1x decode on
  skewed routing (Qwen-A3B), consistent with our own result. GLM-5.2
  confirmed skewed (SharkWipf: 30-50% measured).
- mmap host-pinning fix (ours). One-time load-speed win only, not decode.

## Near-term - targets the GLM-5.2/H200 deployment directly
- Native MTP speculative decoding (merged, PR #25980, needs enabling/testing).
  Comparable DSpark data on Gemma-4: 1.0-1.76x, task-dependent. Likely
  highest-value low-effort win - no external checkpoint needed.
- FR-Spec draft-vocab trimming for GLM's MTP (pattern proven for Qwen,
  issue #25187, needs ~30-line port to glm-dsa.cpp). ~75% draft LM-head
  compute cut, lossless. Small slice of total, but cheap once ported.
- LFRU eviction + hysteresis + decay, replacing our ported plain LRU
  (Colibri's tier.h). No hard number yet - needs A/B. Motivated by
  GLM-5.2's confirmed skew.

## Medium-term - multi-user serving (5-10 agentic users)
- Server-level cross-request prefix/KV reuse (narrower than full
  PagedAttention - maintainer-endorsed path, discussion #21961). No
  throughput number yet; underlying paged-KV experiment showed 26->247
  concurrent-sequence capacity jump at equal VRAM on an A10G.
- MTP batch-gate tuning (close the 9-31 dead zone between MAX_BATCH/
  MIN_BATCH). Tuning fix, not a feature. Unquantified until tested at
  real concurrency.
- Suffix Decode (PR #26283, model-free spec decoding). No number found.
  Additive to MTP, best case is repetitive agentic/tool-call output.

## Longer-term / exploratory - not yet scoped as real builds
- Persistent cross-session usage history (Colibri-style) - avoids
  cold-restart penalty, no number.
- Live skew-detection driving adaptive strategy (our own synthesis) -
  protects against static-beats-dynamic-on-uniform-routing failure case,
  unscoped.
- Live per-expert heatmap UI - debugging/demo value, not performance.
- Static pre-flight capacity/hit-rate planner - planning tool, not
  performance.

## Explicitly decided against
Full PagedAttention (maintainers want the narrower path), multi-LoRA
serving (not our use case), multi-node serving (single H200), SGLang's
expert-parallelism/DeepEP (multi-GPU only).

## Honest calibration on the combined number
RFC-thread testers got low-to-mid-20s tok/s with moe-cache alone on WORSE
hardware (2x3090 + repaired 8-channel DDR4) running the HARDER case
(DeepSeek's uniform routing). Our setup stacks three favorable factors on
top - GLM-5.2's confirmed skew, H200's far higher HBM bandwidth, native
MTP layered on - which is why 40+ tok/s reads as realistic, not optimistic.
Target, not a guarantee, until actually measured.

# Other software worth knowing about (not deep-dived into source yet)

## KTransformers (kvcache-ai/ktransformers) - validates our approach independently

Published research (SOSP 2026, top-tier systems conference), 19.2k stars,
Apache-2.0, active (pushed 2026-08-08). Now integrated as a backend into
SGLang for CPU/GPU hybrid MoE inference. Core strategy confirmed from their
own README: "Heterogeneous expert placement (hot experts on GPU, cold
experts on CPU)" - the same fundamental approach as leloch's moe-cache we
already ported, independently arrived at and peer-reviewed. Useful as
validation that the design direction is sound, not just a hobbyist idea.

Differentiators worth understanding better if we go deeper:
- AMX/AVX-optimized CPU kernels (Intel AMX-Int8/AMX-BF16 specific -
  depends on the office HPC's actual CPU supporting AMX; needs checking,
  not assumed)
- NUMA-aware memory management (relevant if the H200 node is multi-socket)
- **Native multi-concurrency support built in from the start** (per their
  "Apr 2, 2025: Support Multi-concurrency" changelog entry) - directly
  relevant to our 5-10 concurrent user requirement, which is currently an
  open question for our moe-cache port (see MTP x concurrency question
  above), not a solved design there.

Reported: >220 tok/s total throughput on trillion-parameter MoE models in
their hybrid setup (hardware config not verified from this pass - would
need to check if comparable to our 1x H200 + 512GB target before treating
as a benchmark to match).

Not yet: read their actual source for the concurrency/scheduling
mechanism, compared their eviction/placement policy in detail against
leloch's, or checked AMX availability on the target HPC's CPU.

## ik_llama.cpp (ikawrakow/ikawrakow_llama.cpp) - sibling fork, same problem space

llama.cpp fork specifically focused on CPU/GPU hybrid MoE offload
performance. Already has MTP decoding support merged for GLM-4.x MoE,
Qwen 3.5/3.6, Gemma 4, and GLM 5 - i.e. both of our target models already
have MTP support in this fork, same as we just verified for mainline
GLM-5.2. Uses a concrete, specific GPU-offload threshold formula: prompt
processing offloads to GPU above `32 * total_experts / active_experts`
tokens - a different, more MoE-shape-aware heuristic than the flat
default-32 `GGML_OP_OFFLOAD_MIN_BATCH` threshold we've been discussing for
mainline. Worth comparing once we're tuning batch-size gates for our own
deployment - this formula scales with the model's actual expert:active
ratio rather than being a flat constant.

Not yet: read their actual moe-cache-equivalent implementation, if one
exists, or diffed their offload heuristic against leloch's approach in
detail.

## SGLang (sgl-project/sglang) - MoE-first architecture, mixed relevance

31.7k stars, active (pushed 2026-08-13, today). Three real pieces:

1. **RadixAttention** - generalizes prefix caching from vLLM's fixed
   16-token block hash to a radix tree, handling *branching* conversations
   (agent retries, multiple tool-call paths) more naturally than exact
   block matching. Refines our existing prefix-caching candidate above -
   more relevant given our agentic/API-heavy usage pattern specifically.
2. **Expert Parallelism (DeepEP/MoriEP + EPLB + DeepGEMM)** - specialized
   all-to-all GPU communication for MoE token dispatch, dynamic expert
   load rebalancing, and grouped-GEMM kernels for MoE's per-expert
   batches. All about spreading experts across *multiple* GPUs - **not
   applicable to our single H200 deployment**, this solves a different
   problem (multi-GPU expert sharding) than ours (CPU/GPU tiering on one
   device). Noted for completeness, not a candidate for us.
3. **FR-Spec draft-vocab trimming** (`--speculative-token-map` in SGLang,
   production code) - restricts a speculative drafter's LM-head to a
   frequency-ranked vocab subset (e.g. top 32k of 100k+), cutting draft
   compute ~75% with a lossless guarantee (target still verifies full
   vocab). **Already proposed and implemented for llama.cpp**: issue
   #25187 (open, created 2026-07-01, unusually mature - implemented and
   tested on a real branch, commit 047bfa508). Reuses infrastructure
   llama.cpp's own EAGLE-3 implementation already has for a different
   reason (a `d2t` tensor that trims-then-scatters logits, for drafters
   with a smaller native vocab than the target). ~30 lines, no-op for
   existing GGUFs. **Currently implemented only for `qwen35.cpp`
   (Qwen3.5/3.6 dense MTP) - explicitly checked and confirmed NOT yet
   done for GLM's MTP path** (`glm4-moe.cpp`/`glm-dsa.cpp`). Directly
   relevant to us: proven pattern, just needs extending to our target
   architecture. Real, scoped opportunity, not speculative.

## Does SGLang already do what we're building? Checked directly - no, not reliably

Native SGLang GGUF support is "coming soon" per their own docs, not yet
mature. The only path to GGUF + CPU-offload in the SGLang ecosystem is via
the KTransformers integration, and even there, GGUF loading goes through
what they call a "llamafile backend" - i.e. it reuses llama.cpp/llamafile's
own GGUF-reading code, not independently-built infrastructure.

Reliability check: KTransformers issue #1655 (open, filed 2025-12-02, still
unresolved) - a user tested this exact scenario, including an actual
unsloth GGUF (`unsloth/gpt-oss-20b-GGUF`, same publisher/format we're
using). Results: DeepSeek R1 loaded but failed during batch capture,
GPT-OSS failed during loading entirely. A related issue also names
GLM-4.5-AIR (close cousin of our GLM-5.2 target) as broken with GGUF in
their stack. No evidence found of drafter + limited-VRAM + GGUF all three
working together in SGLang/KTransformers.

**Conclusion**: what we're building isn't a solved problem sitting on a
shelf elsewhere. SGLang is attempting something adjacent by borrowing
llama.cpp's own format-handling to do it, and even that combination has
real, open, unresolved bugs on exactly this class of model (large MoE,
unsloth-quantized GGUF). llama.cpp + the moe-cache port is arguably ahead
of SGLang specifically for GGUF + CPU-offload + native-drafter, not
behind it - useful calibration for the value of this whole project.

## Also found while researching: CUDA graphs for multi-slot decode

Issue #27009 (open, **created today**, 2026-08-13) - proposes decoupling
CUDA graph shapes from the number of active decode slots via fixed-size
padding (dummy token ids, -inf masked padding columns, reserved dummy KV
rows), so graphs can be captured once per bucket size and reused via
`cudaGraphLaunch` instead of rebuilding per active-slot-count. Directly
relevant to our open MTP x concurrency question above - if/when this
lands, it changes how multi-slot decode batching and shapes work, which
could interact with (help or complicate) the moe-cache batch-size gates
we're already tracking as an open question. Brand new, unresolved, worth
watching rather than acting on yet.

## Ollama / LM Studio - confirmed downstream, not independent sources

Both wrap llama.cpp rather than implementing independent MoE-offload
architectures. Ollama's MoE handling is transparently llama.cpp
underneath; their own recent additions are MLX (Apple Silicon) engine
support alongside llama.cpp, not a competing scheduling/placement design.
Checked directly (2026-08-13) rather than assumed - nothing to extract
here for this specific concern.

# Open questions, tracked but not yet resolved

## MTP verify-batch size x concurrency, interacting with moe-cache's batch gate

Native MTP (GLM-5.2's NextN head, PR #25980) is architecturally safer than
the external-drafter case the RFC thread found broken - MTP runs in the
same trunk graph, same device/session as the main model, so the "GPU
drafter causes the cache to see zero hits" failure mode (a device/session
binding conflict under shared-draft-device reordering) likely doesn't
apply here.

But a different, untested risk exists specific to our deployment: the RFC
thread found a "dead batch-size zone" - requests sized 9-31 are served by
neither the cache (gated by `GGML_CUDA_MOE_CACHE_MAX_BATCH`, default 8)
nor the bulk-offload path (gated by `GGML_OP_OFFLOAD_MIN_BATCH`, default
32). MTP verify batches are small individually (block_size ~7 in the
Gemma DSpark reference example), but with 5-10 concurrent `--parallel`
slots each potentially drafting-and-verifying at once, the *effective*
batch size the scheduler sees could land in that gap. Nobody in the RFC
thread tested MTP at real multi-user concurrency, only single-stream.

Not resolvable from the Gemma-4/RTX-3060 sandbox test - we won't be
running 5-10 concurrent `--parallel` slots there. Needs explicit testing
once we have access to hardware that can actually run that concurrency
(the real H200 deployment, or a scaled-down concurrency test earlier if
possible). If it turns out to be a real gap, the fix is likely widening
`MAX_BATCH` and/or lowering `MIN_BATCH` so there's no dead zone for the
batch sizes this deployment actually produces - not a redesign, just a
tuning gap to close once measured.

# Real deployment target (drives all priority calls above and below)

Office HPC: 1x H200 (141GB HBM3e), 512GB system RAM, GLM-5.2 (744B, Q4_K_M,
unsloth quant), 5-10 users - but usage is API/agentic (many requests per
user, not just interactive chat), not simple back-and-forth chat. Target:
40+ tok/s decode.

**Why unsloth specifically**: their dynamic-quant approach is what's
needed to hit the 40 tok/s sweet spot at Q4_K_M quality - same reasoning
that put us on the Gemma-4-26B-A4B unsloth GGUF for the sandbox test.

**Capacity math** (rough, Q4_K_M ~0.59 bytes/param average): ~439GB total
GGUF, ~10GB dense/attention (always VRAM-resident), ~429GB routed experts.
512GB RAM holds the entire expert mass resident - no disk-tier streaming
needed at all, unlike Colibri's harder cases. Leaves ~131GB of H200 VRAM
for the moe-cache hot-expert-set, KV cache across 5-10 concurrent slots,
and compute buffers once dense weights are subtracted - workable but needs
deliberate sizing once we're testing on the real hardware.

**Why this reprioritizes things above generic advice:**
- GLM-5.2's confirmed routing skew (SharkWipf: top ~1000 experts dominate,
  30-50% measured speedup) means moe-cache is a strong fit, not a maybe -
  validated for the real target, not just our Gemma-4 sandbox stand-in.
- GLM-5.2's native NextN/MTP speculative decoding is merged in-tree (PR
  #25980) - uses the model's own trained draft head, no external
  checkpoint needed. Likely the single highest-value, lowest-effort win
  for this deployment specifically. **Verified 2026-08-13**: fetched the
  header of `unsloth/GLM-5.2-GGUF` (UD-Q4_K_M, shard 1/11) directly and
  confirmed `glm-dsa.nextn_predict_layers = 1` in the GGUF metadata - the
  MTP layer was not stripped, it's present in the file we'd actually
  deploy. Also from the same header: `glm-dsa.block_count = 79`,
  `glm-dsa.expert_count = 256`, `glm-dsa.leading_dense_block_count = 3`
  (first 3 blocks dense, MoE starts after).
- 5-10 nominal users but agentic/API-heavy traffic pushes effective
  request volume and repetition patterns well past "5-10 people typing in
  a chat box." Prefix caching (agents resending stable system
  prompts/tool schemas every turn) and Suffix Decode (repetitive
  code-edit/tool-call output) both become more directly relevant here
  than they'd be for plain chat - matches the RFC thread's own finding
  that agentic coding workloads are prefill-heavy by construction.
- A single H200 (not multi-GPU) means the RFC thread's multi-device
  pool-distribution findings (sticky layer/device routing, uneven pool
  sizing across cards) don't apply - simpler single-device sizing problem.
- Sanity check against real numbers: RFC-thread testers got low-to-mid-20s
  tok/s decode with moe-cache on 2x RTX 3090 + repaired 8-channel DDR4
  (~100GB/s) running DeepSeek-V4-Flash 284B - the *harder* case (uniform
  routing). GLM-5.2's skew plus an H200's far higher HBM bandwidth plus
  512GB of presumably fast server RAM all point toward 40+ tok/s being a
  realistic target, not optimistic - more so once MTP speculative decoding
  stacks on top.

# vLLM features beyond DSpark/Suffix Decode - priority order

Not yet scoped, just recorded with a rough priority (nearest-term first).
Status verified by reading the actual PRs/discussions/issues, not assumed.

## 1. Automatic prefix caching (next up after DSpark/Suffix Decode)

vLLM hashes KV-cache in 16-token blocks and reuses them across *different*
requests sharing a prefix (system prompt, few-shot preamble, multi-turn
history) - not just within one conversation. A natural extension of
PagedAttention's block-based, reference-counted KV layout: since blocks are
already page-like, sharing them across requests by content hash falls out
almost for free. llama.cpp's prompt caching is per-slot/per-session today,
not a global hash-addressed cache shared across arbitrary concurrent
requests. Relevant for llama-server multi-session workloads (agentic/tool
systems hammering the same system prompt across sessions).

**Real status, verified by reading the threads (2026-08-13):**
- Two prior PagedAttention attempts: #17579 (closed, zero comments, no
  adoption) and #22569 (open draft, dormant since 2026-05, real numbers -
  on an A10G at equal KV budget, unified KV OOMs at 26 concurrent
  sequences, paged mode handles 247).
- The actual design discussion is #21961 ("Paged KV cache and scheduler:
  Phase 1"), still genuinely live - last comment 2026-08-11, two days
  before writing this. No maintainer verdict landed yet, not rejected,
  just unresolved.
- **The real maintainer pushback isn't against the outcome, it's against
  the implementation strategy.** ngxson (maintainer): "paged attn is only
  beneficial when the incoming requests have the same prompt prefix...
  we can obviously implement the same scheduling for cross-request KV
  reuse at server level" - i.e. get prefix-sharing without adopting
  vLLM's whole fixed-block-pool KV architecture. am17an (maintainer)
  separately noted most llama.cpp users don't run enough parallel
  requests for the >25-concurrency win to matter to them specifically.
  Even the original design discussion #21961 already deferred CoW/prefix
  caching itself to an unscoped "Phase 2" - it was never actually in
  scope for the PR that stalled.
- **Implication for us**: don't port PagedAttention wholesale. The
  narrower, maintainer-suggested path - server-level cross-request KV
  block reuse without a full block-pool rewrite - is the one with an
  actual chance of landing, and is a smaller, more scoped project than
  what either stalled PR attempted.

## 2. Chunked prefill + continuous batching (later - look at together)

Chunked prefill splits a long prompt's prefill into chunks interleaved with
other requests' decode steps, so one huge prompt doesn't stall everyone
else's tokens; only makes sense on top of continuous batching (a new batch
formed every iteration rather than waiting for the whole in-flight batch to
finish). Both are multi-tenant serving mechanisms - relevant to llama-server
specifically, not the CLI/single-user path.

**Real status, verified (2026-08-13):**
- Continuous batching is **already shipped** in llama-server, not a gap -
  PR #6358's title ("Allow continuous batching to be disabled") implies
  it's on by default. llama.cpp already has this one; what it lacks is
  vLLM's more sophisticated paged/chunked version layered on top.
- Chunked prefill's tracked PR (#10718) is **stale, not just unmerged** -
  created Dec 2024 as a single-day draft/example, zero updates since. Very
  different situation from the paged-KV discussion, which is still active
  this week. If we want this, we're likely starting closer to scratch than
  picking up existing momentum.

## 3. Multi-LoRA serving (after that)

Serving many different LoRA adapters against one base model simultaneously,
swapping per-request cheaply (S-LoRA-style). Real capability llama.cpp
doesn't match today, but even vLLM's own ecosystem is still fighting an
inherent conflict here as of 2026: prefix caching and multi-LoRA serving
actively fight each other, since KV cache can't be shared across requests
using different adapters - active research problem upstream, not a solved
pattern to copy.

**Real status, verified (2026-08-13):** llama.cpp is hitting the identical
collision right now, live and unfixed - open issue #26207 (filed
2026-07-28, unlabeled/untriaged): with per-request LoRA selection and
`cache_prompt: true`, the server reuses KV computed under adapter A for a
request selecting adapter B whenever the prompt prefix matches - silent
output contamination, nothing logged. The only workaround
(`cache_prompt: false`) means paying full prompt re-processing every
request, i.e. disabling the very feature (prefix caching) that would make
multi-LoRA serving fast. This isn't a "vLLM has it, we don't" gap - it's
the same open problem in both codebases, which changes the framing: fixing
this bug in llama.cpp *is* progress on the shared unsolved problem, not
catch-up on a solved one.

## 4. True distributed / multi-node serving (very end)

Tensor + pipeline + data + expert parallelism combined across a GPU
cluster - vLLM's actual mission territory (cloud-scale multi-tenant
serving, what most RL training loops and inference-as-a-service providers
build on).

**Real status, verified (2026-08-13):** llama.cpp's `rpc-server` is real
and used, not vaporware, but the open issues paint a rough picture:
crashes on multi-node CUDA RPC (#26583, GLM-5.2), no error recovery - one
worker's RPC failure aborts the entire process rather than returning an
error (#25938, with a fix in flight at #26724), and model loading that
serializes read+hash+dispatch on a single host core so a 535GB load takes
~15 minutes with the NIC and 95 other cores idle (#25890). This is real,
shipping infrastructure that people are actively hitting real bugs in -
not a mature, hardened distributed-serving stack the way vLLM's is.
Lowest priority for the same reason as before: this is vLLM's core
differentiator because it solves cloud-scale serving, a different mission
than llama.cpp's portability focus - but if we ever did touch this, fixing
the existing rough edges would matter more than adding new capability.

# vLLM features worth tracking (DSpark, Suffix Decode)

Separate from the Colibri notes below - these are already being actively
ported into `ggml-org/llama.cpp` by the community, not just candidate ideas.

## DSpark (already partially in our checkout)

Block-wise speculative decoding: a small draft model proposes several
tokens per step via a "Markov head" instead of one at a time, with
per-token confidence dynamically scheduling how many drafted tokens to
verify. Originally DeepSeek-V4's technique, in vLLM via
vllm-project/vllm#47093 and the vllm-project/speculators checkpoint format.

- Core support: PR #25173 (merged 2026-07-28, well before our checkout at
  1f368f354) - `src/models/dflash.cpp` and related files already present.
- Newer checkpoint format: PR #26275 "dspark: support speculators-format
  checkpoints" (open, NOT in our checkout - confirmed via grep, no
  "speculators" handling anywhere in conversion/*.py or src/). Needed if we
  want to load checkpoints using the newer vLLM speculators format
  (`sample_from_anchor`, pruned draft vocab + d2t remapping table).

**Directly relevant to us**: PR #26275's own verification benchmark uses
[`makora-ai/gemma4-26b-a4b-dspark`](https://huggingface.co/makora-ai/gemma4-26b-a4b-dspark)
- a public DSpark draft checkpoint whose `speculators_config.verifier`
explicitly targets `google/gemma-4-26B-A4B-it` / `Gemma4ForConditionalGeneration`,
the exact base model our unsloth GGUF is quantized from. Confirmed compatible
on the target side. The draft's own internal transformer body is a small
5-layer **qwen3**-architecture head (not Gemma), block_size=7,
confidence_head_with_markov=true, markov_rank=256 - matches the "small
Markov head" description found via web search earlier. Checkpoint is raw
`safetensors`, needs GGUF conversion; likely routes through
`conversion/qwen.py`'s existing dspark-handling (that file already
references dspark/dflash) since the draft body is qwen3-shaped, though this
isn't confirmed without porting #26275 and trying it.

Reported speedup (task-dependent, from the PR's own numbers): coding 1.76x,
QA/rag ~1.06-1.26x, writing ~1.00x (no gain), acceptance rate 0.24-0.58
depending on category.

**Path to actually testing this**: port PR #26275 (same kind of work as the
moe-cache port, smaller diff), download the makora-ai checkpoint, convert to
GGUF, run against our Gemma-4-26B-A4B target. Not started - this is what
"expert-cache first, DSpark after" (our earlier sequencing decision) points
to next once moe-cache is validated.

## Suffix Decode (open PR, not in our checkout)

PR #26283, open. Model-free speculative decoding - vLLM's
[SuffixDecoding](https://suffix-decoding.github.io/): builds an online
suffix tree from the current request's own generated tokens, best when the
matched suffix is long (repetitive output - code edits, structured output,
agentic tool-call loops). No draft model needed at all, so it composes with
literally anything, including DSpark or moe-cache, without the
device-placement complications DSpark has. vLLM's version optionally keeps
a global cross-request corpus (up to 10k past requests); llama.cpp's port
so far only builds the per-request tree, no global corpus yet.

Worth trying once we have a real agentic/repetitive workload to test against
- our own tool-call-heavy sessions in this very project would be a
reasonable proxy.

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

**Full scoping done, see `docs/moe-cache-planner-scope.md`** - where it
lives (extends `tools/fit-params/fit-params.cpp`, not a new binary), what
new code is needed (an expert-size scanner, resurrected+widened from code
that briefly existed in our own git history), output format, explicit v1
scope boundaries, and a testing plan. Not yet implemented.

**Elevated priority after 2026-08-13 validation - this is no longer just a
nice-to-have.** The RTX 3060 test above needed real debugging (temporary
source-level diagnostics) to discover that `--moe-cache auto`/`on` silently
never engages on a single GPU, `MAX_BATCH` defaults to 1 not 8, and the
default 3GB `RESERVE_MB` can eat nearly all free VRAM on a smaller card -
none of which produced an error, just silent zero-benefit. A pre-flight
planner should compute and print (or directly emit) the correct values for
all three, so nobody else has to repeat that debugging marathon:
- GPU count -> forces explicit numeric `--moe-cache <budget>`, refuses to
  suggest `auto`/`on` when count is 1
- always overrides `MAX_BATCH` to 8 (default is simply wrong for real use,
  barely needs "computing")
- actual free VRAM after estimated model load (dense weights + KV cache
  for N concurrent slots) minus a safety margin, sized against the
  model's real per-expert byte size (from GGUF metadata) so the
  resulting budget clears the ~64x-expert-size minimum pool requirement

**This is static, not dynamic - confirmed 2026-08-13.** GPU count, free
VRAM after load, and per-expert tensor size are fixed facts about the
hardware+model pairing, knowable before the process starts generating and
unchanging for the life of the run. A one-time pre-flight calculation
(exactly this candidate) is the right mechanism - no live monitoring or
running server needed. This is different from candidate #5 below (live
skew detection): routing *concentration* can genuinely vary by workload/
session while the server keeps running (SharkWipf measured skew "per
conversation thread"), which a one-time static calculation can't capture.
Don't conflate the two - static sizing (this section) and dynamic skew
adaptation (candidate #5) are separate mechanisms solving separate
problems, both worth having, not substitutes for each other.

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
- **CORRECTED 2026-08-13, confirmed from our own test + reading the actual
  source: `auto`/`on` mode NEVER engages on a single-GPU system, full
  stop, regardless of free VRAM.** `moe-cache.cu` line ~1365-66:
  `automatic && budget_devices < 2` unconditionally sets the session
  dormant. This contradicts noonghunna's RFC-thread claim ("no
  device-count check, just a 1GiB minimum-slab floor") - that must have
  applied to an earlier revision; the final rework we ported has a real,
  hard single-GPU gate. **Reproduced directly**: ran identical prompts on
  our RTX 3060 with `--moe-cache off` vs `--moe-cache auto`
  (`-ngl 99 -ncmoe 99`, same prompt, same n_predict) - got 27.0 tok/s
  both times, zero difference, no pool/log lines emitted either run.
  **Explicit numeric budgets bypass this gate** (`--moe-cache <MiB>`) -
  this is the only way to engage the cache on a single GPU. **Directly
  affects the real deployment too, not just this sandbox test - the H200
  target is also a single GPU.** Must always use an explicit budget
  there, never `auto`.
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
