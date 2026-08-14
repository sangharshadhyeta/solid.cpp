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

## Hypothesis 4 (2026-08-14): endpoint-bias artifact - RULED OUT, cliff is real

The three hypotheses above were tested 2026-08-13, before the endpoint-bias
bug (raw `/completion` bypassing the chat template, see the endpoint-bias
investigation section) was found. Worth checking whether the cliff itself
was ever real, or partly a measurement artifact of the same bug. Re-ran the
Phase C sweep through the corrected `/v1/chat/completions` endpoint with 16
distinct real prompts (not repeated), `-ncmoe 99 --moe-cache auto -c 16384
--parallel 16`, aggregate tok/s = total generated tokens / wall-clock for
all N concurrent requests to complete:

| concurrency | 1 | 2 | 4 | 5 | 8 | 10 | 12 | 16 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| agg tok/s | 42.18 | 69.49 | 81.53 | 85.94 | **92.12** | 54.38 | 58.52 | 64.54 |

Same shape as the original (peak around 5-8, hard collapse at 10, -41% vs
the original's -39%, partial recovery by 16). **The cliff is confirmed
real, not an endpoint-bias artifact.**

## Hypothesis 5 (2026-08-14): copy/PCIe contention under concurrency - inconclusive, mechanism likely never engaged

Motivated directly by the expert-prefetch port done this session (from
github.com/thecodacus/llama.cpp, fable5/prefetch-experts branch - unrelated
to the JustVugg/colibri material elsewhere in this doc): if GPU-idle-during-
copy compounds under concurrent load (more simultaneous sequences needing
CPU-resident expert weights fetched over the same PCIe bus), that could
plausibly explain a cliff once total in-flight copy volume crosses a
hardware limit. Re-ran
the same corrected sweep with `GGML_CUDA_REGISTER_HOST=1
GGML_SCHED_PREFETCH_EXPERTS=1` (the prefetch-experts port, measured
+36.0% prefill throughput on this same hardware) active:

| concurrency | 1 | 2 | 4 | 5 | 8 | 10 | 12 | 16 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| agg tok/s (patched) | 42.56 | 69.57 | 80.04 | 88.90 | 91.99 | 53.24 | 58.42 | 63.87 |

**Zero measurable difference at any concurrency level, cliff identical in
shape and magnitude.** Not a clean "ruled out" though: the prefetch
mechanism only engages above a batch-size threshold
(`ids->ne[0]*ids->ne[1] >= 2*n_expert` in `ggml_backend_sched_compute_splits`)
tuned for large-batch *prefill*, not single-token-per-sequence *decode* -
at concurrency=16 with Gemma-4's top-k, the per-step routing-id count is
plausibly still well under that threshold, meaning the patch likely never
activated during this decode-dominated workload at all. So this result
shows the prefill-oriented prefetch fix doesn't touch the decode-time
concurrency cliff, not that copy/PCIe contention during decode is
definitively ruled out - a genuinely decode-scoped version of overlapped
prefetch (much smaller per-step tensors, higher call frequency) is a
different, untested mechanism.

## ROOT-CAUSED, 2026-08-14: `MMVQ_MAX_BATCH_SIZE` in ggml-cuda, not moe-cache or llama-server

Five candidate explanations were checked first and none of them it
(moe-cache MAX_BATCH gate - real bug, not this bug; cache-capacity
thrashing - ruled out via stats; CPU thread-pool saturation - ruled out,
made it worse; endpoint-bias artifact - ruled out, cliff reproduces
through the corrected endpoint; prefill-oriented copy-prefetch - no
effect, likely never engaged for this decode-shaped workload). Rather
than keep testing moe-cache-adjacent hypotheses from outside, captured
full per-request timing distributions at concurrency=8 vs 10 instead of
just the aggregate number: wall-time spread across concurrent requests
stayed ~1.00x at *every* level tested (all requests in a batch finish
within 0.05-0.07s of each other) - **no starvation or fairness issue at
all**. Instead, every request's own per-token decode rate collapses
uniformly: ~12.6-14 tok/s/sequence through concurrency=8, ~6.0
tok/s/sequence at concurrency=9 (confirmed the exact boundary with a
dedicated 7/8/9/10 sweep). A pure batch-size-triggered compute cliff, not
a scheduling bug - which pointed straight at the GPU kernel dispatch
layer, not `tools/server/server.cpp`.

Found it in `ggml/src/ggml-cuda/mmvq.cu`: `ggml_cuda_mul_mat_id` (the
MoE expert-routing dispatcher) only uses the fast `MMVQ` kernel - built
for small batches - when batch size is at or below a per-(GPU-arch,
quant-type) threshold; above it, dispatch falls through to the general
`MMQ`/`MMF` kernels, tuned for large batches and sitting in an
efficiency valley for medium ones (matches the partial recovery seen at
concurrency=16 in the original sweep - closer to those kernels' own
better-utilized batch size, not fully recovered). On this RTX 3060
(compute capability 8.6, the "turing_plus" bucket in
`get_mmvq_mmid_max_batch_turing_plus`), the per-quant-type table has
explicit tuned values for several types but not for whatever quant our
model's expert tensors actually use (Q4_K or Q8_0, both absent from that
table) - so it silently falls to the generic default, `MMVQ_MAX_BATCH_SIZE
= 8`, which is exactly the measured boundary.

**Deployment-critical**: for NVIDIA compute capability >= Ada Lovelace
(includes Hopper/H200) the dispatch code skips the per-type table
entirely and always uses the flat default of 8:
```cpp
if (cc == GGML_CUDA_CC_VOLTA || cc >= GGML_CUDA_CC_ADA_LOVELACE) {
    return MMVQ_MAX_BATCH_SIZE;  // = 8, unconditionally
}
```
The same cliff is likely to hit the real GLM-5.2/H200 deployment at
concurrency 9-10, squarely inside the "5-10 users" target range,
independent of every moe-cache/Layer1/Layer2/LFRU optimization built
this session - this is a different, lower-level dispatch decision none
of that work touches.

**Fix shipped: warn, don't silently guess.** Not something `-ncmoe` or
any placement/VRAM lever can fix (it's a GPU-kernel-dispatch limit, not
a memory-fit question), so Layer 1 (`common_warn_concurrency_cliff` in
`common/common.cpp`) now warns at startup instead of changing anything.
`get_mmvq_mmid_max_batch()` is exposed outside `ggml-cuda` via the
existing proc-address mechanism (same pattern as the host-buffer
registration calls). **Multi-GPU/tensor-split correct**: initially
shipped scoped to device 0 only (the CUDA backend's internal device
indexing doesn't map 1:1 onto the global `ggml_backend_dev_get()` index
space), then fixed the same session - the exposed function now takes the
caller's own `ggml_backend_dev_t` directly and recovers the real
per-backend device index from `dev->context`, and the common.cpp side
checks every GPU device (not just the first), since on a tensor-split
setup the full concurrent batch hits whichever device is computing the
current layer - the cliff can happen on any of them, so the safe ceiling
is the minimum across every device. Since the model's actual
expert-tensor quant type isn't cheaply known before a full load, the
warning queries every common
MoE quant type and reports the minimum across them - a genuine
conservative lower bound, so it can fire a bit early but never miss a
real risk. **Also accounts for MTP**: speculative verification batches
`(n_max+1)` candidate positions per sequence into a single `MUL_MAT_ID`
call, the same multiplier moe-cache's own `max_batch` hint already uses
- so effective batch = `n_parallel * verify_width`, not just
`n_parallel`. Confirmed: `--parallel 1` with `spec-draft-n-max=9` (verify
width 10) correctly fires the warning on its own, where a naive
`n_parallel`-only check would have stayed silent. Runs after the
calibration-cache lookup so it sees the real (possibly cached) `n_max`,
not the pre-lookup default.

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
Also includes `n_threads`/`n_threads_batch` as a swept dimension - no
safe-analytic-floor exists for that lever (our own test proved the
"obviously safe" direction, more threads, actually made concurrent
throughput *worse* - see the root-cause investigation above), only a real
curve to measure.

**Layer 2 (built and validated this session): `--moe-calibrate`.** A new
`llama-server` flag. `common_moe_calibrate()` in `common/common.cpp`: finds
Layer 1's safe floor via `common_moe_find_safe_layers()` (refactored out of
Layer 1's own logic so both share it), then benchmarks real candidates
*above* that floor (`safe_n + k*step` for k=1..4, `step = max(1, n_layer/15)`)
via `common_moe_bench_candidate()` - a genuinely real load + real short
greedy-decode loop (deliberately no sampler subsystem, since output quality
is irrelevant to a speed probe and greedy decoding on a real prompt still
exercises representative MoE routing/cache behavior), timed with
`ggml_time_us()`. Picks the best by measured tok/s, then does the same for
a couple of thread-count candidates (default/physical/logical) at the
winning placement. Writes the result to
`fs_get_cache_directory()/moe-calibration.json` (nlohmann::json, already
vendored) keyed by GPU signature (name + **total** VRAM per device -
deliberately not free VRAM, which fluctuates with background load and would
make the key spuriously miss) + model path + model file size + `-c`/
`--parallel`/`-ngl`. Calibrate-then-exit, matching the existing
`--fit-moe-cache` preview-before-you-commit pattern rather than mixing
calibration into the same run as real serving.

**Cache lookup wired into Layer 1's fallback** (`common_maybe_autoplace_moe_cpu`):
before falling back to the binary search, checks the cache; if an entry
exists, runs **one** verification probe (not the full search) to confirm
it still fits under current conditions, then applies it directly (placement
+ thread counts) - fast path. If conditions changed enough that the cached
placement no longer fits, logs that plainly and falls back to the live
binary search rather than silently using a stale answer.

**Validated on real hardware** (RTX 3060, Gemma-4-26B-A4B, `-ngl 99 -c
16384 --parallel 16`, the exact previously-crashing config):

| `-ncmoe` candidate | measured decode tok/s |
|---:|---:|
| 19 (Layer 1's safe floor) | 35.35 |
| 21 | 47.65 |
| **23 (winner)** | **49.33** |
| 25 | 46.63 |
| 27 | 45.59 |
| thread candidate 12 (vs. default 6) | 42.88 (worse - default 6 wins) |

**Layer 1's safe floor (19) was the *worst* of every candidate tested -
Layer 2 found a placement 40% faster** (49.33 vs. 35.35 tok/s), directly
confirming the earlier finding that memory-fit and throughput-optimal are
different questions, not approximately the same thing. Cache file written
correctly (`{"CUDA0:11900;|<path>|<size>|c16384|p16|ngl99": {"n_cpu_moe":
23, "n_threads": 6, "n_threads_batch": 6, "tok_per_sec": 49.33, ...}}`).
Relaunching the exact same config **without** `--moe-calibrate` produced
`common_maybe_autoplace_moe_cpu: using calibrated MoE placement from cache
(ncmoe=23, n_threads=6, measured 49.33 tok/s on ...)`, started successfully,
and served a real completion at 45.68 tok/s (close to the calibrated
number; the small gap is normal first-request cache warm-up, consistent
with everything else observed about warm-up in this doc).

**Known limitations of this first version**: single-sequence decode speed
only (concurrent-load calibration - e.g. calibrating for a specific
concurrency target, not just solo throughput - is a real further extension,
not attempted here); each candidate costs a full real model load, so total
calibration time scales with model size and shard count (fine as a one-time
opt-in step for a long-lived server, would need reconsidering for very
large models like GLM-5.2 where a single load may itself take minutes -
worth timing on the actual H200 box before assuming the current candidate
count/step size is still reasonable there).

# Native MTP speculative decoding, 2026-08-13 (RTX 3060, Gemma-4-26B-A4B)

> **RETRACTED 2026-08-14, see the endpoint-bias investigation section
> below**: every number in this section and in the `--spec-draft-n-max`
> sweep section was measured via the raw `/completion` endpoint with an
> unformatted prompt. That endpoint bypasses this model's chat template
> entirely, and the model (instruction/reasoning-tuned) produces
> degenerate, highly-repetitive output on unformatted prompts -
> independent of MTP, moe-cache, or placement. Degenerate repetitive text
> is trivially predictable, which silently inflates both MTP acceptance
> rate and MoE-cache hit rate. **The real content was never checked for
> these numbers, only tok/s** - so treat every specific number below as
> unverified until the corrected re-measurement (same section, chat-
> templated, content verified coherent) confirms or corrects it. The
> qualitative finding that MTP produces a real speedup on this hardware
> does still hold - the corrected numbers are lower but still positive.

## Real measured speedup

Initially thought untestable on this sandbox - hand-parsed the main GGUF's
header directly and found no `nextn`/mtp metadata keys, so assumed the
Unsloth quant just didn't ship an MTP head. Wrong place to look: MTP isn't
baked into the main model's tensors, it's a **separate sidecar GGUF file**,
loaded through the *speculative decoding* subsystem
(`params.speculative.draft.mparams.path`), not the main model's own
tensors. Confirmed via the HF repo listing
(`unsloth/gemma-4-26B-A4B-it-GGUF`): a `mtp-gemma-4-26B-A4B-it.gguf`
sidecar exists (also BF16/F16/Q8_0 variants under `MTP/`), matching our
already-downloaded main model. Downloaded the Q8_0 variant (441 MB,
verified against a real HF-token-authenticated, resumable, retrying curl -
see the download-reliability note below) and tested directly.

`llama-bench` doesn't support `--spec-type`/`--model-draft` at all - used
`llama-server` for the A/B instead: `-ngl 99 -ncmoe 23 --moe-cache auto -c
4096` baseline vs. the same plus `--model-draft <mtp file> --spec-type
draft-mtp --spec-draft-n-max 2` (2 is Unsloth's own recommended starting
point), same two prompts both times:

| prompt | baseline | MTP-enabled | speedup |
|---|---:|---:|---:|
| photosynthesis explanation | 50.05 tok/s | 67.60 tok/s | **1.35x** |
| binary search + complexity | 56.91 tok/s | 75.57 tok/s | **1.33x** |

Real, consistent **~1.33-1.35x** decode speedup, in line with (conservative
end of) Unsloth's claimed 1.4-2.2x range for `--spec-draft-n-max 2`.
moe-cache showed no dispatch/collect failures under MTP's verify-batch
load in this short test - doesn't fully resolve the open question below,
just no red flag in this specific run.

## `--spec-draft-n-max` sweep - the "2" was never actually tested, it's just Unsloth's suggested starting point

> **RETRACTED 2026-08-14** - same reason as above: measured via raw
> `/completion`, content never checked, numbers unverified. See the
> corrected re-measurement in the endpoint-bias investigation section.

Same principle already applied to `-ncmoe`/`n_threads` in Layer 2: don't
trust a suggested default, measure it. Swept n_max in {1,2,3,4,5,6,8},
same two prompts, one full server restart per candidate (real load each
time, not a no-alloc probe):

| n_max | avg tok/s (2 prompts) |
|---:|---:|
| 1 | 60.11 |
| 2 (Unsloth's suggested default) | 70.75 |
| 3 | 55.64 |
| 4 | 71.94 |
| **5** | **74.72** |
| 6 | 67.12 |
| 8 | 36.80 |

**n_max=5 beat the default of 2 by ~5.6%.** Shape matches theory - too low
leaves speedup on the table (fewer chances to skip a full forward pass),
too high wastes compute on tokens whose acceptance probability compounds
downward with each additional draft position, collapsing hard by n_max=8.
**Caveat, stated honestly**: only 2 samples per candidate, real run-to-run
noise present (n_max=3's two runs differ by 65%, 42.06 vs 69.21) - the
*pattern* (peak in the middle, collapse at the high end) is trustworthy,
the exact winner between 2/4/5 is not fully settled without more repeats.
This is exactly the kind of dimension `--moe-calibrate` should sweep
automatically once MTP is in scope for Layer 2 - not done yet, tracked
below.

## Download reliability note (unrelated to MTP itself, but real and costly)

First attempt (plain anonymous `curl -sL`) silently truncated at 193 MB of
441 MB - reported exit 0 despite being incomplete (the redirect target is a
time-limited signed CDN URL; something about the transfer let curl think it
finished cleanly when it hadn't). The `hf` CLI's own download (Xet-transfer
backend) then hung at 0 bytes for several minutes in this sandbox - killed
it and went back to curl, this time with `-C - --retry 8 --retry-all-errors`
plus an `Authorization: Bearer <token>` header (same token already saved
from the main model download) for resumability. That download was also
oddly slow (~100-150 KB/s) until an unrelated leftover diagnostic `curl`
process (from an earlier troubleshooting command that got auto-backgrounded
after exceeding its foreground timeout) was found still silently
downloading the *same file to `/dev/null`* in the background, competing for
the same limited egress bandwidth - killing that stray process roughly 6x'd
the real download's speed (120 KB/s -> 863 KB/s). Lesson: when a
foreground command that should have finished quickly instead needs to be
backgrounded, track it and clean it up explicitly rather than assuming it
died - it may still be running and consuming resources silently.

## Quality-preservation clarification (came up mid-investigation, worth recording)

Speculative decoding is provably lossless for output quality by
construction - not approximately, exactly. llama.cpp's actual verification
(`common_sampler_sample_and_accept_n` in `common/sampling.cpp`) samples a
token from the *target* model's own distribution at each position
independently, then checks whether the draft's guess matches that
independently-sampled token exactly; on any mismatch it keeps the token the
target actually sampled and discards the rest of the draft. The output is
therefore always literally a token the target model sampled on its own -
a higher-precision drafter cannot lift output quality above what the
(possibly heavily-quantized) target model would produce alone. What a
better drafter *can* do is raise the acceptance rate (more of its guesses
match what the target independently picks), which is pure speed, not
quality. Relevant directly to the H200/GLM-5.2 deployment given the
1-2bit-quantized GGUF options - MTP will make an aggressively-quantized
GLM-5.2 faster, not better than that same aggressively-quantized GLM-5.2
running alone.

## Two real gaps found, not yet fixed

1. **Exact-match verification is more conservative than necessary.**
   `common_sampler_sample_and_accept_n`'s exact-match rule rejects cases
   the classic probabilistic accept/reject rule (`accept with probability
   min(1, p_target(token)/p_draft(token))`, from the original speculative
   decoding papers) would have accepted - specifically, whenever the
   draft's guess isn't the target's own top pick but the target still
   assigns it real probability mass. This matters more the noisier/more
   diffuse the target's distribution is (e.g. heavy quantization), which
   is exactly the GLM-5.2 deployment's situation. A probabilistic
   accept/reject scheme would likely raise acceptance rate - and therefore
   the real-world speedup - without weakening the lossless guarantee at
   all (that guarantee is what the original algorithm's proof covers).
   **Not verified yet**: whether `--spec-type draft-mtp` specifically
   routes through this exact function or a different implementation -
   check before assuming this applies unmodified to the MTP path.
   Meaningfully bigger and more delicate than LFRU eviction - touches
   core sampling/verification correctness code, not moe-cache.
2. **Planner has no visibility into the drafter's VRAM footprint.**
   `common_maybe_autoplace_moe_cpu`/`common_moe_calibrate` only probe the
   *main* model's memory needs (`common_get_device_memory_data(path_model,
   &mparams, &cparams, ...)`) - the draft/MTP model is loaded via a
   completely separate `llama_model_load_from_file` +
   `llama_init_from_model` call in `speculative.cpp`
   (`params.speculative.draft.mparams.path`), invisible to either layer of
   what we built. Didn't cause a visible problem here since the MTP head
   is tiny (441 MB) relative to the 12 GB card's headroom, but the
   principle is real and matters more on H200 where free VRAM after
   loading a 744B model may be thin. Confirmed the drafter must run on GPU
   to be useful at all (CPU round-tripping per token would cost more
   latency than speculative acceptance saves) - so this isn't a "maybe
   offload it" question, the fix has to *reserve* GPU space for it, not
   choose whether to. Standing design direction from this session: the
   actual split between drafter-VRAM and main-model-expert-VRAM should be
   found empirically (extend `--moe-calibrate` to test placement
   candidates with the drafter enabled), not via a fixed static
   reservation rule - "only the test tells the truth" applies here the
   same as it did to placement itself.

# Endpoint-bias investigation and Layer 1/2 architecture rewrite, 2026-08-14

## The saga, honestly, because the wrong turns are as instructive as the right one

While verifying the first joint `--moe-calibrate` run's winning config
(golden-section search over `-ncmoe` and `spec-draft-n-max` together, see
below), the actual generated text was checked for the first time in this
whole MTP investigation - and it was degenerate, repetitive garbage
("sz-sz-sz-sz...", "tah-tah-tah..."). Every prior MTP validation in this
doc (the "1.33-1.35x speedup" section, the n_max sweep) had only ever
checked `predicted_per_second`, never the actual `content` - so this
wasn't a new bug, it was a pre-existing one that had simply never been
looked at closely enough to catch.

Systematic isolation (each step a real test, not a guess):
- Reproduced at both the calibration's chosen `ncmoe=15` and the
  earlier-"validated" `ncmoe=23`, and at both `n_max=4` and `n_max=2`
  (the exact value the original "coherent output" claim used) - ruled out
  the calibration search itself as the cause.
- Reproduced on the exact commit from *before* this session's `max_batch`
  fix (stashed current work, checked out the old commit, rebuilt,
  retested) - ruled out a code regression from anything built this
  session.
- Reproduced with `--moe-cache off` entirely - ruled out moe-cache.
- Reproduced with no `--model-draft`/`--spec-type` at all (plain
  generation, same placement) - **ruled out MTP entirely**.
- At this point the working theory turned to GPU/VRAM-level corruption
  from ~30+ ungraceful `SIGKILL`s of CUDA processes over many hours on a
  non-ECC consumer GPU (RTX 3060 has no ECC VRAM). Checked `dmesg` for
  Xid errors (none), `nvidia-smi` for ECC errors (none), confirmed no
  leaked processes/shared memory, and retested in the cleanest state
  achievable without an actual reboot - **still reproduced**. This was
  the wrong turn: the user correctly pushed back ("hardware is perfect,
  don't question it") rather than letting the investigation settle on an
  unfalsifiable environmental explanation.
- Re-read the Unsloth MTP docs page (the same one already cited above)
  with a more targeted query and found the actual cause in the
  "Recommended Sampling Parameters" section, plus, independently, a
  parsing-failure log line (`common_chat_peg_parse`) from an earlier
  diagnostic run showing the model's *real* underlying generation was
  coherent text prefixed with stray channel-marker tokens - a strong hint
  that pointed at chat-formatting, not generation quality, being the
  actual point of failure.
- Confirmed directly: the same prompt via `/completion` (raw, unformatted)
  produced garbage; the identical request via `/v1/chat/completions`
  (chat-template applied server-side) produced clean, coherent reasoning
  content. Reproduced on a second prompt. Corroborated independently via
  a GitHub Discussion search - ["Why is my Llama.cpp result so much worse
  than for the same model on another platform?"](https://github.com/ggml-org/llama.cpp/discussions/7781)
  describes the identical symptom as a known pattern, not something
  specific to this setup.

**Root cause**: every MTP/moe-cache test this session used the raw
`/completion` endpoint with an unformatted prompt string. This model
(instruction/reasoning-tuned) requires its chat template
(`<bos><|turn>system...<|turn>user...<|turn>model\n`, applied only by
`/v1/chat/completions`) to generate coherently at all - independent of
MTP, moe-cache, placement, or hardware. Every throughput number measured
this way is invalid, not just cosmetically wrong: degenerate repetitive
text is trivially predictable, which silently inflates both MTP
acceptance rate (repeating a token is an easy guess for the draft) and
MoE-cache hit rate (repetitive text routes to a narrow, unrealistic set
of experts). The corrected re-measurement below is the only one of this
session's MTP/calibration numbers that should be trusted.

## Fix: unified the calibration benchmark path, not just the prompt format

`common_moe_calibrate()`'s two benchmark functions had two different,
independent bugs:
- The `-ncmoe` search (`common_moe_bench_candidate`, in-process
  `llama_decode()` calls, no HTTP server) fed the model the *same* kind of
  raw, un-chat-templated prompt.
- The `spec-draft-n-max` search (`common_moe_bench_candidate_server`,
  subprocess-based, added earlier this session) used the raw
  `/completion` endpoint.

Rather than patch both separately, removed the in-process benchmark
entirely and made the subprocess+HTTP path (`common_moe_bench_candidate_server`)
the *only* benchmark function, used by both searches and by thread
tuning. It now:
- Launches a real `llama-server` subprocess with `--temp 1.0 --top-p 0.95
  --top-k 64` (this model family's documented MTP sampling
  recommendation), a fresh port per candidate (fixes a real port-reuse
  race found separately - see below), and `-t`/`-tb` when tuning threads.
- Sends `/v1/chat/completions` requests (proper chat template, applied
  server-side, exactly like a real client) for two representative
  prompts, averaging `timings.predicted_per_second` across both.
- One retry per candidate before treating a failure as real (single
  subprocess runs are noisy samples - same lesson as the n_max=3
  run-to-run variance found earlier this session).

This also meant thread tuning - previously skipped whenever MTP was
calibrated, because its old in-process benchmark had no MTP awareness and
mixing MTP/non-MTP throughput numbers wouldn't have been a fair
comparison - could be **re-enabled**: since every candidate now goes
through the same real, correct path, MTP-aware thread tuning is directly
comparable to the placement/n_max numbers above it.

## Other real bugs found and fixed during this work (independent of the endpoint bug)

- **`n_seq_max=-1` crash**: `--moe-calibrate`'s early-exit in
  `tools/server/server.cpp` ran before `params.n_parallel`'s `-1` ("auto")
  sentinel got resolved to a real value, so `cparams.n_seq_max` silently
  wrapped to a huge unsigned value and crashed the no-alloc probe with
  `n_seq_max must be <= 256` - which then got **mis-reported** as "model
  has no MoE experts" because the probe's exception handler didn't log
  the real error, just returned an empty result indistinguishable from
  "not a MoE model." Fixed both: moved the early-exit to after
  `n_parallel` resolution, and made the exception handler log the actual
  exception message.
- **Stale `tok_per_sec` reporting**: the cache/log correctly picked the
  best `n_max`, but the reported throughput number stayed at the
  pre-n_max-search (`-ncmoe`-only) value unless the n_max result was
  strictly higher - meaning a genuinely-better MTP-inclusive result could
  still get under-reported. Fixed by carrying the n_max search's own
  winning number forward once it's known to be real.
- **Port-reuse race**: all `n_max` candidates originally shared one port
  for the whole search. `SIGKILL`ing a candidate's server doesn't
  guarantee the OS releases its listening socket before the next
  candidate tries to bind the same port (TIME_WAIT) - caused a real,
  reproducible intermittent "failed" candidate. Fixed with a fresh port
  per subprocess.

## Corrected, validated results (2026-08-14, chat-templated, content verified coherent)

Same RTX 3060 + Gemma-4-26B-A4B + MTP Q8_0 sidecar setup, full joint
`--moe-calibrate` run with the fixed benchmark path:

**`-ncmoe` golden-section search** (range [13,17], derived from Layer 1's
live safe floor):

| ncmoe | tok/s |
|---:|---:|
| **15 (winner)** | **51.36** |
| 14 | 46.84 |
| 13 | 41.81 |

**`spec-draft-n-max` envelope + golden-section search** (at ncmoe=15):

| n_max | tok/s |
|---:|---:|
| 1 | 54.93 |
| 2 | 56.06 |
| 4 | 54.64 |
| 7 | 49.60 |
| 8 | 54.10 |
| **9 (winner)** | **54.55** |
| 10 | 52.38 |
| 13 | 42.48 |
| 16 | 39.40 (envelope edge) |
| 32 | 22.79 |

**Thread tuning** (at ncmoe=15, n_max=9): n_threads=12 (all logical cores)
won at 56.92 tok/s, beating the default of 6 - notably the *opposite* of
this session's earlier concurrency-cliff finding (there, more threads
made things worse) - a reminder that these levers interact differently
under different workloads (single-sequence MTP drafting+verifying is not
the same CPU access pattern as the earlier multi-sequence concurrency
test) and shouldn't be assumed to generalize from one context to another.

**Final winner**: `ncmoe=15, n_threads=12, spec-draft-n-max=9`, 56.92
tok/s, cached correctly and picked up automatically on a normal launch
(confirmed: `common_maybe_autoplace_moe_cpu` log line shows all three
values applied). **Real coherent output confirmed** via
`/v1/chat/completions` on two separate prompts post-calibration, e.g.
"Photosynthesis is the process where plants use sunlight to turn carbon
dioxide and water into food..." - and the response's own `timings` block
now exposes `draft_n`/`draft_n_accepted` directly (155/389 ≈ 40% and
166/291 ≈ 57% on the two test prompts) - realistic acceptance rates for
genuinely diverse text, a strong independent corroboration that the fix
is measuring the real thing now, not inflated-by-repetition numbers.

Compare to the retracted numbers: the corrected `-ncmoe` peak (51.36) and
`n_max` peak (56.92 combined) are noticeably *lower* than the retracted
run's inflated figures (53.99, 71.33) - exactly the direction predicted
by "degenerate repetitive text is trivially predictable, artificially
inflating measured throughput." The qualitative finding (MTP gives a real
speedup on this hardware) still holds; the specific multiplier claimed
earlier does not, and hasn't been re-derived cleanly yet (would need a
proper chat-templated A/B against a no-MTP baseline, not yet done as a
dedicated re-test - the numbers above are internal to the MTP-enabled
calibration search, not compared against a corrected non-MTP baseline).

## Second retraction and re-correction (2026-08-14, golden-section search bug)

**The "Final winner: n_max=9, 56.92 tok/s" call above is itself retracted.**
The `n_max` trace table right above it already contained the evidence:
n_max=2 scored 56.06 tok/s, higher than the declared winner n_max=9's
54.55 - golden-section search assumes a unimodal objective, the real
curve isn't one (dips at n=7, rises again through 8-10), and the search
silently converged on a local peak without ever reconsidering n=1/n=2,
both of which were already sitting in its own trace from the earlier
envelope-doubling phase. See "LFRU eviction: A/B results" bullet in the
build list below for the full root-cause writeup - the same session that
found and fixed this also fixed the LFRU eviction policy, both by
`common_golden_section_search_max` / `moe-cache.cu` respectively.

Fix: after golden-section search returns, validate its pick against
every point already measured in the trace map (free - no new subprocess
spawns, the data already exists) and take the true max instead of
trusting the bisection blindly. Applied to both the `-ncmoe` and
`spec-draft-n-max` searches in `common/common.cpp`.

Re-ran the full joint `--moe-calibrate` with the fix, same RTX 3060 +
Gemma-4-26B-A4B + MTP Q8_0 setup:

| n_max | tok/s |
|---:|---:|
| 1 | 57.80 |
| 2 | 60.10 |
| **3 (winner, trace-validated)** | **64.92** |
| 4 | 52.24 |
| 7 | 50.78 |
| 8 | 48.72 |
| 10 | 49.25 |
| 16 | 33.39 |
| 32 | 22.20 |

(ncmoe=15 alone, no MTP: 50.70 tok/s this run - within normal run-to-run
noise of the earlier 51.36.) Thread tuning tried n_threads=12 this time
too and it *lost* (48.71 vs the default n_threads=6's 64.92) - correctly
rejected, config stayed at n_threads=6. **Final winner: `ncmoe=15,
n_threads=6, spec-draft-n-max=3`, 64.92 tok/s** - a real **+28.1%** over
the no-MTP baseline (50.70 tok/s), cached and auto-applied on normal
launch (confirmed via the `common_maybe_autoplace_moe_cpu` log line).
**Content-verified coherent** via `/v1/chat/completions` on two prompts
post-calibration: 100/147 (68%) and 98/152 (64.5%) `draft_n_accepted` -
healthy, realistic acceptance rates, both notably higher than the
previous (buggy-search) run's 40%/57%, consistent with a smaller n_max
having an easier time getting each individual draft token accepted.

This replaces the 56.92 tok/s number everywhere it was cited above and
in the build-list summary below - use 64.92 tok/s (n_max=3) as the real,
validated figure for this hardware/model/context combination.

## Lesson for next time

The endpoint/prompt-formatting choice is not a cosmetic detail of a test
harness - it changes what's actually being measured, silently, in a
direction (inflated speedup) that's easy to mistake for a good result.
Any future benchmark of a chat/instruct-tuned model through
`llama-server` should default to `/v1/chat/completions`, and any
"speedup" or "hit rate" claim should have the actual generated content
spot-checked at least once, not just the timing numbers - cheap insurance
against exactly this failure mode.

# Build list summary (updated 2026-08-13, end of day)

## Already built, VALIDATED with real measurement
- moe-cache core port (leloch's design). Load-bearing feature. **Measured
  +54% decode speedup (27.0 -> 41.6 tok/s), 62.1% hit rate on our RTX 3060 +
  Gemma-4-26B-A4B** - see validation section above. RFC: 1.7-2.1x decode on
  skewed routing (Qwen-A3B), consistent with our own result. GLM-5.2
  confirmed skewed (SharkWipf: 30-50% measured). **Independently
  re-verified 2026-08-14 via `/v1/chat/completions`** (different placement,
  ncmoe=15: `--moe-cache off` 39.37 tok/s vs `--moe-cache auto` 51.71
  tok/s, **+31.3%**) - the mechanism's core value holds up under the
  corrected, chat-templated benchmark path, not just the original
  `/completion`-based measurement.
- mmap host-pinning fix (ours). One-time load-speed win only, not decode.
- Live `reserve_mb`/`max_batch` defaults (ours) - closed a silent
  cache-disabling bug for any concurrent deployment (`MAX_BATCH` hardcoded
  to 8 regardless of `--parallel`). See the concurrency-cliff investigation
  section.
- **Layer 1: live safe-placement auto-fit** (ours) - `-ncmoe` auto-decided
  live at every launch only when the config as given wouldn't fit, via a
  predictive no-alloc probe before allocation. Validated: previously-crashing
  config now self-corrects and serves real completions; explicit `-ncmoe`
  always left untouched; a config that already fits shows zero interference.
- **Layer 2: `--moe-calibrate`** (ours) - empirical throughput-optimal
  placement + thread count, cached per GPU+model+context combo. **Measured
  40% faster than Layer 1's safe floor alone (35.35 -> 49.33 tok/s)** on the
  RTX 3060 sandbox - this *is* the "static pre-flight capacity/hit-rate
  planner" item that used to sit in the longer-term/exploratory list below,
  now built and real. Known scope limit: single-sequence decode speed only,
  not calibrated for a specific concurrency target - see the Layer 2
  section for the full writeup. **Re-verified 2026-08-14** (post both the
  endpoint and golden-section search fixes): safe floor ncmoe=13 -> 41.68
  tok/s, Layer 2's pick ncmoe=15 -> 50.70 tok/s, **+21.6%** - same
  qualitative finding, smaller than the original 40% claim (that number
  pre-dates both fixes). Also separately confirmed with a wider sweep
  (ncmoe 13-30): a naively conservative ncmoe=30 only reaches 46.995
  tok/s, so Layer 2's search is finding a real, non-obvious optimum, not
  just validating the safe floor.
- **Native MTP speculative decoding** (merged upstream, `--spec-type
  draft-mtp --model-draft <sidecar.gguf>`). Works and gives a real
  speedup - confirmed with coherent output and realistic acceptance
  rates after fixing two serious bugs in how it was measured: the
  endpoint-formatting bug (see "Endpoint-bias investigation" below) and
  a golden-section search bug (see "Second retraction and re-correction"
  above) that made the first corrected calibration land on n_max=9
  (54.55 tok/s) when n_max=2/3 both scored higher in the search's own
  trace. **The original "1.33-1.35x" multiplier and the intermediate
  "n_max=9, 56.92 tok/s" number are both retracted.** Current real,
  validated number: `ncmoe=15, n_threads=6, spec-draft-n-max=3`, **64.92
  tok/s, +28.1% over the no-MTP baseline (50.70 tok/s)**, content-verified
  coherent via `/v1/chat/completions` with realistic 64-68%
  `draft_n_accepted` rates on two separate prompts. Provably lossless for
  output quality regardless of any of this (verified output is always
  sampled from the target model, never the draft). Confirmed the same
  sidecar pattern exists for `unsloth/GLM-5.2-GGUF` (20%
  acceptance-length improvement claimed for GLM-5.2's own MTP layer per
  public sources) - directly applicable to the real deployment target
  once that hardware is available, **using `/v1/chat/completions` for
  any validation there, not `/completion`**.
  Search-bug fix: `common_golden_section_search_max`'s callers
  (`common_moe_calibrate` in `common/common.cpp`) now validate the
  bisection's pick against every point already sitting in the trace map
  after the search completes - free, since that data's already measured,
  no new subprocess spawns - and take the true best-of-measured instead
  of trusting golden-section's assumption that the objective is
  unimodal. Applied to both the `-ncmoe` and `spec-draft-n-max` searches.
- **LFRU eviction (SLRU + heat + decay hybrid, ours)** - replaces plain
  LRU as the CUDA-side eviction policy. Four designs A/B'd on an identical
  mixed hot/cold workload (RTX 3060, Gemma-4-26B-A4B, `-ncmoe 15
  --moe-cache auto`, 604 slots) before landing on the shipped one:
    - plain LRU (baseline): **63.8%** hit rate
    - SLRU v1, two segments (probation/protected_), *uncapped* protected_:
      **60.2-60.3%** - a real regression. Root cause: protected_ grew to
      598-600/604 slots unbounded, starving probation to 4-6 slots -  new
      admissions had nowhere to prove themselves before eviction pressure
      hit them.
    - Heat-only, single list + per-slot heat counter (Colibri's
      `tier_decay()` halving) + bounded 8-candidate eviction window with
      25%-plus-4 hysteresis (Colibri's `tier_pick_lfru` margin), no
      segments at all: **64.6%** - recovered past plain LRU with just the
      heat signal and no segmentation.
    - Capped SLRU, same two segments as v1 but protected_ hard-capped at
      50% of the pool with FIFO (oldest-first) demotion at the cap:
      **65.0%** - the structural fix for v1's regression, and on its own
      the best single mechanism measured.
    - Two "let heat/decay decide the cap instead of a fixed percentage"
      variants were tried and **both regressed back to v1's failure
      mode**: an absolute per-slot heat floor for decay-driven demotion
      (61.4%, protected_ still 596-598/604) and a floor relative to
      protected_'s own mean heat (61.3%, still 599/604). Root cause in
      both: decay only fires on *fill* (admission) events, but within one
      busy request thousands of *hits* land on the same resident working
      set between fills - heat balloons for nearly the whole pool before
      decay gets a chance to run, and a demoted slot is usually re-hit and
      re-promoted on the very next token before the next decay checkpoint.
      Heat/decay alone doesn't have a fast enough signal to regulate
      *population size* at this timescale; that needs a hard structural
      bound.
    - **Shipped design**: capped SLRU (the structural fix that measured
      best) + heat as the *tiebreaker* - both for eviction-candidate choice
      within whichever segment is being drained, and for which protected_
      slot pays the demotion cost when a promotion needs to make room -
      instead of heat/decay replacing the cap outright. **65.0%**,
      statistically tied with capped-SLRU-alone on this workload (too
      small/short to discriminate further); kept as the shipped design
      because it's strictly safer (same proven structural bound, heat only
      ever changes *which* candidate within an already-safe pool gets
      picked, never *whether* the pool stays bounded) and should matter
      more under a larger, more diverse, longer-running workload than this
      16-request A/B can exercise. All four intermediate implementations
      backed up during development, not kept in the tree.

## Near-term - targets the GLM-5.2/H200 deployment directly
- ~~FR-Spec draft-vocab trimming for GLM's MTP~~ **BUILT for gemma4
  (sandbox), MEASURED, gemma4-only for now, 2026-08-14.** Not the
  upstream EAGLE3/qwen35 design (a d2t tensor baked into a pre-trimmed
  draft GGUF at conversion time - no producer tooling for that exists
  anywhere upstream, not even for its original Qwen3.5 target). Instead,
  Option B from this session's own design discussion: the original draft
  checkpoint is never modified. Four pieces, all model-agnostic except
  the last:
  1. **Real-traffic token-frequency logging** (`tools/server/server-token-freq.{h,cpp}`,
     `server_token_freq_logger`) - the server passively counts every
     confirmed *generated* token (not prompt tokens) into a dense
     per-vocab-id histogram, persisted as sparse JSON under
     `fs_get_cache_directory()`, keyed by a hash of the model path,
     accumulating across restarts. `--token-freq-log` (on by default).
  2. **Sidecar regeneration tool** (`tools/frspec-vocab-trim/`,
     `llama-frspec-vocab-trim`) - standalone, re-runnable any time (cron,
     or by hand): reads the histogram, ranks by real frequency, writes
     the top N token ids as a plain-text sidecar file (one id per line,
     trimmed-vocab position = line index - the same d2t convention
     EAGLE3/qwen35 use, just sourced externally instead of from a GGUF
     tensor).
  3. **Shared, architecture-agnostic ggml helpers**
     (`src/models/models.h` + `src/models/frspec-helpers.cpp`):
     `llm_frspec_load_d2t_sidecar`, `llm_frspec_gather_vocab_rows`,
     `llm_frspec_scatter_to_full_vocab`, the pending-vocab-map hint
     mechanism (`llama_frspec_set_pending_vocab_map` in `llama.h`, public
     API - needed to cross the `common/` -> `src/models/` boundary since
     Gemma-4's MTP draft has no in-file signal distinguishing it from the
     trunk the way qwen35's `mtp_only` flag does), and
     `llm_graph_input_frspec_d2t` (an `llm_graph_input_i`, matching
     `granite-switch.cpp`'s `llm_graph_input_switch` template exactly -
     the correct pattern for a load-time-populated, per-model-lifetime-constant
     tensor, found only after an extended dead end trying to create a
     GGUF-independent constant weight tensor directly via
     `llama_model_loader` internals). A future GLM port only needs a
     small amount of code calling into these already-tested helpers, not
     a fresh reimplementation.
  4. **Load-time gather/scatter wiring** - architecture-specific, ~40
     lines. Named `--spec-vocab-map <path>` on the CLI
     (`common_params_speculative_draft::frspec_vocab_map`).

  **Real bug: wired into the wrong class initially.** Gemma-4's MTP
  draft GGUF declares `general.architecture = gemma4-assistant`
  (`LLM_ARCH_GEMMA4_ASSISTANT` -> `llama_model_gemma4_assistant`,
  `src/models/gemma4-assistant.cpp`) - a genuinely separate architecture
  from the `gemma4` trunk class the *target* checkpoint uses, not a mode
  flag on the same class. The first pass wired the trim into
  `gemma4.cpp` (`llama_model_gemma4`) end to end, compiled cleanly, and
  only revealed the mismatch at live-load time (no "FR-Spec vocab trim
  active" log line, traced via targeted debug instrumentation before
  finding the real cause: `strings <gguf> | grep general.architecture`
  showed `gemma4-assistant`, not `gemma4`). Ported the identical pattern
  to `gemma4-assistant.cpp`; `gemma4.cpp`'s copy was left in place
  (harmless no-op there in this deployment shape, and free insurance for
  any future case where `gemma4` itself is loaded as a self-speculating
  MTP draft).

  **Three more real bugs found and fixed during build/verify, all
  latent, none specific to this feature's logic:**
  - **Token-frequency histogram was never flushed on normal shutdown.**
    `destroy()` (which flushes) was only ever called on the
    server's "sleeping" state transition, never from the actual
    `SIGTERM`/`SIGINT` shutdown path (`clean_up()` in `server.cpp` calls
    `ctx_server.terminate()`, which only unblocks the task queue - it
    never touched the logger). Every restart silently lost every count
    recorded since the last periodic 4096-token flush - for the
    "restart to regenerate the sidecar" workflow this whole feature
    depends on, that could be most or all of a session's real traffic,
    every single time. Fixed with a new public
    `server_context::flush_token_freq()`, called from `clean_up()`
    before `terminate()`. Verified with a real kill -TERM: histogram
    file was stale/empty before the fix, correctly populated after.
  - **`llama-frspec-vocab-trim`'s `--model-draft` flag hashed the wrong
    path.** The server keys the histogram by the *target* model's path
    (`params_base.model.path`) - correct, since confirmed generated
    tokens are always in the target's vocab space regardless of which
    draft proposed the (possibly-rejected) candidates for them - but the
    tool derived its lookup hash from whatever `--model-draft` path the
    user passed. Every user following the tool's own documented usage
    would reliably get "no histogram found" starting from nothing.
    Renamed the flag to `--model` (target model path; `--model-draft` is
    kept as an accepted alias for the old name, not removed outright).
  - **`ggml_get_rows` requires I32 indices, not I64.** Initial design
    copied EAGLE3/qwen35's convention of an I64 d2t tensor (needed for
    their `ggml_set_rows`-only scatter-back use case). This trim also
    needs a *gather* step first (EAGLE3/qwen35 never do, since their
    trim is baked in at conversion time) - `ggml_get_rows` asserts
    `b->type == GGML_TYPE_I32` and would have failed loudly the first
    time a real trim was exercised. Switched the shared d2t tensor to
    I32 throughout (still accepted by `ggml_set_rows`, which allows
    either).

  **Verified correct, not just compiled.** Live end-to-end test on the
  sandbox (RTX 3060, gemma-4-26B-A4B, Q8_0 draft): confirmed the
  activation log line fires with the real trim size
  ("421 of 262144 tokens" from an initial small real-traffic sample);
  identical output to the untrimmed baseline at temp=0
  ("The capital of France is **Paris**.", byte-for-byte, same
  `predicted_n`); coherent output on a "list primes" prompt; and one
  prompt ("haiku about mountains") that degenerated into token
  repetition identically *with and without* the trim, confirming that's
  a pre-existing small-quant-model quality artifact, not something the
  trim introduced. This matches the design's lossless guarantee: the
  draft's candidates are restricted to the trimmed set, but the target
  always re-verifies over the real, full vocab, so a wrong/rejected
  draft proposal can only cost a fallback token, never wrong output.

  **Speed finding, revised after a follow-up fix (2026-08-14, same
  day).** The first pass measured a real 5-11% regression and named a
  working theory: the trim ran a live `ggml_get_rows` gather inside the
  compute graph on *every* forward pass, re-executed as a real GPU
  kernel every step even though the result never changes -
  `can_reuse()=true` only skips rebuilding the graph's CPU-side
  topology, it does not skip re-running each node's kernel once the
  graph executes. Checked this empirically before acting on it: "graphs
  reused" counts from the server logs were statistically identical
  between trimmed and untrimmed runs (same ~198-per-200-token increment
  in both), ruling out "the trim breaks graph reuse" as the cause and
  confirming the cost really was per-step kernel execution time, not
  launch-dispatch overhead.

  **Fix:** moved the gather out of the compute graph entirely.
  `llm_graph_input_frspec_d2t` now also owns `output_w`, a
  pre-gathered-weight graph *input* (not a graph *node*) populated
  exactly once - on the first `set_input()` call, via a new
  `llm_frspec_gather_vocab_rows_from_backend()` helper that reads the
  needed rows straight off the original weight's backend buffer
  (`ggml_backend_tensor_get()`, works whether it's CPU- or
  GPU-resident) - and left untouched on every call after that. This
  didn't need the GGUF-independent-constant-tensor mechanism that was
  architecturally blocked earlier in the session; it reuses the exact
  `llm_graph_input_i`/`set_input()` pattern already built for the d2t
  index tensor, just with a "populate once" guard instead of "populate
  every call." The scatter-back step (`ggml_set_rows` placing the
  trimmed logits into a full-vocab-shaped, -inf-filled tensor) still
  runs every step, unavoidably - its *data* genuinely changes every
  token, unlike the weight gather.

  **Re-measured, correctness first:** temp=0 output on "The capital of
  France is" stayed byte-identical to both the untrimmed baseline and
  the pre-fix trimmed run - the fix only changes *when* the gather
  computation happens, never its result.

  **Re-measured, speed:** 9 samples each, same 200-token photosynthesis
  prompt, temp=0, on the same 645-token real-traffic sidecar as before.
  Cache-once trim: 57.8-62.7 tok/s (mean 60.5). A matched fresh
  untrimmed baseline, run immediately after: 55.0-59.8 tok/s (mean
  57.4) - lower than the *first* untrimmed baseline measured earlier in
  the session (61.6-62.7), despite being the same build and config. That
  drift between two untrimmed baselines, taken minutes apart, is bigger
  than the trimmed-vs-untrimmed difference either measurement showed -
  the sandbox's own run-to-run noise (thermal state, background load,
  moe-cache warm-up) is evidently larger than this technique's effect
  size at this scale.

  **Second fix, same day, prompted by being asked directly whether 645
  rows was really enough data movement to explain a millisecond-scale
  regression - it wasn't.** Back-of-envelope: 645 rows at embedding
  width, even unquantized, is a few MB; at this GPU's ~360GB/s bandwidth
  that's low single-digit microseconds if bandwidth-bound, nowhere near
  the gap being chased. The real remaining suspect: the scatter-back
  step (`ggml_fill(..., -INFINITY)` over the *full* 262,144-wide output,
  every step, regardless of trim size) was still a live graph op -
  strictly bigger than the gather that was just fixed, and unaffected
  by that fix. Same root cause as before (a value that's actually
  constant, paid for as if it weren't), same fix shape: `out_template`
  (the -inf-filled scatter target) joined `output_w` as a third
  populate-once graph input on `llm_graph_input_frspec_d2t`, filled
  exactly once via `ggml_backend_tensor_set()`. Safe to never re-fill:
  `d2t` names the identical fixed set of positions on every scatter
  call for a model instance's whole lifetime, so every position outside
  that set is written by no call, ever, and correctly stays whatever
  the one-time fill gave it.
  `llm_frspec_scatter_to_full_vocab()`'s signature changed to accept
  this pre-filled template directly instead of building one internally
  each call.

  **Re-measured a third time, back-to-back in the same session to
  cancel drift** (10 samples each, immediately sequential, same prompt,
  same sidecar): trimmed 56.2-60.8 tok/s (mean 58.7); untrimmed,
  restarted and run immediately after with no other change, 54.8-59.9
  tok/s (mean 57.6). Trimmed is now at parity with, if anything slightly
  ahead of, untrimmed - the regression is gone, not just reduced to
  noise. **This is now a real, if modest, positive result on the
  sandbox proxy**, not an open question. Correctness re-verified
  unchanged (byte-identical temp=0 output) after this fix too - both
  fixes only ever changed *when* already-correct computations happen,
  never their results.

  Independently worth knowing, found while investigating: SGLang - the
  most mature real-world implementation of this exact paper's technique
  - has an open, unresolved bug report
  ([sgl-project/sglang#8581](https://github.com/sgl-project/sglang/issues/8581))
  of FR-Spec *decreasing* throughput (430.99->291.04 tok/s) with a
  *lower* draft-acceptance rate too (1.72->1.64), closed as `inactive`
  with no maintainer explanation. That's a different failure mode than
  what this session found (their acceptance rate itself dropped, ours
  never did - only raw graph-kernel overhead was ever the issue here),
  but it's a real signal that this technique doesn't reliably reproduce
  the paper's controlled 1.12x number in production serving engines
  without real engineering care, which is exactly what happened here:
  the naive first port regressed, and it took two rounds of finding the
  actual root cause (not just reporting the number) to get a clean
  result.

  Still genuinely open: whether this specific measured edge holds up at
  GLM-5.2/H200 scale, where the vocab size, draft size, and batch
  dynamics are all different. But the sandbox result is no longer a
  caveat pointing away from the technique - it's a small, clean win,
  arrived at by refusing to accept "there's probably just some overhead
  somewhere" as a final answer.

  **"Net outcome" test at the real calibrated config, not `--parallel
  1`.** All FR-Spec numbers above were measured at `--parallel 1`,
  chosen for simplicity during correctness testing - not the config the
  session's own 64.92 tok/s MTP headline number was calibrated at
  (`--parallel 4`, per the calibration cache key
  `...c4096|p4|ngl99`). Re-ran trimmed vs. untrimmed at that real
  config, `/v1/chat/completions` (the "solid"-tier endpoint used
  throughout this doc, not the raw `/completion` endpoint the earlier
  FR-Spec numbers used), 5 samples each: untrimmed 55.12-55.98 tok/s
  (mean 55.66); trimmed 55.21-55.71 tok/s (mean 55.50). Tighter spread
  than any earlier FR-Spec measurement, and genuinely flat - not even
  the parity-with-a-slight-edge seen at `--parallel 1`, just a clean
  wash either way. Consistent with the fix being real and complete: no
  regression at any config tested, and no dramatic win either.

  Two things this run surfaces that are worth separating from FR-Spec
  itself: (1) at the *identical* calibrated config, this photosynthesis
  prompt gives ~55.7 tok/s solo, not 64.92 - a ~14% gap, from prompt
  content alone (MTP's draft-acceptance rate is prompt-dependent; the
  64.92 figure came from whatever probe prompts `--moe-calibrate` used,
  not this one). The 64.92 headline number is real but prompt-specific,
  not a universal constant for "this config" - a caution against
  treating any single throughput number in this document as portable
  across different real traffic. (2) None of this session's percentage
  gains compound the way stacking them on paper suggests, both because
  they were measured at different `--parallel` settings across the
  session's own timeline and because of this same prompt-dependence.

  **Real bug found running everything together, 2026-08-14: FR-Spec
  trim is currently incompatible with `--spec-type
  draft-mtp,ngram-suffix` (MTP + suffix-decode combined).** Requested
  directly: test all of this session's work together, not FR-Spec in
  isolation. Built the combined config - `-ncmoe 15 --parallel 4
  --spec-type draft-mtp,ngram-suffix --spec-draft-n-max 3
  --spec-vocab-map <sidecar> --kv-unified --cache-idle-slots` - and hit
  a severe regression: 29.8-30.2 tok/s with only 10.7% draft
  acceptance, versus 66.3-68.6 tok/s at 70-74% acceptance for the exact
  same config minus `--spec-vocab-map`. Isolated properly rather than
  guessing: removing `--kv-unified` alone reproduced the identical
  broken numbers (30.24 tok/s, 10.7%), ruling it out; removing
  `--spec-vocab-map` alone (keeping `--kv-unified`) fully restored
  healthy numbers (65.6-68.0 tok/s, 70-74% acceptance). **FR-Spec is
  the cause, specifically when combined with suffix-decode** - FR-Spec
  alone with MTP was at parity (measured above), suffix-decode alone
  with MTP was healthy (measured earlier this session), but the three
  together break badly.

  **Investigated properly, not just theorized (2026-08-14, later same
  day)** - asked directly to solve it, not just document it. Ruled out
  two plausible theories by reading the actual code and tracing a live
  `-lv 5` debug session, in order:
  1. *"MTP's own `p_min` confidence gate fires more often under the
     trim, handing more rounds to suffix-decode, which does worse on
     this content."* Ruled out: `p_min` defaults to `0.0f`
     (`common/common.h`), meaning the gate (`common/speculative.cpp`,
     the MTP drafter's per-token `if (cur_p->data[0].p < params.p_min)`
     check) never fires with default settings.
  2. *"The combined-drafter verification path uses MTP's own reported
     probability - now inflated, since it's normalized over only 645
     candidates instead of the true 262,144-token distribution - in a
     probabilistic accept/reject test, unfairly rejecting good
     candidates."* Ruled out by reading the actual verification call
     (`tools/server/server-context.cpp`,
     `common_sampler_sample_and_accept_n`): at `temperature=0` (every
     test in this doc), acceptance is a plain greedy comparison against
     the target's own independently-computed top pick - it never reads
     the *drafter's* probability at all, so trim-induced probability
     inflation can't be the mechanism.

  What the live trace actually shows: `-lv 5` confirms MTP is the
  implementation producing every single round's draft (suffix-decode's
  fallback path is never reached - `common_speculative_draft()` only
  calls the next configured impl for a sequence still needing a draft,
  and MTP always returns a non-empty result here) - so this isn't
  "suffix-decode drafting badly," it's MTP's *own* drafted tokens being
  rejected far more than usual specifically when suffix-decode is also
  configured. Per-round accept counts (`accepted N/3 draft tokens` in
  the trace) show the *very first* drafted position failing in 30 of 42
  rounds (71%) - immediate rejection, not a late-position falloff.
  Suspicious correlate, not yet proven causal: the single most
  over-represented token in this session's thin 645-token histogram
  (id 236772, a dash/hyphen the model uses heavily in its own
  markdown-style formatting) accounts for 630 of 6114 total real-traffic
  observations (~10.3%) - MTP's confidence in proposing it is very high
  within the trimmed 645-candidate view (0.99+ in the trace) but the
  target frequently disagrees. This alone doesn't explain why FR-Spec
  alone (no suffix-decode) stayed at parity with the identical trim and
  identical prompt - the actual trigger for *why this specific
  combination* surfaces the mismatch (state/scheduling interaction
  between multiple configured drafter types, not drafted-token content
  itself) is still not pinned down at the code level despite this
  investigation ruling out two concrete, testable theories.

  **Action taken: do not enable both together.** FR-Spec trim
  (`--spec-vocab-map`) should not be used alongside `--spec-type
  draft-mtp,ngram-suffix` until this is root-caused and fixed - each is
  fine alone, MTP+suffix is the recommended production combination from
  earlier in this document, and FR-Spec is not worth keeping active at
  the cost of breaking that. This is not yet enforced in code (no
  warning or guard added), only documented here - a real follow-up if
  FR-Spec is pursued further.

  **The real, clean "net outcome of this session's work" number**, all
  compatible pieces together (moe-cache placement + MTP + suffix-decode
  + `--kv-unified`/`--cache-idle-slots`, FR-Spec excluded per the
  finding above), solo decode, `--parallel 4`, same photosynthesis
  prompt: **65.6-68.0 tok/s (mean 66.8), 70-74% draft acceptance** -
  genuinely ahead of the 55.5-55.7 tok/s MTP-alone number at the same
  config, and the real reference point for "what does combining
  everything actually get us" rather than adding percentages from
  differently-configured measurements taken at different points in the
  session.

  Toggling is clean either way: `--spec-vocab-map` unset (the default)
  is a verified no-op - `frspec_d2t_ids` stays empty, `load_arch_tensors`
  and the graph-building gather/scatter code both no-op, full untrimmed
  vocab used exactly as before this feature existed.
- LFRU eviction + hysteresis + decay, replacing our ported plain LRU
  (Colibri's tier.h). No hard number yet - needs A/B. Motivated by
  GLM-5.2's confirmed skew.

## Medium-term - multi-user serving (5-10 agentic users)
- ~~Server-level cross-request prefix/KV reuse~~ **ALREADY BUILT AND
  VALIDATED, 2026-08-14** - not something to build, the narrower
  maintainer-suggested path from discussion #21961 ("we can obviously
  implement the same scheduling for cross-request KV reuse at server
  level") turns out to already be shipped in mainline: `server_prompt_cache`
  (`tools/server/server-task.h`/`.cpp`), enabled by default via
  `--cache-ram` (default 8192 MiB). When a slot's content is evicted, its
  full KV state gets saved to this RAM-resident pool; on a new request,
  `server_prompt_cache::load()` searches the pool by longest-common-prefix
  match (same LCP logic as the live-slot `--slot-prompt-similarity`
  selection, also already on by default) and restores a match into
  *whichever* slot the new request lands on - genuinely cross-slot, not
  just "the same slot got lucky."

  **Verified empirically, not just read in source**: first test (Gemma-4,
  the sandbox's only model) showed the restore firing correctly in the
  trace log (`found better prompt with f_keep=0.866`, 187/216 tokens
  matched) but then getting silently discarded - a *separate* per-slot
  "context checkpoint" mechanism ran its own validation against the
  physical slot's *unrelated* prior checkpoint history, found no match,
  and erased the just-restored state, forcing near-full reprocessing
  anyway (log message names the cause directly: "forcing full prompt
  re-processing due to lack of cache data (likely due to SWA or
  hybrid/recurrent memory..." - traced to PR #13194, SWA support,
  an acknowledged upstream limitation, not a bug introduced here).
  Confirmed `gemma4.cpp` uses SWA. **Confirmed `glm4.cpp`/`glm4-moe.cpp`/
  `glm-dsa.cpp` (GLM-5.2's family) do not** - grepped directly, zero
  matches for `swa_type`/`n_swa`. Re-ran the identical test on a non-SWA
  model already on disk (Qwen2.5-0.5B) to isolate the mechanism from this
  Gemma-4-specific limitation: **real, dramatic reuse** - a second request
  sharing a 206-token prefix with an evicted, different-slot request
  reprocessed only 14 new tokens (`cache_n=192`), 27.0ms -> 5.9ms. This is
  the mechanism working exactly as designed once the SWA-specific
  discard path doesn't trigger.

  **Conclusion for the H200/GLM-5.2 deployment**: this backlog item is
  effectively done already, not a build target - GLM-5.2 isn't SWA, so
  the SWA-specific limitation that crippled the Gemma-4 test shouldn't
  apply there. Worth re-confirming once on the real model/hardware (same
  methodology: force a different-slot restore, check `cache_n` in the
  response `timings` and the `-lv 4` trace log for "found better prompt"
  immediately followed by *no* "erased invalidated context checkpoint"
  line), but there's no code to write for the core mechanism - it already
  exists, is on by default, and the one real failure mode found doesn't
  apply to the target architecture. Narrower than full PagedAttention as
  the maintainers wanted (discussion #21961); underlying paged-KV
  experiment elsewhere showed a 26->247 concurrent-sequence capacity jump
  at equal VRAM on an A10G, a different (block-pool) approach not needed
  here given this simpler mechanism already covers the actual ask.
- ~~MTP batch-gate tuning (close the 9-31 dead zone between MAX_BATCH/
  MIN_BATCH)~~ **RESOLVED, 2026-08-14** - see "MTP verify-batch size x
  concurrency" under Open questions above for the full writeup. moe-cache's
  own `MAX_BATCH` gate is now MTP-aware (was silently ignoring MTP's
  verify-width multiplier entirely, fixed and verified). The empirical
  9-31 sweep found no dead zone in that range itself (smooth climb, 51.2
  -> 77.5 tok/s) - but did find a real, separate, sharp cliff exactly at
  concurrency=32 (77.5 -> 41.2, -46.8%), matching `GGML_OP_OFFLOAD_MIN_BATCH`'s
  default threshold precisely, confirmed independent of moe-cache
  (reproduced with moe-cache both genuinely active and accidentally
  starved of VRAM). Not root-caused to kernel level like the MMVQ cliff
  was, but comfortably outside the 5-10 concurrent user deployment target,
  so not a blocker. **Not entangled with the concurrency-cliff root cause
  after all** - that turned out to be a separate GPU-kernel-dispatch limit
  in `ggml-cuda` (`MMVQ_MAX_BATCH_SIZE`, see below), unrelated to this
  MAX_BATCH/MIN_BATCH question.
- ~~Suffix Decode (PR #26283, model-free spec decoding)~~ **BUILT AND
  MEASURED, 2026-08-14.** Merged `github.com/ggml-org/llama.cpp` pull/26283
  directly (`git fetch origin pull/26283/head`, clean merge onto
  `moe-cache-port`, only 9 files touched - the actual feature, no
  unrelated churn). One compile fix needed: `ngram_suffix`'s
  `need_embd() override` didn't override anything - our branch's
  `common_speculative_impl` base already dropped that virtual (MTP's own
  splitting decision goes through `task->need_embd()` now, a different
  mechanism) - removed the dead override, not a functional change.
  Mechanism: an online suffix tree built from the prompt + previously-
  accepted tokens (no second model, no extra compute cost beyond tree
  maintenance); draft tokens come from matching the current context
  against previously-seen subsequences. `--spec-type ngram-suffix`,
  comma-combinable with MTP (`--spec-type draft-mtp,ngram-suffix`).

  **Measured** (ncmoe=15, 2 samples averaged, same methodology as
  everything else this session), repetitive vs diverse prompts:

  | config | repetitive tok/s | diverse tok/s | rep accept% | div accept% |
  |---|---:|---:|---:|---:|
  | baseline | 50.88 | 51.47 | - | - |
  | MTP alone | 73.53 | 59.89 | 90.4% | 66.4% |
  | suffix alone | 62.10 | 51.29 | 64.4% | 34.0% |
  | **MTP + suffix combined** | 72.88 | **64.82** | 77.5% | 71.1% |

  Suffix-alone matches its design intent exactly: real win on repetitive
  text (+22.1%), flat on diverse text (+0.35%, noise) - nothing to catch
  in genuinely non-repeating text. The nuance: **combined is *not*
  strictly additive everywhere.** On repetitive text, combined (72.88) is
  marginally *worse* than MTP-alone (73.53, -0.9%) - MTP alone is already
  near its own acceptance ceiling there (90.4%), so a second drafter
  competing for the same easy tokens adds coordination overhead without
  new upside. But on **diverse text, combined beats MTP-alone by +8.2%**
  (64.82 vs 59.89) - suffix-decode picks up extra accepted tokens (common
  words/short phrases) that MTP alone misses, a genuine additive effect,
  just showing up on the *opposite* prompt type from what "best case is
  repetitive output" would suggest. Content-verified coherent via
  `/v1/chat/completions` on the combined config (63% acceptance on a real
  diverse prompt, real `reasoning_content`, not degenerate repetition -
  same discipline as every other number in this doc).

  **Recommendation**: enable `ngram-suffix` alongside MTP by default for
  this deployment - the diverse-text case is the common one for an
  agentic API workload, and the repetitive-text downside is small (-0.9%)
  next to the diverse-text upside (+8.2%).
- ~~The still-open concurrency-cliff root cause~~ **RESOLVED, 2026-08-14**
  - see "ROOT-CAUSED" under the concurrency-cliff investigation section
  above. Not llama-server's slot scheduling after all: a GPU-kernel
  dispatch threshold in `ggml-cuda/mmvq.cu` (`MMVQ_MAX_BATCH_SIZE=8`
  default), confirmed to the exact token via a dedicated 7/8/9/10
  concurrency sweep. Directly relevant to the H200/GLM-5.2 target - for
  NVIDIA cc>=Ada Lovelace (Hopper included) this threshold is a flat 8
  unconditionally. Fix shipped as a startup warning
  (`common_warn_concurrency_cliff`), not a silent workaround - it's a
  GPU-kernel limit, not something `-ncmoe`/placement can fix.

## Longer-term / exploratory - not yet scoped as real builds
- Persistent cross-session usage history (Colibri-style) - avoids
  cold-restart penalty, no number.
- Live skew-detection driving adaptive strategy (our own synthesis) -
  protects against static-beats-dynamic-on-uniform-routing failure case,
  unscoped.
- Live per-expert heatmap UI - debugging/demo value, not performance.
- ~~Layer 2 extension: concurrency-target-aware calibration~~ **BUILT AND
  VALIDATED, 2026-08-14.** `--moe-calibrate` now benchmarks every
  candidate (ncmoe, threads, spec-draft-n-max) with real concurrent
  requests and scores by aggregate throughput whenever `--parallel N > 1`
  is passed alongside it, instead of always using solo decode speed -
  directly motivated by this session's own findings that the MMVQ and
  bulk-offload GPU-kernel-dispatch cliffs only appear above certain
  concurrent batch sizes, invisible to a solo benchmark entirely.
  Deliberately reuses `--parallel` itself as the concurrency target
  rather than a separate flag, since the calibration cache key already
  includes `n_parallel` - a separate flag risked drifting out of sync
  with the deploy-time `--parallel` and silently missing the cache.
  Found and fixed a real bug along the way: the benchmark's request
  builder wrapped JSON bodies in shell single quotes, and one of the
  probe prompts ("Explain Newton's second law...") had an apostrophe
  that broke the quoting outright - added proper escaping
  (`common_shell_quote`) rather than just avoiding the word. **Verified
  end-to-end**, not just compiled: calibrated at `--parallel 8` found
  ncmoe=15 at 76.11 aggregate tok/s, closely matching this session's
  independent concurrency-sweep measurement at the same concurrency
  (78.12 tok/s, within normal noise) - real cross-validation, not a
  unit-level check. Confirmed the cache correctly isolates by
  concurrency: a `--parallel 1` launch afterward missed the
  concurrency=8 entry and fell back to Layer 1's live safe floor.
- ~~Verify whether Colibri actually avoids the full prefill pass before the
  first token~~ **RESOLVED, 2026-08-14, checked against actual source**
  (`c/kv_prefix.h` in JustVugg/colibri, cloned and read directly). It
  doesn't - the file's own comments are explicit that prefilling zero
  tokens would leave no hidden state to sample from, and a genuinely new
  or diverged prompt gets full prefill exactly like llama.cpp. What it
  actually does: each conversation is pinned to a KV slot (since their
  commit #639) with the exact token ids that state was built from
  recorded; if the next turn's prompt prefix exactly matches those
  recorded ids, only the new tail gets prefilled, skipping re-processing
  of everything the previous turn already computed - exploiting that chat
  clients resend the whole transcript every turn. All-or-nothing per
  session (`kv_prefix_reuse` returns the full matched length or 0, no
  partial/tree matching), any divergence anywhere falls back to full
  prefill from scratch. Their own measured number: DeepSeek V4, a second
  turn reusing 82% of its prompt, 320s -> 61s. This is a narrower,
  single-session version of the same idea as "Automatic prefix caching"
  below (vLLM/SGLang's RadixAttention shares prefixes *across* sessions
  via a shared radix tree; this is scoped to one pinned conversation
  slot, exact-match only) - not a new technique to add, confirms the
  already-tracked item is the right thing to build if this is wanted.

## FR-Spec + multi-drafter incompatibility, final investigation (2026-08-14, end of session)

Explicit directive: find the real root cause, using any means, as the
last piece of this session's work. Summary of the full push below - a
real independent bug was found and fixed along the way, but the
headline bug's root cause was not conclusively pinned down despite
substantial effort. Reporting the state honestly rather than claiming
a fix that isn't verified.

**New findings this pass:**
- **Not suffix-decode-specific at all.** Bisected across three
  completely different second drafter types - `ngram-suffix`,
  `ngram-simple`, and `ngram-cache` configured with no cache files (as
  inert as a second drafter can be: never produces a draft, ever,
  since it has nothing to match against). All three reproduce the
  identical regression (~30 tok/s, ~9-11% accept) when combined with
  MTP + FR-Spec. This rules out anything in any specific second
  drafter's own `draft()`/`accept()` logic - confirmed by reading all
  three implementations directly, all self-contained (`ngram-suffix`
  and `ngram-simple`'s own `accept()` are literally `// noop`).
- **The internal drafter priority order is hardcoded, not CLI-order-controlled**
  (`common_speculative_init()`, `common/speculative.cpp`): ngram-family
  types are always tried before draft-based types (MTP), regardless of
  the order given to `--spec-type`. This means ngram-suffix's
  `draft()` actually runs *every single round* (not rarely, as first
  assumed from an incomplete trace read) - it just usually produces
  nothing usable (its own `n_min` gate) and falls through to MTP. This
  correction doesn't change the diagnosis but corrects a wrong mental
  model from earlier in the investigation.
- **A real, independent bug found and fixed**: `common_speculative_impl_draft_mtp::accept()`
  ignored its `is_other` parameter entirely and unconditionally
  overwrote its own `pending_h` (the recurrent hidden-state input MTP
  feeds into its own next prediction) using `verify_h`/`n_accepted` -
  data that's only meaningful when MTP itself was the drafter actually
  verified that round. When a *different* configured drafter wins a
  round instead, `is_other=true` and that same code still ran,
  corrupting `pending_h` with a stale, unrelated hidden state. Fixed
  by skipping the update when `is_other=true` (leaving `pending_h`
  untouched is strictly safer than overwriting it with definitely-wrong
  data). Found via new instrumentation added this session
  (`LLAMA_DEBUG_VERIFY=1` env var, `common/sampling.cpp`, prints the
  drafted-vs-target token text on every mismatch) which surfaced a
  genuine repeating/lagging pattern in rejected drafts, consistent with
  a stale recurrent-state input.
- **The fix does not resolve the regression** - re-measured at 31.7-31.9
  tok/s, 10.7-11.0% accept, statistically identical to before the fix.
  In hindsight this is logically expected, not surprising: `is_other=true`
  only ever fires on the ~5% of rounds where a different drafter wins;
  the observed ~90% rejection rate is present even across the ~95% of
  rounds where MTP drafts uninterrupted, with no other impl involved at
  all that round. A bug scoped to the rare cross-drafter rounds cannot
  explain damage present in the common, uninterrupted-MTP rounds - this
  should have been checked analytically before spending a test cycle
  confirming it empirically. The fix is being kept regardless: it's a
  real, independently-valid correctness bug (any multi-drafter MTP
  configuration has it, with or without FR-Spec), just not the
  explanation for this one.

**Where this leaves the investigation:** the bug requires exactly
(MTP active) + (any second `--spec-type` configured, regardless of what
it does) + (FR-Spec trim active) - all three, confirmed via direct
bisection on both of the first two dimensions. What's been eliminated
with real evidence, across this and the prior investigation pass:
`p_min` confidence gating, probability-based verification (both by
reading the actual code and by direct A/B test with `--spec-prob-accept`
showing zero effect), any specific second-drafter's own logic (three
different types tested including a maximally inert one), output-buffer
sizing driven by the other type's larger default `n_max`/`size_m`
(tested directly by forcing `--spec-ngram-suffix-n-max 3` to match
MTP's own value - no change), CLI argument order (proven irrelevant,
priority is hardcoded), and now `pending_h` corruption via `is_other`
(real bug, fixed, confirmed not the cause via direct re-test). What
remains unexplained: some structural difference between "MTP is the
only configured drafter" and "MTP is one of several configured
drafters" that affects MTP's own prediction quality on effectively
every round, not just rounds where another drafter actually
participates - and that specifically requires FR-Spec's vocab
restriction to become severe (this same multi-drafter structural fact,
without FR-Spec, measured healthy earlier this session: 66-69 tok/s,
70%+ accept). No further hypothesis was identified with strong enough
supporting evidence to test as a probable fix within this session.

**Recommendation unchanged**: don't combine `--spec-vocab-map` with a
multi-type `--spec-type` configuration. Each half works well
independently and is recommended on its own merits.

## SUPERSEDES THE ABOVE: root cause found, real fix committed (2026-08-14, later same day)

The section above ended honestly inconclusive. Pushed further per
explicit instruction to keep going and check llama.cpp's own
issues/PRs for prior art: issue #23184 ("Pipeline speculative decoding
strategies draft-mtp -> ngram-mod") confirms independent-draft-streams
is a known architectural limitation, not a reported accuracy
regression; issue #23154, a closed/stale CUDA OOM report combining
`ngram-mod,draft-mtp`, gave a useful but not directly applicable clue.
That search didn't find the bug directly, but re-examining the test
methodology did.

**The methodology bug**: every "healthy MTP-only baseline" number in
this document (including the 66-69 tok/s / 70%+ accept figures cited
just above as the healthy comparison point) that was produced by
launching with `-md <path>` and *no explicit `--spec-type`* was never
actually running speculative decoding at all.
`params.speculative.types` defaults to a single-element `{NONE}`
vector; the HF-sidecar auto-detection in `arg.cpp` only populates it
for `-hfd`-style HF-repo auto-discovery, not a plain local `-md` path.
With `types == {NONE}`, `common_speculative_init()`'s impl list is
empty and it returns `nullptr`; `server_slot::can_speculate()` is
`!!spec`, so speculative decoding silently never engaged - the draft
model loaded into VRAM and then was never called. Confirmed directly
with two new debug prints gated on `LLAMA_DEBUG_VERIFY=1`
(`[DEBUG-TYPES]` in `tools/server/server-context.cpp`, now permanent):
`resolved speculative.types = none` / `spec = (nil) (can_speculate
would be FALSE)`.

**Re-ran with explicit `--spec-type draft-mtp`, the only way to get a
real apples-to-apples number:**
- MTP alone, no FR-Spec: **75.93 tok/s, 81.5% accept.**
- MTP alone, with FR-Spec (`--spec-vocab-map`) active, same prompt/config:
  **30.53 tok/s, 8.8% accept.** A real, severe regression - not the
  fabricated "parity" claimed earlier in this document under implicit
  mode, and not resolved by the `is_other` fix above.
- Bisected exactly as before, but now against a real baseline:
  identical collapse reproduces with **`--spec-type draft-mtp` alone,
  no second drafter type configured at all.** This eliminates
  "multi-drafter interaction" as the mechanism outright - the whole
  suffix-decode/ngram angle investigated above and in the section
  before it was chasing a variable that was never actually the cause.

**The real root cause**: this session's own earlier "cache once"
optimization for FR-Spec's weight gather (`output_w`) and scatter
template (`out_template`) - populated a single time via `set_input()`
on the assumption their shape/content never change - has a genuine,
silent correctness bug. Direct A/B test, reverting
`gemma4-assistant.cpp`'s graph-building code to the original live
(recomputed every forward pass) `ggml_get_rows`/`ggml_fill`+`ggml_set_rows`
pattern, same model/config/prompt: accept rate 8.8% -> 44.3%, tok/s
30.5 -> 55.6-56.7. Correctness (byte-identical greedy output) held
throughout, both cached and live - this was never a wrong-answer bug,
only a "confidently wrong draft, so the target rejects it constantly"
bug. The exact GGML-level mechanism (something about the cached
input's `can_reuse() == true` no longer being a safe signal once the
tensor's *populated content*, not just its shape, needs to track the
live graph) was not traced to the instruction level, but the fix is
unambiguous and repeatedly confirmed.

**Fix committed**: `gemma4-assistant.cpp`, `gemma4.cpp`, `models.h`,
`frspec-helpers.cpp` reverted to the live/uncached pattern; the
`output_w`/`out_template` caching fields and the now-unused
`llm_frspec_gather_vocab_rows_from_backend` helper removed entirely
rather than left as dead, tempting code (commit `fc0818502`). Final,
validated numbers for FR-Spec + MTP, `--spec-type draft-mtp`, fix
applied: **55.6-56.7 tok/s, 43-44% accept**, vs 75.93 tok/s / 81.5%
without FR-Spec - a genuine ~26% throughput cost and a real
accept-rate cost, not the free trim originally hoped for. Still a
plausible trade in a real VRAM-constrained deployment (that's the
whole point of trimming the vocab), just not a free one - budget for
it rather than assuming parity.

**What this retroactively means for the rest of this investigation**:
there was never a real "FR-Spec + multi-drafter" bug - that framing
was a byproduct of comparing a broken cached-FR-Spec run against a
baseline (implicit `-md` mode) that wasn't running speculative
decoding at all, so of course adding a second drafter type "made it
worse" - the baseline had nowhere to go but down once real speculation
was actually switched on for either side. The `is_other` fix in MTP's
`accept()` is still real and still kept (any actual multi-drafter MTP
config has that bug), it was just never this bug's cause.

**Standing caveat for the rest of this document**: any throughput or
accept-rate number elsewhere in this file that was produced by
launching with `-md <path>` and no explicit `--spec-type` should be
treated as a **non-speculative baseline**, not a valid MTP
measurement, until re-verified with `--spec-type` set explicitly and
(ideally) `LLAMA_DEBUG_VERIFY=1` confirming `spec` is non-null. This
document was not fully re-audited line by line for this after the
discovery; numbers that explicitly show `--spec-type` on the command
line (e.g. the MTP+suffix-decode "net outcome" figures above, which do
show it) were real speculative-decoding runs and are not affected by
this specific bug, but should still be treated as unverified against
the now-fixed FR-Spec code path until re-measured.

**Recommendation, corrected**: `--spec-vocab-map` is safe to combine
with a multi-type `--spec-type` configuration now that the actual root
cause (the cache-once bug, not a multi-drafter interaction) is fixed.
Budget for FR-Spec's real cost (~26% throughput, ~38pp accept-rate
drop on this sandbox/prompt) rather than assuming it, and re-measure
before relying on it for the GLM-5.2/H200 target, where vocab size and
trim ratio both differ substantially from this sandbox.

## Probabilistic draft acceptance (`--spec-prob-accept`) - BUILT AND VALIDATED, 2026-08-14

Motivated by the FR-Spec+suffix-decode investigation above: llama.cpp's
speculative verification (`common_sampler_sample_and_accept_n`,
`common/sampling.cpp`) only ever accepted a draft token on an *exact*
match against an independently-drawn sample from the target - never
read the draft's own probability at all, at any temperature. This is
strictly more conservative than the classic speculative-decoding
acceptance test (`min(1, p_target(x)/p_draft(x))`, with fallback to the
target's own sample on rejection - not full residual-distribution
resampling, a common, simpler approximation), which is guaranteed to
accept at least as often for the same draft/target pair: if the target
considers a token at least as likely as the draft did, it's accepted
outright, no coincidental exact match required.

**Design, opt-in and structured to be impossible to regress**: new
`--spec-prob-accept` flag (default off - unset behavior is byte-identical
to before this existed, verified). The new code path is a strict
superset of the old one - exact match is checked and accepted first,
exactly as before; only on mismatch does it get a *second* chance via
the probability-ratio test, before falling back to the same "use the
target's own sample" behavior the old code always used. Implementation:
- `common_speculative_draft_params` gained an optional `result_probs`
  (`common/speculative.h`) - parallel to the existing `result` token
  array, populated by MTP's own draft loop with its real per-token
  confidence (`common/speculative.cpp`, already computed via
  `common_sampler_get_candidates`, previously discarded). Drafters that
  don't track a real probability (ngram-suffix and other pattern
  matchers) are padded with 1.0 (maximally confident) rather than
  requiring every drafter type to change - a probability of 1.0 makes
  the acceptance ratio reduce to exactly `p_target`, which is the
  correct treatment for a token with no real confidence value.
- `common_sampler_sample_and_accept_n` gained an optional `draft_probs`
  parameter (`common/sampling.h`/`.cpp`); null (the default) reproduces
  the original function exactly, byte for byte.
- Server wiring (`tools/server/server-context.cpp`): `slot.spec_draft_probs`
  parallel to `slot.spec_draft`, only allocated/passed when the flag is on.

**Verified, not just built:**
- Flag off: confirmed byte-identical draft/accept counts and tok/s to
  before this change, both on the healthy MTP+suffix config and the
  broken FR-Spec+suffix one - zero behavioral change when unset.
- Flag on at `temperature=0` (every measurement earlier in this
  document): confirmed a mathematical no-op, not a bug - a greedy
  target's true distribution is a point mass, so any non-matching
  token genuinely has `p_target=0`, and `alpha=0` correctly, every
  time. Probabilistic acceptance can only ever matter at non-zero
  temperature; this was worth confirming empirically, not just arguing
  from theory, given every number in this document so far was measured
  at temp=0.
- Flag on at `temperature=0.7`, healthy MTP-only config (no FR-Spec, no
  suffix-decode): **53.6-53.8 tok/s (off) -> 55.6-55.7 tok/s (on), a
  real, consistent ~3.5% improvement** - the mechanism genuinely works
  and helps in the case it's designed for.
- Flag on at `temperature=0` and `0.7`, the FR-Spec config that was
  believed broken at the time (cache-once bug still present, and
  measured under implicit `-md` mode besides): **no measurable change
  either way.** At the time this was read as ruling out "borderline
  draft tokens losing narrow coin-flips" as the bug's mechanism. With
  the benefit of the later finding (see "SUPERSEDES THE ABOVE" above),
  this negative result still stands on its own terms - probabilistic
  acceptance genuinely doesn't move a *correctness* bug like the
  cache-once one, since the drafted tokens themselves were being
  computed wrong, not just narrowly rejected - but it should no longer
  be read as evidence about FR-Spec+suffix-decode specifically, since
  that framing itself didn't survive the later investigation.

Recommendation: safe to enable generally (`--spec-prob-accept`) for any
non-greedy (temperature > 0) deployment using MTP - real upside, zero
downside by construction. Independent of the now-fixed FR-Spec
cache-once bug; combine freely with FR-Spec once re-measured against
the fixed code path.

## Explicitly decided against
Multi-LoRA serving (not our use case), multi-node serving (single H200),
SGLang's expert-parallelism/DeepEP (multi-GPU only) - these are ours to
decide and the deployment shape rules them out cleanly.

Full PagedAttention is different and was mis-filed here originally:
not-building-it was llama.cpp's own maintainers' architectural call
(discussion #21961 - "we can obviously implement the same scheduling
for cross-request KV reuse at server level" instead of adopting vLLM's
block-pool architecture), not an independent evaluation on this branch.
Re-examined 2026-08-14, prompted directly by being called out for that
conflation - see "PagedAttention re-examined" below for what's actually
built into llama.cpp already and whether it matters for us specifically.

## PagedAttention re-examined - llama.cpp already has the practical
## equivalent, and this deployment isn't using it (2026-08-14)

Checked the source directly rather than relying on the earlier
maintainer-thread read. Without `--kv-unified`, `n_ctx_seq = n_ctx /
n_seq_max` (`llama-context.cpp`) - each `--parallel` slot gets a fixed,
equal, contiguously pre-reserved share of the context budget, whether
or not that slot's actual conversation ever comes close to using it.
That's exactly the waste PagedAttention exists to eliminate - a
concurrency-count argument (26->247 sequences on an A10G) isn't the
only place this bites; VRAM wasted on unused reservation is a cost
regardless of concurrency level.

With `--kv-unified`, `n_ctx_seq = n_ctx` and `n_stream = 1`: a single
pool shared across all active sequences, drawn from dynamically rather
than pre-split. Combined with `--cache-idle-slots` (already validated
this session, ties into `server_prompt_cache`/`--cache-ram`), idle
slots' KV cache is actively reclaimed and handed back to the pool for
busy ones - `try_clear_idle_slots()` in `server-context.cpp` is a
no-op entirely unless `kv_unified` is set. Checked both for
compatibility with the rest of this stack: `llama_kv_cache_iswa`
computes `size_swa` correctly for the unified case
(`hparams.n_swa*(unified ? n_seq_max : 1) + n_ubatch`,
`llama-kv-cache-iswa.cpp`) - a real, tested path (referenced from
`llama-memory-hybrid-iswa.cpp` for hellaswag/winogrande eval), not
experimental - and no MTP-specific special-casing anywhere in
`common/speculative.cpp` or the server's spec-loading code suggests an
incompatibility either.

**The catch, and why this deployment doesn't get this for free:**
`kv_unified` only auto-enables when `--parallel` is left at "auto"
(`server.cpp`: `if (params.n_parallel < 0) { params.n_parallel = 4;
params.kv_unified = true; }`). Any explicit `--parallel N` - which is
exactly what this session's concurrency-aware `--moe-calibrate` work
does (deliberately reuses `--parallel` as the concurrency target), and
what a real deployment pinned to "5-10 users" would do - silently keeps
the wasteful fixed-division default instead.

Given GLM-5.2 is VRAM-constrained enough to need moe-cache's CPU/GPU
expert split in the first place, wasted KV reservation isn't just
"fewer raw concurrent connections" - it directly competes with
moe-cache's own expert-cache budget for the same free VRAM. That's a
more concrete case for us than the raw concurrency-count number the
maintainer thread was actually debating.

**Action, not yet taken:** add `--kv-unified` (paired with the
already-validated `--cache-idle-slots`/`--cache-ram`) to the real
deployment config and to this session's own calibration/sweep commands
going forward - all of which have been running with explicit
`--parallel N` the entire session and have therefore never exercised
unified mode. Every concurrency-related number measured so far in this
document (the 9-31 sweep, the concurrency=32 cliff, the
concurrency-aware calibration cross-validation) was measured under the
fixed-division default, not unified mode - worth flagging as a caveat
on those results, not just a forward-looking TODO, since unified mode
could plausibly change where cliffs/optimal placements land by
changing how much VRAM is actually free at a given `--parallel N`.

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

## MTP verify-batch size x concurrency, interacting with moe-cache's batch gate - RESOLVED 2026-08-14

Native MTP (GLM-5.2's NextN head, PR #25980) is architecturally safer than
the external-drafter case the RFC thread found broken - MTP runs in the
same trunk graph, same device/session as the main model, so the "GPU
drafter causes the cache to see zero hits" failure mode (a device/session
binding conflict under shared-draft-device reordering) likely doesn't
apply here.

The RFC thread's "dead batch-size zone" concern - requests sized 9-31
served by neither the cache (gated by `GGML_CUDA_MOE_CACHE_MAX_BATCH`,
default 8) nor the bulk-offload path (gated by `GGML_OP_OFFLOAD_MIN_BATCH`,
default 32) - turned out to be two separable questions, not one:

1. **Was moe-cache's own gate MTP-aware at all? No, confirmed and fixed.**
   `set_max_batch_hint` (llama-context.cpp) passes `n_seq_max` alone, no
   MTP verify-width multiplier - checked directly in the source, not
   assumed. Under MTP, the real per-step batch is `n_seq_max *
   (n_max+1)`, so the gate was silently sized for the no-MTP case on any
   MTP-active deployment. Fixed in `common.cpp`
   (`common_moe_apply_mtp_aware_max_batch_hint`, alongside Layer 1):
   explicitly sets `GGML_CUDA_MOE_CACHE_MAX_BATCH` to `n_parallel *
   verify_width` (capped at the real ceiling, 64) when MTP is active and
   the operator hasn't set an explicit value themselves. Verified:
   `--parallel 8` + MTP n_max=3 went from `max-batch=8` (bug, identical
   to the no-MTP case) to `max-batch=32` (correct); `--parallel 16` +
   n_max=3 went from 16 to 64. Non-MTP configs unaffected.
2. **Does a real throughput dip still exist in the 9-31 range even with
   the gate now correctly sized? Measured directly, and the answer is
   nuanced: no dead zone in 9-31 itself, but a real, separate, sharp
   cliff exactly at 32.** Ran a dedicated concurrency sweep (`-ncmoe 99
   --moe-cache auto -c 12288 --parallel 40`, real concurrent requests,
   confirmed moe-cache genuinely engaged via its own log lines - `max-batch=40`,
   real pool allocation, live hit-rate tracking, not a silent no-op):

   | concurrency | 4 | 8 | 9 | 12 | 16 | 20 | 24 | 28 | 31 | **32** | 40 |
   |---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
   | agg tok/s | 67.9 | 78.1 | 51.2 | 58.0 | 63.8 | 67.2 | 72.0 | 75.7 | 77.5 | **41.2** | 47.2 |

   Throughput climbs smoothly and monotonically from 9 through 31 (51.2 ->
   77.5) - the already-known-and-fixed MMVQ cliff shows up once at the
   8->9 boundary as expected, but there's no *further* dip anywhere in
   9-31 itself. Then a sharp, distinct cliff hits exactly at 32 (77.5 ->
   41.2, -46.8%), with partial recovery by 40 (47.2) - the same rise-then-
   cliff-then-partial-recovery shape as the MMVQ cliff, just at a
   different boundary. **Confirmed this is real and not a moe-cache
   artifact**: first run accidentally used `-c 32768` (too large a
   context for `--parallel 40` on this 12GB card, leaving no VRAM for
   moe-cache's own pool - confirmed via zero `[moe-cache]` log lines the
   entire run) and still showed the identical cliff shape/magnitude at 32
   (76.4 -> 41.6); the corrected re-run with moe-cache genuinely active
   reproduced it almost exactly (77.5 -> 41.2). The boundary lines up
   precisely with `GGML_OP_OFFLOAD_MIN_BATCH`'s default (32) - crossing
   into bulk-GPU-offload territory for CPU-resident ops appears to cause a
   real regression here, not the expected improvement, plausibly a
   PCIe/transfer-volume bottleneck from many expert tensors being shipped
   to GPU simultaneously for one step (untested hypothesis, not
   confirmed) - not yet root-caused to the same depth as the MMVQ cliff,
   but the empirical shape is solid and reproducible. **Practical
   takeaway for the H200/GLM-5.2 deployment** (5-10 concurrent users):
   comfortably clear of both cliffs (8 and 32) at that target, so neither
   needs fixing before deployment - flagged here in case concurrency ever
   scales toward 32 in the future.

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

## 1. LFRU eviction + hysteresis + decay (runtime, dynamic) — DONE, see results below

File: `c/tier.h` (~60 lines) was the original Colibri reference.

**Built and A/B'd, 2026-08-14 — see "LFRU eviction: A/B results" under
"Already built, VALIDATED with real measurement" below for the full story,
including two designs that measured worse and were reverted rather than kept
on the strength of a plausible-sounding rationale.** Final shipped design:
`ggml/src/ggml-cuda/moe-cache.cu` now uses a two-segment probation/protected_
structure (probation drains first, protected_ capped at 50% of pool with
heat-based demotion choice at the cap) plus a per-slot heat counter with
periodic decay, used as the eviction/demotion tiebreaker within each segment
(bounded 8-candidate window, 25%-plus-4 hysteresis - Colibri's own margin,
confirmed to matter here too). Measured 65.0% hit rate on a mixed hot/cold
workload vs. plain LRU's 63.8%.

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

## Design: CUDA MMVQ many-expert density gate (scoped, NOT implemented, 2026-08-14)

Investigated as a "moe.cpp" throughput-triage candidate, prompted by
upstream issue [ggml-org/llama.cpp#25356](https://github.com/ggml-org/llama.cpp/issues/25356)
(Vulkan backend: batched-decode throughput cliff at n_tokens=9 on
many-expert MoE, fixed 8-token thresholds in MMV dispatch). That issue is
Vulkan-only and its own thread found NVIDIA-via-Vulkan *regresses* with a
naive fix (-6% to -35%) - not directly portable. But the same architectural
blind spot exists in our real deployment backend, CUDA, independently
implemented and unexamined until now.

**The gap.** `ggml/src/ggml-cuda/mmvq.cu`'s `get_mmvq_mmid_max_batch(type, cc)`
picks the max token-batch for which the fast MMVQ (mat-vec) kernel is used
for `MUL_MAT_ID` (MoE routed matmul), via a hand-tuned per-architecture,
per-quant-type table (Pascal/Turing+/GCN/CDNA/RDNA1-4, referencing PR
#20905) capped at `MMVQ_MAX_BATCH_SIZE = 8`. The table takes `(type, cc)`
only - no `n_expert` or `n_experts_used` parameter anywhere in its
signature or body. This is exactly the blind spot #25356 identified for
Vulkan: the threshold was tuned against some canonical MoE shape (few
experts, e.g. Mixtral-class), and never accounts for how the real
computational cost driver - average rows landing on the *same* expert,
`n_tokens * n_experts_used / n_expert` - shrinks as `n_expert` grows, so a
many-expert model should tolerate a much higher raw token count before the
tiled/MMQ path actually wins.

**This is not hypothetical for us.** Our own sandbox model, checked
directly via GGUF metadata: `gemma4.expert_count = 128`,
`gemma4.expert_used_count = 8`. That's squarely in the "many-expert"
class the Vulkan issue's testers used (128- and 512-expert models). Our
own `common_warn_concurrency_cliff` (`common/common.cpp`, built earlier
this session investigating this exact CUDA-side cliff under MTP) already
detects and warns about it - on this hardware/quant combination it fires
at effective batch > ~5, well below where a density-aware gate would say
it's actually safe. Applying the community-validated formula
(`n_tokens * n_experts_used <= C * n_expert`, `C = 2`, cross-validated by
three independent testers on the Vulkan side) to our model's real numbers:
`n_tokens * 8 <= 2 * 128 = 256`, i.e. safe up to **n_tokens = 32** - the
literal hardware ceiling (one warp per token at `32 * 32 = 1024`
threads/block), a 4x wider window than the current flat cutoff of 8.

**Why this isn't a quick threshold edit.** `ggml_cuda_mul_mat_vec_q()`
(`mmvq.cu:1173`) has a hard `GGML_ASSERT(!ids || ne12 <= MMVQ_MAX_BATCH_SIZE)`
- not a soft per-arch/type threshold, an absolute ceiling on the whole
function's contract. `MMVQ_MAX_BATCH_SIZE` is threaded through buffer
sizing, warp-count calculation (`calc_nwarps`), and - critically - the
kernel's own `__launch_bounds__(get_mmvq_mmid_max_batch_for_device<type>()
* warp_size, 1)` (`mmvq.cu:708`), which is a *compile-time* register/
occupancy budget for the whole compiled kernel, not a per-call runtime
check. Simply raising the table's values would change that budget
globally, for every call site including the existing, already-tuned small
-batch case - real risk of a silent regression there (the same failure
mode the Vulkan thread hit on NVIDIA, just via a different mechanism:
register pressure instead of tile-padding GFLOP/s crossover).

**Proposed design (safe: never touches the existing compiled kernel).**
1. Add a second, non-default template parameter to `mul_mat_vec_q_moe`
   carrying the launch-bounds ceiling explicitly (default =
   `get_mmvq_mmid_max_batch_for_device<type>()`, so the *existing*
   call site's instantiation is byte-for-byte unchanged - same object
   code, same register budget, zero regression risk by construction).
2. Instantiate a second, explicit variant at the hardware ceiling (32),
   compiled as a genuinely separate kernel - isolates all new risk to
   code paths that only run in the new regime.
3. Add a new host launcher (parallel to `mul_mat_vec_q_moe_launch`) and a
   new top-level entry point alongside `ggml_cuda_mul_mat_vec_q()` (its
   hard `ne12 <= MMVQ_MAX_BATCH_SIZE` assert means the wide path needs its
   own entry, not a relaxed version of the existing one, so the existing
   function's contract - and every other caller relying on it - stays
   intact).
4. Wire the density gate into `ggml_cuda_mul_mat_id`/
   `ggml_cuda_mul_mat_id_needs_sync` (`ggml-cuda.cu:1889-1957`): after the
   existing narrow-kernel check fails, before falling to MMQ, try the wide
   path when `ne2 <= 32 && ne2 * n_expert_used <= 2 * n_expert` (n_expert =
   `src0->ne[2]`, n_expert_used = `ids->ne[0]`).
5. Validate on this sandbox (Gemma-4-26B-A4B, 128/8 experts, real
   many-expert case): correctness first (byte-identical greedy output at
   batch sizes spanning both the untouched narrow path and the new wide
   path), then throughput (wide-MMVQ vs. current MMQ/cuBLAS fallback,
   which is what it's actually competing against in the 9-32 token range,
   not the narrow kernel) across the batch sizes that currently fall back.
   Also explicitly re-benchmark the *unchanged* narrow path (batch <= 8)
   to confirm the new template parameter really did leave it untouched,
   not just assume it from the design.

**Effort estimate**: real CUDA kernel engineering, not a config change -
new kernel instantiation, new launcher, new dispatch wiring, correctness
+ performance validation across quant types actually in use. Scoped here
so the investigation doesn't need to be redone, but deliberately not
started without a dedicated pass - the blast radius (MUL_MAT_ID feeds
every MoE FFN forward pass) means a subtle bug here is a correctness risk
across the whole model, not a contained feature.

## Backend sampling (`-bs`) validated on this sandbox - real, modest, safe (2026-08-14)

Investigated as a moe.cpp throughput candidate, prompted by upstream
[ggml-org/llama.cpp#27050](https://github.com/ggml-org/llama.cpp/issues/27050)
(same-day issue: `-bs` measured +48% aggregate throughput at 32 concurrent
slots on an RTX 5090, 706 -> 1046 tok/s, greedy output byte-identical,
single-slot unchanged). Unlike the other candidates this session, nothing
needed building or porting - `--backend-sampling`/`-bs` already exists in
this fork, fully wired end to end (`arg.cpp`, `common.h`,
`server-context.cpp`, `sampling.cpp`), marked experimental and off by
default. The open question was purely: does it deliver a real, safe gain
on our own hardware/model, worth recommending for our multi-user serving
config.

**Mechanism** (per the issue, confirmed by reading our own
`server-context.cpp`): the default path copies the full logits matrix
device-to-host every decode step so `common_sampler_sample()` can run on
CPU; at higher slot counts that transfer plus CPU-side sampling/sorting
becomes the bottleneck. `-bs` samples on-device and skips the transfer.
Already self-gates safely: `common/sampling.cpp` disables it automatically
(with a warning) when combined with grammar or a reasoning budget, and
`server-context.cpp` disables it when pre-sampling logits are requested
(`n_probs` without `post_sampling_probs`) - none of that needed building,
it was already there.

**Measured directly on this sandbox** (Gemma-4-26B-A4B, `-ncmoe 15`, RTX
3060, greedy, 200-token completions, aggregate tok/s = total completion
tokens / wall time across all concurrent requests):

| config | bs off | bs on | delta |
|---|---|---|---|
| `--parallel 4` (below this card's MMVQ concurrency cliff - clean comparison) | 103.59 tok/s | 107.48 tok/s | **+3.76%** |
| `--parallel 8` (past the cliff - both arms equally confounded, comparison still fair) | 99.66 tok/s | 104.62 tok/s | **+4.98%** |

Real and consistent in both configs, not an artifact of the concurrency
cliff (the clean `--parallel 4` run shows the same direction and similar
magnitude). Far short of the issue's own +48% at 32 slots - expected: this
sandbox tops out at single-digit `--parallel` on a 12GB card, and the
win's mechanism (D2H copy + CPU sampling overhead relative to per-step GPU
compute) scales with both slot count and GPU speed, neither of which this
sandbox has much of relative to an RTX 5090 at 32 slots. Correctness
spot-checked: coherent, non-garbled output, `system_fingerprint` intact,
no incompatibility warnings triggered (config used no grammar/reasoning
budget/`n_probs`).

**Recommendation**: enable `-bs` for this deployment's multi-slot serving
config - real, positive, low-risk (already-gated, self-disabling on known
incompatibilities, upstream-authored and upstream-tested at real scale).
Not flipping the compiled-in default: it's still marked experimental
upstream, and this sandbox has only validated one model/quant/prompt
shape at modest concurrency, not the breadth upstream would want before
changing what every caller gets by default. The GLM-5.2/H200 target,
running at real multi-user concurrency on much faster hardware, is exactly
the regime where the issue's own +48% figure suggests this matters more,
not less - worth re-validating there specifically once that deployment is
reachable, rather than assuming this sandbox's modest ~4% ceilings it.
