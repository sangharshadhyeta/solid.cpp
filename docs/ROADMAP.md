# solid.cpp roadmap and test ledger

Written 1 Oct 2026, updated 2 Oct 2026 01:xx. Numbers measured on this machine (RTX 3060 12 GB, i5-12400F 6C/12T, 30 GB RAM,
NVMe) with the paired method below unless marked "source" (another project's own claim, not measured here).

## Rules that govern the work

- Port hot-rod ONE step at a time; each step built, operator-tested and checked on gemma-4 first. Qwen is run on the final build.
- Never claim a gain that was not measured. A commit message states what was verified and what was not.
- Never weaken a test to make it pass. Never trust a single run: gemma greedy output is not bit-reproducible (same setting twice
  agrees 21-67% depending on the run - always carry a same-build control). Qwen is even noisier at small N (n=5 agreed
  38-55% across settings with no clean separation) - treat small-N Qwen fidelity numbers as unreliable; its large-N aggregate
  counters (cache hit rate, summed over hundreds of thousands of hits/misses) are reliable even when fidelity isn't.
- Take from other projects only what is better than ours; keep our design where it is better.
- A failed build must stop a test chain. A structural plumbing change to a hot dispatch path (changing a callback signature
  used by every call site) does not get rushed in at the end of a session - scope it, document it, do it fresh.
- Kill exact PIDs, never `pkill -f` (it can match your own wrapper's command line and self-abort). One background task per
  wait. Before trusting ANY benchmark number, check nothing else is contending for the same disk/GPU - a duplicated,
  forgotten background download on the same physical NVMe as the model files under test silently degraded an entire Qwen
  run's timing numbers until caught (2 Oct). ThinkVest stays stopped while the model is being worked on.
- Large downloads for comparison files go on a disk OTHER than the one holding the models under active test (here: `/home`,
  not `/mnt/nvme` - verified via `lsblk` to be a genuinely separate physical device, not just a different mountpoint on the
  same disk).

## Method

Paired A/B: same prompt and seed per request across settings, an identical-config control (`*_again` or `exact_ref_again`),
warm median plus cold (first two requests), mean +-se and requests-faster counts. Fidelity checks: an EXACT unbiased
reference (every lossy feature off), greedy decode, prefix-agreement against it, always read against that setting's own
noise-floor control before concluding anything. Harness: `/root/.claude/jobs/743230e4/tmp/ab` (ab2.py, tally.py,
stepcheck.py, longctx.py, gemma256.py, fidelity.py, biascheck.py, biascheck_qwen.py). Chat endpoint with the chat template.

## Wins in the tree (gemma-4, 64K context, 2 slots, unless noted)

| Change | Measured |
|---|---|
| Shipped defaults overall | 77.6/78.2/77.1 tok/s (code+reasoning, was 54.8/56.2); 59.3/62.5/59.9 (mixed, was 29.0/33.5/31.3) |
| Neuron subsetting off unless calibrated on | cost ~33%; ~+50% when off |
| Draft depth 4 (was 8) | depth 8 is -11% +-2 on the shipped stack; +22-32% earlier |
| Draft confidence gate p_min 0.5 | +13% |
| Do not pin `--spec-draft-n-max 8` in ThinkVest's launch | +15% prose, 0% code |
| `-md` naming an MTP head runs as one (header-based, not filename) | -25% without it |
| MoE model that fits 3/5 of total RAM is read into memory, not mapped | -13% without it |
| Expert-cache VRAM reserve floor 1,500 MiB | two ~32K-token prompts complete; no speed cost |
| Draft VRAM reserved in the serving placement probes | launch without `-ncmoe` no longer crashes |
| 64K minimum context | fit floor and fallbacks were 4,096 |
| PART_STATS (read-only counter) | gemma: 5.1% partial expert residency, 8.6% rows miss, 69 MiB/token fetched |
| mmid/mmf race fix | MUL_MAT_ID 869/869, MUL_MAT 1186/1186 pass; +2.8% vs 0.3% control (tie, no regression) |
| Lazy-KV commit-failure yield (`ggml_backend_cuda_moe_cache_yield_vram`) | 2 concurrent ~30K-token prompts: server lives with it, dies without it (confirmed log: pool released, commit retried) |
| Lazy KV default ON for windows >=131072 cells, OFF below | -c 262144 short prompts: +15.1% +-1.6, 1315 vs 835 slots, hit 0.66 vs 0.53. -c 65536: tie (stays off). **3-way verified 2 Oct**: no-env-var auto run (975 slots/hit 0.586/49.8 tok/s) matches forced-ON (986/0.578/52.8) and is clearly unlike forced-OFF (511/0.349/39.5) - auto-detection confirmed correct |
| 9-commit upstream group (loader RAM-peak fix, GGUF load speedup, RMS_NORM+SCALE fusion, MoE-reduction fusion, fusion extended to spec-decode batches, row prefetch, 0-sized-ids skip, draft-context-not-embedding fix) + SWIGLU_CLAMP adaptation | builds clean, all 6 touched operators 100% pass, +3.0% +-1.2 vs 4.9% control (tie - no regression, no proven gain on gemma; this group's payoff is mostly Qwen-side: RAM-peak fix, GGUF load speed, row prefetch) |
| Router bias toward cache-resident experts (`LLAMA_MOE_CACHE_BIAS=beta`, decode-only) | **See "Router bias vs substitution" below - the full story, corrected after an initial test ran prefill-biased by mistake.** |

## Measured, not wins (kept, off or neutral)

Pre-router (tie gemma; Qwen -7% but unverified whether the saved weights even loaded - RECHECK), neuron subsetting (-33%
gemma, -5% Qwen), `-ub 1024` (+5% decode but -22% per request on 5K-token prompts), drafter n-gram cascade, pinned cores,
5 threads, prediction ring, shallow-layer eviction bonus (Fate-style: tie at 2x, -2.4% at 4x on gemma - removed from tree),
adaptive draft-length governor (MoE-SpeQ-style bandit over depths 2-8: **-4.2% +-1.4 vs fixed depth 4, 4 of 16 faster** -
it explores depths that lose and depth 4 is already the best arm on gemma's mix; kept in tree, OFF by default).

## Router bias vs substitution - the real comparison

Substitution's documented +8.11% win (`DEFAULT ON` comment in moe-cache.cu) was measured on **Ornith-1.5-35B-A3B**
(256 fine-grained experts, qwen35moe arch) - never on gemma-4 or Qwen3.8-Flash-Next. That model has far more expert
redundancy (a stand-in is a close approximation) and a much higher natural miss rate than gemma's 0.84-0.86 hit rate.
Substitution was never actually validated in gemma's or Qwen's regime; today's results don't contradict the original
finding, they just show it doesn't generalize to these two models as shipped.

First bias test accidentally biased PREFILL too (not just decode) - corrected (`391f116f0`): bias now only touches
decode-sized batches (`LLAMA_MOE_CACHE_BIAS_MAX_BATCH`, default 8 tokens), matching the paper and the lookahead gate's
own convention.

**Gemma-4** (exact reference = substitution off entirely, n=10 prompts, greedy):

| Setting | Agreement vs exact ref | Speed vs ref | Hit rate |
|---|---|---|---|
| Exact ref, run again (noise floor) | 67.1% | +0% | 0.847 |
| Substitution (shipped default) | 21.0% | +1.6% | 0.848 |
| Bias 0.5, decode-only, subst off | **31.1%** | +2.6% | 0.894 |
| Bias 1.0, decode-only, subst off | 27.8% | +3.9% | 0.917 |
| Bias 1.0 + substitution also on | 20.3% | +1.5% | 0.916 |

**Qwen3.8-Flash-Next** (n=5 prompts - CAUTION, see contamination note below):

| Setting | Agreement vs ref | Speed vs ref | Hit rate (large-N, reliable) |
|---|---|---|---|
| Exact ref | 100.0% | +0% | 0.262 |
| Exact ref again (noise floor) | 41.5% | +2.3% | 0.256 |
| Substitution (shipped) | 54.6% | -1.0% | 0.254 |
| Bias 0.5, subst off | 38.6% | +1.5% | **0.330** |
| Bias 0.5 + substitution fallback | 45.6% | +6.2% | **0.331** |

**What's solid on both models:** substitution never moves the cache hit rate (by construction - it answers a miss
differently, it doesn't prevent one). Bias DOES move it (gemma 0.847->0.917 at beta=1.0; Qwen 0.26->0.33 at beta=0.5) -
a real, large-N-backed, mechanism-consistent result on both models. Stacking bias+substitution is never better than bias
alone (gemma: 20.3% < 27.8%; Qwen: hit rate identical, 0.331 vs 0.330 - exactly what "can't move hit rate by construction"
predicts for the fallback case too).

**What's NOT solid yet:** the Qwen fidelity/speed table above was run while two duplicate background download processes
(an earlier bug - see Rules) were hammering the SAME NVMe drive Qwen's disk-backed experts stream from. Found and fixed
mid-run (2 Oct ~01:00): killed the stale duplicate, moved the download entirely off `/mnt/nvme` to `/home` (separate
physical disk, verified via lsblk). The Qwen speed/fidelity numbers above should be treated as contaminated by I/O
contention (noise-floor control itself swung +2.3% with zero real difference) and **rerun clean before trusting them** -
the hit-rate numbers are fine (large-N aggregate, not sensitive to per-request timing jitter).

**Decision: NOT made a default yet.** Bias beats substitution on every measure that survived testing on both models. It is
a legitimate candidate to replace substitution as the default miss-handling strategy, pending: (1) a clean Qwen
speed/fidelity rerun now that the disk is quiet, (2) an owner decision, since this changes default output on every request.
Both stay opt-in flags; substitution is NOT deleted (kept as a documented fallback option either way this resolves).

## L2 (HOBBIT cumulative-gate-mass skip) - scoped, not attempted this session

Confirmed by reading the code (2 Oct): `rank_bucket` in `moe_cache_plan_impl` is purely positional
(`index % rank_top_k`, i.e. "which slot in the top-k list"), not weighted by the actual softmax gate value. No gate-weight
array reaches the planner at all - only `ids[]` (expert indices). Implementing HOBBIT's cumulative-mass threshold properly
requires passing a parallel weights array through the op callback interface that feeds `moe_cache_plan_impl`, touching
every call site that invokes it in the hot MoE dispatch path. This is a structural change, not a flag - deliberately not
rushed in at the end of a long session per the rules above. Next session: scope the callback signature change first,
build it on the test worktree, verify operator tests before touching main.

## Atlas

Lookahead and eviction protection ON by default; warming (cold-start only) and the admission ring OFF. Not re-measured
this session.

## Hot-rod port status

Merged to main: step 1 (argsort + flash-attention barrier), 2a (sparse FA path), 2b (sparse FA, qwen4exp head 256),
PART_STATS (+ format-string repair), mmid/mmf race fix, the 9-commit upstream group (see table above).
Prepared, not yet merged: `port/hotrod-kv` (KV streaming chain S3a-S3e, qwen4exp-only - gemma can only prove it builds,
passes operator tests, and refuses cleanly on devices that don't support it; `port/hotrod-upstream` is now empty, fully
merged). The hot-rod author's own handoff (as of last check) said everything there was written but NOTHING was built or
run on a GPU - wait for the owner's next push before rebasing.

## Next, in order

1. Clean Qwen speed/fidelity rerun (bias vs substitution) now that the duplicate-download disk contention is fixed.
2. L2 (HOBBIT mass-threshold skip) - scope the op-callback plumbing change properly, build on the worktree first.
3. KV streaming chain (S3a-S3e) port, gemma-checked for build/ops/clean-refusal, Qwen-tested at the end.
4. Concurrent O_DIRECT expert reads + sidecar per-expert pack (SSD-LLaMA/Strata idea) - Qwen-only, gated on PART_STATS
   confirming the SSD is actually the bottleneck there.
5. Final decision + rollout on router bias vs substitution as the default, once the clean Qwen numbers are in and the
   owner has weighed in.
6. Keep `docs/index.html` to tested numbers only (`docs_audit.py` regenerates it); remove the temp worktrees when the
   hot-rod port is fully resolved.

## Environment facts

- The Qwen GSQ-RCO IQ3_S comparison file (ISTA-DASLab, 83.6 GB) is downloading on `/home` (NOT `/mnt/nvme` - that disk
  hosts the models under active test). Network ~1 MB/s observed; resumable (`curl -C -`), sha256-checked on completion.
- Gemma 256K, one slot, dense KV: completes (240K tokens, prefill 422s / 657 tok/s), but only 510 expert-cache slots
  (hit 0.31) and 12.8-15.8 tok/s - the KV starves the cache at full-window usage. The large-window default above targets
  the SHORT-prompt-at-large-window case, not this one; a full 240K prompt is still the one case lazy KV doesn't help.
- ThinkVest launch (`ThinkVest/scripts/services.sh`) no longer pins depth 8 / `--spec-prob-accept` (committed there).
