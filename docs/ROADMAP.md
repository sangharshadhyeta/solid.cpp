# solid.cpp roadmap and test ledger

Written 1 Oct 2026 so nothing agreed in conversation is lost. Update it when a row changes state. Numbers here are measured on
this machine (RTX 3060 12 GB, i5-12400F 6C/12T, 30 GB RAM, NVMe) with the paired method below; "est." marks an estimate.

## Rules that govern the work

- Port hot-rod ONE step at a time; each step built, operator-tested and checked on gemma-4 first. Qwen is run ONCE, at the end, on the combined build.
- Never claim a gain that was not measured. A commit message states what was verified and what was not.
- Never weaken a test to make it pass. Never trust a single run: gemma greedy output is not bit-reproducible (same setting twice agrees 21-57%), so every comparison carries a same-build control.
- Take from other projects only what is better than ours; keep our design where it is better (it is richer than the papers in substitution, prefetch, eviction).
- A failed build must stop a test chain (a chain once tested a stale binary and main was pushed uncompilable for ~25 min).
- Kill exact PIDs, never `pkill -f`. One background task per wait. ThinkVest stays stopped while the model is being worked on.

## Method

Paired A/B: same prompt and seed per request across settings, 3 rounds, an identical-config control (`*_again`), warm median plus cold (first two requests),
mean +-se and requests-faster counts. Harness: `/root/.claude/jobs/743230e4/tmp/ab` (ab2.py, tally.py, stepcheck.py, longctx.py, gemma256.py, fidelity.py, rsscheck.py).
Chat endpoint with the chat template; calibration prompts = code + train problem; mixed set = mostly prose.

## Wins in the tree (gemma-4, 64K context, 2 slots)

| Change | Measured |
|---|---|
| Shipped defaults overall | 77.6 / 78.2 / 77.1 tok/s (code+reasoning, was 54.8/56.2); 59.3 / 62.5 / 59.9 (mixed, was 29.0/33.5/31.3) |
| Neuron subsetting off unless calibrated on | cost ~33%; ~+50% when off |
| Draft depth 4 (was 8) | depth 8 is -11% +-2 on the shipped stack; +22-32% earlier |
| Draft confidence gate p_min 0.5 | +13% |
| Do not pin `--spec-draft-n-max 8` in ThinkVest's launch | +15% prose, 0% code |
| `-md` naming an MTP head runs as one (header-based) | -25% without it |
| MoE model that fits 3/5 of total RAM is read into memory | -13% without it |
| Expert-cache VRAM reserve floor 1,500 MiB | two ~32K-token prompts complete; no speed cost |
| Draft VRAM reserved in the serving placement probes | launch without `-ncmoe` no longer crashes (runtime test of the full no-`-ncmoe` launch still open) |
| 64K minimum context | fit floor and fallbacks were 4,096 |
| PART_STATS (read-only counter) | gemma: 5.1% partial expert residency, 8.6% rows miss, 69 MiB/token fetched |

## Measured, not wins (kept, off or neutral)

Pre-router (tie gemma; Qwen -7% but the run did not show the saved weights loading or saving: RECHECK), neuron subsetting (-33% gemma, -5% Qwen),
substitution (no speed at any floor; output drifts: agreement 33% shipped, 16% rank 4, 7% rank 2 vs 57% control), `-ub 1024` (+5% decode but -22% per
request on 5K-token prompts), drafter n-gram cascade, pinned cores, 5 threads, prediction ring, lazy KV (bigger cache, no speed, killed ~32K prompts: opt-in until the yield below passes).

## Losses to turn into wins (what to learn, and the test)

| Loss | Learn from | Test |
|---|---|---|
| Pre-router | Fate (arXiv 2502.12224): next-layer experts from the current layer's gate input, training-free; HOBBIT | Qwen run with load+save verified in the log; then a training-free cross-layer variant |
| Neuron subsetting | HOBBIT low-precision copy on a miss; SliceMoE bit slices | per-expert adaptive K, or a lower-bit stand-in instead of dropping neurons |
| Substitution drift | Cache-conditional experts (arXiv 2412.00099): bias routing toward resident experts | small router-logit bonus for resident experts at decode + fidelity harness |
| Lazy KV | vLLM: reclaim from the evictable side when blocks run out | `ggml_backend_cuda_moe_cache_yield_vram` (written, uncommitted) - test 2 concurrent ~30K/26K prompts, speed, 240K run |
| `-ub` trade-off | SSD-LLaMA: group prompt rows per expert, one load per expert per pass | check our prefill grouping rather than shrinking `-ub` |
| Prose acceptance | MoE-SpeQ: use the draft pass to predict the target's experts and prefetch | feed the MTP head's hidden state to the predictor |

Already covered by our code (nothing to port): BuddyMoE similarity pairs = atlas-similarity stand-ins; Fate/HOBBIT prefetch = lookahead + next-layer predictor.

## Atlas

Lookahead and eviction protection ON by default; warming (cold-start only) and the admission ring OFF. Not re-measured this session: add an atlas-off setting to the next paired run.

## In flight / next, in order

1. Lazy-KV yield: build main with it, test (chain26: long prompts yield on/off, speed vs default, 240K gemma). Default lazy KV back on ONLY if both long prompts complete and speed holds.
2. Merge the mmid/mmf race fix (a83cc5d24 on `port/hotrod-kv`; MUL_MAT_ID 869/869, MUL_MAT 1186/1186, +2.8% vs control spread 0.3%) AFTER step 1's build.
3. Atlas on/off; f16 vs q8_0 KV at 64K (f16 +3.5-4% on gemma; q8_0 is what allows 256K, so likely f16 up to 64K only).
4. Hot-rod port, gemma-checked each: upstream small fixes + loader RAM peak, MoE fusions one at a time, row prefetch, KV chain S3a-S3e (gemma proves build/ops/clean refusal; `--kv-stream` is qwen4exp only).
5. Placement without `-ncmoe` under `-fit off` gives the cache 0 slots (-19.6%): when the cache is on, do not spend VRAM on resident experts.
6. Loss-to-win variants above, each flag-gated, each a default only if it wins paired.
7. Concurrent O_DIRECT expert reads + sidecar per-expert pack (SSD-LLaMA/Strata). Only provable on Qwen; gated on the Qwen PART_STATS and the SSD actually being the limit.
8. FINAL QWEN RUN (needs the IQ3_S file): UD-IQ4_XS vs ISTA GSQ-RCO IQ3_S (speed, output agreement), KV streaming on/off, `-c 262144` short-prompt start, PART_STATS, n-gram table stays lazy, existing MTP file with IQ3_S, pre-router recheck.
9. Keep docs/index.html to tested numbers only (`docs_audit.py` regenerates it); update memory; remove the temp worktrees; tell the owner what to change in ThinkVest's launch.

## Hot-rod port status

Merged to main: step 1 (argsort + flash-attention barrier), 2a (sparse FA path), 2b (sparse FA, qwen4exp head 256), PART_STATS (+ its format-string repair).
Prepared on `port/hotrod-kv` (12 commits, docs stripped) and `port/hotrod-upstream` (10). The hot-rod author's own handoff says everything there is written and NOTHING was built or run.
Wait for the owner's next hot-rod push before rebasing the port branches. Skip list (renames, UI/CI strip, docs-only) is in `docs/hot-rod-changelog.md` on `hot-rod`.

## Environment facts

- Network ~1-1.8 MB/s: the Qwen IQ3_S file (ISTA-DASLab, 83.6 GB, `hf download ... --include "IQ3_S/*"`, resumable) lands ~10:00 on 2 Oct (est.). DeepSeek-V4.1-Flash GSQ-RCO (416 GB) does not fit the 334 GB free disk.
- Gemma 256K, one slot, dense KV: completed (240K tokens, prefill 422 s), but only 510 expert-cache slots (hit 0.31) and 12.8-15.8 tok/s: the KV starves the cache. That is what the yield and any KV-size work must fix.
- ThinkVest launch (`ThinkVest/scripts/services.sh`) no longer pins depth 8 / `--spec-prob-accept` (committed there).
