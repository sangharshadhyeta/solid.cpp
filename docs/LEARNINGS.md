# Learnings from other implementations that we can implement

1 Oct 2026. Only techniques we do not have, or have a worse version of, written so each can be built. Sources were read in full from their papers; every
number marked "source" is THEIR claim on THEIR hardware, not measured here. Each item names where it goes in our code, the test, and what decides whether it
becomes a default (paired against the shipped default, same-build control, fidelity harness for anything that changes outputs).

Not on this list because ours is already as good or better: atlas-similarity stand-ins (BuddyMoE pairs), lookahead + next-layer predictor (Fate/HOBBIT prefetch),
heat/LFRU eviction (LRU/LFU/ARC), lazy KV commit (vAttention; we only add the reclaim, see L7), whole-expert admit/evict (Strata; gemma measures 5.1% partial residency).

## L1. Bias the router toward resident experts (Cache-Conditional Experts, arXiv 2412.00099)

- Mechanism (source): `z' = z + lambda * delta_avg * m_t`. `z` router logits, `m_t` bitmask of resident experts, `delta_avg` running average of the logit
  range (max-min) over layers and tokens, `lambda` in [0,1]. `z'` is used ONLY to choose the top-k; the forward pass weights come from the ORIGINAL `z`, so a
  chosen expert is computed exactly. Always keep the top-J experts by `z` regardless of residency (J=1 for 8-expert models, 2 for fine-grained). Past lambda ~0.8
  quality degrades. Learnable priors did not beat the fixed one. Source results: miss rate 35%->16% (Qwen1.5-MoE), 28%->7% (DeepSeek-V2-Lite); perplexity +0.1-3%, downstream <0.1%.
- Why it beats our substitution: substitution fixes a miss AFTER the router picked it and computes a different expert with the missed one's weight (output drifts, no
  speed gain measured). The bias changes the SELECTION, every chosen expert is exact, and misses never happen.
- Where: the router top-k in the MoE graph builder (`src/llama-graph.cpp` build_moe_ffn) needs a per-layer residency mask on the device: a bitmask the cache updates
  on admit/evict (the plan path already knows residency), added to the logits before top-k. Keep the router weights computed from unmodified logits.
- Test: gemma-4, lambda in {0.1, 0.2, 0.4}, J=1/2; hit rate, tok/s paired, greedy agreement vs substitution-only. Default only if tok/s up AND agreement no worse than the shipped substitution (33% vs 57% control).

## L2. Importance by cumulative gate mass, with a skip class (HOBBIT, arXiv 2411.01433)

- Mechanism (source): rank the K selected experts by gate weight; unimportance `s_i = sum of the weights of all higher-ranked experts`. `s <= 0.6` full precision,
  `0.6 < s <= 0.9` low precision, `s > 0.9` SKIPPED entirely. Gate magnitude correlates 0.99 with the expert's actual output contribution. Mixtral: 67% full / 30% low / 3% skipped; accuracy loss <1%.
- What we do instead: a rank floor (substitute only ranks >= 6) - rank ignores how much weight the expert carries. On gemma-4 ranks 4-7 hold only 32% of the gate mass.
- Where: the substitution gate in `moe_cache_plan` / `moe_cache_substitute_min_rank` -> a mass threshold; a missed expert past the skip threshold is dropped (weight 0, no fetch, no CPU compute).
- Test: replace the rank floor by `s` thresholds (0.6/0.9 first, then sweep); fidelity + paired speed. Candidate to REPLACE the rank floor rather than add to it.
- Second half (later): a low-precision copy of an expert held in RAM, loaded instead of the full one for the middle class. Needs a requantized copy (extra RAM); only after the first half is measured.

## L3. Adaptive draft length from a cost model (MoE-SpeQ, arXiv 2511.14102)

- Mechanism (source): choose k to maximise `E[accepted tokens] / E[cycle time]`, `cycle(k) = max(T_draft(k), T_io_init) + T_io_new(k) + T_verify(k+1)`, where
  `T_io_new` grows with the number of NEW experts the k drafted tokens need. Acceptance tracked by EMA, costs profiled offline and updated online.
- Why it matters here: depth 8 loses (-11%) and depth 4 wins because verify width widens the expert union; the best k differs by prompt (accept 0.87 code, 0.67 prose). We fix 4.
- Where: `common/speculative.cpp` / the draft loop in `common/sampling.cpp` (n_max, p_min). Track acceptance per draft position (EMA) and the measured verify time per width; pick k each round.
- Test: mixed and calibration sets paired against fixed 4; must not lose on code and should gain on prose. Cheap, gemma-testable, no new kernels.

## L4. Fate's cache and prefetch details (arXiv 2502.12224)

- Shallow-layer-favoring allocation (source): layers 0-3 predict poorly (~60-70% prefetch accuracy vs ~85%+ deep), so cache ALL experts of the first L layers and split the rest
  evenly: `cache_deep = (cache_total - L*n_expert) / (n_layer - L)`. Source hit rate 99.08%. Ours: one shared heat ordering. Test: heat bonus (or reserved slots) for the first N layers; per-layer hit-rate lines already exist in the SUMMARY output.
- Confidence filter on prefetch (source): transfer only predicted experts above the 75th percentile of confidence: accuracy 78.8% -> 97.2%. Ours admits predicted experts into free slots without a confidence cut. Test: add the percentile filter to lookahead/predictor admission; measure wasted fills and hit rate.
- Time-budgeted prefetch (source): `n = T / t_expert`, `T = t_moe + t_attn + t_gate` per layer: only as many experts as can transfer while the layer computes. Ours caps by ring size and free slots, not by time. Test: cap fills per layer by measured layer time over measured fill time.
- Prefill order (source): transfer experts by descending token count (most-demanded first). Where: `moe_cache_prefill_advance` order. Test: first-token latency on 5K-token prompts, which is what ThinkVest sends.

## L5. SSD-LLaMA's read pipeline (arXiv 2609.18110) - for Qwen, where misses come from the SSD

- Mechanism (source): many independent reads in flight in bounded batches with O_DIRECT, each completed read goes to VRAM asynchronously without waiting for the rest; one aligned block per expert;
  all prompt rows for an expert form one task so each expert loads once per prefill pass; RAM-resident experts run on CPU or on GPU after H2D by measured end-to-end latency. Source: SSD bandwidth 77.7% of peak vs 43.2% for llama.cpp; decode 2.1-15.6x (PCIe 5 SSD, RTX 5090).
- What we have: one `pread` per expert behind `GGML_CUDA_MOE_CACHE_EXPLICIT_READ=1`, called from a single fill worker; misses are otherwise read by CPU threads page-faulting the mmap.
- Plan: (1) MEASURE first - NVMe read MB/s and queue depth during Qwen decode (iostat) against the drive's rated speed, and `PART_STATS` miss MiB/token; if the SSD is not near its limit this item is moot.
  (2) At plan time, with the router result known, issue ALL missed experts of the layer at once (bounded batch, O_DIRECT or io_uring) before CPU compute starts, so queue depth > 1.
  (3) Group prefill rows per expert (check what `-ub` splitting does today: each ubatch reloads).
- Test: Qwen only (final run); decode tok/s and MB/s paired against EXPLICIT_READ off/on.
- REJECTED from this paper, with evidence: rANS compression of experts (-33% SSD traffic for them). Our Qwen expert tensors are IQ3_S (gate, up) and IQ4_NL (down); measured order-0 byte entropy is 7.8-7.95 of 8 bits, i.e. 0.6-2.4% saving. Not worth a GPU decoder.

## L6. Smaller items

- HOBBIT's layer-aware eviction term (FLD, farthest layer distance) and "high-precision frequency" (LHU): reported 4.7-8.7% fewer miss-penalty vs LRU. A layer-distance term in `moe_cache_weighted_heat` is a one-line experiment.
- vLLM/vAttention reclaim: implemented as `ggml_backend_cuda_moe_cache_yield_vram` (see ROADMAP step 1).

## Order of work (gemma-testable first)

L1 -> L2 -> L3 -> L4 -> then L5 measurement during the final Qwen run. Each is a flag-gated variant, off until it wins paired.

## Correction recorded here

The Qwen file we run (UD-IQ4_XS) already stores the routed experts mostly as IQ3_S/IQ4_NL, so the GSQ-RCO IQ3_S file will cut bytes per expert by less than the 27% file-size
difference suggests (its saving is largely outside the expert tensors). The expected speedup from it is a hypothesis for the final run, not an estimate to rely on.
