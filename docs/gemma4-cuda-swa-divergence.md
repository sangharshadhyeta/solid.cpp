# Gemma-4 produces different logits on CUDA than on CPU

**Status:** open, root cause narrowed but not fixed. Found 2026-09-04.

## Symptom

A plain chat request to gemma-4-26B-A4B on GPU returns text with raw control
markers in it, and degenerates into loops:

```
"hi" -> '<|channel>thought\n<channel|><channel|><channel|>...'
        'I am Gemma 4, a model trained by Google DeepMind.' x4
        ' way, way, way, way, way, way, ...'
```

The same model, same GGUF, same prompt, same seed, run with `-ngl 0` (pure
CPU) answers correctly and cleanly:

```
"hi"          -> 'Hello! How can I help you today?'
"What's 2+2?" -> '2 + 2 = 4'
```

So the model file, the chat template, the tokenizer, and the gemma4 PEG chat
parser are all fine. The divergence is in the CUDA path.

## Evidence

Greedy (`temperature=0, top_k=1`), same seed, same prompt "hi", comparing the
first generated tokens and their top-5 logprobs:

| token | CPU (`-ngl 0`) | GPU (default) |
|---|---|---|
| 0 | `<|channel>` (0.0) | `<|channel>` (0.0) |
| 1 | `thought` (-0.0) | `thought` (-0.0) |
| 2 | `\n` (-0.0) | `\n` (-0.0) |
| 3 | **`The` (-0.012)**, `<channel\|>` at **-11.204** | **`<channel\|>` (-0.0)**, `The` not in top-5 |

Both sides are *confident*, in opposite directions, with an ~11 nat gap. That
is not floating-point drift near a decision boundary - it is a different
computation. The whole distribution is also compressed on GPU (token 0's
runner-up sits at -25.2 on GPU vs -29.6 on CPU).

## What it is NOT

Each of these was tested and independently ruled out - the divergence persists
with every one of them disabled:

- moe-cache substitution (`GGML_CUDA_MOE_CACHE_SUBSTITUTE_MIN_RANK=1000000`)
- the moe-cache itself (`GGML_CUDA_MOE_CACHE=0`)
- expert prefetch (`GGML_SCHED_PREFETCH_EXPERTS=0`)
- LFRU D2D (`GGML_CUDA_MOE_CACHE_LFRU_D2D=0`)
- flash attention (`-fa off`)
- CUDA graphs (`GGML_CUDA_DISABLE_GRAPHS=1`)
- MMQ kernel selection (`GGML_CUDA_FORCE_MMQ=1`)
- expert placement - all experts on CPU (`--n-cpu-moe 99`) still diverges
- output head placement - `-ngl 10` (head on CPU) still diverges

`test-backend-ops` passes completely against CPU with a free GPU, so no
individual op is wrong in isolation. Note that a busy GPU makes that suite
report spurious `FAIL`s that are really `cudaMalloc ... out of memory` - free
the VRAM before trusting it.

## What it IS: sliding-window attention

`--swa-full` moves the first divergence from token 3 to token 10:

```
CPU:            ... '".', '\n', 'This', ' is', ' a', ' standard', ' greeting', '.'
GPU --swa-full: ... '".', '\n', 'The',  ' user', ' wants', ' to', ' start', ...
GPU (no flag):  diverges at token 3
```

Ten matching tokens before parting is consistent with ordinary fp accumulation;
diverging at token 3 is not. So the SWA KV-cache path is the primary suspect,
and `--swa-full` is a partial workaround (it does not eliminate the residual
divergence, and generation still degrades on longer outputs).

Why SWA is a plausible home for this: gemma-4 carries **two different KV
geometries in one model**, which is unusual and easy for a layout assumption to
get wrong.

| | SWA layers | full-attention layers |
|---|---|---|
| `key_length` / `value_length` | 256 | 512 |
| `head_count_kv` | 8 | 2 |

with `sliding_window = 1024` and a 5-SWA-then-1-full pattern over 30 layers
(`attention.sliding_window_pattern`). `n_embd_head_k(il)` switches on
`is_swa(il)` (`src/llama-hparams.cpp:117`), so any place that resolves the head
dim without the per-layer accessor reads at the wrong stride for 25 of the 30
layers. Ruled out as contributing: `attention.shared_kv_layers` is 0 for this
model, so cross-layer KV sharing is not involved.

## Reproducing

```bash
# broken
llama-server -m gemma-4-26B-A4B-it-UD-Q4_K_M.gguf -c 2048 --port 8099
# correct
llama-server -m gemma-4-26B-A4B-it-UD-Q4_K_M.gguf -c 2048 --port 8099 -ngl 0
# partially fixed
llama-server -m gemma-4-26B-A4B-it-UD-Q4_K_M.gguf -c 2048 --port 8099 --swa-full
```

Compare with logprobs rather than by eye - the failure is a token-selection
difference, and short confident prefixes agree even when the run is broken:

```bash
curl -s localhost:8099/v1/chat/completions -H 'Content-Type: application/json' \
  --data-binary '{"messages":[{"role":"user","content":"hi"}],"max_tokens":6,
                  "temperature":0,"top_k":1,"logprobs":true,"top_logprobs":5,"seed":1234}'
```

## Consequence for calibration

The calibration ladder measured gemma-4 at 39-48 tok/s against output it
believed was healthy. It could not have caught this: its degeneracy guard
(`common_moe_degeneracy_score`) only scores verbatim 4-gram repetition and
single-word dominance, and it only ever ran `probe_prompts[0]` - a long,
substantial instruction - because `n_solo_probes` is 1. Long prompts happen to
be the case where this failure is least visible. The actual user-visible break
was a bare `"hi"`.

Two changes landed for that gap (see `common/common.cpp`): a short-prompt
quality pass (`quality_prompts`, scored but never timed), and an output-fidelity
check that compares each substitution rung against a substitution-free
reference, with the tolerance measured from the model's own resampling rather
than hardcoded. Neither of those would have caught *this* bug either, since it
is present with substitution off - they close a different hole found alongside
it.
