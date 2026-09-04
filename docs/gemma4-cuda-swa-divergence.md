# Gemma-4 MoE inference is nondeterministic on CUDA

**Status:** open, well-bounded, not fixed. Found 2026-09-04.

> **Correction.** An earlier revision of this file named sliding-window
> attention as the cause, on the strength of `--swa-full` appearing to fix the
> first divergent token. That was wrong, and the reason it was wrong matters:
> every one of those comparisons was a *single sample*, and the underlying
> failure is nondeterministic. Re-tested properly, `--swa-full` is nondeterministic
> too - it just happened to match CPU on the run that was measured. Do not
> bisect this bug with single samples.

## Symptom

Four **identical** requests to the same gemma-4 server, greedy
(`temperature=0, top_k=1`), same seed, same prompt `"hi"`:

```
run1: ['<|channel>', 'thought', '\n', '<channel|>', 'thought', '\n', '<channel|>', 'thought']
run2: ['<|channel>', 'thought', '\n', '<channel|>', '\n\n', 'Hello', '!', ' How']
run3: ['<|channel>', 'thought', '\n', '<channel|>', 'thought', '\n', '<channel|>', '<channel|>']
run4: ['<|channel>', 'thought', ' even', ' than', 'ju', '-', '\n\n', 'thought']
```

Greedy decoding is deterministic by construction. Four different answers to one
question means the forward pass is returning different numbers each time.

User-visible, this looks like a quality problem - looping, leaked control
markers, `thought- own- own- own-` - which is why it was first mistaken for a
degenerate-sampling or chat-template issue. It is neither.

## Bounding it

| configuration | deterministic? | correct? |
|---|---|---|
| gemma-4, `-ngl 0` (pure CPU) | **yes**, 4/4 identical | yes |
| gemma-4, GPU, default | no, 4/4 differ | no |
| gemma-4, GPU, substitution off | no | no |
| gemma-4, GPU, `GGML_CUDA_MOE_CACHE=0` | no | no |
| gemma-4, GPU, `--swa-full` | no | no |
| gemma-4, GPU, `GGML_CUDA_DISABLE_GRAPHS=1` | no | no |
| gemma-4, GPU, `-fa off` | no | no |
| gemma-4, GPU, `--n-cpu-moe 99` (no expert copies) | no | no |
| qwen2.5-0.5b (dense), GPU, `-ngl 99` | **yes**, 4/4 identical | yes |
| qwen2.5-0.5b (dense), GPU, `-ngl 10` (CPU/GPU split) | **yes**, 4/4 identical | yes |

So it is not CUDA in general, not the CPU/GPU graph split in general, and not
any of the moe-cache machinery: it survives with the cache, substitution,
prefetch, D2D, flash attention and CUDA graphs all disabled, and with every
expert placed on the CPU. `test-backend-ops` passes clean against CPU on a free
GPU. What is left is MoE inference on the CUDA path for this model.

Also ruled out as contributing: `attention.shared_kv_layers` is 0 here, so
cross-layer KV sharing is not involved.

## Why nondeterminism points somewhere specific

Identical inputs producing different outputs is not a numerical-precision
story. Precision differences are *reproducible* - the same inputs give the same
slightly-different answer every time. Varying run to run means the computation
is reading memory whose contents are not fixed: either uninitialized memory, or
memory being written concurrently by something the reader is not ordered
against.

That makes the sparse expert-copy path and its async copies the natural place
to look next, even though disabling the cache and forcing experts to the CPU
did not clear it - those change *which* copies happen, not whether the split
path leaves regions of a device tensor unwritten between graph nodes.

## Structural notes on gemma-4

Possibly relevant to a layout/stride assumption; not yet tied to the bug.
Gemma-4 carries two KV geometries in one model:

| | SWA layers (25) | full-attention layers (5) |
|---|---|---|
| `key_length` / `value_length` | 256 | 512 |
| `head_count_kv` | 8 | 2 |

`sliding_window = 1024`, 5-SWA-then-1-full over 30 layers. `n_embd_head_k(il)`
switches on `is_swa(il)` (`src/llama-hparams.cpp:117`), so anything resolving a
head dimension without the per-layer accessor is wrong for five layers in six.
It also has per-layer scalar `layer_output_scale.weight` `[1]` tensors and a
`final_logit_softcapping` of 30.0.

## How to test this correctly

Repeat an identical greedy request at least 4 times and compare token
sequences. A single sample proves nothing here - short confident prefixes agree
even on broken runs, and any given run may match CPU by chance.

```bash
for i in 1 2 3 4; do
  curl -s localhost:8099/v1/chat/completions -H 'Content-Type: application/json' \
    --data-binary '{"messages":[{"role":"user","content":"hi"}],"max_tokens":8,
                    "temperature":0,"top_k":1,"logprobs":true,"seed":1234}' \
  | python3 -c "import sys,json;lp=json.load(sys.stdin)['choices'][0]['logprobs'];print([t['token'] for t in lp['content']])"
done
```

Two process-hygiene traps that produced false results while chasing this:

- `pgrep -f llama-server` matches the *shell running the pattern*, so a
  kill-then-relaunch script kills itself. Use `pgrep -x llama-server`.
- Killing only `head -1` of the matches, then waiting on `/health`, can leave a
  stale server holding the port while the new one dies on bind - so the
  measurement lands on the old config. Kill every match, wait for the port to
  clear, then verify the PID that owns the port is the one just launched.

## Consequence for calibration

Calibration measured gemma-4 at 39-48 tok/s against output it believed healthy.
It could not have caught this. Its degeneracy guard scores only verbatim 4-gram
repetition and single-word dominance, and `n_solo_probes` is 1, so the entire
quality verdict came from `probe_prompts[0]` - a long, substantial instruction,
the case where this failure is least visible. The user-visible break was a bare
`"hi"`.

Two changes landed for that gap (`common/common.cpp`): a short-prompt quality
pass (`quality_prompts`, scored but never timed) and an output-fidelity check
against a substitution-free reference, with the tolerance measured from the
model's own resampling rather than hardcoded. Neither would have caught *this*
bug - it is present with substitution off - but a repeat-identical-request
determinism check in calibration would, and does not exist yet.
