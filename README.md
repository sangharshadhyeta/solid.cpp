# solid.cpp

<div align="center">

<img src="solid.cpp.jpeg" alt="solid.cpp" width="360"/>

<b>llama.cpp, hardened: real bugs found, root-caused, and measured before they ship</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Based on llama.cpp](https://img.shields.io/badge/based%20on-llama.cpp-blue.svg)](https://github.com/ggml-org/llama.cpp)

</div>

## What this is

`solid.cpp` is a fork of [`llama.cpp`](https://github.com/ggml-org/llama.cpp) that tracks upstream directly and
layers on a curated set of performance and correctness fixes - triaged from real, reported issues, root-caused
in code, and validated with actual measurements before anything is kept. Not a rewrite, not a speed pun: every
change here is either a measured win or a documented, honest "no" (a fix that was tried, found not to be the
cause, and kept anyway if it was independently valid, or reverted if it wasn't).

The two starting focus areas were MoE expert-cache placement and speculative decoding (draft acceptance,
FR-Spec vocab trimming, MTP correctness), but the scope is the whole inference hot path - kernels, batching,
KV-cache, sampling, quantization - not just MoE. See [`docs/moe-cache-colibri-notes.md`](docs/moe-cache-colibri-notes.md)
for the full, honest working log: what was tried, what worked, what didn't, and why.

## Proven, not promised

The headline claim: MoE models too big for your VRAM don't just run here via CPU offload, they run
*fast* - measured, not assumed. Every row below is a real before/after on the same hardware (RTX 3060,
12GB VRAM - about as constrained as it gets for a 26B-parameter MoE model), same prompt, same config
except the one thing being measured, chat-templated and content-verified so the number reflects real
usage, not a synthetic benchmark.

**[See the live results page →](https://sangharshadhyeta.github.io/solid.cpp/)** for the full table
with methodology notes per row. Full raw numbers and working log in
[`docs/moe-cache-colibri-notes.md`](docs/moe-cache-colibri-notes.md).

| What | Before | After | Change |
|---|---|---|---|
| MoE expert-cache (vs `--moe-cache off`, the naive CPU-offload port) | 39.37 tok/s | 51.71 tok/s | **+31.3%** |
| Real placement optimum (vs a conservative `-ncmoe` guess) | 46.995 tok/s | 51.71 tok/s | **+10.0%** |
| `--moe-calibrate` empirical search (vs safe-floor placement) | 41.68 tok/s | 50.70 tok/s | **+21.6%** |
| MTP speculative decoding | 50.70 tok/s | 64.92 tok/s | **+28.1%** |
| LFRU eviction (vs plain LRU) | 63.8% hit rate | 65.0% hit rate | **+1.2pp** |
| Probabilistic draft acceptance (`--spec-prob-accept`, temp 0.7) | 53.6 tok/s | 55.7 tok/s | **+3.5%** |
| Backend sampling (`-bs`, concurrent load) | 103.6 tok/s | 107.5 tok/s | **+3.8%** |
| FR-Spec draft-vocab trim (measured, not assumed free) | 75.93 tok/s | 56.2 tok/s | **-26%** |

That last row is deliberate. FR-Spec was expected to be a free win and turned out not to be on this
hardware - so it's reported as a loss, not quietly dropped from the table. That's the actual point of
`solid.cpp`: every number here is something that was measured and would be reported honestly either way.
If a claimed improvement doesn't survive contact with a real benchmark, it doesn't ship, and if it does
ship, this is where you can check the receipts.

## What's different from upstream

- **MoE expert-cache** - GPU-resident hot-expert cache for CPU-offloaded MoE layers, with LFRU eviction
  (capped SLRU + heat tiebreak), live auto-placement, and `--moe-calibrate` for empirical, concurrency-aware
  placement search instead of a fixed guess.
- **Probabilistic draft acceptance** (`--spec-prob-accept`) - accepts a draft token whenever the target
  considers it at least as likely as the draft did (`min(1, p_target/p_draft)`), not only on exact match.
  Opt-in, strict superset of the old behavior - verified byte-identical when off.
- **FR-Spec-style MTP draft-vocab trimming**, with its real cost (not assumed-free) measured directly:
  a genuine trade of throughput and accept-rate for VRAM, not a free win - see the notes for the numbers
  and for the cache-related correctness bug that was found and fixed along the way.
- **`p_min` footgun warning** - a diagnostic (not a silent default change) that fires when a drafter's
  confidence early-stop is left disabled while draft width has been raised, mirroring the existing
  concurrency-cliff warning pattern.
- **Backend-sampling validation** - confirmed upstream's on-device sampling path (`-bs`) gives a real,
  positive throughput gain here too, measured directly rather than assumed from upstream's own numbers.
- **Diagnostic instrumentation kept permanently** - `LLAMA_DEBUG_VERIFY=1` gates zero-cost-when-unset
  traces (drafted-vs-target token mismatches, resolved speculative types, draft-context graph-reuse stats)
  that were directly responsible for finding more than one of the fixes above.
- Ongoing triage of upstream's open issue tracker for real, bounded, evidenced performance and correctness
  bugs - accepted, fixed properly, and measured; or investigated, found not to apply, and documented so the
  same dead end isn't walked twice. Several larger, correctly-scoped-but-not-yet-implemented designs (a
  CUDA many-expert MMVQ dispatch gate, among others) are recorded in the notes rather than rushed.

## Quick start

Build from source - see upstream's [build guide](docs/build.md), which still applies unchanged:

```sh
cmake -B build -DGGML_CUDA=ON   # or your backend of choice
cmake --build build --config Release -j
```

```sh
# Launch an OpenAI-compatible API server
./build/bin/llama-server -m <model.gguf> -ngl 99
```

## Supported backends

Unchanged from upstream - see [`docs/build.md`](docs/build.md) for the full table and per-backend build flags
(CUDA, HIP, Vulkan, Metal, SYCL, CANN, and more).

## Documentation

- [`docs/moe-cache-colibri-notes.md`](docs/moe-cache-colibri-notes.md) - the real working log for everything in
  "What's different from upstream": measurements, dead ends, and root causes, not just headline claims.
- Everything else in [`docs/`](docs/) is upstream's own documentation and still applies - build instructions,
  backend guides, server API, model support.

## Relationship to upstream

This repo's git history *is* `llama.cpp`'s history, plus the commits on top. Upstream changes are pulled in
directly rather than re-implemented; nothing here is intended to diverge from upstream's own architecture or
conventions more than a given fix requires. See upstream's own [README](https://github.com/ggml-org/llama.cpp)
and [manifesto](https://github.com/ggml-org/llama.cpp/discussions/205) for the base project this builds on.

## Acknowledgements

Inherited from upstream, still accurate:

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [stb-image](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [miniaudio.h](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
