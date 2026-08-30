# solid.cpp

<div align="center">

<img src="solid.cpp.jpeg" alt="solid.cpp" width="360"/>

**llama.cpp, hardened: real bugs found, root-caused, and measured before they ship**

Big MoE models on hardware people actually own, not a rack of enterprise GPUs — proven live, not just claimed.
A curated fork of [llama.cpp](https://github.com/ggml-org/llama.cpp): MoE expert-cache placement and
speculative-decoding fixes, each one measured before/after on real hardware, not assumed, plus a live Brain/Atlas
view so you can watch the cache work instead of taking it on faith.

**[See the numbers and full writeup → sangharshadhyeta.github.io/solid.cpp](https://sangharshadhyeta.github.io/solid.cpp/)**

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Based on llama.cpp](https://img.shields.io/badge/based%20on-llama.cpp-blue.svg)](https://github.com/ggml-org/llama.cpp)

</div>

## GLM-5.3-Flash on 42 GB of total memory

Ran the 93 GB UD-IQ1_S quant of GLM-5.3-Flash (320B total / 18B active) on an RTX 3060 12 GB + i5-12400F +
30 GB RAM — 42 GB combined, well under Unsloth's own stated practical minimum of ~102 GB for this model. It
loads and generates correctly: 1.39 tok/s prompt processing, 1.18 tok/s generation, all 45 layers' experts
CPU-resident (12,960 experts) with a 3.97 GB GPU expert cache running a 38.7% hit rate. We could not find a
published benchmark of this model running below its stated minimum footprint anywhere else — full breakdown
(Brain/expert-cache stats, placement, hardware) is at the top of the [docs page](https://sangharshadhyeta.github.io/solid.cpp/).

This was only possible after fixing a real bug: `llama.cpp` eagerly `MAP_POPULATE`s the entire model file into
RAM at load time regardless of whether it fits, which OOM-crashes any model bigger than available RAM even
when `-ncmoe` correctly keeps the CPU-offloaded experts lazily paged from disk otherwise. The fix makes that
prefetch conditional on the model actually fitting in available RAM, restoring genuine lazy NVMe-backed expert
weights for oversized models.

**Coming soon:** the same run on an NVIDIA H200 with 512 GB RAM, at Q4 — a fair-hardware comparison against
today's 1-bit-on-a-42GB-rig result.

### Reproduce it yourself

```sh
git clone -b solid https://github.com/sangharshadhyeta/solid.cpp.git
cd solid.cpp
cmake -B build -DGGML_CUDA=ON   # drop -DGGML_CUDA=ON for a CPU-only build
cmake --build build --config Release -j$(nproc)

./build/bin/llama-server \
  -hf unsloth/GLM-5.3-Flash-GGUF:UD-IQ1_S \
  -ncmoe 45 --moe-cache auto \
  -c 2048 --parallel 1 --host 0.0.0.0 --port 8099
```

The eager-mmap-prefetch fix (see below) means this loads and serves correctly even on a machine with under
64 GB of combined VRAM+RAM, where it would otherwise OOM at load time. Full build options are in
[docs/build.md](docs/build.md).
