#include "moe-cache.cuh"

#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)

extern "C" size_t ggml_moe_cache_trim(int device) {
    (void) device;
    return 0;
}

void ggml_moe_cache_register(const void * owner) {
    (void) owner;
}

#else

#include "common.cuh"
#include "mmvq.cuh"
#include "quantize.cuh"
#include "ggml-backend-impl.h"
#include "ggml-cuda.h"
#include "ggml-backend-moe-cache.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <functional>
#include <cerrno>
#include <chrono>
#if defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>
#endif
#include <climits>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <fstream>
#include <unordered_set>
#include <unistd.h>
#include <utility>
#include <vector>

#define MOE_CACHE_LOG(...) GGML_LOG_INFO(__VA_ARGS__)

enum class moe_cache_slot_state : uint8_t {
    free,
    copying,
    valid,
};

struct moe_cache_key {
    const void * tensor = nullptr;
    int32_t expert = -1;

    bool operator==(const moe_cache_key & other) const {
        return tensor == other.tensor && expert == other.expert;
    }
};

struct moe_cache_key_hash {
    size_t operator()(const moe_cache_key & key) const {
        uint64_t value = (uint64_t)(uintptr_t)key.tensor;
        value ^= value >> 33;
        value *= 0xff51afd7ed558ccdULL;
        value ^= (uint64_t)(uint32_t)key.expert * 0x9e3779b97f4a7c15ULL;
        value ^= value >> 29;
        return (size_t)value;
    }
};

// Track 1 steps 4a/4b (docs/plan.md): a real, queryable co-activation edge -
// which two experts were selected together (4a: same tensor, same routing
// decision) or in sequence (4b: layer L's top pick -> layer L+1's), rather
// than only an opaque combined hash. An opaque uint64 was the original
// shape of both counters and turned out to be a real bug, not just a
// missed nicety: a hash can't be reversed back into "layer 3 expert 12 ->
// layer 4 expert 47" for anything that wants to actually look at this data
// (a debug dump, an eventual export API, a visualization) - it was
// observational data that had already thrown away the one thing worth
// observing. Undirected for 4a (order doesn't mean anything within one
// routing decision - normalized so (a,b) and (b,a) collide); directed for
// 4b (from is genuinely earlier than to), so the constructors intentionally
// differ in whether they sort - see moe_cache_edge_undirected/
// moe_cache_edge_directed below.
struct moe_cache_edge {
    moe_cache_key from;
    moe_cache_key to;

    bool operator==(const moe_cache_edge & other) const {
        return from == other.from && to == other.to;
    }
};

struct moe_cache_edge_hash {
    size_t operator()(const moe_cache_edge & e) const {
        moe_cache_key_hash kh;
        size_t h = kh(e.from);
        h ^= kh(e.to) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        return h;
    }
};

static moe_cache_edge moe_cache_edge_undirected(moe_cache_key a, moe_cache_key b) {
    // Same tensor is assumed and asserted by every call site (within-layer
    // co-activation is only ever between two experts of the SAME tensor) -
    // ordering by expert index alone is then sufficient and avoids
    // comparing pointers for a within-tensor comparison.
    if (b.expert < a.expert) {
        std::swap(a, b);
    }
    return {a, b};
}

static moe_cache_edge moe_cache_edge_directed(moe_cache_key from, moe_cache_key to) {
    return {from, to};
}

// LFRU eviction, hybrid design: two O(1) doubly-linked lists (probation/
// protected_, the SLRU structure) PLUS a per-slot heat counter with
// periodic decay (the heat-based design). Measured separately first
// (A/B'd on the same mixed hot/cold workload, RTX 3060 + Gemma-4-26B-A4B,
// -ncmoe 15): plain LRU 63.8%, capped-SLRU-alone 65.0%, heat-alone 64.6%,
// uncapped-SLRU-v1 (the original broken attempt) 60.2-60.3%. Segments
// answer "has this ever proven itself hot" (binary, sticky); heat answers
// "how hot, and how recently" (graded, decaying) - they're complementary,
// not redundant: segments alone can't tell two protected slots apart when
// deciding who to demote at the cap (previously just "oldest", ignoring
// how often either was actually re-hit); heat alone has no admission
// control, so a burst of one-off cold admissions can still churn through
// slots that would otherwise deserve segment-level protection. This
// combines them: promotion to protected_ and eviction/demotion candidate
// selection are both segment-scoped (drain probation before protected_,
// same as capped-SLRU) but *within* whichever segment is being drawn
// from, the choice among a bounded recency window is by heat, not just
// recency order (see moe_cache_colder_enough below).
enum class moe_cache_segment : uint8_t { probation, protected_ };

struct moe_cache_slot {
    moe_cache_key key;
    uint64_t generation = 0;
    int prev = -1;
    int next = -1;
    int readers = 0;
    uint32_t heat = 0;
    moe_cache_slot_state state = moe_cache_slot_state::free;
    moe_cache_segment segment = moe_cache_segment::probation;
};

struct moe_cache_pool {
    size_t expert_size = 0;
    // Physical distance between consecutive slots in `slab`, always
    // >= expert_size. The gap (slot_stride - expert_size) is a per-slot
    // guard region, pre-filled with a canary pattern and never legitimately
    // written by anything - placement designed so that whatever the exact
    // mechanism behind a read/write landing outside its own slot's bounds
    // turns out to be, it lands in dead padding instead of a live
    // neighbour's weights. Checked cheaply after every dispatch (see
    // moe_cache_collect's guard scan) at zero extra sync cost.
    size_t slot_stride = 0;
    int wtype = -1;
    char * slab = nullptr;
    int n_slots = 0;

    std::vector<moe_cache_slot> slots;
    std::vector<int> free_slots;
    std::unordered_map<moe_cache_key, int, moe_cache_key_hash> map;
    // Resident-set bitmask, one bit per expert, keyed by tensor. Exactly the
    // same information `map` already holds - kept alongside it because the two
    // are read very differently. `map` answers "which slot holds this one
    // expert" and costs a hash per question, which is the right shape when the
    // caller asks about the 8 experts the router picked. Substitution has to
    // ask about a whole candidate window (~64 by measurement) in one decision,
    // and 64 hashes on the dispatch path is precisely the cost Track 1.5
    // exists to remove; against the mask that is one hash for the tensor and
    // then 64 bit tests over ~4 words, which stay in L1.
    // Maintained only through moe_cache_map_insert/erase, under the same lock
    // as `map`, so the two cannot drift; moe_cache_mask_audit() proves it.
    std::unordered_map<const void *, std::vector<uint64_t>> resident_mask;

    // Canary allocated immediately after the slab, filled with a known pattern.
    // If a fill writes past its slot the canary is the first thing after the
    // slab in the allocator's address space, so it catches an overrun that the
    // in-bounds arithmetic cannot see. GGML_CUDA_MOE_CACHE_CANARY=1.
    char * canary = nullptr;
    size_t canary_bytes = 0;
    int lru_head = -1;       // probation segment (new/one-off admissions)
    int lru_tail = -1;
    int protected_head = -1; // protected segment (proven hot: re-requested at least once while resident)
    int protected_tail = -1;
    int protected_count = 0; // capped at n_slots * MOE_CACHE_PROTECTED_CAP_PCT / 100 - see moe_cache_promote_to_protected
    long long fills_since_decay = 0;
    // Slow-moving reference scale for this pool's own heat magnitudes -
    // updated on every heat step and halved alongside moe_cache_pool_decay,
    // so it stays proportional without a separate full-pool scan. Lets the
    // align-protect boost's cap be "a fraction of what heat actually looks
    // like in this pool right now" instead of a fraction of the theoretical
    // MOE_CACHE_HEAT_MAX ceiling, which a real pool rarely gets near - see
    // moe_cache_weighted_heat.
    double heat_scale_ema = 4.0; // seeded at MOE_CACHE_HEAT_STEP - avoids a 0-scale window before the first real step
};

struct moe_cache_shape {
    size_t expert_size = 0;
    int wtype = -1;
    int64_t n_expert = 0;
    int64_t n_tensors = 0;
    int pool = -1;
    bool finished = false;
};

struct moe_cache_seen_tensor {
    size_t bytes = 0;
    size_t expert_size = 0;
    int wtype = -1;
};

enum class moe_cache_job_kind : uint8_t {
    fill,           // ordinary admission: contiguous copy of job.bytes from job.source
    reduce_convert, // heat-aware neuron subsetting: gather job.key's top-K rows (see
                     // device.reduced_indices) from the SOURCE tensor into job's
                     // destination slot instead of a single contiguous copy - see
                     // the gather branch in moe_cache_worker.
};

struct moe_cache_job {
    int pool = -1;
    int slot = -1;
    uint64_t generation = 0;
    moe_cache_key key;
    const void * source = nullptr;
    size_t bytes = 0;
    // The whole expert tensor this expert lives in. cudaHostRegister has to be
    // handed a page-aligned range, and adjacent experts share boundary pages,
    // so registration is done per tensor rather than per expert.
    const void * region_base = nullptr;
    size_t region_bytes = 0;
    moe_cache_job_kind kind = moe_cache_job_kind::fill;
};

struct moe_cache_demand {
    uint16_t count = 0;
    size_t expert_size = 0;
};

struct moe_cache_config {
    bool enabled = true;
    bool automatic = true;
    size_t budget_mb = 0;
    // 0 = not explicitly set by the user; moe_cache_prepare_budget() computes
    // a live default from actual free VRAM at that point instead of trusting
    // a fixed constant that can't be right across both a 12GB card and a
    // 141GB one. GGML_CUDA_MOE_CACHE_RESERVE_MB still overrides this exactly
    // as before when explicitly set.
    size_t reserve_mb = 0;
    // Measured, not a round-number guess: Ornith-1.5-35B-A3B (256 experts,
    // Qwen3.5-MoE-style) has 840 KiB per expert - below a naive 1 MiB floor,
    // which silently rejected every expert tensor at moe_cache_begin() with
    // zero log output anywhere. The cache session initialized, reported
    // "explicit MoE cache mode" success, and then did nothing for the entire
    // server lifetime - no hits, no misses, no expert-map data, and no signal
    // to the operator that anything was wrong. GLM-5.2 (256 experts/layer) is
    // in the same size class and would hit the identical silent failure.
    // 256 KiB stays well clear of real many-small-expert architectures while
    // still excluding genuinely too-small slabs (the reason this floor exists
    // at all - per-slot bookkeeping/fragmentation overhead dominating).
    size_t min_expert_bytes = 256u << 10;
    // Default is derived live from g_max_batch_hint (the real n_seq_max of
    // the context, set before session_create() runs) - see
    // moe_cache_read_config(). Floor of 8 so ordinary prefill/decode batches
    // engage the cache even with no hint given; ceiling of
    // MOE_CACHE_MAX_BATCH_CEILING (64) because that's a real buffer-size
    // limit, not a conservative guess. GGML_CUDA_MOE_CACHE_MAX_BATCH
    // overrides this exactly as before when explicitly set.
    int max_batch = 8;
    int inserts_per_plan = 8;
    int admit_after = 2;
    int readmit_after = 8;
    int queue_max = 128;
    size_t queue_mb = 512;
    int stats_every = 0;
    // Page-lock up to this many MiB of expert weights so the fill DMA can read
    // them directly. 0 = off (default): pinning is unreclaimable, so opting in
    // is the caller's decision. Measured: staged copy 506 us/expert vs direct
    // pinned DMA 303 us, on a 24.5 GB/s PCIe 4.0 x16 link.
    size_t hostreg_mb = 0;
    int max_devices = INT_MAX;
    int min_compute_capability = 700;
    bool serial_fill = true;
    std::string fail_stage;
};

struct moe_cache_session;

struct moe_cache_scratch {
    size_t ids = 0;
    size_t act = 0;
    size_t q8 = 0;
    size_t out = 0;
};

// Ping-pong pair of full-tensor-sized device buffers for one shape (see
// moe_cache_device::prefill_slabs). Which of the two slots a given tensor
// lands in is decided ONCE, the first time that tensor is ever registered
// (assigned_slot, alternating 0/1 by registration order), and never changes
// again - NOT a runtime-toggling counter. A stable, tensor-to-slot mapping
// means wait() can look a tensor's slot up directly instead of scanning
// both, and means the *same* slot (and its ready/consumed events) is always
// the one associated with a given logical layer's data across every future
// call - relevant because, empirically, llama.cpp's CUDA graph capture for
// this op_offload path captures roughly once per layer rather than once per
// ubatch (see the design note above moe_cache_prefill_advance), so
// "advance now, join a couple of layers later" almost always crosses a
// capture boundary. `inflight_host`/`inflight_bytes` record which tensor is
// actually resident in each slot so a wait() call can tell a stale copy from
// a fresh one instead of trusting the caller's bookkeeping alone.
//
// `consumed[slot]` closes a real race: without it, prefetching layer L+2 into
// this same slot (the ping-pong period is 2) could start overwriting the
// buffer via cudaMemcpyAsync on prefill_stream while the GEMM reading layer
// L's data is still in flight - and since that GEMM dispatch may belong to a
// different capture than the one advance() is being called from now, this
// wait is a real host sync (cudaEventSynchronize), not a captured
// cudaStreamWaitEvent - see the design note above moe_cache_prefill_advance.
// It costs nothing when the consumer is already done (the overwhelmingly
// common case once prefetch is running one full layer ahead of compute).
// consumed_valid distinguishes "never used yet" (no wait needed) from "really
// has to wait" without a sentinel event of unknown initial state.
struct moe_cache_prefill_slab {
    void * dev[2] = {nullptr, nullptr};
    size_t cap[2] = {0, 0};
    cudaEvent_t ready[2] = {nullptr, nullptr};
    cudaEvent_t consumed[2] = {nullptr, nullptr};
    bool consumed_valid[2] = {false, false};
    const void * inflight_host[2] = {nullptr, nullptr};
    size_t inflight_bytes[2] = {0, 0};
    // Fixed slot assignment per tensor (keyed by the tensor's own host
    // pointer, i.e. what advance() calls next_host_base and wait() calls
    // host_base) - see the struct comment above for why this has to be
    // decided once and stay fixed rather than toggling. Populated by
    // register_successor at build time; assign_next alternates 0/1 as each
    // new distinct tensor is first seen for this slab.
    std::unordered_map<const void *, int> assigned_slot;
    int assign_next = 0;
};

// Each expert's measured topic-affinity position (see ggml_moe_cache.set_atlas).
// At namespace scope rather than nested in moe_cache_device because the Track
// 1.5 warm request below has to name it, and that request has to be a complete
// type before the device that queues it.
struct moe_cache_atlas_cell { float x = 0.0f, y = 0.0f, spec = 0.0f; };
using moe_cache_atlas_row = std::vector<std::pair<int32_t, moe_cache_atlas_cell>>;

// Track 1.5 (docs/plan.md): one deferred atlas-warming pass, handed from the
// dispatch path to the fill worker. Everything the ranking scan needs is
// captured here at enqueue time - the atlas row by shared_ptr (so a
// concurrent set_atlas cannot free it mid-scan) and the request direction by
// value (so the worker ranks against where the request was heading when the
// hint was raised, not wherever the centroid has drifted to by the time it
// gets there).
struct moe_cache_warm_request {
    std::shared_ptr<const moe_cache_atlas_row> row;
    int pool_index = -1;
    const void * host_base = nullptr;
    size_t expert_size = 0;
    int64_t n_expert = 0;
    float req_dir_x = 0.0f;
    float req_dir_y = 0.0f;
};

// Stage A of the live-activation predictor (docs/plan.md): one real
// (hidden-state, actually-selected-experts) example, captured at the
// EXISTING mul_mat_id dispatch hook - by the time this fires the router's
// decision (ids) is already made, so this trains a predictor for FUTURE
// dispatches, it cannot change this one. `acts` is a COPY (the source
// tensor's data is not guaranteed to outlive this dispatch call once the
// hot path moves on), one token only per request - see moe_cache_train's
// comment for why sampling one token per dispatch is enough.
struct moe_cache_train_request {
    const void * host_base = nullptr;
    int64_t n_expert = 0;
    int pool_index = -1;
    size_t expert_size = 0;
    std::vector<float> acts;      // hidden_dim floats, this token's input
    std::vector<int32_t> ids;     // n_ids_per_token selected experts, this token's label
};

struct moe_cache_device {
    explicit moe_cache_device(int physical) : physical(physical) {}

    int physical;
    std::atomic<bool> dead{false};
    std::mutex dispatch_mu;

    std::vector<std::unique_ptr<moe_cache_pool>> pools;
    std::vector<moe_cache_shape> shapes;
    std::unordered_map<const void *, moe_cache_seen_tensor> seen_tensors;
    // host_base -> stable tensor name, for cross-restart persistence keying
    // (see moe_cache_begin's assignment site and moe_cache_predictor_save/
    // _load). host_base itself is a fresh pointer every process restart.
    std::unordered_map<const void *, std::string> tensor_names;

    // Step 0 of Atlas-driven cache warming (see ggml_moe_cache.set_atlas) -
    // each expert's measured topic-affinity position, and a live, decaying
    // centroid of where the current request is trending in that same space.
    // Observational only right now: updated on real demand hits, nowhere
    // yet read to make a promotion or eviction decision.
    using atlas_cell = moe_cache_atlas_cell;
    std::unordered_map<moe_cache_key, atlas_cell, moe_cache_key_hash> atlas;
    // Secondary index for the warming scan (Track 1 step 3): "every expert
    // atlas covers for this tensor", so ranking candidates for one layer
    // doesn't mean scanning the whole model's atlas data on every attempt.
    // Held behind a shared_ptr for Track 1.5: the async ranking pass takes a
    // reference to one row in O(1) while holding the session lock and then
    // scans it with the lock released, so set_atlas has to publish a whole
    // new row rather than mutate one a worker may be walking.
    using atlas_row = moe_cache_atlas_row;
    std::unordered_map<const void *, std::shared_ptr<const atlas_row>> atlas_by_tensor;
    // Track 1.5: deferred warming passes awaiting the fill worker. Bounded to
    // one entry by moe_cache_atlas_warm_enqueue - see the comment there for
    // why dropping, not queueing, is right when the worker is behind.
    std::deque<moe_cache_warm_request> warm_queue;
    // Stage A live-activation predictor: same "bounded to one, drop rather
    // than queue" shape as warm_queue above, and the same reason - a stale
    // training example is not wrong to skip, and letting requests pile up
    // would mean the worker falls further behind under load instead of
    // recovering. GGML_CUDA_MOE_CACHE_TRAIN_PREDICTOR=1, off by default.
    std::deque<moe_cache_train_request> train_queue;
    // One flat row-major (n_expert x hidden_dim) weight matrix per tensor,
    // lazily sized on that tensor's first training example (hidden_dim isn't
    // known before then). Logistic-regression-style online updates only -
    // see moe_cache_train_service. Bias-free by design for this first cut:
    // mean-centering the input would need a running mean, which is one more
    // thing that can be wrong before this mechanism has proven the basic
    // shape works at all.
    std::unordered_map<const void *, std::vector<float>> predictor_weights;
    // Set once per tensor, from that tensor's first training example's own
    // acts.size() (raw activation width + MOE_CACHE_TRAIN_CONTEXT_DIMS).
    // Genuinely PER-TENSOR, not a single model-wide scalar: ffn_gate_exps/
    // ffn_up_exps operate on the residual stream's hidden size, but
    // ffn_down_exps operates on the FFN's intermediate width, a different
    // (usually larger) dimension - caught empirically (2026-08-29) when a
    // single shared scalar corrupted the persisted file's per-tensor
    // n_expert reconstruction (w.size()/hidden_dim came out wrong for every
    // tensor using the wrong tensor's width). Needed at save time to
    // recover n_expert from a flat weight vector's own length, since
    // n_expert isn't otherwise stored per-tensor on the device.
    std::unordered_map<const void *, int64_t> predictor_hidden_dim;
    long long predictor_steps = 0;
    // Sampled, cheap-to-compute running accuracy: was the predictor's single
    // top-scoring expert actually IN the real selected set for that step?
    // Same purpose as the atlas confidence-evidence pattern elsewhere in
    // this file - visible, live proof the mechanism is or is not learning
    // anything, not just silent weight updates nobody can verify.
    long long predictor_top1_hits = 0;
    // Self-tuning outcome evidence for predictor-driven ADMISSION
    // specifically (distinct from predictor_top1_hits above, which measures
    // prediction accuracy, not whether admitting on that prediction actually
    // paid off) - same (for, against) Bayesian-prior shape as the atlas's
    // own atlas_align_evidence_for/against, so the predictor's own
    // aggressiveness can scale off its measured real payoff instead of a
    // hand-tuned constant, the same way align_protect_strength already
    // does. predictor_admitted tracks which (tensor,expert) keys are
    // currently believed resident because predictor warming put them there
    // - a hit on one is "for" evidence, an eviction of one that was never
    // hit is "against" - see moe_cache_predictor_admit_confidence and its
    // recording sites.
    std::unordered_set<moe_cache_key, moe_cache_key_hash> predictor_admitted;
    // Brain UI substitute overlay (see moe_cache_get_substitute_map) - which
    // (tensor,expert) keys served as a resident stand-in since the last
    // poll. Read-and-cleared by that getter, not accumulated forever.
    std::unordered_set<moe_cache_key, moe_cache_key_hash> substituted_recently;
    double predictor_admit_evidence_for = 1.0;
    double predictor_admit_evidence_against = 1.0;
    // Cross-restart persistence staging (see moe_cache_predictor_load_file/
    // _save). Loaded once, lazily, on this device's first training example -
    // name-keyed (host_base is not stable across restarts) flat weight
    // vectors read from GGML_CUDA_MOE_CACHE_TRAIN_STATE_FILE, consulted
    // whenever a tensor is trained for the first time this process instead
    // of zero-initializing it.
    std::unordered_map<std::string, std::vector<float>> predictor_loaded_by_name;
    bool predictor_state_load_attempted = false;
    // Track 1.5 measurement (GGML_CUDA_MOE_CACHE_ATLAS_WARM_TIMING=1): how
    // long the warming block costs *on the dispatch path* specifically.
    // That is the quantity this track is about, and tok/s cannot resolve it
    // - the block runs once every 64 plan() calls, so even a large
    // per-call saving is a rounding error at the throughput level. Measured
    // directly instead of inferred, same reason step 6/7a were measured
    // before anything was built on them.
    long long warm_path_ns = 0;
    long long warm_path_calls = 0;
    float req_dir_x = 0.0f;
    float req_dir_y = 0.0f;
    bool  req_dir_valid = false;
    // Real routing decisions folded into req_dir since it became valid.
    // req_dir SNAPS to a single expert's raw atlas position on the very
    // first update (see the update site) - one noisy sample, not yet an
    // average of anything. Admission gates on this reaching a minimum count
    // (moe_cache_atlas_warm_min_samples) before acting, so it cannot spend
    // a fresh session's scarce initial free-slot budget on a prediction
    // built from a single data point; the EMA itself still starts tracking
    // immediately, since observation alone is free and harmless.
    uint32_t req_dir_updates = 0;
    // Live pacing signal for dynamic substitution gating (see
    // moe_cache_substitute_min_rank below). Was originally an EMA of the
    // wall-clock gap between consecutive moe_cache_plan() calls - abandoned
    // (2026-09-04) after live testing showed it barely ever opened the
    // gate despite decode running at ~1 tok/s: cudaMemcpyAsync does not
    // block the host, so successive dispatch calls race through enqueueing
    // work without ever waiting for it, and the gap between them stays tiny
    // regardless of how backed up the GPU-side fetch pipeline actually is.
    // Replaced with moe_cache_track_fetch_pace's real GPU-side completion
    // timing (issue-to-actual-finish, via a stream callback on every real
    // H2D fetch) - the only thing that genuinely reflects pipeline backlog.
    // That callback runs on a CUDA callback thread, not the dispatch
    // thread, so this field needs its own lock rather than relying on
    // dispatch_mu: hence substitute_pace_mu, touched by nothing else.
    // Two EMAs, same fast/slow-trend SHAPE as the atlas req_dir/req_vel pair
    // above, but their decay is derived from the standard N-period EMA
    // formula (alpha = 2/(N+1), see moe_cache_substitute_pace_alpha_fast/
    // _baseline) rather than borrowed from that UI-tuned tracker:
    // substitute_pace_ema_ms (fast) is "right now"; substitute_pace_baseline_ms
    // (slow, a fixed fetch-count multiple of the fast window) is "what's
    // normal on this hardware/model" - self-calibrated from real
    // observations instead of a hardcoded latency figure, so the gate
    // adapts to whatever machine and model it's actually running on.
    // substitute_pace_samples gates the baseline's validity (against
    // GGML_CUDA_MOE_CACHE_SUBSTITUTE_PACE_MIN_SAMPLES in
    // moe_cache_substitute_min_rank) so an unwarmed baseline of 0 can't make
    // the very first fetches look infinitely-worse-than-normal.
    std::mutex   substitute_pace_mu;
    double       substitute_pace_ema_ms      = 0.0;
    double       substitute_pace_baseline_ms = 0.0;
    uint32_t     substitute_pace_samples     = 0;
    // Rate limit for the warming scan below - same clock spec_evict already
    // uses (device.collect_calls), so this doesn't need its own timer.
    long long atlas_warm_last_calls = -1;
    // Burst detection: req_dir_x/y as of the last warm-throttle check, so
    // each check measures how far the centroid moved SINCE then (a running
    // step-wise distance, not distance from a fixed origin - this is what
    // lets sustained real movement accumulate past the threshold while small
    // noise/jitter between checks does not). See moe_cache_atlas_burst_check.
    float atlas_burst_ref_x = 0.0f;
    float atlas_burst_ref_y = 0.0f;
    bool  atlas_burst_ref_valid = false;
    // Small, deduped, most-recent-first record of tensors moe_cache_plan has
    // actually been called for - not a generic tensor->pool resolver (this
    // cache has none), just enough to let a detected burst warm several
    // DISTINCT tensors (a real topic shift needs correction across many
    // layers, not the one tensor whichever plan() call happened to trip the
    // throttle) using pool/size info the calling code already had in hand
    // when each of those tensors was planned. pool_index only, never a raw
    // pool*, so it's re-resolved through device.pools at use time - same
    // defensive re-validation moe_cache_atlas_warm_service already uses.
    struct atlas_recent_tensor {
        int pool_index; const void * host_base; size_t expert_size; int64_t n_expert;
    };
    std::deque<atlas_recent_tensor> atlas_recent_tensors;
    // Persisted between throttle checks so the interval used to schedule the
    // NEXT check can reflect what the LAST check found - the check itself
    // only runs once per interval, so this is what lets a detected burst
    // shorten its own follow-up interval instead of only speeding up
    // starting from the next steady-interval boundary.
    bool atlas_bursting = false;
    // Continuous burst confidence in [0,1], replacing the discrete
    // atlas_bursting bool as the thing rank_k/admit_k/interval actually key
    // off of - a hard 0.25-distance threshold snaps steady<->burst budgets
    // right at the boundary, which can flip every other check for a
    // centroid drifting near the edge. This ramps toward 1 on a threshold
    // crossing and decays otherwise, so the budgets interpolate instead of
    // discontinuing (see moe_cache_atlas_burst_confidence).
    float atlas_burst_confidence = 0.0f;

    // Holt's linear trend (double exponential smoothing): req_dir_x/y above
    // is the "level" term (already an EMA); these are the "trend" term - an
    // EMA of the level's own per-update delta, i.e. a smoothed estimate of
    // which way and how fast the live topic centroid is currently moving.
    // level + horizon*trend is a genuine forward extrapolation, not just a
    // lagging average of where routing already went - see
    // moe_cache_atlas_lookahead, the thing that makes ranking answer "where
    // is this heading" instead of only "where has this been".
    float req_vel_x = 0.0f;
    float req_vel_y = 0.0f;

    // Evidence pool for the align-protect boost's confidence-weighted
    // strength (Kalman-gain style: the multiplier's real-world influence is
    // computed from this session's own measured track record, not a fixed
    // hand-picked constant). Two independent streams feed the same (for,
    // against) pair - see moe_cache_atlas_align_confidence for how each is
    // recorded: outcome evidence (did an aligned, boosted slot actually get
    // reused before eviction, or evicted without ever being reused) and
    // shadow evidence (collected for free during every real eviction scan:
    // did the boost change which slot got picked, or did plain unboosted
    // LFRU already agree). Bayesian (1,1) prior so confidence starts at 0.5
    // of the configured ceiling before any evidence exists, not 0 or the
    // full ceiling.
    double atlas_align_evidence_for = 1.0;
    double atlas_align_evidence_against = 1.0;

    // Track 1 step 4a (docs/plan.md): within-layer co-activation - which
    // experts get selected *together* in the same real routing decision,
    // independent of topic label. Deliberately scoped to within one plan()
    // call's own ids[] (same tensor, same decision) rather than across
    // layers/tokens: a true cross-layer pathway signal needs a token-
    // boundary signal this cache doesn't have yet (nothing currently tells
    // it "a new token's forward pass just started" vs. "still the same
    // token, next layer") - inventing a heuristic for that risked the same
    // class of bug the bounds-check fix above just caught for real, so it's
    // left as an explicit prerequisite rather than guessed at. Observational
    // only, same as req_dir_* - nothing reads this yet. Keyed by a real
    // moe_cache_edge (see the struct comment there for why this was
    // switched from an opaque hash - that was a real bug, not a style
    // choice), undirected.
    std::unordered_map<moe_cache_edge, uint32_t, moe_cache_edge_hash> co_activation;
    // Per-neuron accumulated |value| for gate_exps/up_exps matmul outputs,
    // keyed by the SAME moe_cache_key as everything else here (tensor +
    // expert) - the intra-expert analogue of co_activation above, one level
    // deeper. Purely observational for now, gated off by default
    // (GGML_CUDA_MOE_CACHE_NEURON_HEAT): accumulated in moe_cache_collect()
    // from device.h_out, which that function already downloads and
    // validates for every gate_exps/up_exps hit - no new CUDA kernel, no
    // new sync, nothing read here changes any existing behavior. Vector
    // length is n_out (moe_intermediate_size) for whichever tensor this
    // key belongs to; lazily sized to that on first sight of a key, since
    // gate/up and down share the key type but not the width.
    std::unordered_map<moe_cache_key, std::vector<float>, moe_cache_key_hash> neuron_heat;
    // Dedicated lock for neuron_heat, deliberately NOT session.mu: that
    // mutex is the cache's core bookkeeping lock (fill worker, eviction,
    // plan()/dispatch() accounting all contend for it constantly). The
    // first cut of the neuron_heat accumulator reused session.mu and
    // measured a real regression (~50-60 tok/s -> ~4 tok/s, GPU util
    // dropped to 0% while CPU held steady) - not from the accumulation
    // work itself being slow, but from adding a new, frequent (up to a
    // few hundred times per token) contender for an already-hot shared
    // lock. A separate mutex, touched by nothing else, removes that
    // contention entirely without changing what's actually synchronized.
    std::mutex neuron_heat_mu;
    // Fast-path mirror of reduced_indices' keyset, guarded by neuron_heat_mu
    // (NOT session.mu) - deliberately separate from reduced_indices itself.
    // The first version of heat-aware neuron subsetting called
    // moe_cache_try_reduce_convert (which takes session.mu unconditionally)
    // from moe_cache_collect for EVERY gate/up hit, on the theory that an
    // early return for an already-decided key would be cheap - measured
    // this directly and it was not: ~35.6 -> ~29.9 tok/s (-16%), the same
    // shape of regression neuron_heat_mu itself already exists to prevent,
    // reintroduced by an extra session.mu acquisition on the hot collect()
    // path for a case (already decided) that is by far the common one after
    // the first handful of tokens. This set is checked FIRST, under the
    // already-held neuron_heat_mu (collect() holds it for the heat
    // accumulation loop right above), so the common "nothing to do" case
    // never touches session.mu at all - only a genuinely new decision
    // (rare: bounded by the number of distinct experts that ever qualify,
    // not by token count) pays for it. Inserted into (alongside
    // reduced_indices) at the end of every moe_cache_try_reduce_convert
    // call that reaches a decision, whether or not that decision converts
    // anything.
    std::unordered_set<moe_cache_key, moe_cache_key_hash> reduce_decided;
    // Per-expert reduced-width neuron index set, once converted (see
    // moe_cache_select_reduced_indices / moe_cache_try_reduce_convert).
    // Protected by session.mu, NOT neuron_heat_mu - it's read/written from
    // the fill/admission path, which already holds session.mu throughout,
    // the same as pool/slot state. When code needs both (deciding whether
    // to convert requires reading neuron_heat), the lock order is always
    // session.mu first, then neuron_heat_mu - the same order
    // moe_cache_neuron_heat_report already uses from
    // moe_cache_session_destroy, so this doesn't introduce a new ordering
    // to get wrong.
    //
    // Presence of a key here is itself the "decided" marker for the
    // convert-once policy in moe_cache_try_reduce_convert: once a key is
    // inserted (even if the conversion below it never actually completes -
    // no free slot, no budget headroom), it is never re-evaluated for the
    // rest of the session. idx is the chosen row set (ascending); the other
    // two fields exist because a reduced-pool slot cannot recover them from
    // its own pool the way a primary slot can (pool.expert_size there is
    // the REDUCED size, not the tensor's real per-expert byte size) - both
    // moe_cache_invalidate_session (to compute the correct source address
    // range) and the fill worker's reduce_convert gather (to know the
    // source stride and per-row size) need the ORIGINAL, full-width values.
    struct reduced_entry {
        std::vector<int32_t> idx;
        size_t src_expert_size = 0; // full (unreduced) per-expert byte size in the source tensor
        size_t row_bytes = 0;       // bytes per selected row = src_expert_size / n_out
        int pool_index = -1;        // index into device.pools of the reduced-width pool
    };
    std::unordered_map<moe_cache_key, reduced_entry, moe_cache_key_hash> reduced_indices;
    // Bounded adjacency list, for coverage-aware eviction. co_activation is
    // keyed by EDGE, which answers "do these two co-fire" in O(1) but cannot
    // enumerate a given expert's partners without scanning every edge. Eviction
    // needs the enumeration ("how many of this expert's partners are still
    // resident"), so the neighbours are mirrored here as they are recorded.
    // Capped per expert: redundancy only needs a representative sample, and an
    // unbounded list would grow with the square of the expert count.
    std::unordered_map<moe_cache_key, std::vector<int32_t>, moe_cache_key_hash> partners;

    // Track 1 step 4b: the missing prerequisite step 4a's comment above
    // flagged - a real token/batch-boundary signal. Reset in
    // moe_cache_session_enter, which is already called exactly once per
    // real ggml_backend_sched_compute_splits invocation (one graph compute
    // = one forward pass/batch) via the existing moe_cache_scope RAII in
    // ggml-backend.cpp - no new API needed, this reuses a call site that
    // already exists and is already exercised by every session. Imperfect
    // for a multi-sequence batched compute call (mixes different
    // sequences' tokens into one shared "last layer" state for that call),
    // same granularity limitation req_dir_x/y already has as a single
    // per-device value in multi-slot serving - not pretending to be more
    // precise than that.
    moe_cache_key last_top_expert{};
    int  last_top_layer = -1;
    bool last_top_expert_valid = false;

    // Track 1 step 6 (docs/plan.md): does the cross-layer co-activation
    // table actually PREDICT the next layer's pick? Router-lookahead's own
    // measured precision is 59.3% / 48.0% / 41.1% at depth 1/2/3, and it
    // pays a real n_embd x n_expert matmul per layer to get it; a table
    // lookup is nearly free, so the question is whether it is competitive.
    // successor_best is an O(1) index (from-expert -> its highest-count
    // successor so far) maintained alongside the edge counts, so scoring a
    // prediction never scans the full edge map. All of this is gated - see
    // moe_cache_measure_pred_enabled - and costs nothing when off.
    std::unordered_map<moe_cache_key, std::pair<moe_cache_key, uint32_t>, moe_cache_key_hash> successor_best;
    long long xlayer_pred_total = 0;
    long long xlayer_pred_hit   = 0;

    // Step 7a measurement: a different question from step 6. Not "which
    // expert comes next across layers" (refuted, 33.8%) but "given one
    // expert in a routing decision, is its most frequent partner also
    // selected in that SAME decision". That is what group-aware admission
    // would rely on: if A and B always co-fire, admitting A alone
    // guarantees a miss on B, independent of how well anything predicts.
    // partner_best mirrors successor_best but for the undirected
    // within-layer table, updated on both endpoints of each edge.
    std::unordered_map<moe_cache_key, std::pair<moe_cache_key, uint32_t>, moe_cache_key_hash> partner_best;
    // How often each expert was selected at all. partner_best holds a raw
    // co-occurrence COUNT, which says nothing on its own - a partner seen
    // twice looks identical to one seen 500 times. Dividing by this turns
    // it into P(partner | anchor fired), which is what a confidence
    // threshold actually needs.
    std::unordered_map<moe_cache_key, uint32_t, moe_cache_key_hash> expert_fire_count;
    // Consecutive copy failures. A cache that cannot copy is not a slow
    // cache, it is a source of wrong weights, so past a small threshold it
    // disables itself rather than continuing to half-work.
    std::atomic<int> copy_failures{0};
    long long group_admits = 0;
    long long partner_pred_total = 0;
    long long partner_pred_hit   = 0;
    long long partner_chance_num = 0; // sum of (n_ids-1), for the chance-level baseline
    long long partner_chance_den = 0; // sum of n_expert
    // Keyed by a real, directed moe_cache_edge (from = earlier layer) -
    // same fix as co_activation above, same reason.
    std::unordered_map<moe_cache_edge, uint32_t, moe_cache_edge_hash> co_activation_cross_layer;

    // Decay rate for the centroid EMA, derived the same way as the
    // substitution-pacing EMAs (moe_cache_substitute_pace_alpha_fast/
    // _baseline): the standard N-period formula, alpha = 2/(N+1), not an
    // independently eyeballed constant. N=8 real routing decisions - enough
    // to smooth single-decision noise while still reacting within a small
    // fraction of one token's worth of atlas-covered picks.
    static constexpr float MOE_CACHE_ATLAS_DECAY = 2.0f / (8.0f + 1.0f); // = 0.2222
    // Trend (Holt's linear smoothing) decay - same formula, N=16 (2x the
    // level's window), since a per-update delta is a second-order quantity
    // and noisier than the level itself; over-reacting to it would make the
    // lookahead point whip around on single-update jitter instead of
    // tracking a real sustained direction of movement. See req_vel_x/y and
    // moe_cache_atlas_lookahead.
    static constexpr float MOE_CACHE_ATLAS_TREND_DECAY = 2.0f / (16.0f + 1.0f); // = 0.1176
    std::unordered_map<moe_cache_key, moe_cache_demand, moe_cache_key_hash> demand_count;
    // Per-CPU-resident-tensor last-use tracking, for the MADV_COLD sweep. An
    // expert that is neither in the VRAM cache nor recently selected is pure
    // page-cache ballast; telling the kernel so lets it reclaim those pages
    // ahead of anything still in use. Indexed by expert so the hot path is an
    // array store, not a hash insert.
    struct cpu_residency {
        size_t expert_size = 0;
        std::vector<uint32_t> last_seen; // coarse seconds, 0 = never selected
        std::vector<bool> is_cold;       // already advised, don't re-advise every sweep
        std::vector<uint32_t> selections; // saturating per-expert selection count
        std::vector<bool> is_pinned;      // held resident with mlock
        // Routing-relative recency: the value of device.plan_epoch when this
        // expert was last selected. Wall-clock idleness turned out to be the
        // wrong unit - at 120s idle swept every 30s, a 40-second benchmark run
        // classified almost nothing, so the kernel was choosing evictions with
        // no information from us at all. What matters is how many routing
        // decisions ago an expert was last wanted, not how many seconds.
        std::vector<uint64_t> last_epoch;
        // This expert's own copy, or nullptr. Allocated per expert at promotion
        // rather than carved from a pre-reserved pool: reserving the whole
        // budget up front took memory from the page cache before anything had
        // been promoted into it, which measured 30% slower. Allocating on
        // promotion means every byte taken is a byte MADV_DONTNEED has just
        // released. Read lock-free from CPU compute threads, written only under
        // session.mu.
        std::vector<std::atomic<void *>> host_slot;
    };
    std::unordered_map<const void *, cpu_residency> residency;
    uint64_t plan_epoch = 0; // one tick per MoE node planned

    // Host hot-expert buffer: memory we own, holding the experts the kernel
    // would otherwise be free to evict.
    size_t host_bytes = 0;      // currently held by promoted experts
    size_t host_promoted = 0;
    // Experts decode has marked worth promoting. The copy itself is done by the
    // fill worker: doing it inline cost a malloc, a 1.4 MiB memcpy and a madvise
    // syscall on the decode path under the session lock, which measured slower
    // than not having the buffer at all. Decode now only records the intent.
    std::deque<std::pair<const void *, int>> host_promote_queue;

    // Evicted blocks awaiting release. A CPU compute thread may hold a pointer
    // it read moments ago, so freeing at eviction would be a use-after-free.
    // Blocks are released once the cache has quiesced instead.
    std::vector<std::pair<void *, size_t>> host_retired;
    std::chrono::steady_clock::time_point residency_epoch{};
    std::chrono::steady_clock::time_point last_cold_sweep{};
    size_t pinned_bytes = 0;

    int stable_visits = 0;
    bool saw_repeat = false;
    bool budget_ready = false;
    size_t budget_limit = 0;
    // Set to the epoch (never a real re-check) so the very first call in
    // moe_cache_prepare_budget() always re-evaluates regardless of clock
    // start time.
    std::chrono::steady_clock::time_point budget_checked_at{};
    size_t allocated_bytes = 0;
    moe_cache_scratch scratch_reserve;

    std::deque<moe_cache_job> queue;
    size_t queued_bytes = 0;
    bool worker_started = false;
    bool inflight = false;
    const void * inflight_source = nullptr;
    size_t inflight_bytes = 0;
    std::thread worker;

    cudaStream_t compute_stream = nullptr;
    int32_t * h_ids = nullptr;
    int32_t * d_ids = nullptr;
    size_t h_ids_cap = 0;
    size_t d_ids_cap = 0;
    float * h_act = nullptr;
    float * d_act = nullptr;
    size_t h_act_cap = 0;
    size_t d_act_cap = 0;
    void * d_act_q8 = nullptr;
    size_t act_q8_cap = 0;
    float * d_out = nullptr;
    size_t d_out_cap = 0;
    // High-water mark of bytes ever written since the buffer's last (re)growth,
    // i.e. the deepest point at which the canary poison from growth is still
    // guaranteed pristine. A dispatch that writes FEWER bytes than a prior
    // dispatch since the last growth leaves real data (not canary) sitting in
    // [this_call_used, hwm) - checking the guard there would flag that
    // leftover real data as a "breach" even though nothing overran anything.
    // Only checkable when this call's usage reaches a fresh high point.
    size_t d_act_hwm = 0;
    size_t act_q8_hwm = 0;
    size_t d_out_hwm = 0;
    float * h_out = nullptr;
    size_t h_out_cap = 0;

    // Heat-aware neuron subsetting: dispatch scratch for the REDUCED-pool
    // subset of a batch, entirely separate from h_ids/d_ids/d_act_q8/d_out
    // above. Same growth discipline (moe_cache_grow_host/moe_cache_grow_device,
    // doubling, canary-poisoned on growth), just a second, smaller set of
    // buffers because a reduced-pool sub-batch has its own row count
    // (n_hits_reduced <= n_hits) and its own row width (K, not n_out) - see
    // the split in moe_cache_dispatch. Allocated lazily on first use; a
    // session that never produces a reduced-pool hit never touches these.
    int32_t * h_ids_reduced = nullptr;
    int32_t * d_ids_reduced = nullptr;
    size_t h_ids_reduced_cap = 0;
    size_t d_ids_reduced_cap = 0;
    void * d_act_q8_reduced = nullptr;
    size_t act_q8_reduced_cap = 0;
    float * d_out_reduced = nullptr;
    size_t d_out_reduced_cap = 0;
    // Per reduced-hit row: which absolute row of the batch it is (so the
    // scatter kernel writes into the right place in d_out) and its K chosen
    // neuron indices (so it writes each computed value to the right column).
    int32_t * h_reduce_row_map = nullptr;
    int32_t * d_reduce_row_map = nullptr;
    size_t h_reduce_row_map_cap = 0;
    size_t d_reduce_row_map_cap = 0;
    int32_t * h_reduce_sel_idx = nullptr;
    int32_t * d_reduce_sel_idx = nullptr;
    size_t h_reduce_sel_idx_cap = 0;
    size_t d_reduce_sel_idx_cap = 0;

    // Full-layer prefill double buffer (see moe_cache_prefill_prefetch/_wait).
    // Keyed by tensor_bytes (a shape proxy: distinct MoE tensors reuse the same
    // slab as long as they're the same total size, same as the decode pool's
    // keying by expert_size) rather than the per-node scratch above, which is
    // sized for at most GGML_MOE_CACHE_MAX_BATCH_ROWS rows - a full layer is
    // usually every expert, i.e. no reasonable row cap applies.
    std::unordered_map<size_t, moe_cache_prefill_slab> prefill_slabs;
    cudaStream_t prefill_stream = nullptr;
    // host_base -> next layer's (host_base, tensor_bytes, n_expert). Populated
    // once per architecture at graph-BUILD time
    // (moe_cache_prefill_register_successor), never at compute time - see
    // that function's comment for why build time is the only point in this
    // whole pipeline that is unconditionally safe for cudaMalloc (graph
    // capture can only ever be active during compute).
    struct prefill_successor {
        const void * next_host_base = nullptr;
        size_t next_tensor_bytes = 0;
        int64_t next_n_expert = 0;
    };
    std::unordered_map<const void *, prefill_successor> prefill_successors;

    // Hit-D2D split diagnostics (see moe_cache_prefill_advance): how many
    // expert rows across all prefetches were served from the decode-time LFRU
    // pool device-to-device, out of how many rows were prefetched in total.
    long long prefill_hit_rows = 0;
    long long prefill_total_rows = 0;

    long long prefetches = 0;
    long long hits = 0;
    long long misses = 0;
    // Hit/miss split by the router's rank for each pick (0 = top choice).
    // Observational only - nothing reads these to make a decision.
    long long rank_hits[GGML_MOE_CACHE_MAX_RANK] = {};
    long long rank_misses[GGML_MOE_CACHE_MAX_RANK] = {};
    // Hit/miss split by LAYER - the actual throughput-relevant signal for
    // "which layer to prefer", as opposed to neuron_heat's concentration
    // stats (a structural property of a layer's activations, not a cost
    // measure). A layer with a flat, unconcentrated neuron distribution
    // can still be cheap if it rarely misses; a layer with a sharp
    // distribution can still be expensive if it misses constantly. This
    // answers "how much wall-clock cost does this layer actually
    // generate", which is the right question for prioritization - miss
    // COUNT is a proxy for cost here (same expert_size per pool on this
    // model, so cost-per-miss is roughly uniform across layers sharing a
    // pool), not a proxy for "how the layer looks". Fixed-size array
    // indexed by node->layer (parsed from the tensor name at moe_cache_
    // begin(), already bounds-checked there) - 256 is far more layers
    // than any real model has, no map/lock overhead needed on this path.
    static constexpr int MAX_LAYER_STAT = 256;
    long long layer_hits[MAX_LAYER_STAT] = {};
    long long layer_misses[MAX_LAYER_STAT] = {};
    // Substitution: a miss served by running a RESIDENT expert in place of the
    // one the router asked for, rather than falling back to CPU compute.
    long long substitutions = 0;      // misses served by a stand-in
    long long substitute_declined = 0;// misses where no acceptable stand-in existed
    long long inserts = 0;
    long long fills = 0;
    long long fill_failures = 0;
    long long evictions = 0;
    long long insert_skips = 0;
    long long admission_skips = 0;
    long long dispatch_failures = 0;
    long long collect_failures = 0;
    long long nodes = 0;
    long long collect_calls = 0;
    long long spec_evictions_this_cycle = 0;   // rate limit for speculative eviction
    long long spec_evict_reset_at_calls = 0;   // collect_calls value at last reset
    // Cross-depth agreement gate for GGML_CUDA_MOE_CACHE_SPEC_EVICT_MODE=agree
    // (see moe_cache_prefetch): a candidate a farther, less-accurate depth
    // wanted but could not get a free slot for is remembered here, along
    // with the depth that suggested it, for the rest of this decode step.
    // Eviction is only permitted when a *strictly closer* (smaller-depth,
    // more accurate) prediction later lands on the same expert - not merely
    // "seen again regardless of which depth". Depth is tracked explicitly
    // because, measured, treating any repeat sighting as corroboration
    // works at depth=2 (only one possible pairing: depth-2 confirmed by
    // depth-1) but measurably hurts at depth=3, where it also accepts a
    // depth-3-confirmed-by-depth-2 pairing - both still under 50% precision
    // - diluting the depth-1-confirms-depth-2 signal that actually works
    // (see docs/index.html). Bounded and reset on the same collect_calls
    // clock spec_evictions_this_cycle uses.
    std::vector<std::pair<moe_cache_key, int>> spec_seen_this_cycle;
    std::atomic<int> error_logs{0};
};

struct moe_cache_session {
    moe_cache_config config;
    std::mutex hostreg_mu;
    // region base -> the page-aligned [begin,end) actually registered. Stored as
    // an interval because GGUF packs tensors on 32-byte boundaries, so the raw
    // base is not page-aligned and cudaHostRegister rejects it outright.
    std::unordered_map<const void *, std::pair<uintptr_t, uintptr_t>> hostreg;
    size_t hostreg_bytes = 0;
    std::vector<std::unique_ptr<moe_cache_device>> devices;
    std::unordered_map<int, int> layer_devices;
    std::unordered_map<const void *, int> tensor_devices;

    // Debug/demo view only (the "Brain" page, tools/ui): host_base -> parsed
    // layer number, reusing moe_cache_layer_number()'s existing name parse
    // rather than re-deriving layer identity. Purely additive bookkeeping -
    // never read by any routing/eviction decision, so it carries none of
    // their correctness risk. n_expert_hint is the widest n_expert seen
    // across begin() calls, used only to size the snapshot grid's column
    // count for the same view.
    std::unordered_map<const void *, int> tensor_layer;
    int64_t n_expert_hint = 0;

    // Cross-run selection history, keyed (layer << 32 | expert). The cache
    // otherwise starts empty on every launch and re-learns the hot set through
    // misses, which is why the first request after a restart is consistently
    // the slowest one measured. Persisting the counts lets a fresh session
    // pre-admit what the last one proved hot. Purely advisory: it seeds
    // admission, never overrides a live eviction or routing decision.
    std::unordered_map<uint64_t, uint32_t> history;
    bool history_loaded = false;
    // Co-activation graph carried across process restarts, exactly like
    // `history` above and for the same reason: device.co_activation starts
    // empty every launch, so writing it out unmerged made each run OVERWRITE
    // the previous run's graph with only its own session's observations. The
    // evolving atlas is supposed to accumulate - a graph that silently resets
    // on restart cannot evolve, it can only ever describe the last session.
    // Keyed by packed (layer_a, expert_a, layer_b, expert_b) rather than by
    // moe_cache_edge, because tensor pointers are fresh addresses every launch
    // and cannot be a stable cross-run identity (the same reason the predictor
    // persists by tensor NAME rather than pointer).
    std::unordered_map<uint64_t, uint64_t> coact_history;
    bool coact_loaded = false;
    std::chrono::steady_clock::time_point last_history_save{};

    std::mutex mu;
    std::mutex fill_mu;
    std::condition_variable cv;
    std::condition_variable idle_cv;
    std::atomic<bool> stopping{false};
    std::atomic<bool> dormant{false};
    bool announced = false;
    int active_scopes = 0;
    int active_nodes = 0;
    struct active_source {
        size_t bytes = 0;
        int references = 0;
    };
    std::unordered_map<const void *, active_source> active_sources;
};

struct moe_cache_pin {
    int slot = -1;
    // slot.generation observed at pin time (readers++ in moe_cache_plan).
    // Compared again in moe_cache_collect after the mandatory post-dispatch
    // cudaStreamSynchronize there - free, since that sync already happens
    // on every collect() call. A mismatch proves the fill worker rewrote
    // this slot while it was supposedly protected by the pin (readers>0),
    // i.e. the row was computed against weights that changed underneath it.
    uint64_t generation = 0;
    // Which pool this slot belongs to. Every existing pin (ordinary hit,
    // substitution) pins a slot in node->pool, so this used to be implicit;
    // heat-aware neuron subsetting introduced a SECOND pool per node (the
    // reduced-width pool, see device.reduced_indices) that a pin can now
    // also point into. Always set explicitly at every pin-creation site -
    // never left to default to node->pool implicitly - so every consumer
    // (moe_cache_collect's guard/gen checks, moe_cache_end's unpin loop)
    // can treat pins uniformly regardless of which pool they came from.
    moe_cache_pool * pool = nullptr;
};

struct moe_cache_node {
    moe_cache_session * session = nullptr;
    moe_cache_device * device = nullptr;
    moe_cache_pool * pool = nullptr;
    int pool_index = -1;
    const void * host_base = nullptr;
    size_t expert_size = 0;
    int64_t n_in = 0;
    int64_t n_out = 0;
    int64_t n_expert = 0;
    // Rows in this node's ids tensor, so plan() can turn a flat ids index
    // back into the router's per-token rank (index % top_k).
    int64_t n_tokens = 0;
    int wtype = -1;
    // Logical layer index, parsed from the tensor name in moe_cache_begin
    // (-1 when the name doesn't follow llama.cpp's "blk.N." convention).
    // Needed because ONE logical layer dispatches SEVERAL expert tensors
    // (gate_up_exps then down_exps) that share a single router decision -
    // without this, cross-layer co-activation treats that pair as a layer
    // transition and records a self-edge. See the tracking block in
    // moe_cache_plan.
    int layer = -1;
    // Heat-aware neuron subsetting eligibility (gate_exps/up_exps only -
    // see the comment where this is computed in moe_cache_begin).
    bool reduce_eligible = false;
    std::unique_lock<std::mutex> dispatch_lock;
    moe_cache_pin pins[GGML_MOE_CACHE_MAX_BATCH_ROWS];
    int n_pins = 0;
    bool planned = false;
    bool dispatched = false;
    // Exact byte extent this dispatch() call actually wrote into each
    // scratch buffer - checked against the guard pattern right after it,
    // in collect(). 0 means "not applicable" (e.g. dispatch never ran).
    size_t act_bytes_used = 0;
    size_t q8_bytes_used = 0;
    size_t out_bytes_used = 0;
};

// GGML_CUDA_MOE_CACHE_LOCK_TRACE=1: a TSan run flagged "double lock of a
// mutex" with moe_cache_plan's session.mu acquisition on the stack, reached
// during model-load warmup via common_context_can_seq_rm. All frames from
// the mutex constructor down to moe_cache_plan collapsed to one binary
// offset in the report (inlining), so the "which line exactly" detail isn't
// trustworthy - but a real same-thread double-lock on a non-recursive
// std::mutex should self-deadlock, which contradicts the server loading
// successfully and serving correctly afterward. This directly tests same-
// thread reentrancy into the two session.mu critical sections the stack
// implicates (begin() and plan()), independent of exactly which frame TSan
// attributed it to - if either is entered while this thread already holds
// one, it logs loudly before the real lock() call, so it is visible even if
// the underlying mutex behavior turns out to mask the reentrancy.
static thread_local int g_moe_cache_session_mu_depth = 0;
struct moe_cache_lock_trace_guard {
    const char * where;
    bool active;
    moe_cache_lock_trace_guard(const char * w) : where(w), active(false) {
        if (!getenv("GGML_CUDA_MOE_CACHE_LOCK_TRACE")) {
            return;
        }
        active = true;
        if (g_moe_cache_session_mu_depth > 0) {
            fprintf(stderr, "[moe-cache] LOCK REENTRANCY at %s: this thread already holds "
                    "session.mu (depth=%d)\n", where, g_moe_cache_session_mu_depth);
            fflush(stderr);
        }
        g_moe_cache_session_mu_depth++;
    }
    ~moe_cache_lock_trace_guard() {
        if (active) {
            g_moe_cache_session_mu_depth--;
        }
    }
};

// Wall-clock seconds since epoch, for correlating debug probe output against
// external system samplers (GPU/CPU/RAM) on the same timeline.
static double moe_cache_wall_clock() {
    return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
}

static std::mutex g_registry_mu;
static std::unordered_set<moe_cache_session *> g_sessions;
static std::atomic<int> g_session_count{0};

// Structural ceiling shared by the scratch buffers below (moe_cache_scratch_requirements'
// max_rows) and the CPU-side stack arrays (ggml-cpu.c's MOE_CACHE_MAX_TOPK) - both are
// already sized for this many rows regardless of max_batch, so this is a real capacity
// limit, not a conservative guess. Defined once, in ggml-backend-moe-cache.h, so the two
// sides of this handshake can't drift apart.
static constexpr int MOE_CACHE_MAX_BATCH_CEILING = GGML_MOE_CACHE_MAX_BATCH_ROWS;

// Per-slot guard region in the slab: dead, canary-filled space between one
// slot's real data and the next slot's real data. One page is generous
// against any plausible overrun (a weight row here is a few hundred bytes)
// while staying negligible against expert_size (hundreds of KiB), so the
// placement cost is not something a budget has to account for precisely.
static constexpr size_t MOE_CACHE_SLOT_GUARD_BYTES = 4096;
static constexpr unsigned char MOE_CACHE_GUARD_PATTERN = 0xA5;

// Live hint for max_batch's default, set by the caller (llama_context, via the
// set_max_batch_hint vtable entry) to the real max concurrent sequence count
// (n_seq_max) before session_create() runs. 0 = no hint given, use the
// hardware-agnostic floor. GGML_CUDA_MOE_CACHE_MAX_BATCH still overrides this
// exactly as before when explicitly set - this only changes the default.
static std::atomic<int> g_max_batch_hint{0};

static void moe_cache_set_max_batch_hint(int n_seq_max) {
    g_max_batch_hint.store(
            std::max(0, std::min(n_seq_max, MOE_CACHE_MAX_BATCH_CEILING)),
            std::memory_order_relaxed);
}
struct moe_cache_scope_frame {
    moe_cache_session * requested = nullptr;
    moe_cache_session * active = nullptr;
};
static thread_local std::vector<moe_cache_scope_frame> g_session_stack;
static thread_local int g_session_suppressed = 0;

static size_t moe_cache_trim_session(
        moe_cache_session & session, int physical_device);

static bool moe_cache_env_i64(
        const char * name, int64_t min_value, int64_t max_value, int64_t & value) {
    const char * text = getenv(name);
    if (!text || !text[0]) {
        return false;
    }

    char * end = nullptr;
    errno = 0;
    const long long parsed = strtoll(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed < min_value || parsed > max_value) {
        MOE_CACHE_LOG("[moe-cache] ignoring invalid %s=%s\n", name, text);
        return false;
    }

    value = parsed;
    return true;
}

static moe_cache_config moe_cache_read_config() {
    moe_cache_config config;
    int64_t value = 0;
    bool mode_off = false;
    bool mode_valid = false;

    if (const char * mode = getenv("GGML_CUDA_MOE_CACHE_MODE")) {
        if (strcmp(mode, "auto") == 0) {
            config.automatic = true;
            mode_valid = true;
        } else if (strcmp(mode, "on") == 0) {
            config.automatic = false;
            mode_valid = true;
        } else if (strcmp(mode, "off") == 0) {
            config.enabled = false;
            mode_off = true;
            mode_valid = true;
        } else {
            MOE_CACHE_LOG("[moe-cache] ignoring invalid GGML_CUDA_MOE_CACHE_MODE=%s\n", mode);
        }
    }
    if (moe_cache_env_i64("GGML_CUDA_MOE_CACHE", 0, 1, value)) {
        config.enabled = value != 0 && !mode_off;
    }
    if (moe_cache_env_i64("GGML_CUDA_MOE_CACHE_BUDGET_MB", 1, 1024 * 1024, value)) {
        config.budget_mb = (size_t)value;
        if (!mode_valid) {
            config.automatic = false;
        }
    }
    if (moe_cache_env_i64("GGML_CUDA_MOE_CACHE_RESERVE_MB", 0, 1024 * 1024, value)) {
        config.reserve_mb = (size_t)value;
    }
    if (moe_cache_env_i64("GGML_CUDA_MOE_CACHE_MIN_EXPERT_KB", 1, 1024 * 1024, value)) {
        config.min_expert_bytes = (size_t)value << 10;
    }
    if (moe_cache_env_i64("GGML_CUDA_MOE_CACHE_MAX_BATCH", 1, MOE_CACHE_MAX_BATCH_CEILING, value)) {
        config.max_batch = (int)value;
    } else {
        const int hint = g_max_batch_hint.load(std::memory_order_relaxed);
        if (hint > 0) {
            config.max_batch = std::max(8, hint);
        }
    }
    if (moe_cache_env_i64("GGML_CUDA_MOE_CACHE_INSERTS", 1, 1024, value)) {
        config.inserts_per_plan = (int)value;
    }
    if (moe_cache_env_i64("GGML_CUDA_MOE_CACHE_ADMIT_AFTER", 1, 255, value)) {
        config.admit_after = (int)value;
    }
    if (moe_cache_env_i64("GGML_CUDA_MOE_CACHE_THROTTLE", 1, 1024, value)) {
        config.readmit_after = (int)value;
    }
    if (moe_cache_env_i64("GGML_CUDA_MOE_CACHE_QUEUE", 1, 65536, value)) {
        config.queue_max = (int)value;
    }
    if (moe_cache_env_i64("GGML_CUDA_MOE_CACHE_QUEUE_MB", 1, 1024 * 1024, value)) {
        config.queue_mb = (size_t)value;
    }
    if (moe_cache_env_i64("GGML_CUDA_MOE_CACHE_STATS", 0, INT_MAX, value)) {
        config.stats_every = (int)value;
    }
    if (moe_cache_env_i64("GGML_CUDA_MOE_CACHE_HOSTREG_MB", 0, 1024 * 1024, value)) {
        config.hostreg_mb = (size_t)value;
    }
    if (moe_cache_env_i64("GGML_CUDA_MOE_CACHE_NDEV", 1, INT_MAX, value)) {
        config.max_devices = (int)value;
    }
    if (moe_cache_env_i64("GGML_CUDA_MOE_CACHE_SERIAL_FILL", 0, 1, value)) {
        config.serial_fill = value != 0;
    }
    if (config.automatic) {
        config.min_compute_capability = 750;
    }
    if (moe_cache_env_i64("GGML_CUDA_MOE_CACHE_MIN_CC", 0, 999, value)) {
        config.min_compute_capability = (int)value;
    }
    if (const char * fail = getenv("GGML_CUDA_MOE_CACHE_FAIL")) {
        config.fail_stage = fail;
    }

    return config;
}

static bool moe_cache_fail(const moe_cache_session & session, const char * stage) {
    const std::string & value = session.config.fail_stage;
    if (value.empty()) {
        return false;
    }
    if (value == "all" || value == stage) {
        return true;
    }

    size_t begin = 0;
    while (begin < value.size()) {
        size_t end = value.find(',', begin);
        if (end == std::string::npos) {
            end = value.size();
        }
        if (value.compare(begin, end - begin, stage) == 0) {
            return true;
        }
        begin = end + 1;
    }
    return false;
}

static bool moe_cache_ranges_overlap(
        const void * lhs, size_t lhs_size, const void * rhs, size_t rhs_size) {
    if (!lhs || !rhs || lhs_size == 0 || rhs_size == 0) {
        return false;
    }
    const uintptr_t l = (uintptr_t)lhs;
    const uintptr_t r = (uintptr_t)rhs;
    return (l <= r ? r - l < lhs_size : l - r < rhs_size);
}

static uint64_t moe_cache_name_hash(const char * text) {
    uint64_t hash = 0xcbf29ce484222325ULL;
    while (*text) {
        hash ^= (unsigned char)*text++;
        hash *= 0x100000001b3ULL;
    }
    return hash;
}

static bool moe_cache_layer_number(const char * name, int & layer) {
    const char * marker = strstr(name, "blk.");
    if (!marker) {
        return false;
    }

    const char * first = marker + 4;
    char * end = nullptr;
    errno = 0;
    const long parsed = strtol(first, &end, 10);
    if (errno != 0 || end == first || parsed < 0 || parsed > INT_MAX) {
        return false;
    }
    if (*end != '.' && *end != '\0') {
        return false;
    }
    layer = (int)parsed;
    return true;
}

static bool moe_cache_type_supported(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_NVFP4:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ4_XS:
            return true;
        default:
            return false;
    }
}

// Generic doubly-linked-list primitives, parameterized by which segment's
// head/tail to operate on - shared by both the probation and protected
// lists below, since a slot is only ever in one of the two at a time and
// its prev/next fields are safe to reuse for whichever list currently owns
// it.
static void moe_cache_list_remove(int & head, int & tail, moe_cache_pool & pool, int index) {
    moe_cache_slot & slot = pool.slots[index];
    if (slot.prev >= 0) {
        pool.slots[slot.prev].next = slot.next;
    } else {
        head = slot.next;
    }
    if (slot.next >= 0) {
        pool.slots[slot.next].prev = slot.prev;
    } else {
        tail = slot.prev;
    }
    slot.prev = -1;
    slot.next = -1;
}

static void moe_cache_list_push_back(int & head, int & tail, moe_cache_pool & pool, int index) {
    moe_cache_slot & slot = pool.slots[index];
    slot.prev = tail;
    slot.next = -1;
    if (tail >= 0) {
        pool.slots[tail].next = index;
    } else {
        head = index;
    }
    tail = index;
}

// Removes a slot from whichever segment it currently occupies (reads
// slot.segment to know which list's head/tail to update).
static void moe_cache_segment_remove(moe_cache_pool & pool, int index) {
    moe_cache_slot & slot = pool.slots[index];
    if (slot.segment == moe_cache_segment::probation) {
        moe_cache_list_remove(pool.lru_head, pool.lru_tail, pool, index);
    } else {
        moe_cache_list_remove(pool.protected_head, pool.protected_tail, pool, index);
        pool.protected_count--;
    }
}

// Pushes a slot onto the tail of the given segment's list, updating
// slot.segment to match. New admissions always go to `probation` (a slot
// has to prove itself hot via a real post-admission hit before it earns
// `protected_` status - see the struct comment above moe_cache_segment).
// Callers wanting the cap+demotion enforced should go through
// moe_cache_promote_to_protected() below instead of pushing to
// protected_ directly.
static void moe_cache_segment_push_back(moe_cache_pool & pool, int index, moe_cache_segment seg) {
    moe_cache_slot & slot = pool.slots[index];
    slot.segment = seg;
    if (seg == moe_cache_segment::probation) {
        moe_cache_list_push_back(pool.lru_head, pool.lru_tail, pool, index);
    } else {
        moe_cache_list_push_back(pool.protected_head, pool.protected_tail, pool, index);
        pool.protected_count++;
    }
}

// Heat gained per hit and the ceiling it saturates at (avoids overflow
// concerns entirely - 20 bits is far more resolution than the hysteresis
// margin below ever needs).
static constexpr uint32_t MOE_CACHE_HEAT_MAX = 1u << 20;
static constexpr uint32_t MOE_CACHE_HEAT_STEP = 4;

// How many fills between heat-decay sweeps, and how many of the
// coldest-recency slots the eviction scan below considers - both cheap,
// bounded costs (decay is O(pool size) but runs rarely; the eviction scan
// is O(window), not O(pool size), on every eviction).
static constexpr long long MOE_CACHE_HEAT_DECAY_EVERY = 512;
static constexpr int MOE_CACHE_EVICT_WINDOW = 8;

// Two things were tried and measured here before landing on this one:
// letting decay alone regulate protected_'s population, with no fixed
// cap, either via an absolute per-slot heat floor (61.4% hit rate,
// protected_ still hit 596-598/604) or a floor relative to protected_'s
// own mean heat (61.3%, 599/604) - both collapsed back to essentially the
// same runaway as the original uncapped-SLRU bug (60.2-60.3%). Root
// cause in both: decay only fires on *fill* events (new admissions), but
// within one busy request thousands of *hits* land on the same resident
// working set between fills - heat (and the mean along with it) balloons
// for nearly the whole pool before decay ever gets a chance to run, and
// even when a slot does get demoted, it's usually re-hit and re-promoted
// on the very next token before the next decay. Heat/decay alone doesn't
// have a fast enough signal to regulate *population size* at this
// timescale - that needs a hard structural bound. So: keep the fixed
// MOE_CACHE_PROTECTED_CAP_PCT cap (measured 65.0% alone, still the best
// single mechanism found), and let heat do the job it's actually good at
// - choosing *which* slot pays the demotion cost when the cap is hit
// (coldest in a bounded recency window, not just oldest) - see
// moe_cache_promote_to_protected below.
static void moe_cache_pool_decay(moe_cache_pool & pool) {
    for (moe_cache_slot & slot : pool.slots) {
        slot.heat >>= 1;
    }
    pool.heat_scale_ema *= 0.5;
}

// Cheap per-hit nudge of the pool's own heat-scale reference (see the
// struct comment on heat_scale_ema) - called from every real heat step
// instead of scanning the pool, so the reference tracks live activity
// between decay sweeps without adding an O(pool size) cost to the hot path.
static void moe_cache_pool_note_heat_step(moe_cache_pool & pool, uint32_t new_heat) {
    pool.heat_scale_ema += 0.02 * ((double) new_heat - pool.heat_scale_ema);
}

// Colibri's "25%-plus-4" hysteresis: candidate `b` only replaces the
// current best eviction pick `a` if b's heat is *meaningfully* colder,
// not just marginally - prevents two similarly-hot slots from flip-
// flopping as the eviction choice from one insertion to the next purely
// from noise. Generalized to the cost-weighted score used below -
// GreedyDual-Size's priority is a real number, not an integer heat count,
// so this needs the double-precision form rather than a second, drifting
// copy of the same 25%-plus-4 margin.
static bool moe_cache_colder_enough(double a_score, double b_score) {
    return b_score + b_score / 4.0 + 4.0 <= a_score;
}

// GreedyDual-Size, cut down to what this cache actually needs: within one
// pool every slot is the same size (one pool per distinct expert_size), so
// the classic cost/size term collapses to plain cost, and there is no
// bin-packing decision left - only "how expensive would it be to bring this
// expert back if we evict it now". That answer is not uniform across a
// pool's own residents, though, and this fork already tracks the signal
// that tells them apart: an expert whose CPU-side backing pages this
// process itself advised POSIX_MADV_DONTNEED on (device.residency.is_cold -
// the CPU host-buffer cold sweep, see moe_cache_cold_sweep) is not sitting
// in the page cache anymore and has to come back from the NVMe device at
// NVMe latency; one that is mlock'd (is_pinned) or was selected recently
// enough that the cold sweep has not touched it is still page-cache-warm,
// RAM-latency to refetch. Weighting the existing heat score by that tier
// multiplier means eviction prefers to let go of a slot that is merely
// less-recently-hit but cheap to bring back over one that is only
// marginally hotter but would cost a real NVMe round trip to restore -
// the actual GreedyDual-Size idea (evict the least total value, not just
// the least recent), applied without touching the SLRU/heat machinery
// itself, which took a lot of separately-measured tuning to land on 65%
// (see the struct comment above moe_cache_slot).
//
// The 5.8x figure is this fork's own measured NVMe-vs-slower-storage
// migration win (see docs/index.html, "NVMe migration") - not a fresh
// benchmark, a reuse of an already-validated number as the tier penalty
// for "this expert's pages are gone, not just idle". Deliberately not
// wired to the H2D/D2D bandwidth profile above: that measures device-side
// contention, a different hop in the path than the host-storage tier this
// weight is about.
static constexpr double MOE_CACHE_COST_TIER_NVME = 5.8;
static constexpr double MOE_CACHE_COST_TIER_RAM  = 1.0;

// Off switch for A/B isolation only - lets the cost-weighted and
// spec-eviction effects be measured independently of each other rather than
// only ever as a bundle. Not meant as a tuning knob for end users.
static bool moe_cache_cost_weight_enabled() {
    static const bool enabled = [] {
        const char * env = getenv("GGML_CUDA_MOE_CACHE_COST_WEIGHT");
        return !env || atoi(env) != 0;
    }();
    return enabled;
}

// is_cold is our own advisory record of having called madvise on an expert's
// pages, not ground truth of whether those pages are actually gone. The
// default demotion mode is soft (MADV_COLD, not MADV_PAGEOUT - see
// moe_cache_cold_sweep above): a soft-advised page "survives when there is
// no pressure and goes first when there is", so an expert we advised cold an
// hour ago, on a box that never came under real memory pressure since, is
// almost certainly still page-cache-resident right now - charging it the
// NVMe tier penalty would be wrong under the *default* demotion mode, which
// is exactly the case that matters most (GGML_CUDA_MOE_CACHE_DEMOTE=pageout
// is opt-in). mincore() gives the real answer instead of trusting the
// advisory flag: it asks the kernel which pages of this exact range are
// resident right now, no assumption needed. Only called for is_cold-flagged
// experts (the minority actually advised), so the common case - most
// experts are never cold-swept at all - never pays this syscall.
static bool moe_cache_pages_actually_resident(const void * host_base, size_t expert_size, size_t expert) {
    static const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0 || expert_size == 0) {
        return true; // can't tell - don't over-penalize on a bad read
    }
    const uintptr_t begin = (uintptr_t) host_base + expert * expert_size;
    const uintptr_t start = begin & ~((uintptr_t) page_size - 1);
    const uintptr_t end   = (begin + expert_size + (uintptr_t) page_size - 1) & ~((uintptr_t) page_size - 1);
    const size_t n_pages = (size_t) ((end - start) / (uintptr_t) page_size);
    if (n_pages == 0) {
        return true;
    }
    // Bounded: a typical expert is well under a few hundred pages: no heap
    // churn for the common case, and a real cap (4096 pages = 16 MiB at 4K
    // pages) so a pathologically large expert can't blow the stack either.
    unsigned char stack_vec[4096];
    std::vector<unsigned char> heap_vec;
    unsigned char * vec = stack_vec;
    if (n_pages > sizeof(stack_vec)) {
        heap_vec.resize(n_pages);
        vec = heap_vec.data();
    }
    if (mincore((void *) start, (size_t) (end - start), vec) != 0) {
        return true; // ENOMEM (unmapped hole) or similar - don't over-penalize on a syscall failure
    }
    for (size_t i = 0; i < n_pages; i++) {
        if (!(vec[i] & 1)) {
            return false; // even one missing page means the read touches disk
        }
    }
    return true;
}

static double moe_cache_cost_tier_weight(const moe_cache_device & device, const moe_cache_key & key) {
    if (key.expert < 0 || !moe_cache_cost_weight_enabled()) {
        return MOE_CACHE_COST_TIER_RAM;
    }
    const auto it = device.residency.find(key.tensor);
    if (it == device.residency.end()) {
        return MOE_CACHE_COST_TIER_RAM; // no host-residency tracking for this tensor - neutral, today's behavior
    }
    const auto & res = it->second;
    const size_t e = (size_t) key.expert;
    if (e < res.is_pinned.size() && res.is_pinned[e]) {
        return MOE_CACHE_COST_TIER_RAM; // mlock'd: always page-cache-resident, never pays the NVMe tier
    }
    if (e < res.is_cold.size() && res.is_cold[e]) {
        // Ground-truth check before trusting our own advisory flag - see the
        // comment on moe_cache_pages_actually_resident above. Toggle exists
        // for A/B isolation only (measuring the mincore() syscall's own
        // overhead against the correctness it buys) - not a tuning knob.
        static const bool verify = [] {
            const char * env = getenv("GGML_CUDA_MOE_CACHE_COST_VERIFY");
            return !env || atoi(env) != 0;
        }();
        if (verify && moe_cache_pages_actually_resident(key.tensor, res.expert_size, e)) {
            return MOE_CACHE_COST_TIER_RAM; // advised cold, but still actually resident - not expensive yet
        }
        return MOE_CACHE_COST_TIER_NVME; // verified gone (or verification disabled): assume NVMe cost
    }
    return MOE_CACHE_COST_TIER_RAM;
}

// Track 1 (docs/plan.md): topic-affinity was also tried here, as an
// eviction-candidate scoring factor ALONGSIDE moe_cache_atlas_warm's
// admission mechanism below, on the theory that the two are complementary
// (admission brings new topically-aligned experts in; this would keep
// already-resident ones around longer). Tested head to head on a real
// topic-switch workload against admission-only and eviction-only variants -
// the combination measured WORSE than either alone (1/4 rounds won, hit
// rate net *negative* vs. baseline, -0.95pp), worse than the eviction-only
// redesign it was meant to complement (3/4 rounds, +0.63pp) and far worse
// than admission-only (4/4 rounds, +2.75pp - the actual best performer of
// all three, and what's still active below). Root cause, best guess: the
// two interfere rather than cooperate - this weight makes topically-aligned
// candidates resist eviction everywhere, including when moe_cache_atlas_
// warm's OWN admission calls need to evict something to make room for a
// DIFFERENT atlas-suggested candidate, so admission ends up fighting itself.
// Removed after the third A/B round settled it - see docs/plan.md for the
// full three-way comparison and the honest reporting of both reversals.

// Whether topic alignment protects a slot from eviction. Reintroduced after
// the removal above, but NOT the same mechanism: this is boost-only (a
// resident slot's effective heat can only go UP when it aligns with the
// live topic direction, never down for one that doesn't - misaligned/
// unmeasured slots score exactly as they always did) and, critically, the
// thing it was found to interfere with - moe_cache_atlas_warm's OWN
// eviction rights - no longer exists (see moe_cache_atlas_admit: reverted
// to free-slot-only after a real, measured regression, docs/plan.md). That
// specific interference is now structurally impossible; this reintroduces
// the "protect what's aligned" half in isolation, never both halves at
// once the way the removed version did.

// #5: Holt's linear trend forecast. req_dir_x/y is the "level" term - a
// lagging EMA of where routing has actually been - and req_vel_x/y (updated
// alongside it, see the update site) is the "trend" term - an EMA of the
// level's own per-update delta, i.e. which way and how fast the centroid is
// currently moving. level + horizon*trend genuinely extrapolates forward
// instead of only describing recent history, which is the whole point:
// ranking against the plain level is "where has this been", not "where is
// this heading". Magnitude does not matter to any consumer of this - both
// moe_cache_atlas_rank and moe_cache_atlas_align_score only ever use it
// through a cosine similarity, which normalizes magnitude away - only the
// direction the horizon bends it toward does.
static bool moe_cache_atlas_lookahead_enabled() {
    static const bool enabled = [] {
        const char * env = getenv("GGML_CUDA_MOE_CACHE_ATLAS_LOOKAHEAD");
        return !env || atoi(env) != 0; // on by default - see docs/plan.md
    }();
    return enabled;
}
static float moe_cache_atlas_lookahead_horizon() {
    static const float v = [] {
        const char * env = getenv("GGML_CUDA_MOE_CACHE_ATLAS_LOOKAHEAD_HORIZON");
        const float f = env ? (float) atof(env) : 3.0f;
        return f < 0.0f ? 0.0f : f;
    }();
    return v;
}
static void moe_cache_atlas_lookahead(const moe_cache_device & device, float & lx, float & ly) {
    lx = device.req_dir_x;
    ly = device.req_dir_y;
    if (moe_cache_atlas_lookahead_enabled()) {
        const float horizon = moe_cache_atlas_lookahead_horizon();
        lx += horizon * device.req_vel_x;
        ly += horizon * device.req_vel_y;
    }
}

static bool moe_cache_atlas_align_protect_enabled() {
    static const bool enabled = [] {
        const char * env = getenv("GGML_CUDA_MOE_CACHE_ATLAS_ALIGN_PROTECT");
        return !env || atoi(env) != 0;
    }();
    return enabled;
}

static float moe_cache_atlas_align_protect_strength() {
    static const float v = [] {
        const char * env = getenv("GGML_CUDA_MOE_CACHE_ATLAS_ALIGN_PROTECT_STRENGTH");
        const float f = env ? (float) atof(env) : 3.0f;
        return f < 0.0f ? 0.0f : f;
    }();
    return v;
}

// Same cosine-similarity-times-specialization formula moe_cache_atlas_rank
// uses to pick admission candidates, evaluated for one specific resident
// slot instead of ranked across a whole tensor's candidates. Self-contained
// (no forward declarations needed) since everything it touches - the atlas
// row lookup, req_dir - is already a plain struct member by this point in
// the file. Returns 0 for "no signal" (atlas disabled, no req_dir yet, no
// measured position for this expert, or genuinely unaligned) - the boost
// this feeds is multiplicative from a floor of 1.0, so 0 means no protection
// at all, never a penalty.
static float moe_cache_atlas_align_score(const moe_cache_device & device, const moe_cache_key & key) {
    if (!device.req_dir_valid || !key.tensor || key.expert < 0) {
        return 0.0f;
    }
    const auto it = device.atlas_by_tensor.find(key.tensor);
    if (it == device.atlas_by_tensor.end() || !it->second || it->second->empty()) {
        return 0.0f;
    }
    // atlas_row is a flat vector (moe_cache_atlas_rank iterates it the same
    // way), not a map - linear scan, bounded by one tensor's atlas entries.
    // Reached from reactive eviction (every real victim pick) and, sampled,
    // from the hit path (moe_cache_atlas_align_note_hit) - the eviction
    // caller is naturally rare; the hit caller samples explicitly so this
    // stays a small, occasional cost rather than landing on every hit.
    const moe_cache_atlas_cell * cellp = nullptr;
    for (const auto & [e, c] : *it->second) {
        if (e == key.expert) {
            cellp = &c;
            break;
        }
    }
    if (!cellp) {
        return 0.0f;
    }
    const auto & cell = *cellp;
    float rx, ry;
    moe_cache_atlas_lookahead(device, rx, ry);
    const float rmag = std::sqrt(rx * rx + ry * ry);
    const float cmag = std::sqrt(cell.x * cell.x + cell.y * cell.y);
    if (rmag < 1e-6f || cmag < 1e-6f) {
        return 0.0f;
    }
    // Not JUST for eviction anymore - see moe_cache_atlas_align_note_hit,
    // which now also calls this (sampled) from the real hit path to gather
    // outcome evidence. Still a small, bounded linear scan either way.
    const float score = ((cell.x * rx + cell.y * ry) / (cmag * rmag)) * cell.spec;
    return score > 0.0f ? score : 0.0f;
}

// #1 outcome evidence "for": a real hit on a currently atlas-aligned key -
// something the boost currently favours turned out to actually be reused.
// Sampled (1 in 16), not called on every hit - hits are the hottest path in
// this whole file, and align_score's linear scan is not free enough to pay
// on all of them just to gather confidence evidence.
static void moe_cache_atlas_align_note_hit(moe_cache_device & device, const moe_cache_key & key) {
    if (!moe_cache_atlas_align_protect_enabled()) {
        return;
    }
    static std::atomic<uint64_t> n{0};
    if ((n.fetch_add(1, std::memory_order_relaxed) & 0xF) != 0) {
        return;
    }
    if (moe_cache_atlas_align_score(device, key) > 0.0f) {
        device.atlas_align_evidence_for += 1.0;
    }
}

// #1: Kalman-gain-style confidence, computed from this session's own
// measured evidence rather than a fixed constant. atlas_align_protect_
// strength() below is now the CEILING; this scales it down when the boost
// has not been earning its keep and lets it approach the ceiling as
// evidence accumulates in its favour. (1,1) prior => 0.5 with no evidence.
static double moe_cache_atlas_align_confidence(const moe_cache_device & device) {
    const double f = device.atlas_align_evidence_for;
    const double a = device.atlas_align_evidence_against;
    return f / (f + a);
}

static float moe_cache_atlas_align_protect_strength_effective(const moe_cache_device & device) {
    return moe_cache_atlas_align_protect_strength() * (float) moe_cache_atlas_align_confidence(device);
}

// #2: how much of the pool's own current heat scale the align-protect boost
// is allowed to ADD, at most. See moe_cache_weighted_heat for why this is
// additive-and-capped rather than the multiplicative form it replaces.
static double moe_cache_atlas_align_protect_cap_fraction() {
    static const double v = [] {
        const char * env = getenv("GGML_CUDA_MOE_CACHE_ATLAS_ALIGN_PROTECT_CAP");
        const double f = env ? atof(env) : 0.5;
        return f < 0.0 ? 0.0 : f;
    }();
    return v;
}

// out_base, if non-null, receives the plain (no-atlas) score - callers that
// need to compare "what would plain LFRU have picked" against "what did the
// boost actually pick" (moe_cache_pick_coldest_unpinned's shadow evidence)
// get it for free instead of recomputing cost_tier_weight a second time.
//
// cap_ref selects what the cap is a fraction OF. Default (-1) falls back to
// pool.heat_scale_ema, a slow, pool-wide EMA - fine for moe_cache_pick_by_
// coverage, whose combination is rank-based anyway (bounded distortion by
// construction), but WRONG for a direct-magnitude comparison: heat_scale_ema
// is seeded low and only grows a little per hit, so for the entire span of
// a fresh session it stays near its seed - measured directly (2026-08-29):
// still ~4-8 after 30 evidence samples (~480 real hits). A magnitude-
// comparing caller (moe_cache_pick_coldest_unpinned, moe_cache_promote_to_
// protected) needs the cap to track the CURRENT competition, not a slowly-
// converging session-wide average, so those pass the local eviction
// window's own max base heat instead - instantaneous, no convergence lag,
// and exactly the scale the comparison is actually being made against.
//
// Additive-and-capped, not multiplicative: `w *= 1 + align*strength` scales
// whatever heat a slot already has, including near-zero - a barely-warmed
// atlas-aligned slot could outrank a genuinely hot, unaligned one purely
// because the multiplier had so little to work with that the distortion was
// invisible at the low end. Adding a bounded amount instead means the boost
// can tip a close call between similarly-warm candidates but can never
// manufacture rank-winning "heat" out of nothing.
static double moe_cache_weighted_heat(const moe_cache_device & device, const moe_cache_pool & pool,
        const moe_cache_slot & slot, double * out_base = nullptr, double cap_ref = -1.0) {
    const double base = (double) slot.heat * moe_cache_cost_tier_weight(device, slot.key);
    if (out_base) {
        *out_base = base;
    }
    if (!moe_cache_atlas_align_protect_enabled()) {
        return base;
    }
    const float align = moe_cache_atlas_align_score(device, slot.key);
    if (align <= 0.0f) {
        return base;
    }
    const double strength = (double) moe_cache_atlas_align_protect_strength_effective(device);
    // cap_ref, when supplied, is already a base score (heat*tier_weight) -
    // value-units-comparable as-is. The heat_scale_ema fallback is raw heat,
    // not value units, so it still needs this slot's own tier weight to
    // land in the same units base is in.
    const double cap = cap_ref >= 0.0
        ? moe_cache_atlas_align_protect_cap_fraction() * cap_ref
        : moe_cache_atlas_align_protect_cap_fraction() * pool.heat_scale_ema *
              moe_cache_cost_tier_weight(device, slot.key);
    const double bias = std::min((double) align * strength * (double) MOE_CACHE_HEAT_STEP, cap);
    return base + std::max(0.0, bias);
}

// protected_ is capped at half the pool - the structural fix for the
// v1 SLRU regression (uncapped: 60.2-60.3% hit rate, protected_ grew to
// 598-600/604, starving probation). When promotion needs to make room,
// the slot that pays for it is chosen by heat, not recency: scan a
// bounded window from protected_head (the segment's own oldest-recency
// end) and demote whichever of those is coldest, with hysteresis so a
// marginal difference doesn't flip-flop the choice run to run. Plain
// oldest-first demotion (capped-SLRU, no heat tiebreak) already measured
// well (65.0%) - this only changes *which* of the stale-recency
// candidates gets picked when several are in contention.
static constexpr int MOE_CACHE_PROTECTED_CAP_PCT = 50;

// Coldest-first, bounded-window pick from a segment, skipping anything with
// active readers. Shared by real-miss eviction and, now, speculative-fill
// eviction: same criterion used to choose what to keep should decide what to
// let go, rather than a separate, unprincipled policy for each path.
static constexpr size_t MOE_CACHE_MAX_PARTNERS = 32;

// Coverage-aware eviction. Measured offline on 909k real decisions at 1.2%
// residency: LRU alone retains 78.30% of the router's intended gate mass;
// ranking the LRU window by coverage(1.0) + atlas(0.5) + score(0.5) retains
// 84.47%, a +6.18pp gain, by cutting the "no resident stand-in" tail from
// 17.69% to 13.06%. It deliberately GIVES UP hit rate (-2.8pp) to do so, which
// is why no hit-rate-scored experiment ever found it.
//
// Weights are not arbitrary: coverage measures STRUCTURE (would evicting this
// orphan a neighbourhood) while atlas and score both measure an individual
// expert's worth, so coverage is weighted equal to their sum. Combining by RANK
// rather than by value is deliberate too - the three quantities have
// incommensurable scales, and every value-space combination tried (raw product,
// calibrated log-additive) lost to rank aggregation because one factor's range
// dominated. See docs/plan.md.
static bool moe_cache_coverage_evict_enabled() {
    static const bool on = [] {
        const char * e = getenv("GGML_CUDA_MOE_CACHE_COVERAGE_EVICT");
        return e && atoi(e) != 0;
    }();
    return on;
}

// Rank-combined victim choice over the same LRU window the heat-only scan uses.
// Three signals, each ranked most-evictable-first, summed with weights:
//   coverage 1.0  how many of this expert's co-firing partners are still
//                 resident - evicting a redundant expert leaves a stand-in
//   atlas    0.5  alignment with the live request direction; misaligned first
//   heat     0.5  the existing weighted heat, coldest first
// Measured +6.18pp retained gate mass at 1.2% residency. See the comment on
// moe_cache_coverage_evict_enabled.
static int moe_cache_pick_by_coverage(const moe_cache_device & device, moe_cache_pool & pool, int head) {
    struct cand_t { int slot; int redundancy; double align; double heat; };
    // Candidates from BOTH segments. The heat-only scan walks probation alone,
    // so every candidate is a fresh one-off admission with near-identical
    // redundancy and the coverage ranking has nothing to discriminate - which
    // is why ranking probation-only did not reproduce the offline +6.18pp. The
    // simulation's pool was the whole resident set, hot experts included, and
    // telling those apart is where coverage earns its gain.
    //
    // This DELIBERATELY makes protected slots evictable, which weakens LFRU's
    // guarantee that a proven-hot expert stays resident. That is the trade the
    // coverage objective asks for: it gives up hit rate to keep a viable
    // stand-in reachable. Off by default, behind COVERAGE_EVICT.
    cand_t cands[2 * MOE_CACHE_EVICT_WINDOW];
    int n = 0;
    const int heads[2] = { head, pool.protected_head };
    for (int h = 0; h < 2; h++) {
        int taken = 0;
        for (int c = heads[h]; c >= 0 && taken < MOE_CACHE_EVICT_WINDOW; c = pool.slots[c].next) {
            if (pool.slots[c].readers > 0) {
                continue;
            }
            taken++;
            const moe_cache_slot & slot = pool.slots[c];
            int redundancy = 0;
            const auto pit = device.partners.find(slot.key);
            if (pit != device.partners.end()) {
                for (int32_t other : pit->second) {
                    if (pool.map.find(moe_cache_key{slot.key.tensor, other}) != pool.map.end()) {
                        redundancy++;
                    }
                }
            }
            double align = 0.0;
            if (device.req_dir_valid) {
                const auto ait = device.atlas.find(slot.key);
                if (ait != device.atlas.end()) {
                    const double mx = device.req_dir_x, my = device.req_dir_y;
                    const double m = std::sqrt(mx*mx + my*my);
                    const double cl = std::sqrt((double)ait->second.x*ait->second.x +
                                                (double)ait->second.y*ait->second.y);
                    if (m > 1e-9 && cl > 1e-9) {
                        align = (ait->second.x*mx + ait->second.y*my) / (m*cl);
                    }
                }
            }
            cands[n++] = { c, redundancy, align, moe_cache_weighted_heat(device, pool, slot) };
        }
    }
    if (n == 0) {
        return -1;
    }
    double total[2 * MOE_CACHE_EVICT_WINDOW] = {};
    int order[2 * MOE_CACHE_EVICT_WINDOW];
    // most redundant first (weight 1.0)
    for (int i = 0; i < n; i++) order[i] = i;
    std::sort(order, order + n, [&](int a, int b){ return cands[a].redundancy > cands[b].redundancy; });
    for (int r = 0; r < n; r++) total[order[r]] += 1.0 * r;
    // least aligned with the request direction first (weight 0.5)
    for (int i = 0; i < n; i++) order[i] = i;
    std::sort(order, order + n, [&](int a, int b){ return cands[a].align < cands[b].align; });
    for (int r = 0; r < n; r++) total[order[r]] += 0.5 * r;
    // coldest first (weight 0.5)
    // NOTE: the simulation's third signal was the ROUTER SCORE EMA, which the
    // cache cannot see - plan() receives ids, never the gate weights. Heat is a
    // proxy for it and may not behave the same, so its weight is tunable and
    // can be set to 0 to test the coverage+atlas pair the simulation scored at
    // +5.01pp on its own.
    static const double heat_w = [] {
        const char * e = getenv("GGML_CUDA_MOE_CACHE_COVERAGE_HEAT_W");
        return e ? atof(e) : 0.5;
    }();
    if (heat_w != 0.0) {
        for (int i = 0; i < n; i++) order[i] = i;
        std::sort(order, order + n, [&](int a, int b){ return cands[a].heat < cands[b].heat; });
        for (int r = 0; r < n; r++) total[order[r]] += heat_w * r;
    }

    int best = 0;
    for (int i = 1; i < n; i++) {
        if (total[i] < total[best]) best = i;
    }
    return cands[best].slot;
}

static int moe_cache_pick_coldest_unpinned(moe_cache_device & device, moe_cache_pool & pool, int head) {
    if (moe_cache_coverage_evict_enabled()) {
        return moe_cache_pick_by_coverage(device, pool, head);
    }
    // Two passes: the align-protect cap needs to know THIS window's own max
    // base heat before it can bound anything (see moe_cache_weighted_heat's
    // cap_ref) - pass 1 gathers plain base heat (no atlas lookup, cheap) and
    // the window max; pass 2 scores each candidate against that max instead
    // of the slow pool-wide heat_scale_ema, so the cap tracks the actual
    // current competition instead of lagging a session behind it.
    int cand[MOE_CACHE_EVICT_WINDOW];
    double cand_base[MOE_CACHE_EVICT_WINDOW];
    int n = 0;
    double window_max_base = 0.0;
    int candidate = head;
    for (int seen = 0; candidate >= 0 && seen < MOE_CACHE_EVICT_WINDOW; candidate = pool.slots[candidate].next) {
        if (pool.slots[candidate].readers > 0) {
            continue;
        }
        seen++;
        const double base = (double) pool.slots[candidate].heat * moe_cache_cost_tier_weight(device, pool.slots[candidate].key);
        cand[n] = candidate;
        cand_base[n] = base;
        if (base > window_max_base) {
            window_max_base = base;
        }
        n++;
    }
    if (n == 0) {
        return -1;
    }
    int best = -1;
    double best_score = 0.0;
    // #3 shadow evidence: the same window's plain (unboosted) pick, tracked
    // alongside the real one for free - see the evidence update below.
    int plain_best = -1;
    double plain_best_score = 0.0;
    for (int i = 0; i < n; i++) {
        const double score = moe_cache_weighted_heat(device, pool, pool.slots[cand[i]], nullptr, window_max_base);
        if (best < 0 || moe_cache_colder_enough(best_score, score)) {
            best = cand[i];
            best_score = score;
        }
        if (plain_best < 0 || moe_cache_colder_enough(plain_best_score, cand_base[i])) {
            plain_best = cand[i];
            plain_best_score = cand_base[i];
        }
    }
    if (best >= 0 && moe_cache_atlas_align_protect_enabled()) {
        // #1 outcome evidence "against": the winner, whatever it is, is
        // about to actually be evicted - if it was aligned enough to have
        // been boosted, the boost did not save it from this.
        const float align = moe_cache_atlas_align_score(device, pool.slots[best].key);
        if (align > 0.0f) {
            device.atlas_align_evidence_against += 1.0;
        }
        // #3 shadow evidence: did the boost change the outcome versus plain
        // heat/cost-tier scoring alone, in the same window? Agreement is
        // free confirmation the boost isn't fighting the base mechanism;
        // disagreement isn't yet known-good or known-bad, so it counts as
        // mild "against" until outcome evidence resolves it - the same
        // conservative-by-default posture the free-slot-only admission
        // redesign already settled on.
        if (plain_best == best) {
            device.atlas_align_evidence_for += 0.25;
        } else {
            device.atlas_align_evidence_against += 0.25;
        }
    }
    // Predictor admission outcome evidence: the winner is about to be
    // evicted - if it is still in predictor_admitted, it was never hit even
    // once since predictor warming put it here (a real hit already resolved
    // and erased it - moe_cache_predictor_note_hit), so this admission did
    // not pay off. The set only ever holds keys still awaiting their first
    // outcome, one way or the other.
    if (best >= 0 && !device.predictor_admitted.empty()) {
        const auto pit = device.predictor_admitted.find(pool.slots[best].key);
        if (pit != device.predictor_admitted.end()) {
            device.predictor_admit_evidence_against += 1.0;
            device.predictor_admitted.erase(pit);
        }
    }
    return best;
}

static void moe_cache_promote_to_protected(const moe_cache_device & device, moe_cache_pool & pool, int index) {
    moe_cache_segment_remove(pool, index);
    const int cap = std::max(1, pool.n_slots * MOE_CACHE_PROTECTED_CAP_PCT / 100);
    if (pool.protected_count >= cap && pool.protected_head >= 0) {
        // Same two-pass window-local cap reference as
        // moe_cache_pick_coldest_unpinned - see that function's comment.
        int cand[MOE_CACHE_EVICT_WINDOW];
        double cand_base[MOE_CACHE_EVICT_WINDOW];
        int n = 0;
        double window_max_base = 0.0;
        int candidate = pool.protected_head;
        for (int seen = 0; candidate >= 0 && seen < MOE_CACHE_EVICT_WINDOW; candidate = pool.slots[candidate].next, seen++) {
            const double base = (double) pool.slots[candidate].heat * moe_cache_cost_tier_weight(device, pool.slots[candidate].key);
            cand[n] = candidate;
            cand_base[n] = base;
            if (base > window_max_base) {
                window_max_base = base;
            }
            n++;
        }
        int demote = cand[0];
        double demote_score = moe_cache_weighted_heat(device, pool, pool.slots[demote], nullptr, window_max_base);
        for (int i = 1; i < n; i++) {
            const double score = moe_cache_weighted_heat(device, pool, pool.slots[cand[i]], nullptr, window_max_base);
            if (moe_cache_colder_enough(demote_score, score)) {
                demote = cand[i];
                demote_score = score;
            }
        }
        moe_cache_segment_remove(pool, demote);
        moe_cache_segment_push_back(pool, demote, moe_cache_segment::probation);
    }
    moe_cache_segment_push_back(pool, index, moe_cache_segment::protected_);
}

// Set or clear one expert's bit. The word vector grows on demand rather than
// being sized from n_expert: not every caller that admits an expert has the
// tensor's expert count to hand, and growing costs one reallocation per tensor
// over the whole run (256 experts = 4 words).
static void moe_cache_mask_set(moe_cache_pool & pool, const moe_cache_key & key, bool resident) {
    if (!key.tensor || key.expert < 0) {
        return;
    }
    const size_t word = (size_t) key.expert >> 6;
    const uint64_t bit = 1ull << ((size_t) key.expert & 63);
    if (!resident) {
        auto it = pool.resident_mask.find(key.tensor);
        if (it != pool.resident_mask.end() && word < it->second.size()) {
            it->second[word] &= ~bit;
        }
        return;
    }
    try {
        auto & words = pool.resident_mask[key.tensor];
        if (word >= words.size()) {
            words.resize(word + 1, 0ull);
        }
        words[word] |= bit;
    } catch (...) {
        // Best-effort, exactly like the other side tables here: the mask is a
        // read accelerator for `map`, never the source of truth, so failing to
        // grow it must not fail a decode.
    }
}

// True only if this expert currently occupies a valid slot. Cheap enough to
// call across a whole candidate window; the caller is expected to hoist the
// per-tensor lookup out of that loop via moe_cache_mask_words().
static const std::vector<uint64_t> * moe_cache_mask_words(const moe_cache_pool & pool, const void * tensor) {
    auto it = pool.resident_mask.find(tensor);
    return it == pool.resident_mask.end() ? nullptr : &it->second;
}

static inline bool moe_cache_mask_test(const std::vector<uint64_t> * words, int32_t expert) {
    if (!words || expert < 0) {
        return false;
    }
    const size_t word = (size_t) expert >> 6;
    return word < words->size() && ((*words)[word] >> ((size_t) expert & 63)) & 1ull;
}

// Single admission choke point, mirroring moe_cache_map_erase below. Returns
// false when the key was already mapped, matching emplace().second so callers
// keep their existing error handling.
static bool moe_cache_map_insert(moe_cache_pool & pool, const moe_cache_key & key, int index) {
    if (!pool.map.emplace(key, index).second) {
        return false;
    }
    moe_cache_mask_set(pool, key, true);
    return true;
}

static void moe_cache_map_erase(moe_cache_pool & pool, int index) {
    moe_cache_slot & slot = pool.slots[index];
    auto it = pool.map.find(slot.key);
    if (it != pool.map.end() && it->second == index) {
        pool.map.erase(it);
        moe_cache_mask_set(pool, slot.key, false);
    }
}

// Proves the mask agrees with `map`, both directions. Off unless
// GGML_CUDA_MOE_CACHE_MASK_AUDIT is set - it walks every slot and every mask
// bit, which is far too heavy for the dispatch path. Returns the number of
// disagreements, so 0 is the assertion.
static int moe_cache_mask_audit(const moe_cache_pool & pool) {
    int bad = 0;
    for (const auto & kv : pool.map) {
        if (!moe_cache_mask_test(moe_cache_mask_words(pool, kv.first.tensor), kv.first.expert)) {
            bad++; // mapped but not marked resident
        }
    }
    for (const auto & kv : pool.resident_mask) {
        for (size_t w = 0; w < kv.second.size(); w++) {
            uint64_t bits = kv.second[w];
            while (bits) {
                const int e = (int) (w << 6) + __builtin_ctzll(bits);
                bits &= bits - 1;
                if (pool.map.find(moe_cache_key{kv.first, e}) == pool.map.end()) {
                    bad++; // marked resident but not mapped
                }
            }
        }
    }
    return bad;
}

static bool moe_cache_pin_audit_enabled() {
    static const bool on = [] {
        const char * e = getenv("GGML_CUDA_MOE_CACHE_PIN_AUDIT");
        return e && atoi(e) != 0;
    }();
    return on;
}

static void moe_cache_slot_reset(moe_cache_pool & pool, int index, bool add_to_free) {
    moe_cache_slot & slot = pool.slots[index];
    // slot.readers is zeroed unconditionally below. If anyone still holds this
    // slot, that pin is silently discarded and the slot can be refilled with a
    // DIFFERENT expert while a dispatch still references it. That is the pin
    // discipline breaking, and it is invisible afterwards because the refcount
    // is gone.
    if (moe_cache_pin_audit_enabled() && slot.readers > 0) {
        static std::atomic<int> n{0};
        if (n.fetch_add(1, std::memory_order_relaxed) < 20) {
            fprintf(stderr, "[moe-cache] *** PIN VIOLATION: slot_reset on slot=%d with readers=%d "
                    "(expert=%d state=%d) - pin discarded\n",
                    index, slot.readers, slot.key.expert, (int) slot.state);
        }
    }
    if (slot.state == moe_cache_slot_state::valid) {
        moe_cache_segment_remove(pool, index);
    }
    moe_cache_map_erase(pool, index);
    slot.key = {};
    slot.generation++;
    slot.readers = 0;
    slot.heat = 0; // heat belongs to the content that was resident, not the physical slot
    slot.state = moe_cache_slot_state::free;
    slot.segment = moe_cache_segment::probation;
    slot.prev = -1;
    slot.next = -1;
    if (add_to_free) {
        pool.free_slots.push_back(index);
    }
}

static bool moe_cache_cuda_ok(
        moe_cache_device & device, cudaError_t error, const char * operation, bool fatal) {
    if (error == cudaSuccess) {
        return true;
    }

    (void)cudaGetLastError();
    if (device.error_logs.fetch_add(1) < 8) {
        MOE_CACHE_LOG("[moe-cache] CUDA%d %s failed: %s\n",
                device.physical, operation, cudaGetErrorString(error));
        if (getenv("MOE_CACHE_DEBUG_GATE")) {
            fprintf(stderr, "[moe-cache-err] CUDA%d %s failed: %s\n",
                    device.physical, operation, cudaGetErrorString(error));
        }
    }
    if (fatal) {
        // cudaErrorMemoryAllocation used to be exempt here, on the reasoning
        // that allocation failures are recoverable (the caller retries with a
        // smaller size). But that exemption also covered OOM on the COPY
        // paths, where there is nothing to retry and the buffer is left
        // holding someone else's bytes - so an out-of-memory device kept
        // serving wrong weights indefinitely. Allocation sites that genuinely
        // want to retry pass fatal=false and are unaffected.
        device.dead.store(true);
    }
    return false;
}

// A failed copy means some buffer now holds the wrong bytes. One can be a
// transient; a run of them means the device is out of memory or otherwise
// unable to serve this cache, and every subsequent "hit" is a chance to
// feed garbage into a GEMM. Disable the cache instead - the offload path
// still works without it, just slower. Logged once, loudly, because the
// symptom otherwise (fluent-looking nonsense) points nowhere near the cause.
// TEST ONLY. Makes every Nth prefill copy report failure, so the recovery
// path can be exercised deterministically instead of hoping to recreate the
// original race. That race (a copy failing under VRAM pressure, the slot
// being published anyway, and the stale buffer then read as expert weights)
// produced fluent-looking garbage and took a long diagnosis precisely
// because it could not be reproduced on demand - 7 configurations came back
// clean before the chat-templated prompt turned out to be the trigger. A
// defensive fix that cannot be tested is not a fix, it is a hope.
// Off unless GGML_CUDA_MOE_CACHE_FAULT_INJECT is set to a positive N.
static bool moe_cache_inject_copy_fault() {
    static const int every = [] {
        const char * env = getenv("GGML_CUDA_MOE_CACHE_FAULT_INJECT");
        const int v = env ? atoi(env) : 0;
        return v < 0 ? 0 : v;
    }();
    if (every <= 0) {
        return false;
    }
    static std::atomic<long long> n{0};
    return (n.fetch_add(1) + 1) % every == 0;
}

static void moe_cache_note_copy_failure(moe_cache_device & device) {
    static const int limit = [] {
        const char * env = getenv("GGML_CUDA_MOE_CACHE_MAX_COPY_FAILURES");
        const int v = env ? atoi(env) : 8;
        return v < 1 ? 1 : v;
    }();
    if (device.copy_failures.fetch_add(1) + 1 >= limit && !device.dead.exchange(true)) {
        MOE_CACHE_LOG("[moe-cache] CUDA%d DISABLED after %d copy failures - "
                "continuing would risk serving wrong expert weights. "
                "Most likely cause: another process is using this GPU's memory. "
                "Inference continues on the uncached offload path.\n",
                device.physical, limit);
        fprintf(stderr, "[moe-cache] CUDA%d DISABLED after %d copy failures "
                "(likely VRAM exhaustion) - falling back to uncached offload\n",
                device.physical, limit);
    }
}

static bool moe_cache_grow_device(
        moe_cache_device & device, void ** pointer, size_t & capacity,
        size_t required, const char * operation) {
    if (capacity >= required) {
        return true;
    }
    if (required > (std::numeric_limits<size_t>::max() - 256) / 2) {
        return false;
    }

    // GGML_CUDA_MOE_CACHE_GROWTH_LOG=1: this cudaMalloc/cudaFree pair runs
    // synchronously on the DECODE thread, from moe_cache_dispatch, whenever a
    // batch's real row count exceeds every previous one - not at build time,
    // not on the fill worker. cudaMalloc during active CUDA graph capture is
    // documented as unsafe; this is the one call site in the whole file that
    // fires conditionally, exactly when a batch is bigger than any seen
    // before, which is exactly the shape of "first n_tokens=4 batch, or first
    // dispatch of a new tensor's (n_in,n_out) shape, corrupts". Logs whether
    // a CUDA graph capture is active on our own compute_stream at this exact
    // moment - the direct test, not an inference from timing.
    static const bool growth_log = [] {
        const char * e = getenv("GGML_CUDA_MOE_CACHE_GROWTH_LOG");
        return e && atoi(e) != 0;
    }();
    if (growth_log) {
        cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
        cudaError_t cap_err = device.compute_stream
            ? cudaStreamIsCapturing(device.compute_stream, &capture_status)
            : cudaErrorInvalidValue;
        fprintf(stderr, "[moe-cache] GROWTH op=%s old_cap=%zu required=%zu "
                "compute_stream=%p capturing=%s(err=%d)\n",
                operation, capacity, required, (void *) device.compute_stream,
                cap_err == cudaSuccess
                    ? (capture_status == cudaStreamCaptureStatusActive ? "ACTIVE" : "none")
                    : "unknown",
                (int) cap_err);
        fflush(stderr);
    }

    // GGML_CUDA_MOE_CACHE_PRESIZE_X=N: multiply the FIRST-ever growth request
    // by N instead of the normal 2x+256, so in practice no second mid-decode
    // cudaMalloc ever fires for the rest of the run - the direct test of
    // "is the mid-decode cudaMalloc itself (not graph capture, not a
    // free-race - both already excluded) the cause", verified via
    // GROWTH_LOG's event count, not assumed from this multiplier alone.
    static const long long presize_x = [] {
        const char * e = getenv("GGML_CUDA_MOE_CACHE_PRESIZE_X");
        return e ? atoll(e) : 1;
    }();
    const size_t requested = capacity == 0 && presize_x > 1
        ? required * (size_t) presize_x + 256
        : required * 2 + 256;
    void * fresh = nullptr;
    if (!moe_cache_cuda_ok(device, cudaMalloc(&fresh, requested), operation, false)) {
        return false;
    }
    // Poison the whole allocation (async, on our own stream - the real write
    // that follows is issued on the same stream right after this call
    // returns, so stream ordering alone guarantees it lands first, no extra
    // sync needed). Placement guard for these scratch buffers, same idea as
    // the per-slot guard in the weight slab: whatever legitimately writes
    // here only ever touches the bytes it asked for, so anything found
    // outside that later is direct proof of an overrun, not an assumption.
    cudaMemsetAsync(fresh, MOE_CACHE_GUARD_PATTERN, requested, device.compute_stream);
    if (*pointer) {
        // GGML_CUDA_MOE_CACHE_GROWTH_SYNC=1: synchronize our own compute_stream
        // before freeing the OLD scratch buffer, so any kernel/copy still
        // reading it (queued earlier on this same stream, or from elsewhere)
        // is provably finished first. Independent of the graph-capture theory
        // (already excluded: GRAPHS DISABLED still corrupts) - this tests
        // whether growth's free/realloc itself races an in-flight reader.
        static const bool growth_sync = [] {
            const char * e = getenv("GGML_CUDA_MOE_CACHE_GROWTH_SYNC");
            return e && atoi(e) != 0;
        }();
        if (growth_sync && device.compute_stream) {
            cudaStreamSynchronize(device.compute_stream);
        }
        cudaFree(*pointer);
    }
    *pointer = fresh;
    capacity = requested;
    return true;
}

// Like moe_cache_grow_device, but allocates exactly `required` bytes instead
// of doubling. moe_cache_grow_device's 2x headroom amortizes the cost of a
// pool that's expected to grow repeatedly over its lifetime - right for the
// resizable pools it's normally used for, wrong for a prefill slab: each
// distinct tensor byte-size gets its own slab (see
// moe_cache_device::prefill_slabs), sized once at registration from the
// tensor's own, fixed, architecture-determined byte count, and never resized
// again. Doubling there would roughly triple this feature's real VRAM
// footprint (two slots per slab, three-ish distinct tensor sizes in a
// typical model) for no benefit - nothing ever needs a slab bigger than the
// one exact size it was created for.
static bool moe_cache_grow_device_exact(
        moe_cache_device & device, void ** pointer, size_t & capacity,
        size_t required, const char * operation) {
    if (capacity >= required) {
        return true;
    }
    void * fresh = nullptr;
    if (!moe_cache_cuda_ok(device, cudaMalloc(&fresh, required), operation, false)) {
        return false;
    }
    if (*pointer) {
        cudaFree(*pointer);
    }
    *pointer = fresh;
    capacity = required;
    return true;
}

static size_t moe_cache_growth_capacity(size_t capacity, size_t required) {
    if (capacity >= required) {
        return capacity;
    }
    if (required > (std::numeric_limits<size_t>::max() - 256) / 2) {
        return 0;
    }
    return required * 2 + 256;
}

// max_rows is the session's actual configured max_batch, not the structural
// ceiling (GGML_MOE_CACHE_MAX_BATCH_ROWS) - reservation must scale with what
// a session can really see, or every session pays worst-case-ceiling scratch
// reservation regardless of its configured batch size. Bounded by the
// ceiling below since config.max_batch is itself clamped to it.
static bool moe_cache_scratch_requirements(
        int64_t n_in, int64_t n_out, int max_batch, moe_cache_scratch & result) {
    if (max_batch <= 0 || max_batch > GGML_MOE_CACHE_MAX_BATCH_ROWS) {
        return false;
    }
    const size_t max_rows = (size_t) max_batch;
    if (n_in <= 0 || n_out <= 0 ||
        n_in > INT64_MAX - (MATRIX_ROW_PADDING - 1)) {
        return false;
    }
    const int64_t padded_n_in =
        ((n_in + MATRIX_ROW_PADDING - 1) / MATRIX_ROW_PADDING) * MATRIX_ROW_PADDING;
    if ((uint64_t)n_in > SIZE_MAX / (max_rows * sizeof(float)) ||
        (uint64_t)n_out > SIZE_MAX / (max_rows * sizeof(float)) ||
        (uint64_t)(padded_n_in / QK8_1) >
            SIZE_MAX / (max_rows * sizeof(block_q8_1))) {
        return false;
    }

    const size_t capacities[] = {
        moe_cache_growth_capacity(0, max_rows * sizeof(int32_t)),
        moe_cache_growth_capacity(0, max_rows * (size_t)n_in * sizeof(float)),
        moe_cache_growth_capacity(
                0, max_rows * (size_t)(padded_n_in / QK8_1) * sizeof(block_q8_1)),
        moe_cache_growth_capacity(0, max_rows * (size_t)n_out * sizeof(float)),
    };
    for (size_t capacity : capacities) {
        if (capacity == 0) {
            return false;
        }
    }
    result = {capacities[0], capacities[1], capacities[2], capacities[3]};
    return true;
}

static size_t moe_cache_scratch_total(
        const moe_cache_scratch & current,
        const moe_cache_scratch * additional = nullptr) {
    const size_t capacities[] = {
        additional ? std::max(current.ids, additional->ids) : current.ids,
        additional ? std::max(current.act, additional->act) : current.act,
        additional ? std::max(current.q8, additional->q8) : current.q8,
        additional ? std::max(current.out, additional->out) : current.out,
    };
    size_t total = 0;
    for (size_t capacity : capacities) {
        if (capacity > SIZE_MAX - total) {
            return SIZE_MAX;
        }
        total += capacity;
    }
    return total;
}

static bool moe_cache_grow_host(
        moe_cache_device & device, void ** pointer, size_t & capacity,
        size_t required, const char * operation) {
    if (capacity >= required) {
        return true;
    }
    if (required > (std::numeric_limits<size_t>::max() - 256) / 2) {
        return false;
    }

    static const long long presize_x = [] {
        const char * e = getenv("GGML_CUDA_MOE_CACHE_PRESIZE_X");
        return e ? atoll(e) : 1;
    }();
    const size_t requested = capacity == 0 && presize_x > 1
        ? required * (size_t) presize_x + 256
        : required * 2 + 256;
    void * fresh = nullptr;
    if (!moe_cache_cuda_ok(device, cudaMallocHost(&fresh, requested), operation, false)) {
        return false;
    }
    if (*pointer) {
        cudaFreeHost(*pointer);
    }
    *pointer = fresh;
    capacity = requested;
    return true;
}

// v1: single-device. Prefill's first access to a tensor may be the very
// first MoE access of the whole run (a pure prompt-processing benchmark
// never calls the decode path at all), so this can't rely on
// session->tensor_devices - that map is only populated by moe_cache_begin(),
// which large (>= op_offload_min_batch_size) batches never reach. Multi-GPU
// tensor-split would need real per-tensor device resolution; out of scope
// for this increment.
static moe_cache_device * moe_cache_prefill_first_device() {
    std::lock_guard<std::mutex> registry_lock(g_registry_mu);
    for (moe_cache_session * session : g_sessions) {
        if (session->stopping) {
            continue;
        }
        for (const auto & device_ptr : session->devices) {
            if (!device_ptr->dead.load()) {
                return device_ptr.get();
            }
        }
    }
    return nullptr;
}

// Same lookup, but also hands back the owning session - needed by the hit-D2D
// split in moe_cache_prefill_advance, which has to read that session's own
// LFRU pools (session->mu protects pool/slot state, not device.dispatch_mu -
// see the lock-ordering note there) to classify each expert row as resident
// or not.
static moe_cache_device * moe_cache_prefill_first_session_and_device(moe_cache_session ** out_session) {
    if (out_session) {
        *out_session = nullptr;
    }
    std::lock_guard<std::mutex> registry_lock(g_registry_mu);
    for (moe_cache_session * session : g_sessions) {
        if (session->stopping) {
            continue;
        }
        for (const auto & device_ptr : session->devices) {
            if (!device_ptr->dead.load()) {
                if (out_session) {
                    *out_session = session;
                }
                return device_ptr.get();
            }
        }
    }
    return nullptr;
}

// EXPERIMENTAL, default OFF - see the matching flag and long comment in
// ggml_cuda_mul_mat_id (ggml-cuda.cu). This function still checks it
// independently rather than trusting callers not to invoke it directly,
// since a stray call would issue real CUDA work (allocation, stream/event
// creation) for nothing.
static bool moe_cache_prefill_buffer_enabled() {
    static const bool enabled = getenv("GGML_CUDA_MOE_PREFILL_BUFFER") != nullptr;
    return enabled;
}

// Registers "when host_base's node is dispatched, kick off a prefetch of
// next_host_base" and eagerly allocates everything that prefetch will need.
// Must be called at graph-BUILD time (once per layer, from the
// architecture's model-building code - see build_moe_prefill_prefetch in
// llama-graph.cpp), never at compute time.
//
// That timing requirement is the whole point of this function existing
// separately from the advance step below: CUDA graph capture can only ever
// be active during compute (llama_context::graph_compute), strictly after
// the ggml graph has already been fully built, so a call made during BUILD
// can never race an active capture - it is unconditionally safe to
// cudaMalloc here. moe_cache_prefill_advance (the thing that actually runs
// during compute, possibly inside a captured region) deliberately never
// allocates anything for exactly this reason; it only ever touches buffers
// this function already created.
static void moe_cache_prefill_register_successor(
        const void * host_base, const void * next_host_base, size_t next_tensor_bytes,
        int64_t next_n_expert) {
    if (!moe_cache_prefill_buffer_enabled() || !host_base || !next_host_base || next_tensor_bytes == 0) {
        return;
    }
    moe_cache_device * device_ptr = moe_cache_prefill_first_device();
    if (!device_ptr) {
        return;
    }
    moe_cache_device & device = *device_ptr;
    std::lock_guard<std::mutex> lock(device.dispatch_mu);

    moe_cache_prefill_slab & slab = device.prefill_slabs[next_tensor_bytes];
    for (int i = 0; i < 2; i++) {
        if (!moe_cache_grow_device_exact(device, &slab.dev[i], slab.cap[i], next_tensor_bytes,
                "prefill slab (register)")) {
            return;
        }
        if (!slab.ready[i] && !moe_cache_cuda_ok(device,
                cudaEventCreateWithFlags(&slab.ready[i], cudaEventDisableTiming),
                "prefill event creation (register)", false)) {
            return;
        }
        if (!slab.consumed[i] && !moe_cache_cuda_ok(device,
                cudaEventCreateWithFlags(&slab.consumed[i], cudaEventDisableTiming),
                "prefill consumed-event creation (register)", false)) {
            return;
        }
    }
    if (!device.prefill_stream && !moe_cache_cuda_ok(device,
            cudaStreamCreateWithFlags(&device.prefill_stream, cudaStreamNonBlocking),
            "prefill stream creation (register)", false)) {
        return;
    }
    // Fixed slot for this tensor - decided once, the first time this exact
    // next_host_base is ever registered, never touched again. See the
    // moe_cache_prefill_slab struct comment for why.
    if (slab.assigned_slot.find(next_host_base) == slab.assigned_slot.end()) {
        slab.assigned_slot[next_host_base] = slab.assign_next & 1;
        slab.assign_next++;
    }
    device.prefill_successors[host_base] = {next_host_base, next_tensor_bytes, next_n_expert};
}

// A cudaEvent_t recorded while a stream was NOT capturing (e.g. during the
// uncaptured warmup pass llama.cpp always runs before it starts capturing a
// shape) can't validly be waited on once that same stream is inside a graph
// capture - the driver rejects it as "a previous error during capture" on
// some later, unrelated kernel launch, since the failure itself isn't
// reported at the cudaStreamWaitEvent call that caused it. consumed[slot]
// crosses exactly that boundary: it's recorded by prefill_release (from the
// consumer's ctx.stream(), which is whatever the warmup pass happened to be
// using) and waited on later by advance(), possibly after capture has since
// started on that same stream.
//
// llama.cpp's own graph splitting for this op_offload scenario turns out to
// capture roughly one region per layer, not one region for the whole
// ubatch/token (confirmed empirically: cudaStreamGetCaptureInfo's capture id
// changes on origin_stream between one MoE layer's dispatch and the next).
// That makes any synchronization scheme built on *captured* cross-stream
// dependencies (cudaStreamWaitEvent recorded as a graph node) unusable for
// this "prefetch N, consume N two layers later" pipeline: producer and
// consumer are essentially always in two different captures, and a captured
// wait can only reference an event recorded within the same capture -
// referencing one from a different (or no) capture is rejected by the
// driver, surfacing later as "operation failed due to a previous error
// during capture" on some unrelated kernel launch rather than at the call
// that actually caused it.
//
// So this pipeline does not try to make prefill_stream's copy part of any
// capture at all. Under cudaStreamCaptureModeRelaxed (what llama.cpp
// captures with), work issued to a stream other than the one being captured
// runs as ordinary, immediate async work, invisible to the capture - exactly
// what's wanted here. The dependency between the copy and the consumer is
// instead enforced with a real, host-blocking cudaEventSynchronize at
// prefill_wait time (see below): CPU-side, unrelated to whatever capture
// origin_stream is or isn't in, and always valid regardless of capture
// boundaries. It only actually blocks if the copy hasn't finished yet, which
// in steady state (prefetch running a full layer or more ahead of compute)
// it already has. The cost of this design is that it only pays off across
// *replays* of a captured region, not within the one-time capture recording
// itself - but capture recording is a one-time event per shape (warmup),
// while replay is the steady-state path token after token, so that's the
// right place to have paid for it once mmap/pinning didn't already cover.

// Userdata for moe_cache_prefill_release_readers_cb below.
struct moe_cache_prefill_pin {
    moe_cache_pool * pool;
    int slot;
    // Generation observed at pin time (mirrors moe_cache_pin, the decode-time
    // hit-serving path's own pin struct - see its comment). Without this, the
    // release callback below has no way to tell "this slot still holds what I
    // pinned" from "this slot index was reset and reused for something else
    // while my copy was in flight" - and unconditional slot.readers = 0 resets
    // exist elsewhere (prewarm_from_history, eviction, teardown) that don't
    // know about this pin at all. A generation-blind release can then
    // decrement a completely unrelated (tensor, expert)'s reader count,
    // corrupting whatever pin protection *that* occupant's own reader was
    // relying on - a real bug found investigating qwen4exp corruption that
    // reproduced with byte-correct D2D copies (see docs/plan.md): the copy
    // itself was always right, but this generation-blind release could let a
    // slot become eviction-eligible while a real reader elsewhere still
    // depended on it staying pinned.
    uint64_t generation;
};

struct moe_cache_prefill_reader_release_ctx {
    moe_cache_session * session;
    std::vector<moe_cache_prefill_pin> pins;
};

// Enqueued on prefill_stream, after every D2D hit-copy that read from a live
// LFRU slot, so it runs (stream-ordered, i.e. only once every copy queued
// ahead of it has actually completed on the GPU) before those slots' reader
// pins are released. Without this, a slot could be evicted - and its bytes
// overwritten by the fill worker - while our async D2D copy from it is still
// in flight, since cudaMemcpyAsync returns as soon as the copy is queued,
// long before session->mu is available to prove it's done. Runs on a CUDA
// driver callback thread, not the calling thread - acquiring a plain mutex
// there is fine, but must stay quick and must never call back into CUDA.
static void moe_cache_prefill_release_readers_cb(void * userdata) {
    auto * ctx = static_cast<moe_cache_prefill_reader_release_ctx *>(userdata);
    {
        std::lock_guard<std::mutex> lock(ctx->session->mu);
        for (auto & pin : ctx->pins) {
            moe_cache_pool * pool = pin.pool;
            const int idx = pin.slot;
            if (idx < 0 || (size_t) idx >= pool->slots.size()) {
                continue;
            }
            moe_cache_slot & slot = pool->slots[idx];
            // Generation-checked: if this slot was reset and reused for a
            // different (tensor, expert) since we pinned it (a forced reset
            // elsewhere that didn't know about this pin - see the struct
            // comment), it is no longer ours to release. Decrementing anyway
            // would steal reader protection from whatever legitimately holds
            // it now.
            if (slot.generation == pin.generation && slot.readers > 0) {
                slot.readers--;
            }
        }
    }
    delete ctx;
}

// See ggml_moe_cache.track_fetch_pace's doc comment in
// ggml-backend-moe-cache.h for why this exists and what replaced it (a
// host-side dispatch-call timer that never saw the real bottleneck, since
// cudaMemcpyAsync does not block the caller). Same stream-callback shape as
// moe_cache_prefill_release_readers_cb above - runs on a CUDA driver
// callback thread once every op queued before it, including the fetch just
// issued, has genuinely finished, not merely been enqueued.
struct moe_cache_fetch_pace_ctx {
    moe_cache_device * device;
    std::chrono::steady_clock::time_point issue_time;
};

// EMA decay follows the standard N-period formula, alpha = 2/(N+1) - not
// borrowed from the atlas UI tracker elsewhere in this file, which is tuned
// for visual smoothness, not statistical validity for a gating decision.
// N_FAST is chosen to smooth ordinary single-fetch scheduling jitter while
// still reacting within a handful of fetches, well under one token's worth
// (a token needs on the order of 10 experts x dozens of offloaded layers of
// fetch attempts). N_BASELINE is defined as a fixed multiple of N_FAST
// rather than an independently guessed number, so the two stay in a
// principled ratio to each other regardless of what N_FAST itself is tuned
// to: the baseline needs to span enough fetches to average out ordinary
// per-token/per-layer variation and reflect genuine steady state, not one
// token's particular mix - BASELINE_MULTIPLE sets how much longer that
// window is, in fetch-count terms, not wall-clock time. Both overridable;
// the multiple, not just the raw N, is what should transfer sensibly
// across models/hardware with very different fetch rates.
static double moe_cache_substitute_pace_alpha_fast() {
    static const double a = [] {
        const char * e = getenv("GGML_CUDA_MOE_CACHE_SUBSTITUTE_PACE_N_FAST");
        const int n = e ? atoi(e) : 8;
        return 2.0 / ((n > 1 ? n : 8) + 1);
    }();
    return a;
}
static double moe_cache_substitute_pace_alpha_baseline() {
    static const double a = [] {
        const char * em = getenv("GGML_CUDA_MOE_CACHE_SUBSTITUTE_PACE_BASELINE_MULTIPLE");
        const double multiple = em ? atof(em) : 32.0;
        const char * en = getenv("GGML_CUDA_MOE_CACHE_SUBSTITUTE_PACE_N_FAST");
        const int n_fast = en ? atoi(en) : 8;
        const double n_baseline = (n_fast > 1 ? n_fast : 8) * (multiple > 1.0 ? multiple : 32.0);
        return 2.0 / (n_baseline + 1.0);
    }();
    return a;
}

static void moe_cache_fetch_pace_cb(void * userdata) {
    auto * ctx = static_cast<moe_cache_fetch_pace_ctx *>(userdata);
    const double delta_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - ctx->issue_time).count();
    {
        std::lock_guard<std::mutex> lock(ctx->device->substitute_pace_mu);
        ctx->device->substitute_pace_ema_ms +=
                moe_cache_substitute_pace_alpha_fast() *
                (delta_ms - ctx->device->substitute_pace_ema_ms);
        ctx->device->substitute_pace_baseline_ms +=
                moe_cache_substitute_pace_alpha_baseline() *
                (delta_ms - ctx->device->substitute_pace_baseline_ms);
        if (ctx->device->substitute_pace_samples < UINT32_MAX) {
            ctx->device->substitute_pace_samples++;
        }
    }
    delete ctx;
}

// Sampled, not measured on every real fetch: this is called from the same
// hot dispatch path copy_experts already runs on, and both the CUDA driver
// call (cudaLaunchHostFunc, ~30-50us of dispatch cost - see the reader-
// release callback's own batching comment above for the same measurement in
// a different context here) and the heap allocation for the context add up
// across the ~10^5 real fetches a single generation can issue. First
// implementation measured every fetch and cost enough of that overhead to
// regress a 1.2-1.7 tok/s baseline down to ~0.7 - the instrumentation meant
// to unlock throughput was paying its own cost on 100% of fetches while the
// gate it feeds only acted on ~12% of them. 1-in-8 mirrors N_FAST's own
// window (moe_cache_substitute_pace_alpha_fast) - the EMA doesn't need
// every sample to stay representative, since it's smoothing over several
// samples internally anyway.
static void moe_cache_track_fetch_pace(void * backend_) {
    static std::atomic<uint64_t> call_count{0};
    if ((call_count.fetch_add(1, std::memory_order_relaxed) & 7) != 0) {
        return; // 1-in-8 sampled
    }
    if (!backend_) {
        return;
    }
    moe_cache_session * session = nullptr;
    moe_cache_device * device_ptr = moe_cache_prefill_first_session_and_device(&session);
    if (!device_ptr || !session) {
        return;
    }
    cudaStream_t stream = ((ggml_backend_cuda_context *) ((ggml_backend *) backend_)->context)->stream();
    auto * ctx = new moe_cache_fetch_pace_ctx{device_ptr, std::chrono::steady_clock::now()};
    if (!moe_cache_cuda_ok(*device_ptr,
            cudaLaunchHostFunc(stream, moe_cache_fetch_pace_cb, ctx),
            "fetch-pace tracking enqueue", false)) {
        delete ctx;
    }
}

// Scheduler-level integration point: ggml_backend_sched_compute_splits'
// existing "copy only the experts that are used" logic (ggml-backend.cpp)
// calls this for each selected expert row before falling back to its own
// H2D copy for it. Finds the same (host_base, expert) key the decode-time
// LFRU pool already uses (moe_cache_key, populated by the CPU-dispatch path
// in ggml-cpu.c) and, if resident, issues a device-to-device copy instead -
// no PCIe cost, no separate cache, upgrading the mechanism that's already
// there rather than running a second one beside it. Does not take
// device.dispatch_mu: this isn't part of the prefill_* pipeline above (no
// prefill_slabs/prefill_successors touched), only session->mu, the same lock
// moe_cache_plan's own decode-time lookup uses to read pool/slot state.
static bool moe_cache_moe_lfru_copy_expert(
        const void * host_base, int32_t expert, size_t expert_size, void * dst_ptr, void * backend_) {
    if (!host_base || !dst_ptr || !backend_ || expert < 0) {
        return false;
    }
    // Same extraction ggml_backend_cuda_set_tensor_async uses, so this copy
    // is ordered on the exact stream the surrounding H2D copies in this
    // split are - required for the later graph_compute_async call to see it
    // as already complete/ordered-before, not a race.
    cudaStream_t stream = ((ggml_backend_cuda_context *) ((ggml_backend *) backend_)->context)->stream();
    moe_cache_session * session = nullptr;
    moe_cache_device * device_ptr = moe_cache_prefill_first_session_and_device(&session);
    if (!device_ptr || !session) {
        return false;
    }
    moe_cache_device & device = *device_ptr;

    const void * src_ptr = nullptr;
    auto release_ctx = std::make_unique<moe_cache_prefill_reader_release_ctx>();
    release_ctx->session = session;
    {
        std::lock_guard<std::mutex> slock(session->mu);
        const moe_cache_key key{host_base, expert};
        for (const auto & pool_ptr : device.pools) {
            if (pool_ptr->expert_size != expert_size) {
                continue;
            }
            auto found = pool_ptr->map.find(key);
            if (found == pool_ptr->map.end()) {
                break; // an expert lives in exactly one pool shape
            }
            moe_cache_slot & pslot = pool_ptr->slots[found->second];
            if (pslot.state == moe_cache_slot_state::valid) {
                src_ptr = pool_ptr->slab + (size_t) found->second * pool_ptr->slot_stride;
                pslot.readers++;
                release_ctx->pins.push_back({pool_ptr.get(), found->second, pslot.generation});
            }
            break;
        }
    }
    if (!src_ptr) {
        return false; // not resident - caller's normal H2D path handles this row
    }

    if (!moe_cache_cuda_ok(device,
            cudaMemcpyAsync(dst_ptr, src_ptr, expert_size, cudaMemcpyDeviceToDevice, stream),
            "lfru expert D2D copy", false)) {
        // the copy never happened - release the pin we just took, no callback needed
        moe_cache_prefill_release_readers_cb(release_ctx.release());
        return false;
    }
    // Ownership passes to the callback, which deletes ctx itself; on a failed
    // enqueue the copy was still issued successfully (queued ahead of the
    // failure), so release the pin immediately rather than leaking it -
    // same tradeoff as moe_cache_prefill_copy_split's own enqueue failure path.
    moe_cache_prefill_reader_release_ctx * raw = release_ctx.release();
    if (!moe_cache_cuda_ok(device,
            cudaLaunchHostFunc(stream, moe_cache_prefill_release_readers_cb, raw),
            "lfru reader-release enqueue", false)) {
        moe_cache_prefill_release_readers_cb(raw);
    }
    if (getenv("MOE_CACHE_DEBUG_GATE")) {
        static std::atomic<long long> hits{0};
        fprintf(stderr, "[lfru-copy-hit] host_base=%p expert=%d cum_hits=%lld\n",
                host_base, expert, (long long) ++hits);
    }
    return true;
}

// Batched form of moe_cache_moe_lfru_copy_expert - see the doc comment on
// ggml_moe_cache.moe_lfru_copy_experts in ggml-backend-moe-cache.h for why
// this exists (one shared reader-release callback for a whole tensor's hits,
// instead of one cudaLaunchHostFunc per expert). Structurally the same
// residency scan and D2D issue as the single-expert version, just over a
// list with one session->mu critical section and one callback for all of it.
static int moe_cache_moe_lfru_copy_experts(
        const void * host_base, size_t expert_size, const int32_t * ids, int n_ids,
        void * dst_base_, uint8_t * out_hit, void * backend_) {
    if (!host_base || !ids || n_ids <= 0 || !dst_base_ || !out_hit || !backend_) {
        return 0;
    }
    char * dst_base = (char *) dst_base_;
    cudaStream_t stream = ((ggml_backend_cuda_context *) ((ggml_backend *) backend_)->context)->stream();
    moe_cache_session * session = nullptr;
    moe_cache_device * device_ptr = moe_cache_prefill_first_session_and_device(&session);
    if (!device_ptr || !session) {
        return 0;
    }
    moe_cache_device & device = *device_ptr;

    auto release_ctx = std::make_unique<moe_cache_prefill_reader_release_ctx>();
    release_ctx->session = session;
    int n_hits = 0;
    {
        std::lock_guard<std::mutex> slock(session->mu);
        for (int i = 0; i < n_ids; i++) {
            const int32_t expert = ids[i];
            if (expert < 0) {
                continue;
            }
            const moe_cache_key key{host_base, expert};
            const void * src_ptr = nullptr;
            for (const auto & pool_ptr : device.pools) {
                if (pool_ptr->expert_size != expert_size) {
                    continue;
                }
                auto found = pool_ptr->map.find(key);
                if (found == pool_ptr->map.end()) {
                    break; // an expert lives in exactly one pool shape
                }
                moe_cache_slot & pslot = pool_ptr->slots[found->second];
                if (pslot.state == moe_cache_slot_state::valid) {
                    src_ptr = pool_ptr->slab + (size_t) found->second * pool_ptr->slot_stride;
                    pslot.readers++;
                    release_ctx->pins.push_back({pool_ptr.get(), found->second, pslot.generation});
                }
                break;
            }
            if (!src_ptr) {
                continue;
            }
            // dst is positioned by expert id, not by this id's position in the
            // ids[] list - the destination tensor is laid out in expert-index
            // order regardless of what order the caller happened to list ids in.
            if (!moe_cache_cuda_ok(device,
                    cudaMemcpyAsync(dst_base + (size_t) expert * expert_size, src_ptr, expert_size,
                            cudaMemcpyDeviceToDevice, stream),
                    "lfru batch expert D2D copy", false)) {
                // this one failed to queue - undo the pin we just took for it,
                // the rest of the batch (already queued) still proceeds
                if (!release_ctx->pins.empty()) {
                    release_ctx->pins.pop_back();
                }
                continue;
            }
            out_hit[i] = 1;
            n_hits++;
        }
    }
    if (release_ctx->pins.empty()) {
        return 0; // nothing pinned, nothing to release
    }
    moe_cache_prefill_reader_release_ctx * raw = release_ctx.release();
    if (!moe_cache_cuda_ok(device,
            cudaLaunchHostFunc(stream, moe_cache_prefill_release_readers_cb, raw),
            "lfru batch reader-release enqueue", false)) {
        moe_cache_prefill_release_readers_cb(raw);
    }
    if (getenv("MOE_CACHE_DEBUG_GATE")) {
        static std::atomic<long long> hits{0};
        hits += n_hits;
        fprintf(stderr, "[lfru-copy-batch-hit] host_base=%p n_ids=%d n_hits=%d cum_hits=%lld\n",
                host_base, n_ids, n_hits, (long long) hits.load());
    }
    return n_hits;
}

// Splits one successor tensor's copy into per-expert-row hit/miss: rows
// already resident in the decode-time LFRU pool are gathered
// device-to-device (no PCIe cost - see moe_cache_prefill_release_readers_cb
// for how residency is protected against a concurrent eviction while that
// copy is in flight); rows that are not resident are H2D'd over PCIe, with
// contiguous runs of missing rows coalesced into one cudaMemcpyAsync each
// rather than one call per row. Falls back to a single whole-tensor H2D copy
// (the pre-hit-D2D behavior) if next_n_expert is 0/unknown or doesn't evenly
// divide next_tensor_bytes - always correct, just never split.
//
// Caller holds device.dispatch_mu. Takes session->mu itself, nested inside
// that - dispatch_mu is always acquired before session->mu in this file
// (see e.g. the teardown path around device.dead.store(true) above), never
// the reverse, so this ordering can't deadlock against it.
// Returns false if ANY copy in the split failed. The caller must then
// refuse to publish the slot: a partially-written prefill buffer still
// holds whatever the previous tenant left behind, and the consumer
// (ggml_cuda_mul_mat_id) reads it as weights - which is silent corruption,
// not a degraded cache. See the incident note in docs/plan.md.
static bool moe_cache_prefill_copy_split(
        moe_cache_device & device, moe_cache_session * session,
        moe_cache_prefill_slab & slab, int slot,
        const void * next_host_base, size_t next_tensor_bytes, int64_t next_n_expert) {
    // See moe_cache_inject_copy_fault - test-only, no effect unless the
    // env var is set. Placed before any copy so both the whole-tensor
    // fast path and the hit/miss split path are covered by one check.
    if (moe_cache_inject_copy_fault()) {
        return false;
    }

    if (!session || next_n_expert <= 0 || next_tensor_bytes % (size_t) next_n_expert != 0) {
        return moe_cache_cuda_ok(device,
                cudaMemcpyAsync(slab.dev[slot], next_host_base, next_tensor_bytes,
                        cudaMemcpyHostToDevice, device.prefill_stream),
                "prefill copy", false);
    }
    const size_t expert_size = next_tensor_bytes / (size_t) next_n_expert;
    bool all_ok = true;
    char * dst_base = (char *) slab.dev[slot];
    const char * src_base = (const char *) next_host_base;

    std::vector<const void *> hit_src((size_t) next_n_expert, nullptr);
    auto release_ctx = std::make_unique<moe_cache_prefill_reader_release_ctx>();
    release_ctx->session = session;
    {
        std::lock_guard<std::mutex> slock(session->mu);
        if (getenv("MOE_CACHE_DEBUG_GATE")) {
            {
                fprintf(stderr, "[prefill-hitd2d-pools] next_host_base=%p n_expert=%lld want_expert_size=%zu n_pools=%zu\n",
                        next_host_base, (long long) next_n_expert, expert_size, device.pools.size());
                for (const auto & pool_ptr : device.pools) {
                    int any_match = 0;
                    for (const auto & kv : pool_ptr->map) {
                        if (kv.first.tensor == next_host_base) {
                            any_match++;
                        }
                    }
                    fprintf(stderr, "  pool expert_size=%zu n_slots=%d map_size=%zu any_match_for_this_tensor=%d\n",
                            pool_ptr->expert_size, pool_ptr->n_slots, pool_ptr->map.size(), any_match);
                }
            }
        }
        for (int64_t e = 0; e < next_n_expert; e++) {
            const moe_cache_key key{next_host_base, (int32_t) e};
            for (const auto & pool_ptr : device.pools) {
                if (pool_ptr->expert_size != expert_size) {
                    continue;
                }
                auto found = pool_ptr->map.find(key);
                if (found == pool_ptr->map.end()) {
                    break; // an expert lives in exactly one pool shape - no point checking others
                }
                moe_cache_slot & pslot = pool_ptr->slots[found->second];
                if (pslot.state == moe_cache_slot_state::valid) {
                    hit_src[(size_t) e] = pool_ptr->slab + (size_t) found->second * pool_ptr->slot_stride;
                    pslot.readers++;
                    release_ctx->pins.push_back({pool_ptr.get(), found->second, pslot.generation});
                }
                break;
            }
        }
    }

    device.prefill_hit_rows += (long long) release_ctx->pins.size();
    device.prefill_total_rows += next_n_expert;
    if (getenv("MOE_CACHE_DEBUG_GATE")) {
        fprintf(stderr, "[prefill-hitd2d-dbg] layer_rows=%lld/%lld cum=%lld/%lld\n",
                (long long) release_ctx->pins.size(), (long long) next_n_expert,
                device.prefill_hit_rows, device.prefill_total_rows);
    }

    for (int64_t e = 0; e < next_n_expert; e++) {
        if (hit_src[(size_t) e]) {
            all_ok = moe_cache_cuda_ok(device,
                    cudaMemcpyAsync(dst_base + (size_t) e * expert_size, hit_src[(size_t) e], expert_size,
                            cudaMemcpyDeviceToDevice, device.prefill_stream),
                    "prefill hit copy", false) && all_ok;
        }
    }
    if (!release_ctx->pins.empty()) {
        // Ownership passes to the callback on success, which deletes ctx
        // itself; on failure (implies the stream/device is already in a bad
        // state - the D2D copies just queued ahead of it are equally
        // suspect) release the pins immediately instead of leaking them
        // permanently unevictable. Trading a already-degraded stream's
        // small residual race for a guaranteed-forever leak is the right
        // call here.
        moe_cache_prefill_reader_release_ctx * raw = release_ctx.release();
        if (!moe_cache_cuda_ok(device,
                cudaLaunchHostFunc(device.prefill_stream, moe_cache_prefill_release_readers_cb, raw),
                "prefill reader-release enqueue", false)) {
            moe_cache_prefill_release_readers_cb(raw);
        }
    }

    int64_t run_start = -1;
    for (int64_t e = 0; e <= next_n_expert; e++) {
        const bool miss = e < next_n_expert && !hit_src[(size_t) e];
        if (miss) {
            if (run_start < 0) {
                run_start = e;
            }
            continue;
        }
        if (run_start < 0) {
            continue;
        }
        const size_t off = (size_t) run_start * expert_size;
        const size_t len = (size_t) (e - run_start) * expert_size;
        all_ok = moe_cache_cuda_ok(device,
                cudaMemcpyAsync(dst_base + off, src_base + off, len,
                        cudaMemcpyHostToDevice, device.prefill_stream),
                "prefill miss copy", false) && all_ok;
        run_start = -1;
    }
    return all_ok;
}

// Called from ggml_cuda_mul_mat_id while it dispatches host_base's own node,
// to kick off the copy for whatever tensor was registered as its successor -
// the "prefetch one layer ahead" pipeline. Never allocates - see
// moe_cache_prefill_register_successor above for why that has to happen
// earlier, at build time, and this function relies on it already having run.
static void moe_cache_prefill_advance(const void * host_base, void * origin_stream_) {
    if (!moe_cache_prefill_buffer_enabled() || !host_base || !origin_stream_) {
        return;
    }
    moe_cache_session * session = nullptr;
    moe_cache_device * device_ptr = moe_cache_prefill_first_session_and_device(&session);
    if (!device_ptr) {
        return;
    }
    moe_cache_device & device = *device_ptr;
    std::lock_guard<std::mutex> lock(device.dispatch_mu);

    if (!device.prefill_stream) {
        return;
    }
    auto succ_it = device.prefill_successors.find(host_base);
    if (succ_it == device.prefill_successors.end()) {
        return;
    }
    const void * next_host_base = succ_it->second.next_host_base;
    const size_t next_tensor_bytes = succ_it->second.next_tensor_bytes;
    const int64_t next_n_expert = succ_it->second.next_n_expert;

    auto slab_it = device.prefill_slabs.find(next_tensor_bytes);
    if (slab_it == device.prefill_slabs.end()) {
        return; // register_successor should have created this - defensive only
    }
    moe_cache_prefill_slab & slab = slab_it->second;
    auto assign_it = slab.assigned_slot.find(next_host_base);
    if (assign_it == slab.assigned_slot.end()) {
        return; // register_successor never assigned this tensor a slot - defensive only
    }
    const int slot = assign_it->second;

    // Already holding a fresh copy of exactly this tensor in this slot (this
    // node visited more than once for the same ubatch, e.g. multiple experts
    // sharing a layer, or a later token replaying a captured region that
    // already has this data resident) - nothing to redo. Weight tensors are
    // immutable for the life of the model, so a completed copy stays valid
    // forever, across any number of future captures/replays, until this slot
    // gets reused for a *different* tensor.
    if (slab.inflight_host[slot] == next_host_base && slab.inflight_bytes[slot] == next_tensor_bytes) {
        return;
    }
    if (!slab.dev[slot] || slab.cap[slot] < next_tensor_bytes || !slab.ready[slot] || !slab.consumed[slot]) {
        return; // not pre-allocated - register_successor was never called for this shape
    }

    // Don't overwrite this slot until whatever previously read it (two
    // advances ago, same slot) has actually finished - see the slab struct
    // comment. A real host block, not a captured wait - see the design note
    // above. Skipped the first time a slot is used.
    if (slab.consumed_valid[slot]) {
        if (!moe_cache_cuda_ok(device, cudaEventSynchronize(slab.consumed[slot]),
                "prefill wait-for-consumed", false)) {
            return;
        }
    }

    const bool copied = moe_cache_prefill_copy_split(
            device, session, slab, slot, next_host_base, next_tensor_bytes, next_n_expert);
    const bool recorded = moe_cache_cuda_ok(device,
            cudaEventRecord(slab.ready[slot], device.prefill_stream),
            "prefill event record", false);

    if (!copied || !recorded) {
        // Do NOT publish this slot. inflight_host is what prefill_wait uses
        // to decide the buffer holds live data for this tensor; setting it
        // after a failed copy is exactly how a stale buffer gets consumed as
        // weights, which surfaces as fluent-looking garbage rather than an
        // error. Leaving it clear makes wait() return NULL and the caller
        // fall back to its normal H2D path - slower, correct.
        slab.inflight_host[slot] = nullptr;
        slab.inflight_bytes[slot] = 0;
        moe_cache_note_copy_failure(device);
        return;
    }

    slab.inflight_host[slot] = next_host_base;
    slab.inflight_bytes[slot] = next_tensor_bytes;
}

// Must be called by the consumer once it has issued (not necessarily
// completed - this just needs to be ordered on the same stream) every read of
// the buffer returned by the matching prefill_wait, so the NEXT prefetch into
// this slot (two calls from now, per the ping-pong period) knows it's safe to
// overwrite. Skipping this call is safe but pessimistic: consumed_valid stays
// false forever for that slot, and moe_cache_prefill_prefetch never gets to
// skip its wait - functionally correct (the memcpy would just queue behind
// whatever's already on prefill_stream) but only if the two streams happen to
// serialize some other way, so callers should always call this.
static void moe_cache_prefill_release(
        const void * host_base, size_t tensor_bytes, int slot, void * consumer_stream_) {
    if (!host_base || tensor_bytes == 0 || slot < 0 || slot > 1 || !consumer_stream_) {
        return;
    }
    cudaStream_t consumer_stream = (cudaStream_t) consumer_stream_;
    moe_cache_device * device_ptr = moe_cache_prefill_first_device();
    if (!device_ptr) {
        return;
    }
    moe_cache_device & device = *device_ptr;
    std::lock_guard<std::mutex> lock(device.dispatch_mu);

    auto it = device.prefill_slabs.find(tensor_bytes);
    if (it == device.prefill_slabs.end()) {
        return;
    }
    moe_cache_prefill_slab & slab = it->second;
    if (slab.inflight_host[slot] != host_base || slab.inflight_bytes[slot] != tensor_bytes ||
        !slab.consumed[slot]) {
        return;
    }
    if (moe_cache_cuda_ok(device, cudaEventRecord(slab.consumed[slot], consumer_stream),
            "prefill release record", false)) {
        slab.consumed_valid[slot] = true;
    }
}

// Blocks the calling (CPU) thread, if needed, until the copy started by the
// matching advance() call has actually finished, then returns the device
// pointer to compute against. A real host sync rather than a captured
// cudaStreamWaitEvent - see the design note above moe_cache_prefill_advance
// for why: this function runs from inside a captured dispatch call as often
// as not, and the capture that recorded slab.ready[slot] is essentially
// never the one being recorded now. Returns nullptr on any mismatch (wrong
// slot, a different tensor landed in that slot since, wrong device) - caller
// falls back to the existing op_offload/host-read path exactly as if
// prefetch had never been called, so a mismatch here is a missed
// optimization, not a correctness problem.
static const void * moe_cache_prefill_wait(
        const void * host_base, size_t tensor_bytes, void * consumer_stream_, int * out_slot) {
    if (out_slot) {
        *out_slot = -1;
    }
    if (!host_base || tensor_bytes == 0 || !consumer_stream_) {
        return nullptr;
    }
    // consumer_stream_ is kept in the signature for API symmetry with
    // prefill_release (whose consumer_stream is genuinely used, to record
    // the matching completion event) even though this function no longer
    // needs a stream itself - see the design note above
    // moe_cache_prefill_advance for why the join here is a host sync, not a
    // captured stream wait.
    moe_cache_device * device_ptr = moe_cache_prefill_first_device();
    if (!device_ptr) {
        return nullptr;
    }
    moe_cache_device & device = *device_ptr;
    std::lock_guard<std::mutex> lock(device.dispatch_mu);

    auto it = device.prefill_slabs.find(tensor_bytes);
    if (it == device.prefill_slabs.end()) {
        return nullptr;
    }
    moe_cache_prefill_slab & slab = it->second;
    // The consumer (a CUDA op dispatch, e.g. ggml_cuda_mul_mat_id) has no
    // channel back to the graph-build-time hook that issued the matching
    // prefetch, so "which slot" can only be recovered by asking the slab -
    // but since assignment is now fixed per tensor (see the slab struct
    // comment), this is a direct lookup rather than a scan of both slots.
    auto assign_it = slab.assigned_slot.find(host_base);
    if (assign_it == slab.assigned_slot.end()) {
        return nullptr;
    }
    const int slot = assign_it->second;
    if (slab.inflight_host[slot] != host_base || slab.inflight_bytes[slot] != tensor_bytes) {
        return nullptr; // registered for this slot, but no live prefetch landed here yet
    }
    if (getenv("MOE_CACHE_DEBUG_GATE")) {
        auto t0 = std::chrono::steady_clock::now();
        cudaError_t err = cudaEventSynchronize(slab.ready[slot]);
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - t0).count();
        fprintf(stderr, "[prefill-sync-dbg] bytes=%zu block_us=%lld\n", tensor_bytes, (long long) us);
        if (!moe_cache_cuda_ok(device, err, "prefill wait", false)) {
            return nullptr;
        }
    } else if (!moe_cache_cuda_ok(device, cudaEventSynchronize(slab.ready[slot]),
            "prefill wait", false)) {
        return nullptr;
    }
    if (out_slot) {
        *out_slot = slot;
    }
    return slab.dev[slot];
}

// Sweep often enough that the classification is still true when the kernel acts
// on it. 30s was chosen when the threshold was wall-clock minutes; with a
// routing-relative threshold the useful signal changes on the order of seconds.
static constexpr std::chrono::seconds MOE_CACHE_COLD_SWEEP_INTERVAL{3};

// How many routing decisions an expert may go unselected before it is declared
// dormant. One tick per MoE node planned, so with ~46 CPU-resident MoE tensors
// this is roughly (value / 46) tokens. Default ~100 tokens' worth: long enough
// that a briefly-unlucky expert is not evicted, short enough that the cold tail
// is identified while it still matters.
static uint64_t moe_cache_cold_after_epochs() {
    static const uint64_t value = [] () -> uint64_t {
        if (const char * env = getenv("GGML_CUDA_MOE_CACHE_COLD_AFTER_EPOCHS")) {
            const long long parsed = atoll(env);
            return parsed <= 0 ? 0 : (uint64_t) parsed;
        }
        // Off by default. The policy itself is sound - it is the same LFRU
        // signal, decay and coldest-pays rule the VRAM tier uses, and that tier
        // holds ~70% hit rate with it. What does not carry over is enforcement:
        // in VRAM we choose the victim slot, whereas here we can only advise,
        // and the kernel's interface is demotion-only (no hint expresses
        // "protect this"; mlock, the only mechanism that does, measured worse by
        // making pages unreclaimable). Measured across several runs, advisory
        // demotion moves cold throughput a few percent against a 21x gap.
        //
        // Kept implemented and one env var away, because the situation it
        // addresses is real - it simply cannot be validated here, where the
        // condition has to be manufactured with a cgroup cap and the models that
        // would genuinely need it are 200GB+.
        return 0;
    }();
    return value;
}
static constexpr std::chrono::seconds MOE_CACHE_HISTORY_SAVE_INTERVAL{60};
static void moe_cache_cold_sweep(moe_cache_device & device, std::chrono::steady_clock::time_point now);
static void moe_cache_host_promote_locked_free(
        moe_cache_device & device, moe_cache_device::cpu_residency & res,
        const void * host_base, int expert, size_t expert_size);
static size_t moe_cache_host_budget_bytes();
static void moe_cache_host_retire(moe_cache_device & device, void * block, size_t bytes);
static void moe_cache_history_save(moe_cache_session & session, const moe_cache_device & device);
static void moe_cache_coact_save(moe_cache_session & session, const moe_cache_device & device);

// Host RAM this process may pin for hot CPU-resident experts.
//
// Why pinning at all: CPU-placed experts are mmap'd, so when RAM cannot hold
// them the kernel reclaims by recency alone. It has no idea that 8 of 128
// experts per layer are selected every token, so a hot expert is exactly as
// likely to be dropped as one never selected since load - and the re-faults
// that follow are small and scattered rather than sequential. Measured on this
// host under an 8 GiB cap: generation fell from 10.80 to 0.51 tok/s, a 21x
// collapse, against an SSD that reads 513 MB/s sequentially. The bandwidth was
// never the problem; the eviction policy was. MADV_COLD (see the sweep below)
// can only lower a page's priority - it cannot hold the hot ones down. mlock
// can, and moe-cache already knows which those are.
//
// Budget: derived live from MemAvailable rather than a constant, for the same
// reason the VRAM budget is - a fixed number is wrong on both a 16 GiB laptop
// and a 512 GiB server. An eighth, clamped hard: pinned pages are by definition
// unreclaimable, so over-pinning under a cgroup cap converts a slow machine
// into an OOM kill. Deliberately far more conservative than the VRAM side.
// GGML_CUDA_MOE_CACHE_PIN_MB overrides (0 disables).
static size_t moe_cache_pin_budget_bytes() {
    static const size_t value = [] () -> size_t {
        if (const char * env = getenv("GGML_CUDA_MOE_CACHE_PIN_MB")) {
            if (strcmp(env, "auto") == 0) {
                // fall through to the derived budget below
            } else {
                const long parsed = strtol(env, nullptr, 10);
                return parsed <= 0 ? 0 : (size_t) parsed << 20;
            }
        } else {
            // Off by default, and this is a measured decision rather than
            // caution. Under a 5 GiB cap with ~12 GiB of CPU-side experts, the
            // derived budget (an eighth of the cap, 640 MiB) covers about 5% of
            // the working set: generation measured 2.03/7.59/10.30/11.40 tok/s
            // with pinning against 2.20/8.73/11.74 without, and reclaim events
            // rose from 69150 to 83058. Pinning a small slice cannot fix a large
            // shortfall, and because pinned pages are unreclaimable it shrinks
            // the pool every other page competes for - so it costs slightly more
            // than it saves. It should help when the shortfall is modest enough
            // that the budget covers a real share of the hot set, which is what
            // GGML_CUDA_MOE_CACHE_PIN_MB=<N>|auto is for.
            return 0;
        }
#if defined(__linux__)
        size_t avail = 0;
        {
            std::ifstream meminfo("/proc/meminfo");
            std::string key, unit;
            size_t value_kib = 0;
            while (meminfo >> key >> value_kib >> unit) {
                if (key == "MemAvailable:") {
                    avail = value_kib << 10;
                    break;
                }
            }
        }
        if (avail == 0) {
            return 0;
        }

        // A cgroup cap is the real ceiling when there is one, and it is not
        // visible in MemAvailable - that reports the host's view, which can be
        // far larger than what this process may actually use. Pinned pages
        // cannot be reclaimed, so budgeting against the host figure inside a
        // small cgroup is precisely how a memory-pressured-but-working server
        // becomes an OOM kill. Take the smaller of the two.
        {
            std::string self_cgroup;
            {
                std::ifstream f("/proc/self/cgroup");
                std::string line;
                while (std::getline(f, line)) {
                    // cgroup v2 has a single "0::<path>" entry
                    const size_t pos = line.rfind("0::");
                    if (pos == 0) {
                        self_cgroup = line.substr(3);
                        break;
                    }
                }
            }
            if (!self_cgroup.empty()) {
                std::ifstream f("/sys/fs/cgroup" + self_cgroup + "/memory.max");
                std::string value;
                if (f >> value && value != "max") {
                    const unsigned long long cap = strtoull(value.c_str(), nullptr, 10);
                    if (cap > 0) {
                        avail = std::min<size_t>(avail, (size_t) cap);
                    }
                }
            }
        }

        const size_t eighth = avail / 8;
        return std::min<size_t>(8ull << 30, std::max<size_t>(256ull << 20, eighth));
#endif
        return 0; // no reliable availability figure - pin nothing rather than guess
    }();
    return value;
}

// Track 1.5: defined further down with the rest of the atlas-warming code,
// declared here because the worker is the thread that runs it.
static bool moe_cache_atlas_warm_service(
        moe_cache_session * session, moe_cache_device * device,
        std::unique_lock<std::mutex> & lock);

// Stage A live-activation predictor: defined further down alongside the
// rest of the predictor code, declared here for the same reason as above.
static void moe_cache_train_service(moe_cache_device & device);

// GGML_CUDA_MOE_CACHE_NO_RT_ALLOC=1: create the fill stream and the pinned
// staging buffer ONCE, here at worker startup, and never allocate again for
// the life of the thread. cudaMallocHost/cudaFreeHost/cudaStreamCreate are
// device-wide synchronizing operations; the default lazy path below runs them
// on this background thread *during decode*, concurrently with the main
// thread's graph capture and dispatch. That is the only channel left that is
// (a) reached by every corrupting arm and (b) short-circuited by FAIL=insert,
// which is the sole clean one. If corruption vanishes under this flag with
// admissions still fully enabled, the runtime allocation is the fault.
// The stage is sized by GGML_CUDA_MOE_CACHE_STAGE_MB (default 64); a job
// larger than it falls through to the unstaged cudaMemcpy path, which is
// correct, just slower - so pinning the size can never break a fill.
// --- weight-immutability guard (GGML_CUDA_MOE_CACHE_WEIGHT_GUARD=1) ---------
// Comparing an expert only against its own previous admission is blind whenever
// experts stay resident once admitted. So record the hash at admission and
// RE-VERIFY EVERY tracked expert on the worker's idle tick, which keeps running
// during and after the degenerate request.
struct moe_weight_rec { const void * src; size_t bytes; uint64_t hash; };
static std::mutex moe_guard_mu;
static std::unordered_map<uint64_t, moe_weight_rec> moe_guard_seen;

static bool moe_cache_weight_guard_on() {
    static const bool on = [] {
        const char * e = getenv("GGML_CUDA_MOE_CACHE_WEIGHT_GUARD");
        return e && atoi(e) != 0;
    }();
    return on;
}

static uint64_t moe_cache_weight_hash(const void * src, size_t bytes) {
    uint64_t h = 1469598103934665603ull;
    const unsigned char * p = (const unsigned char *) src;
    for (size_t i = 0; i < bytes; i++) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

// Re-hash every expert admitted so far. Returns the number that changed.
// GGML_CUDA_MOE_CACHE_MAP_AUDIT=1: pool.map is supposed to be injective - each
// slot index should belong to at most one (tensor, expert) key. If two keys
// ever map to the same slot without the old one being properly evicted
// first, both readers of that slot see whichever content landed last,
// silently. This fits a corruption pattern that is confined to specific
// experts rather than global: with kv_unified=false, a sequence whose
// routing never touches the colliding slot stays clean while one that does
// goes DEGEN, which is exactly what the per-slot probe showed. Walks every
// live session's every pool; heavier than the weight-guard sweep, so kept
// on its own flag and only run from the same idle tick.
// GGML_CUDA_MOE_CACHE_SLOT_NAN_SWEEP=1: full-tensor BYTE comparison of every
// VALID slot's device content against its host source - not a sampled
// 3-window compare (VERIFY_SLOTS, already 0 mismatches, but a corrupted
// region outside its 3x2KB windows would be invisible to it). Byte compare,
// not float reinterpretation: expert weights are block-quantized (Q4_K etc),
// not IEEE floats, so scanning them as float[] for NaN is meaningless -
// caught that mistake before trusting its 100%-bad result, which was purely
// quantized bit patterns coincidentally decoding as non-finite floats, not
// corruption. Heavy - copies every resident slot back to host every idle
// tick - debug-only.
static void moe_cache_slot_nan_sweep() {
    static const bool on = [] {
        const char * e = getenv("GGML_CUDA_MOE_CACHE_SLOT_NAN_SWEEP");
        return e && atoi(e) != 0;
    }();
    if (!on) {
        return;
    }
    static std::atomic<int> reported{0};
    std::lock_guard<std::mutex> registry_lock(g_registry_mu);
    for (moe_cache_session * session : g_sessions) {
        std::lock_guard<std::mutex> lock(session->mu);
        for (const auto & device_ptr : session->devices) {
            ggml_cuda_set_device(device_ptr->physical);
            for (const auto & pool_ptr : device_ptr->pools) {
                moe_cache_pool & pool = *pool_ptr;
                if (!pool.slab || pool.expert_size == 0) {
                    continue;
                }
                std::vector<char> buf(pool.expert_size);
                int scanned = 0, bad_slots = 0;
                for (int i = 0; i < pool.n_slots; i++) {
                    const moe_cache_slot & slot = pool.slots[i];
                    if (slot.state != moe_cache_slot_state::valid || !slot.key.tensor) {
                        continue;
                    }
                    scanned++;
                    if (cudaMemcpy(buf.data(), pool.slab + (size_t) i * pool.slot_stride,
                                   pool.expert_size, cudaMemcpyDeviceToHost) != cudaSuccess) {
                        continue;
                    }
                    const char * host = (const char *) slot.key.tensor +
                            (size_t) slot.key.expert * pool.expert_size;
                    if (memcmp(buf.data(), host, pool.expert_size) != 0) {
                        size_t first = 0;
                        while (first < buf.size() && buf[first] == host[first]) first++;
                        size_t diffs = 0;
                        for (size_t j = 0; j < buf.size(); j++) {
                            if (buf[j] != host[j]) diffs++;
                        }
                        bad_slots++;
                        if (reported.fetch_add(1) < 20) {
                            fprintf(stderr, "[moe-cache] SLOT BYTE MISMATCH slot=%d expert=%d "
                                    "tensor=%p diffs=%zu/%zu first_byte_offset=%zu\n",
                                    i, slot.key.expert, slot.key.tensor,
                                    diffs, pool.expert_size, first);
                        }
                    }
                }
                fprintf(stderr, "[moe-cache] SLOT BYTE SWEEP pool slots=%d scanned=%d bad_slots=%d\n",
                        pool.n_slots, scanned, bad_slots);
            }
        }
    }
    fflush(stderr);
}


static void moe_cache_map_audit() {
    static const bool on = [] {
        const char * e = getenv("GGML_CUDA_MOE_CACHE_MAP_AUDIT");
        return e && atoi(e) != 0;
    }();
    if (!on) {
        return;
    }
    std::lock_guard<std::mutex> registry_lock(g_registry_mu);
    for (moe_cache_session * session : g_sessions) {
        std::lock_guard<std::mutex> lock(session->mu);
        for (const auto & device_ptr : session->devices) {
            for (const auto & pool_ptr : device_ptr->pools) {
                moe_cache_pool & pool = *pool_ptr;
                std::unordered_map<int, std::vector<moe_cache_key>> by_slot;
                for (const auto & kv : pool.map) {
                    by_slot[kv.second].push_back(kv.first);
                }
                long long collisions = 0, state_mismatches = 0, key_mismatches = 0;
                for (const auto & kv : by_slot) {
                    if (kv.second.size() > 1) {
                        collisions++;
                        fprintf(stderr, "[moe-cache] MAP COLLISION slot=%d holds %zu keys:",
                                kv.first, kv.second.size());
                        for (const auto & k : kv.second) {
                            fprintf(stderr, " (tensor=%p expert=%d)", k.tensor, k.expert);
                        }
                        fprintf(stderr, "\n");
                    }
                }
                for (const auto & kv : pool.map) {
                    if (kv.second < 0 || (size_t) kv.second >= pool.slots.size()) {
                        continue;
                    }
                    const moe_cache_slot & slot = pool.slots[kv.second];
                    if (slot.state != moe_cache_slot_state::valid) {
                        state_mismatches++;
                    } else if (!(slot.key == kv.first)) {
                        key_mismatches++;
                    }
                }
                fprintf(stderr, "[moe-cache] MAP AUDIT pool slots=%d map_entries=%zu "
                        "collisions=%lld state_mismatches=%lld key_mismatches=%lld\n",
                        pool.n_slots, pool.map.size(), collisions, state_mismatches, key_mismatches);
            }
        }
    }
    fflush(stderr);
}

static void moe_cache_weight_guard_sweep() {
    moe_cache_map_audit();
    moe_cache_slot_nan_sweep();
    if (!moe_cache_weight_guard_on()) {
        return;
    }
    // A sticky CUDA error explains every observation at once: permanent for the
    // life of the process, cleared only by a restart, weights intact, KV
    // irrelevant, and reached only once an admission has run. The cache swallows
    // statuses in several places ((void)cudaGetLastError()), so a real fault can
    // be discarded here and surface as garbage everywhere else. Peek, do not
    // clear.
    {
        const cudaError_t sticky = cudaPeekAtLastError();
        static cudaError_t reported = cudaSuccess;
        if (sticky != cudaSuccess && sticky != reported) {
            reported = sticky;
            fprintf(stderr, "[moe-cache] STICKY CUDA ERROR %d: %s\n",
                    (int) sticky, cudaGetErrorString(sticky));
            fflush(stderr);
        }
    }
    std::lock_guard<std::mutex> lock(moe_guard_mu);
    if (moe_guard_seen.empty()) {
        return;
    }
    long long changed = 0;
    for (auto & [id, rec] : moe_guard_seen) {
        const uint64_t now = moe_cache_weight_hash(rec.src, rec.bytes);
        if (now != rec.hash) {
            changed++;
            fprintf(stderr, "[moe-cache] WEIGHT MUTATED src=%p bytes=%zu was=%016llx now=%016llx\n",
                    rec.src, rec.bytes,
                    (unsigned long long) rec.hash, (unsigned long long) now);
            rec.hash = now;
        }
    }
    fprintf(stderr, "[moe-cache] weight guard sweep: %zu experts verified, %lld mutated\n",
            moe_guard_seen.size(), changed);
    fflush(stderr);
}

#if defined(__linux__)
// PROTOTYPE, opt-in via GGML_CUDA_MOE_CACHE_EXPLICIT_READ=1: -ncmoe/moe-cache's
// CPU-resident expert weights are mmap'd PROT_READ straight from the GGUF
// (see llama-model-loader.cpp). A cold fill's cudaMemcpyAsync/memcpy touches
// those pages for the first time during the copy itself, which means the
// kernel demand-pages them in one 4KB page at a time with no idea a whole
// multi-hundred-KB expert is about to be read in one shot - no readahead
// matched to the actual access size, no chance to overlap the read with
// anything else. TurboFieldfare (a from-scratch Metal/Swift MoE-on-SSD
// runtime, unrelated project, same underlying problem) measured this exact
// gap on macOS: mmap demand-paging ran at 0.5 tok/s where one explicit read()
// sized to the whole expert hit ~4-6. This tests whether the same gap is
// real here: resolve which file backs a given mmap'd address (via
// /proc/self/maps, since moe-cache only ever sees a raw pointer - the model
// loader doesn't hand it an fd) and pread() the expert's bytes in one call
// into the existing pinned staging buffer, instead of letting the H2D copy's
// own memory access fault pages in one at a time. Falls back to the normal
// path on any resolution or read failure - this cannot change what bytes
// end up in the cache, only how they get read, so it has no correctness
// implications beyond "did the read succeed."
struct moe_cache_file_backing {
    const char * map_start = nullptr;
    const char * map_end = nullptr;
    size_t file_offset_at_map_start = 0;
    int fd = -1;
};

static std::mutex g_file_backing_mu;
// Small (one entry per mmap'd GGUF shard actually touched by moe-cache, i.e.
// single digits to low tens) - linear scan is fine, this only runs once per
// unique mapping, not once per fill.
static std::vector<moe_cache_file_backing> g_file_backings;

static const moe_cache_file_backing * moe_cache_resolve_file_backing(const void * addr) {
    std::lock_guard<std::mutex> lock(g_file_backing_mu);
    for (const auto & b : g_file_backings) {
        if ((const char *) addr >= b.map_start && (const char *) addr < b.map_end) {
            return &b;
        }
    }
    FILE * maps = fopen("/proc/self/maps", "r");
    if (!maps) {
        return nullptr;
    }
    moe_cache_file_backing found;
    bool resolved = false;
    char line[1024];
    while (fgets(line, sizeof(line), maps)) {
        unsigned long long start = 0, end = 0, offset = 0;
        int path_off = 0;
        // "start-end perms offset dev inode  path" - %n captures the byte
        // offset right before the path field so leading pad spaces (the
        // inode column is left-padded) can be skipped without over-trimming
        // a path that starts with a space (theoretical, not a real concern
        // for a GGUF file path, but %n keeps this exact rather than assumed).
        if (sscanf(line, "%llx-%llx %*s %llx %*x:%*x %*u%n", &start, &end, &offset, &path_off) != 3) {
            continue;
        }
        if ((unsigned long long)(uintptr_t) addr < start || (unsigned long long)(uintptr_t) addr >= end) {
            continue;
        }
        const char * p = line + path_off;
        while (*p == ' ') p++;
        const size_t plen = strlen(p);
        if (plen == 0 || p[0] != '/') {
            break; // anonymous/non-file mapping containing addr - can't pread this
        }
        char path[768];
        size_t n = plen < sizeof(path) - 1 ? plen : sizeof(path) - 1;
        memcpy(path, p, n);
        path[n] = '\0';
        if (n > 0 && path[n - 1] == '\n') {
            path[n - 1] = '\0';
        }
        const int fd = open(path, O_RDONLY);
        if (fd < 0) {
            break;
        }
        found.map_start = (const char *) (uintptr_t) start;
        found.map_end   = (const char *) (uintptr_t) end;
        found.file_offset_at_map_start = (size_t) offset;
        found.fd = fd;
        resolved = true;
        break;
    }
    fclose(maps);
    if (!resolved) {
        return nullptr;
    }
    g_file_backings.push_back(found);
    return &g_file_backings.back();
}

// Reads job.bytes starting at job.source into dst via one explicit pread(),
// using the file backing resolved above. Returns false (dst left untouched)
// on any resolution failure, fd issue, or short/partial read - the caller
// falls back to the normal mmap-touching path in that case.
static bool moe_cache_explicit_read_impl(const void * source, size_t bytes, void * dst) {
    static std::atomic<long long> hits{0}, misses{0};
    const bool log = getenv("GGML_CUDA_MOE_CACHE_EXPLICIT_READ_LOG") != nullptr;
    if (!source || !bytes || !dst) {
        return false;
    }
    const moe_cache_file_backing * backing = moe_cache_resolve_file_backing(source);
    if (!backing || backing->fd < 0) {
        const long long m = misses.fetch_add(1) + 1;
        if (log && m <= 20) {
            fprintf(stderr, "[moe-cache] EXPLICIT_READ miss (resolve failed) src=%p bytes=%zu cum_miss=%lld\n",
                    source, bytes, m);
        }
        return false;
    }
    const size_t local_off = (const char *) source - backing->map_start;
    const off_t file_off = (off_t) (backing->file_offset_at_map_start + local_off);
    size_t done = 0;
    // Loop rather than assume one pread() covers the whole request - a short
    // read (signal interruption, or simply the kernel's own discretion) is
    // valid POSIX behaviour, not an error, for a regular file.
    while (done < bytes) {
        const ssize_t got = pread(backing->fd, (char *) dst + done, bytes - done, file_off + (off_t) done);
        if (got < 0) {
            if (errno == EINTR) {
                continue;
            }
            const long long m = misses.fetch_add(1) + 1;
            if (log && m <= 20) {
                fprintf(stderr, "[moe-cache] EXPLICIT_READ miss (pread errno=%d) src=%p bytes=%zu cum_miss=%lld\n",
                        errno, source, bytes, m);
            }
            return false;
        }
        if (got == 0) {
            const long long m = misses.fetch_add(1) + 1;
            if (log && m <= 20) {
                fprintf(stderr, "[moe-cache] EXPLICIT_READ miss (short read) src=%p bytes=%zu cum_miss=%lld\n",
                        source, bytes, m);
            }
            return false; // unexpected EOF - shorter file than the mapping implied
        }
        done += (size_t) got;
    }
    const long long h = hits.fetch_add(1) + 1;
    if (log && (h <= 20 || h % 200 == 0)) {
        fprintf(stderr, "[moe-cache] EXPLICIT_READ hit src=%p bytes=%zu file_off=%lld cum_hit=%lld cum_miss=%lld\n",
                source, bytes, (long long) file_off, h, misses.load());
    }
    return true;
}
#endif // defined(__linux__)

// Unconditional wrapper so the fill worker's copy-path branch doesn't need
// its own #ifdef - a no-op (always "not available, fall back") off Linux.
static bool moe_cache_explicit_read(const void * source, size_t bytes, void * dst) {
#if defined(__linux__)
    return moe_cache_explicit_read_impl(source, bytes, dst);
#else
    (void) source; (void) bytes; (void) dst;
    return false;
#endif
}

// Defined further down (with the rest of the host-page-release code),
// declared here because the fill worker is the thread that runs it for the
// main VRAM admission path - see the call site below for why.
static void moe_cache_release_source_pages(const void * base, size_t offset, size_t bytes);

// On by default: releasing an expert's mmap'd host pages right after VRAM
// admission is provably safe (the copy that read them has already
// completed and synced by the point this runs - see the call site) and
// closes a real gap where the same bytes could sit resident in both the OS
// page cache and VRAM until the kernel got around to reclaiming them under
// pressure. Opt-out knob kept anyway, matching this file's convention of
// making every non-trivial behavior toggleable even once it is trusted.
static bool moe_cache_release_fill_source_enabled() {
    static const bool enabled = [] {
        const char * env = getenv("GGML_CUDA_MOE_CACHE_RELEASE_FILL_SOURCE");
        return !env || atoi(env) != 0;
    }();
    return enabled;
}

static void moe_cache_worker(moe_cache_session * session, moe_cache_device * device) {
    char * stage = nullptr;
    size_t stage_capacity = 0;
    cudaStream_t stream = nullptr;

    static const bool no_rt_alloc = [] {
        const char * e = getenv("GGML_CUDA_MOE_CACHE_NO_RT_ALLOC");
        return e && atoi(e) != 0;
    }();

    if (no_rt_alloc) {
        static const size_t stage_bytes = [] {
            const char * e = getenv("GGML_CUDA_MOE_CACHE_STAGE_MB");
            const long long mb = e ? atoll(e) : 64;
            return (size_t)(mb > 0 ? mb : 64) << 20;
        }();

        ggml_cuda_set_device(device->physical);

        int least_priority = 0;
        int greatest_priority = 0;
        if (cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority) == cudaSuccess) {
            if (cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, least_priority) != cudaSuccess) {
                (void)cudaGetLastError();
                stream = nullptr;
            }
        } else {
            (void)cudaGetLastError();
        }
        if (!stream && cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) {
            (void)cudaGetLastError();
            stream = nullptr;
        }

        char * fresh = nullptr;
        if (cudaMallocHost((void **)&fresh, stage_bytes) == cudaSuccess) {
            stage = fresh;
            stage_capacity = stage_bytes;
        } else {
            (void)cudaGetLastError();
        }

        fprintf(stderr, "[moe-cache] NO_RT_ALLOC stream=%s stage=%zu MiB\n",
                stream ? "preallocated" : "FAILED",
                stage_capacity >> 20);
    }

    for (;;) {
        moe_cache_job job;
        {
            std::unique_lock<std::mutex> lock(session->mu);
            // Bounded wait rather than an indefinite one, so this thread doubles
            // as the cache's periodic maintenance tick. The dormant-page sweep
            // below has to run when nothing is being decoded - that is precisely
            // when experts have gone cold and when reclaiming their pages costs
            // nothing - so hanging it off decode traffic (where it started out)
            // meant it could never fire in the one state it exists for.
            const bool have_work = session->cv.wait_for(lock, MOE_CACHE_COLD_SWEEP_INTERVAL, [&] {
                return session->stopping || device->dead.load() ||
                    !device->queue.empty() || !device->warm_queue.empty();
            });
            if (!have_work) {
                // Idle: nothing is decoding, so a full re-verify costs nothing a
                // token would otherwise have.
                lock.unlock();
                moe_cache_weight_guard_sweep();
                lock.lock();
            }
            // Promotions first, and the expensive part outside the lock. Decode
            // only recorded intent; the malloc, the copy and the madvise happen
            // here, where they cost the fill worker's time rather than a token's.
            if (!device->dead.load() && !device->host_promote_queue.empty()) {
                auto req = device->host_promote_queue.front();
                device->host_promote_queue.pop_front();
                auto it = device->residency.find(req.first);
                if (it != device->residency.end() &&
                    (size_t) req.second < it->second.host_slot.size() &&
                    it->second.host_slot[req.second].load(std::memory_order_relaxed) == nullptr) {
                    const size_t esz = it->second.expert_size;
                    moe_cache_device::cpu_residency & res = it->second;
                    const void * base = req.first;
                    const int    exp  = req.second;
                    // Held under session->mu for its whole body, including the
                    // victim search that walks device.residency: that map is
                    // mutated (via operator[]) by plan() on the decode thread
                    // under the same lock, and an unlocked full-container
                    // iteration racing an insert is undefined behavior, not
                    // just a stale-read risk - it can corrupt the map's own
                    // bucket structure. This path is dormant unless
                    // GGML_CUDA_MOE_CACHE_HOST_MB is set, so holding the lock
                    // for the malloc/memcpy/madvise too costs nothing today;
                    // correctness matters more than an optimization for code
                    // that never runs by default.
                    moe_cache_host_promote_locked_free(*device, res, base, exp, esz);
                }
                continue;
            }

            // Track 1.5: the deferred atlas ranking pass. Placed after
            // promotions and before the fill dequeue for the same reason
            // promotions are placed where they are - this is CPU work the
            // decode path handed over precisely so it would happen here, on
            // the worker's time, while the GPU is busy with the previous
            // layer. It releases the lock for the scan itself (see
            // moe_cache_atlas_warm_service) and may queue fills of its own,
            // so loop round rather than falling through with a stale view of
            // the queue.
            if (!device->dead.load() && !device->warm_queue.empty()) {
                if (moe_cache_atlas_warm_service(session, device, lock)) {
                    session->cv.notify_all();
                }
                continue;
            }

            // Stage A live-activation predictor: cheap (a handful of dot
            // products), so unlike the atlas scan above this does not need
            // the lock released - see moe_cache_train_service.
            if (!device->dead.load() && !device->train_queue.empty()) {
                moe_cache_train_service(*device);
                continue;
            }

            if (!have_work) {
                // Timed out with an empty queue: idle. Safe to sweep here - the
                // session lock is held, and no fill is in flight to contend with.
                if (!device->dead.load()) {
                    const auto now_idle = std::chrono::steady_clock::now();
                    moe_cache_cold_sweep(*device, now_idle);
                    // Idle is also the right moment to write out usage history:
                    // nothing is decoding, and the file is small.
                    if (now_idle - session->last_history_save >= MOE_CACHE_HISTORY_SAVE_INTERVAL) {
                        session->last_history_save = now_idle;
                        moe_cache_history_save(*session, *device);
                        // Same idle moment, same rationale as history: the
                        // graph is only written when nothing is decoding.
                        moe_cache_coact_save(*session, *device);
                    }
                }
                continue;
            }
            // Not idle - but the sweep still has to run. Under real memory
            // pressure this thread never goes idle (every miss queues a fill),
            // which is exactly when residency decisions matter most; gating them
            // on an idle timeout meant they never ran in the one situation they
            // exist for. Measured: a 5 GiB-capped run with 69150 reclaim events
            // completed without a single sweep. So drive it on elapsed time
            // instead, independent of queue state.
            if (!device->dead.load()) {
                const auto now_tick = std::chrono::steady_clock::now();
                if (now_tick - device->last_cold_sweep >= MOE_CACHE_COLD_SWEEP_INTERVAL) {
                    device->last_cold_sweep = now_tick;
                    moe_cache_cold_sweep(*device, now_tick);
                }
            }
            if ((session->stopping || device->dead.load()) &&
                device->queue.empty()) {
                break;
            }

            job = device->queue.front();
            device->queue.pop_front();
            device->queued_bytes = job.bytes <= device->queued_bytes
                ? device->queued_bytes - job.bytes : 0;
            device->inflight = true;
            device->inflight_source = job.source;
            device->inflight_bytes = job.bytes;
        }

        cudaError_t error = cudaSuccess;
        ggml_cuda_set_device(device->physical);

        // DIAGNOSTIC, opt-in via GGML_CUDA_MOE_CACHE_FILL_TIMING_LOG=1: breaks
        // down where a single fill's wall time actually goes (copy+sync vs.
        // canary readback vs. everything else) - added to get real evidence
        // before considering batching multiple fills' copies onto one shared
        // sync (a real correctness-risk change: device->inflight_source/bytes
        // is a single-range tracker moe_cache_invalidate_session blocks on to
        // avoid racing a fill against memory being freed - batching would
        // need it to track a range per in-flight item, not one).
        static const bool fill_timing_log = [] {
            const char * e = getenv("GGML_CUDA_MOE_CACHE_FILL_TIMING_LOG");
            return e && atoi(e) != 0;
        }();
        const double t_job_start = fill_timing_log ? moe_cache_wall_clock() : 0.0;
        double t_copy_done = 0.0, t_canary_done = 0.0;

        if (device->dead.load() || moe_cache_fail(*session, "insert")) {
            error = cudaErrorUnknown;
            // TEMPORARY diagnostic (moe-cache corruption investigation):
            // GGML_CUDA_MOE_CACHE_INSERT_FAIL_DELAY_US=N burns N microseconds
            // right here, on the fill worker thread, doing NOTHING moe-cache-
            // related (no state touched, admission still fully blocked same
            // as plain FAIL=insert, which is the one clean configuration
            // found this session) - the direct test of whether merely the
            // WALL-CLOCK TIMING/scheduling a real fill would introduce, not
            // any of its specific logic (every specific piece of which has
            // been individually ruled out), is what exposes the corruption.
            static const long delay_us = [] {
                const char * e = getenv("GGML_CUDA_MOE_CACHE_INSERT_FAIL_DELAY_US");
                return e ? atol(e) : 0;
            }();
            if (delay_us > 0) {
                std::this_thread::sleep_for(std::chrono::microseconds(delay_us));
            }
        }
        if (error == cudaSuccess && !stream && !no_rt_alloc) {
            int least_priority = 0;
            int greatest_priority = 0;
            error = cudaDeviceGetStreamPriorityRange(
                    &least_priority, &greatest_priority);
            if (error == cudaSuccess) {
                error = cudaStreamCreateWithPriority(
                        &stream, cudaStreamNonBlocking, least_priority);
            } else {
                (void)cudaGetLastError();
                error = cudaStreamCreateWithFlags(
                        &stream, cudaStreamNonBlocking);
            }
        }
        if (error == cudaSuccess && stage_capacity < job.bytes && !no_rt_alloc) {
            char * fresh = nullptr;
            cudaError_t alloc_error = cudaMallocHost((void **)&fresh, job.bytes);
            if (alloc_error == cudaSuccess) {
                if (stage) {
                    cudaFreeHost(stage);
                }
                stage = fresh;
                stage_capacity = job.bytes;
            } else {
                (void)cudaGetLastError();
            }
        }

        moe_cache_pool * pool = nullptr;
        char * destination = nullptr;
        {
            std::lock_guard<std::mutex> lock(session->mu);
            if (job.pool >= 0 && job.pool < (int)device->pools.size()) {
                pool = device->pools[job.pool].get();
                if (pool->slab && job.slot >= 0 && job.slot < pool->n_slots) {
                    destination = pool->slab + (size_t)job.slot * pool->slot_stride;
                }
            }
        }

        if (error == cudaSuccess && !destination) {
            error = cudaErrorInvalidValue;
        }

        // The destination stride is pool->expert_size, but the copy length is
        // job.bytes, carried from the NODE. Nothing validated that the two
        // agree. If job.bytes exceeds the slot stride the fill runs past its
        // slot - into the next slot, or off the end of the slab entirely -
        // which corrupts whatever VRAM follows and is invisible to every
        // read-side check, because the slot it was asked about still reads
        // back correctly. Refuse the fill instead, loudly.
        static const bool log_fills = [] {
            const char * e = getenv("GGML_CUDA_MOE_CACHE_LOG_FILLS");
            return e && atoi(e) != 0;
        }();
        if (pool && log_fills) {
            static std::atomic<int> nlog{0};
            const int k = nlog.fetch_add(1, std::memory_order_relaxed);
            if (k < 40) {
                const char * slab_end = pool->slab + (size_t) pool->n_slots * pool->slot_stride;
                const char * dst_end  = destination ? destination + job.bytes : nullptr;
                fprintf(stderr, "[moe-cache] FILL #%d pool=%d slot=%d/%d slab=%p end=%p "
                        "dst=%p dst_end=%p bytes=%zu stride=%zu %s\n",
                        k, job.pool, job.slot, pool->n_slots,
                        (void*)pool->slab, (void*)slab_end, (void*)destination, (void*)dst_end,
                        job.bytes, pool->expert_size,
                        (destination && dst_end && destination >= pool->slab && dst_end <= slab_end)
                            ? "in-bounds" : "*** OUT OF BOUNDS ***");
            }
        }

        if (error == cudaSuccess && pool && job.bytes > pool->expert_size) {
            static std::atomic<int> reported{0};
            if (reported.fetch_add(1, std::memory_order_relaxed) < 20) {
                fprintf(stderr, "[moe-cache] FILL OVERRUN REFUSED slot=%d job.bytes=%zu "
                        "pool.expert_size=%zu pool=%d n_slots=%d (would overrun by %zu bytes)\n",
                        job.slot, job.bytes, pool->expert_size, job.pool, pool->n_slots,
                        job.bytes - pool->expert_size);
            }
            error = cudaErrorInvalidValue;
        }
        {
            std::unique_lock<std::mutex> fill_lock(
                    session->fill_mu, std::defer_lock);
            if (session->config.serial_fill) {
                fill_lock.lock();
            }
            // Direct DMA out of the model mapping when the region is page-locked.
            // The staging path costs an extra RAM->RAM memcpy (203 us on a 7.43 MB
            // expert) purely so the following copy can start from pinned memory -
            // which cancels most of what pinning is worth. Registering the tensor
            // once removes the copy: 506 us -> 303 us per expert.
            // Record this expert's source hash the first time we admit it; the
            // idle-tick sweep re-verifies all of them.
            // Skipped for reduce_convert: this hash-based weight-mutation
            // guard is keyed by (job.source, job.bytes) meaning "the
            // CONTIGUOUS byte range starting at job.source" - true for an
            // ordinary fill (job.source already points at this expert's own
            // bytes) but not for a reduce_convert job, whose job.source is
            // the whole tensor's base and whose real data is a scattered,
            // non-contiguous gather from it (see the gather branch below).
            // The original full-width fill for this same expert already
            // registered the correct guard entry for its actual bytes;nothing
            // is lost by not registering a second, meaningless one here.
            if (job.kind == moe_cache_job_kind::fill &&
                moe_cache_weight_guard_on() && job.source && job.bytes) {
                const uint64_t id = ((uint64_t) (uintptr_t) job.source) ^ (job.bytes * 2654435761ull);
                std::lock_guard<std::mutex> glock(moe_guard_mu);
                if (moe_guard_seen.find(id) == moe_guard_seen.end()) {
                    moe_guard_seen.emplace(id, moe_weight_rec{
                            job.source, job.bytes, moe_cache_weight_hash(job.source, job.bytes)});
                }
            }
            bool direct = false;
            {
                static std::atomic<int> once{0};
                if (once.fetch_add(1) == 0) {
                    fprintf(stderr, "[moe-cache] first fill: hostreg_mb=%zu region_base=%p region_bytes=%zu\n",
                            session->config.hostreg_mb, job.region_base, job.region_bytes); fflush(stderr);
                }
            }
            if (error == cudaSuccess && session->config.hostreg_mb > 0 &&
                job.region_base && job.region_bytes > 0) {
                static const uintptr_t page = (uintptr_t) sysconf(_SC_PAGESIZE);
                std::lock_guard<std::mutex> reg_lock(session->hostreg_mu);
                auto it_reg = session->hostreg.find(job.region_base);
                if (it_reg == session->hostreg.end()) {
                    // Register the page-aligned *interior*: rounding outward would
                    // overlap the neighbouring tensor's pages and CUDA refuses a
                    // range that is already partly registered. Losing at most one
                    // page at each end costs nothing - those experts simply keep
                    // using the staging path via the range check below.
                    const uintptr_t raw   = (uintptr_t) job.region_base;
                    const uintptr_t begin = (raw + page - 1) & ~(page - 1);
                    const uintptr_t end   = (raw + job.region_bytes) & ~(page - 1);
                    if (end > begin &&
                        session->hostreg_bytes + (end - begin) <= session->config.hostreg_mb << 20) {
                        // The model is mmap'd PROT_READ, and cudaHostRegisterDefault
                        // demands writable pages - it fails "invalid argument" on a
                        // read-only mapping no matter how well aligned. ReadOnly is
                        // the flag for exactly this: device reads only, which is all
                        // an expert weight fill ever does.
                        cudaError_t reg = cudaHostRegister(
                                (void *) begin, (size_t) (end - begin), cudaHostRegisterReadOnly);
                        if (reg != cudaSuccess) {
                            (void) cudaGetLastError();
                            reg = cudaHostRegister(
                                    (void *) begin, (size_t) (end - begin), cudaHostRegisterDefault);
                        }
                        if (reg == cudaSuccess) {
                            session->hostreg.emplace(job.region_base, std::make_pair(begin, end));
                            session->hostreg_bytes += (size_t) (end - begin);
                            it_reg = session->hostreg.find(job.region_base);
                            fprintf(stderr, "[moe-cache] page-locked %zu MiB expert region for direct DMA (%zu MiB total)\n",
                                    (size_t) (end - begin) >> 20, session->hostreg_bytes >> 20); fflush(stderr);
                        } else {
                            fprintf(stderr, "[moe-cache] cudaHostRegister of %zu MiB failed (%s) - staging path\n",
                                    (size_t) (end - begin) >> 20, cudaGetErrorString(reg)); fflush(stderr);
                        // Not fatal - an unregistered region simply keeps using
                        // the staging path. Clear the sticky error either way.
                            (void) cudaGetLastError();
                        }
                    }
                }
                if (it_reg != session->hostreg.end()) {
                    const uintptr_t src = (uintptr_t) job.source;
                    direct = src >= it_reg->second.first &&
                             src + job.bytes <= it_reg->second.second;
                }
            }
            // TESTED AND REVERTED (docs/plan.md): a full cudaDeviceSynchronize()
            // wrapped around this fill's H2D copy (draining the device before AND
            // after) did NOT fix the single-CLI-process 2-prompt reproducer -
            // ruling out "races the fill's H2D copy against other concurrent GPU
            // stream activity" as the mechanism, at least in this form. Left as a
            // negative result rather than kept as a no-op latency cost.
            // GGML_CUDA_MOE_CACHE_SKIP_COPY=1: keep every bit of bookkeeping
            // (slot allocation, state machine, map/bitmask, pins) but never
            // touch device memory, on WHICHEVER of the three copy paths below
            // would otherwise run - previously only guarded the "direct"
            // (hostreg) branch, which cudaHostRegister fails to enter by
            // default (collides with -ncmoe's own host-buffer registration,
            // "already mapped"), so the flag silently had no effect on a
            // default run. Combined with FAIL=dispatch the cached bytes are
            // never read either, so if corruption SURVIVES this the fault is
            // in the bookkeeping; if it VANISHES the fault is one of these
            // copy operations.
            static const bool skip_copy = [] {
                const char * e = getenv("GGML_CUDA_MOE_CACHE_SKIP_COPY");
                return e && atoi(e) != 0;
            }();
            // PROTOTYPE, off by default: GGML_CUDA_MOE_CACHE_EXPLICIT_READ=1 -
            // see moe_cache_explicit_read above for the full rationale. Placed
            // after `direct`, never before it: a hostreg'd region is already
            // guaranteed resident (cudaHostRegister forces and pins every
            // covered page at registration time), so there is no mmap-fault
            // cost left for an explicit read to save there. moe_cache_explicit_read
            // itself returns false on any resolution/read failure, which
            // falls straight through to the ordinary staged-memcpy branch
            // below with nothing extra to unwind.
            static const bool explicit_read_enabled = [] {
                const char * e = getenv("GGML_CUDA_MOE_CACHE_EXPLICIT_READ");
                return e && atoi(e) != 0;
            }();
            if (job.kind == moe_cache_job_kind::reduce_convert) {
                // Heat-aware neuron subsetting: gather this expert's chosen
                // K rows (device.reduced_indices[job.key]) from the SOURCE
                // tensor (job.source == the tensor's own host base - the
                // gather needs the per-expert AND per-row offset, which a
                // single contiguous job.source/job.bytes pair cannot carry)
                // into the pinned `stage` buffer, then one contiguous H2D
                // copy of the gathered result - same staging discipline as
                // an ordinary fill, just with a scattered source instead of
                // a contiguous one. Looked up here rather than carried on
                // the job itself to avoid growing moe_cache_job for a path
                // every OTHER job on the queue never touches.
                std::vector<int32_t> idx;
                size_t row_bytes = 0, src_expert_size = 0;
                {
                    std::lock_guard<std::mutex> lock(session->mu);
                    const auto it = device->reduced_indices.find(job.key);
                    if (it != device->reduced_indices.end()) {
                        idx = it->second.idx;
                        row_bytes = it->second.row_bytes;
                        src_expert_size = it->second.src_expert_size;
                    }
                }
                if (error == cudaSuccess && skip_copy) {
                    // bookkeeping only, same contract as an ordinary fill's
                    // SKIP_COPY branch - fall through with error==cudaSuccess
                } else if (idx.empty() || row_bytes == 0 || src_expert_size == 0 ||
                           idx.size() * row_bytes != job.bytes ||
                           !job.source || job.key.expert < 0) {
                    error = cudaErrorInvalidValue;
                } else if (stage && stage_capacity >= job.bytes) {
                    const char * expert_base = (const char *) job.source +
                        (size_t) job.key.expert * src_expert_size;
                    for (size_t i = 0; i < idx.size(); i++) {
                        memcpy(stage + i * row_bytes,
                               expert_base + (size_t) idx[i] * row_bytes, row_bytes);
                    }
                    error = cudaMemcpyAsync(
                            destination, stage, job.bytes, cudaMemcpyHostToDevice, stream);
                    if (error == cudaSuccess) {
                        error = cudaStreamSynchronize(stream);
                    }
                } else {
                    error = cudaErrorMemoryAllocation;
                }
            } else if (error == cudaSuccess && skip_copy) {
                // bookkeeping only - fall through with error==cudaSuccess
            } else if (error == cudaSuccess && direct) {
                error = cudaMemcpyAsync(
                        destination, job.source, job.bytes, cudaMemcpyHostToDevice, stream);
                if (error == cudaSuccess) {
                    error = cudaStreamSynchronize(stream);
                }
            } else if (error == cudaSuccess && explicit_read_enabled && stage && stage_capacity >= job.bytes &&
                       moe_cache_explicit_read(job.source, job.bytes, stage)) {
                error = cudaMemcpyAsync(
                        destination, stage, job.bytes, cudaMemcpyHostToDevice, stream);
                if (error == cudaSuccess) {
                    error = cudaStreamSynchronize(stream);
                }
            } else if (error == cudaSuccess && stage && stage_capacity >= job.bytes) {
                memcpy(stage, job.source, job.bytes);
                error = cudaMemcpyAsync(
                        destination, stage, job.bytes, cudaMemcpyHostToDevice, stream);
                if (error == cudaSuccess) {
                    error = cudaStreamSynchronize(stream);
                }
            } else if (error == cudaSuccess) {
                error = cudaMemcpy(
                        destination, job.source, job.bytes, cudaMemcpyHostToDevice);
            }
        }
        if (fill_timing_log) {
            t_copy_done = moe_cache_wall_clock();
        }

        {
            std::lock_guard<std::mutex> lock(session->mu);
            device->inflight = false;
            device->inflight_source = nullptr;
            device->inflight_bytes = 0;

            if (pool && job.slot >= 0 && job.slot < pool->n_slots) {
                moe_cache_slot & slot = pool->slots[job.slot];
                if (slot.state == moe_cache_slot_state::copying &&
                    slot.generation == job.generation && slot.key == job.key) {
                    if (error == cudaSuccess) {
                        slot.state = moe_cache_slot_state::valid;
                        // Now that VRAM holds its own copy, the mmap'd host
                        // pages behind it are redundant - the exact reasoning
                        // moe_cache_release_source_pages already applies for
                        // the separate host-hot-expert-buffer promotion path.
                        // This is the main fill path (both reactive-miss and
                        // atlas-warm admissions funnel through here via
                        // device.queue), which never had the equivalent call:
                        // without it, an expert's bytes can sit resident in
                        // both the OS page cache AND VRAM simultaneously until
                        // the kernel gets around to reclaiming them under
                        // pressure (or the periodic cold-sweep catches it once
                        // dormant - see moe_cache_cold_sweep, which explicitly
                        // excludes VRAM-resident experts from its own pass,
                        // since ITS job is prioritising among non-resident
                        // ones, not cleaning up after residency). Safe to call
                        // unconditionally here: the copy that just read these
                        // bytes has already completed and synced (we are past
                        // every copy branch above, all of which end in either
                        // cudaStreamSynchronize or a synchronous cudaMemcpy),
                        // and MADV_DONTNEED on a clean file-backed mapping
                        // only drops cached pages - the virtual mapping stays
                        // valid and re-faults from disk if ever needed again,
                        // never corrupts or invalidates anything.
                        // Not applicable to reduce_convert: job.source there
                        // is the whole tensor's base, and job.bytes is the
                        // GATHERED size, not a real, madvise-able byte range
                        // starting at job.source (see the gather branch
                        // above) - releasing [source, source+bytes) would
                        // target the wrong pages (typically expert 0's), a
                        // wasted call rather than anything unsafe, but
                        // pointless either way since the full-width fill for
                        // this same expert already released its own, correct
                        // range when IT completed.
                        if (job.kind == moe_cache_job_kind::fill &&
                            job.source && job.bytes && moe_cache_release_fill_source_enabled()) {
                            moe_cache_release_source_pages(job.source, 0, job.bytes);
                        }
                        if (pool && pool->canary) {
                            static std::atomic<int> creport{0};
                            std::vector<char> back(pool->canary_bytes);
                            if (cudaMemcpy(back.data(), pool->canary, pool->canary_bytes,
                                           cudaMemcpyDeviceToHost) == cudaSuccess) {
                                size_t bad = 0, first = pool->canary_bytes;
                                for (size_t i = 0; i < back.size(); i++) {
                                    if (back[i] != (char)0xA5) { if (!bad) first = i; bad++; }
                                }
                                if (bad && creport.fetch_add(1, std::memory_order_relaxed) < 10) {
                                    fprintf(stderr, "[moe-cache] *** CANARY CORRUPTED *** after fill slot=%d: "
                                            "%zu/%zu bytes changed, first at +%zu\n",
                                            job.slot, bad, pool->canary_bytes, first);
                                }
                            }
                        }
                        if (fill_timing_log) {
                            t_canary_done = moe_cache_wall_clock();
                        }
                        // Fresh admission - starts in probation regardless
                        // of how many pre-admission misses it took to get
                        // here (that's a different signal from "reused
                        // while resident", which is what earns protected_
                        // status - see moe_cache_segment_push_back).
                        // TEMPORARY diagnostic (moe-cache corruption
                        // investigation): GGML_CUDA_MOE_CACHE_NO_SEGMENT_PUSH=1
                        // skips ONLY this linked-list insertion, isolating it
                        // from state=valid/demand_count/fills - the one
                        // difference identified so far between the FAIL=insert
                        // (clean) and FAIL=dispatch+SKIP_COPY (corrupts) paths.
                        static const bool no_segment_push = [] {
                            const char * e = getenv("GGML_CUDA_MOE_CACHE_NO_SEGMENT_PUSH");
                            return e && atoi(e) != 0;
                        }();
                        if (!no_segment_push) {
                            moe_cache_segment_push_back(*pool, job.slot, moe_cache_segment::probation);
                        }
                        device->demand_count.erase(job.key);
                        device->fills++;
                        if (++pool->fills_since_decay >= MOE_CACHE_HEAT_DECAY_EVERY) {
                            if (getenv("GGML_CUDA_MOE_CACHE_DECAY_LOG")) {
                                fprintf(stderr, "[t=%.3f] [moe-cache] DECAY SWEEP total_fills=%lld\n",
                                        moe_cache_wall_clock(), (long long) device->fills);
                                fflush(stderr);
                            }
                            moe_cache_pool_decay(*pool);
                            pool->fills_since_decay = 0;
                        }
                    } else {
                        moe_cache_slot_reset(*pool, job.slot, true);
                        device->fill_failures++;
                    }
                }
            }
            session->idle_cv.notify_all();
        }

        if (fill_timing_log) {
            const double t_finalize_done = moe_cache_wall_clock();
            static std::atomic<int> logged{0};
            const int n = logged.fetch_add(1);
            if (n < 40 || n % 500 == 0) {
                fprintf(stderr, "[moe-cache] FILL_TIMING bytes=%zu copy_ms=%.3f canary_ms=%.3f "
                        "finalize_ms=%.3f total_ms=%.3f\n",
                        job.bytes,
                        (t_copy_done - t_job_start) * 1000.0,
                        t_canary_done > 0.0 ? (t_canary_done - t_copy_done) * 1000.0 : 0.0,
                        (t_finalize_done - (t_canary_done > 0.0 ? t_canary_done : t_copy_done)) * 1000.0,
                        (t_finalize_done - t_job_start) * 1000.0);
            }
        }

        if (error != cudaSuccess) {
            moe_cache_cuda_ok(*device, error, "expert fill", true);
            moe_cache_trim_session(*session, device->physical);
        }
    }

    if (stream) {
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
    }
    if (stage) {
        cudaFreeHost(stage);
    }
}

static bool moe_cache_start_worker(
        moe_cache_session & session, moe_cache_device & device) {
    if (device.worker_started) {
        return true;
    }
    try {
        device.worker = std::thread(moe_cache_worker, &session, &device);
        device.worker_started = true;
        return true;
    } catch (...) {
        device.dead.store(true);
        MOE_CACHE_LOG("[moe-cache] CUDA%d failed to start fill worker\n", device.physical);
        return false;
    }
}

static int moe_cache_find_pool(
        const moe_cache_device & device, size_t expert_size, int wtype) {
    for (int index = 0; index < (int)device.pools.size(); index++) {
        const moe_cache_pool & pool = *device.pools[index];
        if (pool.expert_size == expert_size && pool.wtype == wtype) {
            return index;
        }
    }
    return -1;
}

// ---------------------------------------------------------------------
// Heat-aware intra-expert neuron subsetting - reduced-width admission.
//
// Scoped per what was actually measured (docs/plan.md, "Heat-aware
// intra-expert neuron subsetting"): gate_exps/up_exps only (down_proj
// deferred - its neuron axis is the Q4_K quantization-block axis, not
// row-addressable without a repack). Per-EXPERT top-K selection, not a
// shared/global index set - measured directly that a shared set only
// retains ~56% of aggregate mass at 50% width, because different experts
// care about different specific neurons; a per-expert set retains ~90% at
// a similar width for a typical expert.
//
// Deliberately NOT layer-scoped, unlike an earlier version of this
// comment. The first pass looked at per-LAYER medians and found layer 0
// alone standing out (30.9% median dead-neuron fraction vs ~0% for every
// other layer) - but re-checking at the individual-expert level found 17
// real outlier experts scattered across layers 1, 2, 3, 4, 37 and 39 too,
// some on large sample counts (not noise). The layer-0 median was masking
// a genuine per-expert effect that exists everywhere, just unevenly
// distributed - restricting to one layer would have thrown those 17 away
// for no principled reason. So every expert in every layer gets the same
// eligibility check; whichever ones actually qualify convert.
//
// Lifecycle: an expert is admitted normally (full width) first, same as
// every other expert - this is also what lets neuron_heat accumulate real
// data for it in the first place (heat is only recorded on GPU-side hits,
// via moe_cache_collect()). Only once it has enough accumulated heat to
// trust a top-K ranking does it become eligible for CONVERSION to a
// reduced-width slot (moe_cache_try_reduce_convert). This is not a special
// admission path, it's a promotion that happens after normal admission has
// already done its job.
//
// The reduced-width pool self-stabilizes the same way the primary pool
// does: once it fills up, a new conversion evicts the coldest unpinned
// slot already in it (same moe_cache_pick_coldest_unpinned policy, same
// segments) rather than going permanently inert - so under real memory
// pressure this pool's own contents track whichever converted experts are
// CURRENTLY hottest, not just whichever converted first. An evicted
// reduced expert's decision is un-decided again (its device.reduced_indices
// entry is dropped, not just its slot), so it is eligible to re-convert
// later if it earns enough heat again - it is only the SELECTION ranking
// for a still-resident key that is convert-once (never re-ranked while
// its slot survives): re-ranking a resident key from the same
// ever-growing heat vector would not change the answer anyway, since heat
// only accumulates and never resets per key.
// DEFAULT ON. Measured +8.1% tok/s at matched placement and warm-up, with
// 50% VRAM saved on every converted expert, and correctness verified across
// multi-topic multi-turn conversation. Set GGML_CUDA_MOE_CACHE_NEURON_REDUCE=0
// to turn it off.
//
// Depends on neuron heat being collected (it is the signal this selects on),
// so moe_cache_neuron_heat_enabled() defaults on for the same reason.
static bool moe_cache_neuron_reduce_enabled() {
    static const bool enabled = [] {
        const char * env = getenv("GGML_CUDA_MOE_CACHE_NEURON_REDUCE");
        return env ? atoi(env) != 0 : true;
    }();
    return enabled;
}

// Fixed VRAM reservation for the reduced-width pool(s), capped further by
// whatever headroom is actually left under the cache's own overall budget
// at the moment the first conversion happens (moe_cache_try_reduce_convert)
// - this is never allowed to push total usage past the existing
// --moe-cache budget, only to claim a slice of what that budget already
// allows and hasn't yet been spent.
static size_t moe_cache_neuron_reduce_budget_bytes() {
    static const size_t bytes = [] {
        const char * env = getenv("GGML_CUDA_MOE_CACHE_NEURON_REDUCE_BUDGET_MB");
        long long mb = env ? atoll(env) : 256;
        if (mb < 0) {
            mb = 0;
        }
        return (size_t) mb << 20;
    }();
    return bytes;
}

// K, the reduced width, in neurons (not a percentage - the measured
// per-expert required-width varies too much for one shared percentage to
// mean the same thing across experts; a fixed neuron count is what
// actually needs to be uniform for mul_mat_id's batching, see the struct
// comment on device.reduced_indices below).
static int moe_cache_neuron_reduce_k() {
    static const int k = [] {
        const char * env = getenv("GGML_CUDA_MOE_CACHE_NEURON_REDUCE_K");
        return env ? atoi(env) : 256;
    }();
    return k;
}

// Minimum accumulated mass (sum of |value| across a key's whole heat
// vector) before its ranking is trusted enough to convert. Without this,
// an expert admitted moments ago with only 1-2 real samples could get
// "reduced" based on almost no evidence - the same class of mistake
// flagged earlier for reactive/session-limited signals generally.
static double moe_cache_neuron_reduce_min_mass() {
    static const double min_mass = [] {
        const char * env = getenv("GGML_CUDA_MOE_CACHE_NEURON_REDUCE_MIN_MASS");
        return env ? atof(env) : 50.0;
    }();
    return min_mass;
}

// Given one expert's full neuron_heat vector, return the top-K neuron
// indices by accumulated |value|, sorted ASCENDING by index (not by
// rank) - ascending order makes the gather-at-fill and scatter-at-collect
// loops simple linear scans instead of needing a reverse lookup, and the
// actual VALUE ordering doesn't matter once the set is chosen, only
// membership does. Returns an empty vector if there isn't enough
// accumulated mass yet to trust the ranking (see min_mass above), or if
// the heat vector is too small to be a real gate/up-width vector.
static std::vector<int32_t> moe_cache_select_reduced_indices(const std::vector<float> & heat, int k) {
    if ((int) heat.size() < 8 || k <= 0 || k >= (int) heat.size()) {
        return {};
    }
    double mass = 0.0;
    for (float v : heat) {
        mass += v;
    }
    if (mass < moe_cache_neuron_reduce_min_mass()) {
        return {};
    }
    std::vector<int32_t> idx(heat.size());
    for (size_t i = 0; i < idx.size(); i++) {
        idx[i] = (int32_t) i;
    }
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
            [&](int32_t a, int32_t b) { return heat[a] > heat[b]; });
    idx.resize(k);
    std::sort(idx.begin(), idx.end());
    return idx;
}

// How often moe_cache_prepare_budget() re-queries live VRAM instead of
// trusting its last answer. This function runs on the hot per-tensor
// dispatch path (moe_cache_begin(), called for every MoE layer access, not
// just once per new shape), so it can't re-issue cudaMemGetInfo - a
// synchronous CUDA API call - on every single call without adding real
// per-token overhead. 2 seconds is frequent enough that a real change in
// device pressure (another process starting, the KV cache growing as a
// conversation gets longer) gets noticed within a couple of seconds, while
// keeping the sync call rate low enough (well under 1/s under any real
// decode throughput) to not show up as measurable overhead.
static constexpr std::chrono::milliseconds MOE_CACHE_BUDGET_RECHECK_INTERVAL{2000};

// Cross-run usage history (see moe_cache_session::history).
//
// Format is deliberately plain text and versioned, following the reference
// implementation described in docs/moe-cache-colibri-notes.md: a header record
// carrying version and an identity hash, then "layer expert count" lines. The
// identity hash covers layer count, expert count and expert size, so a file
// written for one model can never seed another's placement - a mismatch is
// refused outright rather than silently degrading routing.
//
// Written atomically (.tmp + rename) so a crash mid-save cannot leave a
// truncated file that the next launch would half-read.

static constexpr uint32_t MOE_CACHE_HISTORY_VERSION = 1;

static const char * moe_cache_history_path() {
    static const char * path = getenv("GGML_CUDA_MOE_CACHE_HISTORY");
    return path;
}

static uint64_t moe_cache_history_identity(const moe_cache_session & session, const moe_cache_device & device) {
    uint64_t h = 1469598103934665603ull;
    auto mix = [&h](uint64_t v) { h = (h ^ v) * 1099511628211ull; };
    mix((uint64_t) session.tensor_layer.size());
    mix((uint64_t) session.n_expert_hint);
    for (const auto & shape : device.shapes) {
        mix((uint64_t) shape.expert_size);
        mix((uint64_t) shape.wtype);
    }
    return h;
}

static void moe_cache_history_load(moe_cache_session & session, const moe_cache_device & device) {
    if (session.history_loaded) {
        return;
    }
    session.history_loaded = true;

    const char * path = moe_cache_history_path();
    if (!path) {
        return;
    }

    std::ifstream f(path);
    if (!f) {
        return;
    }

    uint32_t version = 0;
    unsigned long long identity = 0;
    std::string tag;
    if (!(f >> tag >> version >> identity) || tag != "moe-cache-history" ||
        version != MOE_CACHE_HISTORY_VERSION) {
        MOE_CACHE_LOG("[moe-cache] usage history '%s' unreadable or wrong version - ignoring\n", path);
        return;
    }
    if (identity != moe_cache_history_identity(session, device)) {
        MOE_CACHE_LOG("%s", "[moe-cache] usage history was written for a different model/shape - ignoring\n");
        return;
    }

    int layer = 0, expert = 0;
    unsigned long long count = 0;
    size_t n = 0;
    while (f >> layer >> expert >> count) {
        if (layer < 0 || expert < 0 || count == 0) {
            continue;
        }
        session.history[((uint64_t) (uint32_t) layer << 32) | (uint32_t) expert] =
            (uint32_t) std::min<unsigned long long>(count, UINT32_MAX);
        n++;
    }
    if (n) {
        MOE_CACHE_LOG("[moe-cache] loaded usage history for %zu expert(s) from '%s'\n", n, path);
    }
}

static void moe_cache_history_save(moe_cache_session & session, const moe_cache_device & device) {
    const char * path = moe_cache_history_path();
    if (!path) {
        return;
    }

    // Merge this run's live selections into whatever was loaded, so history
    // accumulates across runs instead of each launch overwriting the last.
    for (const auto & [host_base, res] : device.residency) {
        auto it = session.tensor_layer.find(host_base);
        if (it == session.tensor_layer.end()) {
            continue;
        }
        for (size_t expert = 0; expert < res.selections.size(); expert++) {
            if (res.selections[expert] == 0) {
                continue;
            }
            const uint64_t key = ((uint64_t) (uint32_t) it->second << 32) | (uint32_t) expert;
            uint32_t & slot = session.history[key];
            const uint64_t sum = (uint64_t) slot + res.selections[expert];
            slot = (uint32_t) std::min<uint64_t>(sum, UINT32_MAX);
        }
    }
    if (session.history.empty()) {
        return;
    }

    const std::string tmp = std::string(path) + ".tmp";
    {
        std::ofstream f(tmp, std::ios::trunc);
        if (!f) {
            return;
        }
        f << "moe-cache-history " << MOE_CACHE_HISTORY_VERSION << " "
          << (unsigned long long) moe_cache_history_identity(session, device) << "\n";
        for (const auto & [key, count] : session.history) {
            f << (uint32_t) (key >> 32) << " " << (uint32_t) (key & 0xffffffffull) << " " << count << "\n";
        }
        if (!f) {
            f.close();
            std::remove(tmp.c_str());
            return;
        }
    }
    if (std::rename(tmp.c_str(), path) != 0) {
        std::remove(tmp.c_str());
    }
}

// Persist the co-activation graph this session has been accumulating, so the
// evolving atlas can be rebuilt from REAL SERVED TRAFFIC instead of from a
// separate offline `llama-moe-trace` run over synthetic prompts.
//
// The data already existed and was simply being discarded: device.co_activation
// holds exactly the within-layer edges scripts/moe-atlas-evolve.py derives from
// trace files (its fold_in() builds edges[(min,max)] += 1 over co-selected
// experts, keyed by a (layer, expert) node), and moe_cache_plan has been
// maintaining it on every routing decision all along - it just never outlived
// the process. Writing it out closes the loop: serve traffic, persist the
// graph, re-run the evolve script against it, feed the result back through
// --expert-atlas-file. The atlas then reflects what this deployment actually
// routes, which is the whole point of a "discovered" atlas and something an
// offline probe over invented prompts cannot do by construction.
//
// Emitted as `layer_a expert_a layer_b expert_b weight`, one edge per line -
// the node identity the evolve script uses, written directly rather than
// re-derived, so nothing has to reconstruct layer numbers from tensor
// pointers on the Python side. Same atomic .tmp + rename and same plain-text
// versioned shape as the history file above, for the same reason: a crash
// mid-save must not leave a half-written graph that the next run silently
// folds in as real observations.
static void moe_cache_coact_save(moe_cache_session & session, const moe_cache_device & device) {
    static const char * path = getenv("GGML_CUDA_MOE_CACHE_COACT_FILE");
    if (!path) {
        return;
    }
    // Load once, so this run ADDS to whatever previous runs recorded instead
    // of replacing it (see session.coact_history).
    if (!session.coact_loaded) {
        session.coact_loaded = true;
        std::ifstream in(path);
        if (in) {
            std::string tag; int version = 0;
            if ((in >> tag >> version) && tag == "moe-cache-coact" && version >= 2) {
                int la, ea, lb, eb; unsigned long long w;
                while (in >> la >> ea >> lb >> eb >> w) {
                    const uint64_t key = ((uint64_t)(uint32_t) la << 48) | ((uint64_t)(uint32_t) ea << 32) |
                                         ((uint64_t)(uint32_t) lb << 16) | (uint64_t)(uint32_t) eb;
                    session.coact_history[key] += w;
                }
            }
        }
    }

    // BOTH edge sets, and the cross-layer one is not optional. Within-layer
    // edges alone leave every layer as its own disconnected component, and a
    // normalised adjacency's eigenvalue 1 then has multiplicity equal to the
    // component count - so the consumer's eigengap heuristic sees N trivial
    // 1.0 eigenvalues and picks an axis count out of pure degeneracy rather
    // than real structure. Confirmed the hard way: a first version of this
    // exported only device.co_activation, and the atlas built from it had its
    // top 12 eigenvalues all exactly 1.0. scripts/moe-atlas-evolve.py's own
    // fold_in() ties the graph together with a cross-layer edge for precisely
    // this reason; co_activation_cross_layer is that same signal, already
    // measured live, so it must be written alongside.
    const std::unordered_map<moe_cache_edge, uint32_t, moe_cache_edge_hash> * maps[2] = {
        &device.co_activation, &device.co_activation_cross_layer
    };
    for (int m = 0; m < 2; m++) {
        for (const auto & [edge, count] : *maps[m]) {
            const auto la = session.tensor_layer.find(edge.from.tensor);
            const auto lb = session.tensor_layer.find(edge.to.tensor);
            // A tensor with no known layer cannot be placed in the graph's
            // (layer, expert) node space at all - skip rather than invent a
            // layer index, which would fabricate structure.
            if (la == session.tensor_layer.end() || lb == session.tensor_layer.end()) {
                continue;
            }
            if (edge.from.expert < 0 || edge.to.expert < 0) {
                continue;
            }
            const uint64_t key = ((uint64_t)(uint32_t) la->second << 48) |
                                 ((uint64_t)(uint32_t) edge.from.expert << 32) |
                                 ((uint64_t)(uint32_t) lb->second << 16) |
                                 (uint64_t)(uint32_t) edge.to.expert;
            session.coact_history[key] += count;
        }
    }
    if (session.coact_history.empty()) {
        return;
    }

    const std::string tmp = std::string(path) + ".tmp";
    {
        std::ofstream f(tmp, std::ios::trunc);
        if (!f) {
            return;
        }
        f << "moe-cache-coact 2\n";
        for (const auto & [key, w] : session.coact_history) {
            f << (uint32_t)((key >> 48) & 0xffff) << " " << (uint32_t)((key >> 32) & 0xffff) << " "
              << (uint32_t)((key >> 16) & 0xffff) << " " << (uint32_t)(key & 0xffff) << " " << w << "\n";
        }
        if (!f) {
            f.close();
            std::remove(tmp.c_str());
            return;
        }
    }
    if (std::rename(tmp.c_str(), path) != 0) {
        std::remove(tmp.c_str());
    }
}

static bool moe_cache_prepare_budget(
        moe_cache_session & session, moe_cache_device & device) {
    const auto now = std::chrono::steady_clock::now();
    if (device.budget_ready && now - device.budget_checked_at < MOE_CACHE_BUDGET_RECHECK_INTERVAL) {
        return device.budget_limit > 0;
    }
    const bool first_check = !device.budget_ready;
    device.budget_ready = true;
    device.budget_checked_at = now;

    ggml_cuda_set_device(device.physical);
    size_t free_memory = 0;
    size_t total_memory = 0;
    cudaError_t error = cudaMemGetInfo(&free_memory, &total_memory);
    if (!moe_cache_cuda_ok(device, error, "memory query", false)) {
        device.dead.store(true);
        return false;
    }

    // Live default when GGML_CUDA_MOE_CACHE_RESERVE_MB wasn't explicitly set:
    // 5% of *actual* free VRAM at this exact moment, clamped to [128, 1024]
    // MiB. A fixed constant can't be right across both a 12GB card (where a
    // 3GB reserve left ~47MB available after loading a 26B model - measured
    // directly, see docs/moe-cache-colibri-notes.md) and a 141GB one (where
    // 3GB is negligible). Computed here, not at config-parse time, because
    // cudaMemGetInfo above is the actual live resource query - this reflects
    // what's free right now, not a stale snapshot from an earlier planning
    // step on a shared multi-tenant machine where availability can shift.
    //
    // free_memory alone already excludes the cache's own pools once any have
    // been allocated (they're real, resident CUDA buffers), so re-deriving
    // the budget straight from free_memory on every re-check would make the
    // cache see its own prior growth as shrinking headroom and throttle
    // itself in response to nothing - a self-inflicted ratchet down to
    // whatever the budget happened to be on the first call, silently
    // defeating the point of re-checking at all. Adding device.allocated_bytes
    // back reconstructs "how much room exists for the cache in total",
    // matching what free_memory already meant on the very first call, when
    // allocated_bytes was still 0 - so the budget only actually moves in
    // response to something *other* than the cache itself: another process,
    // or this device's own KV cache/compute buffers growing or shrinking.
    const size_t free_for_budget = free_memory + device.allocated_bytes;
    size_t reserve_mb = session.config.reserve_mb;
    if (reserve_mb == 0) {
        const size_t free_mb = free_for_budget >> 20;
        reserve_mb = std::min<size_t>(1024, std::max<size_t>(128, free_mb / 20));
    }
    const size_t reserve = reserve_mb << 20;
    size_t available = free_for_budget > reserve ? free_for_budget - reserve : 0;
    if (session.config.budget_mb > 0) {
        available = std::min(available, session.config.budget_mb << 20);
    }
    // Never re-evaluate the budget below what is already allocated. Pools that
    // exist cannot be un-allocated here, so a lower number does not free
    // anything - it only makes every routing weight zero (the eligibility check
    // is `allocated_bytes > slab_limit`), which takes the cache inert: begin()
    // returns NULL for every node, so no hits, no misses, no evictions, and no
    // heat updates are ever recorded again. Found exactly that way: after this
    // live re-check was introduced, an 800-token generation left every counter
    // frozen at its warmup value, because the recomputed budget landed a few MiB
    // under allocated_bytes once the reserve was subtracted. Growth still
    // responds to real pressure; only shrink-below-committed is refused.
    const size_t previous_limit = device.budget_limit;
    const size_t committed = device.allocated_bytes + moe_cache_scratch_total(device.scratch_reserve);

    // Pressure response. `available < committed` means something outside this
    // cache took memory - another process, or this context's own KV cache
    // growing - and the clamp below is about to hide that by refusing to lower
    // the budget. Refusing is still right (a budget under what is already
    // allocated makes the cache inert; that bug is documented above), but doing
    // *only* that means the cache notices pressure and never yields to it.
    //
    // Pool slabs cannot be partially freed - one allocation per pool - so the
    // memory that can actually be returned is the dispatch scratch. It is
    // regrown on demand by moe_cache_grow_device() before the next dispatch, so
    // releasing it costs one reallocation rather than any cached expert. Only
    // done when nothing is in flight and the queue is drained, since a dispatch
    // in progress is reading these very buffers.
    if (available < committed && !device.inflight && device.queue.empty()) {
        const size_t released =
            device.d_ids_cap + device.d_act_cap + device.act_q8_cap + device.d_out_cap;
        if (released > 0) {
            ggml_cuda_set_device(device.physical);
            cudaFree(device.d_ids);   device.d_ids   = nullptr; device.d_ids_cap   = 0;
            cudaFree(device.d_act);   device.d_act   = nullptr; device.d_act_cap   = 0;
            cudaFree(device.d_act_q8);device.d_act_q8= nullptr; device.act_q8_cap  = 0;
            cudaFree(device.d_out);   device.d_out   = nullptr; device.d_out_cap   = 0;
            MOE_CACHE_LOG("[moe-cache] CUDA%d under memory pressure (%zu MiB available vs %zu MiB held) - "
                    "released %zu MiB of dispatch scratch\n",
                    device.physical, available >> 20, committed >> 20, released >> 20);
        }
    }

    device.budget_limit = std::max(available, committed);

    // An EXPLICIT budget is a hard ceiling. The never-shrink-below-committed
    // rule above exists so a live re-check landing under what is already
    // allocated cannot make the cache inert (see its comment), but it also let
    // an explicit GGML_CUDA_MOE_CACHE_BUDGET_MB ratchet upward: once pools grew
    // past the budget, `committed` exceeded it and max() adopted the overrun as
    // the new limit. Measured: BUDGET_MB=256 and 3413 both peaked at 11,797 MiB
    // - the knob did nothing at all.
    //
    // Growth is what escapes the budget: moe_cache_grow_device() cudaMallocs
    // without consulting budget_limit, so the ceiling has to be reasserted here
    // on every re-check rather than trusted to hold from pool creation.
    if (session.config.budget_mb > 0) {
        const size_t hard = session.config.budget_mb << 20;
        if (device.budget_limit > hard) {
            if (first_check) {
                MOE_CACHE_LOG("[moe-cache] CUDA%d budget capped to %zu MiB by "
                        "GGML_CUDA_MOE_CACHE_BUDGET_MB\n", device.physical, hard >> 20);
            }
            device.budget_limit = hard;
        }
    }

    if (!first_check && available != previous_limit) {
        MOE_CACHE_LOG("[moe-cache] CUDA%d cache budget re-evaluated: %zu -> %zu MiB (%zu MiB free, %zu MiB already cached)\n",
                device.physical, previous_limit >> 20, available >> 20,
                free_memory >> 20, device.allocated_bytes >> 20);
    }

    if (available == 0) {
        MOE_CACHE_LOG("[moe-cache] CUDA%d has no cache budget after %zu MiB reserve (%zu MiB free)\n",
                device.physical, reserve_mb, free_memory >> 20);
        return false;
    }
    return true;
}

// Below this many slots a pool is not worth creating at all (it would
// thrash immediately) - moe_cache_allocate_pool refuses outright. Named
// because moe_cache_build_pending's reduce reservation must know the same
// floor to avoid starving a shape below it; it was a bare 64 in three
// places before that reservation existed.
static constexpr size_t MOE_CACHE_MIN_POOL_SLOTS = 64;

static bool moe_cache_allocate_pool(
        moe_cache_session & session, moe_cache_device & device,
        moe_cache_shape & shape, size_t budget) {
    if (shape.expert_size == 0 || shape.n_tensors <= 0 || shape.n_expert <= 0) {
        return false;
    }
    if (shape.expert_size > std::numeric_limits<size_t>::max() - MOE_CACHE_SLOT_GUARD_BYTES) {
        return false;
    }
    // The matvec kernel asserts slot_stride_bytes % type_size(wtype) == 0
    // (it strides through the slab in whole quantization blocks) - a flat
    // +MOE_CACHE_SLOT_GUARD_BYTES is not guaranteed to preserve that (e.g.
    // Q4_K's 144-byte block does not divide 4096), so round the guarded
    // stride up to the next block boundary rather than pick a fixed offset.
    const size_t guard_type_size = ggml_type_size((ggml_type)shape.wtype);
    if (guard_type_size == 0) {
        return false;
    }
    const size_t guard_min = shape.expert_size + MOE_CACHE_SLOT_GUARD_BYTES;
    const size_t slot_stride =
        ((guard_min + guard_type_size - 1) / guard_type_size) * guard_type_size;

    int64_t max_entries = shape.n_tensors;
    if (max_entries > INT_MAX / shape.n_expert) {
        max_entries = INT_MAX;
    } else {
        max_entries *= shape.n_expert;
    }

    size_t slots_by_budget = budget / slot_stride;
    size_t slot_count = std::min<size_t>(slots_by_budget, (size_t)max_entries);
    const size_t type_size = ggml_type_size((ggml_type)shape.wtype);
    if (type_size == 0 || shape.expert_size % type_size != 0) {
        return false;
    }
    const size_t stride_blocks = shape.expert_size / type_size;
    if (stride_blocks == 0) {
        return false;
    }
    slot_count = std::min(slot_count, (size_t)INT_MAX / stride_blocks);
    if (slot_count > INT_MAX) {
        slot_count = INT_MAX;
    }
    if (slot_count < MOE_CACHE_MIN_POOL_SLOTS) {
        if (getenv("MOE_CACHE_DEBUG_GATE")) {
            fprintf(stderr, "[moe-cache-pool-dbg] pool creation SKIPPED: expert_size=%zu budget=%zu "
                    "slots_by_budget=%zu max_entries=%lld slot_count=%zu\n",
                    shape.expert_size, budget, slots_by_budget, (long long) max_entries, slot_count);
        }
        return false;
    }

    ggml_cuda_set_device(device.physical);
    char * slab = nullptr;
    cudaError_t error = cudaSuccess;
    while (slot_count >= MOE_CACHE_MIN_POOL_SLOTS) {
        if (moe_cache_fail(session, "slab")) {
            error = cudaErrorMemoryAllocation;
        } else {
            error = cudaMalloc((void **)&slab, slot_count * slot_stride);
        }
        if (error == cudaSuccess) {
            break;
        }
        (void)cudaGetLastError();
        slot_count /= 2;
    }
    if (error != cudaSuccess || !slab || slot_count < MOE_CACHE_MIN_POOL_SLOTS) {
        MOE_CACHE_LOG("[moe-cache] CUDA%d skipped %zu KiB expert pool: allocation failed\n",
                device.physical, shape.expert_size >> 10);
        return false;
    }
    // Poison the whole slab with the guard pattern before anything real is
    // written. The fill worker only ever overwrites the first expert_size
    // bytes of each slot's slot_stride-sized region, so this pattern
    // persists forever in the per-slot gaps unless something writes there
    // that has no business doing so.
    if (cudaMemset(slab, MOE_CACHE_GUARD_PATTERN, slot_count * slot_stride) != cudaSuccess) {
        (void) cudaGetLastError();
        cudaFree(slab);
        return false;
    }

    std::unique_ptr<moe_cache_pool> pool(new (std::nothrow) moe_cache_pool());
    if (!pool) {
        cudaFree(slab);
        return false;
    }
    try {
        pool->expert_size = shape.expert_size;
        pool->slot_stride = slot_stride;
        pool->wtype = shape.wtype;
        pool->slab = slab;
        pool->n_slots = (int)slot_count;
        if (getenv("GGML_CUDA_MOE_CACHE_CANARY")) {
            const size_t cb = 1u << 20;
            char * canary = nullptr;
            if (cudaMalloc((void **)&canary, cb) == cudaSuccess) {
                std::vector<char> pattern(cb, (char)0xA5);
                if (cudaMemcpy(canary, pattern.data(), cb, cudaMemcpyHostToDevice) == cudaSuccess) {
                    pool->canary = canary; pool->canary_bytes = cb;
                    fprintf(stderr, "[moe-cache] CANARY armed: slab=%p..%p (%zu slots x %zu B, "
                            "stride %zu B) canary=%p+%zu\n",
                            (void*)slab, (void*)(slab + slot_count*slot_stride),
                            slot_count, shape.expert_size, slot_stride, (void*)canary, cb);
                } else {
                    cudaFree(canary);
                }
            }
        }
        pool->slots.resize(slot_count);
        pool->free_slots.reserve(slot_count);
        pool->map.reserve(slot_count);
        for (int index = (int)slot_count - 1; index >= 0; index--) {
            pool->free_slots.push_back(index);
        }
        device.pools.push_back(std::move(pool));
    } catch (...) {
        cudaFree(slab);
        return false;
    }

    shape.pool = (int)device.pools.size() - 1;
    const size_t allocated = slot_count * slot_stride;
    device.allocated_bytes += allocated;
    if (getenv("MOE_CACHE_DEBUG_GATE")) {
        fprintf(stderr, "[moe-cache-pool-dbg] pool CREATED: expert_size=%zu slot_count=%zu allocated=%zu budget=%zu\n",
                shape.expert_size, slot_count, allocated, budget);
    }

    if (!moe_cache_start_worker(session, device)) {
        cudaFree(device.pools.back()->slab);
        device.pools.back()->slab = nullptr;
        device.pools.pop_back();
        device.allocated_bytes -= allocated;
        shape.pool = -1;
        return false;
    }

    if (!session.announced) {
        // reserve_mb==0 means "not explicitly set" - the real value used is
        // computed live per-device from actual free VRAM at that moment (see
        // moe_cache_prepare_budget) and logged there via the "no cache
        // budget after N MiB reserve" message on failure, or implied by
        // pool[]'s total size on success. Printing 0 here would misleadingly
        // read as "no reserve" rather than "computed live, see per-device".
        if (session.config.reserve_mb > 0) {
            MOE_CACHE_LOG("[moe-cache] enabled: mode=%s budget=%s reserve=%zu MiB (explicit) min-expert=%zu KiB admit=%d/%d max-batch=%d\n",
                    session.config.automatic ? "auto" : "on",
                    session.config.budget_mb ? "fixed" : "free-minus-reserve",
                    session.config.reserve_mb, session.config.min_expert_bytes >> 10,
                    session.config.admit_after, session.config.readmit_after,
                    session.config.max_batch);
        } else {
            MOE_CACHE_LOG("[moe-cache] enabled: mode=%s budget=%s reserve=live-per-device(5%% free, 128-1024 MiB) min-expert=%zu KiB admit=%d/%d max-batch=%d\n",
                    session.config.automatic ? "auto" : "on",
                    session.config.budget_mb ? "fixed" : "free-minus-reserve",
                    session.config.min_expert_bytes >> 10,
                    session.config.admit_after, session.config.readmit_after,
                    session.config.max_batch);
        }
        session.announced = true;
    }
    MOE_CACHE_LOG("[moe-cache] CUDA%d pool[%d]: type=%s expert=%zu KiB slots=%zu total=%zu MiB\n",
            device.physical, shape.pool, ggml_type_name((ggml_type)shape.wtype),
            shape.expert_size >> 10, slot_count, allocated >> 20);

    // Marked finished only here, on success - not unconditionally at entry.
    // Every early `return false` above (insufficient budget, slot_count<64,
    // worker start failure) used to also mark it finished, which permanently
    // stuck this shape: a later, bigger tensor of the exact same shape could
    // never re-trigger allocation, since moe_cache_discover_pool only resets
    // `finished` when a shape is being seen for the very first time. That
    // tensor's layer would then just never get cached for the rest of the
    // process, silently. Leaving it false lets moe_cache_build_pending retry
    // on the next call, same as if this shape had never been attempted.
    shape.finished = true;
    return true;
}

// Seed freshly-created pools from the previous run's history.
//
// Deliberately placed here, immediately after pool creation, because every slot
// is still free at this point: admission is a pop from free_slots with no
// eviction, no heat comparison and no LRU surgery. That keeps this out of the
// path where a mistake could corrupt live cache state - it can only ever fill
// slots that nothing is using yet.
//
// Heat-aware neuron subsetting: how moe_cache_plan tells a reduced-pool hit
// apart from an ordinary primary-pool hit in slot_indices[], the same array
// ggml-cpu.c reads as an opaque "which compact slot" value and passes
// straight through to dispatch()/collect() unmodified (ggml-cpu.c only ever
// checks `>= 0`, see the mul_mat_id path in ggml-cpu.c). A primary pool's
// n_slots is bounded by its VRAM budget in bytes divided by its (at least
// hundreds-of-KiB) expert size - nowhere close to this offset for any
// budget this cache could plausibly be given, so encoding as offset+slot is
// unambiguous and reversible with no risk of colliding with a real primary
// slot number.
static constexpr int32_t MOE_CACHE_REDUCED_SLOT_OFFSET = 1 << 24;

// Attempt to convert `key` (a resident, full-width expert whose neuron_heat
// just accumulated another sample - see the caller in moe_cache_collect) to
// a reduced-width slot. Idempotent and convert-once: the very first thing
// this does under session.mu is check device.reduced_indices for `key` and
// return immediately if it's already there, whether or not that earlier
// attempt actually finished converting it (see the comment above
// moe_cache_neuron_reduce_enabled for why a retry policy was deliberately
// left out). Safe to call speculatively for every hit - the common case
// (already decided, or not enough accumulated mass yet) is a single hash
// lookup and an early return.
//
// full_expert_size/wtype/n_expert describe the SOURCE tensor this key
// belongs to (node->expert_size/wtype/n_expert at the call site) - needed
// to size and create the reduced pool and to record the gather geometry,
// none of which is recoverable from `key` alone.
static void moe_cache_try_reduce_convert(
        moe_cache_session & session, moe_cache_device & device,
        const moe_cache_key & key, int64_t n_out,
        size_t full_expert_size, int wtype, int64_t n_expert) {
    std::lock_guard<std::mutex> lock(session.mu);
    if (session.stopping || device.dead.load()) {
        return;
    }
    if (device.reduced_indices.find(key) != device.reduced_indices.end()) {
        return;
    }
    // Records `entry` as this key's decision in BOTH reduced_indices
    // (session.mu-protected, the source of truth for pool/slot bookkeeping)
    // and reduce_decided (neuron_heat_mu-protected, the cheap fast-path
    // membership check moe_cache_collect uses to avoid taking session.mu at
    // all for the overwhelmingly common already-decided case - see that
    // field's own comment for the regression this fixed).
    auto decide = [&](moe_cache_device::reduced_entry entry) {
        device.reduced_indices.emplace(key, std::move(entry));
        std::lock_guard<std::mutex> hlock(device.neuron_heat_mu);
        device.reduce_decided.insert(key);
    };
    if (n_out <= 0 || full_expert_size == 0 || full_expert_size % (size_t) n_out != 0) {
        return;
    }
    const int k = moe_cache_neuron_reduce_k();
    std::vector<int32_t> indices;
    {
        // session.mu -> neuron_heat_mu, the order documented on
        // device.reduced_indices - never the reverse.
        std::lock_guard<std::mutex> hlock(device.neuron_heat_mu);
        const auto hit = device.neuron_heat.find(key);
        if (hit == device.neuron_heat.end() || (int64_t) hit->second.size() != n_out) {
            return;
        }
        indices = moe_cache_select_reduced_indices(hit->second, k);
    }
    if (indices.empty()) {
        // Not enough accumulated mass yet, or k is out of range for this
        // tensor's width - stays undecided, eligible for a later call once
        // more heat has accumulated. Deliberately does NOT record a
        // decided-but-empty entry: unlike every failure mode below this
        // point (which are about VRAM/slot availability, not about whether
        // the ranking itself is trustworthy yet), this one is expected to
        // resolve itself with more data.
        return;
    }

    const size_t row_bytes = full_expert_size / (size_t) n_out;
    if (row_bytes == 0) {
        return;
    }
    const size_t reduced_expert_size = (size_t) indices.size() * row_bytes;

    int pool_index = moe_cache_find_pool(device, reduced_expert_size, wtype);
    if (pool_index < 0) {
        const size_t headroom = device.budget_limit > device.allocated_bytes
            ? device.budget_limit - device.allocated_bytes : 0;
        const size_t budget = std::min(moe_cache_neuron_reduce_budget_bytes(), headroom);
        moe_cache_shape shape{reduced_expert_size, wtype, n_expert, 1, -1, false};
        if (!moe_cache_allocate_pool(session, device, shape, budget)) {
            // Decided (see the decide() call below), never retried this
            // session - see the "convert-once" rationale above.
            decide(moe_cache_device::reduced_entry{
                    std::move(indices), full_expert_size, row_bytes, -1});
            return;
        }
        pool_index = shape.pool;
    }
    moe_cache_pool & rpool = *device.pools[pool_index];
    if (rpool.map.find(key) != rpool.map.end()) {
        // Already resident in the reduced pool (should not happen given the
        // decided-key guard above, but harmless either way).
        decide(moe_cache_device::reduced_entry{
                std::move(indices), full_expert_size, row_bytes, pool_index});
        return;
    }
    int slot_index = -1;
    if (!rpool.free_slots.empty()) {
        slot_index = rpool.free_slots.back();
        rpool.free_slots.pop_back();
    } else {
        // Same reactive, coldest-unpinned-wins eviction the primary pool's
        // own admission path uses (moe_cache_plan) - drain probation before
        // protected_, and within a segment, heat (not recency alone)
        // decides. This is what makes the reduced pool self-stabilizing
        // under its own budget the same way the primary pool already is,
        // rather than a first-come-first-served allocation that goes
        // permanently inert once full: a newly-eligible, hotter expert can
        // displace a colder one that already converted.
        int victim = moe_cache_pick_coldest_unpinned(device, rpool, rpool.lru_head);
        if (victim < 0) {
            victim = moe_cache_pick_coldest_unpinned(device, rpool, rpool.protected_head);
        }
        if (victim < 0) {
            // Every slot is pinned (in-flight reads) - decided, but not
            // placed. Not retried: the same convert-once policy as every
            // other failure path here. A future hit for this key falls
            // through to the ordinary primary-pool path indefinitely (see
            // the struct comment on device.reduced_indices) rather than
            // this function being called again to find the same result.
            decide(moe_cache_device::reduced_entry{
                    std::move(indices), full_expert_size, row_bytes, pool_index});
            return;
        }
        // Unlike a primary-pool eviction, the victim's OWN reduced_indices
        // AND reduce_decided entries must be dropped too (moe_cache_slot_reset
        // only touches the pool/slot, not these device-level maps) - erasing
        // both makes the victim's key eligible for re-conversion later if it
        // earns enough heat again, rather than being permanently exiled from
        // ever re-entering this pool once evicted from it.
        {
            const moe_cache_key victim_key = rpool.slots[victim].key;
            device.reduced_indices.erase(victim_key);
            std::lock_guard<std::mutex> hlock(device.neuron_heat_mu);
            device.reduce_decided.erase(victim_key);
        }
        moe_cache_slot_reset(rpool, victim, false);
        device.evictions++;
        slot_index = victim;
    }
    moe_cache_slot & slot = rpool.slots[slot_index];
    slot.key = key;
    slot.generation++;
    slot.readers = 0;
    slot.state = moe_cache_slot_state::copying;
    if (!moe_cache_map_insert(rpool, key, slot_index)) {
        moe_cache_slot_reset(rpool, slot_index, true);
        decide(moe_cache_device::reduced_entry{
                std::move(indices), full_expert_size, row_bytes, pool_index});
        return;
    }

    // Record BEFORE queuing the job: the worker (moe_cache_worker) looks
    // this entry up by key, under session.mu, to know which rows to gather
    // and what stride to gather them from - see the reduce_convert branch
    // there.
    decide(moe_cache_device::reduced_entry{
            indices, full_expert_size, row_bytes, pool_index});

    moe_cache_job job;
    job.pool = pool_index;
    job.slot = slot_index;
    job.generation = slot.generation;
    job.key = key;
    job.source = key.tensor; // tensor base - the worker computes the per-expert,
                              // per-row offset itself from reduced_indices
    job.bytes = reduced_expert_size; // gathered size; drives stage-buffer growth
                                      // exactly like an ordinary fill job
    job.kind = moe_cache_job_kind::reduce_convert;
    device.queue.push_back(job);
    device.queued_bytes += reduced_expert_size;
    session.cv.notify_all();
}

// Bounded to half of each pool so a stale or over-large history cannot claim the
// whole cache before the live workload has said anything; the remaining half is
// left for what this run actually routes to.
static void moe_cache_prewarm_from_history(
        moe_cache_session & session, moe_cache_device & device) {
    moe_cache_history_load(session, device);
    if (session.history.empty()) {
        return;
    }

    // Rank the recorded experts, hottest first.
    struct warm_candidate { uint32_t count; int layer; int expert; };
    std::vector<warm_candidate> ranked;
    ranked.reserve(session.history.size());
    for (const auto & [key, count] : session.history) {
        ranked.push_back({count, (int) (uint32_t) (key >> 32), (int) (uint32_t) (key & 0xffffffffull)});
    }
    std::sort(ranked.begin(), ranked.end(), [](const warm_candidate & a, const warm_candidate & b) {
        if (a.count != b.count) return a.count > b.count;
        if (a.layer != b.layer) return a.layer < b.layer;
        return a.expert < b.expert;
    });

    // layer -> host_base, so a recorded (layer, expert) can be turned back into
    // an address to copy from.
    std::unordered_map<int, const void *> base_of_layer;
    for (const auto & [host_base, layer] : session.tensor_layer) {
        base_of_layer.emplace(layer, host_base);
    }

    size_t warmed = 0;
    for (const warm_candidate & c : ranked) {
        auto it_base = base_of_layer.find(c.layer);
        if (it_base == base_of_layer.end()) {
            continue;
        }
        const void * host_base = it_base->second;

        auto it_seen = device.seen_tensors.find(host_base);
        if (it_seen == device.seen_tensors.end()) {
            continue;
        }
        const size_t expert_size = it_seen->second.expert_size;
        const int    wtype       = it_seen->second.wtype;

        const int pool_index = moe_cache_find_pool(device, expert_size, wtype);
        if (pool_index < 0) {
            continue;
        }
        moe_cache_pool & pool = *device.pools[pool_index];

        // Half-pool ceiling, and never touch anything but free slots.
        if (pool.free_slots.size() <= pool.n_slots / 2 || pool.free_slots.empty()) {
            continue;
        }
        const moe_cache_key key{host_base, c.expert};
        if (pool.map.find(key) != pool.map.end()) {
            continue;
        }
        if ((int) device.queue.size() >= session.config.queue_max) {
            break;
        }

        const int slot_index = pool.free_slots.back();
        pool.free_slots.pop_back();

        moe_cache_slot & slot = pool.slots[slot_index];
        // Checked BEFORE the reset below, not after (a prior version of this
        // check tested slot.readers > 0 immediately following slot.readers =
        // 0 on the very same slot - always false, dead code, found during
        // the qwen4exp corruption investigation alongside the real bug this
        // was meant to catch: a generation-blind pin release in
        // moe_cache_prefill_reader_release_ctx, see its struct comment.
        if (moe_cache_pin_audit_enabled() && slot.readers > 0) {
            static std::atomic<int> nc{0};
            if (nc.fetch_add(1, std::memory_order_relaxed) < 20) {
                fprintf(stderr, "[moe-cache] *** PIN VIOLATION: refill (state->copying) with readers=%d\n",
                        slot.readers);
            }
        }
        slot.key = key;
        slot.generation++;
        slot.readers = 0;
        slot.state = moe_cache_slot_state::copying;
        try {
            if (!moe_cache_map_insert(pool, key, slot_index)) {
                moe_cache_slot_reset(pool, slot_index, true);
                continue;
            }
            device.queue.push_back({
                    pool_index, slot_index, slot.generation, key,
                    (const char *) host_base + (size_t) c.expert * expert_size,
                    expert_size,
                    host_base, it_seen->second.bytes});
            device.queued_bytes += expert_size;
        } catch (...) {
            moe_cache_slot_reset(pool, slot_index, true);
            continue;
        }
        warmed++;
    }

    if (warmed) {
        session.cv.notify_all();
        MOE_CACHE_LOG("[moe-cache] CUDA%d pre-warmed %zu expert(s) from previous runs' usage history\n",
                device.physical, warmed);
    }

    // Second tier of the same decision: the experts history says are hot but
    // that did not fit in VRAM will be served from CPU RAM, and on a
    // memory-pressured host those pages are faulted in one scattered read at a
    // time - measured as the actual cost of the 21x collapse, against a disk
    // that reads 513 MB/s sequentially. MADV_WILLNEED asks the kernel to read
    // them ahead in bulk instead.
    //
    // Deliberately WILLNEED and not mlock. Pinning was tried and measured worse
    // (see the notes): pinned pages cannot be reclaimed, so they shrink the pool
    // every other page competes for, and a budget small enough to be safe is too
    // small to cover the working set. An advisory hint has no such failure mode -
    // if RAM is short the kernel simply declines, and nothing else is starved.
#if defined(__linux__)
    static const bool readahead_enabled = [] {
        // Off by default, and this is a measured decision. Over 5 interleaved
        // rounds it won 4/5 on the cold request (+4.7% median) but cost ~8% on
        // warm throughput (median 11.10 vs 12.12 tok/s baseline), for 3.4 GiB of
        // page-cache pressure - a bad trade for any workload that serves more
        // than its first request. Kept because the cold-start gain is real and
        // may matter where restarts are frequent.
        const char * env = getenv("GGML_CUDA_MOE_CACHE_READAHEAD");
        return env && atoi(env) != 0;
    }();
    static const long page_size_w = sysconf(_SC_PAGESIZE);
    if (readahead_enabled && page_size_w > 0) {
        size_t advised = 0, advised_bytes = 0;
        for (const warm_candidate & c : ranked) {
            auto it_base = base_of_layer.find(c.layer);
            if (it_base == base_of_layer.end()) {
                continue;
            }
            auto it_seen = device.seen_tensors.find(it_base->second);
            if (it_seen == device.seen_tensors.end()) {
                continue;
            }
            const size_t expert_size = it_seen->second.expert_size;
            const uintptr_t begin = (uintptr_t) it_base->second + (uintptr_t) c.expert * expert_size;
            const uintptr_t end   = begin + expert_size;
            const uintptr_t first = (begin + page_size_w - 1) & ~((uintptr_t) page_size_w - 1);
            const uintptr_t last  = end & ~((uintptr_t) page_size_w - 1);
            if (last <= first) {
                continue;
            }
            if (posix_madvise((void *) first, (size_t) (last - first), POSIX_MADV_WILLNEED) != 0) {
                break; // not supported here - stop asking
            }
            advised++;
            advised_bytes += (size_t) (last - first);
        }
        if (advised) {
            MOE_CACHE_LOG("[moe-cache] CUDA%d asked the kernel to read ahead %zu historically-hot CPU expert(s) (%zu MiB)\n",
                    device.physical, advised, advised_bytes >> 20);
        }
    }
#endif
}

static void moe_cache_build_pending(
        moe_cache_session & session, moe_cache_device & device) {
    if (!moe_cache_prepare_budget(session, device)) {
        for (moe_cache_shape & shape : device.shapes) {
            if (shape.n_tensors > 0) {
                shape.finished = true;
            }
        }
        return;
    }

    const size_t scratch_reserve =
        moe_cache_scratch_total(device.scratch_reserve);
    // Heat-aware neuron subsetting needs real headroom to ever place
    // anything - reserved off the top here, the same way scratch_reserve
    // already is, rather than left to whatever primary admission happens
    // not to consume. Primary admission (the loop below) is greedy: it
    // divides up ALL of `remaining` among pending shapes every time this
    // function runs, so without this reservation the reduced pool's own
    // moe_cache_allocate_pool call (from moe_cache_try_reduce_convert)
    // would NEVER see enough headroom to create its pool - measured
    // directly: a full session produced 20000+ reduce decisions and placed
    // exactly zero of them, paying real per-decision lock overhead for no
    // benefit at all, before this reservation existed.
    //
    // Bounded so it can never starve a PRIMARY shape. Two successive
    // measured failures shaped this, both worth keeping written down:
    //
    //  1. A flat 256 MiB reservation out of a 512 MiB budget left too
    //     little for the pending shapes, and Ornith's 840 KiB down_exps
    //     pool - which this mechanism deliberately never uses at all -
    //     was never created. Hit rate fell 31.3% -> 23.1%, costing far
    //     more than the 72 MiB the reduced pool saved.
    //  2. Subtracting a per-shape floor (MOE_CACHE_MIN_POOL_SLOTS worth
    //     per still-pending shape) did NOT fix it: this function runs
    //     repeatedly as shapes are discovered, so on the call where only
    //     the last shape is still pending, the floor covers just that one
    //     shape and a 256 MiB reservation still swallowed everything the
    //     already-allocated pools had not taken. down_exps stayed absent,
    //     hit rate fell further (19.5%) and ~236 MiB of the budget simply
    //     went unused.
    //
    // So the reservation is also capped as a FRACTION of the whole budget,
    // which is invariant to how many times this runs and to discovery
    // order. The floor subtraction is kept as well - it is the tighter
    // bound in the single-shape case.
    //
    // TODO(calibration): this fraction, MOE_CACHE_MIN_POOL_SLOTS, and K
    // (GGML_CUDA_MOE_CACHE_NEURON_REDUCE_K) are all fixed numbers standing
    // in for values that should be derived per GPU+model+context by
    // --moe-calibrate, the way -ncmoe and thread count already are. A
    // 12 GB 3060 and a 141 GB card have no business sharing them. The
    // reduced pool's real appetite is knowable (converted experts x
    // reduced expert size) and measurable at calibration time; until then
    // this cap exists to make the failure mode "reduce gets less than it
    // wanted" instead of "a primary pool silently ceases to exist".
    static constexpr size_t MOE_CACHE_REDUCE_RESERVE_MAX_NUM = 1;
    static constexpr size_t MOE_CACHE_REDUCE_RESERVE_MAX_DEN = 8;
    size_t reduce_reserve = 0;
    if (moe_cache_neuron_reduce_enabled()) {
        size_t primary_floor = 0;
        for (const moe_cache_shape & shape : device.shapes) {
            if (!shape.finished && shape.n_tensors > 0 && shape.expert_size > 0) {
                // Matches moe_cache_allocate_pool's own slot_stride
                // computation closely enough to be a floor (the real stride
                // is expert_size rounded up past the guard region, i.e.
                // always >= this) - deliberately an UNDER-estimate of the
                // shape's true need, so this reservation errs toward
                // leaving primary admission MORE room, never less.
                primary_floor += shape.expert_size * MOE_CACHE_MIN_POOL_SLOTS;
            }
        }
        const size_t after_scratch = scratch_reserve < device.budget_limit
            ? device.budget_limit - scratch_reserve : 0;
        const size_t spare = after_scratch > device.allocated_bytes
            ? after_scratch - device.allocated_bytes : 0;
        const size_t claimable = spare > primary_floor ? spare - primary_floor : 0;
        const size_t fraction_cap = device.budget_limit / MOE_CACHE_REDUCE_RESERVE_MAX_DEN
                                                        * MOE_CACHE_REDUCE_RESERVE_MAX_NUM;
        reduce_reserve = std::min(moe_cache_neuron_reduce_budget_bytes(),
                                  std::min(claimable, fraction_cap));
    }
    const size_t total_reserve = scratch_reserve <= SIZE_MAX - reduce_reserve
        ? scratch_reserve + reduce_reserve : SIZE_MAX;
    const size_t slab_limit =
        total_reserve < device.budget_limit
            ? device.budget_limit - total_reserve : 0;
    size_t remaining = slab_limit > device.allocated_bytes
        ? slab_limit - device.allocated_bytes : 0;
    if (remaining == 0) {
        for (moe_cache_shape & shape : device.shapes) {
            if (!shape.finished && shape.n_tensors > 0) {
                shape.finished = true;
            }
        }
        return;
    }

    std::vector<moe_cache_shape *> pending;
    double total_weight = 0.0;
    for (moe_cache_shape & shape : device.shapes) {
        if (!shape.finished && shape.n_tensors > 0) {
            pending.push_back(&shape);
            total_weight +=
                (double)shape.expert_size * (double)shape.n_tensors;
        }
    }
    std::sort(pending.begin(), pending.end(), [](const auto * lhs, const auto * rhs) {
        const double lhs_weight =
            (double)lhs->expert_size * (double)lhs->n_tensors;
        const double rhs_weight =
            (double)rhs->expert_size * (double)rhs->n_tensors;
        return lhs_weight > rhs_weight;
    });

    for (moe_cache_shape * shape : pending) {
        const double weight =
            (double)shape->expert_size * (double)shape->n_tensors;
        const size_t share = total_weight > 0.0
            ? (size_t)((double)remaining * weight / total_weight) : 0;
        const size_t before = device.allocated_bytes;
        moe_cache_allocate_pool(session, device, *shape, share);
        const size_t consumed = device.allocated_bytes - before;
        remaining = consumed <= remaining ? remaining - consumed : 0;
        total_weight -= weight;
    }

    moe_cache_prewarm_from_history(session, device);
}

static int moe_cache_discover_pool(
        moe_cache_session & session, moe_cache_device & device,
        const void * host_base, size_t tensor_size, size_t expert_size,
        int wtype, int64_t n_expert) {
    int pool = moe_cache_find_pool(device, expert_size, wtype);
    if (pool >= 0) {
        return pool;
    }

    moe_cache_shape * shape = nullptr;
    for (moe_cache_shape & candidate : device.shapes) {
        if (candidate.expert_size == expert_size && candidate.wtype == wtype) {
            shape = &candidate;
            break;
        }
    }
    if (!shape) {
        device.shapes.push_back({expert_size, wtype, n_expert, 0, -1, false});
        shape = &device.shapes.back();
    } else {
        shape->n_expert = std::max(shape->n_expert, n_expert);
    }

    const bool first_visit = device.seen_tensors.emplace(
            host_base, moe_cache_seen_tensor{tensor_size, expert_size, wtype}).second;
    if (first_visit) {
        if (shape->n_tensors == 0 && shape->pool < 0) {
            shape->finished = false;
        }
        shape->n_tensors++;
        device.stable_visits = 0;
    } else {
        device.saw_repeat = true;
        device.stable_visits++;
    }

    if (!device.saw_repeat || device.stable_visits < 64) {
        return -1;
    }

    moe_cache_build_pending(session, device);
    return moe_cache_find_pool(device, expert_size, wtype);
}

static bool moe_cache_neuron_heat_enabled();

static void moe_cache_log_stats(moe_cache_device & device) {
    size_t used = 0;
    size_t slots = 0;
    size_t protected_used = 0;
    unsigned long long heat_sum = 0;
    size_t heat_n = 0;
    for (const auto & pool_ptr : device.pools) {
        const moe_cache_pool & pool = *pool_ptr;
        slots += pool.n_slots;
        used += pool.n_slots - pool.free_slots.size();
        for (int i = pool.protected_head; i >= 0; i = pool.slots[i].next) {
            protected_used++;
            heat_sum += pool.slots[i].heat;
            heat_n++;
        }
        for (int i = pool.lru_head; i >= 0; i = pool.slots[i].next) {
            heat_sum += pool.slots[i].heat;
            heat_n++;
        }
    }
    const double avg_heat = heat_n ? (double) heat_sum / (double) heat_n : 0.0;
    const long long total = device.hits + device.misses;
    MOE_CACHE_LOG("[moe-cache] CUDA%d hits=%lld/%lld (%.1f%%) used=%zu/%zu (protected=%zu avg_heat=%.1f) enqueued=%lld filled=%lld fill-fail=%lld evictions=%lld skips=%lld admission=%lld queue=%zu jobs/%zu MiB dispatch-fail=%lld collect-fail=%lld\n",
            device.physical, device.hits, total,
            total ? 100.0 * (double)device.hits / (double)total : 0.0,
            used, slots, protected_used, avg_heat, device.inserts, device.fills, device.fill_failures,
            device.evictions, device.insert_skips,
            device.admission_skips, device.queue.size(), device.queued_bytes >> 20,
            device.dispatch_failures, device.collect_failures);

}

// Full concentration report for device.neuron_heat - deliberately NOT
// called from moe_cache_log_stats(). First cut called it from there,
// gated by stats_every like everything else in that function, and
// measured a second real regression on top of the first: tok/s fell to
// ~2 (from the already-fixed ~48-60 baseline) whenever GGML_CUDA_MOE_CACHE_STATS
// was actually set, because this analysis (a full sort of every key's
// heat vector, up to 256 experts x 2048 floats each) runs SYNCHRONOUSLY
// on the compute thread inside moe_cache_collect() - moving it out of
// neuron_heat_mu (the first fix) only helped the lock-contention case,
// it does nothing for "this function itself is slow and blocks the
// thread that called it". Correct fix: don't run it on the hot path at
// all. Called once, from moe_cache_session_destroy(), off any path that
// affects token generation.
static void moe_cache_neuron_heat_report(moe_cache_device & device) {
    if (!moe_cache_neuron_heat_enabled()) {
        return;
    }
    std::vector<std::pair<moe_cache_key, std::vector<float>>> heat_copy;
    {
        std::lock_guard<std::mutex> lock(device.neuron_heat_mu);
        heat_copy.reserve(device.neuron_heat.size());
        for (const auto & [key, vec] : device.neuron_heat) {
            heat_copy.emplace_back(key, vec);
        }
    }
    if (heat_copy.empty()) {
        return;
    }
    float max_val = 0.0f;
    double sum_val = 0.0;
    size_t n_vals = 0;
    // Concentration, done right this time: a VALUE threshold (relative to
    // this expert's own max), not a neuron-COUNT cutoff. "Top 25% of
    // neurons" picks a fixed rank regardless of whether that 25% is
    // actually meaningful for THIS expert - a value threshold instead
    // asks "how much mass is held by neurons that clear this bar", and
    // lets however many neurons that turns out to be fall out as an
    // observation, never an input. This directly answers "how much
    // weight is concentrated vs spread thin", not "what does an
    // arbitrary rank cutoff capture". Averaged across keys with enough
    // accumulated mass to be meaningful (>0), not weighted by how much
    // traffic each key saw - this answers "is weight concentrated within
    // a typical expert", not "concentrated across the whole
    // traffic-weighted population".
    static constexpr int N_THRESH = 6;
    static constexpr double THRESH_FRAC_OF_MAX[N_THRESH] = {0.50, 0.25, 0.10, 0.05, 0.02, 0.01};
    double mass_frac_sum[N_THRESH] = {0};   // fraction of THIS key's mass held above each threshold
    double count_frac_sum[N_THRESH] = {0};  // fraction of THIS key's neuron count that clears each threshold (observed, not chosen)
    size_t n_keys_ranked = 0;
    std::vector<float> sorted_buf;
    // Per-expert "required width": if this expert could pick its own K
    // freely (unconstrained by mul_mat_id's batching requirement), how
    // many neurons (as % of its own width) would it actually need to
    // reach 90% of its own mass? Reported as a distribution across
    // experts, not a single number - directly answers whether a small
    // number of discrete tiers can approximate free per-expert choice
    // (experts cluster near the same required width) or not (required
    // width varies too much for a handful of tiers to fit well).
    static constexpr double TARGET_COVERAGE = 0.90;
    std::vector<double> required_width_pct;
    std::vector<double> dead_frac_pct;
    for (const auto & [key, vec] : heat_copy) {
        double key_sum = 0.0;
        for (float v : vec) {
            max_val = std::max(max_val, v);
            sum_val += v;
            n_vals++;
            key_sum += v;
        }
        if (key_sum <= 0.0 || vec.size() < 8) {
            continue;
        }
        sorted_buf.assign(vec.begin(), vec.end());
        std::sort(sorted_buf.begin(), sorted_buf.end(), std::greater<float>());
        // Per-key max is sorted_buf[0] (descending sort) - the LIVE
        // reference point for this specific expert's own distribution,
        // not a global constant. This is the "heavy weight decides the
        // bar" framing: a threshold relative to THIS expert's strongest
        // neuron, not an absolute number or a shared-across-experts rank.
        const double key_max = (double) sorted_buf[0];
        // Cumulative sum once, then sample it at each value threshold -
        // O(n log n) for the sort plus O(n) here, not O(n * N_THRESH).
        std::vector<double> cum(sorted_buf.size());
        double running = 0.0;
        for (size_t i = 0; i < sorted_buf.size(); i++) {
            running += sorted_buf[i];
            cum[i] = running;
        }
        if (key_max > 0.0) {
            for (int t = 0; t < N_THRESH; t++) {
                const double value_cutoff = THRESH_FRAC_OF_MAX[t] * key_max;
                // sorted_buf is descending, so the first run of values
                // clearing the cutoff is a prefix - find its end.
                size_t n_above = 0;
                while (n_above < sorted_buf.size() && (double) sorted_buf[n_above] >= value_cutoff) {
                    n_above++;
                }
                const double mass_above = n_above > 0 ? cum[n_above - 1] : 0.0;
                mass_frac_sum[t]  += mass_above / key_sum;
                count_frac_sum[t] += (double) n_above / (double) sorted_buf.size();
            }
        }
        const double target = TARGET_COVERAGE * key_sum;
        size_t n_needed = sorted_buf.size();
        for (size_t i = 0; i < cum.size(); i++) {
            if (cum[i] >= target) {
                n_needed = i + 1;
                break;
            }
        }
        required_width_pct.push_back(100.0 * (double) n_needed / (double) sorted_buf.size());
        // Dead-neuron fraction: relative threshold, 1% of this expert's
        // own MEDIAN (not mean) neuron value - mean is a poor reference
        // when the distribution is heavily skewed (measured max/mean
        // ratios well over 100x elsewhere in this report), since a
        // handful of extreme outliers drag the mean up and make the
        // threshold too lenient. Median is far more robust to exactly
        // that skew. sorted_buf is already sorted descending, so the
        // median is just its middle element. This is a genuinely
        // different question from "required width for 90% coverage"
        // above: that measures how much of the USEFUL mass a reduced
        // tier would keep; this measures how much is contributing
        // essentially NOTHING at all - a true pruning candidate, not
        // just a cache-tier exclusion.
        const double key_median = (double) sorted_buf[sorted_buf.size() / 2];
        const double dead_threshold = 0.01 * key_median;
        size_t n_dead = 0;
        for (float v : sorted_buf) {
            if ((double) v < dead_threshold) {
                n_dead++;
            }
        }
        dead_frac_pct.push_back(100.0 * (double) n_dead / (double) sorted_buf.size());
        n_keys_ranked++;
    }
    std::sort(required_width_pct.begin(), required_width_pct.end());
    std::sort(dead_frac_pct.begin(), dead_frac_pct.end());
    auto pct_at = [](const std::vector<double> & v, double p) -> double {
        if (v.empty()) return 0.0;
        size_t idx = (size_t) (p * (double) (v.size() - 1));
        return v[idx];
    };
    // fprintf, not MOE_CACHE_LOG - the server installs a ggml log callback
    // that drops GGML_LOG_INFO (same reason GGML_CUDA_MOE_CACHE_SUMMARY,
    // right above moe_cache_session_destroy, uses fprintf directly), so
    // this would silently never appear there otherwise.
    fprintf(stderr, "[moe-cache] CUDA%d neuron_heat FINAL REPORT: %zu keys, %zu values, mean=%.4f max=%.4f "
            "(max/mean=%.1fx)\n",
            device.physical, heat_copy.size(), n_vals,
            n_vals ? sum_val / (double) n_vals : 0.0, (double) max_val,
            (n_vals && sum_val > 0.0) ? (double) max_val / (sum_val / (double) n_vals) : 0.0);
    fprintf(stderr, "[moe-cache] CUDA%d neuron_heat concentration by VALUE threshold (n=%zu keys) - "
            "\"clears\" means >= this fraction of THIS expert's own max, count%% is an OUTPUT not a chosen cutoff:\n",
            device.physical, n_keys_ranked);
    for (int t = 0; t < N_THRESH; t++) {
        fprintf(stderr, "[moe-cache]   neurons clearing %.0f%% of own max -> %.1f%% of mass, "
                "%.1f%% of neuron count\n",
                100.0 * THRESH_FRAC_OF_MAX[t],
                n_keys_ranked ? 100.0 * mass_frac_sum[t]  / (double) n_keys_ranked : 0.0,
                n_keys_ranked ? 100.0 * count_frac_sum[t] / (double) n_keys_ranked : 0.0);
    }
    fprintf(stderr, "[moe-cache] CUDA%d per-expert required width for %.0f%% coverage "
            "(n=%zu experts, if each could pick freely): "
            "min=%.0f%% p25=%.0f%% median=%.0f%% p75=%.0f%% p90=%.0f%% max=%.0f%%\n",
            device.physical, 100.0 * TARGET_COVERAGE, required_width_pct.size(),
            pct_at(required_width_pct, 0.0), pct_at(required_width_pct, 0.25),
            pct_at(required_width_pct, 0.5), pct_at(required_width_pct, 0.75),
            pct_at(required_width_pct, 0.9), pct_at(required_width_pct, 1.0));
    fprintf(stderr, "[moe-cache] CUDA%d dead-neuron fraction per expert (< 1%% of that expert's "
            "own MEDIAN (not mean - robust to the skew), true pruning candidates - n=%zu experts): "
            "min=%.0f%% p25=%.0f%% median=%.0f%% p75=%.0f%% p90=%.0f%% max=%.0f%%\n",
            device.physical, dead_frac_pct.size(),
            pct_at(dead_frac_pct, 0.0), pct_at(dead_frac_pct, 0.25),
            pct_at(dead_frac_pct, 0.5), pct_at(dead_frac_pct, 0.75),
            pct_at(dead_frac_pct, 0.9), pct_at(dead_frac_pct, 1.0));
    fflush(stderr);

    // Full per-neuron dump, for offline analysis (per-topic isolation
    // comparison, down_proj-weighted true-contribution reweighting) that
    // needs the raw per-key vectors, not just aggregate percentiles.
    // Plain text, one key per line: tensor_name expert_id n_values v0 v1 ...
    // Tensor name resolved via device.tensor_names (populated at
    // moe_cache_begin() for every tensor this device has ever seen) so
    // the dump is meaningful across separate process runs, unlike the
    // raw host_base pointer which is not stable across restarts.
    const char * dump_path = getenv("GGML_CUDA_MOE_CACHE_NEURON_HEAT_DUMP");
    if (dump_path) {
        FILE * f = fopen(dump_path, "w");
        if (f) {
            for (const auto & [key, vec] : heat_copy) {
                const auto name_it = device.tensor_names.find(key.tensor);
                const std::string name = name_it != device.tensor_names.end()
                    ? name_it->second : "unknown";
                fprintf(f, "%s %d %zu", name.c_str(), key.expert, vec.size());
                for (float v : vec) {
                    fprintf(f, " %.6f", v);
                }
                fprintf(f, "\n");
            }
            fclose(f);
            fprintf(stderr, "[moe-cache] CUDA%d neuron_heat full dump written to %s (%zu keys)\n",
                    device.physical, dump_path, heat_copy.size());
        } else {
            fprintf(stderr, "[moe-cache] CUDA%d neuron_heat dump: failed to open %s\n",
                    device.physical, dump_path);
        }
    }
}

// Defined near the end of the file, alongside the rest of the bandwidth
// profiling machinery it belongs with; forward-declared here since
// moe_cache_session_create is the natural one-shot-per-device trigger point.
static void moe_cache_maybe_profile_bandwidth(moe_cache_device & device);

static void * moe_cache_session_create(void * const * backends, int n_backends) {
    try {
        moe_cache_config config = moe_cache_read_config();
        if (!config.enabled) {
            return nullptr;
        }

        std::unique_ptr<moe_cache_session> session(new (std::nothrow) moe_cache_session());
        if (!session) {
            return nullptr;
        }
        session->config = std::move(config);

        std::unordered_set<int> seen_devices;
        for (int index = 0; index < n_backends &&
                            (int)session->devices.size() < session->config.max_devices; index++) {
            ggml_backend_t backend = (ggml_backend_t)backends[index];
            if (!backend || !ggml_backend_is_cuda(backend)) {
                continue;
            }
            ggml_backend_cuda_context * context =
                (ggml_backend_cuda_context *)backend->context;
            const int physical = context->device;
            if (!seen_devices.insert(physical).second) {
                continue;
            }

            cudaDeviceProp properties;
            ggml_cuda_set_device(physical);
            cudaError_t error = cudaGetDeviceProperties(&properties, physical);
            if (error != cudaSuccess) {
                (void)cudaGetLastError();
                MOE_CACHE_LOG("[moe-cache] CUDA%d skipped: device query failed\n", physical);
                continue;
            }
            const int capability = properties.major * 100 + properties.minor * 10;
            if (capability < session->config.min_compute_capability) {
                MOE_CACHE_LOG("[moe-cache] CUDA%d skipped: compute capability %d.%d is below %d.%d\n",
                        physical, properties.major, properties.minor,
                        session->config.min_compute_capability / 100,
                        (session->config.min_compute_capability % 100) / 10);
                continue;
            }

            session->devices.emplace_back(new moe_cache_device(physical));
            moe_cache_maybe_profile_bandwidth(*session->devices.back());
        }

        // A single eligible device is enough for automatic mode: the live
        // reserve computed in moe_cache_prepare_budget() from actual free
        // VRAM at that moment already accounts for headroom safety on a
        // single card, so there's no longer a reason to require a second
        // device just to try. Previously required devices.size() >= 2 for
        // `automatic`, which meant --moe-cache auto/on silently never
        // engaged on any single-GPU system regardless of free VRAM -
        // measured directly on an RTX 3060 (see
        // docs/moe-cache-colibri-notes.md) and directly relevant to any
        // single-GPU deployment (e.g. one H200).
        if (session->devices.empty()) {
            return nullptr;
        }

        moe_cache_session * result = session.get();
        {
            std::lock_guard<std::mutex> lock(g_registry_mu);
            g_sessions.insert(result);
            g_session_count.fetch_add(1, std::memory_order_release);
        }
        session.release();
        return result;
    } catch (...) {
        MOE_CACHE_LOG("[moe-cache] failed to create cache session\n");
        return nullptr;
    }
}

static void moe_cache_cancel_queue_locked(
        moe_cache_device & device, const void * base, size_t size, bool all) {
    for (auto it = device.queue.begin(); it != device.queue.end();) {
        if (all || moe_cache_ranges_overlap(it->source, it->bytes, base, size)) {
            device.queued_bytes = it->bytes <= device.queued_bytes
                ? device.queued_bytes - it->bytes : 0;
            if (it->pool >= 0 && it->pool < (int)device.pools.size()) {
                moe_cache_pool & pool = *device.pools[it->pool];
                if (it->slot >= 0 && it->slot < pool.n_slots) {
                    moe_cache_slot & slot = pool.slots[it->slot];
                    if (slot.state == moe_cache_slot_state::copying &&
                        slot.generation == it->generation && slot.key == it->key) {
                        moe_cache_slot_reset(pool, it->slot, true);
                    }
                }
            }
            it = device.queue.erase(it);
        } else {
            ++it;
        }
    }
}

static void moe_cache_free_device(moe_cache_device & device) {
    ggml_cuda_set_device(device.physical);
    if (device.compute_stream) {
        cudaStreamSynchronize(device.compute_stream);
    }
    for (auto & pool_ptr : device.pools) {
        if (pool_ptr->slab) {
            cudaFree(pool_ptr->slab);
            pool_ptr->slab = nullptr;
        }
    }
    if (device.d_ids) {
        cudaFree(device.d_ids);
        device.d_ids = nullptr;
    }
    if (device.d_act) {
        cudaFree(device.d_act);
        device.d_act = nullptr;
    }
    if (device.d_act_q8) {
        cudaFree(device.d_act_q8);
        device.d_act_q8 = nullptr;
    }
    if (device.d_out) {
        cudaFree(device.d_out);
        device.d_out = nullptr;
    }
    if (device.h_ids) {
        cudaFreeHost(device.h_ids);
        device.h_ids = nullptr;
    }
    if (device.h_act) {
        cudaFreeHost(device.h_act);
        device.h_act = nullptr;
    }
    if (device.h_out) {
        cudaFreeHost(device.h_out);
        device.h_out = nullptr;
    }
    if (device.d_ids_reduced) {
        cudaFree(device.d_ids_reduced);
        device.d_ids_reduced = nullptr;
    }
    if (device.d_act_q8_reduced) {
        cudaFree(device.d_act_q8_reduced);
        device.d_act_q8_reduced = nullptr;
    }
    if (device.d_out_reduced) {
        cudaFree(device.d_out_reduced);
        device.d_out_reduced = nullptr;
    }
    if (device.d_reduce_row_map) {
        cudaFree(device.d_reduce_row_map);
        device.d_reduce_row_map = nullptr;
    }
    if (device.d_reduce_sel_idx) {
        cudaFree(device.d_reduce_sel_idx);
        device.d_reduce_sel_idx = nullptr;
    }
    if (device.h_ids_reduced) {
        cudaFreeHost(device.h_ids_reduced);
        device.h_ids_reduced = nullptr;
    }
    if (device.h_reduce_row_map) {
        cudaFreeHost(device.h_reduce_row_map);
        device.h_reduce_row_map = nullptr;
    }
    if (device.h_reduce_sel_idx) {
        cudaFreeHost(device.h_reduce_sel_idx);
        device.h_reduce_sel_idx = nullptr;
    }
    if (device.compute_stream) {
        cudaStreamDestroy(device.compute_stream);
        device.compute_stream = nullptr;
    }
    if (device.prefill_stream) {
        cudaStreamSynchronize(device.prefill_stream);
    }
    for (auto & [tensor_bytes, slab] : device.prefill_slabs) {
        (void) tensor_bytes;
        for (int i = 0; i < 2; i++) {
            if (slab.dev[i]) {
                cudaFree(slab.dev[i]);
            }
            if (slab.ready[i]) {
                cudaEventDestroy(slab.ready[i]);
            }
            if (slab.consumed[i]) {
                cudaEventDestroy(slab.consumed[i]);
            }
        }
    }
    device.prefill_slabs.clear();
    device.prefill_successors.clear();
    if (device.prefill_stream) {
        cudaStreamDestroy(device.prefill_stream);
        device.prefill_stream = nullptr;
    }
    device.pools.clear();
    device.allocated_bytes = 0;
    device.d_ids_cap = 0;
    device.d_act_cap = 0;
    device.act_q8_cap = 0;
    device.d_out_cap = 0;
    device.h_ids_cap = 0;
    device.h_act_cap = 0;
    device.h_out_cap = 0;
    device.d_ids_reduced_cap = 0;
    device.act_q8_reduced_cap = 0;
    device.d_out_reduced_cap = 0;
    device.d_reduce_row_map_cap = 0;
    device.d_reduce_sel_idx_cap = 0;
    device.h_ids_reduced_cap = 0;
    device.h_reduce_row_map_cap = 0;
    device.h_reduce_sel_idx_cap = 0;
    device.reduced_indices.clear();
    device.reduce_decided.clear();
}

static void moe_cache_session_destroy(void * opaque) {
    moe_cache_session * session = (moe_cache_session *)opaque;
    if (!session) {
        return;
    }

    // Residency readback. /experts gives this on the server, but a
    // non-server binary (llama-perplexity, llama-bench) had no way to report
    // where the cache actually landed - which made a residency sweep
    // unmeasurable: RESERVE_MB was used as a proxy without any way to confirm
    // it produced a graded ladder, and it did not (one arm allocated nothing
    // at all and its null result was mistaken for a low-residency datapoint).
    // fprintf, not MOE_CACHE_LOG, because the server installs a ggml log
    // callback that drops GGML_LOG_INFO.
    if (getenv("GGML_CUDA_MOE_CACHE_SUMMARY")) {
        std::lock_guard<std::mutex> lock(session->mu);
        for (const auto & device_ptr : session->devices) {
            const moe_cache_device & d = *device_ptr;
            size_t used = 0, total = 0;
            for (const auto & pool_ptr : d.pools) {
                total += (size_t) pool_ptr->n_slots;
                used  += (size_t) pool_ptr->n_slots - pool_ptr->free_slots.size();
            }
            const long long lookups = d.hits + d.misses;
            fprintf(stderr, "[moe-cache] SUMMARY CUDA%d slots=%zu/%zu (%.2f%% of pool) "
                    "allocated=%zu MiB budget=%zu MiB hits=%lld misses=%lld hit_rate=%.4f "
                    "substitutions=%lld declined=%lld evictions=%lld fills_failed=%lld\n",
                    d.physical, used, total,
                    total ? 100.0 * (double) used / (double) total : 0.0,
                    d.allocated_bytes >> 20, d.budget_limit >> 20,
                    d.hits, d.misses,
                    lookups ? (double) d.hits / (double) lookups : 0.0,
                    d.substitutions, d.substitute_declined, d.evictions, d.fill_failures);
            // Per-pool breakdown, in creation order (index 0 is always the
            // first shape the model actually presented - a real pool, not
            // necessarily "the" primary one on a model with several
            // distinct expert byte sizes). Distinguishes an ordinary pool
            // from a heat-aware-reduced one only by comparing its
            // expert_size against every OTHER pool's - a reduced pool is,
            // by construction, always smaller than the full-width pool for
            // the same (wtype, tensor family), see moe_cache_try_reduce_convert.
            for (size_t pi = 0; pi < d.pools.size(); pi++) {
                const moe_cache_pool & p = *d.pools[pi];
                const size_t pused = (size_t) p.n_slots - p.free_slots.size();
                fprintf(stderr, "[moe-cache] SUMMARY CUDA%d pool[%zu] expert_size=%zuKiB "
                        "slots=%zu/%d (%.2f%%) allocated=%zuMiB\n",
                        d.physical, pi, p.expert_size >> 10, pused, p.n_slots,
                        p.n_slots ? 100.0 * (double) pused / (double) p.n_slots : 0.0,
                        ((size_t) p.n_slots * p.slot_stride) >> 20);
            }
            // Heat-aware neuron subsetting: how many experts were DECIDED
            // (device.reduced_indices has an entry - whether or not that
            // decision resulted in an actual resident reduced slot; see the
            // "convert-once" note on moe_cache_neuron_reduce_enabled) versus
            // actually PLACED (pool_index >= 0, meaning a real reduced-pool
            // slot exists for it right now), and the aggregate VRAM this
            // mechanism is saving on the ones actually placed: for each
            // placed key, src_expert_size (the full width it would
            // otherwise cost) minus idx.size()*row_bytes (the reduced width
            // it actually costs).
            if (moe_cache_neuron_reduce_enabled()) {
                size_t decided = 0, placed = 0;
                long long saved_bytes = 0, full_bytes = 0;
                for (const auto & [rkey, entry] : d.reduced_indices) {
                    (void) rkey;
                    decided++;
                    if (entry.pool_index >= 0) {
                        placed++;
                        const size_t reduced_bytes = entry.idx.size() * entry.row_bytes;
                        full_bytes  += (long long) entry.src_expert_size;
                        saved_bytes += (long long) entry.src_expert_size - (long long) reduced_bytes;
                    }
                }
                fprintf(stderr, "[moe-cache] SUMMARY CUDA%d neuron-reduce: decided=%zu placed=%zu "
                        "full_width_equiv=%.2fMiB saved=%.2fMiB (%.1f%% of that width)\n",
                        d.physical, decided, placed,
                        full_bytes / (1024.0 * 1024.0), saved_bytes / (1024.0 * 1024.0),
                        full_bytes > 0 ? 100.0 * (double) saved_bytes / (double) full_bytes : 0.0);
            }
            // Per-layer miss rate: the throughput-relevant "which layer to
            // prefer" signal (see device.layer_hits/layer_misses's own
            // comment) - miss COUNT/RATE, not neuron_heat's concentration
            // stats. Cheap (plain array scan, no sort), printed
            // unconditionally under the same GGML_CUDA_MOE_CACHE_SUMMARY
            // flag rather than gated behind NEURON_HEAT, since it needs
            // none of that machinery.
            for (int l = 0; l < moe_cache_device::MAX_LAYER_STAT; l++) {
                const long long lh = d.layer_hits[l];
                const long long lm = d.layer_misses[l];
                const long long lt = lh + lm;
                if (lt == 0) {
                    continue;
                }
                fprintf(stderr, "[moe-cache] SUMMARY CUDA%d layer=%3d hits=%lld misses=%lld "
                        "miss_rate=%.4f\n",
                        d.physical, l, lh, lm, (double) lm / (double) lt);
            }
        }
    }

    // Off the hot path entirely - see moe_cache_neuron_heat_report's own
    // comment for why it must never be called from anywhere token
    // generation touches. Only fires (and only does real work) when
    // GGML_CUDA_MOE_CACHE_NEURON_HEAT is set; a no-op otherwise.
    {
        std::lock_guard<std::mutex> lock(session->mu);
        for (const auto & device_ptr : session->devices) {
            moe_cache_neuron_heat_report(*device_ptr);
        }
    }

    {
        std::unique_lock<std::mutex> lock(session->mu);
        session->stopping = true;
        for (auto & device_ptr : session->devices) {
            moe_cache_cancel_queue_locked(*device_ptr, nullptr, 0, true);
        }
        session->cv.notify_all();
        session->idle_cv.wait(lock, [&] {
            return session->active_scopes == 0 && session->active_nodes == 0;
        });
    }

    for (auto & device_ptr : session->devices) {
        if (device_ptr->worker_started && device_ptr->worker.joinable()) {
            device_ptr->worker.join();
        }
    }
    {
        std::lock_guard<std::mutex> registry_lock(g_registry_mu);
        if (g_sessions.erase(session) > 0) {
            g_session_count.fetch_sub(1, std::memory_order_release);
        }
    }

    for (auto & device_ptr : session->devices) {
        if (device_ptr->nodes > 0 || device_ptr->dispatch_failures > 0 ||
            device_ptr->collect_failures > 0) {
            moe_cache_log_stats(*device_ptr);
        }
    }

    for (auto & device_ptr : session->devices) {
        std::unique_lock<std::mutex> dispatch_lock(device_ptr->dispatch_mu);
        moe_cache_free_device(*device_ptr);
    }

    // Give back every page-locked expert region. Pinned pages are unreclaimable,
    // so leaking them would shrink usable RAM for the rest of the process's life.
    {
        std::lock_guard<std::mutex> reg_lock(session->hostreg_mu);
        for (const auto & entry : session->hostreg) {
            if (cudaHostUnregister((void *) entry.second.first) != cudaSuccess) {
                (void) cudaGetLastError();
            }
        }
        session->hostreg.clear();
        session->hostreg_bytes = 0;
    }

    delete session;
}

static void moe_cache_session_enter(void * opaque) {
    if (g_session_suppressed > 0) {
        g_session_suppressed++;
        return;
    }

    moe_cache_session * session = (moe_cache_session *)opaque;
    if (!session || session->dormant.load()) {
        if (g_session_stack.empty()) {
            return;
        }
        try {
            g_session_stack.push_back({session, nullptr});
        } catch (...) {
            g_session_suppressed++;
        }
        return;
    }
    std::lock_guard<std::mutex> lock(session->mu);
    if (session->dormant.load()) {
        if (g_session_stack.empty()) {
            return;
        }
        try {
            g_session_stack.push_back({session, nullptr});
        } catch (...) {
            g_session_suppressed++;
        }
        return;
    }
    if (session->stopping) {
        if (g_session_stack.empty()) {
            return;
        }
        try {
            g_session_stack.push_back({session, nullptr});
        } catch (...) {
            g_session_suppressed++;
        }
        return;
    }
    try {
        g_session_stack.push_back({session, session});
    } catch (...) {
        g_session_suppressed++;
        return;
    }
    session->active_scopes++;

    // Track 1 step 4b's token/batch-boundary signal (see the struct comment
    // on moe_cache_device::last_top_expert_valid): this scope wraps exactly
    // one real graph compute, so its start is exactly "a new forward pass
    // is beginning" - reset here means the first MoE layer plan() sees
    // inside this compute can never link back to the previous compute's
    // last layer as a false cross-layer edge.
    for (const auto & device_ptr : session->devices) {
        device_ptr->last_top_expert_valid = false;
        device_ptr->last_top_layer = -1;
    }
}

static void moe_cache_session_leave(void * opaque) {
    if (g_session_suppressed > 0) {
        g_session_suppressed--;
        return;
    }
    moe_cache_session * expected = (moe_cache_session *)opaque;
    auto found = std::find_if(
            g_session_stack.rbegin(), g_session_stack.rend(),
            [expected](const moe_cache_scope_frame & frame) {
                return frame.requested == expected;
            });
    if (found == g_session_stack.rend()) {
        return;
    }
    moe_cache_session * active = found->active;
    g_session_stack.erase(std::next(found).base());
    if (active) {
        std::lock_guard<std::mutex> lock(active->mu);
        if (active->active_scopes > 0) {
            active->active_scopes--;
        }
        active->idle_cv.notify_all();
    }
}

static void * moe_cache_begin(
        const char * name, const void * host_base, size_t expert_size,
        int64_t n_in, int64_t n_out, int wtype, int64_t n_expert, int64_t n_tokens) {
    if (g_session_suppressed > 0 || g_session_stack.empty()) {
        return nullptr;
    }
    moe_cache_session * session = g_session_stack.back().active;
    if (!session || session->stopping || session->dormant || !name || !host_base) {
        return nullptr;
    }
    // Logged once, loudly, rather than silently: this specific rejection
    // used to be indistinguishable from "cache working as intended" from the
    // outside - the session still logs a successful "explicit MoE cache
    // mode" line and never registers a single expert. Found on a real model
    // (Ornith-1.5-35B-A3B, 840 KiB/expert against a then-1 MiB floor) where
    // it cost a long diagnosis with zero pointer from the logs.
    if (name && host_base && strstr(name, "_exps") && n_expert > 0 &&
        expert_size > 0 && expert_size < session->config.min_expert_bytes) {
        static std::atomic<bool> warned{false};
        if (!warned.exchange(true)) {
            MOE_CACHE_LOG("[moe-cache] expert tensor '%s' is %zu KiB/expert, below the "
                    "%zu KiB minimum (GGML_CUDA_MOE_CACHE_MIN_EXPERT_KB) - this model's "
                    "experts will never be cached; lower the minimum if this is expected\n",
                    name, expert_size >> 10, session->config.min_expert_bytes >> 10);
        }
    }
    if (!strstr(name, "_exps") || n_tokens < 1 ||
        n_tokens > session->config.max_batch ||
        expert_size < session->config.min_expert_bytes ||
        n_in <= 0 || n_out <= 0 || n_expert <= 0 ||
        !moe_cache_type_supported((ggml_type)wtype)) {
        return nullptr;
    }

    const size_t row_size = ggml_row_size((ggml_type)wtype, n_in);
    if (row_size == 0 || (uint64_t)n_out > SIZE_MAX / row_size ||
        expert_size != (size_t)n_out * row_size ||
        (uint64_t)n_expert > SIZE_MAX / expert_size ||
        expert_size > SIZE_MAX / 64) {
        return nullptr;
    }
    const size_t tensor_size = (size_t)n_expert * expert_size;
    moe_cache_scratch scratch_requirements;
    if (!moe_cache_scratch_requirements(
            n_in, n_out, session->config.max_batch, scratch_requirements)) {
        return nullptr;
    }

    moe_cache_device * selected = nullptr;
    int pool_index = -1;
    moe_cache_pool * pool = nullptr;
    {
        std::unique_lock<std::mutex> lock(session->mu);
        if (session->stopping) {
            return nullptr;
        }

        int budget_devices = 0;
        int eligible_devices = 0;
        const size_t minimum_pool = expert_size * 64;
        for (const auto & device_ptr : session->devices) {
            moe_cache_device & candidate = *device_ptr;
            if (candidate.dead.load() ||
                !moe_cache_prepare_budget(*session, candidate)) {
                continue;
            }
            budget_devices++;
            const size_t reserved = moe_cache_scratch_total(
                    candidate.scratch_reserve, &scratch_requirements);
            const size_t slab_limit =
                candidate.budget_limit > reserved
                    ? candidate.budget_limit - reserved : 0;
            if (slab_limit >= minimum_pool) {
                eligible_devices++;
            }
        }
        // A single device with a real budget/eligible pool is enough - see
        // the matching note above on why `automatic` no longer requires 2+
        // devices.
        if (budget_devices == 0) {
            session->dormant.store(true);
            lock.unlock();
            for (const auto & device_ptr : session->devices) {
                moe_cache_trim_session(*session, device_ptr->physical);
            }
            return nullptr;
        }
        if (eligible_devices == 0) {
            return nullptr;
        }

        auto route_weight = [&](moe_cache_device & candidate) -> size_t {
            if (candidate.dead.load() || !candidate.budget_ready) {
                return 0;
            }
            const size_t reserved = moe_cache_scratch_total(
                    candidate.scratch_reserve, &scratch_requirements);
            const size_t slab_limit =
                candidate.budget_limit > reserved
                    ? candidate.budget_limit - reserved : 0;
            if (candidate.allocated_bytes > slab_limit) {
                return 0;
            }
            if (moe_cache_find_pool(candidate, expert_size, wtype) >= 0) {
                return slab_limit;
            }
            const size_t available = slab_limit - candidate.allocated_bytes;
            return available >= minimum_pool ? available : 0;
        };

        int layer = -1;
        const bool has_layer = moe_cache_layer_number(name, layer);
        bool tensor_override = !has_layer;

        // Brain-view bookkeeping (see the struct comment) - independent of
        // routing, so record it unconditionally whenever a layer number was
        // parsed, regardless of how the routing decision below turns out.
        if (has_layer) {
            session->tensor_layer.try_emplace(host_base, layer);
        }
        session->n_expert_hint = std::max(session->n_expert_hint, n_expert);
        auto find_routed_device = [&](int physical) -> moe_cache_device * {
            for (const auto & device_ptr : session->devices) {
                if (device_ptr->physical == physical) {
                    return device_ptr.get();
                }
            }
            return nullptr;
        };

        auto tensor_route = session->tensor_devices.find(host_base);
        if (tensor_route != session->tensor_devices.end()) {
            selected = find_routed_device(tensor_route->second);
            if (!selected || selected->dead.load() ||
                route_weight(*selected) == 0) {
                session->tensor_devices.erase(tensor_route);
                selected = nullptr;
            } else {
                tensor_override = true;
            }
        }

        if (!selected && has_layer) {
            auto layer_route = session->layer_devices.find(layer);
            if (layer_route != session->layer_devices.end()) {
                selected = find_routed_device(layer_route->second);
                if (!selected || selected->dead.load()) {
                    session->layer_devices.erase(layer);
                    selected = nullptr;
                } else if (route_weight(*selected) == 0) {
                    selected = nullptr;
                    tensor_override = true;
                }
            }
        }

        uint64_t total_weight = 0;
        if (!selected) {
            for (const auto & device_ptr : session->devices) {
                const size_t weight = route_weight(*device_ptr);
                if (weight > UINT64_MAX - total_weight) {
                    total_weight = UINT64_MAX;
                } else {
                    total_weight += weight;
                }
            }
        }
        if (!selected && total_weight == 0) {
            return nullptr;
        }

        const uint64_t hash = has_layer
            ? ((uint64_t)(uint32_t)layer + 1) * 0x9e3779b97f4a7c15ULL
            : moe_cache_name_hash(name);
        if (!selected) {
            const uint64_t target = hash % total_weight;
            uint64_t cumulative = 0;
            for (const auto & device_ptr : session->devices) {
                moe_cache_device & candidate = *device_ptr;
                const size_t weight = route_weight(candidate);
                if (weight == 0) {
                    continue;
                }
                cumulative = weight > UINT64_MAX - cumulative
                    ? UINT64_MAX : cumulative + weight;
                if (target < cumulative) {
                    selected = &candidate;
                    break;
                }
            }
            if (!selected) {
                return nullptr;
            }
            try {
                if (tensor_override) {
                    session->tensor_devices.emplace(host_base, selected->physical);
                } else {
                    session->layer_devices.emplace(layer, selected->physical);
                }
            } catch (...) {
                return nullptr;
            }
        }

        const size_t selected_scratch = moe_cache_scratch_total(
                selected->scratch_reserve, &scratch_requirements);
        if (selected_scratch >= selected->budget_limit ||
            minimum_pool > selected->budget_limit - selected_scratch) {
            return nullptr;
        }
        selected->scratch_reserve.ids = std::max(
                selected->scratch_reserve.ids, scratch_requirements.ids);
        selected->scratch_reserve.act = std::max(
                selected->scratch_reserve.act, scratch_requirements.act);
        selected->scratch_reserve.q8 = std::max(
                selected->scratch_reserve.q8, scratch_requirements.q8);
        selected->scratch_reserve.out = std::max(
                selected->scratch_reserve.out, scratch_requirements.out);

        try {
            pool_index = moe_cache_discover_pool(
                    *session, *selected, host_base, tensor_size, expert_size, wtype, n_expert);
        } catch (...) {
            MOE_CACHE_LOG("[moe-cache] disabled one node after host allocation failure\n");
            return nullptr;
        }
        if (pool_index < 0 || pool_index >= (int)selected->pools.size() ||
            !selected->pools[pool_index]->slab) {
            return nullptr;
        }
        pool = selected->pools[pool_index].get();
        // Stable identity for cross-restart persistence (predictor weights,
        // see moe_cache_predictor_save/_load): host_base is a raw pointer,
        // a fresh, different address every process restart, so it cannot be
        // the persisted key - the tensor's own name (e.g.
        // "blk.5.ffn_gate_exps.weight") is stable across restarts of the
        // same model and is exactly what's available here.
        selected->tensor_names[host_base] = name;
        moe_cache_session::active_source * source = nullptr;
        try {
            source = &session->active_sources[host_base];
        } catch (...) {
            return nullptr;
        }
        source->bytes = std::max(source->bytes, tensor_size);
        source->references++;
        session->active_nodes++;
        if (getenv("GGML_CUDA_MOE_CACHE_CONCURRENCY_LOG")) {
            static std::atomic<int> hwm{0};
            int cur = session->active_nodes;
            int prev = hwm.load();
            while (cur > prev && !hwm.compare_exchange_weak(prev, cur)) {}
            if (cur > 1) {
                fprintf(stderr, "[moe-cache] CONCURRENT active_nodes=%d hwm=%d tid=%zu\n",
                        cur, hwm.load(),
                        std::hash<std::thread::id>{}(std::this_thread::get_id()));
                fflush(stderr);
            }
        }
    }

    moe_cache_device & device = *selected;
    std::unique_lock<std::mutex> dispatch_lock;
    try {
        dispatch_lock = std::unique_lock<std::mutex>(
                device.dispatch_mu, std::try_to_lock);
    } catch (...) {
        std::lock_guard<std::mutex> lock(session->mu);
        auto source = session->active_sources.find(host_base);
        if (source != session->active_sources.end() && --source->second.references == 0) {
            session->active_sources.erase(source);
        }
        session->active_nodes--;
        session->idle_cv.notify_all();
        return nullptr;
    }
    if (!dispatch_lock.owns_lock() || device.dead.load()) {
        if (getenv("GGML_CUDA_MOE_CACHE_CONCURRENCY_LOG") && !dispatch_lock.owns_lock()) {
            static std::atomic<long long> contended{0};
            fprintf(stderr, "[moe-cache] DISPATCH_MU CONTENDED count=%lld tid=%zu\n",
                    (long long) ++contended,
                    std::hash<std::thread::id>{}(std::this_thread::get_id()));
            fflush(stderr);
        }
        std::lock_guard<std::mutex> lock(session->mu);
        auto source = session->active_sources.find(host_base);
        if (source != session->active_sources.end() && --source->second.references == 0) {
            session->active_sources.erase(source);
        }
        session->active_nodes--;
        session->idle_cv.notify_all();
        return nullptr;
    }

    std::unique_ptr<moe_cache_node> node(new (std::nothrow) moe_cache_node());
    if (!node) {
        std::lock_guard<std::mutex> lock(session->mu);
        auto source = session->active_sources.find(host_base);
        if (source != session->active_sources.end() && --source->second.references == 0) {
            session->active_sources.erase(source);
        }
        session->active_nodes--;
        session->idle_cv.notify_all();
        return nullptr;
    }
    node->session = session;
    node->device = &device;
    node->pool = pool;
    node->pool_index = pool_index;
    node->host_base = host_base;
    node->expert_size = expert_size;
    node->n_in = n_in;
    node->n_out = n_out;
    node->n_expert = n_expert;
    node->n_tokens = n_tokens;
    node->wtype = wtype;
    // llama.cpp names every per-layer tensor "blk.<layer>.<what>.weight"
    // (llm_tn / LLM_TENSOR_NAMES), so the logical layer is already carried
    // by the name this API is handed - no new parameter needed.
    node->layer = -1;
    if (const char * blk = strstr(name, "blk.")) {
        const char * digits = blk + 4;
        if (*digits >= '0' && *digits <= '9') {
            node->layer = atoi(digits);
        }
    }
    // Heat-aware neuron subsetting is explicitly gate_exps/up_exps only
    // (docs/plan.md: down_exps's row axis is hidden_size, an OUTPUT
    // dimension, not an intermediate "neuron" - selecting top-K rows there
    // and zero-filling the rest would silently delete real output
    // dimensions rather than approximate an activation sparsity pattern,
    // which is a correctness bug, not a smaller approximation). Computed
    // once here, from the same `name` string moe_cache_begin already has,
    // rather than re-checked per hit: an ALLOWLIST (only gate_exps/up_exps
    // qualify), not a denylist, so any tensor whose name doesn't match
    // either pattern - including down_exps, and anything not yet
    // anticipated - defaults to ineligible.
    node->reduce_eligible = strstr(name, "gate_exps") != nullptr ||
                             strstr(name, "up_exps") != nullptr;
    node->dispatch_lock = std::move(dispatch_lock);
    return node.release();
}

// Third memory tier, by hint rather than by copy.
//
// MoE weights placed on CPU (-ncmoe / --moe-cache auto) are mmap'd straight
// from the GGUF, so they already live on a VRAM -> RAM -> NVMe ladder: the
// page cache holds what's been touched and the kernel re-faults the rest from
// the file on demand. What was missing is any say in *which* pages get dropped
// when RAM gets tight. llama.cpp issues posix_madvise exactly twice, both at
// load time and both blanket calls over the whole mapping, so the kernel's LRU
// is the only thing deciding - and it can't distinguish an expert that fires
// on most tokens from one that hasn't been selected since startup.
//
// moe-cache already knows the difference: it sees every expert selection. This
// sweep tells the kernel, for experts that are neither VRAM-resident nor
// recently selected, that their pages are reclaim candidates (MADV_COLD -
// deactivate, don't free). Nothing is evicted outright and correctness doesn't
// depend on it; if RAM is plentiful the pages simply stay. The effect only
// shows up under pressure, which is exactly when a dormant expert should lose
// to a hot one instead of to blind recency.
//
// Linux-only (MADV_COLD is 5.4+); a no-op everywhere else, and harmless on an
// older kernel, where madvise just returns EINVAL and the sweep is disabled.
#if defined(__linux__) && defined(MADV_COLD)
#define MOE_CACHE_HAS_MADV_COLD 1
#else
#define MOE_CACHE_HAS_MADV_COLD 0
#endif

// Seconds an expert must go unselected before its pages are advised cold, and
// how often the sweep runs at all. Both deliberately coarse: this is a hint
// whose whole value is being right about "dormant for a long time", and a
// sweep walks every CPU-resident expert, so it should be rare next to decode.
static uint32_t moe_cache_cold_after_s() {
    static const uint32_t value = [] {
        const char * env = getenv("GGML_CUDA_MOE_CACHE_COLD_AFTER_S");
        if (!env) {
            return 120u;
        }
        const long parsed = strtol(env, nullptr, 10);
        return parsed < 0 ? 0u : (uint32_t) parsed; // 0 disables
    }();
    return value;
}

static void moe_cache_cold_sweep(moe_cache_device & device, std::chrono::steady_clock::time_point now) {
#if MOE_CACHE_HAS_MADV_COLD
    static const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) {
        return;
    }

    // Two independent halves, deliberately not sharing a gate. Advising cold
    // pages needs an expert to have been unused for a while, so it waits on the
    // dormancy threshold. Pinning hot ones does not: which experts are hot is
    // known from the first tokens, and under memory pressure the pinning is
    // wanted immediately - gating it behind a 120s dormancy timer meant it never
    // ran in short-lived or heavily-loaded sessions, which are exactly the ones
    // that need it.
    const uint32_t cold_after = moe_cache_cold_after_s();
    const uint32_t age_now = (uint32_t) std::chrono::duration_cast<std::chrono::seconds>(
            now - device.residency_epoch).count();
    const uint64_t cold_epochs = moe_cache_cold_after_epochs();
    // Routing-relative when available (the useful signal), wall-clock only as a
    // fallback if epoch classification is disabled.
    const bool do_cold = cold_epochs > 0
        ? device.plan_epoch > cold_epochs
        : (cold_after > 0 && age_now >= cold_after);

    // Heat decay on the same schedule as the pools', so the RAM tier ages its
    // signal exactly as the VRAM tier does, then take the cut point. Demoting
    // the coldest half leaves the hot set standing without needing a "protect"
    // hint the kernel does not offer.
    uint32_t heat_cut = 0;
    if (cold_epochs > 0) {
        std::vector<uint32_t> heats;
        for (auto & [host_base_d, res_d] : device.residency) {
            for (uint32_t & h : res_d.selections) {
                h >>= 1;
                heats.push_back(h);
            }
        }
        if (!heats.empty()) {
            std::sort(heats.begin(), heats.end());
            heat_cut = heats[heats.size() / 2]; // median: demote the colder half
        }
    }

    size_t advised_experts = 0;
    size_t advised_bytes = 0;
    for (auto & [host_base, res] : device.residency) {
        if (!do_cold) break;
        for (size_t expert = 0; expert < res.last_seen.size(); expert++) {
            if (res.is_cold[expert]) {
                continue; // already advised; a later hit clears this
            }
            // Demote by rank, not by a threshold: anything at or below the heat
            // cut computed for this sweep. Same LFRU signal and the same
            // "coldest pays" rule the VRAM tier uses, applied one tier down.
            if (cold_epochs > 0) {
                if (res.selections[expert] > heat_cut) {
                    continue;
                }
            } else if (res.last_seen[expert] != 0 && age_now - res.last_seen[expert] < cold_after) {
                continue;
            }
            // Skip anything currently held in VRAM: its host pages back a live
            // cache entry and re-reading them is exactly what a refill does.
            const moe_cache_key key{host_base, (int32_t) expert};
            bool resident = false;
            for (const auto & pool_ptr : device.pools) {
                if (pool_ptr->map.find(key) != pool_ptr->map.end()) {
                    resident = true;
                    break;
                }
            }
            if (resident) {
                continue;
            }

            // Align outward to page boundaries, but only advise whole pages
            // that lie strictly inside this expert - a partial page at either
            // end is shared with a neighbouring expert that may well be hot.
            const uintptr_t begin = (uintptr_t) host_base + (uintptr_t) expert * res.expert_size;
            const uintptr_t end   = begin + res.expert_size;
            const uintptr_t first = (begin + page_size - 1) & ~((uintptr_t) page_size - 1);
            const uintptr_t last  = end & ~((uintptr_t) page_size - 1);
            if (last <= first) {
                continue; // expert smaller than a page, or straddling one
            }
            // MADV_COLD only moves pages to the inactive list and hopes the
            // kernel gets round to them; MADV_PAGEOUT reclaims immediately,
            // which is what actually frees room for the hot set under pressure.
            // Prefer it, falling back when the kernel is too old to support it.
            // Hard eviction versus soft preference is a real choice, not an
            // optimisation detail. MADV_PAGEOUT reclaims the page immediately -
            // decisive, but if the classification is wrong the re-read is paid
            // unconditionally. MADV_COLD only moves the page to the front of the
            // eviction queue: it survives when there is no pressure and goes
            // first when there is, so a misclassification costs nothing unless
            // memory is actually short. Default is the soft form; set
            // GGML_CUDA_MOE_CACHE_DEMOTE=pageout for the hard one.
            static const bool hard_demote = [] {
                const char * env = getenv("GGML_CUDA_MOE_CACHE_DEMOTE");
                return env && strcmp(env, "pageout") == 0;
            }();
            int adv_rc = -1;
#ifdef MADV_PAGEOUT
            static bool pageout_ok = true;
            if (hard_demote && pageout_ok) {
                adv_rc = madvise((void *) first, (size_t) (last - first), MADV_PAGEOUT);
                if (adv_rc != 0 && errno == EINVAL) {
                    pageout_ok = false; // not supported here - use MADV_COLD from now on
                }
            }
#endif
            if (adv_rc != 0) {
                adv_rc = madvise((void *) first, (size_t) (last - first), MADV_COLD);
            }
            if (adv_rc != 0) {
                if (errno == EINVAL) {
                    // kernel without MADV_COLD - stop trying for this process
                    MOE_CACHE_LOG("%s", "[moe-cache] MADV_COLD unsupported on this kernel, cold-page hinting disabled\n");
                    setenv("GGML_CUDA_MOE_CACHE_COLD_AFTER_S", "0", 1);
                    return;
                }
                continue; // transient (e.g. ENOMEM on an unmapped hole) - skip
            }
            res.is_cold[expert] = true;
            advised_experts++;
            advised_bytes += (size_t) (last - first);
        }
    }

    if (advised_experts > 0) {
        MOE_CACHE_LOG("[moe-cache] CUDA%d advised %zu dormant CPU expert(s) (%zu MiB) as reclaimable\n",
                device.physical, advised_experts, advised_bytes >> 20);
    }

    // Second half of the same policy: having told the kernel what it may drop,
    // hold down what it must not. Rank every tracked expert by how often it has
    // actually been selected and pin the top of that ranking with mlock, up to
    // the budget. This is the part MADV_COLD cannot do - an advisory hint lowers
    // priority, it never protects.
    const size_t pin_budget = moe_cache_pin_budget_bytes();
    if (pin_budget == 0) {
        return;
    }

    struct pin_candidate {
        uint32_t selections;
        const void * host_base;
        size_t expert;
    };
    std::vector<pin_candidate> candidates;
    for (auto & [host_base, res] : device.residency) {
        for (size_t expert = 0; expert < res.selections.size(); expert++) {
            if (res.selections[expert] > 0) {
                candidates.push_back({res.selections[expert], host_base, expert});
            }
        }
    }
    // Descending by selection count; ties broken by address so the ordering is
    // stable across sweeps and an expert doesn't churn in and out of the pinned
    // set for no reason.
    std::sort(candidates.begin(), candidates.end(), [](const pin_candidate & a, const pin_candidate & b) {
        if (a.selections != b.selections) {
            return a.selections > b.selections;
        }
        if (a.host_base != b.host_base) {
            return a.host_base < b.host_base;
        }
        return a.expert < b.expert;
    });

    std::unordered_set<const void *> keep; // packed (base, expert) identities to retain
    keep.reserve(candidates.size());
    size_t want_bytes = 0;
    size_t want_count = 0;
    for (const pin_candidate & c : candidates) {
        auto & res = device.residency[c.host_base];
        const uintptr_t begin = (uintptr_t) c.host_base + (uintptr_t) c.expert * res.expert_size;
        const uintptr_t end   = begin + res.expert_size;
        const uintptr_t first = (begin + page_size - 1) & ~((uintptr_t) page_size - 1);
        const uintptr_t last  = end & ~((uintptr_t) page_size - 1);
        if (last <= first) {
            continue;
        }
        const size_t len = (size_t) (last - first);
        if (want_bytes + len > pin_budget) {
            break; // ranking is descending, so everything after this is colder
        }
        want_bytes += len;
        want_count++;
        keep.insert((const void *) first);

        if (res.is_pinned[c.expert]) {
            continue; // already held
        }
        if (mlock((void *) first, len) != 0) {
            // ENOMEM (RLIMIT_MEMLOCK or cgroup) or EPERM: stop trying this
            // sweep rather than hammering the syscall for every candidate.
            // Not fatal - unpinned simply means "same behaviour as before".
            MOE_CACHE_LOG("[moe-cache] CUDA%d could not pin hot experts (%s) - continuing without pinning\n",
                    device.physical, strerror(errno));
            break;
        }
        res.is_pinned[c.expert] = true;
        device.pinned_bytes += len;
    }

    // Release anything pinned that no longer makes the cut, so the pinned set
    // tracks the workload instead of only ever growing.
    size_t released = 0;
    for (auto & [host_base, res] : device.residency) {
        for (size_t expert = 0; expert < res.is_pinned.size(); expert++) {
            if (!res.is_pinned[expert]) {
                continue;
            }
            const uintptr_t begin = (uintptr_t) host_base + (uintptr_t) expert * res.expert_size;
            const uintptr_t end   = begin + res.expert_size;
            const uintptr_t first = (begin + page_size - 1) & ~((uintptr_t) page_size - 1);
            const uintptr_t last  = end & ~((uintptr_t) page_size - 1);
            if (last <= first || keep.count((const void *) first)) {
                continue;
            }
            munlock((void *) first, (size_t) (last - first));
            res.is_pinned[expert] = false;
            device.pinned_bytes -= std::min(device.pinned_bytes, (size_t) (last - first));
            released++;
        }
    }

    static size_t last_reported = SIZE_MAX;
    if (want_count > 0 && device.pinned_bytes != last_reported) {
        last_reported = device.pinned_bytes;
        MOE_CACHE_LOG("[moe-cache] CUDA%d pinned %zu hot CPU expert(s) (%zu MiB of %zu MiB budget, %zu released)\n",
                device.physical, want_count, device.pinned_bytes >> 20, pin_budget >> 20, released);
    }
#else
    (void) device; (void) now;
#endif
}

// Track 1 step 3 (docs/plan.md): Atlas-driven cache warming's actual
// warming action. Off by default - this is the first thing in the Atlas
// track that touches real cache state, and it hasn't been A/B'd yet, so it
// follows the same rule every other predictive mechanism here was held to
// before earning trust: prefetch-only, free-slot-only, NEVER evicts. A
// wrong topic read costs one wasted free-slot fill, same ceiling a wrong
// router-lookahead guess already has - never a displaced, already-earned
// resident expert.
// Step 6: opt-in measurement only. Off by default, checked once.
static bool moe_cache_measure_pred_enabled() {
    static const bool on = [] {
        const char * env = getenv("GGML_CUDA_MOE_CACHE_MEASURE_PRED");
        return env && atoi(env) != 0;
    }();
    return on;
}

// Step 7a follow-on: group-aware admission. Off by default.
static bool moe_cache_group_admit_enabled() {
    static const bool on = [] {
        const char * env = getenv("GGML_CUDA_MOE_CACHE_GROUP_ADMIT");
        return env && atoi(env) != 0;
    }();
    return on;
}

// partner_best has two consumers now - the 7a measurement and group
// admission - so it must be maintained if either is on.
static bool moe_cache_partner_index_enabled() {
    return moe_cache_measure_pred_enabled() || moe_cache_group_admit_enabled();
}

// CHANGES MODEL OUTPUT. Off by default and deliberately not folded into any
// other flag: every other knob in this cache is output-identical to mainline,
// and this one is not. When a router pick is not resident, run the best
// resident expert of the same tensor instead of paying a CPU fallback.
//
// Why this is even defensible: the picks that miss are the ones the router
// weights least (measured - rank 7 misses 44% of the time against rank 0's
// 20%, and ranks 4-7 carry only 32% of the gate mass), so the substituted
// weight is small. Whether the output damage is acceptable is a question for
// perplexity, which has NOT been run yet.
// DEFAULT ON. Measured +8.11% tok/s, winning 4 of 4 rounds on Ornith-1.5-35B,
// with hit rate unchanged (it cannot move that metric by construction - it
// serves a resident stand-in instead of paying CPU fallback, rather than
// making the miss into a hit). Rank-gated by moe_cache_substitute_min_rank()
// so the router's most confident picks always get exact compute.
// GGML_CUDA_MOE_CACHE_SUBSTITUTE=0 turns it off.
static bool moe_cache_substitute_enabled() {
    static const bool enabled = [] {
        const char * env = getenv("GGML_CUDA_MOE_CACHE_SUBSTITUTE");
        return env ? atoi(env) != 0 : true;
    }();
    return enabled;
}

// Purely observational per-neuron heat accumulation (device.neuron_heat),
// off by default. See that field's comment for what it measures and why
// this costs no new sync/kernel. Off by default because it is not read by
// anything yet - this is step one (measure) of a two-step build, not the
// full mechanism.
// DEFAULT ON. Purely observational (accumulates |value| per neuron over data
// moe_cache_collect has already downloaded and validated - no new kernel, no
// new sync) and it is the signal reduced-width serving selects on, so it has
// to be collected for that to work at all. Its own cost was measured and the
// two regressions it caused were fixed (a dedicated mutex rather than
// session.mu; the expensive report moved off the hot path entirely).
// GGML_CUDA_MOE_CACHE_NEURON_HEAT=0 turns it off.
static bool moe_cache_neuron_heat_enabled() {
    static const bool enabled = [] {
        const char * env = getenv("GGML_CUDA_MOE_CACHE_NEURON_HEAT");
        return env ? atoi(env) != 0 : true;
    }();
    return enabled;
}

// Substitution gate (2026-09-04, revised three times same day - this is the
// settled version). The rank floor a miss must clear to accept a hot-loaded
// stand-in is a DYNAMIC variable driven by real, currently-measured backlog,
// not a fixed cutoff and not "always substitute":
//
//   - Unconditional (min_rank forced to 0, every rank substitutable) was
//     tried live: fast (13.25 tok/s) but broke output - the reasoning
//     channel got stuck in a verbatim repetition loop. Confirms the
//     router's most-confident picks genuinely need close-to-exact compute;
//     substituting them at volume corrupts coherence, regardless of how
//     good the stand-in-selection method is.
//   - A static cutoff (MIN_RANK=4) is reliable but leaves throughput on the
//     table when the fetch pipeline has real slack (PPL cost was real:
//     3.54 -> 5.62, non-overlapping CI, even gated).
//   - What's actually wanted: exact compute when the pipeline is keeping
//     up, substitution opening up (starting with the router's
//     least-confident picks, same as the static cutoff's floor) only when
//     it's genuinely falling behind. That requires a REAL backlog signal.
//
// moe_cache_track_fetch_pace supplies that signal: real GPU-side fetch
// completion latency (issue to actual finish, via a stream callback - see
// its own comment), not host-side dispatch-call gaps, which an earlier
// version of this gate used and which stayed small even at ~1 tok/s live
// because cudaMemcpyAsync never blocks the host between enqueueing copies
// (substitutions=279 of 32,073 misses on a real 8099 session - the gate
// barely opened despite the pipeline being nowhere near keeping up).
//
// The "falling behind" threshold is itself not a hardcoded latency figure:
// device.substitute_pace_ema_ms (fast) is compared against
// device.substitute_pace_baseline_ms (slow) - this system's own recently-
// typical pace, self-calibrated from real observations on whatever
// hardware/model is actually running, not a guessed millisecond constant.
// See the struct comment on those fields and moe_cache_fetch_pace_cb for
// how both are maintained.
//
// GGML_CUDA_MOE_CACHE_SUBSTITUTE_MIN_RANK, if the user sets it, still wins
// outright - same "never override an explicit choice" rule as everywhere
// else in this file.
static int moe_cache_substitute_min_rank(moe_cache_device & device) {
    const char * pinned_env = getenv("GGML_CUDA_MOE_CACHE_SUBSTITUTE_MIN_RANK");
    if (pinned_env) {
        const int n = atoi(pinned_env);
        return n < 0 ? 0 : n;
    }

    // How far above this system's own recent-normal pace counts as "falling
    // behind" - a dimensionless ratio, not an absolute latency, so it scales
    // automatically with whatever hardware/model is actually running rather
    // than encoding a guessed millisecond figure. Override via
    // GGML_CUDA_MOE_CACHE_SUBSTITUTE_OVERRUN_RATIO; 1.5 (50% slower than
    // typical) is the starting point pending real calibration - still a
    // choice, but a scale-free one, unlike an absolute-ms target.
    static const double overrun_ratio = [] {
        const char * e = getenv("GGML_CUDA_MOE_CACHE_SUBSTITUTE_OVERRUN_RATIO");
        const double v = e ? atof(e) : 1.5;
        return v > 1.0 ? v : 1.5;
    }();
    static const uint32_t min_samples = [] {
        const char * e = getenv("GGML_CUDA_MOE_CACHE_SUBSTITUTE_PACE_MIN_SAMPLES");
        const int v = e ? atoi(e) : 32;
        return (uint32_t) (v > 0 ? v : 32);
    }();

    double ema, baseline;
    uint32_t samples;
    {
        std::lock_guard<std::mutex> lock(device.substitute_pace_mu);
        ema      = device.substitute_pace_ema_ms;
        baseline = device.substitute_pace_baseline_ms;
        samples  = device.substitute_pace_samples;
    }

    if (samples < min_samples || baseline <= 0.0) {
        return 10; // baseline not warmed up yet - no signal to act on, stay exact
    }
    const double target = baseline * overrun_ratio;
    if (ema <= target) {
        return 10; // keeping pace relative to this system's own normal - never substitute
    }
    // Falling behind this system's own baseline: ramp the floor down as the
    // overrun grows, so a small overrun costs only the router's
    // least-confident picks (the same rank floor the static cutoff used)
    // and a large one opens substitution up further - but real, self-
    // measured backlog earns that, a fixed policy doesn't.
    const double overrun = ema / target; // > 1 here
    const int rank = (int) (10.0 / overrun);
    return rank < 0 ? 0 : (rank > 10 ? 10 : rank);
}

// How many resident candidates the stand-in search may examine. The offline
// cost model put the useful window near 64; beyond that the scan cost on the
// dispatch path outweighs what a better stand-in is worth.
static int moe_cache_substitute_scan() {
    static const int n = [] {
        const char * env = getenv("GGML_CUDA_MOE_CACHE_SUBSTITUTE_SCAN");
        const int v = env ? atoi(env) : 64;
        return v < 1 ? 1 : (v > 256 ? 256 : v);
    }();
    return n;
}

// Minimum co-activation count before a resident expert is accepted as a
// stand-in. 0 would let an arbitrary resident expert stand in for one it has
// never fired alongside, which is the case most likely to damage output.
static uint32_t moe_cache_substitute_min_coact() {
    static const uint32_t n = [] {
        const char * env = getenv("GGML_CUDA_MOE_CACHE_SUBSTITUTE_MIN_COACT");
        const int v = env ? atoi(env) : 1;
        return (uint32_t) (v < 0 ? 0 : v);
    }();
    return n;
}

// Sampled integrity check: does the slot actually hold the expert it claims?
// Set GGML_CUDA_MOE_CACHE_VERIFY_SLOTS=N to check one hit in every N. Reads
// three windows (head, middle, tail) rather than one, because a partial copy
// leaves the head correct and only diverges later - checking the first bytes
// alone would report everything healthy.
//
// This is a debugging tool, not a guard: it holds the session lock across a
// synchronous D2H copy, so any N small enough to be useful is far too slow
// for production.
static int moe_cache_verify_slots_rate() {
    static const int n = [] {
        const char * env = getenv("GGML_CUDA_MOE_CACHE_VERIFY_SLOTS");
        return env ? atoi(env) : 0;
    }();
    return n;
}

static void moe_cache_verify_slot(const moe_cache_pool & pool, int slot_index,
                                  const moe_cache_key & key, int device_id) {
    const int rate = moe_cache_verify_slots_rate();
    if (rate <= 0 || !key.tensor || pool.expert_size == 0 || !pool.slab) {
        return;
    }
    static std::atomic<long long> seen{0};
    if ((seen.fetch_add(1, std::memory_order_relaxed) % rate) != 0) {
        return;
    }
    const size_t win = std::min<size_t>(2048, pool.expert_size);
    const size_t offsets[3] = { 0, (pool.expert_size / 2) & ~size_t(15),
                                pool.expert_size > win ? pool.expert_size - win : 0 };
    const char * host = (const char *) key.tensor + (size_t) key.expert * pool.expert_size;
    const char * dev  = pool.slab + (size_t) slot_index * pool.slot_stride;
    std::vector<char> buf(win);
    ggml_cuda_set_device(device_id);
    for (int w = 0; w < 3; w++) {
        if (cudaMemcpy(buf.data(), dev + offsets[w], win, cudaMemcpyDeviceToHost) != cudaSuccess) {
            continue;
        }
        if (memcmp(buf.data(), host + offsets[w], win) != 0) {
            size_t first = 0;
            while (first < win && buf[first] == host[offsets[w] + first]) first++;
            static std::atomic<int> reported{0};
            if (reported.fetch_add(1, std::memory_order_relaxed) < 20) {
                fprintf(stderr, "[moe-cache] SLOT MISMATCH slot=%d expert=%d tensor=%p "
                        "expert_size=%zu window@%zu first_diff=+%zu dev=%02x%02x%02x%02x "
                        "host=%02x%02x%02x%02x\n",
                        slot_index, key.expert, key.tensor, pool.expert_size,
                        offsets[w], first,
                        (unsigned char)buf[first], (unsigned char)buf[first+1 < win ? first+1 : first],
                        (unsigned char)buf[first+2 < win ? first+2 : first],
                        (unsigned char)buf[first+3 < win ? first+3 : first],
                        (unsigned char)host[offsets[w]+first],
                        (unsigned char)host[offsets[w]+first+1],
                        (unsigned char)host[offsets[w]+first+2],
                        (unsigned char)host[offsets[w]+first+3]);
            }
            return;
        }
    }
}

static void moe_cache_verify_rows(void * opaque, int n_hits, const int32_t * slot_idx,
                                  const int32_t * experts) {
    static const bool on = [] {
        const char * env = getenv("GGML_CUDA_MOE_CACHE_VERIFY_ROWS");
        return env && atoi(env) != 0;
    }();
    if (!on || !opaque || !slot_idx || !experts) {
        return;
    }
    moe_cache_node * node = (moe_cache_node *) opaque;
    if (!node->pool) {
        return;
    }
    moe_cache_pool & pool = *node->pool;
    std::lock_guard<std::mutex> lock(node->session->mu);
    static std::atomic<int> reported{0};
    for (int i = 0; i < n_hits; i++) {
        const int si = slot_idx[i];
        if (si < 0 || si >= (int) pool.slots.size()) {
            if (reported.fetch_add(1) < 20) {
                fprintf(stderr, "[moe-cache] ROW BAD SLOT i=%d slot=%d n_slots=%zu\n",
                        i, si, pool.slots.size());
            }
            continue;
        }
        const moe_cache_slot & slot = pool.slots[si];
        if (slot.key.expert != experts[i] || slot.key.tensor != node->host_base ||
            slot.state != moe_cache_slot_state::valid) {
            if (reported.fetch_add(1) < 20) {
                fprintf(stderr, "[moe-cache] ROW MISMATCH i=%d/%d slot=%d "
                        "slot_expert=%d wanted_expert=%d slot_tensor=%p node_tensor=%p state=%d\n",
                        i, n_hits, si, slot.key.expert, experts[i],
                        slot.key.tensor, node->host_base, (int) slot.state);
            }
        }
    }
}

static bool moe_cache_mask_audit_enabled() {
    static const bool enabled = [] {
        const char * env = getenv("GGML_CUDA_MOE_CACHE_MASK_AUDIT");
        return env && atoi(env) != 0;
    }();
    return enabled;
}

static bool moe_cache_atlas_warm_enabled() {
    static const bool enabled = [] {
        const char * env = getenv("GGML_CUDA_MOE_CACHE_ATLAS_WARM");
        return env && atoi(env) != 0;
    }();
    return enabled;
}

// Track 1.5 (docs/plan.md): atlas warming has two halves with very different
// costs. The ranking pass below is a cosine-similarity scan over every
// atlas-covered expert of one tensor (a sqrt and a few flops per candidate,
// up to n_expert of them) and touches no cache state at all; the admission
// half that follows is a handful of hash lookups and a queue push. Only the
// second half has to happen under the session lock, and neither half has to
// happen on the thread that is about to issue a GPU op. Splitting them lets
// the expensive one run on the fill worker while the GPU is busy with the
// previous layer, leaving the dispatch path with just an O(1) enqueue - the
// CPU/GPU split FreeToken's architecture keeps and this port never carried
// over. Both halves are unchanged in what they compute; only where they run
// is configurable (GGML_CUDA_MOE_CACHE_ATLAS_WARM_ASYNC).
struct moe_cache_atlas_cand { int32_t expert; float score; };

// How many candidates the ranking pass keeps, and how many of those may
// actually be admitted. These differ now, which is the one real behavior
// change in this split: the original scan skipped already-resident experts
// inline, which needs pool.map and therefore the lock. The lock-free pass
// cannot, so it keeps a deeper list and lets the admission half - which
// holds the lock and already re-checked residency anyway - walk down it,
// admitting the same top-2 non-resident candidates in the same score order.
// Only when 3+ of the top 4 are already resident do the two differ, and
// then only by not looking further down the list than the old code would
// have. Noted rather than hidden: the topic-switch result in docs/plan.md
// was measured with the old scan and would need re-validating if this ever
// goes on by default.
// Array bounds only now (see moe_cache_atlas_warm_rank_k/admit_k below for
// the actual runtime values, which are <= these). Widened from the original
// 4/2 after the multi-round topic-switch benchmark (docs/plan.md) showed
// that budget cannot move enough of the cache per firing to matter: after a
// real topic switch, reactive demand-driven admission (immediate, no
// throttle) finishes reshaping the cache before a 2-candidate warming pass
// makes a dent, so the "chance-level map performs identically to a real
// map" finding was budget-starved, not evidence the ranking signal itself
// is worthless.
static constexpr int MOE_CACHE_ATLAS_RANK_K  = 32;
static constexpr int MOE_CACHE_ATLAS_ADMIT_K = 16;

// The expensive half. Pure: reads only the (immutable, shared_ptr-held)
// atlas row and the request direction snapshot it is handed, writes only
// best[]. No session lock required, and none taken.
// resident_pool is the one concession to re-validating the pre-Track-1.5
// result: when non-null the scan filters already-resident experts inline,
// exactly as the old code did, which means the caller must hold session.mu.
// Only the legacy comparison arm passes it - see
// moe_cache_atlas_warm_legacy_rank_enabled.
static int moe_cache_atlas_rank(
        const moe_cache_device::atlas_row & candidates,
        float rx, float ry, int64_t n_expert,
        moe_cache_atlas_cand * best, int k,
        const moe_cache_pool * resident_pool = nullptr,
        const void * host_base = nullptr) {
    const float rmag = std::sqrt(rx * rx + ry * ry);
    if (rmag < 1e-6f || k <= 0) {
        return 0; // decaying centroid hasn't moved off the origin yet - nothing to rank against
    }
    int n_best = 0;
    for (const auto & [expert, cell] : candidates) {
        // Atlas data can be stale (a different model/shape than what's
        // actually loaded, or an architecture change since the offline
        // probe ran) or, in this session's own testing, was deliberately
        // registered from a different model's atlas to exercise this path -
        // in both cases an out-of-range expert index here is a real,
        // previously-unguarded out-of-bounds read: host_base + expert *
        // expert_size would point past this tensor's real allocation, and
        // the fill worker's copy from that address segfaults. Every other
        // caller (moe_cache_plan, moe_cache_prefetch) bounds-checks against
        // n_expert before trusting an index; this one has to too, since its
        // indices come from an external file, not the model's own router.
        if (expert < 0 || expert >= n_expert) {
            continue;
        }
        if (resident_pool) {
            const moe_cache_key key{host_base, expert};
            if (resident_pool->map.find(key) != resident_pool->map.end()) {
                continue; // already resident or already in flight (legacy arm only)
            }
        }
        const float cmag = std::sqrt(cell.x * cell.x + cell.y * cell.y);
        if (cmag < 1e-6f) {
            continue; // this expert never showed real affinity to any probed category
        }
        // Cosine similarity to the live request direction, discounted by
        // the candidate's own specialization confidence (a high-cosine
        // match to a barely-specialized expert is a much weaker signal than
        // the same cosine to a sharply-specialized one).
        const float score = ((cell.x * rx + cell.y * ry) / (cmag * rmag)) * cell.spec;
        if (score <= 0.0f) {
            continue; // only warm candidates genuinely aligned with the live direction
        }
        // Bounded top-K, no allocation.
        if (n_best < k) {
            best[n_best++] = {expert, score};
            std::sort(best, best + n_best, [](const moe_cache_atlas_cand & a, const moe_cache_atlas_cand & b) { return a.score > b.score; });
        } else if (score > best[k - 1].score) {
            best[k - 1] = {expert, score};
            std::sort(best, best + k, [](const moe_cache_atlas_cand & a, const moe_cache_atlas_cand & b) { return a.score > b.score; });
        }
    }
    return n_best;
}

// The cheap half. Must hold session.mu: touches pool.map, the slot table,
// the LRU list and the fill queue. Returns whether anything was queued, so
// the caller can fold it into its own worker wake.
// Live counterpart to GGML_CUDA_MOE_CACHE_READAHEAD (see its comment ~5380):
// that one only fires once, at session warm-start, from persisted cross-run
// history. This fires on every admission this function ever makes - atlas
// scan and predictor warming alike - for a candidate device.residency has
// already flagged cold (POSIX_MADV_DONTNEED'd by the cold sweep, or never
// resident this run). moe_cache_atlas_admit only queues the copy (see
// device.queue.push_back below); the actual host read happens later, on the
// fill worker, so a WILLNEED issued here has a real window to complete
// before that read arrives instead of racing it.
//
// On by default, unlike the one-shot version - and this is a measured
// decision, not an assumption that "more prefetch" generalizes from that
// other result. A/B'd 5 interleaved rounds each way (Gemma-4-26B-A4B,
// -ncmoe 20, under a real 15 GiB cgroup memory cap to force genuine cold
// misses): off = {34.86, 20.11, 16.81, 36.66, 37.07} tok/s (mean 29.1,
// wide spread with real stalls down to ~17); on = {40.50, 33.83, 40.30,
// 42.46, 41.90} (mean 39.8, tight). +37% on the mean, and - the more telling
// part - on eliminates the worst-case stalls rather than just raising the
// average, exactly the failure mode this is meant to fix (a synchronous
// cold-page read on the admit path, now hidden behind compute instead).
// Unlike the one-shot version - which blindly prefetches whatever cross-run
// history says was hot, whether or not anything currently wants it - this
// only fires for a candidate the live predictor is already about to admit
// AND that device.residency has already flagged cold, so it does not carry
// the "page-cache pressure from stuff nothing asked for" cost that made the
// one-shot version a net loss warm. Re-measure if that reasoning stops
// holding (e.g. after other changes to the admit path).
#if defined(__linux__)
static bool moe_cache_live_prefetch_enabled() {
    static const bool enabled = [] {
        const char * env = getenv("GGML_CUDA_MOE_CACHE_LIVE_PREFETCH");
        return !env || atoi(env) != 0;
    }();
    return enabled;
}
#endif

static bool moe_cache_atlas_admit(
        moe_cache_device & device, moe_cache_pool & pool, int pool_index,
        const void * host_base, size_t expert_size,
        const moe_cache_atlas_cand * best, int n_best,
        int admit_k, int evict_cap_override) {
    // Free-slot-only, deliberately, and permanently - this is a SUGGESTION,
    // not a mandate. An eviction-capable version of this was built and
    // measured (2026-08-23 through the multi-round burst-mode benchmark,
    // docs/plan.md): the free-slot-only design was originally found "flat,
    // no effect either way" because pool.free_slots empties permanently
    // within the first few tokens of any real run, so eviction rights were
    // added to give it more room to act. That was the wrong fix. The
    // reactive/demand-driven admission path's own eviction is always
    // justified - it evicts to satisfy a request that just actually
    // happened. This path's "eviction" would sacrifice something the
    // reactive system already decided was worth keeping, to make room for a
    // PREDICTION - and when the prediction is wrong (or the evicted expert
    // is asked for again two tokens later, which the multi-switch benchmark
    // showed happens constantly), that is a real, direct loss, not a
    // neutral bet. Measured directly: widening the eviction budget for a
    // detected burst (up to 6 evictions/pass, vs. the original cap of 1)
    // made a real regression six times worse (post-switch hit rate delta
    // vs. no warming: -0.31pp thin-eviction -> -6.55pp wide-eviction, warm
    // losing on every single request across all 4 rounds) - unambiguous
    // confirmation that the eviction path itself, not its budget, was the
    // problem. Reverted to free-slot-only permanently; the wider
    // candidate pool, burst-triggered frequency, and multi-tensor sweep
    // added alongside the (now-removed) eviction rights are kept, since
    // none of those can ever actively hurt - they only mean more chances to
    // fill a slot that is ALREADY free, never a chance to take one away.
    int admitted = 0;
    bool woke = false;
    admit_k = admit_k > 0 ? std::min(admit_k, MOE_CACHE_ATLAS_ADMIT_K) : MOE_CACHE_ATLAS_ADMIT_K;
    (void) evict_cap_override; // no longer used - kept in the signature so callers need no further changes
    for (int i = 0; i < n_best && admitted < admit_k; i++) {
        const int32_t expert = best[i].expert;
        const moe_cache_key key{host_base, expert};
        if (pool.map.find(key) != pool.map.end()) {
            continue; // already resident or in flight - the only residency check now, see MOE_CACHE_ATLAS_RANK_K
        }
        if (pool.free_slots.empty()) {
            break; // nothing free right now - a suggestion with nowhere to land is a no-op, not a reason to evict
        }
        const int slot_index = pool.free_slots.back();
        pool.free_slots.pop_back();
        moe_cache_slot & slot = pool.slots[slot_index];
        slot.key = key;
        slot.generation++;
        slot.readers = 0;
        slot.state = moe_cache_slot_state::copying;
        try {
            if (!moe_cache_map_insert(pool, key, slot_index)) {
                moe_cache_slot_reset(pool, slot_index, true);
                continue;
            }
            const char * host_ptr = (const char *) host_base + (size_t) expert * expert_size;
            device.queue.push_back({
                    pool_index, slot_index, slot.generation, key,
                    host_ptr, expert_size});
            device.queued_bytes += expert_size;
            device.prefetches++;
            admitted++;
            woke = true;
#if defined(__linux__)
            // See moe_cache_live_prefetch_enabled()'s comment above. Only
            // for a candidate device.residency has actually flagged cold -
            // an already-warm expert's host pages need no hint, and hinting
            // on every admission regardless of residency would be exactly
            // the "more prefetch, unconditionally" mistake that comment
            // warns against.
            if (moe_cache_live_prefetch_enabled()) {
                const auto rit = device.residency.find(host_base);
                if (rit != device.residency.end() &&
                    (size_t) expert < rit->second.is_cold.size() &&
                    rit->second.is_cold[expert]) {
                    static const long page_size = sysconf(_SC_PAGESIZE);
                    if (page_size > 0) {
                        const uintptr_t begin = (uintptr_t) host_ptr;
                        const uintptr_t end   = begin + expert_size;
                        const uintptr_t first = (begin + page_size - 1) & ~((uintptr_t) page_size - 1);
                        const uintptr_t last  = end & ~((uintptr_t) page_size - 1);
                        if (last > first) {
                            posix_madvise((void *) first, (size_t) (last - first), POSIX_MADV_WILLNEED);
                        }
                    }
                }
            }
#endif
        } catch (...) {
            moe_cache_slot_reset(pool, slot_index, true);
        }
    }
    return woke;
}

// Measurement-only arm: reproduces the pre-Track-1.5 scan exactly (residency
// filtered inline, top-2 kept) so the +2.75pp topic-switch result recorded in
// docs/plan.md can be re-validated head to head against the new scan in one
// build, one session, one interleaving - rather than compared across
// sessions, which would not be evidence of much. Sync path only: the filter
// needs pool.map and therefore the lock, which is the whole reason the new
// scan does not have it.
static bool moe_cache_atlas_warm_legacy_rank_enabled() {
    static const bool enabled = [] {
        const char * env = getenv("GGML_CUDA_MOE_CACHE_ATLAS_WARM_LEGACY_RANK");
        return env && atoi(env) != 0;
    }();
    return enabled;
}

// Burst-mode budget knobs. Steady-state values match the original
// conservative design (verified flat-to-good there, no reason to change);
// burst values are what actually fire once a topic shift is detected - see
// moe_cache_atlas_req_dir_shifted below for the detector. All separately
// env-overridable so steady/burst can be tuned independently, same as every
// other A/B knob in this file.
// #4: all four budget knobs below used to key off a discrete bursting bool,
// which could flip every other throttle check for a centroid oscillating
// right at the distance threshold (steady and burst budgets are ~6x apart,
// so that flip is not a rounding error). They now take the continuous
// atlas_burst_confidence in [0,1] (see moe_cache_atlas_burst_check) and
// interpolate - gain scheduling, not a switch - so a centroid hovering near
// the boundary gets a budget that hovers too, instead of alternating
// between the two extremes.
static float moe_cache_atlas_burst_confidence_clamp(float c) {
    return c < 0.0f ? 0.0f : (c > 1.0f ? 1.0f : c);
}
static int moe_cache_atlas_rank_k(float confidence) {
    static const int steady = [] {
        const char * e = getenv("GGML_CUDA_MOE_CACHE_ATLAS_WARM_RANK_K");
        const int v = e ? atoi(e) : 4;
        return v > 0 ? std::min(v, MOE_CACHE_ATLAS_RANK_K) : 4;
    }();
    static const int burst = [] {
        const char * e = getenv("GGML_CUDA_MOE_CACHE_ATLAS_WARM_BURST_RANK_K");
        const int v = e ? atoi(e) : 24;
        return v > 0 ? std::min(v, MOE_CACHE_ATLAS_RANK_K) : 24;
    }();
    confidence = moe_cache_atlas_burst_confidence_clamp(confidence);
    return steady + (int) std::lround((double) (burst - steady) * confidence);
}
static int moe_cache_atlas_admit_k(float confidence) {
    static const int steady = [] {
        const char * e = getenv("GGML_CUDA_MOE_CACHE_ATLAS_WARM_ADMIT_K");
        const int v = e ? atoi(e) : 2;
        return v > 0 ? std::min(v, MOE_CACHE_ATLAS_ADMIT_K) : 2;
    }();
    static const int burst = [] {
        const char * e = getenv("GGML_CUDA_MOE_CACHE_ATLAS_WARM_BURST_ADMIT_K");
        const int v = e ? atoi(e) : 12;
        return v > 0 ? std::min(v, MOE_CACHE_ATLAS_ADMIT_K) : 12;
    }();
    confidence = moe_cache_atlas_burst_confidence_clamp(confidence);
    return steady + (int) std::lround((double) (burst - steady) * confidence);
}
static int moe_cache_atlas_evict_cap(float confidence) {
    confidence = moe_cache_atlas_burst_confidence_clamp(confidence);
    if (confidence <= 0.0f) {
        return -1; // no override - moe_cache_atlas_admit falls back to GGML_CUDA_MOE_CACHE_ATLAS_WARM_EVICT_CAP
    }
    static const int burst = [] {
        const char * e = getenv("GGML_CUDA_MOE_CACHE_ATLAS_WARM_BURST_EVICT_CAP");
        const int v = e ? atoi(e) : 6;
        return v < 0 ? 0 : v;
    }();
    return (int) std::lround((double) burst * confidence);
}

// Cold-start guard for burst detection specifically - separate from, and a
// much higher bar than, moe_cache_atlas_warm_min_samples (which only gates
// plain admission). Root-caused empirically (2026-08-29, ablation): a fresh
// req_dir doesn't just start noisy, it keeps SWINGING for a while as its EMA
// settles toward a representative direction - real, sustained movement, not
// a single bad sample, so the plain min-samples gate does not catch it. Each
// swing past the burst threshold during that settling window used to read as
// a genuine topic shift and fire the widened multi-tensor sweep, competing
// with real first-time demand for the scarcest free-slot budget a session
// ever has. Isolated by disabling burst alone (GGML_CUDA_MOE_CACHE_ATLAS_
// WARM_BURST_THRESHOLD=999): req0 hit rate on a fresh session went from
// 0.5816 (burst enabled, defaults) to 0.7157 (burst disabled) against a
// 0.7259 true baseline - almost the entire regression was burst firing on
// convergence noise, not the align-protect mechanism it was first blamed on.
static uint32_t moe_cache_atlas_burst_min_samples() {
    static const uint32_t n = [] {
        const char * e = getenv("GGML_CUDA_MOE_CACHE_ATLAS_WARM_BURST_MIN_SAMPLES");
        const int v = e ? atoi(e) : 64;
        return (uint32_t) (v > 0 ? v : 64);
    }();
    return n;
}

// Euclidean distance the live req_dir centroid has moved since the
// reference point saved at the last throttle check, in the atlas's own
// [-1,1]x[-1,1]-ish coordinate space. Updates the reference unconditionally
// (this is a step-wise "since last check" measurement, not "since session
// start"). A crossing ramps device.atlas_burst_confidence up (fast - a real
// shift deserves an immediate response); every non-crossing check decays it
// back down (slower, so a couple of quiet checks right after a shift don't
// immediately snap budgets back to steady). Returns confidence >= 0.5 for
// callers that still need a plain yes/no for a structural branch (sync
// multi-tensor sweep vs. the async single-tensor hint queue) - that
// threshold is now the entry/exit point of a smoothed signal instead of the
// raw instantaneous distance, which is what actually kills the flip-flop.
// First call after req_dir becomes valid always reports non-bursting
// (nothing to compare against yet), and no crossing is ever honoured before
// moe_cache_atlas_burst_min_samples updates have accumulated (the reference
// point still tracks throughout, so detection resumes cleanly once past it,
// comparing against a recent point rather than a stale pre-convergence one).
static bool moe_cache_atlas_burst_check(moe_cache_device & device) {
    static const float threshold = [] {
        const char * e = getenv("GGML_CUDA_MOE_CACHE_ATLAS_WARM_BURST_THRESHOLD");
        const float v = e ? (float) atof(e) : 0.25f;
        return v < 0.0f ? 0.0f : v;
    }();
    static const float ramp_up = [] {
        const char * e = getenv("GGML_CUDA_MOE_CACHE_ATLAS_WARM_BURST_CONF_RAMP");
        const float v = e ? (float) atof(e) : 0.6f;
        return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
    }();
    static const float decay = [] {
        const char * e = getenv("GGML_CUDA_MOE_CACHE_ATLAS_WARM_BURST_CONF_DECAY");
        const float v = e ? (float) atof(e) : 0.65f;
        return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
    }();
    if (!device.req_dir_valid) {
        return false;
    }
    const bool converged = device.req_dir_updates >= moe_cache_atlas_burst_min_samples();
    if (device.atlas_burst_ref_valid) {
        const float dx = device.req_dir_x - device.atlas_burst_ref_x;
        const float dy = device.req_dir_y - device.atlas_burst_ref_y;
        const bool crossed = converged && std::sqrt(dx * dx + dy * dy) >= threshold;
        if (crossed) {
            device.atlas_burst_confidence += ramp_up * (1.0f - device.atlas_burst_confidence);
        } else {
            device.atlas_burst_confidence *= decay;
        }
    }
    device.atlas_burst_ref_x = device.req_dir_x;
    device.atlas_burst_ref_y = device.req_dir_y;
    device.atlas_burst_ref_valid = true;
    return device.atlas_burst_confidence >= 0.5f;
}

// How many plan() calls between throttle checks - steady is the original
// 64; burst is far shorter so a detected shift gets several corrective
// passes in quick succession instead of waiting out the same interval that
// was sized for "nothing is changing". Interpolated by confidence like the
// other three budgets above.
static long long moe_cache_atlas_warm_interval(float confidence) {
    static const long long steady = [] {
        const char * e = getenv("GGML_CUDA_MOE_CACHE_ATLAS_WARM_INTERVAL");
        const long long v = e ? atoll(e) : 64;
        return v > 0 ? v : 64;
    }();
    static const long long burst = [] {
        const char * e = getenv("GGML_CUDA_MOE_CACHE_ATLAS_WARM_BURST_INTERVAL");
        const long long v = e ? atoll(e) : 4;
        return v > 0 ? v : 4;
    }();
    confidence = moe_cache_atlas_burst_confidence_clamp(confidence);
    return steady + (long long) std::lround((double) (burst - steady) * confidence);
}

// How many DISTINCT tensors get a warming pass in one burst firing (steady
// mode only ever does the one tensor node happens to be on - a real topic
// shift needs correction across many layers at once, not one every 64
// calls). Bounded and capped by how many tensors the atlas actually covers.
static int moe_cache_atlas_burst_tensor_sweep() {
    static const int n = [] {
        const char * e = getenv("GGML_CUDA_MOE_CACHE_ATLAS_WARM_BURST_TENSORS");
        const int v = e ? atoi(e) : 8;
        return v > 0 ? v : 8;
    }();
    return n;
}

// req_dir_valid flips true on the FIRST atlas-covered routing decision, at
// which point req_dir_x/y is that one expert's raw atlas position, not yet
// an average of anything - the EMA (decay=0.15) only starts pulling it
// toward a representative direction over the next several updates. Acting
// on admission that early means competing for the limited, non-refilling
// free_slots budget on a single noisy sample. Gate warming until the EMA
// has had a real chance to converge.
static uint32_t moe_cache_atlas_warm_min_samples() {
    static const uint32_t n = [] {
        const char * e = getenv("GGML_CUDA_MOE_CACHE_ATLAS_WARM_MIN_SAMPLES");
        const int v = e ? atoi(e) : 8;
        return (uint32_t) (v > 0 ? v : 8);
    }();
    return n;
}

// Synchronous path (GGML_CUDA_MOE_CACHE_ATLAS_WARM_ASYNC=0, the default
// until the split is measured): both halves inline on the dispatch thread,
// session lock held throughout, exactly as before Track 1.5.
// Congestion-avoidance dampener for burst's widened admission budget: only
// safe to spend when the pool actually has free capacity to spare. Free
// slots are the scarcest resource a fresh, still-filling pool has - every
// real first-time routing decision needs one, and they never refill except
// at pool creation (reactive eviction's moe_cache_slot_reset calls never
// add back to free_slots). Root-caused empirically (2026-08-29): gating
// burst on req_dir history alone (moe_cache_atlas_burst_min_samples) was
// necessary but not sufficient - burst kept firing throughout an entire
// cold-fill request, not just at the very start, because the actual
// determinant of whether a burst can be spent safely is how much free-slot
// headroom is left RIGHT NOW in THIS pool, not how long req_dir has been
// tracking on the device as a whole. Below the floor fraction, confidence
// scales toward 0 (falls back to steady's already-small budget); at or
// above it, full confidence passes through unchanged - the same back-off-
// under-contention shape AIMD uses, applied to admission instead of a
// send window.
static float moe_cache_atlas_burst_headroom_floor() {
    static const float v = [] {
        const char * e = getenv("GGML_CUDA_MOE_CACHE_ATLAS_WARM_BURST_HEADROOM_FLOOR");
        // 0.15 measured near-useless (real pool sizes are in the thousands
        // of slots; one request's cold fill never gets that scarce) - 0.95
        // measured best in single-shot req0 ablation (2026-08-29), 1.0
        // measured statistically indistinguishable from it (noise, not a
        // further gain). Effectively "dampen unless the pool is still
        // almost entirely untouched."
        const float f = e ? (float) atof(e) : 0.95f;
        return f < 0.0f ? 0.0f : (f > 1.0f ? 1.0f : f);
    }();
    return v;
}
static float moe_cache_atlas_burst_headroom_scale(const moe_cache_pool & pool) {
    const float floor = moe_cache_atlas_burst_headroom_floor();
    if (floor <= 0.0f || pool.n_slots <= 0) {
        return 1.0f;
    }
    const float free_frac = (float) pool.free_slots.size() / (float) pool.n_slots;
    return free_frac >= floor ? 1.0f : std::max(0.0f, free_frac / floor);
}

static bool moe_cache_atlas_warm(
        moe_cache_device & device, moe_cache_pool & pool, int pool_index,
        const void * host_base, size_t expert_size, int64_t n_expert,
        float burst_confidence = 0.0f) {
    if (!device.req_dir_valid || n_expert <= 0 || device.req_dir_updates < moe_cache_atlas_warm_min_samples()) {
        return false;
    }
    const auto it = device.atlas_by_tensor.find(host_base);
    if (it == device.atlas_by_tensor.end() || !it->second || it->second->empty()) {
        return false;
    }
    burst_confidence *= moe_cache_atlas_burst_headroom_scale(pool);
    const bool legacy = moe_cache_atlas_warm_legacy_rank_enabled();
    const int rank_k = moe_cache_atlas_rank_k(burst_confidence);
    float lx, ly;
    moe_cache_atlas_lookahead(device, lx, ly);
    moe_cache_atlas_cand best[MOE_CACHE_ATLAS_RANK_K];
    const int n_best = moe_cache_atlas_rank(
            *it->second, lx, ly, n_expert,
            best, legacy ? moe_cache_atlas_admit_k(burst_confidence) : rank_k,
            legacy ? &pool : nullptr, host_base);
    if (n_best <= 0) {
        return false;
    }
    return moe_cache_atlas_admit(device, pool, pool_index, host_base, expert_size, best, n_best,
            moe_cache_atlas_admit_k(burst_confidence), moe_cache_atlas_evict_cap(burst_confidence));
}

// Opt-in, and read once - see device.warm_path_ns for what it measures and
// why tok/s is the wrong instrument for it.
static bool moe_cache_atlas_warm_timing_enabled() {
    static const bool enabled = [] {
        const char * env = getenv("GGML_CUDA_MOE_CACHE_ATLAS_WARM_TIMING");
        return env && atoi(env) != 0;
    }();
    return enabled;
}

static bool moe_cache_atlas_warm_async_enabled() {
    static const bool enabled = [] {
        const char * env = getenv("GGML_CUDA_MOE_CACHE_ATLAS_WARM_ASYNC");
        return env && atoi(env) != 0;
    }();
    return enabled;
}

// Asynchronous path: everything moe_cache_plan does for warming, which is
// now a bounds check, a hash lookup and a push_back onto a one-deep queue.
// The row is captured by shared_ptr here (O(1), under the lock the caller
// already holds) so the worker never has to re-find it, and so a concurrent
// set_atlas cannot pull it out from under the scan.
static bool moe_cache_atlas_warm_enqueue(
        moe_cache_device & device, int pool_index,
        const void * host_base, size_t expert_size, int64_t n_expert) {
    if (!device.req_dir_valid || n_expert <= 0 || device.req_dir_updates < moe_cache_atlas_warm_min_samples()) {
        return false;
    }
    const auto it = device.atlas_by_tensor.find(host_base);
    if (it == device.atlas_by_tensor.end() || !it->second || it->second->empty()) {
        return false;
    }
    // One deep on purpose. A warming request is a hint about where the
    // request is heading *now*; if the worker is far enough behind that one
    // is still pending, a second is already stale by the time it would run,
    // and queueing it would only make the worker do the expensive scan twice
    // for the same answer. Dropping is the correct behavior, not a shortcut.
    if (!device.warm_queue.empty()) {
        return false;
    }
    float lx, ly;
    moe_cache_atlas_lookahead(device, lx, ly);
    device.warm_queue.push_back({
            it->second, pool_index, host_base, expert_size, n_expert,
            lx, ly});
    return true;
}

// Worker side of the async path. Called with the session lock held and an
// empty fill queue is NOT required - this runs before the worker blocks on
// its next job, so the ranking overlaps whatever the GPU is doing. The lock
// is released for the scan itself and retaken for admission; both the pool
// index and the atlas row are re-validated afterward, since anything can
// have happened to the cache while the lock was down.
static bool moe_cache_atlas_warm_service(
        moe_cache_session * session, moe_cache_device * device,
        std::unique_lock<std::mutex> & lock) {
    if (device->warm_queue.empty() || device->dead.load()) {
        return false;
    }
    const moe_cache_warm_request req = device->warm_queue.front();
    device->warm_queue.pop_front();
    if (!req.row) {
        return false;
    }

    moe_cache_atlas_cand best[MOE_CACHE_ATLAS_RANK_K];
    int n_best = 0;
    {
        // The whole point of Track 1.5: this scan runs with the session lock
        // released, off the dispatch path, on the fill worker's own time.
        // req.row is a shared_ptr, so it stays alive and immutable here even
        // if set_atlas republishes the tensor's row in the meantime.
        lock.unlock();
        n_best = moe_cache_atlas_rank(
                *req.row, req.req_dir_x, req.req_dir_y, req.n_expert,
                best, MOE_CACHE_ATLAS_RANK_K);
        lock.lock();
    }
    if (n_best <= 0 || session->stopping || device->dead.load()) {
        return false;
    }
    if (req.pool_index < 0 || (size_t) req.pool_index >= device->pools.size()) {
        return false; // pools can only grow today, but this is not the thread that guarantees it
    }
    moe_cache_pool * pool = device->pools[req.pool_index].get();
    // expert_size is re-checked, not assumed: admission writes a job whose
    // byte count has to match the slab's slot stride, and unlike the inline
    // path this one carries a size captured before the lock was ever
    // released. Pools are append-only and keyed by (expert_size, wtype)
    // today, so this should never fire - it costs a compare, and a wrong
    // stride here would be a silent overwrite of a neighbouring slot.
    if (!pool || !pool->slab || pool->expert_size != req.expert_size) {
        return false;
    }
    // Async path never participates in burst mode (see the sync call site's
    // comment - the one-deep hint queue has no room for a sweep's worth of
    // requests, and ASYNC is off by default anyway) - always steady budget.
    return moe_cache_atlas_admit(*device, *pool, req.pool_index,
            req.host_base, req.expert_size, best, n_best,
            moe_cache_atlas_admit_k(0.0f), moe_cache_atlas_evict_cap(0.0f));
}

// Stage A live-activation predictor (docs/plan.md, 2026-08-29): everything
// tried so far to discover expert relationships from HISTORICAL data
// (spectral graph embedding, incremental SGNS) lost decisively to the
// hand-labeled probe atlas on held-out link-prediction AUC (0.57/0.49 vs
// 0.71) - confirmed by web research to be a known, general result: a
// FUNCTIONAL signal (what causes an expert to activate) generalizes to
// future decisions better than a CORRELATIONAL one (what happened to fire
// alongside it), and recent (2025) MoE literature has moved past graph
// embeddings entirely to lightweight linear predictors trained directly on
// the model's own live hidden-state activations, reporting 93-97% next-
// expert accuracy. This is that: a predictor trained on the same tensor
// mul_mat_id already reads for the real fill/hit decision, not a separate
// offline artifact.
//
// Genuinely a two-stage build. This is Stage A only: the (activation, real
// ids) training PAIR is captured at the existing dispatch hook, but by the
// time that hook fires the router's decision for THIS token is already
// made - training happens here, but nothing here can influence what this
// token's own dispatch does. Stage B (a hook earlier in the graph, before
// the router's own gate matmul, so a prediction is available in time to
// influence admission) is a separate, larger change to graph construction
// (llama-graph.cpp) and is not part of this pass.
static bool moe_cache_train_predictor_enabled() {
    static const bool enabled = [] {
        const char * env = getenv("GGML_CUDA_MOE_CACHE_TRAIN_PREDICTOR");
        return env && atoi(env) != 0;
    }();
    return enabled;
}
static float moe_cache_train_lr() {
    static const float v = [] {
        const char * env = getenv("GGML_CUDA_MOE_CACHE_TRAIN_LR");
        const float f = env ? (float) atof(env) : 0.01f;
        return f > 0.0f ? f : 0.01f;
    }();
    return v;
}
static int moe_cache_train_negs() {
    static const int v = [] {
        const char * env = getenv("GGML_CUDA_MOE_CACHE_TRAIN_NEGS");
        const int n = env ? atoi(env) : 4;
        return n > 0 ? n : 4;
    }();
    return v;
}

// Warming knobs, deliberately named and shaped like the atlas's steady-state
// RANK_K/ADMIT_K (moe_cache_atlas_rank_k/admit_k) rather than reusing them -
// this is a materially stronger signal (40%+ measured top-1 accuracy vs.
// ~3% chance, see moe_cache_train_service) so it earns its own tuning space
// instead of inheriting budgets sized for the much weaker cosine-similarity
// signal.
static int moe_cache_train_rank_k() {
    static const int v = [] {
        const char * env = getenv("GGML_CUDA_MOE_CACHE_TRAIN_RANK_K");
        const int n = env ? atoi(env) : 8;
        return n > 0 ? std::min(n, MOE_CACHE_ATLAS_RANK_K) : 8;
    }();
    return v;
}
static int moe_cache_train_admit_k() {
    static const int v = [] {
        const char * env = getenv("GGML_CUDA_MOE_CACHE_TRAIN_ADMIT_K");
        const int n = env ? atoi(env) : 4;
        return n > 0 ? std::min(n, MOE_CACHE_ATLAS_ADMIT_K) : 4;
    }();
    return v;
}
// Hard floor on free slots, not a rate limit: warm at full speed as long as
// a pool's free_slots stays above this fraction of its own n_slots, stop
// entirely once it does not. Training itself (the SGD update) stays on
// every step regardless, cheap and always safe - only the admission side is
// gated. free_slots never refill during normal operation
// (moe_cache_slot_reset's reactive-eviction callers always pass
// add_to_free=false), so this directly caps the TOTAL cumulative amount of
// a session's free-slot budget predictor-warming can ever consume,
// independent of how often it fires - a step-count interval only bounds
// firing RATE, and two earlier attempts at that (a per-call interval, then
// a per-firing admit_k scaled by absolute free-slot headroom) both measured
// as barely moving the prefetch count (7,388-7,436 unthrottled) because
// with pools of thousands of slots neither ever actually bound until very
// late. A floor on the pool's OWN remaining budget does not have that
// problem: above it, nothing is throttled at all; below it, warming stops
// completely, permanently, for the rest of the session.
static float moe_cache_train_warm_slot_floor_frac() {
    static const float v = [] {
        const char * env = getenv("GGML_CUDA_MOE_CACHE_TRAIN_WARM_FLOOR");
        const float f = env ? (float) atof(env) : 0.5f;
        return f < 0.0f ? 0.0f : (f > 1.0f ? 1.0f : f);
    }();
    return v;
}

// Feed the atlas's live context signal (req_dir) into the predictor's OWN
// input features, rather than hand-blending the atlas's and predictor's
// scores after the fact outside the model. req_dir is the same [-1,1]-ish
// 2D live-topic-direction centroid moe_cache_atlas_rank already ranks
// against (device.req_dir_x/y, decaying EMA of real routing decisions) -
// appended once per token to the captured hidden-state vector (see
// moe_cache_train below), so it becomes part of what EVERY expert's row of
// W learns to weigh, end to end, via the exact same online SGD already
// training on the raw activations. This lets the model discover its own
// relationship between "where the live topic is trending" and "which
// experts that favours" - the atlas's own cosine-similarity ranking makes
// that relationship by hand; this lets gradient descent find it instead,
// and fuses it with the raw activation signal in whatever proportion the
// data actually supports, rather than a fixed or manually-scheduled weight.
static constexpr int64_t MOE_CACHE_TRAIN_CONTEXT_DIMS = 2; // req_dir_x, req_dir_y

// Self-tuning: how well predictor-driven admission has actually paid off,
// measured from real outcomes (device.predictor_admit_evidence_for/against)
// rather than guessed - same Bayesian-ratio shape as
// moe_cache_atlas_align_confidence. Used to scale admit_k in the warming
// block below, the same way that confidence scales align_protect_strength:
// starts at 0.5 (no evidence yet), climbs toward 1 as admitted content
// proves itself reused, falls toward 0 if it is mostly getting evicted
// unused - no hand-tuned constant governs how aggressive predictor warming
// is, its own measured track record does.
static double moe_cache_predictor_admit_confidence(const moe_cache_device & device) {
    const double f = device.predictor_admit_evidence_for;
    const double a = device.predictor_admit_evidence_against;
    return f / (f + a);
}

// Hit-time "for" evidence: called from the real hit path (moe_cache_plan,
// same site moe_cache_atlas_align_note_hit already samples) - a hit on a
// key predictor warming believes it admitted means that admission actually
// got used. Resolved and ERASED on the first hit, not left for further
// hits to keep counting - the verdict ("this paid off") is already settled,
// and leaving it in the set would let the SAME admission also count as
// "against" later when it eventually gets evicted (everything resident
// eventually is), double-counting one outcome as both good and bad.
static void moe_cache_predictor_note_hit(moe_cache_device & device, const moe_cache_key & key) {
    if (!moe_cache_train_predictor_enabled() || device.predictor_admitted.empty()) {
        return;
    }
    const auto it = device.predictor_admitted.find(key);
    if (it != device.predictor_admitted.end()) {
        device.predictor_admit_evidence_for += 1.0;
        device.predictor_admitted.erase(it);
    }
}

// Cross-restart persistence. No default path - opt-in, since this writes a
// file, and every prior mechanism this session that touched disk (the atlas
// file, the trace files) was an explicit, named path, never an implicit
// default location. Answers "shouldn't the confidence floor be decided by
// the system only" (2026-08-29): the (1,1) evidence prior is unavoidable
// before ANY evidence exists, but with this, that only happens once, ever,
// on this file's very first run - every later restart resumes from what was
// actually measured, not a fresh guess.
static const char * moe_cache_predictor_state_path() {
    return getenv("GGML_CUDA_MOE_CACHE_TRAIN_STATE_FILE");
}
static long long moe_cache_predictor_save_interval() {
    static const long long v = [] {
        const char * env = getenv("GGML_CUDA_MOE_CACHE_TRAIN_STATE_SAVE_INTERVAL");
        const long long n = env ? atoll(env) : 2000;
        return n > 0 ? n : 2000;
    }();
    return v;
}

static constexpr uint32_t MOE_CACHE_PREDICTOR_STATE_MAGIC = 0x4D435031u; // "MCP1"

// Loaded once, lazily, on this device's first training example - see
// device.predictor_state_load_attempted. Populates predictor_loaded_by_name
// (consulted per-tensor as each is first trained, since host_base is not
// yet known for tensors not yet seen this run) and the global evidence/step
// counters directly (those are not per-tensor, safe to apply immediately).
static void moe_cache_predictor_load_file(moe_cache_device & device) {
    device.predictor_state_load_attempted = true;
    const char * path = moe_cache_predictor_state_path();
    if (!path) {
        return;
    }
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        return; // no prior state yet - first run, or a fresh path - not an error
    }
    uint32_t magic = 0;
    f.read((char *) &magic, sizeof(magic));
    if (!f || magic != MOE_CACHE_PREDICTOR_STATE_MAGIC) {
        MOE_CACHE_LOG("[moe-cache] predictor state file '%s' unreadable or wrong format - ignoring\n", path);
        return;
    }
    uint32_t n_tensors = 0;
    f.read((char *) &n_tensors, sizeof(n_tensors));
    for (uint32_t t = 0; f && t < n_tensors; t++) {
        uint32_t name_len = 0;
        f.read((char *) &name_len, sizeof(name_len));
        if (!f || name_len > (1u << 20)) { // sanity ceiling, not a real limit
            return;
        }
        std::string name(name_len, '\0');
        f.read(name.data(), name_len);
        uint32_t n_expert = 0, hidden_dim = 0;
        f.read((char *) &n_expert, sizeof(n_expert));
        f.read((char *) &hidden_dim, sizeof(hidden_dim));
        if (!f) {
            return;
        }
        const size_t count = (size_t) n_expert * (size_t) hidden_dim;
        std::vector<float> w(count);
        if (count > 0) {
            f.read((char *) w.data(), count * sizeof(float));
        }
        if (!f) {
            return;
        }
        device.predictor_loaded_by_name[name] = std::move(w);
    }
    f.read((char *) &device.predictor_admit_evidence_for, sizeof(double));
    f.read((char *) &device.predictor_admit_evidence_against, sizeof(double));
    f.read((char *) &device.predictor_steps, sizeof(long long));
    f.read((char *) &device.predictor_top1_hits, sizeof(long long));
    if (!f) {
        // Per-tensor weights loaded fine above; only the trailing scalars
        // are suspect - leave evidence at its safe (1,1) prior rather than
        // trust a partially-read value.
        device.predictor_admit_evidence_for = 1.0;
        device.predictor_admit_evidence_against = 1.0;
    }
    MOE_CACHE_LOG("[moe-cache] predictor state resumed from '%s': %u tensors, "
            "admit_confidence=%.3f, %lld prior steps\n",
            path, n_tensors, moe_cache_predictor_admit_confidence(device), device.predictor_steps);
}

// Called periodically (moe_cache_predictor_save_interval steps), not only
// at shutdown - this session's own testing kills servers with SIGKILL
// routinely (pkill, not a graceful stop), which a save-on-destroy design
// would simply never reach. Overwrites the whole file each time: simple,
// and the file only ever holds one process's state at a time by design (see
// the single-writer caveat below) so there is nothing to merge.
static void moe_cache_predictor_save(const moe_cache_device & device) {
    const char * path = moe_cache_predictor_state_path();
    if (!path) {
        return;
    }
    const std::string tmp_path = std::string(path) + ".tmp";
    std::ofstream f(tmp_path, std::ios::binary | std::ios::trunc);
    if (!f) {
        return;
    }
    uint32_t magic = MOE_CACHE_PREDICTOR_STATE_MAGIC;
    f.write((const char *) &magic, sizeof(magic));
    // A tensor is writable only with BOTH a known stable name and a known
    // per-tensor hidden_dim (see that field's comment - ffn_down_exps and
    // ffn_gate_exps/up_exps genuinely differ here, so there is no
    // model-wide fallback to use if a specific tensor's is missing) - the
    // counting and writing loops below must use the exact same condition,
    // or the file's declared n_tensors will not match what is actually
    // written and every read after the mismatch point misparses.
    auto writable = [&](const void * host_base) {
        return device.tensor_names.count(host_base) && device.predictor_hidden_dim.count(host_base);
    };
    uint32_t n_tensors = 0;
    for (const auto & [host_base, w] : device.predictor_weights) {
        if (writable(host_base)) {
            n_tensors++;
        }
    }
    f.write((const char *) &n_tensors, sizeof(n_tensors));
    for (const auto & [host_base, w] : device.predictor_weights) {
        if (!writable(host_base)) {
            continue; // no stable name or no known hidden_dim - cannot be resumed later, skip rather than write an orphan
        }
        const std::string & name = device.tensor_names.at(host_base);
        const uint32_t name_len = (uint32_t) name.size();
        f.write((const char *) &name_len, sizeof(name_len));
        f.write(name.data(), name_len);
        const uint32_t hidden_dim = (uint32_t) device.predictor_hidden_dim.at(host_base);
        const uint32_t n_expert = hidden_dim > 0 ? (uint32_t) (w.size() / hidden_dim) : 0;
        f.write((const char *) &n_expert, sizeof(n_expert));
        f.write((const char *) &hidden_dim, sizeof(hidden_dim));
        if (!w.empty()) {
            f.write((const char *) w.data(), w.size() * sizeof(float));
        }
    }
    f.write((const char *) &device.predictor_admit_evidence_for, sizeof(double));
    f.write((const char *) &device.predictor_admit_evidence_against, sizeof(double));
    f.write((const char *) &device.predictor_steps, sizeof(long long));
    f.write((const char *) &device.predictor_top1_hits, sizeof(long long));
    f.close();
    if (!f) {
        return;
    }
    // Atomic-ish swap: rename is atomic on the same filesystem, so a reader
    // (the next process's load, or a concurrent inspection) never sees a
    // half-written file - the failure mode this session's own corruption
    // investigation spent weeks on was exactly this class of thing
    // elsewhere in the codebase, worth avoiding here on the first attempt.
    std::rename(tmp_path.c_str(), path);
}

// Worker side: one online logistic-regression step per queued example -
// positives are the real selected experts (push their score up toward this
// token's actual activation), negatives are a small random same-tensor
// sample (push unrelated experts' scores down) - same shape as
// moe-incremental-atlas.py's SGNS training, now on the real mechanistic
// signal instead of a co-activation graph proxy, and cheap enough (a
// handful of dot products) to run under the session lock rather than
// needing it released like the atlas scan above.
static void moe_cache_train_service(moe_cache_device & device) {
    if (device.train_queue.empty()) {
        return;
    }
    const moe_cache_train_request req = std::move(device.train_queue.front());
    device.train_queue.pop_front();
    const int64_t hidden_dim = (int64_t) req.acts.size();
    if (hidden_dim <= 0 || req.n_expert <= 0 || req.ids.empty()) {
        return;
    }
    device.predictor_hidden_dim[req.host_base] = hidden_dim;
    if (!device.predictor_state_load_attempted) {
        moe_cache_predictor_load_file(device);
    }
    std::vector<float> & W = device.predictor_weights[req.host_base];
    const size_t need = (size_t) req.n_expert * (size_t) hidden_dim;
    if (W.size() != need) {
        // Tensor's first training example this process, or n_expert/
        // hidden_dim changed (a different model loaded into the same
        // process) - consult persisted state (name-keyed, since host_base
        // is not stable across restarts) before falling back to zero.
        // Zero-initialized logistic regression is still a fine starting
        // point when nothing was persisted; nothing here depends on
        // symmetry-breaking randomness.
        bool restored = false;
        const auto nit = device.tensor_names.find(req.host_base);
        if (nit != device.tensor_names.end()) {
            const auto lit = device.predictor_loaded_by_name.find(nit->second);
            if (lit != device.predictor_loaded_by_name.end() && lit->second.size() == need) {
                W = lit->second;
                restored = true;
            }
        }
        if (!restored) {
            W.assign(need, 0.0f);
        }
    }
    const float * h = req.acts.data();
    const float lr = moe_cache_train_lr();
    auto sgd_step = [&](int32_t e, float label) {
        if (e < 0 || e >= req.n_expert) {
            return;
        }
        float * w = W.data() + (size_t) e * (size_t) hidden_dim;
        double score = 0.0;
        for (int64_t d = 0; d < hidden_dim; d++) {
            score += (double) w[d] * (double) h[d];
        }
        const double sig = 1.0 / (1.0 + std::exp(-std::max(-30.0, std::min(30.0, score))));
        const float grad = (float) ((double) label - sig) * lr;
        for (int64_t d = 0; d < hidden_dim; d++) {
            w[d] += grad * h[d];
        }
        return;
    };
    // Live accuracy sample: does the predictor's single highest-scoring
    // expert (BEFORE this step's update, so it is judged on what it knew
    // going in) actually land in the real selected set?
    {
        int32_t best_e = -1;
        double best_score = -1e300;
        for (int32_t e = 0; e < req.n_expert; e++) {
            const float * w = W.data() + (size_t) e * (size_t) hidden_dim;
            double score = 0.0;
            for (int64_t d = 0; d < hidden_dim; d++) {
                score += (double) w[d] * (double) h[d];
            }
            if (score > best_score) {
                best_score = score;
                best_e = e;
            }
        }
        if (best_e >= 0 && std::find(req.ids.begin(), req.ids.end(), best_e) != req.ids.end()) {
            device.predictor_top1_hits++;
        }
    }
    for (int32_t e : req.ids) {
        sgd_step(e, 1.0f);
    }
    static thread_local std::mt19937 rng(0xC0FFEE);
    std::uniform_int_distribution<int32_t> dist(0, (int32_t) req.n_expert - 1);
    const int negs = moe_cache_train_negs();
    for (int i = 0; i < negs; i++) {
        int32_t e = dist(rng);
        if (std::find(req.ids.begin(), req.ids.end(), e) == req.ids.end()) {
            sgd_step(e, 0.0f);
        }
    }
    // Warming: rank every expert by the (just-updated) predictor score and
    // hand the top few to the SAME free-slot-only admission function the
    // atlas uses (moe_cache_atlas_admit) - no new admission logic, this
    // reuses every safety property already proven there (suggestion-only,
    // never evicts, no-op when nothing is free). req.pool_index/expert_size
    // were captured at enqueue time (moe_cache_train, dispatch thread); both
    // are re-validated here since this runs later, on the worker.
    //
    // Gated on the pool's OWN remaining free-slot budget, not a firing-rate
    // interval - see moe_cache_train_warm_slot_floor_frac for why a hard
    // floor is the right shape here (two rate-based attempts, a step-count
    // interval and a per-firing size cap, both measured as barely moving
    // anything against pools of thousands of slots).
    if (req.pool_index >= 0 && (size_t) req.pool_index < device.pools.size()) {
        moe_cache_pool * pool = device.pools[req.pool_index].get();
        if (pool && pool->slab && pool->expert_size == req.expert_size &&
            pool->n_slots > 0 &&
            pool->free_slots.size() > (size_t) (pool->n_slots * moe_cache_train_warm_slot_floor_frac())) {
            const int rank_k = moe_cache_train_rank_k();
            moe_cache_atlas_cand cands[MOE_CACHE_ATLAS_RANK_K];
            int n_cands = 0;
            for (int32_t e = 0; e < (int32_t) req.n_expert; e++) {
                const float * w = W.data() + (size_t) e * (size_t) hidden_dim;
                double score = 0.0;
                for (int64_t d = 0; d < hidden_dim; d++) {
                    score += (double) w[d] * (double) h[d];
                }
                if (score <= 0.0) {
                    continue; // only warm candidates the predictor actually favours
                }
                const moe_cache_atlas_cand cand{e, (float) score};
                if (n_cands < rank_k) {
                    cands[n_cands++] = cand;
                    std::sort(cands, cands + n_cands, [](const moe_cache_atlas_cand & a, const moe_cache_atlas_cand & b) { return a.score > b.score; });
                } else if (cand.score > cands[rank_k - 1].score) {
                    cands[rank_k - 1] = cand;
                    std::sort(cands, cands + rank_k, [](const moe_cache_atlas_cand & a, const moe_cache_atlas_cand & b) { return a.score > b.score; });
                }
            }
            if (n_cands > 0) {
                // Self-tuning admit_k: scaled by predictor_admit_confidence
                // (measured real payoff - see that function's comment),
                // never below 1 so a confidence still settling from its 0.5
                // prior isn't silently zeroed out.
                const double confidence = moe_cache_predictor_admit_confidence(device);
                const int admit_k = std::max(1, (int) std::lround((double) moe_cache_train_admit_k() * confidence));
                // Residency BEFORE admitting, so the after-check below can
                // tell "this call just admitted it" apart from "it was
                // already resident" (moe_cache_atlas_admit silently skips
                // already-resident candidates) - attributing an
                // already-resident key's eventual eviction to predictor
                // warming would be a real mislabel, not just noise.
                bool was_resident[MOE_CACHE_ATLAS_RANK_K];
                for (int i = 0; i < n_cands; i++) {
                    was_resident[i] = pool->map.find(moe_cache_key{req.host_base, cands[i].expert}) != pool->map.end();
                }
                moe_cache_atlas_admit(device, *pool, req.pool_index, req.host_base, req.expert_size,
                        cands, n_cands, admit_k, -1);
                for (int i = 0; i < n_cands && i < admit_k; i++) {
                    if (was_resident[i]) {
                        continue;
                    }
                    const moe_cache_key key{req.host_base, cands[i].expert};
                    if (pool->map.find(key) != pool->map.end()) {
                        device.predictor_admitted.insert(key);
                    }
                }
            }
        }
    }

    device.predictor_steps++;
    if (getenv("MOE_CACHE_DEBUG_GATE") && device.predictor_steps % 2000 == 0) {
        fprintf(stderr, "[predictor-dbg] steps=%lld top1_acc=%.4f (chance ~= %.4f) "
                "admit_confidence=%.3f (for=%.1f against=%.1f, tracked=%zu)\n",
                device.predictor_steps,
                (double) device.predictor_top1_hits / (double) device.predictor_steps,
                (double) 8 /* typical top-k */ / (double) req.n_expert,
                moe_cache_predictor_admit_confidence(device),
                device.predictor_admit_evidence_for, device.predictor_admit_evidence_against,
                device.predictor_admitted.size());
    }
    // Periodic, not save-on-shutdown-only - see moe_cache_predictor_save's
    // comment for why (this session's own testing kills servers with
    // SIGKILL routinely).
    if (moe_cache_predictor_state_path() && device.predictor_steps % moe_cache_predictor_save_interval() == 0) {
        moe_cache_predictor_save(device);
    }
}

// Entry point: called from the existing mul_mat_id dispatch hook (Stage A
// wiring, ggml-cpu.c) with the router's ALREADY-made decision - see the
// Stage A comment above moe_cache_train_predictor_enabled for what this
// can and cannot influence. Copies exactly one token (the first in this
// batch) into the bounded train_queue - see that field's comment for why
// bounded-and-dropping, not queued, is correct here, same as warm_queue.
static void moe_cache_train(
        void * opaque, const int32_t * ids, int n_ids_per_token, int n_tokens,
        const float * acts, size_t act_stride, int64_t hidden_dim) {
    if (!moe_cache_train_predictor_enabled()) {
        return;
    }
    moe_cache_node * node = (moe_cache_node *) opaque;
    if (!node || !node->device || !ids || !acts || n_ids_per_token <= 0 ||
        n_tokens <= 0 || hidden_dim <= 0 || node->n_expert <= 0) {
        return;
    }
    moe_cache_device & device = *node->device;
    std::lock_guard<std::mutex> lock(node->session->mu);
    if (node->session->stopping || device.dead.load()) {
        return;
    }
    if (!device.train_queue.empty()) {
        return; // one deep, drop rather than fall behind - see the field comment
    }
    moe_cache_train_request req;
    req.host_base = node->host_base;
    req.n_expert = node->n_expert;
    req.pool_index = node->pool_index;
    req.expert_size = node->expert_size;
    req.acts.assign(acts, acts + hidden_dim); // token 0 only; act_stride unused when sampling just the first row
    // Append the live context signal (see MOE_CACHE_TRAIN_CONTEXT_DIMS's
    // comment) - lookahead-adjusted the same way the atlas's own ranking
    // uses it, so both mechanisms condition on the identical live-direction
    // estimate. 0,0 (the same value req_dir starts at before validity)
    // when nothing is known yet - a neutral input, not a wrong one.
    {
        float rx = 0.0f, ry = 0.0f;
        if (device.req_dir_valid) {
            moe_cache_atlas_lookahead(device, rx, ry);
        }
        req.acts.push_back(rx);
        req.acts.push_back(ry);
    }
    req.ids.assign(ids, ids + n_ids_per_token);
    (void) act_stride;
    (void) n_tokens;
    device.train_queue.push_back(std::move(req));
}

// Pick a resident stand-in for `missed`, or -1. Walks this tensor's resident
// bitmask (one hash for the tensor, then bit tests over ~4 words - the whole
// reason the mask exists) and scores each candidate by how often it has fired
// in the SAME routing decision as the expert it would replace. Co-activation
// rather than atlas position or router score, because those two rank experts
// context-free and both lost their eviction A/Bs to plain recency; co-firing
// is the only signal here that says "these two do the same job for this
// input". Ties break toward the hotter slot, which is the cheaper one to keep.
//
// Caller must hold the session lock, as everything reading pool state does.
static int moe_cache_substitute_pick(
        moe_cache_device & device, moe_cache_pool & pool,
        const void * host_base, int32_t missed, int64_t n_expert) {
    const std::vector<uint64_t> * words = moe_cache_mask_words(pool, host_base);
    if (!words) {
        return -1;
    }
    const moe_cache_key mkey{host_base, missed};
    const uint32_t min_coact = moe_cache_substitute_min_coact();
    const int scan_cap = moe_cache_substitute_scan();
    int best_slot = -1;
    uint32_t best_count = 0;
    uint16_t best_heat = 0;
    int examined = 0;
    for (size_t w = 0; w < words->size() && examined < scan_cap; w++) {
        uint64_t bits = (*words)[w];
        while (bits && examined < scan_cap) {
            const int cand = (int) (w << 6) + __builtin_ctzll(bits);
            bits &= bits - 1;
            if (cand == missed || cand >= (int) n_expert) {
                continue;
            }
            examined++;
            const auto cit = device.co_activation.find(
                    moe_cache_edge_undirected(mkey, moe_cache_key{host_base, cand}));
            if (cit == device.co_activation.end() || cit->second < min_coact) {
                continue;
            }
            const auto found = pool.map.find(moe_cache_key{host_base, cand});
            if (found == pool.map.end()) {
                continue; // mask and map disagree - trust the map, it is the truth
            }
            const moe_cache_slot & slot = pool.slots[found->second];
            if (slot.state != moe_cache_slot_state::valid) {
                continue;
            }
            if (cit->second > best_count ||
                (cit->second == best_count && slot.heat > best_heat)) {
                best_count = cit->second;
                best_heat  = slot.heat;
                best_slot  = found->second;
            }
        }
    }
    return best_slot;
}

// Pick the hottest resident stand-in for `missed`, breaking ties by
// co-activation count (how often it has fired alongside the expert it would
// replace, tracked live and continuously across this session's own real
// requests - the dynamic "these two do the same job for this input" signal,
// not the static, offline 9-category --expert-atlas-file probe data, which
// is explicitly NOT what this uses: co-activation grows and adapts with
// actual traffic, the atlas file is a fixed snapshot from 27 canned probes
// run once at build time. An atlas-position-based version of this was tried
// and reverted the same day - the dynamic, live-updating signal is the
// wanted one).
//   - heat is the PRIMARY key, co-activation only a tiebreak - "hot loaded"
//     candidates are preferred outright, not merely used to break ties
//     between similarly-co-activated ones.
//   - no min_coact floor: a candidate with zero recorded co-activation is
//     still eligible (scored as the lowest tiebreak tier), rather than
//     skipped outright. That floor was what caused most declines; removing
//     it means this only returns -1 when NOTHING for this tensor is
//     resident within the scan at all, not merely when nothing resident
//     happens to be a co-activation match yet.
// Caller must hold the session lock, as everything reading pool state does.
static int moe_cache_substitute_pick_hot(
        moe_cache_device & device, moe_cache_pool & pool,
        const void * host_base, int32_t missed, int64_t n_expert) {
    const std::vector<uint64_t> * words = moe_cache_mask_words(pool, host_base);
    if (!words) {
        return -1;
    }
    const moe_cache_key mkey{host_base, missed};
    const int scan_cap = moe_cache_substitute_scan();
    int best_slot = -1;
    uint16_t best_heat = 0;
    uint32_t best_count = 0;
    int examined = 0;
    for (size_t w = 0; w < words->size() && examined < scan_cap; w++) {
        uint64_t bits = (*words)[w];
        while (bits && examined < scan_cap) {
            const int cand = (int) (w << 6) + __builtin_ctzll(bits);
            bits &= bits - 1;
            if (cand == missed || cand >= (int) n_expert) {
                continue;
            }
            examined++;
            const auto found = pool.map.find(moe_cache_key{host_base, cand});
            if (found == pool.map.end()) {
                continue; // mask and map disagree - trust the map, it is the truth
            }
            const moe_cache_slot & slot = pool.slots[found->second];
            if (slot.state != moe_cache_slot_state::valid) {
                continue;
            }
            const auto cit = device.co_activation.find(
                    moe_cache_edge_undirected(mkey, moe_cache_key{host_base, cand}));
            const uint32_t count = cit == device.co_activation.end() ? 0 : cit->second;
            if (slot.heat > best_heat || (slot.heat == best_heat && count > best_count)) {
                best_heat  = slot.heat;
                best_count = count;
                best_slot  = found->second;
            }
        }
    }
    return best_slot;
}

// Pick a stand-in from the SAME token's router picks. ids arrive rank-ordered,
// so the first resident one is the highest-scoring available substitute - the
// router's own judgement about which expert suits this input, which is exactly
// what a co-activation score is only a proxy for.
//
// Measured against the alternatives on 151,584 real decisions, scoring each by
// the router's own score for the stand-in it chose:
//   random resident        0.01387
//   pairwise co-activation 0.02130   (what this replaces)
//   THIS (top-k resident)  0.03856   +81%
//   full-probs oracle      0.03790   - needs ffn_moe_probs, which is computed
//                                      on the GPU and not reachable from this
//                                      CPU dispatch path at all
//
// It declines more often than the co-activation scan (it only considers this
// token's k picks, not every resident expert of the tensor), and those rows
// fall back to CPU compute - correct, just slower. That trade is deliberate:
// a wrong stand-in costs output quality, a declined one costs only time.
static int moe_cache_substitute_pick_rank(
        moe_cache_pool & pool, const void * host_base, const int32_t * ids,
        int self_index, int top_k, int64_t n_expert) {
    if (top_k <= 1) {
        return -1;
    }
    const int token_start = self_index - (self_index % top_k);
    const int32_t missed = ids[self_index];
    for (int j = token_start; j < token_start + top_k; j++) {
        if (j == self_index) {
            continue;
        }
        const int32_t cand = ids[j];
        if (cand < 0 || cand >= n_expert || cand == missed) {
            continue;
        }
        const auto found = pool.map.find(moe_cache_key{host_base, cand});
        if (found == pool.map.end()) {
            continue;
        }
        if (pool.slots[found->second].state != moe_cache_slot_state::valid) {
            continue;
        }
        return found->second;   // ids are rank-ordered: first resident wins
    }
    return -1;
}

// Pruning for the live session-tracking maps (co_activation,
// co_activation_cross_layer, neuron_heat) - all insert-only, no eviction,
// growing with every newly-observed expert pair or (tensor,expert) combo
// for the life of the session. Measured cause of a real, repeatable memory
// leak (2026-09-04): swap grew ~500MB per 150-token generation on qwen4exp,
// RSS fell while VmSize kept climbing - the signature of heap that's never
// freed forcing the kernel to swap out colder pages to stay under the
// cgroup limit. None of these maps individually looked catastrophic, but
// all shared the same insert-only shape.
//
// co_activation's key space is the real risk - up to ~130K pairs/layer x
// n_layer, not bounded by n_expert alone the way a per-(tensor,expert) map
// is. Pruned by VALUE (the co-activation count): single-observation pairs
// first (cheapest to justify losing - noise, not a repeated signal, and
// normally the large majority of low-value entries), falling back to a
// real nth_element threshold only if that alone didn't reach the target.
static uint32_t moe_cache_coact_max_entries() {
    static const uint32_t n = [] {
        const char * e = getenv("GGML_CUDA_MOE_CACHE_COACT_MAX_ENTRIES");
        const long v = e ? atol(e) : 2000000L;
        return (uint32_t) (v > 1000 ? v : 2000000L);
    }();
    return n;
}

template <typename CoactMap>
static void moe_cache_prune_coact_map(CoactMap & m) {
    const size_t cap = moe_cache_coact_max_entries();
    if (m.size() <= cap) {
        return;
    }
    const size_t target = cap * 4 / 5; // hysteresis - don't re-trigger every call once at the boundary
    for (auto it = m.begin(); it != m.end() && m.size() > target; ) {
        it = it->second <= 1 ? m.erase(it) : std::next(it);
    }
    if (m.size() > target) {
        std::vector<uint32_t> counts;
        counts.reserve(m.size());
        for (const auto & kv : m) {
            counts.push_back(kv.second);
        }
        const size_t remove_n = m.size() - target;
        std::nth_element(counts.begin(), counts.begin() + (ptrdiff_t) remove_n, counts.end());
        const uint32_t threshold = counts[remove_n];
        for (auto it = m.begin(); it != m.end() && m.size() > target; ) {
            it = it->second <= threshold ? m.erase(it) : std::next(it);
        }
    }
}

// neuron_heat has no cheap per-entry "value" the way co_activation's own
// count does (heat lives per-neuron inside the vector, not per-key), so
// this bounds total memory by evicting in iteration order once over cap
// rather than a true LRU/least-valuable policy - strictly better than the
// unbounded status quo even without one. Its natural key space (n_expert x
// n_layer for one loaded model) is normally well under the default cap; the
// cap mainly guards larger-expert-count architectures and very long
// sessions rather than triggering routinely here.
static uint32_t moe_cache_neuron_heat_max_entries() {
    static const uint32_t n = [] {
        const char * e = getenv("GGML_CUDA_MOE_CACHE_NEURON_HEAT_MAX_ENTRIES");
        const long v = e ? atol(e) : 50000L;
        return (uint32_t) (v > 100 ? v : 50000L);
    }();
    return n;
}

static void moe_cache_prune_neuron_heat(moe_cache_device & device) {
    auto & m = device.neuron_heat;
    const size_t cap = moe_cache_neuron_heat_max_entries();
    if (m.size() <= cap) {
        return;
    }
    const size_t target = cap * 4 / 5;
    for (auto it = m.begin(); it != m.end() && m.size() > target; ) {
        it = m.erase(it);
    }
}

// Rate limit for the pruning above - a full-map pass is not free at
// multi-million-entry scale, so this only even checks .size() (cheap) most
// calls and only does real work on the rare call where that check trips.
static void moe_cache_prune_session_maps(moe_cache_device & device) {
    static const uint32_t interval = [] {
        const char * e = getenv("GGML_CUDA_MOE_CACHE_PRUNE_INTERVAL");
        const long v = e ? atol(e) : 4096L;
        return (uint32_t) (v > 0 ? v : 4096L);
    }();
    if (device.plan_epoch % interval != 0) {
        return;
    }
    moe_cache_prune_coact_map(device.co_activation);
    moe_cache_prune_coact_map(device.co_activation_cross_layer);
    moe_cache_prune_neuron_heat(device);
}

static int moe_cache_plan(
        void * opaque, const int32_t * ids, int n_ids, int32_t * slot_indices) {
    moe_cache_node * node = (moe_cache_node *)opaque;
    if (!node || !ids || !slot_indices || n_ids < 0 ||
        n_ids > GGML_MOE_CACHE_MAX_BATCH_ROWS || node->planned) {
        return 0;
    }
    node->planned = true;
    for (int index = 0; index < n_ids; index++) {
        slot_indices[index] = -1;
    }

    moe_cache_session & session = *node->session;
    moe_cache_device & device = *node->device;
    moe_cache_pool & pool = *node->pool;
    int hits = 0;
    int inserts_left = session.config.inserts_per_plan;
    bool wake_worker = false;
    // Group admission may evict at most once per plan() call - see the
    // block after the demand-fill queue below.
    bool group_evicted_this_call = false;

    // See device.layer_hits/layer_misses's own comment for what this
    // measures and why (throughput cost per layer, not activation shape).
    // session.mu is already held for this whole function, same as the
    // plain (non-atomic) rank_hits/rank_misses increments below - no
    // extra locking needed.
    auto note_layer = [&](bool hit) {
        if (node->layer < 0 || node->layer >= moe_cache_device::MAX_LAYER_STAT) {
            return;
        }
        if (hit) {
            device.layer_hits[node->layer]++;
        } else {
            device.layer_misses[node->layer]++;
        }
    };

    moe_cache_lock_trace_guard lock_trace("plan");
    std::unique_lock<std::mutex> lock(session.mu);
    if (session.stopping) {
        return 0;
    }

    // Record which experts were selected, for the dormant-page sweep below.
    // One hash lookup per tensor per call, then plain array stores - the entry
    // is found once here rather than per expert.
    moe_cache_device::cpu_residency * residency = nullptr;
    uint32_t age_now = 0;
    device.plan_epoch++;
    moe_cache_prune_session_maps(device);
    if (moe_cache_cold_after_s() > 0 && node->n_expert > 0) {
        const auto now = std::chrono::steady_clock::now();
        if (device.residency_epoch.time_since_epoch().count() == 0) {
            device.residency_epoch = now;
            device.last_cold_sweep = now;
        }
        // +1 so a just-selected expert is never confused with "never selected"
        age_now = 1 + (uint32_t) std::chrono::duration_cast<std::chrono::seconds>(
                now - device.residency_epoch).count();
        try {
            auto & entry = device.residency[node->host_base];
            if (entry.last_seen.empty()) {
                entry.expert_size = node->expert_size;
                entry.last_seen.assign((size_t) node->n_expert, 0);
                entry.is_cold.assign((size_t) node->n_expert, false);
                entry.selections.assign((size_t) node->n_expert, 0);
                entry.is_pinned.assign((size_t) node->n_expert, false);
                entry.last_epoch.assign((size_t) node->n_expert, 0);
                entry.host_slot = std::vector<std::atomic<void *>>((size_t) node->n_expert);
                for (auto & hs : entry.host_slot) {
                    hs.store(nullptr, std::memory_order_relaxed);
                }
            }
            residency = &entry;
        } catch (...) {
            residency = nullptr; // tracking is best-effort, never fail a decode for it
        }

    }

    // Router rank for each flat ids index. ggml-cpu.c lays the ids out as
    // [token][rank] (see the expert_ids fill in ggml_compute_forward_mul_mat_id),
    // and the router emits them rank-ordered, so index % top_k recovers the
    // rank. Guarded: a zero/inconsistent n_tokens folds everything into
    // rank 0 rather than dividing by zero.
    // GGML_CUDA_MOE_CACHE_WIDTH=1: per-node routing width. If the early layers
    // really are selecting every expert, n_ids is not n_tokens*top_k but something
    // far larger, and several 4096-sized arrays on both sides of the handshake are
    // undersized for it.
    if (getenv("GGML_CUDA_MOE_CACHE_WIDTH")) {
        // Ring buffer of the last 48 layer-2 dispatches, dumped at the first
        // identity flip so we can see what the cache was doing in the moments
        // BEFORE the fault, not after it.
        static std::mutex ring_mu;
        static std::vector<std::string> ring;
        static size_t ring_pos = 0;
        if (node->layer == 2) {
            char line[512];
            const int k = node->n_tokens > 0 ? (int) (n_ids / node->n_tokens) : 0;
            int off = snprintf(line, sizeof(line),
                    "n_tokens=%lld n_ids=%d top_k=%d hits=%lld misses=%lld evict=%lld inflight=%d ids=",
                    (long long) node->n_tokens, (int) n_ids, k,
                    device.hits, device.misses, device.evictions,
                    device.inflight ? 1 : 0);
            for (int i = 0; i < 40 && i < (int) n_ids && off < (int) sizeof(line) - 8; i++) {
                off += snprintf(line + off, sizeof(line) - off, "%d,", ids[i]);
            }
            std::lock_guard<std::mutex> rlock(ring_mu);
            if (ring.size() < 48) {
                ring.push_back(line);
            } else {
                ring[ring_pos] = line;
            }
            ring_pos = (ring_pos + 1) % 48;
        }

        // Identity routing (ids == 0,1,2..top_k-1) is what top-k returns when the
        // router logits are all NaN/equal. It is the fingerprint of the failure.
        // Log the FIRST layer that flips, which is where the NaN originates.
        {
            const int k = node->n_tokens > 0 ? (int) (n_ids / node->n_tokens) : 0;
            bool identity = k > 0 && n_ids >= k;
            for (int i = 0; identity && i < k; i++) {
                if (ids[i] != i) identity = false;
            }
            static std::atomic<int> flipped{0};
            if (identity && flipped.fetch_add(1) < 12) {
                fprintf(stderr, "[t=%.3f] [moe-cache] IDENTITY ROUTING layer=%d top_k=%d n_tokens=%lld\n",
                        moe_cache_wall_clock(), node->layer, k, (long long) node->n_tokens);
                static std::atomic<int> dumped{0};
                if (dumped.fetch_add(1) == 0) {
                    std::lock_guard<std::mutex> rlock(ring_mu);
                    fprintf(stderr, "[moe-cache] --- last %zu layer-2 dispatches before first flip ---\n",
                            ring.size());
                    for (size_t i = 0; i < ring.size(); i++) {
                        const size_t idx = (ring_pos + i) % ring.size();
                        fprintf(stderr, "[moe-cache] L2[%02zu] %s\n", i, ring[idx].c_str());
                    }
                }
                fflush(stderr);
            }
        }
        static std::atomic<int> shots{0};
        const int shot = shots.fetch_add(1);
        if (shot < 240) {
            int distinct = 0;
            {
                std::vector<unsigned char> seen_e((size_t) node->n_expert, 0);
                for (int i = 0; i < (int) n_ids; i++) {
                    const int e = ids[i];
                    if (e >= 0 && e < node->n_expert && !seen_e[(size_t) e]) {
                        seen_e[(size_t) e] = 1;
                        distinct++;
                    }
                }
            }
            fprintf(stderr, "[t=%.3f] [moe-cache] WIDTH layer=%d n_tokens=%lld n_ids=%d "
                    "top_k=%d distinct=%d/%d\n",
                    moe_cache_wall_clock(), node->layer, (long long) node->n_tokens, (int) n_ids,
                    node->n_tokens > 0 ? (int) (n_ids / node->n_tokens) : -1,
                    distinct, node->n_expert);
            fflush(stderr);
        }
    }
    const int rank_top_k = (node->n_tokens > 0 && n_ids >= node->n_tokens)
        ? (int) (n_ids / node->n_tokens) : 0;

    // Heat-aware neuron subsetting: resolve the reduced-width pool for
    // THIS tensor's shape once per plan() call, not once per id - the
    // reduced expert_size is deterministic from node->expert_size/n_out and
    // the global K (moe_cache_neuron_reduce_k()), so every id in this call
    // shares the same answer. nullptr (the common case: reduction disabled,
    // or no reduced pool exists yet because nothing has converted) means
    // every id below falls straight through to the ordinary primary-pool
    // lookup, unchanged.
    moe_cache_pool * reduced_pool_ptr = nullptr;
    if (moe_cache_neuron_reduce_enabled() && node->reduce_eligible && node->n_out > 0 &&
        node->expert_size % (size_t) node->n_out == 0) {
        const int reduce_k = moe_cache_neuron_reduce_k();
        const size_t row_bytes = node->expert_size / (size_t) node->n_out;
        if (reduce_k > 0 && row_bytes > 0) {
            const size_t reduced_expert_size = (size_t) reduce_k * row_bytes;
            const int ridx = moe_cache_find_pool(device, reduced_expert_size, node->wtype);
            if (ridx >= 0) {
                reduced_pool_ptr = device.pools[ridx].get();
            }
        }
    }

    for (int index = 0; index < n_ids; index++) {
        const int32_t expert = ids[index];
        if (expert < 0 || expert >= node->n_expert || device.dead.load()) {
            continue;
        }
        const int rank_bucket = std::min(
            rank_top_k > 0 ? index % rank_top_k : 0, GGML_MOE_CACHE_MAX_RANK - 1);

        if (residency && (size_t) expert < residency->last_seen.size()) {
            residency->last_seen[expert] = age_now;
            residency->last_epoch[expert] = device.plan_epoch;
            // The same heat signal the VRAM tier uses - +STEP per hit, saturating
            // at HEAT_MAX, halved on decay - rather than a raw cumulative count.
            // LFRU's parameters were settled by a lot of measurement upstream;
            // reusing them is the point, and a cumulative counter would drift
            // from that policy by growing steadily less responsive over time.
            residency->selections[expert] =
                std::min(MOE_CACHE_HEAT_MAX, residency->selections[expert] + MOE_CACHE_HEAT_STEP);
            // Mark for promotion once an expert has proven itself over several
            // selections, so a single fluke does not claim a slot. The copy is
            // the worker's job - see host_promote_queue. Bounded so a burst
            // cannot grow the queue without limit.
            if (moe_cache_host_budget_bytes() > 0 &&
                residency->selections[expert] >= MOE_CACHE_HEAT_STEP * 4 &&
                residency->host_slot[expert].load(std::memory_order_relaxed) == nullptr &&
                device.host_promote_queue.size() < 64) {
                device.host_promote_queue.emplace_back(node->host_base, expert);
                wake_worker = true;
            }
            // A selected expert is live again: clear the cold mark so a later
            // dormant stretch re-advises rather than being skipped forever.
            residency->is_cold[expert] = false;
        }

        const moe_cache_key key{node->host_base, expert};

        // Step 0 of Atlas-driven cache warming: fold this real, live routing
        // decision into the decaying request-direction centroid, regardless
        // of hit/miss - this is about what the router actually selected,
        // not about cache state. No-op (and cheap: one hash lookup) for any
        // expert the atlas has no measured position for, which is expected
        // and fine - coverage is whatever the offline probes actually hit.
        // Purely observational: nothing downstream reads req_dir_* yet.
        {
            const auto atlas_it = device.atlas.find(key);
            if (atlas_it != device.atlas.end()) {
                const auto & cell = atlas_it->second;
                if (!device.req_dir_valid) {
                    device.req_dir_x = cell.x;
                    device.req_dir_y = cell.y;
                    device.req_dir_valid = true;
                    // No prior level to take a delta against yet - trend
                    // starts at zero (no movement assumed) rather than a
                    // spurious jump from the origin.
                } else {
                    const float prev_x = device.req_dir_x, prev_y = device.req_dir_y;
                    device.req_dir_x += moe_cache_device::MOE_CACHE_ATLAS_DECAY * (cell.x - device.req_dir_x);
                    device.req_dir_y += moe_cache_device::MOE_CACHE_ATLAS_DECAY * (cell.y - device.req_dir_y);
                    // Holt's trend update: EMA of the level's own step, i.e.
                    // how fast and which way the centroid is currently
                    // drifting - see moe_cache_atlas_lookahead.
                    const float step_x = device.req_dir_x - prev_x;
                    const float step_y = device.req_dir_y - prev_y;
                    device.req_vel_x += moe_cache_device::MOE_CACHE_ATLAS_TREND_DECAY * (step_x - device.req_vel_x);
                    device.req_vel_y += moe_cache_device::MOE_CACHE_ATLAS_TREND_DECAY * (step_y - device.req_vel_y);
                }
                device.req_dir_updates++;
                if (getenv("MOE_CACHE_DEBUG_GATE")) {
                    static std::atomic<long long> n{0};
                    if (++n % 200 == 0) {
                        fprintf(stderr, "[atlas-dbg] req_dir=(%.3f,%.3f) vel=(%.4f,%.4f) last_expert=(%.3f,%.3f) spec=%.3f "
                                "evidence_for=%.1f evidence_against=%.1f confidence=%.3f heat_scale_ema=%.2f\n",
                                device.req_dir_x, device.req_dir_y, device.req_vel_x, device.req_vel_y,
                                cell.x, cell.y, cell.spec,
                                device.atlas_align_evidence_for, device.atlas_align_evidence_against,
                                moe_cache_atlas_align_confidence(device), pool.heat_scale_ema);
                    }
                }
            }
        }

        // Heat-aware neuron subsetting: prefer a reduced-width hit over the
        // primary pool once one exists for this key. A row served this way
        // is fully handled here (dispatch/collect will read its K real
        // values out of the reduced pool and reconstruct an n_out-wide row
        // with everything else zero - see the split in moe_cache_dispatch) -
        // it must NOT also fall through to the primary lookup below, or the
        // row would end up pinned in two pools for one id.
        if (reduced_pool_ptr) {
            auto rfound = reduced_pool_ptr->map.find(key);
            if (rfound != reduced_pool_ptr->map.end()) {
                moe_cache_slot & rslot = reduced_pool_ptr->slots[rfound->second];
                if (rslot.state == moe_cache_slot_state::valid &&
                    node->n_pins < GGML_MOE_CACHE_MAX_BATCH_ROWS) {
                    rslot.readers++;
                    rslot.heat = std::min(MOE_CACHE_HEAT_MAX, rslot.heat + MOE_CACHE_HEAT_STEP);
                    moe_cache_pool_note_heat_step(*reduced_pool_ptr, rslot.heat);
                    node->pins[node->n_pins++] = {rfound->second, rslot.generation, reduced_pool_ptr};
                    slot_indices[index] = MOE_CACHE_REDUCED_SLOT_OFFSET + rfound->second;
                    // moe_cache_verify_slot is deliberately NOT called here:
                    // it assumes a resident slot is a byte-for-byte
                    // contiguous copy of [host_base + expert*pool.expert_size,
                    // ...), which is true for every other pool but not this
                    // one - a reduced-pool slot holds a GATHERED, non-
                    // contiguous subset of rows, so that comparison would
                    // compare against the wrong host bytes entirely and
                    // report a false mismatch on every sampled hit.
                    device.hits++;
                    device.rank_hits[rank_bucket]++;
                    note_layer(true);
                    hits++;
                    continue;
                }
            }
        }

        auto found = pool.map.find(key);
        if (found != pool.map.end()) {
            moe_cache_slot & slot = pool.slots[found->second];
            // The substitution path below already bounded its pins[] write
            // (checks node->n_pins < ceiling before pushing); this one used
            // to increment slot.readers and push to pins[] in the wrong
            // order, so an overflow left the reader count bumped with
            // nothing recorded to release it - a permanent pin leak - and
            // still handed the caller a slot_indices entry pointing at a
            // slot that was never actually pinned, which the caller could
            // then read after it gets evicted/refilled for someone else.
            // Checked first now, before any of that happens: on overflow the
            // row degrades to an ordinary miss rather than a half-served
            // hit. Currently unreachable - n_ids is capped at
            // GGML_MOE_CACHE_MAX_BATCH_ROWS at moe_cache_plan's entry and
            // n_pins increments at most once per id, so n_pins can never
            // exceed n_ids - but worth being correct if that cap ever
            // loosens.
            if (slot.state == moe_cache_slot_state::valid &&
                node->n_pins >= GGML_MOE_CACHE_MAX_BATCH_ROWS) {
                static std::atomic<int> once{0};
                if (once.fetch_add(1) == 0) {
                    fprintf(stderr,
                            "[moe-cache] PIN OVERFLOW n_pins=%d ceiling=%d n_ids=%d "
                            "n_tokens=%lld - hit not pinned\n",
                            node->n_pins, GGML_MOE_CACHE_MAX_BATCH_ROWS,
                            (int) n_ids, (long long) node->n_tokens);
                    fflush(stderr);
                }
                device.misses++;
                device.rank_misses[rank_bucket]++;
                note_layer(false);
                continue;
            }
            if (slot.state == moe_cache_slot_state::valid) {
                slot.readers++;
                slot.heat = std::min(MOE_CACHE_HEAT_MAX, slot.heat + MOE_CACHE_HEAT_STEP);
                moe_cache_pool_note_heat_step(pool, slot.heat);
                moe_cache_atlas_align_note_hit(device, key);
                moe_cache_predictor_note_hit(device, key);
                // Any real hit, from either segment, promotes to (or
                // refreshes within) protected_ - a resident slot being
                // requested again is exactly the "genuinely hot, not just
                // a one-off admission" signal that earns real protection
                // from eviction churn. Heat then decides how long that
                // protection actually lasts, at the next decay sweep.
                // TEMPORARY diagnostic (moe-cache corruption investigation):
                // GGML_CUDA_MOE_CACHE_NO_PROMOTE=1 skips ONLY the segment
                // promotion on a hit, keeping readers++/heat/pins - isolates
                // moe_cache_promote_to_protected specifically, the last
                // remaining candidate after admission's copy, dispatch,
                // segment-push, and row-accounting were all individually
                // ruled out by direct testing.
                static const bool no_promote = [] {
                    const char * e = getenv("GGML_CUDA_MOE_CACHE_NO_PROMOTE");
                    return e && atoi(e) != 0;
                }();
                if (!no_promote) {
                    moe_cache_promote_to_protected(device, pool, found->second);
                }
                node->pins[node->n_pins++] = {found->second, slot.generation, &pool};
                // GGML_CUDA_MOE_CACHE_NO_HITS=1: do every bit of accounting a
                // hit normally does (heat, promotion, pin, counters) but never
                // report the hit to the caller, so ggml-cpu.c routes the row
                // through the ordinary CPU path instead of the compact arrays.
                // Splits "plan()'s internal bookkeeping" from "the row-routing
                // that a reported hit triggers".
                static const bool no_hits = [] {
                    const char * e = getenv("GGML_CUDA_MOE_CACHE_NO_HITS");
                    return e && atoi(e) != 0;
                }();
                if (!no_hits) {
                    slot_indices[index] = found->second;
                }
                moe_cache_verify_slot(pool, found->second, key, device.physical);
                device.hits++;
                device.rank_hits[rank_bucket]++;
                note_layer(true);
                hits++;
            } else {
                device.misses++;
                device.rank_misses[rank_bucket]++;
                note_layer(false);
            }
            continue;
        }

        device.misses++;
        device.rank_misses[rank_bucket]++;
        note_layer(false);

        // Substitution. Deliberately does NOT skip the admission bookkeeping
        // below: the router still wanted this expert, and admission must keep
        // following what the router wanted rather than what it was forced to
        // run. Feeding admission from substituted picks would close a
        // self-confirming loop where resident experts are the only ones ever
        // used and therefore the only ones ever kept - the confirmation-drift
        // trap docs/plan.md already names for the dynamic atlas.
        //
        // Unconditional (2026-09-04): every miss attempts a hot-loaded
        // stand-in first - generation only ever uses what's actually
        // resident in VRAM, never waits on a real fetch by choice. A real
        // PCIe fetch only happens when moe_cache_substitute_pick_hot itself
        // returns -1, i.e. nothing for this tensor is resident at all
        // (measured well under 5% of attempts) - that is a hard floor, not
        // a pacing decision. The earlier rank/pace-gated version is kept
        // available via GGML_CUDA_MOE_CACHE_SUBSTITUTE_MIN_RANK for anyone
        // who wants the router's most-confident picks to always pay exact
        // compute regardless of residency.
        if (moe_cache_substitute_enabled() && slot_indices[index] < 0 &&
            rank_bucket >= moe_cache_substitute_min_rank(device) &&
            node->n_pins < GGML_MOE_CACHE_MAX_BATCH_ROWS) {
            // Hot-loaded-first by default (2026-09-04): picks the hottest
            // currently-resident expert for this tensor, tiebroken by
            // cross-request co-activation, with no min_coact floor to
            // decline against - see moe_cache_substitute_pick_hot's own
            // comment. The narrower per-token-rank method and the older
            // co-activation-primary method both stay available for
            // comparison via GGML_CUDA_MOE_CACHE_SUBSTITUTE_STRICT_RANK=1 /
            // GGML_CUDA_MOE_CACHE_SUBSTITUTE_COACT=1 respectively - strict
            // rank wins if both are set, since it is the more conservative
            // of the two opt-outs.
            static const bool use_strict_rank = [] {
                const char * env = getenv("GGML_CUDA_MOE_CACHE_SUBSTITUTE_STRICT_RANK");
                return env && atoi(env) != 0;
            }();
            static const bool use_coact = [] {
                const char * env = getenv("GGML_CUDA_MOE_CACHE_SUBSTITUTE_COACT");
                return env && atoi(env) != 0;
            }();
            const int sub = use_strict_rank
                ? moe_cache_substitute_pick_rank(pool, node->host_base, ids, index,
                                                 rank_top_k, node->n_expert)
                : use_coact
                ? moe_cache_substitute_pick(device, pool, node->host_base, expert, node->n_expert)
                : moe_cache_substitute_pick_hot(device, pool, node->host_base, expert, node->n_expert);
            if (sub >= 0) {
                moe_cache_slot & ssl = pool.slots[sub];
                ssl.readers++;
                // Heat, but no promotion to protected: it was genuinely used,
                // so it earns recency, but it was never actually demanded by
                // the router and must not claim eviction protection on the
                // strength of standing in.
                ssl.heat = std::min(MOE_CACHE_HEAT_MAX, ssl.heat + MOE_CACHE_HEAT_STEP);
                moe_cache_pool_note_heat_step(pool, ssl.heat);
                node->pins[node->n_pins++] = {sub, ssl.generation, &pool};
                slot_indices[index] = sub;
                device.substitutions++;
                // For the Brain UI's substitute overlay (yellow) - see
                // moe_cache_get_substitute_map. The SUBSTITUTE's own key
                // (the expert actually doing the extra work), not the
                // missed one - read-and-cleared by that getter, so this is
                // "since last poll" event tracking, not a persistent flag.
                device.substituted_recently.insert(ssl.key);
                hits++;   // dispatchable row, though NOT counted in device.hits
            } else {
                device.substitute_declined++;
            }
        }

        moe_cache_demand * demand = nullptr;
        try {
            demand = &device.demand_count[key];
        } catch (...) {
            device.insert_skips++;
            continue;
        }
        demand->expert_size = node->expert_size;
        if (demand->count < std::numeric_limits<uint16_t>::max()) {
            demand->count++;
        }
        const int admit_after = pool.free_slots.empty()
            ? std::max(session.config.admit_after, session.config.readmit_after)
            : session.config.admit_after;
        if (demand->count < admit_after) {
            device.admission_skips++;
            continue;
        }
        const size_t queue_limit = session.config.queue_mb << 20;
        if (inserts_left <= 0 || (int)device.queue.size() >= session.config.queue_max ||
            node->expert_size > queue_limit - std::min(queue_limit, device.queued_bytes)) {
            device.insert_skips++;
            continue;
        }

        int slot_index = -1;
        if (!pool.free_slots.empty()) {
            slot_index = pool.free_slots.back();
            pool.free_slots.pop_back();
        } else {
            // Drain probation first - only touch protected_ (proven-hot
            // slots, reused at least once while resident) once probation
            // has genuinely nothing left to sacrifice. This is the segment
            // half of the LFRU protection: a burst of one-off cold
            // admissions can fully cycle through probation without ever
            // touching something with a real repeated-access history.
            // Within whichever segment is being drawn from, though, the
            // pick among a bounded recency window is by heat (coldest
            // wins, with hysteresis) rather than plain oldest-first - the
            // heat half of the hybrid.
            int candidate = moe_cache_pick_coldest_unpinned(device, pool, pool.lru_head);
            if (candidate < 0) {
                candidate = moe_cache_pick_coldest_unpinned(device, pool, pool.protected_head);
            }
            if (candidate < 0) {
                device.insert_skips++;
                continue;
            }
            slot_index = candidate;
            moe_cache_slot_reset(pool, slot_index, false);
            device.evictions++;
        }

        moe_cache_slot & slot = pool.slots[slot_index];
        slot.key = key;
        slot.generation++;
        slot.readers = 0;
        slot.state = moe_cache_slot_state::copying;
        try {
            const bool inserted_ok = moe_cache_map_insert(pool, key, slot_index);
            if (!inserted_ok) {
                moe_cache_slot_reset(pool, slot_index, true);
                device.insert_skips++;
                continue;
            }

            const void * source =
                (const char *)node->host_base + (size_t)expert * node->expert_size;
            device.queue.push_back({
                    node->pool_index, slot_index, slot.generation,
                    key, source, node->expert_size,
                    node->host_base, (size_t) node->n_expert * node->expert_size});
            device.queued_bytes += node->expert_size;
        } catch (...) {
            moe_cache_slot_reset(pool, slot_index, true);
            device.insert_skips++;
            continue;
        }
        device.inserts++;
        inserts_left--;
        wake_worker = true;

        // Step 7a follow-on: group-aware admission. We just took a REAL
        // demand miss on `expert` - not a guess - and measurement says its
        // most frequent partner is selected in the same routing decision
        // 67.6% of the time (vs 2.8% chance). Admitting `expert` alone
        // therefore leaves a ~2-in-3 miss on the partner that the current
        // one-expert-at-a-time policy takes structurally. So admit the
        // partner alongside it.
        //
        // Deliberately piggybacked on a demand fill rather than run as its
        // own speculative pass: the trigger is a confirmed need, which is
        // what makes this different from atlas warming (a topic guess, and
        // measured flat). Still bounded hard - one partner per demand fill,
        // one eviction per plan() call, probation only, never protected_ -
        // because a 67.6% partner hit rate does not by itself prove the
        // slot is better spent on the partner than on whatever it displaces.
        // The A/B decides that, not the precision number.
        const size_t group_queue_limit = session.config.queue_mb << 20;
        if (moe_cache_group_admit_enabled() && inserts_left > 0 &&
            (int) device.queue.size() < session.config.queue_max &&
            // Honour the in-flight BYTE budget too, not just the entry
            // count - the demand path above guards both, and skipping this
            // let group admission push queued_bytes past the configured
            // queue_mb cap.
            node->expert_size <= group_queue_limit - std::min(group_queue_limit, device.queued_bytes)) {
            const auto pit = device.partner_best.find(key);
            if (pit != device.partner_best.end()) {
                const moe_cache_key pkey = pit->second.first;
                // pkey.tensor is expected to equal node->host_base -
                // within-layer co-activation only ever pairs experts of the
                // same tensor, so partner_best[{host_base, a}] can only hold
                // a partner built from that same host_base. Checked rather
                // than assumed: the slot is keyed by pkey but FILLED from
                // node->host_base below, so if the two ever diverged the
                // slot would hold another tensor's weights - silent
                // corruption, and the same shape of unchecked assumption
                // that caused a real segfault earlier in this work.
                // Confidence gate. The first version of this policy admitted
                // the top partner unconditionally and measured -0.5% tok/s
                // for +0.39pp hit rate: it was paying a fill and an
                // eviction for partners whose co-occurrence might be
                // incidental. P(partner | anchor fired) is the honest
                // measure; a raw count is not.
                static const double min_p = [] {
                    const char * env = getenv("GGML_CUDA_MOE_CACHE_GROUP_ADMIT_MIN_P");
                    const double v = env ? atof(env) : 0.5;
                    return v < 0.0 ? 0.0 : v;
                }();
                const auto fit = device.expert_fire_count.find(key);
                const uint32_t fired = fit == device.expert_fire_count.end() ? 0 : fit->second;
                const double conf = fired > 0 ? (double) pit->second.second / (double) fired : 0.0;
                if (conf >= min_p &&
                    pkey.tensor == node->host_base &&
                    pkey.expert >= 0 && pkey.expert < node->n_expert &&
                    pool.map.find(pkey) == pool.map.end()) {
                    int pslot = -1;
                    if (!pool.free_slots.empty()) {
                        pslot = pool.free_slots.back();
                        pool.free_slots.pop_back();
                    } else if (!group_evicted_this_call) {
                        // free_slots empties permanently within the first
                        // few tokens of any real run (it means "never used
                        // since allocation", not "currently unoccupied"),
                        // so without this the whole mechanism would be
                        // inert after warmup - the exact trap the first
                        // version of atlas warming fell into.
                        const int cand = moe_cache_pick_coldest_unpinned(device, pool, pool.lru_head);
                        // Only displace something genuinely cold. The first
                        // version evicted whatever was coldest even when
                        // "coldest" was still warm, so a confident partner
                        // could throw out an expert that was itself being
                        // used - paying twice (a fill now, a refill later)
                        // for one speculative admission.
                        static const uint32_t max_victim_heat = [] {
                            const char * env = getenv("GGML_CUDA_MOE_CACHE_GROUP_ADMIT_VICTIM_HEAT");
                            return (uint32_t) (env ? atoi(env) : MOE_CACHE_HEAT_STEP * 2);
                        }();
                        if (cand >= 0 && pool.slots[cand].heat <= max_victim_heat) {
                            moe_cache_slot_reset(pool, cand, false);
                            pslot = cand;
                            device.evictions++;
                            group_evicted_this_call = true;
                        }
                    }
                    if (pslot >= 0) {
                        moe_cache_slot & pslot_ref = pool.slots[pslot];
                        pslot_ref.key = pkey;
                        pslot_ref.generation++;
                        pslot_ref.readers = 0;
                        pslot_ref.state = moe_cache_slot_state::copying;
                        try {
                            if (moe_cache_map_insert(pool, pkey, pslot)) {
                                const void * psrc = (const char *) node->host_base +
                                    (size_t) pkey.expert * node->expert_size;
                                device.queue.push_back({
                                        node->pool_index, pslot, pslot_ref.generation,
                                        pkey, psrc, node->expert_size,
                                        node->host_base, (size_t) node->n_expert * node->expert_size});
                                device.queued_bytes += node->expert_size;
                                device.group_admits++;
                                inserts_left--;
                            } else {
                                moe_cache_slot_reset(pool, pslot, true);
                            }
                        } catch (...) {
                            moe_cache_slot_reset(pool, pslot, true);
                        }
                    }
                }
            }
        }
    }

    // Track 1 step 4a: within-layer co-activation tracking. Capped to
    // small ids[] batches (decode-scale, not a large prefill chunk) since
    // this is an O(n^2) pass over the batch's own selected experts - a
    // real prefill batch can carry thousands of ids, and n^2 on that would
    // be a genuine cost, not a rounding error. n_ids <= 32 covers ordinary
    // decode (n_expert_used is a handful) with real margin.
    if (n_ids >= 2 && n_ids <= 32) {
        // Step 7a scoring, BEFORE this decision updates the table - same
        // discipline as step 6, so a prediction is never graded against
        // evidence it just supplied.
        if (moe_cache_measure_pred_enabled() && ids[0] >= 0) {
            const moe_cache_key anchor{node->host_base, ids[0]};
            const auto it = device.partner_best.find(anchor);
            if (it != device.partner_best.end()) {
                device.partner_pred_total++;
                device.partner_chance_num += n_ids - 1;
                device.partner_chance_den += node->n_expert > 0 ? node->n_expert : 1;
                for (int b = 1; b < n_ids; b++) {
                    if (ids[b] == it->second.first.expert) {
                        device.partner_pred_hit++;
                        break;
                    }
                }
                if (device.partner_pred_total % 2000 == 0) {
                    const double chance = device.partner_chance_den > 0
                        ? 100.0 * (double) device.partner_chance_num / (double) device.partner_chance_den : 0.0;
                    fprintf(stderr,
                            "[partner-pred] within-layer precision@1 %.1f%% (%lld/%lld) vs %.1f%% chance\n",
                            100.0 * (double) device.partner_pred_hit / (double) device.partner_pred_total,
                            device.partner_pred_hit, device.partner_pred_total, chance);
                }
            }
        }
        for (int a = 0; a < n_ids; a++) {
            if (ids[a] < 0) continue;
            if (moe_cache_partner_index_enabled()) {
                device.expert_fire_count[{node->host_base, ids[a]}]++;
            }
            for (int b = a + 1; b < n_ids; b++) {
                if (ids[b] < 0 || ids[b] == ids[a]) continue;
                const moe_cache_key ka{node->host_base, ids[a]};
                const moe_cache_key kb{node->host_base, ids[b]};
                const uint32_t n = ++device.co_activation[moe_cache_edge_undirected(ka, kb)];
                {
                    // Populated unconditionally: gating this on the eviction
                    // flag left the adjacency EMPTY at the moment the cache
                    // first fills, so every candidate scored redundancy 0 and
                    // the coverage term silently contributed nothing exactly
                    // when it mattered most. Cost is one bounded vector per
                    // expert.
                    auto add_partner = [&](const moe_cache_key & self, int32_t other) {
                        auto & v = device.partners[self];
                        if (v.size() >= MOE_CACHE_MAX_PARTNERS) return;
                        for (int32_t x : v) { if (x == other) return; }
                        v.push_back(other);
                    };
                    add_partner(ka, kb.expert);
                    add_partner(kb, ka.expert);
                }
                if (moe_cache_partner_index_enabled()) {
                    // Undirected: this count is evidence for BOTH endpoints.
                    auto & ba = device.partner_best[ka];
                    if (n > ba.second) { ba.first = kb; ba.second = n; }
                    auto & bb = device.partner_best[kb];
                    if (n > bb.second) { bb.first = ka; bb.second = n; }
                }
            }
        }
    }

    // Track 1 step 4b: cross-layer co-activation, now that
    // moe_cache_session_enter resets last_top_expert_valid at the real
    // token/batch boundary. ids[0] is used as this layer's single
    // representative expert (real top-k router outputs are rank-ordered,
    // index 0 = highest score) rather than the full selected set - the
    // combinatorial cost of tracking every cross-layer pair would be
    // (this layer's n_ids) x (previous layer's n_ids), unbounded in a way
    // step 4a's own within-layer n_ids<=32 cap doesn't cover, since it's a
    // product across two different calls, not a self-pairing within one.
    if (n_ids > 0 && ids[0] >= 0 && ids[0] < node->n_expert) {
        const moe_cache_key top_key{node->host_base, ids[0]};
        // Only record an edge across a REAL layer boundary. One logical
        // layer dispatches several expert tensors (gate_up_exps then
        // down_exps) off a single router decision, so without this check
        // that pair looks like a transition and lands as a self-edge -
        // measured at 45 of 64 exported cross-layer edges on Gemma and 55
        // of 64 on Ornith, i.e. most of the signal was this artifact.
        // Unknown layer (-1, tensor name off-convention) is treated as
        // "can't tell" and records nothing rather than guessing, which is
        // the same choice every other unknown in this file makes.
        const bool real_transition =
            device.last_top_expert_valid &&
            node->layer >= 0 && device.last_top_layer >= 0 &&
            node->layer != device.last_top_layer;
        if (real_transition) {
            // Step 6 scoring, BEFORE this observation updates the table -
            // otherwise the prediction would be graded against evidence it
            // just supplied, which would inflate it.
            if (moe_cache_measure_pred_enabled()) {
                const auto pred = device.successor_best.find(device.last_top_expert);
                if (pred != device.successor_best.end()) {
                    device.xlayer_pred_total++;
                    if (pred->second.first == top_key) {
                        device.xlayer_pred_hit++;
                    }
                    if (device.xlayer_pred_total % 2000 == 0) {
                        fprintf(stderr,
                                "[xlayer-pred] depth-1 precision %.1f%% (%lld/%lld) over %zu tracked edges\n",
                                100.0 * (double) device.xlayer_pred_hit / (double) device.xlayer_pred_total,
                                device.xlayer_pred_hit, device.xlayer_pred_total,
                                device.co_activation_cross_layer.size());
                    }
                }
            }
            const auto edge = moe_cache_edge_directed(device.last_top_expert, top_key);
            const uint32_t n = ++device.co_activation_cross_layer[edge];
            if (moe_cache_measure_pred_enabled()) {
                // O(1) index upkeep: only this edge's count changed, so it
                // can only have become the new best for its own from-expert.
                auto & best = device.successor_best[device.last_top_expert];
                if (n > best.second) {
                    best.first  = top_key;
                    best.second = n;
                }
            }
        }
        // Advance regardless: the next call's "previous layer" should be
        // this one even when this call itself recorded nothing, otherwise
        // a same-layer pair in the middle would also swallow the following
        // genuine transition.
        device.last_top_expert = top_key;
        device.last_top_layer = node->layer;
        device.last_top_expert_valid = true;
    }

    // Track 1 step 3: Atlas-driven warming. Rate-limited to once every
    // atlas_warm_interval() plan() calls (device.nodes, unconditionally
    // incremented once per call just below - deliberately not plan_epoch,
    // whose own increment above is conditional on cold-sweep being enabled)
    // so ranking a tensor's whole atlas-covered candidate set never runs on
    // every single token by default - see moe_cache_atlas_warm's own comment
    // for why this is prefetch-only and off by default.
    //
    // Burst extension: the interval, per-firing admit/evict budget, and how
    // many distinct tensors get a pass are all widened together once
    // moe_cache_atlas_burst_check reports the live req_dir centroid has
    // moved a real distance since the last check - i.e. an actual topic
    // shift in progress, not steady-state noise. Steady-state behavior
    // (interval=64, one tensor, small budget) is unchanged from the
    // original design, which measured flat-to-good there; the multi-round
    // multi-switch benchmark that motivated this found the ORIGINAL burst
    // response (2 admits, 1 eviction, one tensor, still throttled to every
    // 64 calls) too thin to beat reactive demand-driven admission to the
    // punch - this widens exactly that response without touching
    // steady-state at all.
    if (moe_cache_atlas_warm_enabled()) {
        // Record this call's tensor into the recent-tensor ring
        // unconditionally (cheap - a linear dedup scan over a capped-size
        // deque, no allocation once warmed up), so a burst detected on ANY
        // later call has real pool/size info for whichever OTHER tensors
        // were planned recently, not just whichever one trips the throttle.
        {
            auto & recent = device.atlas_recent_tensors;
            for (auto it = recent.begin(); it != recent.end(); ++it) {
                if (it->host_base == node->host_base) {
                    recent.erase(it);
                    break;
                }
            }
            recent.push_front({node->pool_index, node->host_base, node->expert_size, node->n_expert});
            static const size_t max_recent = (size_t) std::max(8, moe_cache_atlas_burst_tensor_sweep() * 2);
            while (recent.size() > max_recent) {
                recent.pop_back();
            }
        }

        const long long interval = moe_cache_atlas_warm_interval(device.atlas_burst_confidence);
        if ((long long) device.nodes - device.atlas_warm_last_calls >= interval) {
            device.atlas_warm_last_calls = (long long) device.nodes;
            const bool bursting = moe_cache_atlas_burst_check(device);
            if (bursting != device.atlas_bursting && getenv("MOE_CACHE_DEBUG_GATE")) {
                fprintf(stderr, "[atlas-burst] %s at node=%lld req_dir=(%.3f,%.3f) recent_tensors=%zu\n",
                        bursting ? "ENTER burst" : "exit burst",
                        (long long) device.nodes, device.req_dir_x, device.req_dir_y,
                        device.atlas_recent_tensors.size());
                fflush(stderr);
            }
            device.atlas_bursting = bursting;

            // Track 1.5: with ASYNC on, all this path does is hand the worker a
            // hint (a hash lookup and a push_back); the cosine scan and the
            // admission both happen there, off the thread that has to issue the
            // next GPU op. Either way the worker still has to be woken - for the
            // fills the sync path just queued, or for the ranking pass the async
            // path just deferred to it. Bursting always takes the sync,
            // multi-tensor path - the async single-tensor hint queue (one
            // deep, by design) has no room to carry a whole sweep's worth of
            // requests, and async is off by default anyway.
            const bool timing = moe_cache_atlas_warm_timing_enabled();
            const auto warm_t0 = timing ? std::chrono::steady_clock::now()
                                        : std::chrono::steady_clock::time_point{};
            bool queued = false;
            if (bursting) {
                int swept = 0;
                // Scale how many DISTINCT tensors get swept by the same
                // headroom dampener moe_cache_atlas_warm applies to each
                // individual call's own admit/rank budget - without this,
                // a dampened per-tensor budget still got spent 8 separate
                // times (once per swept tensor), which is most of why the
                // per-call dampening alone (2026-08-29 ablation) only
                // partly closed the cold-start regression: 8 small draws
                // add up to roughly the same total damage one big draw
                // does. Keyed off `pool` (the tensor that actually
                // triggered this check), same as the confidence already
                // is - the other swept tensors' own pools get their own
                // per-call dampening inside moe_cache_atlas_warm regardless.
                const float sweep_confidence = moe_cache_atlas_burst_confidence_clamp(
                        device.atlas_burst_confidence * moe_cache_atlas_burst_headroom_scale(pool));
                const int sweep_max = moe_cache_atlas_burst_tensor_sweep();
                const int sweep_cap = 1 + (int) std::lround((double) (sweep_max - 1) * sweep_confidence);
                for (const auto & rt : device.atlas_recent_tensors) {
                    if (swept >= sweep_cap) {
                        break;
                    }
                    if (rt.pool_index < 0 || (size_t) rt.pool_index >= device.pools.size()) {
                        continue; // pools are append-only, but re-check anyway - see moe_cache_atlas_warm_service
                    }
                    moe_cache_pool * rt_pool = device.pools[rt.pool_index].get();
                    if (!rt_pool || !rt_pool->slab) {
                        continue;
                    }
                    if (moe_cache_atlas_warm(device, *rt_pool, rt.pool_index, rt.host_base, rt.expert_size, rt.n_expert, device.atlas_burst_confidence)) {
                        queued = true;
                    }
                    swept++;
                }
            } else {
                queued = moe_cache_atlas_warm_async_enabled()
                    ? moe_cache_atlas_warm_enqueue(device, node->pool_index, node->host_base, node->expert_size, node->n_expert)
                    : moe_cache_atlas_warm(device, pool, node->pool_index, node->host_base, node->expert_size, node->n_expert, device.atlas_burst_confidence);
            }
            if (timing) {
                device.warm_path_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - warm_t0).count();
                if (++device.warm_path_calls % 256 == 0) {
                    fprintf(stderr, "[atlas-warm] %s: %.1f us avg on the dispatch path (%lld calls)\n",
                            moe_cache_atlas_warm_async_enabled() ? "async" : "sync",
                            (double) device.warm_path_ns / (double) device.warm_path_calls / 1000.0,
                            device.warm_path_calls);
                    fflush(stderr);
                }
            }
            if (queued) {
                wake_worker = true;
            }
        }
    }

    device.nodes++;
    lock.unlock();
    if (wake_worker) {
        session.cv.notify_all();
    }
    return hits;
}

// Heat-aware neuron subsetting: scatter the reduced-pool sub-batch's dense
// K-wide results into their correct, sparse positions within the full
// n_out-wide d_out rows the primary mmv call already produced (junk there,
// from a dummy slot-0 substitution - see the split in moe_cache_dispatch).
// Zeroes the whole row first: every position NOT in this row's own K
// selected indices is defined to contribute nothing to the following
// down_proj matmul - the same "only the neurons that matter" approximation
// this whole mechanism is built on, made explicit here instead of implicit
// in a real value. One block per reduced row, threads cooperate within it -
// n_out is typically a few thousand (moe_intermediate_size) and k a few
// hundred, both comfortably coverable by a small number of grid-stride
// passes per block.
static __global__ void moe_cache_reduce_scatter_kernel(
        float * __restrict__ d_out, int64_t n_out,
        const float * __restrict__ d_out_reduced, int32_t k,
        const int32_t * __restrict__ d_row_map,
        const int32_t * __restrict__ d_sel_idx) {
    const int r = blockIdx.x;
    const int64_t orig_row = d_row_map[r];
    float * out_row = d_out + orig_row * n_out;
    for (int64_t j = threadIdx.x; j < n_out; j += blockDim.x) {
        out_row[j] = 0.0f;
    }
    __syncthreads();
    for (int j = threadIdx.x; j < k; j += blockDim.x) {
        const int32_t idx = d_sel_idx[(size_t) r * (size_t) k + (size_t) j];
        if (idx >= 0 && (int64_t) idx < n_out) {
            out_row[idx] = d_out_reduced[(size_t) r * (size_t) k + (size_t) j];
        }
    }
}

static int moe_cache_dispatch(
        void * opaque, int wtype, int64_t n_in, int64_t n_out, int n_hits,
        const int32_t * slot_indices, const float * const * act_rows) {
    moe_cache_node * node = (moe_cache_node *)opaque;
    if (!node || !node->planned || node->dispatched || n_hits <= 0 ||
        n_hits > GGML_MOE_CACHE_MAX_BATCH_ROWS || n_hits != node->n_pins || !slot_indices || !act_rows ||
        wtype != node->wtype || n_in != node->n_in || n_out != node->n_out ||
        n_in > INT_MAX || n_out > INT_MAX ||
        n_in > INT64_MAX - (MATRIX_ROW_PADDING - 1)) {
        return 0;
    }

    // GGML_CUDA_MOE_CACHE_DISPATCH_DELAY_US=N: artificial delay right at
    // dispatch entry, before any real work. Tests directly whether merely
    // SLOWING this window (the way compute-sanitizer's per-launch
    // instrumentation does, incidentally, as a side effect of what it's
    // actually checking) reproduces corruption on its own - a fast,
    // controlled way to test a timing-dependent theory without the hours of
    // sanitizer/TSan overhead.
    {
        static const long delay_us = [] {
            const char * e = getenv("GGML_CUDA_MOE_CACHE_DISPATCH_DELAY_US");
            return e ? atol(e) : 0;
        }();
        if (delay_us > 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(delay_us));
        }
    }

    moe_cache_session & session = *node->session;
    moe_cache_device & device = *node->device;
    moe_cache_pool & pool = *node->pool;
    if (device.dead.load() || moe_cache_fail(session, "dispatch")) {
        std::lock_guard<std::mutex> lock(session.mu);
        device.dispatch_failures++;
        return 0;
    }

    ggml_cuda_set_device(device.physical);
    if (!device.compute_stream) {
        if (!moe_cache_cuda_ok(device,
                    cudaStreamCreateWithFlags(&device.compute_stream, cudaStreamNonBlocking),
                    "compute stream creation", true)) {
            std::lock_guard<std::mutex> lock(session.mu);
            device.dispatch_failures++;
            return 0;
        }
    }

    bool shared_activation = true;
    for (int index = 1; index < n_hits; index++) {
        if (act_rows[index] != act_rows[0]) {
            shared_activation = false;
            break;
        }
    }
    const int activation_rows = shared_activation ? 1 : n_hits;

    // Heat-aware neuron subsetting: a row's slot_indices value can now
    // point into EITHER node->pool (an ordinary hit) or a second,
    // reduced-width pool (see MOE_CACHE_REDUCED_SLOT_OFFSET and the
    // encoding in moe_cache_plan). Classify every row here, cross-checked
    // against node->pins[] - positionally aligned with slot_indices[],
    // since both are built by the same ids-order loop in
    // moe_cache_plan/ggml-cpu.c - rather than trusting the raw integer
    // alone, so an inconsistent slot_indices value fails dispatch instead
    // of being silently misinterpreted as pointing into the wrong pool.
    int n_reduced = 0;
    moe_cache_pool * rpool = nullptr;
    for (int index = 0; index < n_hits; index++) {
        if (!act_rows[index]) {
            return 0;
        }
        const moe_cache_pin & pin = node->pins[index];
        if (slot_indices[index] >= MOE_CACHE_REDUCED_SLOT_OFFSET) {
            const int32_t rslot = slot_indices[index] - MOE_CACHE_REDUCED_SLOT_OFFSET;
            if (!pin.pool || pin.pool == &pool || pin.slot != rslot ||
                rslot < 0 || rslot >= pin.pool->n_slots) {
                return 0;
            }
            if (rpool && rpool != pin.pool) {
                // Can only happen if this tensor somehow had two different
                // reduced pools active at once, which nothing in this file
                // does (K and wtype are both fixed per session) - refuse
                // rather than guess which one is right.
                return 0;
            }
            rpool = pin.pool;
            n_reduced++;
        } else {
            if (slot_indices[index] < 0 || slot_indices[index] >= pool.n_slots ||
                pin.pool != &pool || pin.slot != slot_indices[index]) {
                return 0;
            }
        }
    }

    const int64_t padded_n_in =
        ((n_in + MATRIX_ROW_PADDING - 1) / MATRIX_ROW_PADDING) * MATRIX_ROW_PADDING;
    const size_t type_size = ggml_type_size((ggml_type)wtype);
    if (type_size == 0 || node->expert_size % type_size != 0 ||
        ggml_row_size((ggml_type)wtype, n_in) % type_size != 0 ||
        node->expert_size / type_size > INT_MAX ||
        ggml_row_size((ggml_type)wtype, n_in) / type_size > INT_MAX ||
        padded_n_in / QK8_1 > INT_MAX ||
        (uint64_t)activation_rows * (padded_n_in / QK8_1) > INT_MAX ||
        (uint64_t)n_out * n_hits > INT_MAX ||
        (uint64_t)(node->expert_size / type_size) * pool.n_slots > INT_MAX) {
        return 0;
    }

    if (n_in > INT64_MAX / activation_rows ||
        (uint64_t)n_in * activation_rows > SIZE_MAX / sizeof(float) ||
        n_out > INT64_MAX / n_hits ||
        (uint64_t)n_out * n_hits > SIZE_MAX / sizeof(float) ||
        (uint64_t)activation_rows * (padded_n_in / QK8_1) >
            SIZE_MAX / sizeof(block_q8_1)) {
        return 0;
    }
    const size_t ids_bytes = (size_t)n_hits * sizeof(int32_t);
    const size_t act_bytes = (size_t)activation_rows * n_in * sizeof(float);
    const size_t q8_bytes =
        (size_t)activation_rows * (padded_n_in / QK8_1) * sizeof(block_q8_1);
    const size_t out_bytes = (size_t)n_hits * n_out * sizeof(float);
    node->act_bytes_used = act_bytes;
    node->q8_bytes_used = q8_bytes;
    node->out_bytes_used = out_bytes;

    // Heat-aware neuron subsetting: reduced-pool sub-batch scratch sizing.
    // Every size below is 0 when n_reduced == 0 (the common case - no
    // reduced-pool hit this call), so none of the growth calls below fire
    // and no reduced-path buffer is ever touched by a session that never
    // produces one.
    const int64_t s11_blocks = padded_n_in / QK8_1;
    const size_t row_bytes = (n_reduced > 0 && n_out > 0 &&
            node->expert_size % (size_t) n_out == 0)
        ? node->expert_size / (size_t) n_out : 0;
    const int reduce_k = (n_reduced > 0 && rpool && row_bytes > 0)
        ? (int) (rpool->expert_size / row_bytes) : 0;
    if (n_reduced > 0 && (!rpool || row_bytes == 0 || reduce_k <= 0 ||
            (size_t) reduce_k * row_bytes != rpool->expert_size)) {
        return 0;
    }
    const size_t reduced_ids_bytes     = (size_t) n_reduced * sizeof(int32_t);
    const size_t reduced_q8_bytes      = (size_t) n_reduced * (size_t) s11_blocks * sizeof(block_q8_1);
    const size_t reduced_out_bytes     = (size_t) n_reduced * (size_t) reduce_k * sizeof(float);
    const size_t reduced_row_map_bytes = (size_t) n_reduced * sizeof(int32_t);
    const size_t reduced_sel_bytes     = (size_t) n_reduced * (size_t) reduce_k * sizeof(int32_t);

    const size_t desired_caps[] = {
        moe_cache_growth_capacity(device.d_ids_cap, ids_bytes),
        moe_cache_growth_capacity(device.d_act_cap, act_bytes),
        moe_cache_growth_capacity(device.act_q8_cap, q8_bytes),
        moe_cache_growth_capacity(device.d_out_cap, out_bytes),
    };
    size_t scratch_bytes = 0;
    for (size_t capacity : desired_caps) {
        if (capacity == 0 || capacity > SIZE_MAX - scratch_bytes) {
            std::lock_guard<std::mutex> lock(session.mu);
            device.dispatch_failures++;
            return 0;
        }
        scratch_bytes += capacity;
    }
    // Separate loop, only entered when n_reduced > 0: moe_cache_growth_capacity
    // returns 0 both for "already sufficient at 0 bytes required" and for
    // "overflow", and the primary loop above relies on 0 always meaning
    // failure - reusing it for entries that are legitimately 0 whenever
    // n_reduced == 0 would reject every ordinary dispatch call.
    if (n_reduced > 0) {
        const size_t reduced_caps[] = {
            moe_cache_growth_capacity(device.d_ids_reduced_cap, reduced_ids_bytes),
            moe_cache_growth_capacity(device.act_q8_reduced_cap, reduced_q8_bytes),
            moe_cache_growth_capacity(device.d_out_reduced_cap, reduced_out_bytes),
            moe_cache_growth_capacity(device.d_reduce_row_map_cap, reduced_row_map_bytes),
            moe_cache_growth_capacity(device.d_reduce_sel_idx_cap, reduced_sel_bytes),
        };
        for (size_t capacity : reduced_caps) {
            if (capacity == 0 || capacity > SIZE_MAX - scratch_bytes) {
                std::lock_guard<std::mutex> lock(session.mu);
                device.dispatch_failures++;
                return 0;
            }
            scratch_bytes += capacity;
        }
    }
    {
        std::lock_guard<std::mutex> lock(session.mu);
        if (scratch_bytes > device.budget_limit ||
            device.allocated_bytes > device.budget_limit - scratch_bytes) {
            device.dispatch_failures++;
            return 0;
        }
    }

    const size_t d_act_cap_before   = device.d_act_cap;
    const size_t act_q8_cap_before  = device.act_q8_cap;
    const size_t d_out_cap_before   = device.d_out_cap;

    if (!moe_cache_grow_host(device, (void **)&device.h_ids, device.h_ids_cap,
                             ids_bytes, "ids host allocation") ||
        !moe_cache_grow_device(device, (void **)&device.d_ids, device.d_ids_cap,
                               ids_bytes, "ids device allocation") ||
        !moe_cache_grow_host(device, (void **)&device.h_act, device.h_act_cap,
                             act_bytes, "activation host allocation") ||
        !moe_cache_grow_device(device, (void **)&device.d_act, device.d_act_cap,
                               act_bytes, "activation device allocation") ||
        !moe_cache_grow_device(device, &device.d_act_q8, device.act_q8_cap,
                               q8_bytes, "q8 activation allocation") ||
        !moe_cache_grow_device(device, (void **)&device.d_out, device.d_out_cap,
                               out_bytes, "output device allocation") ||
        !moe_cache_grow_host(device, (void **)&device.h_out, device.h_out_cap,
                             out_bytes, "output host allocation")) {
        std::lock_guard<std::mutex> lock(session.mu);
        device.dispatch_failures++;
        device.dead.store(true);
        return 0;
    }
    // Reduced-path growth failure is NOT fatal to the whole cache (unlike
    // the primary scratch above): these buffers are tiny (bounded by
    // n_reduced <= GGML_MOE_CACHE_MAX_BATCH_ROWS and reduce_k, typically
    // low hundreds of KiB at most), so a failure here is far more likely to
    // be transient allocator pressure than a real out-of-memory condition -
    // failing just this one dispatch call (which safely falls back to the
    // ordinary CPU path in ggml-cpu.c for every row of the batch) is the
    // proportionate response, not killing an otherwise-healthy cache.
    if (n_reduced > 0 && (
        !moe_cache_grow_host(device, (void **)&device.h_ids_reduced, device.h_ids_reduced_cap,
                             reduced_ids_bytes, "reduced ids host allocation") ||
        !moe_cache_grow_device(device, (void **)&device.d_ids_reduced, device.d_ids_reduced_cap,
                               reduced_ids_bytes, "reduced ids device allocation") ||
        !moe_cache_grow_device(device, &device.d_act_q8_reduced, device.act_q8_reduced_cap,
                               reduced_q8_bytes, "reduced q8 activation allocation") ||
        !moe_cache_grow_device(device, (void **)&device.d_out_reduced, device.d_out_reduced_cap,
                               reduced_out_bytes, "reduced output device allocation") ||
        !moe_cache_grow_host(device, (void **)&device.h_reduce_row_map, device.h_reduce_row_map_cap,
                             reduced_row_map_bytes, "reduce row-map host allocation") ||
        !moe_cache_grow_device(device, (void **)&device.d_reduce_row_map, device.d_reduce_row_map_cap,
                               reduced_row_map_bytes, "reduce row-map device allocation") ||
        !moe_cache_grow_host(device, (void **)&device.h_reduce_sel_idx, device.h_reduce_sel_idx_cap,
                             reduced_sel_bytes, "reduce sel-idx host allocation") ||
        !moe_cache_grow_device(device, (void **)&device.d_reduce_sel_idx, device.d_reduce_sel_idx_cap,
                               reduced_sel_bytes, "reduce sel-idx device allocation"))) {
        std::lock_guard<std::mutex> lock(session.mu);
        device.dispatch_failures++;
        return 0;
    }
    // A (re)growth repoisons the WHOLE buffer with canary (moe_cache_grow_device),
    // so the high-water mark of real-data bytes written resets to 0 along with it -
    // otherwise the guard below would compare against a stale hwm from before the
    // realloc and treat freshly-poisoned canary bytes as unreachable "checkable" ones.
    if (device.d_act_cap != d_act_cap_before) {
        device.d_act_hwm = 0;
    }
    if (device.act_q8_cap != act_q8_cap_before) {
        device.act_q8_hwm = 0;
    }
    if (device.d_out_cap != d_out_cap_before) {
        device.d_out_hwm = 0;
    }

    for (int index = 0; index < n_hits; index++) {
        // Reduced rows get a dummy, always-valid primary slot (0 - every
        // pool that exists has at least 64 slots, see moe_cache_allocate_pool)
        // so the primary mmv call below can still run as ONE ordinary batched
        // call over the full n_hits, unchanged in every other respect - its
        // result for a reduced row is discarded, overwritten below by the
        // reduced mmv call + scatter kernel. This avoids compacting/
        // reordering the primary rows at all.
        device.h_ids[index] = slot_indices[index] >= MOE_CACHE_REDUCED_SLOT_OFFSET
            ? 0 : slot_indices[index];
    }
    for (int index = 0; index < activation_rows; index++) {
        memcpy(device.h_act + (size_t)index * n_in,
               act_rows[shared_activation ? 0 : index], n_in * sizeof(float));
    }

    // Compact per-reduced-row bookkeeping: which absolute row of the batch
    // it is (for the scatter kernel to write to the right place in d_out)
    // and its expert's chosen K neuron indices (for the scatter kernel to
    // write each computed value to the right column). Looked up from
    // device.reduced_indices under session.mu - see that field's comment
    // for why this is never touched from any thread other than the compute
    // thread except under this lock.
    if (n_reduced > 0) {
        int c = 0;
        for (int index = 0; index < n_hits && c < n_reduced; index++) {
            if (slot_indices[index] < MOE_CACHE_REDUCED_SLOT_OFFSET) {
                continue;
            }
            const moe_cache_pin & pin = node->pins[index];
            device.h_ids_reduced[c] = pin.slot;
            device.h_reduce_row_map[c] = index;
            bool filled = false;
            {
                std::lock_guard<std::mutex> lock(session.mu);
                const moe_cache_key rkey = pin.pool->slots[pin.slot].key;
                const auto it = device.reduced_indices.find(rkey);
                if (it != device.reduced_indices.end() &&
                    (int) it->second.idx.size() == reduce_k) {
                    for (int j = 0; j < reduce_k; j++) {
                        device.h_reduce_sel_idx[(size_t) c * reduce_k + j] = it->second.idx[j];
                    }
                    filled = true;
                }
            }
            if (!filled) {
                // Should not happen - a valid, pinned reduced-pool slot only
                // exists after a successful conversion recorded its indices
                // (moe_cache_try_reduce_convert records reduced_indices
                // BEFORE queuing the fill job that brings the slot to
                // `valid`). Fail safe rather than scatter into undefined
                // positions.
                std::lock_guard<std::mutex> lock(session.mu);
                device.dispatch_failures++;
                return 0;
            }
            c++;
        }
    }

    if (getenv("GGML_CUDA_MOE_CACHE_ACT_MAGNITUDE_LOG")) {
        static std::atomic<int> logged{0};
        float amax = 0.0f;
        size_t nan_count = 0, zero_count = 0;
        const size_t total = (size_t)activation_rows * n_in;
        for (size_t i = 0; i < total; i++) {
            const float v = device.h_act[i];
            if (std::isnan(v)) { nan_count++; continue; }
            if (v == 0.0f) { zero_count++; }
            // std::max silently ignores NaN (NaN < x is always false), so NaN
            // values were previously invisible here - counted separately above,
            // excluded from amax explicitly rather than relying on max's NaN
            // behaviour to do the right thing by accident.
            amax = std::max(amax, std::fabs(v));
        }
        if (nan_count > 0) {
            fprintf(stderr, "[moe-cache] ACT MAGNITUDE layer=%d NaN in activation itself: "
                    "%zu/%zu elements are NaN (not the dot-product output - the INPUT)\n",
                    node->layer, nan_count, total);
            fflush(stderr);
        }
        // FP16 max finite magnitude ~65504; the GPU's q8_1 quantizer stores its
        // per-block scale (amax/127) as a half, so any block whose amax alone
        // already exceeds ~65504*127 turns that scale into +Inf when cast down -
        // check the raw activation magnitude here, upstream of any quantization,
        // against that threshold directly.
        if (amax > 60000.0f && logged.fetch_add(1) < 20) {
            fprintf(stderr, "[moe-cache] ACT MAGNITUDE layer=%d amax=%g n_hits=%d "
                    "activation_rows=%d n_in=%lld (fp16 quantizer scale overflow risk above ~65504*127)\n",
                    node->layer, (double) amax, n_hits, activation_rows, (long long) n_in);
            fflush(stderr);
        } else if (node->layer >= 20) {
            static std::atomic<int> quiet{0};
            if (quiet.fetch_add(1) < 40) {
                fprintf(stderr, "[moe-cache] ACT MAGNITUDE layer=%d amax=%g zero_count=%zu/%zu nan_count=%zu (sample)\n",
                        node->layer, (double) amax, zero_count, total, nan_count);
            }
        }
    }

    (void)cudaGetLastError();
    bool ok =
        moe_cache_cuda_ok(device, cudaMemcpyAsync(
                device.d_ids, device.h_ids, ids_bytes,
                cudaMemcpyHostToDevice, device.compute_stream), "ids upload", true) &&
        moe_cache_cuda_ok(device, cudaMemcpyAsync(
                device.d_act, device.h_act, act_bytes,
                cudaMemcpyHostToDevice, device.compute_stream), "activation upload", true);
    if (ok) {
        quantize_row_q8_1_cuda(
                device.d_act, nullptr, device.d_act_q8, (ggml_type)wtype,
                n_in, n_in, (int64_t)activation_rows * n_in,
                (int64_t)activation_rows * n_in, padded_n_in,
                activation_rows, 1, 1, device.compute_stream);
        ok = moe_cache_cuda_ok(
                device, cudaPeekAtLastError(), "activation quantization", true);
    }
    if (ok) {
        ggml_cuda_moe_cache_mmv(
                pool.slab, (ggml_type)wtype, (const char *)device.d_act_q8,
                device.d_ids, device.d_out, n_in, n_out, pool.n_slots,
                (int64_t)pool.slot_stride, n_hits, activation_rows,
                device.compute_stream);
        ok = moe_cache_cuda_ok(
                device, cudaPeekAtLastError(), "expert matvec launch", true);
    }

    // Heat-aware neuron subsetting: the reduced-pool subset of this batch
    // (if any) needs its own matvec call - one launch can only address one
    // pool/slab (see docs/plan.md, "ggml_mul_mat_id batching constraint") -
    // followed by a scatter of its dense K-wide results into the sparse
    // positions of the SAME d_out rows the call above already produced
    // (junk, from the dummy slot-0 substitution) for those rows.
    if (ok && n_reduced > 0) {
        ok = moe_cache_cuda_ok(device, cudaMemcpyAsync(
                    device.d_ids_reduced, device.h_ids_reduced, reduced_ids_bytes,
                    cudaMemcpyHostToDevice, device.compute_stream),
                "reduced ids upload", true) &&
            moe_cache_cuda_ok(device, cudaMemcpyAsync(
                    device.d_reduce_row_map, device.h_reduce_row_map, reduced_row_map_bytes,
                    cudaMemcpyHostToDevice, device.compute_stream),
                "reduce row-map upload", true) &&
            moe_cache_cuda_ok(device, cudaMemcpyAsync(
                    device.d_reduce_sel_idx, device.h_reduce_sel_idx, reduced_sel_bytes,
                    cudaMemcpyHostToDevice, device.compute_stream),
                "reduce sel-idx upload", true);
    }
    if (ok && n_reduced > 0) {
        // Compact the reduced rows' quantized activation out of the
        // already-quantized device.d_act_q8 (built for the FULL batch just
        // above) via one small device-to-device copy per reduced row -
        // correct whether or not shared_activation is true (a shared
        // source is just the same source row copied n_reduced times).
        // Bounded by n_reduced <= GGML_MOE_CACHE_MAX_BATCH_ROWS; a real
        // scatter/gather kernel would remove this loop, but a reduced-pool
        // sub-batch is a fraction of an already-small decode batch, and
        // correctness came first for this landing - see docs/plan.md's
        // "no tok/s claim" note on this mechanism.
        const size_t s11_bytes = (size_t) s11_blocks * sizeof(block_q8_1);
        for (int c = 0; c < n_reduced && ok; c++) {
            const int orig_row = device.h_reduce_row_map[c];
            const size_t src_off = (shared_activation ? 0 : (size_t) orig_row) * s11_bytes;
            ok = moe_cache_cuda_ok(device, cudaMemcpyAsync(
                        (char *) device.d_act_q8_reduced + (size_t) c * s11_bytes,
                        (const char *) device.d_act_q8 + src_off, s11_bytes,
                        cudaMemcpyDeviceToDevice, device.compute_stream),
                    "reduced activation compact", true);
        }
    }
    if (ok && n_reduced > 0) {
        ggml_cuda_moe_cache_mmv(
                rpool->slab, (ggml_type)wtype, (const char *)device.d_act_q8_reduced,
                device.d_ids_reduced, device.d_out_reduced, n_in, (int64_t) reduce_k,
                rpool->n_slots, (int64_t) rpool->slot_stride, n_reduced, n_reduced,
                device.compute_stream);
        ok = moe_cache_cuda_ok(
                device, cudaPeekAtLastError(), "reduced expert matvec launch", true);
    }
    if (ok && n_reduced > 0) {
        moe_cache_reduce_scatter_kernel<<<n_reduced, 256, 0, device.compute_stream>>>(
                device.d_out, n_out, device.d_out_reduced, reduce_k,
                device.d_reduce_row_map, device.d_reduce_sel_idx);
        ok = moe_cache_cuda_ok(
                device, cudaPeekAtLastError(), "reduce scatter launch", true);
    }

    // GGML_CUDA_MOE_CACHE_VERIFY_MMV=1: run the SAME rows through the SAME
    // kernel one at a time and diff against the batched result. Same weights,
    // same activations, same slots - only the batching differs, so any
    // disagreement is the batching itself. This is the instrument for the
    // open corruption bug, where max_batch=1 is clean and every value >= 2
    // corrupts.
    // Verify the slots AT DISPATCH TIME, not at plan time. plan() pins each
    // slot (readers++) and checks it then, but the fill worker runs
    // concurrently - if it copies a new expert into a slot that is pinned and
    // about to be read, the weights change underneath the kernel. That race
    // would be invisible to every check made at plan time, and would get more
    // likely as more rows are in flight, which matches max_batch=1 being clean.
    static const bool verify_at_dispatch = [] {
        const char * env = getenv("GGML_CUDA_MOE_CACHE_VERIFY_DISPATCH");
        return env && atoi(env) != 0;
    }();
    if (ok && verify_at_dispatch) {
        std::lock_guard<std::mutex> lock(session.mu);
        static std::atomic<int> reported{0};
        for (int i = 0; i < n_hits; i++) {
            const int si = slot_indices[i];
            if (si < 0 || si >= (int) pool.slots.size()) continue;
            const moe_cache_slot & slot = pool.slots[si];
            if (slot.state != moe_cache_slot_state::valid || slot.readers <= 0) {
                if (reported.fetch_add(1) < 20) {
                    fprintf(stderr, "[moe-cache] DISPATCH SLOT UNPINNED/INVALID row=%d slot=%d "
                            "state=%d readers=%d\n", i, si, (int) slot.state, slot.readers);
                }
                continue;
            }
            moe_cache_verify_slot(pool, si, slot.key, device.physical);
        }
    }

    static const bool verify_mmv = [] {
        const char * env = getenv("GGML_CUDA_MOE_CACHE_VERIFY_MMV");
        return env && atoi(env) != 0;
    }();
    if (ok && verify_mmv && n_hits > 1) {
        const int64_t ne10_padded = ((n_in + MATRIX_ROW_PADDING - 1) / MATRIX_ROW_PADDING) * MATRIX_ROW_PADDING;
        const int64_t s11_blocks = ne10_padded / QK8_1;
        float * d_ref = nullptr;
        std::vector<float> batched((size_t) n_hits * n_out), single((size_t) n_hits * n_out);
        if (cudaMalloc((void **) &d_ref, (size_t) n_hits * n_out * sizeof(float)) == cudaSuccess) {
            for (int i = 0; i < n_hits; i++) {
                const char * act_i = (const char *) device.d_act_q8 +
                    (activation_rows > 1 ? (size_t) i * s11_blocks * sizeof(block_q8_1) : 0);
                ggml_cuda_moe_cache_mmv(
                        pool.slab, (ggml_type)wtype, act_i,
                        device.d_ids + i, d_ref + (size_t) i * n_out,
                        n_in, n_out, pool.n_slots, (int64_t)pool.slot_stride,
                        /*n_hits=*/1, /*act_rows=*/1, device.compute_stream);
            }
            cudaMemcpyAsync(batched.data(), device.d_out, batched.size()*sizeof(float),
                            cudaMemcpyDeviceToHost, device.compute_stream);
            cudaMemcpyAsync(single.data(), d_ref, single.size()*sizeof(float),
                            cudaMemcpyDeviceToHost, device.compute_stream);
            cudaStreamSynchronize(device.compute_stream);
            static std::atomic<int> reported{0};
            for (int i = 0; i < n_hits; i++) {
                double worst = 0.0; int worst_c = -1;
                for (int64_t c = 0; c < n_out; c++) {
                    const double d = std::fabs(batched[(size_t)i*n_out+c] - single[(size_t)i*n_out+c]);
                    if (d > worst) { worst = d; worst_c = (int) c; }
                }
                if (worst > 1e-3 && reported.fetch_add(1) < 20) {
                    fprintf(stderr, "[moe-cache] MMV BATCH MISMATCH row=%d/%d slot=%d "
                            "act_rows=%d col=%d batched=%.6f single=%.6f absdiff=%.6f\n",
                            i, n_hits, slot_indices[i], activation_rows, worst_c,
                            batched[(size_t)i*n_out+worst_c], single[(size_t)i*n_out+worst_c], worst);
                }
            }
            cudaFree(d_ref);
        }
    }

    if (!ok) {
        cudaStreamSynchronize(device.compute_stream);
        std::lock_guard<std::mutex> lock(session.mu);
        device.dispatch_failures++;
        return 0;
    }

    node->dispatched = true;
    return 1;
}

static int moe_cache_collect(
        void * opaque, int n_hits, float * const * dst_rows, int64_t n_out) {
    moe_cache_node * node = (moe_cache_node *)opaque;
    if (!node || !node->dispatched || n_hits <= 0 ||
        n_hits > GGML_MOE_CACHE_MAX_BATCH_ROWS ||
        n_hits != node->n_pins || !dst_rows || n_out != node->n_out) {
        return 0;
    }
    for (int index = 0; index < n_hits; index++) {
        if (!dst_rows[index]) {
            return 0;
        }
    }

    moe_cache_session & session = *node->session;
    moe_cache_device & device = *node->device;
    ggml_cuda_set_device(device.physical);

    bool ok = !device.dead.load();
    if (moe_cache_fail(session, "collect")) {
        ok = false;
    }
    const size_t bytes = (size_t)n_hits * n_out * sizeof(float);
    if (ok) {
        ok = moe_cache_cuda_ok(device, cudaMemcpyAsync(
                device.h_out, device.d_out, bytes,
                cudaMemcpyDeviceToHost, device.compute_stream), "output download", true);
    }

    // Sample each pinned slot's guard region (the dead, canary-filled gap
    // right after its real expert_size bytes) into host memory, batched
    // onto the same stream/sync as the output download above - one extra
    // async copy, no extra sync. Checked after the sync below. A mismatch
    // here is direct, physical proof that something wrote past this slot's
    // own bounds during this dispatch, independent of whatever the exact
    // mechanism turns out to be.
    static constexpr size_t GUARD_SAMPLE_BYTES = 64;
    // Costs n_hits+3 extra cudaMemcpyAsync launches per collect() call (real
    // per-call driver overhead, even though they share the existing sync) -
    // was worth it while this session was actively chasing the `////...`
    // corruption bug (root-caused and fixed in ggml-backend.cpp - see
    // docs/plan.md), not something a production hot path should pay by
    // default now that it's closed. GGML_CUDA_MOE_CACHE_GUARD_CHECK=1
    // re-enables both this and the scratch-buffer guard below, e.g. to rule
    // out an out-of-bounds write as the cause of some future regression.
    static const bool guard_check_enabled = [] {
        const char * e = getenv("GGML_CUDA_MOE_CACHE_GUARD_CHECK");
        return e && atoi(e) != 0;
    }();
    // Checkability is now per-pin, not a single flag for the whole call:
    // each pin carries its own pool (moe_cache_pin), and heat-aware neuron
    // subsetting means that pool can be either node->pool or the
    // reduced-width pool, which can have its own (possibly different, or
    // absent) guard gap.
    std::vector<char> guard_sample;
    std::vector<bool> guard_checked(n_hits, false);
    const bool guard_checkable = guard_check_enabled && ok;
    if (guard_checkable) {
        guard_sample.resize((size_t) n_hits * GUARD_SAMPLE_BYTES);
        for (int index = 0; index < n_hits; index++) {
            const moe_cache_pin & pin = node->pins[index];
            const moe_cache_pool * gpool = pin.pool;
            if (!gpool || pin.slot < 0 || pin.slot >= gpool->n_slots || !gpool->slab ||
                gpool->slot_stride <= gpool->expert_size ||
                (gpool->slot_stride - gpool->expert_size) < GUARD_SAMPLE_BYTES) {
                continue;
            }
            const char * guard_src = gpool->slab +
                (size_t) pin.slot * gpool->slot_stride + gpool->expert_size;
            cudaMemcpyAsync(guard_sample.data() + (size_t) index * GUARD_SAMPLE_BYTES,
                    guard_src, GUARD_SAMPLE_BYTES, cudaMemcpyDeviceToHost, device.compute_stream);
            guard_checked[index] = true;
        }
    }

    // Same idea for the scratch buffers quantization and the matvec kernel
    // actually write into (d_act, d_act_q8, d_out) - these are grown via
    // moe_cache_grow_device, which poisons the whole allocation on every
    // (re)growth. But growth is amortised: most dispatches reuse an
    // already-large-enough buffer without regrowing it, and successive calls'
    // `used` sizes vary with n_hits - so bytes beyond THIS call's `used` are
    // only guaranteed to still be pristine canary if this call reached at
    // least as deep as every call since the last growth (tracked in
    // device.*_hwm). Otherwise that region holds a PRIOR call's real,
    // legitimately-written data, not canary - flagging it would be a false
    // "breach" against leftover data that nothing overran.
    char scratch_guard[3][GUARD_SAMPLE_BYTES];
    bool scratch_guard_checkable[3] = {false, false, false};
    if (ok && guard_check_enabled) {
        struct { const char * base; size_t used; size_t cap; size_t * hwm; } bufs[3] = {
            { (const char *) device.d_act,    node->act_bytes_used, device.d_act_cap,  &device.d_act_hwm },
            { (const char *) device.d_act_q8, node->q8_bytes_used,  device.act_q8_cap, &device.act_q8_hwm },
            { (const char *) device.d_out,    node->out_bytes_used, device.d_out_cap,  &device.d_out_hwm },
        };
        for (int b = 0; b < 3; b++) {
            const bool at_high_water = bufs[b].used >= *bufs[b].hwm;
            if (bufs[b].used > *bufs[b].hwm) {
                *bufs[b].hwm = bufs[b].used;
            }
            if (at_high_water && bufs[b].base && bufs[b].used > 0 &&
                bufs[b].used + GUARD_SAMPLE_BYTES <= bufs[b].cap) {
                scratch_guard_checkable[b] = true;
                cudaMemcpyAsync(scratch_guard[b], bufs[b].base + bufs[b].used,
                        GUARD_SAMPLE_BYTES, cudaMemcpyDeviceToHost, device.compute_stream);
            }
        }
    }

    if (ok) {
        ok = moe_cache_cuda_ok(
                device, cudaStreamSynchronize(device.compute_stream),
                "output synchronization", true);
    } else {
        cudaStreamSynchronize(device.compute_stream);
    }
    node->dispatched = false;

    if (guard_checkable && ok) {
        static std::atomic<int> greported{0};
        for (int index = 0; index < n_hits; index++) {
            if (!guard_checked[index]) {
                continue;
            }
            const moe_cache_pin & pin = node->pins[index];
            const char * sample = guard_sample.data() + (size_t) index * GUARD_SAMPLE_BYTES;
            for (size_t b = 0; b < GUARD_SAMPLE_BYTES; b++) {
                if ((unsigned char) sample[b] != MOE_CACHE_GUARD_PATTERN) {
                    if (greported.fetch_add(1) < 20) {
                        fprintf(stderr, "[moe-cache] GUARD BREACH slot=%d layer=%d byte_offset=+%zu "
                                "expected=0x%02x actual=0x%02x - something wrote past this slot's "
                                "own bounds\n",
                                pin.slot, node->layer, b,
                                (unsigned) MOE_CACHE_GUARD_PATTERN, (unsigned char) sample[b]);
                        fflush(stderr);
                    }
                    ok = false;
                    break;
                }
            }
        }
    }

    if (ok) {
        static const char * const scratch_names[3] = { "d_act", "d_act_q8", "d_out" };
        static std::atomic<int> sreported{0};
        for (int b = 0; b < 3; b++) {
            if (!scratch_guard_checkable[b]) {
                continue;
            }
            for (size_t i = 0; i < GUARD_SAMPLE_BYTES; i++) {
                if ((unsigned char) scratch_guard[b][i] != MOE_CACHE_GUARD_PATTERN) {
                    if (sreported.fetch_add(1) < 20) {
                        fprintf(stderr, "[moe-cache] SCRATCH GUARD BREACH buf=%s layer=%d "
                                "byte_offset=+%zu expected=0x%02x actual=0x%02x - something wrote "
                                "past this call's own scratch usage\n",
                                scratch_names[b], node->layer, i,
                                (unsigned) MOE_CACHE_GUARD_PATTERN, (unsigned char) scratch_guard[b][i]);
                        fflush(stderr);
                    }
                    ok = false;
                    break;
                }
            }
        }
    }

    // GGML_CUDA_MOE_CACHE_GEN_CHECK=1: prove or rule out the race the
    // fill-worker-vs-pinned-read comment in moe_cache_dispatch warns about,
    // at zero extra sync cost - the cudaStreamSynchronize above already
    // happened unconditionally, so by the time we get here the matvec kernel
    // has already read every pinned slot's weights; comparing generation now
    // just checks whether the fill worker rewrote a slot out from under a
    // reader that plan() believed was protected by readers>0.
    static const bool gen_check = [] {
        const char * e = getenv("GGML_CUDA_MOE_CACHE_GEN_CHECK");
        return e && atoi(e) != 0;
    }();
    if (ok && gen_check) {
        std::lock_guard<std::mutex> lock(session.mu);
        static std::atomic<int> reported{0};
        for (int index = 0; index < n_hits; index++) {
            const moe_cache_pin & pin = node->pins[index];
            if (!pin.pool || pin.slot < 0 || pin.slot >= pin.pool->n_slots) {
                continue;
            }
            const moe_cache_slot & slot = pin.pool->slots[pin.slot];
            if (slot.generation != pin.generation && reported.fetch_add(1) < 20) {
                fprintf(stderr, "[moe-cache] GENERATION MISMATCH row=%d/%d slot=%d "
                        "pinned_gen=%llu now_gen=%llu readers=%d\n",
                        index, n_hits, pin.slot,
                        (unsigned long long) pin.generation,
                        (unsigned long long) slot.generation, slot.readers);
                fflush(stderr);
            }
        }
    }

    if (ok) {
        for (int index = 0; index < n_hits; index++) {
            memcpy(dst_rows[index], device.h_out + (size_t)index * n_out,
                   n_out * sizeof(float));
        }
    }

    // Per-neuron heat: read-only over data moe_cache_collect already
    // downloaded and validated above (h_out, post-sync) - accumulates
    // |value| per neuron per (tensor, expert) key, keyed the same way as
    // everything else here. gate_exps/up_exps calls land here with
    // n_out == moe_intermediate_size; down_exps calls also pass through
    // (n_out == hidden size) and get their own, separate heat vector under
    // the same key - harmless, just an unused signal for now, since only
    // gate/up are ever meant to be read from this map.
    // Keys whose heat vector was just updated this call AND are not yet a
    // decided reduce-conversion candidate (device.reduce_decided, checked
    // right here under the neuron_heat_mu this loop already holds - see
    // that field's comment for why this fast check, not
    // moe_cache_try_reduce_convert itself, is what runs on every hit).
    // Checked for reduce-conversion eligibility AFTER neuron_heat_mu is
    // released below (moe_cache_try_reduce_convert needs session.mu, and
    // the documented order is session.mu -> neuron_heat_mu, never nested
    // the other way).
    std::vector<moe_cache_key> heat_touched_keys;
    const bool reduce_enabled_this_call = moe_cache_neuron_reduce_enabled() && node->reduce_eligible;
    if (ok && moe_cache_neuron_heat_enabled()) {
        // device.neuron_heat_mu, NOT session.mu - see that field's comment
        // for why (measured regression from contending with the cache's
        // core bookkeeping lock). Reading pin.pool->slots[pin.slot].key
        // without session.mu is safe here specifically: the pin mechanism
        // already guarantees a pinned slot's key cannot change out from
        // under this call (that is the whole point of pinning), and
        // pool.slots itself is a fixed-size array sized once at pool
        // creation - never resized - so no other thread's write to a
        // DIFFERENT slot can invalidate this read. Each pin carries its OWN
        // pool now (see moe_cache_pin) rather than assuming node->pool -
        // heat-aware neuron subsetting introduced pins into a second
        // (reduced-width) pool.
        std::lock_guard<std::mutex> lock(device.neuron_heat_mu);
        for (int index = 0; index < n_hits; index++) {
            const moe_cache_pin & pin = node->pins[index];
            if (!pin.pool || pin.slot < 0 || pin.slot >= pin.pool->n_slots) {
                continue;
            }
            const moe_cache_key key = pin.pool->slots[pin.slot].key;
            std::vector<float> & heat = device.neuron_heat[key];
            if ((int64_t) heat.size() != n_out) {
                heat.assign((size_t) n_out, 0.0f);
            }
            const float * row = device.h_out + (size_t) index * n_out;
            for (int64_t j = 0; j < n_out; j++) {
                heat[(size_t) j] += fabsf(row[j]);
            }
            if (reduce_enabled_this_call && device.reduce_decided.find(key) == device.reduce_decided.end()) {
                heat_touched_keys.push_back(key);
            }
        }
    }
    if (ok && reduce_enabled_this_call) {
        for (const moe_cache_key & key : heat_touched_keys) {
            moe_cache_try_reduce_convert(
                    session, device, key, n_out, node->expert_size, node->wtype, node->n_expert);
        }
    }

    {
        std::lock_guard<std::mutex> lock(session.mu);
        if (!ok) {
            device.collect_failures++;
        }
        device.collect_calls++;
        if (session.config.stats_every > 0 &&
            device.collect_calls % session.config.stats_every == 0) {
            moe_cache_log_stats(device);
        }
    }
    return ok ? 1 : 0;
}

static void moe_cache_end(void * opaque) {
    std::unique_ptr<moe_cache_node> node((moe_cache_node *)opaque);
    if (!node) {
        return;
    }

    if (node->dispatched) {
        ggml_cuda_set_device(node->device->physical);
        moe_cache_cuda_ok(
                *node->device, cudaStreamSynchronize(node->device->compute_stream),
                "end synchronization", true);
        node->dispatched = false;
    }

    moe_cache_session & session = *node->session;
    const bool trim = node->device->dead.load();
    {
        std::lock_guard<std::mutex> lock(session.mu);
        for (int index = 0; index < node->n_pins; index++) {
            const moe_cache_pin & pin = node->pins[index];
            if (pin.pool && pin.slot >= 0 && pin.slot < pin.pool->n_slots) {
                moe_cache_slot & slot = pin.pool->slots[pin.slot];
                if (slot.readers > 0) {
                    slot.readers--;
                }
            }
        }
        auto source = session.active_sources.find(node->host_base);
        if (source != session.active_sources.end()) {
            if (--source->second.references == 0) {
                session.active_sources.erase(source);
            }
        }
        session.active_nodes--;
        session.idle_cv.notify_all();
    }
    if (trim && node->dispatch_lock.owns_lock()) {
        node->dispatch_lock.unlock();
        moe_cache_trim_session(session, node->device->physical);
    }
}

static void moe_cache_invalidate_session(
        moe_cache_session & session, const void * base, size_t size) {
    std::unique_lock<std::mutex> lock(session.mu);
    session.tensor_devices.erase(base);
    for (auto & device_ptr : session.devices) {
        moe_cache_cancel_queue_locked(*device_ptr, base, size, false);
    }

    session.idle_cv.wait(lock, [&] {
        for (const auto & active : session.active_sources) {
            if (active.second.references > 0 &&
                moe_cache_ranges_overlap(active.first, active.second.bytes, base, size)) {
                return false;
            }
        }
        for (const auto & device_ptr : session.devices) {
            if (device_ptr->inflight &&
                moe_cache_ranges_overlap(
                    device_ptr->inflight_source, device_ptr->inflight_bytes, base, size)) {
                return false;
            }
        }
        return true;
    });

    for (auto & device_ptr : session.devices) {
        moe_cache_cancel_queue_locked(*device_ptr, base, size, false);
        moe_cache_device & device = *device_ptr;
        for (auto & pool_ptr : device.pools) {
            moe_cache_pool & pool = *pool_ptr;
            for (int index = 0; index < pool.n_slots; index++) {
                moe_cache_slot & slot = pool.slots[index];
                const void * source = slot.key.expert >= 0
                    ? (const char *)slot.key.tensor +
                        (size_t)slot.key.expert * pool.expert_size
                    : nullptr;
                if (slot.state != moe_cache_slot_state::free &&
                    moe_cache_ranges_overlap(source, pool.expert_size, base, size)) {
                    moe_cache_slot_reset(pool, index, true);
                }
            }
        }
        // Heat-aware neuron subsetting: the generic per-pool loop just
        // above computes each slot's source range as
        // key.expert*pool.expert_size, which is correct for every ordinary
        // pool but WRONG for a reduced-width pool - pool.expert_size there
        // is the REDUCED size, not the real per-expert byte span in the
        // source tensor a reduced slot was gathered from (a scattered
        // subset of a much larger range). Left alone, that means a
        // reduced-pool slot could survive an invalidation event it should
        // have been caught by. device.reduced_indices carries the real
        // (src_expert_size) span for exactly this reason - use it here
        // instead, for the reduced pool specifically.
        for (auto it = device.reduced_indices.begin(); it != device.reduced_indices.end();) {
            const moe_cache_key & rkey = it->first;
            const auto & entry = it->second;
            const void * source = rkey.expert >= 0 && entry.src_expert_size > 0
                ? (const char *) rkey.tensor + (size_t) rkey.expert * entry.src_expert_size
                : nullptr;
            if (moe_cache_ranges_overlap(source, entry.src_expert_size, base, size)) {
                if (entry.pool_index >= 0 && entry.pool_index < (int) device.pools.size()) {
                    moe_cache_pool & rpool = *device.pools[entry.pool_index];
                    const auto mit = rpool.map.find(rkey);
                    if (mit != rpool.map.end()) {
                        moe_cache_slot_reset(rpool, mit->second, true);
                    }
                }
                {
                    std::lock_guard<std::mutex> hlock(device.neuron_heat_mu);
                    device.reduce_decided.erase(rkey);
                }
                it = device.reduced_indices.erase(it);
            } else {
                ++it;
            }
        }
        for (auto it = device.seen_tensors.begin(); it != device.seen_tensors.end();) {
            if (moe_cache_ranges_overlap(it->first, it->second.bytes, base, size)) {
                session.tensor_devices.erase(it->first);
                for (moe_cache_shape & shape : device.shapes) {
                    if (shape.expert_size == it->second.expert_size &&
                        shape.wtype == it->second.wtype) {
                        shape.n_tensors = std::max<int64_t>(shape.n_tensors - 1, 0);
                        if (shape.n_tensors == 0 && shape.pool < 0) {
                            shape.finished = false;
                        }
                        break;
                    }
                }
                it = device.seen_tensors.erase(it);
                device.stable_visits = 0;
            } else {
                ++it;
            }
        }
        for (auto it = device.demand_count.begin(); it != device.demand_count.end();) {
            const void * source = it->first.expert >= 0
                ? (const char *)it->first.tensor +
                    (size_t)it->first.expert * it->second.expert_size
                : nullptr;
            if (moe_cache_ranges_overlap(
                    source, it->second.expert_size, base, size)) {
                it = device.demand_count.erase(it);
            } else {
                ++it;
            }
        }
    }
}

static void moe_cache_invalidate(const void * base, size_t size) {
    if (!base || size == 0 ||
        g_session_count.load(std::memory_order_acquire) == 0) {
        return;
    }
    std::lock_guard<std::mutex> registry_lock(g_registry_mu);
    for (moe_cache_session * session : g_sessions) {
        moe_cache_invalidate_session(*session, base, size);
    }
}

static size_t moe_cache_trim_session(
        moe_cache_session & session, int physical_device) {
    moe_cache_device * selected = nullptr;
    for (auto & device_ptr : session.devices) {
        if (device_ptr->physical == physical_device) {
            selected = device_ptr.get();
            break;
        }
    }
    if (!selected) {
        return 0;
    }

    std::unique_lock<std::mutex> dispatch_lock(selected->dispatch_mu);
    std::unique_lock<std::mutex> lock(session.mu);
    selected->dead.store(true);
    moe_cache_cancel_queue_locked(*selected, nullptr, 0, true);
    session.cv.notify_all();
    session.idle_cv.wait(lock, [&] {
        return !selected->inflight;
    });

    size_t freed = 0;
    for (const auto & pool_ptr : selected->pools) {
        if (pool_ptr->slab) {
            freed += (size_t)pool_ptr->n_slots * pool_ptr->slot_stride;
        }
    }
    freed += selected->d_out_cap + selected->d_act_cap +
             selected->act_q8_cap + selected->d_ids_cap;
    lock.unlock();
    moe_cache_free_device(*selected);

    if (freed > 0) {
        MOE_CACHE_LOG("[moe-cache] CUDA%d trimmed %zu MiB after a cache failure or allocator pressure\n",
                physical_device, freed >> 20);
    }
    return freed;
}

extern "C" size_t ggml_moe_cache_trim(int device) {
    if (g_session_count.load(std::memory_order_acquire) == 0) {
        return 0;
    }
    size_t freed = 0;
    std::lock_guard<std::mutex> registry_lock(g_registry_mu);
    for (moe_cache_session * session : g_sessions) {
        freed += moe_cache_trim_session(*session, device);
    }
    return freed;
}

static void moe_cache_get_stats(long long * out_hits, long long * out_misses) {
    long long hits = 0, misses = 0;
    if (g_session_count.load(std::memory_order_acquire) > 0) {
        std::lock_guard<std::mutex> registry_lock(g_registry_mu);
        for (moe_cache_session * session : g_sessions) {
            for (const auto & device_ptr : session->devices) {
                hits += device_ptr->hits;
                misses += device_ptr->misses;
            }
        }
    }
    if (out_hits)   *out_hits = hits;
    if (out_misses) *out_misses = misses;
}

// Snapshot the live per-(layer,expert) cache state for a debugging/demo UI
// view (the "Brain" page, tools/ui - ported from Colibri's, see
// docs/moe-cache-colibri-notes.md). One byte per (layer,expert) cell: top 2
// bits = tier (0=not cached right now, 1=probation/warm, 2=protected/hot),
// bottom 6 bits = heat, saturated. Read-only and best-effort: takes each
// session's mutex only briefly per session, same as the hot dispatch path
// already does, so this competes for the same lock but doesn't add a new
// one. Grid shape comes from bookkeeping populated in moe_cache_begin();
// if no session has cached anything yet, returns 0 with rows=cols=0 - the
// caller (the server's /experts handler) treats that as "not ready yet",
// not an error.
static int moe_cache_get_expert_map(uint8_t * out_bytes, int max_bytes, int * out_rows, int * out_cols) {
    if (out_rows) *out_rows = 0;
    if (out_cols) *out_cols = 0;
    if (g_session_count.load(std::memory_order_acquire) == 0) {
        return 0;
    }
    std::lock_guard<std::mutex> registry_lock(g_registry_mu);
    for (moe_cache_session * session : g_sessions) {
        std::lock_guard<std::mutex> lock(session->mu);
        if (session->tensor_layer.empty() || session->n_expert_hint <= 0) {
            continue;
        }
        int n_rows = 0;
        for (const auto & kv : session->tensor_layer) {
            n_rows = std::max(n_rows, kv.second + 1);
        }
        const int n_cols = (int) std::min<int64_t>(session->n_expert_hint, INT_MAX);
        if (n_rows <= 0 || n_cols <= 0) {
            continue;
        }
        const long long need = (long long) n_rows * (long long) n_cols;
        if (need > max_bytes) {
            // Buffer too small for this session's grid - report the real
            // shape anyway so the caller can size a retry, but don't write
            // out of bounds.
            if (out_rows) *out_rows = n_rows;
            if (out_cols) *out_cols = n_cols;
            return 0;
        }
        std::fill(out_bytes, out_bytes + need, (uint8_t) 0);
        for (const auto & kv : session->tensor_layer) {
            const void * host_base = kv.first;
            const int    row       = kv.second;
            for (int expert = 0; expert < n_cols; expert++) {
                const moe_cache_key key{host_base, expert};
                for (const auto & device_ptr : session->devices) {
                    for (const auto & pool_ptr : device_ptr->pools) {
                        auto found = pool_ptr->map.find(key);
                        if (found == pool_ptr->map.end()) {
                            continue;
                        }
                        const moe_cache_slot & slot = pool_ptr->slots[found->second];
                        if (slot.state != moe_cache_slot_state::valid) {
                            continue;
                        }
                        const uint8_t tier = slot.segment == moe_cache_segment::protected_ ? 2 : 1;
                        // MOE_CACHE_HEAT_MAX is 2^20; a slot realistically
                        // saturates the 6-bit display range long before
                        // that (64 steps of 4 = 256 hits), which is exactly
                        // the point - this is a "how hot lately" indicator,
                        // not a raw counter.
                        const uint8_t heat = (uint8_t) std::min<uint32_t>(63, slot.heat / 4);
                        out_bytes[(size_t) row * n_cols + expert] = (uint8_t)((tier << 6) | heat);
                        goto next_expert;
                    }
                }
                next_expert:;
            }
        }
        if (out_rows) *out_rows = n_rows;
        if (out_cols) *out_cols = n_cols;
        return 1;
    }
    return 0;
}

// Brain UI substitute overlay: which (layer,expert) cells served as a
// resident stand-in since the last poll, as a BITSET (1 bit/cell, same
// shape and hex-encoding convention the server already uses for its
// diffed hit-bitmask). Unlike moe_cache_get_expert_map (a live snapshot,
// re-readable any time with the same answer), substitution events are
// READ-AND-CLEARED: each call reports what happened since the previous
// call, then empties every device's substituted_recently set - correct
// for exactly one poller, which is what the /experts handler is.
static int moe_cache_get_substitute_map(uint8_t * out_bits, int max_bytes, int * out_rows, int * out_cols) {
    if (out_rows) *out_rows = 0;
    if (out_cols) *out_cols = 0;
    if (g_session_count.load(std::memory_order_acquire) == 0) {
        return 0;
    }
    std::lock_guard<std::mutex> registry_lock(g_registry_mu);
    for (moe_cache_session * session : g_sessions) {
        std::lock_guard<std::mutex> lock(session->mu);
        if (session->tensor_layer.empty() || session->n_expert_hint <= 0) {
            continue;
        }
        int n_rows = 0;
        for (const auto & kv : session->tensor_layer) {
            n_rows = std::max(n_rows, kv.second + 1);
        }
        const int n_cols = (int) std::min<int64_t>(session->n_expert_hint, INT_MAX);
        if (n_rows <= 0 || n_cols <= 0) {
            continue;
        }
        const long long n_cells = (long long) n_rows * (long long) n_cols;
        const long long need = (n_cells + 7) / 8;
        if (need > max_bytes) {
            // Same convention as get_expert_map: report the real shape so
            // the caller can size a retry, write nothing out of bounds.
            if (out_rows) *out_rows = n_rows;
            if (out_cols) *out_cols = n_cols;
            return 0;
        }
        std::fill(out_bits, out_bits + need, (uint8_t) 0);
        for (auto & device_ptr : session->devices) {
            for (const auto & key : device_ptr->substituted_recently) {
                const auto rit = session->tensor_layer.find(key.tensor);
                if (rit == session->tensor_layer.end() || key.expert < 0 || key.expert >= n_cols) {
                    continue;
                }
                const long long i = (long long) rit->second * n_cols + key.expert;
                out_bits[i >> 3] |= (uint8_t) (1u << (i & 7));
            }
            device_ptr->substituted_recently.clear();
        }
        if (out_rows) *out_rows = n_rows;
        if (out_cols) *out_cols = n_cols;
        return 1;
    }
    return 0;
}

// Live per-expert concentration, for the Brain/Atlas view (dot size, not
// color - color already carries tier + substitute state). Deliberately a
// CHEAP O(n) proxy (dead-fraction: share of neurons below 1% of this
// expert's own mean), not the full sorted "required width for 90%
// coverage" curve moe_cache_neuron_heat_report computes - that needs an
// O(n log n) sort per key, fine for a one-time end-of-session dump, not
// for something the Brain UI polls every ~1.5s across potentially
// thousands of keys. Same shape convention as moe_cache_get_expert_map/
// moe_cache_get_substitute_map (session->tensor_layer for rows,
// n_expert_hint for cols), but writes a float per cell instead of a bit -
// -1.0 for a cell with no neuron_heat data yet (never touched, or
// GGML_CUDA_MOE_CACHE_NEURON_HEAT not enabled), 0.0-1.0 dead-fraction
// otherwise.
static int moe_cache_get_neuron_concentration_map(float * out_values, int max_floats, int * out_rows, int * out_cols) {
    if (out_rows) *out_rows = 0;
    if (out_cols) *out_cols = 0;
    if (!moe_cache_neuron_heat_enabled() || g_session_count.load(std::memory_order_acquire) == 0) {
        return 0;
    }
    std::lock_guard<std::mutex> registry_lock(g_registry_mu);
    for (moe_cache_session * session : g_sessions) {
        std::lock_guard<std::mutex> lock(session->mu);
        if (session->tensor_layer.empty() || session->n_expert_hint <= 0) {
            continue;
        }
        int n_rows = 0;
        for (const auto & kv : session->tensor_layer) {
            n_rows = std::max(n_rows, kv.second + 1);
        }
        const int n_cols = (int) std::min<int64_t>(session->n_expert_hint, INT_MAX);
        if (n_rows <= 0 || n_cols <= 0) {
            continue;
        }
        const long long n_cells = (long long) n_rows * (long long) n_cols;
        if (n_cells > max_floats) {
            if (out_rows) *out_rows = n_rows;
            if (out_cols) *out_cols = n_cols;
            return 0;
        }
        std::fill(out_values, out_values + n_cells, -1.0f);
        for (auto & device_ptr : session->devices) {
            std::lock_guard<std::mutex> heat_lock(device_ptr->neuron_heat_mu);
            for (const auto & [key, vec] : device_ptr->neuron_heat) {
                const auto rit = session->tensor_layer.find(key.tensor);
                if (rit == session->tensor_layer.end() || key.expert < 0 || key.expert >= n_cols ||
                    vec.size() < 8) {
                    continue;
                }
                double sum = 0.0;
                for (float v : vec) sum += v;
                if (sum <= 0.0) {
                    continue;
                }
                const double mean = sum / (double) vec.size();
                const double thresh = 0.01 * mean;
                size_t n_dead = 0;
                for (float v : vec) {
                    if ((double) v < thresh) n_dead++;
                }
                const long long i = (long long) rit->second * n_cols + key.expert;
                out_values[i] = (float) n_dead / (float) vec.size();
            }
        }
        if (out_rows) *out_rows = n_rows;
        if (out_cols) *out_cols = n_cols;
        return 1;
    }
    return 0;
}

// Aggregate cache health, summed across every currently-live session's
// devices - the exact same fields moe_cache_log_stats() already computes
// per device (MOE_CACHE_LOG), just returned instead of only logged. Used
// by the Brain view's stats sidepanel (tools/ui) and available generally
// for anyone else who wants real numbers instead of grepping server logs.
static void moe_cache_get_summary(ggml_moe_cache_summary * out) {
    ggml_moe_cache_summary sum{};
    unsigned long long heat_sum_total = 0;
    size_t heat_n_total = 0;
    if (g_session_count.load(std::memory_order_acquire) > 0) {
        std::lock_guard<std::mutex> registry_lock(g_registry_mu);
        for (moe_cache_session * session : g_sessions) {
            std::lock_guard<std::mutex> lock(session->mu);
            for (const auto & device_ptr : session->devices) {
                const moe_cache_device & device = *device_ptr;
                sum.hits            += device.hits;
                sum.misses          += device.misses;
                sum.evictions       += device.evictions;
                sum.fill_failures   += device.fill_failures;
                sum.admission_skips += device.admission_skips;
                sum.prefetches      += device.prefetches;
                sum.substitutions       += device.substitutions;
                sum.substitute_declined += device.substitute_declined;
                for (int r = 0; r < GGML_MOE_CACHE_MAX_RANK; r++) {
                    sum.rank_hits[r]   += device.rank_hits[r];
                    sum.rank_misses[r] += device.rank_misses[r];
                }
                sum.allocated_bytes += device.allocated_bytes;
                sum.budget_bytes    += device.budget_limit;

                if (device.req_dir_valid) {
                    sum.req_dir_x     = device.req_dir_x;
                    sum.req_dir_y     = device.req_dir_y;
                    sum.req_dir_valid = 1;
                }

                for (const auto & pool_ptr : device.pools) {
                    const moe_cache_pool & pool = *pool_ptr;
                    // GGML_CUDA_MOE_CACHE_MASK_AUDIT=1 verifies the resident
                    // bitmask still agrees with pool.map, both directions.
                    // Runs here rather than on the dispatch path because it
                    // walks every slot and every set bit - fine on a stats
                    // poll, ruinous per plan() call. Silent when clean.
                    if (moe_cache_mask_audit_enabled()) {
                        const int bad = moe_cache_mask_audit(pool);
                        if (bad > 0) {
                            // fprintf, not MOE_CACHE_LOG: the server installs a
                            // ggml log callback that drops GGML_LOG_INFO, so the
                            // macro is invisible here - the same reason the
                            // "first fill" line above writes to stderr directly.
                            fprintf(stderr, "[moe-cache] MASK AUDIT FAILED: %d disagreement(s) between "
                                    "resident_mask and map in pool %zu B/expert wtype=%d\n",
                                    bad, pool.expert_size, pool.wtype);
                        } else {
                            fprintf(stderr, "[moe-cache] mask audit clean: %zu tensor(s), %zu mapped experts\n",
                                    pool.resident_mask.size(), pool.map.size());
                        }
                    }
                    sum.slots_total += (size_t) pool.n_slots;
                    sum.slots_used  += (size_t) pool.n_slots - pool.free_slots.size();
                    for (int i = pool.protected_head; i >= 0; i = pool.slots[i].next) {
                        sum.protected_slots++;
                        heat_sum_total += pool.slots[i].heat;
                        heat_n_total++;
                    }
                    for (int i = pool.lru_head; i >= 0; i = pool.slots[i].next) {
                        heat_sum_total += pool.slots[i].heat;
                        heat_n_total++;
                    }
                }
            }
        }
    }
    // Single division over the true combined sum/count across every device,
    // not an average-of-per-device-averages (which would silently misweight
    // devices with different slot counts).
    sum.avg_heat = heat_n_total ? (double) heat_sum_total / (double) heat_n_total : 0.0;
    *out = sum;
}


// ---------------------------------------------------------------------------
// Host hot-expert buffer: own the memory the hot set lives in.
//
// mmap and the page cache stay exactly as they are - they remain the backing
// store, they still serve everything not promoted, and the load path is
// untouched. What changes is that the *hot* experts are copied into memory this
// cache owns, where LFRU is enforced rather than merely advised. That is the one
// thing no amount of madvise could deliver: the kernel interface is
// demotion-only, with no way to say "protect this", so under pressure it evicts
// a expert that fires every token as readily as one never selected. Measured,
// that costs 21x.
//
// Promotion releases the corresponding mmap pages (MADV_DONTNEED). Without that
// the same weights are resident twice - once in the page cache, once here - and
// the duplication costs exactly what this was built to save. That step is what
// makes the two mechanisms a mixture rather than a stack: each holds the part it
// is good at, and neither holds the other's part.
//
// Budget is derived, never constant, on the same rule already used for the VRAM
// budget and the prompt cache: a share of what is actually available, clamped by
// any cgroup limit, because a fixed number is wrong on both a 16GB laptop and a
// 1TB server. GGML_CUDA_MOE_CACHE_HOST_MB overrides; 0 disables.
static size_t moe_cache_host_budget_bytes() {
    static const size_t value = [] () -> size_t {
        if (const char * env = getenv("GGML_CUDA_MOE_CACHE_HOST_MB")) {
            const long long parsed = atoll(env);
            return parsed <= 0 ? 0 : (size_t) parsed << 20;
        }
        return 0; // opt-in: this allocates real host memory, never by surprise
    }();
    return value;
}

// Release the mmap pages behind an expert now that we hold our own copy.
// Page-aligned inward so a partial page shared with a neighbouring expert is
// never discarded.
static void moe_cache_release_source_pages(const void * base, size_t offset, size_t bytes) {
#if defined(__linux__)
    static const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) {
        return;
    }
    const uintptr_t begin = (uintptr_t) base + offset;
    const uintptr_t end   = begin + bytes;
    const uintptr_t first = (begin + page_size - 1) & ~((uintptr_t) page_size - 1);
    const uintptr_t last  = end & ~((uintptr_t) page_size - 1);
    if (last > first) {
        // DONTNEED on a clean file-backed mapping simply drops the pages; the
        // data is still on disk and re-faults if this expert is ever demoted.
        madvise((void *) first, (size_t) (last - first), MADV_DONTNEED);
    }
#else
    (void) base; (void) offset; (void) bytes;
#endif
}

// Called under session.mu from the planning path, where heat has just been
// updated. Promotes `expert` into the host buffer if it is hot enough and there
// is room, evicting the coldest resident expert when there is not.
// Runs on the fill worker with the session lock released. The only shared
// state it mutates is the atomic slot pointer (published after the copy) and
// the byte counters, which only this thread writes.
static void moe_cache_host_promote_locked_free(
        moe_cache_device & device, moe_cache_device::cpu_residency & res,
        const void * host_base, int expert, size_t expert_size) {
    const size_t budget = moe_cache_host_budget_bytes();
    if (budget == 0 || expert_size == 0 || (size_t) expert >= res.host_slot.size()) {
        return;
    }
    if (res.host_slot[expert].load(std::memory_order_relaxed) != nullptr) {
        return; // already held
    }

    // Make room by evicting colder experts, and only ones genuinely colder than
    // the candidate - otherwise a cold newcomer churns a hot incumbent out on
    // every miss. Freeing here releases host memory immediately, so the page
    // cache gets it back rather than the buffer sitting on a reservation.
    while (device.host_bytes + expert_size > budget) {
        uint32_t victim_heat = UINT32_MAX;
        moe_cache_device::cpu_residency * victim_res = nullptr;
        size_t victim_expert = 0;
        for (auto & [vb, vres] : device.residency) {
            for (size_t e = 0; e < vres.host_slot.size(); e++) {
                if (vres.host_slot[e].load(std::memory_order_relaxed) == nullptr) {
                    continue;
                }
                if (vres.selections[e] < victim_heat) {
                    victim_heat  = vres.selections[e];
                    victim_res   = &vres;
                    victim_expert = e;
                }
            }
        }
        if (!victim_res || victim_heat >= res.selections[expert]) {
            return; // nothing colder than the candidate - leave it to mmap
        }
        void * old = victim_res->host_slot[victim_expert].exchange(nullptr, std::memory_order_acq_rel);
        if (!old) {
            return;
        }
        // A CPU thread may still be mid-read of this pointer. Freeing it now
        // would be a use-after-free, so evicted blocks are retired rather than
        // released - see moe_cache_host_retire below.
        moe_cache_host_retire(device, old, victim_res->expert_size);
    }

    void * copy = std::malloc(expert_size);
    if (!copy) {
        return;
    }
    std::memcpy(copy, (const char *) host_base + (size_t) expert * expert_size, expert_size);
    // Publish only after the copy is complete: a reader that observes the
    // pointer must find valid weights behind it.
    res.host_slot[expert].store(copy, std::memory_order_release);
    device.host_bytes += expert_size;
    device.host_promoted++;

    // Now that we hold our own copy, let the kernel reclaim the mapped pages.
    // This is what keeps the two mechanisms complementary rather than additive:
    // without it the same weights are resident twice and the buffer costs
    // exactly what it was built to save.
    moe_cache_release_source_pages(host_base, (size_t) expert * expert_size, expert_size);

    if (device.host_promoted == 1) {
        MOE_CACHE_LOG("[moe-cache] host hot-expert buffer active (budget %zu MiB, grown on promotion)\n",
                budget >> 20);
    }
}

// Retire an evicted block. Released later, when no dispatch is in flight and
// the queue is drained, which is the same quiescence condition the scratch
// release uses. Bounded: if the list grows large the oldest are freed first.
static void moe_cache_host_retire(moe_cache_device & device, void * block, size_t bytes) {
    device.host_bytes -= std::min(device.host_bytes, bytes);
    device.host_retired.emplace_back(block, bytes);
    if (device.host_retired.size() > 256 && !device.inflight && device.queue.empty()) {
        for (auto & [p, n] : device.host_retired) {
            std::free(p);
        }
        device.host_retired.clear();
    }
}

// Hot path, called per expert per MoE node from CPU compute threads. The
// atomic host_slot offset itself is lock-free by construction (published via
// release, read via acquire) - but reaching it requires a `.find()` on
// device.residency, which plan() mutates (via operator[], which can rehash)
// under session->mu on the decode thread. Reading the map under a DIFFERENT
// lock (g_registry_mu only guards session/device list membership, not each
// session's own map) let that find() race a concurrent insert - undefined
// behavior on the map's bucket structure, not just a staleness risk, and the
// exact shape of "CPU compute reads a garbage pointer as expert weight
// bytes" this cache exists to avoid. Each session's own residency map is now
// read under that session's own session->mu, matching every other access to
// it. Only taken when this feature is active (the budget==0 fast path above
// returns before any lock), so the default-off case stays lock-free.
static const void * moe_cache_host_ptr(const void * host_base, int expert) {
    if (moe_cache_host_budget_bytes() == 0 ||
        g_session_count.load(std::memory_order_acquire) <= 0 || expert < 0) {
        return nullptr;
    }
    std::lock_guard<std::mutex> registry_lock(g_registry_mu);
    for (moe_cache_session * session : g_sessions) {
        std::lock_guard<std::mutex> lock(session->mu);
        for (const auto & device_ptr : session->devices) {
            auto it = device_ptr->residency.find(host_base);
            if (it == device_ptr->residency.end()) {
                continue;
            }
            if ((size_t) expert >= it->second.host_slot.size()) {
                continue;
            }
            void * held = it->second.host_slot[expert].load(std::memory_order_acquire);
            if (held) {
                return held;
            }
        }
    }
    return nullptr;
}


// Speculative prefetch, from a router run one layer ahead of execution.
//
// The fill machinery is the same one misses already use; the difference is
// timing. A reactive fill is issued after the miss that needed it, too late for
// the token that caused it. This is issued before, so the transfer overlaps the
// current layer's compute - which is the entire point, and why the literature
// treats prefetch as a pipeline stage rather than a bigger cache.
//
// Two rules keep speculation honest:
//  - it only ever takes free slots, never evicting a resident expert. At ~77%
//    accuracy a wrong guess that displaced a known-hot expert would cost more
//    than a right one saves.
//  - it respects the same queue bounds as demand fills, so a bad prediction
//    cannot starve real work.
static void moe_cache_prefetch(const void * host_base, const int32_t * ids, int n_ids, int depth) {
    static const bool enabled = [] {
        const char * env = getenv("GGML_CUDA_MOE_CACHE_PREFETCH");
        return !env || atoi(env) != 0; // on unless explicitly disabled
    }();
    if (getenv("MOE_CACHE_DEBUG_GATE")) {
        static std::atomic<long long> n{0};
        fprintf(stderr, "[router-lookahead-dbg] host_base=%p n_ids=%d enabled=%d call=%lld\n",
                host_base, n_ids, (int) enabled, (long long) ++n);
    }
    if (!enabled || !host_base || !ids || n_ids <= 0 ||
        g_session_count.load(std::memory_order_acquire) <= 0) {
        return;
    }

    std::lock_guard<std::mutex> registry_lock(g_registry_mu);
    for (moe_cache_session * session : g_sessions) {
        std::unique_lock<std::mutex> lock(session->mu, std::try_to_lock);
        if (!lock.owns_lock() || session->stopping) {
            continue; // never block decode for a guess
        }
        for (const auto & device_ptr : session->devices) {
            moe_cache_device & device = *device_ptr;
            if (device.dead.load()) {
                continue;
            }
            auto seen = device.seen_tensors.find(host_base);
            if (seen == device.seen_tensors.end()) {
                continue;
            }
            const size_t expert_size = seen->second.expert_size;
            const int    pool_index  = moe_cache_find_pool(device, expert_size, seen->second.wtype);
            if (pool_index < 0) {
                continue;
            }
            moe_cache_pool & pool = *device.pools[pool_index];

            bool woke = false;
            for (int i = 0; i < n_ids; i++) {
                const int32_t expert = ids[i];
                if (expert < 0) {
                    continue;
                }
                if ((int) device.queue.size() >= session->config.queue_max) {
                    break;
                }
                // "any": the original policy - evict for any single speculative
                // guess, rate-limited only by count. Measured harmful (-35% /
                // -16% / -15% across three regimes, see docs) because the harm
                // is churn frequency, not eviction-target quality - kept only
                // for A/B comparison against "agree" below, not recommended.
                // "agree": only evict when a *closer, more-accurate* depth's
                // prediction lands on the same expert a farther depth already
                // wanted but could not get a free slot for this same decode
                // step - two independent router-state samples agreeing, which
                // the measured per-depth precision (41.1% / 48.0% / 59.3% for
                // depth 3/2/1) says should be much rarer and much more often
                // right than either depth alone. Default off either way.
                enum class spec_evict_mode { off, any, agree };
                static const spec_evict_mode mode = [] {
                    const char * env = getenv("GGML_CUDA_MOE_CACHE_SPEC_EVICT_MODE");
                    if (!env) {
                        // back-compat with the older on/off-only flag
                        const char * legacy = getenv("GGML_CUDA_MOE_CACHE_SPEC_EVICT");
                        if (legacy) {
                            return atoi(legacy) != 0 ? spec_evict_mode::any : spec_evict_mode::off;
                        }
                        // Default: agree, not off. Safe even though it's now
                        // the default - agree can only ever fire when a
                        // farther depth exists to confirm against
                        // (GGML_CUDA_MOE_LOOKAHEAD_DEPTH>=2), and that itself
                        // still defaults to 1, so this is a no-op until the
                        // caller opts into deeper lookahead. any is never
                        // defaulted - measured harmful in every regime tested.
                        return spec_evict_mode::agree;
                    }
                    if (!strcmp(env, "any")) return spec_evict_mode::any;
                    if (!strcmp(env, "agree")) return spec_evict_mode::agree;
                    return spec_evict_mode::off;
                }();
                const moe_cache_key key{host_base, expert};
                int slot_index = -1;
                if (!pool.free_slots.empty()) {
                    slot_index = pool.free_slots.back();
                    pool.free_slots.pop_back();
                } else if (mode == spec_evict_mode::off) {
                    break; // default: no eviction for speculation - measured harmful
                            // under no real memory pressure (probation churn); gate
                            // this on to re-test once demand genuinely exceeds
                            // capacity, where the calculus may differ.
                } else {
                    // Reset once per real decode step (device.collect_calls) -
                    // shared clock for both the rate limit and the agreement
                    // memory below, so a stale guess from tokens ago can never
                    // count as "agreement" with a fresh one.
                    if (device.spec_evict_reset_at_calls != device.collect_calls) {
                        device.spec_evict_reset_at_calls = device.collect_calls;
                        device.spec_evictions_this_cycle = 0;
                        device.spec_seen_this_cycle.clear();
                    }
                    if (mode == spec_evict_mode::agree) {
                        auto & seen = device.spec_seen_this_cycle;
                        auto it = std::find_if(seen.begin(), seen.end(),
                                [&](const std::pair<moe_cache_key, int> & p) { return p.first == key; });
                        // Corroborated only if a *strictly closer* (smaller,
                        // more accurate) depth than whatever suggested it
                        // before now confirms it - not merely seen twice. A
                        // same-or-farther-depth repeat is not new evidence
                        // (see the struct comment on spec_seen_this_cycle for
                        // why this distinction is the depth=3 fix).
                        const bool corroborated = it != seen.end() && depth < it->second;
                        if (!corroborated) {
                            if (it == seen.end()) {
                                // First sighting this step: remember it (and
                                // its depth) in case a later, closer call
                                // lands on the same expert.
                                if (seen.size() < 64) {
                                    seen.emplace_back(key, depth);
                                }
                            }
                            // else: a same-or-farther-depth repeat of an
                            // already-seen, still-unconfirmed candidate - no
                            // new information, leave the closer depth already
                            // on record as the one that still needs confirming.
                            break; // uncorroborated: behave exactly like mode=off
                        }
                        seen.erase(it); // confirmed - remove so it can't double-fire this step
                    }
                    // Rate-limited regardless of mode: uncapped, this fires up
                    // to n_expert_used times per layer per token (measured
                    // harmful under mode=any - see above). Capped small so
                    // eviction is reserved for genuinely infrequent swaps.
                    static const long long spec_evict_cap = [] {
                        const char * env = getenv("GGML_CUDA_MOE_CACHE_SPEC_EVICT_CAP");
                        const long long v = env ? atoll(env) : 4;
                        return v < 0 ? 0 : v;
                    }();
                    if (device.spec_evictions_this_cycle >= spec_evict_cap) {
                        break; // budget spent for this token - fall back to free-only
                    }
                    device.spec_evictions_this_cycle++;
                    // No free slot: evict the least-preferred probation-tier
                    // occupant, by the same heat criterion that decides who
                    // gets to STAY (moe_cache_pick_coldest_unpinned - shared
                    // with the real-miss path). Protected is never touched
                    // here, on purpose: an item earns protected status via a
                    // real hit or a cross-depth-confirmed prediction (see the
                    // promote_to_protected call above, on repeated prefetch
                    // agreement) - exactly the "keep many loaded" half of the
                    // design. Speculative admission may only rotate the
                    // unconfirmed probation tier, never something already
                    // proven wanted.
                    const int candidate = moe_cache_pick_coldest_unpinned(device, pool, pool.lru_head);
                    if (candidate < 0) {
                        break; // probation has nothing left to sacrifice
                    }
                    moe_cache_slot_reset(pool, candidate, false);
                    slot_index = candidate;
                    device.evictions++;
                }

                // Feed the same prediction to the host buffer. Its promotion
                // signal was per-expert heat - backward-looking, and it tried to
                // hold the whole hot set, which is the capacity approach that
                // measured worse three times. A prediction says what is about to
                // be needed, which turns the buffer into a small staging area
                // for imminent work rather than a cache of what was recently
                // popular: the same capacity-to-timing shift, one tier down.
                //
                // Queued for the worker rather than copied here - this runs on
                // the decode path and a 1.4 MiB memcpy inline is exactly what
                // made the buffer slow before.
                if (moe_cache_host_budget_bytes() > 0) {
                    auto res_it = device.residency.find(host_base);
                    if (res_it != device.residency.end() &&
                        (size_t) expert < res_it->second.host_slot.size() &&
                        res_it->second.host_slot[expert].load(std::memory_order_relaxed) == nullptr &&
                        device.host_promote_queue.size() < 64) {
                        device.host_promote_queue.emplace_back(host_base, expert);
                        woke = true;
                    }
                }

                {
                    auto existing = pool.map.find(key);
                    if (existing != pool.map.end()) {
                        // A closer, fresher-depth prediction re-targeting what a
                        // farther, weaker-depth guess already brought in (or
                        // queued) is real evidence, not a duplicate to discard.
                        // Every layer re-predicts on its own turn (build_moe_
                        // lookahead fires at every layer), so by the time we are
                        // one layer out a candidate may have been suggested by
                        // depth 3, then depth 2, then depth 1 - each strictly
                        // more accurate (measured: 41.1% / 48.0% / 59.3%
                        // precision) than the last. Treat repeated agreement the
                        // same way a real re-request already does: promote it,
                        // so it survives the probation churn other one-off
                        // speculative admissions are exposed to, rather than
                        // silently doing nothing with the stronger signal.
                        moe_cache_slot & existing_slot = pool.slots[existing->second];
                        if (existing_slot.state == moe_cache_slot_state::valid) {
                            moe_cache_promote_to_protected(device, pool, existing->second);
                        }
                        continue; // already resident or already in flight
                    }
                }

                moe_cache_slot & slot = pool.slots[slot_index];
                slot.key = key;
                slot.generation++;
                slot.readers = 0;
                slot.state = moe_cache_slot_state::copying;
                try {
                    if (!moe_cache_map_insert(pool, key, slot_index)) {
                        moe_cache_slot_reset(pool, slot_index, true);
                        continue;
                    }
                    device.queue.push_back({
                            pool_index, slot_index, slot.generation, key,
                            (const char *) host_base + (size_t) expert * expert_size,
                            expert_size});
                    device.queued_bytes += expert_size;
                    device.prefetches++;
                    woke = true;
                } catch (...) {
                    moe_cache_slot_reset(pool, slot_index, true);
                }
            }
            if (woke) {
                session->cv.notify_all();
            }
        }
    }
}

// Bandwidth profiling: standalone vs. contended H2D throughput for a
// realistic expert-sized transfer. Standalone-then-subtract badly
// over/under-penalizes a real fetch (see FreeToken's benchbw.py
// measure_overlap_bw for the same lesson in a different contention pair,
// CPU compute vs. PCIe gather) - the number that actually matters is what
// bandwidth a fetch achieves while the GPU is ALSO busy, since that's the
// condition every real re-fetch of an evicted expert happens under. This
// is the missing cost term a future cost-aware (GreedyDual-Size-style)
// eviction policy needs - "how expensive, in real achieved time, is it to
// re-fetch this expert" - not yet wired into eviction decisions, which
// still use plain LFRU recency; this only measures and reports.
struct moe_cache_bandwidth_profile {
    double standalone_gbs = 0.0;
    double contended_gbs = 0.0;
};

static bool moe_cache_measure_bandwidth(moe_cache_device & device, moe_cache_bandwidth_profile & out) {
    ggml_cuda_set_device(device.physical);

    const size_t sample_bytes = 256u << 20;    // realistic gate_up-sized expert tensor
    const size_t contender_bytes = 512u << 20; // kept busy on its own stream throughout

    void * h_src = nullptr;
    if (cudaMallocHost(&h_src, sample_bytes) != cudaSuccess) {
        (void) cudaGetLastError();
        return false;
    }
    void * d_dst = nullptr;
    void * d_contender_a = nullptr;
    void * d_contender_b = nullptr;
    cudaStream_t measure_stream = nullptr;
    cudaStream_t contend_stream = nullptr;
    const bool ok =
            cudaMalloc(&d_dst, sample_bytes) == cudaSuccess &&
            cudaMalloc(&d_contender_a, contender_bytes) == cudaSuccess &&
            cudaMalloc(&d_contender_b, contender_bytes) == cudaSuccess &&
            cudaStreamCreateWithFlags(&measure_stream, cudaStreamNonBlocking) == cudaSuccess &&
            cudaStreamCreateWithFlags(&contend_stream, cudaStreamNonBlocking) == cudaSuccess;
    if (!ok) {
        (void) cudaGetLastError();
        if (d_dst) cudaFree(d_dst);
        if (d_contender_a) cudaFree(d_contender_a);
        if (d_contender_b) cudaFree(d_contender_b);
        if (measure_stream) cudaStreamDestroy(measure_stream);
        if (contend_stream) cudaStreamDestroy(contend_stream);
        cudaFreeHost(h_src);
        return false;
    }

    // contend: when true, re-queues one D2D copy on contend_stream right
    // before each H2D copy on measure_stream, so the contention traffic is
    // interleaved with (not just issued before) the measured window - an
    // earlier version queued all the D2D contention up front, which could
    // finish draining before the H2D measurement even started and measured
    // near-zero contention effect as a result (100% of standalone,
    // confirmed live). Bounded by construction: exactly `runs` D2D copies
    // per call (runs * 512 MiB = 4 GiB for the default runs=8), so this can
    // never queue an unbounded amount regardless of actual D2D throughput.
    auto time_h2d_runs = [&](int runs, bool contend) -> double {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaStreamSynchronize(measure_stream);
        cudaEventRecord(start, measure_stream);
        for (int i = 0; i < runs; i++) {
            if (contend) {
                cudaMemcpyAsync(d_contender_b, d_contender_a, contender_bytes,
                        cudaMemcpyDeviceToDevice, contend_stream);
            }
            cudaMemcpyAsync(d_dst, h_src, sample_bytes, cudaMemcpyHostToDevice, measure_stream);
        }
        cudaEventRecord(stop, measure_stream);
        cudaEventSynchronize(stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return (double) sample_bytes * runs / (ms / 1000.0) / 1e9;
    };

    const int runs = 8;
    time_h2d_runs(2, false); // warm (page faults, clocks) - not measured

    out.standalone_gbs = time_h2d_runs(runs, false);
    out.contended_gbs = time_h2d_runs(runs, true);

    cudaStreamSynchronize(contend_stream); // drain before teardown - bounded by time_h2d_runs' own call count
    cudaFree(d_dst);
    cudaFree(d_contender_a);
    cudaFree(d_contender_b);
    cudaStreamDestroy(measure_stream);
    cudaStreamDestroy(contend_stream);
    cudaFreeHost(h_src);
    return true;
}

// Opt-in (GGML_CUDA_MOE_BANDWIDTH_PROFILE=1), runs once per physical device
// for the life of the process. Logged directly with fprintf, not gated
// behind MOE_CACHE_DEBUG_GATE - this is a deliberate diagnostic invocation
// the caller asked for, not incidental debug noise.
//
// Latched by physical device id in a static set, not a moe_cache_device
// member: moe_cache_session_create runs multiple times per process (once
// per graph-reserve/probe pass during model load - the same pattern this
// file's other one-shot-per-device logic has had to account for
// elsewhere), each time constructing a FRESH moe_cache_device object, so a
// per-object flag resets every time and never actually latches. Measured
// live: without this fix the profile ran nine times in a single launch.
static void moe_cache_maybe_profile_bandwidth(moe_cache_device & device) {
    static const bool requested = getenv("GGML_CUDA_MOE_BANDWIDTH_PROFILE") != nullptr;
    if (!requested) {
        return;
    }
    static std::mutex latch_mu;
    static std::unordered_set<int> profiled_physical_devices;
    {
        std::lock_guard<std::mutex> lock(latch_mu);
        if (!profiled_physical_devices.insert(device.physical).second) {
            return; // already profiled this physical device this process
        }
    }
    moe_cache_bandwidth_profile profile;
    if (!moe_cache_measure_bandwidth(device, profile)) {
        fprintf(stderr, "[moe-cache-bandwidth] CUDA%d: measurement failed\n", device.physical);
        return;
    }
    fprintf(stderr,
            "[moe-cache-bandwidth] CUDA%d: standalone=%.2f GB/s contended=%.2f GB/s "
            "(%.0f%% of standalone under concurrent GPU memory traffic)\n",
            device.physical, profile.standalone_gbs, profile.contended_gbs,
            profile.standalone_gbs > 0.0 ? 100.0 * profile.contended_gbs / profile.standalone_gbs : 0.0);
}

// Step 0 of Atlas-driven cache warming - see the header doc comment on
// ggml_moe_cache.set_atlas. Registers into every live session's every
// device, same broadcast pattern moe_cache_get_summary already uses to
// reach all of them, since a caller here has no session/device handle of
// its own (the atlas is a model-wide, not a per-session, property).
static void moe_cache_set_atlas(
        const void * host_base, const int32_t * expert,
        const float * x, const float * y, const float * spec, int n_cells) {
    if (!host_base || n_cells <= 0 || !expert || !x || !y || !spec) {
        return;
    }
    if (g_session_count.load(std::memory_order_acquire) <= 0) {
        return;
    }
    std::lock_guard<std::mutex> registry_lock(g_registry_mu);
    for (moe_cache_session * session : g_sessions) {
        std::lock_guard<std::mutex> lock(session->mu);
        for (const auto & device_ptr : session->devices) {
            moe_cache_device & device = *device_ptr;
            // A second registration for this host_base replaces its
            // entries in both indices. The row is rebuilt off to the side
            // and published by pointer swap, never mutated in place - a
            // Track 1.5 ranking pass may be scanning the previous row right
            // now with the session lock released, and its shared_ptr keeps
            // that row alive until it is done with it.
            auto row = std::make_shared<moe_cache_device::atlas_row>();
            row->reserve((size_t) n_cells);
            for (int i = 0; i < n_cells; i++) {
                if (expert[i] < 0) {
                    continue;
                }
                const moe_cache_device::atlas_cell cell{x[i], y[i], spec[i]};
                const moe_cache_key key{host_base, expert[i]};
                device.atlas[key] = cell;
                row->emplace_back(expert[i], cell);
            }
            device.atlas_by_tensor[host_base] = std::move(row);
        }
    }
}

// Exports Track 1 steps 4a/4b's co-activation data as real (tensor,
// expert) identity - see the header doc comment on get_co_activation.
// Aggregates across every live session's device (same broadcast pattern
// moe_cache_get_summary/moe_cache_set_atlas already use), then a partial
// sort for just the top max_entries by count - this is a top-K query, not
// a full dump, on purpose (a visualization only ever wants the
// significant edges, and a long-running server's edge count is otherwise
// unbounded).
static int moe_cache_get_co_activation(
        int cross_layer, ggml_moe_cache_co_activation_entry * out, int max_entries) {
    if (!out || max_entries <= 0 || g_session_count.load(std::memory_order_acquire) <= 0) {
        return 0;
    }
    std::vector<std::pair<moe_cache_edge, uint32_t>> all;
    {
        std::lock_guard<std::mutex> registry_lock(g_registry_mu);
        for (moe_cache_session * session : g_sessions) {
            std::lock_guard<std::mutex> lock(session->mu);
            for (const auto & device_ptr : session->devices) {
                const auto & map = cross_layer ? device_ptr->co_activation_cross_layer
                                                : device_ptr->co_activation;
                for (const auto & [edge, count] : map) {
                    all.emplace_back(edge, count);
                }
            }
        }
    }
    const int n = std::min((int) all.size(), max_entries);
    std::partial_sort(all.begin(), all.begin() + n, all.end(),
            [](const auto & a, const auto & b) { return a.second > b.second; });
    for (int i = 0; i < n; i++) {
        out[i] = {all[i].first.from.tensor, all[i].first.from.expert,
                  all[i].first.to.tensor,   all[i].first.to.expert,
                  all[i].second};
    }
    return n;
}

void ggml_moe_cache_register(const void * owner) {
    if (ggml_moe_cache.owner && ggml_moe_cache.owner != owner) {
        return;
    }
    ggml_moe_cache.owner = owner;
    ggml_moe_cache.session_create = moe_cache_session_create;
    ggml_moe_cache.session_destroy = moe_cache_session_destroy;
    ggml_moe_cache.session_enter = moe_cache_session_enter;
    ggml_moe_cache.session_leave = moe_cache_session_leave;
    ggml_moe_cache.begin = moe_cache_begin;
    ggml_moe_cache.plan = moe_cache_plan;
    ggml_moe_cache.train = moe_cache_train;
    ggml_moe_cache.dispatch = moe_cache_dispatch;
    ggml_moe_cache.collect = moe_cache_collect;
    ggml_moe_cache.end = moe_cache_end;
    ggml_moe_cache.invalidate = moe_cache_invalidate;
    ggml_moe_cache.set_max_batch_hint = moe_cache_set_max_batch_hint;
    ggml_moe_cache.get_stats = moe_cache_get_stats;
    ggml_moe_cache.get_expert_map = moe_cache_get_expert_map;
    ggml_moe_cache.get_substitute_map = moe_cache_get_substitute_map;
    ggml_moe_cache.get_neuron_concentration_map = moe_cache_get_neuron_concentration_map;
    ggml_moe_cache.verify_rows    = moe_cache_verify_rows;
    ggml_moe_cache.get_summary = moe_cache_get_summary;
    ggml_moe_cache.set_atlas = moe_cache_set_atlas;
    ggml_moe_cache.get_co_activation = moe_cache_get_co_activation;
    ggml_moe_cache.host_ptr = moe_cache_host_ptr;
    ggml_moe_cache.prefetch = moe_cache_prefetch;
    ggml_moe_cache.prefill_register_successor = moe_cache_prefill_register_successor;
    ggml_moe_cache.prefill_advance = moe_cache_prefill_advance;
    ggml_moe_cache.prefill_wait = moe_cache_prefill_wait;
    ggml_moe_cache.prefill_release = moe_cache_prefill_release;
    ggml_moe_cache.moe_lfru_copy_expert = moe_cache_moe_lfru_copy_expert;
    ggml_moe_cache.moe_lfru_copy_experts = moe_cache_moe_lfru_copy_experts;
    ggml_moe_cache.track_fetch_pace = moe_cache_track_fetch_pace;
}

#endif
