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
#include <cerrno>
#include <chrono>
#if defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>
#endif
#include <climits>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <thread>
#include <unordered_map>
#include <fstream>
#include <unordered_set>
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
    int wtype = -1;
    char * slab = nullptr;
    int n_slots = 0;

    std::vector<moe_cache_slot> slots;
    std::vector<int> free_slots;
    std::unordered_map<moe_cache_key, int, moe_cache_key_hash> map;
    int lru_head = -1;       // probation segment (new/one-off admissions)
    int lru_tail = -1;
    int protected_head = -1; // protected segment (proven hot: re-requested at least once while resident)
    int protected_tail = -1;
    int protected_count = 0; // capped at n_slots * MOE_CACHE_PROTECTED_CAP_PCT / 100 - see moe_cache_promote_to_protected
    long long fills_since_decay = 0;
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

struct moe_cache_job {
    int pool = -1;
    int slot = -1;
    uint64_t generation = 0;
    moe_cache_key key;
    const void * source = nullptr;
    size_t bytes = 0;
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
    size_t min_expert_bytes = 1u << 20;
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

struct moe_cache_device {
    explicit moe_cache_device(int physical) : physical(physical) {}

    int physical;
    std::atomic<bool> dead{false};
    std::mutex dispatch_mu;

    std::vector<std::unique_ptr<moe_cache_pool>> pools;
    std::vector<moe_cache_shape> shapes;
    std::unordered_map<const void *, moe_cache_seen_tensor> seen_tensors;
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
    };
    std::unordered_map<const void *, cpu_residency> residency;
    uint64_t plan_epoch = 0; // one tick per MoE node planned
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
    float * h_out = nullptr;
    size_t h_out_cap = 0;

    long long hits = 0;
    long long misses = 0;
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
    std::atomic<int> error_logs{0};
};

struct moe_cache_session {
    moe_cache_config config;
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
    int wtype = -1;
    std::unique_lock<std::mutex> dispatch_lock;
    moe_cache_pin pins[64];
    int n_pins = 0;
    bool planned = false;
    bool dispatched = false;
};

static std::mutex g_registry_mu;
static std::unordered_set<moe_cache_session *> g_sessions;
static std::atomic<int> g_session_count{0};

// Structural ceiling shared by the scratch buffers below (moe_cache_scratch_requirements'
// max_rows) and the CPU-side stack arrays (ggml-cpu.c's MOE_CACHE_MAX_TOPK) - both are
// already sized for this many rows regardless of max_batch, so this is a real capacity
// limit, not a conservative guess.
static constexpr int MOE_CACHE_MAX_BATCH_CEILING = 64;

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
}

// Colibri's "25%-plus-4" hysteresis: candidate `b` only replaces the
// current best eviction pick `a` if b's heat is *meaningfully* colder,
// not just marginally - prevents two similarly-hot slots from flip-
// flopping as the eviction choice from one insertion to the next purely
// from noise.
static bool moe_cache_colder_enough(uint32_t a_heat, uint32_t b_heat) {
    return (uint64_t) b_heat + b_heat / 4 + 4 <= a_heat;
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

static void moe_cache_promote_to_protected(moe_cache_pool & pool, int index) {
    moe_cache_segment_remove(pool, index);
    const int cap = std::max(1, pool.n_slots * MOE_CACHE_PROTECTED_CAP_PCT / 100);
    if (pool.protected_count >= cap && pool.protected_head >= 0) {
        int demote = pool.protected_head;
        uint32_t demote_heat = pool.slots[demote].heat;
        int candidate = pool.slots[demote].next;
        for (int seen = 1; candidate >= 0 && seen < MOE_CACHE_EVICT_WINDOW; candidate = pool.slots[candidate].next, seen++) {
            const uint32_t heat = pool.slots[candidate].heat;
            if (moe_cache_colder_enough(demote_heat, heat)) {
                demote = candidate;
                demote_heat = heat;
            }
        }
        moe_cache_segment_remove(pool, demote);
        moe_cache_segment_push_back(pool, demote, moe_cache_segment::probation);
    }
    moe_cache_segment_push_back(pool, index, moe_cache_segment::protected_);
}

static void moe_cache_map_erase(moe_cache_pool & pool, int index) {
    moe_cache_slot & slot = pool.slots[index];
    auto it = pool.map.find(slot.key);
    if (it != pool.map.end() && it->second == index) {
        pool.map.erase(it);
    }
}

static void moe_cache_slot_reset(moe_cache_pool & pool, int index, bool add_to_free) {
    moe_cache_slot & slot = pool.slots[index];
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
    }
    if (fatal && error != cudaErrorMemoryAllocation) {
        device.dead.store(true);
    }
    return false;
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

    const size_t requested = required * 2 + 256;
    void * fresh = nullptr;
    if (!moe_cache_cuda_ok(device, cudaMalloc(&fresh, requested), operation, false)) {
        return false;
    }
    if (*pointer) {
        cudaFree(*pointer);
    }
    *pointer = fresh;
    capacity = requested;
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

static bool moe_cache_scratch_requirements(
        int64_t n_in, int64_t n_out, moe_cache_scratch & result) {
    constexpr size_t max_rows = 64;
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

    const size_t requested = required * 2 + 256;
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
static void moe_cache_history_save(moe_cache_session & session, const moe_cache_device & device);

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

static void moe_cache_worker(moe_cache_session * session, moe_cache_device * device) {
    char * stage = nullptr;
    size_t stage_capacity = 0;
    cudaStream_t stream = nullptr;

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
                    !device->queue.empty();
            });
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

        if (device->dead.load() || moe_cache_fail(*session, "insert")) {
            error = cudaErrorUnknown;
        }
        if (error == cudaSuccess && !stream) {
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
        if (error == cudaSuccess && stage_capacity < job.bytes) {
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
                    destination = pool->slab + (size_t)job.slot * pool->expert_size;
                }
            }
        }

        if (error == cudaSuccess && !destination) {
            error = cudaErrorInvalidValue;
        }
        {
            std::unique_lock<std::mutex> fill_lock(
                    session->fill_mu, std::defer_lock);
            if (session->config.serial_fill) {
                fill_lock.lock();
            }
            if (error == cudaSuccess && stage && stage_capacity >= job.bytes) {
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
                        // Fresh admission - starts in probation regardless
                        // of how many pre-admission misses it took to get
                        // here (that's a different signal from "reused
                        // while resident", which is what earns protected_
                        // status - see moe_cache_segment_push_back).
                        moe_cache_segment_push_back(*pool, job.slot, moe_cache_segment::probation);
                        device->demand_count.erase(job.key);
                        device->fills++;
                        if (++pool->fills_since_decay >= MOE_CACHE_HEAT_DECAY_EVERY) {
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
    device.budget_limit = std::max(available, committed);

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

static bool moe_cache_allocate_pool(
        moe_cache_session & session, moe_cache_device & device,
        moe_cache_shape & shape, size_t budget) {
    if (shape.expert_size == 0 || shape.n_tensors <= 0 || shape.n_expert <= 0) {
        return false;
    }
    shape.finished = true;

    int64_t max_entries = shape.n_tensors;
    if (max_entries > INT_MAX / shape.n_expert) {
        max_entries = INT_MAX;
    } else {
        max_entries *= shape.n_expert;
    }

    size_t slots_by_budget = budget / shape.expert_size;
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
    if (slot_count < 64) {
        return false;
    }

    ggml_cuda_set_device(device.physical);
    char * slab = nullptr;
    cudaError_t error = cudaSuccess;
    while (slot_count >= 64) {
        if (moe_cache_fail(session, "slab")) {
            error = cudaErrorMemoryAllocation;
        } else {
            error = cudaMalloc((void **)&slab, slot_count * shape.expert_size);
        }
        if (error == cudaSuccess) {
            break;
        }
        (void)cudaGetLastError();
        slot_count /= 2;
    }
    if (error != cudaSuccess || !slab || slot_count < 64) {
        MOE_CACHE_LOG("[moe-cache] CUDA%d skipped %zu KiB expert pool: allocation failed\n",
                device.physical, shape.expert_size >> 10);
        return false;
    }

    std::unique_ptr<moe_cache_pool> pool(new (std::nothrow) moe_cache_pool());
    if (!pool) {
        cudaFree(slab);
        return false;
    }
    try {
        pool->expert_size = shape.expert_size;
        pool->wtype = shape.wtype;
        pool->slab = slab;
        pool->n_slots = (int)slot_count;
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
    const size_t allocated = slot_count * shape.expert_size;
    device.allocated_bytes += allocated;

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
        slot.key = key;
        slot.generation++;
        slot.readers = 0;
        slot.state = moe_cache_slot_state::copying;
        try {
            if (!pool.map.emplace(key, slot_index).second) {
                moe_cache_slot_reset(pool, slot_index, true);
                continue;
            }
            device.queue.push_back({
                    pool_index, slot_index, slot.generation, key,
                    (const char *) host_base + (size_t) c.expert * expert_size,
                    expert_size});
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
    const size_t slab_limit =
        scratch_reserve < device.budget_limit
            ? device.budget_limit - scratch_reserve : 0;
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
    if (device.compute_stream) {
        cudaStreamDestroy(device.compute_stream);
        device.compute_stream = nullptr;
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
}

static void moe_cache_session_destroy(void * opaque) {
    moe_cache_session * session = (moe_cache_session *)opaque;
    if (!session) {
        return;
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
    if (!session || session->stopping || session->dormant || !name || !host_base ||
        !strstr(name, "_exps") || n_tokens < 1 ||
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
            n_in, n_out, scratch_requirements)) {
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
        moe_cache_session::active_source * source = nullptr;
        try {
            source = &session->active_sources[host_base];
        } catch (...) {
            return nullptr;
        }
        source->bytes = std::max(source->bytes, tensor_size);
        source->references++;
        session->active_nodes++;
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
    node->wtype = wtype;
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

static int moe_cache_plan(
        void * opaque, const int32_t * ids, int n_ids, int32_t * slot_indices) {
    moe_cache_node * node = (moe_cache_node *)opaque;
    if (!node || !ids || !slot_indices || n_ids < 0 || n_ids > 64 || node->planned) {
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
            }
            residency = &entry;
        } catch (...) {
            residency = nullptr; // tracking is best-effort, never fail a decode for it
        }

    }

    for (int index = 0; index < n_ids; index++) {
        const int32_t expert = ids[index];
        if (expert < 0 || expert >= node->n_expert || device.dead.load()) {
            continue;
        }

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
            // A selected expert is live again: clear the cold mark so a later
            // dormant stretch re-advises rather than being skipped forever.
            residency->is_cold[expert] = false;
        }

        const moe_cache_key key{node->host_base, expert};
        auto found = pool.map.find(key);
        if (found != pool.map.end()) {
            moe_cache_slot & slot = pool.slots[found->second];
            if (slot.state == moe_cache_slot_state::valid) {
                slot.readers++;
                slot.heat = std::min(MOE_CACHE_HEAT_MAX, slot.heat + MOE_CACHE_HEAT_STEP);
                // Any real hit, from either segment, promotes to (or
                // refreshes within) protected_ - a resident slot being
                // requested again is exactly the "genuinely hot, not just
                // a one-off admission" signal that earns real protection
                // from eviction churn. Heat then decides how long that
                // protection actually lasts, at the next decay sweep.
                moe_cache_promote_to_protected(pool, found->second);
                node->pins[node->n_pins++] = {found->second};
                slot_indices[index] = found->second;
                device.hits++;
                hits++;
            } else {
                device.misses++;
            }
            continue;
        }

        device.misses++;
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
            auto pick_coldest_unpinned = [&](int head) -> int {
                int best = -1;
                uint32_t best_heat = 0;
                int candidate = head;
                for (int seen = 0; candidate >= 0 && seen < MOE_CACHE_EVICT_WINDOW; candidate = pool.slots[candidate].next) {
                    if (pool.slots[candidate].readers > 0) {
                        continue;
                    }
                    seen++;
                    const uint32_t heat = pool.slots[candidate].heat;
                    if (best < 0 || moe_cache_colder_enough(best_heat, heat)) {
                        best = candidate;
                        best_heat = heat;
                    }
                }
                return best;
            };
            int candidate = pick_coldest_unpinned(pool.lru_head);
            if (candidate < 0) {
                candidate = pick_coldest_unpinned(pool.protected_head);
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
            const auto inserted = pool.map.emplace(key, slot_index);
            if (!inserted.second) {
                moe_cache_slot_reset(pool, slot_index, true);
                device.insert_skips++;
                continue;
            }

            const void * source =
                (const char *)node->host_base + (size_t)expert * node->expert_size;
            device.queue.push_back({
                    node->pool_index, slot_index, slot.generation,
                    key, source, node->expert_size});
            device.queued_bytes += node->expert_size;
        } catch (...) {
            moe_cache_slot_reset(pool, slot_index, true);
            device.insert_skips++;
            continue;
        }
        device.inserts++;
        inserts_left--;
        wake_worker = true;
    }
    device.nodes++;
    lock.unlock();
    if (wake_worker) {
        session.cv.notify_all();
    }
    return hits;
}

static int moe_cache_dispatch(
        void * opaque, int wtype, int64_t n_in, int64_t n_out, int n_hits,
        const int32_t * slot_indices, const float * const * act_rows) {
    moe_cache_node * node = (moe_cache_node *)opaque;
    if (!node || !node->planned || node->dispatched || n_hits <= 0 ||
        n_hits > 64 || n_hits != node->n_pins || !slot_indices || !act_rows ||
        wtype != node->wtype || n_in != node->n_in || n_out != node->n_out ||
        n_in > INT_MAX || n_out > INT_MAX ||
        n_in > INT64_MAX - (MATRIX_ROW_PADDING - 1)) {
        return 0;
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
    for (int index = 0; index < n_hits; index++) {
        if (slot_indices[index] < 0 || slot_indices[index] >= pool.n_slots ||
            !act_rows[index]) {
            return 0;
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
    {
        std::lock_guard<std::mutex> lock(session.mu);
        if (scratch_bytes > device.budget_limit ||
            device.allocated_bytes > device.budget_limit - scratch_bytes) {
            device.dispatch_failures++;
            return 0;
        }
    }

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

    for (int index = 0; index < n_hits; index++) {
        device.h_ids[index] = slot_indices[index];
    }
    for (int index = 0; index < activation_rows; index++) {
        memcpy(device.h_act + (size_t)index * n_in,
               act_rows[shared_activation ? 0 : index], n_in * sizeof(float));
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
                (int64_t)pool.expert_size, n_hits, activation_rows,
                device.compute_stream);
        ok = moe_cache_cuda_ok(
                device, cudaPeekAtLastError(), "expert matvec launch", true);
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
    if (!node || !node->dispatched || n_hits <= 0 || n_hits > 64 ||
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
    if (ok) {
        ok = moe_cache_cuda_ok(
                device, cudaStreamSynchronize(device.compute_stream),
                "output synchronization", true);
    } else {
        cudaStreamSynchronize(device.compute_stream);
    }
    node->dispatched = false;

    if (ok) {
        for (int index = 0; index < n_hits; index++) {
            memcpy(dst_rows[index], device.h_out + (size_t)index * n_out,
                   n_out * sizeof(float));
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
            if (pin.slot >= 0 && pin.slot < node->pool->n_slots) {
                moe_cache_slot & slot = node->pool->slots[pin.slot];
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
            freed += (size_t)pool_ptr->n_slots * pool_ptr->expert_size;
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
                sum.allocated_bytes += device.allocated_bytes;
                sum.budget_bytes    += device.budget_limit;

                for (const auto & pool_ptr : device.pools) {
                    const moe_cache_pool & pool = *pool_ptr;
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
    ggml_moe_cache.dispatch = moe_cache_dispatch;
    ggml_moe_cache.collect = moe_cache_collect;
    ggml_moe_cache.end = moe_cache_end;
    ggml_moe_cache.invalidate = moe_cache_invalidate;
    ggml_moe_cache.set_max_batch_hint = moe_cache_set_max_batch_hint;
    ggml_moe_cache.get_stats = moe_cache_get_stats;
    ggml_moe_cache.get_expert_map = moe_cache_get_expert_map;
    ggml_moe_cache.get_summary = moe_cache_get_summary;
}

#endif
