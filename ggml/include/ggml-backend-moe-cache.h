#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Aggregate cache health, summed across every currently-live session's
// devices - the same numbers moe-cache.cu already logs per device
// (MOE_CACHE_LOG), exposed for a UI/monitoring caller instead of only the
// log. All-zero when no session has recorded anything yet.
struct ggml_moe_cache_summary {
	long long hits;
	long long misses;
	long long evictions;
	long long fill_failures;
	long long admission_skips;
	size_t slots_used;
	size_t slots_total;
	size_t protected_slots;
	double avg_heat;
	size_t allocated_bytes;
	size_t budget_bytes;
};

struct ggml_moe_cache_api {
    const void * owner;

    // The scheduler owns one cache session. backends contains the scheduler's
    // actual backend set, so the provider can use only selected CUDA devices.
    void * (*session_create)(void * const * backends, int n_backends);
    void   (*session_destroy)(void * session);
    // NULL and dormant sessions still create a suppressing thread-local scope.
    void   (*session_enter)(void * session);
    void   (*session_leave)(void * session);

    // Begin one CPU MUL_MAT_ID node. Returns an opaque plan, or NULL when the
    // stock CPU path should handle the complete node.
    void * (*begin)(const char * tensor_name, const void * host_base, size_t expert_size,
                    int64_t n_in, int64_t n_out, int wtype, int64_t n_expert, int64_t n_tokens);

    // Mark cache hits and enqueue bounded demand fills for misses. A nonnegative
    // slot index means that the row may be omitted from CPU work only if
    // dispatch subsequently succeeds.
    int (*plan)(void * node, const int32_t * ids, int n_ids, int32_t * slot_idx);

    // Dispatch all planned hit rows. Returns 1 only after the complete GPU
    // operation has been accepted. On 0, the caller must restore every row to
    // the normal CPU mapping before worker threads start.
    int (*dispatch)(void * node, int wtype, int64_t n_in, int64_t n_out, int n_hits,
                    const int32_t * slot_idx, const float * const * act_rows);

    // Copy GPU results into dst_rows. On 0, the caller must recompute every
    // skipped row on the CPU.
    int (*collect)(void * node, int n_hits, float * const * dst_rows, int64_t n_out);

    // Releases slot pins and all per-node ownership. Must be called exactly
    // once for every non-NULL begin result, on every success or failure path.
    void (*end)(void * node);

    // Host buffer mutation or teardown notification. Sessions cancel or finish
    // any fill that still reads the supplied range before this call returns.
    void (*invalidate)(const void * base, size_t size);

    // Live hint for the default max batch size a cache-eligible MUL_MAT_ID
    // call may have (the provider still clamps to its own structural
    // ceiling). Callers should pass the real max concurrent sequence count
    // (e.g. llama_context's n_seq_max) before the next session_create(), so
    // the cache doesn't silently disable itself once real concurrency
    // exceeds a default sized for a single interactive session. NULL-safe:
    // callers must check for NULL before calling, same as other optional
    // entries here.
    void (*set_max_batch_hint)(int n_seq_max);

    // Aggregate hit/miss counts summed across every currently-live session's
    // devices. Meant for calibration/benchmarking callers that create one
    // context at a time (the common case there), not general production
    // monitoring - with multiple concurrent sessions this sums across all of
    // them, not just "the" one a caller might have in mind. Writes 0/0 if no
    // session has recorded anything yet. NULL-safe like the other optional
    // entries here.
    void (*get_stats)(long long * out_hits, long long * out_misses);

    // Snapshot the live per-(layer,expert) cache state for a debugging/demo
    // UI view (the "Brain" page). One byte per (layer,expert) cell packed
    // as (tier << 6) | heat - see moe_cache_get_expert_map()'s own comment
    // in moe-cache.cu for the exact encoding. out_rows/out_cols are always
    // written (0 on failure). Returns 1 on success, 0 if no session has
    // cached anything yet or out_bytes is too small for the real grid (in
    // which case out_rows/out_cols still report the real shape so the
    // caller can retry with a bigger buffer). NULL-safe like get_stats.
    int (*get_expert_map)(uint8_t * out_bytes, int max_bytes, int * out_rows, int * out_cols);

    // Aggregate cache health summary - see ggml_moe_cache_summary. Always
    // writes *out (zeroed on failure). NULL-safe like the other optional
    // entries here.
    void (*get_summary)(struct ggml_moe_cache_summary * out);
};

extern struct ggml_moe_cache_api ggml_moe_cache;
void ggml_moe_cache_unregister(const void * owner);

#ifdef __cplusplus
}
#endif
