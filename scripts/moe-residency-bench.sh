#!/usr/bin/env bash
# Repeated-measurement harness for moe-cache residency policies.
#
# Exists because single-run comparisons on this rig proved unusable: nominally
# identical configurations ranged 7.59-13.11 and 10.30-33.37 tok/s, i.e. the
# run-to-run spread was larger than the effects being claimed. Every policy
# decision made from one sample is a coin flip dressed as a measurement.
#
# Method: N repetitions per configuration, page cache dropped and the server
# restarted between every repetition so each one is a genuine cold start, first
# repetition of each *series* discarded, median and min/max reported.
set -u

MODEL=${MODEL:-/home/Projects/llama.cpp/models/moe-test2/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf}
BIN=${BIN:-/home/Projects/llama.cpp/build/bin/llama-server}
PORT=${PORT:-8095}
CAP=${CAP:-5G}
REPS=${REPS:-5}
NCMOE=${NCMOE:-23}
CTX=${CTX:-8192}
HIST=${HIST:-/root/.claude/jobs/a804561e/tmp/moe-history.txt}
OUT=${OUT:-/root/.claude/jobs/a804561e/tmp/bench}
PROMPT='{"messages":[{"role":"user","content":"Write three sentences about the ocean and three about mountains."}],"max_tokens":80}'

mkdir -p "$OUT"

stop_server() {
    systemctl stop "llama-bench-$PORT.scope" >/dev/null 2>&1
    pkill -9 -f "port $PORT" >/dev/null 2>&1
    sleep 2
}

# $1 = label, $2..$n = env assignments
run_config() {
    local label=$1; shift
    local results=()
    for rep in $(seq 1 "$REPS"); do
        stop_server
        sync; echo 3 > /proc/sys/vm/drop_caches 2>/dev/null

        systemd-run --scope --quiet -p MemoryMax="$CAP" -p MemorySwapMax=0 \
            --unit="llama-bench-$PORT" env "$@" \
            "$BIN" -m "$MODEL" -ncmoe "$NCMOE" --jinja --moe-cache auto \
            -c "$CTX" -fit off --host 127.0.0.1 --port "$PORT" \
            > "$OUT/$label-$rep.log" 2>&1 &
        disown

        local up=0
        for _ in $(seq 1 100); do
            curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { up=1; break; }
            sleep 3
        done
        [ "$up" = 1 ] || { echo "  $label rep$rep: server did not start"; continue; }

        # One warm-up request, then the measured one - so we time steady-state
        # decode rather than whatever the very first tokens happened to cost.
        curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H "Content-Type: application/json" \
            -d "$PROMPT" -o /dev/null --max-time 900
        curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H "Content-Type: application/json" \
            -d "$PROMPT" -o "$OUT/$label-$rep.json" --max-time 900

        local v
        v=$(python3 -c "
import json
try:
    print('%.2f' % json.load(open('$OUT/$label-$rep.json'))['timings']['predicted_per_second'])
except Exception:
    print('nan')")
        echo "  $label rep$rep: $v tok/s"
        [ "$v" != "nan" ] && results+=("$v")
    done
    stop_server

    printf '%s\n' "${results[@]}" | python3 -c "
import sys, statistics
xs=[float(x) for x in sys.stdin if x.strip()]
if not xs:
    print('$label: no data'); raise SystemExit
xs.sort()
print('RESULT %-22s n=%d  median %6.2f   min %6.2f   max %6.2f   spread %.0f%%'
      % ('$label', len(xs), statistics.median(xs), xs[0], xs[-1],
         100*(xs[-1]-xs[0])/max(xs[0],1e-9)))"
}

echo "=== moe-cache residency benchmark: cap=$CAP reps=$REPS ncmoe=$NCMOE ctx=$CTX ==="
run_config baseline          GGML_CUDA_MOE_CACHE_HISTORY=/nonexistent-baseline GGML_CUDA_MOE_CACHE_READAHEAD=0
run_config history           GGML_CUDA_MOE_CACHE_HISTORY="$HIST" GGML_CUDA_MOE_CACHE_READAHEAD=0
run_config history+readahead GGML_CUDA_MOE_CACHE_HISTORY="$HIST" GGML_CUDA_MOE_CACHE_READAHEAD=1
run_config history+ra+pin    GGML_CUDA_MOE_CACHE_HISTORY="$HIST" GGML_CUDA_MOE_CACHE_READAHEAD=1 GGML_CUDA_MOE_CACHE_PIN_MB=auto
