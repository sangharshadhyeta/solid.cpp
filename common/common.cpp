#include "ggml.h"
#include "gguf.h"
#include "ggml-backend-moe-cache.h"

#include "build-info.h"
#include "common.h"
#include "fit.h"
#include "log.h"
#include "llama.h"
#include "sampling.h"
#include "speculative.h"
#include "unicode.h"

#include <algorithm>
#include <cinttypes>
#include <climits>
#include <cmath>
#include <atomic>
#include <chrono>
#include <cstdarg>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <list>
#include <map>
#include <mutex>
#include <regex>
#include <sys/wait.h>

#define JSON_ASSERT GGML_ASSERT
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#if defined(__APPLE__) && defined(__MACH__)
#include <sys/types.h>
#include <sys/sysctl.h>
#endif

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <locale>
#include <windows.h>
#include <string.h>
#include <fcntl.h>
#include <io.h>
#else
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#if defined(__linux__)
#include <sys/types.h>
#include <pwd.h>
#endif

#if defined(_AIX)
#include <sys/systemcfg.h>
#endif

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

common_time_meas::common_time_meas(int64_t & t_acc, bool disable) : t_start_us(disable ? -1 : ggml_time_us()), t_acc(t_acc) {}

common_time_meas::~common_time_meas() {
    if (t_start_us >= 0) {
        t_acc += ggml_time_us() - t_start_us;
    }
}

//
// CPU utils
//

int32_t common_cpu_get_num_physical_cores() {
#if defined(_AIX)
    int32_t logical_cpus = _system_configuration.ncpus;
    int32_t smt_threads = _system_configuration.smt_threads;
    if (smt_threads > 0) {
        return static_cast<int32_t>(logical_cpus / smt_threads);
    }
    if (logical_cpus > 0) {
        return static_cast<int32_t>(logical_cpus);
    }
#elif defined(__linux__)
    // enumerate the set of thread siblings, num entries is num cores
    std::unordered_set<std::string> siblings;
    for (uint32_t cpu=0; cpu < UINT32_MAX; ++cpu) {
        std::ifstream thread_siblings("/sys/devices/system/cpu/cpu"
            + std::to_string(cpu) + "/topology/thread_siblings");
        if (!thread_siblings.is_open()) {
            break; // no more cpus
        }
        std::string line;
        if (std::getline(thread_siblings, line)) {
            siblings.insert(line);
        }
    }
    if (!siblings.empty()) {
        return static_cast<int32_t>(siblings.size());
    }
#elif defined(__APPLE__) && defined(__MACH__)
    int32_t num_physical_cores;
    size_t len = sizeof(num_physical_cores);
    int result = sysctlbyname("hw.perflevel0.physicalcpu", &num_physical_cores, &len, NULL, 0);
    if (result == 0) {
        return num_physical_cores;
    }
    result = sysctlbyname("hw.physicalcpu", &num_physical_cores, &len, NULL, 0);
    if (result == 0) {
        return num_physical_cores;
    }
#elif defined(_WIN32) && (_WIN32_WINNT >= 0x0601) && !defined(__MINGW64__) // windows 7 and later
    // TODO: windows + arm64 + mingw64
    unsigned int n_threads_win = std::thread::hardware_concurrency();
    unsigned int default_threads = n_threads_win > 0 ? (n_threads_win <= 4 ? n_threads_win : n_threads_win / 2) : 4;

    DWORD buffer_size = 0;
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &buffer_size)) {
        if (GetLastError() != ERROR_INSUFFICIENT_BUFFER) {
            return default_threads;
        }
    }

    std::vector<char> buffer(buffer_size);
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data()), &buffer_size)) {
        return default_threads;
    }

    int32_t num_physical_cores = 0;
    PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data());
    while (buffer_size > 0) {
        if (info->Relationship == RelationProcessorCore) {
            num_physical_cores += info->Processor.GroupCount;
        }
        buffer_size -= info->Size;
        info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(reinterpret_cast<char*>(info) + info->Size);
    }

    return num_physical_cores > 0 ? num_physical_cores : default_threads;
#endif
    unsigned int n_threads = std::thread::hardware_concurrency();
    return n_threads > 0 ? (n_threads <= 4 ? n_threads : n_threads / 2) : 4;
}

#if defined(__x86_64__) && defined(__linux__) && !defined(__ANDROID__)
#include <pthread.h>

static void cpuid(unsigned leaf, unsigned subleaf,
                  unsigned *eax, unsigned *ebx, unsigned *ecx, unsigned *edx) {
    __asm__("movq\t%%rbx,%%rsi\n\t"
            "cpuid\n\t"
            "xchgq\t%%rbx,%%rsi"
            : "=a"(*eax), "=S"(*ebx), "=c"(*ecx), "=d"(*edx)
            : "0"(leaf), "2"(subleaf));
}

static int pin_cpu(int cpu) {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(cpu, &mask);
    return pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask);
}

static bool is_hybrid_cpu(void) {
    unsigned eax, ebx, ecx, edx;
    cpuid(7, 0, &eax, &ebx, &ecx, &edx);
    return !!(edx & (1u << 15));
}

static bool is_running_on_efficiency_core(void) {
    unsigned eax, ebx, ecx, edx;
    cpuid(0x1a, 0, &eax, &ebx, &ecx, &edx);
    int intel_atom = 0x20;
    int core_type = (eax & 0xff000000u) >> 24;
    return core_type == intel_atom;
}

static int cpu_count_math_cpus(int n_cpu) {
    int result = 0;
    for (int cpu = 0; cpu < n_cpu; ++cpu) {
        if (pin_cpu(cpu)) {
            return -1;
        }
        if (is_running_on_efficiency_core()) {
            continue; // efficiency cores harm lockstep threading
        }
        ++cpu; // hyperthreading isn't useful for linear algebra
        ++result;
    }
    return result;
}

#endif // __x86_64__ && __linux__

/**
 * Returns number of CPUs on system that are useful for math.
 */
int32_t common_cpu_get_num_math() {
#if defined(__x86_64__) && defined(__linux__) && !defined(__ANDROID__)
    int n_cpu = sysconf(_SC_NPROCESSORS_ONLN);
    if (n_cpu < 1) {
        return common_cpu_get_num_physical_cores();
    }
    if (is_hybrid_cpu()) {
        cpu_set_t affinity;
        if (!pthread_getaffinity_np(pthread_self(), sizeof(affinity), &affinity)) {
            int result = cpu_count_math_cpus(n_cpu);
            pthread_setaffinity_np(pthread_self(), sizeof(affinity), &affinity);
            if (result > 0) {
                return result;
            }
        }
    }
#elif defined(__powerpc64__) || defined(__powerpc__)
    int32_t smt_factor = 1;
    int phy_cpus = common_cpu_get_num_physical_cores();
    int logical_cpus = sysconf(_SC_NPROCESSORS_ONLN);
    if (phy_cpus > 0 && logical_cpus > phy_cpus) {
        smt_factor = logical_cpus / phy_cpus;
    }
    return phy_cpus * std::min(smt_factor, 2);
#endif
    return common_cpu_get_num_physical_cores();
}

// Helper for setting process priority

#if defined(_WIN32)

bool set_process_priority(enum ggml_sched_priority prio) {
    if (prio == GGML_SCHED_PRIO_NORMAL) {
        return true;
    }

    DWORD p = NORMAL_PRIORITY_CLASS;
    switch (prio) {
        case GGML_SCHED_PRIO_LOW:      p = BELOW_NORMAL_PRIORITY_CLASS; break;
        case GGML_SCHED_PRIO_NORMAL:   p = NORMAL_PRIORITY_CLASS;       break;
        case GGML_SCHED_PRIO_MEDIUM:   p = ABOVE_NORMAL_PRIORITY_CLASS; break;
        case GGML_SCHED_PRIO_HIGH:     p = HIGH_PRIORITY_CLASS;         break;
        case GGML_SCHED_PRIO_REALTIME: p = REALTIME_PRIORITY_CLASS;     break;
    }

    if (!SetPriorityClass(GetCurrentProcess(), p)) {
        COM_WRN("failed to set process priority class %d : (%d)\n", prio, (int) GetLastError());
        return false;
    }

    return true;
}

#else // MacOS and POSIX
#include <sys/types.h>
#include <sys/resource.h>

bool set_process_priority(enum ggml_sched_priority prio) {
    if (prio == GGML_SCHED_PRIO_NORMAL) {
        return true;
    }

    int p = 0;
    switch (prio) {
        case GGML_SCHED_PRIO_LOW:      p =  5;  break;
        case GGML_SCHED_PRIO_NORMAL:   p =  0;  break;
        case GGML_SCHED_PRIO_MEDIUM:   p = -5;  break;
        case GGML_SCHED_PRIO_HIGH:     p = -10; break;
        case GGML_SCHED_PRIO_REALTIME: p = -20; break;
    }

    if (setpriority(PRIO_PROCESS, 0, p) != 0) {
        COM_WRN("failed to set process priority %d : %s (%d)\n", prio, strerror(errno), errno);
        return false;
    }
    return true;
}

#endif

//
// CLI argument parsing
//


void postprocess_cpu_params(common_cpu_params & cpuparams, const common_cpu_params * role_model) {
    int32_t n_set = 0;

    if (cpuparams.n_threads < 0) {
        // Assuming everything about cpuparams is invalid
        if (role_model != nullptr) {
            cpuparams = *role_model;
        } else {
            cpuparams.n_threads = common_cpu_get_num_math();
        }
    }

    for (int32_t i = 0; i < GGML_MAX_N_THREADS; i++) {
        if (cpuparams.cpumask[i]) {
            n_set++;
        }
    }

    if (n_set && n_set < cpuparams.n_threads) {
        // Not enough set bits, may experience performance issues.
        COM_WRN("Not enough set bits in CPU mask (%d) to satisfy requested thread count: %d\n", n_set, cpuparams.n_threads);
    }
}

bool parse_cpu_range(const std::string & range, bool (&boolmask)[GGML_MAX_N_THREADS]) {
    size_t dash_loc = range.find('-');
    if (dash_loc == std::string::npos) {
        COM_ERR("%s", "Format of CPU range is invalid! Expected [<start>]-[<end>].\n");
        return false;
    }

    size_t start_i;
    size_t end_i;

    if (dash_loc == 0) {
        start_i = 0;
    } else {
        start_i = std::stoull(range.substr(0, dash_loc));
        if (start_i >= GGML_MAX_N_THREADS) {
            COM_ERR("%s", "Start index out of bounds!\n");
            return false;
        }
    }

    if (dash_loc == range.length() - 1) {
        end_i = GGML_MAX_N_THREADS - 1;
    } else {
        end_i = std::stoull(range.substr(dash_loc + 1));
        if (end_i >= GGML_MAX_N_THREADS) {
            COM_ERR("%s", "End index out of bounds!\n");
            return false;
        }
    }

    for (size_t i = start_i; i <= end_i; i++) {
        boolmask[i] = true;
    }

    return true;
}

bool parse_cpu_mask(const std::string & mask, bool (&boolmask)[GGML_MAX_N_THREADS]) {
    // Discard potential 0x prefix
    size_t start_i = 0;
    if (mask.length() >= 2 && mask.substr(0, 2) == "0x") {
        start_i = 2;
    }

    size_t num_digits = mask.length() - start_i;
    num_digits = std::min<size_t>(num_digits, 128);

    size_t end_i = num_digits + start_i;

    for (size_t i = start_i, n = (num_digits*4 - 1); i < end_i; i++, n-=4) {
        char c = mask.at(i);
        int8_t id = c;

        if ((c >= '0' && c <= '9')) {
            id -= '0';
        } else if (c >= 'a' && c <= 'f') {
            id -= 'a' - 10;
        } else if (c >= 'A' && c <= 'F') {
            id -= 'A' - 10;
        } else {
            COM_ERR("Invalid hex character '%c' at position %d\n", c, int32_t(i));
            return false;
        }

        boolmask[  n  ] = boolmask[  n  ] || ((id & 8) != 0);
        boolmask[n - 1] = boolmask[n - 1] || ((id & 4) != 0);
        boolmask[n - 2] = boolmask[n - 2] || ((id & 2) != 0);
        boolmask[n - 3] = boolmask[n - 3] || ((id & 1) != 0);
    }

    return true;
}

void common_init() {
#if defined(_WIN32)
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif

    common_log_set_prefix(common_log_main(), true);
    common_log_set_timestamps(common_log_main(), true);

    llama_log_set(common_log_default_callback, NULL);
}

void common_params_print_info(const common_params & params, bool print_devices) {
#ifdef NDEBUG
    const char * build_type = "";
#else
    const char * build_type = " (debug)";
#endif
    COM_TRC("%s: build %d (%s) with %s for %s%s\n", __func__, llama_build_number(), llama_commit(), llama_compiler(), llama_build_target(), build_type);

    COM_INF("%s: verbosity = %d (adjust with the `-lv N` CLI arg)\n", __func__, common_log_get_verbosity_thold());

    // device enumeration creates a primary context on CUDA backends, skip it when the caller does not own any device
    if (print_devices) {
        COM_TRC("%s", "device_info:\n");
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            auto * dev = ggml_backend_dev_get(i);
            size_t free, total;
            ggml_backend_dev_memory(dev, &free, &total);
            COM_TRC("  - %-8s: %s (%zu MiB, %zu MiB free)\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev), total / 1024 / 1024, free / 1024 / 1024);
        }
    }
    COM_TRC("%s\n", common_params_get_system_info(params).c_str());
}

std::string common_params_get_system_info(const common_params & params) {
    std::ostringstream os;

    os << "system_info: n_threads = " << params.cpuparams.n_threads;
    if (params.cpuparams_batch.n_threads != -1) {
        os << " (n_threads_batch = " << params.cpuparams_batch.n_threads << ")";
    }
#if defined(_WIN32) && (_WIN32_WINNT >= 0x0601) && !defined(__MINGW64__) // windows 7 and later
    // TODO: windows + arm64 + mingw64
    DWORD logicalProcessorCount = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
    os << " / " << logicalProcessorCount << " | " << llama_print_system_info();
#else
    os << " / " << std::thread::hardware_concurrency() << " | " << llama_print_system_info();
#endif

    return os.str();
}

//
// String utils
//

std::string string_format(const char * fmt, ...) {
    va_list ap;
    va_list ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    int size = vsnprintf(NULL, 0, fmt, ap);
    GGML_ASSERT(size >= 0 && size < INT_MAX); // NOLINT
    std::vector<char> buf(size + 1);
    int size2 = vsnprintf(buf.data(), size + 1, fmt, ap2);
    GGML_ASSERT(size2 == size);
    va_end(ap2);
    va_end(ap);
    return std::string(buf.data(), size);
}

std::string string_strip(const std::string & str) {
    size_t start = 0;
    size_t end = str.size();
    while (start < end && std::isspace(str[start])) {
        start++;
    }
    while (end > start && std::isspace(str[end - 1])) {
        end--;
    }
    return str.substr(start, end - start);
}

std::string string_lcs(std::string_view a, std::string_view b) {
    if (a.empty() || b.empty()) return {};

    std::vector<std::vector<size_t>> dp(a.size() + 1, std::vector<size_t>(b.size() + 1, 0));
    size_t best_len = 0;
    size_t best_end_a = 0;

    for (size_t i = 1; i <= a.size(); ++i) {
        for (size_t j = 1; j <= b.size(); ++j) {
            if (a[i - 1] == b[j - 1]) {
                dp[i][j] = dp[i - 1][j - 1] + 1;
                if (dp[i][j] > best_len) {
                    best_len = dp[i][j];
                    best_end_a = i;
                }
            }
        }
    }
    return std::string(a.substr(best_end_a - best_len, best_len));
}

std::string string_get_sortable_timestamp() {
    using clock = std::chrono::system_clock;

    const clock::time_point current_time = clock::now();
    const time_t as_time_t = clock::to_time_t(current_time);
    char timestamp_no_ns[100];
    std::strftime(timestamp_no_ns, 100, "%Y_%m_%d-%H_%M_%S", std::localtime(&as_time_t));

    const int64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        current_time.time_since_epoch() % 1000000000).count();
    char timestamp_ns[11];
    snprintf(timestamp_ns, 11, "%09" PRId64, ns);

    return std::string(timestamp_no_ns) + "." + std::string(timestamp_ns);
}

void string_replace_all(std::string & s, const std::string & search, const std::string & replace) {
    if (search.empty()) {
        return;
    }
    std::string builder;
    builder.reserve(s.length());
    size_t pos = 0;
    size_t last_pos = 0;
    while ((pos = s.find(search, last_pos)) != std::string::npos) {
        builder.append(s, last_pos, pos - last_pos);
        builder.append(replace);
        last_pos = pos + search.length();
    }
    builder.append(s, last_pos, std::string::npos);
    s = std::move(builder);
}

std::string regex_escape(const std::string & s) {
    static const std::regex special_chars("[.^$|()*+?\\[\\]{}\\\\]");
    return std::regex_replace(s, special_chars, "\\$&");
}

std::string string_join(const std::vector<std::string> & values, const std::string & separator) {
    std::ostringstream result;
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            result << separator;
        }
        result << values[i];
    }
    return result.str();
}

std::vector<std::string> string_split(const std::string & str, const std::string & delimiter) {
    std::vector<std::string> parts;
    size_t start = 0;
    size_t end = str.find(delimiter);

    while (end != std::string::npos) {
        parts.push_back(str.substr(start, end - start));
        start = end + delimiter.length();
        end = str.find(delimiter, start);
    }

    parts.push_back(str.substr(start));

    return parts;
}

std::string string_repeat(const std::string & str, size_t n) {
    if (n == 0) {
        return "";
    }

    std::string result;
    result.reserve(str.length() * n);

    for (size_t i = 0; i < n; ++i) {
        result += str;
    }

    return result;
}

std::string string_from(bool value) {
    return value ? "true" : "false";
}

std::string string_from(const std::vector<int> & values) {
    std::stringstream buf;

    buf << "[ ";
    bool first = true;
    for (auto e : values) {
        if (first) {
            first = false;
        } else {
            buf << ", ";
        }
        buf << std::to_string(e);
    }
    buf << " ]";

    return buf.str();
}

std::string string_from(const struct llama_context * ctx, const std::vector<llama_token> & tokens) {
    std::stringstream buf;

    buf << "[ ";

    bool first = true;
    for (const auto & token : tokens) {
        if (!first) {
            buf << ", ";
        } else {
            first = false;
        }

        auto detokenized = common_token_to_piece(ctx, token);

        buf << "'" << detokenized << "'"
            << ":" << std::to_string(token);
    }

    buf << " ]";

    return buf.str();
}

std::string string_from(const struct llama_context * ctx, const struct llama_batch & batch) {
    std::stringstream buf;

    buf << "[ ";

    bool first = true;
    for (int i = 0; i < batch.n_tokens; ++i) {
        if (!first) {
            buf << ", ";
        } else {
            first = false;
        }

        auto detokenized = common_token_to_piece(ctx, batch.token[i]);

        buf << "\n"          << std::to_string(i)
            << ", token '"   << detokenized << "'"
            << ", pos "      << std::to_string(batch.pos[i])
            << ", n_seq_id " << std::to_string(batch.n_seq_id[i])
            << ", seq_id "   << std::to_string(batch.seq_id[i][0])
            << ", logits "   << std::to_string(batch.logits[i]);
    }

    buf << " ]";

    return buf.str();
}

void string_process_escapes(std::string & input) {
    std::size_t input_len = input.length();
    std::size_t output_idx = 0;

    for (std::size_t input_idx = 0; input_idx < input_len; ++input_idx) {
        if (input[input_idx] == '\\' && input_idx + 1 < input_len) {
            switch (input[++input_idx]) {
                case 'n':  input[output_idx++] = '\n'; break;
                case 'r':  input[output_idx++] = '\r'; break;
                case 't':  input[output_idx++] = '\t'; break;
                case '\'': input[output_idx++] = '\''; break;
                case '\"': input[output_idx++] = '\"'; break;
                case '\\': input[output_idx++] = '\\'; break;
                case 'x':
                    // Handle \x12, etc
                    if (input_idx + 2 < input_len) {
                        const char x[3] = { input[input_idx + 1], input[input_idx + 2], 0 };
                        char *err_p = nullptr;
                        const long val = std::strtol(x, &err_p, 16);
                        if (err_p == x + 2) {
                            input_idx += 2;
                            input[output_idx++] = char(val);
                            break;
                        }
                    }
                    // fall through
                default:   input[output_idx++] = '\\';
                           input[output_idx++] = input[input_idx]; break;
            }
        } else {
            input[output_idx++] = input[input_idx];
        }
    }

    input.resize(output_idx);
}

bool string_parse_kv_override(const char * data, std::vector<llama_model_kv_override> & overrides) {
    const char * sep = strchr(data, '=');
    if (sep == nullptr || sep - data >= 128) {
        COM_ERR("%s: malformed KV override '%s'\n", __func__, data);
        return false;
    }
    llama_model_kv_override kvo;
    std::strncpy(kvo.key, data, sep - data);
    kvo.key[sep - data] = 0;
    sep++;
    if (strncmp(sep, "int:", 4) == 0) {
        sep += 4;
        kvo.tag = LLAMA_KV_OVERRIDE_TYPE_INT;
        kvo.val_i64 = std::atol(sep);
    } else if (strncmp(sep, "float:", 6) == 0) {
        sep += 6;
        kvo.tag = LLAMA_KV_OVERRIDE_TYPE_FLOAT;
        kvo.val_f64 = std::atof(sep);
    } else if (strncmp(sep, "bool:", 5) == 0) {
        sep += 5;
        kvo.tag = LLAMA_KV_OVERRIDE_TYPE_BOOL;
        if (std::strcmp(sep, "true") == 0) {
            kvo.val_bool = true;
        } else if (std::strcmp(sep, "false") == 0) {
            kvo.val_bool = false;
        } else {
            COM_ERR("%s: invalid boolean value for KV override '%s'\n", __func__, data);
            return false;
        }
    } else if (strncmp(sep, "str:", 4) == 0) {
        sep += 4;
        kvo.tag = LLAMA_KV_OVERRIDE_TYPE_STR;
        if (strlen(sep) > 127) {
            COM_ERR("%s: malformed KV override '%s', value cannot exceed 127 chars\n", __func__, data);
            return false;
        }
        strncpy(kvo.val_str, sep, 127);
        kvo.val_str[127] = '\0';
    } else {
        COM_ERR("%s: invalid type for KV override '%s'\n", __func__, data);
        return false;
    }
    overrides.emplace_back(std::move(kvo));
    return true;
}

static inline bool glob_class_match(const char c, const char * pattern, const char * class_end) {
    const char * class_start = pattern;
    bool negated = false;

    if (*class_start == '!') {
        negated = true;
        class_start++;
    }

    // If first character after negation is ']' or '-', treat it as literal
    if (*class_start == ']' || *class_start == '-') {
        if (class_start < class_end && *class_start == c) {
            return !negated;
        }
        class_start++;
    }

    bool matched = false;

    while (class_start < class_end) {
        if (class_start + 2 < class_end && class_start[1] == '-' && class_start[2] != ']') {
            char start_char = *class_start;
            char end_char = class_start[2];
            if (c >= start_char && c <= end_char) {
                matched = true;
                break;
            }
            class_start += 3;
        } else {
            if (*class_start == c) {
                matched = true;
                break;
            }
            class_start++;
        }
    }

    return negated ? !matched : matched;
}

// simple glob: * matches non-/ chars, ** matches anything including /, [] matches character class
static inline bool glob_match(const char * pattern, const char * str) {
    if (*pattern == '\0') {
        return *str == '\0';
    }
    if (pattern[0] == '*' && pattern[1] == '*') {
        const char * p = pattern + 2;
        if (glob_match(p, str)) return true;
        if (*str != '\0') return glob_match(pattern, str + 1);
        return false;
    }
    if (*pattern == '*') {
        const char * p = pattern + 1;
        for (; *str != '\0' && *str != '/'; str++) {
            if (glob_match(p, str)) return true;
        }
        return glob_match(p, str);
    }
    if (*pattern == '?' && *str != '\0' && *str != '/') {
        return glob_match(pattern + 1, str + 1);
    }
    if (*pattern == '[') {
        const char * class_end = pattern + 1;
        // If first character after '[' is ']' or '-', treat it as literal
        if (*class_end == ']' || *class_end == '-') {
            class_end++;
        }
        while (*class_end != '\0' && *class_end != ']') {
            class_end++;
        }
        if (*class_end == ']') {
            if (*str == '\0') return false;
            bool matched = glob_class_match(*str, pattern + 1, class_end);
            return matched && glob_match(class_end + 1, str + 1);
        } else {
            if (*str == '[') {
                return glob_match(pattern + 1, str + 1);
            }
            return false;
        }
    }
    if (*pattern == *str) {
        return glob_match(pattern + 1, str + 1);
    }
    return false;
}

bool glob_match(const std::string & pattern, const std::string & str) {
    return glob_match(pattern.c_str(), str.c_str());
}

//
// Filesystem utils
//

// Validate if a filename is safe to use
// To validate a full path, split the path by the OS-specific path separator, and validate each part with this function
bool fs_validate_filename(const std::string & filename, bool allow_subdirs) {
    if (!filename.length()) {
        // Empty filename invalid
        return false;
    }
    if (filename.length() > 255) {
        // Limit at common largest possible filename on Linux filesystems
        // to avoid unnecessary further validation
        // (On systems with smaller limits it will be caught by the OS)
        return false;
    }

    size_t offset = 0;
    while (offset < filename.size()) {
        utf8_parse_result result = common_parse_utf8_codepoint(filename, offset);

        if (result.status != utf8_parse_result::SUCCESS) {
            return false;
        }
        uint32_t c = result.codepoint;

        if ((result.bytes_consumed == 2 && c < 0x80) ||
            (result.bytes_consumed == 3 && c < 0x800) ||
            (result.bytes_consumed == 4 && c < 0x10000)) {
            return false;
        }

        // Check for forbidden codepoints:
        // - Control characters
        // - Unicode equivalents of illegal characters
        // - UTF-16 surrogate pairs
        // - UTF-8 replacement character
        // - Byte order mark (BOM)
        // - Illegal characters: / \ : * ? " < > |
        if (c <= 0x1F // Control characters (C0)
            || c == 0x7F // Control characters (DEL)
            || (c >= 0x80 && c <= 0x9F) // Control characters (C1)
            || c == 0xFF0E // Fullwidth Full Stop (period equivalent)
            || c == 0x2215 // Division Slash (forward slash equivalent)
            || c == 0x2216 // Set Minus (backslash equivalent)
            || (c >= 0xD800 && c <= 0xDFFF) // UTF-16 surrogate pairs
            || c > 0x10FFFF // Max Unicode limit
            || c == 0xFFFD // Replacement Character (UTF-8)
            || c == 0xFEFF // Byte Order Mark (BOM)
            || c == ':' || c == '*' // Illegal characters
            || c == '?' || c == '"' || c == '<' || c == '>' || c == '|') {
            return false;
        }
        if (!allow_subdirs && (c == '/' || c == '\\')) {
            // Subdirectories not allowed, reject path separators
            return false;
        }
        offset += result.bytes_consumed;
    }

    // Reject any leading or trailing ' ', or any trailing '.', these are stripped on Windows and will cause a different filename
    // Unicode and other whitespace is not affected, only 0x20 space
    if (filename.front() == ' ' || filename.back() == ' ' || filename.back() == '.') {
        return false;
    }

    // Reject any ".." (currently stricter than necessary, it should be fine to just check for == ".." instead)
    if (filename.find("..") != std::string::npos) {
        return false;
    }

    // Reject "."
    if (filename == ".") {
        return false;
    }

    return true;
}

#include <iostream>


#ifdef _WIN32
static std::wstring utf8_to_wstring(const std::string & str) {
    if (str.empty()) {
        return std::wstring();
    }

    int size = MultiByteToWideChar(CP_UTF8, 0, str.c_str(), (int)str.size(), NULL, 0);

    if (size <= 0) {
        return std::wstring();
    }

    std::wstring wstr(size, 0);
    MultiByteToWideChar(CP_UTF8, 0, str.c_str(), (int)str.size(), &wstr[0], size);

    return wstr;
}
#endif

// returns true if successful, false otherwise
bool fs_create_directory_with_parents(const std::string & path) {
#ifdef _WIN32
    std::wstring wpath = utf8_to_wstring(path);

    // if the path already exists, check whether it's a directory
    const DWORD attributes = GetFileAttributesW(wpath.c_str());
    if ((attributes != INVALID_FILE_ATTRIBUTES) && (attributes & FILE_ATTRIBUTE_DIRECTORY)) {
        return true;
    }

    size_t pos_slash = 0;

    // process path from front to back, procedurally creating directories
    while ((pos_slash = path.find('\\', pos_slash)) != std::string::npos) {
        const std::wstring subpath = wpath.substr(0, pos_slash);

        pos_slash += 1;

        // skip the drive letter, in some systems it can return an access denied error
        if (subpath.length() == 2 && subpath[1] == ':') {
            continue;
        }

        const bool success = CreateDirectoryW(subpath.c_str(), NULL);

        if (!success) {
            const DWORD error = GetLastError();

            // if the path already exists, ensure that it's a directory
            if (error == ERROR_ALREADY_EXISTS) {
                const DWORD attributes = GetFileAttributesW(subpath.c_str());
                if (attributes == INVALID_FILE_ATTRIBUTES || !(attributes & FILE_ATTRIBUTE_DIRECTORY)) {
                    return false;
                }
            } else {
                return false;
            }
        }
    }

    return true;
#else
    // if the path already exists, check whether it's a directory
    struct stat info;
    if (stat(path.c_str(), &info) == 0) {
        return S_ISDIR(info.st_mode);
    }

    size_t pos_slash = 1; // skip leading slashes for directory creation

    // process path from front to back, procedurally creating directories
    while ((pos_slash = path.find('/', pos_slash)) != std::string::npos) {
        const std::string subpath = path.substr(0, pos_slash);
        struct stat info;

        // if the path already exists, ensure that it's a directory
        if (stat(subpath.c_str(), &info) == 0) {
            if (!S_ISDIR(info.st_mode)) {
                return false;
            }
        } else {
            // create parent directories
            const int ret = mkdir(subpath.c_str(), 0755);
            if (ret != 0) {
                return false;
            }
        }

        pos_slash += 1;
    }

    return true;
#endif // _WIN32
}

bool fs_is_directory(const std::string & path) {
    std::filesystem::path dir(path);
    return std::filesystem::exists(dir) && std::filesystem::is_directory(dir);
}

std::string common_get_env(const std::string & name) {
    const char * value = std::getenv(name.c_str());
    return value == nullptr ? "" : value;
}

void common_set_env(const std::string & name, const std::string & value) {
#if defined(_WIN32)
    _putenv_s(name.c_str(), value.c_str());
#else
    if (value.empty()) {
        unsetenv(name.c_str());
    } else {
        setenv(name.c_str(), value.c_str(), 1);
    }
#endif
}

std::string fs_get_cache_directory() {
    std::string cache_directory = "";
    auto ensure_trailing_slash = [](std::string p) {
        // Make sure to add trailing slash
        if (p.empty() || p.back() != DIRECTORY_SEPARATOR) {
            p += DIRECTORY_SEPARATOR;
        }
        return p;
    };
    cache_directory = common_get_env("LLAMA_CACHE");
    if (cache_directory.empty()) {
#if defined(__linux__) || defined(__FreeBSD__) || defined(_AIX) || \
        defined(__OpenBSD__) || defined(__NetBSD__)
        const std::string xdg_cache_home = common_get_env("XDG_CACHE_HOME");
        const std::string home           = common_get_env("HOME");
        if (!xdg_cache_home.empty()) {
            cache_directory = xdg_cache_home;
        } else if (!home.empty()) {
            cache_directory = home + "/.cache/";
        } else {
#if defined(__linux__)
            /* no $HOME is defined, fallback to getpwuid */
            struct passwd *pw = getpwuid(getuid());
            if ((!pw) || (!pw->pw_dir)) {
                throw std::runtime_error("Failed to find $HOME directory");
            }

            cache_directory = std::string(pw->pw_dir) + std::string("/.cache/");
#else /* defined(__linux__) */
            throw std::runtime_error("Failed to find $HOME directory");
#endif /* defined(__linux__) */
        }
#elif defined(__APPLE__)
        cache_directory = common_get_env("HOME");
        if (cache_directory.empty()) {
            throw std::runtime_error("Failed to find $HOME directory");
        }
        cache_directory += "/Library/Caches/";
#elif defined(_WIN32)
        cache_directory = common_get_env("LOCALAPPDATA");
        if (cache_directory.empty()) {
            throw std::runtime_error("Failed to find %LOCALAPPDATA% directory");
        }
#elif defined(__EMSCRIPTEN__)
        GGML_ABORT("not implemented on this platform");
#else
#  error Unknown architecture
#endif
        cache_directory = ensure_trailing_slash(cache_directory);
        cache_directory += "llama.cpp";
    }
    return ensure_trailing_slash(cache_directory);
}

std::string fs_get_config_directory() {
    std::string config_directory = "";
    auto ensure_trailing_slash = [](std::string p) {
        if (p.empty() || p.back() != DIRECTORY_SEPARATOR) {
            p += DIRECTORY_SEPARATOR;
        }
        return p;
    };
#if defined(__linux__) || defined(__FreeBSD__) || defined(_AIX) || \
        defined(__OpenBSD__) || defined(__NetBSD__) || defined(__APPLE__)
    const std::string xdg_config_home = common_get_env("XDG_CONFIG_HOME");
    const std::string home            = common_get_env("HOME");
    if (!xdg_config_home.empty()) {
        config_directory = xdg_config_home;
    } else if (!home.empty()) {
        config_directory = home + "/.config/";
    } else {
#if defined(__linux__)
        /* no $HOME is defined, fallback to getpwuid */
        struct passwd *pw = getpwuid(getuid());
        if ((!pw) || (!pw->pw_dir)) {
            throw std::runtime_error("Failed to find $HOME directory");
        }

        config_directory = std::string(pw->pw_dir) + std::string("/.config/");
#else
        throw std::runtime_error("Failed to find $HOME directory");
#endif
    }
#elif defined(_WIN32)
    config_directory = common_get_env("APPDATA");
    if (config_directory.empty()) {
        throw std::runtime_error("Failed to find %APPDATA% directory");
    }
#elif defined(__EMSCRIPTEN__)
    // caller decides what to do when there is no config directory
    throw std::runtime_error("not implemented on this platform");
#else
#  error Unknown architecture
#endif
    config_directory = ensure_trailing_slash(config_directory);
    config_directory += "llama.cpp";
    return ensure_trailing_slash(config_directory);
}

std::string fs_get_cache_file(const std::string & filename) {
    GGML_ASSERT(filename.find(DIRECTORY_SEPARATOR) == std::string::npos);
    std::string cache_directory = fs_get_cache_directory();
    const bool success = fs_create_directory_with_parents(cache_directory);
    if (!success) {
        throw std::runtime_error("failed to create cache directory: " + cache_directory);
    }
    return cache_directory + filename;
}

std::vector<common_file_info> fs_list(const std::string & path, bool include_directories) {
    std::vector<common_file_info> files;
    if (path.empty()) return files;

    std::filesystem::path dir(path);
    if (!std::filesystem::exists(dir) || !std::filesystem::is_directory(dir)) {
        return files;
    }

    for (const auto & entry : std::filesystem::directory_iterator(dir)) {
        try {
            // Only include regular files (skip directories)
            const auto & p = entry.path();
            if (std::filesystem::is_regular_file(p)) {
                common_file_info info;
                info.path   = p.string();
                info.name   = p.filename().string();
                info.is_dir = false;
                try {
                    info.size = static_cast<size_t>(std::filesystem::file_size(p));
                } catch (const std::filesystem::filesystem_error &) {
                    info.size = 0;
                }
                files.push_back(std::move(info));
            } else if (include_directories && std::filesystem::is_directory(p)) {
                common_file_info info;
                info.path   = p.string();
                info.name   = p.filename().string();
                info.size   = 0; // Directories have no size
                info.is_dir = true;
                files.push_back(std::move(info));
            }
        } catch (const std::filesystem::filesystem_error &) {
            // skip entries we cannot inspect
            continue;
        }
    }

    return files;
}

std::ifstream fs_open_ifstream(const std::string & fname, std::ios_base::openmode mode) {
#ifdef _WIN32
    int wlen = MultiByteToWideChar(CP_UTF8, 0, fname.c_str(), -1, NULL, 0);
    if (!wlen) { return std::ifstream(); }
    std::vector<wchar_t> wfname(wlen);
    (void)MultiByteToWideChar(CP_UTF8, 0, fname.c_str(), -1, wfname.data(), wlen);
    return std::ifstream(wfname.data(), mode);
#else
    return std::ifstream(fname, mode);
#endif
}

//
// TTY utils
//

bool tty_can_use_colors() {
    // Check NO_COLOR environment variable (https://no-color.org/)
    if (const char * no_color = std::getenv("NO_COLOR")) {
        if (no_color[0] != '\0') {
            return false;
        }
    }

    // Check TERM environment variable
    if (const char * term = std::getenv("TERM")) {
        if (std::strcmp(term, "dumb") == 0) {
            return false;
        }
    }

    // Check if stdout and stderr are connected to a terminal
    // We check both because log messages can go to either
    bool stdout_is_tty = isatty(fileno(stdout));
    bool stderr_is_tty = isatty(fileno(stderr));

    return stdout_is_tty || stderr_is_tty;
}

//
// Model utils
//

// TODO: move to common/sampling
static void common_init_sampler_from_model(
    const llama_model * model,
    common_params_sampling & sparams) {

    const uint64_t config = sparams.user_sampling_config;

    auto get_int32 = [&](const char * key, int32_t & dst, uint64_t user_config) {
        if (config & user_config) {
            return;
        }

        char buf[64] = {0};
        if (llama_model_meta_val_str(model, key, buf, sizeof(buf)) > 0) {
            char * end = nullptr;
            int32_t v = strtol(buf, &end, 10);
            if (end && end != buf) {
                dst = v;
            }
        }
    };

    auto get_float = [&](const char * key, float & dst, uint64_t user_config) {
        if (config & user_config) {
            return;
        }

        char buf[128] = {0};
        if (llama_model_meta_val_str(model, key, buf, sizeof(buf)) > 0) {
            char * end = nullptr;
            float v = strtof(buf, &end);
            if (end && end != buf) {
                dst = v;
            }
        }
    };

    // Sampling sequence
    if (!(config & common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_SAMPLERS)) {
        char buf[512] = {0};
        if (llama_model_meta_val_str(model, llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_SEQUENCE), buf, sizeof(buf)) > 0) {
            const std::vector<std::string> sampler_names = string_split<std::string>(std::string(buf), ';');
            if (!sampler_names.empty()) {
                sparams.samplers = common_sampler_types_from_names(sampler_names);
            }
        }
    }

    get_int32(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_TOP_K),           sparams.top_k,           common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_TOP_K);
    get_float(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_TOP_P),           sparams.top_p,           common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_TOP_P);
    get_float(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_MIN_P),           sparams.min_p,           common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_MIN_P);
    get_float(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_XTC_PROBABILITY), sparams.xtc_probability, common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_XTC_PROBABILITY);
    get_float(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_XTC_THRESHOLD),   sparams.xtc_threshold,   common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_XTC_THRESHOLD);
    get_float(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_TEMP),            sparams.temp,            common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_TEMP);
    get_int32(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_PENALTY_LAST_N),  sparams.penalty_last_n,  common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_PENALTY_LAST_N);
    get_float(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_PENALTY_REPEAT),  sparams.penalty_repeat,  common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_PENALTY_REPEAT);
    get_int32(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT),        sparams.mirostat,        common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_MIROSTAT);
    get_float(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT_TAU),    sparams.mirostat_tau,    common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_MIROSTAT_TAU);
    get_float(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT_ETA),    sparams.mirostat_eta,    common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_MIROSTAT_ETA);
}

struct common_init_result::impl {
    impl() = default;
    ~impl() = default;

    // note: the order in which model, context, etc. are declared matters because their destructors will be called bottom-to-top

    llama_model_ptr   model;
    llama_context_ptr context;

    std::vector<llama_adapter_lora_ptr> lora;

    std::vector<common_sampler_ptr> samplers;
    std::vector<llama_sampler_seq_config> samplers_seq_config;
};

// True when the real (non-probe) load will keep CPU-resident model weights
// on a plain lazy mmap: LOAD_MODE_AUTO resolves to mmap whenever the device
// supports it (the common case), and LOAD_MODE_MMAP asks for it explicitly.
// MLOCK/MMAP_MLOCK force full residency (that's the point of mlock), and
// NONE/DIRECT_IO never mmap at all, so those still need the real byte count.
static bool common_mmap_is_lazy(const llama_model_params & mparams) {
    return mparams.load_mode == LLAMA_LOAD_MODE_AUTO || mparams.load_mode == LLAMA_LOAD_MODE_MMAP;
}

// margin: bytes that must remain free per device on top of what's needed.
// Defaults to 0 (bare fit) for callers that only ask "would this load at all".
// A caller whose answer will be re-judged by common_fit_params() must pass the
// same margin that check will demand - otherwise placement approves a config
// that fit then rejects, and the two silently disagree.
//
// host_model_lazy_mmap: whether the host (CPU) entry's `model` bytes are a
// lazily-paged zero-copy mapping rather than a real allocation - see
// common_mmap_is_lazy(). When true, those bytes are excluded from what must
// fit in available RAM: the kernel reclaims clean, unmodified file pages
// under pressure instead of committing them, so CPU-offloaded ("cold") MoE
// experts don't need to be simultaneously resident the way a real copy would
// - only the host's KV cache and compute-buffer bytes (context/compute) are
// genuine allocations that still have to fit.
static bool common_device_memory_data_fits(
        const common_device_memory_data_vec & data, int64_t margin = 0, bool host_model_lazy_mmap = false) {
    for (size_t i = 0; i < data.size(); i++) {
        const auto & d = data[i];
        if (d.total <= 0) {
            continue; // not a real device (host aggregate, or a device with unknown budget)
        }
        // by construction (common_get_device_memory_data_impl), the host
        // aggregate is always the last entry
        const bool is_host = (i + 1 == data.size());
        const size_t model_bytes = (is_host && host_model_lazy_mmap) ? 0 : d.model;
        const int64_t needed = (int64_t) model_bytes + (int64_t) d.context + (int64_t) d.compute;
        if (needed + margin > d.free) {
            return false;
        }
    }
    return true;
}

// The buffer type -ncmoe places its overridden experts in. Plain
// ggml_backend_cpu_buffer_type() is host memory the CPU backend can compute
// on, but it's invisible to op_offload: op_offload's backend-assignment path
// (ggml_backend_sched_backend_id_from_cur) only considers diverting a node to
// GPU for buffers a GPU device's own get_host_buffer_type() produced (the
// same buffer type llama-model.cpp's make_cpu_buft_list already uses for
// tensors that overflow VRAM naturally, for exactly this reason - see its
// own comment: "useful when processing of large batches is offloaded to a
// GPU device, since it reduces the time spent on data transfers"). Plain CPU
// buft tensors stay CPU-computed forever, at any batch size - verified
// empirically: -ncmoe layers never engaged op_offload even at n_tokens on
// the order of thousands. Resolving the same host buffer type here (falling
// back to plain CPU buft if no device offers one, e.g. CPU-only builds)
// makes explicit -ncmoe placements op_offload-eligible during prefill the
// same way naturally-overflowing tensors already are, without touching
// ggml_moe_cache at all - its CPU-dispatch gate checks buffer *usage*
// (WEIGHTS), not buffer *type*, so decode-time behavior is unaffected.
static ggml_backend_buffer_type_t common_moe_cpu_override_buft() {
    static ggml_backend_buffer_type_t buft = [] () -> ggml_backend_buffer_type_t {
        for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
                continue;
            }
            ggml_backend_buffer_type_t host_buft = ggml_backend_dev_host_buffer_type(dev);
            if (host_buft) {
                if (getenv("MOE_CACHE_DEBUG_GATE")) {
                    fprintf(stderr, "[ncmoe-buft-dbg] resolved HOST buft=%p (name=%s) from dev=%s\n",
                            (void*)host_buft, ggml_backend_buft_name(host_buft), ggml_backend_dev_name(dev));
                }
                return host_buft;
            }
        }
        if (getenv("MOE_CACHE_DEBUG_GATE")) {
            fprintf(stderr, "[ncmoe-buft-dbg] no GPU host buft found, falling back to plain CPU buft\n");
        }
        return ggml_backend_cpu_buffer_type();
    }();
    return buft;
}

// Shared by Layer 1 (use the safe floor as-is) and Layer 2/--moe-calibrate
// (use the safe floor as the starting point for empirical throughput
// candidates above it). Keeps its own C-string storage alive for the
// lifetime of the process, same as the override lists it hands back.
static std::vector<llama_model_tensor_buft_override> common_moe_build_cpu_overrides(uint32_t n) {
    static std::list<std::string> pattern_storage;
    std::vector<llama_model_tensor_buft_override> ov;
    ggml_backend_buffer_type_t buft = common_moe_cpu_override_buft();
    for (uint32_t i = 0; i < n; i++) {
        pattern_storage.push_back(llm_ffn_exps_block_regex((int) i));
        ov.push_back({pattern_storage.back().c_str(), buft});
    }
    ov.push_back({nullptr, nullptr});
    return ov;
}

static bool common_moe_fits_with_n(
        const char * path_model, const llama_model_params & mparams_base,
        const llama_context_params & cparams, uint32_t n,
        std::vector<ggml_backend_dev_t> & devs, uint32_t & hp_ngl, uint32_t & hp_n_ctx_train, uint32_t & hp_n_expert,
        int64_t margin = 0) {
    std::vector<llama_model_tensor_buft_override> trial = common_moe_build_cpu_overrides(n);
    llama_model_params mparams_trial = mparams_base;
    mparams_trial.tensor_buft_overrides = trial.data();
    try {
        common_device_memory_data_vec trial_data = common_get_device_memory_data(
                path_model, &mparams_trial, &cparams, devs, hp_ngl, hp_n_ctx_train, hp_n_expert, GGML_LOG_LEVEL_ERROR);
        return common_device_memory_data_fits(trial_data, margin, common_mmap_is_lazy(mparams_base));
    } catch (const std::exception &) {
        return false;
    }
}

struct common_moe_fit_probe_result {
    bool     is_moe        = false;
    bool     already_fits  = false;
    bool     found_safe_n  = false;
    uint32_t safe_n        = 0;
    uint32_t n_layer       = 0;
};

// Binary-searches the minimal MoE CPU-offload layer count N such that the
// model fits in available device memory - fit is monotonic in N (more
// offload never needs more memory), so this is O(log n_layer) no-alloc
// probes, not O(n_layer). A live, per-launch computation: the answer
// depends on -c/--parallel/free VRAM at that moment, not a cached constant.
static common_moe_fit_probe_result common_moe_find_safe_layers(
        const char * path_model, const llama_model_params & mparams_base, const llama_context_params & cparams,
        int64_t margin = 0) {
    common_moe_fit_probe_result result;

    std::vector<ggml_backend_dev_t> devs;
    uint32_t hp_ngl = 0, hp_n_ctx_train = 0, hp_n_expert = 0;
    common_device_memory_data_vec data;
    try {
        data = common_get_device_memory_data(path_model, &mparams_base, &cparams, devs, hp_ngl, hp_n_ctx_train, hp_n_expert,
                GGML_LOG_LEVEL_ERROR);
    } catch (const std::exception & e) {
        // Deliberately logged, not swallowed: an empty result here reads
        // identically to "not is_moe" to every caller, which previously
        // misreported a real probe crash (invalid n_seq_max, OOM, etc.) as
        // "model has no MoE experts" - found by hitting exactly that with
        // --moe-calibrate + --model-draft together.
        LOG_ERR("%s: no-alloc probe failed: %s\n", __func__, e.what());
        return result;
    }
    result.n_layer = hp_ngl;
    result.is_moe  = hp_n_expert > 0;

    if (common_device_memory_data_fits(data, margin, common_mmap_is_lazy(mparams_base))) {
        result.already_fits = true;
        return result;
    }
    if (!result.is_moe) {
        return result;
    }
    if (!common_moe_fits_with_n(path_model, mparams_base, cparams, hp_ngl, devs, hp_ngl, hp_n_ctx_train, hp_n_expert, margin)) {
        return result; // nothing fits even with everything offloaded
    }

    uint32_t lo = 1, hi = hp_ngl;
    while (lo < hi) {
        const uint32_t mid = lo + (hi - lo) / 2;
        if (common_moe_fits_with_n(path_model, mparams_base, cparams, mid, devs, hp_ngl, hp_n_ctx_train, hp_n_expert, margin)) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    result.found_safe_n = true;
    result.safe_n        = lo;
    return result;
}

// Bumped when calibration gains a check that changes which configurations it
// is willing to record. An entry written before a gate existed was never
// tested by it, so the fields that gate protects must not be applied from it.
// 1 = measured with the degeneracy guard, the short-prompt quality pass, the
// output-fidelity check against a substitution-free reference, and the
// repeat-identical-request reproducibility check.
static constexpr int COMMON_MOE_CALIBRATION_GATES_VERSION = 1;

struct common_moe_calibration_entry {
    int         n_cpu_moe       = 0;
    int         n_threads       = -1;
    int         n_threads_batch = -1;
    int         spec_n_max      = -1; // -1 = no MTP calibration recorded
    int         concurrency     = 1;  // > 1: tok_per_sec is aggregate throughput at this many concurrent requests, not solo
    double      tok_per_sec     = 0.0;
    int         moe_cache_mb    = -1; // -1 = not calibrated, use --moe-cache auto
    int         substitute_min_rank = -1; // -1 = not calibrated, use the runtime default gate
    // Stand-in quality bar, in standard deviations below the mean
    // co-activation of the residents examined. This is the continuous form of
    // the wait-or-substitute balance: high = wait for the real expert, low =
    // run with whatever is resident. NaN/unset = not calibrated. Searched
    // separately from the rank floor because they gate different things -
    // rank is about how much the ROUTER wanted this expert, sigma is about
    // how good the available replacement actually is.
    double      substitute_quality_sigma = std::numeric_limits<double>::quiet_NaN();
    // Per-device fit margin (-fitt/--fit-target, MiB). -1 = not calibrated.
    //
    // This is not a minor knob: common_maybe_raise_moe_for_ctx reserves
    // 3 x fit_target of VRAM before deciding placement, and that reservation
    // is what silently RAISES a requested -ncmoe until the context "fits".
    // Measured on this fork (Ornith-1.5-35B-Q4_K_M, RTX 3060 12 GB, -c 4096,
    // requesting -ncmoe 8):
    //
    //     fitt=1024 (default) -> forced to ncmoe 27, 47.32 tok/s,  8951 MiB used
    //     fitt=640            -> forced to ncmoe 24, 50.37 tok/s, 10307 MiB used
    //     fitt=448            -> forced to ncmoe 23, 51.61 tok/s, 10741 MiB used
    //
    // i.e. the default margin left ~3 GiB of a 12 GiB card unused and pushed
    // 4 extra layers of experts onto the CPU, which is where that throughput
    // went. The margin genuinely cannot be dropped to zero (it covers real
    // weight loading and lazy CUDA graph capture that the no-alloc fit probe
    // cannot see - removing it reproduced a hard 4k context collapse), so the
    // right value is neither "3 GiB always" nor "as small as possible": it is
    // hardware- and model-specific, which makes it exactly the kind of thing
    // this calibration exists to measure instead of guess.
    int         fit_target_mb   = -1;
    // Heat-aware intra-expert neuron subsetting (GGML_CUDA_MOE_CACHE_NEURON_REDUCE
    // and friends). -1 = not calibrated / leave to the environment.
    //
    // Recorded here because this mechanism changes the VRAM arithmetic every
    // other value in this struct was measured under: a converted expert costs
    // K rows instead of the tensor's full width (measured 50% for K=256 on
    // Ornith's 576 KiB gate/up experts), so the same hit rate needs less cache
    // VRAM than it used to. A calibration taken with reduction off is not
    // valid for a run with it on, and vice versa - hence storing it rather
    // than letting the two drift silently apart.
    int         neuron_reduce_k         = -1; // > 0 also means "reduction was enabled"
    int         neuron_reduce_budget_mb = -1;
    // How many layers common_moe_calibrate found best kept resident on GPU
    // (the -ngl a real launch should use), -1 = not calibrated / use whatever
    // -ngl was requested. Distinct from n_cpu_moe: this trades GPU-resident
    // dense/attention compute for VRAM the expert cache converts into hit
    // rate, the same "extra CPU offload pays for itself in cache hits"
    // effect n_cpu_moe's own search already knows about (see the ncmoe_hi
    // extension comment above) - just on the layer-residency axis instead of
    // the expert-placement axis. Only searched/applied when the model didn't
    // already fit on GPU as the model's own full layer count (see
    // common_moe_calibrate): a model with room to spare has nothing to trade.
    int         n_gpu_layers            = -1;
    std::string calibrated_at;
    // False when this entry was found by the -ngl-relaxed fallback below, i.e.
    // it was measured at a different -ngl than this launch is using. The
    // fields that depend on layer residency (placement, VRAM sizing) are then
    // not applicable and must be skipped; the ones that don't are still real
    // measurements of this machine and this model. See
    // common_moe_calibration_lookup.
    bool        ngl_exact               = true;
    // 0 for any entry written before gate versioning existed.
    int         gates_version           = 0;
};

// Key identifies "the same launch, calibrated before": GPU signature (name +
// total VRAM per device - not free VRAM, which fluctuates and would make
// the cache miss on unrelated background load), model file identity (path +
// size, cheap proxy for content without hashing multi-GB files), and the
// context shape that determines KV-cache footprint. Any change here (new
// GPU, different model, different -c/--parallel/-ngl) is a real cache miss,
// not staleness - a stale entry is caught separately by the fits-check in
// the lookup path, not by this key.
static std::string common_moe_calibration_key(const char * path_model, const common_params & params) {
    std::string gpu_sig;
    const size_t ndev = ggml_backend_dev_count();
    for (size_t i = 0; i < ndev; i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
            continue;
        }
        size_t dfree = 0, dtotal = 0;
        ggml_backend_dev_memory(dev, &dfree, &dtotal);
        // Round to the nearest GiB. Total VRAM was chosen over free precisely
        // because it should not move - but the value the backend reports does
        // drift by a few MiB run to run (observed 11900 and 11909 MiB for the
        // same card, against nvidia-smi's 12288), and at MiB precision that
        // silently rewrites the key and orphans every entry cached under the
        // old one. Qwen re-calibrated on every launch for exactly this reason,
        // whatever -c and --parallel it was given. A GiB bucket is still
        // specific enough to separate real hardware while being immune to the
        // jitter.
        const size_t dtotal_gib = (dtotal + (512ull << 20)) >> 30;
        gpu_sig += string_format("%s:%zuG;", ggml_backend_dev_name(dev), dtotal_gib);
    }
    long long model_size = 0;
    struct stat st;
    if (stat(path_model, &st) == 0) {
        model_size = (long long) st.st_size;
    }
    // Resolve the context the same way every other consumer here does. Left
    // raw, a bare launch (n_ctx == 0, "the context the model was trained
    // with") built a key of c0 that could never match an entry cached at a
    // resolved context, so it re-calibrated from scratch every start and then
    // could not find what it had just written.
    const uint32_t key_ctx = params.n_ctx > 0 ? params.n_ctx : 4096;
    return string_format("%s|%s|%lld|c%u|p%u|ngl%d",
            gpu_sig.c_str(), path_model, model_size, key_ctx, params.n_parallel, params.n_gpu_layers);
}

// The same key with the -ngl component removed. Used to find an entry
// measured on this GPU, this model and this context shape but at a different
// layer residency - see common_moe_calibration_lookup for why that is worth
// finding rather than treating as a plain miss.
static std::string common_moe_calibration_key_prefix(const char * path_model, const common_params & params) {
    // Everything up to and including the model size - i.e. GPU + model
    // identity, without the context/parallel/-ngl shape fields that drift.
    const std::string key = common_moe_calibration_key(path_model, params);
    const size_t cut = key.find("|c");
    return cut == std::string::npos ? key : key.substr(0, cut);
}

static void common_moe_apply_quality_knobs(const common_moe_calibration_entry & cal) {
    // An entry recorded before the output gates existed says nothing about
    // output. The substitution floor is the clearest case: a gemma-4 entry
    // measured at 12:46 recorded rank 2 purely because it was fastest, and
    // when the fidelity check was added later it scored that same rank 2 at
    // 0.33 - the output diverged from the substitution-free reference after a
    // third of its tokens while still reading as fluent English, which is
    // exactly what the old degeneracy-only guard could not see. Applying it
    // anyway is how a stale measurement quietly degrades every later launch.
    //
    // Deliberately narrow: this skips only the fields the new gates protect.
    // Placement, thread count, cache size and fit margin are throughput
    // measurements that the gates never had an opinion about, so they stay
    // valid and keep being applied from the same entry.
    if (cal.gates_version < COMMON_MOE_CALIBRATION_GATES_VERSION) {
        if (cal.substitute_min_rank >= 0 || !std::isnan(cal.substitute_quality_sigma)) {
            LOG_WRN("%s: ignoring the cached substitution settings - that entry (%s) predates the output "
                    "fidelity and reproducibility checks, so they were never applied to it. Re-run "
                    "--moe-calibrate to measure them; everything else from the entry still applies\n",
                    __func__, cal.calibrated_at.empty() ? "undated" : cal.calibrated_at.c_str());
        }
        return;
    }
    if (cal.substitute_min_rank >= 0 && !getenv("GGML_CUDA_MOE_CACHE_SUBSTITUTE_MIN_RANK")) {
#if defined(_WIN32)
        _putenv_s("GGML_CUDA_MOE_CACHE_SUBSTITUTE_MIN_RANK",
                std::to_string(cal.substitute_min_rank).c_str());
#else
        setenv("GGML_CUDA_MOE_CACHE_SUBSTITUTE_MIN_RANK",
                std::to_string(cal.substitute_min_rank).c_str(), 1);
#endif
        LOG_WRN("%s: using calibrated substitution floor of rank %d\n", __func__, cal.substitute_min_rank);
    }
    if (!std::isnan(cal.substitute_quality_sigma) &&
        !getenv("GGML_CUDA_MOE_CACHE_SUBSTITUTE_QUALITY_SIGMA")) {
#if defined(_WIN32)
        _putenv_s("GGML_CUDA_MOE_CACHE_SUBSTITUTE_QUALITY_SIGMA",
                std::to_string(cal.substitute_quality_sigma).c_str());
#else
        setenv("GGML_CUDA_MOE_CACHE_SUBSTITUTE_QUALITY_SIGMA",
                std::to_string(cal.substitute_quality_sigma).c_str(), 1);
#endif
        LOG_WRN("%s: using calibrated stand-in quality bar of %.1f sigma\n",
                __func__, cal.substitute_quality_sigma);
    }
    if (cal.neuron_reduce_k > 0 && cal.neuron_reduce_budget_mb > 0 &&
        !getenv("GGML_CUDA_MOE_CACHE_NEURON_REDUCE")) {
#if defined(_WIN32)
        _putenv_s("GGML_CUDA_MOE_CACHE_NEURON_REDUCE", "1");
        _putenv_s("GGML_CUDA_MOE_CACHE_NEURON_REDUCE_K", std::to_string(cal.neuron_reduce_k).c_str());
        _putenv_s("GGML_CUDA_MOE_CACHE_NEURON_REDUCE_BUDGET_MB",
                std::to_string(cal.neuron_reduce_budget_mb).c_str());
#else
        setenv("GGML_CUDA_MOE_CACHE_NEURON_REDUCE", "1", 1);
        setenv("GGML_CUDA_MOE_CACHE_NEURON_REDUCE_K", std::to_string(cal.neuron_reduce_k).c_str(), 1);
        setenv("GGML_CUDA_MOE_CACHE_NEURON_REDUCE_BUDGET_MB",
                std::to_string(cal.neuron_reduce_budget_mb).c_str(), 1);
#endif
        LOG_WRN("%s: using calibrated neuron-reduce k=%d / %d MiB\n",
                __func__, cal.neuron_reduce_k, cal.neuron_reduce_budget_mb);
    }
}

static std::string common_moe_calibration_cache_path() {
    return fs_get_cache_directory() + "moe-calibration.json";
}

static bool common_moe_calibration_lookup(
        const char * path_model, const common_params & params, common_moe_calibration_entry & out) {
    std::ifstream f(common_moe_calibration_cache_path());
    static const bool trace_lookup = getenv("LLAMA_MOE_CALIB_TRACE") != nullptr;
    if (trace_lookup) {
        // fprintf, not LOG_WRN: the logging macros are gated on a verbosity
        // threshold that is not necessarily set yet this early in startup, and
        // a diagnostic that can be silently swallowed is worse than none - it
        // reads as "this code did not run" when it did.
        fprintf(stderr, "[calib-trace] path='%s' good=%d\n",
                common_moe_calibration_cache_path().c_str(), (int) f.good());
        fflush(stderr);
    }
    if (!f.good()) {
        return false;
    }
    try {
        nlohmann::json j;
        f >> j;
        const std::string key = common_moe_calibration_key(path_model, params);
        bool exact = j.contains(key);
        if (!exact) {
            // Print the key on a miss. Three separate theories about why a
            // cached entry was not being found (context, parallelism, GPU
            // signature) were each wrong in turn, and every one of them would
            // have been settled immediately by seeing the string itself.
            if (trace_lookup) {
                fprintf(stderr, "[calib-trace] MISS key='%s' (cache has %zu entries)\n",
                        key.c_str(), (size_t) j.size());
                fflush(stderr);
            }
        }
        std::string use_key = key;
        if (!exact) {
            // Relaxed retry: same GPU, model and context shape, different
            // -ngl. Pinning -ngl used to discard the entire entry, because
            // -ngl is part of the key - so `-ngl 20` on a box calibrated at
            // the default threw away the measured thread count, substitution
            // floor and neuron-reduce settings along with the placement, and
            // silently fell back to runtime defaults for all of them. Only
            // some of those values depend on layer residency. Find the entry
            // anyway and let the apply path take the parts that still hold
            // (see ngl_exact). Prefer the fastest such entry when several
            // -ngl values have been calibrated - they were all measured on
            // this same machine and model.
            // Match on GPU + model + size only, ignoring the context,
            // parallelism and -ngl fields. Those three have each drifted in
            // practice - the GPU signature moved 11900 vs 11909 MiB for one
            // card, calibration rewrites n_ctx and n_parallel as it adapts, and
            // -ngl is pinned by the user - and any one of them changing orphans
            // the entry, so the run re-calibrates and then cannot find what it
            // just wrote. An entry for this model on this machine is worth more
            // than a perfect key match: the apply path already refuses anything
            // that does not transfer (see ngl_exact), so a relaxed hit costs
            // nothing beyond the values that legitimately carry over.
            const std::string prefix = common_moe_calibration_key_prefix(path_model, params);
            const size_t model_at = prefix.find('|');
            const std::string model_part = model_at == std::string::npos
                    ? prefix : prefix.substr(model_at);
            double best = -1.0;
            for (const auto & item : j.items()) {
                const size_t item_at = item.key().find('|');
                const std::string item_model = item_at == std::string::npos
                        ? item.key() : item.key().substr(item_at);
                const bool same_prefix = item.key().compare(0, prefix.size(), prefix) == 0;
                // model_part starts at the first '|' and covers path + size, so
                // this pins the model file identity while letting the GPU
                // signature and the trailing shape fields vary.
                const bool same_model = item_model.compare(0, model_part.size(), model_part) == 0;
                if (!same_prefix && !same_model) {
                    continue;
                }
                const double tps = item.value().value("tok_per_sec", 0.0);
                if (tps > best) {
                    best    = tps;
                    use_key = item.key();
                }
            }
            if (best < 0.0) {
                return false;
            }
        }
        const auto & e = j.at(use_key);
        out.ngl_exact = exact;
        out.n_cpu_moe       = e.value("n_cpu_moe", 0);
        out.n_threads       = e.value("n_threads", -1);
        out.n_threads_batch = e.value("n_threads_batch", -1);
        out.spec_n_max      = e.value("spec_n_max", -1);
        out.concurrency     = e.value("concurrency", 1);
        out.tok_per_sec     = e.value("tok_per_sec", 0.0);
        out.moe_cache_mb    = e.value("moe_cache_mb", -1);
        out.substitute_min_rank = e.value("substitute_min_rank", -1);
        out.substitute_quality_sigma = e.value("substitute_quality_sigma",
                std::numeric_limits<double>::quiet_NaN());
        // Absent in entries written before these were calibrated - the
        // value() defaults keep such an entry loadable rather than making
        // it a hard cache miss, since the fields it DOES carry are still
        // valid measurements.
        out.fit_target_mb           = e.value("fit_target_mb", -1);
        out.neuron_reduce_k         = e.value("neuron_reduce_k", -1);
        out.neuron_reduce_budget_mb = e.value("neuron_reduce_budget_mb", -1);
        out.n_gpu_layers            = e.value("n_gpu_layers", -1);
        out.calibrated_at   = e.value("calibrated_at", std::string());
        out.gates_version   = e.value("gates_version", 0);
        return true;
    } catch (const std::exception &) {
        return false;
    }
}

static void common_moe_calibration_save(
        const char * path_model, const common_params & params, const common_moe_calibration_entry & entry) {
    const std::string path = common_moe_calibration_cache_path();
    nlohmann::json j = nlohmann::json::object();
    {
        std::ifstream f(path);
        if (f.good()) {
            try {
                f >> j;
            } catch (const std::exception &) {
                j = nlohmann::json::object();
            }
        }
    }
    const std::string key = common_moe_calibration_key(path_model, params);
    j[key] = {
        {"n_cpu_moe",       entry.n_cpu_moe},
        {"n_threads",       entry.n_threads},
        {"n_threads_batch", entry.n_threads_batch},
        {"spec_n_max",      entry.spec_n_max},
        {"concurrency",     entry.concurrency},
        {"tok_per_sec",     entry.tok_per_sec},
        {"moe_cache_mb",    entry.moe_cache_mb},
        {"substitute_min_rank", entry.substitute_min_rank},
        {"substitute_quality_sigma", entry.substitute_quality_sigma},
        {"gates_version", COMMON_MOE_CALIBRATION_GATES_VERSION},
        {"fit_target_mb",           entry.fit_target_mb},
        {"neuron_reduce_k",         entry.neuron_reduce_k},
        {"neuron_reduce_budget_mb", entry.neuron_reduce_budget_mb},
        {"n_gpu_layers",    entry.n_gpu_layers},
        {"calibrated_at",   entry.calibrated_at},
    };
    fs_create_directory_with_parents(fs_get_cache_directory());
    std::ofstream out(path);
    out << j.dump(2);
}

bool common_moe_should_auto_calibrate(common_params & params) {
    // Same rule as common_maybe_autoplace_moe_cpu(): an explicit -ncmoe (or
    // any other explicit tensor-buffer-type override) is a deliberate user
    // choice and auto-calibration must never second-guess it.
    for (const auto & o : params.tensor_buft_overrides) {
        if (o.pattern != nullptr) {
            return false;
        }
    }

    const char * path_model = params.model.path.c_str();
    common_moe_calibration_entry cached;
    if (common_moe_calibration_lookup(path_model, params, cached)) {
        return false; // already have a cached answer for this combination
    }

    // Cheap no-alloc probe (GGUF header + device memory, no tensor data
    // loaded) - only trigger the (multi-minute) automatic calibration run
    // for models that actually need MoE CPU-offload placement decided.
    // Dense models, and MoE models that already fit as configured, have
    // nothing for --moe-calibrate to usefully tune.
    auto mparams = common_model_params_to_llama(params);
    auto cparams = common_context_params_to_llama(params);
    common_moe_fit_probe_result probe = common_moe_find_safe_layers(path_model, mparams, cparams);
    if (!probe.is_moe || probe.already_fits) {
        return false;
    }
    return true;
}

// Golden-section search over integers in [lo, hi] for the argmax of a
// unimodal (single-peak) function - the shape our own -ncmoe and
// spec-draft-n-max sweeps actually showed empirically (a real interior
// peak, not monotonic), not a guess. O(log(hi-lo)) evaluations instead of
// O(hi-lo) for a full grid - matters because each evaluation here is a
// real model load, not a cheap probe. Memoizes every point actually
// measured (golden-section revisits nearby points) and returns it via
// `trace` for logging/debugging.
template <typename F>
static int common_golden_section_search_max(
        int lo, int hi, F && measure, std::map<int, double> & trace) {
    if (lo >= hi) {
        if (trace.find(lo) == trace.end()) {
            trace[lo] = measure(lo);
        }
        return lo;
    }
    auto measured = [&](int x) -> double {
        auto it = trace.find(x);
        if (it != trace.end()) {
            return it->second;
        }
        const double v = measure(x);
        trace[x] = v;
        return v;
    };

    const double gr = 0.6180339887498949; // 1/phi
    int x1 = lo + (int) std::lround((1.0 - gr) * (hi - lo));
    int x2 = lo + (int) std::lround(gr * (hi - lo));
    x1 = std::max(lo, std::min(hi, x1));
    x2 = std::max(lo, std::min(hi, x2));
    double f1 = measured(x1);
    double f2 = measured(x2);

    while (hi - lo > 2) {
        if (f1 < f2) {
            lo = x1;
            x1 = x2; f1 = f2;
            x2 = std::max(lo, std::min(hi, lo + (int) std::lround(gr * (hi - lo))));
            if (x2 == x1) {
                break;
            }
            f2 = measured(x2);
        } else {
            hi = x2;
            x2 = x1; f2 = f1;
            x1 = std::max(lo, std::min(hi, lo + (int) std::lround((1.0 - gr) * (hi - lo))));
            if (x1 == x2) {
                break;
            }
            f1 = measured(x1);
        }
    }
    // small remaining range - just check every point directly, cheap now
    int best = lo;
    double best_val = measured(lo);
    for (int x = lo + 1; x <= hi; x++) {
        const double v = measured(x);
        if (v > best_val) {
            best_val = v;
            best = x;
        }
    }
    return best;
}

// Spawns a real llama-server subprocess with the given placement/n_max/
// thread count, waits for it to become healthy, sends real chat-completion
// requests (averaged over a couple of representative prompts), and returns
// measured decode tok/s (already net-of-rejection when MTP is active - see
// the MTP section of docs/moe-cache-colibri-notes.md: predicted_per_second
// is tokens_predicted/elapsed, and tokens_predicted only counts tokens that
// survived speculative verification, so rejected draft attempts are
// already excluded from the numerator while their wasted compute is
// captured in the denominator).
//
// This is the *only* benchmark path in this file - an earlier in-process
// version (direct llama_decode() calls, bypassing the HTTP server) existed
// for the non-MTP -ncmoe search, on the theory that avoiding a real
// draft/verify reimplementation for MTP was the main correctness risk.
// That was true but incomplete: the in-process version also fed the model
// a raw, un-chat-templated prompt, and this model family (instruction/
// reasoning-tuned) produces degenerate, highly-repetitive output on
// unformatted prompts *independent of MTP, moe-cache, or placement* -
// confirmed directly by testing (see docs). Degenerate repetitive text is
// trivially predictable, which would have silently inflated every
// measurement taken with it: MTP acceptance rate (repeating the same token
// is an easy guess for the draft), and MoE-cache hit rate (repetitive text
// routes to a narrow, unrealistic set of experts). Spawning the real
// server and using /v1/chat/completions (which applies the GGUF's own
// chat template server-side, exactly like a real client would) avoids
// both the draft/verify-reimplementation risk and the prompt-formatting
// risk in one path, at the cost of a real subprocess per candidate.
// Wraps a string in single quotes for safe use as one shell argument,
// escaping any embedded single quotes via the standard '\'' trick (close
// the quote, emit an escaped literal quote, reopen). Without this, a
// prompt containing an apostrophe (e.g. "Newton's second law" - a real
// prompt in this file's own probe pool, found the hard way when it broke
// the concurrency-benchmark path with a shell syntax error) corrupts the
// surrounding --data-binary '...' argument and the request silently
// becomes a shell parse error instead of an HTTP call.
static std::string common_shell_quote(const std::string & s) {
    std::string out = "'";
    for (char c : s) {
        if (c == '\'') {
            out += "'\\''";
        } else {
            out += c;
        }
    }
    out += "'";
    return out;
}

// Fetches one /v1/chat/completions response and returns (predicted_n, predicted_ms),
// or (-1, 0) on any failure - used by both the solo and concurrent benchmark paths below.
// Degeneracy score for one candidate's generated text, in [0,1] where higher
// is worse. Calibration maximises tok/s, and without this it would happily
// pick a configuration that is fast BECAUSE it is broken: forcing
// substitution on for every router pick measured 11.95 tok/s (vs ~0.7
// otherwise) while emitting word salad ("a question to how a steps
// explaining on the how quicksort, e shouldicym is"), and an earlier
// variant produced verbatim repetition loops. Both were only caught by
// reading the output by hand - exactly the kind of thing a throughput-only
// objective cannot see.
//
// Two cheap checks, no extra model passes and no logprobs needed, each
// aimed at one of the two failure modes actually observed:
//   - repeated 4-grams (a loop repeats long spans verbatim)
//   - single-word dominance (both failure modes degenerate into hammering a
//     few function words - "to how a ... to how a")
// Deliberately not a language-quality judgement, just a degeneracy floor:
// it has to reject output that is obviously broken, not rank prose.
static double common_moe_degeneracy_score(const std::string & text) {
    std::vector<std::string> words;
    {
        std::istringstream iss(text);
        std::string w;
        while (iss >> w) {
            words.push_back(w);
        }
    }
    if (words.size() < 16) {
        return 0.0; // too short to judge - don't reject on noise
    }
    std::unordered_map<std::string, int> word_counts;
    for (const auto & w : words) {
        word_counts[w]++;
    }
    int max_word = 0;
    for (const auto & [w, c] : word_counts) {
        max_word = std::max(max_word, c);
    }
    const double max_word_share = (double) max_word / (double) words.size();

    std::unordered_map<std::string, int> gram_counts;
    size_t n_grams = 0;
    for (size_t i = 0; i + 4 <= words.size(); i++) {
        gram_counts[words[i] + " " + words[i+1] + " " + words[i+2] + " " + words[i+3]]++;
        n_grams++;
    }
    const double distinct_gram_ratio = n_grams > 0 ? (double) gram_counts.size() / (double) n_grams : 1.0;

    // Worst of the two, each normalised so ~1.0 means clearly degenerate.
    const double rep_score  = 1.0 - distinct_gram_ratio;      // 0 = all 4-grams unique
    const double dom_score  = max_word_share;                  // 0 = perfectly varied vocabulary
    return std::max(rep_score, dom_score);
}

// At or above this score a candidate is rejected outright. 0.45 sits well
// clear of healthy prose (normal answers score low on both components - long
// verbatim 4-gram repeats are rare, and no single word takes ~half the
// output) while catching both observed failures, which are not marginal:
// they repeat spans verbatim or hammer one word for a large fraction of the
// text. Override with GGML_MOE_DEGENERACY_REJECT if a model legitimately
// trips it (a format-constrained generator emitting highly repetitive
// structure, say).
static double common_moe_degeneracy_reject_threshold() {
    static const double v = [] {
        const char * e = getenv("GGML_MOE_DEGENERACY_REJECT");
        const double x = e ? atof(e) : 0.45;
        return x > 0.0 ? x : 0.45;
    }();
    return v;
}

// Agreement between a candidate's greedy output and a substitution-free
// reference's, as the fraction of the reference that the candidate reproduces
// before the two part ways.
//
// This exists because the degeneracy guard above cannot see the failure mode
// substitution actually causes. That guard is a lexical health check: it
// catches verbatim 4-gram repeats and single-word hammering. Serving expert
// A's weights where the router asked for B does not produce either - it
// produces fluent, varied, grammatical text that says the wrong thing, which
// scores near zero on both components and passes the gate untouched. A
// throughput ladder gated only on degeneracy will therefore happily pick a
// rung that hallucinates, which is exactly what was observed on gemma-4 after
// calibration settled on rank 2.
//
// Greedy decoding makes divergence sticky: once two runs pick different
// tokens the contexts differ and they rarely reconverge. So the length of the
// shared prefix is the natural measure of "at what point did substitution
// change the model's mind", and it needs no notion of what a good answer is -
// only that this configuration says what the unsubstituted one would have
// said. Compared word-wise rather than byte-wise so that whitespace and
// tokenizer boundaries don't register as disagreement.
static double common_moe_output_fidelity(const std::string & reference, const std::string & candidate) {
    auto split = [](const std::string & s) {
        std::vector<std::string> out;
        std::istringstream iss(s);
        std::string w;
        while (iss >> w) {
            out.push_back(w);
        }
        return out;
    };
    const std::vector<std::string> a = split(reference);
    const std::vector<std::string> b = split(candidate);
    if (a.empty()) {
        return 1.0; // no reference to disagree with - don't reject on noise
    }
    size_t common = 0;
    while (common < a.size() && common < b.size() && a[common] == b[common]) {
        common++;
    }
    return (double) common / (double) a.size();
}

struct common_moe_bench_result {
    double predicted_n  = -1.0;
    double predicted_ms = 0.0;
    double degeneracy   = 0.0;
    // What the model actually emitted, for the fidelity comparison in the
    // substitution ladder. See common_moe_output_fidelity().
    std::string text;
};

static common_moe_bench_result common_moe_bench_one_request_full(int port, const char * prompt, int n_predict, int seed = 1234, bool greedy = false);

static std::pair<double, double> common_moe_bench_one_request(int port, const char * prompt, int n_predict) {
    const auto r = common_moe_bench_one_request_full(port, prompt, n_predict);
    return {r.predicted_n, r.predicted_ms};
}

static common_moe_bench_result common_moe_bench_one_request_full(int port, const char * prompt, int n_predict, int seed, bool greedy) {
    nlohmann::json req = {
        {"messages", nlohmann::json::array({
            {{"role", "user"}, {"content", prompt}}
        })},
        {"max_tokens", n_predict},
        // Seeded, but sampling params deliberately left alone: whatever the
        // model ships as its recommended sampling (unsloth's GGUF metadata for
        // these quants, or the user's own flags) is the condition it will
        // actually be served under, so that is the condition the ladder has to
        // measure. Pinning the seed is enough to make two runs of the SAME
        // config comparable, which is all the fidelity check needs - and it
        // avoids tuning the substitution floor against a greedy regime nobody
        // runs. Residual run-to-run wobble is measured, not assumed: see the
        // reference self-fidelity noise floor in the substitution ladder.
        {"seed", seed},
    };
    if (greedy) {
        // Only for the determinism check: pinning the sampler to argmax takes
        // it out of the picture entirely, so any variation left is the forward
        // pass. Deliberately not used for the throughput or fidelity probes,
        // which have to measure the sampling the model actually ships with.
        req["temperature"] = 0.0;
        req["top_k"] = 1;
    }
    const std::string req_body = req.dump();
    char req_cmd[4096];
    snprintf(req_cmd, sizeof(req_cmd),
        "curl -s http://127.0.0.1:%d/v1/chat/completions -H 'Content-Type: application/json' --data-binary %s",
        port, common_shell_quote(req_body).c_str());
    FILE * rp = popen(req_cmd, "r");
    if (!rp) {
        return {};
    }
    std::string body;
    char buf[4096];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), rp)) > 0) {
        body.append(buf, n);
    }
    pclose(rp);
    try {
        auto j = nlohmann::json::parse(body);
        if (j.contains("timings") && j["timings"].contains("predicted_n")) {
            common_moe_bench_result r;
            r.predicted_n  = j["timings"]["predicted_n"].get<double>();
            r.predicted_ms = j["timings"]["predicted_ms"].get<double>();
            // Score whatever the model actually emitted. Reasoning models put
            // most of a short probe's tokens in reasoning_content rather than
            // content, and a degenerate run is degenerate in either channel -
            // so both are scored and the worse one wins.
            if (j.contains("choices") && !j["choices"].empty() && j["choices"][0].contains("message")) {
                const auto & msg = j["choices"][0]["message"];
                double worst = 0.0;
                for (const char * field : {"content", "reasoning_content"}) {
                    if (msg.contains(field) && msg[field].is_string()) {
                        const std::string s = msg[field].get<std::string>();
                        worst = std::max(worst, common_moe_degeneracy_score(s));
                        if (!s.empty()) {
                            if (!r.text.empty()) {
                                r.text += ' ';
                            }
                            r.text += s;
                        }
                    }
                }
                r.degeneracy = worst;
            }
            return r;
        }
    } catch (const std::exception &) {
        // falls through to the failure return below
    }
    return {};
}

// Wall-clock deadline for the whole calibration run, set by
// common_moe_calibrate() and checked here rather than at each search's own
// call site. Every candidate in every stage funnels through this function,
// which is what makes it the correct choke point: a first attempt put the
// check in one stage's helper lambda and several stages (the -ngl search
// among them) call this directly instead, so a 600s budget sailed 26 minutes
// past its deadline with 89 more projected. Zero means "no deadline set".
static std::atomic<long long> g_moe_calibrate_deadline_ms{0};
static std::atomic<bool>      g_moe_calibrate_budget_warned{false};

static long long common_moe_steady_now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
}

static bool common_moe_calibrate_budget_spent() {
    const long long deadline = g_moe_calibrate_deadline_ms.load(std::memory_order_relaxed);
    if (deadline == 0 || common_moe_steady_now_ms() < deadline) {
        return false;
    }
    if (!g_moe_calibrate_budget_warned.exchange(true)) {
        LOG_WRN("%s: calibration time budget spent - skipping remaining candidates and keeping the best "
                "configuration measured so far (raise GGML_MOE_CALIBRATE_BUDGET_S for a longer search)\n",
                __func__);
        common_moe_calibration_status_set("time budget spent - finalizing best measured configuration");
    }
    return true;
}

static double common_moe_bench_candidate_server(
        const std::string & self_exe, const std::string & path_model, const std::string & mtp_path,
        uint32_t n_cpu_moe, int n_max, int n_threads, int port, uint32_t n_ctx, int n_predict,
        int n_concurrency = 1, int moe_cache_mb = -1, int fit_target_mb = -1, int n_gpu_layers = 99,
        int substitute_min_rank = -1, std::string * out_sample = nullptr, int probe_seed = 1234,
        double substitute_quality_sigma = std::numeric_limits<double>::quiet_NaN()) {
    if (common_moe_calibrate_budget_spent()) {
        return -1.0; // reported as a failed candidate; every search here keeps its best measured point
    }
    std::string mtp_args;
    if (!mtp_path.empty()) {
        char buf[2048];
        snprintf(buf, sizeof(buf), "--model-draft '%s' --spec-type draft-mtp --spec-draft-n-max %d ",
                mtp_path.c_str(), n_max);
        mtp_args = buf;
    }
    std::string threads_args;
    if (n_threads > 0) {
        char buf[128];
        snprintf(buf, sizeof(buf), "-t %d -tb %d ", n_threads, n_threads);
        threads_args = buf;
    }
    // At concurrency > 1, --parallel must match so the candidate server can
    // actually hold n_concurrency slots, and -c needs enough headroom for
    // all of them at once (n_predict + a real prompt, per slot) - reusing
    // the caller's n_ctx here would starve most slots down to a handful of
    // tokens each and force truncation/failure well before this is a real
    // concurrent-throughput measurement.
    std::string parallel_args;
    uint32_t ctx_for_launch = n_ctx;
    if (n_concurrency > 1) {
        char buf[64];
        snprintf(buf, sizeof(buf), "--parallel %d ", n_concurrency);
        parallel_args = buf;
        ctx_for_launch = std::max<uint32_t>(n_ctx, (uint32_t) n_concurrency * 384);
    }
    char cmd[4096];
    // --temp/--top-p/--top-k match this model family's documented MTP
    // sampling recommendation (Unsloth's llama.cpp MTP guide) - found to
    // matter for output correctness, not just MTP: benchmarking with
    // mismatched sampling produced degenerate, highly-repetitive
    // generations that (being trivially predictable) artificially inflate
    // both MoE-cache hit rate and MTP acceptance rate, biasing every
    // throughput number measured this way. Real, representative numbers
    // require realistic, non-degenerate generation.
    // --no-token-freq-log: this candidate's probe traffic is a handful of
    // fixed prompts repeated across many candidates, not representative of
    // real deployment usage - would badly skew a histogram meant to guide
    // FR-Spec vocab trimming toward this benchmark's own prompt pool
    // instead of real traffic.
    char cache_arg[32];
    snprintf(cache_arg, sizeof(cache_arg), moe_cache_mb > 0 ? "%d" : "auto", moe_cache_mb);
    // -fitt controls the VRAM margin common_maybe_raise_moe_for_ctx reserves
    // before deciding placement, and that margin is what silently raises
    // -ncmoe until the requested context fits. Passing it through means a
    // candidate is benchmarked at the placement it actually asked for
    // instead of whatever the default 3 x 1024 MiB margin forced it to -
    // without this, sweeping -ncmoe below the margin's floor measures the
    // same effective configuration several times over (confirmed: -ncmoe
    // 22/16/10 all silently became 27 and returned near-identical tok/s).
    std::string fit_args;
    if (fit_target_mb > 0) {
        char buf[64];
        snprintf(buf, sizeof(buf), "-fitt %d ", fit_target_mb);
        fit_args = buf;
    }
    // Substitution aggressiveness is an env var, not a flag, so it is passed
    // as a shell assignment on the candidate's own command line - the
    // subprocess must see the value being benchmarked, not whatever this
    // calibrating process happens to have inherited.
    std::string subst_env;
    if (!std::isnan(substitute_quality_sigma)) {
        char qbuf[96];
        snprintf(qbuf, sizeof(qbuf), "GGML_CUDA_MOE_CACHE_SUBSTITUTE_QUALITY_SIGMA=%g ",
                substitute_quality_sigma);
        subst_env += qbuf;
    }
    if (substitute_min_rank >= 0) {
        char buf[96];
        snprintf(buf, sizeof(buf), "GGML_CUDA_MOE_CACHE_SUBSTITUTE_MIN_RANK=%d ", substitute_min_rank);
        subst_env = buf;
    }
    snprintf(cmd, sizeof(cmd),
        "%s'%s' -m '%s' -ngl %d -ncmoe %u --moe-cache %s -c %u %s%s%s%s"
        "--temp 1.0 --top-p 0.95 --top-k 64 --no-token-freq-log "
        "--port %d --no-webui > /dev/null 2>&1 & echo $!",
        subst_env.c_str(), self_exe.c_str(), path_model.c_str(), n_gpu_layers, n_cpu_moe, cache_arg, ctx_for_launch,
        mtp_args.c_str(), threads_args.c_str(), parallel_args.c_str(), fit_args.c_str(), port);
    FILE * pf = popen(cmd, "r");
    if (!pf) {
        return -1.0;
    }
    char pidbuf[32] = {0};
    const bool got_pid = fgets(pidbuf, sizeof(pidbuf), pf) != nullptr;
    pclose(pf);
    if (!got_pid) {
        return -1.0;
    }
    const pid_t pid = (pid_t) atol(pidbuf);
    if (pid <= 0) {
        return -1.0;
    }

    auto cleanup = [&]() {
        kill(pid, SIGKILL);
        int status = 0;
        waitpid(pid, &status, 0);
    };

    char health_cmd[256];
    snprintf(health_cmd, sizeof(health_cmd),
        "curl -s -o /dev/null -w '%%{http_code}' http://127.0.0.1:%d/health 2>/dev/null", port);
    bool ready = false;
    for (int i = 0; i < 30; i++) {
        FILE * hp = popen(health_cmd, "r");
        if (hp) {
            char code[8] = {0};
            const bool got = fgets(code, sizeof(code), hp) != nullptr;
            pclose(hp);
            if (got && strncmp(code, "200", 3) == 0) {
                ready = true;
                break;
            }
        }
        struct timespec ts{2, 0};
        nanosleep(&ts, nullptr);
    }
    if (!ready) {
        cleanup();
        return -1.0;
    }

    // /v1/chat/completions, not /completion: the raw completion endpoint
    // bypasses the model's chat template entirely, and this model family
    // (instruction/reasoning-tuned) produces degenerate output on
    // unformatted raw prompts independent of MTP/moe-cache/placement -
    // confirmed directly by testing (see docs). The chat endpoint applies
    // the GGUF's own template server-side, matching how any real client
    // would actually talk to this server.
    static const char * const probe_prompts[] = {
        "Explain how photosynthesis works in three sentences, then describe the role of chlorophyll.",
        "Write a Python function that implements binary search, then explain its time complexity.",
        "Summarize the plot of a mystery novel in two sentences, then suggest a twist ending.",
        "Describe the water cycle briefly, then explain how it relates to weather patterns.",
        "Explain Newton's second law of motion, then give a real-world example.",
        "Describe how a car engine works in simple terms, then list its main components.",
        "What are the main causes of inflation, and how do central banks respond?",
        "Explain how vaccines work, then describe herd immunity.",
    };
    static constexpr int n_probe_prompts = sizeof(probe_prompts) / sizeof(probe_prompts[0]);

    // Short, low-context prompts, scored for quality but never for speed.
    //
    // Every prompt above is a substantial instruction, and a model with a
    // paragraph of task to hold onto is the EASY case for degeneracy: there is
    // plenty of context anchoring the next token. The failure that actually
    // reached a user on gemma-4 was a bare "hi" answered with the same
    // sentence four times over plus leaked "Input:/Output:" scaffolding -
    // classic repetition the guard above catches easily, on a kind of prompt
    // the ladder had never once put in front of it. Cheap to add: the prompts
    // are tiny, and a healthy reply to them is short, so the only runs that
    // cost real time here are the broken ones worth catching.
    static const char * const quality_prompts[] = {
        "hi",
        "thanks!",
        "What's 2 + 2?",
    };
    static constexpr int n_quality_prompts = sizeof(quality_prompts) / sizeof(quality_prompts[0]);

    double result_tps = -1.0;
    if (n_concurrency <= 1) {
        // Solo path: average per-request predicted_per_second across
        // n_solo_probes sequential probes. Deliberately reduced from 2 to 1
        // (see n_samples_per_candidate's comment below - the same
        // speed-vs-noise trade, made together) on a slow-decoding model
        // where every probe is expensive; this drops the averaging that
        // partially absorbed run-to-run variance within a single candidate.
        constexpr int n_solo_probes = 1;
        double sum_tps = 0.0;
        int n_ok = 0;
        double worst_degeneracy = 0.0;
        for (int i = 0; i < n_solo_probes; i++) {
            const auto r = common_moe_bench_one_request_full(port, probe_prompts[i], n_predict, probe_seed);
            if (r.predicted_n > 0 && r.predicted_ms > 0) {
                sum_tps += r.predicted_n / (r.predicted_ms / 1000.0);
                worst_degeneracy = std::max(worst_degeneracy, r.degeneracy);
                n_ok++;
            }
        }
        result_tps = n_ok > 0 ? sum_tps / n_ok : -1.0;
        // Quality-only pass over the short prompts. Kept out of the timing
        // average deliberately: these generate few tokens when healthy, so
        // folding them into tok/s would measure prompt length, not decode
        // rate. The fidelity sample is taken from here rather than from the
        // timing probe because this is the harder case - with almost no
        // context to anchor on, a configuration that has damaged the model's
        // routing shows it here first.
        if (result_tps > 0) {
            for (int i = 0; i < n_quality_prompts; i++) {
                const auto q = common_moe_bench_one_request_full(port, quality_prompts[i], n_predict, probe_seed);
                if (q.predicted_n <= 0) {
                    continue;
                }
                worst_degeneracy = std::max(worst_degeneracy, q.degeneracy);
                if (out_sample && i == 0) {
                    *out_sample = q.text;
                }
            }
        }
        // Reject rather than rank: a candidate that generates degenerate text
        // is not a slower-but-valid point on the throughput curve, it is not a
        // usable configuration at all, so it must not be able to win on speed.
        // Reported as a failed candidate (-1), the same as one whose server
        // never came up.
        if (result_tps > 0 && worst_degeneracy >= common_moe_degeneracy_reject_threshold()) {
            LOG_WRN("%s: candidate rejected - output is degenerate (score %.2f >= %.2f) at %.2f tok/s; "
                    "throughput bought with broken generation is not a valid result\n",
                    __func__, worst_degeneracy, common_moe_degeneracy_reject_threshold(), result_tps);
            common_moe_calibration_status_note("output check", "degeneracy",
                    string_format("rejected - score %.2f (was %.2f tok/s)", worst_degeneracy, result_tps), false);
            result_tps = -1.0;
        }
        // Determinism gate: same request, same seed, repeated. A configuration
        // that answers one question two different ways is not a slower-but-
        // valid point on the throughput curve - the forward pass is returning
        // different numbers each time, and no amount of tok/s redeems that.
        //
        // This exists because calibration certified a gemma-4 configuration at
        // 48 tok/s and shipped it as the default, when repeating one identical
        // greedy request against it gave 9 distinct answers out of 12. Nothing
        // already here could see that: the degeneracy guard scores each reply
        // on its own and every one of those nine was lexically healthy, and the
        // fidelity check compares a candidate against a reference but never
        // compares a candidate against itself. Cheap to close - the replies are
        // short and a broken config usually diverges within a few tokens.
        // Majority agreement, not unanimity. Requiring all N identical was
        // tried first and is wrong: measured on this hardware, even a pure-CPU
        // run (-ngl 0) disagrees with itself occasionally at this length, so a
        // unanimity gate rejects every configuration including the good ones
        // and calibration finds nothing. Majority is the natural
        // parameter-free line - it asks whether there IS a single answer this
        // configuration mostly gives, which cleanly separates the failure that
        // prompted this (9 distinct answers in 12 requests) from healthy noise
        // (11 of 12 agreeing), without a tolerance anybody had to pick.
        if (result_tps > 0 && !common_moe_calibrate_budget_spent()) {
            constexpr int n_det_probes = 5;
            std::vector<std::string> answers;
            for (int i = 0; i < n_det_probes; i++) {
                const auto r = common_moe_bench_one_request_full(port, quality_prompts[0], 12, probe_seed, /* greedy */ true);
                if (r.predicted_n > 0) {
                    answers.push_back(r.text);
                }
            }
            int modal = 0;
            for (const auto & a : answers) {
                modal = std::max(modal, (int) std::count(answers.begin(), answers.end(), a));
            }
            if (!answers.empty() && modal * 2 <= (int) answers.size()) {
                LOG_WRN("%s: candidate rejected - not reproducible: the same request at the same seed, "
                        "sampled greedily, agreed only %d times in %zu at %.2f tok/s. A forward pass that "
                        "varies run to run is reading memory it does not own, and no throughput redeems it\n",
                        __func__, modal, answers.size(), result_tps);
                common_moe_calibration_status_note("output check", "reproducibility",
                        string_format("rejected - %d of %zu agreed (was %.2f tok/s)",
                                modal, answers.size(), result_tps), false);
                result_tps = -1.0;
            }
        }
    } else {
        // Concurrent path: fire n_concurrency real requests at once (cycling
        // through the prompt pool so it's not the same prompt N times, which
        // would bias moe-cache/MTP toward an unrealistically easy repeated
        // pattern - see the endpoint-bias lesson elsewhere in this file),
        // measure wall-clock for all of them to complete, and report
        // aggregate tok/s = total generated tokens / wall time. This is the
        // actual quantity a concurrent deployment cares about, not the
        // average of what each individual request would have gotten alone.
        std::vector<double> predicted_n_per_req(n_concurrency, -1.0);
        std::vector<std::thread> threads;
        threads.reserve(n_concurrency);
        const auto t_start = std::chrono::steady_clock::now();
        for (int i = 0; i < n_concurrency; i++) {
            const char * prompt = probe_prompts[i % n_probe_prompts];
            threads.emplace_back([&predicted_n_per_req, i, port, prompt, n_predict]() {
                const auto [predicted_n, predicted_ms] = common_moe_bench_one_request(port, prompt, n_predict);
                predicted_n_per_req[i] = predicted_n;
                (void) predicted_ms; // wall-clock comes from the outer timer, not per-request timing
            });
        }
        for (auto & t : threads) {
            t.join();
        }
        const auto t_end = std::chrono::steady_clock::now();
        const double wall_s = std::chrono::duration<double>(t_end - t_start).count();

        double total_tokens = 0.0;
        int n_ok = 0;
        for (double predicted_n : predicted_n_per_req) {
            if (predicted_n > 0) {
                total_tokens += predicted_n;
                n_ok++;
            }
        }
        // Require every concurrent request to have succeeded - a partial
        // failure under real concurrency is itself a signal this candidate
        // can't actually sustain the target concurrency, not just noise to
        // average past.
        result_tps = (n_ok == n_concurrency && wall_s > 0) ? total_tokens / wall_s : -1.0;
    }

    cleanup();
    return result_tps;
}

static std::string common_self_exe_path() {
    char buf[4096];
    const ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n <= 0) {
        return "";
    }
    buf[n] = '\0';
    return std::string(buf);
}

// --moe-calibrate: empirically finds the throughput-optimal -ncmoe and
// thread count for this exact GPU+model+context combination via a handful
// of real short benchmarks, caches the result, then the caller exits
// (matches the existing --fit-moe-cache preview-before-you-commit pattern
// rather than mixing calibration into the same run as real serving).
// -ncmoe is searched via golden-section search (interior-peak-aware, not a
// fixed grid). When a draft/MTP model is configured, spec-draft-n-max is
// also searched the same way, within an envelope found first by doubling
// until cache health collapses (see the n_max=8 finding in the docs - a
// real, measured collapse, not a guess at where the envelope ends).
//
// Concurrency-aware: if params.n_parallel > 1, every candidate is
// benchmarked with that many real concurrent requests (aggregate
// throughput), not solo decode speed - the optimal placement/threads/n_max
// can differ meaningfully at real concurrency (confirmed this session: the
// MMVQ and bulk-offload GPU-kernel-dispatch cliffs both only appear above
// certain concurrent batch sizes, invisible to a solo benchmark entirely).
// cparams.n_seq_max already equals params.n_parallel via
// common_context_params_to_llama, so the safe-floor probe below already
// accounts for the larger KV-cache footprint N concurrent slots need - no
// separate concurrency-specific probe required. Deliberately reuses
// --parallel itself as the concurrency target (not a separate flag): the
// calibration cache key already includes n_parallel, so calibrating and
// deploying with the same --parallel value is what makes the cache
// lookup find this entry later - a separate flag that could drift out of
// sync with --parallel would be a real footgun here.
static std::mutex                    g_moe_calib_status_mutex;
static std::string                   g_moe_calib_status_stage = "starting";
static std::chrono::steady_clock::time_point g_moe_calib_status_start = std::chrono::steady_clock::now();
static int                           g_moe_calib_status_done  = 0;
static int                           g_moe_calib_status_total = 0; // 0 = no estimate yet
static long long                     g_moe_calib_status_eta_s = -1; // frozen at each candidate_done(), not live

void common_moe_calibration_status_start() {
    std::lock_guard<std::mutex> lock(g_moe_calib_status_mutex);
    g_moe_calib_status_stage = "starting";
    g_moe_calib_status_start = std::chrono::steady_clock::now();
    g_moe_calib_status_done  = 0;
    g_moe_calib_status_total = 0;
    g_moe_calib_status_eta_s = -1;
}

void common_moe_calibration_status_set(const std::string & stage) {
    std::lock_guard<std::mutex> lock(g_moe_calib_status_mutex);
    g_moe_calib_status_stage = stage;
}

void common_moe_calibration_status_set_total(int total_candidates) {
    std::lock_guard<std::mutex> lock(g_moe_calib_status_mutex);
    g_moe_calib_status_total = total_candidates;
}

static std::vector<common_moe_calibration_decision> g_moe_calib_decisions;

void common_moe_calibration_status_note(
        const std::string & lever, const std::string & value,
        const std::string & result, bool accepted, bool chosen) {
    std::lock_guard<std::mutex> lock(g_moe_calib_status_mutex);
    // Bounded: a long run on a slow model can produce a lot of these, and the
    // page only ever renders them. Oldest first out, newest kept.
    if (g_moe_calib_decisions.size() >= 64) {
        g_moe_calib_decisions.erase(g_moe_calib_decisions.begin());
    }
    if (chosen) {
        // Only one winner per lever: a later commit for the same lever
        // supersedes an earlier one (the confirmation step can step the
        // substitution floor back after the ladder already picked it).
        for (auto & d : g_moe_calib_decisions) {
            if (d.lever == lever) {
                d.chosen = false;
            }
        }
    }
    g_moe_calib_decisions.push_back({lever, value, result, accepted, chosen});
}

void common_moe_calibration_status_candidate_done() {
    std::lock_guard<std::mutex> lock(g_moe_calib_status_mutex);
    g_moe_calib_status_done++;
    // Recompute the ETA only here, at a real completion instant - not live
    // on every status poll, which would inflate the estimate every second
    // spent waiting on the current (still in-flight) candidate purely
    // because the numerator (elapsed) keeps growing while the denominator
    // (done) sits still until it actually finishes. Confirmed happening in
    // practice: watched a live run's ETA climb from 248m to 371m across a
    // single still-in-progress candidate before this fix.
    if (g_moe_calib_status_total > 0 && g_moe_calib_status_done < g_moe_calib_status_total) {
        const long long elapsed_s = (long long) std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - g_moe_calib_status_start).count();
        g_moe_calib_status_eta_s = (long long) (((double) elapsed_s / g_moe_calib_status_done) *
                (g_moe_calib_status_total - g_moe_calib_status_done));
        // Never promise longer than the time budget allows. The extrapolation
        // above assumes every remaining candidate gets measured, but the
        // budget cuts the run the moment it expires - so on a slow model the
        // naive number is not just pessimistic, it is describing a search
        // that will not be permitted to happen: a real run showed "~70m57s
        // left" on a 600s budget that would stop it inside 10 minutes.
        const long long deadline_ms = g_moe_calibrate_deadline_ms.load(std::memory_order_relaxed);
        if (deadline_ms != 0) {
            const long long remaining_budget_s = (deadline_ms - common_moe_steady_now_ms()) / 1000;
            g_moe_calib_status_eta_s = std::min(g_moe_calib_status_eta_s, std::max(0LL, remaining_budget_s));
        }
    } else {
        g_moe_calib_status_eta_s = -1;
    }
}

common_moe_calibration_status common_moe_calibration_status_get_struct() {
    common_moe_calibration_status s;
    std::chrono::steady_clock::time_point start;
    {
        std::lock_guard<std::mutex> lock(g_moe_calib_status_mutex);
        s.stage   = g_moe_calib_status_stage;
        start     = g_moe_calib_status_start;
        s.done    = g_moe_calib_status_done;
        s.total   = g_moe_calib_status_total;
        s.eta_s   = g_moe_calib_status_eta_s; // frozen at the last candidate_done() - see its comment
        s.decisions = g_moe_calib_decisions;
    }
    s.elapsed_s = (long long) std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - start).count();
    return s;
}

std::string common_moe_calibration_status_get() {
    const common_moe_calibration_status s = common_moe_calibration_status_get_struct();

    std::string progress_note;
    if (s.total > 0) {
        // done can exceed total - the estimate is a single upfront guess
        // (see common_moe_calibrate), not a hard ceiling; a stage that
        // genuinely needed more candidates than guessed just shows 100%+
        // rather than a nonsensical negative "time left".
        const int pct = (int) std::lround(100.0 * std::min(s.done, s.total) / s.total);
        std::string eta;
        if (s.eta_s >= 0) {
            eta = string_format(", ~%lldm%02llds left", s.eta_s / 60, s.eta_s % 60);
        }
        progress_note = string_format(" [candidate %d/~%d, %d%%%s]", s.done, s.total, pct, eta.c_str());
    }

    char buf[420];
    snprintf(buf, sizeof(buf),
            "Calibrating MoE placement for maximum throughput: %s%s (elapsed %lldm%02llds). "
            "This runs once per hardware+model+context combination and is cached - a normal "
            "launch afterward (without --moe-calibrate) starts instantly using the cached result.",
            s.stage.c_str(), progress_note.c_str(), s.elapsed_s / 60, s.elapsed_s % 60);
    return buf;
}

// Rough evaluation-count estimate for common_golden_section_search_max over
// [lo, hi] - used only to size the status page's progress bar/ETA, not the
// search itself (which is exact and memoized regardless of this guess being
// off). Mirrors the real loop's shrink rate without running it.
static int common_golden_section_eval_estimate(int lo, int hi) {
    if (lo >= hi) {
        return 1;
    }
    const double gr = 1.618033988749895;
    int evals = 2; // initial x1, x2
    double width = hi - lo;
    while (width > 2.0) {
        width /= gr;
        evals++;
    }
    return evals + 3; // close-out linear scan of the final <=3-point range
}

static void common_enforce_moe_cache_parallel_limit(common_params & params, llama_context_params & cparams);

void common_moe_calibrate(common_params & params) {
    common_moe_calibration_status_start();
    const char * path_model = params.model.path.c_str();
    auto mparams = common_model_params_to_llama(params);
    auto cparams = common_context_params_to_llama(params);
    // Clamp concurrency BEFORE measuring anything. Serving applies this limit
    // on its own (the moe-cache corrupts output above it), so calibrating
    // above it measures a configuration that can never run - and worse, every
    // candidate carries the memory cost of slots that will be thrown away.
    // Observed on Qwen3.8-Flash-Next: calibration benchmarked at --parallel 4,
    // reported 0.17 aggregate tok/s, and then every -ngl candidate after the
    // first failed outright, while serving clamped the very same run to 2. The
    // cached entry described a concurrency the server refuses to use.
    common_enforce_moe_cache_parallel_limit(params, cparams);
    const int concurrency = std::max(1, (int) params.n_parallel);

    LOG_INF("%s: probing safe MoE CPU-offload floor for this GPU+model+context combination ...\n", __func__);
    common_moe_calibration_status_set("probing safe MoE CPU-offload floor");
    // Probe with the SAME margin common_maybe_raise_moe_for_ctx will demand at
    // serving time (3x the per-device fit target - see its own comment for why
    // a bare fit is not enough there). Probing with the bare-fit default here
    // instead put the search floor below anything serving would accept, so
    // calibration spent its entire budget measuring placements that were then
    // discarded: a real run measured ncmoe 46/45/44, picked 44 as the fastest,
    // and the fit guard immediately overrode it with "calibrated placement of
    // 44 CPU layer(s) is below the safe minimum for this context - keeping the
    // fit search's more conservative answer", serving 47. Every candidate in
    // that budget was unusable by construction.
    const int64_t fit_margin = 3 * (int64_t) params.fit_params_target[0];
    common_moe_fit_probe_result probe = common_moe_find_safe_layers(path_model, mparams, cparams, fit_margin);
    if (!probe.is_moe) {
        LOG_WRN("%s: model has no MoE experts - nothing for --moe-calibrate to do\n", __func__);
        return;
    }

    uint32_t safe_n = probe.already_fits ? 0 : probe.safe_n;
    if (!probe.already_fits && !probe.found_safe_n) {
        // Halve the context until it fits, rather than giving up. Bailing here
        // meant a bare launch on a model whose default context is larger than
        // this card can hold skipped calibration entirely and served
        // unconfigured - observed on Qwen3.8-Flash-Next, where the whole
        // "point it at a model" path silently produced nothing because the
        // model's own default context does not fit a 12GB card even with
        // every expert on the CPU. The serving path already resolves this the
        // same way (it trades context for placement and says so); there is no
        // reason calibration should be the one component that refuses.
        //
        // Halving rather than a fine search on purpose: each step costs a real
        // probe, this only has to find a context that FITS so the actual
        // search can start, and the fit machinery downstream still gets the
        // final say on placement at whatever context this lands on.
        // Resolve both the way the rest of this file does. A bare launch
        // leaves n_ctx at 0 meaning "the model's own trained context", and
        // reading it raw made `requested / 2` zero, so the loops below never
        // executed even once - which is why this still reported "does not fit
        // even at 512" while an explicit -c 2048 --parallel 1 fitted
        // immediately. Same for n_seq_max.
        const uint32_t requested     = cparams.n_ctx     > 0 ? cparams.n_ctx     : 4096;
        const uint32_t requested_par = cparams.n_seq_max > 0 ? cparams.n_seq_max : 1;
        bool found = false;
        // Concurrency first, and context only after. On a hybrid model the
        // recurrent-state cache is sized by n_seq_max, not by n_ctx, so on
        // Qwen3.8-Flash-Next the allocation that actually failed was the rs
        // cache and halving the context could not have helped however far it
        // went - the first version of this fallback shrank the context to 512
        // and still gave up, which is what showed the variable was wrong.
        for (uint32_t try_par = requested_par / 2; try_par >= 1 && !found; try_par /= 2) {
            cparams.n_seq_max = try_par;
            probe = common_moe_find_safe_layers(path_model, mparams, cparams, fit_margin);
            if (probe.already_fits || probe.found_safe_n) {
                LOG_WRN("%s: %u concurrent slots do not fit this device even with every expert on the CPU "
                        "- calibrating at %u instead, which does. Pass --parallel explicitly to pin a "
                        "different number\n", __func__, requested_par, try_par);
                params.n_parallel = (int32_t) try_par;
                safe_n = probe.already_fits ? 0 : probe.safe_n;
                found = true;
            }
            if (try_par == 1) {
                break;
            }
        }
        for (uint32_t try_ctx = requested / 2; try_ctx >= 512 && !found; try_ctx /= 2) {
            cparams.n_ctx = try_ctx;
            probe = common_moe_find_safe_layers(path_model, mparams, cparams, fit_margin);
            if (probe.already_fits || probe.found_safe_n) {
                LOG_WRN("%s: the requested %u-token context does not fit this device even with every expert "
                        "on the CPU - calibrating at %u instead, which does. Pass -c explicitly to pin a "
                        "different one\n", __func__, requested, try_ctx);
                params.n_ctx = try_ctx;
                safe_n = probe.already_fits ? 0 : probe.safe_n;
                found = true;
                break;
            }
        }
        if (!found) {
            LOG_ERR("%s: config does not fit in available device memory even with all MoE experts on CPU, "
                    "one slot and a 512-token context; add VRAM before calibrating\n", __func__);
            return;
        }
    }

    const int n_threads_default = params.cpuparams.n_threads > 0 ? params.cpuparams.n_threads : common_cpu_get_num_math();
    // 32, not 64: per-candidate cost is dominated by this probe on a
    // CPU-offloaded model (64 tokens at ~0.6 tok/s is ~107s, against ~25s to
    // spawn and load), and halving it roughly doubles how many levers fit in
    // the time budget. Still well above the degeneracy guard's 16-word floor,
    // so candidates that generate broken text are still caught.
    const int n_predict = 32;

    // Hard wall-clock budget for the whole run. Per-candidate cost varies by
    // orders of magnitude across models (a candidate is a full server spawn
    // plus a real generation probe: seconds on a small model that fits in
    // VRAM, ~90s+ on one decoding at ~1 tok/s), so a fixed candidate count
    // cannot bound runtime - the same search that finishes in a minute on
    // gemma-4 projected to 8-9 HOURS on qwen4exp, which is not a calibration
    // anyone waits through before their first chat.
    //
    // A deadline degrades gracefully where a smaller fixed count would not:
    // fast models still run the full search, slow ones get a truncated but
    // valid one, since stages run in priority order (placement first, then
    // the refinements) and each candidate is skipped once the budget is
    // spent. Skipped candidates report as failures, which every search here
    // already handles by keeping the best point actually measured.
    const double calibration_budget_s = [] {
        const char * e = getenv("GGML_MOE_CALIBRATE_BUDGET_S");
        const double v = e ? atof(e) : 600.0; // 10 minutes
        return v > 0.0 ? v : 600.0;
    }();
    g_moe_calibrate_budget_warned.store(false);
    g_moe_calibrate_deadline_ms.store(
            common_moe_steady_now_ms() + (long long) (calibration_budget_s * 1000.0));
    LOG_INF("%s: time budget for this run: %.0fs (GGML_MOE_CALIBRATE_BUDGET_S)\n", __func__, calibration_budget_s);

    const std::string self_exe = common_self_exe_path();
    if (self_exe.empty()) {
        LOG_ERR("%s: could not resolve /proc/self/exe - --moe-calibrate needs this to spawn benchmark "
                "subprocesses (real llama-server instances, so benchmarks go through the same chat-template "
                "and sampling path a real client would use, not a raw/unformatted prompt)\n", __func__);
        return;
    }
    // A fresh port per subprocess, not one reused across the whole run:
    // SIGKILLing a candidate's server doesn't guarantee the OS releases its
    // listening socket before the next candidate tries to bind the same
    // port (TIME_WAIT) - reusing one port caused a real intermittent
    // "failed" candidate when this was tested.
    int port_counter = 18900 + (int) (getpid() % 500);
    auto next_port = [&]() { return port_counter++; };
    const uint32_t ctx = cparams.n_ctx > 0 ? cparams.n_ctx : 4096;

    // Two independent samples per candidate, averaged - a single subprocess
    // run is one noisy sample, and this isn't a theoretical concern: a
    // documented n_max=3 run-to-run swing of 65% (42.06 vs 69.21 tok/s)
    // elsewhere in this history, and a real thread-tuning decision that
    // shipped on a single n_threads=12 sample of 48.71 tok/s when the true
    // 3-repeat mean was 61.82 - an outlier large enough to flip a closer
    // decision, confirmed by re-measuring after the fact. Each sample gets
    // its own one-retry-on-failure (a launch failure is a different problem
    // than ordinary timing noise); a sample that still fails after its
    // retry is dropped from the average rather than failing the whole
    // candidate, so one bad launch doesn't cost two good measurements.
    //
    // Reduced from 2 to 1 on a deliberate speed-for-noise trade (chosen by
    // the user, not a default): with this model decoding at ~1 tok/s, each
    // extra sample costs several real minutes, and calibration's own search
    // ranges grew (the -ngl search above did not exist when the 2-sample
    // default and the swings cited above were measured). Single-sample
    // candidates reopen exactly the outlier risk those swings describe -
    // still guarded by the retry-on-failure below (a launch failure, not
    // ordinary timing noise) and by golden-section's own "always trust the
    // best of what was truly measured" check, but with no averaging to
    // absorb a plain bad-luck sample within one candidate.
    constexpr int n_samples_per_candidate = 1;
    // The substitution floor decided so far, carried into every later stage's
    // candidates. Stages are supposed to build on each other's decisions, not
    // each measure a different machine: without this the ladder settles on a
    // rank, and then the -ngl, thread, cache-size and fit-margin stages all
    // benchmark with substitution back at the runtime default. Seen plainly on
    // gemma-4 - the ladder peaked at 42.70 tok/s and every stage after it
    // measured ~36, not because anything degraded but because they had quietly
    // dropped the setting the ladder had just established, and then tuned
    // cache size against a configuration that is not the one being served.
    int active_min_rank = -1;
    // Carried into the later stages the same way active_min_rank is, so
    // everything measured after this point is measured at the balance that
    // will actually be served.
    double active_quality_sigma = std::numeric_limits<double>::quiet_NaN();
    // Same reasoning for the GPU-resident layer count: it is decided by its
    // own stage and then every later stage passed a hardcoded 99 (full
    // residency), so thread count, cache size and fit margin were all tuned
    // against a layer placement the server will not be using. 99 remains the
    // pre-decision default, matching the parameter's own default.
    int active_ngl = 99;
    auto bench_one_sample = [&](uint32_t n_cpu_moe, int n_max, const std::string & mtp_path, int n_threads) -> double {
        double tps = common_moe_bench_candidate_server(
                self_exe, path_model, mtp_path, n_cpu_moe, n_max, n_threads, next_port(), ctx, n_predict, concurrency,
                -1, -1, active_ngl, active_min_rank);
        if (tps < 0) {
            LOG_WRN("%s:   candidate sample failed, retrying ...\n", __func__);
            tps = common_moe_bench_candidate_server(
                    self_exe, path_model, mtp_path, n_cpu_moe, n_max, n_threads, next_port(), ctx, n_predict, concurrency,
                    -1, -1, active_ngl, active_min_rank);
        }
        return tps;
    };
    auto bench_with_retry = [&](uint32_t n_cpu_moe, int n_max, const std::string & mtp_path, int n_threads) -> double {
        // The time budget is enforced inside common_moe_bench_candidate_server
        // itself - the one function every stage's candidates funnel through,
        // including the several that bypass this lambda entirely.
        double sum = 0.0;
        int n_ok = 0;
        for (int i = 0; i < n_samples_per_candidate; i++) {
            const double tps = bench_one_sample(n_cpu_moe, n_max, mtp_path, n_threads);
            if (tps > 0) {
                sum += tps;
                n_ok++;
            }
        }
        return n_ok > 0 ? sum / n_ok : -1.0;
    };

    // Search range: from the safe floor up to a bound past which more
    // conservative placement essentially never helps (our own sweep never
    // found the peak more than ~15% of n_layer above the floor) - wide
    // enough to contain the peak, narrow enough that golden-section search
    // stays cheap.
    uint32_t ncmoe_hi = std::min<uint32_t>(probe.n_layer, safe_n + std::max<uint32_t>(4, probe.n_layer / 8));

    const uint32_t ncmoe_span = ncmoe_hi > safe_n ? ncmoe_hi - safe_n : 0;
    static const uint32_t ncmoe_search_min_span = [] {
        const char * e = getenv("GGML_MOE_CALIBRATE_NCMOE_MIN_SPAN");
        const long v = e ? atol(e) : 3;
        return (uint32_t) (v >= 0 ? v : 3);
    }();
    // Single upfront, deliberately generous estimate of the total candidate
    // count for the whole calibration run (ncmoe search, the -ngl search and
    // its possible ncmoe re-search, spec-draft-n-max if MTP is configured,
    // plus the fixed-size thread/cache/fit-margin stages later) - purely for
    // the status page's progress bar and ETA. Not recomputed per branch
    // (whether -ngl actually wins and triggers a re-search, whether a
    // boundary extends, whether a fit-margin candidate fails early): erring
    // generous means "time left" only ever counts down, never climbs back up
    // as stages run, at the cost of finishing a bit before it reaches 100%.
    {
        // Placement contributes one baseline measurement when the fit probe
        // fixed it (the common case on a tightly-constrained model), not a
        // whole search - counting the search anyway made the bar claim ~34
        // candidates for a run that only ever intended to measure a handful.
        const bool ncmoe_will_search = ncmoe_span >= ncmoe_search_min_span;
        int est = ncmoe_will_search
                ? common_golden_section_eval_estimate((int) safe_n, (int) ncmoe_hi)
                : 1;
        est += 7; // substitution-floor ladder (6 rungs) + long-probe confirmation
        const uint32_t ngl_hi_est = (uint32_t) probe.n_layer + 1;
        const uint32_t ngl_lo_est = probe.n_layer > 3 ? probe.n_layer - probe.n_layer / 3 : 0;
        est += common_golden_section_eval_estimate((int) ngl_lo_est, (int) ngl_hi_est); // -ngl search
        if (ncmoe_will_search) {
            est += common_golden_section_eval_estimate((int) safe_n, (int) ncmoe_hi);   // possible ncmoe re-search at new -ngl
        }
        if (params.speculative.has_dft()) {
            est += 6;                                                // spec-draft-n-max envelope doubling
            est += common_golden_section_eval_estimate(1, 32);        // spec-draft-n-max golden-section
        }
        est += 2; // thread-count candidates
        est += 8; // expert-cache size knee (fixed candidate list)
        est += 4; // fit-margin search (upper bound; stops early on first failure)
        common_moe_calibration_status_set_total(est);
    }

    // Placement is a feasibility question first and a throughput question a
    // distant second, so a narrow feasible range is decided by the (free,
    // no-alloc) fit probe rather than benchmarked. Measured on qwen4exp: once
    // the probe used the same margin serving demands, the whole range was
    // [47, 48] and the two options differed by 3% (0.61 vs 0.59 tok/s) - yet
    // benchmarking them consumed 7 of a 9-minute budget, starving the
    // substitution search, whose own range spans 20x. Spending the budget on
    // the levers that actually move is the entire point of having one.
    //
    // The threshold is on candidate COUNT, not on any measured spread: the
    // spread is only knowable by paying for the very benchmarks this is
    // deciding whether to skip. Above it the search runs as before, since a
    // genuinely wide range can hide a real optimum.
    if (ncmoe_span < ncmoe_search_min_span) {
        LOG_INF("%s: MoE CPU-offload depth fixed at %u by the fit probe - only %u placement(s) fit this "
                "context with the serving margin, not enough spread to be worth benchmarking; spending the "
                "budget on the levers that move instead\n", __func__, safe_n, ncmoe_span + 1);
    }

    if (ncmoe_span >= ncmoe_search_min_span) {
        LOG_INF("%s: golden-section search for -ncmoe in [%u, %u] at n_threads=%d%s (real llama-server subprocess, "
                "chat-templated prompts, per candidate) ...\n", __func__, safe_n, ncmoe_hi, n_threads_default,
                concurrency > 1 ? string_format(", concurrency=%d (aggregate throughput)", concurrency).c_str() : "");
        common_moe_calibration_status_set(string_format("searching MoE CPU-offload depth (-ncmoe) in [%u, %u]", safe_n, ncmoe_hi));
    } else {
        common_moe_calibration_status_set(string_format("measuring baseline at the probe-fixed placement (-ncmoe %u)", safe_n));
    }

    std::map<int, double> ncmoe_trace;
    auto measure_ncmoe = [&](int n) -> double {
        // Memoized: the boundary-extension search below re-probes points the
        // first search already measured, and a sample costs a full server spawn.
        auto it = ncmoe_trace.find(n);
        if (it != ncmoe_trace.end()) {
            return it->second;
        }
        const double tps = bench_with_retry((uint32_t) n, 0, "", n_threads_default);
        LOG_INF("%s:   ncmoe=%d -> %s\n", __func__, n, tps > 0 ? string_format("%.2f tok/s", tps).c_str() : "failed");
        common_moe_calibration_status_candidate_done();
        return tps; // failed candidates measure as -1, golden-section still works (just avoids them)
    };
    uint32_t best_n;
    double best_tps;
    if (ncmoe_span < ncmoe_search_min_span) {
        // Probe-decided (see the comment above): take the fit probe's own
        // floor and measure it ONCE, so later stages still have a real
        // baseline tok/s to compare against without paying for a search.
        best_n   = safe_n;
        best_tps = measure_ncmoe((int) safe_n);
    } else {
        best_n   = (uint32_t) common_golden_section_search_max((int) safe_n, (int) ncmoe_hi, measure_ncmoe, ncmoe_trace);
        best_tps = ncmoe_trace.at((int) best_n);
    }
    // Golden-section search assumes a unimodal objective and only ever
    // bisects toward whichever side of its *current* probe pair scores
    // higher - it never revisits a point once the search has narrowed
    // away from it, even if that point is sitting right there in the
    // trace map already measured. Confirmed to matter in practice for
    // the spec-draft-n-max search below (see docs/moe-cache-colibri-notes.md,
    // "SECOND BUG FOUND": n_max=2 scored higher than the declared winner
    // n_max=9, silently ignored because the search had already bisected
    // past it). This costs zero extra subprocess spawns - ncmoe_trace
    // already holds every point actually measured - so always trust the
    // best of what was truly measured over what golden-section converged
    // to, not just for n_max but here too as a general safety net.
    for (const auto & kv : ncmoe_trace) {
        if (kv.second > best_tps) {
            best_tps = kv.second;
            best_n = (uint32_t) kv.first;
        }
    }
    if (best_tps < 0) {
        LOG_ERR("%s: every placement candidate failed to benchmark; not writing a cache entry\n", __func__);
        return;
    }

    // The initial hi bound is a heuristic ("the peak is never more than ~15%
    // of n_layer above the safe floor"), and a heuristic that is wrong shows
    // itself in exactly one way: the winner sits *on* the boundary, meaning
    // throughput was still climbing when the search ran out of room. Measured
    // on Nemotron 3.5 Lightning, where [37,43] returned 43 with the trace
    // rising monotonically to it - because on this model extra CPU offload
    // frees VRAM that the expert cache converts straight back into hit rate.
    // Extend to the hard ceiling and keep going rather than shipping a bound
    // artifact as an optimum.
    if (best_n == ncmoe_hi && ncmoe_hi < (uint32_t) probe.n_layer) {
        const uint32_t prev_hi = ncmoe_hi;
        ncmoe_hi = (uint32_t) probe.n_layer;   // the hard ceiling: one extension is all there is
        LOG_INF("%s: peak landed on the search boundary (%u) - throughput was still rising, "
                "extending the range to [%u, %u]\n", __func__, prev_hi, prev_hi, ncmoe_hi);
        common_golden_section_search_max((int) prev_hi, (int) ncmoe_hi, measure_ncmoe, ncmoe_trace);
        for (const auto & kv : ncmoe_trace) {
            if (kv.second > best_tps) {
                best_tps = kv.second;
                best_n   = (uint32_t) kv.first;
            }
        }
    }

    // Trade GPU-resident dense/attention-layer compute for VRAM the expert
    // cache converts into hit rate. The ncmoe search above only ever moved
    // MoE EXPERT weights off GPU - every layer's attention/dense compute
    // (the majority of the graph, on a model this sparsely routed) stayed
    // GPU-resident regardless, at whatever VRAM the fit search left for it.
    // If the CPU/NVMe path for cold experts is the actual bottleneck, giving
    // up some GPU-resident layers to grow the expert cache's own (auto-sized,
    // free-VRAM-minus-reserve) budget can pay for itself, the same way extra
    // -ncmoe offload sometimes does above - just on the layer-residency axis
    // instead of the expert-placement axis. Reaching this point already means
    // the model did not fit fully on GPU (probe.already_fits was false), so
    // there is always something to trade here.
    // Substitution aggressiveness, measured BEFORE the -ngl / thread / cache
    // stages rather than after them. Serving a cache miss with a resident
    // stand-in is the largest throughput lever measured on this model
    // (0.6 -> ~12 tok/s at its most aggressive) AND the one that destroys
    // output when overused, so it is also the one most worth spending a
    // constrained budget on. Ordering it last meant a slow model never
    // reached it: a real clean-slate run spent its whole budget on the
    // placement and -ngl searches and recorded substitute_min_rank=-1,
    // having measured the biggest lever not at all while fully exploring
    // -ngl, which moved almost nothing on the same model.
    //
    // Benchmarked at the winning placement with the other knobs still at
    // their defaults, which is sound because this is a router-quality
    // boundary (which picks may be served approximately) rather than a
    // memory-sizing decision - it does not depend on the thread count or
    // cache budget chosen later.
    int best_min_rank = -1;
    double best_min_rank_tps = 0.0;
    if (!common_moe_calibrate_budget_spent()) {
        LOG_INF("%s: measuring substitution aggressiveness at ncmoe=%u (degenerate output is rejected, "
                "not ranked) ...\n", __func__, best_n);
        common_moe_calibration_status_set("measuring substitution aggressiveness");
        // Coarse ladder rather than a golden-section search: the response is
        // a quality cliff, not a smooth curve, and each point costs a full
        // server spawn on a budget that is already tight for slow models.
        // Descend to 0, not to a floor picked in advance. An earlier version
        // stopped at 2 because live testing had shown rank 0 producing word
        // salad - but that made the floor an assumption baked into the code
        // rather than something this run measured, on a machine and model it
        // may not hold for. The degeneracy guard rejects whatever actually
        // breaks, so the ladder can safely ask the question instead.
        // Establish what this model says when nothing is substituted, and how
        // much its own sampling moves the answer around. A rank floor above
        // top_k can never fire, so this is a genuinely substitution-free run.
        //
        // Two reference runs, same config, different seeds: the first is the
        // reference text, the second measures how far apart two legitimate
        // answers from this model already sit. That distance becomes the
        // tolerance - a candidate has to stay at least as close to the
        // reference as the model's own resampling does. Measuring the bar
        // instead of picking one keeps it honest across models and hardware:
        // a model that answers near-identically run to run gets a strict bar
        // automatically, a chattier one gets a loose one, and neither number
        // is written down here.
        constexpr int subst_off_rank = 1000000;
        std::string ref_text;
        std::string ref_alt_text;
        double fidelity_bar = -1.0;
        const double ref_tps = common_moe_bench_candidate_server(
                self_exe, path_model, "", best_n, 0, n_threads_default, next_port(), ctx, n_predict,
                concurrency, -1, -1, active_ngl, subst_off_rank, &ref_text, 1234);
        common_moe_calibration_status_candidate_done();
        if (ref_tps > 0 && !ref_text.empty() && !common_moe_calibrate_budget_spent()) {
            common_moe_bench_candidate_server(
                    self_exe, path_model, "", best_n, 0, n_threads_default, next_port(), ctx, n_predict,
                    concurrency, -1, -1, active_ngl, subst_off_rank, &ref_alt_text, 5678);
            common_moe_calibration_status_candidate_done();
            if (!ref_alt_text.empty()) {
                fidelity_bar = common_moe_output_fidelity(ref_text, ref_alt_text);
                // A bar of 1.0 is not a strict measurement, it is the absence
                // of one. It means the two reference runs came back identical,
                // so this model showed no answer-to-answer variation to
                // calibrate a tolerance from - and a bar of exactly 1.0 then
                // demands that a candidate reproduce the reference token for
                // token, which no substitution can do however good its
                // stand-ins are. Observed live: the bar measured 1.00 and the
                // ladder rejected every rung at fidelity 0.62, including ones
                // that may well have been fine. With no evidence about natural
                // variation, the honest move is to not gate on it and let the
                // degeneracy and reproducibility checks carry the decision,
                // rather than to invent a tolerance here.
                if (fidelity_bar >= 1.0) {
                    LOG_INF("%s:   the two reference runs were identical, so there is no measured "
                            "answer-to-answer variation to set a fidelity tolerance from - judging the "
                            "ladder on degeneracy and reproducibility alone\n", __func__);
                    fidelity_bar = -1.0;
                }
            }
        }
        if (ref_tps > 0) {
            LOG_INF("%s:   substitution off -> %.2f tok/s (reference)%s\n", __func__, ref_tps,
                    fidelity_bar >= 0.0
                        ? string_format(", fidelity bar %.2f from the model's own resampling", fidelity_bar).c_str()
                        : ", no fidelity bar - candidates judged on degeneracy alone");
            best_min_rank_tps = ref_tps;
            best_min_rank     = subst_off_rank;
        }

        for (const int rank : {10, 6, 4, 2, 1, 0}) {
            if (common_moe_calibrate_budget_spent()) {
                break;
            }
            std::string cand_text;
            const double tps = common_moe_bench_candidate_server(
                    self_exe, path_model, "", best_n, 0, n_threads_default, next_port(), ctx, n_predict,
                    concurrency, -1, -1, active_ngl, rank, &cand_text, 1234);
            common_moe_calibration_status_candidate_done();
            // Fidelity is a rejection, not a ranking term - same rule as the
            // degeneracy guard. A rung that drifts further from the reference
            // than the model's own resampling does is not a faster point on
            // the quality curve, it is a different (and unasked-for) model.
            double fidelity = -1.0;
            if (tps > 0 && fidelity_bar >= 0.0 && !cand_text.empty()) {
                fidelity = common_moe_output_fidelity(ref_text, cand_text);
                if (fidelity < fidelity_bar) {
                    LOG_WRN("%s:   substitute-min-rank=%d rejected - output diverges from the "
                            "substitution-free reference (fidelity %.2f < %.2f) at %.2f tok/s; this is the "
                            "fluent-but-wrong failure the degeneracy guard cannot see\n",
                            __func__, rank, fidelity, fidelity_bar, tps);
                    common_moe_calibration_status_note("substitution floor",
                            string_format("rank %d", rank),
                            string_format("rejected - fidelity %.2f < %.2f (was %.2f tok/s)",
                                    fidelity, fidelity_bar, tps), false);
                    continue;
                }
            }
            LOG_INF("%s:   substitute-min-rank=%d -> %s%s\n", __func__, rank,
                    tps > 0 ? string_format("%.2f tok/s", tps).c_str() : "failed or rejected as degenerate",
                    fidelity >= 0.0 ? string_format(" (fidelity %.2f)", fidelity).c_str() : "");
            common_moe_calibration_status_note("substitution floor",
                    string_format("rank %d", rank),
                    tps > 0 ? string_format("%.2f tok/s%s", tps,
                            fidelity >= 0.0 ? string_format(", fidelity %.2f", fidelity).c_str() : "")
                            : std::string("failed or degenerate"),
                    tps > 0);
            if (tps > best_min_rank_tps) {
                best_min_rank_tps = tps;
                best_min_rank     = rank;
            }
        }
        // Confirm the winner over a LONGER generation before committing it.
        // The ladder's own probes are short (n_predict, kept small so more
        // levers fit the budget), and short-probe coherence does not prove
        // long-generation coherence: every degenerate output observed by hand
        // on this model appeared at 150-250 tokens, well past where these
        // probes stop looking. Rather than leave that as a known blind spot,
        // re-measure the winner at 4x the probe length and, if the guard
        // rejects it there, fall back one rung toward the safe end and
        // confirm that instead.
        while (best_min_rank >= 0 && !common_moe_calibrate_budget_spent()) {
            const int confirm_predict = n_predict * 4;
            // Say so. This step re-runs the winner at 4x the probe length and
            // on a slow model that is several minutes during which the stage
            // line and the decisions table both sat unchanged, which reads as
            // a stall rather than as the longest single check in the run.
            common_moe_calibration_status_set(string_format(
                    "confirming substitution floor rank %d over %d tokens",
                    best_min_rank, confirm_predict));
            const double tps = common_moe_bench_candidate_server(
                    self_exe, path_model, "", best_n, 0, n_threads_default, next_port(), ctx,
                    confirm_predict, concurrency, -1, -1, active_ngl, best_min_rank);
            common_moe_calibration_status_candidate_done();
            if (tps > 0) {
                LOG_INF("%s:   rank %d confirmed over %d tokens -> %.2f tok/s\n",
                        __func__, best_min_rank, confirm_predict, tps);
                common_moe_calibration_status_note("substitution floor",
                        string_format("rank %d", best_min_rank),
                        string_format("confirmed over %d tokens - %.2f tok/s", confirm_predict, tps), true);
                break;
            }
            // Degenerate (or failed) at length: step toward the safe end.
            const int safer = best_min_rank < 2 ? 2 : (best_min_rank < 4 ? 4 : (best_min_rank < 6 ? 6 : 10));
            LOG_WRN("%s:   rank %d did NOT hold over %d tokens - stepping back to rank %d\n",
                    __func__, best_min_rank, confirm_predict, safer);
            common_moe_calibration_status_note("substitution floor",
                    string_format("rank %d", best_min_rank),
                    string_format("did not hold over %d tokens - stepping back to rank %d",
                            confirm_predict, safer), false);
            if (safer == best_min_rank || safer > 10) {
                best_min_rank = -1; // nothing survived confirmation; leave the runtime default in place
                break;
            }
            best_min_rank = safer;
        }
        if (best_min_rank >= 0) {
            LOG_INF("%s: substitution floor: rank %d at %.2f tok/s - carried into the remaining stages\n",
                    __func__, best_min_rank, best_min_rank_tps);
            common_moe_calibration_status_note("substitution floor",
                    string_format("rank %d", best_min_rank),
                    string_format("SELECTED - %.2f tok/s, carried into the remaining stages", best_min_rank_tps),
                    true, /* chosen */ true);
            // Everything measured from here on is measured at this floor, so
            // the later knobs are tuned against the configuration that will
            // actually be served.
            active_min_rank = best_min_rank;
            best_tps = std::max(best_tps, best_min_rank_tps);
        }

        // Stand-in quality bar, measured at the floor just chosen. The rank
        // floor answers "how much did the router want this expert"; this
        // answers "is the replacement on offer actually any good", and it is
        // the one that decides when inference waits for a real fetch and when
        // it runs on what is already resident. Descending order because the
        // strict end is the safe end: a high sigma demands a stand-in far
        // better than typical and mostly declines, which costs throughput and
        // nothing else, while the permissive end is where output dies - on
        // this hardware an ungated run reaches 51 tok/s emitting "* * * *"
        // and nothing more. Every rung goes through the same degeneracy,
        // fidelity and reproducibility gates as the ladder above, so the
        // permissive end is rejected on evidence rather than avoided by a
        // number written here.
        double best_sigma = std::numeric_limits<double>::quiet_NaN();
        double best_sigma_tps = best_min_rank_tps;
        if (!common_moe_calibrate_budget_spent()) {
            LOG_INF("%s: measuring stand-in quality bar at rank %d (this is the wait-or-substitute "
                    "balance) ...\n", __func__, active_min_rank);
            common_moe_calibration_status_set("measuring stand-in quality bar");
            for (const double sigma : {2.0, 1.0, 0.0, -1.0, -2.0}) {
                if (common_moe_calibrate_budget_spent()) {
                    break;
                }
                std::string cand_text;
                const double tps = common_moe_bench_candidate_server(
                        self_exe, path_model, "", best_n, 0, n_threads_default, next_port(), ctx, n_predict,
                        concurrency, -1, -1, active_ngl, active_min_rank, &cand_text, 1234, sigma);
                common_moe_calibration_status_candidate_done();
                double fidelity = -1.0;
                if (tps > 0 && fidelity_bar >= 0.0 && !cand_text.empty()) {
                    fidelity = common_moe_output_fidelity(ref_text, cand_text);
                    if (fidelity < fidelity_bar) {
                        LOG_WRN("%s:   quality-sigma=%.1f rejected - diverges from the substitution-free "
                                "reference (fidelity %.2f < %.2f) at %.2f tok/s\n",
                                __func__, sigma, fidelity, fidelity_bar, tps);
                        common_moe_calibration_status_note("stand-in quality bar",
                                string_format("%+.1f sigma", sigma),
                                string_format("rejected - fidelity %.2f < %.2f (was %.2f tok/s)",
                                        fidelity, fidelity_bar, tps), false);
                        continue;
                    }
                }
                LOG_INF("%s:   quality-sigma=%.1f -> %s%s\n", __func__, sigma,
                        tps > 0 ? string_format("%.2f tok/s", tps).c_str() : "failed or rejected",
                        fidelity >= 0.0 ? string_format(" (fidelity %.2f)", fidelity).c_str() : "");
                common_moe_calibration_status_note("stand-in quality bar",
                        string_format("%+.1f sigma", sigma),
                        tps > 0 ? string_format("%.2f tok/s%s", tps,
                                fidelity >= 0.0 ? string_format(", fidelity %.2f", fidelity).c_str() : "")
                                : std::string("failed or rejected"),
                        tps > 0);
                if (tps > best_sigma_tps) {
                    best_sigma_tps = tps;
                    best_sigma     = sigma;
                }
            }
            if (!std::isnan(best_sigma)) {
                LOG_INF("%s: stand-in quality bar: %.1f sigma at %.2f tok/s\n",
                        __func__, best_sigma, best_sigma_tps);
                common_moe_calibration_status_note("stand-in quality bar",
                        string_format("%+.1f sigma", best_sigma),
                        string_format("SELECTED - %.2f tok/s", best_sigma_tps), true, /* chosen */ true);
                best_tps = std::max(best_tps, best_sigma_tps);
            }
        }
        active_quality_sigma = best_sigma;
    }

    uint32_t best_ngl = (uint32_t) probe.n_layer + 1; // "all" - mirrors llama_model::n_gpu_layers()'s own +1
    {
        const uint32_t ngl_hi = best_ngl;
        const uint32_t ngl_lo = probe.n_layer > 3 ? probe.n_layer - probe.n_layer / 3 : 0;
        LOG_INF("%s: golden-section search for -ngl in [%u, %u] at ncmoe=%u (trading GPU-resident layers "
                "for expert-cache VRAM) ...\n", __func__, ngl_lo, ngl_hi, best_n);
        common_moe_calibration_status_set(string_format("searching GPU-resident layer count (-ngl) in [%u, %u]", ngl_lo, ngl_hi));

        std::map<int, double> ngl_trace;
        ngl_trace[(int) ngl_hi] = best_tps; // already measured above - full residency is the ncmoe search's own baseline
        // bench_with_retry doesn't take an ngl override, so this wraps
        // common_moe_bench_candidate_server directly instead.
        auto measure_ngl = [&](int n) -> double {
            double sum = 0.0; int n_ok = 0;
            for (int i = 0; i < n_samples_per_candidate; i++) {
                double tps = common_moe_bench_candidate_server(
                        self_exe, path_model, "", best_n, 0, n_threads_default, next_port(), ctx, n_predict,
                        concurrency, -1, -1, n, active_min_rank);
                if (tps < 0) {
                    tps = common_moe_bench_candidate_server(
                            self_exe, path_model, "", best_n, 0, n_threads_default, next_port(), ctx, n_predict,
                            concurrency, -1, -1, n, active_min_rank);
                }
                if (tps > 0) { sum += tps; n_ok++; }
            }
            const double result = n_ok > 0 ? sum / n_ok : -1.0;
            LOG_INF("%s:   ngl=%d -> %s\n", __func__, n, result > 0 ? string_format("%.2f tok/s", result).c_str() : "failed");
            common_moe_calibration_status_candidate_done();
            return result;
        };

        best_ngl = (uint32_t) common_golden_section_search_max((int) ngl_lo, (int) ngl_hi, measure_ngl, ngl_trace);
        double best_ngl_tps = ngl_trace.at((int) best_ngl);
        for (const auto & kv : ngl_trace) {
            if (kv.second > best_ngl_tps) {
                best_ngl_tps = kv.second;
                best_ngl     = (uint32_t) kv.first;
            }
        }
        if (best_ngl == ngl_lo && ngl_lo > 0) {
            LOG_INF("%s: -ngl peak landed on the search boundary (%u) - throughput was still rising as more "
                    "layers moved to CPU, extending the range to [0, %u]\n", __func__, ngl_lo, ngl_lo);
            common_golden_section_search_max(0, (int) ngl_lo, measure_ngl, ngl_trace);
            for (const auto & kv : ngl_trace) {
                if (kv.second > best_ngl_tps) {
                    best_ngl_tps = kv.second;
                    best_ngl     = (uint32_t) kv.first;
                }
            }
        }

        if (best_ngl_tps > best_tps && best_ngl < ngl_hi) {
            LOG_INF("%s: -ngl=%u wins over full GPU residency (%.2f vs %.2f tok/s) - re-searching ncmoe at "
                    "this layer residency, since the safe floor and available VRAM both just changed\n",
                    __func__, best_ngl, best_ngl_tps, best_tps);
            common_moe_calibration_status_set(string_format("re-searching -ncmoe at -ngl=%u", best_ngl));
            best_tps = best_ngl_tps;

            llama_model_params mparams_ngl = mparams;
            mparams_ngl.n_gpu_layers = (int) best_ngl;
            // Same serving-time margin as the initial probe above - a re-search
            // floor computed on a bare fit would reintroduce exactly the
            // mismatch that made the first search's winner unusable.
            common_moe_fit_probe_result probe_ngl =
                    common_moe_find_safe_layers(path_model, mparams_ngl, cparams, fit_margin);
            const uint32_t safe_n_ngl = probe_ngl.already_fits ? 0 :
                    (probe_ngl.found_safe_n ? probe_ngl.safe_n : safe_n);
            const uint32_t ncmoe_hi_ngl = std::min<uint32_t>(probe.n_layer, safe_n_ngl + std::max<uint32_t>(4, probe.n_layer / 8));

            if (safe_n_ngl < ncmoe_hi_ngl) {
                std::map<int, double> ncmoe_trace2;
                auto measure_ncmoe2 = [&](int n) -> double {
                    double tps = common_moe_bench_candidate_server(
                            self_exe, path_model, "", (uint32_t) n, 0, n_threads_default, next_port(), ctx, n_predict,
                            concurrency, -1, -1, (int) best_ngl, active_min_rank);
                    if (tps < 0) {
                        tps = common_moe_bench_candidate_server(
                                self_exe, path_model, "", (uint32_t) n, 0, n_threads_default, next_port(), ctx, n_predict,
                                concurrency, -1, -1, (int) best_ngl, active_min_rank);
                    }
                    LOG_INF("%s:   ncmoe=%d (ngl=%u) -> %s\n", __func__, n, best_ngl,
                            tps > 0 ? string_format("%.2f tok/s", tps).c_str() : "failed");
                    common_moe_calibration_status_candidate_done();
                    return tps;
                };
                uint32_t best_n2 = (uint32_t) common_golden_section_search_max(
                        (int) safe_n_ngl, (int) ncmoe_hi_ngl, measure_ncmoe2, ncmoe_trace2);
                double best_tps2 = ncmoe_trace2.at((int) best_n2);
                for (const auto & kv : ncmoe_trace2) {
                    if (kv.second > best_tps2) {
                        best_tps2 = kv.second;
                        best_n2   = (uint32_t) kv.first;
                    }
                }
                if (best_tps2 > best_tps) {
                    LOG_INF("%s: ncmoe=%u wins at ngl=%u (%.2f tok/s, was %.2f)\n",
                            __func__, best_n2, best_ngl, best_tps2, best_tps);
                    best_n   = best_n2;
                    best_tps = best_tps2;
                }
            }
        } else {
            LOG_INF("%s: full GPU residency still wins (%.2f tok/s) - -ngl left at default\n", __func__, best_tps);
            best_ngl = ngl_hi;
        }
        // Carry it forward, so the thread / cache-size / fit-margin stages
        // below measure at the layer residency actually chosen.
        if (best_ngl <= (uint32_t) probe.n_layer) {
            active_ngl = (int) best_ngl;
        }
    }

    int best_n_max = -1;
    if (params.speculative.has_dft()) {
        {
            // Find the envelope: double n_max until throughput drops below
            // half the n_max=1 baseline (the real n_max=8 collapse we
            // measured went from ~80% cache health to ~14-21% - a >2x
            // throughput cliff, not a gentle decline, so "less than half"
            // is a safe, real signal for "past the edge" rather than
            // ordinary run-to-run noise).
            LOG_INF("%s: finding spec-draft-n-max envelope (doubling until collapse) via real llama-server subprocesses ...\n", __func__);
            common_moe_calibration_status_set("searching speculative-decoding depth (spec-draft-n-max)");
            std::map<int, double> nmax_trace;
            double baseline = -1.0;
            int last_good = 1;
            for (int n = 1; n <= 32; n *= 2) {
                const double tps = bench_with_retry(best_n, n, params.speculative.draft.mparams.path, n_threads_default);
                nmax_trace[n] = tps;
                LOG_INF("%s:   spec-draft-n-max=%d -> %s\n", __func__, n, tps > 0 ? string_format("%.2f tok/s", tps).c_str() : "failed");
                common_moe_calibration_status_candidate_done();
                if (n == 1) {
                    baseline = tps;
                }
                if (tps < 0 || (baseline > 0 && tps < baseline * 0.5)) {
                    break;
                }
                last_good = n;
            }
            const int nmax_hi = std::max(1, last_good);

            if (baseline > 0) {
                LOG_INF("%s: golden-section search for spec-draft-n-max in [1, %d] at ncmoe=%u ...\n",
                        __func__, nmax_hi, best_n);
                auto measure_nmax = [&](int n) -> double {
                    auto it = nmax_trace.find(n);
                    if (it != nmax_trace.end()) {
                        return it->second;
                    }
                    const double tps = bench_with_retry(best_n, n, params.speculative.draft.mparams.path, n_threads_default);
                    LOG_INF("%s:   spec-draft-n-max=%d -> %s\n", __func__, n, tps > 0 ? string_format("%.2f tok/s", tps).c_str() : "failed");
                    common_moe_calibration_status_candidate_done();
                    return tps;
                };
                best_n_max = common_golden_section_search_max(1, nmax_hi, measure_nmax, nmax_trace);
                // Validate against the full trace, not just what
                // golden-section converged to - see the comment on the
                // identical check after the ncmoe search above. This is
                // the fix for a confirmed real bug: the envelope-doubling
                // phase above already measures n=1,2,4,8,... before
                // golden-section ever runs, but golden-section's own
                // bisection can (and, measured once, did: n_max=2 scored
                // 56.06 tok/s vs the bisection's own pick of n_max=9 at
                // 54.55) narrow away from those low values without ever
                // reconsidering them.
                for (const auto & kv : nmax_trace) {
                    if (kv.second > nmax_trace.at(best_n_max)) {
                        best_n_max = kv.first;
                    }
                }
                LOG_INF("%s: spec-draft-n-max=%d wins (%.2f tok/s)\n", __func__, best_n_max, nmax_trace.at(best_n_max));
                // The n_max search's own winning number (MTP active) is the
                // real answer for this deployment, not the earlier ncmoe-only
                // number (MTP off) - carry it forward so the final report and
                // cache entry don't undersell what was actually found.
                if (nmax_trace.at(best_n_max) > best_tps) {
                    best_tps = nmax_trace.at(best_n_max);
                }
            } else {
                LOG_WRN("%s: spec-draft-n-max=1 itself failed to benchmark - skipping n_max calibration\n", __func__);
            }
        }
    }

    // Now uses the same correct subprocess/chat-template path as ncmoe and
    // n_max above, so - unlike before this fix - it's safe to also tune
    // threads under MTP when MTP was calibrated: same real generation
    // conditions, numbers are directly comparable to best_tps above.
    const std::string mtp_path_for_threads = best_n_max > 0 ? params.speculative.draft.mparams.path : std::string();
    const int n_max_for_threads = best_n_max > 0 ? best_n_max : 0;

    const int n_threads_physical = common_cpu_get_num_physical_cores();
    const int n_threads_logical  = (int) std::thread::hardware_concurrency();
    std::vector<int> thread_candidates = { n_threads_default };
    if (n_threads_physical > 0 && n_threads_physical != n_threads_default) {
        thread_candidates.push_back(n_threads_physical);
    }
    if (n_threads_logical > 0 && n_threads_logical != n_threads_default && n_threads_logical != n_threads_physical) {
        thread_candidates.push_back(n_threads_logical);
    }

    int    best_threads = n_threads_default;
    double best_threads_tps = best_tps;
    if (thread_candidates.size() > 1) {
        common_moe_calibration_status_set(string_format("benchmarking %zu thread-count candidate(s)", thread_candidates.size()));
        LOG_INF("%s: benchmarking %zu thread-count candidate(s) at ncmoe=%u%s ...\n", __func__, thread_candidates.size(), best_n,
                best_n_max > 0 ? string_format(", spec-draft-n-max=%d", best_n_max).c_str() : "");
        for (int nt : thread_candidates) {
            if (nt == n_threads_default) {
                continue; // already measured above as best_tps
            }
            const double tps = bench_with_retry(best_n, n_max_for_threads, mtp_path_for_threads, nt);
            LOG_INF("%s:   n_threads=%d -> %s\n", __func__, nt, tps > 0 ? string_format("%.2f tok/s", tps).c_str() : "failed");
            common_moe_calibration_status_candidate_done();
            if (tps > best_threads_tps) {
                best_threads_tps = tps;
                best_threads     = nt;
            }
        }
    }

    // Expert-cache size: same question as -ncmoe, same answer - measure this
    // model's own knee rather than assume one. Our own sweep found the
    // relationship is not "more is better": Nemotron flattened at 4 GiB (of a
    // 7-8 GiB default) with no loss, while a too-small request starved the
    // fit search's VRAM margin and collapsed concurrent slots entirely. Start
    // small and grow only while it still pays; stop and take the smallest
    // point once growth stops paying, mirroring the n_max envelope-doubling
    // approach below rather than a full sweep of every candidate.
    LOG_INF("%s: finding expert-cache size knee (growing while it still helps) at ncmoe=%u ...\n",
            __func__, best_n);
    common_moe_calibration_status_set("searching expert-cache VRAM budget size");
    static const int cache_candidates_mb[] = {512, 1024, 2048, 4096, 6144, 8192, 12288, 16384};
    // Test every candidate rather than stopping at the first non-improving
    // step: the curve is not guaranteed monotonic below the knee - our own
    // sweep found a placement cliff at small cache sizes (too little VRAM
    // margin starves the fit search of concurrent slots, e.g. 4 -> 1, a
    // collapse unrelated to cache-hit-rate that a local stop rule would
    // mistake for "smaller is fine"). Picking the smallest point within 3%
    // of the *global* max, after seeing the whole curve, is robust to that
    // dip in a way a running best-so-far comparison is not.
    std::vector<std::pair<int, double>> cache_results;
    for (size_t i = 0; i < sizeof(cache_candidates_mb) / sizeof(cache_candidates_mb[0]); i++) {
        const int mb = cache_candidates_mb[i];
        double tps = common_moe_bench_candidate_server(
                self_exe, path_model, mtp_path_for_threads, best_n, n_max_for_threads,
                best_threads, next_port(), ctx, n_predict, concurrency, mb, -1, active_ngl, active_min_rank);
        if (tps < 0) {
            tps = common_moe_bench_candidate_server(
                    self_exe, path_model, mtp_path_for_threads, best_n, n_max_for_threads,
                    best_threads, next_port(), ctx, n_predict, concurrency, mb, -1, active_ngl, active_min_rank);
        }
        LOG_INF("%s:   moe-cache=%dMiB -> %s\n", __func__, mb,
                tps > 0 ? string_format("%.2f tok/s", tps).c_str() : "failed");
        common_moe_calibration_status_candidate_done();
        if (tps > 0) {
            cache_results.emplace_back(mb, tps);
        }
    }
    int    best_cache_mb  = -1; // -1 stays "auto" if the sweep found nothing usable
    double best_cache_tps = -1.0;
    for (const auto & r : cache_results) {
        best_cache_tps = std::max(best_cache_tps, r.second);
    }
    if (best_cache_tps > 0) {
        for (const auto & r : cache_results) {
            if (r.second >= best_cache_tps * 0.97) {
                best_cache_mb = r.first; // smallest candidate within 3% of the global max
                break;
            }
        }
    }

    // Fit margin (-fitt). Searched LAST and deliberately re-searching
    // -ncmoe underneath it, because the two are not independent: the margin
    // is a floor on how little offload is allowed, so lowering it does not
    // just free VRAM, it unlocks placements the -ncmoe search above was
    // never permitted to evaluate. Measured on Ornith-1.5-35B-Q4_K_M /
    // RTX 3060 12 GB / -c 4096, requesting -ncmoe 8:
    //
    //     fitt=1024 (default) -> forced to ncmoe 27, 47.32 tok/s
    //     fitt=640            -> forced to ncmoe 24, 50.37 tok/s
    //     fitt=448            -> forced to ncmoe 23, 51.61 tok/s
    //     fitt=320            -> forced to ncmoe 22, 52.85 tok/s
    //
    // +11.7% end to end, entirely from VRAM the default margin was holding
    // in reserve and CPU work that reservation forced. Descending (largest
    // margin first, i.e. safest first) and stopping at the first candidate
    // that fails to launch or produces no improvement: below some point the
    // margin stops covering the post-probe allocations it exists for (real
    // weight loading, lazy CUDA graph capture) and the server either fails
    // outright or collapses its context - a documented, reproduced failure,
    // so this walks toward the edge and stops rather than bisecting across
    // it. Each candidate asks for the minimum offload (safe_n) and lets the
    // fit logic raise it to whatever that margin actually permits, which is
    // the quantity being measured.
    int    best_fit_mb  = -1;
    // Baseline MUST be the best throughput already measured at the DEFAULT
    // margin under the SAME cache size and thread count these candidates
    // run with - i.e. the cache-knee winner, not best_threads_tps. Getting
    // this wrong is not a subtle accounting slip: best_threads_tps was
    // measured during the thread sweep at a different (auto) cache size, so
    // comparing margin candidates against it compares across two different
    // configurations at once. Measured consequence on gemma-4-26B-A4B,
    // concurrency 4: the cache knee found 71.86 tok/s at 4096 MiB with the
    // default 1024 MiB margin, every tightened margin scored ~50, and this
    // search still declared fitt=448 the winner because it was only being
    // compared against the thread stage's 43.57. That entry would have
    // shipped a ~29% REGRESSION as a calibrated optimum. Seeding with the
    // cache-stage best means -1 ("keep the default") correctly survives
    // whenever tightening does not actually help, which for this model it
    // does not.
    double best_fit_tps = best_cache_tps > 0 ? best_cache_tps : best_threads_tps;
    {
        const int default_fit_mb = (int) (params.fit_params_target[0] / (1024 * 1024));
        static const int fit_candidates_mb[] = {640, 448, 320, 256};
        common_moe_calibration_status_set("searching fit margin (-fitt)");
        LOG_INF("%s: searching fit margin (-fitt) below the default of %d MiB at ncmoe>=%u ...\n",
                __func__, default_fit_mb, safe_n);
        for (size_t i = 0; i < sizeof(fit_candidates_mb) / sizeof(fit_candidates_mb[0]); i++) {
            const int mb = fit_candidates_mb[i];
            if (mb >= default_fit_mb) {
                continue; // only ever tighten below the default, never loosen past it
            }
            const double tps = common_moe_bench_candidate_server(
                    self_exe, path_model, mtp_path_for_threads, safe_n, n_max_for_threads,
                    best_threads, next_port(), ctx, n_predict, concurrency, best_cache_mb, mb, active_ngl, active_min_rank);
            LOG_INF("%s:   fitt=%dMiB -> %s\n", __func__, mb,
                    tps > 0 ? string_format("%.2f tok/s", tps).c_str() : "failed");
            common_moe_calibration_status_candidate_done();
            if (tps <= 0) {
                LOG_INF("%s:   fitt=%dMiB did not come up - stopping here, this is the edge the "
                        "margin exists to stay clear of\n", __func__, mb);
                break;
            }
            if (tps > best_fit_tps) {
                best_fit_tps = tps;
                best_fit_mb  = mb;
            }
        }
    }

    time_t now = time(nullptr);
    char timebuf[32];
    strftime(timebuf, sizeof(timebuf), "%Y-%m-%d %H:%M:%S", localtime(&now));

    common_moe_calibration_entry entry;
    entry.n_cpu_moe       = (int) best_n;
    entry.n_threads       = best_threads;
    entry.n_threads_batch = best_threads;
    entry.spec_n_max      = best_n_max;
    entry.concurrency     = concurrency;
    // best_fit_tps is already seeded from the cache-knee best, so it is the
    // running maximum across every stage that measured a full candidate -
    // reporting anything lower here would understate what this entry's own
    // settings actually achieved (the pre-existing code reported only
    // best_threads_tps, which ignored the cache sweep entirely and on
    // gemma-4 understated the result by 71.86 -> 43.57).
    // Include the substitution ladder's own best: substitute_min_rank is part
    // of the configuration this entry caches, so a figure that ignores it
    // understates what the cached config actually delivers. Measured on
    // qwen4exp: the ladder found 1.64 tok/s at rank 2 while the entry
    // reported 0.65, the placement/-ngl number - the same understatement the
    // cache-sweep comment above already describes for a different stage.
    entry.tok_per_sec     = std::max({best_threads_tps, best_fit_tps, best_min_rank_tps});
    entry.moe_cache_mb    = best_cache_mb;
    entry.substitute_min_rank = best_min_rank;
    entry.substitute_quality_sigma = active_quality_sigma;
    entry.fit_target_mb   = best_fit_mb;
    // -1 ("not calibrated / use default") when full GPU residency won its own
    // search above - only recorded as an explicit override when giving up
    // some GPU-resident layers actually measured faster.
    entry.n_gpu_layers    = best_ngl <= (uint32_t) probe.n_layer ? (int) best_ngl : -1;
    // Record the reduction settings this calibration was actually measured
    // under, read from the same environment the cache itself reads (rather
    // than from params, which has no field for them) - see the field
    // comments on common_moe_calibration_entry for why an entry measured
    // with reduction on is not valid for a run with it off.
    {
        const char * en = getenv("GGML_CUDA_MOE_CACHE_NEURON_REDUCE");
        if (en && atoi(en) != 0) {
            const char * k  = getenv("GGML_CUDA_MOE_CACHE_NEURON_REDUCE_K");
            const char * bm = getenv("GGML_CUDA_MOE_CACHE_NEURON_REDUCE_BUDGET_MB");
            entry.neuron_reduce_k         = k  ? atoi(k)  : 256;
            entry.neuron_reduce_budget_mb = bm ? atoi(bm) : 256;
        }
    }
    entry.calibrated_at   = timebuf;
    common_moe_calibration_save(path_model, params, entry);

    const char * tps_label = concurrency > 1 ? "aggregate tok/s" : "tok/s";
    const std::string ngl_note = entry.n_gpu_layers >= 0 ?
            string_format(", ngl=%d (some layers moved to CPU to grow the expert cache)", entry.n_gpu_layers) : "";
    if (best_n_max > 0) {
        LOG_INF("%s: calibration complete - ncmoe=%d, n_threads=%d, spec-draft-n-max=%d%s%s, measured %.2f %s. "
                "Cached to %s - launch normally (without --moe-calibrate) to use it.\n",
                __func__, entry.n_cpu_moe, entry.n_threads, entry.spec_n_max,
                concurrency > 1 ? string_format(", concurrency=%d", concurrency).c_str() : "", ngl_note.c_str(),
                entry.tok_per_sec, tps_label, common_moe_calibration_cache_path().c_str());
    } else {
        LOG_INF("%s: calibration complete - ncmoe=%d, n_threads=%d%s%s, measured %.2f %s. "
                "Cached to %s - launch normally (without --moe-calibrate) to use it.\n",
                __func__, entry.n_cpu_moe, entry.n_threads,
                concurrency > 1 ? string_format(", concurrency=%d", concurrency).c_str() : "", ngl_note.c_str(),
                entry.tok_per_sec, tps_label, common_moe_calibration_cache_path().c_str());
    }
}

// Runs as a fallback *after* the general --fit system has already had its
// chance (whether --fit is on or off): --fit's only levers are -c and -ngl,
// so as soon as a user pins both explicitly (a completely ordinary thing to
// do - e.g. -ngl 99 -c 16384) it has nothing left to try and gives up,
// while common_init_result() discards that failure status and attempts the
// real load anyway. This fills the specific gap --fit doesn't cover: partial
// MoE-expert CPU offload while leaving -ngl/-c exactly as the user set them.
// Only steps in when nothing has already decided placement (-ncmoe, -ot,
// -cmoe, or --fit itself already writing real overrides all leave this
// alone) and the config as given would actually fail to fit - this predicts
// the failure via the same no-alloc probe --fit-moe-cache uses, instead of
// catching the crash after a real allocation fails (CUDA allocator state
// after a real failed cudaMalloc isn't something to rely on being cleanly
// retriable). No-op (and no added startup cost beyond one no-alloc probe)
// whenever the config already fits.
//
// Checks the --moe-calibrate cache first (single verification probe, not
// the full binary search) - if a prior calibration run still fits under
// current conditions, its throughput-optimal answer is used directly
// instead of just the memory-safe floor. Falls back to the live binary
// search if there's no cache entry, or if conditions changed enough that
// the cached placement no longer fits.
// MUL_MAT_ID (MoE expert routing) silently drops off the fast MMVQ kernel
// onto a slower general-batch path above a per-(GPU-arch, quant-type)
// batch-size threshold - confirmed empirically this session as a real,
// severe throughput cliff (>2x slower per token, RTX 3060 + Gemma-4-26B-A4B:
// ~12.6 tok/s/sequence at concurrency=8, ~6.0 at concurrency=9, the exact
// boundary of MMVQ_MAX_BATCH_SIZE=8 in ggml/src/ggml-cuda/mmvq.cu), not a
// gentle diminishing return - and for NVIDIA cc>=Ada Lovelace (which
// includes H200/Hopper) the threshold is a flat 8 regardless of quant type.
// This is a GPU-kernel-dispatch limit, not a placement/VRAM question, so
// unlike the rest of Layer 1 it can't be fixed by choosing a different
// -ncmoe - the only real lever is the operator's own --parallel choice, so
// this only ever warns, never silently changes anything.
//
// The real threshold depends on the model's actual expert-tensor quant
// type, which isn't cheaply available before a full model load. Instead of
// guessing one type, this queries every quant type MoE GGUFs commonly ship
// with and reports the minimum across them - a true conservative lower
// bound (the real answer for whatever type this model actually uses can
// only be >= this), so the warning can under-fire slightly early but can
// never miss a real risk by assuming a too-generous type.
//
// Checks every GPU device, not just the first: on a tensor-split multi-GPU
// setup, layers (and therefore MUL_MAT_ID calls) are spread across devices,
// so the full concurrent batch hits whichever device is computing the
// current layer - the cliff can happen on any of them, and the safe ceiling
// for the whole deployment is the minimum across every device, not just
// device 0's. Each device gets its own ggml_backend_dev_t passed straight
// through to the backend's own proc-address function, which recovers that
// backend's real internal device index from it - this file never assumes
// global device index i lines up with the backend's own numbering.
static int common_moe_min_mmvq_max_batch(void) {
    static const ggml_type candidate_types[] = {
        GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K,
        GGML_TYPE_Q4_0, GGML_TYPE_Q5_0, GGML_TYPE_Q8_0, GGML_TYPE_IQ4_XS, GGML_TYPE_MXFP4,
    };

    int min_batch = INT_MAX;
    bool found_any = false;

    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
            continue;
        }
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        if (!reg) {
            continue;
        }
        auto get_max_batch = (int (*)(int, ggml_backend_dev_t)) ggml_backend_reg_get_proc_address(
                reg, "ggml_backend_get_mmid_mmvq_max_batch");
        if (!get_max_batch) {
            continue; // this device's backend doesn't expose this (not CUDA) - nothing known for it
        }
        found_any = true;
        for (ggml_type t : candidate_types) {
            min_batch = std::min(min_batch, get_max_batch((int) t, dev));
        }
    }

    return found_any ? min_batch : -1;
}

// MTP/speculative verification batches (n_max+1) candidate positions per
// sequence into a single MUL_MAT_ID call, not just 1. Shared by both the
// MMVQ-cliff warning below and the moe-cache max_batch fix right after it -
// moe-cache's own hint (llama-context.cpp's set_max_batch_hint) does NOT
// apply this multiplier, it passes n_seq_max alone (checked directly in
// the source, not assumed) - a real, separate gap this session found while
// investigating the MMVQ cliff, fixed by common_moe_apply_mtp_aware_max_batch_hint
// below rather than touching llama_context's public API surface.
static int common_moe_verify_width(const common_params & params) {
    return params.speculative.has_dft() ? (params.speculative.draft.n_max + 1) : 1;
}

// Hard safety clamp on concurrent slots while the MoE expert cache is active.
//
// This is a CORRECTNESS guard, not a throughput knob, which is why it clamps
// instead of warning: above the limit the cache produces silently corrupted
// output - the model emits garbage vocabulary tokens (`<unused49>` on
// gemma-4) instead of text - and the session never recovers, so every later
// request on that server is affected too, including sequential ones.
//
// Measured directly, gemma-4-26B-A4B / RTX 3060 / -c 4096, 8 concurrent
// requests x 3 rounds per configuration, counting responses containing
// garbage tokens:
//
//     --parallel 2, cache on        ->  0/24   clean
//     --parallel 3, cache on        -> 21/24   corrupt, and permanent
//     --parallel 4, cache on        -> 20/24   corrupt, and permanent
//     --parallel 8, cache on        -> 24/24   corrupt, and permanent
//     --parallel 4, --moe-cache off ->  0/24   clean
//
// The last row is what makes this the cache's own bug rather than a general
// concurrency problem in the server, and it reproduced with neuron
// subsetting, neuron heat and the atlas all disabled - so it is pre-existing
// rather than anything the heat-aware work introduced. Root cause is not yet
// found; until it is, refusing to run in the configuration that corrupts is
// strictly better than producing wrong output at speed. See docs/plan.md.
//
// GGML_CUDA_MOE_CACHE_MAX_PARALLEL overrides the limit for anyone
// deliberately investigating the bug (set it higher to reproduce), and a
// calibrated entry can raise it for a machine where a higher value has
// actually been verified clean - but the DEFAULT is the measured-safe value,
// because the failure mode is silent.
static void common_enforce_moe_cache_parallel_limit(common_params & params, llama_context_params & cparams) {
    const char * mode = getenv("GGML_CUDA_MOE_CACHE_MODE");
    const char * en   = getenv("GGML_CUDA_MOE_CACHE");
    const bool cache_off = (mode && std::string(mode) == "off") || (en && std::string(en) == "0");
    if (cache_off || params.n_parallel <= 0) {
        return;
    }
    int limit = 2; // measured safe ceiling - see the table above
    if (const char * ov = getenv("GGML_CUDA_MOE_CACHE_MAX_PARALLEL")) {
        const int v = atoi(ov);
        if (v > 0) {
            limit = v;
        }
    }
    if (params.n_parallel <= limit) {
        return;
    }
    LOG_WRN("%s: --parallel %d exceeds the MoE expert cache's measured-safe concurrency limit of %d - "
            "clamping to %d. Above this limit the cache corrupts output silently and permanently "
            "(garbage vocabulary tokens, and the session never recovers). This is a correctness "
            "guard, not a tuning choice. Run with --moe-cache off to use higher concurrency, or set "
            "GGML_CUDA_MOE_CACHE_MAX_PARALLEL to override if you have verified a higher value clean "
            "on this machine.\n",
            __func__, params.n_parallel, limit, limit);
    params.n_parallel = limit;
    // cparams.n_seq_max must move with it, not just the reported slot count.
    // The per-slot context has already been divided by the ORIGINAL slot
    // count by this point, so clamping params alone leaves the server with
    // (say) 2 slots each holding 4096/8 = 512 tokens instead of 4096/2 =
    // 2048 - silently costing three quarters of each slot's context as a
    // side effect of a safety guard, which would be its own bug.
    if (cparams.n_seq_max > (uint32_t) limit) {
        cparams.n_seq_max = (uint32_t) limit;
    }
}

static void common_warn_concurrency_cliff(const common_params & params) {
    // Ignoring the MTP multiplier here would under-warn: even --parallel 1
    // can hit the real cliff once MTP's own verify-width alone exceeds the
    // threshold, so it has to be applied before deciding whether there's
    // anything to check at all, not gated behind an n_parallel>1 fast path.
    const int verify_width    = common_moe_verify_width(params);
    const int effective_batch = params.n_parallel * verify_width;
    if (effective_batch <= 1) {
        return;
    }
    const int max_batch = common_moe_min_mmvq_max_batch();
    if (max_batch > 0 && effective_batch > max_batch) {
        LOG_WRN("%s: --parallel %d%s = effective concurrent batch %d exceeds this GPU's MoE "
                "routing fast-path batch limit (<=%d for at least one common quant type) - "
                "concurrent decode above that limit hits a real GPU-kernel-dispatch throughput "
                "cliff (measured >2x slower per token above the boundary, not a gentle decline). "
                "This is not a moe-cache/placement issue and -ncmoe can't fix it; consider "
                "keeping the effective batch (--parallel%s) at or below %d, or accept the "
                "throughput drop above it.\n",
                __func__, params.n_parallel,
                verify_width > 1 ? string_format(" x MTP verify-width %d", verify_width).c_str() : "",
                effective_batch, max_batch,
                verify_width > 1 ? " x MTP verify-width" : "",
                max_batch);
    }
}

// The shared p_min confidence gate (common_params_speculative_draft::p_min -
// read by draft-simple, EAGLE3, DFlash/DSpark, and MTP; the ngram-family
// types don't use it, they're pattern matchers with no per-token
// confidence) defaults to 0.0f. That default can never actually gate
// anything: every real sampled probability satisfies p >= 0.0, so the
// "only collect very high-confidence draft tokens" early-stop in each of
// those drafters' draft() loops (e.g. `if (cur_p->data[0].p < params.p_min)`)
// is dead code at the default. At n_max's own default of 3 this is bounded
// (at most a couple of low-confidence forward passes wasted, already
// measured this session as immaterial for MTP specifically) - which is
// presumably why it went unnoticed here. But raising n_max on any
// p_min-honoring drafter without also raising p_min removes the only
// mechanism that stops a drafter once its confidence has collapsed, and it
// then burns compute drafting every remaining slot up to n_max regardless,
// tokens verification is going to reject anyway. Reported upstream with
// measured costs at exactly this combination: 2.6x slower at n_max=16,
// 3.7x at n_max=48, 8.6x at n_max=64, for the same output
// (ggml-org/llama.cpp#25908, #26100). Purely diagnostic, mirrors
// common_warn_concurrency_cliff just below - never changes behavior or
// silently picks a value on the operator's behalf.
static void common_warn_p_min_disabled(const common_params & params) {
    if (!params.speculative.has_dft()) {
        return; // no p_min-honoring drafter configured
    }
    if (params.speculative.draft.p_min > 0.0f) {
        return; // gate is live - an explicit choice already made here
    }
    if (params.speculative.draft.n_max <= 3) {
        return; // at or below the validated default width; cost is bounded
    }
    LOG_WRN("%s: --spec-draft-n-max %d with --spec-draft-p-min left at its default (0.0) - "
            "the confidence early-stop can never fire at 0.0 (every real probability is >= 0.0), "
            "so the draft model will draft all %d tokens every round even once its own confidence "
            "has collapsed, burning compute on tokens verification is going to reject anyway. "
            "Measured cost elsewhere at this kind of width: 2.6x-8.6x slower wall time for the same "
            "output (see ggml-org/llama.cpp#25908). Consider setting --spec-draft-p-min explicitly "
            "(e.g. 0.5-0.75) now that n_max is above its default of 3.\n",
            __func__, params.speculative.draft.n_max, params.speculative.draft.n_max);
}

bool common_moe_cache_get_expert_map(std::vector<uint8_t> & out_bytes, int & out_rows, int & out_cols) {
    out_rows = 0;
    out_cols = 0;
    if (!ggml_moe_cache.get_expert_map) {
        return false; // no CUDA backend registered, or built without it
    }
    // First call with a zero-capacity probe to learn the real shape (the
    // provider reports rows/cols even on a "buffer too small" 0 return),
    // then size the real buffer and fetch for real - avoids guessing a
    // grid size up front for a model we haven't measured yet.
    int rows = 0, cols = 0;
    ggml_moe_cache.get_expert_map(nullptr, 0, &rows, &cols);
    if (rows <= 0 || cols <= 0) {
        return false; // nothing cached yet
    }
    out_bytes.assign((size_t) rows * (size_t) cols, 0);
    if (!ggml_moe_cache.get_expert_map(out_bytes.data(), (int) out_bytes.size(), &rows, &cols)) {
        out_bytes.clear();
        return false;
    }
    out_rows = rows;
    out_cols = cols;
    return true;
}

bool common_moe_cache_get_substitute_map(std::vector<uint8_t> & out_bits, int & out_rows, int & out_cols) {
    out_rows = 0;
    out_cols = 0;
    if (!ggml_moe_cache.get_substitute_map) {
        return false;
    }
    int rows = 0, cols = 0;
    ggml_moe_cache.get_substitute_map(nullptr, 0, &rows, &cols);
    if (rows <= 0 || cols <= 0) {
        return false;
    }
    const size_t need = ((size_t) rows * (size_t) cols + 7) / 8;
    out_bits.assign(need, 0);
    if (!ggml_moe_cache.get_substitute_map(out_bits.data(), (int) out_bits.size(), &rows, &cols)) {
        out_bits.clear();
        return false;
    }
    out_rows = rows;
    out_cols = cols;
    return true;
}

bool common_moe_cache_get_neuron_concentration_map(std::vector<float> & out_values, int & out_rows, int & out_cols) {
    out_rows = 0;
    out_cols = 0;
    if (!ggml_moe_cache.get_neuron_concentration_map) {
        return false;
    }
    int rows = 0, cols = 0;
    ggml_moe_cache.get_neuron_concentration_map(nullptr, 0, &rows, &cols);
    if (rows <= 0 || cols <= 0) {
        return false;
    }
    const size_t need = (size_t) rows * (size_t) cols;
    out_values.assign(need, -1.0f);
    if (!ggml_moe_cache.get_neuron_concentration_map(out_values.data(), (int) out_values.size(), &rows, &cols)) {
        out_values.clear();
        return false;
    }
    out_rows = rows;
    out_cols = cols;
    return true;
}

bool common_moe_cache_get_summary(common_moe_cache_summary & out) {
    out = common_moe_cache_summary{};
    if (!ggml_moe_cache.get_summary) {
        return false; // no CUDA backend registered, or built without it
    }
    ggml_moe_cache_summary raw{};
    ggml_moe_cache.get_summary(&raw);
    out.hits             = raw.hits;
    out.misses           = raw.misses;
    out.evictions        = raw.evictions;
    out.fill_failures    = raw.fill_failures;
    out.admission_skips  = raw.admission_skips;
    out.prefetches       = raw.prefetches;
    out.slots_used       = raw.slots_used;
    out.slots_total      = raw.slots_total;
    out.protected_slots  = raw.protected_slots;
    out.avg_heat         = raw.avg_heat;
    out.allocated_bytes  = raw.allocated_bytes;
    out.budget_bytes     = raw.budget_bytes;
    out.substitutions       = raw.substitutions;
    out.substitute_declined = raw.substitute_declined;
    out.rank_hits.assign(raw.rank_hits, raw.rank_hits + GGML_MOE_CACHE_MAX_RANK);
    out.rank_misses.assign(raw.rank_misses, raw.rank_misses + GGML_MOE_CACHE_MAX_RANK);
    out.req_dir_x        = raw.req_dir_x;
    out.req_dir_y        = raw.req_dir_y;
    out.req_dir_valid    = raw.req_dir_valid != 0;
    return raw.slots_total > 0;
}

// Registers one MoE tensor's measured topic-affinity cells (from
// --expert-atlas-file) with moe-cache's step-0 request-direction tracking.
// host_base identifies the tensor the same way moe_cache_key already does;
// the caller (server_context::init, which has both the loaded model and the
// parsed atlas file) is responsible for resolving that pointer and calling
// this once per MoE tensor found for each atlas-covered layer.
bool common_moe_cache_set_atlas(
        const void * host_base, const std::vector<int32_t> & expert,
        const std::vector<float> & x, const std::vector<float> & y, const std::vector<float> & spec,
        const std::vector<float> & dims, int n_dims) {
    if (!ggml_moe_cache.set_atlas || !host_base || expert.empty()) {
        return false;
    }
    if (expert.size() != x.size() || expert.size() != y.size() || expert.size() != spec.size()) {
        return false;
    }
    // Only pass the embedding through if it is exactly the expected shape -
    // a short or ragged array would be read past its end per cell. A mismatch
    // drops to the 2D path rather than failing the registration, since the
    // 2D pair is still valid data.
    const bool dims_ok = n_dims > 0 && dims.size() == expert.size() * (size_t) n_dims;
    ggml_moe_cache.set_atlas(host_base, expert.data(), x.data(), y.data(), spec.data(),
            dims_ok ? dims.data() : nullptr, dims_ok ? n_dims : 0, (int) expert.size());
    return true;
}

std::vector<common_moe_cache_co_activation_entry> common_moe_cache_get_co_activation(
        bool cross_layer, int max_entries) {
    std::vector<common_moe_cache_co_activation_entry> out;
    if (!ggml_moe_cache.get_co_activation || max_entries <= 0) {
        return out;
    }
    std::vector<ggml_moe_cache_co_activation_entry> raw(max_entries);
    const int n = ggml_moe_cache.get_co_activation(cross_layer ? 1 : 0, raw.data(), max_entries);
    out.reserve((size_t) std::max(n, 0));
    for (int i = 0; i < n; i++) {
        out.push_back({raw[i].tensor_from, raw[i].expert_from,
                        raw[i].tensor_to,   raw[i].expert_to, raw[i].count});
    }
    return out;
}

// moe-cache's own admission gate (GGML_CUDA_MOE_CACHE_MAX_BATCH) defaults
// from a live hint set by llama_context (real n_seq_max, floored at 8,
// ceilinged at 64 - see moe-cache.cu's MOE_CACHE_MAX_BATCH_CEILING) - but
// that hint is n_seq_max alone, with no MTP verify-width multiplier applied
// (checked directly at the call site, llama-context.cpp's reserve()).
// Under MTP, the real per-step batch moe-cache actually sees is
// n_seq_max * verify_width, so without this fix the cache gate stays sized
// for the no-MTP case and silently stops helping (falls back to
// uncached CPU compute, not a crash) for any MTP-active deployment whose
// effective batch exceeds whatever n_seq_max alone would have set.
//
// Fixed without touching llama_context's public API: this file already
// knows params.speculative.draft.n_max (the context layer doesn't), so it
// sets GGML_CUDA_MOE_CACHE_MAX_BATCH explicitly when MTP is active,
// reusing the existing "explicit env var always wins over the hint"
// precedence in moe_cache_read_config() rather than adding a new one.
// Never overrides an operator-set value - only steps in for the specific
// case the live hint under-serves.
static void common_moe_apply_mtp_aware_max_batch_hint(const common_params & params) {
    if (getenv("GGML_CUDA_MOE_CACHE_MAX_BATCH")) {
        return; // explicit operator choice - never override it
    }
    const int verify_width = common_moe_verify_width(params);
    if (verify_width <= 1) {
        return; // no MTP - the plain n_seq_max hint already covers this correctly
    }
    // MOE_CACHE_MAX_BATCH_CEILING in moe-cache.cu - duplicated here since
    // that constant is internal to the CUDA backend, not exposed publicly;
    // both need to move together if the real ceiling ever changes.
    constexpr int moe_cache_max_batch_ceiling = 64;
    const int effective = std::max(1, std::min(moe_cache_max_batch_ceiling, params.n_parallel * verify_width));
    setenv("GGML_CUDA_MOE_CACHE_MAX_BATCH", std::to_string(effective).c_str(), 1);
    LOG_INF("%s: MTP active (spec-draft-n-max=%d) - set GGML_CUDA_MOE_CACHE_MAX_BATCH=%d "
            "(n_parallel=%d x verify-width=%d) so moe-cache's own admission gate matches the "
            "real effective batch size, not just n_seq_max alone\n",
            __func__, params.speculative.draft.n_max, effective, params.n_parallel, verify_width);
}

static bool common_maybe_autoplace_moe_cpu(
        const char * path_model, common_params & params,
        llama_model_params & mparams, const llama_context_params & cparams) {
    for (const auto & o : params.tensor_buft_overrides) {
        if (o.pattern != nullptr) {
            return false;
        }
    }

    common_moe_calibration_entry cached;
    if (common_moe_calibration_lookup(path_model, params, cached)) {
        std::vector<ggml_backend_dev_t> devs;
        uint32_t hp_ngl = 0, hp_n_ctx_train = 0, hp_n_expert = 0;
        // Placement and VRAM sizing are only meaningful at the -ngl they were
        // measured under - how many layers sit on the GPU is precisely what
        // decides how much room is left for experts and cache. The rest of the
        // entry is not layer-residency dependent: thread count is a property
        // of this CPU, the substitution floor is a router-quality boundary
        // (see the ladder's own comment), and neuron-reduce is a magnitude
        // threshold on expert weights. Those stay valid, so a relaxed match
        // applies them rather than dropping the whole entry on the floor and
        // silently reverting every knob to its default.
        const bool use_placement = cached.ngl_exact &&
            common_moe_fits_with_n(path_model, mparams, cparams, (uint32_t) cached.n_cpu_moe,
                                    devs, hp_ngl, hp_n_ctx_train, hp_n_expert);
        if (use_placement || !cached.ngl_exact) {
            if (use_placement) {
                params.tensor_buft_overrides = common_moe_build_cpu_overrides((uint32_t) cached.n_cpu_moe);
                mparams.tensor_buft_overrides = params.tensor_buft_overrides.data();
            }
            if (cached.n_threads > 0) {
                params.cpuparams.n_threads = cached.n_threads;
            }
            if (cached.n_threads_batch > 0) {
                params.cpuparams_batch.n_threads = cached.n_threads_batch;
            }
            // Only apply the calibrated n_max if the user is already running
            // with a draft model configured and didn't pin n_max themselves
            // - this never turns MTP on for someone who didn't ask for it,
            // it only tunes n_max for someone who already did.
            if (cached.spec_n_max > 0 && params.speculative.has_dft() &&
                params.speculative.draft.n_max == 3 /* default, see common_params_speculative_draft */) {
                params.speculative.draft.n_max = cached.spec_n_max;
            }
            // Same rule as spec_n_max above: only refine --moe-cache's own
            // "auto" intent, never override a size the user explicitly
            // requested. GGML_CUDA_MOE_CACHE_MODE=auto is the marker for
            // that - "off" or an explicit numeric budget means the user
            // already decided and calibration should not second-guess it.
            const char * cache_mode_env = getenv("GGML_CUDA_MOE_CACHE_MODE");
            const bool cache_mode_is_auto = !cache_mode_env || std::string(cache_mode_env) == "auto";
            if (cached.moe_cache_mb > 0 && cache_mode_is_auto && use_placement) {
#if defined(_WIN32)
                _putenv_s("GGML_CUDA_MOE_CACHE", "1");
                _putenv_s("GGML_CUDA_MOE_CACHE_MODE", "on");
                _putenv_s("GGML_CUDA_MOE_CACHE_BUDGET_MB", std::to_string(cached.moe_cache_mb).c_str());
#else
                setenv("GGML_CUDA_MOE_CACHE", "1", 1);
                setenv("GGML_CUDA_MOE_CACHE_MODE", "on", 1);
                setenv("GGML_CUDA_MOE_CACHE_BUDGET_MB", std::to_string(cached.moe_cache_mb).c_str(), 1);
#endif
                params.moe_cache_force = true;
            }
            // Same rule again: only apply calibrated neuron-reduce settings
            // when the user hasn't already set the env var themselves - an
            // explicit GGML_CUDA_MOE_CACHE_NEURON_REDUCE (on or off) is a
            // deliberate choice calibration should not second-guess. This
            // was a real, confirmed gap until now: calibration has recorded
            // these fields since neuron_reduce_k/_budget_mb were added to
            // common_moe_calibration_entry, but nothing ever read them back
            // on a normal launch - an entry with real measured values (e.g.
            // gemma-4's k=256, budget_mb=256) sat unused every time.
            // Same rule again: a calibrated substitution floor is applied only
            // when the user hasn't pinned one themselves. Measured per model
            // because the safe point genuinely differs - it is the boundary
            // between "fast" and "generates word salad", and the degeneracy
            // guard in calibration is what makes recording it trustworthy.
            const bool subst_rank_is_default = !getenv("GGML_CUDA_MOE_CACHE_SUBSTITUTE_MIN_RANK");
            const bool applied_subst_rank = cached.substitute_min_rank >= 0 && subst_rank_is_default;
            if (applied_subst_rank) {
#if defined(_WIN32)
                _putenv_s("GGML_CUDA_MOE_CACHE_SUBSTITUTE_MIN_RANK",
                        std::to_string(cached.substitute_min_rank).c_str());
#else
                setenv("GGML_CUDA_MOE_CACHE_SUBSTITUTE_MIN_RANK",
                        std::to_string(cached.substitute_min_rank).c_str(), 1);
#endif
            }
            const bool neuron_reduce_is_default = !getenv("GGML_CUDA_MOE_CACHE_NEURON_REDUCE");
            const bool applied_neuron_reduce =
                    cached.neuron_reduce_k > 0 && cached.neuron_reduce_budget_mb > 0 && neuron_reduce_is_default;
            if (applied_neuron_reduce) {
#if defined(_WIN32)
                _putenv_s("GGML_CUDA_MOE_CACHE_NEURON_REDUCE", "1");
                _putenv_s("GGML_CUDA_MOE_CACHE_NEURON_REDUCE_K", std::to_string(cached.neuron_reduce_k).c_str());
                _putenv_s("GGML_CUDA_MOE_CACHE_NEURON_REDUCE_BUDGET_MB",
                        std::to_string(cached.neuron_reduce_budget_mb).c_str());
#else
                setenv("GGML_CUDA_MOE_CACHE_NEURON_REDUCE", "1", 1);
                setenv("GGML_CUDA_MOE_CACHE_NEURON_REDUCE_K", std::to_string(cached.neuron_reduce_k).c_str(), 1);
                setenv("GGML_CUDA_MOE_CACHE_NEURON_REDUCE_BUDGET_MB",
                        std::to_string(cached.neuron_reduce_budget_mb).c_str(), 1);
#endif
            }
            // Same rule again: only apply the calibrated -ngl when the user
            // left it at the default (-1, "auto"/all) - an explicit -ngl is a
            // deliberate choice calibration should never silently override.
            // Safe to apply after the fits-check above without re-verifying:
            // fewer GPU-resident layers only ever *reduces* VRAM demand, so a
            // placement that already fit at the (higher) default -ngl still
            // fits at the calibrated (lower) one.
            const bool applied_ngl = use_placement && cached.n_gpu_layers >= 0 && params.n_gpu_layers == -1;
            if (applied_ngl) {
                params.n_gpu_layers  = cached.n_gpu_layers;
                mparams.n_gpu_layers = cached.n_gpu_layers;
            }
            if (!use_placement) {
                // Relaxed match: the machine-level knobs above are applied,
                // but placement was measured at a different -ngl and does not
                // transfer. Fall through to the live probe to decide it -
                // returning here would leave the model unplaced.
                LOG_WRN("%s: -ngl was pinned to %d, but this machine/model was calibrated at a different "
                        "-ngl; applied the values that do not depend on it (n_threads=%d%s%s%s) and "
                        "placing the experts live - run --moe-calibrate with this -ngl for a measured placement\n",
                        __func__, params.n_gpu_layers, cached.n_threads,
                        cached.spec_n_max > 0 ? string_format(", spec-draft-n-max=%d", cached.spec_n_max).c_str() : "",
                        applied_neuron_reduce ? string_format(", neuron-reduce=k%d/%dMiB", cached.neuron_reduce_k, cached.neuron_reduce_budget_mb).c_str() : "",
                        applied_subst_rank ? string_format(", substitute-min-rank=%d", cached.substitute_min_rank).c_str() : "");
            } else {
            LOG_WRN("%s: using calibrated MoE placement from cache (ncmoe=%d, n_threads=%d%s%s%s%s%s%s, measured %.2f %s on %s) "
                    "- run --moe-calibrate again if hardware/model/context changed\n",
                    __func__, cached.n_cpu_moe, cached.n_threads,
                    cached.spec_n_max > 0 ? string_format(", spec-draft-n-max=%d", cached.spec_n_max).c_str() : "",
                    cached.moe_cache_mb > 0 && cache_mode_is_auto ? string_format(", moe-cache=%dMiB", cached.moe_cache_mb).c_str() : "",
                    applied_ngl ? string_format(", ngl=%d", cached.n_gpu_layers).c_str() : "",
                    applied_neuron_reduce ? string_format(", neuron-reduce=k%d/%dMiB", cached.neuron_reduce_k, cached.neuron_reduce_budget_mb).c_str() : "",
                    applied_subst_rank ? string_format(", substitute-min-rank=%d", cached.substitute_min_rank).c_str() : "",
                    cached.concurrency > 1 ? string_format(", concurrency=%d", cached.concurrency).c_str() : "",
                    cached.tok_per_sec, cached.concurrency > 1 ? "aggregate tok/s" : "tok/s", cached.calibrated_at.c_str());
            return true;
            }
        } else {
            LOG_WRN("%s: cached calibration no longer fits current conditions - recalibrating placement live\n", __func__);
        }
    }

    common_moe_fit_probe_result probe = common_moe_find_safe_layers(path_model, mparams, cparams);
    if (probe.already_fits || !probe.is_moe) {
        return false;
    }
    if (!probe.found_safe_n) {
        LOG_WRN("%s: could not find any MoE CPU-offload placement that fits in available device memory; "
                "proceeding with the original configuration, load will likely fail\n", __func__);
        return false;
    }

    LOG_WRN("%s: model does not fit in available device memory as configured; "
            "auto-placed the MoE experts of the first %u layers on CPU to fit "
            "(pass -ncmoe explicitly to override, or run --moe-calibrate once for a throughput-optimal placement)\n",
            __func__, probe.safe_n);
    params.tensor_buft_overrides = common_moe_build_cpu_overrides(probe.safe_n);
    mparams.tensor_buft_overrides = params.tensor_buft_overrides.data();
    return true;
}

// Raise MoE CPU offload until the *requested* context fits, before --fit ever
// considers shrinking that context.
//
// Order of operations is the whole point here. common_fit_params() runs first
// and, finding the model doesn't fit, shrinks n_ctx until it does - at whatever
// MoE placement it was handed. common_maybe_autoplace_moe_cpu() then runs and
// asks only "does the model fit?", which is now trivially true (at the reduced
// context), so it does nothing. Each step locally succeeds while together they
// silently trade away context the user explicitly asked for, to preserve a
// placement nobody deliberately chose. Measured on a 12GB card with a 26B MoE:
// `-ncmoe 15 -c 65536` collapsed to 4096 ctx / 1 slot, while `-ncmoe 30` (the
// same model, more experts in CPU RAM) served the full 65536 at 4 slots and
// ~90% of the throughput - the context was never the thing that had to give.
//
// So placement is chosen here, against the requested context, and only if no
// placement can host it does --fit fall back to reducing context as before
// (a genuine hardware limit rather than a self-inflicted one).
//
// -ncmoe acts as a floor rather than an off-switch: an explicit value is never
// lowered, but it is raised when the requested context demands it. This
// deliberately differs from common_maybe_autoplace_moe_cpu()'s "bail if any
// override is set" rule - that early-out means passing -ncmoe silently disables
// the very logic meant to make this decision (the same trap as -ngl disabling
// --fit entirely), which is exactly how the collapse above went unnoticed.
static bool common_maybe_raise_moe_for_ctx(
        const char * path_model, common_params & params,
        llama_model_params & mparams, const llama_context_params & cparams) {
    // Quality knobs first, and independently of whether the placement below is
    // usable. Neither depends on layer residency: the substitution floor is a
    // router-quality boundary and neuron-reduce is a magnitude threshold on
    // expert weights, so both survive an -ngl the entry was not measured at.
    // They used to live inside the placement branch, which meant pinning -ngl
    // (or any placement mismatch) silently dropped them - the same class of
    // gap as the one recorded in that branch's own comment, where a measured
    // substitute_min_rank was cached and then never applied at serving.
    {
        common_moe_calibration_entry cal_q;
        if (common_moe_calibration_lookup(path_model, params, cal_q)) {
            common_moe_apply_quality_knobs(cal_q);
        }
    }

    // Only meaningful for an explicit request. n_ctx == 0 ("auto") means the
    // user expressed no preference, so there is no context to protect and the
    // existing post-fit autoplace path already handles it.
    //
    // The quality knobs above are deliberately applied BEFORE this returns.
    // They do not depend on the context at all - the substitution floor is a
    // router-quality boundary and neuron-reduce is a magnitude threshold - and
    // leaving them behind this guard meant a launch with no -c at all silently
    // discarded every one of them. That is exactly the launch this project
    // tells people to use: measured on a bare `llama-server -m model.gguf`,
    // substitutions were 0 and declined 0, i.e. the gate never even ran, while
    // the same binary with `-c 4096` applied the floor and used it. The
    // fewer flags you passed, the more of the calibration was thrown away.
    if (params.n_ctx == 0) {
        return false;
    }

    // Any non-MoE-block override pattern means the user hand-placed tensors
    // with -ot; rebuilding the override list from a layer count would discard
    // that. Only the uniform "first N layers" shape -ncmoe produces is safe to
    // extend, so bail on anything else.
    uint32_t current_n = 0;
    for (const auto & o : params.tensor_buft_overrides) {
        if (o.pattern == nullptr) {
            continue;
        }
        if (o.pattern != llm_ffn_exps_block_regex((int) current_n)) {
            return false;
        }
        current_n++;
    }

    // Must match what common_fit_params() will demand of this same config, or
    // placement approves a layout fit then rejects - see fit.cpp's step0_margin,
    // which requires 3x the configured per-device target to cover costs its
    // no-alloc probe can't see (real weight loading, lazy CUDA graph capture).
    // Judging placement by a bare fit here is what let a 65536-token request
    // get "fixed" by a single extra offloaded layer and still collapse to 4096.
    // A calibrated fit margin, if one was measured for this exact
    // GPU+model+context, replaces the default before the margin is computed
    // - it has to be applied HERE rather than alongside the placement
    // decision below, because this margin is precisely what determines how
    // far -ncmoe gets raised. Applying it afterwards would compute the
    // placement against the untuned default and then record a margin that
    // never influenced anything. Only ever tightens: a calibrated value
    // above the default is ignored, on the same "calibration is a floor,
    // never a licence to undershoot safety" principle the placement lookup
    // below already follows.
    {
        common_moe_calibration_entry cal_fit;
        if (common_moe_calibration_lookup(path_model, params, cal_fit) && cal_fit.fit_target_mb > 0 &&
            cal_fit.ngl_exact) {
            const size_t want = (size_t) cal_fit.fit_target_mb * 1024 * 1024;
            if (want < params.fit_params_target[0]) {
                LOG_WRN("%s: using calibrated fit margin of %d MiB per device (default %zu MiB) - "
                        "measured %.2f tok/s on %s\n", __func__, cal_fit.fit_target_mb,
                        params.fit_params_target[0] / (1024 * 1024), cal_fit.tok_per_sec,
                        cal_fit.calibrated_at.c_str());
                std::fill(params.fit_params_target.begin(), params.fit_params_target.end(), want);
            }
        }
    }
    const int64_t margin = 3 * (int64_t) params.fit_params_target[0];

    // Prefer a calibrated placement when one exists. The search below answers
    // "what is the least offload that makes this context fit?", which is a
    // sufficiency question, not a throughput one - and the two have different
    // answers. Every layer left resident holds weights competing with the expert
    // cache for the same VRAM, and cached experts serve the offloaded layers, so
    // trading resident weights for cache capacity keeps paying well past the
    // point where the context merely fits. Measured on Nemotron 3.5 (53 layers):
    // the fit search stopped at 46 for 1239 cache slots and a 57.1% hit rate,
    // while offloading all 53 gave 2307 slots and 68.3%.
    //
    // --moe-calibrate already measures the throughput optimum empirically. This
    // only consults what it recorded; it does not guess past "fits" on its own,
    // because the right number is hardware- and model-specific and the honest
    // way to find it is to measure it.
    {
        common_moe_calibration_entry cal;
        if (common_moe_calibration_lookup(path_model, params, cal) && cal.n_cpu_moe > 0 && cal.ngl_exact) {
            std::vector<ggml_backend_dev_t> cdevs;
            uint32_t c_ngl = 0, c_nct = 0, c_nex = 0;
            // Keep the full fit margin here. Dropping it to 0 -- on the
            // reasoning that a calibrated entry is a measurement rather than a
            // prediction, so it only needs to still fit -- was tried and
            // reproduced the 4k collapse outright: ncmoe=40 passed a zero-margin
            // probe, then real weight loading plus lazy CUDA graph capture
            // overran VRAM and the downstream search clamped the server to
            // n_ctx=4096 with a single slot. The margin covers allocation that
            // happens *after* the probe, so a measurement taken elsewhere does
            // not excuse skipping it.
            //
            // That makes calibration a floor, never a licence to undershoot: it
            // can only move placement to be *more* CPU-offloaded than the safe
            // minimum (which is where it pays -- freed VRAM turns into expert
            // cache hit rate). When it recommends less, the fit search's more
            // conservative answer wins.
            if (common_moe_fits_with_n(path_model, mparams, cparams, (uint32_t) cal.n_cpu_moe,
                                       cdevs, c_ngl, c_nct, c_nex, margin)) {
                LOG_WRN("%s: using calibrated placement of %d CPU layer(s) (measured %.2f tok/s on %s) "
                        "rather than the %s that merely fits\n",
                        __func__, cal.n_cpu_moe, cal.tok_per_sec, cal.calibrated_at.c_str(),
                        "minimum");
                // Same expert-cache-size consultation as common_maybe_autoplace_moe_cpu -
                // this is the placement path that actually runs when the user asks
                // for a context and lets -ncmoe auto-raise to fit it (--moe-cache
                // auto, no explicit -ncmoe), which is most real launches. Only
                // refines --moe-cache's own "auto" intent, same guard as there.
                {
                    const char * cache_mode_env = getenv("GGML_CUDA_MOE_CACHE_MODE");
                    const bool cache_mode_is_auto = !cache_mode_env || std::string(cache_mode_env) == "auto";
                    if (cal.moe_cache_mb > 0 && cache_mode_is_auto) {
#if defined(_WIN32)
                        _putenv_s("GGML_CUDA_MOE_CACHE", "1");
                        _putenv_s("GGML_CUDA_MOE_CACHE_MODE", "on");
                        _putenv_s("GGML_CUDA_MOE_CACHE_BUDGET_MB", std::to_string(cal.moe_cache_mb).c_str());
#else
                        setenv("GGML_CUDA_MOE_CACHE", "1", 1);
                        setenv("GGML_CUDA_MOE_CACHE_MODE", "on", 1);
                        setenv("GGML_CUDA_MOE_CACHE_BUDGET_MB", std::to_string(cal.moe_cache_mb).c_str(), 1);
#endif
                        params.moe_cache_force = true;
                        LOG_WRN("%s: using calibrated expert-cache size of %d MiB\n", __func__, cal.moe_cache_mb);
                    }
                }
                // Substitution floor and neuron-reduce, for the same reason the
                // cache size is applied here: this is the path most real
                // launches take (context auto-raises -ncmoe, no explicit flag),
                // and it returns before common_maybe_autoplace_moe_cpu ever
                // runs - so settings applied only there were measured, cached,
                // and then silently ignored at serving. Caught live: a run
                // measured substitute_min_rank=2 at 1.64 tok/s, cached it, and
                // then served 0.66 tok/s with substitutions=0 because this
                // branch never applied it. Same "only fill in what the user
                // left at default" rule as everywhere else.
                params.tensor_buft_overrides  = common_moe_build_cpu_overrides((uint32_t) cal.n_cpu_moe);
                mparams.tensor_buft_overrides = params.tensor_buft_overrides.data();
                // c_ngl was filled by the fit probe above with the model's real
                // layer count - report against that, not the override count.
                if (c_ngl > 0) {
                    params.placed_n_layer = (int32_t) c_ngl;
                }
                const int32_t n_layer_cal = c_ngl > 0 ? (int32_t) c_ngl : cal.n_cpu_moe;
                params.placed_n_cpu_moe_req   = std::min((int32_t) current_n, n_layer_cal);
                params.placed_n_cpu_moe_final = std::min(cal.n_cpu_moe, n_layer_cal);
                params.placed_n_ctx_req       = params.n_ctx;
                return true;
            }
            LOG_WRN("%s: calibrated placement of %d CPU layer(s) is below the safe minimum for this "
                    "context - keeping the fit search's more conservative answer\n",
                    __func__, cal.n_cpu_moe);
        }
    }

    const common_moe_fit_probe_result probe =
        common_moe_find_safe_layers(path_model, mparams, cparams, margin);

    // Record what was asked for regardless of the outcome below, so the Brain
    // view can distinguish "23 layers because you asked for it" from "23 layers
    // because 65536 context demanded it".
    // Clamp what gets reported to the layers that actually exist. `-ncmoe 99`
    // on a 53-layer model is a legitimate "offload everything" idiom - the extra
    // patterns simply match nothing - but reporting 99 CPU layers to /props (and
    // from there to the model-info dialog and the Brain placement view) states
    // something untrue about the model.
    const int32_t n_layer_real = probe.n_layer > 0 ? (int32_t) probe.n_layer : (int32_t) current_n;
    params.placed_n_cpu_moe_req   = std::min((int32_t) current_n, n_layer_real);
    params.placed_n_cpu_moe_final = std::min((int32_t) current_n, n_layer_real);
    params.placed_n_ctx_req       = params.n_ctx;
    if (probe.n_layer > 0) {
        params.placed_n_layer = (int32_t) probe.n_layer;
    }

    if (!probe.is_moe || probe.already_fits) {
        return false; // dense model, or the requested context already fits as configured
    }
    if (!probe.found_safe_n || probe.safe_n <= current_n) {
        // Nothing fits even with every expert offloaded, or more offload than
        // already configured wouldn't help - a real hardware ceiling. Leave it
        // to --fit to reduce context, which is the correct last resort.
        return false;
    }

    LOG_WRN("%s: requested context of %d does not fit with the MoE experts of %u layer(s) on CPU - "
            "raised to %u layer(s) so the requested context fits, instead of reducing the context to fit "
            "the placement (pass -fit off to disable this)\n",
            __func__, params.n_ctx, current_n, probe.safe_n);

    params.tensor_buft_overrides  = common_moe_build_cpu_overrides(probe.safe_n);
    mparams.tensor_buft_overrides = params.tensor_buft_overrides.data();
    params.placed_n_cpu_moe_final = std::min((int32_t) probe.safe_n, n_layer_real);
    return true;
}

common_init_result::common_init_result(common_params & params, bool model_only) :
    pimpl(new impl{}) {
    auto mparams = common_model_params_to_llama(params);
    auto cparams = common_context_params_to_llama(params);

    if (params.fit_params) {
        // must run before common_fit_params() - see the function's own comment
        common_maybe_raise_moe_for_ctx(params.model.path.c_str(), params, mparams, cparams);

        COM_TRC("%s", "fitting params to device memory ...\n");
        COM_TRC("%s", "(for bugs during this step try to reproduce them with -fit off, or provide --verbose logs if the bug only occurs with -fit on)\n");
        common_fit_params(params.model.path.c_str(), &mparams, &cparams,
            params.tensor_split,
            params.tensor_buft_overrides.data(),
            params.fit_params_target.data(),
            params.fit_params_min_ctx,
            params.verbosity >= LOG_LEVEL_DEBUG ? GGML_LOG_LEVEL_DEBUG : GGML_LOG_LEVEL_ERROR);
    }

    common_maybe_autoplace_moe_cpu(params.model.path.c_str(), params, mparams, cparams);

    // mparams.n_gpu_layers is otherwise a local copy that gets discarded once the real model/context
    // are built below - write the resolved value back so callers (e.g. the server's /props endpoint)
    // can report what --fit actually decided, not just the -1/"auto" the user passed in. Mirrors the
    // same write-back already done for tensor_buft_overrides above.
    params.n_gpu_layers = mparams.n_gpu_layers;

    // cparams.n_seq_max can also be reduced by --fit (see the context-vs-concurrency priority logic
    // in fit.cpp: the requested context size is kept fixed and concurrent slots give way to it, not
    // the other way around). Writing it back to params.n_parallel is not just for reporting purposes
    // this time - the server creates one slot object per params.n_parallel (server-context.cpp) after
    // this constructor returns, independent of the llama_context that was actually built. Without this
    // write-back the server would create more slots than the context's real n_seq_max supports, a real
    // mismatch under concurrent requests, not just a stale number in a dialog.
    if (cparams.n_seq_max != (uint32_t) params.n_parallel && params.n_parallel > 0) {
        LOG_WRN("%s: --fit reduced concurrent slots from %d to %u to keep the requested context size\n",
                __func__, params.n_parallel, cparams.n_seq_max);
    }
    params.n_parallel = (int32_t) cparams.n_seq_max;

    // Both run after autoplace, not before: the calibration-cache lookup
    // inside it can override params.speculative.draft.n_max to the real
    // calibrated value, and both of these need to see that final value, not
    // the pre-lookup default - otherwise an MTP setup with a cached n_max
    // would be checked/sized against the wrong (understated) effective
    // batch size.
    common_warn_concurrency_cliff(params);
    common_enforce_moe_cache_parallel_limit(params, cparams);
    common_moe_apply_mtp_aware_max_batch_hint(params);
    common_warn_p_min_disabled(params);

    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), mparams);
    if (model == NULL) {
        return;
    }

    pimpl->model.reset(model);

    if (model_only) {
        return;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    // load and optionally apply lora adapters
    for (auto & la : params.lora_adapters) {
        llama_adapter_lora_ptr lora;
        lora.reset(llama_adapter_lora_init(model, la.path.c_str()));
        if (lora == nullptr) {
            COM_ERR("failed to load lora adapter '%s'\n", la.path.c_str());
            return;
        }

        char buf[1024];
        la.ptr = lora.get();
        llama_adapter_meta_val_str(la.ptr, "adapter.lora.task_name", buf, sizeof(buf));
        la.task_name = buf;
        llama_adapter_meta_val_str(la.ptr, "adapter.lora.prompt_prefix", buf, sizeof(buf));
        la.prompt_prefix = buf;
        pimpl->lora.emplace_back(std::move(lora)); // copy to list of loaded adapters
    }

    // updates params.sampling
    // TODO: fix naming
    common_init_sampler_from_model(model, params.sampling);

    if (params.sampling.ignore_eos && llama_vocab_eos(vocab) == LLAMA_TOKEN_NULL) {
        COM_WRN("%s", "vocab does not have an EOS token, ignoring --ignore-eos\n");
        params.sampling.ignore_eos = false;
    }

    // initialize once
    for (llama_token i = 0; i < llama_vocab_n_tokens(vocab); i++) {
        if (llama_vocab_is_eog(vocab, i)) {
            COM_TRC("added %s logit bias = %f\n", common_token_to_piece(vocab, i).c_str(), -INFINITY);
            params.sampling.logit_bias_eog.push_back({i, -INFINITY});
        }
    }

    if (params.sampling.ignore_eos) {
        // add EOG biases to the active set of logit biases
        params.sampling.logit_bias.insert(
                params.sampling.logit_bias.end(),
                params.sampling.logit_bias_eog.begin(), params.sampling.logit_bias_eog.end());
    }

    // init the backend samplers as part of the context creation
    pimpl->samplers.resize(cparams.n_seq_max);
    pimpl->samplers_seq_config.resize(cparams.n_seq_max);

    for (int i = 0; i < (int) cparams.n_seq_max; ++i) {
        pimpl->samplers[i].reset(common_sampler_init(model, params.sampling));
        pimpl->samplers_seq_config[i] = { i, common_sampler_get(pimpl->samplers[i].get()) };
    }

    if (params.sampling.backend_sampling) {
        cparams.samplers   = pimpl->samplers_seq_config.data();
        cparams.n_samplers = pimpl->samplers_seq_config.size();
    }

    llama_context * lctx = llama_init_from_model(model, cparams);
    if (lctx == NULL) {
        COM_ERR("failed to create context with model '%s'\n", params.model.path.c_str());
        return;
    }

    pimpl->context.reset(lctx);
}

llama_model * common_init_result::model() {
    return pimpl->model.get();
}

llama_context * common_init_result::context() {
    return pimpl->context.get();
}

common_sampler * common_init_result::sampler(llama_seq_id seq_id) {
    if (seq_id < 0 || seq_id >= (int) pimpl->samplers.size()) {
        return nullptr;
    }
    return pimpl->samplers[seq_id].get();
}

void common_init_result::reset_samplers() {
    for (int i = 0; i < (int) pimpl->samplers.size(); ++i) {
        llama_sampler_reset(common_sampler_get(pimpl->samplers[i].get()));
    }
}

std::vector<llama_adapter_lora_ptr> & common_init_result::lora() {
    return pimpl->lora;
}

common_init_result_ptr common_init_from_params(common_params & params, bool model_only) {
    common_init_result_ptr res(new common_init_result(params, model_only));

    llama_model * model = res->model();
    if (model == NULL) {
        COM_ERR("failed to load model '%s'\n", params.model.path.c_str());
        return res;
    }

    if (model_only) {
        return res;
    }

    llama_context * lctx = res->context();
    if (lctx == NULL) {
        COM_ERR("failed to create context with model '%s'\n", params.model.path.c_str());
        return res;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    if (params.ctx_shift && !llama_memory_can_shift(llama_get_memory(lctx))) {
        COM_WRN("%s", "KV cache shifting is not supported for this context, disabling KV cache shifting\n");
        params.ctx_shift = false;
    }

    if (!params.control_vectors.empty()) {
        if (params.control_vector_layer_start <= 0) params.control_vector_layer_start = 1;
        if (params.control_vector_layer_end   <= 0) params.control_vector_layer_end   = llama_model_n_layer(model);

        const auto cvec = common_control_vector_load(params.control_vectors);
        if (cvec.n_embd == -1) {
            return res;
        }

        int err = llama_set_adapter_cvec(
                lctx,
                cvec.data.data(),
                cvec.data.size(),
                cvec.n_embd,
                params.control_vector_layer_start,
                params.control_vector_layer_end);
        if (err) {
            return res;
        }
    }

    if (llama_pooling_type(lctx) == LLAMA_POOLING_TYPE_RANK) {
        bool ok = true;

        if (llama_vocab_bos(vocab) == LLAMA_TOKEN_NULL) {
            COM_WRN("%s", "vocab does not have a  BOS token, reranking will not work\n");
            ok = false;
        }

        bool has_eos = llama_vocab_eos(vocab) != LLAMA_TOKEN_NULL;
        bool has_sep = llama_vocab_sep(vocab) != LLAMA_TOKEN_NULL;
        bool has_rerank_prompt = llama_model_chat_template(model, "rerank") != NULL;

        if (!has_eos && !has_sep && !has_rerank_prompt) {
            COM_WRN("%s", "vocab does not have an EOS token, SEP token, or rerank prompt. Reranking will not work\n");
            ok = false;
        } else if (!has_eos) {
            COM_WRN("%s", "vocab does not have an EOS token, using SEP token as fallback\n");
        }

        if (!ok) {
            return res;
        }
    }

    if (!params.lora_init_without_apply) {
        common_set_adapter_lora(lctx, params.lora_adapters);
    }

    if (params.warmup) {
        COM_TRC("%s", "warming up the model with an empty run - please wait ... (--no-warmup to disable)\n");

        std::vector<llama_token> tmp;
        llama_token bos = llama_vocab_bos(vocab);
        llama_token eos = llama_vocab_eos(vocab);

        // some models (e.g. T5) don't have a BOS token
        if (bos != LLAMA_TOKEN_NULL) {
            tmp.push_back(bos);
        }
        if (eos != LLAMA_TOKEN_NULL) {
            tmp.push_back(eos);
        }
        if (tmp.empty()) {
            tmp.push_back(0);
        }

        if (llama_model_has_encoder(model)) {
            llama_encode(lctx, llama_batch_get_one(tmp.data(), tmp.size()));
            llama_token decoder_start_token_id = llama_model_decoder_start_token(model);
            if (decoder_start_token_id == LLAMA_TOKEN_NULL) {
                decoder_start_token_id = bos;
            }
            tmp.clear();
            tmp.push_back(decoder_start_token_id);
        }
        if (llama_model_has_decoder(model)) {
            llama_decode(lctx, llama_batch_get_one(tmp.data(), std::min(tmp.size(), (size_t) params.n_batch)));
        }
        llama_memory_clear(llama_get_memory(lctx), true);
        llama_synchronize(lctx);
        llama_perf_context_reset(lctx);

        // reset samplers to reset RNG state after warmup to the seeded state
        res->reset_samplers();
    }

    return res;
}

common_init_result::~common_init_result() = default;

std::string common_get_model_endpoint() {
    std::string endpoint = common_get_env("MODEL_ENDPOINT");
    if (endpoint.empty()) {
        // the HF_ENDPOINT variable is respected for backward compatibility
        endpoint = common_get_env("HF_ENDPOINT");
    }
    if (endpoint.empty()) {
        return "https://huggingface.co/";
    }
    if (endpoint.back() != '/') {
        endpoint += '/';
    }
    return endpoint;
}

char * common_get_model_or_exit(int argc, char * argv[]) {
    if (argc > 1) {
        return argv[1];
    }

    char * path = getenv("LLAMACPP_TEST_MODELFILE");
    if (!path || strlen(path) == 0) {
        fprintf(stderr, "\033[33mWARNING: No model file provided. Skipping this test. Set LLAMACPP_TEST_MODELFILE=<gguf_model_path> to silence this warning and run this test.\n\033[0m");
        exit(EXIT_SUCCESS);
    }

    return path;
}

common_context_seq_rm_type common_context_can_seq_rm(llama_context * ctx) {
    auto * mem = llama_get_memory(ctx);
    if (mem == nullptr) {
        return COMMON_CONTEXT_SEQ_RM_TYPE_NO;
    }

    common_context_seq_rm_type res = COMMON_CONTEXT_SEQ_RM_TYPE_PART;

    llama_memory_clear(mem, true);

    // eval 2 tokens to check if the context is compatible
    std::vector<llama_token> tmp;
    tmp.push_back(0);
    tmp.push_back(0);

    int ret = llama_decode(ctx, llama_batch_get_one(tmp.data(), tmp.size()));
    if (ret != 0) {
        COM_ERR("llama_decode() failed: %d\n", ret);
        res = COMMON_CONTEXT_SEQ_RM_TYPE_NO;
        goto done;
    }

    if (llama_n_rs_seq(ctx) > 0) {
        COM_TRC("%s", "the context supports bounded partial sequence removal\n");
        res = COMMON_CONTEXT_SEQ_RM_TYPE_RS;
        goto done;
    }

    // try to remove the last tokens
    if (!llama_memory_seq_rm(mem, 0, 1, -1)) {
        COM_TRC("%s", "the context does not support partial sequence removal\n");
        res = COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
        goto done;
    }

done:
    llama_memory_clear(mem, true);
    llama_synchronize(ctx);

    return res;
}

static void common_context_seq_rm(llama_context * ctx, llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    auto * mem = llama_get_memory(ctx);
    if (!llama_memory_seq_rm(mem, seq_id, p0, p1)) {
        GGML_ABORT("%s", string_format("failed to remove sequence %d with p0=%d, p1=%d\n", seq_id, p0, p1).c_str());
    }
}

static void common_context_seq_cp(llama_context * ctx, llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    auto * mem = llama_get_memory(ctx);
    llama_memory_seq_cp(mem, seq_id_src, seq_id_dst, p0, p1);
}

static void common_context_seq_add(llama_context * ctx, llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos delta) {
    auto * mem = llama_get_memory(ctx);
    llama_memory_seq_add(mem, seq_id, p0, p1, delta);
}

void common_memory::init(llama_context * ctx_tgt, llama_context * ctx_dft) {
    this->ctx_tgt = ctx_tgt;
    this->ctx_dft = ctx_dft;
}

void common_memory::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) const {
    common_context_seq_rm(ctx_tgt, seq_id, p0, p1);
    if (ctx_dft) {
        common_context_seq_rm(ctx_dft, seq_id, p0, p1);
    }
}

void common_memory::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) const {
    common_context_seq_cp(ctx_tgt, seq_id_src, seq_id_dst, p0, p1);
    if (ctx_dft) {
        common_context_seq_cp(ctx_dft, seq_id_src, seq_id_dst, p0, p1);
    }
}

void common_memory::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos delta) const {
    common_context_seq_add(ctx_tgt, seq_id, p0, p1, delta);
    if (ctx_dft) {
        common_context_seq_add(ctx_dft, seq_id, p0, p1, delta);
    }
}

void common_set_adapter_lora(struct llama_context * ctx, std::vector<common_adapter_lora_info> & lora) {
    std::vector<llama_adapter_lora *> loras;
    std::vector<float> scales;

    for (auto & la: lora) {
        loras.push_back(la.ptr);
        scales.push_back(la.scale);
    }

    llama_set_adapters_lora(ctx, loras.data(), loras.size(), scales.data());
}

struct llama_model_params common_model_params_to_llama(common_params & params) {
    auto mparams = llama_model_default_params();

    if (!params.devices.empty()) {
        mparams.devices = params.devices.data();
    }

    mparams.n_gpu_layers    = params.n_gpu_layers;
    mparams.main_gpu        = params.main_gpu;
    mparams.split_mode      = params.split_mode;
    mparams.load_mode       = params.load_mode;
    mparams.lazy_mode = params.lazy_mode;
    mparams.tensor_split    = params.tensor_split;
    mparams.check_tensors   = params.check_tensors;
    mparams.use_extra_bufts = !params.no_extra_bufts;
    mparams.no_host         = params.no_host;

    if (params.kv_overrides.empty()) {
        mparams.kv_overrides = NULL;
    } else {
        GGML_ASSERT(params.kv_overrides.back().key[0] == 0 && "KV overrides not terminated with empty key");
        mparams.kv_overrides = params.kv_overrides.data();
    }

    if (params.tensor_buft_overrides.empty()) {
        mparams.tensor_buft_overrides = NULL;
    } else {
        GGML_ASSERT(params.tensor_buft_overrides.back().pattern == nullptr && "Tensor buffer overrides not terminated with empty pattern");
        mparams.tensor_buft_overrides = params.tensor_buft_overrides.data();
    }

    mparams.progress_callback           = params.load_progress_callback;
    mparams.progress_callback_user_data = params.load_progress_callback_user_data;
    mparams.no_alloc                    = params.no_alloc;
    mparams.load_mtp                    = std::find(params.speculative.types.begin(), params.speculative.types.end(), COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != params.speculative.types.end();

    return mparams;
}

struct llama_context_params common_context_params_to_llama(const common_params & params) {
    auto cparams = llama_context_default_params();

    cparams.n_ctx             = params.n_ctx;
    cparams.n_seq_max         = params.n_parallel;
    cparams.n_rs_seq          = params.speculative.need_n_rs_seq();
    cparams.n_outputs_max     = std::max(params.n_outputs_max, 0);
    cparams.n_outputs_max_per_seq = std::max(params.n_outputs_max_per_seq, 0);
    cparams.n_batch           = params.n_batch;
    cparams.n_ubatch          = params.n_ubatch;
    cparams.n_threads         = params.cpuparams.n_threads;
    cparams.n_threads_batch   = params.cpuparams_batch.n_threads == -1 ?
                                params.cpuparams.n_threads : params.cpuparams_batch.n_threads;
    cparams.embeddings        = params.embedding;
    cparams.rope_scaling_type = params.rope_scaling_type;
    cparams.rope_freq_base    = params.rope_freq_base;
    cparams.rope_freq_scale   = params.rope_freq_scale;
    cparams.yarn_ext_factor   = params.yarn_ext_factor;
    cparams.yarn_attn_factor  = params.yarn_attn_factor;
    cparams.yarn_beta_fast    = params.yarn_beta_fast;
    cparams.yarn_beta_slow    = params.yarn_beta_slow;
    cparams.yarn_orig_ctx     = params.yarn_orig_ctx;
    cparams.pooling_type      = params.pooling_type;
    cparams.attention_type    = params.attention_type;
    cparams.flash_attn_type   = params.flash_attn_type;
    cparams.cb_eval           = params.cb_eval;
    cparams.cb_eval_user_data = params.cb_eval_user_data;
    cparams.offload_kqv       = !params.no_kv_offload;
    cparams.no_perf           = params.no_perf;
    cparams.op_offload        = !params.no_op_offload;
    cparams.swa_full          = params.swa_full;
    cparams.kv_unified        = params.kv_unified;

    cparams.type_k = params.cache_type_k;
    cparams.type_v = params.cache_type_v;
    cparams.type_r = params.cache_type_r;
    cparams.type_s = params.cache_type_s;

    return cparams;
}

struct ggml_threadpool_params ggml_threadpool_params_from_cpu_params(const common_cpu_params & params) {
    struct ggml_threadpool_params tpp;

    ggml_threadpool_params_init(&tpp, params.n_threads); // setup the defaults

    if (params.mask_valid) {
        std::memcpy(&tpp.cpumask, &params.cpumask, GGML_MAX_N_THREADS);
    }

    tpp.prio       = params.priority;
    tpp.poll       = params.poll;
    tpp.strict_cpu = params.strict_cpu;

    return tpp;
}

//
// Batch utils
//

void common_batch_clear(struct llama_batch & batch) {
    batch.n_tokens = 0;
}

void common_batch_add(
                 struct llama_batch & batch,
                        llama_token   id,
                          llama_pos   pos,
    const std::vector<llama_seq_id> & seq_ids,
                               bool   logits) {
    GGML_ASSERT(batch.seq_id[batch.n_tokens] && "llama_batch size exceeded");

    batch.token   [batch.n_tokens] = id;
    batch.pos     [batch.n_tokens] = pos;
    batch.n_seq_id[batch.n_tokens] = seq_ids.size();
    for (size_t i = 0; i < seq_ids.size(); ++i) {
        batch.seq_id[batch.n_tokens][i] = seq_ids[i];
    }
    batch.logits  [batch.n_tokens] = logits;

    batch.n_tokens++;
}

//
// Vocab utils
//

std::vector<llama_token> common_tokenize(
  const struct llama_context * ctx,
           const std::string & text,
                        bool   add_special,
                        bool   parse_special) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    return common_tokenize(vocab, text, add_special, parse_special);
}

std::vector<llama_token> common_tokenize(
    const struct llama_vocab * vocab,
           const std::string & text,
                        bool   add_special,
                        bool   parse_special) {
    // upper limit for the number of tokens
    int n_tokens = text.length() + 2 * add_special;
    std::vector<llama_token> result(n_tokens);
    n_tokens = llama_tokenize(vocab, text.data(), text.length(), result.data(), result.size(), add_special, parse_special);
    if (n_tokens == std::numeric_limits<int32_t>::min()) {
        throw std::runtime_error("Tokenization failed: input text too large, tokenization result exceeds int32_t limit");
    }
    if (n_tokens < 0) {
        result.resize(-n_tokens);
        int check = llama_tokenize(vocab, text.data(), text.length(), result.data(), result.size(), add_special, parse_special);
        GGML_ASSERT(check == -n_tokens);
    } else {
        result.resize(n_tokens);
    }
    return result;
}

std::string common_token_to_piece(const struct llama_context * ctx, llama_token token, bool special) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    return common_token_to_piece(vocab, token, special);
}

std::string common_token_to_piece(const struct llama_vocab * vocab, llama_token token, bool special) {
    std::string piece;
    piece.resize(piece.capacity());  // using string internal cache, 15 bytes + '\n'
    const int n_chars = llama_token_to_piece(vocab, token, &piece[0], piece.size(), 0, special);
    if (n_chars < 0) {
        piece.resize(-n_chars);
        int check = llama_token_to_piece(vocab, token, &piece[0], piece.size(), 0, special);
        GGML_ASSERT(check == -n_chars);
    }
    else {
        piece.resize(n_chars);
    }

    return piece;
}

std::string common_detokenize(const struct llama_context * ctx, const std::vector<llama_token> & tokens, bool special) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    return common_detokenize(vocab, tokens, special);
}

std::string common_detokenize(const struct llama_vocab * vocab, const std::vector<llama_token> & tokens, bool special) {
    std::string text;
    text.resize(std::max(text.capacity(), tokens.size()));
    int32_t n_chars = llama_detokenize(vocab, tokens.data(), (int32_t)tokens.size(), &text[0], (int32_t)text.size(), false, special);
    if (n_chars < 0) {
        text.resize(-n_chars);
        n_chars = llama_detokenize(vocab, tokens.data(), (int32_t)tokens.size(), &text[0], (int32_t)text.size(), false, special);
        GGML_ASSERT(n_chars <= (int32_t)text.size());  // whitespace trimming is performed after per-token detokenization
    }

    text.resize(n_chars);

    // NOTE: the original tokenizer decodes bytes after collecting the pieces.
    return text;
}

//
// Embedding utils
//

void common_embd_normalize(const float * inp, float * out, int n, int embd_norm) {
    double sum = 0.0;

    switch (embd_norm) {
        case -1: // no normalisation
            sum = 1.0;
            break;
        case 0: // max absolute
            for (int i = 0; i < n; i++) {
                if (sum < std::abs(inp[i])) {
                    sum = std::abs(inp[i]);
                }
            }
            sum /= 32760.0; // make an int16 range
            break;
        case 2: // euclidean
            for (int i = 0; i < n; i++) {
                sum += inp[i] * inp[i];
            }
            sum = std::sqrt(sum);
            break;
        default: // p-norm (euclidean is p-norm p=2)
            for (int i = 0; i < n; i++) {
                sum += std::pow(std::abs(inp[i]), embd_norm);
            }
            sum = std::pow(sum, 1.0 / embd_norm);
            break;
    }

    const float norm = sum > 0.0 ? 1.0 / sum : 0.0f;

    for (int i = 0; i < n; i++) {
        out[i] = inp[i] * norm;
    }
}

float common_embd_similarity_cos(const float * embd1, const float * embd2, int n){
    double sum  = 0.0;
    double sum1 = 0.0;
    double sum2 = 0.0;

    for (int i = 0; i < n; i++) {
        sum  += embd1[i] * embd2[i];
        sum1 += embd1[i] * embd1[i];
        sum2 += embd2[i] * embd2[i];
    }

    // Handle the case where one or both vectors are zero vectors
    if (sum1 == 0.0 || sum2 == 0.0) {
        if (sum1 == 0.0 && sum2 == 0.0) {
            return 1.0f; // two zero vectors are similar
        }
        return 0.0f;
    }

    return sum / (sqrt(sum1) * sqrt(sum2));
}

//
// Control vector utils
//

static common_control_vector_data common_control_vector_load_one(const common_control_vector_load_info & load_info) {
    common_control_vector_data result = { -1, {} };

    ggml_context * ctx = nullptr;
    struct gguf_init_params meta_gguf_params = {
        /* .no_alloc = */ false,
        /* .ctx      = */ &ctx,
    };
    struct gguf_context * ctx_gguf = gguf_init_from_file(load_info.fname.c_str(), meta_gguf_params);
    if (!ctx_gguf) {
        COM_ERR("failed to load control vector file from %s\n", load_info.fname.c_str());
        return result;
    }

    int32_t n_tensors = gguf_get_n_tensors(ctx_gguf);
    if (n_tensors == 0) {
        COM_WRN("no direction tensors found in %s\n", load_info.fname.c_str());
    }

    for (int i = 0; i < n_tensors; i++) {
        std::string name = gguf_get_tensor_name(ctx_gguf, i);

        int layer_idx = -1;

        // split on '.'
        size_t dotpos = name.find('.');
        if (dotpos != std::string::npos && name.substr(0, dotpos) == "direction") {
            try {
                layer_idx = std::stoi(name.substr(dotpos + 1));
            } catch (...) {
                layer_idx = -1;
            }
        }
        if (layer_idx < 0) {
            COM_ERR("invalid/unparsable direction tensor layer index in %s\n", load_info.fname.c_str());
            result.n_embd = -1;
            break;
        } else if (layer_idx == 0) {
            COM_ERR("invalid (zero) direction tensor layer index in %s\n", load_info.fname.c_str());
            result.n_embd = -1;
            break;
        }

        struct ggml_tensor * tensor = ggml_get_tensor(ctx, name.c_str());
        if (tensor->type != GGML_TYPE_F32) {
            COM_ERR("invalid (non-F32) direction tensor type in %s\n", load_info.fname.c_str());
            result.n_embd = -1;
            break;
        }
        if (ggml_n_dims(tensor) != 1) {
            COM_ERR("invalid (non-1D) direction tensor shape in %s\n", load_info.fname.c_str());
            result.n_embd = -1;
            break;
        }

        if (result.n_embd == -1) {
            result.n_embd = ggml_nelements(tensor);
        } else if (ggml_nelements(tensor) != result.n_embd) {
            COM_ERR("direction tensor in %s does not match previous dimensions\n", load_info.fname.c_str());
            result.n_embd = -1;
            break;
        }

        // extend if necessary - do not store data for layer 0 (it's not used)
        result.data.resize(std::max(result.data.size(), static_cast<size_t>(result.n_embd * layer_idx)), 0.0f);

        const float * src = (const float *) tensor->data;
        float * dst = result.data.data() + result.n_embd * (layer_idx - 1);  // layer 1 at [0]
        for (int j = 0; j < result.n_embd; j++) {
            dst[j] += src[j] * load_info.strength;  // allows multiple directions for same layer in same file
        }

    }

    if (result.n_embd == -1) {
        COM_WRN("skipping %s due to invalid direction tensors\n", load_info.fname.c_str());
        result.data.clear();
    }

    gguf_free(ctx_gguf);
    ggml_free(ctx);

    return result;
}

common_control_vector_data common_control_vector_load(const std::vector<common_control_vector_load_info> & load_infos) {
    common_control_vector_data result = { -1, {} };

    for (const auto & info : load_infos) {
        auto cur = common_control_vector_load_one(info);

        if (cur.n_embd == -1) {
            result.n_embd = -1;
            break;
        }
        if (result.n_embd != -1 && result.n_embd != cur.n_embd) {
            COM_ERR("control vectors in %s does not match previous dimensions\n", info.fname.c_str());
            result.n_embd = -1;
            break;
        }

        if (result.n_embd == -1) {
            result = std::move(cur);
        } else {
            result.data.resize(std::max(result.data.size(), cur.data.size()), 0.0f);  // extend if necessary
            for (size_t i = 0; i < cur.data.size(); i++) {
                result.data[i] += cur.data[i];
            }
        }
    }

    if (result.n_embd == -1) {
        COM_ERR("%s", "no valid control vector files passed\n");
        result.data.clear();
    }

    return result;
}

ggml_opt_dataset_t common_opt_dataset_init(struct llama_context * ctx, const std::vector<llama_token> & tokens, int64_t stride) {
    const int64_t ne_datapoint = llama_n_ctx(ctx);
    const int64_t ndata        = (tokens.size() - ne_datapoint - 1) / stride;
    ggml_opt_dataset_t result = ggml_opt_dataset_init(
        GGML_TYPE_I32, GGML_TYPE_I32, ne_datapoint, ne_datapoint, ndata, /*ndata_shard =*/ 1);

    llama_token * data   = (llama_token *) ggml_opt_dataset_data(result)->data;
    llama_token * labels = (llama_token *) ggml_opt_dataset_labels(result)->data;

    for (int64_t idata = 0; idata < ndata; ++idata) {
        memcpy(data   + idata*ne_datapoint, tokens.data() + idata*stride + 0, ne_datapoint*sizeof(llama_token));
        memcpy(labels + idata*ne_datapoint, tokens.data() + idata*stride + 1, ne_datapoint*sizeof(llama_token));
    }

    return result;
}

ggml_opt_optimizer_params common_opt_lr_pars(void * userdata) {
    ggml_opt_optimizer_params result = ggml_opt_get_default_optimizer_params(nullptr);
    const lr_opt &            d      = *(lr_opt *) userdata;
    result.adamw.alpha = result.sgd.alpha = d.get_lr(d.epoch);
    result.sgd.wd = result.adamw.wd = d.wd;
    return result;
}

// TODO make all command line args case-insensitive
static inline bool eq_case_insensitive(char const* a, char const* b) {
    return !
#if defined(_MSC_VER)
        _stricmp
#else
        strcasecmp
#endif // defined(_MSC_VER)
        (a, b);
}

enum ggml_opt_optimizer_type common_opt_get_optimizer(const char * n) {
    if (eq_case_insensitive("adamw", n)) {
        return GGML_OPT_OPTIMIZER_TYPE_ADAMW;
    }
    if (eq_case_insensitive("sgd", n)) {
        return GGML_OPT_OPTIMIZER_TYPE_SGD;
    }
    return GGML_OPT_OPTIMIZER_TYPE_COUNT;
}

// TODO simplify to use just log and exp
static float const k_log_2 = std::log(2.f);

void lr_opt::init() {
    if (lr_min > 0 && lr_min < lr0) {
        float nhalf = std::log(lr0 / lr_min) / k_log_2;
        float e     = epochs;
        if (decay_epochs > 0 && decay_epochs < e) {
            e = decay_epochs;
        } else {
            decay_epochs = e;
        }
        scale_epoch = nhalf / e;
    }
}

float lr_opt::get_lr(float epoch) const {
    float r = lr_min <= 0 ? lr0 :
        epoch >= decay_epochs ? lr_min :
        lr0 * std::pow(0.5f, epoch * scale_epoch);
    LOG_INF("epoch %.2g lr=%.2g\n", epoch, r);
    return r;
}

bool common_replay_last_token(struct llama_context * ctx, llama_token last_token, int32_t pos) {
    llama_batch batch = llama_batch_get_one(&last_token, 1);
    batch.pos = &pos;
    if (llama_decode(ctx, batch)) {
        LOG_ERR("%s: failed to replay last token\n", __func__);
        return false;
    }
    return true;
}

bool common_prompt_batch_decode(
              struct llama_context * ctx,
    const std::vector<llama_token> & all_tokens,
                               int   n_new,
                               int & n_past,
                               int   n_batch,
                  std::string_view   state_path,
                              bool   save_state) {
    if (n_new == 0) {
        return true;
    }
    const int offset = all_tokens.size() - n_new;

    if (save_state && n_new > 1) {
        const int n_tokens_before_last = n_new - 1;

        GGML_ASSERT(n_new <= n_batch);

        // Decode all but the last token so we can save the memory state before decoding the last token.
        // This is done so we can restore the session state later and replay the last token.
        // Memory implementations in recurrent/hybrid models don't support removing tokens from their
        // memory, so we can't just remove the last token from the memory and replay the last token which
        // is the reason for this logic.
        if (llama_decode(ctx, llama_batch_get_one(const_cast<llama_token*>(all_tokens.data() + offset), n_tokens_before_last))) {
            COM_ERR("%s", "failed to eval\n");
            return false;
        }
        n_past += n_tokens_before_last;

        llama_state_save_file(ctx, state_path.data(), all_tokens.data(), all_tokens.size());
        COM_INF("saved session before last token to %s, n_new = %zu\n", state_path.data(), all_tokens.size());

        llama_token last_token = all_tokens.back();
        llama_batch batch = llama_batch_get_one(&last_token, 1);
        int32_t pos = n_past;
        batch.pos = &pos;

        if (llama_decode(ctx, batch)) {
            COM_ERR("%s", "failed to eval last token\n");
            return false;
        }
        n_past++;
    } else {
        if (llama_decode(ctx, llama_batch_get_one(const_cast<llama_token*>(all_tokens.data() + offset), n_new))) {
            COM_ERR("%s", "failed to eval\n");
            return false;
        }
        n_past += n_new;
    }

    return true;
}

size_t common_prompt_checkpoint::size() const {
    return data_tgt.size() + data_dft.size() + data_spec.size();
}

bool common_prompt_checkpoint::empty() const {
    return data_tgt.empty();
}

void common_prompt_checkpoint::clear() {
    n_tokens = 0;

    pos_min = 0;
    pos_max = 0;

    data_tgt.clear();
    data_dft.clear();
    data_spec.clear();
}

void common_prompt_checkpoint::update_pos(
        int64_t n_tokens,
        llama_pos pos_min,
        llama_pos pos_max) {
    this->n_tokens = n_tokens;
    this->pos_min  = pos_min;
    this->pos_max  = pos_max;
}

void common_prompt_checkpoint::update_tgt(
        llama_context * ctx,
        llama_seq_id seq_id,
        llama_state_seq_flags flags) {
    if (ctx == nullptr) {
        return;
    }

    const size_t ckpt_size = llama_state_seq_get_size_ext(ctx, seq_id, flags);

    data_tgt.resize(ckpt_size);

    const size_t n = llama_state_seq_get_data_ext(ctx, data_tgt.data(), ckpt_size, seq_id, flags);
    if (n != ckpt_size) {
        GGML_ABORT("checkpoint size mismatch: expected %zu, got %zu\n", ckpt_size, n);
    }
}

void common_prompt_checkpoint::update_dft(
        llama_context * ctx,
        llama_seq_id seq_id,
        llama_state_seq_flags flags) {
    if (ctx == nullptr) {
        return;
    }

    const size_t ckpt_size = llama_state_seq_get_size_ext(ctx, seq_id, flags);

    data_dft.resize(ckpt_size);

    const size_t n = llama_state_seq_get_data_ext(ctx, data_dft.data(), ckpt_size, seq_id, flags);
    if (n != ckpt_size) {
        GGML_ABORT("checkpoint size mismatch: expected %zu, got %zu\n", ckpt_size, n);
    }
}

void common_prompt_checkpoint::load_tgt(
        llama_context * ctx,
        llama_seq_id seq_id,
        llama_state_seq_flags flags) const {
    if (ctx == nullptr) {
        return;
    }

    if (data_tgt.empty()) {
        return;
    }

    const size_t n = llama_state_seq_set_data_ext(ctx, data_tgt.data(), data_tgt.size(), seq_id, flags);
    if (n != data_tgt.size()) {
        GGML_ABORT("checkpoint size mismatch: expected %zu, got %zu\n", data_tgt.size(), n);
    }
}

void common_prompt_checkpoint::load_dft(
        llama_context * ctx,
        llama_seq_id seq_id,
        llama_state_seq_flags flags) const {
    if (ctx == nullptr) {
        return;
    }

    if (data_dft.empty()) {
        return;
    }

    const size_t n = llama_state_seq_set_data_ext(ctx, data_dft.data(), data_dft.size(), seq_id, flags);
    if (n != data_dft.size()) {
        GGML_ABORT("checkpoint size mismatch: expected %zu, got %zu\n", data_dft.size(), n);
    }
}

void common_prompt_checkpoint::clear_tgt() {
    data_tgt.clear();
}

void common_prompt_checkpoint::clear_dft() {
    data_dft.clear();
    data_spec.clear();
}
