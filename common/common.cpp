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

// margin: bytes that must remain free per device on top of what's needed.
// Defaults to 0 (bare fit) for callers that only ask "would this load at all".
// A caller whose answer will be re-judged by common_fit_params() must pass the
// same margin that check will demand - otherwise placement approves a config
// that fit then rejects, and the two silently disagree.
static bool common_device_memory_data_fits(const common_device_memory_data_vec & data, int64_t margin = 0) {
    for (const auto & d : data) {
        if (d.total <= 0) {
            continue; // not a real device (host aggregate, or a device with unknown budget)
        }
        const int64_t needed = (int64_t) d.model + (int64_t) d.context + (int64_t) d.compute;
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
        return common_device_memory_data_fits(trial_data, margin);
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

    if (common_device_memory_data_fits(data, margin)) {
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

struct common_moe_calibration_entry {
    int         n_cpu_moe       = 0;
    int         n_threads       = -1;
    int         n_threads_batch = -1;
    int         spec_n_max      = -1; // -1 = no MTP calibration recorded
    int         concurrency     = 1;  // > 1: tok_per_sec is aggregate throughput at this many concurrent requests, not solo
    double      tok_per_sec     = 0.0;
    int         moe_cache_mb    = -1; // -1 = not calibrated, use --moe-cache auto
    std::string calibrated_at;
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
        gpu_sig += string_format("%s:%zu;", ggml_backend_dev_name(dev), dtotal >> 20);
    }
    long long model_size = 0;
    struct stat st;
    if (stat(path_model, &st) == 0) {
        model_size = (long long) st.st_size;
    }
    return string_format("%s|%s|%lld|c%u|p%u|ngl%d",
            gpu_sig.c_str(), path_model, model_size, params.n_ctx, params.n_parallel, params.n_gpu_layers);
}

static std::string common_moe_calibration_cache_path() {
    return fs_get_cache_directory() + "moe-calibration.json";
}

static bool common_moe_calibration_lookup(
        const char * path_model, const common_params & params, common_moe_calibration_entry & out) {
    std::ifstream f(common_moe_calibration_cache_path());
    if (!f.good()) {
        return false;
    }
    try {
        nlohmann::json j;
        f >> j;
        const std::string key = common_moe_calibration_key(path_model, params);
        if (!j.contains(key)) {
            return false;
        }
        const auto & e = j.at(key);
        out.n_cpu_moe       = e.value("n_cpu_moe", 0);
        out.n_threads       = e.value("n_threads", -1);
        out.n_threads_batch = e.value("n_threads_batch", -1);
        out.spec_n_max      = e.value("spec_n_max", -1);
        out.concurrency     = e.value("concurrency", 1);
        out.tok_per_sec     = e.value("tok_per_sec", 0.0);
        out.moe_cache_mb    = e.value("moe_cache_mb", -1);
        out.calibrated_at   = e.value("calibrated_at", std::string());
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
        {"calibrated_at",   entry.calibrated_at},
    };
    fs_create_directory_with_parents(fs_get_cache_directory());
    std::ofstream out(path);
    out << j.dump(2);
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
static std::pair<double, double> common_moe_bench_one_request(int port, const char * prompt, int n_predict) {
    nlohmann::json req = {
        {"messages", nlohmann::json::array({
            {{"role", "user"}, {"content", prompt}}
        })},
        {"max_tokens", n_predict},
    };
    const std::string req_body = req.dump();
    char req_cmd[4096];
    snprintf(req_cmd, sizeof(req_cmd),
        "curl -s http://127.0.0.1:%d/v1/chat/completions -H 'Content-Type: application/json' --data-binary %s",
        port, common_shell_quote(req_body).c_str());
    FILE * rp = popen(req_cmd, "r");
    if (!rp) {
        return {-1.0, 0.0};
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
            return {j["timings"]["predicted_n"].get<double>(), j["timings"]["predicted_ms"].get<double>()};
        }
    } catch (const std::exception &) {
        // falls through to the failure return below
    }
    return {-1.0, 0.0};
}

static double common_moe_bench_candidate_server(
        const std::string & self_exe, const std::string & path_model, const std::string & mtp_path,
        uint32_t n_cpu_moe, int n_max, int n_threads, int port, uint32_t n_ctx, int n_predict,
        int n_concurrency = 1, int moe_cache_mb = -1) {
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
    snprintf(cmd, sizeof(cmd),
        "'%s' -m '%s' -ngl 99 -ncmoe %u --moe-cache %s -c %u %s%s%s"
        "--temp 1.0 --top-p 0.95 --top-k 64 --no-token-freq-log "
        "--port %d --no-webui > /dev/null 2>&1 & echo $!",
        self_exe.c_str(), path_model.c_str(), n_cpu_moe, cache_arg, ctx_for_launch,
        mtp_args.c_str(), threads_args.c_str(), parallel_args.c_str(), port);
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

    double result_tps = -1.0;
    if (n_concurrency <= 1) {
        // Solo path: average per-request predicted_per_second across a
        // couple of sequential probes - unchanged from before concurrency
        // support was added, so existing solo calibration behavior is
        // bit-for-bit identical.
        double sum_tps = 0.0;
        int n_ok = 0;
        for (int i = 0; i < 2; i++) {
            const auto [predicted_n, predicted_ms] = common_moe_bench_one_request(port, probe_prompts[i], n_predict);
            if (predicted_n > 0 && predicted_ms > 0) {
                sum_tps += predicted_n / (predicted_ms / 1000.0);
                n_ok++;
            }
        }
        result_tps = n_ok > 0 ? sum_tps / n_ok : -1.0;
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
void common_moe_calibrate(common_params & params) {
    const char * path_model = params.model.path.c_str();
    auto mparams = common_model_params_to_llama(params);
    auto cparams = common_context_params_to_llama(params);
    const int concurrency = std::max(1, (int) params.n_parallel);

    LOG_INF("%s: probing safe MoE CPU-offload floor for this GPU+model+context combination ...\n", __func__);
    common_moe_fit_probe_result probe = common_moe_find_safe_layers(path_model, mparams, cparams);
    if (!probe.is_moe) {
        LOG_WRN("%s: model has no MoE experts - nothing for --moe-calibrate to do\n", __func__);
        return;
    }

    const uint32_t safe_n = probe.already_fits ? 0 : probe.safe_n;
    if (!probe.already_fits && !probe.found_safe_n) {
        LOG_ERR("%s: config does not fit in available device memory even with all MoE experts on CPU; "
                "reduce -c/--parallel or add VRAM before calibrating\n", __func__);
        return;
    }

    const int n_threads_default = params.cpuparams.n_threads > 0 ? params.cpuparams.n_threads : common_cpu_get_num_math();
    const int n_predict = 64;

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
    constexpr int n_samples_per_candidate = 2;
    auto bench_one_sample = [&](uint32_t n_cpu_moe, int n_max, const std::string & mtp_path, int n_threads) -> double {
        double tps = common_moe_bench_candidate_server(
                self_exe, path_model, mtp_path, n_cpu_moe, n_max, n_threads, next_port(), ctx, n_predict, concurrency);
        if (tps < 0) {
            LOG_WRN("%s:   candidate sample failed, retrying ...\n", __func__);
            tps = common_moe_bench_candidate_server(
                    self_exe, path_model, mtp_path, n_cpu_moe, n_max, n_threads, next_port(), ctx, n_predict, concurrency);
        }
        return tps;
    };
    auto bench_with_retry = [&](uint32_t n_cpu_moe, int n_max, const std::string & mtp_path, int n_threads) -> double {
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

    LOG_INF("%s: golden-section search for -ncmoe in [%u, %u] at n_threads=%d%s (real llama-server subprocess, "
            "chat-templated prompts, per candidate) ...\n", __func__, safe_n, ncmoe_hi, n_threads_default,
            concurrency > 1 ? string_format(", concurrency=%d (aggregate throughput)", concurrency).c_str() : "");

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
        return tps; // failed candidates measure as -1, golden-section still works (just avoids them)
    };
    uint32_t best_n = (uint32_t) common_golden_section_search_max((int) safe_n, (int) ncmoe_hi, measure_ncmoe, ncmoe_trace);
    double best_tps = ncmoe_trace.at((int) best_n);
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
            std::map<int, double> nmax_trace;
            double baseline = -1.0;
            int last_good = 1;
            for (int n = 1; n <= 32; n *= 2) {
                const double tps = bench_with_retry(best_n, n, params.speculative.draft.mparams.path, n_threads_default);
                nmax_trace[n] = tps;
                LOG_INF("%s:   spec-draft-n-max=%d -> %s\n", __func__, n, tps > 0 ? string_format("%.2f tok/s", tps).c_str() : "failed");
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
        LOG_INF("%s: benchmarking %zu thread-count candidate(s) at ncmoe=%u%s ...\n", __func__, thread_candidates.size(), best_n,
                best_n_max > 0 ? string_format(", spec-draft-n-max=%d", best_n_max).c_str() : "");
        for (int nt : thread_candidates) {
            if (nt == n_threads_default) {
                continue; // already measured above as best_tps
            }
            const double tps = bench_with_retry(best_n, n_max_for_threads, mtp_path_for_threads, nt);
            LOG_INF("%s:   n_threads=%d -> %s\n", __func__, nt, tps > 0 ? string_format("%.2f tok/s", tps).c_str() : "failed");
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
                best_threads, next_port(), ctx, n_predict, concurrency, mb);
        if (tps < 0) {
            tps = common_moe_bench_candidate_server(
                    self_exe, path_model, mtp_path_for_threads, best_n, n_max_for_threads,
                    best_threads, next_port(), ctx, n_predict, concurrency, mb);
        }
        LOG_INF("%s:   moe-cache=%dMiB -> %s\n", __func__, mb,
                tps > 0 ? string_format("%.2f tok/s", tps).c_str() : "failed");
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

    time_t now = time(nullptr);
    char timebuf[32];
    strftime(timebuf, sizeof(timebuf), "%Y-%m-%d %H:%M:%S", localtime(&now));

    common_moe_calibration_entry entry;
    entry.n_cpu_moe       = (int) best_n;
    entry.n_threads       = best_threads;
    entry.n_threads_batch = best_threads;
    entry.spec_n_max      = best_n_max;
    entry.concurrency     = concurrency;
    entry.tok_per_sec     = best_threads_tps;
    entry.moe_cache_mb    = best_cache_mb;
    entry.calibrated_at   = timebuf;
    common_moe_calibration_save(path_model, params, entry);

    const char * tps_label = concurrency > 1 ? "aggregate tok/s" : "tok/s";
    if (best_n_max > 0) {
        LOG_INF("%s: calibration complete - ncmoe=%d, n_threads=%d, spec-draft-n-max=%d%s, measured %.2f %s. "
                "Cached to %s - launch normally (without --moe-calibrate) to use it.\n",
                __func__, entry.n_cpu_moe, entry.n_threads, entry.spec_n_max,
                concurrency > 1 ? string_format(", concurrency=%d", concurrency).c_str() : "",
                entry.tok_per_sec, tps_label, common_moe_calibration_cache_path().c_str());
    } else {
        LOG_INF("%s: calibration complete - ncmoe=%d, n_threads=%d%s, measured %.2f %s. "
                "Cached to %s - launch normally (without --moe-calibrate) to use it.\n",
                __func__, entry.n_cpu_moe, entry.n_threads,
                concurrency > 1 ? string_format(", concurrency=%d", concurrency).c_str() : "",
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
        const std::vector<float> & x, const std::vector<float> & y, const std::vector<float> & spec) {
    if (!ggml_moe_cache.set_atlas || !host_base || expert.empty()) {
        return false;
    }
    if (expert.size() != x.size() || expert.size() != y.size() || expert.size() != spec.size()) {
        return false;
    }
    ggml_moe_cache.set_atlas(host_base, expert.data(), x.data(), y.data(), spec.data(), (int) expert.size());
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
        if (common_moe_fits_with_n(path_model, mparams, cparams, (uint32_t) cached.n_cpu_moe,
                                    devs, hp_ngl, hp_n_ctx_train, hp_n_expert)) {
            params.tensor_buft_overrides = common_moe_build_cpu_overrides((uint32_t) cached.n_cpu_moe);
            mparams.tensor_buft_overrides = params.tensor_buft_overrides.data();
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
            if (cached.moe_cache_mb > 0 && cache_mode_is_auto) {
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
            LOG_WRN("%s: using calibrated MoE placement from cache (ncmoe=%d, n_threads=%d%s%s%s, measured %.2f %s on %s) "
                    "- run --moe-calibrate again if hardware/model/context changed\n",
                    __func__, cached.n_cpu_moe, cached.n_threads,
                    cached.spec_n_max > 0 ? string_format(", spec-draft-n-max=%d", cached.spec_n_max).c_str() : "",
                    cached.moe_cache_mb > 0 && cache_mode_is_auto ? string_format(", moe-cache=%dMiB", cached.moe_cache_mb).c_str() : "",
                    cached.concurrency > 1 ? string_format(", concurrency=%d", cached.concurrency).c_str() : "",
                    cached.tok_per_sec, cached.concurrency > 1 ? "aggregate tok/s" : "tok/s", cached.calibrated_at.c_str());
            return true;
        }
        LOG_WRN("%s: cached calibration no longer fits current conditions - recalibrating placement live\n", __func__);
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
    // Only meaningful for an explicit request. n_ctx == 0 ("auto") means the
    // user expressed no preference, so there is no context to protect and the
    // existing post-fit autoplace path already handles it.
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
        if (common_moe_calibration_lookup(path_model, params, cal) && cal.n_cpu_moe > 0) {
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
