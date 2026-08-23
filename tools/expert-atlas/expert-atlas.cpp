// llama-expert-atlas: measures which MoE experts fire for which topics.
//
// Runs a small bundle of labeled probe prompts through greedy prefill (no
// sampling, no speculative decoding - see docs/moe-cache-colibri-notes.md
// for why those confound routing measurements) and hooks the eval callback
// to read the ffn_moe_topk-<layer> tensor that llm_graph_context::build_moe_ffn
// already produces for every MoE layer. From the per-(layer,expert,category)
// selection counts it derives a topic-affinity position (weighted centroid of
// category anchors on a unit circle) and a specialization score, following
// the same math as Colibri's tools/expert_atlas/analyze.py. The output JSON
// is consumed by the server's --expert-atlas-file flag to plot the Brain
// view's points by measured topic affinity instead of a fixed grid.

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstring>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

using json = nlohmann::ordered_json;

struct atlas_probe {
    const char * category;
    const char * text;
};

// 9 categories x 3 prompts each. Three per category so the replication gate
// below (an expert's affinity to a category only counts if it fired on at
// least 2 of that category's 3 independent prompts) has something to gate.
static const atlas_probe k_probes[] = {
    { "code",     "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    return quicksort(left) + [pivot] + quicksort([x for x in arr if x > pivot])" },
    { "code",     "SELECT c.name, SUM(o.total) AS revenue FROM customers c JOIN orders o ON c.id = o.customer_id WHERE o.total > 100 GROUP BY c.name ORDER BY revenue DESC LIMIT 10;" },
    { "code",     "public class BinaryTree<T extends Comparable<T>> {\n    private Node<T> root;\n    public void insert(T value) {\n        root = insertRec(root, value);\n    }\n}" },
    { "math",     "Prove that the square root of 2 is irrational. Assume for contradiction that sqrt(2) = p/q where p and q are coprime integers, then derive a contradiction." },
    { "math",     "Let f(x) = x^3 - 3x + 1. Find all critical points by computing f'(x), solving f'(x) = 0, then classify each as a local maximum or minimum." },
    { "math",     "Evaluate the definite integral of x*e^x from 0 to 1 using integration by parts, showing each step of the substitution and the final numeric result." },
    { "science",  "Explain how mitochondria produce ATP through oxidative phosphorylation, describing the electron transport chain and the role of the proton gradient." },
    { "science",  "Describe stellar nucleosynthesis, from hydrogen fusion in main-sequence stars to the formation of heavier elements in supernova explosions." },
    { "science",  "What causes the greenhouse effect, and how does rising atmospheric CO2 change the Earth's radiative energy balance over time?" },
    { "law",      "Summarize the doctrine of promissory estoppel and explain the elements a plaintiff must prove to enforce a promise made without consideration." },
    { "law",      "Explain the difference between a motion to dismiss and a motion for summary judgment in civil procedure, and when a court grants each." },
    { "law",      "Describe the elements of a valid contract under common law: offer, acceptance, consideration, capacity, and legality." },
    { "medicine", "Describe the pathophysiology of type 2 diabetes, including insulin resistance, beta-cell dysfunction, and downstream vascular complications." },
    { "medicine", "Explain the mechanism of action of ACE inhibitors in treating hypertension and their effect on the renin-angiotensin-aldosterone system." },
    { "medicine", "What is the difference between a Type I and Type II error in a clinical trial, and how does statistical power relate to each?" },
    { "creative", "Write a short poem about the changing of seasons, using vivid imagery of autumn leaves falling and winter's first snow." },
    { "creative", "Continue this story: the old lighthouse keeper hadn't seen another soul in three months, until the morning a small boat appeared on the horizon." },
    { "creative", "Write a formal email declining a job offer, thanking the company for their time while explaining you have accepted a different position." },
    { "casual",   "Hey, how's it going? I was just thinking about grabbing some coffee later, want to join?" },
    { "casual",   "What's your favorite way to spend a rainy weekend? I usually just curl up with a book or binge a show." },
    { "casual",   "lol that's hilarious, did you see the game last night? can't believe that ending honestly" },
    { "format",   "{\n  \"name\": \"example\",\n  \"version\": \"1.0.0\",\n  \"dependencies\": {\n    \"lodash\": \"^4.17.21\"\n  }\n}" },
    { "format",   "<config>\n  <server>\n    <host>0.0.0.0</host>\n    <port>8080</port>\n  </server>\n</config>" },
    { "format",   "name: ci\non:\n  push:\n    branches: [main]\njobs:\n  build:\n    runs-on: ubuntu-latest\n    steps:\n      - uses: actions/checkout@v4" },
    { "history",  "Describe the causes of the fall of the Western Roman Empire, including economic decline, military pressure, and political fragmentation." },
    { "history",  "Explain the significance of the Treaty of Westphalia in 1648 for the modern concept of state sovereignty." },
    { "history",  "What were the main causes and consequences of the Industrial Revolution in 18th and 19th century Britain?" },
};

static const size_t k_n_probes = sizeof(k_probes) / sizeof(k_probes[0]);

static void print_usage(int, char ** argv) {
    LOG("\nexample usage:\n");
    LOG("\n    %s -m model.gguf -ngl 99 [-o expert-atlas.json]\n\n", argv[0]);
}

struct expert_stats {
    std::unordered_map<std::string, int64_t> count_by_category;
    std::unordered_map<std::string, int>     probes_hit_by_category; // distinct probes that routed here at least once
};

struct atlas_collector {
    int n_layer = 0;
    std::vector<std::unordered_map<int32_t, expert_stats>> layer_experts; // [layer][expert]
    std::vector<std::unordered_map<std::string, int64_t>>   layer_category_totals; // [layer][category]

    std::string cur_category;
    // per-probe touched set, to update probes_hit once per probe (not per token)
    std::vector<std::unordered_map<int32_t, bool>> cur_probe_touched;

    void begin_probe(const std::string & category) {
        cur_category = category;
        cur_probe_touched.assign(n_layer, {});
    }

    void end_probe() {
        for (int il = 0; il < n_layer; ++il) {
            for (const auto & [expert, _] : cur_probe_touched[il]) {
                layer_experts[il][expert].probes_hit_by_category[cur_category]++;
            }
        }
    }

    void record(int il, int32_t expert) {
        if (il < 0 || il >= n_layer) {
            return;
        }
        layer_experts[il][expert].count_by_category[cur_category]++;
        layer_category_totals[il][cur_category]++;
        cur_probe_touched[il][expert] = true;
    }
};

static atlas_collector g_collector;

static bool atlas_cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    GGML_UNUSED(user_data);

    if (g_collector.cur_category.empty()) {
        return true;
    }

    static const char k_prefix[] = "ffn_moe_topk-";
    static const size_t k_prefix_len = sizeof(k_prefix) - 1;

    if (strncmp(t->name, k_prefix, k_prefix_len) != 0) {
        return ask ? false : true;
    }

    if (ask) {
        return true;
    }

    const int il = atoi(t->name + k_prefix_len);

    const bool is_host = ggml_backend_buffer_is_host(t->buffer);
    std::vector<uint8_t> tmp;
    const uint8_t * data;
    if (is_host) {
        data = (const uint8_t *) t->data;
    } else {
        tmp.resize(ggml_nbytes(t));
        ggml_backend_tensor_get(t, tmp.data(), 0, tmp.size());
        data = tmp.data();
    }

    // selected_experts: [n_expert_used, n_tokens], I32
    const int64_t n_used = t->ne[0];
    const int64_t n_tok   = t->ne[1];

    for (int64_t tok = 0; tok < n_tok; ++tok) {
        for (int64_t u = 0; u < n_used; ++u) {
            const int32_t expert = *(const int32_t *) (data + tok * t->nb[1] + u * t->nb[0]);
            g_collector.record(il, expert);
        }
    }

    return true;
}

static json build_atlas_json() {
    // fixed category order so anchor angles are deterministic between runs
    std::vector<std::string> categories;
    {
        std::unordered_map<std::string, bool> seen;
        for (size_t i = 0; i < k_n_probes; ++i) {
            if (!seen[k_probes[i].category]) {
                seen[k_probes[i].category] = true;
                categories.emplace_back(k_probes[i].category);
            }
        }
    }

    const double two_pi = 2.0 * M_PI;
    std::unordered_map<std::string, double> angle_by_category;
    for (size_t c = 0; c < categories.size(); ++c) {
        angle_by_category[categories[c]] = two_pi * (double) c / (double) categories.size();
    }

    json cells = json::array();

    for (int il = 0; il < g_collector.n_layer; ++il) {
        for (const auto & [expert, stats] : g_collector.layer_experts[il]) {
            int64_t total_n = 0;
            for (const auto & [cat, n] : stats.count_by_category) {
                total_n += n;
            }
            if (total_n == 0) {
                continue;
            }

            // replication gate: an expert's affinity to a category only counts
            // if it fired on at least 2 of that category's independent probes,
            // otherwise a single lucky prompt can plant it far from center
            std::unordered_map<std::string, double> f;
            double f_sum = 0.0;
            for (const auto & [cat, n] : stats.count_by_category) {
                const auto it_hit = stats.probes_hit_by_category.find(cat);
                const int probes_hit = it_hit == stats.probes_hit_by_category.end() ? 0 : it_hit->second;
                if (probes_hit < 2) {
                    continue;
                }
                const int64_t cat_total = g_collector.layer_category_totals[il].at(cat);
                const double  fc = (double) n / (double) std::max<int64_t>(1, cat_total);
                f[cat] = fc;
                f_sum += fc;
            }

            double x = 0.0, y = 0.0, spec = 0.0;
            // Per-category probability vector, kept rather than discarded.
            // x/y below collapse it onto a circle, which is lossy in two
            // specific ways worth naming: (1) an N-category profile is
            // squashed to 2 numbers, so two experts with entirely different
            // topic mixes can land on the same point (a 50/50 code+history
            // expert and an even generalist both sit near the origin); and
            // (2) the circle places categories at equal angles in *array
            // order*, which invents an adjacency that was never measured -
            // "code" ends up next to "math" and, wrapping around, next to
            // "history", purely because of where they sit in k_probes[].
            // Emitting the raw vector lets a consumer project it however it
            // likes (2D circle, 3D sphere, or a re-projection after the
            // category set grows) without this tool having to pick one, and
            // without another schema change later. Sparse: categories an
            // expert never fired on are omitted, which is most of them for
            // any genuinely specialized expert.
            json cats = json::object();
            if (f_sum > 0.0) {
                const double log_c = std::log((double) categories.size());
                double entropy = 0.0;
                for (const auto & [cat, fc] : f) {
                    const double p = fc / f_sum;
                    x += p * std::cos(angle_by_category[cat]);
                    y += p * std::sin(angle_by_category[cat]);
                    if (p > 0.0) {
                        entropy -= p * std::log(p);
                        cats[cat] = std::round(p * 1000.0) / 1000.0;
                    }
                }
                spec = log_c > 0.0 ? 1.0 - entropy / log_c : 0.0;
            }

            json cell;
            cell["layer"]  = il;
            cell["expert"] = expert;
            cell["n"]      = total_n;
            cell["spec"]   = std::round(spec * 1000.0) / 1000.0;
            // x/y stay exactly as before - the existing 2D atlas view and
            // moe-cache's own req_dir tracking both read them, and changing
            // their meaning would silently invalidate every atlas file on
            // disk. cats is purely additive.
            cell["x"]      = std::round(x * 1000.0) / 1000.0;
            cell["y"]      = std::round(y * 1000.0) / 1000.0;
            cell["cats"]   = std::move(cats);
            cells.push_back(std::move(cell));
        }
    }

    json out;
    out["categories"] = categories;
    out["n_layer"]     = g_collector.n_layer;
    out["n_probes"]    = (int) k_n_probes;
    out["cells"]       = std::move(cells);
    return out;
}

int main(int argc, char ** argv) {
    common_params params;
    params.n_ctx   = 4096;
    params.warmup  = false;
    params.escape  = false;
    params.out_file = "expert-atlas.json";

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_IMATRIX, print_usage)) {
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    params.cb_eval           = atlas_cb_eval;
    params.cb_eval_user_data = nullptr;

    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();

    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s: failed to init\n", __func__);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const bool add_bos = llama_vocab_get_add_bos(vocab);

    g_collector.n_layer = llama_model_n_layer(model);
    g_collector.layer_experts.resize(g_collector.n_layer);
    g_collector.layer_category_totals.resize(g_collector.n_layer);

    LOG_INF("%s: running %zu probe prompts across %d layers\n", __func__, k_n_probes, g_collector.n_layer);

    llama_batch batch = llama_batch_init(params.n_batch, 0, 1);

    for (size_t i = 0; i < k_n_probes; ++i) {
        const atlas_probe & probe = k_probes[i];

        llama_memory_clear(llama_get_memory(ctx), true);

        std::vector<llama_token> tokens = common_tokenize(ctx, probe.text, add_bos, true);
        if ((int) tokens.size() > params.n_batch) {
            LOG_WRN("%s: probe %zu (%s) has %zu tokens, truncating to n_batch=%d\n",
                    __func__, i, probe.category, tokens.size(), params.n_batch);
            tokens.resize(params.n_batch);
        }

        common_batch_clear(batch);
        for (size_t j = 0; j < tokens.size(); ++j) {
            common_batch_add(batch, tokens[j], (int32_t) j, { 0 }, false);
        }

        g_collector.begin_probe(probe.category);

        if (llama_decode(ctx, batch)) {
            LOG_ERR("%s: decode failed for probe %zu (%s)\n", __func__, i, probe.category);
            g_collector.end_probe();
            continue;
        }

        g_collector.end_probe();

        LOG_INF("%s: [%zu/%zu] %-10s %4zu tokens\n", __func__, i + 1, k_n_probes, probe.category, tokens.size());
    }

    llama_batch_free(batch);

    json atlas = build_atlas_json();

    std::ofstream out(params.out_file);
    out << atlas.dump(2);

    LOG_INF("%s: wrote %zu cells to %s\n", __func__, atlas["cells"].size(), params.out_file.c_str());

    llama_backend_free();

    return 0;
}
