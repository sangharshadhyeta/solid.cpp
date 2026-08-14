#pragma once

#include "llama-model.h"
#include "llama-graph.h"
#include "llama-model-loader.h"

// note: almost all graphs require at least sqrtf, so include cmath globally
#include <cmath>

//
// base classes
//

struct llm_build_mamba_base : public llm_graph_context {
    llm_build_mamba_base(const llm_graph_params & params);

    virtual ~llm_build_mamba_base() = default;

    ggml_tensor * build_mamba_layer(llm_graph_input_rs * inp, ggml_tensor * cur, const llama_model & model, const llama_ubatch & ubatch, int il);
    ggml_tensor * build_mamba2_layer(llm_graph_input_rs * inp, ggml_tensor * cur, const llama_model & model, const llama_ubatch & ubatch, int il) const;

};

struct llm_build_delta_net_base : public llm_graph_context {
    llm_build_delta_net_base(const llm_graph_params & params);

    virtual ~llm_build_delta_net_base() = default;

    // returns pair of output and new state
    std::pair<ggml_tensor *, ggml_tensor *> build_delta_net_chunking(
                ggml_tensor * q,
                ggml_tensor * k,
                ggml_tensor * v,
                ggml_tensor * g,
                ggml_tensor * b,
                ggml_tensor * s,
                        int   il);

    // returns pair of output and new state
    std::pair<ggml_tensor *, ggml_tensor *> build_delta_net_autoregressive(
                ggml_tensor * q,
                ggml_tensor * k,
                ggml_tensor * v,
                ggml_tensor * g,
                ggml_tensor * b,
                ggml_tensor * s,
                int           il);

    // use the ggml_gated_delta_net fused operator (K=1; state has shape [S_v, S_v, H_v, n_seqs])
    std::pair<ggml_tensor *, ggml_tensor *> build_delta_net_fused(
                ggml_tensor * q,
                ggml_tensor * k,
                ggml_tensor * v,
                ggml_tensor * g,
                ggml_tensor * b,
                ggml_tensor * s,
                        int   il);

    // choose one of two implementations above based on the number of tokens
    std::pair<ggml_tensor *, ggml_tensor *> build_delta_net(
                ggml_tensor * q,
                ggml_tensor * k,
                ggml_tensor * v,
                ggml_tensor * g,
                ggml_tensor * b,
                ggml_tensor * s,
                        int   il);

    // read conv state from cache, concat with qkv_mixed, write back (single slot or per-token)
    // qkv_mixed: (qkv_dim, n_seq_tokens, n_seqs); returns conv_input: (kernel_size + n_seq_tokens - 1, channels, n_seqs)
    ggml_tensor * build_conv_state(
            llm_graph_input_rs * inp,
            ggml_tensor *        conv_states_all,
            ggml_tensor *        qkv_mixed,
            int64_t              conv_kernel_size,
            int64_t              conv_channels,
            int                  il);

    // run delta-net attention and write the new recurrent state(s) back to ssm_states_all
    // s: (head_v_dim, head_v_dim, num_v_heads, n_seqs); returns output: (head_v_dim, num_v_heads, n_seq_tokens, n_seqs)
    ggml_tensor * build_recurrent_attn(
            llm_graph_input_rs * inp,
            ggml_tensor *        ssm_states_all,
            ggml_tensor *        q,
            ggml_tensor *        k,
            ggml_tensor *        v,
            ggml_tensor *        g,
            ggml_tensor *        b,
            ggml_tensor *        s,
            int                  il);
};

struct llm_build_rwkv6_base : public llm_graph_context {
    const llama_model & model;

    llm_build_rwkv6_base(const llama_model & model, const llm_graph_params & params);

    virtual ~llm_build_rwkv6_base() = default;

    ggml_tensor * build_rwkv6_channel_mix(const llama_layer * layer,
                                          ggml_tensor *       cur,
                                          ggml_tensor *       x_prev,
                                          llm_arch            arch) const;

    ggml_tensor * build_rwkv6_time_mix(llm_graph_input_rs * inp,
                                       ggml_tensor *        cur,
                                       ggml_tensor *        x_prev,
                                       const llama_ubatch & ubatch,
                                       int                  il) const;
};

// Base class for RWKV7-related models
struct llm_build_rwkv7_base : public llm_graph_context {
    const llama_model & model;

    llm_build_rwkv7_base(const llama_model & model, const llm_graph_params & params);

    virtual ~llm_build_rwkv7_base() = default;

    // RWKV7-specific graph building methods
    ggml_tensor * build_rwkv7_channel_mix(const llama_layer * layer,
                                          ggml_tensor *       cur,
                                          ggml_tensor *       x_prev,
                                          llm_arch            arch) const;
    ggml_tensor * build_rwkv7_time_mix(llm_graph_input_rs * inp,
                                       ggml_tensor *        cur,
                                       ggml_tensor *        x_prev,
                                       ggml_tensor *&       first_layer_value,
                                       const llama_ubatch & ubatch,
                                       int                  il) const;
};

//
// FR-Spec-style MTP draft-vocab trim (shared helpers)
//
// A standalone MTP-only draft can score a frequency-ranked subset of the
// vocabulary instead of the full one, cutting the dominant draft-time cost
// (the LM-head projection) losslessly - verification still runs over the
// full vocab on the target model, so output is unchanged. These helpers
// are architecture-agnostic (they only touch tensor shapes/data, not
// anything model-specific), so each architecture's own load_arch_tensors/
// graph_mtp only needs a few lines calling into these, not a fresh
// reimplementation of the underlying ggml graph operations - see
// docs/moe-cache-colibri-notes.md, "FR-Spec draft-vocab trimming", for the
// full design writeup and the one architecture (gemma4) that currently
// uses this.
//
// Unlike upstream's EAGLE3/qwen35 precedent (a d2t tensor baked into the
// draft GGUF itself, produced by a conversion-time tool that doesn't
// exist upstream yet), this version reads the trim from an external
// sidecar file at load time and gathers the relevant rows out of the
// already-loaded full-vocab weight - the original draft GGUF is never
// modified, and the sidecar can be regenerated from real serving traffic
// (see server-token-freq.h) without touching the checkpoint at all.

// Reads a sidecar vocab-trim mapping file (see server-token-freq.h /
// the frspec regeneration tool for the producer side). Returns the
// ordered list of original (full-vocab) token ids, one per trimmed-vocab
// position - i.e. exactly the d2t convention EAGLE3/qwen35 already use,
// just sourced from an external file instead of a GGUF tensor. Returns
// an empty vector (not an error) if the path doesn't exist or is
// unreadable - callers should treat that as "no trim configured", never
// fail model load over it.
std::vector<int64_t> llm_frspec_load_d2t_sidecar(const std::string & path);

// Gathers just the rows named by d2t_ids out of a full-vocab weight
// tensor (n_embd * n_vocab_full elements of full_weight->type, row-major
// by vocab position - i.e. tok_embd/output.weight's usual layout),
// producing a new, smaller buffer for just the trimmed vocab, in
// d2t_ids' order. Reads through ggml_backend_tensor_get(), so it works
// whether full_weight actually lives in CPU or GPU memory. Row-granularity
// gather is safe for block-quantized types (Q8_0 etc.) as long as n_embd
// is a whole number of quant blocks, which it always is for real
// vocab/embedding dimensions - never splits a quant block across a row
// boundary.
//
// Called exactly once per model load (from set_input(), the first time
// it runs - see llm_graph_input_frspec_d2t below), not from inside the
// compute graph: gathering via a live ggml_get_rows() graph op instead
// re-ran the gather as a real GPU kernel on every single forward pass -
// a real, reproducible decode-speed regression, measured on the sandbox
// (see docs/moe-cache-colibri-notes.md) - since ordinary ggml graph-node
// reuse only skips CPU-side graph *construction*, not re-executing each
// node's kernel once the graph runs.
std::vector<uint8_t> llm_frspec_gather_vocab_rows_from_backend(
        const ggml_tensor * full_weight, const std::vector<int64_t> & d2t_ids);

// Scatters compressed (trimmed-vocab-sized) logits back into a
// full-vocab-shaped tensor (positions outside the trimmed set filled
// -inf), so downstream verify/sampling code never has to know the draft
// scored a reduced vocab. Identical pattern to eagle3.cpp/qwen35.cpp's
// own d2t handling, extracted here so it's written and tested once, not
// once per architecture. `d2t` must be an I32 tensor of the same
// trimmed-vocab size as `cur`'s first dimension (I32, not I64, so the
// same tensor doubles as the index argument to ggml_get_rows() when
// gathering the trimmed weight rows - ggml_get_rows() requires I32).
// `out_template` must already be F32 [1, n_vocab_full, n_outputs] and
// -inf-filled (see llm_graph_input_frspec_d2t::out_template below) -
// this never fills it itself, so it's safe to call every step against
// the same persistent template without repaying that fill each time:
// d2t names the same fixed set of positions on every call for a given
// model instance, so every position outside that set is written by no
// call, ever, and can safely stay whatever out_template already had.
ggml_tensor * llm_frspec_scatter_to_full_vocab(
        ggml_context * ctx0, ggml_tensor * cur, ggml_tensor * d2t, ggml_tensor * out_template);

// Internal-only read side of the pending-vocab-map hint - see
// llama_frspec_set_pending_vocab_map() in llama.h for the public setter
// and the full explanation of why this side channel exists (Gemma-4's
// MTP draft has no in-file signal distinguishing it from the trunk, unlike
// qwen35's mtp_only detection). Consumes (clears) the hint on read, so a
// later, unrelated model load never accidentally inherits a stale value.
std::string llm_frspec_take_pending_vocab_map();

// Graph input for FR-Spec vocab trimming. Shared so each architecture's
// own graph-building code only needs a few lines (create this, use
// output_w in build_lora_mm instead of the full weight, scatter back via
// out_template) rather than reimplementing the input-tensor lifecycle -
// see gemma4-assistant.cpp for the one current user. Three different
// lifetimes bundled into one input:
//   - d2t (the index tensor): constant for the model's whole lifetime,
//     cheap to rewrite, rewritten on every set_input() call regardless.
//   - output_w (the pre-gathered trimmed weight) and out_template (the
//     -inf-filled full-vocab scatter target): both also constant for the
//     model's whole lifetime, but genuinely expensive to recompute (a
//     real row-gather, and a full-vocab-width fill, respectively) -
//     populated exactly once, on the first set_input() call, then left
//     alone. Doing either as a live graph op every step instead (the
//     original design for both) re-ran real GPU kernels on every single
//     decode step - can_reuse()=true only skips rebuilding the graph's
//     CPU-side topology, it does not skip executing each node's kernel
//     once the graph runs, so "constant" ops sitting in the graph are
//     silently paid for on every step regardless. Both sidestep that by
//     never appearing as graph nodes in the first place - they're plain
//     inputs, like d2t itself. out_template stays correct without ever
//     being re-filled because d2t names the exact same fixed positions
//     on every scatter call for a given model instance - every position
//     outside that set is written by no call, ever, so it can safely
//     keep whatever the one-time fill gave it.
// can_reuse() always returns true for all three: none ever changes in a
// way that should defeat graph/CUDA-graph reuse across decode steps the
// way a genuinely per-token-varying input would.
class llm_graph_input_frspec_d2t : public llm_graph_input_i {
public:
    llm_graph_input_frspec_d2t(const std::vector<int64_t> & ids, const ggml_tensor * full_weight) :
        ids(ids), full_weight(full_weight) {}
    virtual ~llm_graph_input_frspec_d2t() = default;

    void set_input(const llama_ubatch * ubatch) override;
    bool can_reuse(const llm_graph_params & params) override;

    ggml_tensor * d2t          = nullptr; // I32 [n_trimmed_vocab]
    ggml_tensor * output_w     = nullptr; // same type as full_weight, [n_embd, n_trimmed_vocab]
    ggml_tensor * out_template = nullptr; // F32 [1, n_vocab_full, n_outputs], -inf outside the trimmed set

    const std::vector<int64_t> & ids;
    const ggml_tensor * full_weight; // source for output_w's one-time gather; never itself modified

private:
    bool output_w_populated     = false;
    bool out_template_populated = false;
};

//
// models
//

struct llama_model_llama : public llama_model_base {
    llama_model_llama(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    template <bool embed>
    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_llama4 : public llama_model_base {
    llama_model_llama4(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    template <bool iswa>
    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_llama_embed : public llama_model_llama {
    llama_model_llama_embed(const struct llama_model_params & params) : llama_model_llama(params) {}
    // reuse load_arch_hparams and load_arch_tensors from llama_model_llama

    template <bool embed>
    using graph = llama_model_llama::graph<embed>;

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_maincoder : public llama_model_base {
    llama_model_maincoder(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_talkie : public llama_model_base {
    llama_model_talkie(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_deci : public llama_model_base {
    llama_model_deci(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_baichuan : public llama_model_base {
    llama_model_baichuan(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_falcon : public llama_model_base {
    llama_model_falcon(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_grok : public llama_model_base {
    llama_model_grok(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_starcoder : public llama_model_base {
    llama_model_starcoder(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_refact : public llama_model_base {
    llama_model_refact(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_bert : public llama_model_base {
    llama_model_bert(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_jina_bert_v2 : public llama_model_base {
    llama_model_jina_bert_v2(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    using graph = llama_model_bert::graph;

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_jina_bert_v3 : public llama_model_base {
    llama_model_jina_bert_v3(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    using graph = llama_model_bert::graph;

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_nomic_bert : public llama_model_base {
    llama_model_nomic_bert(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    using graph = llama_model_bert::graph;

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_nomic_bert_moe : public llama_model_base {
    llama_model_nomic_bert_moe(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    using graph = llama_model_bert::graph;

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_modern_bert : public llama_model_base {
    llama_model_modern_bert(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_neo_bert : public llama_model_base {
    llama_model_neo_bert(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_eurobert : public llama_model_base {
    llama_model_eurobert(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_bloom : public llama_model_base {
    llama_model_bloom(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


// Quant-only stub for mmproj GGUFs
// none of these are ever called, they only exist to satisfy the llama_model_base interface
struct llama_model_clip : public llama_model_base {
    llama_model_clip(const struct llama_model_params & params) : llama_model_base(params) {}

    [[noreturn]]
    void load_arch_hparams(llama_model_loader & ml) override;

    [[noreturn]]
    void load_arch_tensors(llama_model_loader & ml) override;

    [[noreturn]]
    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_mpt : public llama_model_base {
    llama_model_mpt(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_stablelm : public llama_model_base {
    llama_model_stablelm(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llama_model_mellum : public llama_model_base {
    llama_model_mellum(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    template <bool iswa>
    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llama_model_nanbeige : public llama_model_base {
    llama_model_nanbeige(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    int  n_loops = 1;
    int  n_layer_phys = 0;
    bool skip_loop_final_norm = false;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llama_model_qwen : public llama_model_base {
    llama_model_qwen(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_qwen2 : public llama_model_base {
    llama_model_qwen2(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_dream : public llama_model_base {
    llama_model_dream(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_llada : public llama_model_base {
    llama_model_llada(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_llada_moe : public llama_model_base {
    llama_model_llada_moe(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_rnd1 : public llama_model_base {
    llama_model_rnd1(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_qwen2vl : public llama_model_base {
    llama_model_qwen2vl(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_qwen2moe : public llama_model_base {
    llama_model_qwen2moe(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_qwen3 : public llama_model_base {
    llama_model_qwen3(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_qwen3moe : public llama_model_base {
    llama_model_qwen3moe(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_qwen3vl : public llama_model_base {
    llama_model_qwen3vl(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_qwen3vlmoe : public llama_model_base {
    llama_model_qwen3vlmoe(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_qwen3tts : public llama_model_qwen3vl {
    llama_model_qwen3tts(const struct llama_model_params & params) : llama_model_qwen3vl(params) {}
};


struct llama_model_phi2 : public llama_model_base {
    llama_model_phi2(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_phi3 : public llama_model_base {
    llama_model_phi3(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    template <bool iswa>
    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_phimoe : public llama_model_base {
    llama_model_phimoe(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    template <bool iswa>
    using graph = llama_model_phi3::graph<iswa>;

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_plamo : public llama_model_base {
    llama_model_plamo(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_plamo2 : public llama_model_base {
    llama_model_plamo2(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_build_mamba_base {
        graph(const llama_model & model, const llm_graph_params & params);
        private:
            ggml_tensor * build_plamo2_mamba_layer(llm_graph_input_rs * inp, ggml_tensor * cur, const llama_model & model, const llama_ubatch & ubatch, int il);
            ggml_tensor * build_plamo2_attn_layer(llm_graph_input_attn_kv * inp, ggml_tensor * inp_pos, ggml_tensor * cur,
                                                    const llama_model & model, int il);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_plamo3 : public llama_model_base {
    llama_model_plamo3(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    template <bool iswa>
    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_gpt2 : public llama_model_base {
    llama_model_gpt2(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_pockettts : public llama_model_base {
    llama_model_pockettts(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_codeshell : public llama_model_base {
    llama_model_codeshell(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_orion : public llama_model_base {
    llama_model_orion(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_internlm2 : public llama_model_base {
    llama_model_internlm2(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_minicpm3 : public llama_model_base {
    llama_model_minicpm3(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_gemma : public llama_model_base {
    llama_model_gemma(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_gemma2 : public llama_model_base {
    llama_model_gemma2(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_gemma3 : public llama_model_base {
    llama_model_gemma3(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    template <bool iswa>
    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_gemma3n : public llama_model_base {
    llama_model_gemma3n(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        const llama_model & model;

        const int64_t n_embd_head;
        const int64_t n_embd_altup;
        const int64_t n_altup;
        const int     i_altup_act;
        const int     n_layer_sparsity = 10; // number of layers using activation sparsity
        const float   f_sparsity_std_mul = 1.6448533535003662f; // std_multiplier = normal_dist.icdf(0.95)

        graph(const llama_model & model, const llm_graph_params & params);
        ggml_tensor * calc_magnitude(ggml_tensor * x);

        // TODO: refactor in common "per-layer" functionality [TAG_PER_LAYER]
        ggml_tensor * build_inp_per_layer();
        ggml_tensor * project_per_layer_inputs(ggml_tensor * inp_batch, ggml_tensor * inp_per_layer);

        ggml_tensor * gaussian_topk(ggml_tensor * x);
        ggml_tensor * altup_compute_router_modalities(ggml_tensor * x, int il);
        ggml_tensor * altup_predict(ggml_tensor * cur, int il);
        ggml_tensor * laurel(ggml_tensor * cur, int il);
        ggml_tensor * altup_correct(ggml_tensor * predictions, ggml_tensor * activated, int il);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_gemma4 : public llama_model_base {
    llama_model_gemma4(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    // FR-Spec-style draft-vocab trim (see docs/moe-cache-colibri-notes.md).
    // Populated once at load time from an external sidecar file (never
    // from the GGUF itself - this model instance's own weights are never
    // modified) when llama_frspec_set_pending_vocab_map() was called
    // before this load. Empty (the common case) = no trim, this model's
    // own output.weight is used unmodified at its full size.
    std::vector<int64_t> frspec_d2t_ids;

    struct graph : public llm_graph_context {
        const llama_model & model;

        const int64_t n_embd_per_layer;

        graph(const llama_model & model, const llm_graph_params & params);

        // TODO: refactor in common "per-layer" functionality [TAG_PER_LAYER]
        ggml_tensor * build_inp_per_layer();
        ggml_tensor * project_per_layer_inputs(ggml_tensor * inp_batch, ggml_tensor * inp_per_layer);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_gemma4_assistant : public llama_model_base {
    llama_model_gemma4_assistant(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    // FR-Spec-style draft-vocab trim (see docs/moe-cache-colibri-notes.md).
    // This is the class that actually loads as Gemma-4's MTP draft (GGUF
    // arch string "gemma4-assistant" - a genuinely separate architecture
    // from the "gemma4" trunk, not a mode flag on it), so this is where
    // the trim hint set by llama_frspec_set_pending_vocab_map() actually
    // gets consumed in practice. Empty (the common case) = no trim.
    std::vector<int64_t> frspec_d2t_ids;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_gemma_embedding : public llama_model_base {
    llama_model_gemma_embedding(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_starcoder2 : public llama_model_base {
    llama_model_starcoder2(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_mamba : public llama_model_base {
    llama_model_mamba(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_build_mamba_base {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_mamba2 : public llama_model_base {
    llama_model_mamba2(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    using graph = llama_model_mamba::graph;

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_jamba : public llama_model_base {
    llama_model_jamba(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_build_mamba_base {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_xverse : public llama_model_base {
    llama_model_xverse(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_command_r : public llama_model_base {
    llama_model_command_r(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_cohere2 : public llama_model_base {
    llama_model_cohere2(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_cohere2moe : public llama_model_base {
    llama_model_cohere2moe(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    struct graph_mtp : public llm_graph_context {
        graph_mtp(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_dbrx : public llama_model_base {
    llama_model_dbrx(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_olmo : public llama_model_base {
    llama_model_olmo(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_olmo2 : public llama_model_base {
    llama_model_olmo2(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    template <bool iswa>
    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_olmoe : public llama_model_base {
    llama_model_olmoe(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_muse_glimmer : public llama_model_base {
    llama_model_muse_glimmer(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_openelm : public llama_model_base {
    llama_model_openelm(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_gptneox : public llama_model_base {
    llama_model_gptneox(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_arctic : public llama_model_base {
    llama_model_arctic(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_deepseek : public llama_model_base {
    llama_model_deepseek(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_deepseek2 : public llama_model_base {
    llama_model_deepseek2(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    struct graph_mtp : public llm_graph_context {
        graph_mtp(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_deepseek32 : public llama_model_base {
    llama_model_deepseek32(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    struct graph_mtp : public llm_graph_context {
        graph_mtp(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_deepseek4 : public llama_model_base {
    llama_model_deepseek4(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params) : llm_graph_context(params) {}
        graph(const llama_model & model, const llm_graph_params & params);

        ggml_tensor * build_hc_pre(
                ggml_tensor * x,
                ggml_tensor * hc_fn,
                ggml_tensor * hc_scale,
                ggml_tensor * hc_base,
                ggml_tensor ** post,
                ggml_tensor ** comb,
                int il) const;

        ggml_tensor * build_hc_post(
                ggml_tensor * x,
                ggml_tensor * residual,
                ggml_tensor * post,
                ggml_tensor * comb,
                int il) const;

        ggml_tensor * build_hc_head(
                ggml_tensor * x,
                ggml_tensor * hc_fn,
                ggml_tensor * hc_scale,
                ggml_tensor * hc_base) const;

        ggml_tensor * build_attention(
                const llama_model & model,
                llm_graph_input_dsv4 * inp_dsv4,
                ggml_tensor * cur,
                ggml_tensor * inp_pos,
                int il) const;

        ggml_tensor * build_attention(
                const llama_model & model,
                llm_graph_input_attn_k_iswa * inp_mtp,
                ggml_tensor * cur,
                ggml_tensor * inp_pos,
                int il) const;

        ggml_tensor * build_attention_impl(
                const llama_model & model,
                llm_graph_input_dsv4 * inp_dsv4,
                llm_graph_input_attn_k_iswa * inp_mtp,
                ggml_tensor * cur,
                ggml_tensor * inp_pos,
                int il) const;

        ggml_tensor * build_hca_compressed_kv_from_state(
                ggml_tensor * kv_state,
                ggml_tensor * score_state,
                ggml_tensor * state_read_idxs,
                ggml_tensor * comp_pos,
                ggml_tensor * norm,
                int64_t n_embd_head,
                const char * name,
                int il) const;

        ggml_tensor * build_overlap_compressed_kv_from_state(
                ggml_tensor * kv_state,
                ggml_tensor * score_state,
                ggml_tensor * state_read_idxs,
                ggml_tensor * comp_pos,
                ggml_tensor * norm,
                int64_t ratio,
                int64_t n_embd_head,
                const char * name,
                int il) const;

        ggml_tensor * build_lid_top_k(
                const llama_model & model,
                llm_graph_input_dsv4 * inp_dsv4,
                ggml_tensor * qr,
                ggml_tensor * cur,
                ggml_tensor * inp_pos,
                int il) const;

        ggml_tensor * build_top_k_mask(
                ggml_tensor * kq_mask,
                ggml_tensor * top_k,
                const char * name,
                int il) const;

        ggml_tensor * build_csa_lid_attention(
                const llama_model & model,
                llm_graph_input_dsv4 * inp_dsv4,
                llm_graph_input_dsv4_raw * inp_attn,
                ggml_tensor * q,
                ggml_tensor * kv,
                ggml_tensor * qr,
                ggml_tensor * cur,
                ggml_tensor * inp_pos,
                ggml_tensor * sinks,
                float kq_scale,
                int il) const;

        ggml_tensor * build_hca_attention(
                llm_graph_input_dsv4 * inp_dsv4,
                llm_graph_input_dsv4_raw * inp_attn,
                ggml_tensor * q,
                ggml_tensor * kv,
                ggml_tensor * sinks,
                float kq_scale,
                int il) const;

        ggml_tensor * build_raw_attention(
                llm_graph_input_dsv4_raw * inp_attn,
                ggml_tensor * q,
                ggml_tensor * kv,
                ggml_tensor * sinks,
                float kq_scale,
                int il) const;

        ggml_tensor * build_hc_pre(
                ggml_tensor * x,
                ggml_tensor * weights,
                int il) const;

        ggml_tensor * build_hc_sinkhorn(
                ggml_tensor * comb,
                int il) const;
    };

    struct graph_mtp : public graph {
        graph_mtp(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_deepseek2ocr : public llama_model_base {
    llama_model_deepseek2ocr(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    using graph = llama_model_deepseek2::graph;

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_glm_dsa : public llama_model_base {
    llama_model_glm_dsa(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    struct graph_mtp : public llm_graph_context {
        graph_mtp(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llama_model_eagle3 : public llama_model_base {
    llama_model_eagle3(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    template <bool is_enc>
    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);

        ggml_tensor * build_inp_embd_enc() const;
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_dflash : public llama_model_base {
    llama_model_dflash(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    template <bool is_enc>
    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);

        ggml_tensor * build_inp_embd_enc() const;
    };

    struct graph_dsv4 : public llama_model_deepseek4::graph {
        graph_dsv4(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_mistral4 : public llama_model_deepseek2 {
    llama_model_mistral4(const struct llama_model_params & params) : llama_model_deepseek2(params) {}
    // reuse load_arch_hparams and load_arch_tensors from llama_model_deepseek2

    using graph = llama_model_deepseek2::graph;

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_chatglm : public llama_model_base {
    llama_model_chatglm(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_glm4 : public llama_model_base {
    llama_model_glm4(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_glm4_moe : public llama_model_base {
    llama_model_glm4_moe(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_bitnet : public llama_model_base {
    llama_model_bitnet(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_t5 : public llama_model_base {
    llama_model_t5(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    template <bool is_enc>
    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_t5encoder : public llama_model_base {
    llama_model_t5encoder(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    using graph = llama_model_t5::graph<true>;

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_jais : public llama_model_base {
    llama_model_jais(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_jais2 : public llama_model_base {
    llama_model_jais2(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_nemotron : public llama_model_base {
    llama_model_nemotron(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_nemotron_h : public llama_model_base {
    llama_model_nemotron_h(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_build_mamba_base {
        graph(const llama_model & model, const llm_graph_params & params);
        ggml_tensor * build_ffn_layer(ggml_tensor * cur, const llama_model & model, int il);
        ggml_tensor * build_attention_layer(ggml_tensor * cur, llm_graph_input_attn_kv * inp_attn,
            const llama_model & model, int64_t n_embd_head, int il);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_nemotron_h_moe : public llama_model_nemotron_h {
    llama_model_nemotron_h_moe(const struct llama_model_params & params) : llama_model_nemotron_h(params) {}
    // reuse load_arch_hparams and load_arch_tensors from llama_model_nemotron_h

    using graph = llama_model_nemotron_h::graph;

    struct graph_mtp : public llm_graph_context {
        graph_mtp(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_exaone : public llama_model_base {
    llama_model_exaone(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_exaone4 : public llama_model_base {
    llama_model_exaone4(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    template <bool iswa>
    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_exaone_moe : public llama_model_base {
    llama_model_exaone_moe(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_rwkv6 : public llama_model_base {
    llama_model_rwkv6(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_build_rwkv6_base {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_rwkv6qwen2 : public llama_model_base {
    llama_model_rwkv6qwen2(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_build_rwkv6_base {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_rwkv7 : public llama_model_base {
    llama_model_rwkv7(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_build_rwkv7_base {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_arwkv7 : public llama_model_base {
    llama_model_arwkv7(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_build_rwkv7_base {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_granite : public llama_model_base {
    llama_model_granite(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);

    private:
        ggml_tensor * build_attention_layer(
                  ggml_tensor             * cur,
                  ggml_tensor             * inp_pos,
                  llm_graph_input_attn_kv * inp_attn,
            const llama_model             & model,
            const int64_t                 n_embd_head,
            const int                     il);

        ggml_tensor * build_layer_ffn(
                  ggml_tensor       * cur,
                  ggml_tensor       * inpSA,
            const llama_model       & model,
            const int                 il);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_granite_moe : public llama_model_base {
    llama_model_granite_moe(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    using graph = llama_model_granite::graph;

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_granite_switch : public llama_model_base {
    llama_model_granite_switch(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    uint32_t n_adapters    = 0;
    uint32_t max_lora_rank = 0;
    float    router_gain   = 15.0f;

    std::unordered_map<llama_token, int32_t>     adapter_token_to_slot;
    std::unordered_map<llama_token, llama_token> adapter_token_to_substitute;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);

    private:
        ggml_tensor * build_switched_lora_delta(
                  ggml_tensor * lora_a,
                  ggml_tensor * lora_b,
                  ggml_tensor * cur,
                  ggml_tensor * ids);

        ggml_tensor * build_switched_lora_mm(
                  ggml_tensor * w,
                  ggml_tensor * lora_a,
                  ggml_tensor * lora_b,
                  ggml_tensor * cur,
                  ggml_tensor * ids);

        ggml_tensor * build_attention_layer(
                  ggml_tensor             * cur,
                  ggml_tensor             * inp_pos,
                  ggml_tensor             * adapter_ids,
                  llm_graph_input_attn_kv * inp_attn,
            const llama_model             & model,
            const int64_t                 n_embd_head,
            const int                     il);

        ggml_tensor * build_layer_ffn(
                  ggml_tensor       * cur,
                  ggml_tensor       * inpSA,
                  ggml_tensor       * adapter_ids,
            const llama_model       & model,
            const int                 il);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_minicpm : public llama_model_base {
    llama_model_minicpm(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    using graph = llama_model_granite::graph;

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_granite_hybrid : public llama_model_base {
    llama_model_granite_hybrid(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_build_mamba_base {
        graph(const llama_model & model, const llm_graph_params & params);
        ggml_tensor * build_layer_ffn(ggml_tensor * cur, ggml_tensor * inpSA, const llama_model & model, const int il);
        ggml_tensor * build_attention_layer(ggml_tensor * cur, ggml_tensor * inp_pos, llm_graph_input_attn_kv * inp_attn,
            const llama_model & model,const int64_t n_embd_head, const int il);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_chameleon : public llama_model_base {
    llama_model_chameleon(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_wavtokenizer_dec : public llama_model_base {
    llama_model_wavtokenizer_dec(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_plm : public llama_model_base {
    llama_model_plm(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_bailingmoe : public llama_model_base {
    llama_model_bailingmoe(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_bailingmoe2 : public llama_model_base {
    llama_model_bailingmoe2(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_seed_oss : public llama_model_base {
    llama_model_seed_oss(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_dots1 : public llama_model_base {
    llama_model_dots1(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_arcee : public llama_model_base {
    llama_model_arcee(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_afmoe : public llama_model_base {
    llama_model_afmoe(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_laguna : public llama_model_base {
    llama_model_laguna(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_ernie4_5 : public llama_model_base {
    llama_model_ernie4_5(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_ernie4_5_moe : public llama_model_ernie4_5 {
    llama_model_ernie4_5_moe(const struct llama_model_params & params) : llama_model_ernie4_5(params) {}
    // reuse load_arch_hparams and load_arch_tensors from llama_model_ernie4_5

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_paddleocr : public llama_model_ernie4_5 {
    llama_model_paddleocr(const struct llama_model_params & params) : llama_model_ernie4_5(params) {}
    // reuse load_arch_hparams and load_arch_tensors from llama_model_ernie4_5

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_hunyuan_moe : public llama_model_base {
    llama_model_hunyuan_moe(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llama_model_hy_v3 : public llama_model_base {
    llama_model_hy_v3(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    struct graph_mtp : public llm_graph_context {
        graph_mtp(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_hunyuan_vl : public llama_model_base {
    llama_model_hunyuan_vl(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_hunyuan_dense : public llama_model_hunyuan_vl {
    llama_model_hunyuan_dense(const struct llama_model_params & params) : llama_model_hunyuan_vl(params) {}
    // reuse load_arch_hparams and load_arch_tensors from llama_model_hunyuan_vl

    using graph = llama_model_hunyuan_vl::graph;

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_smollm3 : public llama_model_base {
    llama_model_smollm3(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_openai_moe : public llama_model_base {
    llama_model_openai_moe(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_falcon_h1 : public llama_model_base {
    llama_model_falcon_h1(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_build_mamba_base {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_lfm2 : public llama_model_base {
    llama_model_lfm2(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    template <bool iswa>
    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_lfm2moe : public llama_model_base {
    llama_model_lfm2moe(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    template <bool iswa>
    using graph = llama_model_lfm2::graph<iswa>;

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_smallthinker : public llama_model_base {
    llama_model_smallthinker(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    template <bool iswa>
    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_grovemoe : public llama_model_base {
    llama_model_grovemoe(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_apertus : public llama_model_base {
    llama_model_apertus(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_minimax_m2 : public llama_model_base {
    llama_model_minimax_m2(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct msa_params {
    int blk;
    int topk_blocks;
    int local;
};

struct llama_model_minimax_m3 : public llama_model_base {
    llama_model_minimax_m3(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;
    msa_params msa_p;
    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);

        ggml_tensor * build_attn_msa_fa(
                ggml_tensor * q_cur,   // [D, HQ, S] f32
                ggml_tensor * k,       // [D, n_keys, 1, C]  C = HKV or HKV*n_stream
                ggml_tensor * v,       // [D, n_keys, 1, C]
                ggml_tensor * mask,    // [n_keys, R, 1, C] f16, R = HQ*T/(Gp*C)
                int64_t Gp, float kq_scale, int il) const;
    };
    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llama_model_cogvlm : public llama_model_base {
    llama_model_cogvlm(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_pangu_embed : public llama_model_base {
    llama_model_pangu_embed(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_qwen3next : public llama_model_base {
    llama_model_qwen3next(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_build_delta_net_base {
        graph(const llama_model & model, const llm_graph_params & params);
    private:
        ggml_tensor * build_layer_attn(
        llm_graph_input_attn_kv * inp_attn,
                    ggml_tensor * cur,
                    ggml_tensor * inp_pos,
                            int   il);

        ggml_tensor * build_layer_attn_linear(
             llm_graph_input_rs * inp,
                    ggml_tensor * cur,
                            int   il);

        ggml_tensor * build_layer_ffn(
                    ggml_tensor * cur,
                            int   il);

        ggml_tensor * build_norm_gated(
                    ggml_tensor * input,
                    ggml_tensor * weights,
                    ggml_tensor * gate,
                            int   layer);

        // returns pair of qkv, z
        std::pair<ggml_tensor *, ggml_tensor *> build_qkvz(
                    ggml_tensor * input,
                            int   il);

        const llama_model & model;
    };

    struct graph_mtp : public llm_graph_context {
        graph_mtp(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_qwen35 : public llama_model_base {
    llama_model_qwen35(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_build_delta_net_base {
        graph(const llama_model & model, const llm_graph_params & params);
    private:
        ggml_tensor * build_layer_attn(
        llm_graph_input_attn_kv * inp_attn,
                    ggml_tensor * cur,
                    ggml_tensor * inp_pos,
                            int * sections,
                            int   il);

        ggml_tensor * build_layer_attn_linear(
             llm_graph_input_rs * inp,
                    ggml_tensor * cur,
                            int   il);

        ggml_tensor * build_layer_ffn(
                    ggml_tensor * cur,
                            int   il);

        ggml_tensor * build_norm_gated(
                    ggml_tensor * input,
                    ggml_tensor * weights,
                    ggml_tensor * gate,
                            int   layer);

        // returns pair of qkv, z
        std::pair<ggml_tensor *, ggml_tensor *> build_qkvz(
                    ggml_tensor * input,
                            int   il);

        const llama_model & model;
    };

    struct graph_mtp : public llm_graph_context {
        graph_mtp(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_qwen35moe : public llama_model_base {
    llama_model_qwen35moe(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_build_delta_net_base {
        graph(const llama_model & model, const llm_graph_params & params);
    private:
        ggml_tensor * build_layer_attn(
        llm_graph_input_attn_kv * inp_attn,
                    ggml_tensor * cur,
                    ggml_tensor * inp_pos,
                            int * sections,
                            int   il);

        ggml_tensor * build_layer_attn_linear(
             llm_graph_input_rs * inp,
                    ggml_tensor * cur,
                            int   il);

        ggml_tensor * build_layer_ffn(
                    ggml_tensor * cur,
                            int   il);

        ggml_tensor * build_norm_gated(
                    ggml_tensor * input,
                    ggml_tensor * weights,
                    ggml_tensor * gate,
                            int   layer);

        // returns pair of qkv, z
        std::pair<ggml_tensor *, ggml_tensor *> build_qkvz(
                    ggml_tensor * input,
                            int   il);

        const llama_model & model;
    };

    struct graph_mtp : public llm_graph_context {
        graph_mtp(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_mistral3 : public llama_model_base {
    llama_model_mistral3(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_mimo2 : public llama_model_base {
    llama_model_mimo2(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    struct graph_mtp : public llm_graph_context {
        graph_mtp(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_kimi_linear : public llama_model_base {
    llama_model_kimi_linear(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_build_delta_net_base {
        graph(const llama_model & model, const llm_graph_params & params);

        std::pair<ggml_tensor *, ggml_tensor *> build_kda_autoregressive(
                    ggml_tensor * q,
                    ggml_tensor * k,
                    ggml_tensor * v,
                    ggml_tensor * gk,
                    ggml_tensor * beta,
                    ggml_tensor * state,
                            int   il);

        std::pair<ggml_tensor *, ggml_tensor *> build_kda_chunking(
                    ggml_tensor * q,
                    ggml_tensor * k,
                    ggml_tensor * v,
                    ggml_tensor * gk,
                    ggml_tensor * beta,
                    ggml_tensor * state,
                    ggml_tensor * causal_mask,
                    ggml_tensor * identity,
                    ggml_tensor * diag_mask,
                            int   il);

        const llama_model & model;
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};


struct llama_model_step35 : public llama_model_base {
    llama_model_step35(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };

    struct graph_mtp : public llm_graph_context {
        graph_mtp(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};
