#pragma once

// micro-engram: hashed n-gram memory added to the output of some decoder layers
// ref: https://github.com/som1tokmynam/micro_engram (micro_engram/module.py)

#include "llama.h"

#include <cstdint>
#include <functional>
#include <initializer_list>
#include <vector>

struct ggml_tensor;
struct llama_hparams;
struct llama_ubatch;
struct llm_graph_context;
struct llama_model_loader;
class  llm_graph_input_micro_engram;

struct llama_micro_engram {
    enum gate_type : uint32_t {
        GATE_SCALAR      = 0, // sigmoid(rms(h) . rms(W_K e) / sqrt(d))
        GATE_SCALAR_SQRT = 1, // same with learned norm scales and a signed sqrt on the score
        GATE_PLE         = 2, // W_up (gelu(W_down rms(h)) * e)
    };

    struct engram {
        int32_t   layer      = -1;
        uint32_t  table_size = 0;
        uint32_t  dim        = 0;
        uint32_t  n_tables   = 0;
        gate_type gate       = GATE_SCALAR;
        int64_t   pad_id     = 0;

        std::vector<uint32_t> orders; // [n_tables] n-gram order of each table
        std::vector<uint64_t> mults;  // [n_tables*max_order]

        ggml_tensor * tables  = nullptr; // [dim, n_tables*table_size]
        ggml_tensor * key     = nullptr; // [n_tables*dim, n_embd]
        ggml_tensor * value   = nullptr; // [n_tables*dim, n_embd]
        ggml_tensor * q_scale = nullptr; // [n_embd]
        ggml_tensor * k_scale = nullptr; // [n_embd]
        ggml_tensor * down    = nullptr; // [n_embd, n_tables*dim]
        ggml_tensor * up      = nullptr; // [n_tables*dim, n_embd]

        int64_t mem_dim() const { return (int64_t) n_tables*dim; }
    };

    std::vector<engram>  engrams;   // in layer order
    std::vector<int32_t> canon;     // token id -> canonical id (token merging)
    uint32_t             max_order = 0;

    bool enabled() const { return !engrams.empty(); }

    // index into engrams for layer il, or -1
    int index_of(int il) const;

    // read the micro_engram.* metadata; no metadata = disabled
    void load_hparams(llama_model_loader & ml, uint32_t n_layer);

    using create_fn = std::function<ggml_tensor * (bool is_vec, const char * suffix, int il, std::initializer_list<int64_t> ne)>;
    void create_tensors(const create_fn & create, int64_t n_embd);

    // rows to gather for every token: out[i*n_tables + t] = t*table_size + row
    void hash(const engram & e, const llama_ubatch & ubatch, const std::vector<llama_token> & prev, std::vector<int32_t> & out) const;
};

// per-graph state: the gathered memory of each engram
struct llm_micro_engram_graph {
    llm_micro_engram_graph(llm_graph_context & g, const llama_micro_engram & me);

    // add the gated memory of the engram at layer il to cur (no-op for other layers)
    ggml_tensor * inject(ggml_tensor * cur, int il, ggml_tensor * inp_out_ids);

private:
    llm_graph_context & g;
    const llama_micro_engram & me;
    std::vector<ggml_tensor *> mem; // [n_tables*dim, n_tokens] per engram
    ggml_tensor * keep = nullptr;   // F32 [1, n_tokens], 0 for embedding rows (mixed ubatch only)
};
