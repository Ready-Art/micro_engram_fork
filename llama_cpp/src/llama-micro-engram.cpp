#include "llama-micro-engram.h"

#include "llama-batch.h"
#include "llama-graph.h"
#include "llama-impl.h"
#include "llama-kv-cache.h"
#include "llama-kv-cache-iswa.h"
#include "llama-model-loader.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

static const uint64_t MENGRAM_PRIME = 2147483647ull; // 2^31 - 1

// torch F.rms_norm default eps for fp32
static const float MENGRAM_RMS_EPS = 1.1920929e-07f;

// read with the gguf API: the loader's string-key getters are not instantiated for these types
template <typename T>
static bool mengram_arr(const gguf_context * meta, const char * key, gguf_type type, std::vector<T> & out, bool required = true) {
    const int64_t id = gguf_find_key(meta, key);
    if (id < 0) {
        if (required) {
            throw std::runtime_error(format("micro_engram: missing key %s", key));
        }
        return false;
    }
    if (gguf_get_kv_type(meta, id) != GGUF_TYPE_ARRAY || gguf_get_arr_type(meta, id) != type) {
        throw std::runtime_error(format("micro_engram: key %s has the wrong type", key));
    }
    const T * data = (const T *) gguf_get_arr_data(meta, id);
    out.assign(data, data + gguf_get_arr_n(meta, id));
    return true;
}

int llama_micro_engram::index_of(int il) const {
    for (size_t e = 0; e < engrams.size(); ++e) {
        if (engrams[e].layer == il) {
            return (int) e;
        }
    }
    return -1;
}

void llama_micro_engram::load_hparams(llama_model_loader & ml, uint32_t n_layer) {
    const gguf_context * meta = ml.metadata;

    std::vector<uint32_t> layers;
    if (!mengram_arr(meta, "micro_engram.layers", GGUF_TYPE_UINT32, layers, false) || layers.empty()) {
        return;
    }

    std::vector<uint32_t> table_size, dim, n_tables, gate, orders;
    std::vector<int32_t>  pad_id;
    std::vector<uint64_t> mults;
    mengram_arr(meta, "micro_engram.table_size",   GGUF_TYPE_UINT32, table_size);
    mengram_arr(meta, "micro_engram.dim",          GGUF_TYPE_UINT32, dim);
    mengram_arr(meta, "micro_engram.n_tables",     GGUF_TYPE_UINT32, n_tables);
    mengram_arr(meta, "micro_engram.gate",         GGUF_TYPE_UINT32, gate);
    mengram_arr(meta, "micro_engram.pad_id",       GGUF_TYPE_INT32,  pad_id);
    mengram_arr(meta, "micro_engram.table_orders", GGUF_TYPE_UINT32, orders);
    mengram_arr(meta, "micro_engram.mults",        GGUF_TYPE_UINT64, mults);
    mengram_arr(meta, "micro_engram.canon",        GGUF_TYPE_INT32,  canon);

    const int64_t id_order = gguf_find_key(meta, "micro_engram.max_order");
    if (id_order < 0 || gguf_get_kv_type(meta, id_order) != GGUF_TYPE_UINT32) {
        throw std::runtime_error("micro_engram: missing key micro_engram.max_order");
    }
    max_order = gguf_get_val_u32(meta, id_order);

    const size_t n_e = layers.size();
    if (table_size.size() != n_e || dim.size() != n_e || n_tables.size() != n_e || gate.size() != n_e ||
            pad_id.size() != n_e || canon.empty() || max_order < 1) {
        throw std::runtime_error("micro_engram metadata is inconsistent");
    }

    size_t t_off = 0;
    for (size_t i = 0; i < n_e; ++i) {
        engram e;
        e.layer      = (int32_t) layers[i];
        e.table_size = table_size[i];
        e.dim        = dim[i];
        e.n_tables   = n_tables[i];
        e.gate       = (gate_type) gate[i];
        e.pad_id     = pad_id[i];

        if (layers[i] >= n_layer || (i > 0 && layers[i] <= layers[i - 1])) {
            throw std::runtime_error(format("micro_engram layer %u is out of range or out of order", layers[i]));
        }
        if (e.table_size == 0 || e.dim == 0 || e.n_tables == 0 || gate[i] > GATE_PLE ||
                (uint64_t) e.table_size*e.n_tables > (uint64_t) INT32_MAX) {
            throw std::runtime_error(format("micro_engram layer %u: bad table settings", layers[i]));
        }
        if (orders.size() < t_off + e.n_tables || mults.size() < (t_off + e.n_tables)*max_order) {
            throw std::runtime_error("micro_engram metadata is inconsistent");
        }
        e.orders.assign(orders.begin() + t_off, orders.begin() + t_off + e.n_tables);
        e.mults.assign(mults.begin() + t_off*max_order, mults.begin() + (t_off + e.n_tables)*max_order);
        for (uint32_t n : e.orders) {
            if (n < 1 || n > max_order) {
                throw std::runtime_error(format("micro_engram layer %u: bad n-gram order %u", layers[i], n));
            }
        }
        t_off += e.n_tables;
        engrams.push_back(std::move(e));
    }

    LLAMA_LOG_INFO("%s: micro_engram: %zu engram(s), max order %u, %zu canonical token ids\n", __func__, n_e, max_order, canon.size());
}

void llama_micro_engram::create_tensors(const create_fn & create, int64_t n_embd) {
    for (auto & e : engrams) {
        const int il = e.layer;
        e.tables = create(false, "mengram_tables", il, {(int64_t) e.dim, (int64_t) e.n_tables*e.table_size});
        if (e.gate == GATE_PLE) {
            e.down = create(false, "mengram_down", il, {n_embd, e.mem_dim()});
            e.up   = create(false, "mengram_up",   il, {e.mem_dim(), n_embd});
        } else {
            e.key   = create(false, "mengram_key",   il, {e.mem_dim(), n_embd});
            e.value = create(false, "mengram_value", il, {e.mem_dim(), n_embd});
            if (e.gate == GATE_SCALAR_SQRT) {
                e.q_scale = create(true, "mengram_q_scale", il, {n_embd});
                e.k_scale = create(true, "mengram_k_scale", il, {n_embd});
            }
        }
    }
}

// same as MicroEngram.hash_ids: canonical ids, missing earlier tokens = raw pad id, sum of id*mult per table
void llama_micro_engram::hash(const engram & e, const llama_ubatch & ubatch, const std::vector<llama_token> & prev, std::vector<int32_t> & out) const {
    const uint32_t n_prev = max_order - 1;
    const int64_t  n_cano = (int64_t) canon.size();

    const auto cid = [&](llama_token t) -> uint64_t {
        return (uint64_t) canon[std::clamp<int64_t>(t, 0, n_cano - 1)];
    };

    out.resize((size_t) ubatch.n_tokens*e.n_tables);
    std::vector<uint64_t> toks(max_order);
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        const bool is_embd = !ubatch.token || (ubatch.is_mixed() && ubatch.type[i]);
        toks[0] = is_embd ? (uint64_t) e.pad_id : cid(ubatch.token[i]);
        for (uint32_t s = 1; s < max_order; ++s) {
            const llama_token t = prev[(size_t) i*n_prev + (n_prev - s)];
            toks[s] = t == LLAMA_TOKEN_NULL ? (uint64_t) e.pad_id : cid(t);
        }
        for (uint32_t k = 0; k < e.n_tables; ++k) {
            // ids < 2^31 and mults < 2^31, so the sum fits in 64 bits
            uint64_t x = 0;
            for (uint32_t s = 0; s < e.orders[k]; ++s) {
                x += toks[s]*e.mults[(size_t) k*max_order + s];
            }
            out[(size_t) i*e.n_tables + k] = (int32_t) ((uint64_t) k*e.table_size + (x % MENGRAM_PRIME) % e.table_size);
        }
    }
}

static const llama_kv_cache_context * mengram_kv_base(const llama_memory_context_i * mctx) {
    if (const auto * c = dynamic_cast<const llama_kv_cache_iswa_context *>(mctx)) {
        return c->get_base();
    }
    return dynamic_cast<const llama_kv_cache_context *>(mctx);
}

class llm_graph_input_micro_engram : public llm_graph_input_i {
public:
    llm_graph_input_micro_engram(const llama_micro_engram & me, const llama_memory_context_i * mctx) : me(me), mctx(mctx) {}

    void set_input(const llama_ubatch * ubatch) override {
        if (ids.empty()) {
            return;
        }
        const llama_kv_cache_context * kv = mengram_kv_base(mctx);
        GGML_ASSERT(kv != nullptr && "micro_engram needs a KV cache for the token history");

        std::vector<llama_token> prev;
        kv->get_prev_tokens(*ubatch, me.max_order - 1, prev);

        std::vector<int32_t> rows;
        for (size_t e = 0; e < ids.size(); ++e) {
            me.hash(me.engrams[e], *ubatch, prev, rows);
            if (ids[e]->buffer) {
                ggml_backend_tensor_set(ids[e], rows.data(), 0, rows.size()*sizeof(int32_t));
            }
        }
    }

    bool can_reuse(const llm_graph_params & params) override {
        mctx = params.mctx;
        if (!params.ubatch.token) {
            return ids.empty();
        }
        if (ids.size() != me.engrams.size()) {
            return false;
        }
        for (size_t e = 0; e < ids.size(); ++e) {
            if (ids[e]->ne[0] != (int64_t) me.engrams[e].n_tables*params.ubatch.n_tokens) {
                return false;
            }
        }
        return true;
    }

    std::vector<ggml_tensor *> ids; // I32 [n_tables*n_tokens] per engram

    const llama_micro_engram & me;
    const llama_memory_context_i * mctx;
};

llm_micro_engram_graph::llm_micro_engram_graph(llm_graph_context & g, const llama_micro_engram & me) : g(g), me(me) {
    if (!me.enabled()) {
        return;
    }
    auto inp = std::make_unique<llm_graph_input_micro_engram>(me, g.mctx);
    // embedding (image) input has no token ids: no engram, as the gathered rows would be meaningless
    if (g.ubatch.token) {
        for (const auto & e : me.engrams) {
            ggml_tensor * ids = ggml_new_tensor_1d(g.ctx0, GGML_TYPE_I32, (int64_t) e.n_tables*g.n_tokens);
            ggml_set_name(ids, ("mengram_ids-" + std::to_string(e.layer)).c_str());
            ggml_set_input(ids);
            inp->ids.push_back(ids);

            ggml_tensor * rows = ggml_get_rows(g.ctx0, e.tables, ids);
            rows = ggml_reshape_2d(g.ctx0, rows, e.mem_dim(), g.n_tokens);
            g.cb(rows, "mengram_rows", e.layer);
            mem.push_back(rows);
        }
    }
    g.res->add_input(std::move(inp));
}

ggml_tensor * llm_micro_engram_graph::inject(ggml_tensor * cur, int il, ggml_tensor * inp_out_ids) {
    const int ie = me.index_of(il);
    if (ie < 0 || mem.empty()) {
        return cur;
    }
    const auto & e = me.engrams[ie];
    ggml_context * ctx0 = g.ctx0;

    ggml_tensor * m = mem[ie];
    if (m->ne[1] != cur->ne[1]) {
        // the layer works on pruned rows (outputs only)
        m = ggml_get_rows(ctx0, m, inp_out_ids);
    }

    ggml_tensor * h = ggml_rms_norm(ctx0, cur, MENGRAM_RMS_EPS);
    ggml_tensor * out;
    if (e.gate == llama_micro_engram::GATE_PLE) {
        ggml_tensor * gt = ggml_gelu(ctx0, ggml_mul_mat(ctx0, e.down, h));
        out = ggml_mul_mat(ctx0, e.up, ggml_mul(ctx0, gt, m));
    } else {
        ggml_tensor * k = ggml_rms_norm(ctx0, ggml_mul_mat(ctx0, e.key, m), MENGRAM_RMS_EPS);
        if (e.gate == llama_micro_engram::GATE_SCALAR_SQRT) {
            h = ggml_mul(ctx0, h, e.q_scale);
            k = ggml_mul(ctx0, k, e.k_scale);
        }
        ggml_tensor * s = ggml_scale(ctx0, ggml_sum_rows(ctx0, ggml_mul(ctx0, h, k)), 1.0f/sqrtf((float) cur->ne[0]));
        if (e.gate == llama_micro_engram::GATE_SCALAR_SQRT) {
            // sign(s) * sqrt(max(|s|, 1e-6))
            s = ggml_mul(ctx0, ggml_sgn(ctx0, s), ggml_sqrt(ctx0, ggml_clamp(ctx0, ggml_abs(ctx0, s), 1e-6f, INFINITY)));
        }
        g.cb(s, "mengram_score", il);
        out = ggml_mul(ctx0, ggml_mul_mat(ctx0, e.value, m), ggml_sigmoid(ctx0, s));
    }
    g.cb(out, "mengram_out", il);

    return ggml_add(ctx0, cur, out);
}
