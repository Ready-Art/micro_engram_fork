// Logits of a token sequence, for comparing llama.cpp + micro_engram with the PyTorch reference.
//
//   engram_logits MODEL.gguf OUT.bin MODE N_UBATCH N_SPLIT TOK...
//
// MODE:
//   full   - all tokens in one llama_decode (split into ubatches of N_UBATCH)
//   decode - first N_SPLIT tokens in one call, then one token per call
//   two    - two sequences in one batch: tokens [0, N_SPLIT) are seq 0, the rest seq 1
//   state  - like decode, but seq 0 is saved, the cache cleared and the seq restored before the rest
// OUT.bin: float32 logits [n_tokens, n_vocab], in token order (seq 0 then seq 1 for "two")

#include "llama.h"
#include "ggml-backend.h"

#include <cmath>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static void log_cb(enum ggml_log_level level, const char * text, void *) {
    if (level >= GGML_LOG_LEVEL_WARN || getenv("ENGRAM_VERBOSE")) {
        fputs(text, stderr);
    }
}

// ENGRAM_DUMP=1: print the first token's column of every l_out-N / result_norm tensor (for debugging)
static bool dump_cb(struct ggml_tensor * t, bool ask, void *) {
    const std::string name = t->name;
    const bool want = name.rfind("l_out-", 0) == 0 || name == "result_norm" || name.rfind("inp_scaled", 0) == 0 ||
                      name.rfind("attn_out-", 0) == 0 || name.rfind("ffn_out-", 0) == 0 || name.rfind("mengram_out-", 0) == 0;
    if (ask) {
        return want;
    }
    if (want && t->type == GGML_TYPE_F32) {
        std::vector<float> col(t->ne[0]);
        ggml_backend_tensor_get(t, col.data(), 0, col.size()*sizeof(float));
        double ss = 0;
        for (float v : col) {
            ss += (double) v*v;
        }
        fprintf(stderr, "DUMP %-16s norm %.6e first %.6e %.6e %.6e\n", name.c_str(), sqrt(ss), col[0], col[1], col[2]);
    }
    return true;
}

struct runner {
    llama_context * ctx;
    int n_vocab;
    std::vector<float> out;

    // decode toks[i] at pos0+i for seq; keep the logits of every token
    void run(const std::vector<llama_token> & toks, const std::vector<llama_seq_id> & seqs, const std::vector<llama_pos> & pos) {
        llama_batch b = llama_batch_init((int32_t) toks.size(), 0, 1);
        for (size_t i = 0; i < toks.size(); ++i) {
            b.token[i]     = toks[i];
            b.pos[i]       = pos[i];
            b.n_seq_id[i]  = 1;
            b.seq_id[i][0] = seqs[i];
            b.logits[i]    = 1;
        }
        b.n_tokens = (int32_t) toks.size();
        if (llama_decode(ctx, b) != 0) {
            fprintf(stderr, "llama_decode failed\n");
            exit(1);
        }
        for (size_t i = 0; i < toks.size(); ++i) {
            const float * l = llama_get_logits_ith(ctx, (int32_t) i);
            out.insert(out.end(), l, l + n_vocab);
        }
        llama_batch_free(b);
    }

    void run_seq(const std::vector<llama_token> & toks, size_t begin, size_t end, llama_seq_id seq) {
        std::vector<llama_token> t(toks.begin() + begin, toks.begin() + end);
        std::vector<llama_seq_id> s(t.size(), seq);
        std::vector<llama_pos> p;
        for (size_t i = begin; i < end; ++i) {
            p.push_back((llama_pos) i);
        }
        run(t, s, p);
    }
};

int main(int argc, char ** argv) {
    if (argc < 7) {
        fprintf(stderr, "usage: %s MODEL OUT MODE N_UBATCH N_SPLIT TOK...\n", argv[0]);
        return 1;
    }
    const std::string mode = argv[3];
    const int n_ubatch = atoi(argv[4]);
    const size_t n_split = (size_t) atoi(argv[5]);
    std::vector<llama_token> toks;
    for (int i = 6; i < argc; ++i) {
        toks.push_back(atoi(argv[i]));
    }

    llama_log_set(log_cb, nullptr);
    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) {
        fprintf(stderr, "failed to load %s\n", argv[1]);
        return 1;
    }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx      = 1024;
    cp.n_batch    = 1024;
    cp.n_ubatch   = n_ubatch;
    cp.n_seq_max  = 2;
    cp.kv_unified = true;
    cp.type_k     = GGML_TYPE_F32; // f16 cache rounding would hide small errors
    cp.type_v     = GGML_TYPE_F32;
    if (getenv("ENGRAM_DUMP")) {
        cp.cb_eval = dump_cb;
    }
    llama_context * ctx = llama_init_from_model(model, cp);

    runner r{ctx, llama_vocab_n_tokens(llama_model_get_vocab(model)), {}};

    if (mode == "full") {
        r.run_seq(toks, 0, toks.size(), 0);
    } else if (mode == "decode" || mode == "state") {
        r.run_seq(toks, 0, n_split, 0);
        if (mode == "state") {
            std::vector<uint8_t> buf(llama_state_seq_get_size(ctx, 0));
            llama_state_seq_get_data(ctx, buf.data(), buf.size(), 0);
            llama_memory_clear(llama_get_memory(ctx), true);
            if (llama_state_seq_set_data(ctx, buf.data(), buf.size(), 0) == 0) {
                fprintf(stderr, "state restore failed\n");
                return 1;
            }
        }
        for (size_t i = n_split; i < toks.size(); ++i) {
            r.run_seq(toks, i, i + 1, 0);
        }
    } else if (mode == "two") {
        std::vector<llama_seq_id> s;
        std::vector<llama_pos> p;
        for (size_t i = 0; i < toks.size(); ++i) {
            s.push_back(i < n_split ? 0 : 1);
            p.push_back((llama_pos) (i < n_split ? i : i - n_split));
        }
        r.run(toks, s, p);
    } else {
        fprintf(stderr, "unknown mode %s\n", mode.c_str());
        return 1;
    }

    FILE * f = fopen(argv[2], "wb");
    fwrite(r.out.data(), sizeof(float), r.out.size(), f);
    fclose(f);

    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
