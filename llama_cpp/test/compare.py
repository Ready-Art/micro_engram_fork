"""
End-to-end check of the llama.cpp micro_engram patch against the PyTorch engram (CPU).

    python llama_cpp/test/compare.py --llama-cpp /path/to/patched/llama.cpp \\
        --driver /path/to/engram_logits --work /tmp/engram_test

Builds a tiny random Gemma 4 (transformers) plus random engrams, converts it with
convert_hf_to_gguf.py (no tokenizer: token ids are fed directly), adds the engrams with
gguf_add_engram.py, then compares llama.cpp logits with the reference (HF model +
infer_engram.EngramRunner, the same code as training) for: one batch, small ubatches,
prefill + token-by-token decode, two sequences in one batch, and a state save/restore.
Run from the micro_engram repo root.

Build first: apply llama_cpp/apply.sh to a llama.cpp checkout, build libllama on CPU
(cmake -B build -DGGML_CUDA=OFF && cmake --build build --target llama), then the driver:
    g++ -O2 -std=c++17 -I$L/include -I$L/ggml/include llama_cpp/test/engram_logits.cpp \
        -L$L/build/bin -lllama -lggml -lggml-base -Wl,-rpath,$L/build/bin -o engram_logits
Needs torch, transformers (with Gemma 4), safetensors and numpy.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0, os.getcwd())
from infer_engram import EngramRunner  # noqa: E402
from micro_engram.module import MicroEngram  # noqa: E402
from micro_engram.weights import default_seed  # noqa: E402

V, D, NL = 512, 64, 6
LAYER_TYPES = ["sliding_attention"] * 5 + ["full_attention"]


def make_model(out_dir):
    from transformers import Gemma4ForCausalLM, Gemma4TextConfig

    cfg = Gemma4TextConfig(vocab_size=V, hidden_size=D, intermediate_size=128, num_hidden_layers=NL,
                           num_attention_heads=4, num_key_value_heads=2, head_dim=16, global_head_dim=32,
                           layer_types=LAYER_TYPES, sliding_window=8, hidden_size_per_layer_input=0,
                           vocab_size_per_layer_input=V, max_position_embeddings=1024,
                           final_logit_softcapping=30.0, initializer_range=0.08)
    cfg.architectures = ["Gemma4ForCausalLM"]
    torch.manual_seed(0)
    model = Gemma4ForCausalLM(cfg).float().eval()
    with torch.no_grad():  # norms start at 1/0 in HF; randomize so they matter
        for n, p in model.named_parameters():
            if p.dim() == 1:
                p.copy_(1 + 0.2 * torch.randn_like(p))
    model.save_pretrained(out_dir, safe_serialization=True)
    # the Gemma converter only counts the vocab from this file
    with open(os.path.join(out_dir, "tokenizer.json"), "w") as f:
        json.dump({"model": {"vocab": {f"t{i}": i for i in range(V)}}, "added_tokens": []}, f)
    return model


def make_engram(layer, gate, seed_shift=0, table_sizes=None):
    canon = torch.arange(V) & ~1  # merge every odd id into the even one below (idempotent)
    eng = MicroEngram(model_dim=D, table_size=97, engram_dim=8, num_heads=2, pad_id=0, seed=default_seed(layer),
                      vocab_size=V, canon=canon, gate=gate, table_sizes=table_sizes)
    g = torch.Generator().manual_seed(100 + layer + seed_shift)
    with torch.no_grad():
        for n, p in eng.named_parameters():
            if n.startswith("tables."):
                p.copy_(torch.randn(p.shape, generator=g) * 0.5)
            elif n in ("q_scale", "k_scale"):
                p.copy_(0.5 + torch.rand(p.shape, generator=g))
            else:
                p.copy_(torch.randn(p.shape, generator=g) * 0.3)
    return eng


def save_engrams(engrams, out_dir):
    entries = []
    for layer, eng in engrams:
        name = f"engram_L{layer}.pt"
        torch.save(eng.state_dict(), os.path.join(out_dir, name))
        entries.append({"layer": layer, "table_size": eng.table_size, "table_sizes": dict(eng.order_sizes),
                        "engram_dim": eng.tables[0].embedding_dim,
                        "num_heads": len(eng.tables) // len(eng.orders), "orders": list(eng.orders),
                        "gate": eng.gate, "seed": default_seed(layer), "pad_id": 0, "token_merge": True,
                        "weights": name})
    conf = {"model_dim": D, "vocab_size": V, **{k: v for k, v in entries[0].items() if k != "weights"},
            "engrams": entries}
    with open(os.path.join(out_dir, "micro_engram_config.json"), "w") as f:
        json.dump(conf, f, indent=2)


def run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-3000:])
        raise SystemExit(f"failed: {' '.join(map(str, cmd[:4]))} ...")
    return r


def convert(hf_dir, out, llama_cpp):
    code = f"""
import sys
sys.path.insert(0, {llama_cpp!r}); sys.path.insert(1, {os.path.join(llama_cpp, 'gguf-py')!r})
sys.argv = ["convert_hf_to_gguf.py", {hf_dir!r}, "--outfile", {out!r}, "--outtype", "f32"]
import convert_hf_to_gguf as c
import conversion.gemma as gm
def set_vocab(self):  # no tokenizer in the test: token ids are fed directly
    self.gguf_writer.add_tokenizer_model("no_vocab")
    self.gguf_writer.add_vocab_size(self.hparams["vocab_size"])
gm.Gemma4Model.set_vocab = set_vocab
c.main()
"""
    run([sys.executable, "-c", code])


def llama_logits(driver, gguf, mode, ubatch, split, toks, work):
    out = os.path.join(work, "logits.bin")
    run([driver, gguf, out, mode, str(ubatch), str(split)] + [str(t) for t in toks])
    return np.fromfile(out, dtype=np.float32).reshape(-1, V)


def reference(model, engrams, toks):
    runner = EngramRunner(model, [(eng, layer) for layer, eng in engrams]) if engrams else None
    with torch.no_grad():
        if runner:
            runner.reset()
        logits = model(input_ids=torch.tensor([toks])).logits[0].float().numpy()
    return logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llama-cpp", required=True)
    ap.add_argument("--driver", required=True)
    ap.add_argument("--work", required=True)
    args = ap.parse_args()
    os.makedirs(args.work, exist_ok=True)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([os.getcwd(), os.path.join(args.llama_cpp, "gguf-py")])}

    from transformers import Gemma4ForCausalLM

    hf_dir = os.path.join(args.work, "tiny_hf")
    make_model(hf_dir)
    base_gguf = os.path.join(args.work, "tiny.gguf")
    convert(hf_dir, base_gguf, args.llama_cpp)

    g = torch.Generator().manual_seed(7)
    seq_a = torch.randint(1, V, (24,), generator=g).tolist()
    seq_a = seq_a + seq_a[3:15] + seq_a[:4]  # repeated n-grams, 40 tokens
    seq_b = torch.randint(1, V, (25,), generator=g).tolist()

    results = []

    def check(name, got, want, tol):
        err = float(np.abs(got - want).max())
        scale = float(np.abs(want).max())
        ok = err <= tol * max(scale, 1.0)
        results.append((name, ok))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: max |diff| {err:.2e} (logits up to {scale:.1f})")

    # llama.cpp's CPU kernels differ from PyTorch by ~1e-5 per layer even without an engram; that drift is the
    # yardstick: with engrams the error must stay of the same size (a hashing/gate/history bug is far larger)
    plain = Gemma4ForCausalLM.from_pretrained(hf_dir, dtype=torch.float32).eval()
    ref_plain = reference(plain, None, seq_a)
    ref_plain_b = reference(plain, None, seq_b)
    base_a = float(np.abs(llama_logits(args.driver, base_gguf, "full", 512, 0, seq_a, args.work) - ref_plain).max())
    base_b = float(np.abs(llama_logits(args.driver, base_gguf, "full", 512, 0, seq_b, args.work) - ref_plain_b).max())
    base_err = max(base_a, base_b)
    print(f"no engram: llama.cpp vs HF drift up to {base_err:.2e} (CPU kernel numerics, not the engram)")

    def check_ref(name, got, want):
        err = float(np.abs(got - want).max())
        ok = err <= 3 * base_err + 1e-4
        results.append((name, ok))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: max |diff| {err:.2e} (allowed {3 * base_err + 1e-4:.2e})")
        return err

    variants = [
        ("scalar@0 + scalar_sqrt@3", [(0, "scalar"), (3, "scalar_sqrt")], "f32"),
        ("ple@0 + scalar_sqrt@5", [(0, "ple"), (5, "scalar_sqrt")], "f32"),
        ("scalar_sqrt@0 + ple@3, bf16 storage", [(0, "scalar_sqrt"), (3, "ple")], "bf16"),
        # rows per order differ (and 4-grams: 3 earlier tokens from the KV history)
        ("4-grams + sizes 97-131-61 @0 + scalar@3", [(0, "scalar_sqrt", {2: 97, 3: 131, 4: 61}), (3, "scalar")], "f32"),
    ]
    for label, layout, dtype in variants:
        edir = os.path.join(args.work, "engram_" + label.split(",")[0].replace(" ", "").replace("+", "_").replace("@", ""))
        os.makedirs(edir, exist_ok=True)
        engrams = [(entry[0], make_engram(entry[0], entry[1], table_sizes=entry[2] if len(entry) > 2 else None))
                   for entry in layout]
        save_engrams(engrams, edir)
        gguf_path = os.path.join(args.work, "tiny_engram.gguf")
        run([sys.executable, os.path.join("llama_cpp", "gguf_add_engram.py"), "--gguf", base_gguf, "--engram", edir,
             "--out", gguf_path, "--dtype", dtype, "--force"], env=env)

        if dtype == "bf16":  # the reference uses the same rounded weights (norm scales stay f32)
            with torch.no_grad():
                for _, eng in engrams:
                    for n, p in eng.named_parameters():
                        if n not in ("q_scale", "k_scale"):
                            p.copy_(p.bfloat16().float())

        model = Gemma4ForCausalLM.from_pretrained(hf_dir, dtype=torch.float32).eval()
        runner = EngramRunner(model, [(eng, layer) for layer, eng in engrams])
        with torch.no_grad():
            runner.reset()
            ref_a = model(input_ids=torch.tensor([seq_a])).logits[0].float().numpy()
            runner.reset()
            ref_b = model(input_ids=torch.tensor([seq_b])).logits[0].float().numpy()
        effect = float(np.abs(ref_a - ref_plain).max())
        print(f"--- {label}: the engrams change the logits by up to {effect:.2f}")
        results.append((f"{label}: the engrams have a real effect", effect > 0.1))

        full_a = llama_logits(args.driver, gguf_path, "full", 512, 0, seq_a, args.work)
        full_b = llama_logits(args.driver, gguf_path, "full", 512, 0, seq_b, args.work)
        err = check_ref(f"{label}: matches the PyTorch engram (seq A)", full_a, ref_a)
        check_ref(f"{label}: matches the PyTorch engram (seq B)", full_b, ref_b)

        # negative control: the same engrams without token merging must clearly fail the same tolerance
        with torch.no_grad():
            for _, eng in engrams:
                eng.canon.copy_(torch.arange(V))
            runner.reset()
            wrong = model(input_ids=torch.tensor([seq_a])).logits[0].float().numpy()
        bad_err = float(np.abs(full_a - wrong).max())
        ok = bad_err > 3 * (3 * base_err + 1e-4)
        results.append((f"{label}: negative control (hashing without merging) is caught", ok))
        print(f"[{'PASS' if ok else 'FAIL'}] {label}: negative control (no token merging) max |diff| {bad_err:.2e} "
              f"vs {err:.2e} with the right hashing")

        # token history: every way of feeding the tokens must match the reference too. A token that lost
        # its predecessors would hash to other rows in all tables, an error of the size of the engram effect.
        # (llama.cpp itself differs by up to ~1e-2 between these modes even without an engram.)
        check_ref(f"{label}: ubatch of 7 (chunked prompt)",
                  llama_logits(args.driver, gguf_path, "full", 7, 0, seq_a, args.work), ref_a)
        check_ref(f"{label}: prefill 13 + token-by-token decode",
                  llama_logits(args.driver, gguf_path, "decode", 512, 13, seq_a, args.work), ref_a)
        check_ref(f"{label}: state save/clear/restore, then decode",
                  llama_logits(args.driver, gguf_path, "state", 512, 13, seq_a, args.work), ref_a)
        two = llama_logits(args.driver, gguf_path, "two", 512, len(seq_a), seq_a + seq_b, args.work)
        check_ref(f"{label}: two sequences in one batch", two, np.concatenate([ref_a, ref_b]))

    bad = sum(not ok for _, ok in results)
    print(f"\n{len(results) - bad} passed, {bad} failed")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
