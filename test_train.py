"""CPU tests for micro_engram.train (the standalone trainer). Run: python test_train.py

Tiny random Gemma 4 + a word-level tokenizer, 2 processes on gloo:
  1. data: completion chunks (Axolotl's rules), chat masks (assistant content + end-of-turn
     token only), packing (position_ids restart per document), cache round-trip
  2. config: LoRA keys and unknown keys are errors; Axolotl-only keys are ignored
  3. first-step loss == a plain HF forward (full logits, unsharded) on the same micro-batches,
     with gradient accumulation and 2 GPUs: per-token mean over the whole step
  4. no-grad prefix gives the same engram gradients as a full backward
  5. save -> resume -> next step is identical to the uninterrupted run
  6. the checkpoint loads through the usual loaders (engram_specs / load_engram_weights)
  7. cpu_ram_efficient_loading (rank 1 on meta) trains exactly like a normal load
"""
import json
import os
import shutil
import sys
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["MICRO_ENGRAM_CPU"] = "1"

V = 400
WORDS = [f"w{i}" for i in range(V - 8)]
SPECIAL = ["<pad>", "<eos>", "<bos>", "<|turn>", "<turn|>", "user", "assistant", "system"]
TEMPLATE = ("{% for m in messages %}<|turn>{{ m.role }} {{ m.content }}<turn|> {% endfor %}"
            "{% if add_generation_prompt %}<|turn>assistant {% endif %}")
FAIL = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  ({detail})" if detail and not ok else ""), flush=True)
    if not ok:
        FAIL.append(name)


def make_assets(root):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import Gemma4ForCausalLM, Gemma4TextConfig, PreTrainedTokenizerFast

    vocab = {t: i for i, t in enumerate(SPECIAL + WORDS)}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="<pad>"))
    tk.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="<pad>", eos_token="<eos>", bos_token="<bos>",
                                  additional_special_tokens=["<|turn>", "<turn|>"])
    tok.chat_template = TEMPLATE
    model_dir = os.path.join(root, "base")
    cfg = Gemma4TextConfig(vocab_size=V, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                           num_attention_heads=4, num_key_value_heads=2, head_dim=16, global_head_dim=32,
                           layer_types=["sliding_attention", "sliding_attention", "full_attention", "sliding_attention"],
                           sliding_window=8, hidden_size_per_layer_input=0, vocab_size_per_layer_input=V,
                           max_position_embeddings=1024, final_logit_softcapping=30.0, initializer_range=0.08,
                           pad_token_id=0, eos_token_id=1, bos_token_id=2)
    cfg.architectures = ["Gemma4ForCausalLM"]
    torch.manual_seed(0)
    m = Gemma4ForCausalLM(cfg)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if p.dim() == 1:
                p.copy_(1 + 0.2 * torch.randn_like(p))
    m.to(torch.bfloat16).save_pretrained(model_dir, safe_serialization=True)
    tok.save_pretrained(model_dir)
    g = torch.Generator().manual_seed(1)
    rows = []
    for k in range(40):
        n = int(torch.randint(5, 60, (1,), generator=g))
        rows.append({"text": "<bos> " + " ".join(WORDS[int(i)] for i in torch.randint(0, len(WORDS), (n,), generator=g))})
    data = os.path.join(root, "cpt.jsonl")
    with open(data, "w") as f:
        f.write("\n".join(json.dumps(r) for r in rows))
    canon = torch.arange(V) & ~1
    torch.save(canon, os.path.join(root, "canon.pt"))
    return model_dir, data


def base_cfg(root, model_dir, data, out, **over):
    c = {
        "base_model": model_dir, "datasets": [{"path": data, "type": "completion"}], "sequence_len": 32,
        "output_dir": out, "micro_batch_size": 1, "gradient_accumulation_steps": 2, "num_epochs": 1,
        "learning_rate": 5e-3, "lr_scheduler": "constant", "max_grad_norm": 2.0, "seed": 7,
        "attn_implementation": "sdpa", "fused_attn_kernel": False, "gradient_checkpointing": True,
        "micro_engram_layer": 1, "micro_engram_table_size": 97, "micro_engram_dim": 8, "micro_engram_heads": 2,
        "micro_engram_gate": "scalar_sqrt", "micro_engram_extra": [{"layer": 3}],
        "micro_engram_optimizer": "rowwise_adam", "micro_engram_dtype": "fp32",
        "micro_engram_canon_path": os.path.join(root, "canon.pt"), "micro_engram_stats_steps": 1,
        "loss_chunk_tokens": 16, "save_steps": None,
    }
    c.update(over)
    return c


# ----------------------------------------------------------------------------- 1-2. data / config
def test_data(root, model_dir, data):
    import numpy as np
    from transformers import AutoTokenizer

    from micro_engram.train.config import DatasetConfig, load_config
    from micro_engram.train.data import EpochPlan, Samples, Tokenizing, collate, pack, prepare

    tok = AutoTokenizer.from_pretrained(model_dir)
    cfg = load_config(base_cfg(root, model_dir, data, os.path.join(root, "o")))
    t = Tokenizing(tok, cfg)
    long = "<bos> " + " ".join(WORDS[:70])
    ch = t.completion(long)
    ids = tok(long)["input_ids"]
    check("completion: split into sequence_len chunks, EOS appended",
          [len(c) for c, _ in ch] == [32, 32, len(ids) + 1 - 64] and ch[-1][0][-1] == tok.eos_token_id and ch[0][0] == ids[:32])

    cfg2 = load_config(base_cfg(root, model_dir, data, os.path.join(root, "o"), eot_tokens=["<turn|>"]))
    t2 = Tokenizing(tok, cfg2)
    msgs = [{"role": "system", "content": "w1 w2"}, {"role": "user", "content": "w3 w4"},
            {"role": "assistant", "content": "w5 w6 w7"}, {"role": "user", "content": "w8"},
            {"role": "assistant", "content": "w9"}]
    ids, mask = t2.chat(msgs, {"assistant"})
    trained = [tok.convert_ids_to_tokens(i) for i, m in zip(ids, mask) if m]
    check("chat: only assistant content + its end-of-turn token are trained",
          trained == ["w5", "w6", "w7", "<turn|>", "w9", "<turn|>"], trained)

    ds = DatasetConfig(path="x", type="chat_template", field_messages="conversations",
                       message_property_mappings={"role": "from", "content": "value"},
                       roles={"assistant": ["gpt"], "user": ["human"]})
    from micro_engram.train.data import messages_of

    got = messages_of({"conversations": [{"from": "human", "value": "a"}, {"from": "gpt", "value": "b"}]}, ds)
    check("chat: ShareGPT mapping", got == [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}], got)

    s = Samples.from_list([([1, 2, 3], None), ([4, 5], np.array([False, True])), ([6] * 30, None)])
    bins = pack(s.lengths(), 32, seed=0)
    check("pack: every sample once, no bin over capacity",
          sorted(i for b in bins for i in b) == [0, 1, 2] and all(sum(s.lengths()[i] for i in b) <= 32 for b in bins))
    b = collate(s, [[0, 1]], 8, pad_id=0)
    check("collate: position_ids restart per document, padding continues the last one",
          b["position_ids"][0].tolist() == [0, 1, 2, 0, 1, 2, 3, 4])
    check("collate: labels shifted, never across documents, untrained -100",
          b["labels"][0].tolist() == [2, 3, -100, 5, -100, -100, -100, -100], b["labels"][0].tolist())

    prep = os.path.join(root, "prep")
    c3 = load_config(base_cfg(root, model_dir, data, os.path.join(root, "o"), dataset_prepared_path=prep))
    a = prepare(c3, tok, model_dir)
    b2 = prepare(c3, tok, model_dir)
    check("cache: second prepare reads the same samples",
          np.array_equal(a.tokens, b2.tokens) and np.array_equal(a.offsets, b2.offsets)
          and len(os.listdir(prep)) == 1)
    plan = EpochPlan(a, c3, world=2)
    e0, e1 = plan.bins(0), plan.bins(1)
    check("epochs are packed in a different order", e0 != e1)
    mb = [plan.micro_batches(0, 0, r) for r in (0, 1)]
    flat = [tuple(x) for r in mb for m in r for x in m]
    check("ranks and micro-batches get distinct bins", len(flat) == len(set(flat)) == 4)

    from micro_engram.train.head import VocabShard

    W = torch.zeros(262_144, 8)
    chunks = {VocabShard(W, r, 5).chunk for r in range(5)}  # 52,429 rows on ranks 0-3, 52,428 on rank 4
    check("vocab shards: same loss chunk on every rank, also with an uneven split", len(chunks) == 1, chunks)

    for bad, why in (({"adapter": "qlora"}, "adapter"), ({"lora_r": 1}, "lora_r"),
                     ({"unfrozen_parameters": [".+x"]}, "unfrozen"), ({"no_such_key": 1}, "unknown")):
        try:
            load_config(base_cfg(root, model_dir, data, "o", **bad))
            check(f"config: {why} key is an error", False)
        except Exception:  # noqa: BLE001
            check(f"config: {why} key is an error", True)
    c4 = load_config(base_cfg(root, model_dir, data, "o", plugins=["x"], strict=False, bf16="auto",
                              fsdp_version=2, fsdp_config={"offload_params": True, "state_dict_type": "SHARDED"}))
    check("config: Axolotl-only keys ignored, fsdp_config read", c4.fsdp_config.offload_params)


# ----------------------------------------------------------------------------- 3-7. distributed
def _init(rank, world, port):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE=str(world),
                      LOCAL_RANK=str(rank))


def _trainer(cfg_dict):
    from micro_engram.train.config import load_config
    from micro_engram.train.loop import Trainer

    t = Trainer(load_config(cfg_dict))
    t.build()
    return t


def _randomize_dense(t):
    g = torch.Generator().manual_seed(5)
    with torch.no_grad():
        for s in t.slots:
            for n, p in s.engram.named_parameters():
                p.copy_(torch.randn(p.shape, generator=g) * 0.05)


def _engram_state(t):
    out = {}
    for s in t.slots:
        for n, p in s.engram.named_parameters():
            out[f"{s.layer}.{n}"] = p.detach().clone()
        for k, p in enumerate(s.engram.row_shards):
            out[f"{s.layer}.rows{k}"] = p.detach().clone()
    return out


def _dense_grad(p):
    """A table shard's gradient as a dense tensor, whether it is stored sparse or dense."""
    sg = getattr(p, "sparse_grad", None)
    if sg is not None:
        g = torch.zeros(p.shape, dtype=torch.float32)
        g.index_add_(0, sg[0], sg[1])
        return g
    return None if p.grad is None else p.grad.float().clone()


def worker(rank, world, port, root, model_dir, data, q):
    _init(rank, world, port)
    from transformers import Gemma4ForCausalLM

    res = {}
    out = os.path.join(root, "run_a")
    cfg = base_cfg(root, model_dir, data, out)

    # 3. first-step loss vs a plain HF forward
    t = _trainer(cfg)
    micro = t.plan.micro_batches(0, 0, rank)
    batches = [t._micro_batch(b) for b in micro]
    ref = Gemma4ForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, attn_implementation="sdpa")
    s_sum, s_cnt = torch.zeros(()), torch.zeros(())
    with torch.no_grad():
        for b in batches:
            logits = ref(input_ids=b["input_ids"], position_ids=b["position_ids"]).logits.float()
            lab = b["labels"]
            s_sum += torch.nn.functional.cross_entropy(logits.reshape(-1, V), lab.reshape(-1), ignore_index=-100,
                                                       reduction="sum")
            s_cnt += (lab != -100).sum()
    dist.all_reduce(s_sum)
    dist.all_reduce(s_cnt)
    logs = t.train_step(micro)
    res["loss"], res["ref_loss"] = logs["loss"], float(s_sum / s_cnt)

    # 4. no-grad prefix: same gradients
    t.cfg.no_grad_prefix = True
    _randomize_dense(t)
    b = t._micro_batch(t.plan.micro_batches(0, 1, rank)[0])

    def grads(prefix):
        t.cfg.no_grad_prefix = prefix
        for p in t.dense:
            p.grad = None
        t.rows.zero_grad()
        t.forward_loss(b, 1.0)
        return [None if p.grad is None else p.grad.clone() for p in t.dense] + [_dense_grad(p) for p in t.rows.params()]

    g1, g0 = grads(True), grads(False)
    res["prefix_same"] = all((a is None and c is None) or (a is not None and c is not None and torch.allclose(a, c, atol=1e-6))
                             for a, c in zip(g1, g0))
    res["prefix_nonzero"] = any(a is not None and a.abs().sum() > 0 for a in g1)
    t.cfg.no_grad_prefix = True
    for p in t.dense:
        p.grad = None
    t.rows.zero_grad()

    # 5. save -> resume == uninterrupted
    t.global_step, t.step_in_epoch = 1, 1
    t.save_checkpoint()
    t.train_step(t.plan.micro_batches(0, 1, rank))
    after = _engram_state(t)
    del t
    t2 = _trainer({**cfg, "resume_from_checkpoint": os.path.join(out, "checkpoint-1")})
    t2._maybe_resume()
    t2.train_step(t2.plan.micro_batches(0, t2.step_in_epoch, rank))
    after2 = _engram_state(t2)
    res["resume_same"] = all(torch.allclose(after[k], after2[k], atol=1e-6) for k in after)
    res["resume_step"] = t2.global_step
    del t2

    # 7. cpu_ram_efficient_loading: rank 1 builds on meta
    t3 = _trainer({**cfg, "output_dir": os.path.join(root, "run_b"),
                   "fsdp_config": {"cpu_ram_efficient_loading": True}})
    res["meta_loss"] = t3.train_step(t3.plan.micro_batches(0, 0, rank))["loss"]
    del t3

    # 8. sparse row gradients == dense ones: same tables after a few steps (clipping included)
    states = []
    for sparse in (True, False):
        tt = _trainer({**cfg, "output_dir": os.path.join(root, f"run_sp{int(sparse)}"), "sparse_row_grads": sparse,
                       "max_grad_norm": 0.05})
        _randomize_dense(tt)
        res[f"sparse_flag{int(sparse)}"] = tt.rows.sparse
        norms = [tt.train_step(tt.plan.micro_batches(0, k, rank))["table_grad_norm"] for k in range(3)]
        states.append((_engram_state(tt), norms))
        del tt
    (a, na), (b, nb) = states
    res["sparse_same"] = all(torch.allclose(a[k], b[k], atol=1e-6) for k in a) and \
        all(abs(x - y) < 1e-5 * max(1, abs(y)) for x, y in zip(na, nb))
    res["sparse_norms"] = na
    q.put((rank, res))
    dist.destroy_process_group()


def worker_full(rank, world, port, root, model_dir, data):
    """A whole run through Trainer.train(): checkpoints, rotation, final output."""
    _init(rank, world, port)
    cfg = base_cfg(root, model_dir, data, os.path.join(root, "run_full"), save_steps=2, save_total_limit=2,
                   num_epochs=2, lr_scheduler="cosine", warmup_ratio=0.2, profile_step=2)
    t = _trainer(cfg)
    t.train()
    dist.destroy_process_group()


def main():
    root = tempfile.mkdtemp(prefix="me_train_")
    try:
        model_dir, data = make_assets(root)
        test_data(root, model_dir, data)
        world = 2
        q = mp.get_context("spawn").SimpleQueue()
        mp.spawn(worker, args=(world, 29620, root, model_dir, data, q), nprocs=world)
        res = dict(q.get() for _ in range(world))
        r0 = res[0]
        check("first-step loss == plain HF forward (2 GPUs, grad accumulation, per-token mean)",
              abs(r0["loss"] - r0["ref_loss"]) < 2e-2 * r0["ref_loss"], f"{r0['loss']} vs {r0['ref_loss']}")
        check("no-grad prefix: same engram gradients as a full backward",
              all(res[r]["prefix_same"] for r in res) and r0["prefix_nonzero"])
        check("save -> resume -> next step identical to the uninterrupted run",
              all(res[r]["resume_same"] for r in res) and r0["resume_step"] == 1)
        check("sparse row gradients: same tables and table grad norms as dense ones (3 steps, clipped)",
              all(res[r]["sparse_same"] for r in res) and r0["sparse_flag1"] and not r0["sparse_flag0"]
              and all(n and n > 0 for n in r0["sparse_norms"]), r0["sparse_norms"])
        check("cpu_ram_efficient_loading (meta on rank 1) trains like a normal load",
              abs(r0["meta_loss"] - r0["loss"]) < 1e-4, f"{r0['meta_loss']} vs {r0['loss']}")

        mp.spawn(worker_full, args=(world, 29640, root, model_dir, data), nprocs=world)
        out = os.path.join(root, "run_full")
        ckpts = sorted(d for d in os.listdir(out) if d.startswith("checkpoint-"))
        check("save_total_limit keeps the newest checkpoints", len(ckpts) == 2, ckpts)
        from micro_engram.weights import engram_specs, load_engram_weights

        conf_path = os.path.join(out, "micro_engram_config.json")
        with open(conf_path) as f:
            specs = engram_specs(json.load(f), conf_path)
        sds = [load_engram_weights(s["weights"], layer=s["layer"]) for s in specs]
        ok = len(specs) == 2 and all(f"tables.{k}.weight" in sd for sd in sds for k in range(4)) \
            and all(sd["tables.0.weight"].shape[0] == 97 for sd in sds) and all("value.weight" in sd for sd in sds)
        check("final output loads through engram_specs / load_engram_weights (both engrams, full tables)", ok,
              [s["weights"] for s in specs])
        check("profile_step writes a kernel table and a timeline",
              all(os.path.isfile(os.path.join(out, "profile", f"rank0_step2.{x}")) for x in ("txt", "json")))
        with open(os.path.join(out, "train_log.jsonl")) as f:
            lines = [json.loads(x) for x in f]
        check("training log has loss, grad norms and engram stats",
              lines and all("loss" in x and "table_grad_norm" in x for x in lines)
              and any(k.startswith("engram_L1/") for k in lines[-1]))
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print("\nALL PASSED" if not FAIL else f"\n{len(FAIL)} FAILED: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
