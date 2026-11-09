"""
Several engrams, init_from and frozen, on CPU:  python test_multi.py

Engrams are set up the way micro_engram.train does it (runconfig.engram_requests +
build_slot). Run A trains one engram at layer 0 and saves a checkpoint like the old
Axolotl/FSDP runs did (DCP folder + micro_engram_config.json), which init_from must keep
reading. Run B loads it FROZEN and adds a new engram at layer 2. Then: the config lists
both, inference with both matches training, the frozen one stays frozen,
assemble_engram_model.py writes both, and engram_stats.py reports the trained one.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types

import torch
import torch.nn as nn

import logging  # noqa: E402

import torch.distributed.checkpoint as dcp  # noqa: E402

from micro_engram.module import MicroEngram  # noqa: E402
from micro_engram.runconfig import (_params, build_config, build_slot, engram_requests,  # noqa: E402
                                    write_config)
from micro_engram.weights import (engram_layers_in, engram_specs, load_engram_weights,  # noqa: E402
                                  resolve_engram_source)

D, V, NL = 32, 500, 4
TMP = tempfile.mkdtemp(prefix="engram_multi_")
results = []


def check(name, ok):
    results.append((name, bool(ok)))


class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(D, D)

    def forward(self, h, **kw):
        return h + torch.tanh(self.lin(h))


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(123)  # identical backbone every time
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.embed_tokens = nn.Embedding(V, D)
        self.model.language_model.layers = nn.ModuleList(Layer() for _ in range(NL))
        self.config = types.SimpleNamespace(hidden_size=D, vocab_size=V, pad_token_id=0)

    def forward(self, input_ids=None, position_ids=None, **kw):
        h = self.model.language_model.embed_tokens(input_ids)
        for layer in self.model.language_model.layers:
            h = layer(h)
        return h


class DictDefault(dict):
    """A mapping that answers every missing attribute with None (engram_requests must not use hasattr)."""

    def __getattr__(self, name):
        return self.get(name)


class Cfg(types.SimpleNamespace):
    def __getattr__(self, name):  # unset settings are None
        return None


class _Grab(logging.Handler):
    def __init__(self):
        super().__init__()
        self.msgs = []

    def emit(self, record):
        self.msgs.append(record.getMessage())


grab = _Grab()
logging.getLogger("micro_engram").addHandler(grab)


class Setup:
    """Engram setup as micro_engram.train does it: build every requested engram, freeze the
    base, train only the engrams that aren't frozen."""

    def __init__(self):
        self.slots, self.base_conf = [], None

    def post_model_load(self, c, model):
        self.slots = [build_slot(spec, init_from, frozen, model, model.config, None, dtype=c.micro_engram_dtype,
                                 sharding=c.micro_engram_sharding)
                      for spec, init_from, frozen in engram_requests(c)]
        for p in model.parameters():
            p.requires_grad_(False)
        for s_ in self.slots:
            for p in _params(s_.engram):
                p.requires_grad_(not s_.frozen)
        self.base_conf = {"model_dim": D, "vocab_size": V, "canon_path": None, "base_model": None,
                          "base_fingerprint": None}


def cfg(**kw):
    base = dict(micro_engram_layer=0, micro_engram_table_size=257, micro_engram_dim=8,
                micro_engram_heads=2, micro_engram_gate="scalar_sqrt", micro_engram_stats_steps=10,
                micro_engram_save_steps=0, fsdp_config={"fsdp_version": 2}, fsdp_version=2)
    base.update(kw)
    return Cfg(**base)


def fake_checkpoint(model, setup, out_dir, step):
    """What an FSDP2 trainable-only save leaves: DCP of the trainable weights + config."""
    ckpt = os.path.join(out_dir, f"checkpoint-{step}")
    # through state_dict(), as FSDP's get_model_state_dict does (it writes the scales, not their deltas)
    full = model.state_dict()
    keep = {n for n, p in model.named_parameters() if p.requires_grad}
    keep |= {n for n, _ in model.named_buffers() if "micro_engram" in n}
    sd = {k: v.detach().clone() for k, v in full.items() if k in keep}
    dcp.save(sd, checkpoint_id=os.path.join(ckpt, "pytorch_model_fsdp_0"), no_dist=True)
    write_config(build_config(setup.base_conf, setup.slots, lambda s_: "pytorch_model_fsdp_0"), ckpt)
    write_config(build_config(setup.base_conf, setup.slots, lambda s_: f"checkpoint-{step}/pytorch_model_fsdp_0"),
                 out_dir)
    open(os.path.join(ckpt, "trainer_state.json"), "w").write("{}")
    return ckpt


# ----------------------------------------------------------------------------- run A
run_a = os.path.join(TMP, "runA")
model_a = Toy()
plug_a = Setup()
plug_a.post_model_load(cfg(), model_a)
eng_a = plug_a.slots[0].engram
with torch.no_grad():  # "training"
    eng_a.value.weight.normal_(std=0.05)
    eng_a.tables[1].weight.add_(0.3)
    eng_a.q_scale.add_(0.7)  # delta: the scale becomes 1.7
check("run A: one engram at layer 0, seed 0 (same hashing as before multi-engram)",
      len(plug_a.slots) == 1 and plug_a.slots[0].spec["seed"] == 0)
fake_checkpoint(model_a, plug_a, run_a, 10)
fake_checkpoint(model_a, plug_a, run_a, 20)
os.remove(os.path.join(run_a, "checkpoint-20", "trainer_state.json"))  # still being written
with open(os.path.join(run_a, "micro_engram_config.json")) as f:
    top_conf = json.load(f)
check("output_dir config points to the newest checkpoint (servable as is, weights not null)",
      top_conf["weights"] == "checkpoint-20/pytorch_model_fsdp_0"
      and top_conf["engrams"][0]["weights"] == "checkpoint-20/pytorch_model_fsdp_0")

w, spec = resolve_engram_source(run_a, 0)
check("init_from an output_dir picks the newest COMPLETE checkpoint",
      w.endswith(os.path.join("checkpoint-10", "pytorch_model_fsdp_0")) and spec["gate"] == "scalar_sqrt")

# ----------------------------------------------------------------------------- run B
run_b = os.path.join(TMP, "runB")
model_b = Toy()
plug_b = Setup()
cfg_b = cfg(micro_engram_init_from=run_a, micro_engram_frozen=True,
            micro_engram_table_size=999,  # wrong on purpose: the trained engram's settings win
            micro_engram_extra=[DictDefault(layer=2, gate="scalar")])
plug_b.post_model_load(cfg_b, model_b)
e0, e2 = plug_b.slots[0].engram, plug_b.slots[1].engram
check("run B: frozen layer-0 engram equals run A's (trained settings used, not the YAML's)",
      e0.table_size == 257 and torch.equal(e0.tables[1].weight, eng_a.tables[1].weight)
      and torch.allclose(e0.q_scale, eng_a.q_scale, atol=1e-6) and torch.equal(e0.mults, eng_a.mults))
check("run B: layer-2 engram is new, its own gate, seed 10007*2, starts as a no-op",
      e2.gate == "scalar" and plug_b.slots[1].spec["seed"] == 20014
      and not torch.equal(e2.mults, e0.mults) and e2.value.weight.abs().sum() == 0)

check("frozen engram stays frozen; the new one trains",
      not any(p.requires_grad for p in e0.parameters()) and all(p.requires_grad for p in e2.parameters()))

ckpt_b = fake_checkpoint(model_b, plug_b, run_b, 5)
conf_b = json.load(open(os.path.join(ckpt_b, "micro_engram_config.json")))
specs_b = engram_specs(conf_b, os.path.join(ckpt_b, "micro_engram_config.json"))
check("config lists both engrams; frozen one points to run A, trained one to this checkpoint",
      [s["layer"] for s in specs_b] == [0, 2]
      and specs_b[0]["weights"].endswith(os.path.join("checkpoint-10", "pytorch_model_fsdp_0"))
      and specs_b[1]["weights"] == os.path.join(ckpt_b, "pytorch_model_fsdp_0")
      and specs_b[0]["frozen"] and not specs_b[1]["frozen"] and specs_b[1]["seed"] == 20014)
check("top-level fields still describe the first engram (older loaders)",
      conf_b["layer"] == 0 and conf_b["gate"] == "scalar_sqrt" and conf_b["seed"] == 0)
# buffers (mults, canon) are not parameters, so a trainable-only save may still hold the
# frozen engram's; its trained tables are not there, which is why the config points to run A
in_b = load_engram_weights(os.path.join(ckpt_b, "pytorch_model_fsdp_0"), layer=0)
check("trainable-only checkpoint has the layer-2 tables, not the frozen layer-0 ones",
      2 in engram_layers_in(os.path.join(ckpt_b, "pytorch_model_fsdp_0"))
      and not any(k.startswith("tables.") for k in in_b))

# ----------------------------------------------------------------------------- inference
with torch.no_grad():
    e2.value.weight.normal_(std=0.05)  # "train" the new one
ckpt_b = fake_checkpoint(model_b, plug_b, run_b, 6)
ids = torch.randint(5, V, (1, 12))
model_b.eval()
with torch.no_grad():
    ref = model_b(input_ids=ids)  # training-time path: both engrams via their forward hooks

sys.argv = ["x"]
from infer_engram import attach_planned, plan_engrams  # noqa: E402

args = types.SimpleNamespace(model=TMP, engram_pt=None, engram_config=os.path.join(ckpt_b, "micro_engram_config.json"),
                             layer=None, table_size=None, dim=None, heads=None, gate=None)
plans = plan_engrams(args)
fresh = Toy().eval()
runner = attach_planned(plans, fresh)
with torch.no_grad():
    runner.reset()
    full = fresh(input_ids=ids)
    runner.reset()  # prefill + one-token decode steps, as generate() does
    parts = [fresh(input_ids=ids[:, :8])] + [fresh(input_ids=ids[:, t:t + 1]) for t in range(8, 12)]
check("inference loads both engrams (layer 0 from run A, layer 2 from run B)",
      [lay for _, lay in runner.engrams] == [0, 2])
check("inference with both engrams == training forward (full sequence)", torch.allclose(full, ref, atol=1e-6))
check("inference with both engrams == training forward (prefill + decode)",
      torch.allclose(torch.cat(parts, dim=1), ref, atol=1e-6))
runner.enabled = False
with torch.no_grad():
    runner.reset()
    off = fresh(input_ids=ids)
check("engrams off changes the output", not torch.allclose(off, ref))

# ----------------------------------------------------------------------------- assemble + stats
base_dir = os.path.join(TMP, "base")
os.makedirs(base_dir)
from safetensors.torch import save_file  # noqa: E402

save_file({f"model.language_model.layers.{i}.self_attn.q_proj.weight": torch.randn(4, 4) for i in range(NL)},
          os.path.join(base_dir, "model.safetensors"))
out_dir = os.path.join(TMP, "assembled")
r = subprocess.run([sys.executable, "assemble_engram_model.py", "--base", base_dir,
                    "--engram", os.path.join(ckpt_b, "pytorch_model_fsdp_0"), "--out", out_dir],
                   capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__)),
                   env={**os.environ, "PYTHONPATH": os.path.dirname(os.path.abspath(__file__))})
ok = r.returncode == 0
if ok:
    a0, a2 = load_engram_weights(out_dir, layer=0), load_engram_weights(out_dir, layer=2)
    conf_out = json.load(open(os.path.join(out_dir, "micro_engram_config.json")))
    ok = (torch.equal(a0["tables.1.weight"], eng_a.tables[1].weight)
          and torch.equal(a2["value.weight"], e2.value.weight)
          and all("weights" not in e for e in conf_out["engrams"]))
check("assemble_engram_model.py writes both engrams into one folder" + ("" if r.returncode == 0 else
      f" ({r.stderr.strip().splitlines()[-1] if r.stderr.strip() else r.returncode})"), ok)

r = subprocess.run([sys.executable, "engram_stats.py", run_b, "--all"], capture_output=True, text=True,
                   cwd=os.path.dirname(os.path.abspath(__file__)),
                   env={**os.environ, "PYTHONPATH": os.path.dirname(os.path.abspath(__file__))})
lines = [ln for ln in r.stdout.splitlines() if ln.startswith("step")]
check(f"engram_stats.py reports only the trained engram ({lines[-1] if lines else r.stderr[-200:]})",
      r.returncode == 0 and lines and all(" L2 " in ln for ln in lines))

# ----------------------------------------------------------------------------- micro_engram_dtype
bf_model = Toy().to(torch.bfloat16)
pd = Setup()
pd.post_model_load(cfg(micro_engram_dtype="fp32", micro_engram_extra=[DictDefault(layer=2)]), bf_model)
check("micro_engram_dtype fp32: engrams keep fp32 weights in a bf16 model",
      all(next(s.engram.parameters()).dtype == torch.float32 for s in pd.slots)
      and next(bf_model.model.language_model.layers[0].lin.parameters()).dtype == torch.bfloat16)
with torch.no_grad():
    out_bf = bf_model(input_ids=ids)
check("fp32 engram in a bf16 model: forward works, output stays bf16", out_bf.dtype == torch.bfloat16)
pb = Setup()
pb.post_model_load(cfg(), Toy().to(torch.bfloat16))
check("micro_engram_dtype unset: engram follows its layer (bf16)", next(pb.slots[0].engram.parameters()).dtype == torch.bfloat16)

# ----------------------------------------------------------------------------- legacy
legacy = {"layer": 1, "table_size": 64, "engram_dim": 8, "num_heads": 2, "weights": "micro_engram.pt"}
sp = engram_specs(legacy, "/x/micro_engram_config.json")
check("old config (no 'engrams', no 'seed'): one engram, seed 0, weights resolved",
      len(sp) == 1 and sp[0]["seed"] == 0 and sp[0]["layer"] == 1 and sp[0]["weights"] == "/x/micro_engram.pt")
try:
    Setup().post_model_load(cfg(micro_engram_extra=[{"layer": 0}]), Toy())
    check("two engrams in one layer are rejected", False)
except ValueError:
    check("two engrams in one layer are rejected", True)
try:
    Setup().post_model_load(cfg(micro_engram_frozen=True), Toy())
    check("frozen without init_from is rejected", False)
except ValueError:
    check("frozen without init_from is rejected", True)

# base fingerprint inside a server: vLLM builds models with CUDA as the default device, and
# safetensors' get_slice follows it ("can't convert cuda tensor to numpy"). meta stands in.
from micro_engram.weights import base_fingerprint  # noqa: E402

fp_dir = os.path.join(TMP, "fp_base")
os.makedirs(fp_dir)
save_file({"model.embed_tokens.weight": torch.randn(64, 8),
           "model.layers.0.self_attn.q_proj.weight": torch.randn(8, 8)},
          os.path.join(fp_dir, "model.safetensors"))
fp_cpu = base_fingerprint(fp_dir)
torch.set_default_device("meta")
try:
    fp_meta = base_fingerprint(fp_dir)
except Exception as e:  # noqa: BLE001
    fp_meta = f"{type(e).__name__}: {e}"
finally:
    torch.set_default_device("cpu")
check(f"base fingerprint is the same with a non-CPU default device ({fp_meta})", fp_cpu and fp_meta == fp_cpu)

# one file shared by several engrams must carry layer numbers (else both get the same tensors)
local_pt = os.path.join(TMP, "local_names.pt")
torch.save({"tables.0.weight": torch.zeros(4, 2), "value.weight": torch.zeros(2, 2)}, local_pt)
full_pt = os.path.join(TMP, "full_names.pt")
torch.save({f"model.layers.{i}.micro_engram.tables.0.weight": torch.full((4, 2), float(i)) for i in (0, 29)}, full_pt)
try:
    load_engram_weights(local_pt, layer=29, need_layer=True)
    shared_err = None
except ValueError as e:
    shared_err = str(e)
check("shared file with local names: error instead of the same weights for every engram",
      shared_err is not None and "layer number" in shared_err
      and "tables.0.weight" in load_engram_weights(local_pt, layer=29))  # alone: still loads
check("shared file with full names: each engram gets its own layer's tensors",
      float(load_engram_weights(full_pt, layer=29, need_layer=True)["tables.0.weight"][0, 0]) == 29.0
      and float(load_engram_weights(full_pt, layer=0, need_layer=True)["tables.0.weight"][0, 0]) == 0.0)

# micro_engram_sharding: rows, single process (the multi-GPU exchange is in test_rows.py)
ids_r = torch.randint(5, V, (2, 12))
plain = Toy()
p_plain = Setup()
p_plain.post_model_load(cfg(micro_engram_init_from=run_a), plain)
rowm = Toy()
p_rows = Setup()
p_rows.post_model_load(cfg(micro_engram_init_from=run_a, micro_engram_sharding="rows",
                           micro_engram_extra=[{"layer": 2}]), rowm)
e_rows = p_rows.slots[0].engram
check("rows: tables leave the model's parameters and state dict (FSDP never sees them)",
      not any(".tables." in n for n, _ in rowm.named_parameters())
      and not any(".tables." in k for k in rowm.state_dict()) and len(e_rows.row_shards) == 4)
check("rows: init_from rows are the trained tables; trainable; same output as without sharding",
      torch.equal(e_rows.row_shards[1], p_plain.slots[0].engram.tables[1].weight)
      and all(p.requires_grad for p in e_rows.row_shards)
      and torch.allclose(p_plain.slots[0].engram(torch.ones(2, 12, D), ids_r), e_rows(torch.ones(2, 12, D), ids_r)))
x_rows = p_rows.slots[1].engram.row_shards[0]
check(f"rows: a new engram's rows start at N(0, 0.02) (std {float(x_rows.std()):.4f}), where = its layer",
      abs(float(x_rows.std()) - 0.02) < 0.004 and p_rows.slots[1].where == "model.language_model.layers.2")
rowm(input_ids=ids_r).pow(2).sum().backward()
check("rows: gradients reach the rows through the model", e_rows.row_shards[0].grad is not None
      and e_rows.row_shards[0].grad.abs().sum() > 0)
mixed = MicroEngram(model_dim=D, table_size=257, engram_dim=8, num_heads=2, vocab_size=V, gate="scalar_sqrt",
                    build_tables=False)
mixed.shard_rows(0, 1, dtype=torch.float32)
mixed.to(torch.bfloat16)  # as FSDP's mixed precision computes key/value; the rows stay fp32
try:
    mixed(torch.randn(2, 12, D, dtype=torch.bfloat16), ids_r).float().pow(2).sum().backward()
    mixed_ok = mixed.row_shards[0].dtype == torch.float32 and mixed.row_shards[0].grad.dtype == torch.float32
except RuntimeError as e:
    mixed_ok = False
    print("   mixed:", e)
check("rows: fp32 rows (micro_engram_dtype: fp32) with bf16 key/value compute; fp32 gradients", mixed_ok)
p_fr = Setup()
p_fr.post_model_load(cfg(micro_engram_init_from=run_a, micro_engram_frozen=True, micro_engram_sharding="rows"), Toy())
check("rows: a frozen engram's rows are frozen too", not any(p.requires_grad for p in p_fr.slots[0].engram.row_shards))

# per-order table sizes (micro_engram_table_sizes): 4-grams, dropping an order, inheritance
from micro_engram.module import engram_from_spec  # noqa: E402
from micro_engram.weights import engram_specs as _specs  # noqa: E402

p_sz = Setup()
p_sz.post_model_load(cfg(micro_engram_table_sizes={2: 101, 3: 307, 4: 53},
                         micro_engram_extra=[{"layer": 2}, {"layer": 3, "table_size": 61}]), Toy())
e_main, e_inh, e_own = (s_.engram for s_ in p_sz.slots)
check("table_sizes: orders 2/3/4 with their own rows (4-grams added), table_size None when mixed",
      e_main.orders == (2, 3, 4) and e_main.table_rows == [101, 101, 307, 307, 53, 53] and e_main.table_size is None
      and [t.weight.shape[0] for t in e_main.tables] == e_main.table_rows and tuple(e_main.mults.shape) == (6, 4))
idx_sz = e_main.hash_ids(torch.randint(0, V, (2, 40)))
check("table_sizes: every table's hashed rows stay inside that table",
      all(int(i.max()) < r and int(i.min()) >= 0 for i, r in zip(idx_sz, e_main.table_rows)))
check("extra engram without sizes inherits the main's per-order sizes; with its own table_size, one size",
      e_inh.table_rows == e_main.table_rows and e_own.orders == (2, 3) and e_own.table_rows == [61] * 4)
p_drop = Setup()
p_drop.post_model_load(cfg(micro_engram_table_sizes={2: 0, 3: 307}), Toy())
p_null = Setup()
p_null.post_model_load(cfg(micro_engram_table_sizes={"2": None, "4": None}), Toy())
check("an order set to 0 or null is not used (keys may be strings, as from JSON)",
      p_drop.slots[0].engram.orders == (3,) and p_null.slots[0].engram.orders == (3,)
      and p_null.slots[0].engram.table_rows == [257, 257])
conf_sz = build_config(p_sz.base_conf, p_sz.slots, lambda s_: None)
conf_sz = json.loads(json.dumps(conf_sz))  # through JSON: order keys become strings
specs_sz = _specs(conf_sz)
rebuilt = engram_from_spec(specs_sz[0], D, V, 0)
check("config round trip (JSON): the loaders rebuild the same tables and hashing",
      rebuilt.table_rows == e_main.table_rows and torch.equal(rebuilt.mults, e_main.mults)
      and conf_sz["engrams"][0]["table_size"] is None and specs_sz[2]["table_sizes"] == {2: 61, 3: 61})
grab.msgs.clear()
p_keep = Setup()
p_keep.post_model_load(cfg(micro_engram_init_from=run_a, micro_engram_table_sizes={2: 50, 3: 900}), Toy())
check("init_from a uniform run keeps the trained sizes over the YAML's table_sizes (and says so)",
      p_keep.slots[0].engram.table_rows == [257] * 4 and any("table_sizes" in m for m in grab.msgs))
p_rs = Setup()
toy_rs = Toy()
p_rs.post_model_load(cfg(micro_engram_table_sizes={2: 101, 3: 307, 4: 53}, micro_engram_sharding="rows"), toy_rs)
e_rs = p_rs.slots[0].engram
check("row sharding with mixed sizes: one shard per table, each its own size",
      [p_.shape[0] for p_ in e_rs.row_shards] == [101, 101, 307, 307, 53, 53] and e_rs.row_info[2] == [0] * 6)
toy_rs(input_ids=torch.randint(5, V, (2, 12))).pow(2).sum().backward()
check("row sharding with mixed sizes: forward and backward run, every table gets gradients",
      all(p_.grad is not None for p_ in e_rs.row_shards))

toy_sz = Toy()
p_sz2 = Setup()
p_sz2.post_model_load(cfg(micro_engram_table_sizes={2: 101, 3: 307, 4: 53}), toy_sz)
with torch.no_grad():
    p_sz2.slots[0].engram.value.weight.normal_(std=0.05)
ck_sz = fake_checkpoint(toy_sz, p_sz2, os.path.join(TMP, "runSZ"), 4)
ids4 = torch.randint(5, V, (1, 14))
toy_sz.eval()
with torch.no_grad():
    ref_sz = toy_sz(input_ids=ids4)
args_sz = types.SimpleNamespace(model=TMP, engram_pt=None, engram_config=os.path.join(ck_sz, "micro_engram_config.json"),
                                layer=None, table_size=None, dim=None, heads=None, gate=None)
fresh_sz = Toy().eval()
runner_sz = attach_planned(plan_engrams(args_sz), fresh_sz)
with torch.no_grad():
    runner_sz.reset()
    full_sz = fresh_sz(input_ids=ids4)
    runner_sz.reset()
    parts_sz = [fresh_sz(input_ids=ids4[:, :9])] + [fresh_sz(input_ids=ids4[:, t:t + 1]) for t in range(9, 14)]
check("4-grams + sizes 101/307/53 saved, reloaded by infer_engram == training forward (full and prefill + decode)",
      torch.allclose(full_sz, ref_sz, atol=1e-6) and torch.allclose(torch.cat(parts_sz, dim=1), ref_sz, atol=1e-6))

shutil.rmtree(TMP, ignore_errors=True)
bad = 0
for name, ok in results:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    bad += not ok
sys.exit(1 if bad else 0)
