"""
Several engrams, init_from and frozen, on CPU:  python test_multi.py
(needs transformers; Axolotl is stubbed, so this runs outside the training venv too)

Run A trains one engram at layer 0 and saves a checkpoint like FSDP does (DCP folder +
micro_engram_config.json). Run B loads it FROZEN and adds a new engram at layer 2.
Then: the config lists both, inference with both matches training, the frozen one
stays frozen even when unfrozen_parameters matches it, assemble_engram_model.py writes
both, and engram_stats.py reports the trained one.
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

# --- stub Axolotl (only BasePlugin is imported) -----------------------------------------
_ax = types.ModuleType("axolotl")
_integ = types.ModuleType("axolotl.integrations")
_base = types.ModuleType("axolotl.integrations.base")
_base.BasePlugin = type("BasePlugin", (), {"__init__": lambda self: None})
sys.modules.setdefault("axolotl", _ax)
sys.modules.setdefault("axolotl.integrations", _integ)
sys.modules.setdefault("axolotl.integrations.base", _base)

import torch.distributed.checkpoint as dcp  # noqa: E402

from infer_engram import EngramRunner  # noqa: E402
from micro_engram.module import MicroEngram  # noqa: E402
from micro_engram.plugin import (MicroEngramPlugin, _WriteConfigCallback,  # noqa: E402
                                 build_config)
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
    """Like Axolotl's DictDefault: every missing attribute is None (so hasattr() is always True)."""

    def __getattr__(self, name):
        return self.get(name)


class Cfg(types.SimpleNamespace):
    def __getattr__(self, name):  # unset settings are None, like Axolotl's DictDefault
        return None


def cfg(**kw):
    base = dict(micro_engram_layer=0, micro_engram_table_size=257, micro_engram_dim=8,
                micro_engram_heads=2, micro_engram_gate="scalar_sqrt", micro_engram_stats_steps=10,
                micro_engram_save_steps=0, fsdp_config={"fsdp_version": 2}, fsdp_version=2)
    base.update(kw)
    return Cfg(**base)


def fake_checkpoint(model, plugin, out_dir, step):
    """What an FSDP2 trainable-only save leaves: DCP of the trainable weights + config."""
    ckpt = os.path.join(out_dir, f"checkpoint-{step}")
    sd = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    sd.update({n: b.clone() for n, b in model.named_buffers() if "micro_engram" in n})
    dcp.save(sd, checkpoint_id=os.path.join(ckpt, "pytorch_model_fsdp_0"), no_dist=True)
    cb = _WriteConfigCallback(plugin.base_conf, plugin.slots, "pytorch_model_fsdp_0")
    state = types.SimpleNamespace(is_world_process_zero=True, global_step=step)
    cb.on_train_begin(types.SimpleNamespace(output_dir=out_dir), state, None)
    cb.on_save(types.SimpleNamespace(output_dir=out_dir), state, None)
    open(os.path.join(ckpt, "trainer_state.json"), "w").write("{}")
    return ckpt


def axolotl_unfreeze(model, pattern):  # utils/freeze.py: dots literal unless followed by '+'
    import re
    rx = re.compile(re.sub(r"\.(?!\+)", "\\.", pattern))
    for n, p in model.named_parameters():
        p.requires_grad = bool(rx.match(n))


# ----------------------------------------------------------------------------- run A
run_a = os.path.join(TMP, "runA")
model_a = Toy()
plug_a = MicroEngramPlugin()
plug_a.post_model_load(cfg(unfrozen_parameters=[".+micro_engram"]), model_a)
axolotl_unfreeze(model_a, ".+micro_engram")
eng_a = plug_a.slots[0].engram
with torch.no_grad():  # "training"
    eng_a.value.weight.normal_(std=0.05)
    eng_a.tables[1].weight.add_(0.3)
    eng_a.q_scale.mul_(1.7)
check("run A: one engram at layer 0, seed 0 (same hashing as before multi-engram)",
      len(plug_a.slots) == 1 and plug_a.slots[0].spec["seed"] == 0)
fake_checkpoint(model_a, plug_a, run_a, 10)
fake_checkpoint(model_a, plug_a, run_a, 20)
os.remove(os.path.join(run_a, "checkpoint-20", "trainer_state.json"))  # still being written

w, spec = resolve_engram_source(run_a, 0)
check("init_from an output_dir picks the newest COMPLETE checkpoint",
      w.endswith(os.path.join("checkpoint-10", "pytorch_model_fsdp_0")) and spec["gate"] == "scalar_sqrt")

# ----------------------------------------------------------------------------- run B
run_b = os.path.join(TMP, "runB")
model_b = Toy()
plug_b = MicroEngramPlugin()
cfg_b = cfg(unfrozen_parameters=[".+micro_engram"], micro_engram_init_from=run_a, micro_engram_frozen=True,
            micro_engram_table_size=999,  # wrong on purpose: the trained engram's settings win
            micro_engram_extra=[DictDefault(layer=2, gate="scalar")])  # as Axolotl passes it
plug_b.post_model_load(cfg_b, model_b)
e0, e2 = plug_b.slots[0].engram, plug_b.slots[1].engram
check("run B: frozen layer-0 engram equals run A's (trained settings used, not the YAML's)",
      e0.table_size == 257 and torch.equal(e0.tables[1].weight, eng_a.tables[1].weight)
      and torch.equal(e0.q_scale, eng_a.q_scale) and torch.equal(e0.mults, eng_a.mults))
check("run B: layer-2 engram is new, its own gate, seed 10007*2, starts as a no-op",
      e2.gate == "scalar" and plug_b.slots[1].spec["seed"] == 20014
      and not torch.equal(e2.mults, e0.mults) and e2.value.weight.abs().sum() == 0)

axolotl_unfreeze(model_b, ".+micro_engram")  # Axolotl's freezing runs after post_model_load...
plug_b.post_trainer_create(cfg_b, None)        # ...and the plugin re-freezes the frozen engram
check("frozen engram stays frozen although '.+micro_engram' matches it; the new one trains",
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
    ref = model_b(input_ids=ids)  # training-time path: both engrams via the plugin hooks

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

# ----------------------------------------------------------------------------- log callbacks
import logging  # noqa: E402

from micro_engram.plugin import _ReportTrainableCallback, _StatsCallback  # noqa: E402

model_b.train()
model_b(input_ids=ids)  # records gate stats
logs = {"loss": 1.0}
_StatsCallback(plug_b.slots, 10).on_log(None, types.SimpleNamespace(is_world_process_zero=True, global_step=10),
                                        None, logs=logs)
check("stats: per-engram keys (engram_L2/...), none for the frozen engram",
      "engram_L2/out_abs_mean" in logs and "engram_L2/gate_mean" in logs
      and not any(k.startswith("engram_L0/") or k.startswith("engram/") for k in logs))


class _Grab(logging.Handler):
    def __init__(self):
        super().__init__()
        self.msgs = []

    def emit(self, record):
        self.msgs.append(record.getMessage())


grab = _Grab()
logging.getLogger("micro_engram").addHandler(grab)
_ReportTrainableCallback(plug_b.slots).on_train_begin(
    None, types.SimpleNamespace(is_world_process_zero=True), None, model=model_b)
report = " | ".join(grab.msgs)
check("train-start report lists both engrams, the frozen one as frozen, and no errors",
      "layer 0: 0.0M of" in report and "(frozen)" in report and "layer 2:" in report
      and "FROZEN" not in report and "should be frozen" not in report)

# ----------------------------------------------------------------------------- adapter checkpoints
# With an adapter (placeholder QLoRA) the Trainer saves FSDP checkpoints adapter_only=True;
# Accelerate then kept only the LoRA and the engram was lost (~100 KB checkpoints).
try:
    from peft import LoraConfig, get_peft_model
    from accelerate.utils import fsdp_utils
    from torch.distributed.checkpoint.state_dict import StateDictOptions
    have_peft = True
except ImportError as e:
    print(f"   (adapter checkpoint checks skipped: {e})")
    have_peft = False

if have_peft:

    def peft_with_engram():
        m = get_peft_model(Toy(), LoraConfig(r=1, lora_alpha=1, target_modules=["model.language_model.layers.3.lin"]))
        p = MicroEngramPlugin()
        p.post_model_load(cfg(adapter="qlora", unfrozen_parameters=[".+micro_engram"]), m)
        axolotl_unfreeze(m, ".+micro_engram")
        return m, p.slots[0].engram

    opts = StateDictOptions(full_state_dict=False)  # SHARDED_STATE_DICT, as in your runs
    unpatched = fsdp_utils._get_model_state_dict  # the plugin installs the fix when it loads
    pm, pe = peft_with_engram()
    before = unpatched(pm, adapter_only=True, sd_options=opts)
    check("reproduced: unpatched adapter_only checkpoint has the LoRA but no engram",
          any("lora" in k for k in before) and not any("micro_engram" in k for k in before))

    check("the plugin installs the fix itself on an adapter + FSDP run",
          getattr(fsdp_utils._get_model_state_dict, "_micro_engram_patched", False))
    after = fsdp_utils._get_model_state_dict(pm, adapter_only=True, sd_options=opts)
    check("fixed: adapter_only checkpoint holds the engram, not the frozen LoRA",
          any(k.endswith("micro_engram.tables.0.weight") for k in after)
          and not any("lora" in k for k in after))

    with torch.no_grad():
        pe.value.weight.normal_(std=0.05)
        pe.tables[0].weight.add_(0.5)
    ck = os.path.join(TMP, "adapter_ckpt")
    dcp.save(fsdp_utils._get_model_state_dict(pm, adapter_only=True, sd_options=opts), checkpoint_id=ck, no_dist=True)
    pm2, pe2 = peft_with_engram()  # fresh run resuming: same steps as load_fsdp_model
    template = fsdp_utils._get_model_state_dict(pm2, adapter_only=True, sd_options=opts)
    dcp.load(template, checkpoint_id=ck, no_dist=True)
    fsdp_utils._set_model_state_dict(pm2, template, adapter_only=True, sd_options=opts)
    check("resume from an adapter_only checkpoint restores the engram",
          torch.equal(pe2.tables[0].weight, pe.tables[0].weight) and torch.equal(pe2.value.weight, pe.value.weight))
    check("assemble/inference loaders read the engram from it (PEFT key prefix stripped)",
          torch.equal(load_engram_weights(ck, layer=0)["value.weight"], pe.value.weight))

# ----------------------------------------------------------------------------- legacy
legacy = {"layer": 1, "table_size": 64, "engram_dim": 8, "num_heads": 2, "weights": "micro_engram.pt"}
sp = engram_specs(legacy, "/x/micro_engram_config.json")
check("old config (no 'engrams', no 'seed'): one engram, seed 0, weights resolved",
      len(sp) == 1 and sp[0]["seed"] == 0 and sp[0]["layer"] == 1 and sp[0]["weights"] == "/x/micro_engram.pt")
try:
    MicroEngramPlugin().post_model_load(cfg(micro_engram_extra=[{"layer": 0}]), Toy())
    check("two engrams in one layer are rejected", False)
except ValueError:
    check("two engrams in one layer are rejected", True)
try:
    MicroEngramPlugin().post_model_load(cfg(micro_engram_frozen=True), Toy())
    check("frozen without init_from is rejected", False)
except ValueError:
    check("frozen without init_from is rejected", True)

shutil.rmtree(TMP, ignore_errors=True)
bad = 0
for name, ok in results:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    bad += not ok
sys.exit(1 if bad else 0)
