"""zero_ngram_order.py (zeroed orders, --scale) on CPU: python test_zero_order.py

Two engrams with 2/3/4-gram tables saved as a DCP checkpoint + config (as micro_engram.train
writes them). After zeroing order 4: the 4-gram tables are zero, everything else is
unchanged, the config points to the new file, every loader reads it, and the engram's
output equals the original engram with its 4-gram lookups removed.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

import torch
import torch.distributed.checkpoint as dcp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from micro_engram.module import MicroEngram, engram_from_spec  # noqa: E402
from micro_engram.weights import engram_specs, load_engram_weights  # noqa: E402

D, V = 16, 300
FAIL = []


def check(name, ok):
    print(("PASS " if ok else "FAIL ") + name)
    if not ok:
        FAIL.append(name)


tmp = tempfile.mkdtemp(prefix="zero4_")
try:
    torch.manual_seed(0)
    engs, entries = {}, []
    for layer, seed in ((0, 0), (29, 290203)):
        e = MicroEngram(model_dim=D, table_size=0, table_sizes={2: 31, 3: 37, 4: 41}, engram_dim=4, num_heads=2,
                        vocab_size=V, gate="scalar_sqrt", seed=seed, orders=(2, 3, 4))
        with torch.no_grad():
            for p in e.parameters():
                p.normal_(std=0.3)
        engs[layer] = e
        entries.append({"layer": layer, "table_size": None, "table_sizes": {"2": 31, "3": 37, "4": 41},
                        "engram_dim": 4, "num_heads": 2, "orders": [2, 3, 4], "gate": "scalar_sqrt", "seed": seed,
                        "pad_id": 0, "weights": "micro_engram_rows"})
    ckpt = os.path.join(tmp, "out", "checkpoint-5")
    sd = {f"model.language_model.layers.{L}.micro_engram.{k}": v for L, e in engs.items()
          for k, v in e.state_dict().items()}
    dcp.save(sd, checkpoint_id=os.path.join(ckpt, "micro_engram_rows"), no_dist=True)
    conf = {**{k: v for k, v in entries[0].items() if k != "weights"}, "model_dim": D, "vocab_size": V,
            "weights": "micro_engram_rows", "engrams": entries}
    with open(os.path.join(ckpt, "micro_engram_config.json"), "w") as f:
        json.dump(conf, f)
    with open(os.path.join(ckpt, "trainer_state.json"), "w") as f:
        f.write("{}")

    out = os.path.join(tmp, "zeroed")
    here = os.path.dirname(os.path.abspath(__file__))
    r = subprocess.run([sys.executable, os.path.join(here, "zero_ngram_order.py"), os.path.join(tmp, "out"),
                        "--order", "4", "--out", out], capture_output=True, text=True)
    print(r.stdout.strip())
    check("script runs on a training output_dir (newest checkpoint, DCP)", r.returncode == 0 or print(r.stderr))

    conf2_path = os.path.join(out, "micro_engram_config.json")
    specs = engram_specs(json.load(open(conf2_path)), conf2_path)
    ok_zero, ok_same, ok_out = True, True, True
    ids = torch.randint(1, V, (2, 20))
    h = torch.randn(2, 20, D)
    for s in specs:
        new = load_engram_weights(s["weights"], layer=s["layer"], need_layer=True)
        old = engs[s["layer"]].state_dict()
        for k, v in new.items():
            if k.startswith("tables."):
                order = (2, 3, 4)[int(k.split(".")[1]) // 2]
                ok_zero &= (order != 4) or bool((v == 0).all())
                ok_same &= (order == 4) or torch.equal(v, old[k])
            else:
                ok_same &= torch.equal(v, old[k])
        e2 = engram_from_spec(s, D, V)
        e2.load_state_dict(new)
        ref = engs[s["layer"]]
        with torch.no_grad():
            idx = ref.hash_ids(ids)
            mem = ref.lookup(idx)
            for k, n in enumerate(ref._table_orders):
                if n == 4:
                    mem[k] = torch.zeros_like(mem[k])
            ref.lookup = lambda _i, m=mem: m  # the original engram, 4-gram lookups removed
            ok_out &= torch.allclose(e2(h, ids), ref(h, ids), atol=1e-6)
    check("4-gram tables are all zero, 2/3-gram tables and everything else unchanged", ok_zero and ok_same)
    check("config points to consolidated.pth; both engrams load through engram_specs", len(specs) == 2
          and all(s["weights"].endswith("consolidated.pth") for s in specs))
    check("output == the original engram with its 4-gram lookups removed", ok_out)
    # --scale / --layer-scale: what the engram adds to h is scaled, nothing else changes
    out2 = os.path.join(tmp, "scaled")
    r = subprocess.run([sys.executable, os.path.join(here, "zero_ngram_order.py"), os.path.join(tmp, "out"),
                        "--scale", "0.5", "--layer-scale", "29=0.25", "--out", out2], capture_output=True, text=True)
    print(r.stdout.strip())
    conf3 = os.path.join(out2, "micro_engram_config.json")
    ok_scale = r.returncode == 0
    for s in engram_specs(json.load(open(conf3)), conf3):
        e3 = engram_from_spec(s, D, V)
        e3.load_state_dict(load_engram_weights(s["weights"], layer=s["layer"], need_layer=True))
        src = engram_from_spec(s, D, V)
        src.load_state_dict({k: v for k, v in engs[s["layer"]].state_dict().items()})
        want = 0.5 if s["layer"] == 0 else 0.25
        with torch.no_grad():
            ok_scale &= torch.allclose(e3(h, ids) - h, want * (src(h, ids) - h), atol=1e-5)
    check("--scale 0.5 / --layer-scale 29=0.25: each engram adds exactly that share of its original output", ok_scale)
finally:
    shutil.rmtree(tmp, ignore_errors=True)
print("\nALL PASSED" if not FAIL else f"\nFAILED: {FAIL}")
sys.exit(1 if FAIL else 0)
