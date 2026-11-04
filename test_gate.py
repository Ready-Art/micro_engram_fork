"""
Checks both gate modes on CPU. Run in the venv:  python test_gate.py
"""
import sys

import torch

from micro_engram.module import MicroEngram

V, D = 1000, 32


def make(gate, **kw):
    return MicroEngram(model_dim=D, table_size=257, engram_dim=4, num_heads=2,
                       vocab_size=V, gate=gate, **kw)


results = []
torch.manual_seed(0)
ids = torch.randint(5, V, (2, 12))
hidden = torch.randn(2, 12, D) * 50  # large, like Gemma's residual stream

for gate in MicroEngram.GATES:
    eng = make(gate)
    out = eng(hidden, ids)
    results.append((f"{gate}: step 0 is an exact no-op", torch.equal(out, hidden)))

    # at step 0 only the zero-init output layer gets a gradient (same as the paper's setup)
    h = hidden.clone().requires_grad_(True)
    eng(h, ids).pow(2).sum().backward()
    out_layer = eng.up if gate == "ple" else eng.value
    results.append((f"{gate}: output layer gets a gradient at step 0",
                    out_layer.weight.grad is not None and out_layer.weight.grad.abs().sum() > 0))

    # once the output layer is non-zero, the tables learn too and the output changes
    eng.zero_grad()
    with torch.no_grad():
        out_layer.weight.normal_(std=0.02)
    eng(hidden, ids).pow(2).sum().backward()
    results.append((f"{gate}: tables get a gradient after the first update",
                    eng.tables[0].weight.grad.abs().sum() > 0))
    results.append((f"{gate}: output differs from input once trained",
                    not torch.equal(eng(hidden, ids), hidden)))

    # bf16, as in training
    eb = make(gate).to(torch.bfloat16)
    ob = eb(hidden.to(torch.bfloat16), ids)
    results.append((f"{gate}: bf16 forward keeps dtype and shape",
                    ob.dtype == torch.bfloat16 and ob.shape == hidden.shape))

s, p = make("scalar"), make("ple")
results.append(("scalar keeps its old parameter names (key, value)",
                {"key.weight", "value.weight"} <= set(s.state_dict())))
results.append(("ple has down/up and no key/value",
                {"down.weight", "up.weight"} <= set(p.state_dict()) and "key.weight" not in p.state_dict()))
results.append(("both modes have the same parameter count",
                sum(x.numel() for x in s.parameters()) == sum(x.numel() for x in p.parameters())))
results.append(("an old scalar checkpoint still loads strictly",
                not make("scalar").load_state_dict(s.state_dict(), strict=True).missing_keys))
q = make("scalar_sqrt")
results.append(("scalar_sqrt has key/value plus learnable q_scale/k_scale (init 1)",
                {"key.weight", "value.weight", "q_scale", "k_scale"} <= set(q.state_dict())
                and torch.equal(q.q_scale, torch.ones(D)) and q.q_scale.requires_grad))
results.append(("scalar_sqrt: old scalar checkpoint has no scales (so it never loads by accident)",
                set(make("scalar_sqrt").load_state_dict(s.state_dict(), strict=False).missing_keys)
                == {"q_scale", "k_scale"}))

# the signed square root keeps the gate away from saturation for the same alignment
def gate_of(eng, h, ids):
    eng.train()
    eng._gate_stats = None
    eng(h, ids)
    return eng.pop_gate_stats()

torch.manual_seed(1)
sq, sc = make("scalar_sqrt"), make("scalar")
with torch.no_grad():
    sc.key.weight.copy_(sq.key.weight)
    sc.tables.load_state_dict(sq.tables.state_dict())
    for e in (sq, sc):
        e.value.weight.normal_(std=0.02)
# make h point along the key, so the score is large (cos ~ 1)
idx = sq.hash_ids(ids)
mem = torch.cat([t(i) for t, i in zip(sq.tables, idx)], dim=-1)
h_aligned = sq.key(mem).detach() * 30
g_sq, sat_sq = gate_of(sq, h_aligned, ids)
g_sc, sat_sc = gate_of(sc, h_aligned, ids)
results.append((f"aligned input: plain gate saturates ({g_sc:.4f}), sqrt gate less ({g_sq:.4f})",
                g_sc > g_sq > 0.5 and sat_sc >= sat_sq))
results.append(("gate stats are only recorded in training mode",
                (sq.eval(), sq(h_aligned, ids), sq.pop_gate_stats())[2] is None))

# q/k scales learn
sq.train()
sq.zero_grad()
sq(h_aligned + torch.randn_like(h_aligned), ids).pow(2).sum().backward()
results.append(("scalar_sqrt: q_scale and k_scale get gradients",
                sq.q_scale.grad is not None and sq.q_scale.grad.abs().sum() > 0
                and sq.k_scale.grad.abs().sum() > 0))

# weight stats: fresh tables are at the init RMS, and none count as learned
from micro_engram.stats import format_stats, module_stats, weight_stats  # noqa: E402

big = MicroEngram(model_dim=D, table_size=4096, engram_dim=64, num_heads=2, vocab_size=V,  # 64 wide, as in real runs
                  gate="scalar_sqrt")
st = module_stats(big)
results.append((f"fresh tables: rms ~0.02 ({st['engram/table_rms_2g']:.4f}), 0% learned, out|w| 0",
                abs(st["engram/table_rms_2g"] - 0.02) < 0.002 and st["engram/rows_learned_3g"] < 0.001
                and st["engram/out_abs_mean"] == 0 and abs(st["engram/q_scale_mean"] - 1) < 1e-6))
with torch.no_grad():
    big.tables[2].weight[:1024] *= 3  # table 2 is a 3-gram table (2 heads per order)
st2 = module_stats(big)
results.append((f"grown rows are counted per order (3g learned {st2['engram/rows_learned_3g']:.1%}, 2g "
                f"{st2['engram/rows_learned_2g']:.1%})",
                0.10 < st2["engram/rows_learned_3g"] < 0.15 and st2["engram/rows_learned_2g"] < 0.001))
results.append(("stats from saved tensors == stats from the module",
                weight_stats(big.state_dict(), big.orders, 2) == st2))
print("   example line:", format_stats({**st2, "engram/gate_mean": 0.42, "engram/gate_saturated": 0.03}))

try:
    make("bogus")
    results.append(("unknown gate is rejected", False))
except ValueError:
    results.append(("unknown gate is rejected", True))

bad = 0
for name, ok in results:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    bad += not ok
sys.exit(1 if bad else 0)
