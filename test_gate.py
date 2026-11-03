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
    out_layer = eng.value if gate == "scalar" else eng.up
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
