"""
LazyAdamW (micro_engram_optimizer: lazy_adam) on CPU:  python test_optim.py
"""
import os
import sys
import types

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from micro_engram.optim import LazyAdamW  # noqa: E402

results = []


def check(name, ok):
    results.append((name, bool(ok)))


def run(opt_cls, grads, init, lazy=True, wd=0.0, dtype=torch.float32, **kw):
    p = nn.Parameter(init.clone().to(dtype))
    if opt_cls is LazyAdamW:
        kw["lazy"] = lazy  # (an earlier version forgot this, so "lazy" runs were plain AdamW)
    opt = opt_cls([p], lr=1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=wd, **kw)
    for g in grads:
        opt.zero_grad()
        p.grad = g.clone().to(dtype)
        opt.step()
    return p.detach().float(), opt


torch.manual_seed(0)
R, C = 16, 8
init = torch.randn(R, C)
dense_grads = [torch.randn(R, C) for _ in range(5)]

# 1. every row gets a gradient every step: identical to torch AdamW
for wd in (0.0, 0.1):
    a, _ = run(torch.optim.AdamW, dense_grads, init, wd=wd, foreach=False)
    b, _ = run(LazyAdamW, dense_grads, init, wd=wd, lazy=True)
    check(f"all rows touched (wd {wd}): LazyAdamW == AdamW", torch.allclose(a, b, atol=1e-6))

# 2. lazy=False is plain AdamW, also with sparse gradients
sparse_grads = []
for i in range(5):
    g = torch.randn(R, C)
    g[4:] = 0
    if i > 0:
        g[0] = 0  # row 0 only touched at step 1
    sparse_grads.append(g)
a, _ = run(torch.optim.AdamW, sparse_grads, init, foreach=False)
b, _ = run(LazyAdamW, sparse_grads, init, lazy=False)
check("lazy=False == AdamW (sparse gradients too)", torch.allclose(a, b, atol=1e-6))

# 3. lazy: a row with no gradient this step does not move; always-touched rows match AdamW
c, opt = run(LazyAdamW, sparse_grads, init, lazy=True)
after_step1, _ = run(LazyAdamW, sparse_grads[:1], init, lazy=True)
check("never-touched rows are exactly unchanged", torch.equal(c[4:], init[4:]))
check("a row used only at step 1 stays where step 1 left it (no momentum drift)", torch.equal(c[0], after_step1[0]))
check("AdamW, by contrast, keeps moving that row", not torch.allclose(a[0], after_step1[0]))
check("rows touched every step match AdamW", torch.allclose(c[1:4], a[1:4], atol=1e-6))
st = opt.state[opt.param_groups[0]["params"][0]]
check("state names as torch AdamW (step, exp_avg, exp_avg_sq)", set(st) == {"step", "exp_avg", "exp_avg_sq"})
check("moments of untouched rows stay zero", torch.equal(st["exp_avg"][4:], torch.zeros(R - 4, C)))

# 4. bf16 parameters (how the engram trains by default)
d, optb = run(LazyAdamW, sparse_grads, init, lazy=True, dtype=torch.bfloat16)
stb = optb.state[optb.param_groups[0]["params"][0]]
check("bf16: untouched rows bit-identical, touched rows move, state bf16",
      torch.equal(d[4:], init[4:].bfloat16().float()) and not torch.equal(d[1:4], init[1:4].bfloat16().float())
      and stb["exp_avg"].dtype == torch.bfloat16)

# 5. under FSDP2 (single process, gloo): sharded params, optimizer state save/load
try:
    import torch.distributed as dist
    from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict, set_optimizer_state_dict
    from torch.distributed.fsdp import fully_shard

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29533")
    dist.init_process_group("gloo", rank=0, world_size=1)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(C, C)
            self.tables = nn.Embedding(R, C)

        def forward(self, ids):
            return self.lin(self.tables(ids))

    torch.manual_seed(1)
    blk = Block()
    fully_shard(blk)
    opt = LazyAdamW([{"params": [blk.tables.weight], "lazy": True}, {"params": [blk.lin.weight, blk.lin.bias]}], lr=1e-2)
    before = blk.tables.weight.full_tensor().clone()
    row1_after_step1 = None
    for i in range(3):
        opt.zero_grad()
        ids = [[1, 2, 3]] if i == 0 else [[2, 3]]  # row 1 only in the first step
        blk(torch.tensor(ids)).pow(2).sum().backward()
        opt.step()
        if i == 0:
            row1_after_step1 = blk.tables.weight.full_tensor()[1].clone()
    w = blk.tables.weight.full_tensor()
    check("FSDP2: rows used by the batch move, the others are unchanged",
          not torch.equal(w[2:4], before[2:4]) and torch.equal(w[4:], before[4:]) and torch.equal(w[0], before[0]))
    check("FSDP2: a row used only in step 1 stays put afterwards (lazy, on sharded DTensors)",
          not torch.equal(row1_after_step1, before[1]) and torch.equal(w[1], row1_after_step1))
    sd = get_optimizer_state_dict(blk, opt)
    opt2 = LazyAdamW([{"params": [blk.tables.weight], "lazy": True}, {"params": [blk.lin.weight, blk.lin.bias]}], lr=1e-2)
    set_optimizer_state_dict(blk, opt2, sd)
    s1, s2 = opt.state[blk.tables.weight], opt2.state[blk.tables.weight]
    check("FSDP2: optimizer state saves and loads (DCP state-dict API)",
          torch.equal(s1["exp_avg"].full_tensor(), s2["exp_avg"].full_tensor()) and float(s2["step"]) == 3)
    dist.destroy_process_group()
except Exception as e:  # noqa: BLE001
    check(f"FSDP2 checks ({type(e).__name__}: {e})", False)

# 6. plugin: the trainer builds LazyAdamW lazily, tables in the lazy group (Axolotl stubbed)
for n in ("axolotl", "axolotl.integrations", "axolotl.integrations.base"):
    sys.modules.setdefault(n, types.ModuleType(n))
sys.modules["axolotl.integrations.base"].BasePlugin = type("BasePlugin", (), {"__init__": lambda self: None})
from micro_engram.module import MicroEngram  # noqa: E402
from micro_engram.plugin import _install_lazy_adam  # noqa: E402

model = nn.Module()
model.layer = nn.Module()
model.layer.micro_engram = MicroEngram(model_dim=8, table_size=31, engram_dim=4, num_heads=1, vocab_size=50,
                                       gate="scalar_sqrt")
model.frozen = nn.Linear(8, 8)
model.frozen.requires_grad_(False)
trainer = types.SimpleNamespace(optimizer=None, model=model, args=types.SimpleNamespace(
    learning_rate=1e-3, adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-8, weight_decay=0.0),
    create_optimizer=lambda model=None: None)
_install_lazy_adam(trainer)
check("plugin: nothing is built before the trainer asks (FSDP shards first)", trainer.optimizer is None)
opt = trainer.create_optimizer()
lazy = {id(p) for g in opt.param_groups if g["lazy"] for p in g["params"]}
tables = {id(t.weight) for t in model.layer.micro_engram.tables}
others = {id(p) for n, p in model.named_parameters() if p.requires_grad and "tables" not in n}
dense = {id(p) for g in opt.param_groups if not g["lazy"] for p in g["params"]}
check("plugin: exactly the tables are lazy; key/value/scales are AdamW; frozen params left out",
      lazy == tables and dense == others and id(model.frozen.weight) not in lazy | dense)

bad = 0
for name, ok in results:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    bad += not ok
sys.exit(1 if bad else 0)
