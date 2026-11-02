"""
Checks the token-merging logic on CPU. Run in the venv:  python test_canon.py
"""
import sys

import torch
import torch.nn.functional as F

from micro_engram.module import MicroEngram, _PRIME

V = 1000


def make(canon=None):
    return MicroEngram(model_dim=16, table_size=257, engram_dim=4, num_heads=2,
                       vocab_size=V, canon=canon)


def reference_hash(eng, input_ids):
    """The ORIGINAL hashing (before token merging existed), written out independently."""
    ids = input_ids.long()
    T = ids.shape[-1]
    max_n = max(eng.orders)
    padded = F.pad(ids, (max_n - 1, 0), value=eng.pad_id)
    shifted = [padded[:, max_n - 1 - i: max_n - 1 - i + T] for i in range(max_n)]
    out = []
    for k, n in enumerate(eng._table_orders):
        x = shifted[0] * eng.mults[k, 0]
        for i in range(1, n):
            x = x + shifted[i] * eng.mults[k, i]
        out.append((x % _PRIME) % eng.table_size)
    return out


def same(a, b):
    return all(torch.equal(x, y) for x, y in zip(a, b))


results = []
torch.manual_seed(0)

# 1. without a map, hashing is identical to the original code
ids = torch.randint(5, V, (3, 20))
eng = make()
results.append(("no map == original hashing", same(eng.hash_ids(ids), reference_hash(eng, ids))))

# 2. merged tokens hash the same; unmerged ones still differ
canon = torch.arange(V)
canon[11] = 10
canon[501] = 500
eng = make(canon)
a = torch.tensor([[3, 10, 7, 500, 9, 10, 4]])
b = torch.tensor([[3, 11, 7, 501, 9, 11, 4]])  # merged variants of a
c = torch.tensor([[3, 12, 7, 500, 9, 10, 4]])  # 12 is NOT merged with 10
results.append(("merged variants hash identically", same(eng.hash_ids(a), eng.hash_ids(b))))
results.append(("unmerged tokens still differ", not same(eng.hash_ids(a), eng.hash_ids(c))))

# 3. the map is saved in the state dict, and an old checkpoint (no canon) still loads
results.append(("canon is in the state dict", "canon" in eng.state_dict()))
old = {k: v for k, v in make().state_dict().items() if k != "canon"}
res = make().load_state_dict(old, strict=False)
results.append(("old checkpoint: only canon missing",
                list(res.missing_keys) == ["canon"] and not res.unexpected_keys))

# 4. the map round-trips through a checkpoint
e2 = make()
e2.load_state_dict(eng.state_dict())
results.append(("canon round-trips", torch.equal(e2.canon, canon)))

bad = 0
for name, ok in results:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    bad += not ok
sys.exit(1 if bad else 0)