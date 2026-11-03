"""
CPU checks for the vLLM path. Run in the venv:  python test_vllm_ids.py

Simulates vLLM batches (chunked prefill, prefix-cache hits, decode steps, mixed
batches, shuffled request slots, CUDA-graph padding) and checks that the engram sees
the same tokens and produces the same output as in training.
"""
import random
import sys

import torch

from micro_engram.module import MicroEngram
from micro_engram_vllm.ids import fill_engram_ids

V, D, MAX_REQS, MAX_LEN = 1000, 16, 8, 64
random.seed(0)
torch.manual_seed(0)
results = []


def make(gate="scalar"):
    eng = MicroEngram(model_dim=D, table_size=257, engram_dim=4, num_heads=2,
                      vocab_size=V, pad_id=0, gate=gate)
    with torch.no_grad():
        (eng.value if gate == "scalar" else eng.up).weight.normal_()
    canon = torch.arange(V)
    canon[11], canon[501] = 10, 500  # some token merging
    eng.canon.copy_(canon)
    return eng


def run_step(chunks, histories, slots):
    """chunks: list of (request, first position, count). Returns the filled buffer rows."""
    input_ids, positions, qsl = [], [], [0]
    for r, start, n in chunks:
        input_ids += histories[r][start:start + n]
        positions += list(range(start, start + n))
        qsl.append(qsl[-1] + n)
    num_tokens = len(input_ids)
    padded = num_tokens + 3  # pretend CUDA graphs pad the batch
    # history table: real tokens before each chunk, garbage from the chunk on, to prove
    # the current chunk is never read from history
    table = torch.full((MAX_REQS, MAX_LEN), 999, dtype=torch.int32)
    for r, start, _ in chunks:
        table[slots[r], :start] = torch.tensor(histories[r][:start], dtype=torch.int32)
    out = torch.full((num_tokens + 10, 3), 777, dtype=torch.int32)
    fill_engram_ids(
        out,
        torch.tensor(input_ids + [5] * 3, dtype=torch.int32),
        torch.tensor(positions + [0] * 3),
        torch.tensor([slots[r] for r, _, _ in chunks], dtype=torch.int32),
        torch.tensor(qsl, dtype=torch.int32),
        table, num_reqs=len(chunks), num_tokens=num_tokens, num_tokens_padded=padded,
    )
    return out, num_tokens, padded


def expected(chunks, histories):
    rows = []
    for r, start, n in chunks:
        h = histories[r]
        for p in range(start, start + n):
            rows.append([h[p], h[p - 1] if p >= 1 else -1, h[p - 2] if p >= 2 else -1])
    return torch.tensor(rows, dtype=torch.int32)


# 1. hash_prev gives the same table rows as training's hash_ids
eng = make()
for doc_breaks in (False, True):
    ids = torch.randint(5, V, (1, 30))
    ids[0, 3], ids[0, 9] = 11, 501  # merged tokens
    pos = None
    if doc_breaks:  # packed batch: documents start at 0, 12 and 25
        pos = torch.cat([torch.arange(12), torch.arange(13), torch.arange(5)]).unsqueeze(0)
    ref = eng.hash_ids(ids, pos)
    p = pos[0] if pos is not None else torch.arange(30)
    rows = [[int(ids[0, t]),
             int(ids[0, t - 1]) if p[t] >= 1 else -1,
             int(ids[0, t - 2]) if p[t] >= 2 else -1] for t in range(30)]
    got = eng.hash_prev(torch.tensor(rows))
    results.append((f"hash_prev == hash_ids ({'packed docs' if doc_breaks else 'one doc'})",
                    all(torch.equal(a[0], b) for a, b in zip(ref, got))))

# 2. the right previous tokens in every kind of vLLM step
hist = {r: [random.randrange(5, V) for _ in range(MAX_LEN)] for r in range(4)}
slots = {0: 5, 1: 2, 2: 7, 3: 0}  # requests sit in arbitrary rows of the history table
cases = {
    "full prefill": [(0, 0, 20)],
    "chunked prefill, 2nd chunk": [(0, 20, 15)],
    "prefix-cache hit (chunk starts at 33)": [(1, 33, 4)],
    "decode steps": [(0, 40, 1), (1, 37, 1), (2, 2, 1)],
    "decode at position 0 and 1": [(3, 0, 1), (2, 1, 1)],
    "mixed prefill + decode": [(0, 41, 1), (1, 0, 9), (2, 10, 2), (3, 50, 1)],
}
for name, chunks in cases.items():
    out, n, padded = run_step(chunks, hist, slots)
    ok = torch.equal(out[:n], expected(chunks, hist))
    ok &= bool((out[n:padded] == -1).all()) and bool((out[padded:] == 777).all())
    results.append((f"token ids: {name}", ok))

# 3. end to end: a request served step by step in vLLM matches the training forward
for gate in ("scalar", "ple"):
    eng = make(gate)
    seq = torch.tensor(hist[0][:30])
    seq[4], seq[17] = 11, 501
    hidden = torch.randn(30, D) * 30
    train = eng(hidden.unsqueeze(0), seq.unsqueeze(0))[0]  # training: whole sequence at once
    h = {0: seq.tolist()}
    served = []
    for chunk in [(0, 0, 13), (0, 13, 9)] + [(0, p, 1) for p in range(22, 30)]:
        out, n, _ = run_step([chunk], h, {0: 3})
        _, start, cnt = chunk
        served.append(eng.inject(hidden[start:start + cnt], eng.hash_prev(out[:n])))
    results.append((f"{gate}: chunked prefill + decode == training forward",
                    torch.allclose(torch.cat(served), train, atol=1e-5)))

bad = 0
for name, ok in results:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    bad += not ok
sys.exit(1 if bad else 0)
