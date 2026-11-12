"""Vocab-parallel embedding + chunked softcapped loss (micro_engram/train/head.py) vs a plain
full-logit reference, on 1 and 3 processes (gloo, CPU): loss, gradients wrt the hidden states (with a
different loss weight per rank), and the embedding lookup.  python test_head.py
"""
import os, sys, torch, torch.distributed as dist, torch.multiprocessing as mp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
def run(rank, world, port):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    from micro_engram.train.head import VocabShard, reference_loss_sum
    torch.manual_seed(0)
    V, d, N = 103, 16, 37
    W = torch.randn(V, d, dtype=torch.float64)
    Hs = torch.randn(world, N, d, dtype=torch.float64)
    Ls = torch.randint(0, V, (world, N)); Ls[:, ::5] = -100
    for cap in (None, 3.0):
        sh = VocabShard(W, rank, world, embed_scale=2.0, softcap=cap, chunk_tokens=8)
        h = Hs[rank].clone().requires_grad_()
        tot, cnt = sh.loss_sum(h, Ls[rank])
        (tot * (rank + 1)).backward()
        ref_h = Hs[rank].clone().requires_grad_()
        ref = reference_loss_sum(ref_h, Ls[rank], W, cap)
        (ref * (rank + 1)).backward()
        assert torch.allclose(tot, ref), (tot, ref)
        assert torch.allclose(h.grad, ref_h.grad, atol=1e-5), (h.grad - ref_h.grad).abs().max()
        assert int(cnt) == int((Ls[rank] != -100).sum())
    ids = torch.randint(0, V, (2, 9)) + rank
    e = sh.embed(ids)
    assert torch.allclose(e, W[ids] * 2.0)
    if rank == 0: print("head ok")
    dist.destroy_process_group()
if __name__ == "__main__":
    for world in (1, 3):
        mp.spawn(run, args=(world, 29511 + world), nprocs=world)
