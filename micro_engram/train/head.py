"""Vocab-parallel embedding and loss: the tied embedding / lm_head split by rows over the GPUs.

Gemma's embedding is tied to the output head (262,144 x 5,376, ~2.7 GiB in bf16). Under
FSDP it sat whole on every GPU through the forward and backward (root unit). Here each
GPU keeps only its own rows (1/world), outside FSDP, frozen:

  embedding  every GPU asks the owners for the rows of its tokens (all_to_all, the same
             exchange as the row-sharded engram tables), then scales by sqrt(hidden).
  loss       chunks of tokens: every GPU gathers the chunk's hidden states from all GPUs,
             computes logits against its own vocab rows (with Gemma's final logit softcap),
             and the GPUs combine max / sum-exp / target logit with three small all_reduces.
             The backward recomputes the chunk's logits and reduce-scatters the hidden-state
             gradients back to their GPUs. Full logits never exist; no GPU holds the whole head.

Loss terms are summed (not averaged): the trainer divides by the number of trained tokens of
the whole optimizer step, across micro-batches and GPUs, so every token weighs the same.
"""
import torch
import torch.distributed as dist

from ..rows import _RowLookup, _group, row_range

IGNORE = -100


class VocabShard:
    def __init__(self, weight, rank, world, embed_scale=None, softcap=None, chunk_tokens=2048,
                 chunk_bytes=512 << 20, device=None, dtype=None):
        """weight: the full [V, d] embedding (rank's slice is copied) or None with `local` set later."""
        self.V, self.d = int(weight.shape[0]), int(weight.shape[1])
        self.rank, self.world = rank, world
        self.start, self.rows = row_range(self.V, rank, world)
        self.per = -(-self.V // world)
        self.weight = weight[self.start:self.start + self.rows].detach().to(
            device=device, dtype=dtype or weight.dtype).clone()
        self.embed_scale = embed_scale
        self.softcap = softcap
        # tokens per chunk: world * chunk logits [world*c, rows] in fp32 stay under chunk_bytes
        fit = max(64, chunk_bytes // max(1, world * max(self.rows, 1) * 4))
        self.chunk = max(1, min(chunk_tokens, fit))

    @property
    def group(self):
        return _group(self.world)

    @torch.no_grad()
    def embed(self, ids):
        """[*ids.shape, d] embeddings (frozen, no autograd), scaled like Gemma4TextScaledWordEmbedding."""
        flat = ids.reshape(-1).to(self.weight.device)
        out = _RowLookup.apply(flat, ([0, self.V], [self.per], self.rank, self.world), self.weight)
        out = out.reshape(*ids.shape, self.d)
        if self.embed_scale is not None:
            out = out * torch.tensor(self.embed_scale, dtype=self.weight.dtype, device=out.device)
        return out

    def loss_sum(self, hidden, labels):
        """(sum of token losses of this GPU's tokens, number of its trained tokens).
        hidden [N, d] (requires grad), labels [N] with -100 for untrained. Collective: every
        GPU calls it with the same N."""
        count = (labels != IGNORE).sum()
        total = _VocabParallelCE.apply(hidden, labels.to(hidden.device), self)
        return total, count


def _gather(x, group, world):
    if world == 1:
        return x
    out = x.new_empty((world * x.shape[0],) + tuple(x.shape[1:]))
    dist.all_gather_into_tensor(out, x.contiguous(), group=group)
    return out


def _chunk_logits(h, w, softcap):
    z = (h @ w.t()).float()
    if softcap:
        t = torch.tanh(z / softcap)
        return t * softcap, t
    return z, None


class _VocabParallelCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, labels, shard):
        world, group, w = shard.world, shard.group, shard.weight
        N = hidden.shape[0]
        c = shard.chunk
        h_local = hidden.detach().to(w.dtype)
        lse_all, tot = [], torch.zeros((), dtype=torch.float32, device=hidden.device)
        for a in range(0, N, c):
            b = min(N, a + c)
            H = _gather(h_local[a:b], group, world)                # [world*n, d], rank-major
            L = _gather(labels[a:b].contiguous(), group, world)    # [world*n]
            z, _ = _chunk_logits(H, w, shard.softcap)               # [world*n, rows]
            m = z.max(dim=1).values
            if world > 1:
                dist.all_reduce(m, op=dist.ReduceOp.MAX, group=group)
            s = torch.exp(z - m[:, None]).sum(dim=1)
            loc = L - shard.start
            mine = (L != IGNORE) & (loc >= 0) & (loc < shard.rows)
            tgt = torch.zeros_like(m)
            idx = mine.nonzero().squeeze(1)
            if idx.numel():
                tgt[idx] = z[idx, loc[idx]]
            if world > 1:
                st = torch.stack([s, tgt])
                dist.all_reduce(st, group=group)
                s, tgt = st[0], st[1]
            lse = m + torch.log(s)
            n = b - a
            own = slice(shard.rank * n, (shard.rank + 1) * n)
            valid = L[own] != IGNORE
            tot = tot + torch.where(valid, lse[own] - tgt[own], torch.zeros_like(lse[own])).sum()
            lse_all.append(lse)
            del z, H
        ctx.shard = shard
        ctx.save_for_backward(hidden.detach(), labels, *lse_all)
        return tot

    @staticmethod
    def backward(ctx, g):
        shard = ctx.shard
        hidden, labels, *lse_all = ctx.saved_tensors
        world, group, w = shard.world, shard.group, shard.weight
        N = hidden.shape[0]
        c = shard.chunk
        # every GPU's own upstream gradient (the loss weight of its tokens)
        gs = _gather(g.detach().float().reshape(1), group, world)  # [world]
        h_local = hidden.to(w.dtype)
        grad = torch.empty(N, shard.d, dtype=hidden.dtype, device=hidden.device)
        for k, a in enumerate(range(0, N, c)):
            b = min(N, a + c)
            n = b - a
            H = _gather(h_local[a:b], group, world)
            L = _gather(labels[a:b].contiguous(), group, world)
            z, t = _chunk_logits(H, w, shard.softcap)
            p = torch.exp(z - lse_all[k][:, None])                # softmax over the whole vocab, our columns
            loc = L - shard.start
            mine = (L != IGNORE) & (loc >= 0) & (loc < shard.rows)
            idx = mine.nonzero().squeeze(1)
            if idx.numel():
                p[idx, loc[idx]] -= 1
            weight = gs.repeat_interleave(n) * (L != IGNORE).float()  # [world*n]
            p.mul_(weight[:, None])
            if t is not None:
                p.mul_(1 - t * t)                                  # d softcap*tanh(x/softcap) / dx
            dH = (p.to(w.dtype) @ w).float()                       # [world*n, d], this GPU's vocab part
            if world > 1:
                out = torch.empty(n, shard.d, dtype=torch.float32, device=dH.device)
                dist.reduce_scatter_tensor(out, dH, group=group)
            else:
                out = dH
            grad[a:b] = out.to(grad.dtype)
            del z, p, dH, H
        return grad, None, None


def reference_loss_sum(hidden, labels, weight, softcap=None):
    """Plain full-logit version (tests)."""
    z = (hidden.to(weight.dtype) @ weight.t()).float()
    if softcap:
        z = torch.tanh(z / softcap) * softcap
    valid = labels != IGNORE
    lse = torch.logsumexp(z, dim=-1)
    tgt = z.gather(1, labels.clamp_min(0)[:, None]).squeeze(1)
    return torch.where(valid, lse - tgt, torch.zeros_like(lse)).sum()

