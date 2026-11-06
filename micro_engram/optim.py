"""LazyAdamW: AdamW where some parameter groups only update the rows that got a gradient.

For the engram tables. A batch touches a small share of the rows; with plain AdamW every
row that was used recently keeps moving for a while on its leftover momentum, even in
steps where its n-gram is absent. With `lazy=True` a row with an all-zero gradient is left
exactly as it is (parameter and both moments), like torch.optim.SparseAdam / TensorFlow's
LazyAdam, but on ordinary dense gradients, so it works under FSDP2. FSDP2 shards along
rows (dim 0), so each rank decides on its own rows with no communication.

Everything else is plain AdamW (decoupled weight decay, the same bias correction with a
global step count). State names match torch.optim.AdamW (step, exp_avg, exp_avg_sq), so
checkpoints can switch between the two.
"""
import torch


def _local(t):
    """The local shard of an FSDP2 DTensor (same storage), else t itself."""
    to_local = getattr(t, "to_local", None)
    return to_local() if to_local is not None else t


class LazyAdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, lazy=False):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, lazy=lazy))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr, eps, wd = group["lr"], group["eps"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:
                    state["step"] = torch.zeros((), dtype=torch.float32)
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state["step"] += 1
                step = float(state["step"])
                bc1, bc2 = 1 - beta1 ** step, 1 - beta2 ** step

                P, G = _local(p), _local(p.grad)
                M, V = _local(state["exp_avg"]), _local(state["exp_avg_sq"])
                rows = None
                if group["lazy"] and P.dim() >= 2:
                    rows = (G != 0).flatten(1).any(dim=1).nonzero().squeeze(1)
                    if rows.numel() == 0:
                        continue
                    p_, g = P.index_select(0, rows).float(), G.index_select(0, rows).float()
                    m, v = M.index_select(0, rows).float(), V.index_select(0, rows).float()
                else:
                    p_, g, m, v = P.float(), G.float(), M.float(), V.float()

                if wd:
                    p_.mul_(1 - lr * wd)
                m.mul_(beta1).add_(g, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                denom = (v.sqrt() / bc2 ** 0.5).add_(eps)
                p_.addcdiv_(m, denom, value=-lr / bc1)

                if rows is None:
                    P.copy_(p_)
                    M.copy_(m)
                    V.copy_(v)
                else:
                    P.index_copy_(0, rows, p_.to(P.dtype))
                    M.index_copy_(0, rows, m.to(M.dtype))
                    V.index_copy_(0, rows, v.to(V.dtype))
        return loss
