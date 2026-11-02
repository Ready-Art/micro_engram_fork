import torch
import torch.nn as nn
import torch.nn.functional as F

_PRIME = 2_147_483_647  # 2^31 - 1


class MicroEngram(nn.Module):
    """Hashed n-gram memory injected into the residual stream (Engram-style).

    At each position t, the module looks up the 2-grams and 3-grams that END at t
    (causal). Ids are hashed straight from input_ids, so the data pipeline needs no
    changes. With sample packing, position_ids keep n-grams inside one document.

    Token merging: `canon` maps every token id to a canonical id (e.g. case variants
    share one), applied before hashing. It is a buffer, so it is saved in checkpoints.
    canon=None means identity (no merging).
    """

    def __init__(self, model_dim, table_size=65_536, engram_dim=128,
                 num_heads=4, orders=(2, 3), pad_id=0, seed=0,
                 vocab_size=262_144, canon=None):
        super().__init__()
        self.orders = tuple(orders)
        self.table_size = table_size
        self.pad_id = pad_id
        self._table_orders = [n for n in self.orders for _ in range(num_heads)]
        n_tables = len(self._table_orders)

        self.tables = nn.ModuleList(nn.Embedding(table_size, engram_dim) for _ in range(n_tables))
        for t in self.tables:
            nn.init.normal_(t.weight, std=0.02)

        mem_dim = engram_dim * n_tables
        self.key = nn.Linear(mem_dim, model_dim, bias=False)    # used only by the gate
        self.value = nn.Linear(mem_dim, model_dim, bias=False)  # what gets injected
        nn.init.zeros_(self.value.weight)  # step 0: exact no-op

        # Deterministic from `seed`, so it can be rebuilt if a checkpoint drops buffers.
        g = torch.Generator().manual_seed(seed)
        mults = torch.randint(1, _PRIME, (n_tables, max(self.orders)), generator=g) | 1
        self.register_buffer("mults", mults)

        if canon is None:
            canon = torch.arange(vocab_size)
        else:
            canon = torch.as_tensor(canon).long().clone()
            if canon.shape != (vocab_size,):
                raise ValueError(f"canon must have shape ({vocab_size},), got {tuple(canon.shape)}")
        self.register_buffer("canon", canon)

    def hash_ids(self, input_ids, position_ids=None):
        ids = input_ids.long()
        canon = self.canon.to(ids.device)  # buffers may sit on CPU under param offload
        ids = canon[ids.clamp(0, canon.numel() - 1)]  # token merging
        T = ids.shape[-1]
        max_n = max(self.orders)
        padded = F.pad(ids, (max_n - 1, 0), value=self.pad_id)
        # shifted[i][:, t] == token at position t - i
        shifted = [padded[:, max_n - 1 - i: max_n - 1 - i + T] for i in range(max_n)]
        if position_ids is not None:
            # Packed batches: position_ids restart at 0 for every document, so token
            # t-i is in the same document only when pos[t] >= i.
            pos = position_ids.to(ids.device).expand_as(ids)
            shifted = [s if i == 0 else torch.where(pos >= i, s, torch.full_like(s, self.pad_id))
                       for i, s in enumerate(shifted)]
        mults = self.mults.to(ids.device)  # buffers may sit on CPU under param offload
        out = []
        for k, n in enumerate(self._table_orders):
            x = shifted[0] * mults[k, 0]
            for i in range(1, n):
                x = x + shifted[i] * mults[k, i]  # < 2^51, safe in int64
            out.append((x % _PRIME) % self.table_size)
        return out

    def forward(self, hidden, input_ids, position_ids=None):
        idx = self.hash_ids(input_ids.to(hidden.device), position_ids)
        mem = torch.cat([tab(i) for tab, i in zip(self.tables, idx)], dim=-1)
        d = hidden.shape[-1]
        h = F.rms_norm(hidden.float(), (d,))
        k = F.rms_norm(self.key(mem).float(), (d,))
        alpha = torch.sigmoid((h * k).sum(-1, keepdim=True) / d ** 0.5)
        return hidden + (alpha * self.value(mem).float()).to(hidden.dtype)


def find_text_layers(model):
    """Return the decoder ModuleList of the language model (not the vision tower)."""
    cands = [(n, m) for n, m in model.named_modules()
             if isinstance(m, nn.ModuleList) and n.endswith("layers")]
    for n, m in cands:
        if "language_model" in n:
            return n, m
    for n, m in cands:
        if "vision" not in n and "audio" not in n:
            return n, m
    raise RuntimeError("Could not find text decoder layers; print(model) and set the path by hand.")


def attach_engram(model, engram, layer_idx):
    """Nest the engram inside decoder layer `layer_idx` and inject after its forward.

    Living inside the layer means FSDP2 shards, gathers, offloads and checkpoints it
    together with that Gemma4TextDecoderLayer, and auto-resume restores it.
    """
    name, layers = find_text_layers(model)
    target = layers[layer_idx]
    ref = next(target.parameters())
    # FSDP2 requires a single dtype within one wrapped unit; ints (mults, canon) are untouched.
    engram.to(device=ref.device, dtype=ref.dtype)
    target.add_module("micro_engram", engram)
    state = {}

    def grab_ids(_mod, args, kwargs):
        ids = kwargs.get("input_ids")
        if ids is None and args:
            ids = args[0]
        state["input_ids"] = ids
        state["position_ids"] = kwargs.get("position_ids")

    def inject(_mod, _args, output):
        ids = state.get("input_ids")
        if ids is None:
            return output
        pos = state.get("position_ids")
        if isinstance(output, tuple):
            return (engram(output[0], ids, pos),) + tuple(output[1:])
        return engram(output, ids, pos)

    handles = [
        model.register_forward_pre_hook(grab_ids, with_kwargs=True),
        target.register_forward_hook(inject),
    ]
    return f"{name}.{layer_idx}", handles