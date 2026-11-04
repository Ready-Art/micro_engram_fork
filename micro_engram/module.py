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

    Gate modes:
      "scalar" (paper): one sigmoid per token, from rms(h) . rms(W_K e); injects alpha * W_V e.
      "scalar_sqrt" (DeepSeek's released Engram code): like "scalar", but the score s goes
            through a signed square root, sigmoid(sign(s) * sqrt(|s|)), and both RMSNorms
            have learnable scales (q_scale, k_scale). The score is about cos * sqrt(d), so with
            Gemma's d = 5376 the plain sigmoid saturates already at cos ~ 0.1; the square root
            keeps the gate responsive, and the scales let it learn its own sharpness.
      "ple" (Gemma 3n per-layer-embedding style): a per-dimension gate from the hidden
            state, g = gelu(W_down rms(h)), injects W_up (g * e). Each head's slice of e
            gets its own slice of the gate.
    """

    GATES = ("scalar", "ple", "scalar_sqrt")
    SCALAR_GATES = ("scalar", "scalar_sqrt")  # both have key/value

    def __init__(self, model_dim, table_size=65_536, engram_dim=128,
                 num_heads=4, orders=(2, 3), pad_id=0, seed=0,
                 vocab_size=262_144, canon=None, gate="scalar"):
        super().__init__()
        if gate not in self.GATES:
            raise ValueError(f"gate must be one of {self.GATES}, got {gate!r}")
        self.gate = gate
        self.orders = tuple(orders)
        self.table_size = table_size
        self.pad_id = pad_id
        self._table_orders = [n for n in self.orders for _ in range(num_heads)]
        n_tables = len(self._table_orders)

        self.tables = nn.ModuleList(nn.Embedding(table_size, engram_dim) for _ in range(n_tables))
        for t in self.tables:
            nn.init.normal_(t.weight, std=0.02)

        mem_dim = engram_dim * n_tables
        if gate in self.SCALAR_GATES:
            self.key = nn.Linear(mem_dim, model_dim, bias=False)    # used only by the gate
            self.value = nn.Linear(mem_dim, model_dim, bias=False)  # what gets injected
            nn.init.zeros_(self.value.weight)  # step 0: exact no-op
            if gate == "scalar_sqrt":
                # learnable RMSNorm scales for the query (hidden state) and the key
                self.q_scale = nn.Parameter(torch.ones(model_dim))
                self.k_scale = nn.Parameter(torch.ones(model_dim))
        else:
            self.down = nn.Linear(model_dim, mem_dim, bias=False)   # hidden -> per-dim gate
            self.up = nn.Linear(mem_dim, model_dim, bias=False)     # gated memory -> hidden
            nn.init.zeros_(self.up.weight)  # step 0: exact no-op

        # Deterministic from `seed`, so it can be rebuilt if a checkpoint drops buffers.
        g = torch.Generator().manual_seed(seed)
        # device="cpu": the generator is a CPU one, and servers (vLLM) build models with
        # CUDA as the default device. Same values either way; .to() moves it later.
        mults = torch.randint(1, _PRIME, (n_tables, max(self.orders)), generator=g,
                              device="cpu") | 1
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

    def hash_prev(self, ids):
        """Same hashes as `hash_ids`, from a flat list of tokens with their predecessors.

        `ids` is [N, max order]: column 0 holds the token at t, column i the token at
        t - i, and -1 where there is none (before the start of the document). Used by
        servers (vLLM) that see flattened batches instead of whole sequences.
        """
        ids = ids.long()
        canon = self.canon.to(ids.device)
        merged = canon[ids.clamp(0, canon.numel() - 1)]  # token merging
        # missing tokens get the raw pad id, exactly like the padding in hash_ids
        merged = torch.where(ids >= 0, merged, torch.full_like(merged, self.pad_id))
        mults = self.mults.to(ids.device)
        out = []
        for k, n in enumerate(self._table_orders):
            x = merged[:, 0] * mults[k, 0]
            for i in range(1, n):
                x = x + merged[:, i] * mults[k, i]
            out.append((x % _PRIME) % self.table_size)
        return out

    def inject(self, hidden, idx):
        """Look up the hashed rows `idx` and add the gated memory to `hidden`."""
        mem = torch.cat([tab(i) for tab, i in zip(self.tables, idx)], dim=-1)
        d = hidden.shape[-1]
        h = F.rms_norm(hidden.float(), (d,))
        if self.gate in self.SCALAR_GATES:
            if self.gate == "scalar_sqrt":
                h = F.rms_norm(hidden.float(), (d,), weight=self.q_scale.float())
                k = F.rms_norm(self.key(mem).float(), (d,), weight=self.k_scale.float())
            else:
                k = F.rms_norm(self.key(mem).float(), (d,))
            s = (h * k).sum(-1, keepdim=True) / d ** 0.5
            if self.gate == "scalar_sqrt":
                s = s.sign() * s.abs().clamp_min(1e-6).sqrt()  # as in DeepSeek's Engram code
            alpha = torch.sigmoid(s)
            if self.training:
                self._record_gate(alpha)
            out = alpha * self.value(mem).float()
        else:
            # Normalising h keeps the gate well-scaled (Gemma's residual stream is large).
            g = F.gelu(self.down(h.to(self.down.weight.dtype)), approximate="tanh")
            out = self.up(g * mem).float()
        return hidden + out.to(hidden.dtype)

    def forward(self, hidden, input_ids, position_ids=None):
        idx = self.hash_ids(input_ids.to(hidden.device), position_ids)
        return self.inject(hidden, idx)

    # --- gate statistics for training logs (scalar gates only; no GPU sync until read) ---
    def _record_gate(self, alpha):
        a = alpha.detach().float()
        stats = getattr(self, "_gate_stats", None)
        if stats is None or stats.device != a.device:
            stats = torch.zeros(3, device=a.device)  # sum of alpha, count, count of alpha > 0.99
        self._gate_stats = stats + torch.stack([a.sum(), torch.tensor(float(a.numel()), device=a.device),
                                                (a > 0.99).float().sum()])

    def pop_gate_stats(self):
        """(mean gate, fraction of tokens with gate > 0.99) since the last call, or None."""
        stats = getattr(self, "_gate_stats", None)
        self._gate_stats = None
        if stats is None or stats[1] == 0:
            return None
        s, n, sat = stats.tolist()
        return s / n, sat / n


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