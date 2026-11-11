"""Data for micro_engram.train: tokenize, cache, pack.

Samples are token lists with a per-token "trained" flag (label = token, or -100):
  completion     same rules as Axolotl's `type: completion`: the text is tokenized (cut at
                 sequence_len * 64 tokens), EOS appended when shorter than that, then split
                 into sequence_len chunks; every token is trained.
  chat_template  the conversation rendered with the chat template (chat_template_jinja +
                 chat_template_kwargs); the content of trained turns (roles_to_train, default
                 assistant) is trained, up to and including its end-of-turn token
                 (eot_tokens), not the role header nor what follows the end-of-turn token.

Tokenized samples are cached under dataset_prepared_path/<hash>/ as flat numpy arrays.
`axolotl_prepared_dataset` reads an Axolotl prepared dataset instead (exact same tokens
and labels as an Axolotl run).

Packing: every epoch, samples are shuffled with seed + epoch and packed first-fit into
bins of sequence_len tokens; position_ids restart at 0 for each document (that is what
keeps attention and the engram's n-grams inside a document).
"""
import hashlib
import json
import logging
import os

import numpy as np

LOG = logging.getLogger("micro_engram.train")

DATA_VERSION = 1  # bump when tokenization rules change (invalidates caches)
MAX_OPEN_BINS = 64  # first-fit looks at this many partially filled bins


# ----------------------------------------------------------------------------- reading
def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"{path}:{n}: not valid JSON ({e})") from e


def _role_map(ds):
    """name used in the file -> our role, from `roles` (our role: [names])."""
    out = {"human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant",
           "model": "assistant", "system": "system", "tool": "tool"}
    for role, names in (ds.roles or {}).items():
        for n in names:
            out[n] = role
    return out


def messages_of(obj, ds):
    """List of {role, content, ...} from one chat sample, with message_property_mappings applied."""
    field = ds.field_messages or "messages"
    msgs = obj.get(field)
    if msgs is None:
        for alt in ("messages", "conversations", "conversation"):
            if isinstance(obj.get(alt), list):
                msgs = obj[alt]
                break
    if not isinstance(msgs, list) or not msgs:
        return None
    mapping = {"role": "role", "content": "content"}
    mapping.update(ds.message_property_mappings or {})
    if ds.message_property_mappings is None:  # ShareGPT files without explicit mappings
        if "from" in msgs[0] and "role" not in msgs[0]:
            mapping.update(role="from", content="value")
    roles = _role_map(ds)
    out = []
    for m in msgs:
        msg = {}
        for ours, theirs in mapping.items():
            if theirs in m and m[theirs] is not None:
                msg[ours] = m[theirs]
        if "role" not in msg or "content" not in msg:
            return None
        msg["role"] = roles.get(msg["role"], msg["role"])
        out.append(msg)
    return out


# ----------------------------------------------------------------------------- tokenizing
class Tokenizing:
    """Turns raw samples into (tokens, trained mask)."""

    def __init__(self, tokenizer, cfg):
        self.tok = tokenizer
        self.cfg = cfg
        self.seq = cfg.sequence_len
        self.kwargs = dict(cfg.chat_template_kwargs or {})
        self.template = load_template(cfg.chat_template_jinja)
        self.eot = list(cfg.eot_tokens or [])

    # completion: Axolotl's CompletionPromptTokenizingStrategy + PromptTokenizingStrategy._tokenize
    def completion(self, text):
        if not text:
            return []
        max_len = self.seq * 64
        ids = self.tok(text, truncation=True, max_length=max_len, padding=False)["input_ids"]
        if not ids:
            return []
        eos = self.tok.eos_token_id
        if ids[-1] != eos and len(ids) < max_len:
            ids = ids + [eos]
        return [(ids[i:i + self.seq], None) for i in range(0, len(ids), self.seq)]  # None: all trained

    def render(self, msgs, add_generation_prompt=False):
        kw = dict(self.kwargs)
        if self.template is not None:
            kw["chat_template"] = self.template
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=add_generation_prompt, **kw)

    def chat(self, msgs, roles_to_train):
        full, spans = chat_spans(self.render, msgs, roles_to_train, self.eot)
        enc = self.tok(full, add_special_tokens=False, return_offsets_mapping=True)
        ids, offs = enc["input_ids"], enc["offset_mapping"]
        mask = np.zeros(len(ids), dtype=bool)
        for (a, b) in spans:
            for j, (s, e) in enumerate(offs):
                if s >= a and e <= b and e > s:
                    mask[j] = True
        return ids, mask


def chat_spans(render, msgs, roles_to_train, eot=()):
    """(the whole rendered conversation, [(start char, end char)] of the trained turns).

    A trained turn's span is its content: from after the role header (what the generation
    prompt adds) up to and including its end-of-turn token (`eot`), not what follows it.
    `render(msgs, add_generation_prompt=False)` renders with the chat template. Shared by
    training (labels) and render_chat.py (eval_logprob.py --assistant-only scoring)."""
    full = render(msgs)
    spans, prev = [], ""
    for i, m in enumerate(msgs):
        cur = render(msgs[:i + 1])
        if not full.startswith(cur):
            raise ValueError("the chat template is not prefix-stable (rendering the first turns is not a prefix "
                             "of the whole conversation), so turns can't be located; check chat_template_kwargs "
                             "(e.g. preserve_thinking) or use axolotl_prepared_dataset")
        if m["role"] in roles_to_train:
            start = len(prev)
            head = render(msgs[:i], add_generation_prompt=True) if i else None
            if head and cur.startswith(head) and len(head) > start:
                start = len(head)  # skip the role header the generation prompt adds
            end = len(cur)
            if eot:
                best = -1
                for t in eot:
                    k = cur.rfind(t, start, end)
                    if k >= 0:
                        best = max(best, k + len(t))
                if best > 0:
                    end = best  # up to and including the end-of-turn token, not what follows
            spans.append((start, end))
        prev = cur
    return full, spans


def load_template(value):
    if not value:
        return None
    if os.path.isfile(value):
        with open(value, encoding="utf-8") as f:
            return f.read()
    if "{%" in value or "{{" in value:
        return value
    raise FileNotFoundError(f"chat_template_jinja: no such file {value}")


def _tok_worker(args):
    lines, kind, cfg_dict, tok_path, ds_dict = args
    from transformers import AutoTokenizer

    from .config import DatasetConfig, TrainConfig

    cfg = TrainConfig(**cfg_dict)
    ds = DatasetConfig(**ds_dict)
    t = Tokenizing(AutoTokenizer.from_pretrained(tok_path), cfg)
    return _tokenize_lines(t, lines, kind, ds)


def _tokenize_lines(t, lines, kind, ds):
    out, stats = [], {"empty": 0, "untrained": 0, "too_long": 0, "bad": 0}
    roles = set(ds.roles_to_train or ["assistant"])
    for obj in lines:
        if kind == "completion":
            text = obj.get(ds.field or "text")
            chunks = t.completion(text) if isinstance(text, str) else []
            if not chunks:
                stats["empty"] += 1
            out.extend(chunks)
        else:
            msgs = messages_of(obj, ds)
            if msgs is None:
                stats["bad"] += 1
                continue
            ids, mask = t.chat(msgs, roles)
            if len(ids) > t.seq:
                stats["too_long"] += 1  # Axolotl's default too: dropped
                continue
            if not mask.any():
                stats["untrained"] += 1
                continue
            out.append((ids, mask))
    return out, stats


# ----------------------------------------------------------------------------- cache
class Samples:
    """Flat token storage: tokens[int32], trained[bool], offsets[int64] (sample i is offsets[i]:offsets[i+1])."""

    def __init__(self, tokens, trained, offsets):
        self.tokens, self.trained, self.offsets = tokens, trained, offsets

    def __len__(self):
        return len(self.offsets) - 1

    def lengths(self):
        return np.diff(self.offsets)

    def get(self, i):
        a, b = self.offsets[i], self.offsets[i + 1]
        return self.tokens[a:b], self.trained[a:b]

    def save(self, folder):
        os.makedirs(folder, exist_ok=True)
        tmp = os.path.join(folder, "samples.tmp.npz")
        np.savez(tmp, tokens=self.tokens, trained=self.trained, offsets=self.offsets)
        os.replace(tmp, os.path.join(folder, "samples.npz"))

    @classmethod
    def load(cls, folder):
        z = np.load(os.path.join(folder, "samples.npz"))
        return cls(z["tokens"], z["trained"], z["offsets"])

    @classmethod
    def from_list(cls, items):
        lens = [len(ids) for ids, _ in items]
        offsets = np.zeros(len(items) + 1, dtype=np.int64)
        np.cumsum(lens, out=offsets[1:])
        tokens = np.empty(offsets[-1], dtype=np.int32)
        trained = np.empty(offsets[-1], dtype=bool)
        for k, (ids, mask) in enumerate(items):
            a, b = offsets[k], offsets[k + 1]
            tokens[a:b] = ids
            trained[a:b] = True if mask is None else mask
        return cls(tokens, trained, offsets)


def _file_sig(path):
    st = os.stat(path)
    return [os.path.abspath(path), st.st_size, int(st.st_mtime)]


def data_hash(cfg, tok_path):
    h = {
        "v": DATA_VERSION,
        "datasets": [{**d.model_dump(), "_file": _file_sig(d.path)} for d in cfg.datasets],
        "tokenizer": os.path.abspath(tok_path),
        "seq": cfg.sequence_len,
        "template": load_template(cfg.chat_template_jinja),
        "kwargs": cfg.chat_template_kwargs,
        "eot": cfg.eot_tokens,
    }
    return hashlib.sha256(json.dumps(h, sort_keys=True, default=str).encode()).hexdigest()[:16]


def tokenize_all(cfg, tokenizer, tok_path):
    items, total = [], {}
    nproc = max(1, int(cfg.dataset_num_proc or 1))
    for ds in cfg.datasets:
        rows = list(read_jsonl(ds.path))
        if nproc > 1 and len(rows) > 64:
            from multiprocessing import get_context

            chunk = -(-len(rows) // (nproc * 4))
            jobs = [(rows[i:i + chunk], ds.type, cfg.model_dump(), tok_path, ds.model_dump())
                    for i in range(0, len(rows), chunk)]
            with get_context("spawn").Pool(nproc) as pool:
                results = pool.map(_tok_worker, jobs)
        else:
            results = [_tokenize_lines(Tokenizing(tokenizer, cfg), rows, ds.type, ds)]
        n0 = len(items)
        for got, stats in results:
            items.extend(got)
            for k, v in stats.items():
                total[k] = total.get(k, 0) + v
        LOG.warning(f"[micro_engram] {ds.path}: {len(rows):,} rows -> {len(items) - n0:,} samples ({ds.type})")
    dropped = {k: v for k, v in total.items() if v}
    if dropped:
        LOG.warning(f"[micro_engram] dropped rows: {dropped}")
    if not items:
        raise ValueError("no training samples")
    return Samples.from_list(items)


def read_axolotl_prepared(path):
    """input_ids/labels of an Axolotl prepared dataset (datasets.load_from_disk folder)."""
    from datasets import load_from_disk

    ds = load_from_disk(path)
    if hasattr(ds, "keys") and "train" in ds:
        ds = ds["train"]
    items = []
    for row in ds.select_columns(["input_ids", "labels"]):
        ids = row["input_ids"]
        items.append((ids, np.asarray(row["labels"]) != -100))
    LOG.warning(f"[micro_engram] read {len(items):,} samples from the Axolotl prepared dataset {path}")
    return Samples.from_list(items)


def prepare(cfg, tokenizer, tok_path, rank=0, barrier=None):
    """Samples for this run: from the cache, else tokenized on rank 0 and cached."""
    if cfg.axolotl_prepared_dataset:
        return read_axolotl_prepared(cfg.axolotl_prepared_dataset)
    folder = None
    if cfg.dataset_prepared_path:
        folder = os.path.join(cfg.dataset_prepared_path, "micro_engram-" + data_hash(cfg, tok_path))
    if folder and os.path.isfile(os.path.join(folder, "samples.npz")):
        LOG.warning(f"[micro_engram] tokenized data from cache {folder}")
        return Samples.load(folder)
    if folder is None:
        return tokenize_all(cfg, tokenizer, tok_path)  # every rank (no shared cache)
    if rank == 0:
        s = tokenize_all(cfg, tokenizer, tok_path)
        s.save(folder)
        LOG.warning(f"[micro_engram] tokenized data cached in {folder}")
    if barrier is not None:
        barrier()
    return Samples.load(folder)


# ----------------------------------------------------------------------------- packing
def pack(lengths, capacity, seed, shuffle=True):
    """Bins (lists of sample indices) of at most `capacity` tokens: samples in a shuffled order
    (seed), first-fit over the last MAX_OPEN_BINS open bins. Samples longer than capacity are
    skipped (the data step already keeps them out)."""
    order = np.arange(len(lengths))
    if shuffle:
        np.random.default_rng(seed).shuffle(order)
    bins, room, open_ = [], [], []
    for i in order:
        n = int(lengths[i])
        if n > capacity or n == 0:
            continue
        for b in open_:
            if room[b] >= n:
                bins[b].append(int(i))
                room[b] -= n
                if room[b] == 0:
                    open_.remove(b)
                break
        else:
            bins.append([int(i)])
            room.append(capacity - n)
            if room[-1] > 0:
                open_.append(len(bins) - 1)
                if len(open_) > MAX_OPEN_BINS:
                    open_.pop(0)
    if shuffle:  # bins in random order too (else the long-sample bins come in clumps)
        perm = np.random.default_rng(seed + 1).permutation(len(bins))
        bins = [bins[k] for k in perm]
    return bins


def collate(samples, bins, seq_len, pad_id):
    """Micro-batch [B, seq_len]: input_ids, labels (-100 untrained/pad), position_ids
    restarting per document. Padding continues the last document's positions: it sits at the
    end, is untrained, and causal attention never lets a real token see it, so it needs no
    isolation, and a one-document row with padding still counts as one document (so attention
    can take the plain causal path instead of a block mask)."""
    import torch

    B = len(bins)
    ids = torch.full((B, seq_len), pad_id, dtype=torch.long)
    labels = torch.full((B, seq_len), -100, dtype=torch.long)
    pos = torch.zeros((B, seq_len), dtype=torch.long)
    for r, b in enumerate(bins):
        t, last = 0, 0
        for i in b:
            tok, tr = samples.get(i)
            n = len(tok)
            last = n
            tt = torch.from_numpy(tok.astype(np.int64))
            ids[r, t:t + n] = tt
            lab = tt.clone()
            lab[~torch.from_numpy(tr)] = -100
            labels[r, t:t + n] = lab
            pos[r, t:t + n] = torch.arange(n)
            t += n
        if t < seq_len:
            pos[r, t:] = torch.arange(last, last + seq_len - t)
    # labels are next-token targets: shift left by one inside each row; the token after a
    # document's last token belongs to the next document, so it is never a target
    shifted = torch.full_like(labels, -100)
    shifted[:, :-1] = labels[:, 1:]
    shifted[:, :-1][pos[:, 1:] == 0] = -100
    return {"input_ids": ids, "labels": shifted, "position_ids": pos}


class EpochPlan:
    """Which bins every rank gets in every step of an epoch."""

    def __init__(self, samples, cfg, world):
        self.samples, self.cfg, self.world = samples, cfg, world
        self.per_step = world * cfg.micro_batch_size * cfg.gradient_accumulation_steps
        self._cache = {}

    def bins(self, epoch):
        if epoch not in self._cache:
            self._cache = {epoch: pack(self.samples.lengths(), self.cfg.sequence_len, self.cfg.seed + epoch,
                                       shuffle=self.cfg.shuffle_each_epoch or epoch == 0)}
        return self._cache[epoch]

    def steps_per_epoch(self, epoch=0):
        return len(self.bins(epoch)) // self.per_step

    def micro_batches(self, epoch, step, rank):
        """The gradient_accumulation_steps micro-batches (lists of bins) of `rank` for `step`."""
        b = self.bins(epoch)
        base = step * self.per_step
        mbs, ga = self.cfg.micro_batch_size, self.cfg.gradient_accumulation_steps
        out = []
        for a in range(ga):
            k = base + (a * self.world + rank) * mbs
            out.append(b[k:k + mbs])
        return out
