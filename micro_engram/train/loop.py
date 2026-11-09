"""micro_engram.train: the training loop (pure PyTorch + FSDP2, no HF Trainer/Accelerate).

What runs where, per GPU:
  base model      frozen; FSDP2 per decoder layer (bf16, or NF4 with bitsandbytes), optional
                  CPU offload of its parameters
  embedding/head  vocab rows split over the GPUs (head.VocabShard), outside FSDP
  engram tables   row-sharded (rows.RowTables): own optimizer and clipping, as before
  engram dense    key/value/gate/scales replicated on every GPU, outside FSDP; gradients
                  all-reduced; AdamW (LazyAdamW non-lazy, stochastic rounding for bf16)
"""
import json
import logging
import math
import os
import shutil
import time

import torch
import torch.distributed as dist

from ..rows import ROWS_DIR, ROWS_OPTIM_DIR, RowTables, dist_rank_world
from ..runconfig import (CONFIG_NAME, _params, _safe_fingerprint, build_config, build_slot, check_canon,
                         engram_requests, epoch_step_scheduler, write_config)
from ..stats import module_stats
from . import model as M
from .data import EpochPlan, collate, prepare
from .offload import ActivationOffload

LOG = logging.getLogger("micro_engram.train")
STATE_FILE = "trainer_state.json"     # written last: marks a checkpoint complete
DENSE_FILE = "micro_engram_dense_optim.pt"


def _log0(msg):
    if dist_rank_world()[0] == 0:
        LOG.warning(msg)


def setup_dist():
    rank, world = dist_rank_world()
    local = int(os.environ.get("LOCAL_RANK", 0))
    cuda = torch.cuda.is_available() and os.environ.get("MICRO_ENGRAM_CPU") != "1"
    if cuda:
        torch.cuda.set_device(local)
        device = torch.device("cuda", local)
    else:
        device = torch.device("cpu")
    if not dist.is_initialized():
        dist.init_process_group("nccl" if cuda else "gloo", rank=rank, world_size=world,
                                **({"device_id": device} if cuda else {}))
    return rank, world, local, device


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.rank, self.world, self.local, self.device = setup_dist()
        self.ctx = {"ids": None, "pos": None}
        self.global_step, self.epoch, self.step_in_epoch = 0, 0, 0
        self.wandb = None
        if cfg.tf32 and self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    # ------------------------------------------------------------------ setup
    def build(self):
        cfg = self.cfg
        torch.manual_seed(cfg.seed)
        from transformers import AutoTokenizer

        tok_path = cfg.tokenizer_config or cfg.base_model
        self.tokenizer = AutoTokenizer.from_pretrained(tok_path)
        self.samples = prepare(cfg, self.tokenizer, tok_path, rank=self.rank, barrier=dist.barrier)
        self.plan = EpochPlan(self.samples, cfg, self.world)
        self.pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

        M.apply_kernel_patches(cfg, self.device.type == "cuda")
        model, meta = M.load_base(cfg, self.rank, self.local, self.device)
        self.model = model
        self.text_name, self.text = M.text_model(model)
        M.hybrid_attention(self.text, cfg)
        text_cfg = self.text.config

        self.vocab = M.take_vocab(model, self.text, self.rank, self.world, self.device,
                                  broadcast=bool(cfg.fsdp_config.cpu_ram_efficient_loading and not cfg.load_in_4bit),
                                  chunk_tokens=cfg.loss_chunk_tokens)

        # engrams
        canon = None
        if cfg.micro_engram_canon_path:
            canon = check_canon(torch.load(cfg.micro_engram_canon_path, map_location="cpu"), text_cfg.vocab_size)
            _log0(f"[micro_engram] token merging ON: {canon.numel():,} ids -> {int(torch.unique(canon).numel()):,} "
                  f"classes ({cfg.micro_engram_canon_path})")
        self.slots = []
        for spec, init_from, frozen in engram_requests(cfg):
            spec["canon_path"] = cfg.micro_engram_canon_path
            self.slots.append(build_slot(spec, init_from, frozen, model, text_cfg, canon, dtype=cfg.micro_engram_dtype,
                                         sharding="rows", get_ids=lambda: (self.ctx["ids"], self.ctx["pos"])))
        for s in self.slots:
            if s.frozen:
                for p in _params(s.engram):
                    p.requires_grad_(False)
        self._place_engrams(meta)
        self.dense = [p for s in self.slots for p in s.engram.parameters() if p.requires_grad]
        trained_layers = [s.layer for s in self.slots if not s.frozen]
        self.first_trained = min(trained_layers) if trained_layers else None
        if self.first_trained is None:
            raise ValueError("every engram is frozen: nothing to train")
        self.base_conf = {
            "model_dim": text_cfg.hidden_size, "vocab_size": text_cfg.vocab_size,
            "canon_path": cfg.micro_engram_canon_path, "base_model": cfg.base_model,
            "base_fingerprint": _safe_fingerprint(cfg.base_model) if self.rank == 0 else None,
        }

        no_grad_upto = self.first_trained if cfg.no_grad_prefix else None
        M.checkpointing(model, self.text, cfg, no_grad_upto)
        model.train()

        from torch.distributed.device_mesh import init_device_mesh

        mesh = init_device_mesh(self.device.type, (self.world,))
        efficient = meta or (cfg.fsdp_config.cpu_ram_efficient_loading and not cfg.load_in_4bit)
        full = None
        if efficient and self.rank == 0:  # rank 0's weights, plain tensors, before sharding
            full = {k: v for k, v in self.text.state_dict().items() if ".micro_engram." not in k}
        M.shard_model(self.text, cfg, mesh, [p for s in self.slots for p in s.engram.parameters()])
        if efficient:
            self._load_from_rank0(full)
            del full

        # optimizers
        from ..optim import LazyAdamW

        a = cfg
        self.dense_opt = LazyAdamW([{"params": self.dense, "lazy": False}], lr=a.learning_rate,
                                   betas=(a.adam_beta1, a.adam_beta2), eps=a.adam_epsilon,
                                   weight_decay=a.weight_decay,
                                   stochastic_rounding=a.micro_engram_stochastic_rounding is not False)
        opt = cfg.micro_engram_optimizer
        self.rows = RowTables([(f"{s.where}.micro_engram.", s.engram) for s in self.slots],
                              lr=a.learning_rate, betas=(a.adam_beta1, a.adam_beta2), eps=a.adam_epsilon,
                              weight_decay=a.weight_decay, lazy=opt in ("lazy_adam", "rowwise_adam"),
                              rowwise=opt == "rowwise_adam",
                              stochastic_rounding=a.micro_engram_stochastic_rounding is not False)
        if cfg.optimizer and not str(cfg.optimizer).startswith("adamw"):
            _log0(f"[micro_engram] optimizer: {cfg.optimizer} is not used; key/value/scales use AdamW")

        self.total_steps = self._count_steps()
        self.scheduler = self._scheduler()
        self.offload = ActivationOffload(self.device) if cfg.activation_offloading else None
        self._report()

    def _place_engrams(self, meta):
        """Engram dense params and buffers onto this GPU, identical on every rank (rank 0's)."""
        for s in self.slots:
            e = s.engram
            if any(t.is_meta for t in list(e.parameters()) + list(e.buffers())):
                e.to_empty(device=self.device)
            else:
                e.to(self.device)
            if self.world > 1:
                for t in list(e.parameters()) + list(e.buffers()):
                    dist.broadcast(t.data, src=0)

    def _load_from_rank0(self, full):
        """cpu_ram_efficient_loading: rank 0 loaded the weights (`full`, taken before sharding),
        the others built the model on meta. Broadcast into the sharded model."""
        from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict

        for mod in self.text.modules():  # materialize meta shards before receiving
            if any(p.is_meta for p in mod.parameters(recurse=False)) or \
                    any(b.is_meta for b in mod.buffers(recurse=False)):
                mod.to_empty(device=self.device, recurse=False)
        set_model_state_dict(self.text, full or {}, options=StateDictOptions(
            full_state_dict=True, broadcast_from_rank0=True, strict=False))
        for name, b in self.text.named_buffers():  # non-persistent buffers (RoPE...) are not in state dicts
            if ".micro_engram." not in name:
                dist.broadcast(b.data, src=0)

    def _count_steps(self):
        cfg = self.cfg
        if cfg.max_steps:
            return int(cfg.max_steps)
        full, frac = int(math.floor(cfg.num_epochs)), cfg.num_epochs - math.floor(cfg.num_epochs)
        n = sum(self.plan.steps_per_epoch(e) for e in range(full))
        if frac > 0:
            n += int(self.plan.steps_per_epoch(full) * frac)
        if n <= 0:
            raise ValueError(f"no optimizer steps: {len(self.samples):,} samples pack into "
                             f"{len(self.plan.bins(0)):,} bins, fewer than one step needs ({self.plan.per_step})")
        return n

    def _scheduler(self):
        cfg = self.cfg
        warm = cfg.warmup_steps if cfg.warmup_steps is not None else (
            math.ceil(cfg.warmup_ratio * self.total_steps) if cfg.warmup_ratio else 0)
        self.warmup = warm
        if cfg.micro_engram_lr_epoch_decay:
            return epoch_step_scheduler(self.dense_opt, self.total_steps, warm, float(cfg.num_epochs),
                                        cfg.micro_engram_lr_epoch_decay)
        from transformers import get_scheduler

        name, kw = cfg.lr_scheduler, dict(cfg.lr_scheduler_kwargs or {})
        if name == "cosine" and cfg.cosine_min_lr_ratio is not None:
            name, kw = "cosine_with_min_lr", {**kw, "min_lr_rate": cfg.cosine_min_lr_ratio}
        if name == "constant":
            warm = 0
        return get_scheduler(name, self.dense_opt, num_warmup_steps=warm, num_training_steps=self.total_steps,
                             scheduler_specific_kwargs=kw or None)

    def _report(self):
        n_dense = sum(p.numel() for p in self.dense)
        n_rows = sum(p.numel() for p in self.rows.params() if p.requires_grad)
        n_base_train = sum(p.numel() for p in self.text.parameters() if p.requires_grad) - n_dense
        n_4bit = sum(1 for m in self.text.modules() if type(m).__name__ == "Linear4bit")
        parts = []
        for s in self.slots:
            parts.append(f"layer {s.layer}{' (frozen)' if s.frozen else ''}: "
                         f"{sum(p.numel() for p in s.engram.row_shards) / 1e6:.1f}M table params on this GPU, "
                         f"{sum(p.numel() for p in s.engram.parameters()) / 1e6:.1f}M dense")
        _log0(f"[micro_engram] trainable: engram only ({n_rows / 1e6:.1f}M table params on this GPU + "
              f"{n_dense / 1e6:.1f}M dense); base: frozen{', ' + str(n_4bit) + ' NF4 layers' if n_4bit else ''}; "
              + "; ".join(parts))
        if n_base_train:
            raise RuntimeError(f"{n_base_train} base parameters are trainable")
        _log0(f"[micro_engram] {len(self.samples):,} samples, {len(self.plan.bins(0)):,} bins of "
              f"{self.cfg.sequence_len} tokens, {self.plan.per_step} bins per step, {self.total_steps} steps; "
              f"no-grad prefix through layer {self.first_trained if self.cfg.no_grad_prefix else '-'}; "
              f"vocab rows {self.vocab.rows:,} of {self.vocab.V:,} on this GPU (loss chunk {self.vocab.chunk} tokens)")
        if self.device.type == "cuda":
            _log0(f"[micro_engram] after setup: {torch.cuda.memory_allocated(self.device) / 2**30:.2f} GiB allocated "
                  f"on GPU {self.local}")

    # ------------------------------------------------------------------ one step
    def _micro_batch(self, bins):
        b = collate(self.samples, bins, self.cfg.sequence_len, self.pad_id)
        return {k: v.to(self.device, non_blocking=True) for k, v in b.items()}

    def forward_loss(self, batch, scale):
        """Sum of this GPU's token losses for one micro-batch; backward of loss * scale."""
        ids, pos, labels = batch["input_ids"], batch["position_ids"], batch["labels"]
        self.ctx["ids"], self.ctx["pos"] = ids, pos
        emb = self.vocab.embed(ids)
        kw = {}
        if getattr(self.text, "hidden_size_per_layer_input", 0):
            with torch.no_grad():
                kw["per_layer_inputs"] = self.text.get_per_layer_inputs(ids)
        off = self.offload.context() if self.offload else _null()
        with off, M.grad_from_layer(self.text, self.first_trained, self.cfg.no_grad_prefix):
            out = self.text(inputs_embeds=emb, position_ids=pos, attention_mask=None, use_cache=False, **kw)
        h = out.last_hidden_state.reshape(-1, out.last_hidden_state.shape[-1])
        total, _ = self.vocab.loss_sum(h, labels.reshape(-1))
        (total * scale).backward()
        if self.offload:
            self.offload.end_step()
        self.ctx["ids"] = self.ctx["pos"] = None
        return total.detach()

    def train_step(self, micro):
        """One optimizer step over the gradient_accumulation_steps micro-batches."""
        batches = [self._micro_batch(b) for b in micro]
        count = torch.stack([(b["labels"] != -100).sum() for b in batches]).sum().float()
        if self.world > 1:
            dist.all_reduce(count)
        count = count.clamp_min(1)
        # the row tables and the dense all-reduce average over GPUs: x world keeps the
        # gradient that of the per-token mean over the whole step
        scale = self.world / count
        loss = torch.zeros((), device=self.device)
        for b in batches:
            loss += self.forward_loss(b, scale)
        if self.world > 1:
            dist.all_reduce(loss)
            self._reduce_dense()
        dense_norm = torch.nn.utils.clip_grad_norm_(self.dense, self.cfg.max_grad_norm or float("inf"))
        table_norm = self.rows.clip(self.cfg.max_grad_norm)
        lr = self.dense_opt.param_groups[0]["lr"]
        self.dense_opt.step()
        self.rows.step(lr)
        self.dense_opt.zero_grad(set_to_none=True)
        self.scheduler.step()
        mean = float(loss / count)
        return {"loss": mean, "ppl": math.exp(min(mean, 50.0)), "grad_norm": float(dense_norm),
                "table_grad_norm": table_norm,
                "learning_rate": lr, "tokens": int(count)}

    def _reduce_dense(self):
        grads = [p.grad for p in self.dense if p.grad is not None]
        if not grads:
            return
        flat = torch.cat([g.reshape(-1).float() for g in grads])
        dist.all_reduce(flat)
        flat /= self.world
        k = 0
        for g in grads:
            n = g.numel()
            g.copy_(flat[k:k + n].view_as(g))
            k += n

    # ------------------------------------------------------------------ loop
    def train(self):
        cfg = self.cfg
        self._maybe_resume()
        if self.rank == 0:
            os.makedirs(cfg.output_dir, exist_ok=True)
            write_config(build_config(self.base_conf, self.slots, lambda s: None), cfg.output_dir)
            self._init_wandb()
        t_last, toks = time.time(), 0
        self.pbar = None
        if self.rank == 0:
            from tqdm import tqdm

            self.pbar = tqdm(total=self.total_steps, initial=self.global_step, desc="train", dynamic_ncols=True,
                             unit="step")
        while self.global_step < self.total_steps:
            n_ep = self.plan.steps_per_epoch(self.epoch)
            if self.step_in_epoch >= n_ep:
                self.epoch, self.step_in_epoch = self.epoch + 1, 0
                continue
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)
            micro = self.plan.micro_batches(self.epoch, self.step_in_epoch, self.rank)
            logs = self.train_step(micro)
            self.global_step += 1
            self.step_in_epoch += 1
            toks += logs.pop("tokens")
            if self.pbar is not None:
                self.pbar.update(1)
                self.pbar.set_postfix(loss=f"{logs['loss']:.4f}", ppl=f"{logs['ppl']:.1f}",
                                      lr=f"{logs['learning_rate']:.2e}", epoch=self.epoch, refresh=True)
            if self.global_step % cfg.logging_steps == 0 or self.global_step == self.total_steps:
                dt = time.time() - t_last
                logs.update(self._memory())
                logs.update(self._engram_logs())
                logs.update({"epoch": round(self.epoch + self.step_in_epoch / max(1, n_ep), 4),
                             "step": self.global_step, "s_per_step": round(dt / cfg.logging_steps, 2),
                             "trained_tokens_per_s": round(toks / max(dt, 1e-9), 1)})
                self._log(logs)
                t_last, toks = time.time(), 0
            if cfg.save_steps and self.global_step % cfg.save_steps == 0:
                self.save_checkpoint()
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None
        if not cfg.save_steps or self.global_step % cfg.save_steps:
            self.save_checkpoint()
        self.rows.save(os.path.join(cfg.output_dir, ROWS_DIR))
        if self.rank == 0:
            write_config(build_config(self.base_conf, self.slots, lambda s: ROWS_DIR), cfg.output_dir)
            _log0(f"[micro_engram] done: {cfg.output_dir}/{ROWS_DIR} holds the engram; serve it with "
                  f"--additional-config '{{\"micro_engram\": \"{cfg.output_dir}/{CONFIG_NAME}\"}}'")
            if self.wandb:
                self.wandb.finish()

    def _memory(self):
        if self.device.type != "cuda":
            return {}
        m = torch.tensor([torch.cuda.max_memory_allocated(self.device), torch.cuda.max_memory_reserved(self.device)],
                         dtype=torch.float64, device=self.device) / 2**30
        allm = [torch.zeros_like(m) for _ in range(self.world)]
        dist.all_gather(allm, m)
        allm = torch.stack(allm)
        return {"mem_peak_alloc_gib": round(float(allm[:, 0].max()), 2),
                "mem_peak_reserved_gib": round(float(allm[:, 1].max()), 2),
                "mem_peak_alloc_per_gpu": [round(float(x), 2) for x in allm[:, 0]]}

    def _engram_logs(self):
        out = {}
        stats_every = self.cfg.micro_engram_stats_steps
        gates = [s.engram.pop_gate_stats() for s in self.slots]
        if self.rank != 0 or not stats_every or self.global_step % stats_every:
            return out
        for s, gate in zip(self.slots, gates):
            if s.frozen:
                continue
            try:
                st = module_stats(s.engram)
                if gate is not None:
                    st["engram/gate_mean"], st["engram/gate_saturated"] = gate
            except Exception as e:  # noqa: BLE001  -- stats must not stop a run
                LOG.warning(f"[micro_engram] stats failed: {e}")
                continue
            if len(self.slots) > 1:
                st = {k.replace("engram/", f"engram_L{s.layer}/", 1): v for k, v in st.items()}
            out.update(st)
        return out

    def _log(self, logs):
        if self.rank != 0:
            return
        line = json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in logs.items()})
        if getattr(self, "pbar", None) is not None:
            self.pbar.write(line)  # above the progress bar
        else:
            print(line, flush=True)
        with open(os.path.join(self.cfg.output_dir, "train_log.jsonl"), "a") as f:
            f.write(line + "\n")
        if self.wandb:
            self.wandb.log({k: v for k, v in logs.items() if isinstance(v, (int, float))}, step=self.global_step)

    def _init_wandb(self):
        cfg = self.cfg
        if not cfg.wandb_project or cfg.wandb_mode == "disabled":
            return
        import wandb

        self.wandb = wandb
        wandb.init(project=cfg.wandb_project, name=cfg.wandb_name, entity=cfg.wandb_entity, mode=cfg.wandb_mode,
                   config=cfg.model_dump(), resume="allow")

    # ------------------------------------------------------------------ checkpoints
    def save_checkpoint(self):
        cfg = self.cfg
        folder = os.path.join(cfg.output_dir, f"checkpoint-{self.global_step}")
        self.rows.save(os.path.join(folder, ROWS_DIR), os.path.join(folder, ROWS_OPTIM_DIR))
        if self.rank == 0:
            torch.save({"dense_optimizer": self.dense_opt.state_dict(), "scheduler": self.scheduler.state_dict(),
                        "rng": torch.get_rng_state()}, os.path.join(folder, DENSE_FILE))
            write_config(build_config(self.base_conf, self.slots, lambda s: ROWS_DIR), folder)
            write_config(build_config(self.base_conf, self.slots,
                                      lambda s: f"checkpoint-{self.global_step}/{ROWS_DIR}"), cfg.output_dir)
            with open(os.path.join(folder, STATE_FILE), "w") as f:
                json.dump({"global_step": self.global_step, "epoch": self.epoch, "step_in_epoch": self.step_in_epoch,
                           "total_steps": self.total_steps, "trainer": "micro_engram.train"}, f, indent=2)
            self._rotate()
        dist.barrier()
        _log0(f"[micro_engram] saved {folder}")

    def _rotate(self):
        limit = self.cfg.save_total_limit
        if not limit:
            return
        import glob
        import re

        done = sorted((int(m.group(1)), c) for c in glob.glob(os.path.join(self.cfg.output_dir, "checkpoint-*"))
                      if (m := re.search(r"checkpoint-(\d+)$", c)) and os.path.isfile(os.path.join(c, STATE_FILE)))
        for _, c in done[:-limit]:
            shutil.rmtree(c, ignore_errors=True)

    def _maybe_resume(self):
        from ..weights import latest_checkpoint

        cfg = self.cfg
        ckpt = cfg.resume_from_checkpoint
        if ckpt is None and cfg.auto_resume_from_checkpoints and os.path.isdir(cfg.output_dir):
            ckpt = latest_checkpoint(cfg.output_dir)
        if not ckpt:
            return
        with open(os.path.join(ckpt, STATE_FILE)) as f:
            st = json.load(f)
        if st.get("trainer") != "micro_engram.train":
            raise ValueError(f"{ckpt} was written by another trainer (Axolotl?): start from it with "
                             f"micro_engram_init_from instead of resuming")
        n, opt = self.rows.load(os.path.join(ckpt, ROWS_DIR), os.path.join(ckpt, ROWS_OPTIM_DIR))
        self._load_dense(os.path.join(ckpt, ROWS_DIR))
        blob = torch.load(os.path.join(ckpt, DENSE_FILE), map_location="cpu", weights_only=False)
        self.dense_opt.load_state_dict(blob["dense_optimizer"])
        self.scheduler.load_state_dict(blob["scheduler"])
        self.global_step, self.epoch, self.step_in_epoch = st["global_step"], st["epoch"], st["step_in_epoch"]
        if st.get("total_steps") not in (None, self.total_steps):
            _log0(f"[micro_engram] resume: the run now has {self.total_steps} steps (checkpoint said "
                  f"{st['total_steps']}); the LR schedule follows the new count")
        _log0(f"[micro_engram] resumed from {ckpt}: step {self.global_step}, epoch {self.epoch} "
              f"(+{self.step_in_epoch} steps), {n} tables" + ("" if opt else ", NO table optimizer state"))

    def _load_dense(self, folder):
        """Key/value/gate/scales of the trained engrams from a micro_engram_rows checkpoint."""
        from ..weights import load_engram_weights

        for s in self.slots:
            if s.frozen:
                continue
            sd = {k: v for k, v in load_engram_weights(folder, layer=s.layer).items() if not k.startswith("tables.")}
            tables_missing = {f"tables.{k}.weight" for k in range(len(s.engram.row_shards))}
            res = s.engram.load_state_dict(sd, strict=False)
            bad = [k for k in res.missing_keys if k not in tables_missing and k != "mults"] + list(res.unexpected_keys)
            if bad:
                raise ValueError(f"[micro_engram] resume: engram at layer {s.layer} does not match {folder}: {bad[:10]}")


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def main(argv=None):
    import argparse

    from .config import load_config

    ap = argparse.ArgumentParser(description="Train micro_engram on a frozen base (torchrun -m micro_engram.train cfg.yaml)")
    ap.add_argument("config")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")
    import warnings

    # The text model's output is only read (reshape + detach in the loss), never changed in place,
    # so FSDP2's pre-backward hook on it stays intact; the warning is about in-place ops only.
    warnings.filterwarnings("ignore", message=r"FSDP2-wrapped module .* returned a view tensor")
    if os.environ.get("RANK", "0") != "0":  # one copy of the setup log, from rank 0; errors from all
        logging.getLogger("micro_engram").setLevel(logging.ERROR)
    cfg = load_config(args.config)
    t = Trainer(cfg)
    try:
        t.build()
        t.train()
    except BaseException:
        # Print the error NOW and exit hard: destroy_process_group() would wait for the other
        # GPUs, which are stuck in a collective this rank never reaches, so the error would never
        # show and the run would hang. Exiting lets torchrun stop the others.
        import sys
        import traceback

        rank = os.environ.get("RANK", "?")
        print(f"\n[micro_engram] rank {rank} FAILED:", file=sys.stderr, flush=True)
        traceback.print_exc()
        sys.stderr.flush()
        if torch.cuda.is_available():
            dev = torch.cuda.current_device()
            print(f"[micro_engram] rank {rank}: {torch.cuda.memory_allocated(dev) / 2**30:.2f} GiB allocated, "
                  f"{torch.cuda.memory_reserved(dev) / 2**30:.2f} GiB reserved", file=sys.stderr, flush=True)
        os._exit(1)
    if dist.is_initialized():
        dist.destroy_process_group()
