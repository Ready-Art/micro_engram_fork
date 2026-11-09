"""Model side of micro_engram.train: load the frozen base (bf16 or bitsandbytes NF4, no
LoRA), apply the Gemma 4 kernels, attach the engrams, shard with FSDP2."""
import inspect
import logging
import os
from contextlib import contextmanager

import torch
import torch.nn as nn

from ..module import find_text_layers

LOG = logging.getLogger("micro_engram.train")

_MM_PARTS = ("vision_tower", "audio_tower", "multi_modal_projector", "embed_vision", "embed_audio")


# ----------------------------------------------------------------------------- patches (before load)
def apply_kernel_patches(cfg, on_cuda):
    """Gemma 4 kernels, as Axolotl applied them. Triton kernels only on CUDA."""
    done = []
    if cfg.fused_attn_kernel and on_cuda:
        from .vendor.gemma4_fused_attn import patch_gemma4_fused_attn

        # the shared-KV side channel avoids a VRAM leak under checkpointing / offloading
        patch_gemma4_fused_attn(install_shared_kv_workaround=bool(cfg.gradient_checkpointing
                                                                  or cfg.activation_offloading))
        done.append("fused q/k/v norm + RoPE")
    if cfg.gemma4_hybrid_attn_impl:
        from .vendor.gemma4_hybrid_mask import patch_gemma4_hybrid_mask
        from .vendor.large_head import set_large_head_packed, set_large_head_policy

        patch_gemma4_hybrid_mask()
        set_large_head_policy(cfg.large_head_policy)
        # None = check every call whether a row really holds several documents (we run eager, so the
        # probe costs one tiny sync). Axolotl declared every batch packed, which forced an explicit
        # S x S block mask on single-document rows too: SDPA then can't skip the causal triangle,
        # ~2x the work on the 512-dim global layers (34% of the step in the stage-1 profile).
        set_large_head_packed(None)
        done.append(f"hybrid attention (large-head policy {cfg.large_head_policy})")
    if (cfg.liger_rms_norm or cfg.liger_glu_activation) and on_cuda:
        done += _apply_liger(cfg)
    if done:
        LOG.warning(f"[micro_engram] kernels: {', '.join(done)}")


def _apply_liger(cfg):
    from transformers.models.gemma4 import modeling_gemma4 as mg

    done = []
    if cfg.liger_rms_norm:
        from liger_kernel.ops.rms_norm import LigerRMSNormFunction

        names = list(inspect.signature(LigerRMSNormFunction.forward).parameters)[2:]  # after ctx, X
        orig = mg.Gemma4RMSNorm.forward

        def rms_forward(self, x):
            if not self.with_scale:
                return orig(self, x)
            # HF Gemma 4 computes everything in fp32 then casts: Liger's "gemma" casting mode
            kw = {"W": self.weight, "eps": self.eps, "offset": 0.0, "casting_mode": "gemma", "in_place": False}
            return LigerRMSNormFunction.apply(x, *[kw[n] for n in names if n in kw])

        mg.Gemma4RMSNorm.forward = rms_forward
        done.append("Liger RMSNorm")
    if cfg.liger_glu_activation:
        from liger_kernel.ops.geglu import LigerGELUMulFunction

        def mlp_forward(self, x):  # keeps the module's own (possibly 4-bit) linears
            return self.down_proj(LigerGELUMulFunction.apply(self.gate_proj(x), self.up_proj(x)))

        mg.Gemma4TextMLP.forward = mlp_forward
        done.append("Liger GEGLU")
    return done


# ----------------------------------------------------------------------------- loading
def _model_class(path):
    from transformers import AutoConfig, AutoModelForCausalLM

    conf = AutoConfig.from_pretrained(path)
    try:
        AutoModelForCausalLM._model_mapping[type(conf)]
        return AutoModelForCausalLM, conf
    except KeyError:
        from transformers import AutoModelForImageTextToText

        return AutoModelForImageTextToText, conf


def load_base(cfg, rank, local_rank, device):
    """The frozen base model. 4-bit: NF4 straight onto this GPU (no adapter). bf16: on CPU
    (cpu_ram_efficient_loading: rank 0 only, the others build it on meta and receive the
    weights after sharding)."""
    cls, conf = _model_class(cfg.base_model)
    kw = {"dtype": torch.bfloat16, "attn_implementation": cfg.attn_implementation}
    meta = False
    if cfg.load_in_4bit:
        from transformers import BitsAndBytesConfig

        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_storage=torch.bfloat16,
            llm_int8_skip_modules=list(_MM_PARTS) + ["lm_head", "embed_tokens"])
        kw["device_map"] = {"": device}
        model = cls.from_pretrained(cfg.base_model, **kw)
    elif cfg.fsdp_config.cpu_ram_efficient_loading and rank != 0:
        with torch.device("meta"):
            model = cls.from_config(conf, dtype=torch.bfloat16, attn_implementation=cfg.attn_implementation)
        meta = True
    else:
        model = cls.from_pretrained(cfg.base_model, device_map={"": "cpu"}, **kw)
    for p in model.parameters():
        p.requires_grad_(False)
    _drop_multimodal(model)
    return model, meta


def _drop_multimodal(model):
    """Training is text-only: free the vision/audio towers."""
    for _, mod in list(model.named_modules()):
        for part in _MM_PARTS:
            if isinstance(getattr(mod, part, None), nn.Module):
                setattr(mod, part, None)


def text_model(model):
    """(name, module) of the text decoder stack's owner (Gemma4TextModel)."""
    name, _ = find_text_layers(model)
    parent = name.rsplit(".", 1)[0] if "." in name else ""
    return parent, (model.get_submodule(parent) if parent else model)


def hybrid_attention(text, cfg):
    """Global (512-dim) layers onto the packing-aware SDPA impl, sliding ones stay on FA2."""
    if not cfg.gemma4_hybrid_attn_impl:
        return
    import copy

    from .vendor.gemma4_hybrid_mask import GLOBAL_PACKED_SDPA

    types = text.config.layer_types
    n = 0
    for i, layer in enumerate(text.layers):
        if types[i] != "sliding_attention":
            c = copy.copy(layer.self_attn.config)
            c._attn_implementation = GLOBAL_PACKED_SDPA
            layer.self_attn.config = c
            n += 1
    LOG.warning(f"[micro_engram] hybrid attention: {n} global layers on packed SDPA, "
                f"{len(text.layers) - n} sliding layers on {cfg.attn_implementation}")


# ----------------------------------------------------------------------------- vocab rows
def take_vocab(model, text, rank, world, device, broadcast, chunk_tokens=2048):
    """Split the tied embedding/lm_head by rows over the GPUs (VocabShard) and remove both
    from the model. broadcast: only rank 0 has the weights (cpu_ram_efficient_loading)."""
    import torch.distributed as dist

    from .head import VocabShard

    emb = text.embed_tokens
    W = emb.weight.detach()
    softcap = getattr(text.config, "final_logit_softcapping", None)
    scale = getattr(emb, "scalar_embed_scale", None)
    if scale is None and getattr(emb, "embed_scale", None) is not None and not emb.embed_scale.is_meta:
        scale = float(emb.embed_scale)
    if scale is None:
        scale = float(text.config.hidden_size) ** 0.5  # Gemma's scaled embedding
    lm_head = getattr(model, "lm_head", None)
    if lm_head is not None and not W.is_meta and lm_head.weight is not emb.weight \
            and not torch.equal(lm_head.weight.detach().to(W.device), W):
        raise ValueError("lm_head is not tied to the embedding: the vocab-parallel loss assumes it is")
    if broadcast and world > 1:
        buf = W.to(device=device, dtype=torch.bfloat16) if rank == 0 else \
            torch.empty(W.shape, dtype=torch.bfloat16, device=device)
        dist.broadcast(buf, src=0)
        W = buf
    shard = VocabShard(W, rank, world, embed_scale=scale, softcap=softcap, chunk_tokens=chunk_tokens,
                       device=device, dtype=torch.bfloat16)
    del W
    text.embed_tokens = _NoEmbedding(shard.V, shard.d)
    if lm_head is not None:
        model.lm_head = None
    return shard


class _NoEmbedding(nn.Module):
    """Stands in for the embedding the trainer took out (the text model gets inputs_embeds)."""

    def __init__(self, num, dim):
        super().__init__()
        self.num_embeddings, self.embedding_dim = num, dim

    def forward(self, ids):  # pragma: no cover - never called by the trainer
        raise RuntimeError("micro_engram.train feeds inputs_embeds; the embedding lives in VocabShard")


# ----------------------------------------------------------------------------- no-grad prefix
@contextmanager
def grad_from_layer(text, first_layer, enabled):
    """Run the embedding and decoder layers up to and including `first_layer` without autograd:
    nothing before the first trained engram needs a gradient. Grad turns on right before that
    layer's engram injects (prepended forward hook, so it runs before the engram's hook)."""
    if not enabled or first_layer is None:
        yield
        return
    layer = text.layers[first_layer]

    def grad_on(*_):
        torch.set_grad_enabled(True)  # returns None: the layer's output is unchanged

    h = layer.register_forward_hook(grad_on, prepend=True)
    prev = torch.is_grad_enabled()
    torch.set_grad_enabled(False)
    try:
        yield
    finally:
        h.remove()
        torch.set_grad_enabled(prev)


def checkpointing(model, text, cfg, no_grad_upto):
    """HF-style per-layer non-reentrant checkpointing (what Axolotl's gradient_checkpointing does);
    layers that run without autograd (the no-grad prefix) don't need it."""
    if not cfg.gradient_checkpointing:
        return
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if no_grad_upto is not None:
        for i in range(no_grad_upto + 1):
            text.layers[i].gradient_checkpointing = False


# ----------------------------------------------------------------------------- FSDP2
def shard_model(text, cfg, mesh, engram_params):
    """fully_shard every decoder layer, then the text model (root). The engrams' dense
    parameters stay outside FSDP (replicated; the trainer all-reduces their gradients); their
    tables are row-sharded on their own already."""
    from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard

    if cfg.load_in_4bit:
        from .vendor.fsdp2_4bit import (apply_init_dtype_attrs_patch, apply_init_sharded_param_patch,
                                        apply_init_unsharded_param_patch)

        apply_init_sharded_param_patch()
        apply_init_unsharded_param_patch()
        apply_init_dtype_attrs_patch()
        if not (getattr(apply_init_sharded_param_patch, "_axolotl_patched", False)
                and getattr(apply_init_unsharded_param_patch, "_axolotl_patched", False)):
            raise RuntimeError("load_in_4bit: the FSDP2 4-bit patch does not match this torch version "
                               f"({torch.__version__}); FSDP would lose the NF4 quantization state")
    kw = {"mesh": mesh, "mp_policy": MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)}
    if cfg.fsdp_config.offload_params:
        kw["offload_policy"] = CPUOffloadPolicy(pin_memory=os.environ.get("FSDP_CPU_OFFLOAD_PIN_MEMORY", "1") != "0")
    ignored = set(engram_params)
    for layer in text.layers:
        fully_shard(layer, reshard_after_forward=cfg.fsdp_config.reshard_after_forward,
                    ignored_params=ignored & set(layer.parameters()) or None, **kw)
    fully_shard(text, reshard_after_forward=cfg.fsdp_config.reshard_after_forward,
                ignored_params=ignored & set(text.parameters()) or None, **kw)
    return text
