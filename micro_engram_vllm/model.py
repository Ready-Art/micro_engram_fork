"""Gemma 4 with a micro-engram, for vLLM (text-only, Model Runner V2).

Layer 0 gets the engram and a fixed token-id buffer. Before every step the model
state fills that buffer with each token and the ones before it; inside the compiled
model, layer 0 hashes them and adds the gated memory to its output, exactly like
training. Without a micro_engram_config.json next to the weights this class behaves
like stock Gemma 4.
"""
import json
import os

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.gemma4 import Gemma4DecoderLayer, Gemma4ForCausalLM
from vllm.model_executor.models.utils import PPMissingLayer

from micro_engram.module import MicroEngram
from micro_engram.weights import base_mismatch, load_engram_weights, resolve_weights_path

# Under "vllm." so vLLM's log handler prints it (other logger names are dropped).
logger = init_logger("vllm.micro_engram")

CONFIG_NAME = "micro_engram_config.json"
_WRAP = "._checkpoint_wrapped_module."  # can appear in keys saved with checkpointing


class EngramGemma4DecoderLayer(Gemma4DecoderLayer):
    """Gemma 4 decoder layer that adds the engram memory to its output."""

    def forward(self, positions, hidden_states, residual, per_layer_input=None, **kwargs):
        hidden_states, residual = super().forward(
            positions, hidden_states, residual, per_layer_input=per_layer_input, **kwargs
        )
        if not self.micro_engram_on:  # fixed at load time, so compile sees a constant
            return hidden_states, residual
        ids = self.micro_engram_ids[: hidden_states.shape[0]]
        idx = self.micro_engram.hash_prev(ids)
        return self.micro_engram.inject(hidden_states, idx), residual


def _resolve_engram(vllm_config: VllmConfig):
    """Which engram to load: (config, path to separate weights or None) or (None, None).

    1. --additional-config '{"micro_engram": "<micro_engram_config.json or its folder>"}'
       (or the MICRO_ENGRAM_CONFIG environment variable): the weights come from the
       file the config's "weights" field points to; the base model is left untouched.
    2. A micro_engram_config.json next to the model weights (an assembled folder):
       the weights come from the checkpoint itself.
    3. Neither: plain Gemma 4.
    """
    extra = vllm_config.additional_config
    chosen = (extra.get("micro_engram") if isinstance(extra, dict) else None) or os.environ.get(
        "MICRO_ENGRAM_CONFIG"
    )
    if chosen:
        path = os.path.join(chosen, CONFIG_NAME) if os.path.isdir(chosen) else chosen
        if not os.path.isfile(path):
            raise ValueError(f"[micro_engram] engram config not found: {path}")
        with open(path) as f:
            conf = json.load(f)
        weights = resolve_weights_path(conf, path)
        if weights is None or not os.path.exists(weights):
            raise ValueError(
                f"[micro_engram] {path} must say where the engram is: add \"weights\": "
                f"\"<file or folder, relative to the config>\" (found: {weights})"
            )
        return conf, weights

    path = os.path.join(vllm_config.model_config.model, CONFIG_NAME)
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f), None
    return None, None


class EngramGemma4ForCausalLM(Gemma4ForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.micro_engram_layer = None
        self.micro_engram_weights = None  # separate engram file, or None: it is in the checkpoint
        self.micro_engram_conf = None

        conf, weights = _resolve_engram(vllm_config)
        if conf is None:
            logger.info("[micro_engram] no engram configured: plain Gemma 4")
            return
        self.micro_engram_weights = weights
        self.micro_engram_conf = conf
        if (msg := base_mismatch(conf, vllm_config.model_config.model)):
            logger.warning("[micro_engram] %s", msg)

        if vllm_config.model_config.architecture != "Gemma4ForCausalLM":
            raise ValueError(
                "[micro_engram] serve the text-only architecture, otherwise the engram is "
                "never fed: add --hf-overrides '{\"architectures\": [\"Gemma4ForCausalLM\"]}'"
            )
        if not vllm_config.use_v2_model_runner:
            raise ValueError(
                "[micro_engram] needs Model Runner V2 (vLLM's default). Something forced "
                "V1, e.g. VLLM_USE_V2_MODEL_RUNNER=0 or a feature V2 does not support."
            )

        layer_idx = conf.get("layer", 0)
        layer = self.model.layers[layer_idx]
        if isinstance(layer, PPMissingLayer):
            return  # pipeline parallel: layer 0 lives on another rank

        engram = MicroEngram(
            model_dim=self.config.hidden_size,
            table_size=conf["table_size"],
            engram_dim=conf["engram_dim"],
            num_heads=conf["num_heads"],
            orders=tuple(conf.get("orders", (2, 3))),
            pad_id=conf.get("pad_id", self.config.pad_token_id or 0),
            vocab_size=self.config.vocab_size,
            gate=conf.get("gate", "scalar"),  # runs from before the gate option are scalar
        )
        ref = next(layer.parameters())
        engram.to(device=ref.device, dtype=ref.dtype)  # ints (canon, mults) keep their dtype
        layer.add_module("micro_engram", engram)
        layer.register_buffer(
            "micro_engram_ids",
            torch.full(
                (vllm_config.scheduler_config.max_num_batched_tokens, max(engram.orders)),
                -1,
                dtype=torch.int32,
                device=ref.device,
            ),
            persistent=False,
        )
        # MICRO_ENGRAM_OFF=1 loads the engram weights but skips it, for A/B tests.
        layer.micro_engram_on = os.environ.get("MICRO_ENGRAM_OFF", "0") != "1"
        layer.__class__ = EngramGemma4DecoderLayer
        self.micro_engram_layer = layer
        logger.info(
            "[micro_engram] attached at layer %d: %s gate, %d tables x %d rows x %d dims%s, from %s",
            layer_idx, engram.gate, len(engram.tables), engram.table_size,
            engram.tables[0].embedding_dim,
            "" if layer.micro_engram_on else " (OFF: MICRO_ENGRAM_OFF=1)",
            weights or "the model checkpoint",
        )

    @property
    def micro_engram_ids(self) -> torch.Tensor | None:
        layer = self.micro_engram_layer
        return None if layer is None else layer.micro_engram_ids

    def get_model_state_cls(self):
        from .model_state import EngramModelState

        return EngramModelState

    def load_weights(self, weights):
        weights = ((name.replace(_WRAP, "."), w) for name, w in weights)
        if self.micro_engram_weights is None or self.micro_engram_layer is None:
            return super().load_weights(weights)

        # Separate engram: ignore any engram already in the checkpoint, then load the chosen one.
        skipped = []

        def without_engram():
            for name, w in weights:
                if ".micro_engram." in name:
                    skipped.append(name)
                else:
                    yield name, w

        loaded = super().load_weights(without_engram())
        if skipped:
            logger.warning("[micro_engram] the checkpoint has its own engram (%d tensors); "
                           "using %s instead", len(skipped), self.micro_engram_weights)
        return loaded | self._load_separate_engram()

    def _load_separate_engram(self) -> set[str]:
        engram = self.micro_engram_layer.micro_engram
        sd = load_engram_weights(self.micro_engram_weights)
        if not any(k.startswith("tables.") for k in sd):
            raise ValueError(f"[micro_engram] no engram weights in {self.micro_engram_weights}")
        if "canon" not in sd and self.micro_engram_conf.get("token_merge"):
            canon_path = self.micro_engram_conf.get("canon_path")
            if not canon_path or not os.path.isfile(canon_path):
                raise ValueError("[micro_engram] the run used token merging but the weights have no "
                                 f"'canon' and the canon file is missing ({canon_path})")
            sd["canon"] = torch.load(canon_path, map_location="cpu")
        result = engram.load_state_dict(sd, strict=False)  # copies onto the GPU, in its dtype
        problems = [k for k in result.missing_keys if k != "mults"] + list(result.unexpected_keys)
        if problems:  # mults is rebuilt from its seed; anything else is a wrong file
            raise ValueError(f"[micro_engram] {self.micro_engram_weights} does not match the "
                             f"config: {problems[:10]}")
        prefix = next(n for n, m in self.named_modules() if m is engram)
        logger.info("[micro_engram] loaded %d engram tensors from %s", len(sd), self.micro_engram_weights)
        return {f"{prefix}.{n}" for n, _ in engram.named_parameters()}
