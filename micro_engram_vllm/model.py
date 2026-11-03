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


def _load_engram_config(model_path: str) -> dict | None:
    path = os.path.join(model_path, CONFIG_NAME)
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


class EngramGemma4ForCausalLM(Gemma4ForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.micro_engram_layer = None

        conf = _load_engram_config(vllm_config.model_config.model)
        if conf is None:
            logger.info("[micro_engram] no %s next to the weights: plain Gemma 4", CONFIG_NAME)
            return

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
            "[micro_engram] attached at layer %d: %s gate, %d tables x %d rows x %d dims%s",
            layer_idx, engram.gate, len(engram.tables), engram.table_size,
            engram.tables[0].embedding_dim,
            "" if layer.micro_engram_on else " (OFF: MICRO_ENGRAM_OFF=1)",
        )

    @property
    def micro_engram_ids(self) -> torch.Tensor | None:
        layer = self.micro_engram_layer
        return None if layer is None else layer.micro_engram_ids

    def get_model_state_cls(self):
        from .model_state import EngramModelState

        return EngramModelState

    def load_weights(self, weights):
        return super().load_weights((name.replace(_WRAP, "."), w) for name, w in weights)
