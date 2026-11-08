"""Gemma 4 with a micro-engram, for vLLM (text-only, Model Runner V2).

Each engram layer (layer 0, plus any others the config lists) gets its engram and a
view of one shared, fixed token-id buffer. Before every step the model state fills that
buffer with each token and the ones before it; inside the compiled model, each engram
layer hashes them and adds its gated memory to its output, exactly like training.
Without a micro_engram_config.json next to the weights this class behaves like stock
Gemma 4.
"""
import json
import os

import torch

from vllm.config import VllmConfig
from vllm.distributed import (get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_reduce)
from vllm.logger import init_logger
from vllm.model_executor.models.gemma4 import Gemma4DecoderLayer, Gemma4ForCausalLM
from vllm.model_executor.models.gemma4_mm import Gemma4ForConditionalGeneration
from vllm.model_executor.models.utils import PPMissingLayer

from micro_engram.module import engram_from_spec
from micro_engram.weights import _local_name, base_mismatch, engram_specs, layer_of_key, load_engram_weights

from .ids import mask_media
from .tp import load_rows, tp_lookup

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
        keep = None
        if self.micro_engram_media is not None:  # multimodal model: set at load time
            ids, keep = mask_media(ids, self.micro_engram_media)
        idx = self.micro_engram.hash_prev(ids)
        return self.micro_engram.inject(hidden_states, idx, keep), residual


def _offload_wanted(vllm_config: VllmConfig) -> bool:
    """Engram tables in pinned host memory: --additional-config '{"micro_engram_offload": true}'
    or MICRO_ENGRAM_OFFLOAD=1."""
    extra = vllm_config.additional_config
    if isinstance(extra, dict) and "micro_engram_offload" in extra:
        return bool(extra["micro_engram_offload"])
    return os.environ.get("MICRO_ENGRAM_OFFLOAD", "0") == "1"


def _host_views(shards):
    """GPU views (UVA) of pinned CPU rows: kernels read only the rows they index, over PCIe,
    at fixed addresses (CUDA graphs keep working). Same mechanism as vLLM's own Engram offload."""
    from vllm.utils.platform_utils import is_uva_available
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

    if not is_uva_available():
        raise RuntimeError("[micro_engram] micro_engram_offload needs UVA (pinned host memory readable "
                           "by the GPU); this platform has none: serve without offload")
    return [get_accelerator_view_from_cpu_tensor(p.data) for p in shards]


def _resolve_engram(vllm_config: VllmConfig):
    """Which engram(s) to load: (config, config path, separate weights?) or (None, None, False).

    1. --additional-config '{"micro_engram": "<micro_engram_config.json or its folder>"}'
       (or the MICRO_ENGRAM_CONFIG environment variable): each engram's weights come from
       the file its "weights" field points to; the base model is left untouched.
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
        for spec in engram_specs(conf, path):
            if not spec["weights"]:
                raise ValueError(
                    f"[micro_engram] {path}: the engram at layer {spec['layer']} has no \"weights\". "
                    f"This is probably the copy training writes to output_dir at the start (before any "
                    f"checkpoint exists): point --additional-config at checkpoint-N/{CONFIG_NAME} "
                    f"instead, or set \"weights\": \"<file or folder; relative paths are relative to "
                    f"this config's folder>\""
                )
            if not os.path.exists(spec["weights"]):
                raise ValueError(
                    f"[micro_engram] {path}: the weights of the engram at layer {spec['layer']} are not "
                    f"there: {spec['weights']} (a relative \"weights\" is relative to the config's "
                    f"folder, so a config copied elsewhere needs an absolute path)"
                )
        return conf, path, True

    path = os.path.join(vllm_config.model_config.model, CONFIG_NAME)
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f), path, False
    return None, None, False


class EngramGemma4ForCausalLM(Gemma4ForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.micro_engram_slots = []      # [(spec, layer)] for the engrams on this rank
        self.micro_engram_separate = False  # True: weights from files, not the checkpoint
        self._micro_engram_ids = None
        self.micro_engram_conf = None

        conf, conf_path, separate = _resolve_engram(vllm_config)
        if conf is None:
            logger.info("[micro_engram] no engram configured: plain Gemma 4")
            return
        self.micro_engram_separate = separate
        self.micro_engram_conf = conf
        if (msg := base_mismatch(conf, vllm_config.model_config.model)):
            logger.warning("[micro_engram] %s", msg)

        # Inside the multimodal model the language model sees the top-level architecture;
        # EngramGemma4ForConditionalGeneration feeds it. Other wrappers would not.
        arch = vllm_config.model_config.architecture
        if arch not in ("Gemma4ForCausalLM", "Gemma4ForConditionalGeneration"):
            raise ValueError(
                f"[micro_engram] {arch} is not supported (the engram would never be fed); "
                "serve Gemma4ForConditionalGeneration or Gemma4ForCausalLM"
            )
        if not vllm_config.use_v2_model_runner:
            raise ValueError(
                "[micro_engram] needs Model Runner V2 (vLLM's default). Something forced "
                "V1, e.g. VLLM_USE_V2_MODEL_RUNNER=0 or a feature V2 does not support."
            )

        specs = engram_specs(conf, conf_path)
        max_order = max(max(s["orders"]) for s in specs)
        # The tables are split by rows across the tensor-parallel GPUs (like the vocabulary
        # embedding): each holds 1/TP of them, a lookup is masked + one all-reduce (tp.py).
        tp_rank, tp_size = get_tensor_model_parallel_rank(), get_tensor_model_parallel_world_size()
        lookup = tp_lookup(tensor_model_parallel_all_reduce)
        offload = _offload_wanted(vllm_config)
        on = os.environ.get("MICRO_ENGRAM_OFF", "0") != "1"  # =1: load but skip, for A/B tests
        for spec in specs:
            layer = self.model.layers[spec["layer"]]
            if isinstance(layer, PPMissingLayer):
                continue  # pipeline parallel: this layer lives on another rank
            engram = engram_from_spec(spec, self.config.hidden_size, self.config.vocab_size,
                                      self.config.pad_token_id or 0, build_tables=False)
            ref = next(layer.parameters())
            engram.to(device=ref.device, dtype=ref.dtype)  # ints (canon, mults) keep their dtype
            # this GPU's rows, filled by load_weights (not registered: vLLM's loader never sees them);
            # with offload they stay in pinned host memory and the GPU reads them through UVA views
            if offload:
                engram.shard_rows(tp_rank, tp_size, device="cpu", dtype=ref.dtype, requires_grad=False,
                                  empty=True, pin_memory=True)
                engram.__dict__["row_views"] = _host_views(engram.row_shards)
            else:
                engram.shard_rows(tp_rank, tp_size, device=ref.device, dtype=ref.dtype, requires_grad=False,
                                  empty=True)
            engram.row_lookup_fn = lookup
            layer.add_module("micro_engram", engram)
            if self._micro_engram_ids is None:
                # one buffer for every engram layer: filled once per step, read in place
                self._micro_engram_ids = torch.full(
                    (vllm_config.scheduler_config.max_num_batched_tokens, max_order),
                    -1, dtype=torch.int32, device=ref.device,
                )
            layer.register_buffer("micro_engram_ids", self._micro_engram_ids, persistent=False)
            layer.micro_engram_on = on
            layer.micro_engram_media = None  # set by the multimodal wrapper
            layer.__class__ = EngramGemma4DecoderLayer
            self.micro_engram_slots.append((spec, layer))
            logger.info(
                "[micro_engram] attached at layer %d: %s gate, %d tables of %d dims, rows per order %s "
                "(this rank holds 1/%d of each: %.2f GiB %s), seed %d%s, from %s",
                spec["layer"], engram.gate, len(engram.row_shards), engram.engram_dim, engram.order_sizes,
                tp_size, sum(p.numel() * p.element_size() for p in engram.row_shards) / 2**30,
                "in pinned host memory" if offload else "on the GPU", spec["seed"],
                "" if on else " (OFF: MICRO_ENGRAM_OFF=1)",
                spec["weights"] if separate else "the model checkpoint",
            )

    @property
    def micro_engram_ids(self) -> torch.Tensor | None:
        return self._micro_engram_ids

    def get_model_state_cls(self):
        from .model_state import EngramModelState

        return EngramModelState

    def load_weights(self, weights):
        if not self.micro_engram_slots:
            return super().load_weights(weights)

        # Engram tensors never go to vLLM's loader (it copies by name, and the tables are split
        # across the TP GPUs): they are collected per layer and loaded by load_rows.
        found = {}

        def without_engram():
            for name, w in weights:
                name = name.replace(_WRAP, ".")
                if ".micro_engram." in name:
                    found.setdefault(layer_of_key(name), {})[_local_name(name)] = w
                else:
                    yield name, w

        loaded = super().load_weights(without_engram())
        if self.micro_engram_separate:
            if found:
                logger.warning("[micro_engram] the checkpoint has its own engram (%d tensors); "
                               "using the configured engram weights instead", sum(map(len, found.values())))
            # vLLM's AutoWeightsLoader groups names with itertools.groupby, so with vision weights
            # between the shards this runs once per run of language-model names: load only once
            if not getattr(self, "_micro_engram_loaded", False):
                self._micro_engram_loaded = True
                for spec, layer in self.micro_engram_slots:
                    loaded |= self._load_separate_engram(spec, layer)
            return loaded

        # engram inside the checkpoint (assembled folder): may arrive over several calls
        done = self.__dict__.setdefault("_micro_engram_tables_done", {})
        for spec, layer in self.micro_engram_slots:
            part = found.get(spec["layer"])
            if not part:
                continue
            engram = layer.micro_engram
            got = done.setdefault(spec["layer"], set())
            got |= load_rows(engram, part, f"the checkpoint (layer {spec['layer']})", complete=False)
            loaded |= self._engram_param_names(engram, part)
            n = len(engram.row_shards)
            if len(got) == n:
                logger.info("[micro_engram] layer %d: loaded from the checkpoint", spec["layer"])
            else:
                logger.info("[micro_engram] layer %d: %d of %d tables so far", spec["layer"], len(got), n)
        return loaded

    def _engram_param_names(self, engram, keys=None) -> set[str]:
        prefix = next(n for n, m in self.named_modules() if m is engram)
        return {f"{prefix}.{n}" for n, _ in engram.named_parameters() if keys is None or n in keys}

    def _load_separate_engram(self, spec, layer) -> set[str]:
        engram = layer.micro_engram
        shared = sum(s["weights"] == spec["weights"] for s, _ in self.micro_engram_slots) > 1
        sd = load_engram_weights(spec["weights"], layer=spec["layer"], need_layer=shared)
        if not any(k.startswith("tables.") for k in sd):
            raise ValueError(f"[micro_engram] no engram weights for layer {spec['layer']} in {spec['weights']}")
        if "canon" not in sd and spec.get("token_merge"):
            canon_path = spec.get("canon_path")
            if not canon_path or not os.path.isfile(canon_path):
                raise ValueError("[micro_engram] the run used token merging but the weights have no "
                                 f"'canon' and the canon file is missing ({canon_path})")
            sd["canon"] = torch.load(canon_path, map_location="cpu")
        load_rows(engram, sd, spec["weights"])  # this GPU's rows + the rest, onto the GPU in its dtype
        logger.info("[micro_engram] layer %d: loaded %d engram tensors from %s",
                    spec["layer"], len(sd), spec["weights"])
        return self._engram_param_names(engram)


class EngramGemma4ForConditionalGeneration(Gemma4ForConditionalGeneration):
    """Multimodal Gemma 4 with the engram.

    The language model inside is built through the registry as "Gemma4ForCausalLM", so
    it is already EngramGemma4ForCausalLM. vLLM takes the model state from the top-level
    model, so this class hands it the engram's model state and token-id buffer, and
    tells the engram layers which token ids are media placeholders (no engram there).
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        slots = getattr(self.language_model, "micro_engram_slots", None)
        if not slots:
            return
        media = sorted({int(t) for t in (getattr(self.config, n, None)
                                         for n in ("image_token_id", "video_token_id", "audio_token_id"))
                        if t is not None})
        for _, layer in slots:
            layer.__dict__.pop("micro_engram_media", None)  # the None placeholder; register_buffer refuses existing names
            layer.register_buffer(
                "micro_engram_media",
                torch.tensor(media, dtype=torch.int32, device=layer.micro_engram_ids.device),
                persistent=False,
            )
        logger.info("[micro_engram] multimodal model: no engram on media placeholder tokens %s", media)

    @property
    def micro_engram_ids(self) -> torch.Tensor | None:
        return getattr(self.language_model, "micro_engram_ids", None)

    def get_model_state_cls(self):
        from .model_state import EngramModelState

        return EngramModelState
