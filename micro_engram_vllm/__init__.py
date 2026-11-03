"""vLLM plugin for micro-engram (loaded automatically through the vllm.general_plugins entry point)."""


def register():
    from vllm import ModelRegistry

    # Same name as the stock class, so vLLM's Gemma 4 settings (e.g. the Triton attention
    # backend on GPUs without FA4) still apply. Models without a micro_engram_config.json
    # run as plain Gemma 4.
    ModelRegistry.register_model(
        "Gemma4ForCausalLM", "micro_engram_vllm.model:EngramGemma4ForCausalLM"
    )
