"""micro-engram: a hashed n-gram memory for frozen decoder models.

Train with `torchrun -m micro_engram.train cfg.yaml` (micro_engram/train). The engram module,
weight loaders and run config live in module.py, weights.py and runconfig.py; vLLM serving in
the separate micro_engram_vllm package. The Axolotl plugin was removed (git tag
axolotl-plugin-final has the last version).
"""
from .runconfig import CONFIG_NAME  # noqa: F401
