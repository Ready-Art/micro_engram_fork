"""micro-engram: a hashed n-gram memory for frozen decoder models.

The Axolotl training plugin lives in `plugin.py` and needs Axolotl. It is loaded here
only when Axolotl is installed, so `micro_engram.module` can also be imported on
machines without it (e.g. a vLLM server).
"""
try:
    from .plugin import (  # noqa: F401
        CONFIG_NAME,
        MicroEngramPlugin,
        _strip_wrapper_prefix,
        save_engram,
    )
except ImportError as _err:  # Axolotl (or another training-only dependency) is missing
    _PLUGIN_IMPORT_ERROR = _err

    def __getattr__(name):
        if name in ("CONFIG_NAME", "MicroEngramPlugin", "_strip_wrapper_prefix", "save_engram"):
            raise ImportError(
                f"micro_engram.{name} needs the training dependencies (Axolotl): "
                f"{_PLUGIN_IMPORT_ERROR}"
            ) from _PLUGIN_IMPORT_ERROR
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
