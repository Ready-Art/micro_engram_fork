# Vendored from Axolotl 0.19.0 (Apache-2.0, https://github.com/axolotl-ai-cloud/axolotl), src/axolotl/kernels/op_registry.py.
# Changes: imports point at micro_engram.train.vendor; torch op namespace "micro_engram" instead of "axolotl".
"""Registration helper for axolotl's custom torch ops.

Kernel entry points registered through here become dispatcher-visible:
``torch.compile`` treats them as opaque ops instead of graph-breaking, and
policy-based selective checkpointing can match them by name. Registration
failure (exotic torch builds) falls back to the raw python callable so the
kernel keeps working — it just stays invisible to the dispatcher.
"""

from __future__ import annotations

from typing import Callable, Sequence

import torch

from logging import getLogger as get_logger

LOG = get_logger(__name__)

NAMESPACE = "micro_engram"

_warned_registration_failure = False


class _UnregisteredOp:
    """Callable shim with the CustomOpDef surface we use, for fallback."""

    def __init__(self, fn: Callable):
        self._fn = fn

    def __call__(self, *args, **kwargs):
        return self._fn(*args, **kwargs)

    def register_fake(self, fn: Callable) -> Callable:
        return fn


def register_kernel_op(name: str, mutates_args: Sequence[str] = ()) -> Callable:
    """Decorator: register the function as ``torch.ops.axolotl.<name>``.

    Returns the registered op (callable), or a passthrough shim wrapping the
    raw function if registration is unavailable.
    """

    def decorator(fn: Callable):
        try:
            return torch.library.custom_op(
                f"{NAMESPACE}::{name}", mutates_args=tuple(mutates_args)
            )(fn)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            global _warned_registration_failure
            message = (
                f"custom op registration failed for {NAMESPACE}::{name} "
                f"({type(exc).__name__}: {exc}); kernel will run unregistered "
                "(functional, but invisible to torch.compile and selective "
                "checkpointing)"
            )
            if _warned_registration_failure:
                LOG.debug(message)
            else:
                _warned_registration_failure = True
                LOG.warning(message)
            return _UnregisteredOp(fn)

    return decorator


def detab_code(code):
    """(code with its common leading indent removed, that indent); from axolotl.monkeypatch.utils."""
    import re

    m = re.match(r"([\s\t]{1,})", code)
    if m is None:
        return code, ""
    spaces = m.group(0)
    return re.sub(r"^" + spaces, "", code, flags=re.MULTILINE), spaces
