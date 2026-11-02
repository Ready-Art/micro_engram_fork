"""
CPU-only test: does the plugin's key-rename hook make resume loading work?
Run from the folder that contains the micro_engram package:
    python test_resume_keys.py
"""
import torch
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from torch.distributed.checkpoint.state_dict import get_model_state_dict, set_model_state_dict

from micro_engram import _strip_wrapper_prefix  # the exact hook the plugin uses

WRAP = "_checkpoint_wrapped_module"


class FakeEngram(nn.Module):
    def __init__(self, d=8):
        super().__init__()
        self.tables = nn.ModuleList(nn.Embedding(16, 4) for _ in range(2))
        self.key = nn.Linear(8, d, bias=False)
        self.value = nn.Linear(8, d, bias=False)
        self.register_buffer("mults", torch.arange(6).view(2, 3))


class Block(nn.Module):
    def __init__(self, d=8):
        super().__init__()
        self.lin = nn.Linear(d, d)


class LM(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(Block() for _ in range(4))


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = LM()


def build(with_hook):
    m = Toy()
    layers = m.model.language_model.layers
    # same order as the plugin: attach the engram first, wrap for checkpointing after
    layers[1].add_module("micro_engram", FakeEngram())
    for i in range(len(layers)):
        layers[i] = checkpoint_wrapper(layers[i])
    if with_hook:
        m._register_load_state_dict_pre_hook(_strip_wrapper_prefix)
    return m


def engram_weights(m):
    return {k: v.clone() for k, v in m.state_dict().items() if "micro_engram" in k}


def make_checkpoint(wrapped_keys):
    src = build(with_hook=False)
    with torch.no_grad():
        for k, p in src.named_parameters():
            if "micro_engram" in k:
                p.normal_()
    sd = get_model_state_dict(src)
    if wrapped_keys:
        sd = {k.replace("layers.1.micro_engram", f"layers.1.{WRAP}.micro_engram"): v
              for k, v in sd.items()}
    return src, sd


def attempt(label, wrapped_keys, with_hook):
    src, sd = make_checkpoint(wrapped_keys)
    dst = build(with_hook)
    try:
        set_model_state_dict(dst, sd)
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] {label}\n       {type(e).__name__}: {str(e)[:300]}\n")
        return False
    a, b = engram_weights(src), engram_weights(dst)
    same = all(torch.equal(a[k], b[k]) for k in a if k in b) and set(a) == set(b)
    print(f"[{'OK  ' if same else 'WRONG'}] {label}"
          f"{'' if same else '  (loaded without error but values differ)'}\n")
    return same


if __name__ == "__main__":
    print("Keys the toy model produces when saving:")
    for k in get_model_state_dict(build(False)):
        if "micro_engram" in k and "tables.0" in k:
            print("   ", k)
    print()
    attempt("checkpoint keys WITHOUT wrapper name, no fix", False, False)
    attempt("checkpoint keys WITH wrapper name,    no fix", True, False)
    attempt("checkpoint keys WITHOUT wrapper name, with fix", False, True)
    attempt("checkpoint keys WITH wrapper name,    with fix", True, True)