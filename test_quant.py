"""
Engram on a 4-bit (bitsandbytes NF4) base with a placeholder QLoRA, on CPU.
Needs bitsandbytes and peft:  python test_quant.py

Mirrors the Axolotl setup: adapter: qlora (a rank-1 LoRA on one module, only so Axolotl
loads the base in 4-bit), the engram attached after PEFT, and
unfrozen_parameters: ['.+micro_engram'] applied with Axolotl's own pattern rule.
"""
import re
import sys

import torch
import torch.nn as nn

try:
    import bitsandbytes as bnb
    from peft import LoraConfig, get_peft_model
except ImportError as e:
    sys.exit(f"skipped: {e}")

from micro_engram.module import MicroEngram, attach_engram, layer_device_dtype

D, V = 64, 1000


def q4(storage):
    lin = bnb.nn.Linear4bit(D, D, bias=False, compute_dtype=torch.bfloat16, quant_type="nf4",
                            quant_storage=storage)
    lin.weight = bnb.nn.Params4bit(torch.randn(D, D) * 0.05, requires_grad=False, quant_type="nf4",
                                   quant_storage=storage)
    return lin


class Layer(nn.Module):
    def __init__(self, storage):
        super().__init__()
        self.q_proj = q4(storage)  # first parameter is the packed 4-bit weight, like Gemma
        self.o_proj = q4(storage)
        self.input_layernorm = nn.RMSNorm(D, dtype=torch.bfloat16)

    def forward(self, h, **kw):
        return h + self.o_proj(torch.tanh(self.q_proj(self.input_layernorm(h))))


class Inner(nn.Module):
    def __init__(self, storage):
        super().__init__()
        self.embed_tokens = nn.Embedding(V, D, dtype=torch.bfloat16)
        self.layers = nn.ModuleList(Layer(storage) for _ in range(3))


class Wrap(nn.Module):
    def __init__(self, storage):
        super().__init__()
        self.language_model = Inner(storage)


class Toy(nn.Module):
    def __init__(self, storage):
        super().__init__()
        self.model = Wrap(storage)
        self.lm_head = nn.Linear(D, V, bias=False, dtype=torch.bfloat16)
        self.config = type("C", (), {"use_return_dict": True, "tie_word_embeddings": False,
                                     "to_dict": lambda self: {}})()

    def forward(self, input_ids=None, **kw):
        h = self.model.language_model.embed_tokens(input_ids)
        for layer in self.model.language_model.layers:
            h = layer(h)
        return self.lm_head(h)


def axolotl_pattern(p):  # utils/freeze.py LayerNamePattern: dots literal unless followed by '+'
    return re.compile(re.sub(r"\.(?!\+)", "\\.", p))


results = []
torch.manual_seed(0)
for storage in (torch.bfloat16, torch.uint8):
    tag = f"[{str(storage).split('.')[-1]} storage]"
    base = Toy(storage).to("cpu")  # moving to a device is what quantizes Params4bit
    first = next(base.model.language_model.layers[0].parameters())
    peft_model = get_peft_model(base, LoraConfig(r=1, lora_alpha=1, target_modules=[
        "model.language_model.layers.2.o_proj"]))
    eng = MicroEngram(model_dim=D, table_size=257, engram_dim=8, num_heads=2, vocab_size=V,
                      gate="scalar_sqrt")
    attach_engram(peft_model, eng, 0)  # after PEFT, as in post_model_load

    results.append((f"{tag} first layer param is the packed weight ({first.dtype})",
                    type(first).__name__ == "Params4bit"))
    results.append((f"{tag} engram is bf16 (compute dtype), not the storage dtype",
                    eng.value.weight.dtype == torch.bfloat16 and eng.tables[0].weight.dtype == torch.bfloat16
                    and eng.mults.dtype == torch.int64))

    pat = axolotl_pattern(".+micro_engram")
    for name, p in peft_model.named_parameters():
        p.requires_grad = bool(pat.match(name))
    trainable = [n for n, p in peft_model.named_parameters() if p.requires_grad]
    results.append((f"{tag} '.+micro_engram' unfreezes exactly the engram ({len(trainable)} tensors, "
                    f"e.g. {trainable[0]})",
                    trainable and all("micro_engram" in n for n in trainable)
                    and len(trainable) == len(list(eng.parameters()))
                    and trainable[0].startswith("base_model.model.")))
    results.append((f"{tag} old pattern (no prefix) would match nothing under PEFT",
                    not any(axolotl_pattern("model.language_model.layers.0.(_checkpoint_wrapped_module.)?"
                                            "micro_engram..*").match(n)
                            for n, _ in peft_model.named_parameters())))

    ids = torch.randint(5, V, (2, 10))
    with torch.no_grad():
        eng.value.weight.normal_(std=0.02)  # past step 0, so the tables get gradients too
    loss = peft_model(input_ids=ids).float().pow(2).mean()
    loss.backward()
    g = eng.tables[0].weight.grad
    results.append((f"{tag} gradient reaches the engram through the 4-bit layers",
                    g is not None and g.abs().sum() > 0 and eng.q_scale.grad is not None))
    results.append((f"{tag} 4-bit base and LoRA get no gradient",
                    all(p.grad is None for n, p in peft_model.named_parameters() if "micro_engram" not in n)))

# the LoRA really is a no-op: same output with and without the adapter layers
torch.manual_seed(1)
base = Toy(torch.bfloat16).to("cpu")
ids = torch.randint(5, V, (2, 10))
with torch.no_grad():
    ref = base(input_ids=ids)
    pm = get_peft_model(base, LoraConfig(r=1, lora_alpha=1, target_modules=["model.language_model.layers.2.o_proj"]))
    results.append(("placeholder LoRA (B = 0) leaves the 4-bit model's output unchanged",
                    torch.equal(pm(input_ids=ids), ref)))

# plain bf16 layers still pick bf16 / fp32 as before
lay = nn.Sequential(nn.Linear(4, 4, dtype=torch.float32))
results.append(("unquantized layer: dtype from its first weight, as before",
                 layer_device_dtype(lay)[1] == torch.float32))

bad = 0
for name, ok in results:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    bad += not ok
sys.exit(1 if bad else 0)
