"""
GPU tests for the Triton kernels micro_engram.train uses (vendored from Axolotl): run on the server.

    python test_kernels_gpu.py            # needs CUDA, triton, flash-attn (for the hybrid test)

Each kernel is compared with plain PyTorch on random inputs, forward AND gradients:
  1. flash_d512 (head_dim 512, the Gemma 4 global layers): packed rows (several documents,
     varlen from position_ids) and one-document rows, against fp32 attention with the
     block-diagonal causal mask.
  2. fused q/k/v RMSNorm + RoPE (fused_attn_kernel): the patched Gemma4TextAttention.forward
     against the original one, on a sliding layer and a global layer (partial rotary).
  3. end to end, a tiny Gemma 4 with head_dim-512 global layers on a packed batch: the
     trainer's kernel setup (fused attention + hybrid attention + FA2 + large_head_attention:
     auto) against stock HF SDPA: same loss, same gradient into the inputs, and the Triton
     kernel and the fused RoPE actually ran.

bf16 everywhere, so the comparisons use relative error (|a-b| / |b| over the whole tensor).
"""
import os
import sys
import types

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if not torch.cuda.is_available():
    print("skipped: needs a CUDA GPU")
    sys.exit(0)

DEV = torch.device("cuda")
FAIL = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"   [{detail}]" if detail else ""), flush=True)
    if not ok:
        FAIL.append(name)


def rel(a, b):
    a, b = a.float(), b.float()
    return float((a - b).norm() / b.norm().clamp_min(1e-12))


# ----------------------------------------------------------------------------- 1. flash_d512
def ref_attention(q, k, v, pos, scale):
    """fp32 causal attention, documents isolated by position_ids restarting at 0. q/k/v [B,H,S,D]."""
    q, k, v = q.float(), k.float(), v.float()
    S = q.shape[2]
    doc = (pos == 0).cumsum(-1)                                   # [B,S]
    allow = (doc[:, :, None] == doc[:, None, :]) & torch.ones(S, S, dtype=torch.bool, device=q.device).tril()
    s = (q @ k.transpose(-1, -2)) * scale
    s = s.masked_fill(~allow[:, None], float("-inf"))
    return torch.softmax(s, dim=-1) @ v


def test_flash_d512():
    from micro_engram.train.vendor.flash_attn_d512 import flash_d512

    torch.manual_seed(0)
    B, H, S, D = 2, 4, 1024, 512
    lens = [[300, 500, 224], [1024]]                              # row 0 packed, row 1 one document
    pos = torch.cat([torch.cat([torch.arange(n) for n in row])[None] for row in lens]).to(DEV)
    for scale, amp in ((D ** -0.5, 1.0), (1.0, 0.15)):            # generic, and Gemma's scaling=1.0 (q/k normed)
        q = (torch.randn(B, H, S, D, device=DEV) * amp).bfloat16().requires_grad_()
        k = (torch.randn(B, H, S, D, device=DEV) * amp).bfloat16().requires_grad_()
        v = torch.randn(B, H, S, D, device=DEV).bfloat16().requires_grad_()
        g = torch.randn(B, H, S, D, device=DEV).bfloat16()
        out = flash_d512(q, k, v, True, position_ids=pos, scale=scale)
        if out.shape != q.shape:                                   # [B,H,S,D] expected
            check(f"flash_d512 output shape (scale {scale:g})", False, f"{tuple(out.shape)}")
            continue
        (out.float() * g.float()).sum().backward()
        dq, dk, dv = q.grad, k.grad, v.grad
        qr, kr, vr = (t.detach().float().requires_grad_() for t in (q, k, v))
        ref = ref_attention(qr, kr, vr, pos, scale)
        (ref * g.float()).sum().backward()
        errs = [rel(out, ref), rel(dq, qr.grad), rel(dk, kr.grad), rel(dv, vr.grad)]
        check(f"flash_d512 packed + one-doc rows, scale {scale:g}: out/dq/dk/dv vs fp32 masked attention",
              all(e < 3e-2 for e in errs), "rel err " + " ".join(f"{e:.1e}" for e in errs))
        # no leak across documents: changing doc 1 of row 0 must not change doc 0's output
        with torch.no_grad():
            k2, v2 = k.detach().clone(), v.detach().clone()
            k2[0, :, 300:800] += 1.0
            v2[0, :, 300:800] += 1.0
            out2 = flash_d512(q.detach(), k2, v2, True, position_ids=pos, scale=scale)
        leak = float((out2[0, :, :300].float() - out.detach()[0, :, :300].float()).abs().max())
        check(f"flash_d512 keeps documents apart (scale {scale:g})", leak < 1e-3, f"max change in doc 0: {leak:.1e}")


# ----------------------------------------------------------------------------- model helpers
def tiny_config(**over):
    from transformers import Gemma4TextConfig

    c = dict(vocab_size=512, hidden_size=256, intermediate_size=512, num_hidden_layers=6, num_attention_heads=4,
             num_key_value_heads=2, head_dim=64, global_head_dim=512, sliding_window=128,
             layer_types=["sliding_attention"] * 5 + ["full_attention"], hidden_size_per_layer_input=0,
             vocab_size_per_layer_input=512, max_position_embeddings=4096, final_logit_softcapping=30.0,
             initializer_range=0.05, pad_token_id=0, eos_token_id=1, bos_token_id=2)
    c.update(over)
    return Gemma4TextConfig(**c)


def build_model(attn):
    from transformers import Gemma4ForCausalLM

    torch.manual_seed(1)
    m = Gemma4ForCausalLM._from_config(tiny_config(), attn_implementation=attn, dtype=torch.bfloat16)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if p.dim() == 1:
                p.copy_(1 + 0.2 * torch.randn_like(p))         # norms away from 1, so they matter
    return m.to(DEV, torch.bfloat16).eval()


# ----------------------------------------------------------------------------- 2. fused RMSNorm + RoPE
def test_fused_rope():
    from transformers.models.gemma4 import modeling_gemma4 as mg

    model = build_model("sdpa")
    text = model.model
    orig = mg.Gemma4TextAttention.forward
    from micro_engram.train.vendor.gemma4_fused_attn import _make_fused_forward

    fused = _make_fused_forward(orig)
    torch.manual_seed(2)
    S = 256
    pos = torch.arange(S, device=DEV)[None]
    for li in (0, 5):                                             # a sliding layer, the global layer
        attn = text.layers[li].self_attn
        ltype = text.config.layer_types[li]
        h = (torch.randn(1, S, 256, device=DEV) * 0.5).bfloat16()
        pe = text.rotary_emb(h, pos, ltype)
        outs, grads = [], []
        for fwd in (orig, fused):
            x = h.clone().requires_grad_()
            o, _ = fwd(attn, x, pe, None, shared_kv_states={}, position_ids=pos)
            o.float().pow(2).sum().backward()
            outs.append(o.detach())
            grads.append(x.grad)
        e_out, e_grad = rel(outs[1], outs[0]), rel(grads[1], grads[0])
        check(f"fused q/k/v norm + RoPE == original attention ({ltype}, layer {li}): output and input grad",
              e_out < 2e-2 and e_grad < 3e-2, f"rel err {e_out:.1e} {e_grad:.1e}")


# ----------------------------------------------------------------------------- 3. end to end
def loss_and_grad(model, ids, pos, labels):
    emb = model.model.embed_tokens(ids).detach().requires_grad_()
    # use_cache=False as in training: with a KV cache the global layers keep the model's own mask
    logits = model(inputs_embeds=emb, position_ids=pos, use_cache=False).logits.float()
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1),
                                             ignore_index=-100)
    loss.backward()
    return float(loss), emb.grad.detach()


def packed_batch(lens_per_row, S, seed):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(3, 512, (len(lens_per_row), S), generator=g)
    pos = torch.cat([torch.cat([torch.arange(n) for n in row])[None] for row in lens_per_row])
    labels = torch.full_like(ids, -100)
    labels[:, :-1] = ids[:, 1:]
    labels[:, :-1][pos[:, 1:] == 0] = -100                         # never across documents
    return ids.to(DEV), pos.to(DEV), labels.to(DEV)


def test_end_to_end():
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        check("end to end (needs flash-attn for the sliding layers)", False, "flash_attn not installed")
        return
    S = 1024
    batches = {
        "packed rows (Triton kernel)": packed_batch([[300, 500, 224], [100, 924]], S, 3),
        "one-document rows (causal SDPA)": packed_batch([[1024], [1024]], S, 4),
    }
    ref_model = build_model("sdpa")                                # stock HF: builds the block masks itself
    refs = {k: loss_and_grad(ref_model, *b) for k, b in batches.items()}
    del ref_model

    # the trainer's setup: count kernel calls
    from micro_engram.train.vendor import flash_attn_d512 as fd, gemma4_fused_rope as fr

    calls = {"flash_d512": 0, "fused_rope": 0}

    def counting(name, fn):
        def wrapped(*a, **k):
            calls[name] += 1
            return fn(*a, **k)
        return wrapped

    fd.flash_d512 = counting("flash_d512", fd.flash_d512)
    fr.fused_rms_norm_rope = counting("fused_rope", fr.fused_rms_norm_rope)
    from micro_engram.train import model as M

    cfg = types.SimpleNamespace(fused_attn_kernel=True, gradient_checkpointing=False, activation_offloading=False,
                                gemma4_hybrid_attn_impl=True, large_head_policy="auto", sample_packing=True,
                                liger_rms_norm=False, liger_glu_activation=False, attn_implementation="flash_attention_2")
    M.apply_kernel_patches(cfg, True)
    test_model = build_model("flash_attention_2")
    M.hybrid_attention(test_model.model, cfg)
    for name, b in batches.items():
        before = calls["flash_d512"]
        loss, grad = loss_and_grad(test_model, *b)
        rloss, rgrad = refs[name]
        e = rel(grad, rgrad)
        check(f"end to end, {name}: loss and input gradient == stock HF SDPA",
              abs(loss - rloss) < 1e-2 * max(1.0, abs(rloss)) and e < 5e-2,
              f"loss {loss:.4f} vs {rloss:.4f}, grad rel err {e:.1e}")
        used = calls["flash_d512"] - before
        if "packed" in name:
            check("  ...the Triton head_dim-512 kernel ran on the packed rows", used > 0, f"{used} calls")
        else:
            check("  ...one-document rows took the causal SDPA path (no Triton call)", used == 0, f"{used} calls")
    check("fused RMSNorm + RoPE kernel ran", calls["fused_rope"] > 0, f"{calls['fused_rope']} calls")
    from micro_engram.train.vendor import large_head

    check("no 'Triton kernel failed' fallback happened", not large_head._WARNED)


if __name__ == "__main__":
    print(f"GPU: {torch.cuda.get_device_name()} | torch {torch.__version__}")
    for t in (test_flash_d512, test_fused_rope, test_end_to_end):
        try:
            t()
        except Exception as e:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            check(f"{t.__name__} crashed ({type(e).__name__}: {e})", False)
    print("\nALL PASSED" if not FAIL else f"\n{len(FAIL)} FAILED: {FAIL}")
    sys.exit(1 if FAIL else 0)
