"""
GPU tests for the Triton kernels micro_engram.train uses (vendored from Axolotl): run on the server.

    python test_kernels_gpu.py            # needs CUDA, triton, flash-attn (for the hybrid test)

Each kernel is compared with plain PyTorch on random inputs, forward AND gradients:
  1. flash_d512 (head_dim 512, the Gemma 4 global layers): packed rows (several documents,
     varlen from position_ids) and one-document rows, against fp32 attention with the
     block-diagonal causal mask.
  2. fused q/k/v RMSNorm + RoPE (fused_attn_kernel): the patched Gemma4TextAttention.forward
     against the original one, on a sliding layer and a global layer (partial rotary).
  3. end to end, a tiny Gemma 4 with head_dim-512 global layers on packed and one-document
     batches: hybrid attention (FA2 + Triton/causal globals), then + fused norm/RoPE, against
     stock HF; and that the Triton kernel and the fused RoPE actually ran.

Tests 2 and 3 use an fp32 run of stock HF as the truth: a bf16 kernel passes if its error vs
fp32 is not clearly worse than plain bf16 HF's own error (bf16 HF is not exact either).
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
    a, b = a.detach().float(), b.detach().float()
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


def build_model(attn, dtype=torch.bfloat16):
    """Same weights every time (seeded, made in fp32, then cast), so fp32 and bf16 copies match."""
    from transformers import Gemma4ForCausalLM

    torch.manual_seed(1)
    m = Gemma4ForCausalLM._from_config(tiny_config(), attn_implementation=attn, dtype=torch.float32)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if p.dim() == 1:
                p.copy_(1 + 0.2 * torch.randn_like(p))         # norms away from 1, so they matter
                if n.endswith(("q_norm.weight", "k_norm.weight")):
                    # Gemma 4 uses scaling=1.0 on RMS-normed q/k: with norm weights ~1 the scores have
                    # std ~8 (head_dim 64) to ~22 (512), softmax is nearly argmax and bf16 vs fp32
                    # differs by ~130% in the input gradient. 0.3x (a trained-model-like temperature)
                    # brings that floor to ~4%, so the checks below can see a real kernel bug.
                    p.mul_(0.3)
    return m.to(DEV, dtype).eval()


def no_worse(err, floor, slack=1.5, eps=1e-3):
    """bf16 kernels can't be compared with bf16 HF as if HF were exact: both are compared with an
    fp32 run, and the kernel passes if its error is not clearly worse than plain bf16 HF's."""
    return err <= slack * floor + eps


# ----------------------------------------------------------------------------- 2. fused RMSNorm + RoPE
def test_fused_rope():
    from transformers.models.gemma4 import modeling_gemma4 as mg

    from micro_engram.train.vendor.gemma4_fused_attn import _make_fused_forward

    orig = mg.Gemma4TextAttention.forward
    fused = _make_fused_forward(orig)
    m16, m32 = build_model("sdpa"), build_model("sdpa", torch.float32)
    torch.manual_seed(2)
    S = 256
    pos = torch.arange(S, device=DEV)[None]
    for li in (0, 5):                                             # a sliding layer, the global layer
        ltype = m16.model.config.layer_types[li]
        h16 = (torch.randn(1, S, 256, device=DEV) * 0.5).bfloat16()

        def run(model, fwd, h):
            attn = model.model.layers[li].self_attn
            x = h.clone().requires_grad_()
            pe = model.model.rotary_emb(x, pos, ltype)
            o, _ = fwd(attn, x, pe, None, shared_kv_states={}, position_ids=pos)
            o.float().pow(2).sum().backward()
            return o.detach().float(), x.grad.detach().float()

        t_out, t_grad = run(m32, orig, h16.float())                # truth: fp32 original
        b_out, b_grad = run(m16, orig, h16)                        # bf16 original (the noise floor)
        f_out, f_grad = run(m16, fused, h16)                       # bf16 fused kernel
        fo, fg, bo, bg = rel(f_out, t_out), rel(f_grad, t_grad), rel(b_out, t_out), rel(b_grad, t_grad)
        check(f"fused q/k/v norm + RoPE ({ltype}, layer {li}): no worse than bf16 HF vs fp32",
              no_worse(fo, bo) and no_worse(fg, bg),
              f"vs fp32: fused out {fo:.1e} grad {fg:.1e} | bf16 HF out {bo:.1e} grad {bg:.1e}")


# ----------------------------------------------------------------------------- 3. end to end
def loss_and_grad(model, ids, pos, labels):
    emb = model.model.embed_tokens(ids).detach().requires_grad_()
    # use_cache=False as in training: with a KV cache the global layers keep the model's own mask
    logits = model(inputs_embeds=emb, position_ids=pos, use_cache=False).logits.float()
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1),
                                             ignore_index=-100)
    loss.backward()
    return float(loss), emb.grad.detach().float()


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
    # ONE row per batch, as micro_engram.train feeds the model (it flattens micro-batches): transformers'
    # FA2 only isolates packed documents for a batch of one row (_is_packed_sequence), see the last check
    batches = {
        "packed rows": packed_batch([[300, 500, 224, 100, 924]], 2048, 3),
        "one-document rows": packed_batch([[1024]], 1024, 4),
    }
    # truth: stock HF in fp32 (SDPA, block masks built by HF); floor: stock HF in bf16
    m32 = build_model("sdpa", torch.float32)
    truth = {k: loss_and_grad(m32, *b) for k, b in batches.items()}
    del m32
    m16 = build_model("sdpa")
    floor = {k: loss_and_grad(m16, *b) for k, b in batches.items()}
    del m16

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

    from micro_engram.train.vendor.large_head import set_large_head_policy

    def compare(label, model, expect_triton=True):
        for name, b in batches.items():
            before = calls["flash_d512"]
            loss, grad = loss_and_grad(model, *b)
            tl, tg = truth[name]
            fl, fg = floor[name]
            el, eg = abs(loss - tl), rel(grad, tg)
            fel, feg = abs(fl - tl), rel(fg, tg)
            check(f"{label}, {name}: no worse than bf16 HF vs fp32",
                  no_worse(el, fel, eps=2e-3) and no_worse(eg, feg, eps=5e-3),
                  f"vs fp32: loss diff {el:.1e} grad {eg:.1e} | bf16 HF loss diff {fel:.1e} grad {feg:.1e}")
            used = calls["flash_d512"] - before
            if name == "packed rows" and expect_triton:
                check("  ...the Triton head_dim-512 kernel ran on the packed rows", used > 0, f"{used} calls")
            elif name == "packed rows":
                check("  ...no Triton call (block mask instead)", used == 0, f"{used} calls")
            else:
                check("  ...one-document rows took the causal SDPA path (no Triton call)", used == 0, f"{used} calls")

    def setup(attn, policy):
        cfg = types.SimpleNamespace(fused_attn_kernel=False, gradient_checkpointing=False, activation_offloading=False,
                                    gemma4_hybrid_attn_impl=True, large_head_policy=policy, sample_packing=True,
                                    liger_rms_norm=False, liger_glu_activation=False, attn_implementation=attn)
        M.apply_kernel_patches(cfg, True)                          # idempotent; sets the policy
        set_large_head_policy(policy)
        model = build_model(attn)
        M.hybrid_attention(model.model, cfg)
        return model

    # One piece at a time, so a failure points at it (patches are process-wide, order matters):
    # a) all SDPA (sliding layers get HF's packed sliding masks), global layers on the Triton kernel
    compare("a) SDPA sliding + Triton globals", setup("sdpa", "auto"))
    # b) FA2 varlen sliding layers, global layers on the block mask (no Triton)
    compare("b) FA2 sliding + block-mask globals", setup("flash_attention_2", "sdpa"), expect_triton=False)
    # c) FA2 sliding + Triton globals (the trainer's attention setup)
    model = setup("flash_attention_2", "auto")
    compare("c) FA2 sliding + Triton globals", model)
    from micro_engram.train.vendor.gemma4_fused_attn import patch_gemma4_fused_attn

    patch_gemma4_fused_attn(install_shared_kv_workaround=False)  # d) + fused norm/RoPE (class-level)
    compare("d) c + fused norm/RoPE (the trainer's full setup)", model)

    # e) documents the transformers limit the trainer works around: the same documents as 2 rows
    two_rows = packed_batch([[300, 500, 224], [100, 924]], 1024, 3)
    t_loss, _ = loss_and_grad(build_model("sdpa", torch.float32), *two_rows)
    f_loss, _ = loss_and_grad(model, *two_rows)
    print(f"note: FA2 with 2 packed rows per batch: loss {f_loss:.4f} vs fp32 {t_loss:.4f} "
          f"(diff {abs(f_loss - t_loss):.1e}; large = documents leak across in FA2's sliding layers, "
          f"which is why micro_engram.train flattens micro-batches into one row)")
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
