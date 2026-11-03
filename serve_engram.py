"""
OpenAI-compatible server for a Gemma 4 model trained with micro-engram.

Usage:
  python serve_engram.py --model /path/to/merged_model --port 8000
  python serve_engram.py --model /path/to/base_model --engram-pt /path/to/micro_engram.pt

Then point your frontend at  http://<this-machine>:8000/v1
Models it offers:
  <name>             engram ON
  <name>-noengram    engram OFF  (same weights, for A/B comparison)

Notes:
  * One request at a time (the engram keeps a token history, so requests are locked).
  * Every request re-reads the full prompt, which is exactly what the engram needs.
  * No API key / auth. Default host is localhost only; use --host 0.0.0.0 for your LAN.
  * For a fair A/B, use the same seed (or temperature 0) in your frontend.
  * Thinking is ON by default, as in training (enable_thinking + preserve_thinking).
    Start with --no-thinking to flip the default, or send per request:
      "chat_template_kwargs": {"enable_thinking": false}
  * Chat replies put the thinking in `reasoning_content` (like vLLM's reasoning parser),
    so frontends show it as a thinking block; `content` is only the answer.

Put this file next to infer_engram.py and the micro_engram package folder.
"""
import argparse
import json
import os
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, StoppingCriteria,
                          StoppingCriteriaList, TextIteratorStreamer)

from micro_engram.module import MicroEngram
from infer_engram import EngramRunner, load_engram_weights, pick, resolve_settings

S = SimpleNamespace()          # model, tok, runner, name, lock
LOCK = threading.Lock()


# ----------------------------------------------------------------------------- loading
def load_everything(args):
    sd, layer_from_key = load_engram_weights(args.model, args.engram_pt)
    if not sd:
        raise SystemExit("No micro_engram weights found in the checkpoint. Wrong path, or they were not saved.")
    settings = resolve_settings(args, sd, layer_from_key)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="auto")
    model.eval()

    text_cfg = getattr(model.config, "text_config", model.config)
    if settings["model_dim"] and settings["model_dim"] != text_cfg.hidden_size:
        raise SystemExit(f"Config says model_dim {settings['model_dim']} but the model has "
                         f"hidden_size {text_cfg.hidden_size}. Wrong config for this model?")

    canon = None
    if "canon" not in sd and settings["token_merge"]:
        path = settings["canon_path"]
        if not path or not os.path.isfile(path):
            raise SystemExit(f"Training used token merging but the weights have no 'canon' tensor and "
                             f"the canon file is not available ({path}).")
        canon = torch.load(path, map_location="cpu")
        print(f"[engram] token-merge map loaded from {path}")

    engram = MicroEngram(
        model_dim=text_cfg.hidden_size,
        table_size=settings["table_size"],
        engram_dim=settings["engram_dim"],
        num_heads=settings["num_heads"],
        orders=settings["orders"],
        pad_id=pick(settings["pad_id"], text_cfg.pad_token_id, 0),
        vocab_size=text_cfg.vocab_size,
        canon=canon,
        gate=settings["gate"],
    )
    runner = EngramRunner(model, engram, settings["layer"])
    missing, unexpected = engram.load_state_dict(sd, strict=False)
    print(f"[engram] loaded {len(sd)} tensors | missing: {missing} | unexpected: {unexpected}")
    out_layer = engram.value if engram.gate == "scalar" else engram.up
    print(f"[engram] output-weight magnitude ({engram.gate} gate): "
          f"{out_layer.weight.float().abs().mean().item():.6f} "
          f"(0.000000 would mean it never trained or did not load)")

    S.model, S.tok, S.runner = model, tok, runner


# ----------------------------------------------------------------------------- prompts
def flat_content(c):
    if c is None:
        return ""
    if isinstance(c, str):
        return c
    if isinstance(c, list):  # OpenAI "content parts"
        return "".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text")
    return str(c)


def encode_chat(messages, template_kwargs):
    msgs = []
    for m in messages:
        msg = {"role": m.get("role", "user"), "content": flat_content(m.get("content"))}
        reasoning = m.get("reasoning_content") or m.get("reasoning")
        if reasoning:  # earlier thinking sent back by the frontend (used with preserve_thinking)
            msg["reasoning_content"] = reasoning
        msgs.append(msg)
    text = S.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, **template_kwargs)
    # the template already contains <bos>, so don't add a second one
    return S.tok(text, add_special_tokens=False, return_tensors="pt").input_ids


def encode_text(prompt):
    bos = S.tok.bos_token or ""
    add = not (bos and prompt.startswith(bos))
    return S.tok(prompt, add_special_tokens=add, return_tensors="pt").input_ids


# ----------------------------------------------------------------------------- generation
class CancelFlag(StoppingCriteria):
    def __init__(self):
        self.cancelled = False

    def __call__(self, input_ids, scores, **kwargs):
        return torch.full((input_ids.shape[0],), self.cancelled, dtype=torch.bool, device=input_ids.device)


def raw_pieces(body, input_ids, engram_on, cancel, max_new, skip_special=True):
    """Yield decoded text pieces from model.generate running in a thread. Caller holds LOCK."""
    S.runner.enabled = engram_on
    S.runner.reset()

    temp = body.get("temperature")
    temp = 0.8 if temp is None else float(temp)
    kw = dict(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        max_new_tokens=max_new,
        do_sample=temp > 0,
        use_cache=True,
        stopping_criteria=StoppingCriteriaList([cancel]),
    )
    if temp > 0:
        kw["temperature"] = temp
        kw["top_p"] = float(body.get("top_p") if body.get("top_p") is not None else 1.0)
        if body.get("top_k"):
            kw["top_k"] = int(body["top_k"])
        if body.get("min_p"):
            kw["min_p"] = float(body["min_p"])
    if body.get("repetition_penalty"):
        kw["repetition_penalty"] = float(body["repetition_penalty"])
    if body.get("seed") is not None:
        torch.manual_seed(int(body["seed"]))

    streamer = TextIteratorStreamer(S.tok, skip_prompt=True, skip_special_tokens=skip_special)
    kw["streamer"] = streamer
    errors = []

    def work():
        try:
            with torch.no_grad():  # grad mode is per-thread, so set it here
                S.model.generate(**kw)
        except Exception as e:  # make sure the consumer never hangs
            errors.append(e)
            streamer.on_finalized_text("", stream_end=True)

    th = threading.Thread(target=work, daemon=True)
    th.start()
    try:
        for piece in streamer:
            yield piece
    finally:
        cancel.cancelled = True
        th.join()
    if errors:
        raise errors[0]


THINK_START, THINK_END = "<|channel>thought", "<channel|>"


def split_reasoning(pieces, specials):
    """Turn raw text (special tokens kept) into ("reasoning" | "content", text) pairs.

    Text between <|channel>thought and <channel|> is reasoning, the rest is content.
    Other special tokens (<turn|>, <eos>, ...) are dropped. Text that might be the start
    of a marker is held back until it is complete.
    """
    specials = [s for s in specials if s and s not in (THINK_START, THINK_END, "<|channel>")]
    hold = max([len(THINK_START), len(THINK_END)] + [len(s) for s in specials]) - 1
    buf, mode, strip_nl = "", "content", False

    def clean(text):
        for s in specials:
            text = text.replace(s, "")
        return text

    try:
        for p in pieces:
            buf += p
            while True:
                if strip_nl and buf:
                    buf, strip_nl = buf.removeprefix("\n"), False
                marker = THINK_START if mode == "content" else THINK_END
                i = buf.find(marker)
                if i >= 0:
                    head = clean(buf[:i])
                    if head:
                        yield mode, head
                    buf = buf[i + len(marker):]
                    mode = "reasoning" if mode == "content" else "content"
                    strip_nl = True  # the template puts a newline after both markers
                    continue
                buf = clean(buf)
                if len(buf) > hold:
                    yield mode, buf[:len(buf) - hold]
                    buf = buf[len(buf) - hold:]
                break
        buf = clean(buf)
        if buf:
            yield mode, buf
    finally:
        close = getattr(pieces, "close", None)
        if close:
            close()


def with_stops(pieces, stops, cancel):
    """Cut the stream at the first stop string in the answer (reasoning is never cut).

    Takes and yields ("reasoning" | "content", text) pairs. Holds back answer text that
    might be the start of a stop string.
    """
    if not stops:
        yield from pieces
        return
    hold = max(len(s) for s in stops) - 1
    buf = ""
    try:
        for kind, p in pieces:
            if kind != "content":
                yield kind, p
                continue
            buf += p
            hits = [buf.find(s) for s in stops if s in buf]
            if hits:
                yield "content", buf[:min(hits)]
                cancel.cancelled = True
                return
            if len(buf) > hold:
                yield "content", buf[:len(buf) - hold]
                buf = buf[len(buf) - hold:]
        yield "content", buf
    finally:
        close = getattr(pieces, "close", None)
        if close:
            close()


# ----------------------------------------------------------------------------- HTTP
def parse_stops(body):
    stop = body.get("stop")
    if not stop:
        return []
    return [stop] if isinstance(stop, str) else [s for s in stop if s]


class Handler(BaseHTTPRequestHandler):
    server_version = "engram-server"

    def log_message(self, fmt, *a):  # quiet the default per-request log
        pass

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

    def _json(self, obj, code=200):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self._cors()
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        if self.path.rstrip("/") in ("/v1/models", "/models"):
            now = int(time.time())
            self._json({"object": "list", "data": [
                {"id": S.name, "object": "model", "created": now, "owned_by": "local"},
                {"id": S.name + "-noengram", "object": "model", "created": now, "owned_by": "local"},
            ]})
        elif self.path.rstrip("/") in ("/health", ""):
            self._json({"status": "ok"})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        path = self.path.rstrip("/")
        if path not in ("/v1/chat/completions", "/v1/completions"):
            return self._json({"error": "not found"}, 404)
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            chat = path.endswith("chat/completions")
            if chat:
                # server defaults (thinking on, as in training), overridable per request like vLLM
                tmpl = dict(S.template_kwargs)
                tmpl.update(body.get("chat_template_kwargs") or {})
                input_ids = encode_chat(body.get("messages", []), tmpl)
            else:
                prompt = body.get("prompt", "")
                input_ids = encode_text(prompt if isinstance(prompt, str) else "".join(prompt))
        except Exception as e:
            return self._json({"error": f"bad request: {e}"}, 400)

        model_name = body.get("model") or S.name
        engram_on = not model_name.endswith("-noengram")
        max_new = int(body.get("max_tokens") or body.get("max_completion_tokens") or 512)
        stops = parse_stops(body)
        stream = bool(body.get("stream"))
        input_ids = input_ids.to("cuda:0")
        rid = ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex[:24]
        created = int(time.time())

        def chunk(delta_text, finish=None, first=False, kind="content"):
            if chat:
                delta = {}
                if first:
                    delta["role"] = "assistant"
                if delta_text:
                    delta["reasoning_content" if kind == "reasoning" else "content"] = delta_text
                return {"id": rid, "object": "chat.completion.chunk", "created": created,
                        "model": model_name,
                        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
            return {"id": rid, "object": "text_completion", "created": created, "model": model_name,
                    "choices": [{"index": 0, "text": delta_text, "finish_reason": finish}]}

        t0 = time.time()
        text_out, reasoning_out = "", ""
        with LOCK:
            cancel = CancelFlag()
            if chat:  # keep special tokens so the thinking channel can be split off
                raw = raw_pieces(body, input_ids, engram_on, cancel, max_new, skip_special=False)
                tagged = split_reasoning(raw, S.tok.all_special_tokens + list(S.tok.get_added_vocab()))
            else:  # plain completions: raw text, as before
                raw = raw_pieces(body, input_ids, engram_on, cancel, max_new)
                tagged = (("content", p) for p in raw)
            gen = with_stops(tagged, stops, cancel)
            try:
                if stream:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self._cors()
                    self.end_headers()
                    if chat:
                        self.wfile.write(f"data: {json.dumps(chunk('', first=True))}\n\n".encode())
                    for kind, piece in gen:
                        if not piece:
                            continue
                        if kind == "reasoning":
                            reasoning_out += piece
                        else:
                            text_out += piece
                        self.wfile.write(f"data: {json.dumps(chunk(piece, kind=kind))}\n\n".encode())
                        self.wfile.flush()
                else:
                    for kind, piece in gen:
                        if kind == "reasoning":
                            reasoning_out += piece
                        else:
                            text_out += piece
            except (BrokenPipeError, ConnectionResetError):
                cancel.cancelled = True  # client hit stop
                print("[server] client disconnected, generation cancelled")
            except Exception as e:
                cancel.cancelled = True
                print(f"[server] generation error: {e!r}")
                if not stream:
                    return self._json({"error": str(e)}, 500)
            finally:
                gen.close()

        all_out = reasoning_out + text_out
        n_new = len(S.tok(all_out, add_special_tokens=False).input_ids) if all_out else 0
        finish = "length" if n_new >= max_new else "stop"
        dt = max(time.time() - t0, 1e-6)
        print(f"[server] {'ENGRAM' if engram_on else 'no-engram'} | prompt {input_ids.shape[1]} tok | "
              f"~{n_new} new tok | {n_new / dt:.1f} tok/s (incl. prompt) | {finish}")

        if stream:
            try:
                self.wfile.write(f"data: {json.dumps(chunk('', finish=finish))}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            return

        usage = {"prompt_tokens": int(input_ids.shape[1]), "completion_tokens": n_new,
                 "total_tokens": int(input_ids.shape[1]) + n_new}
        if chat:
            self._json({"id": rid, "object": "chat.completion", "created": created, "model": model_name,
                        "choices": [{"index": 0, "message": {"role": "assistant", "content": text_out,
                                                             "reasoning_content": reasoning_out or None},
                                     "finish_reason": finish}], "usage": usage})
        else:
            self._json({"id": rid, "object": "text_completion", "created": created, "model": model_name,
                        "choices": [{"index": 0, "text": text_out, "finish_reason": finish}],
                        "usage": usage})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="merged model dir (or base model dir if using --engram-pt)")
    ap.add_argument("--engram-pt", default=None, help="optional separate micro_engram.pt file")
    ap.add_argument("--engram-config", default=None, help="path to micro_engram_config.json")
    ap.add_argument("--name", default=None, help="model name shown to the frontend")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    # optional overrides; normally found automatically
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--table-size", type=int, default=None)
    ap.add_argument("--dim", type=int, default=None)
    ap.add_argument("--heads", type=int, default=None)
    ap.add_argument("--gate", choices=MicroEngram.GATES, default=None)
    ap.add_argument("--no-thinking", action="store_true",
                    help="default to enable_thinking=false (requests can still override via chat_template_kwargs)")
    args = ap.parse_args()

    # Same defaults as training (chat_template_kwargs in the Axolotl YAML).
    S.template_kwargs = {"enable_thinking": not args.no_thinking, "preserve_thinking": True}
    S.name = args.name or os.path.basename(os.path.normpath(args.model))
    load_everything(args)

    print(f"\n[server] ready: http://{args.host}:{args.port}/v1")
    print(f"[server] models: '{S.name}' (engram on)  |  '{S.name}-noengram' (engram off)")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()