"""
Where does the engram push? Score one reply token by token with the engram on and off,
then color every token by how much more likely the engram made it. Standard library only.

1. Put the conversation in a file and the reply to score in another:
     chat.json   OpenAI-style messages, e.g. [{"role": "system", "content": "..."},
                 {"role": "user", "content": "..."}], everything BEFORE the reply
                 (or --context raw.txt: the exact prompt text, chat template already applied)
     reply.txt   the reply text to score (e.g. the engram-on output)

2. Score it with whichever config is served (same server flags otherwise):
     python engram_token_diff.py run --url http://localhost:8000 --messages chat.json --reply reply.txt --out on.json
     (restart vLLM without --additional-config)
     python engram_token_diff.py run --url http://localhost:8000 --messages chat.json --reply reply.txt --out off.json

3. Compare:
     python engram_token_diff.py render on.json off.json --out diff.html

Green tokens: the engram made them more likely; red: less likely; white: no change.
Hover a token for both probabilities. The terminal shows the biggest pushes both ways.

The score is the log-probability the model gives to each real token of the reply, given
the conversation and the reply so far (vLLM prompt_logprobs; no sampling, deterministic).
The engram changes probabilities at every token; this shows where the change mattered.
"""
import argparse
import html
import json
import math
import os
import sys
import urllib.error
import urllib.request


AUTH = {}  # {"Authorization": "Bearer <key>"} when the server runs with --api-key


def _open(req, timeout):
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        if e.code == 401:
            sys.exit("the server wants an API key: pass --api-key (or set VLLM_API_KEY)")
        raise
    except urllib.error.URLError as e:
        url = req.full_url if isinstance(req, urllib.request.Request) else req
        sys.exit(f"no server answered at {url} ({e.reason}): is vLLM up (\"Application startup complete\"), "
                 f"on this host and port, and started with --host 0.0.0.0 if you use another machine's address?")


def post(url, payload, timeout):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json", **AUTH})
    with _open(req, timeout) as r:
        return json.loads(r.read())


def served_model(base, timeout):
    with _open(urllib.request.Request(f"{base}/v1/models", headers=AUTH), timeout) as r:
        return json.loads(r.read())["data"][0]["id"]


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def context_ids(base, model, args):
    if args.messages:
        msgs = json.loads(read(args.messages))
        if not isinstance(msgs, list) or not all(isinstance(m, dict) and "role" in m for m in msgs):
            sys.exit(f"{args.messages}: expected a JSON list of {{'role': ..., 'content': ...}}")
        if msgs and msgs[-1]["role"] == "assistant":
            sys.exit(f"{args.messages}: the last message is the assistant's; leave the reply out "
                     f"(it goes in --reply)")
        r = post(f"{base}/tokenize", {"model": model, "messages": msgs, "add_generation_prompt": True}, args.timeout)
    else:
        r = post(f"{base}/tokenize", {"model": model, "prompt": read(args.context), "add_special_tokens": True},
                 args.timeout)
    return r["tokens"]


def cmd_run(args):
    base = args.url.rstrip("/")
    model = args.model or served_model(base, args.timeout)
    ctx = context_ids(base, model, args)
    reply_text = read(args.reply)
    if args.strip:
        reply_text = reply_text.strip()
    reply = post(f"{base}/tokenize", {"model": model, "prompt": reply_text, "add_special_tokens": False},
                 args.timeout)["tokens"]
    if not reply:
        sys.exit(f"{args.reply} is empty")
    resp = post(f"{base}/v1/completions", {
        "model": model,
        "prompt": ctx + reply,       # token ids: both runs score exactly the same tokens
        "max_tokens": 1,
        "temperature": 0,
        "prompt_logprobs": 0,        # the log-prob of the token that is actually there
    }, args.timeout)
    plps = resp["choices"][0]["prompt_logprobs"]
    tokens = []
    for pos, tok in enumerate(reply, start=len(ctx)):
        entry = plps[pos][str(tok)]
        tokens.append({"id": tok, "text": entry.get("decoded_token") or f"<{tok}>", "logprob": entry["logprob"]})
    total = sum(t["logprob"] for t in tokens)
    print(f"{model}: {len(tokens)} reply tokens after {len(ctx)} context tokens | "
          f"mean log-prob {total / len(tokens):.4f} | perplexity {math.exp(-total / len(tokens)):.3f}")
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"model": model, "url": base, "context_tokens": len(ctx), "context_ids": ctx,
                   "tokens": tokens}, f, ensure_ascii=False, indent=1)
    print(f"saved {args.out}")


# ----------------------------------------------------------------------------- render
def _color(d, scale):
    """Green for d > 0, red for d < 0, white at 0; full color at |d| = scale."""
    a = min(abs(d) / scale, 1.0)
    return (f"rgba(22, 163, 74, {a:.3f})" if d > 0 else f"rgba(220, 38, 38, {a:.3f})") if a > 0.02 else "transparent"


def _show(text):
    return text.replace("\n", "↵\n").replace(" ", "·") if text.strip() == "" else text


def cmd_render(args):
    on, off = (json.load(open(p, encoding="utf-8")) for p in (args.on, args.off))
    if on["context_ids"] != off["context_ids"] or [t["id"] for t in on["tokens"]] != [t["id"] for t in off["tokens"]]:
        sys.exit("the two runs scored different tokens (different text, tokenizer or chat template)")
    rows = []
    for a, b in zip(on["tokens"], off["tokens"]):
        rows.append({"text": a["text"], "on": a["logprob"], "off": b["logprob"], "d": a["logprob"] - b["logprob"]})
    n = len(rows)
    lo_on, lo_off = -sum(r["on"] for r in rows) / n, -sum(r["off"] for r in rows) / n
    print(f"{n} tokens | loss with engram {lo_on:.4f}, without {lo_off:.4f} "
          f"(perplexity {math.exp(lo_on):.3f} vs {math.exp(lo_off):.3f})")
    helped = sum(1 for r in rows if r["d"] > 0.05)
    hurt = sum(1 for r in rows if r["d"] < -0.05)
    print(f"tokens the engram made clearly more likely: {helped} ({helped / n:.0%}), less likely: {hurt} ({hurt / n:.0%})")

    def ctx_of(i):
        left = "".join(r["text"] for r in rows[max(0, i - 6):i])
        return (left[-40:] + "[" + rows[i]["text"] + "]").replace("\n", " ")

    order = sorted(range(n), key=lambda i: rows[i]["d"])
    k = min(args.top, n)
    print(f"\nbiggest pushes TOWARD the text (engram on - off, in log-prob; prob off -> on):")
    for i in reversed(order[-k:]):
        r = rows[i]
        print(f"  {r['d']:+.2f}  {math.exp(r['off']):6.1%} -> {math.exp(r['on']):6.1%}   {ctx_of(i)}")
    print(f"\nbiggest pushes AWAY from the text:")
    for i in order[:k]:
        r = rows[i]
        print(f"  {r['d']:+.2f}  {math.exp(r['off']):6.1%} -> {math.exp(r['on']):6.1%}   {ctx_of(i)}")

    scale = args.scale
    spans = []
    for r in rows:
        tip = (f"with engram {math.exp(r['on']):.1%} | without {math.exp(r['off']):.1%} | "
               f"log-prob {r['d']:+.2f}")
        spans.append(f'<span style="background:{_color(r["d"], scale)}" title="{html.escape(tip)}">'
                     f'{html.escape(_show(r["text"]))}</span>')
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Engram token diff</title>
<style>
:root {{ --bg: #ffffff; --fg: #1f2328; --muted: #59636e; --card: #f6f8fa; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg: #0d1117; --fg: #e6edf3; --muted: #9198a1; --card: #161b22; }} }}
body {{ background: var(--bg); color: var(--fg); font: 16px/1.7 system-ui, sans-serif; margin: 0; padding: 24px 16px; }}
main {{ max-width: 860px; margin: 0 auto; }}
h1 {{ font-size: 1.3rem; margin: 0 0 4px; }}
p.meta {{ color: var(--muted); margin: 0 0 16px; font-size: 0.9rem; }}
.text {{ background: var(--card); border-radius: 8px; padding: 16px; white-space: pre-wrap; word-wrap: break-word; }}
.text span {{ border-radius: 3px; }}
.legend span {{ display: inline-block; padding: 0 8px; border-radius: 3px; margin-right: 6px; font-size: 0.85rem; }}
</style></head><body><main>
<h1>Where the engram pushes</h1>
<p class="meta">{n} tokens &middot; loss with engram {lo_on:.4f}, without {lo_off:.4f} &middot;
more likely with the engram: {helped} tokens, less likely: {hurt} &middot; full color at &plusmn;{scale:g} log-prob &middot; hover a token for its probabilities</p>
<p class="legend"><span style="background:{_color(scale, scale)}">engram made it more likely</span><span style="background:{_color(-scale, scale)}">less likely</span></p>
<div class="text">{"".join(spans)}</div>
</main></body></html>
"""
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"\nwrote {args.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="score the reply with the config being served")
    r.add_argument("--url", default="http://localhost:8000")
    r.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY"), help="if vLLM runs with --api-key")
    r.add_argument("--model", default=None, help="served model name (default: ask the server)")
    src = r.add_mutually_exclusive_group(required=True)
    src.add_argument("--messages", help="JSON list of chat messages before the reply (chat template applied)")
    src.add_argument("--context", help="raw prompt text, chat template already applied")
    r.add_argument("--reply", required=True, help="text file with the reply to score")
    r.add_argument("--strip", action="store_true", help="strip whitespace around the reply")
    r.add_argument("--out", required=True)
    r.add_argument("--timeout", type=float, default=600)
    c = sub.add_parser("render", help="compare two runs and write the HTML heat map")
    c.add_argument("on", help="run with the engram")
    c.add_argument("off", help="run without it")
    c.add_argument("--out", default="engram_diff.html")
    c.add_argument("--top", type=int, default=15, help="tokens listed per direction")
    c.add_argument("--scale", type=float, default=2.0, help="log-prob difference shown at full color")
    args = ap.parse_args()
    if getattr(args, "api_key", None):
        AUTH["Authorization"] = f"Bearer {args.api_key}"
    {"run": cmd_run, "render": cmd_render}[args.cmd](args)


if __name__ == "__main__":
    main()
