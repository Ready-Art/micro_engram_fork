"""
How well does a model predict held-out domain text? Compare two vLLM configs (e.g. with
and without the engram) on the same texts. No dependencies beyond the standard library.

1. Score the texts with whichever config is currently served (one at a time is fine):
     python eval_logprob.py run --url http://localhost:8000 --data heldout/ --out engram.json
     (switch the server to the other config)
     python eval_logprob.py run --url http://localhost:8000 --data heldout/ --out base.json

2. Compare:
     python eval_logprob.py compare engram.json base.json

--data is a folder of .txt files, a single .txt file, or a .jsonl file with a "text"
field per line. Use text the models were NOT trained on. Long texts are cut into chunks
of about --max-chars characters at paragraph breaks, the same way on every run, so both
configs score exactly the same pieces.

Uses vLLM's /v1/completions with prompt_logprobs: the score is the log-probability the
model gives to each real token of the text (no generation, no sampling, deterministic).
"""
import argparse
import hashlib
import json
import math
import os
import random
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor


# ----------------------------------------------------------------------------- data
def read_texts(path):
    """Yield (source name, text) from a folder of .txt, a .txt file, or a .jsonl file."""
    if os.path.isdir(path):
        for name in sorted(os.listdir(path)):
            if name.endswith((".txt", ".jsonl")):
                yield from read_texts(os.path.join(path, name))
    elif path.endswith(".jsonl"):
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if line.strip():
                    yield f"{os.path.basename(path)}:{i}", json.loads(line)["text"]
    else:
        with open(path, encoding="utf-8") as f:
            yield os.path.basename(path), f.read()


def chunk(text, max_chars):
    """Cut at paragraph breaks into pieces of about max_chars (a single longer paragraph
    stays whole). Deterministic, so every run produces the same chunks."""
    out, cur = [], ""
    for para in text.split("\n\n"):
        if cur and len(cur) + len(para) + 2 > max_chars:
            out.append(cur)
            cur = para
        else:
            cur = f"{cur}\n\n{para}" if cur else para
    if cur.strip():
        out.append(cur)
    return [c for c in out if c.strip()]


def load_chunks(path, max_chars, min_chars):
    chunks = []
    for source, text in read_texts(path):
        for i, piece in enumerate(chunk(text.strip(), max_chars)):
            if len(piece) >= min_chars:
                chunks.append({"id": f"{source}#{i}", "text": piece})
    return chunks


# ----------------------------------------------------------------------------- scoring
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


def served_model(base_url, timeout):
    with _open(urllib.request.Request(f"{base_url}/v1/models", headers=AUTH), timeout) as r:
        return json.loads(r.read())["data"][0]["id"]


def score(base_url, model, piece, timeout):
    """Sum of log-probs of every real token after the first, plus the token ids."""
    resp = post(f"{base_url}/v1/completions", {
        "model": model,
        "prompt": piece["text"],
        "max_tokens": 1,
        "temperature": 0,
        "prompt_logprobs": 0,       # only the log-prob of the token that is actually there
        "return_token_ids": True,   # to check both configs scored the same tokens
    }, timeout)
    choice = resp["choices"][0]
    ids, plps = choice["prompt_token_ids"], choice["prompt_logprobs"]
    total, n = 0.0, 0
    for tok, entry in zip(ids[1:], plps[1:]):  # the first token has nothing to predict it
        total += entry[str(tok)]["logprob"]
        n += 1
    token_hash = hashlib.sha1(json.dumps(ids).encode()).hexdigest()[:16]
    return {"id": piece["id"], "sum_logprob": total, "n_tokens": n, "token_hash": token_hash}


def cmd_run(args):
    base = args.url.rstrip("/")
    model = args.model or served_model(base, args.timeout)
    chunks = load_chunks(args.data, args.max_chars, args.min_chars)
    if not chunks:
        sys.exit(f"no text found in {args.data}")
    print(f"scoring {len(chunks)} chunks from {args.data} with '{model}' at {base} ...")
    done = []
    with ThreadPoolExecutor(args.parallel) as pool:  # vLLM batches parallel requests
        for i, res in enumerate(pool.map(lambda c: score(base, model, c, args.timeout), chunks)):
            done.append(res)
            if (i + 1) % 20 == 0 or i + 1 == len(chunks):
                print(f"  {i + 1}/{len(chunks)}")
    n = sum(r["n_tokens"] for r in done)
    nll = -sum(r["sum_logprob"] for r in done) / n
    print(f"{model}: {n:,} tokens | loss {nll:.4f} | perplexity {math.exp(nll):.3f}")
    with open(args.out, "w") as f:
        json.dump({"model": model, "url": base, "data": args.data, "max_chars": args.max_chars,
                   "chunks": done}, f, indent=1)
    print(f"saved {args.out}")


# ----------------------------------------------------------------------------- compare
def cmd_compare(args):
    a, b = (json.load(open(p)) for p in (args.a, args.b))
    rb = {r["id"]: r for r in b["chunks"]}
    pairs = [(r, rb[r["id"]]) for r in a["chunks"] if r["id"] in rb]
    if not pairs:
        sys.exit("no chunks in common: were both runs made on the same --data and --max-chars?")
    mismatched = [ra["id"] for ra, rb_ in pairs if ra["token_hash"] != rb_["token_hash"]]
    if mismatched:
        print(f"WARNING: {len(mismatched)} chunks were tokenized differently (different "
              f"tokenizer or chat template?), e.g. {mismatched[:3]}. They are left out.")
        pairs = [(x, y) for x, y in pairs if x["token_hash"] == y["token_hash"]]

    def loss(rows):
        return -sum(r["sum_logprob"] for r in rows) / sum(r["n_tokens"] for r in rows)

    la, lb = loss([x for x, _ in pairs]), loss([y for _, y in pairs])
    n_tok = sum(x["n_tokens"] for x, _ in pairs)
    print(f"{len(pairs)} chunks, {n_tok:,} tokens\n")
    print(f"  A {a['model']:<32} loss {la:.4f}  perplexity {math.exp(la):.3f}   ({args.a})")
    print(f"  B {b['model']:<32} loss {lb:.4f}  perplexity {math.exp(lb):.3f}   ({args.b})")
    diff = lb - la  # positive: A predicts the text better
    print(f"\n  A - B: loss {-diff:+.4f} per token, perplexity {math.exp(la) / math.exp(lb) - 1:+.1%}")

    # per-chunk: how often is A better, and how sure are we (bootstrap over chunks)
    # d = loss of A minus loss of B on one chunk, as a sum over its tokens (negative: A better)
    per = [(y["sum_logprob"] - x["sum_logprob"], x["n_tokens"]) for x, y in pairs]
    wins = sum(1 for d, _ in per if d < 0)
    print(f"  A is better on {wins}/{len(per)} chunks ({wins / len(per):.0%})")
    rng = random.Random(0)
    boots = []
    for _ in range(2000):
        sample = [per[rng.randrange(len(per))] for _ in per]
        boots.append(sum(d for d, _ in sample) / sum(n for _, n in sample))
    boots.sort()
    lo, hi = boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots)) - 1]
    print(f"  95% interval for the loss difference A - B: [{lo:+.4f}, {hi:+.4f}]")
    if hi < 0:
        print("  -> A is reliably better on this text")
    elif lo > 0:
        print("  -> B is reliably better on this text")
    else:
        print("  -> no reliable difference (more text, or the effect is small)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="score the texts with the currently served model")
    r.add_argument("--url", default="http://localhost:8000")
    r.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY"), help="if vLLM runs with --api-key")
    r.add_argument("--model", default=None, help="served model name (default: ask the server)")
    r.add_argument("--data", required=True, help="folder of .txt, a .txt, or a .jsonl with 'text'")
    r.add_argument("--out", required=True, help="where to save the scores (.json)")
    r.add_argument("--max-chars", type=int, default=6000, help="chunk size, about 1.5k tokens")
    r.add_argument("--min-chars", type=int, default=200, help="skip shorter pieces")
    r.add_argument("--parallel", type=int, default=8, help="requests in flight at once")
    r.add_argument("--timeout", type=float, default=600)
    c = sub.add_parser("compare", help="compare two saved runs")
    c.add_argument("a", help="first run, e.g. engram.json")
    c.add_argument("b", help="second run, e.g. base.json")
    args = ap.parse_args()
    if getattr(args, "api_key", None):
        AUTH["Authorization"] = f"Bearer {args.api_key}"
    cmd_run(args) if args.cmd == "run" else cmd_compare(args)


if __name__ == "__main__":
    main()
