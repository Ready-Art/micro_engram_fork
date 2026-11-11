"""
eval_logprob.py assistant-only scoring against a fake vLLM server (standard library only):
    python test_eval_spans.py

The fake model gives assistant words probability 0.9 and everything else 0.1. With spans
(render_chat.py output) the loss must be exactly -log(0.9): only assistant tokens count,
the user turns and role headers are context only. Without spans every token counts.
"""
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
results = []
VOCAB = {}


def check(name, ok):
    results.append((name, bool(ok)))


def tok(text):
    """Markers, words and whitespace are tokens; stable ids."""
    parts = re.findall(r"<bos>|<end>|<user>|<assistant>|\s|[^\s<]+", text)
    return [VOCAB.setdefault(t, len(VOCAB) + 10) for t in parts]


INV = lambda i: next(k for k, v in VOCAB.items() if v == i)  # noqa: E731


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send({"data": [{"id": "fake"}]})

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        ids = tok(req["prompt"])
        if self.path == "/tokenize":
            self._send({"tokens": ids, "count": len(ids)})
            return
        plps = [None]
        for i in ids[1:]:
            w = INV(i)
            p = 0.9 if (w.startswith("aa") or w == "<end>") else 0.1
            plps.append({str(i): {"logprob": math.log(p), "rank": 1}})
        self._send({"choices": [{"text": "", "prompt_token_ids": ids, "prompt_logprobs": plps}]})


srv = HTTPServer(("127.0.0.1", 0), H)
threading.Thread(target=srv.serve_forever, daemon=True).start()
url = f"http://127.0.0.1:{srv.server_address[1]}"

tmp = tempfile.mkdtemp(prefix="eval_spans_")
rows = []
for k in range(4):
    text = "<bos>"
    spans = []
    for t in range(3):
        text += f"<user>uu{k} said something here\n"
        text += "<assistant>"
        a = len(text)
        text += f"aa{k} aa{t} aaa<end>"
        spans.append([a, len(text)])
        text += "\n"
    rows.append({"text": text, "spans": spans})
with_spans = os.path.join(tmp, "spans.jsonl")
with open(with_spans, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
no_spans = os.path.join(tmp, "nospans.jsonl")
with open(no_spans, "w") as f:
    for r in rows:
        f.write(json.dumps({"text": r["text"]}) + "\n")

script = os.path.join(HERE, "eval_logprob.py")


def run(data, out):
    r = subprocess.run([sys.executable, script, "run", "--url", url, "--data", data, "--out", out,
                        "--parallel", "1", "--min-chars", "1"], capture_output=True, text=True)
    res = json.load(open(out)) if os.path.isfile(out) else None
    return r, res


r1, a = run(with_spans, os.path.join(tmp, "a.json"))
if a:
    n = sum(c["n_tokens"] for c in a["chunks"])
    loss = -sum(c["sum_logprob"] for c in a["chunks"]) / n
    # per assistant turn: "aaK", " ", "aaT", " ", "aaa", "<end>" -> the spaces are 0.1 tokens inside the span
    per_turn = [math.log(0.9), math.log(0.1), math.log(0.9), math.log(0.1), math.log(0.9), math.log(0.9)]
    want = -sum(per_turn) / len(per_turn)
    check(f"with spans: only assistant tokens scored (loss {loss:.4f}, expected {want:.4f}, {n} tokens)",
          abs(loss - want) < 1e-9 and n == 4 * 3 * 6)
else:
    check(f"with spans run ({r1.stderr[-300:]})", False)
r2, b = run(no_spans, os.path.join(tmp, "b.json"))
if b:
    nb = sum(c["n_tokens"] for c in b["chunks"])
    check("without spans: every token after the first is scored (more tokens, user turns included)", nb > 4 * 3 * 6)
else:
    check(f"without spans run ({r2.stderr[-300:]})", False)

srv.shutdown()
bad = 0
for name, ok in results:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    bad += not ok
sys.exit(1 if bad else 0)
