"""
engram_token_diff.py against a fake vLLM server (no GPU, standard library):  python test_token_diff.py
"""
import json
import math
import os
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
results = []


def check(name, ok):
    results.append((name, bool(ok)))


VOCAB = {}


def tok(text):
    """Words and single spaces/newlines as tokens; ids stable across both servers."""
    out, cur = [], ""
    for ch in text:
        if ch in " \n":
            if cur:
                out.append(cur)
            out.append(ch)
            cur = ""
        else:
            cur += ch
    if cur:
        out.append(cur)
    return [VOCAB.setdefault(t, len(VOCAB) + 10) for t in out]


INV = lambda i: next(k for k, v in VOCAB.items() if v == i)  # noqa: E731
BOOST = {"snout", "Mountain"}  # the fake "engram" makes these likelier


def make_handler(engram):
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
            self._send({"data": [{"id": "fake-gemma"}]})

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/tokenize":
                if "messages" in req:
                    text = "".join(f"<{m['role']}>{m['content']}\n" for m in req["messages"]) + "<assistant>"
                else:
                    text = req["prompt"]
                ids = ([2] if req.get("add_special_tokens", True) and "messages" not in req else []) + tok(text)
                self._send({"tokens": ids, "count": len(ids)})
            else:
                ids = req["prompt"]
                plps = [None]
                for i in ids[1:]:
                    word = INV(i) if i in VOCAB.values() else "<bos>"
                    lp = math.log(0.2)
                    if engram and word.strip(".?") in BOOST:
                        lp = math.log(0.9)
                    plps.append({str(i): {"logprob": lp, "rank": 1, "decoded_token": word}})
                self._send({"choices": [{"text": "", "prompt_logprobs": plps}]})
    return H


def serve(engram):
    srv = HTTPServer(("127.0.0.1", 0), make_handler(engram))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


tmp = tempfile.mkdtemp(prefix="token_diff_")
chat = os.path.join(tmp, "chat.json")
reply = os.path.join(tmp, "reply.txt")
json.dump([{"role": "user", "content": "this thing? I am princess celeste"}], open(chat, "w"))
open(reply, "w").write("She nudges Celeste's neck with her snout.\nWhat brings you to Jade Mountain?")

script = os.path.join(HERE, "engram_token_diff.py")
outs = {}
for name, engram in (("on", True), ("off", False)):
    srv, url = serve(engram)
    outs[name] = os.path.join(tmp, f"{name}.json")
    r = subprocess.run([sys.executable, script, "run", "--url", url, "--messages", chat, "--reply", reply,
                        "--out", outs[name]], capture_output=True, text=True)
    srv.shutdown()
    check(f"run ({name}) succeeds", r.returncode == 0 and os.path.isfile(outs[name]))
    if r.returncode:
        print(r.stderr)

on = json.load(open(outs["on"]))
check("only the reply is scored (context tokens left out), decoded tokens kept",
      on["tokens"][0]["text"] == "She" and "".join(t["text"] for t in on["tokens"]).endswith("Mountain?")
      and on["context_tokens"] > 0)
html_out = os.path.join(tmp, "diff.html")
r = subprocess.run([sys.executable, script, "render", outs["on"], outs["off"], "--out", html_out],
                   capture_output=True, text=True)
check("render succeeds", r.returncode == 0)
page = open(html_out, encoding="utf-8").read() if os.path.isfile(html_out) else ""
lines = r.stdout.splitlines()
toward = lines[lines.index(next(x for x in lines if "TOWARD" in x)) + 1] if r.returncode == 0 else ""
check(f"boosted tokens lead the 'toward' list ({toward.strip()[:60]})",
      toward.strip().startswith("+1.50") and ("snout" in toward or "Mountain" in toward))
check("HTML: boosted tokens green with both probabilities on hover, others uncolored",
      'rgba(22, 163, 74, 0.75' in page and "with engram 90.0% | without 20.0%" in page
      and page.count("rgba(22, 163, 74") == 3 and "<title>Engram token diff</title>" in page)  # 2 tokens + legend
check("HTML escapes the text", "Celeste&#x27;s" in page)

# different texts must be refused
other = os.path.join(tmp, "reply2.txt")
open(other, "w").write("Something else entirely.")
srv, url = serve(False)
o2 = os.path.join(tmp, "off2.json")
subprocess.run([sys.executable, script, "run", "--url", url, "--messages", chat, "--reply", other, "--out", o2],
               capture_output=True, text=True)
srv.shutdown()
r = subprocess.run([sys.executable, script, "render", outs["on"], o2, "--out", html_out], capture_output=True, text=True)
check("render refuses runs that scored different tokens", r.returncode != 0 and "different tokens" in r.stderr)

json.dump([{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}], open(chat, "w"))
r = subprocess.run([sys.executable, script, "run", "--url", "http://127.0.0.1:9", "--messages", chat, "--reply",
                    reply, "--out", o2, "--model", "m"], capture_output=True, text=True)
check("messages ending with the assistant's reply are refused (it belongs in --reply)",
      r.returncode != 0 and "leave the reply out" in r.stderr)

bad = 0
for name, ok in results:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    bad += not ok
sys.exit(1 if bad else 0)
