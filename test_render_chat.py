"""
render_chat.py with a tiny tokenizer and template (needs transformers, tokenizers):  python test_render_chat.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast

HERE = os.path.dirname(os.path.abspath(__file__))
results = []


def check(name, ok):
    results.append((name, bool(ok)))


tmp = tempfile.mkdtemp(prefix="render_chat_")
tok = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(models.WordLevel({"<unk>": 0, "<bos>": 2}, unk_token="<unk>")),
                              bos_token="<bos>", unk_token="<unk>")
tok.chat_template = "{{ bos_token }}{% for m in messages %}[{{ m.role }}]{{ m.content }}\n{% endfor %}"
tok.save_pretrained(os.path.join(tmp, "model"))
jinja = os.path.join(tmp, "step2.jinja")
open(jinja, "w").write("{{ bos_token }}{% if think %}<think>{% endif %}"
                       "{% for m in messages %}<{{ m.role }}>{{ m.content }}<end>\n{% endfor %}")

data = os.path.join(tmp, "combined.jsonl")
with open(data, "w") as f:
    for i in range(10):
        f.write(json.dumps({"messages": [{"role": "user", "content": f"hi {i}"},
                                         {"role": "assistant", "content": f'say "{i}"\nok'}]}) + "\n")
    f.write(json.dumps({"conversations": [{"from": "human", "value": "sharegpt q"},
                                          {"from": "gpt", "value": "sharegpt a"}]}) + "\n")
    f.write(json.dumps({"text": "not a conversation"}) + "\n")

script = os.path.join(HERE, "render_chat.py")


def run(*extra, out="out.jsonl"):
    path = os.path.join(tmp, out)
    r = subprocess.run([sys.executable, script, "--data", data, "--tokenizer", os.path.join(tmp, "model"),
                        "--out", path, *extra], capture_output=True, text=True)
    lines = [json.loads(l)["text"] for l in open(path)] if os.path.isfile(path) else []
    return r, lines


r, lines = run("--n", "100")
check("all 11 conversations rendered, the non-chat line skipped and reported",
      r.returncode == 0 and len(lines) == 11 and "1 lines without a conversation skipped" in r.stdout)
check("tokenizer's own template used by default; text starts with <bos>",
      all(t.startswith("<bos>[") for t in lines))
check("ShareGPT roles mapped (human -> user, gpt -> assistant)",
      any("[user]sharegpt q" in t and "[assistant]sharegpt a" in t for t in lines))
check("quotes and newlines survive the round trip", any('[assistant]say "3"\nok' in t for t in lines))

r, lines = run("--n", "3", "--template", jinja, "--kwargs", '{"think": true}', out="t.jsonl")
check("--template replaces the tokenizer's template, --kwargs reach it",
      len(lines) == 3 and all(t.startswith("<bos><think><") for t in lines))
_, again = run("--n", "3", "--template", jinja, "--kwargs", '{"think": true}', out="t2.jsonl")
_, other = run("--n", "3", "--template", jinja, "--seed", "7", out="t3.jsonl")
check("same --seed picks the same samples (both configs score the same texts)", lines == again)
check("another seed picks others", [t.replace("<think>", "") for t in lines] != other)

# spans: the assistant turns' content + end-of-turn token, not the role header or user turns
gen = os.path.join(tmp, "gen.jinja")
open(gen, "w").write("{{ bos_token }}{% for m in messages %}<{{ m.role }}>{{ m.content }}<end>\n{% endfor %}"
                     "{% if add_generation_prompt %}<assistant>{% endif %}")
path = os.path.join(tmp, "spans.jsonl")
r = subprocess.run([sys.executable, script, "--data", data, "--tokenizer", os.path.join(tmp, "model"), "--out", path,
                    "--template", gen, "--eot", "<end>", "--n", "100"], capture_output=True, text=True)
rows = [json.loads(l) for l in open(path)] if os.path.isfile(path) else []
got = [[row["text"][a:b] for a, b in row["spans"]] for row in rows if row.get("spans")]
check("spans = each assistant turn's content + <end>, nothing else" + ("" if r.returncode == 0 else f" ({r.stderr[-300:]})"),
      len(got) == 11 and ['say "3"\nok<end>'] in got and ["sharegpt a<end>"] in got
      and all(len(g) == 1 for g in got))
path2 = os.path.join(tmp, "nospans.jsonl")
subprocess.run([sys.executable, script, "--data", data, "--tokenizer", os.path.join(tmp, "model"), "--out", path2,
                "--no-spans", "--n", "2"], capture_output=True, text=True)
check("--no-spans writes text only", all("spans" not in json.loads(l) for l in open(path2)))

shutil.rmtree(tmp, ignore_errors=True)
bad = 0
for name, ok in results:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    bad += not ok
sys.exit(1 if bad else 0)
