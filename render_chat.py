"""
Render chat samples into plain text with the model's chat template, for eval_logprob.py.

    python render_chat.py --data combined.jsonl --tokenizer /path/to/model \\
        --template /path/to/model/chat_template_step2.jinja \\
        --kwargs '{"enable_thinking": true, "preserve_thinking": true}' --n 80 --out chat_eval.jsonl

Reads a .jsonl where each line holds a conversation under "messages" (role/content) or
"conversations" / "conversation" (ShareGPT from/value: human, gpt, system). Picks --n of
them at random (fixed --seed, so both configs get the same samples) and writes
{"text": <the conversation exactly as the template renders it>, "spans": [[start, end], ...]}
per line. "spans" are the characters of the assistant turns (content + --eot, not the role
header): what training trains on and what the model writes. eval_logprob.py scores only
those when they are there (the whole conversation is still the context):

    python eval_logprob.py run --url ... --data chat_eval.jsonl --out on.json --parallel 1

Without spans (--no-spans) the whole text is scored: system prompt, character card and user
turns too. A chat model was never trained to predict those, so that mixes in a different
skill (predicting any text) and can make a document-style model look far better than it is.
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROLE = {"human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant", "model": "assistant",
        "system": "system", "tool": "tool"}


def messages_of(obj):
    for key in ("messages", "conversations", "conversation"):
        msgs = obj.get(key)
        if isinstance(msgs, list) and msgs:
            out = []
            for m in msgs:
                role = m.get("role", m.get("from"))
                content = m.get("content", m.get("value"))
                if role is None or content is None:
                    return None
                out.append({"role": ROLE.get(role, role), "content": content})
            return out
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--tokenizer", required=True, help="model folder (its tokenizer and chat template)")
    ap.add_argument("--template", default=None, help="a .jinja chat template that replaces the tokenizer's "
                    "(use the same one as training: chat_template_jinja)")
    ap.add_argument("--kwargs", default="{}", help="chat template variables as JSON (training's chat_template_kwargs)")
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--eot", action="append", default=None,
                    help='end-of-turn token scored with each assistant turn (training\'s eot_tokens), e.g. "<turn|>"; repeatable')
    ap.add_argument("--roles", default="assistant", help="comma-separated roles whose turns are scored")
    ap.add_argument("--no-spans", action="store_true", help="score the whole text (old behavior)")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    if args.template:
        with open(args.template, encoding="utf-8") as f:
            tok.chat_template = f.read()
    kwargs = json.loads(args.kwargs)

    convs, skipped = [], 0
    with open(args.data, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            msgs = messages_of(json.loads(line))
            if msgs is None:
                skipped += 1
            else:
                convs.append(msgs)
    if not convs:
        sys.exit(f"{args.data}: no conversations found (expected 'messages' or 'conversations' per line)")
    picked = random.Random(args.seed).sample(convs, min(args.n, len(convs)))

    from micro_engram.train.data import chat_spans

    def render(m, add_generation_prompt=False):
        return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=add_generation_prompt, **kwargs)

    roles = {r.strip() for r in args.roles.split(",") if r.strip()}
    no_bos, no_spans = 0, 0
    with open(args.out, "w", encoding="utf-8") as g:
        for msgs in picked:
            row = {}
            if args.no_spans:
                row["text"] = render(msgs)
            else:
                try:
                    row["text"], spans = chat_spans(render, msgs, roles, args.eot or ())
                    row["spans"] = [list(sp) for sp in spans]
                except ValueError:
                    no_spans += 1  # template not prefix-stable for this one: whole text, flagged
                    row["text"] = render(msgs)
                if not row.get("spans"):
                    no_spans += "spans" in row
                    row.pop("spans", None)
            no_bos += bool(tok.bos_token) and not row["text"].startswith(tok.bos_token)
            g.write(json.dumps(row, ensure_ascii=False) + "\n")
    avg = sum(len(json.loads(l)["text"]) for l in open(args.out, encoding="utf-8")) / len(picked)
    print(f"{len(picked)} of {len(convs)} conversations rendered to {args.out} (average {avg:,.0f} characters)"
          + (f"; {skipped} lines without a conversation skipped" if skipped else ""))
    if no_spans:
        print(f"note: {no_spans} conversations have no scorable {sorted(roles)} turns (or the template is not "
              f"prefix-stable for them); eval_logprob.py skips them in assistant-only scoring")
    if no_bos:
        print(f"note: {no_bos} rendered texts do not start with {tok.bos_token!r}; pass --prefix "
              f"\"{tok.bos_token}\" to eval_logprob.py so every piece gets one")


if __name__ == "__main__":
    main()
