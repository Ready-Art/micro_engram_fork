"""
Render chat samples into plain text with the model's chat template, for eval_logprob.py.

    python render_chat.py --data combined.jsonl --tokenizer /path/to/model \\
        --template /path/to/model/chat_template_step2.jinja \\
        --kwargs '{"enable_thinking": true, "preserve_thinking": true}' --n 80 --out chat_eval.jsonl

Reads a .jsonl where each line holds a conversation under "messages" (role/content) or
"conversations" / "conversation" (ShareGPT from/value: human, gpt, system). Picks --n of
them at random (fixed --seed, so both configs get the same samples) and writes
{"text": <the conversation exactly as the template renders it>} per line.

Then score it once per config (each text starts with the template's <bos>; long ones are
cut into pieces, so give every piece its own <bos> with --prefix):
    python eval_logprob.py run --url ... --data chat_eval.jsonl --out on.json --max-chars 6000 --parallel 1 --prefix "<bos>"
"""
import argparse
import json
import random
import sys

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

    no_bos = 0
    with open(args.out, "w", encoding="utf-8") as g:
        for msgs in picked:
            text = tok.apply_chat_template(msgs, tokenize=False, **kwargs)
            no_bos += bool(tok.bos_token) and not text.startswith(tok.bos_token)
            g.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
    avg = sum(len(json.loads(l)["text"]) for l in open(args.out, encoding="utf-8")) / len(picked)
    print(f"{len(picked)} of {len(convs)} conversations rendered to {args.out} (average {avg:,.0f} characters)"
          + (f"; {skipped} lines without a conversation skipped" if skipped else ""))
    if no_bos:
        print(f"note: {no_bos} rendered texts do not start with {tok.bos_token!r}; pass --prefix "
              f"\"{tok.bos_token}\" to eval_logprob.py so every piece gets one")


if __name__ == "__main__":
    main()
