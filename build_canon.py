"""
Build the token-merging map for the engram.

    python build_canon.py --tokenizer /path/to/gemma-4-31B-it --out canon.pt
    python build_canon.py --tokenizer /path/to/gemma-4-31B-it --out canon.pt --merge-space

Default: merge case variants (NFKC + lowercase), keeping the leading-space marker,
so "?The" and "?the" share a row but "?the" and the word-piece "the" do not.
--merge-space also merges space-prefixed and non-prefixed pieces ("?the" with "the").
Special tokens, added tokens and byte tokens are never merged.
The script prints how many classes each option gives, so you can pick.
"""
import argparse
import random
import re
import unicodedata
from collections import defaultdict

import torch
from transformers import AutoTokenizer

MARKERS = ("\u2581", "\u0120")  # SentencePiece space marker, byte-level BPE space marker
SPACE = "\u2581"
BYTE_TOKEN = re.compile(r"^<0x[0-9A-Fa-f]{2}>$")


def merge_key(piece, merge_space):
    lead = ""
    if piece and piece[0] in MARKERS:
        lead, piece = SPACE, piece[1:]
    body = unicodedata.normalize("NFKC", piece).lower()
    if body == "":
        return None  # nothing to merge on
    return body if merge_space else lead + body


def build(pieces, protected, vocab_size, merge_space):
    canon = list(range(vocab_size))
    groups = defaultdict(list)
    for i, p in enumerate(pieces):
        if i in protected or p is None or BYTE_TOKEN.match(p):
            continue
        key = merge_key(p, merge_space)
        if key is not None:
            groups[key].append(i)  # ids arrive in increasing order
    for ids in groups.values():
        rep = ids[0]  # smallest id represents the group
        for i in ids:
            canon[i] = rep
    return canon, groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vocab-size", type=int, default=262_144, help="model vocab size (Gemma 4: 262144)")
    ap.add_argument("--merge-space", action="store_true")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    n = min(len(tok), args.vocab_size)
    pieces = tok.convert_ids_to_tokens(list(range(n)))
    protected = set(tok.get_added_vocab().values()) | set(tok.all_special_ids)
    print(f"tokenizer pieces: {n:,} | protected (special/added): {len(protected):,}")

    results = {}
    for name, flag in (("case only", False), ("case + space", True)):
        canon, groups = build(pieces, protected, args.vocab_size, flag)
        classes = len(set(canon[:n]))
        results[flag] = (canon, groups)
        print(f"{name:13s}: {n:,} tokens -> {classes:,} classes ({1 - classes / n:.1%} fewer)")

    canon, groups = results[args.merge_space]
    print(f"\nchosen: {'case + space' if args.merge_space else 'case only'}")
    multi = [ids for ids in groups.values() if len(ids) >= 3]
    random.seed(0)
    print("\nsample of merged groups (check these look right):")
    for ids in random.sample(multi, min(12, len(multi))):
        print("   ", [pieces[i] for i in ids[:8]])

    t = torch.tensor(canon, dtype=torch.long)
    assert torch.equal(t[t], t), "map is not idempotent"
    torch.save(t, args.out)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()