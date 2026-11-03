"""
Build the token-merging map for the engram.

    python build_canon.py --tokenizer /path/to/gemma-4-31B-it --out canon.pt --paper
    python build_canon.py --tokenizer /path/to/gemma-4-31B-it --out canon.pt   # case only

Default: merge case variants (NFKC + lowercase), keeping the leading-space marker,
so "▁The" and "▁the" share a row but "▁the" and the word-piece "the" do not.

Extra options (each one merges more):
  --merge-space        also merge space-prefixed and bare pieces ("▁the" with "the")
  --strip-accents      also merge accented Latin/Greek/Cyrillic letters with plain ones ("é" with "e")
  --merge-whitespace   all whitespace-only pieces (spaces, tabs, newlines, runs of them) share one id
  --paper              all three, as in the Engram paper's "tokenizer compression"

  --protect FILE       names to keep apart from the ordinary word ("Peril" the dragon vs
                       "peril"): one name per line, # comments. Repeatable.

Special tokens, added tokens and byte tokens are never merged.
The script prints how many classes each option gives, so you can compare.
"""
import argparse
import random
import re
import unicodedata
from collections import defaultdict

import torch
from transformers import AutoTokenizer

MARKERS = ("▁", "Ġ")  # SentencePiece space marker, byte-level BPE space marker
SPACE = "▁"
BYTE_TOKEN = re.compile(r"^<0x[0-9A-Fa-f]{2}>$")
# Visible stand-ins tokenizers use for whitespace -> the real character.
WS_STANDINS = str.maketrans({"▁": " ", "Ġ": " ", "Ċ": "\n", "ĉ": "\t"})
WS_KEY = "<whitespace>"


def strip_accents(s):
    """Drop combining diacritics (U+0300-U+036F) only, so other scripts' marks are kept."""
    decomposed = unicodedata.normalize("NFKD", s)
    kept = "".join(c for c in decomposed if not 0x300 <= ord(c) <= 0x36F)
    return unicodedata.normalize("NFKC", kept)


def merge_key(piece, merge_space, accents=False, whitespace=False):
    if whitespace:
        real = piece.translate(WS_STANDINS)
        if real and real.isspace():
            return WS_KEY
    lead = ""
    if piece and piece[0] in MARKERS:
        lead, piece = SPACE, piece[1:]
    body = unicodedata.normalize("NFKC", piece).lower()
    if accents:
        body = strip_accents(body)
    if body == "":
        return None  # nothing to merge on
    return body if merge_space else lead + body


def read_names(path):
    with open(path, encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.lstrip().startswith("#")]


def name_token_ids(names, pieces):
    """Ids of the tokens that ARE a name (bare or with a leading-space marker).

    Returns (ids to protect, names that are a single token, names split into several
    tokens). A split name is left alone: its pieces are not the name.
    """
    index = {p: i for i, p in enumerate(pieces) if p is not None}
    ids, single, split = set(), [], []
    for name in names:
        found = [index[v] for v in (name, *(m + name for m in MARKERS)) if v in index]
        if found:
            ids.update(found)
            single.append(name)
        else:
            split.append(name)
    return ids, single, split


def build(pieces, protected, vocab_size, merge_space, accents=False, whitespace=False):
    canon = list(range(vocab_size))
    groups = defaultdict(list)
    for i, p in enumerate(pieces):
        if i in protected or p is None or BYTE_TOKEN.match(p):
            continue
        key = merge_key(p, merge_space, accents, whitespace)
        if key is not None:
            groups[key].append(i)  # ids arrive in increasing order
    for ids in groups.values():
        rep = ids[0]  # smallest id represents the group
        for i in ids:
            canon[i] = rep
    return canon, groups


OPTION_SETS = (  # (label, merge_space, strip_accents, merge_whitespace), each adds one step
    ("case only", False, False, False),
    ("+ space", True, False, False),
    ("+ accents", True, True, False),
    ("+ whitespace (paper)", True, True, True),
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vocab-size", type=int, default=262_144, help="model vocab size (Gemma 4: 262144)")
    ap.add_argument("--merge-space", action="store_true")
    ap.add_argument("--strip-accents", action="store_true")
    ap.add_argument("--merge-whitespace", action="store_true")
    ap.add_argument("--paper", action="store_true", help="same as all three options above")
    ap.add_argument("--protect", action="append", default=[], metavar="FILE",
                    help="file of names whose own tokens are never merged (repeatable)")
    args = ap.parse_args()
    if args.paper:
        args.merge_space = args.strip_accents = args.merge_whitespace = True
    chosen = (args.merge_space, args.strip_accents, args.merge_whitespace)

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    n = min(len(tok), args.vocab_size)
    pieces = tok.convert_ids_to_tokens(list(range(n)))
    protected = set(tok.get_added_vocab().values()) | set(tok.all_special_ids)
    print(f"tokenizer pieces: {n:,} | protected (special/added): {len(protected):,}")
    for path in args.protect:
        ids, single, split = name_token_ids(read_names(path), pieces)
        protected |= ids
        print(f"{path}: {len(single)} names kept apart ({len(ids)} tokens)")
        if split:
            print(f"  not single tokens, left alone ({len(split)}): {', '.join(split)}")

    for name, *flags in OPTION_SETS:
        canon, _ = build(pieces, protected, args.vocab_size, *flags)
        classes = len(set(canon[:n]))
        mark = "  <- chosen" if tuple(flags) == chosen else ""
        print(f"{name:22s}: {n:,} tokens -> {classes:,} classes ({1 - classes / n:.1%} fewer){mark}")

    canon, groups = build(pieces, protected, args.vocab_size, *chosen)
    print(f"\nchosen: merge_space={args.merge_space} strip_accents={args.strip_accents} "
          f"merge_whitespace={args.merge_whitespace}")
    if WS_KEY in groups:
        ws = groups[WS_KEY]
        print(f"whitespace class: {len(ws)} pieces, e.g. {[pieces[i] for i in ws[:8]]}")
    multi = [ids for k, ids in groups.items() if len(ids) >= 3 and k != WS_KEY]
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
