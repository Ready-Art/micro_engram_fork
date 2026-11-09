"""
Compare micro_engram.train's tokenization with an Axolotl prepared dataset for the same YAML.

    python compare_axolotl_data.py stage2.yaml /path/to/dataset_prepared_path/<axolotl hash folder> [--n 200]

Tokenizes the YAML's datasets the micro_engram.train way and checks, sample by sample (in
order), that the tokens and the trained-token masks match Axolotl's input_ids / labels.
Axolotl drops some samples (too long, nothing to train), so the samples are matched by their
tokens first; unmatched ones are listed. Prints a summary and the first differences. If it
reports differences that matter, train with `axolotl_prepared_dataset: <that folder>`.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("axolotl_folder")
    ap.add_argument("--n", type=int, default=0, help="only the first N of our samples (0: all)")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    from micro_engram.train.config import load_config
    from micro_engram.train.data import read_axolotl_prepared, tokenize_all

    cfg = load_config(args.config)
    tok_path = cfg.tokenizer_config or cfg.base_model
    tok = AutoTokenizer.from_pretrained(tok_path)
    ours = tokenize_all(cfg, tok, tok_path)
    theirs = read_axolotl_prepared(args.axolotl_folder)

    index = {}
    for i in range(len(theirs)):
        t, _ = theirs.get(i)
        index.setdefault(t.tobytes(), []).append(i)
    n = len(ours) if not args.n else min(args.n, len(ours))
    same, mask_diff, missing = 0, [], []
    for i in range(n):
        t, m = ours.get(i)
        hits = index.get(t.tobytes())
        if not hits:
            missing.append(i)
            continue
        _, m2 = theirs.get(hits[0])
        if np.array_equal(m, m2):
            same += 1
        else:
            mask_diff.append((i, hits[0]))
    print(f"ours: {len(ours):,} samples, Axolotl: {len(theirs):,}; compared {n:,}")
    print(f"  identical tokens + mask: {same:,}")
    print(f"  same tokens, different trained mask: {len(mask_diff):,}")
    print(f"  tokens not found in Axolotl's data: {len(missing):,}")
    for i, j in mask_diff[:3]:
        t, m = ours.get(i)
        _, m2 = theirs.get(j)
        d = np.nonzero(m != m2)[0]
        a, b = max(0, d[0] - 5), min(len(t), d[0] + 15)
        print(f"\n sample {i}: {len(d)} tokens differ, first at {d[0]} (+ = we train, A = Axolotl trains):")
        print("  " + " ".join(f"{tok.decode([int(x)])!r}{'+' if m[k] else ''}{'A' if m2[k] else ''}"
                              for k, x in zip(range(a, b), t[a:b])))
    for i in missing[:3]:
        t, _ = ours.get(i)
        print(f"\n sample {i} not in Axolotl's data ({len(t)} tokens): {tok.decode(t[:40].tolist())!r}...")
    sys.exit(0 if not mask_diff and not missing else 1)


if __name__ == "__main__":
    main()
