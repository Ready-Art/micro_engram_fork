"""
How much of the data would the engram learn from, and how crowded would its tables be?

    python engram_data_stats.py /path/to/last_run_prepared/<hash>
    python engram_data_stats.py /path/to/last_run_prepared/<hash> --max-tokens 250000000

Reads Axolotl's prepared dataset (columns input_ids and labels, where -100 = masked).
Position t counts as 'trained' when the token AFTER it (t+1) is not masked,
because the engram output at t is used to predict token t+1.
"""
import argparse

import numpy as np
from datasets import DatasetDict, load_from_disk

V = 1 << 18  # 262144 = Gemma vocab size; three ids fit in 54 bits, so keys fit in int64
ORDERS = (2, 3)
TABLE_SIZES = (65_536, 262_144, 1_048_576)
MIN_REPEATS = 3  # an n-gram seen fewer times than this is mostly noise to memorize


def collect(ds, max_tokens, batch_size=1000):
    keys = {n: [] for n in ORDERS}
    sup = {n: [] for n in ORDERS}
    total_pos, trained_pos, rows = 0, 0, 0
    for batch in ds.iter(batch_size=batch_size):
        for ids, labels in zip(batch["input_ids"], batch["labels"]):
            ids = np.asarray(ids, dtype=np.int64)
            labels = np.asarray(labels, dtype=np.int64)
            T = len(ids)
            if T < 4:
                continue
            nxt_trained = np.zeros(T, dtype=bool)
            nxt_trained[:-1] = labels[1:] != -100  # last position has no next token
            total_pos += T
            trained_pos += int(nxt_trained.sum())
            rows += 1
            for n in ORDERS:
                k = ids[n - 1:].copy()
                for i in range(1, n):
                    k = k + ids[n - 1 - i: T - i] * (V ** i)
                keys[n].append(k)
                sup[n].append(nxt_trained[n - 1:])
        if total_pos >= max_tokens:
            break
    return keys, sup, total_pos, trained_pos, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="Axolotl prepared dataset folder")
    ap.add_argument("--max-tokens", type=int, default=50_000_000,
                    help="stop after about this many tokens (default 50M; 250M needs a lot of RAM)")
    args = ap.parse_args()

    ds = load_from_disk(args.path)
    if isinstance(ds, DatasetDict):
        ds = ds["train"] if "train" in ds else ds[list(ds.keys())[0]]
    cols = ds.column_names
    if "input_ids" not in cols or "labels" not in cols:
        raise SystemExit(f"Need columns input_ids and labels, found: {cols}")
    ds = ds.select_columns(["input_ids", "labels"]).shuffle(seed=0)

    keys, sup, total_pos, trained_pos, rows = collect(ds, args.max_tokens)
    print(f"\nsampled {rows:,} samples, {total_pos:,} tokens")
    print(f"positions with a trained next token: {trained_pos / total_pos:.1%} "
          f"({1 - trained_pos / total_pos:.1%} masked)\n")

    for n in ORDERS:
        k = np.concatenate(keys[n])
        s = np.concatenate(sup[n])
        uniq, counts = np.unique(k, return_counts=True)
        sup_uniq = np.unique(k[s])

        # which positions have an n-gram that NEVER appears at a trained position?
        idx = np.clip(np.searchsorted(sup_uniq, k), 0, len(sup_uniq) - 1)
        never_trained_pos = float(np.mean(sup_uniq[idx] != k))

        only_masked = len(uniq) - len(sup_uniq)
        worth = int((counts >= MIN_REPEATS).sum())
        print(f"--- {n}-grams ---")
        print(f"distinct: {len(uniq):,}")
        print(f"seen at least {MIN_REPEATS}x (worth memorizing): {worth:,}")
        print(f"distinct that only occur in masked text: {only_masked:,} "
              f"({only_masked / len(uniq):.1%})")
        print(f"positions whose n-gram is never trained directly: {never_trained_pos:.1%}")
        print("worth-memorizing n-grams per table row (each table holds all of them):")
        for M in TABLE_SIZES:
            print(f"    {M:>9,} rows -> {worth / M:6.2f} per row")
        print()

    print("Notes: counts come from the sample, so a bigger sample means more distinct n-grams.\n"
          "Roughly, 1 per row or less is comfortable; 5+ per row means heavy sharing.")


if __name__ == "__main__":
    main()