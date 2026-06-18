#!/usr/bin/env python3
"""
Stage 1 — Random morph pipeline.

Generates a 60/20/20 identity-level split and random pair CSVs.
Output files written to --out_dir:
  random_split_{train,val,test}.txt
  pairs_random_{train,val,test}.csv   (columns: id_a, id_b)
"""

import os
import csv
import json
import random
import argparse
from pathlib import Path


def canon(a: str, b: str):
    """Canonical (sorted) pair key for symmetric deduplication — matches make_split.py logic."""
    ia, ib = int(a), int(b)
    return (ia, ib) if ia < ib else (ib, ia)


def generate_pairs(split_ids, n_partners, rng):
    raw = []
    for id_a in split_ids:
        candidates = [x for x in split_ids if x != id_a]
        partners = rng.sample(candidates, min(n_partners, len(candidates)))
        for id_b in partners:
            raw.append((id_a, id_b))
    return raw


def deduplicate(pairs):
    seen = set()
    deduped = []
    for a, b in pairs:
        key = canon(a, b)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((a, b))
    return deduped


def main():
    ap = argparse.ArgumentParser(description="Stage 1: random identity split + pair generation")
    ap.add_argument("--dataset_root", required=True,
                    help="Root folder containing one subdirectory per identity")
    ap.add_argument("--out_dir", required=True,
                    help="Output directory (RAND_PAIRS_DIR)")
    ap.add_argument("--n_partners", type=int, default=5,
                    help="Partners sampled per identity")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_ratio", type=float, default=0.6)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    identities = sorted(
        d for d in os.listdir(dataset_root)
        if (dataset_root / d).is_dir()
    )
    print(f"Found {len(identities)} identities in {dataset_root}")

    rng = random.Random(args.seed)
    shuffled = identities[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)

    train_ids = shuffled[:n_train]
    val_ids = shuffled[n_train:n_train + n_val]
    test_ids = shuffled[n_train + n_val:]

    print(f"Split: {len(train_ids)} train / {len(val_ids)} val / {len(test_ids)} test")

    train_set, val_set, test_set = set(train_ids), set(val_ids), set(test_ids)
    assert not (train_set & val_set),   "train/val overlap — seed collision?"
    assert not (train_set & test_set),  "train/test overlap — seed collision?"
    assert not (val_set & test_set),    "val/test overlap — seed collision?"
    assert len(train_set) + len(val_set) + len(test_set) == n, "identity count mismatch"

    for name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        (out_dir / f"random_split_{name}.txt").write_text("\n".join(ids) + "\n")
        # JSON format expected by train_adapter_eer_stop.py (sorted integers, matching make_split.py)
        (out_dir / f"{name}_ids.json").write_text(json.dumps(sorted(int(i) for i in ids)))
        print(f"Written random_split_{name}.txt + {name}_ids.json  ({len(ids)} identities)")

    splits = {"train": train_ids, "val": val_ids, "test": test_ids}

    for split_name, split_ids in splits.items():
        split_set = set(split_ids)

        raw = generate_pairs(split_ids, args.n_partners, rng)
        n_raw = len(raw)

        deduped = deduplicate(raw)
        n_deduped = len(deduped)

        print(f"[{split_name}] pairs before dedup: {n_raw:,}  "
              f"after: {n_deduped:,}  removed: {n_raw - n_deduped:,}")

        # Inline sanity checks (abort early on violation rather than silently continuing)
        for a, b in deduped:
            assert a != b, f"self-pair in {split_name}: {a}"
            assert a in split_set and b in split_set, \
                f"cross-split pair in {split_name}: ({a}, {b})"
        unique_keys = {canon(a, b) for a, b in deduped}
        assert len(unique_keys) == len(deduped), f"dedup key mismatch in {split_name}"

        out_csv = out_dir / f"pairs_random_{split_name}.csv"
        with open(out_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id_a", "id_b"])
            for a, b in deduped:
                writer.writerow([a, b])

        print(f"[{split_name}] Written {out_csv}  ({n_deduped:,} pairs) — all checks OK")

    print("\nStage 1 complete.")


if __name__ == "__main__":
    main()