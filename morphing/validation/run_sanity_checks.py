#!/usr/bin/env python3
"""
Step 9 — Sanity checks for the random morph pipeline.

Reads the Stage 1 split files + pairs CSVs and the Stage 2 pairlist CSVs,
then writes sanity_check_log.txt with all results.

Run after both Stage 1 and Stage 2 have completed.

Usage:
  python run_sanity_checks.py \
    --pairs_dir  $RAND_PAIRS_DIR \
    --out_root   $RAND_OUT_ROOT \
    --out_log    $RAND_OUT_ROOT/sanity_check_log.txt
"""

import os
import csv
import argparse
import numpy as np
from pathlib import Path

SPLITS = ("train", "val", "test")


def log(lines, msg=""):
    print(msg)
    lines.append(msg)


def check_splits_and_pairs(pairs_dir, lines):
    log(lines, "\n=== CHECK 1-4, 6: SPLITS AND PAIRS ===")
    split_ids = {}

    for split in SPLITS:
        txt = Path(pairs_dir) / f"random_split_{split}.txt"
        if not txt.exists():
            log(lines, f"  ERROR: {txt} not found")
            continue
        ids = [l.strip() for l in txt.read_text().splitlines() if l.strip()]
        split_ids[split] = set(ids)
        log(lines, f"  {split}: {len(ids)} identities")

    # Check 1: split disjointness
    if len(split_ids) == 3:
        tv = split_ids["train"] & split_ids["val"]
        tt = split_ids["train"] & split_ids["test"]
        vt = split_ids["val"] & split_ids["test"]
        ok = not tv and not tt and not vt
        log(lines, f"\nCHECK 1 (split disjointness): {'PASSED' if ok else 'FAILED'}")
        if not ok:
            log(lines, f"  train∩val={len(tv)}, train∩test={len(tt)}, val∩test={len(vt)}")

    for split in SPLITS:
        csv_path = Path(pairs_dir) / f"pairs_random_{split}.csv"
        if not csv_path.exists():
            log(lines, f"\n  ERROR: {csv_path} not found")
            continue

        pairs = []
        with open(csv_path, "r", newline="") as f:
            for row in csv.DictReader(f):
                pairs.append((row["id_a"], row["id_b"]))

        split_set = split_ids.get(split, set())

        # Check 3: no self-pairs
        self_pairs = [p for p in pairs if p[0] == p[1]]
        # Check 2: same-split constraint
        cross = [p for p in pairs if p[0] not in split_set or p[1] not in split_set]
        # Check 4: no duplicates
        seen = set()
        dups = []
        for a, b in pairs:
            key = (min(int(a), int(b)), max(int(a), int(b)))
            if key in seen:
                dups.append((a, b))
            seen.add(key)

        # Check 6: pair counts
        log(lines, f"\n  [{split}] pairs_random_{split}.csv — {len(pairs):,} pairs")
        log(lines, f"  CHECK 2 same-split: {'PASSED' if not cross else f'FAILED ({len(cross)} cross-split)'}")
        log(lines, f"  CHECK 3 no self-pairs: {'PASSED' if not self_pairs else f'FAILED ({len(self_pairs)} self-pairs)'}")
        log(lines, f"  CHECK 4 no duplicates: {'PASSED' if not dups else f'FAILED ({len(dups)} duplicates)'}")


def check_morph_files(out_root, lines):
    log(lines, "\n=== CHECK 5: MORPH FILE EXISTENCE ===")
    for split in SPLITS:
        pl = Path(out_root) / f"pairlist_{split}.csv"
        if not pl.exists():
            log(lines, f"  [{split}] pairlist not found: {pl}")
            continue

        total = 0
        rows_with_missing = 0
        with open(pl, "r", newline="") as f:
            for row in csv.DictReader(f):
                total += 1
                for col in ("out_full", "out_inA", "out_inB"):
                    p = row.get(col, "")
                    if p and not os.path.exists(p):
                        rows_with_missing += 1
                        break

        ok = rows_with_missing == 0
        log(lines, f"  [{split}] {total:,} rows, {rows_with_missing} with missing morph files "
                   f"— CHECK 5: {'PASSED' if ok else 'FAILED'}")


def check_cosine_distribution(out_root, lines):
    log(lines, "\n=== CHECK 7: COSINE SIMILARITY DISTRIBUTION ===")
    for split in SPLITS:
        pl = Path(out_root) / f"pairlist_{split}.csv"
        if not pl.exists():
            log(lines, f"  [{split}] pairlist not found")
            continue

        sims = []
        with open(pl, "r", newline="") as f:
            reader = csv.DictReader(f)
            if "cosine_similarity" not in (reader.fieldnames or []):
                log(lines, f"  [{split}] no cosine_similarity column — CHECK 7: SKIPPED "
                           f"(run add_cosine_similarity.py first)")
                continue
            for row in reader:
                v = row.get("cosine_similarity", "")
                try:
                    sims.append(float(v))
                except (ValueError, TypeError):
                    pass

        if not sims:
            log(lines, f"  [{split}] no numeric cosine_similarity values — CHECK 7: SKIPPED")
            continue

        arr = np.array(sims)
        log(lines, f"  [{split}] n={len(arr):,}  mean={arr.mean():.4f} ± {arr.std():.4f}  "
                   f"min={arr.min():.4f}  max={arr.max():.4f}")

        if arr.mean() > 0.5:
            log(lines, f"  [{split}] FLAG: mean={arr.mean():.4f} is unexpectedly high for random pairs — "
                       f"CHECK 7: FLAG (expected notably lower than similarity-driven)")
        else:
            log(lines, f"  [{split}] CHECK 7: PASSED")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_dir", required=True, help="artifacts_random/ directory (Stage 1 output)")
    ap.add_argument("--out_root", required=True, help="random_morphs_3way/ directory (Stage 2 output)")
    ap.add_argument("--out_log", default="sanity_check_log.txt")
    args = ap.parse_args()

    lines = []
    log(lines, "=== RANDOM MORPH PIPELINE SANITY CHECK ===")
    log(lines, f"pairs_dir : {args.pairs_dir}")
    log(lines, f"out_root  : {args.out_root}")

    check_splits_and_pairs(args.pairs_dir, lines)
    check_morph_files(args.out_root, lines)
    check_cosine_distribution(args.out_root, lines)

    log(lines, "\n=== ALL CHECKS COMPLETE ===")

    Path(args.out_log).write_text("\n".join(lines) + "\n")
    print(f"\nLog written to {args.out_log}")


if __name__ == "__main__":
    main()