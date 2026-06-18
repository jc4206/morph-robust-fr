#!/usr/bin/env python3
"""
Tag per-ratio pairlist CSVs with blend_ratio and morph_root columns,
then create merged pairlist CSVs for Phase 2 (asymmetric) and Phase 3 (all ratios).

Run AFTER all four ratios have been generated so their pairlist CSVs exist.

Usage:
  python -m morphing.pairing.make_blend_pairlists \
      --morph_root       /path/to/blended_morphs \
      --ref_pairlist_dir /path/to/sim_morphs_3way \
      --out_dir          /path/to/blended_morphs/merged_pairlists

Output files (written to --out_dir):
  pairlist_{split}_alpha_{ratio}.csv        — tagged single-ratio pairlists
  pairlist_{split}_mixed_asymmetric.csv     — Phase 2: 40/60/30/70 only
  pairlist_{split}_mixed_all.csv            — Phase 3: 40/60/30/70 + 50:50
"""

import argparse
import sys
import pandas as pd
from pathlib import Path

SPLITS = ("train", "val", "test")
NEW_RATIOS = [40, 60, 30, 70]


def tag(df: pd.DataFrame, ratio: int, morph_root: str) -> pd.DataFrame:
    df = df.copy()
    df["blend_ratio"] = ratio
    df["morph_root"]  = morph_root
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--morph_root",       required=True,
                    help="Parent dir containing alpha_40/, alpha_60/, ... (blended_morphs/)")
    ap.add_argument("--ref_pairlist_dir", required=True,
                    help="Dir containing the 50:50 pairlist CSVs (sim_morphs_3way/)")
    ap.add_argument("--out_dir",          required=True,
                    help="Destination for tagged and merged pairlist CSVs")
    args = ap.parse_args()

    morph_root       = Path(args.morph_root)
    ref_pairlist_dir = Path(args.ref_pairlist_dir)
    out_dir          = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    errors = []

    for split in SPLITS:
        print(f"\n--- {split} ---")
        tagged_new: dict[int, pd.DataFrame] = {}

        # Tag each new ratio's pairlist
        for ratio in NEW_RATIOS:
            src = morph_root / f"alpha_{ratio}" / f"pairlist_{split}.csv"
            if not src.exists():
                msg = f"  WARNING: {src} not found — ratio {ratio} skipped for {split}"
                print(msg)
                errors.append(msg)
                continue

            df = tag(
                pd.read_csv(src),
                ratio,
                str(morph_root / f"alpha_{ratio}" / split),
            )
            dst = out_dir / f"pairlist_{split}_alpha_{ratio}.csv"
            df.to_csv(dst, index=False)
            tagged_new[ratio] = df
            print(f"  alpha_{ratio:>2}: {len(df):>8,} rows  ->  {dst.name}")

        # Tag the 50:50 reference pairlist
        ref_src = ref_pairlist_dir / f"pairlist_{split}.csv"
        df_50 = None
        if not ref_src.exists():
            msg = f"  WARNING: 50:50 pairlist not found at {ref_src}"
            print(msg)
            errors.append(msg)
        else:
            df_50 = tag(
                pd.read_csv(ref_src),
                50,
                str(ref_pairlist_dir / split),
            )
            dst_50 = out_dir / f"pairlist_{split}_alpha_50.csv"
            df_50.to_csv(dst_50, index=False)
            print(f"  alpha_50:  {len(df_50):>8,} rows  ->  {dst_50.name}")

        # Phase 2: asymmetric only (40 / 60 / 30 / 70, no 50:50)
        avail_new = [tagged_new[r] for r in NEW_RATIOS if r in tagged_new]
        if avail_new:
            merged_asym = pd.concat(avail_new, ignore_index=True)
            dst = out_dir / f"pairlist_{split}_mixed_asymmetric.csv"
            merged_asym.to_csv(dst, index=False)
            print(f"  mixed_asymmetric: {len(merged_asym):>8,} rows  ->  {dst.name}")

        # Phase 3: all ratios (40 / 60 / 30 / 70 + 50:50)
        all_dfs = avail_new + ([df_50] if df_50 is not None else [])
        if all_dfs:
            merged_all = pd.concat(all_dfs, ignore_index=True)
            dst = out_dir / f"pairlist_{split}_mixed_all.csv"
            merged_all.to_csv(dst, index=False)
            print(f"  mixed_all:        {len(merged_all):>8,} rows  ->  {dst.name}")

    print("\n=== Done ===")
    if errors:
        print("\nWarnings / errors encountered:")
        for e in errors:
            print(f"  {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()