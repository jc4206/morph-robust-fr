"""
merge_splits.py

Merges train/val/test identity splits into a single all-IDs split for final
full-data adapter training.  The existing split directory is never modified.

Usage:
    python merge_splits.py \
        --split_dir  /path/to/splits \
        --out_dir    /path/to/splits_full

Output (in --out_dir):
    train_ids.json        — union of all input splits, sorted
    merge_stats.json      — provenance: source counts, total, input paths

Train with the merged split (no validation):
    python train_adapter.py \
        --split_dir  /path/to/splits_full \
        --split      train \
        --val_split  train \
        --val_every  999999 \
        ...
    (--val_split train re-uses train_ids.json for the val loader so no file is
     missing; --val_every > --steps ensures validation never fires.)
"""

import argparse
import json
from pathlib import Path


SPLIT_FILES = ("train_ids.json", "val_ids.json", "test_ids.json")


def load_ids(path: Path) -> list[int]:
    return [int(x) for x in json.loads(path.read_text())]


def save_ids(path: Path, ids: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sorted(ids), indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Merge train/val/test identity splits into one all-IDs split."
    )
    ap.add_argument(
        "--split_dir", required=True,
        help="Directory containing the existing train_ids.json, val_ids.json, test_ids.json.",
    )
    ap.add_argument(
        "--out_dir", required=True,
        help="Output directory for the merged split (created if it does not exist). "
             "Must differ from --split_dir.",
    )
    args = ap.parse_args()

    split_dir = Path(args.split_dir).resolve()
    out_dir   = Path(args.out_dir).resolve()

    if out_dir == split_dir:
        ap.error("--out_dir must differ from --split_dir to avoid overwriting existing splits.")

    # ------------------------------------------------------------------
    # Load available split files
    # ------------------------------------------------------------------
    per_split: dict[str, list[int]] = {}
    for fname in SPLIT_FILES:
        p = split_dir / fname
        if p.exists():
            per_split[fname] = load_ids(p)
            print(f"[Load] {p}  →  {len(per_split[fname])} IDs")
        else:
            print(f"[Skip] {p}  (not found)")

    if not per_split:
        ap.error(f"No split files found in {split_dir}. Check --split_dir.")

    # ------------------------------------------------------------------
    # Integrity check: splits must be pairwise disjoint
    # ------------------------------------------------------------------
    names  = list(per_split.keys())
    splits = list(per_split.values())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = set(splits[i]) & set(splits[j])
            if overlap:
                raise RuntimeError(
                    f"Splits are not disjoint: {names[i]} ∩ {names[j]} = "
                    f"{len(overlap)} IDs (e.g. {sorted(overlap)[:5]})"
                )
    print("[OK] All input splits are pairwise disjoint.")

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------
    all_ids: list[int] = sorted({id_ for ids in splits for id_ in ids})
    print(f"[Merge] {' + '.join(str(len(s)) for s in splits)} = {len(all_ids)} total IDs")

    out_dir.mkdir(parents=True, exist_ok=True)
    save_ids(out_dir / "train_ids.json", all_ids)

    stats = {
        "total_ids": len(all_ids),
        "source_split_dir": str(split_dir),
        "sources": {fname: len(ids) for fname, ids in per_split.items()},
    }
    (out_dir / "merge_stats.json").write_text(json.dumps(stats, indent=2))

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    print()
    print("[Written]")
    print(f"  {out_dir / 'train_ids.json'}   ({len(all_ids)} IDs)")
    print(f"  {out_dir / 'merge_stats.json'}")
    print()
    print("[Next step] run train_adapter.py with:")
    print(f"  --split_dir  {out_dir}")
    print( "  --split      train")
    print( "  --val_split  train   # re-uses train_ids.json; no separate val file needed")
    print( "  --val_every  999999  # larger than --steps → validation never fires")


if __name__ == "__main__":
    main()