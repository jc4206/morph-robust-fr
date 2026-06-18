import argparse
import json
import random
from pathlib import Path

def load_ids(path: Path):
    ids = json.loads(path.read_text())
    return [int(x) for x in ids]

def save_ids(path: Path, ids):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sorted(ids), indent=2))

def main():
    ap = argparse.ArgumentParser(description="Create a 60/20/20 split from existing train/val IDs.")
    ap.add_argument("--train_ids", required=True, help="Path to existing train_ids.json (80%). ")
    ap.add_argument("--val_ids", required=True, help="Path to existing val_ids.json (20%).")
    ap.add_argument("--out_dir", required=True, help="Output directory for new train/val/test jsons.")
    ap.add_argument("--train_ratio", type=float, default=0.60)
    ap.add_argument("--val_ratio", type=float, default=0.20)
    ap.add_argument("--test_ratio", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    if abs((args.train_ratio + args.val_ratio + args.test_ratio) - 1.0) > 1e-8:
        raise ValueError("Ratios must sum to 1.0")

    old_train = load_ids(Path(args.train_ids))
    old_val = load_ids(Path(args.val_ids))

    # Basic integrity checks
    inter = set(old_train).intersection(set(old_val))
    if inter:
        raise RuntimeError(f"Input train/val are not disjoint. Overlap count: {len(inter)}")

    total_ids = len(old_train) + len(old_val)
    target_val = round(total_ids * args.val_ratio)
    target_test = round(total_ids * args.test_ratio)
    target_train = total_ids - target_val - target_test

    # Keep old val fixed (as requested), move some IDs from old train to test.
    if len(old_val) != target_val:
        print(
            f"[WARN] Existing val size={len(old_val)} differs from target_val={target_val}. "
            "Keeping existing val fixed."
        )

    rng = random.Random(args.seed)
    train_pool = list(old_train)
    rng.shuffle(train_pool)

    # Compute test size so final train is as close as possible to target_train
    # while keeping val unchanged.
    new_test_size = total_ids - len(old_val) - target_train
    if new_test_size < 0:
        raise RuntimeError("Computed negative test size. Check ratios/inputs.")
    if new_test_size > len(train_pool):
        raise RuntimeError("Not enough train IDs to create requested test split.")

    test_ids = train_pool[:new_test_size]
    new_train_ids = train_pool[new_test_size:]
    new_val_ids = list(old_val)

    # Final checks
    s_train, s_val, s_test = set(new_train_ids), set(new_val_ids), set(test_ids)
    if s_train & s_val or s_train & s_test or s_val & s_test:
        raise RuntimeError("Output splits are not disjoint.")

    union_out = s_train | s_val | s_test
    union_in = set(old_train) | set(old_val)
    if union_out != union_in:
        raise RuntimeError("Output IDs do not match input ID universe.")

    out_dir = Path(args.out_dir)
    save_ids(out_dir / "train_ids.json", new_train_ids)
    save_ids(out_dir / "val_ids.json", new_val_ids)
    save_ids(out_dir / "test_ids.json", test_ids)

    stats = {
        "total_ids": total_ids,
        "seed": args.seed,
        "ratios_requested": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio,
        },
        "counts": {
            "train": len(new_train_ids),
            "val": len(new_val_ids),
            "test": len(test_ids),
        },
        "ratios_actual": {
            "train": len(new_train_ids) / total_ids,
            "val": len(new_val_ids) / total_ids,
            "test": len(test_ids) / total_ids,
        },
    }
    (out_dir / "split_stats_3way.json").write_text(json.dumps(stats, indent=2))

    print("[OK] Wrote:")
    print(f" {out_dir / 'train_ids.json'}")
    print(f" {out_dir / 'val_ids.json'}")
    print(f" {out_dir / 'test_ids.json'}")
    print(f" {out_dir / 'split_stats_3way.json'}")
    print("[INFO] counts:", stats["counts"])
    print("[INFO] ratios_actual:", stats["ratios_actual"])


if __name__ == "__main__":
    main()