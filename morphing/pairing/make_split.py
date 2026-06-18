import json
import argparse
from pathlib import Path
import pandas as pd
import random

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--col_a", type=str, default="id1")
    ap.add_argument("--col_b", type=str, default="id2")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.pairs_csv)

    # Collect IDs
    ids = sorted(set(df[args.col_a].astype(int).tolist()) | set(df[args.col_b].astype(int).tolist()))
    rng = random.Random(args.seed)
    rng.shuffle(ids)

    n_val = int(len(ids) * args.val_ratio)
    val_ids = set(ids[:n_val])
    train_ids = set(ids[n_val:])

    # canonicalize pairs + filtering
    def canon(a, b):
        a = int(a); b = int(b)
        return (a, b) if a < b else (b, a)

    seen = set()
    train_rows = []
    val_rows = []
    cross = 0

    for _, r in df.iterrows():
        a, b = canon(r[args.col_a], r[args.col_b])
        if (a, b) in seen:
            continue
        seen.add((a, b))

        in_train = (a in train_ids) and (b in train_ids)
        in_val = (a in val_ids) and (b in val_ids)

        if in_train:
            train_rows.append((a, b))
        elif in_val:
            val_rows.append((a, b))
        else:
            cross += 1


    # write outputs
    (out_dir / "train_ids.json").write_text(json.dumps(sorted(train_ids)))
    (out_dir / "val_ids.json").write_text(json.dumps(sorted(val_ids)))

    pd.DataFrame(train_rows, columns=["id_a", "id_b"]).to_csv(out_dir / "train_pairs.csv", index=False)
    pd.DataFrame(val_rows, columns=["id_a", "id_b"]).to_csv(out_dir / "val_pairs.csv", index=False)

    stats = {
        "n_ids_total": len(ids),
        "n_train_ids": len(train_ids),
        "n_val_ids": len(val_ids),
        "pairs_total_unique": len(seen),
        "pairs_train": len(train_rows),
        "pairs_val": len(val_rows),
        "pairs_cross_dropped": cross,
        "effective_val_pair_ratio": (len(val_rows) / max(1, (len(train_rows)+len(val_rows))))
    }
    (out_dir / "split_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))

if __name__ == "__main__":
    main()
