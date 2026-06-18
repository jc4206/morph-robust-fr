import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

def load_knn(path: Path) -> Dict[int, List[Tuple[int, float, int]]]:
    """
    Load kNN CSV (src_id, nbr_id, score, rank) into a dict:
    {src_id: [(nbr_id, score, rank), ...]} sorted by rank then score then id.
    """
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    by_src: Dict[int, List[Tuple[int, float, int]]] = {}
    for row in rows:
        src = int(row["src_id"])
        nbr = int(row["nbr_id"])
        score = float(row["score"])
        rank = int(row["rank"])
        by_src.setdefault(src, []).append((nbr, score, rank))

    for src, items in by_src.items():
        items.sort(key=lambda x: (x[2], -x[1], x[0]))
    return by_src


def main() -> None:
    """
    Stage 5: Sample morph pairs from kNN lists within a split.
    Output pairs_{split}.csv with id_a, id_b, score (+ optional src_id).
    """
    parser = argparse.ArgumentParser(description="Stage 5: Sample morph pairs from kNN lists.")
    parser.add_argument("--knn", type=Path, required=True, help="kNN CSV file.")
    parser.add_argument("--split-ids", type=Path, required=True, help="Split IDs JSON.")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts"), help="Output directory.")
    parser.add_argument("--split-name", type=str, default= "train", help="Split name for output filename.")
    parser.add_argument("--topk", type=int, default=50,  help="Candidate pool size (top-k neighbors).")
    parser.add_argument("--pairs-per-id", type=int, default=3, help="Number of partners to sample per identity.")
    parser.add_argument("--strategy", choices=["uniform", "weighted"], default="uniform")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--include-source", action="store_true", help="Include src_id column in output.")
    parser.add_argument("--max-degree", type=int, default=0, help="Max pairs per identity (0 = unlimited).")
    args = parser.parse_args()

    knn = load_knn(args.knn)
    split_ids = sorted(set(int(i) for i in json.loads(args.split_ids.read_text())))

    rng = np.random.default_rng(args.seed)
    seen_pairs = set()
    degree = {identity_id: 0 for identity_id in split_ids}

    rows = []
    skipped_no_candidates = 0
    skipped_max_degree = 0
    duplicate_drops = 0

    for src_id in split_ids:
        if args.max_degree and degree[src_id] >= args.max_degree:
            skipped_max_degree += 1
            continue
        candidates = knn.get(src_id, [])[: args.topk]
        if not candidates:
            skipped_no_candidates += 1
            continue

        nbr_ids = np.array([c[0] for c in candidates], dtype=int)
        scores = np.array([c[1] for c in candidates], dtype=float)

        if args.strategy == "weighted":
            weights = np.maximum(scores, 0.0)
            if weights.sum() == 0:
                weights = np.ones_like(weights)
            probs = weights / weights.sum()
        else:
            probs = None

        selected = []
        attempts = 0
        max_attempts = len(nbr_ids) * 2

        while len(selected) < args.pairs_per_id and attempts < max_attempts:
            attempts += 1
            if probs is None:
                idx = rng.integers(0, len(nbr_ids))
            else:
                idx = int(rng.choice(len(nbr_ids), p=probs))

            partner = int(nbr_ids[idx])
            pair = (src_id, partner) if src_id <= partner else (partner, src_id)

            if args.max_degree:
                if degree[src_id] >= args.max_degree or degree[partner] >= args.max_degree:
                    skipped_max_degree += 1
                    continue

            if pair in seen_pairs:
                duplicate_drops += 1
                continue

            seen_pairs.add(pair)
            degree[src_id] += 1
            degree[partner] += 1
            selected.append((partner, float(scores[idx])))

        for partner, score in selected:
            row = {
                "id_a": min(src_id, partner),
                "id_b": max(src_id, partner),
                "score": score,
            }
            if args.include_source:
                row["src_id"] = src_id
            rows.append(row)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"pairs_{args.split_name}.csv"
    stats_path = args.out_dir / f"pairs_stats_{args.split_name}.json"

    fieldnames = ["id_a", "id_b", "score"] + (["src_id"] if args.include_source else [])
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    stats = {
        "n_pairs": len(rows),
        "pairs_per_id": args.pairs_per_id,
        "topk": args.topk,
        "strategy": args.strategy,
        "max_degree": args.max_degree,
        "duplicates_dropped": duplicate_drops,
        "skipped_no_candidates": skipped_no_candidates,
        "skipped_max_degree": skipped_max_degree,
    }
    stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True))

    manifest = {
        "name": "sample_pairs",
        "params": {
            "pairs_per_id": args.pairs_per_id,
            "topk": args.topk,
            "strategy": args.strategy,
            "seed": args.seed,
            "max_degree": args.max_degree,
        },
        "inputs": {"knn": str(args.knn), "split_ids": str(args.split_ids)},
        "outputs": {"pairs": str(out_path), "stats": str(stats_path)},
    }
    (args.out_dir / f"manifest_pairs_{args.split_name}.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True)
    )



if __name__ == "__main__":
    main()