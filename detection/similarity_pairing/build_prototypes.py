import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict

import numpy as np

def l2_normalize(vectors: np.ndarray, axis: int = 1, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=axis, keepdims=True)
    norms = np.maximum(norms, eps)
    return vectors / norms

def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2: Build identity prototypes.")
    parser.add_argument("--embeddings", type=Path, required=True, help="Embeddings .npy file.")
    parser.add_argument("--emb-meta", type=Path, required=True, help="Embeddings metadata CSV.")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts"), help="Output directory.")
    parser.add_argument("--normalize-input", action="store_true", help="L2-normalize image embeddings before mean.")
    args = parser.parse_args()

    embeddings = np.load(args.embeddings)
    with args.emb_meta.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if embeddings.shape[0] != len(rows):
        raise SystemExit("Embedding count does not match emb_meta rows.")

    by_id = defaultdict(list)
    for idx, row in enumerate(rows):
        identity_id = int(row["identity_id"])
        by_id[identity_id].append(idx)

    ids = sorted(by_id.keys())
    prototypes = np.zeros((len(ids), embeddings.shape[1]), dtype=np.float32)
    norms_before = []

    for i, identity_id in enumerate(ids):
        idxs = by_id[identity_id]
        vecs = embeddings[idxs]
        if args.normalize_input:
            vecs = l2_normalize(vecs)
        mean = vecs.mean(axis=0)
        norms_before.append(float(np.linalg.norm(mean)))
        prototypes[i] = mean

    prototypes = l2_normalize(prototypes)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    proto_path = args.out_dir / "prototypes.npy"
    id_map_path = args.out_dir / "id_map.json"
    stats_path = args.out_dir / "prototype_stats.json"

    np.save(proto_path, prototypes)
    with id_map_path.open("w", encoding="utf-8") as f:
        json.dump(ids, f, indent=2, sort_keys=True)

    stats = {
        "mean_norm_before": float(np.mean(norms_before)),
        "min_norm_before": float(np.min(norms_before)),
        "max_norm_before": float(np.max(norms_before)),
        "n_ids": len(ids),
    }
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, sort_keys=True)

    manifest = {
        "name": "build_prototypes",
        "params": {"normalize_input": args.normalize_input},
        "inputs": {"embeddings": str(args.embeddings), "emb_meta": str(args.emb_meta)},
        "outputs": {
            "prototypes": str(proto_path),
            "id_map": str(id_map_path),
            "stats": str(stats_path)
        },
    }
    with (args.out_dir / "manifest_prototypes.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)



if __name__ == "__main__":
    main()


