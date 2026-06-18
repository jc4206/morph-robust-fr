import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

def _knn_faiss(vectors: np.ndarray, ids: List[int], k: int) -> List[Tuple[int, int, float, int]]:
    """
    Compute exact kNN using FAISS (IndexFlatIP) on L2-normalized vectors.
    Returns (src_id, nbr_id, score, rank) rows excluding self
    """
    try:
        import faiss
    except Exception as exc:
        raise ImportError("faiss not available") from exc

    d = vectors.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(vectors.astype(np.float32))
    scores, neighbors = index.search(vectors.astype(np.float32), k + 1)

    rows = []
    id_arr = np.array(ids, dtype=int)
    for i, src_id in enumerate(ids):
        candidates = []
        for score, nbr_idx in zip(scores[i], neighbors[i]):
            nbr_id = int(id_arr[nbr_idx])
            if nbr_id == src_id:
                continue
            candidates.append((nbr_id, float(score)))
        candidates.sort(key=lambda x: (-x[1], x[0]))
        for rank, (nbr_id, score) in enumerate(candidates[:k], start=1):
            rows.append((src_id, nbr_id, score, rank))
    return rows


def _knn_numpy(vectors: np.ndarray, ids: List[int], k: int, block: int = 512) -> List[Tuple[int, int, float, int]]:
    """
    Compute exact kNN using NumPy dot products (blockwise).
    Returns (src_id, nbr_id, score, rank) rows excluding self.
    """
    rows = []
    ids_arr = np.array(ids, dtype=int)
    n = vectors.shape[0]

    for start in range(0, n, block):
        end = min(start + block, n)
        sims = vectors[start:end] @ vectors.T
        for i in range(end - start):
            src_idx = start + i
            src_id = int(ids_arr[src_idx])
            scores = sims[i]
            scores[src_idx] = -np.inf
            order = np.lexsort((ids_arr, -scores))
            top = order[:k]
            for rank, nbr_idx in enumerate(top, start=1):
                rows.append((src_id, int(ids_arr[nbr_idx]), float(scores[nbr_idx]), rank))

    return rows

def _build_knn_for_split(name: str, split_ids: List[int], prototypes: np.ndarray, id_to_idx: Dict[int, int], k: int) -> List[Tuple[int, int, float, int]]:
    """
    Restrict prototypes to split_ids and compute kNN within that split only.
    """
    idxs = [id_to_idx[i] for i in split_ids]
    split_vectors = prototypes[idxs]
    try:
        rows = _knn_faiss(split_vectors, split_ids, k)
    except Exception:
        rows = _knn_numpy(split_vectors, split_ids, k)
    return rows

def main() -> None:
    """
    Stage 4: Compute within-split kNN lists for train, val and test identities
    Outputs knn_train.csv, knn_val.csv and knn_test.csv with src_id, nbr_id, score, rank.
    """
    parser = argparse.ArgumentParser(description="Stage 4: Build kNN lists within each split.")
    parser.add_argument("--prototypes", type=Path, required=True, help="Prototypes .npy.")
    parser.add_argument("--id_map", type=Path, required=True, help="ID map JSON aligned with prototypes.")
    parser.add_argument("--train-ids", type=Path, required=True, help="Train IDs JSON.")
    parser.add_argument("--val-ids", type=Path, required=True, help="Val IDs JSON.")
    parser.add_argument("--test-ids", type=Path, required=True, help="Test IDs JSON.")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts"), help="Output directory.")
    parser.add_argument("--k", type=int, default=50, help="Number of nearest neighbors.")
    args = parser.parse_args()

    prototypes = np.load(args.prototypes)
    ids = sorted(set(int(i) for i in json.loads(args.id_map.read_text())))
    id_to_idx = {identity_id: idx for idx, identity_id in enumerate(ids)}

    train_ids = sorted(set(int(i) for i in json.loads(args.train_ids.read_text())))
    val_ids = sorted(set(int(i) for i in json.loads(args.val_ids.read_text())))
    test_ids = sorted(set(int(i) for i in json.loads(args.test_ids.read_text())))

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    train_rows = _build_knn_for_split("train", train_ids, prototypes, id_to_idx, args.k)
    val_rows = _build_knn_for_split("val", val_ids, prototypes, id_to_idx, args.k)
    test_rows = _build_knn_for_split("test", test_ids, prototypes, id_to_idx, args.k)

    train_path = out_dir / "knn_train.csv"
    val_path = out_dir / "knn_val.csv"
    test_path = out_dir / "knn_test.csv"

    with train_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["src_id", "nbr_id", "score", "rank"])
        writer.writeheader()
        for row in train_rows:
            writer.writerow(dict(zip(["src_id", "nbr_id", "score", "rank"], row)))

    with val_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["src_id", "nbr_id", "score", "rank"])
        writer.writeheader()
        for row in val_rows:
            writer.writerow(dict(zip(["src_id", "nbr_id", "score", "rank"], row)))

    with test_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["src_id", "nbr_id", "score", "rank"])
        writer.writeheader()
        for row in test_rows:
            writer.writerow(dict(zip(["src_id", "nbr_id", "score", "rank"], row)))

    manifest = {
        "name": "build_knn",
        "params": {"k": args.k},
        "inputs": {
            "prototypes": str(args.prototypes),
            "id_map": str(args.id_map),
            "train_ids": str(args.train_ids),
            "val_ids": str(args.val_ids),
            "test_ids": str(args.test_ids),
        },
        "outputs": {"knn_train": str(train_path), "knn_val": str(val_path), "knn_test": str(test_path),
    },
    }
    (out_dir / "manifest_knn.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))



if __name__ == "__main__":
    main() 

