import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np


def random_split(ids: List[int], val_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    rng = np.random.default_rng(seed)
    ids_arr = np.array(ids, dtype=int)
    rng.shuffle(ids_arr)
    n_val = int(round(len(ids_arr) * val_ratio))
    val_ids = sorted(ids_arr[:n_val].tolist())
    train_ids = sorted(ids_arr[n_val:].tolist())
    return train_ids, val_ids


def cluster_split(ids: List[int], prototypes: np.ndarray, val_ratio: float, seed: int, n_clusters: int):
    try:
        from sklearn.cluster import KMeans
    except Exception as exc:
        raise SystemExit("scikit-learn is required for --method cluster") from exc

    rng = np.random.RandomState(seed)
    kmeans = KMeans(n_clusters=n_clusters, random_state=rng, n_init=10)
    labels = kmeans.fit_predict(prototypes)

    id_to_label = {identity_id: int(label) for identity_id, label in zip(ids, labels)}
    train_ids: List[int] = []
    val_ids: List[int] = []
    rng = np.random.RandomState(seed)

    for label in sorted(set(labels)):
        cluster_ids = [i for i in ids if id_to_label[i] == label]
        rng.shuffle(cluster_ids)
        n_val = int(round(len(cluster_ids) * val_ratio))
        val_ids.extend(cluster_ids[:n_val])
        train_ids.extend(cluster_ids[n_val:])

    return sorted(train_ids), sorted(val_ids)


def cluster_holdout_split(ids: List[int], prototypes: np.ndarray, val_ratio: float, seed: int, n_clusters: int):
    """
    Split by holding out entire clusters: a fraction of clusters go to val.
    Example: val_ratio=0.2 and n_clusters=100 -> ~20 clusters in val, 80 in train.
    """
    try:
        from sklearn.cluster import KMeans
    except Exception as exc:
        raise SystemExit("scikit-learn is required for --method cluster_holdout") from exc

    rng = np.random.RandomState(seed)
    kmeans = KMeans(n_clusters=n_clusters, random_state=rng, n_init=10)
    labels = kmeans.fit_predict(prototypes)

    unique_labels = sorted(set(int(l) for l in labels))
    rng.shuffle(unique_labels)
    n_val_clusters = int(round(len(unique_labels) * val_ratio))
    val_clusters = set(unique_labels[:n_val_clusters])

    train_ids: List[int] = []
    val_ids: List[int] = []
    for identity_id, label in zip(ids, labels):
        if int(label) in val_clusters:
            val_ids.append(identity_id)
        else:
            train_ids.append(identity_id)

    return sorted(train_ids), sorted(val_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 3: Make identity-disjoint train/val split.")
    parser.add_argument("--ids", type=Path, required=True, help="JSON list of identity IDs.")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts"), help="Output directory.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed.")
    parser.add_argument("--method", choices=["random", "cluster", "cluster_holdout"], default="random")
    parser.add_argument("--prototypes", type=Path, help="Prototypes .npy (required for cluster method).")
    parser.add_argument("--id_map", type=Path, help="ID map JSON aligned with prototypes.")
    parser.add_argument("--n-clusters", type=int, default=100, help="Number of clusters for cluster split.")
    args = parser.parse_args()

    ids = sorted(set(int(i) for i in json.loads(args.ids.read_text())))

    if args.method == "random":
        train_ids, val_ids = random_split(ids, args.val_ratio, args.seed)
    elif args.method == "cluster":
        if not args.prototypes or not args.id_map:
            raise SystemExit("--prototypes and --id-map are required for cluster split")
        proto_ids = sorted(set(int(i) for i in json.loads(args.id_map.read_text())))
        if proto_ids != ids:
            raise SystemExit("IDs in id_map do not match provided ids.")
        prototypes = np.load(args.prototypes)
        train_ids, val_ids = cluster_split(ids, prototypes, args.val_ratio, args.seed, args.n_clusters)
    else:
        if not args.prototypes or not args.id_map:
            raise SystemExit("--prototypes and --id-map are required for cluster_holdout split")
        proto_ids = sorted(set(int(i) for i in json.loads(args.id_map.read_text())))
        if proto_ids != ids:
            raise SystemExit("IDs in id_map do not match provided ids.")
        prototypes = np.load(args.prototypes)
        train_ids, val_ids = cluster_holdout_split(ids, prototypes, args.val_ratio, args.seed, args.n_clusters)

    if set(train_ids) & set(val_ids):
        raise SystemExit("Split is not identity-disjoint.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.out_dir / "train_ids.json"
    val_path = args.out_dir / "val_ids.json"
    stats_path = args.out_dir / "split_stats.json"

    train_path.write_text(json.dumps(train_ids, indent=2, sort_keys=True))
    val_path.write_text(json.dumps(val_ids, indent=2, sort_keys=True))
    stats = {
        "n_ids": len(ids),
        "n_train": len(train_ids),
        "n_val": len(val_ids),
        "val_ratio": args.val_ratio,
        "method": args.method,
        "n_clusters": args.n_clusters if args.method in {"cluster", "cluster_holdout"} else None,
    }
    stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True))

    manifest = {
        "name": "make_split_ids",
        "params": {
            "val_ratio": args.val_ratio,
            "seed": args.seed,
            "method": args.method,
            "n_clusters": args.n_clusters if args.method in {"cluster", "cluster_holdout"} else None,
        },
        "inputs": {"ids": str(args.ids)},
        "outputs": {"train_ids": str(train_path), "val_ids": str(val_path), "stats": str(stats_path)},
    }
    (args.out_dir / "manifest_split.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


