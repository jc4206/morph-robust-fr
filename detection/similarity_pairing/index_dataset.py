import argparse
import csv
import json
from pathlib import Path

from detection.data.index_parser import build_real_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 0: Index dataset images by identity.")
    parser.add_argument("--real_root", type=Path, required=True, help="Root folder with identity subfolders.")
    parser.add_argument("--out_dir", type=Path, default=Path("artifacts"), help="Output directory.")
    args = parser.parse_args()

    real_index = build_real_index(args.real_root)
    ids = sorted(set(int(i) for i in real_index.keys()))

    rows = []
    for identity_id in ids:
        for local_idx, path in enumerate(real_index[identity_id]):
            rows.append({
                "identity_id": identity_id,
                "image_path": str(path),
                "local_index": local_idx,
            })

    args.out_dir.mkdir(parents=True, exist_ok=True)
    index_path = args.out_dir / "index.csv"
    ids_path = args.out_dir / "ids.json"

    with index_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["identity_id", "image_path", "local_index"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    with ids_path.open("w", encoding="utf-8") as f:
        json.dump(ids, f, indent=2, sort_keys=True)

    manifest = {
        "name": "index_dataset",
        "params": {},
        "inputs": {"real_root": str(args.real_root)},
        "outputs": {"index_csv": str(index_path), "ids_json": str(ids_path)},
    }
    with (args.out_dir / "manifest_index.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
