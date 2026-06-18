import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from detection.models.adaface.loader import load_adaface_ir50
from detection.data.preprocessing import transform

def build_transform(image_size: int = 112):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

def l2_normalize(vectors: np.ndarray, axis: int=1, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=axis, keepdims=True)
    norms = np.maximum(norms, eps)
    return vectors / norms

def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1: Extract face embeddings.")
    parser.add_argument("--index-csv", type=Path, required=True, help="Index CSV from stage 0.")
    parser.add_argument("--out_dir", type=Path, default=Path("artifacts"), help="Output directory.")
    parser.add_argument("--ckpt", type=Path, default=Path("pretrained/adaface_ir50_ms1mv2.ckpt"))
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--dim", type=int, default=512, help="Embedding dimension.")
    parser.add_argument("--normalize", action="store_true", help="L2-normalize embeddings.")
    args = parser.parse_args()

    with args.index_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit("Index CSV is empty.")

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    model = load_adaface_ir50(args.ckpt, device)

    embeddings = np.zeros((len(rows), args.dim), dtype=np.float32)
    meta_rows = []

    batch_imgs = []
    batch_indices = []

    for i, row in enumerate(rows):
        image_path = row["image_path"]
        img = Image.open(image_path).convert("RGB")
        tensor = transform(img)
        batch_imgs.append(tensor)
        batch_indices.append(i)

        if len(batch_imgs) == args.batch_size or i == len(rows) - 1:
            batch = torch.stack(batch_imgs).to(device)
            with torch.no_grad():
                feats = model(batch).float().cpu().numpy()
            embeddings[batch_indices] = feats
            batch_imgs = []
            batch_indices = []

        meta_rows.append({
            "identity_id": int(row["identity_id"]),
            "image_path": image_path,
            "local_index": int(row["local_index"]),
        })

    if args.normalize:
        embeddings = l2_normalize(embeddings)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    emb_path = args.out_dir / "embeddings.npy"
    meta_path = args.out_dir / "emb_meta.csv"
    np.save(emb_path, embeddings)
    with meta_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["identity_id", "image_path", "local_index"])
        writer.writeheader()
        for row in meta_rows:
            writer.writerow(row)

    manifest = {
        "name": "extract_embeddings_adaface",
        "params": {
            "ckpt": str(args.ckpt),
            "device": device,
            "batch_size": args.batch_size,
            "dim": args.dim,
            "normalize": args.normalize,
        },
        "inputs": {"index_csv": str(args.index_csv)},
        "outputs": {"embeddings": str(emb_path), "emb_meta": str(meta_path)},
    }
    with (args.out_dir / "manifest_embeddings.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()