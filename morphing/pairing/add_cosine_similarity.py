#!/usr/bin/env python3
"""
Step 6 — Add cosine_similarity column to random morph pairlists.

Computes prototype embeddings (mean over all images per identity) using the
AdaFace IR-50 backbone, then appends a cosine_similarity column to each pairlist.

If --embeddings_cache points to an existing .npy file (dict id -> 1-D embedding),
those are reused and only missing identities trigger model inference.

Usage:
  python add_cosine_similarity.py \
    --rand_out_root $RAND_OUT_ROOT \
    --dataset_root  $DATASET_ROOT \
    --model_path    /path/to/adaface_ir50.ckpt \
    --embeddings_cache $SIM_CACHE   # optional
"""

import os
import csv
import argparse
import numpy as np
from pathlib import Path

IMG_EXTS = (".jpg", ".jpeg", ".png")


def load_cache(path, dataset_root=None):
    p = Path(path)
    if not p.exists():
        return {}
    data = np.load(p, allow_pickle=True)
    # Plain float array (N, D): rows are in sorted identity folder order
    if data.ndim == 2 and np.issubdtype(data.dtype, np.floating):
        if dataset_root is None:
            raise RuntimeError(
                "prototypes.npy is a plain (N, D) array — need --dataset_root to map rows to identity IDs."
            )
        identities = sorted(
            d for d in os.listdir(dataset_root)
            if Path(dataset_root, d).is_dir()
        )
        if len(identities) != data.shape[0]:
            raise RuntimeError(
                f"Row count mismatch: {data.shape[0]} embeddings vs {len(identities)} identity folders."
            )
        result = {identity: data[i] for i, identity in enumerate(identities)}
        print(f"Loaded {len(result)} cached embeddings (float array) from {p}")
        return result
    # Dict wrapped in 0-d object array
    result = data.item()
    print(f"Loaded {len(result)} cached embeddings (dict) from {p}")
    return result


def cosine_sim(a, b):
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))


def compute_prototypes(dataset_root, identities, model, transform, device, batch_size=64):
    import torch
    from PIL import Image as PILImage

    prototypes = {}
    model.eval()
    with torch.no_grad():
        for identity in identities:
            folder = Path(dataset_root) / identity
            img_paths = sorted(
                str(folder / f) for f in os.listdir(folder)
                if f.lower().endswith(IMG_EXTS)
            )
            if not img_paths:
                continue

            embeddings = []
            for i in range(0, len(img_paths), batch_size):
                tensors = []
                for p in img_paths[i:i + batch_size]:
                    try:
                        img = PILImage.open(p).convert("RGB")
                        tensors.append(transform(img))
                    except Exception:
                        continue
                if not tensors:
                    continue
                batch = __import__("torch").stack(tensors).to(device)
                feats, _ = model(batch)
                embeddings.append(feats.cpu().numpy())

            if embeddings:
                all_emb = np.concatenate(embeddings, axis=0)
                prototypes[identity] = all_emb.mean(axis=0)

    return prototypes


def add_column(pairlist_path, prototypes):
    pairlist_path = Path(pairlist_path)
    rows = []
    with open(pairlist_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    if "cosine_similarity" in fieldnames:
        print(f"  {pairlist_path.name}: cosine_similarity already present, skipping")
        return

    missing = 0
    for row in rows:
        id_a = f"{int(row['idA']):05d}"
        id_b = f"{int(row['idB']):05d}"
        if id_a in prototypes and id_b in prototypes:
            row["cosine_similarity"] = f"{cosine_sim(prototypes[id_a], prototypes[id_b]):.6f}"
        else:
            row["cosine_similarity"] = ""
            missing += 1

    new_fieldnames = fieldnames + ["cosine_similarity"]
    tmp = pairlist_path.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=new_fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(pairlist_path)

    warn = f"  WARNING: {missing} rows missing embeddings" if missing else ""
    print(f"  {pairlist_path.name}: {len(rows)} rows updated{warn}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rand_out_root", required=True,
                    help="random_morphs_3way root (contains train/val/test subdirs)")
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--model_path", default="",
                    help="AdaFace IR-50 checkpoint. Not needed if cache covers all identities.")
    ap.add_argument("--embeddings_cache", default="",
                    help="Optional .npy file: dict mapping identity_str -> 1-D embedding. "
                         "Reuses embeddings from the similarity-driven pipeline if available.")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    rand_out_root = Path(args.rand_out_root)
    pairlists = {
        split: rand_out_root / f"pairlist_{split}.csv"
        for split in ("train", "val", "test")
    }

    needed_ids = set()
    for pl in pairlists.values():
        if not pl.exists():
            continue
        with open(pl, "r", newline="") as f:
            for row in csv.DictReader(f):
                needed_ids.add(f"{int(row['idA']):05d}")
                needed_ids.add(f"{int(row['idB']):05d}")

    print(f"Need embeddings for {len(needed_ids)} unique identities")

    prototypes = load_cache(args.embeddings_cache, args.dataset_root) if args.embeddings_cache else {}
    missing_ids = needed_ids - set(prototypes)

    if missing_ids:
        if not args.model_path:
            raise RuntimeError(
                f"{len(missing_ids)} identities have no cached embeddings "
                "and --model_path was not provided."
            )
        print(f"Running AdaFace IR-50 inference for {len(missing_ids)} identities...")

        import torch
        import torchvision.transforms as T

        device = torch.device(args.device if torch.cuda.is_available() else "cpu")

        try:
            from detection.models.adaface import net as adaface_net
        except ImportError:
            raise ImportError(
                "Cannot import the AdaFace 'net' module. "
                "Run this script from the repo root so 'detection' is importable."
            )

        ckpt = torch.load(args.model_path, map_location="cpu")
        raw_sd = ckpt.get("state_dict", ckpt)
        sd = {
            (k[len("module."):] if k.startswith("module.") else k): v
            for k, v in raw_sd.items()
        }
        model = adaface_net.build_model("ir_50")
        model.load_state_dict(sd)
        model = model.to(device).eval()

        transform = T.Compose([
            T.Resize((112, 112)),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        new_protos = compute_prototypes(
            args.dataset_root, sorted(missing_ids), model, transform, device
        )
        prototypes.update(new_protos)
        print(f"Computed {len(new_protos)} new embeddings ({len(prototypes)} total)")

        cache_out = (
            Path(args.embeddings_cache) if args.embeddings_cache
            else rand_out_root / "prototype_embeddings_random.npy"
        )
        np.save(cache_out, prototypes)
        print(f"Saved extended cache to {cache_out}")

    for split, pl in pairlists.items():
        if not pl.exists():
            print(f"  Skipping {split}: pairlist not found at {pl}")
            continue
        print(f"[{split}]")
        add_column(pl, prototypes)

    print("\nDone.")


if __name__ == "__main__":
    main()