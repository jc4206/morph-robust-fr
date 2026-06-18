import os
import csv
import random
import argparse
import numpy as np
import cv2
from PIL import Image

from morphing.morph_function_final import morph  # morph(imgA, imgB, lmA, lmB, alpha, beta)

IMG_EXTS = (".jpg", ".jpeg", ".png")


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def list_images_for_id(dataset_root: str, identity: str):
    folder = os.path.join(dataset_root, identity)
    if not os.path.isdir(folder):
        return []
    files = sorted([f for f in os.listdir(folder) if f.lower().endswith(IMG_EXTS)])
    return [os.path.join(folder, f) for f in files]


def pick_image(dataset_root: str, identity: str, mode: str, rng: random.Random):
    imgs = list_images_for_id(dataset_root, identity)
    if not imgs:
        return None
    if mode == "first":
        return imgs[0]
    return rng.choice(imgs)


def lm_path_for_image(img_path: str, dataset_root: str, lm_root: str) -> str:
    rel = os.path.relpath(img_path, dataset_root)
    rel_no_ext = os.path.splitext(rel)[0]
    return os.path.join(lm_root, rel_no_ext + ".landmarks.npy")


def load_rgb(img_path: str):
    bgr = cv2.imread(img_path)
    if bgr is None:
        return None
    return bgr[..., ::-1]


def safe_load_landmarks(lm_path: str):
    if not os.path.exists(lm_path):
        return None
    try:
        lm = np.load(lm_path)
    except Exception:
        return None
    if lm is None or lm.ndim != 2 or lm.shape[1] != 2:
        return None
    if lm.shape[0] < 68:
        return None
    return lm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--lm_root", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--pairs_csv", required=True, help="CSV with columns id_a,id_b,score")

    ap.add_argument("--pairlist_csv", default="", help="Optional: write morph pairlist CSV with imgA/imgB paths")

    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=0.5)

    ap.add_argument("--pick", choices=["first", "random"], default="first")
    ap.add_argument("--save_variants", choices=["full", "all"], default="full")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    dataset_root = os.path.abspath(args.dataset_root)
    lm_root = os.path.abspath(args.lm_root)
    out_root = os.path.abspath(args.out_root)
    ensure_dir(out_root)

    # setup pairlist writer
    pairlist_fh = None
    pairlist_writer = None
    if args.pairlist_csv:
        pairlist_path = os.path.abspath(args.pairlist_csv)
        ensure_dir(os.path.dirname(pairlist_path))
        pairlist_fh = open(pairlist_path, "w", newline="")
        pairlist_writer = csv.writer(pairlist_fh)
        pairlist_writer.writerow([
            "idA", "idB", "imgA", "imgB",
            "lmA", "lmB", "alpha", "beta",
            "out_full", "out_inA", "out_inB"
        ])

    # Load pairs
    with open(args.pairs_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        pairs = [(row["id_a"], row["id_b"]) for row in reader]

    # normalize to 5-digit folder names
    pairs = [(f"{int(a):05d}", f"{int(b):05d}") for a, b in pairs]

    total = 0
    skipped = 0

    for idA, idB in pairs:
        imgA_path = pick_image(dataset_root, idA, args.pick, rng)
        imgB_path = pick_image(dataset_root, idB, args.pick, rng)
        if imgA_path is None or imgB_path is None:
            skipped += 1
            continue

        imgA = load_rgb(imgA_path)
        imgB = load_rgb(imgB_path)
        if imgA is None or imgB is None:
            skipped += 1
            continue

        lmA_path = lm_path_for_image(imgA_path, dataset_root, lm_root)
        lmB_path = lm_path_for_image(imgB_path, dataset_root, lm_root)
        lmA = safe_load_landmarks(lmA_path)
        lmB = safe_load_landmarks(lmB_path)
        if lmA is None or lmB is None:
            skipped += 1
            continue

        if imgA.shape != imgB.shape:
            imgB = cv2.resize(imgB, (imgA.shape[1], imgA.shape[0]))

        try:
            morph_full, morph_inA, morph_inB = morph(imgA, imgB, lmA, lmB, args.alpha, args.beta)
        except Exception:
            skipped += 1
            continue

        out_full = os.path.join(out_root, f"{idA}__{idB}_full.png")
        Image.fromarray(morph_full).save(out_full)

        if args.save_variants == "all":
            out_inA = os.path.join(out_root, f"{idA}__{idB}_inA.png")
            out_inB = os.path.join(out_root, f"{idA}__{idB}_inB.png")
            Image.fromarray(morph_inA).save(out_inA)
            Image.fromarray(morph_inB).save(out_inB)
        else:
            out_inA, out_inB = "", ""

        # write pairlist row
        if pairlist_writer:
            pairlist_writer.writerow([
                idA, idB, imgA_path, imgB_path,
                lmA_path, lmB_path, args.alpha, args.beta,
                out_full, out_inA, out_inB
            ])

        total += 1
        if total % 500 == 0:
            print(f"Generated {total} morphs")

    if pairlist_fh:
        pairlist_fh.close()

    print(f"\nDone. Generated {total} morph pairs. Skipped {skipped}.")


if __name__ == "__main__":
    main()
