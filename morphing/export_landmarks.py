# export_landmarks.py
# Computes dlib 68-point landmarks (+4 corners) for ALL images in identity folders.
# Optional: restrict to a split (train/val) via --split-ids and write to out_root/<split-name>/...

import os
import argparse
import json
import re
import numpy as np
from typing import Optional, List
from concurrent.futures import ProcessPoolExecutor, as_completed

IMG_EXTS = (".jpg", ".jpeg", ".png")
ID_DIR_RE = re.compile(r"^\d{5}$")


def is_image_filename(name: str) -> bool:
    return name.lower().endswith(IMG_EXTS)


def list_images_in_folder(folder: str) -> List[str]:
    try:
        files = sorted([f for f in os.listdir(folder) if is_image_filename(f)])
    except FileNotFoundError:
        return []
    return [os.path.join(folder, f) for f in files]


def landmark_out_path(img_path: str, out_root: str, dataset_root: str) -> str:
    """
    Mirror structure under out_root:
      dataset_root/00001/img_01.jpg  ->  out_root/00001/img_01.landmarks.npy
    """
    rel = os.path.relpath(img_path, dataset_root)
    rel_no_ext = os.path.splitext(rel)[0]
    return os.path.join(out_root, rel_no_ext + ".landmarks.npy")


def ensure_parent_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def process_one(img_path: str,
                predictor_path: str,
                out_path: str,
                upsample: int,
                min_landmarks: int,
                fallback_full_bbox: bool,
                fallback_inset: float):
    """
    Worker function (runs in subprocess):
      - loads image via dlib
      - detects largest face
      - predicts 68 landmarks
      - optionally falls back to full-image bbox if detector finds no face
      - appends 4 corner points
      - saves .npy
    Returns tuple: (img_path, status, out_path_or_none)
    """
    import dlib  # import inside worker to avoid multiprocessing/fork issues

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    img = dlib.load_rgb_image(img_path)
    H, W = img.shape[0], img.shape[1]

    dets = detector(img, upsample)

    if len(dets) == 0:
        if not fallback_full_bbox:
            return (img_path, "no_face", None)

        # Fallback: use (optionally inset) full image as bbox
        inset = max(0.0, min(float(fallback_inset), 0.45))
        left = int(round(W * inset))
        top = int(round(H * inset))
        right = int(round(W * (1.0 - inset))) - 1
        bottom = int(round(H * (1.0 - inset))) - 1

        left = max(0, min(left, W - 2))
        top = max(0, min(top, H - 2))
        right = max(left + 1, min(right, W - 1))
        bottom = max(top + 1, min(bottom, H - 1))

        d = dlib.rectangle(left=left, top=top, right=right, bottom=bottom)
        shape = predictor(img, d)
        status_prefix = "ok_fallback_bbox"
    else:
        d = max(dets, key=lambda r: (r.right() - r.left()) * (r.bottom() - r.top()))
        shape = predictor(img, d)
        status_prefix = "ok"

    pts = np.array([(p.x, p.y) for p in shape.parts()], dtype=np.float32)
    if pts.shape[0] < min_landmarks:
        return (img_path, "few_landmarks", None)

    corners = np.array([[0, 0], [0, H - 1], [W - 1, 0], [W - 1, H - 1]], dtype=np.float32)
    pts = np.vstack([pts, corners])

    ensure_parent_dir(out_path)
    np.save(out_path, pts)
    return (img_path, status_prefix, out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True,
                    help="Root folder with identity subfolders (00001, 00002, ...)")
    ap.add_argument("--out_root", required=True,
                    help="Where to write landmarks (mirrors folder structure)")
    ap.add_argument("--predictor", required=True,
                    help="Path to shape_predictor_68_face_landmarks.dat")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4,
                    help="Number of parallel processes")
    ap.add_argument("--upsample", type=int, default=1,
                    help="dlib detector upsample (0-2 typical)")
    ap.add_argument("--min_landmarks", type=int, default=60)
    ap.add_argument("--fallback_full_bbox", action="store_true",
                    help="If detector finds no face, run predictor on full-image bbox.")
    ap.add_argument("--fallback_inset", type=float, default=0.0,
                    help="Inset ratio for fallback bbox (e.g., 0.05 shrinks bbox by 5% on each side).")
    ap.add_argument("--skip_existing", action="store_true")
    ap.add_argument("--split-ids", type=str, default=None,
                    help="Optional JSON with IDs for split (train/val).")
    ap.add_argument("--split-name", type=str, default=None,
                    help="Optional split name; if set, outputs go to out_root/<split-name>/")
    args = ap.parse_args()

    dataset_root = os.path.abspath(args.dataset_root)
    out_root = os.path.abspath(args.out_root)
    predictor = os.path.abspath(args.predictor)

    if not os.path.exists(dataset_root):
        raise FileNotFoundError(f"dataset_root not found: {dataset_root}")
    if not os.path.exists(predictor):
        raise FileNotFoundError(f"predictor not found: {predictor}")

    if args.split_name:
        out_root = os.path.join(out_root, args.split_name)

    os.makedirs(out_root, exist_ok=True)

    # collect identity folders
    ids = sorted([d for d in os.listdir(dataset_root)
                  if os.path.isdir(os.path.join(dataset_root, d)) and ID_DIR_RE.match(d)])

    # optional: filter by split ids
    if args.split_ids:
        split_ids = set(int(i) for i in json.loads(Path(args.split_ids).read_text()))
        ids = [d for d in ids if int(d) in split_ids]

    print(f"Found {len(ids)} identity folders in {dataset_root}")

    # choose ALL images per identity
    chosen_imgs = []
    for ident in ids:
        folder = os.path.join(dataset_root, ident)
        imgs = list_images_in_folder(folder)
        chosen_imgs.extend(imgs)

    chosen_imgs.sort()
    print(f"Will compute landmarks for {len(chosen_imgs)} images (all per identity).")

    tasks = []
    for img_path in chosen_imgs:
        out_path = landmark_out_path(img_path, out_root, dataset_root)
        if args.skip_existing and os.path.exists(out_path):
            continue
        tasks.append((img_path, out_path))

    print(f"Tasks to run: {len(tasks)} (skip_existing={args.skip_existing})")
    print(f"Writing landmarks to: {out_root}")
    print(f"Fallback bbox enabled: {args.fallback_full_bbox} (inset={args.fallback_inset})")

    ok = 0
    ok_fallback = 0
    no_face = 0
    few = 0
    failed = 0

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [
            ex.submit(process_one, img_path, predictor, out_path, args.upsample, args.min_landmarks,
                      args.fallback_full_bbox, args.fallback_inset)
            for img_path, out_path in tasks
        ]

        for fut in as_completed(futures):
            try:
                _, status, _ = fut.result()
                if status == "ok":
                    ok += 1
                elif status == "ok_fallback_bbox":
                    ok_fallback += 1
                elif status == "no_face":
                    no_face += 1
                elif status == "few_landmarks":
                    few += 1
                else:
                    failed += 1
            except Exception:
                failed += 1

    print("\nDone.")
    print(f"  ok           : {ok}")
    print(f"  ok_fallback  : {ok_fallback}")
    print(f"  no_face      : {no_face}")
    print(f"  few_landmarks: {few}")
    print(f"  failed       : {failed}")
    print(f"\nLandmarks written under: {out_root}")


if __name__ == "__main__":
    main()



