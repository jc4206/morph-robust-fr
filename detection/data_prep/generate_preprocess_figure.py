"""
Generate a 4-panel thesis figure illustrating the MTCNN-based face alignment
and cropping pipeline used as input to the AdaFace IR-50 backbone.

Panels:
  (a) Raw input          — centre-cropped original
  (b) MTCNN landmarks    — same crop + 5 landmark dots + eye-axis line
  (c) Pose alignment     — rotation-only warp (eye axis levelled, full scale)
  (d) 112×112 crop       — full similarity-transform output (AdaFace input)

Usage:
  python generate_preprocess_figure.py --image  <path/to/face.jpg>  [--out_dir ...]
  python generate_preprocess_figure.py --data_dir <dir/>             [--out_dir ...]
"""

import subprocess
import sys


def _ensure(pkg, import_name=None):
    name = import_name or pkg
    try:
        __import__(name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])


_ensure("facenet-pytorch", "facenet_pytorch")
_ensure("Pillow", "PIL")
_ensure("matplotlib")
_ensure("numpy")
_ensure("opencv-python-headless", "cv2")
_ensure("scikit-image", "skimage")

import argparse
import glob
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image
import torch
from facenet_pytorch import MTCNN
from skimage.transform import SimilarityTransform

# ---------------------------------------------------------------------------
# Canonical 112×112 reference landmarks (InsightFace / ArcFace / AdaFace)
# ---------------------------------------------------------------------------
REFERENCE_112 = np.array([
    [38.2946, 51.6963],   # left eye
    [73.5318, 51.5014],   # right eye
    [56.0252, 71.7366],   # nose tip
    [41.5493, 92.3655],   # left mouth
    [70.7299, 92.2041],   # right mouth
], dtype=np.float32)

LM_COLORS = ["#00FF00", "#00FF00", "#FF9900", "#00BFFF", "#00BFFF"]

# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def detect_landmarks(img_pil: Image.Image):
    """Return (lm5 float32 (5,2), box float32 (4,)) or (None, None)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mtcnn = MTCNN(keep_all=False, device=device, min_face_size=40)
    boxes, _, landmarks = mtcnn.detect(img_pil, landmarks=True)
    if landmarks is None or len(landmarks) == 0:
        return None, None
    lm5 = np.array(landmarks[0], dtype=np.float32)
    box = np.array(boxes[0], dtype=np.float32)   # [x1, y1, x2, y2]
    return lm5, box


def load_single(path: str):
    img_pil = Image.open(path).convert("RGB")
    lm5, box = detect_landmarks(img_pil)
    if lm5 is None:
        sys.exit(f"MTCNN detected no face in {path}")
    return img_pil, lm5, box


def find_in_dir(data_dir: str, max_tries: int = 10):
    candidates = sorted(
        glob.glob(os.path.join(data_dir, "*.jpg")) +
        glob.glob(os.path.join(data_dir, "*.png"))
    )
    if not candidates:
        sys.exit(f"No .jpg/.png files found in {data_dir}")
    tried = []
    for path in candidates[:max_tries]:
        tried.append(path)
        img_pil = Image.open(path).convert("RGB")
        lm5, box = detect_landmarks(img_pil)
        if lm5 is not None:
            return img_pil, path, lm5, box
    sys.exit(
        f"MTCNN found no face in {len(tried)} files:\n  "
        + "\n  ".join(os.path.basename(p) for p in tried)
    )

# ---------------------------------------------------------------------------
# Alignment helpers (cv2-based)
# ---------------------------------------------------------------------------

def _similarity_matrix(lm5: np.ndarray) -> np.ndarray:
    """2×3 affine matrix mapping lm5 → REFERENCE_112."""
    tform = SimilarityTransform()
    tform.estimate(lm5, REFERENCE_112)
    return tform.params[:2, :].astype(np.float64)


def align_112(img_bgr: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Apply similarity transform M → 112×112 output (panel d)."""
    return cv2.warpAffine(img_bgr, M, (112, 112), flags=cv2.INTER_LINEAR)


def rotation_warp(img_bgr: np.ndarray, lm5: np.ndarray):
    """
    Rotation-only warp: level the eye axis, keep original canvas size (panel c).

    Returns (warped_bgr, angle_deg).
    angle_deg > 0  → original face tilted CW (right eye lower), corrected CCW.
    """
    eye_left  = lm5[0]
    eye_right = lm5[1]
    dx = float(eye_right[0] - eye_left[0])
    dy = float(eye_right[1] - eye_left[1])

    # Angle of the eye axis from horizontal, in y-down image coords.
    # arctan2(dy, dx) > 0 when right eye is lower (CW tilt on screen).
    # cv2.getRotationMatrix2D with this positive angle applies a CCW rotation
    # that visually levels the eyes.
    angle_deg = float(np.degrees(np.arctan2(dy, dx)))

    cx = float((eye_left[0] + eye_right[0]) / 2)
    cy = float((eye_left[1] + eye_right[1]) / 2)
    H, W = img_bgr.shape[:2]
    M_rot = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    warped = cv2.warpAffine(
        img_bgr, M_rot, (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return warped, angle_deg


def centre_crop_rgb(img_rgb: np.ndarray) -> tuple:
    """Return (square_crop, off_x, off_y)."""
    H, W = img_rgb.shape[:2]
    CROP = min(H, W)
    off_y = (H - CROP) // 2
    off_x = (W - CROP) // 2
    return img_rgb[off_y:off_y + CROP, off_x:off_x + CROP], off_x, off_y

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(img_pil: Image.Image, lm5: np.ndarray, box: np.ndarray, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    img_rgb = np.array(img_pil)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # --- alignment ---
    M_sim                  = _similarity_matrix(lm5)
    img_rot_bgr, angle_deg = rotation_warp(img_bgr, lm5)
    img_112_bgr            = align_112(img_bgr, M_sim)

    print(f"Eye-axis angle from horizontal : {angle_deg:+.2f}°")
    print(f"Rotation applied to panel (c)  : {angle_deg:+.2f}° CCW")

    # --- display crops ---
    crop_raw, off_x, off_y = centre_crop_rgb(img_rgb)
    crop_rot, _, _         = centre_crop_rgb(cv2.cvtColor(img_rot_bgr, cv2.COLOR_BGR2RGB))
    img_112_rgb            = cv2.cvtColor(img_112_bgr, cv2.COLOR_BGR2RGB)

    # Landmarks and bounding box in crop coordinates
    lm_crop = lm5 - np.array([[off_x, off_y]], dtype=np.float32)
    bx1, by1, bx2, by2 = (box[0] - off_x, box[1] - off_y,
                           box[2] - off_x, box[3] - off_y)

    # --- layout ---
    CROP    = crop_raw.shape[0]
    panel_w = 1.0
    ratio_d = 0.45 * panel_w   # panel (d) ~45% width — smaller but readable
    arrow_w = 0.15

    fig = plt.figure(figsize=(15, 4.2), facecolor="white")
    widths = [panel_w, arrow_w, panel_w, arrow_w, panel_w, arrow_w, ratio_d]
    gs = GridSpec(1, 7, figure=fig, width_ratios=widths,
                  left=0.02, right=0.98, top=0.88, bottom=0.02, wspace=0.0)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 2])
    ax_c = fig.add_subplot(gs[0, 4])
    ax_d = fig.add_subplot(gs[0, 6])
    arrow_axes = [
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 3]),
        fig.add_subplot(gs[0, 5]),
    ]

    font_title  = {"fontname": "serif", "fontsize": 11}
    ARROW_COLOR = "#333333"

    # (a) raw
    ax_a.imshow(crop_raw)
    ax_a.set_title("(a)  Raw input", pad=5, **font_title)

    # (b) landmarks
    ax_b.imshow(crop_raw)
    ax_b.set_title("(b)  MTCNN landmarks", pad=5, **font_title)
    ax_b.add_patch(plt.Rectangle(
        (bx1, by1), bx2 - bx1, by2 - by1,
        linewidth=2.5, edgecolor="white", facecolor="none", zorder=3,
    ))
    for (x, y), color in zip(lm_crop, LM_COLORS):
        ax_b.plot(x, y, "o", color=color, markersize=7,
                  markeredgecolor="black", markeredgewidth=0.6, zorder=5)
    ax_b.plot(
        [lm_crop[0, 0], lm_crop[1, 0]],
        [lm_crop[0, 1], lm_crop[1, 1]],
        linestyle="--", color="red", linewidth=1.8, zorder=4,
    )
    legend_elems = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#00FF00",
               markeredgecolor="k", markeredgewidth=0.6, markersize=7, label="Eye"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#FF9900",
               markeredgecolor="k", markeredgewidth=0.6, markersize=7, label="Nose"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#00BFFF",
               markeredgecolor="k", markeredgewidth=0.6, markersize=7, label="Mouth"),
        Line2D([0], [0], linestyle="--", color="red", linewidth=1.5, label="Eye axis"),
        plt.Rectangle((0, 0), 1, 1, linewidth=1.5,
                       edgecolor="#333333", facecolor="white", label="Detection box"),
    ]
    ax_b.legend(handles=legend_elems, loc="lower left", fontsize=7,
                framealpha=0.75, edgecolor="grey", fancybox=False)

    # (c) rotation corrected
    ax_c.imshow(crop_rot)
    ax_c.set_title("(c)  Pose alignment", pad=5, **font_title)

    # (d) 112×112 — upscaled to panel size, nearest-neighbour to avoid blur
    ax_d.imshow(img_112_rgb, interpolation="nearest")
    ax_d.set_title(r"(d)  $112\!\times\!112$ crop", pad=5, **font_title)

    for ax in (ax_a, ax_b, ax_c, ax_d):
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    for ax in arrow_axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.annotate(
            "",
            xy=(0.92, 0.5), xytext=(0.08, 0.5),
            xycoords="axes fraction", textcoords="axes fraction",
            arrowprops=dict(arrowstyle="-|>", color=ARROW_COLOR,
                            lw=1.6, mutation_scale=16),
        )

    pdf_path = os.path.join(out_dir, "preprocess_pipeline.pdf")
    png_path = os.path.join(out_dir, "preprocess_pipeline.png")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white", dpi=300)
    fig.savefig(png_path, bbox_inches="tight", facecolor="white", dpi=300)
    plt.close(fig)
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Generate preprocessing pipeline figure.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--image",    type=str,
                     help="Path to a single face image (.jpg / .png)")
    src.add_argument("--data_dir", type=str,
                     help="Directory of face images — first MTCNN detection is used")
    ap.add_argument("--out_dir", type=str,
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures"),
                    help="Output directory (default: <script_dir>/figures)")
    args = ap.parse_args()

    if args.image:
        print(f"Loading: {args.image}")
        img_pil, lm5, box = load_single(args.image)
        print(f"Face detected in: {os.path.basename(args.image)}")
    else:
        print(f"Scanning: {args.data_dir}")
        img_pil, chosen, lm5, box = find_in_dir(args.data_dir)
        print(f"Face detected in: {os.path.basename(chosen)}")

    make_figure(img_pil, lm5, box, args.out_dir)


if __name__ == "__main__":
    main()