# plot_score_histogram.py
#
# Generates a score-distribution histogram for one model configuration (frozen
# backbone or adapter-augmented) using FRLL/AMSL data.  The three distributions
# (impostor, morph, genuine) and the FMR-calibrated threshold tau are computed
# by run_frll_eval() in eval.py — no scoring logic is reimplemented here.
#
# Run (frozen backbone):
#   python plot_score_histogram.py \
#     --backbone adaface \
#     --base_ckpt <ckpt> \
#     --bonafide_dir <frll_all_4ArcFace/> \
#     --morph_dir <morph_amsl_4ArcFace/> \
#     --output score_hist_backbone.pdf \
#     --title "Frozen Backbone"
#
# Run (bona fide adapter):
#   python plot_score_histogram.py \
#     --backbone adaface \
#     --base_ckpt <ckpt> \
#     --adapter_ckpt <adapter.pt> \
#     --bonafide_dir <frll_all_4ArcFace/> \
#     --morph_dir <morph_amsl_4ArcFace/> \
#     --output score_hist_bonafide.pdf \
#     --title "Bona Fide Adapter"

import matplotlib
matplotlib.use('Agg')   # set before any other matplotlib/pyplot import
import matplotlib.pyplot as plt

import argparse
from pathlib import Path

import numpy as np
import torch

# run_frll_eval encapsulates the full FRLL/AMSL scoring pipeline.
# Importing eval sets its module-level matplotlib backend (also Agg), which is
# harmless — both scripts write to files, not an interactive display.
from detection.evaluation.eval import run_frll_eval


# ---------------------------------------------------------------------------
# Fixed visual parameters — identical for both figures so the reader can
# compare the two side by side.
# ---------------------------------------------------------------------------
COL_IMP   = '#7F7F7F'   # grey   — impostor
COL_MORPH = '#D62728'   # red    — morph
COL_GEN   = '#2CA02C'   # green  — genuine

LS_IMP    = '-'         # solid   — primary line style for greyscale printing
LS_MORPH  = '--'        # dashed
LS_GEN    = ':'         # dotted

X_MIN_DEFAULT = -0.2
X_MAX_DEFAULT =  1.0
N_BINS        =  50     # uniform bin width applied to all three distributions


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_histogram(
    impostor_scores,
    genuine_scores,
    morph_scores,
    tau,
    mmpmr,
    d_eer,
    title,
    output,
    x_min=X_MIN_DEFAULT,
    x_max=X_MAX_DEFAULT,
):
    """
    Draw three density-normalised step histograms on a single axes, mark tau,
    annotate key metrics, and save as both PDF and PNG.

    impostor_scores / genuine_scores / morph_scores : torch.Tensor (float32)
    """
    bins = np.linspace(x_min, x_max, N_BINS + 1)

    imp  = impostor_scores.numpy()
    morph = morph_scores.numpy()
    gen  = genuine_scores.numpy()

    fig, ax = plt.subplots(figsize=(8, 5))

    # --- Semi-transparent fills (overlap regions remain visible) ---
    ax.hist(imp,  bins=bins, density=True, histtype='stepfilled',
            color=COL_IMP,   alpha=0.35, linewidth=0)
    ax.hist(morph, bins=bins, density=True, histtype='stepfilled',
            color=COL_MORPH, alpha=0.35, linewidth=0)
    ax.hist(gen,  bins=bins, density=True, histtype='stepfilled',
            color=COL_GEN,   alpha=0.35, linewidth=0)

    # --- Outlined steps: distinct line styles for greyscale readability ---
    ax.hist(imp,  bins=bins, density=True, histtype='step',
            color=COL_IMP,   linewidth=1.5, linestyle=LS_IMP,
            label=f'Impostor  (n={len(imp):,})')
    ax.hist(morph, bins=bins, density=True, histtype='step',
            color=COL_MORPH, linewidth=1.5, linestyle=LS_MORPH,
            label=f'Morph     (n={len(morph):,})')
    ax.hist(gen,  bins=bins, density=True, histtype='step',
            color=COL_GEN,   linewidth=1.5, linestyle=LS_GEN,
            label=f'Genuine   (n={len(gen):,})')

    # --- Threshold line ---
    ax.axvline(tau, color='black', linewidth=1.5, linestyle='--',
               label=f'τ = {tau:.4f}  (FMR=0.1%)')

    # --- Y-axis headroom: 20% above the tallest bin across all three distributions ---
    # This creates a clear strip at the top of the axes for the stats box.
    counts_imp,   _ = np.histogram(imp,  bins=bins, density=True)
    counts_morph, _ = np.histogram(morph, bins=bins, density=True)
    counts_gen,   _ = np.histogram(gen,  bins=bins, density=True)
    ymax_data = max(counts_imp.max(), counts_morph.max(), counts_gen.max())
    ax.set_ylim(0, ymax_data * 1.20)

    # --- Stats box: upper-left, inside the axes, in the headroom above the peaks ---
    # Upper-left is clear of the genuine distribution (which sits right in both figures)
    # and the 20% headroom ensures the box sits above the impostor peak (which is left).
    stats_text = (
        f'D-EER = {d_eer:.1%}\n'
        f'MMPMR = {mmpmr:.1%}\n'
        f'τ = {tau:.4f}'
    )
    ax.text(
        0.02, 0.97, stats_text,
        transform=ax.transAxes,
        verticalalignment='top', horizontalalignment='left',
        fontsize=8,
        bbox=dict(
            boxstyle='round', facecolor='white', alpha=0.85,
            edgecolor='#CCCCCC', linewidth=0.8,
        ),
    )

    # --- Axes formatting ---
    ax.set_xlim(x_min, x_max)
    ax.set_xlabel('Cosine similarity', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='both', alpha=0.25, linewidth=0.5)

    # --- Legend: outside the axes, anchored to the right edge ---
    # bbox_inches='tight' in savefig captures it; no manual right-margin adjustment needed.
    ax.legend(
        fontsize=9,
        loc='upper left',
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0,
        frameon=True,
    )

    fig.tight_layout()

    # --- Save as PDF (for LaTeX) and PNG (for preview) ---
    out = Path(output)
    pdf_path = out.with_suffix('.pdf')
    png_path = out.with_suffix('.png')
    fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"[Saved] {pdf_path}")
    print(f"[Saved] {png_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=(
            'Plot impostor / morph / genuine score distributions for one model '
            'configuration using FRLL/AMSL data.  Scoring reuses eval.py logic exactly.'
        )
    )

    # Model
    ap.add_argument('--backbone',     type=str, required=True,
                    choices=['adaface', 'arcface'],
                    help='Backbone architecture')
    ap.add_argument('--base_ckpt',    type=str, required=True,
                    help='Backbone checkpoint (.ckpt / .pth)')
    ap.add_argument('--adapter_ckpt', type=str, default='',
                    help='Adapter checkpoint — omit or leave empty for backbone-only evaluation')

    # Data
    ap.add_argument('--bonafide_dir', type=str, required=True,
                    help='FRLL bona fide image directory (frll_all_4ArcFace/)')
    ap.add_argument('--morph_dir',    type=str, required=True,
                    help='AMSL morph image directory (morph_amsl_4ArcFace/)')

    # Output
    ap.add_argument('--output', type=str, required=True,
                    help='Output path — extension is replaced; both .pdf and .png are written')
    ap.add_argument('--title',  type=str, default='Score Distributions',
                    help='Figure title (e.g. "Frozen Backbone" or "Bona Fide Adapter")')

    # Figure geometry — keep identical across both runs for a fair comparison
    ap.add_argument('--x_min', type=float, default=X_MIN_DEFAULT,
                    help=f'x-axis lower limit (default: {X_MIN_DEFAULT})')
    ap.add_argument('--x_max', type=float, default=X_MAX_DEFAULT,
                    help=f'x-axis upper limit (default: {X_MAX_DEFAULT})')

    # Compute
    ap.add_argument('--batch_size',  type=int,   default=256)
    ap.add_argument('--num_workers', type=int,   default=4)
    ap.add_argument('--pin_memory',  action='store_true')
    ap.add_argument('--amp',         action='store_true')
    ap.add_argument('--fmr',         type=float, default=0.001,
                    help='Target FMR for threshold tau (default: 0.001 = 0.1%%)')

    args = ap.parse_args()

    device  = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_amp = bool(args.amp and device == 'cuda')

    # --- Run scoring (same logic as eval.py --dataset frll_amsl) ---
    result = run_frll_eval(
        backbone_type=args.backbone,
        base_ckpt=args.base_ckpt,
        adapter_ckpt=args.adapter_ckpt,
        bonafide_dir=args.bonafide_dir,
        morph_dir=args.morph_dir,
        device=device,
        use_amp=use_amp,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        fmr=args.fmr,
    )

    # --- Sanity print: compare against reference values in task spec ---
    print()
    print('[Sanity check — compare against known-good eval.py output]')
    print(f'  genuine_mean   = {result["genuine_mean"]:.4f}')
    print(f'  morph_mean     = {result["morph_score_mean"]:.4f}')
    print(f'  impostor_mean  = {result["impostor_mean"]:.4f}')
    print(f'  tau            = {result["tau"]:.4f}')
    print(f'  d_eer          = {result["d_eer"]:.4f}')
    print(f'  mmpmr          = {result["mmpmr"]:.4f}')
    print(f'  eer            = {result["eer"]:.4f}')
    print()
    print('  Expected (frozen backbone):    genuine~0.879  morph~0.425  imp~0.021  tau~0.264  d_eer=0.0')
    print('  Expected (bona fide adapter):  genuine~0.414  morph~0.192  imp~0.006  tau~0.201  d_eer~0.020')
    print()

    # --- Plot ---
    plot_histogram(
        impostor_scores=result['impostor_scores'],
        genuine_scores=result['genuine_scores'],
        morph_scores=result['morph_scores'],
        tau=result['tau'],
        mmpmr=result['mmpmr'],
        d_eer=result['d_eer'],
        title=args.title,
        output=args.output,
        x_min=args.x_min,
        x_max=args.x_max,
    )


if __name__ == '__main__':
    main()