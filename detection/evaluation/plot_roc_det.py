import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_curve

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


def plot_roc_det(
    genuine_scores,
    impostor_scores,
    output_dir='figures/',
    filename_stem='background_roc_det',
    label='AdaFace + Adapter',
    log_wandb=True,
):
    """
    Generate side-by-side ROC and DET curves from real score arrays.
    Saves PDF and PNG to output_dir. Optionally logs PNG preview to W&B.

    Args:
        genuine_scores : 1D np.ndarray of cosine similarities for genuine pairs
        impostor_scores: 1D np.ndarray of cosine similarities for impostor pairs
        output_dir     : directory to save figures (created if absent)
        filename_stem  : base filename without extension
        label          : system label for legend
        log_wandb      : whether to log PNG preview to W&B
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── EER computation ───────────────────────────────────────────────────────
    labels = np.concatenate([
        np.ones(len(genuine_scores)),
        np.zeros(len(impostor_scores)),
    ])
    scores = np.concatenate([genuine_scores, impostor_scores])
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1.0 - tpr

    eer_idx = np.nanargmin(np.abs(fnr - fpr))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)
    eer_threshold = float(thresholds[eer_idx])
    print(f"[ROC/DET] EER: {eer * 100:.2f}%  |  threshold at EER: {eer_threshold:.4f}")

    # ── Style ─────────────────────────────────────────────────────────────────
    plt.rcParams.update({
        'font.family'    : 'serif',
        'font.size'      : 11,
        'axes.titlesize' : 12,
        'axes.labelsize' : 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
    })

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    eps = 1e-4

    # ── ROC ───────────────────────────────────────────────────────────────────
    ax = axes[0]

    ax.plot(fpr, tpr, color='#2166ac', lw=2, label=label)
    ax.plot([0, 1], [1, 0], 'k--', lw=1, label='EER reference')
    ax.plot(eer, 1.0 - eer, 'o', color='#d6191b', markersize=8, label='EER')

    ax.text(-0.01, 1.045,
            'Ideal system: upper-left corner',
            transform=ax.transAxes,
            fontsize=9, color='dimgrey', style='italic')

    ax.set_xlabel('False Match Rate (FMR)')
    ax.set_ylabel('True Match Rate (TMR)')
    ax.set_title('Receiver Operating Characteristic (ROC)', pad=20)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False)

    # ── DET ───────────────────────────────────────────────────────────────────
    ax = axes[1]

    fpr_c = np.clip(fpr, eps, 1 - eps)
    fnr_c = np.clip(fnr, eps, 1 - eps)
    ax.plot(norm.ppf(fpr_c), norm.ppf(fnr_c), color='#2166ac', lw=2, label=label)

    # Full-range diagonal spanning the entire visible axis
    axis_min_p = norm.ppf(0.0008)
    axis_max_p = norm.ppf(0.42)
    diag = np.linspace(axis_min_p, axis_max_p, 2000)
    ax.plot(diag, diag, 'k--', lw=1, label='EER reference (FNMR = FMR)')

    # EER marker
    eer_c = float(np.clip(eer, eps, 1 - eps))
    ax.plot(norm.ppf(eer_c), norm.ppf(eer_c),
            'o', color='#d6191b', markersize=8, label='EER')

    # ABC operating point
    ax.axvline(norm.ppf(0.001), color='dimgrey', linestyle='--', lw=1.2,
               label='FMR = 0.1% (ABC operating point)')

    # Probit ticks
    ticks_pct   = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40]
    tick_labels = ['0.1%', '0.5%', '1%', '2%', '5%', '10%', '20%', '40%']
    probit_ticks = [norm.ppf(t) for t in ticks_pct]
    ax.set_xticks(probit_ticks)
    ax.set_xticklabels(tick_labels)
    ax.set_yticks(probit_ticks)
    ax.set_yticklabels(tick_labels)
    ax.set_xlim([axis_min_p, axis_max_p])
    ax.set_ylim([axis_min_p, axis_max_p])

    ax.text(-0.01, 1.045,
            'Ideal system: lower-left corner',
            transform=ax.transAxes,
            fontsize=9, color='dimgrey', style='italic')

    ax.set_xlabel('False Match Rate (FMR)')
    ax.set_ylabel('False Non-Match Rate (FNMR)')
    ax.set_title('Detection Error Trade-off (DET)', pad=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False)

    # ── Save ──────────────────────────────────────────────────────────────────
    plt.tight_layout(rect=[0, 0.10, 1, 1])

    pdf_path = os.path.join(output_dir, f'{filename_stem}.pdf')
    png_path = os.path.join(output_dir, f'{filename_stem}.png')

    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"[ROC/DET] Saved PDF: {pdf_path}")
    print(f"[ROC/DET] Saved PNG: {png_path}")

    # ── W&B logging ───────────────────────────────────────────────────────────
    if log_wandb:
        if not _WANDB_AVAILABLE:
            print("[ROC/DET] W&B logging skipped: wandb not installed.")
        else:
            try:
                wandb.log({
                    'figures/roc_det': wandb.Image(
                        png_path,
                        caption=f'ROC + DET — EER {eer * 100:.2f}%',
                    )
                })
                print("[ROC/DET] Logged to W&B.")
            except Exception as e:
                print(f"[ROC/DET] W&B logging skipped: {e}")

    plt.close()
    return eer, eer_threshold