import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import argparse
import sys
from pathlib import Path
import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_curve

from detection.evaluation.eval import run_frll_eval


# ── Configurable line-style defaults ────────────────────────────────────────
#
# Mode A (single backbone, --adapters): styles assigned by position.
DEFAULT_STYLES = [
    ('#000000', 'solid',    1.8),   # 0 black solid        → Bona Fide adapter
    ('#2166ac', 'dashed',   1.8),   # 1 blue dashed        → TetraLoss
    ('#4dac26', 'dashdot',  1.8),   # 2 green dashdot      → TripletRepulsion
    ('#d6191b', 'solid',    2.5),   # 3 red solid thicker  → DirectedRepulsion
    ('#f77f00', 'dotted',   2.0),   # 4 orange dotted      → TripletWC
]

# Backbone-only reference line (--show_backbone in Mode A)
BACKBONE_STYLE = ('#888888', 'dotted', 1.5)

# Mode B (--curves): fallback when no color/linestyle given in the spec
DEFAULT_STYLES_B = [
    ('#2166ac', 'solid',    1.8),
    ('#2166ac', 'dashed',   1.8),
    ('#d6191b', 'solid',    1.8),
    ('#d6191b', 'dashed',   1.8),
    ('#4dac26', 'solid',    1.8),
    ('#4dac26', 'dashed',   1.8),
]


# ── Spec parsers ─────────────────────────────────────────────────────────────

def _parse_adapter_spec(spec: str):
    """
    Parse a --adapters entry: 'path/to/adapter.pt:Display Label'
    Returns (ckpt_path, label).
    """
    colon = spec.rfind(':')
    if colon <= 0:
        raise argparse.ArgumentTypeError(
            f"--adapters spec must be 'path:label', got: {spec!r}"
        )
    return spec[:colon], spec[colon + 1:]


def _parse_curve_spec(spec: str):
    """
    Parse a --curves entry:
        backbone:base_ckpt:adapter_ckpt:label[:color[:linestyle]]

    backbone      — 'adaface' or 'arcface'
    base_ckpt     — path to backbone checkpoint
    adapter_ckpt  — path to adapter checkpoint; use 'none' for backbone-only
    label         — display label (may contain spaces if the whole spec is quoted)
    color         — optional matplotlib color string
    linestyle     — optional matplotlib linestyle name

    Returns a dict with those keys (adapter_ckpt is None when the field is 'none').
    """
    parts = spec.split(':')
    if len(parts) < 4:
        raise argparse.ArgumentTypeError(
            f"--curves spec needs at least 4 colon-separated fields "
            f"(backbone:base_ckpt:adapter_ckpt:label), got: {spec!r}"
        )
    backbone     = parts[0].strip()
    base_ckpt    = parts[1].strip()
    adapter_ckpt = parts[2].strip()
    label        = parts[3].strip()
    color        = parts[4].strip() if len(parts) >= 5 else None
    linestyle    = parts[5].strip() if len(parts) >= 6 else None

    if backbone not in ('adaface', 'arcface'):
        raise argparse.ArgumentTypeError(
            f"backbone must be 'adaface' or 'arcface', got: {backbone!r}"
        )
    if adapter_ckpt.lower() == 'none':
        adapter_ckpt = None

    return dict(backbone=backbone, base_ckpt=base_ckpt,
                adapter_ckpt=adapter_ckpt, label=label,
                color=color, linestyle=linestyle)


# ── Scoring ──────────────────────────────────────────────────────────────────

def _run_eval(backbone, base_ckpt, adapter_ckpt,
              bonafide_dir, morph_dir, use_amp, batch_size, num_workers, seed):
    res = run_frll_eval(
        backbone_type=backbone,
        base_ckpt=base_ckpt,
        adapter_ckpt=adapter_ckpt,
        bonafide_dir=bonafide_dir,
        morph_dir=morph_dir,
        use_amp=use_amp,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
    )
    return res['genuine_scores'].numpy(), res['impostor_scores'].numpy()


def _det_curve(genuine_scores: np.ndarray, attack_scores: np.ndarray):
    """
    Compute probit-clipped (FPR, FNR) and EER from score arrays.
    Identical to the computation in the original plot_roc_det.py:
    sklearn roc_curve → FNR = 1 − TPR, then both arrays clipped to (eps, 1−eps).
    """
    eps = 1e-4
    labels = np.concatenate([np.ones(len(genuine_scores)),
                              np.zeros(len(attack_scores))])
    scores = np.concatenate([genuine_scores, attack_scores])
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1.0 - tpr

    eer_idx = np.nanargmin(np.abs(fnr - fpr))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)
    eer_threshold = float(thresholds[eer_idx])

    return np.clip(fpr, eps, 1 - eps), np.clip(fnr, eps, 1 - eps), eer, eer_threshold


# ── Plot ─────────────────────────────────────────────────────────────────────

def _plot_det(curves, output_path, title=None, annotate=False):
    """
    Render DET figure and save PDF + PNG.

    curves: list of dicts, each with:
        label  : str
        fpr_c  : np.ndarray  (clipped)
        fnr_c  : np.ndarray  (clipped)
        eer    : float
        style  : (color, linestyle, linewidth)
    """
    plt.rcParams.update({
        'font.family'    : 'serif',
        'font.size'      : 11,
        'axes.titlesize' : 12,
        'axes.labelsize' : 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
    })

    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    eps = 1e-4

    # Probit axis range — identical to plot_roc_det.py
    axis_min_p = norm.ppf(0.0008)
    axis_max_p = norm.ppf(0.42)

    # EER reference diagonal
    diag = np.linspace(axis_min_p, axis_max_p, 2000)
    ax.plot(diag, diag, 'k--', lw=1, label='EER reference (FNMR = FMR)', zorder=1)

    # ABC operating point
    ax.axvline(norm.ppf(0.001), color='#888888', linestyle='dotted', lw=1.0,
               label='FMR = 0.1% (ABC operating point)', zorder=1)

    annotation_pts = []  # (x_probit, y_probit) for overlap avoidance

    for entry in curves:
        color, ls, lw = entry['style']
        label = entry['label']
        eer   = entry['eer']

        ax.plot(norm.ppf(entry['fpr_c']), norm.ppf(entry['fnr_c']),
                color=color, linestyle=ls, linewidth=lw,
                label=f'{label}  (EER {eer * 100:.1f}%)',
                zorder=2)

        # EER marker
        eer_c = float(np.clip(eer, eps, 1 - eps))
        ax.plot(norm.ppf(eer_c), norm.ppf(eer_c),
                'o', color=color, markersize=6, zorder=3)

        if annotate:
            # Label at the curve point closest to FMR = 1 %
            target = 0.01
            idx = int(np.argmin(np.abs(entry['fpr_c'] - target)))
            xp  = float(norm.ppf(entry['fpr_c'][idx]))
            yp  = float(norm.ppf(entry['fnr_c'][idx]))
            # Nudge down for each nearby existing label to reduce overlap
            y_shift = 0.0
            for px, py in annotation_pts:
                if abs(xp - px) < 0.4 and abs(yp - py - y_shift) < 0.35:
                    y_shift += 0.35
            ax.annotate(
                label,
                xy=(xp, yp),
                xytext=(xp + 0.12, yp - 0.25 - y_shift),
                fontsize=8,
                color=color,
                arrowprops=dict(arrowstyle='->', color=color, lw=0.7),
            )
            annotation_pts.append((xp, yp))

    # Probit tick marks — identical to plot_roc_det.py
    ticks_pct   = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40]
    tick_labels = ['0.1%', '0.5%', '1%', '2%', '5%', '10%', '20%', '40%']
    probit_ticks = [norm.ppf(t) for t in ticks_pct]
    ax.set_xticks(probit_ticks)
    ax.set_xticklabels(tick_labels)
    ax.set_yticks(probit_ticks)
    ax.set_yticklabels(tick_labels)
    ax.set_xlim([axis_min_p, axis_max_p])
    ax.set_ylim([axis_min_p, axis_max_p])

    ax.set_xlabel('False Match Rate (FMR)')
    ax.set_ylabel('False Non-Match Rate (FNMR)')
    ax.set_title(title or 'Detection Error Trade-off (DET)', pad=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    n_entries = len(curves) + 2  # +2: EER diagonal + ABC line
    ncol = 2 if n_entries >= 4 else 1
    leg = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18),
                    ncol=ncol, frameon=True,
                    edgecolor='black', facecolor='white',
                    fontsize='small')
    leg.get_frame().set_linewidth(1.0)

    fig.subplots_adjust(bottom=0.30)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = out.with_suffix('.pdf')
    png_path = out.with_suffix('.png')
    fig.savefig(str(pdf_path), dpi=300, bbox_inches='tight')
    fig.savefig(str(png_path), dpi=300, bbox_inches='tight')
    print(f'[DET] Saved PDF: {pdf_path}')
    print(f'[DET] Saved PNG: {png_path}')
    plt.close(fig)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description='Multi-adapter / multi-backbone DET comparison (FRLL/AMSL).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Mode A — single backbone, multiple adapters:
  python plot_det.py --backbone adaface --base_ckpt adaface.ckpt \\
    --bonafide_dir /path/frll --morph_dir /path/amsl \\
    --adapters "bonafide.pt:Bona Fide" "tetraloss.pt:TetraLoss" \\
    --show_backbone --output det_adaface.pdf --title "AMSL — AdaFace"

Mode B — per-curve backbone (backbone:base_ckpt:adapter_ckpt:label[:color[:linestyle]]):
  python plot_det.py \\
    --bonafide_dir /path/frll --morph_dir /path/amsl \\
    --curves \\
      "adaface:adaface.ckpt:ada_bonafide.pt:AdaFace Bona Fide:blue:solid" \\
      "adaface:adaface.ckpt:ada_dirrep.pt:AdaFace DirectedRepulsion:blue:dashed" \\
      "arcface:arcface.ckpt:arc_bonafide.pt:ArcFace Bona Fide:red:solid" \\
      "arcface:arcface.ckpt:arc_dirrep.pt:ArcFace DirectedRepulsion:red:dashed" \\
    --output det_backbone.pdf --title "AMSL — Backbone Comparison"
""",
    )

    # Dataset paths (both modes)
    ap.add_argument('--bonafide_dir', type=str, required=True,
                    help='FRLL bona fide image directory')
    ap.add_argument('--morph_dir',    type=str, required=True,
                    help='AMSL morph image directory')

    # Mode A
    mode_a = ap.add_argument_group('Mode A — single backbone')
    mode_a.add_argument('--backbone',  type=str, choices=['adaface', 'arcface'],
                        help='Backbone architecture')
    mode_a.add_argument('--base_ckpt', type=str,
                        help='Backbone checkpoint')
    mode_a.add_argument('--adapters',  type=str, nargs='*', default=[],
                        metavar='PATH:LABEL',
                        help='Adapter specs: path/to/ckpt.pt:"Label"')
    mode_a.add_argument('--show_backbone', action='store_true',
                        help='Add frozen backbone-only reference curve (grey dotted)')

    # Mode B
    mode_b = ap.add_argument_group('Mode B — per-curve backbone')
    mode_b.add_argument('--curves', type=str, nargs='*', default=[],
                        metavar='BACKBONE:BASE_CKPT:ADAPTER_CKPT:LABEL[:COLOR[:LINESTYLE]]',
                        help='Per-curve spec; quote each entry if the label contains spaces')

    # Plot
    ap.add_argument('--output',   type=str, default='det_comparison.pdf')
    ap.add_argument('--title',    type=str, default=None)
    ap.add_argument('--annotate', action='store_true',
                    help='Place inline text label on each curve at FMR ≈ 1%%')

    # Eval
    ap.add_argument('--batch_size',  type=int, default=256)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--amp',         action='store_true')
    ap.add_argument('--seed',        type=int, default=42)

    args = ap.parse_args()

    use_curves_mode   = bool(args.curves)
    use_adapters_mode = bool(args.adapters) or args.show_backbone

    if use_curves_mode and use_adapters_mode:
        ap.error('--curves and --adapters / --show_backbone are mutually exclusive.')
    if not use_curves_mode and not use_adapters_mode and not (args.backbone and args.base_ckpt):
        ap.error('Provide --curves (Mode B) or --backbone + --base_ckpt [+ --adapters] (Mode A).')
    if not use_curves_mode and (not args.backbone or not args.base_ckpt):
        ap.error('Mode A requires both --backbone and --base_ckpt.')

    eval_kw = dict(
        bonafide_dir=args.bonafide_dir,
        morph_dir=args.morph_dir,
        use_amp=args.amp,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    curves_out = []

    # ── Mode B ────────────────────────────────────────────────────────────────
    if use_curves_mode:
        for i, spec in enumerate(args.curves):
            cd = _parse_curve_spec(spec)
            print(f'[Run {i+1}/{len(args.curves)}] '
                  f'backbone={cd["backbone"]}  '
                  f'adapter={cd["adapter_ckpt"] or "none"}  '
                  f'label={cd["label"]!r}')
            g, imp = _run_eval(
                backbone=cd['backbone'],
                base_ckpt=cd['base_ckpt'],
                adapter_ckpt=cd['adapter_ckpt'],
                **eval_kw,
            )
            print(f'[DET-diag] genuine_mean={g.mean():.5f}  genuine_std={g.std():.5f}  n_genuine={len(g)}')
            print(f'[DET-diag] impostor_mean={imp.mean():.5f}  impostor_std={imp.std():.5f}  n_impostor={len(imp)}')
            fpr_c, fnr_c, eer, _ = _det_curve(g, imp)
            print(f'[DET-diag] EER={eer:.5f}')
            print(f'  EER = {eer * 100:.2f}%')

            # Explicit color/linestyle in spec take precedence over defaults
            if cd['color'] and cd['linestyle']:
                style = (cd['color'], cd['linestyle'], 1.8)
            elif cd['color']:
                fallback_ls = DEFAULT_STYLES_B[i % len(DEFAULT_STYLES_B)][1]
                style = (cd['color'], fallback_ls, 1.8)
            else:
                style = DEFAULT_STYLES_B[i % len(DEFAULT_STYLES_B)]

            curves_out.append(dict(label=cd['label'],
                                   fpr_c=fpr_c, fnr_c=fnr_c, eer=eer,
                                   style=style))

    # ── Mode A ────────────────────────────────────────────────────────────────
    else:
        style_idx = 0
        total = (1 if (args.show_backbone or not args.adapters) else 0) + len(args.adapters)
        run_idx = 0

        if args.show_backbone or not args.adapters:
            run_idx += 1
            print(f'[Run {run_idx}/{total}] backbone-only (no adapter) ...')
            g, imp = _run_eval(backbone=args.backbone, base_ckpt=args.base_ckpt,
                               adapter_ckpt=None, **eval_kw)
            print(f'[DET-diag] genuine_mean={g.mean():.5f}  genuine_std={g.std():.5f}  n_genuine={len(g)}')
            print(f'[DET-diag] impostor_mean={imp.mean():.5f}  impostor_std={imp.std():.5f}  n_impostor={len(imp)}')
            fpr_c, fnr_c, eer, _ = _det_curve(g, imp)
            print(f'[DET-diag] EER={eer:.5f}')
            print(f'  EER = {eer * 100:.2f}%')
            curves_out.append(dict(label='Backbone (no adapter)',
                                   fpr_c=fpr_c, fnr_c=fnr_c, eer=eer,
                                   style=BACKBONE_STYLE))

        for spec in args.adapters:
            ckpt_path, label = _parse_adapter_spec(spec)
            run_idx += 1
            print(f'[Run {run_idx}/{total}] adapter={ckpt_path}  label={label!r} ...')
            g, imp = _run_eval(backbone=args.backbone, base_ckpt=args.base_ckpt,
                               adapter_ckpt=ckpt_path, **eval_kw)
            print(f'[DET-diag] genuine_mean={g.mean():.5f}  genuine_std={g.std():.5f}  n_genuine={len(g)}')
            print(f'[DET-diag] impostor_mean={imp.mean():.5f}  impostor_std={imp.std():.5f}  n_impostor={len(imp)}')
            fpr_c, fnr_c, eer, _ = _det_curve(g, imp)
            print(f'[DET-diag] EER={eer:.5f}')
            print(f'  EER = {eer * 100:.2f}%')
            style = DEFAULT_STYLES[style_idx % len(DEFAULT_STYLES)]
            style_idx += 1
            curves_out.append(dict(label=label,
                                   fpr_c=fpr_c, fnr_c=fnr_c, eer=eer,
                                   style=style))

    if not curves_out:
        print('[ERROR] No curves collected.')
        sys.exit(1)

    _plot_det(curves_out, output_path=args.output,
              title=args.title, annotate=args.annotate)


if __name__ == '__main__':
    main()