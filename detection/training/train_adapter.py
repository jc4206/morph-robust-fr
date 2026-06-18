# train_adapter.py
import time
import json
import argparse
import random
from pathlib import Path
import os
import numpy as np
from sklearn.metrics import roc_curve
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from detection.models.adaface.loader import load_adaface_ir50, load_arcface_ir50
from detection.losses import TetraLoss, TetraLossExt, TetraLossDirected, TetraLossWorstCase, TripletWorstCase, TetraLossBalanced, TetraLossWorstCaseBalanced, TripletRepulsionLoss, DirectedRepulsionLoss
from detection.data.dataset import TetraQuadrupleDataset
from detection.data.index_parser import build_real_index, build_morph_index
from detection.data.preprocessing import transform
from detection.models.mlp_adapter import MLPAdapter
import csv

try:
    import wandb
except Exception:
    wandb = None


# -----------------------------------
# Utils
# -----------------------------------
def set_requires_grad(module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag


def count_trainable_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def encode_backbone(backbone, imgs, device, use_amp: bool, non_blocking: bool = True):
    """
    Returns L2-normalized embeddings from frozen backbone.
    AMP is supported for faster GPU inference.
    """
    imgs = imgs.to(device, non_blocking=non_blocking)
    with torch.cuda.amp.autocast(enabled=use_amp):
        emb, _ = backbone(imgs)
        emb = F.normalize(emb, p=2, dim=1)
    return emb

def encode_adapter(backbone, adapter, imgs, device, use_amp: bool, non_blocking: bool = True):
    """
    backbone forward is no-grad (frozen), adapter is trainable.
    Returns L2-normalized adapter embeddings.
    """
    base_emb = encode_backbone(backbone, imgs, device, use_amp=use_amp, non_blocking=non_blocking)  # (B,512)
    out = adapter(base_emb)  # MLPAdapter.forward already L2-normalizes its output
    return out


def encode_backbone_grad(backbone, imgs, device, use_amp: bool, non_blocking: bool = True):
    """
    Like encode_backbone but WITHOUT @torch.no_grad — required when the backbone
    output layer is unfrozen so that gradients flow back into it.
    Body layers still receive no gradient (their params have requires_grad=False),
    but the computation graph is kept alive through the output projection.
    """
    imgs = imgs.to(device, non_blocking=non_blocking)
    with torch.cuda.amp.autocast(enabled=use_amp):
        emb, _ = backbone(imgs)
        emb = F.normalize(emb, p=2, dim=1)
    return emb


def encode_adapter_with_grad(backbone, adapter, imgs, device, use_amp: bool, non_blocking: bool = True):
    """
    Full encoding path with gradients through the backbone output layer AND adapter.
    Used only when --unfreeze_output_layer is active.
    """
    base_emb = encode_backbone_grad(backbone, imgs, device, use_amp=use_amp, non_blocking=non_blocking)
    out = adapter(base_emb)  # MLPAdapter.forward already L2-normalizes its output
    return out


def unfreeze_output_layer(backbone, backbone_type: str):
    """
    Unfreezes only the final projection layer of the frozen IR-50 backbone.

    AdaFace (net.py Backbone):
        backbone.output_layer = Sequential(BN2d, Dropout, Flatten, Linear(25088→512), BN1d(affine=False))
        Trainable after unfreeze: BN2d affine params + Linear weight (~13.1 M params total)

    ArcFace (iresnet.py IResNet):
        backbone.fc = Linear(25088→512)
        backbone.features = BN1d — features.weight.requires_grad=False is hardcoded in __init__,
        so only backbone.fc is unfrozen here.

    Returns the unfrozen module (for logging / param-group construction).
    """
    module = backbone.output_layer if backbone_type == "adaface" else backbone.fc
    for p in module.parameters():
        p.requires_grad = True
    return module


def unfreeze_last_block(backbone, backbone_type: str):
    """
    Unfreezes the last residual block group + output projection of the frozen IR-50 backbone.

    AdaFace (net.py Backbone):
        backbone.body[-3:]    — last 3 BasicBlockIR (body[21], body[22], body[23])
        backbone.output_layer — Sequential(BN2d, Dropout, Flatten, Linear(25088→512), BN1d)
        backbone.body[-3:] preserves original key names ("21", "22", "23") in the returned
        Sequential, so state_dict / load_state_dict round-trips correctly.

    ArcFace (iresnet.py IResNet):
        backbone.layer4 — last residual block group (nn.Sequential)
        backbone.fc     — Linear(25088→512)
        backbone.features.weight stays frozen (hardcoded requires_grad=False in __init__).

    Returns list of unfrozen module objects (for .train() mode restoration and param-group
    construction).
    """
    if backbone_type == "adaface":
        modules = list(backbone.body[-3:]) + [backbone.output_layer]
    else:
        modules = [backbone.layer4, backbone.fc]
    for m in modules:
        for p in m.parameters():
            p.requires_grad = True
    return modules


def log_freeze_state(backbone, adapter):
    """Logs frozen vs trainable parameter counts at training start."""
    bb_train  = [(n, p) for n, p in backbone.named_parameters() if p.requires_grad]
    bb_frozen = sum(1 for _, p in backbone.named_parameters() if not p.requires_grad)
    ad_total  = sum(p.numel() for p in adapter.parameters())
    bb_unfrz  = sum(p.numel() for _, p in bb_train)

    print(f"[FreezeState] backbone : {bb_frozen} frozen tensors | "
          f"{len(bb_train)} trainable tensors ({bb_unfrz:,} params)")
    for n, p in bb_train:
        print(f"  [TRAINABLE backbone]  {n:55s}  {list(p.shape)}")
    print(f"[FreezeState] adapter  : all {sum(1 for _ in adapter.parameters())} tensors trainable "
          f"({ad_total:,} params)")


@torch.no_grad()
def quick_reports_adapter(
    backbone,
    adapter,
    batch,
    device,
    use_amp: bool,
    non_blocking: bool,
    adapter_on_positive: bool,
):
    """
    Logs cosine sims + euclidean distances for (anchor, pos/neg/morph).

    If adapter_on_positive=False (paper-aligned):
      - za, zn, zm are adapter embeddings
      - zp is backbone embedding (no adapter)

    If adapter_on_positive=True:
      - za, zp, zn, zm are adapter embeddings
    """
    was_bb_training = backbone.training
    was_ad_training = adapter.training

    backbone.eval()
    adapter.eval()

    def enc_adapted(x):
        x = x.to(device, non_blocking=non_blocking)
        with torch.cuda.amp.autocast(enabled=use_amp):
            emb, _ = backbone(x)
            emb = F.normalize(emb, p=2, dim=1)
            emb = adapter(emb)
            emb = F.normalize(emb, p=2, dim=1)
        return emb

    def enc_base(x):
        x = x.to(device, non_blocking=non_blocking)
        with torch.cuda.amp.autocast(enabled=use_amp):
            emb, _ = backbone(x)
            emb = F.normalize(emb, p=2, dim=1)
        return emb

    # Always adapted for anchor/negative/morph
    za = enc_adapted(batch["anchor"])
    zn = enc_adapted(batch["negative"])
    zm = enc_adapted(batch["morph"])

    # Positive depends on mode
    if adapter_on_positive:
        zp = enc_adapted(batch["positive"])
    else:
        zp = enc_base(batch["positive"])

    # cosine
    s_ap = (za * zp).sum(dim=1).mean().item()
    s_an = (za * zn).sum(dim=1).mean().item()
    s_am = (za * zm).sum(dim=1).mean().item()

    # euclidean (unit sphere)
    d_ap = torch.norm(za - zp, dim=1).mean().item()
    d_an = torch.norm(za - zn, dim=1).mean().item()
    d_am = torch.norm(za - zm, dim=1).mean().item()

    # Backbone-only baseline (all four, no adapter) — frozen so these stay flat
    #bb_a = enc_base(batch["anchor"])
    #bb_p = enc_base(batch["positive"])
    #bb_n = enc_base(batch["negative"])
    #bb_m = enc_base(batch["morph"])

    #bb_s_ap = (bb_a * bb_p).sum(dim=1).mean().item()
    #bb_s_an = (bb_a * bb_n).sum(dim=1).mean().item()
    #bb_s_am = (bb_a * bb_m).sum(dim=1).mean().item()

    #bb_d_ap = torch.norm(bb_a - bb_p, dim=1).mean().item()
    #bb_d_an = torch.norm(bb_a - bb_n, dim=1).mean().item()
    #bb_d_am = torch.norm(bb_a - bb_m, dim=1).mean().item()

    backbone.train(was_bb_training)
    adapter.train(was_ad_training)

    return (s_ap, s_an, s_am), (d_ap, d_an, d_am)



def save_ckpt(out_dir, step, adapter, opt, scaler, args, scheduler=None,
              backbone_extra_sds=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "step": step,
        "adapter": adapter.state_dict(),
        "optimizer": opt.state_dict(),
        "scaler": None if scaler is None else scaler.state_dict(),
        "lr_scheduler": None if scheduler is None else scheduler.state_dict(),
        "args": vars(args),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "py_rng": random.getstate(),
    }
    if backbone_extra_sds:
        ckpt.update(backbone_extra_sds)

    torch.save(ckpt, out_dir / f"ckpt_step{step}.pt")
    torch.save(adapter.state_dict(), out_dir / f"adapter_step{step}.pt")
    print(f"[Save] step {step}")

def _backbone_extra_sds(backbone, args):
    """Returns dict of backbone state dicts to embed in checkpoint (empty when fully frozen)."""
    result = {}
    if getattr(args, "unfreeze_output_layer", False):
        module = backbone.output_layer if args.backbone == "adaface" else backbone.fc
        result["backbone_output_layer"] = module.state_dict()
    elif getattr(args, "unfreeze_last_block", False):
        if args.backbone == "adaface":
            result["backbone_layer4"]       = backbone.body[-3:].state_dict()
            result["backbone_output_layer"] = backbone.output_layer.state_dict()
        else:
            result["backbone_layer4"]       = backbone.layer4.state_dict()
            result["backbone_output_layer"] = backbone.fc.state_dict()
    return result

## VALIDATION, update: 20-02-2026
@torch.no_grad()
def compute_val_metrics(backbone, adapter, loss_fn, val_dl, device, use_amp, adapter_on_positive, max_batches,
                        need_contributor_b=False, need_directed=False):
    was_bb_training = backbone.training
    was_ad_training = adapter.training

    backbone.eval()
    adapter.eval()

    vals = []
    val_triplet_vals = []
    val_repel_vals = []
    val_directed_a_vals = []
    val_directed_b_vals = []
    cos_mn_vals = []
    genuine_scores = []
    impostor_scores = []
    morph_scores = []

    for bi, batch in enumerate(val_dl):
        if bi >= max_batches:
            break

        with torch.cuda.amp.autocast(enabled=use_amp):
            za = encode_adapter(backbone, adapter, batch["anchor"], device, use_amp=use_amp)
            zn = encode_adapter(backbone, adapter, batch["negative"], device, use_amp=use_amp)
            zm = encode_adapter(backbone, adapter, batch["morph"], device, use_amp=use_amp)

            if adapter_on_positive:
                zp = encode_adapter(backbone, adapter, batch["positive"], device, use_amp=use_amp)
                zp_backbone_metric = encode_backbone(backbone, batch["positive"], device, use_amp=use_amp)
            else:
                zp = encode_backbone(backbone, batch["positive"], device, use_amp=use_amp)
                zp_backbone_metric = zp  # already backbone-only

            if need_contributor_b:
                zb = encode_adapter(backbone, adapter, batch["contributor_b"], device, use_amp=use_amp)
                result = loss_fn(za, zp, zn, zm, zb)
            else:
                result = loss_fn(za, zp, zn, zm)

            if isinstance(result, tuple):
                vals.append(result[0].item())
                if need_directed and len(result) == 5:
                    val_triplet_vals.append(result[1].item())
                    val_repel_vals.append(result[2].item())
                    val_directed_a_vals.append(result[3].item())
                    val_directed_b_vals.append(result[4].item())
            else:
                vals.append(result.item())

            if need_directed:
                cos_mn_vals.append((zm * zn).sum(dim=1).mean().item())

        # Asymmetric scoring — mirrors eval.py deployment convention:
        #   reference side (document / may be morphed) → adapter embedding
        #   probe side     (live capture, bona fide)   → backbone embedding only
        #
        # genuine:  adapter(anchor_A) · backbone(positive_A)  — already zp_backbone_metric
        # impostor: adapter(anchor_A) · backbone(negative_B)  — need backbone-only negative
        # morph:    adapter(morph)    · backbone(probe_A/B)
        zn_backbone   = encode_backbone(backbone, batch["negative"], device, use_amp=use_amp)
        zp_B_backbone = encode_backbone(backbone, batch["probe_b"],  device, use_amp=use_amp)
        s_g = (za * zp_backbone_metric).sum(dim=1).detach().cpu().numpy()
        s_i = (za * zn_backbone).sum(dim=1).detach().cpu().numpy()
        s_m_A = (zm * zp_backbone_metric).sum(dim=1)
        s_m_B = (zm * zp_B_backbone).sum(dim=1)
        s_m = torch.minimum(s_m_A, s_m_B).detach().cpu().numpy()

        genuine_scores.append(s_g)
        impostor_scores.append(s_i)
        morph_scores.append(s_m)

    backbone.train(was_bb_training)
    adapter.train(was_ad_training)

    # --- compute_val_loss safeguard ---
    if len(vals) == 0:
        raise RuntimeError(
            "Validation produced 0 batches. Check val split/morph roots/pairlist and val_batches."
        )

    genuine_scores = np.concatenate(genuine_scores) if genuine_scores else np.array([], dtype=np.float32)
    impostor_scores = np.concatenate(impostor_scores) if impostor_scores else np.array([], dtype=np.float32)
    morph_scores = np.concatenate(morph_scores) if morph_scores else np.array([], dtype=np.float32)

    tau_val = float(np.quantile(impostor_scores, 0.999)) if len(impostor_scores) else float("nan")
    mmpmr   = float((morph_scores >= tau_val).mean())    if len(morph_scores)    else float("nan")

    out = {
        "val_loss": float(sum(vals) / len(vals)),
        "eer": compute_eer(genuine_scores, impostor_scores),
        "d_eer": compute_eer(genuine_scores, morph_scores),
        "mmpmr": mmpmr,
        "genuine_scores": genuine_scores,
        "impostor_scores": impostor_scores,
        "morph_scores": morph_scores,
        "genuine_mean": float(genuine_scores.mean()) if len(genuine_scores) else float("nan"),
        "impostor_mean": float(impostor_scores.mean()) if len(impostor_scores) else float("nan"),
        "morph_mean": float(morph_scores.mean()) if len(morph_scores) else float("nan"),
    }
    if need_directed and val_triplet_vals:
        out["val_loss_triplet"]    = float(sum(val_triplet_vals)    / len(val_triplet_vals))
        out["val_loss_repel"]      = float(sum(val_repel_vals)      / len(val_repel_vals))
        out["val_loss_directed_a"] = float(sum(val_directed_a_vals) / len(val_directed_a_vals))
        out["val_loss_directed_b"] = float(sum(val_directed_b_vals) / len(val_directed_b_vals))
        out["val_cos_mn"]          = float(sum(cos_mn_vals) / len(cos_mn_vals))
    return out

# Logging: Train Loss quadruplet-wise to determine hard-morphs

def update_morph_loss_stats(morph_stats, morph_paths, loss_vec, step):
    """
    morph_paths: list[str] length B
    loss_vec: torch.Tensor shape (B,) on any device
    """
    losses = loss_vec.detach().float().cpu().tolist()
    for mp, lv in zip(morph_paths, losses):
        key = str(mp)
        s = morph_stats.get(key)
        if s is None:
            morph_stats[key] = {
                "count_seen": 1,
                "last_loss": float(lv),
                "last_step": int(step),
            }
        else:
            s["count_seen"] += 1
            s["last_loss"] = float(lv)
            s["last_step"] = int(step)

def flush_morph_loss_stats_csv(path, morph_stats):
    if not path:
        return
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["morph_path", "count_seen", "last_loss", "last_step"])
        # sort by current/latest loss descending
        for morph_path, s in sorted(
            morph_stats.items(),
            key=lambda kv: kv[1]["last_loss"],
            reverse=True,
        ):
            w.writerow([
                morph_path,
                s["count_seen"],
                s["last_loss"],
                s["last_step"],
            ])

# WANDB (weights & biases)
def init_wandb(args):
    if not args.use_wandb:
        return None
    if wandb is None:
        print("[W&B] not installed -> disabled")
        return None
    try:
        run = wandb.init(
            project=args.wandb_project,
            entity=(args.wandb_entity or None),
            name=(args.wandb_run_name or None),
            config=vars(args),
            mode=args.wandb_mode,
            dir=(args.wandb_dir or None),
        )
        print(f"[W&B] initialized (mode={args.wandb_mode})")
        return run
    except Exception as e:
        print(f"[W&B] init failed -> disabled ({e})")
        return None

def wandb_log(run, data: dict, step: int):
    if run is None:
        return
    try:
        wandb.log(data, step=step)
    except Exception as e:
        print(f"[W&B] log failed at step={step}: {e}")

def compute_eer(genuine_scores: np.ndarray, impostor_scores: np.ndarray) -> float:
    if len(genuine_scores) == 0 or len(impostor_scores) == 0:
        return float("nan")

    y_true = np.concatenate([
        np.ones(len(genuine_scores), dtype=np.int32),
        np.zeros(len(impostor_scores), dtype=np.int32),
    ])
    y_score = np.concatenate([genuine_scores, impostor_scores])

    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
    fnr = 1.0 - tpr

    i = np.argmin(np.abs(fnr - fpr))
    eer = 0.5 * (fnr[i] + fpr[i])
    return float(eer)


# -----------------------------------------------------
# Main
# ---------------------------------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--real_root", required=True)
    ap.add_argument("--morph_root", required=True, nargs="+",
                    help="One or more morph image directories (e.g. one per blending ratio). "
                         "All are pooled into a single morph index.")
    ap.add_argument("--val_morph_root", nargs="*", default=[],
                    help="Optional separate morph root(s) for validation split. "
                         "Defaults to --morph_root if omitted.")
    ap.add_argument("--base_ckpt", required=True)
    ap.add_argument("--backbone", type=str, default="adaface", choices=["adaface", "arcface"],
                    help="Pretrained backbone to use (default: adaface).")

    # adapter checkpoint (instead of random initalization we load a pretrained adapter)
    ap.add_argument("--init_adapter_ckpt", type=str, default="", help="Optional adapter checkpoint to initialize from.")


    ap.add_argument("--pairlist_csv", type=str, default="")

    # enable hard morph sampling
    ap.add_argument("--hard_stats_csv", type=str, default="")
    ap.add_argument("--hard_mix", type=float, default=0.0)
    ap.add_argument("--hard_alpha", type=float, default=1.0)

    ap.add_argument("--split_dir", required=True)
    ap.add_argument("--split", choices=["train", "val"], default="train")
    ap.add_argument("--val_split", choices=["train", "val"], default="val")

    ap.add_argument("--out_dir", default="checkpoints_palma/adapter_run1")

    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=12)
    ap.add_argument("--prefetch_factor", type=int, default=4)
    ap.add_argument("--persistent_workers", action="store_true")

    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--save_every", type=int, default=2000)

    # Paper used lr=1e-1 with SGD+Nesterov; keep your defaults, override via CLI when needed
    ap.add_argument("--lr", type=float, default=1e-1)
    ap.add_argument("--weight_decay", type=float, default=0.0)  # keep simple; no scheduler
    ap.add_argument("--loss", type=str, default="tetra_ext",
                    choices=["tetra", "tetra_ext", "tetra_directed", "worst_case", "triplet_wc",
                             "balanced", "balanced_wc", "triplet_repulsion", "directed_repulsion"],
                    help="Loss function. tetra=plain TetraLoss, tetra_ext=TetraLoss+attraction, "
                         "tetra_directed=TetraLoss+hinged-attraction (self-limiting once d_mn < d_am), "
                         "worst_case=TetraLoss+worst-case morph embedding hinge, "
                         "triplet_wc=TripletLoss+worst-case hinge (decoupled bona fide and morph objectives), "
                         "balanced=TetraLoss with 50:50 random hardest selection (no WC term), "
                         "balanced_wc=balanced selection + worst-case hinge, "
                         "directed_repulsion=TripletRepulsion + directed impostor-attraction hinge.")
    ap.add_argument("--margin", type=float, default=3.0)
    ap.add_argument("--margin2", type=float, default=0.2,
                    help="Second margin for worst-case hinge term (TetraLossWorstCase only). "
                         "Recommended start: 0.2; increase to 0.3–0.5 if loss_wc goes to zero too early.")
    ap.add_argument("--lam", type=float, default=0.0,
                    help="Weight for secondary loss term (TetraLossExt / TetraLossDirected / TetraLossWorstCase). "
                         "0.0 = plain TetraLoss. Recommended: {0.01,0.1,0.5} for ext; {0.1,0.5,1.0} for directed/worst_case.")
    ap.add_argument("--lam_directed", type=float, default=0.3,
                    help="Weight on the A-side directed term in DirectedRepulsionLoss. "
                         "Default: 0.3 (conservative). Sweep {0.1,0.3,0.5,0.7} after first stable run.")
    ap.add_argument("--lam_dir_b", type=float, default=0.0,
                    help="Weight on the B-side directed term in DirectedRepulsionLoss. "
                         "Default 0.0 = original behaviour. Set equal to --lam_directed for symmetric directed.")
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--dataset_length", type=int, default=500000)

    # Validation
    ap.add_argument("--val_length", type=int, default=100000)
    ap.add_argument("--val_every", type=int, default=500)
    ap.add_argument("--val_batches", type=int, default=100)
    ap.add_argument("--patience_evals", type=int, default=8)
    ap.add_argument("--min_delta", type=float, default=1e-4)
    ap.add_argument("--early_stop", type=str, default="loss", choices=["loss", "eer_deer"],
                    help="Legacy early stopping criterion (used when --early_stop_metric is not eer_floor). "
                         "loss=best val_loss; eer_deer=stop when BOTH EER and D-EER stop improving.")
    ap.add_argument("--early_stop_metric", type=str, default="eer_floor",
                    choices=["eer_floor", "d_eer", "eer", "loss"],
                    help="Primary early stopping criterion. 'eer_floor' (default): saves best D-EER among "
                         "checkpoints satisfying val_eer <= (best_val_eer + eer_tolerance); stops when EER "
                         "violates the tolerance band or plateaus for eer_patience validations. "
                         "'d_eer': best val_d_eer regardless of EER. 'eer': best val_eer. "
                         "'loss': best val_loss. For the legacy eer_deer behaviour use "
                         "--early_stop_metric loss --early_stop eer_deer.")
    ap.add_argument("--eer_tolerance", type=float, default=0.005,
                    help="EER tolerance band (eer_floor only): val_eer may rise by at most this much above "
                         "its historical minimum before counting as a constraint violation. Default 0.005.")
    ap.add_argument("--eer_patience", type=int, default=5,
                    help="Consecutive validations with EER violation OR EER plateau before stopping "
                         "(eer_floor only). Default 5.")
    ap.add_argument("--val_pairlist_csv", type=str, default="")

    # Update loss values morph-wise (preparation for hard-morph mining)
    ap.add_argument("--morph_stats_every", type=int, default=1000)
    ap.add_argument("--morph_stats_csv", type=str, default="")

    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--unfreeze_output_layer", action="store_true",
                    help="Unfreeze the backbone output projection layer (Linear 25088→512) during "
                         "adapter training. All other backbone layers remain frozen. "
                         "Uses --output_layer_lr for that layer; adapter uses --lr as usual. "
                         "Default: False — fully frozen backbone, identical to standard runs.")
    ap.add_argument("--output_layer_lr", type=float, default=1e-6,
                    help="LR for the unfrozen backbone output layer (--unfreeze_output_layer only). "
                         "Default: 1e-6.")
    ap.add_argument("--unfreeze_last_block", action="store_true",
                    help="Unfreeze the last residual block group + output projection of the backbone. "
                         "AdaFace: backbone.body[-3:] (last 3 BasicBlockIR) + backbone.output_layer. "
                         "ArcFace: backbone.layer4 + backbone.fc. "
                         "Uses --backbone_lr for those layers; adapter uses --lr. "
                         "Mutually exclusive with --unfreeze_output_layer.")
    ap.add_argument("--backbone_lr", type=float, default=1e-6,
                    help="LR for the unfrozen backbone layers (--unfreeze_last_block only). Default: 1e-6.")

    # Paper vs. alternative: adapter on positive
    ap.add_argument(
        "--adapter_on_positive",
        action="store_true",
        help="If set, apply adapter to positive embeddings as well. Default: False (paper-aligned).",
    )

    # NEW: SGD momentum + optional Nesterov (paper: momentum=0.9, nesterov=True)
    ap.add_argument("--momentum", type=float, default=0.9, help="SGD momentum (paper: 0.9)")
    ap.add_argument("--nesterov", action="store_true", help="Use Nesterov momentum (paper: enabled)")

    # [LR-SCHED CHANGE] scheduler config
    ap.add_argument("--lr_sched", type=str, default="step", choices=["none", "step", "plateau"])
    ap.add_argument("--lr_step_size", type=int, default=12000, help="StepLR: decay every N steps (paper: every 3 epochs; default = 3 * (500000/128) ≈ 12000)")
    ap.add_argument("--lr_gamma", type=float, default=0.1, help="Decay factor (<1.0)")
    ap.add_argument("--lr_patience", type=int, default=3, help="ReduceLROnPlateau patience (in val evals)")
    ap.add_argument("--lr_min", type=float, default=1e-6, help="Minimum LR for plateau scheduler")

    # NEW: Weights & Biases
    ap.add_argument("--use_wandb", action="store_true")
    ap.add_argument("--wandb_project", type=str, default="ada_tetra")
    ap.add_argument("--wandb_entity", type=str, default="")
    ap.add_argument("--wandb_mode", type=str, default=os.getenv("WANDB_MODE", "offline"),
                    choices=["online", "offline", "disabled"])
    ap.add_argument("--wandb_run_name", type=str, default="")
    ap.add_argument("--wandb_dir", type=str, default=os.getenv("WANDB_DIR", ""))

    args = ap.parse_args()

    if getattr(args, "unfreeze_output_layer", False) and getattr(args, "unfreeze_last_block", False):
        ap.error("--unfreeze_output_layer and --unfreeze_last_block are mutually exclusive.")

    # --- val morph root selection (after args parse, before indexing) ---
    val_morph_root = args.val_morph_root if args.val_morph_root else args.morph_root

    # ---------------------------
    # Repro
    # --------------------------
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")

    # Performance toggles (GPU)
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    pin_memory = (device == "cuda")
    non_blocking = True  # safe; only truly helps when pin_memory=True

    # AMP
    use_amp = args.amp and (device == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    print(f"[AMP] enabled={use_amp}")

    # ------------------------------------
    # Indexing (split-aware)
    # -------------------------------------
    split_path = Path(args.split_dir) / ("train_ids.json" if args.split == "train" else "val_ids.json")
    split_ids = set(json.loads(split_path.read_text()))
    print(f"[Split] {args.split}: loaded {len(split_ids)} IDs from {split_path}")

    real_index_all = build_real_index(args.real_root)
    real_index = {i: real_index_all[i] for i in split_ids if i in real_index_all}
    morphs_by_id = build_morph_index(args.morph_root, allowed_ids=split_ids)

    print(f"[Index:{args.split}] real_ids={len(real_index)} morph_ids={len(morphs_by_id)}")

    # ------------------------------------
    # Validation Indexing (split-aware)
    # ------------------------------------
    val_split_path = Path(args.split_dir) / ("train_ids.json" if args.val_split == "train" else "val_ids.json")
    val_split_ids = set(json.loads(val_split_path.read_text()))
    print(f"[Val Split] {args.val_split}: loaded {len(val_split_ids)} IDs from {val_split_path}")

    val_real_index = {i: real_index_all[i] for i in val_split_ids if i in real_index_all}
    val_morphs_by_id = build_morph_index(val_morph_root, allowed_ids=val_split_ids)

    print(f"[Index:{args.val_split}] real_ids={len(val_real_index)} morph_ids={len(val_morphs_by_id)}")

    # ------------------------------------
    # Dataset / DataLoader
    # ------------------------------------
    ds = TetraQuadrupleDataset(
        real_index,
        morphs_by_id,
        transform=transform,
        length=args.dataset_length,
        seed=args.seed,
        pairlist_csv=args.pairlist_csv,
        hard_stats_csv=args.hard_stats_csv,
        hard_mix=args.hard_mix,
        hard_alpha=args.hard_alpha,
        load_contributor_b=(args.loss in ("worst_case", "triplet_wc", "balanced_wc", "triplet_repulsion", "directed_repulsion")),
    )

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=(args.persistent_workers and args.num_workers > 0),
        prefetch_factor=(args.prefetch_factor if args.num_workers > 0 else None),
        drop_last=True,
    )

    val_ds = TetraQuadrupleDataset(
        val_real_index,
        val_morphs_by_id,
        transform=transform,
        length=args.val_length,
        seed=args.seed + 1337,
        pairlist_csv=args.val_pairlist_csv,
        load_contributor_b=(args.loss in ("worst_case", "triplet_wc", "balanced_wc", "triplet_repulsion", "directed_repulsion")),
        load_probe_b=True,
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=(args.persistent_workers and args.num_workers > 0),
        prefetch_factor=(args.prefetch_factor if args.num_workers > 0 else None),
        drop_last=False,
    )

    # --------------------------------
    # Models
    # --------------------------------
    if args.backbone == "arcface":
        backbone = load_arcface_ir50(args.base_ckpt, device=device, strict=False)
    else:
        backbone = load_adaface_ir50(args.base_ckpt, device=device, strict=False)
    print(f"[Model] backbone={args.backbone}")
    adapter = MLPAdapter(embedding_size=512)

    # New: load pretrained adapter (triplet baseline)
    if args.init_adapter_ckpt:
        init_path = Path(args.init_adapter_ckpt)
        if not init_path.exists():
            raise FileNotFoundError(f"init_adapter_ckpt not found: {init_path}")
        state = torch.load(init_path, map_location="cpu")
        # support both plain state_dict and wrapped ckpt.dict
        if isinstance(state, dict) and "adapter" in state:
            state = state["adapter"]
        adapter.load_state_dict(state, strict=True)
        print(f"[Init] loaded adapter weights from {init_path}")

    set_requires_grad(backbone, False)
    set_requires_grad(adapter, True)

    backbone.to(device).eval()  # frozen by default
    adapter.to(device).train()

    # List of backbone module objects put in train() mode — used for mode restoration
    # after validation calls that internally call backbone.eval().
    unfrozen_bb_modules = []
    if args.unfreeze_output_layer:
        module = unfreeze_output_layer(backbone, args.backbone)
        unfrozen_bb_modules = [module]
        module.train()
        print(f"[Unfreeze] backbone output layer unfrozen | output_layer_lr={args.output_layer_lr}")
    elif args.unfreeze_last_block:
        unfrozen_bb_modules = unfreeze_last_block(backbone, args.backbone)
        for m in unfrozen_bb_modules:
            m.train()
        n_ub = sum(p.numel() for m in unfrozen_bb_modules for p in m.parameters())
        print(f"[Unfreeze] last block + output layer unfrozen ({n_ub:,} params) | backbone_lr={args.backbone_lr}")

    log_freeze_state(backbone, adapter)
    print(f"[Params] adapter trainable: {count_trainable_params(adapter):,}")
    print(
        "[Mode] adapter_on_positive="
        f"{args.adapter_on_positive} "
        "(False=paper-aligned: adapter only on anchor/negative/morph; True=adapter on all)"
    )

    # --------------------------------
    # Optim / Loss (SGD + optional Nesterov)
    # ---------------------------------
    if args.loss == "tetra":
        loss_fn = TetraLoss(margin=args.margin).to(device)
    elif args.loss == "tetra_ext":
        loss_fn = TetraLossExt(margin=args.margin, lam=args.lam).to(device)
    elif args.loss == "tetra_directed":
        loss_fn = TetraLossDirected(margin=args.margin, lam=args.lam).to(device)
    elif args.loss == "worst_case":
        loss_fn = TetraLossWorstCase(margin=args.margin, margin2=args.margin2, lam=args.lam).to(device)
    elif args.loss == "triplet_wc":
        loss_fn = TripletWorstCase(margin=args.margin, margin2=args.margin2, lam=args.lam).to(device)
    elif args.loss == "balanced":
        loss_fn = TetraLossBalanced(margin=args.margin).to(device)
    elif args.loss == "balanced_wc":
        loss_fn = TetraLossWorstCaseBalanced(margin=args.margin, margin2=args.margin2, lam=args.lam).to(device)
    elif args.loss == "triplet_repulsion":
        loss_fn = TripletRepulsionLoss(margin=args.margin, margin_repel=args.margin2, lam=args.lam).to(device)
    elif args.loss == "directed_repulsion":
        loss_fn = DirectedRepulsionLoss(
            margin=args.margin,
            margin_repel=args.margin2,
            lam_repel=args.lam,
            lam_directed=args.lam_directed,
            lam_directed_b=args.lam_dir_b,
        ).to(device)
    print(f"[Loss] {args.loss} margin={args.margin} margin2={args.margin2} lam={args.lam}"
          + (f" lam_directed={args.lam_directed} lam_dir_b={args.lam_dir_b}" if args.loss == "directed_repulsion" else ""))

    # For triplet_wc, validate on the same TripletWorstCase loss — early stopping
    # should reflect the actual training objective, not TetraLoss geometry.
    # All other loss variants validate on plain TetraLoss for cross-run comparability.
    if args.loss == "triplet_wc":
        val_loss_fn = TripletWorstCase(margin=args.margin, margin2=args.margin2, lam=args.lam).to(device)
    elif args.loss == "balanced_wc":
        val_loss_fn = TetraLossWorstCaseBalanced(margin=args.margin, margin2=args.margin2, lam=args.lam).to(device)
    elif args.loss == "triplet_repulsion":
        val_loss_fn = TripletRepulsionLoss(margin=args.margin, margin_repel=args.margin2, lam=args.lam).to(device)
    elif args.loss == "directed_repulsion":
        val_loss_fn = DirectedRepulsionLoss(
            margin=args.margin,
            margin_repel=args.margin2,
            lam_repel=args.lam,
            lam_directed=args.lam_directed,
            lam_directed_b=args.lam_dir_b,
        ).to(device)
    else:
        val_loss_fn = TetraLoss(margin=args.margin).to(device)

    if args.unfreeze_last_block:
        bb_trainable = [p for p in backbone.parameters() if p.requires_grad]
        param_groups = [
            {"params": list(adapter.parameters()), "lr": args.lr},
            {"params": bb_trainable,               "lr": args.backbone_lr},
        ]
        opt = torch.optim.SGD(
            param_groups,
            momentum=args.momentum,
            nesterov=args.nesterov,
            weight_decay=args.weight_decay,
        )
        print(f"[Opt] SGD  adapter lr={args.lr}  backbone lr={args.backbone_lr}  "
              f"momentum={args.momentum} nesterov={args.nesterov} weight_decay={args.weight_decay}")
    elif args.unfreeze_output_layer:
        bb_output_params = [p for p in backbone.parameters() if p.requires_grad]
        param_groups = [
            {"params": list(adapter.parameters()), "lr": args.lr},
            {"params": bb_output_params,           "lr": args.output_layer_lr},
        ]
        opt = torch.optim.SGD(
            param_groups,
            momentum=args.momentum,
            nesterov=args.nesterov,
            weight_decay=args.weight_decay,
        )
        print(f"[Opt] SGD  adapter lr={args.lr}  output_layer lr={args.output_layer_lr}  "
              f"momentum={args.momentum} nesterov={args.nesterov} weight_decay={args.weight_decay}")
    else:
        opt = torch.optim.SGD(
            adapter.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            nesterov=args.nesterov,
            weight_decay=args.weight_decay,
        )
        print(f"[Opt] SGD lr={args.lr} momentum={args.momentum} nesterov={args.nesterov} weight_decay={args.weight_decay}")
    print("[Paper] typical: --lr 1e-1 --momentum 0.9 --nesterov (and possibly --weight_decay 1e-1)")

    # [LR-SCHED CHANGE] create optional scheduler
    scheduler = None
    if args.lr_sched == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            opt,
            step_size=args.lr_step_size,
            gamma=args.lr_gamma,
        )
    elif args.lr_sched == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min",
            factor=args.lr_gamma,
            patience=args.lr_patience,
            min_lr=args.lr_min,
        )

    print(f"[Sched] type={args.lr_sched} gamma={args.lr_gamma}")

    # save run config
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "run_args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    wandb_run = init_wandb(args)

    # -------------------------
    # Training Loop
    # -------------------------
    # Select adapted-encoding function once — avoids per-step branching inside the loop.
    # encode_adapter_with_grad keeps the computation graph alive through any unfrozen
    # backbone layers so they receive gradients.
    _enc_adapted = (encode_adapter_with_grad
                    if (args.unfreeze_output_layer or args.unfreeze_last_block)
                    else encode_adapter)

    # All trainable parameters — used for gradient clipping across adapter + backbone.
    _all_trainable = (list(adapter.parameters()) +
                      [p for p in backbone.parameters() if p.requires_grad])

    it = iter(dl)
    seen = 0
    t0 = time.time()

    best_val  = float("inf")
    best_eer  = float("inf")
    best_deer = float("inf")
    bad_evals = 0
    best_step = 0
    last_step = 0

    # eer_floor criterion state
    best_val_d_eer_under_constraint = float("inf")
    eer_violation_count = 0
    eer_no_improve_count = 0

    morph_stats = {}
    loss_tetra_vec = None
    loss_attract_vec = None
    zb = None  # contributor B embedding (worst_case loss only)

    for step in range(1, args.steps + 1):
        last_step = step
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dl)
            batch = next(it)

        opt.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            # anchor / negative / morph always go through adapter (+ output layer if unfrozen)
            za = _enc_adapted(backbone, adapter, batch["anchor"],   device, use_amp=use_amp, non_blocking=non_blocking)
            zn = _enc_adapted(backbone, adapter, batch["negative"], device, use_amp=use_amp, non_blocking=non_blocking)
            zm = _enc_adapted(backbone, adapter, batch["morph"],    device, use_amp=use_amp, non_blocking=non_blocking)

            # Positive depends on mode:
            # - paper-aligned (default): probe / backbone only — output layer not touched from this side
            # - adapter_on_positive: also goes through the full adapted path
            if args.adapter_on_positive:
                zp = _enc_adapted(backbone, adapter, batch["positive"], device, use_amp=use_amp, non_blocking=non_blocking)
            else:
                zp = encode_backbone(backbone, batch["positive"], device, use_amp=use_amp, non_blocking=non_blocking)

            if args.loss in ("worst_case", "triplet_wc", "balanced_wc", "triplet_repulsion", "directed_repulsion"):
                zb = _enc_adapted(backbone, adapter, batch["contributor_b"], device, use_amp=use_amp, non_blocking=non_blocking)
                result = loss_fn(za, zp, zn, zm, zb, reduction="none")
            else:
                result = loss_fn(za, zp, zn, zm, reduction="none")

            loss_directed_a_vec = loss_directed_b_vec = None
            if isinstance(result, tuple):
                if args.loss == "directed_repulsion":
                    loss_vec, loss_tetra_vec, loss_attract_vec, loss_directed_a_vec, loss_directed_b_vec = result
                else:
                    loss_vec, loss_tetra_vec, loss_attract_vec = result
            else:
                loss_vec = result
                loss_tetra_vec = loss_attract_vec = None
            loss = loss_vec.mean()

            # DataLoader collates tuple fields column-wise: paths = (anchor_paths, pos_paths, morph_paths, neg_paths)
            morph_paths = batch["paths"][2]
            update_morph_loss_stats(morph_stats, morph_paths, loss_vec, step)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(_all_trainable, args.grad_clip)
            scaler.step(opt)
            scaler.update()

            # [LR-SCHED CHANGE] step scheduler after optimizer update
            if scheduler is not None and args.lr_sched == "step":
                scheduler.step()

        else:
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(_all_trainable, args.grad_clip)
            opt.step()

            # [LR-SCHED CHANGE] step scheduler after optimizer update
            if scheduler is not None and args.lr_sched == "step":
                scheduler.step()

        seen += args.batch_size

        if step % args.log_every == 0:
            (s_ap, s_an, s_am), (d_ap, d_an, d_am) = quick_reports_adapter(
                backbone,
                adapter,
                batch,
                device,
                use_amp=use_amp,
                non_blocking=non_blocking,
                adapter_on_positive=args.adapter_on_positive,
            )
            # quick_reports_adapter calls backbone.eval() and restores backbone.training (False),
            # which resets unfrozen layers to eval — re-apply train mode.
            for m in unfrozen_bb_modules:
                m.train()
            dt = time.time() - t0
            sub_loss_str = ""
            if loss_tetra_vec is not None:
                if args.loss == "triplet_wc":
                    sub_loss_str = (f" | loss_triplet={loss_tetra_vec.mean().item():.4f}"
                                    f" loss_wc={loss_attract_vec.mean().item():.4f}")
                elif args.loss in ("worst_case", "balanced_wc"):
                    sub_loss_str = (f" | loss_tetra={loss_tetra_vec.mean().item():.4f}"
                                    f" loss_wc={loss_attract_vec.mean().item():.4f}")
                elif args.loss == "triplet_repulsion":
                    sub_loss_str = (f" | loss_triplet={loss_tetra_vec.mean().item():.4f}"
                                    f" loss_repel={loss_attract_vec.mean().item():.4f}")
                elif args.loss == "directed_repulsion":
                    sub_loss_str = (f" | loss_triplet={loss_tetra_vec.mean().item():.4f}"
                                    f" loss_repel={loss_attract_vec.mean().item():.4f}"
                                    f" loss_dir_a={loss_directed_a_vec.mean().item():.4f}"
                                    f" loss_dir_b={loss_directed_b_vec.mean().item():.4f}")
                else:
                    sub_loss_str = (f" | loss_tetra={loss_tetra_vec.mean().item():.4f}"
                                    f" loss_attract={loss_attract_vec.mean().item():.4f}")

            # Worst-case diagnostics: d(morph, y*) and cos(morph, y*)
            wc_diag_str = ""
            if args.loss in ("worst_case", "triplet_wc", "balanced_wc") and zb is not None:
                with torch.no_grad():
                    y_star = torch.nn.functional.normalize(za.detach() + zb.detach(), p=2, dim=1)
                    d_m_ystar = torch.norm(zm.detach() - y_star, p=2, dim=1).mean().item()
                    cos_m_ystar = (zm.detach() * y_star).sum(dim=1).mean().item()
                wc_diag_str = f" | d_m_y*={d_m_ystar:.3f} cos_m_y*={cos_m_ystar:.3f}"

            print(
                f"step={step:6d}/{args.steps} loss={loss.item():.4f}{sub_loss_str}{wc_diag_str} | "
                f"cos: ap={s_ap:.3f} an={s_an:.3f} am={s_am:.3f} | "
                f"L2: ap={d_ap:.3f} an={d_an:.3f} am={d_am:.3f} | "
                f"img/s={seen/max(dt, 1e-6):.1f}"
            )
            log_dict = {
                "train/loss": float(loss.item()),
                "train/cos_ap": float(s_ap),
                "train/cos_an": float(s_an),
                "train/cos_am": float(s_am),
                "train/l2_ap": float(d_ap),
                "train/l2_an": float(d_an),
                "train/l2_am": float(d_am),
                "train/img_per_s": float(seen/max(dt, 1e-6)),
                "optim/lr": float(opt.param_groups[0]["lr"]),
            }
            if loss_tetra_vec is not None:
                if args.loss == "triplet_wc":
                    log_dict["train/loss_triplet"] = float(loss_tetra_vec.mean().item())
                    log_dict["train/loss_wc"] = float(loss_attract_vec.mean().item())
                elif args.loss in ("worst_case", "balanced_wc"):
                    log_dict["train/loss_tetra"] = float(loss_tetra_vec.mean().item())
                    log_dict["train/loss_wc"] = float(loss_attract_vec.mean().item())
                elif args.loss == "triplet_repulsion":
                    log_dict["train/loss_triplet"] = float(loss_tetra_vec.mean().item())
                    log_dict["train/loss_repel"]    = float(loss_attract_vec.mean().item())
                elif args.loss == "directed_repulsion":
                    log_dict["train/loss_triplet"]    = float(loss_tetra_vec.mean().item())
                    log_dict["train/loss_repel"]      = float(loss_attract_vec.mean().item())
                    log_dict["train/loss_directed_a"] = float(loss_directed_a_vec.mean().item())
                    log_dict["train/loss_directed_b"] = float(loss_directed_b_vec.mean().item())
                else:
                    log_dict["train/loss_tetra"] = float(loss_tetra_vec.mean().item())
                    log_dict["train/loss_attract"] = float(loss_attract_vec.mean().item())
            if args.loss in ("worst_case", "triplet_wc", "balanced_wc") and zb is not None:
                log_dict["train/d_m_ystar"] = float(d_m_ystar)
                log_dict["train/cos_m_ystar"] = float(cos_m_ystar)
            # triplet_repulsion diagnostics: cos/L2 for contributor B, per-term active fractions
            if args.loss == "triplet_repulsion" and zb is not None:
                with torch.no_grad():
                    _za = za.detach(); _zp = zp.detach(); _zn = zn.detach()
                    _zm = zm.detach(); _zb = zb.detach()
                    cos_bm = (_zb * _zm).sum(dim=1).mean().item()
                    l2_bm  = torch.norm(_zb - _zm, dim=1).mean().item()
                    _dap = torch.norm(_za - _zp, p=2, dim=1)
                    _dan = torch.norm(_za - _zn, p=2, dim=1)
                    _dam = torch.norm(_za - _zm, p=2, dim=1)
                    _dbm = torch.norm(_zb - _zm, p=2, dim=1)
                    m1   = loss_fn.margin
                    m2   = loss_fn.margin_repel
                    frac_t  = ((_dap + m1 - _dan) > 0).float().mean().item()
                    frac_ra = ((_dap + m2 - _dam) > 0).float().mean().item()
                    frac_rb = ((_dap + m2 - _dbm) > 0).float().mean().item()
                    loss_repel_a_val = F.relu(_dap + m2 - _dam).mean().item()
                    loss_repel_b_val = F.relu(_dap + m2 - _dbm).mean().item()
                log_dict["train/cos_bm"]         = float(cos_bm)
                log_dict["train/l2_bm"]          = float(l2_bm)
                log_dict["train/loss_repel_a"]   = float(loss_repel_a_val)
                log_dict["train/loss_repel_b"]   = float(loss_repel_b_val)
                log_dict["train/active_triplet"] = float(frac_t)
                log_dict["train/active_repel_a"] = float(frac_ra)
                log_dict["train/active_repel_b"] = float(frac_rb)
                print(f"  [repulsion] cos_bm={cos_bm:.3f} L2_bm={l2_bm:.3f} "
                      f"active: t={frac_t:.2f} ra={frac_ra:.2f} rb={frac_rb:.2f} "
                      f"repel_a={loss_repel_a_val:.4f} repel_b={loss_repel_b_val:.4f}")
            # directed_repulsion diagnostics: cos/L2 for B and n, per-term active fractions
            if args.loss == "directed_repulsion" and zb is not None:
                with torch.no_grad():
                    _za = za.detach(); _zp = zp.detach(); _zn = zn.detach()
                    _zm = zm.detach(); _zb = zb.detach()
                    cos_bm = (_zb * _zm).sum(dim=1).mean().item()
                    cos_mn = (_zm * _zn).sum(dim=1).mean().item()
                    l2_bm  = torch.norm(_zb - _zm, dim=1).mean().item()
                    _dap = torch.norm(_za - _zp, p=2, dim=1)
                    _dan = torch.norm(_za - _zn, p=2, dim=1)
                    _dam = torch.norm(_za - _zm, p=2, dim=1)
                    _dbm = torch.norm(_zb - _zm, p=2, dim=1)
                    _dmn = torch.norm(_zm - _zn, p=2, dim=1)
                    m1   = loss_fn.margin
                    m2   = loss_fn.margin_repel
                    frac_t  = ((_dap + m1 - _dan)  > 0).float().mean().item()
                    frac_ra = ((_dap + m2 - _dam)  > 0).float().mean().item()
                    frac_rb = ((_dap + m2 - _dbm)  > 0).float().mean().item()
                    frac_d_a = ((_dmn - _dam) > 0).float().mean().item()
                    frac_d_b = ((_dmn - _dbm) > 0).float().mean().item()
                    loss_repel_a_val = F.relu(_dap + m2 - _dam).mean().item()
                    loss_repel_b_val = F.relu(_dap + m2 - _dbm).mean().item()
                log_dict["train/cos_bm"]            = float(cos_bm)
                log_dict["train/cos_mn"]            = float(cos_mn)
                log_dict["train/l2_bm"]             = float(l2_bm)
                log_dict["train/loss_repel_a"]      = float(loss_repel_a_val)
                log_dict["train/loss_repel_b"]      = float(loss_repel_b_val)
                log_dict["train/active_triplet"]    = float(frac_t)
                log_dict["train/active_repel_a"]    = float(frac_ra)
                log_dict["train/active_repel_b"]    = float(frac_rb)
                log_dict["train/active_directed_a"] = float(frac_d_a)
                log_dict["train/active_directed_b"] = float(frac_d_b)
                print(f"  [dir_repulsion] cos_bm={cos_bm:.3f} cos_mn={cos_mn:.3f} L2_bm={l2_bm:.3f} "
                      f"active: t={frac_t:.2f} ra={frac_ra:.2f} rb={frac_rb:.2f} "
                      f"da={frac_d_a:.2f} db={frac_d_b:.2f} "
                      f"repel_a={loss_repel_a_val:.4f} repel_b={loss_repel_b_val:.4f}")
            wandb_log(wandb_run, log_dict, step=step)

        if step % args.val_every == 0:
            val_metrics = compute_val_metrics(
                backbone=backbone,
                adapter=adapter,
                loss_fn=val_loss_fn,
                val_dl=val_dl,
                device=device,
                use_amp=use_amp,
                adapter_on_positive=args.adapter_on_positive,
                max_batches=args.val_batches,
                need_contributor_b=(args.loss in ("triplet_wc", "balanced_wc", "triplet_repulsion", "directed_repulsion")),
                need_directed=(args.loss == "directed_repulsion"),
            )
            # compute_val_metrics calls backbone.eval() internally and restores backbone.training
            # (False), which resets unfrozen layers to eval — re-apply train mode.
            for m in unfrozen_bb_modules:
                m.train()
            vloss = val_metrics["val_loss"]
            eer   = val_metrics["eer"]
            d_eer = val_metrics["d_eer"]
            mmpmr = val_metrics["mmpmr"]

            if args.early_stop_metric == "eer_floor":
                print(f"[VAL] step={step} val_loss={vloss:.4f} eer={eer:.4f} d_eer={d_eer:.4f} mmpmr={mmpmr:.4f}")
            elif args.early_stop == "eer_deer":
                print(f"[VAL] step={step} val_loss={vloss:.4f} eer={eer:.4f} d_eer={d_eer:.4f} mmpmr={mmpmr:.4f} "
                      f"best_eer={best_eer:.4f} best_deer={best_deer:.4f} bad_evals={bad_evals}")
            else:
                print(f"[VAL] step={step} val_loss={vloss:.4f} eer={eer:.4f} d_eer={d_eer:.4f} mmpmr={mmpmr:.4f} "
                      f"best_val={best_val:.4f} bad_evals={bad_evals}")

            if scheduler is not None and args.lr_sched == "plateau":
                scheduler.step(vloss)

            # W&B val scalars + distributions
            if wandb_run is not None:
                val_log = {
                    "val/loss": float(vloss),
                    "val/eer": float(eer),
                    "val/d_eer": float(d_eer),
                    "val/mmpmr": float(mmpmr),
                    "val/best_loss": float(best_val),
                    "val/bad_evals": int(bad_evals),
                    "optim/lr": float(opt.param_groups[0]["lr"]),
                    "scores/genuine_mean": val_metrics["genuine_mean"],
                    "scores/impostor_mean": val_metrics["impostor_mean"],
                    "scores/morph_mean": val_metrics["morph_mean"],
                    "scores/genuine_hist": wandb.Histogram(val_metrics["genuine_scores"]),
                    "scores/impostor_hist": wandb.Histogram(val_metrics["impostor_scores"]),
                    "scores/morph_hist": wandb.Histogram(val_metrics["morph_scores"]),
                }
                if args.loss == "directed_repulsion" and "val_loss_triplet" in val_metrics:
                    val_log["val/loss_triplet"]    = val_metrics["val_loss_triplet"]
                    val_log["val/loss_repel"]      = val_metrics["val_loss_repel"]
                    val_log["val/loss_directed_a"] = val_metrics["val_loss_directed_a"]
                    val_log["val/loss_directed_b"] = val_metrics["val_loss_directed_b"]
                    val_log["val/cos_mn"]          = val_metrics["val_cos_mn"]
                wandb_log(wandb_run, val_log, step=step)

            # -------------------------------------------------------
            # Early stopping logic
            # -------------------------------------------------------
            if args.early_stop_metric == "eer_floor":
                # Track best EER
                if eer < best_eer:
                    best_eer = eer
                    eer_no_improve_count = 0
                else:
                    eer_no_improve_count += 1

                eer_satisfies_constraint = eer <= (best_eer + args.eer_tolerance)

                if not eer_satisfies_constraint:
                    eer_violation_count += 1
                else:
                    eer_violation_count = 0
                    # Within EER constraint: save if this is the best D-EER so far
                    if d_eer < best_val_d_eer_under_constraint:
                        best_val_d_eer_under_constraint = d_eer
                        best_step = step
                        _extra = _backbone_extra_sds(backbone, args)
                        if _extra:
                            d = {"adapter": adapter.state_dict()}
                            d.update(_extra)
                            torch.save(d, out_dir / "adapter_best_val.pt")
                        else:
                            torch.save(adapter.state_dict(), out_dir / "adapter_best_val.pt")
                        best_val_ckpt = {
                            "step": step,
                            "val_eer": eer,
                            "val_d_eer": d_eer,
                            "best_val_eer_at_save": best_eer,
                            "eer_tolerance": args.eer_tolerance,
                            "early_stop_metric": args.early_stop_metric,
                            "adapter": adapter.state_dict(),
                            "optimizer": opt.state_dict(),
                            "scaler": None if scaler is None else scaler.state_dict(),
                            "lr_scheduler": None if scheduler is None else scheduler.state_dict(),
                            "args": vars(args),
                        }
                        best_val_ckpt.update(_extra)
                        torch.save(best_val_ckpt, out_dir / "ckpt_best_val.pt")
                        print(f"  → [EERFloor] new best: val_eer={eer:.4f} val_d_eer={d_eer:.4f} "
                              f"(best_eer={best_eer:.4f} tol={args.eer_tolerance})")

                print(
                    f"[VAL/eer_floor] step={step} val_eer={eer:.4f} val_d_eer={d_eer:.4f} "
                    f"best_eer={best_eer:.4f} "
                    f"cstr={'OK' if eer_satisfies_constraint else 'VIOLATED'} "
                    f"viol={eer_violation_count}/{args.eer_patience} "
                    f"no_impr={eer_no_improve_count}/{args.eer_patience} "
                    f"best_d_eer_cstr={best_val_d_eer_under_constraint:.4f}"
                )

                if wandb_run is not None:
                    wandb_log(wandb_run, {
                        "val/best_eer": float(best_eer),
                        "val/best_d_eer_constrained": float(best_val_d_eer_under_constraint),
                        "val/eer_violation_count": int(eer_violation_count),
                        "val/eer_constraint_satisfied": int(eer_satisfies_constraint),
                        "val/eer_no_improve_count": int(eer_no_improve_count),
                    }, step=step)

                if eer_violation_count >= args.eer_patience:
                    print(f"[EarlyStop/eer_floor] step={step}: val_eer exceeded tolerance "
                          f"({args.eer_tolerance}) for {args.eer_patience} consecutive validations. "
                          f"best_step={best_step} best_d_eer_cstr={best_val_d_eer_under_constraint:.4f}")
                    break
                if eer_no_improve_count >= args.eer_patience:
                    print(f"[EarlyStop/eer_floor] step={step}: val_eer plateaued for "
                          f"{args.eer_patience} consecutive validations. "
                          f"best_step={best_step} best_eer={best_eer:.4f} "
                          f"best_d_eer_cstr={best_val_d_eer_under_constraint:.4f}")
                    break

            else:
                # Simple / legacy criteria
                if args.early_stop_metric == "d_eer":
                    improved = d_eer < (best_deer - args.min_delta)
                    if improved:
                        best_deer = d_eer
                elif args.early_stop_metric == "eer":
                    improved = eer < (best_eer - args.min_delta)
                    if improved:
                        best_eer = eer
                elif args.early_stop == "eer_deer":
                    eer_improved  = eer  < (best_eer  - args.min_delta)
                    deer_improved = d_eer < (best_deer - args.min_delta)
                    improved = eer_improved or deer_improved
                    if eer_improved:
                        best_eer = eer
                    if deer_improved:
                        best_deer = d_eer
                else:  # loss
                    improved = vloss < (best_val - args.min_delta)
                    if improved:
                        best_val = vloss

                if improved:
                    best_step = step
                    bad_evals = 0
                    _extra = _backbone_extra_sds(backbone, args)
                    if _extra:
                        d = {"adapter": adapter.state_dict()}
                        d.update(_extra)
                        torch.save(d, out_dir / "adapter_best_val.pt")
                    else:
                        torch.save(adapter.state_dict(), out_dir / "adapter_best_val.pt")
                    best_val_ckpt = {
                        "step": step,
                        "best_val": best_val,
                        "best_eer": best_eer,
                        "best_deer": best_deer,
                        "adapter": adapter.state_dict(),
                        "optimizer": opt.state_dict(),
                        "scaler": None if scaler is None else scaler.state_dict(),
                        "lr_scheduler": None if scheduler is None else scheduler.state_dict(),
                        "args": vars(args),
                    }
                    best_val_ckpt.update(_extra)
                    torch.save(best_val_ckpt, out_dir / "ckpt_best_val.pt")
                    if args.early_stop_metric == "d_eer":
                        print(f"[Best] step={step} d_eer={d_eer:.4f}")
                    elif args.early_stop_metric == "eer":
                        print(f"[Best] step={step} eer={eer:.4f}")
                    elif args.early_stop == "eer_deer":
                        print(f"[Best] step={step} eer={eer:.4f} d_eer={d_eer:.4f}")
                    else:
                        print(f"[Best] step={step} val_loss={vloss:.4f}")
                else:
                    bad_evals += 1
                    if bad_evals >= args.patience_evals:
                        if args.early_stop_metric == "d_eer":
                            print(f"[EarlyStop] step={step} best_step={best_step} best_d_eer={best_deer:.4f}")
                        elif args.early_stop_metric == "eer":
                            print(f"[EarlyStop] step={step} best_step={best_step} best_eer={best_eer:.4f}")
                        elif args.early_stop == "eer_deer":
                            print(f"[EarlyStop] step={step} best_step={best_step} "
                                  f"best_eer={best_eer:.4f} best_deer={best_deer:.4f}")
                        else:
                            print(f"[EarlyStop] step={step} best_step={best_step} best_val={best_val:.4f}")
                        break

        if step % args.save_every == 0:
            save_ckpt(args.out_dir, step, adapter, opt, scaler, args, scheduler=scheduler,
                      backbone_extra_sds=_backbone_extra_sds(backbone, args))

        if args.morph_stats_csv and (step % args.morph_stats_every == 0):
            flush_morph_loss_stats_csv(args.morph_stats_csv, morph_stats)
            print(f"[MorphStats] flushed {len(morph_stats)} morph entries to {args.morph_stats_csv}")

    # final
    _extra = _backbone_extra_sds(backbone, args)
    if _extra:
        d = {"adapter": adapter.state_dict()}
        d.update(_extra)
        torch.save(d, out_dir / "adapter_final.pt")
    else:
        torch.save(adapter.state_dict(), out_dir / "adapter_final.pt")
    final_ckpt = {
        "step": last_step,
        "adapter": adapter.state_dict(),
        "optimizer": opt.state_dict(),
        "scaler": None if scaler is None else scaler.state_dict(),
        "lr_scheduler": None if scheduler is None else scheduler.state_dict(),
        "args": vars(args),
    }
    final_ckpt.update(_extra)
    torch.save(final_ckpt, out_dir / "ckpt_final.pt")

    if args.morph_stats_csv:
        flush_morph_loss_stats_csv(args.morph_stats_csv, morph_stats)
        print(f"[MorphStats] final flush -> {args.morph_stats_csv}")

    if wandb_run is not None:
        wandb.finish()

    print("[Done] Training complete")


if __name__ == "__main__":
    main()



