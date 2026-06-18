# train_adapter_triplet.py
import time
import json
import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


from detection.models.adaface.loader import load_adaface_ir50, load_arcface_ir50
from detection.losses import TripletLoss
from detection.data.dataset_triplet import TripletDataset
from detection.data.index_parser import build_real_index
from detection.data.preprocessing import transform
from detection.models.mlp_adapter import MLPAdapter


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

    # Positive depends on mode
    if adapter_on_positive:
        zp = enc_adapted(batch["positive"])
    else:
        zp = enc_base(batch["positive"])

    # cosine
    s_ap = (za * zp).sum(dim=1).mean().item()
    s_an = (za * zn).sum(dim=1).mean().item()

    # euclidean (unit sphere)
    d_ap = torch.norm(za - zp, dim=1).mean().item()
    d_an = torch.norm(za - zn, dim=1).mean().item()

    backbone.train(was_bb_training)
    adapter.train(was_ad_training)

    return (s_ap, s_an), (d_ap, d_an)


def save_ckpt(out_dir, step, adapter, opt, scaler, args):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "step": step,
        "adapter": adapter.state_dict(),
        "optimizer": opt.state_dict(),
        "scaler": None if scaler is None else scaler.state_dict(),
        "args": vars(args),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "py_rng": random.getstate(),
    }

    torch.save(ckpt, out_dir / f"ckpt_step{step}.pt")
    torch.save(adapter.state_dict(), out_dir / f"adapter_step{step}.pt")
    print(f"[Save] step {step}")

def compute_eer(genuine: torch.Tensor, impostor: torch.Tensor, n_thresh: int = 4001) -> float:
    if genuine.numel() == 0 or impostor.numel() == 0:
        return float("nan")
    g   = torch.sort(genuine.float()).values
    imp = torch.sort(impostor.float()).values
    th  = torch.quantile(torch.cat([g, imp]), torch.linspace(0.0, 1.0, n_thresh)).float()
    fnmr = torch.searchsorted(g,   th, right=False).float() / g.numel()
    fmr  = 1.0 - torch.searchsorted(imp, th, right=False).float() / imp.numel()
    idx  = torch.argmin(torch.abs(fnmr - fmr))
    return float((0.5 * (fnmr[idx] + fmr[idx])).item())


## VALIDATION
@torch.no_grad()
def compute_val_metrics(backbone, adapter, loss_fn, val_dl, device, use_amp, adapter_on_positive, max_batches):
    was_bb_training = backbone.training
    was_ad_training = adapter.training

    backbone.eval()
    adapter.eval()

    vals = []
    genuine_scores  = []
    impostor_scores = []

    for bi, batch in enumerate(val_dl):
        if bi >= max_batches:
            break
        with torch.cuda.amp.autocast(enabled=use_amp):
            za = encode_adapter(backbone, adapter, batch["anchor"],   device, use_amp=use_amp)
            zn = encode_adapter(backbone, adapter, batch["negative"], device, use_amp=use_amp)

            if adapter_on_positive:
                zp = encode_adapter(backbone, adapter, batch["positive"], device, use_amp=use_amp)
            else:
                zp = encode_backbone(backbone, batch["positive"], device, use_amp=use_amp)

            vals.append(loss_fn(za, zp, zn).item())

        # Asymmetric scoring — mirrors train_adapter.py / eval.py deployment convention:
        #   reference side (document) → adapter embedding
        #   probe side     (live capture, bona fide) → backbone embedding only
        # genuine:  adapter(anchor) · backbone(positive)  — already zp when not adapter_on_positive
        # impostor: adapter(anchor) · backbone(negative)  — re-encode negative without adapter
        zn_backbone = encode_backbone(backbone, batch["negative"], device, use_amp=use_amp)
        genuine_scores.append((za * zp).sum(dim=1).detach().cpu())
        impostor_scores.append((za * zn_backbone).sum(dim=1).detach().cpu())

    backbone.train(was_bb_training)
    adapter.train(was_ad_training)

    if len(vals) == 0:
        raise RuntimeError("Validation produced 0 batches. Check val split and val_batches.")

    g_cat   = torch.cat(genuine_scores)  if genuine_scores  else torch.empty(0)
    imp_cat = torch.cat(impostor_scores) if impostor_scores else torch.empty(0)

    return {
        "val_loss": float(sum(vals) / len(vals)),
        "eer":      compute_eer(g_cat, imp_cat),
    }

# -----------------------------------------------------
# Main
# ---------------------------------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--real_root", required=True)
    ap.add_argument("--base_ckpt", required=True)
    ap.add_argument("--backbone", type=str, default="adaface", choices=["adaface", "arcface"],
                    help="Pretrained backbone to use (default: adaface).")

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
    ap.add_argument("--margin", type=float, default=3.0)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--dataset_length", type=int, default=500000)

    # Validation
    ap.add_argument("--val_length", type=int, default=100000)
    ap.add_argument("--val_every", type=int, default=500)
    ap.add_argument("--val_batches", type=int, default=100)
    ap.add_argument("--patience_evals", type=int, default=8)
    ap.add_argument("--min_delta", type=float, default=1e-4
                    )
    ap.add_argument("--stop_metric", choices=["eer", "val_loss"], default="eer",
                    help="Metric used for early stopping and best-checkpoint selection.")

    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--seed", type=int, default=42)

    # Paper vs. alternative: adapter on positive
    ap.add_argument(
        "--adapter_on_positive",
        action="store_true",
        help="If set, apply adapter to positive embeddings as well. Default: False (paper-aligned).",
    )

    # NEW: SGD momentum + optional Nesterov (paper: momentum=0.9, nesterov=True)
    ap.add_argument("--momentum", type=float, default=0.9, help="SGD momentum (paper: 0.9)")
    ap.add_argument("--nesterov", action="store_true", help="Use Nesterov momentum (paper: enabled)")

    args = ap.parse_args()

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

    print(f"[Index:{args.split}] real_ids={len(real_index)}")

    # ------------------------------------
    # Validation Indexing (split-aware)
    # ------------------------------------
    val_split_path = Path(args.split_dir) / ("train_ids.json" if args.val_split == "train" else "val_ids.json")
    val_split_ids = set(json.loads(val_split_path.read_text()))
    print(f"[Val Split] {args.val_split}: loaded {len(val_split_ids)} IDs from {val_split_path}")

    val_real_index = {i: real_index_all[i] for i in val_split_ids if i in real_index_all}

    print(f"[Index:{args.val_split}] real_ids={len(val_real_index)}")

    # ------------------------------------
    # Dataset / DataLoader
    # ------------------------------------
    ds = TripletDataset(
        real_index,
        transform=transform,
        length=args.dataset_length,
        seed=args.seed
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

    val_ds = TripletDataset(
        val_real_index,
        transform=transform,
        length=args.val_length,
        seed=args.seed + 1337
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

    set_requires_grad(backbone, False)
    set_requires_grad(adapter, True)

    backbone.to(device).eval()  # frozen
    adapter.to(device).train()

    print(f"[Params] adapter trainable: {count_trainable_params(adapter):,}")
    print(
        "[Mode] adapter_on_positive="
        f"{args.adapter_on_positive} "
        "(False=paper-aligned: adapter only on anchor/negative; True=adapter on all)"
    )

    # --------------------------------
    # Optim / Loss (SGD + optional Nesterov)
    # ---------------------------------
    loss_fn = TripletLoss(margin=args.margin).to(device)

    opt = torch.optim.SGD(
        adapter.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        nesterov=args.nesterov,
        weight_decay=args.weight_decay,
    )

    print(f"[Opt] SGD lr={args.lr} momentum={args.momentum} nesterov={args.nesterov} weight_decay={args.weight_decay}")
    print("[Paper] typical: --lr 1e-1 --momentum 0.9 --nesterov (and possibly --weight_decay 1e-1)")

    # save run config
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "run_args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # -------------------------
    # Training Loop
    # -------------------------
    it = iter(dl)
    seen = 0
    t0 = time.time()

    best_stop = float("inf")
    bad_evals = 0
    best_step = 0
    last_step = 0


    for step in range(1, args.steps + 1):
        last_step = step
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dl)
            batch = next(it)

        opt.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            # Always adapted (paper + your original mode)
            za = encode_adapter(backbone, adapter, batch["anchor"], device, use_amp=use_amp, non_blocking=non_blocking)
            zn = encode_adapter(backbone, adapter, batch["negative"], device, use_amp=use_amp, non_blocking=non_blocking)

            # Positive depends on mode:
            # - paper-aligned: zp is backbone only (no adapter)
            # - alternative mode: zp is adapted as well
            if args.adapter_on_positive:
                zp = encode_adapter(backbone, adapter, batch["positive"], device, use_amp=use_amp, non_blocking=non_blocking)
            else:
                zp = encode_backbone(backbone, batch["positive"], device, use_amp=use_amp, non_blocking=non_blocking)

            loss_vec = loss_fn(za, zp, zn, reduction="none")
            loss = loss_vec.mean()

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), args.grad_clip)
            opt.step()

        seen += args.batch_size

        if step % args.log_every == 0:
            (s_ap, s_an), (d_ap, d_an) = quick_reports_adapter(
                backbone,
                adapter,
                batch,
                device,
                use_amp=use_amp,
                non_blocking=non_blocking,
                adapter_on_positive=args.adapter_on_positive,
            )
            dt = time.time() - t0
            print(
                f"step={step:6d}/{args.steps} loss={loss.item():.4f} | "
                f"cos: ap={s_ap:.3f} an={s_an:.3f} | "
                f"L2: ap={d_ap:.3f} an={d_an:.3f} | "
                f"img/s={seen/max(dt, 1e-6):.1f}"
            )

        if step % args.val_every == 0:
            val_metrics = compute_val_metrics(
                backbone=backbone,
                adapter=adapter,
                loss_fn=loss_fn,
                val_dl=val_dl,
                device=device,
                use_amp=use_amp,
                adapter_on_positive=args.adapter_on_positive,
                max_batches=args.val_batches,
            )
            vloss = val_metrics["val_loss"]
            eer   = val_metrics["eer"]
            current_stop = eer if args.stop_metric == "eer" else vloss
            print(f"[VAL] step={step} val_loss={vloss:.4f} eer={eer:.4f} "
                  f"best_{args.stop_metric}={best_stop:.4f} bad_evals={bad_evals}")

            if current_stop < (best_stop - args.min_delta):
                best_stop = current_stop
                best_step = step
                bad_evals = 0

                torch.save(adapter.state_dict(), out_dir / "adapter_best_val.pt")
                torch.save(
                    {
                        "step": step,
                        "stop_metric": args.stop_metric,
                        "best_stop": best_stop,
                        "val_loss": vloss,
                        "eer": eer,
                        "adapter": adapter.state_dict(),
                        "optimizer": opt.state_dict(),
                        "scaler": None if scaler is None else scaler.state_dict(),
                        "args": vars(args),
                    },
                    out_dir / "ckpt_best_val.pt",
                )
                print(f"[Best] step={step} {args.stop_metric}={current_stop:.4f} "
                      f"val_loss={vloss:.4f} eer={eer:.4f}")
            else:
                bad_evals += 1
                if bad_evals >= args.patience_evals:
                    print(f"[EarlyStop] step={step} best_step={best_step} "
                          f"{args.stop_metric}={best_stop:.4f}")
                    break

        if step % args.save_every == 0:
            save_ckpt(args.out_dir, step, adapter, opt, scaler, args)

    # final
    torch.save(adapter.state_dict(), out_dir / "adapter_final.pt")
    torch.save(
        {
            "step": last_step,
            "adapter": adapter.state_dict(),
            "optimizer": opt.state_dict(),
            "scaler": None if scaler is None else scaler.state_dict(),
            "args": vars(args),
        },
        out_dir / "ckpt_final.pt",
    )

    print("[Done] Training complete")


if __name__ == "__main__":
    main()
