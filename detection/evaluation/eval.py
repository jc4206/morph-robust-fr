# eval.py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: F401 — imported here so backend is set before any downstream import

import argparse
import json
import random
import re
from pathlib import Path
import numpy as np
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from detection.models.adaface.loader import load_adaface_ir50, load_arcface_ir50
from detection.data.index_parser import build_real_index, build_morph_index
from detection.data.preprocessing import transform
from detection.models.mlp_adapter import MLPAdapter
from detection.data.dataset import load_morph_meta_map


# -----------------------
# Dataset for embedding extraction
# -----------------------
class PathDataset(Dataset):
    def __init__(self, items, tfm):
        """items: list of (path: Path, meta: dict)"""
        self.items = items
        self.tfm = tfm

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        p, meta = self.items[idx]
        img = Image.open(p).convert("RGB")
        img = self.tfm(img) if self.tfm else img
        return img, meta

def load_adapter(adapter_ckpt_path, device):
    """
    Supports:
      - pure adapter state_dict  (adapter_final.pt / adapter_step*.pt)
      - full ckpt dict with key 'adapter'  (ckpt_step*.pt / ckpt_final.pt)
    """
    adapter = MLPAdapter(embedding_size=512).to(device)
    sd = torch.load(adapter_ckpt_path, map_location=device)
    if isinstance(sd, dict) and "adapter" in sd:
        adapter.load_state_dict(sd["adapter"])
    else:
        adapter.load_state_dict(sd)
    adapter.eval()
    return adapter


def patch_backbone_output_layer(backbone, adapter_ckpt_path, backbone_type, device):
    """
    Loads any backbone weights stored in the checkpoint back into the backbone before
    inference.  Handles checkpoints from --unfreeze_output_layer and --unfreeze_last_block.

    Keys checked:
      'backbone_layer4'       — last residual block group weights
                                AdaFace: backbone.body[-3:]  (keys "21.*","22.*","23.*")
                                ArcFace: backbone.layer4
      'backbone_output_layer' — output projection weights
                                AdaFace: backbone.output_layer
                                ArcFace: backbone.fc

    Returns True if any backbone weights were patched, False for standard frozen checkpoints.
    """
    sd = torch.load(adapter_ckpt_path, map_location=device)
    if not isinstance(sd, dict):
        return False

    patched = False

    if "backbone_layer4" in sd:
        if backbone_type == "adaface":
            backbone.body[-3:].load_state_dict(sd["backbone_layer4"])
        else:
            backbone.layer4.load_state_dict(sd["backbone_layer4"])
        print(f"[Backbone] last block (layer4) patched from checkpoint ({backbone_type})")
        patched = True

    if "backbone_output_layer" in sd:
        module = backbone.output_layer if backbone_type == "adaface" else backbone.fc
        module.load_state_dict(sd["backbone_output_layer"])
        print(f"[Backbone] output layer patched from checkpoint ({backbone_type})")
        patched = True

    if patched:
        backbone.eval()
    return patched


# -----------------------
# Encoding
# Mirrors train_adapter.py:
#   encode_adapter  -> backbone -> normalize -> adapter  (MLPAdapter normalizes internally)
#   encode_backbone -> backbone -> normalize
# -----------------------
@torch.no_grad()
def encode_backbone(backbone, imgs, device, use_amp):
    imgs = imgs.to(device, non_blocking=True)
    with torch.cuda.amp.autocast(enabled=use_amp):
        emb, _ = backbone(imgs)
        emb = F.normalize(emb, p=2, dim=1)
    return emb


@torch.no_grad()
def encode_adapter(backbone, adapter, imgs, device, use_amp):
    emb = encode_backbone(backbone, imgs, device, use_amp)
    with torch.cuda.amp.autocast(enabled=use_amp):
        emb = adapter(emb)   # MLPAdapter.forward already L2-normalizes output
    return emb


# -----------------------
# Bulk embedding helper
# -----------------------
def embed_image_set(items, backbone, device, use_amp, adapter, batch_size, num_workers, pin_memory):
    """
    Embeds a list of (path, meta) items. meta must contain key 'id' (int).

    If adapter is given:  encode_adapter path  (reference / document space)
    If adapter is None:   encode_backbone path (probe / live-capture space)

    Returns:
        embs_by_id : dict[int, list[tuple[str, Tensor(D)]]]  — per-identity (path, embedding) pairs
        all_embs   : Tensor(N, D)                             — flat embedding matrix
        all_ids    : list[int]                                — identity label for each row
    """
    ds = PathDataset(items, transform)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    embs_by_id = defaultdict(list)
    all_embs = []
    all_ids = []

    with torch.no_grad():
        for imgs, metas in dl:
            if adapter is not None:
                emb = encode_adapter(backbone, adapter, imgs, device, use_amp)
            else:
                emb = encode_backbone(backbone, imgs, device, use_amp)
            emb_cpu = emb.float().cpu()

            ids = metas["id"]
            for i in range(emb_cpu.shape[0]):
                id_i = int(ids[i])
                v = emb_cpu[i]
                embs_by_id[id_i].append((metas["path"][i], v))
                all_embs.append(v)
                all_ids.append(id_i)

    all_embs = torch.stack(all_embs, dim=0)
    return embs_by_id, all_embs, all_ids


# -----------------------
# Metrics
# -----------------------
def cosine_sim(a, b):
    """Element-wise cosine similarity, assumes L2-normalized inputs."""
    return (a * b).sum(dim=-1)


def agg_sim(morph_emb, probe_embs, agg="max"):
    """
    Aggregate similarity of one morph embedding against K probe embeddings.
    morph_emb  : (D,)
    probe_embs : (K, D)
    """
    sims = probe_embs @ morph_emb   # (K,)
    if agg == "max":
        return sims.max().item()
    if agg == "min":
        return sims.min().item()
    raise ValueError(f"Unknown agg={agg}")


def compute_eer(genuine_scores: torch.Tensor, impostor_scores: torch.Tensor, n_thresh: int = 4001):
    """
    EER in similarity score space.
      FNMR(t) = P(genuine  < t)     — genuine pairs falsely rejected
      FMR(t)  = P(impostor >= t)    — impostors falsely accepted
      EER     = operating point minimising |FNMR - FMR|

    Works for both biometric EER (genuine vs impostor)
    and detection D-EER (genuine vs morph scores).

    Returns: eer, tau_eer, fmr_at_tau_eer, fnmr_at_tau_eer
    """
    if genuine_scores.numel() == 0 or impostor_scores.numel() == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")

    g = torch.sort(genuine_scores.float()).values
    i = torch.sort(impostor_scores.float()).values

    th = torch.quantile(torch.cat([g, i]), torch.linspace(0.0, 1.0, n_thresh)).float()

    fnmr = torch.searchsorted(g, th, right=False).float() / g.numel()
    fmr  = 1.0 - torch.searchsorted(i, th, right=False).float() / i.numel()

    idx = torch.argmin(torch.abs(fnmr - fmr))
    eer = 0.5 * (fnmr[idx] + fmr[idx])

    return float(eer.item()), float(th[idx].item()), float(fmr[idx].item()), float(fnmr[idx].item())


# -----------------------
# FRLL/AMSL dataset loaders
# -----------------------
def load_frll_bonafide(bonafide_dir):
    """
    Parses frll_all_4ArcFace/.
    Filename: {id:03d}_{seq:02d}.png   seq=03 source, seq=08 probe.
    Returns dict[int, {"source": str, "probe": str}] for identities with both images.
    """
    result = {}
    pattern = re.compile(r'^(\d{3})_(\d{2})\.png$', re.IGNORECASE)
    for f in sorted(Path(bonafide_dir).iterdir()):
        m = pattern.match(f.name)
        if m is None:
            continue
        id_int = int(m.group(1))
        seq    = int(m.group(2))
        if id_int not in result:
            result[id_int] = {}
        if seq == 3:
            result[id_int]["source"] = str(f)
        elif seq == 8:
            result[id_int]["probe"] = str(f)
    complete = {k: v for k, v in result.items() if "source" in v and "probe" in v}
    dropped = len(result) - len(complete)
    if dropped:
        print(f"[FRLL] dropped {dropped} identities missing source or probe image")
    return complete


def load_frll_morphs(morph_dir, valid_ids):
    """
    Parses morph_amsl_4ArcFace/.
    Filename: {id_a:03d}_{id_b:03d}.png
    Returns list of (path_str, id_a, id_b) filtered to valid_ids.
    """
    result = []
    pattern = re.compile(r'^(\d{3})_(\d{3})\.png$', re.IGNORECASE)
    for f in sorted(Path(morph_dir).iterdir()):
        m = pattern.match(f.name)
        if m is None:
            continue
        id_a = int(m.group(1))
        id_b = int(m.group(2))
        if id_a in valid_ids and id_b in valid_ids:
            result.append((str(f), id_a, id_b))
    return result


# -----------------------
# MMPMR scoring helper
# -----------------------
def _run_mmpmr_scoring(
    morph_embs, morph_pairs, morph_path_strs,
    probe_embs_by_id, morph_meta, morph_agg, morph_rule, tau,
    rng=None, do_sanity_check=True,
):
    """
    One MMPMR scoring pass over all morphs.
    rng: np.random.Generator, required when morph_agg='random'.
    Returns (morph_score_list: list[float], mmpmr_accept: int).
    """
    morph_score_list = []
    mmpmr_accept = 0

    for m_idx in range(morph_embs.shape[0]):
        a, b = morph_pairs[m_idx]
        if a not in probe_embs_by_id or b not in probe_embs_by_id:
            continue

        m = morph_embs[m_idx]

        morph_path_str = morph_path_strs[m_idx]
        meta = morph_meta.get(morph_path_str, {})
        if meta:
            if meta.get("idA") == a:
                src_a, src_b = meta.get("imgA"), meta.get("imgB")
            else:
                src_a, src_b = meta.get("imgB"), meta.get("imgA")
        else:
            src_a = src_b = None
            if morph_meta:
                print(f"[WARNING] no metadata for morph {morph_path_str}; disjointness not enforced for this trial")

        probesA_list = [v for path, v in probe_embs_by_id[a] if path != src_a]
        probesB_list = [v for path, v in probe_embs_by_id[b] if path != src_b]

        if m_idx == 0 and do_sanity_check and morph_meta and src_a is not None:
            n_excl_a = len(probe_embs_by_id[a]) - len(probesA_list)
            n_excl_b = len(probe_embs_by_id[b]) - len(probesB_list)
            assert n_excl_a == 1, (
                f"Disjointness sanity check FAILED: expected to exclude 1 source image from "
                f"identity {a}, but excluded {n_excl_a}. Check path format consistency between "
                f"--pairlist_csv imgA/imgB values and real image paths."
            )
            assert n_excl_b == 1, (
                f"Disjointness sanity check FAILED: expected to exclude 1 source image from "
                f"identity {b}, but excluded {n_excl_b}. Check path format consistency between "
                f"--pairlist_csv imgA/imgB values and real image paths."
            )
            print(f"[Disjointness] sanity check passed: excluded 1 source image per side.")

        if len(probesA_list) == 0:
            probesA_list = [v for _, v in probe_embs_by_id[a]]
        if len(probesB_list) == 0:
            probesB_list = [v for _, v in probe_embs_by_id[b]]

        probesA = torch.stack(probesA_list, dim=0)
        probesB = torch.stack(probesB_list, dim=0)

        sims_A = probesA @ m  # (K,)
        sims_B = probesB @ m  # (K,)

        if morph_agg == "max":
            score_A = sims_A.max().item()
            score_B = sims_B.max().item()
        else:  # "random"
            score_A = sims_A[int(rng.integers(0, len(sims_A)))].item()
            score_B = sims_B[int(rng.integers(0, len(sims_B)))].item()

        score = min(score_A, score_B)
        morph_score_list.append(score)

        if morph_rule == "both":
            if score_A >= tau and score_B >= tau:
                mmpmr_accept += 1
        else:
            if score >= tau:
                mmpmr_accept += 1

    return morph_score_list, mmpmr_accept


# -----------------------
# Worst-case morph metrics
# -----------------------
def compute_wc_metrics(
    ref_embs_by_id,
    probe_embs_by_id,
    genuine_scores,
    tau,
    morph_pairs_ids=None,
    morph_meta=None,
):
    """
    Compute worst-case morph metrics (wc-MMPMR, wc-D-EER).

    For each unique contributor pair (A, B):
        z_a  = adapter(backbone(src_A))  — from ref_embs_by_id (adapter/reference space)
        z_b  = adapter(backbone(src_B))  — idem
        z_wc = normalize(z_a + z_b)      — geometric midpoint on the unit hypersphere
        score_wc = min(max-sim of z_wc vs probes of A,
                       max-sim of z_wc vs probes of B)   (backbone/probe space)

    Exactly one of morph_pairs_ids or morph_meta must be provided:

    frll_amsl  →  morph_pairs_ids=morph_pairs  (list of (id_a, id_b))
        Each identity has one source image (_03, in ref_embs_by_id, adapter space) and
        one probe image (_08, in probe_embs_by_id, backbone space). Disjointness is
        structural — _03 and _08 are different files by dataset design; no path exclusion
        needed. This mirrors the enforcement already used in MMPMR/genuine scoring.

    synface    →  morph_meta=morph_meta  (dict[morph_path, {idA,idB,imgA,imgB}])
        Identities have multiple images; source image identified via pairlist CSV.
        Probe embeddings filtered to exclude the specific source image path.

    Args:
        ref_embs_by_id   : dict[int, list[(path_str, Tensor)]] — adapter/reference space
        probe_embs_by_id : dict[int, list[(path_str, Tensor)]] — backbone/probe space
        genuine_scores   : Tensor — genuine cosine scores (for wc-D-EER FNMR curve)
        tau              : float  — FMR-calibrated threshold (for wc-MMPMR)
        morph_pairs_ids  : list[(int, int)] — frll_amsl pairs (id_a, id_b)
        morph_meta       : dict[str, {idA,idB,imgA,imgB}] — synface pairlist map

    Returns:
        dict with keys: wc_pairs, wc_mmpmr, wc_d_eer, tau_wc_d_eer,
                        mar_at_wc_d_eer, fnmr_at_wc_d_eer, wc_score_mean, wc_score_std.
    """
    assert (morph_pairs_ids is None) != (morph_meta is None), (
        "Exactly one of morph_pairs_ids (frll_amsl) or morph_meta (synface) must be provided."
    )

    wc_scores = []
    skipped   = 0

    if morph_pairs_ids is not None:
        # ------------------------------------------------------------------
        # FRLL/AMSL path
        # Pairs and source images are fully determined by the dataset structure:
        #   ref_embs_by_id[id]   → single entry: _03 source image, adapter space
        #   probe_embs_by_id[id] → single entry: _08 probe image,  backbone space
        # No path exclusion needed — _03 and _08 are structurally disjoint.
        # ------------------------------------------------------------------
        unique_pairs = list({(a, b) for a, b in morph_pairs_ids})
        print(f"[WC] frll_amsl — unique contributor pairs: {len(unique_pairs)}")

        for id_a, id_b in unique_pairs:
            if id_a not in ref_embs_by_id or id_b not in ref_embs_by_id:
                skipped += 1
                continue
            if id_a not in probe_embs_by_id or id_b not in probe_embs_by_id:
                skipped += 1
                continue

            z_a  = ref_embs_by_id[id_a][0][1]   # (D,) — _03 source, adapter space
            z_b  = ref_embs_by_id[id_b][0][1]   # (D,) — _03 source, adapter space
            z_wc = F.normalize((z_a + z_b).unsqueeze(0), p=2, dim=-1).squeeze(0)

            # All probes for each identity are _08 images — structurally disjoint from _03
            probesA = torch.stack([v for _, v in probe_embs_by_id[id_a]], dim=0)
            probesB = torch.stack([v for _, v in probe_embs_by_id[id_b]], dim=0)

            sim_a = (probesA @ z_wc).max().item()
            sim_b = (probesB @ z_wc).max().item()
            wc_scores.append(min(sim_a, sim_b))

        if skipped:
            print(f"[WC] skipped {skipped}/{len(unique_pairs)} pairs — ID not in embedding index.")

    else:
        # ------------------------------------------------------------------
        # Synface path
        # Source image identified via pairlist CSV (imgA/imgB paths).
        # Probe embeddings exclude the source image path (same logic as _run_mmpmr_scoring).
        # ------------------------------------------------------------------
        path_to_adapter_emb = {
            path_str: v
            for vecs in ref_embs_by_id.values()
            for path_str, v in vecs
        }

        unique_pairs_full = list({
            (m["idA"], m["idB"], m["imgA"], m["imgB"])
            for m in morph_meta.values()
        })
        print(f"[WC] synface — unique contributor pairs: {len(unique_pairs_full)}")

        for idA, idB, imgA, imgB in unique_pairs_full:
            if imgA not in path_to_adapter_emb or imgB not in path_to_adapter_emb:
                skipped += 1
                continue
            if idA not in probe_embs_by_id or idB not in probe_embs_by_id:
                skipped += 1
                continue

            z_a  = path_to_adapter_emb[imgA]
            z_b  = path_to_adapter_emb[imgB]
            z_wc = F.normalize((z_a + z_b).unsqueeze(0), p=2, dim=-1).squeeze(0)

            # Exclude source image from probe pool (path-based disjointness)
            probesA_list = [v for path, v in probe_embs_by_id[idA] if path != imgA]
            probesB_list = [v for path, v in probe_embs_by_id[idB] if path != imgB]

            # Fallback (shouldn't occur with real data)
            if len(probesA_list) == 0:
                probesA_list = [v for _, v in probe_embs_by_id[idA]]
            if len(probesB_list) == 0:
                probesB_list = [v for _, v in probe_embs_by_id[idB]]

            probesA = torch.stack(probesA_list, dim=0)
            probesB = torch.stack(probesB_list, dim=0)

            sim_a = (probesA @ z_wc).max().item()
            sim_b = (probesB @ z_wc).max().item()
            wc_scores.append(min(sim_a, sim_b))

        if skipped:
            print(f"[WC] skipped {skipped}/{len(unique_pairs_full)} pairs — source image path not "
                  f"found in embedding index. Check path format consistency between "
                  f"--pairlist_csv imgA/imgB values and the paths stored by build_real_index.")

    wc_scores_t = (torch.tensor(wc_scores, dtype=torch.float32)
                   if wc_scores else torch.empty(0, dtype=torch.float32))
    n = wc_scores_t.numel()

    wc_mmpmr = float((wc_scores_t > tau).float().mean()) if n > 0 else float("nan")
    wc_d_eer, tau_wc, mar_wc, fnmr_wc = compute_eer(genuine_scores, wc_scores_t)

    print(f"[WC-MMPMR] pairs={n} | wc_mmpmr={wc_mmpmr:.6f}")
    print(f"[WC-D-EER] {wc_d_eer:.6f}  "
          f"(tau={tau_wc:.6f}, MAR={mar_wc:.6f}, FNMR={fnmr_wc:.6f})")

    return {
        "wc_pairs":         n,
        "wc_mmpmr":         wc_mmpmr,
        "wc_d_eer":         float(wc_d_eer),
        "tau_wc_d_eer":     float(tau_wc),
        "mar_at_wc_d_eer":  float(mar_wc),
        "fnmr_at_wc_d_eer": float(fnmr_wc),
        "wc_score_mean":    float(wc_scores_t.mean()) if n > 0 else float("nan"),
        "wc_score_std":     float(wc_scores_t.std())  if n > 0 else float("nan"),
    }


# -----------------------
# FRLL/AMSL scoring pipeline (exposed for external callers, e.g. plot_score_histogram.py)
# -----------------------
def run_frll_eval(
    backbone_type,
    base_ckpt,
    adapter_ckpt,
    bonafide_dir,
    morph_dir,
    device=None,
    use_amp=False,
    batch_size=256,
    num_workers=4,
    pin_memory=False,
    fmr=0.001,
    morph_agg="max",
    morph_rule="min_of_two",
    seed=42,
):
    """
    Full FRLL/AMSL scoring pipeline for a single model configuration.

    Runs exactly the same computation as ``eval.py --dataset frll_amsl`` and exposes
    the three score arrays and tau so they can be used by histogram plotting or other
    downstream analysis without re-implementing the scoring logic.

    Returns a dict with the following keys:

    Score arrays (torch.Tensor, float32):
        impostor_scores  — all exhaustive cross-identity pair scores (N×(N-1) entries)
        genuine_scores   — one score per mated source→probe pair
        morph_scores     — per-morph min(max-sim-A, max-sim-B) scores

    Scalars:
        tau              — threshold at FMR=fmr (the 99.9th-percentile of impostor dist.)
        mmpmr            — morphing attack potential at tau
        d_eer            — detection equal-error rate (genuine vs. morph)
        tau_d_eer        — threshold at D-EER
        mar_at_d_eer     — MAR at D-EER
        fnmr_at_d_eer    — FNMR at D-EER
        fnmr_at_tau      — FNMR of genuine scores at the FMR-calibrated tau
        eer              — biometric EER (genuine vs. impostor)
        tau_eer          — threshold at EER
        n_ids            — number of FRLL identities used
        impostor_mean / impostor_std
        genuine_mean  / genuine_std
        morph_score_mean / morph_score_std

    Internals (needed by wc-metrics if called by main or externally):
        ref_embs_by_id, probe_embs_by_id, all_ref_embs, all_ref_ids,
        all_probe_embs, all_probe_ids, morph_pairs, morph_path_strs,
        morph_embs, backbone, adapter, asymmetric
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if device == "cuda" and not pin_memory:
        pin_memory = True

    # --- Load FRLL data ---
    bonafide = load_frll_bonafide(bonafide_dir)
    morph_paths_list = load_frll_morphs(morph_dir, set(bonafide.keys()))
    n_ids = len(bonafide)
    print(f"[FRLL] bonafide identities={n_ids} | morphs={len(morph_paths_list)}")
    for id_k, v in bonafide.items():
        assert Path(v["probe"]).stem.endswith("_08"), (
            f"FRLL disjointness: probe for identity {id_k} is not sequence 08: {v['probe']}"
        )

    # --- Load model(s) ---
    if backbone_type == "arcface":
        backbone = load_arcface_ir50(base_ckpt, device=device, strict=False)
    else:
        backbone = load_adaface_ir50(base_ckpt, device=device, strict=False)
    print(f"[Model] backbone={backbone_type}")
    backbone.to(device).eval()

    adapter = None
    if adapter_ckpt:
        adapter = load_adapter(adapter_ckpt, device=device)
        patched = patch_backbone_output_layer(backbone, adapter_ckpt, backbone_type, device)
        if not patched:
            print("[Backbone] no output layer weights in checkpoint — using pretrained backbone as-is")

    asymmetric = adapter is not None
    print(f"[Model] adapter={'yes' if adapter_ckpt else 'no'} | "
          f"encoding={'asymmetric' if asymmetric else 'symmetric'} | "
          f"amp={use_amp} | device={device}")

    # --- Embed source images (_03) -> reference space (backbone [+adapter]) ---
    frll_source_items = [(Path(v["source"]), {"id": k, "path": v["source"]}) for k, v in bonafide.items()]
    frll_probe_items  = [(Path(v["probe"]),  {"id": k, "path": v["probe"]})  for k, v in bonafide.items()]

    print("[Embed] FRLL source images (_03) -> reference space ...")
    ref_embs_by_id, all_ref_embs, all_ref_ids = embed_image_set(
        frll_source_items, backbone, device, use_amp,
        adapter=adapter,
        batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,
    )
    print(f"  N={all_ref_embs.shape[0]}, D={all_ref_embs.shape[1]}")

    # --- Embed probe images (_08) -> probe space (backbone only) ---
    print("[Embed] FRLL probe images (_08) -> probe space (backbone only) ...")
    probe_embs_by_id, all_probe_embs, all_probe_ids = embed_image_set(
        frll_probe_items, backbone, device, use_amp,
        adapter=None,
        batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,
    )
    print(f"  N={all_probe_embs.shape[0]}, D={all_probe_embs.shape[1]}")

    # --- Embed morphs -> reference space (backbone [+adapter]) ---
    morph_items = [(p, {"a": int(a), "b": int(b), "path": str(p)}) for p, a, b in morph_paths_list]
    morph_ds = PathDataset(morph_items, transform)
    morph_dl = DataLoader(morph_ds, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=pin_memory, drop_last=False)

    morph_embs_list  = []
    morph_pairs      = []
    morph_path_strs  = []
    print("[Embed] morphs -> reference space ...")
    with torch.no_grad():
        for imgs, metas in morph_dl:
            if adapter is not None:
                emb = encode_adapter(backbone, adapter, imgs, device, use_amp)
            else:
                emb = encode_backbone(backbone, imgs, device, use_amp)
            emb_cpu = emb.float().cpu()
            for i in range(emb_cpu.shape[0]):
                morph_embs_list.append(emb_cpu[i])
                morph_pairs.append((int(metas["a"][i]), int(metas["b"][i])))
                morph_path_strs.append(str(metas["path"][i]))

    morph_embs = (torch.stack(morph_embs_list, dim=0) if morph_embs_list
                  else torch.empty((0, all_ref_embs.shape[1]), dtype=torch.float32))
    print(f"  M={morph_embs.shape[0]}")

    # --- Impostor scores: exhaustive cross-identity scoring ---
    print(f"[Eval] FRLL/AMSL: computing all impostor scores exhaustively "
          f"({all_ref_embs.shape[0]} × {all_probe_embs.shape[0] - 1} = "
          f"{all_ref_embs.shape[0] * (all_probe_embs.shape[0] - 1)} pairs) ...")

    assert all_ref_ids == all_probe_ids, (
        "FRLL/AMSL impostor scoring: ref and probe identity lists are not aligned. "
        "Both must be built from the same identity iteration order."
    )

    R = all_ref_embs
    P = all_probe_embs
    S = R @ P.T
    n_frll = R.shape[0]
    off_diag_mask = ~torch.eye(n_frll, dtype=torch.bool, device=S.device)
    impostor_scores = S[off_diag_mask].float().cpu()
    print(f"  impostor_scores: {impostor_scores.numel()} pairs | "
          f"mean={impostor_scores.mean():.6f} | std={impostor_scores.std():.6f}")

    tau = float(torch.quantile(impostor_scores, 1.0 - fmr))
    print(f"  tau @ FMR={fmr} => {tau:.6f}")

    # --- Genuine scores: one source→probe pair per identity ---
    print("[Eval] computing genuine scores ...")
    genuine_chunks = []
    for id_val, ref_vecs in ref_embs_by_id.items():
        if id_val not in probe_embs_by_id:
            continue
        ref_v   = ref_vecs[0][1]
        probe_v = probe_embs_by_id[id_val][0][1]
        genuine_chunks.append(cosine_sim(ref_v, probe_v).unsqueeze(0))

    genuine_scores = (torch.cat(genuine_chunks) if genuine_chunks
                      else torch.empty((0,), dtype=torch.float32))

    fnmr_at_tau = float((genuine_scores < tau).float().mean()) if genuine_scores.numel() > 0 else float("nan")
    print(f"  genuine pairs={genuine_scores.numel()} | FNMR@tau={fnmr_at_tau:.6f}")

    eer, tau_eer, fmr_at_eer, fnmr_at_eer = compute_eer(genuine_scores, impostor_scores)
    print(f"[EER]  {eer:.6f}  (tau={tau_eer:.6f}, FMR={fmr_at_eer:.6f}, FNMR={fnmr_at_eer:.6f})")

    # --- Morph scores: min(max-sim-A, max-sim-B) per morph, max aggregation ---
    # morph_meta={} for FRLL/AMSL — structural disjointness (_03 source vs _08 probe)
    # means the source image is never in the probe set; no pairlist exclusion needed.
    print("[Eval] computing MMPMR ...")
    score_list, accept = _run_mmpmr_scoring(
        morph_embs, morph_pairs, morph_path_strs,
        probe_embs_by_id, {},
        morph_agg, morph_rule, tau,
        rng=None, do_sanity_check=True,
    )
    morph_scores = torch.tensor(score_list, dtype=torch.float32)
    mmpmr = float(accept / max(1, morph_scores.numel()))

    d_eer, tau_d_eer, mar_at_d_eer, fnmr_at_d_eer = compute_eer(genuine_scores, morph_scores)
    print(f"[MMPMR] trials={morph_scores.numel()} | agg={morph_agg} rule={morph_rule} | MMPMR={mmpmr:.6f}")
    print(f"[D-EER] {d_eer:.6f}  (tau={tau_d_eer:.6f}, MAR={mar_at_d_eer:.6f}, FNMR={fnmr_at_d_eer:.6f})")

    return {
        # The three score arrays + threshold used directly by the histogram script
        "impostor_scores":   impostor_scores,
        "genuine_scores":    genuine_scores,
        "morph_scores":      morph_scores,
        "tau":               tau,
        # Detection and verification metrics
        "mmpmr":             mmpmr,
        "d_eer":             float(d_eer),
        "tau_d_eer":         float(tau_d_eer),
        "mar_at_d_eer":      float(mar_at_d_eer),
        "fnmr_at_d_eer":     float(fnmr_at_d_eer),
        "fnmr_at_tau":       fnmr_at_tau,
        "eer":               float(eer),
        "tau_eer":           float(tau_eer),
        "fmr_at_eer":        float(fmr_at_eer),
        "fnmr_at_eer":       float(fnmr_at_eer),
        "n_ids":             n_ids,
        # Distribution summary stats
        "impostor_mean":     float(impostor_scores.mean()),
        "impostor_std":      float(impostor_scores.std()),
        "genuine_mean":      float(genuine_scores.mean()) if genuine_scores.numel() > 0 else float("nan"),
        "genuine_std":       float(genuine_scores.std())  if genuine_scores.numel() > 0 else float("nan"),
        "morph_score_mean":  float(morph_scores.mean()) if morph_scores.numel() > 0 else float("nan"),
        "morph_score_std":   float(morph_scores.std())  if morph_scores.numel() > 0 else float("nan"),
        # Internals needed by compute_wc_metrics if called after this function
        "ref_embs_by_id":    ref_embs_by_id,
        "probe_embs_by_id":  probe_embs_by_id,
        "all_ref_embs":      all_ref_embs,
        "all_ref_ids":       all_ref_ids,
        "all_probe_embs":    all_probe_embs,
        "all_probe_ids":     all_probe_ids,
        "morph_pairs":       morph_pairs,
        "morph_path_strs":   morph_path_strs,
        "morph_embs":        morph_embs,
        "backbone":          backbone,
        "adapter":           adapter,
        "asymmetric":        asymmetric,
    }


# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--dataset", type=str, default="synface", choices=["synface", "frll_amsl"],
                    help="evaluation dataset: 'synface' (default) or 'frll_amsl'")

    ap.add_argument("--real_root",  type=str, default="",
                    help="synface: root dir of real identity images (required for --dataset synface)")
    ap.add_argument("--morph_root", type=str, default="",
                    help="synface: root dir of morph images (required for --dataset synface)")

    # model weights
    ap.add_argument("--base_ckpt", type=str, required=True,
                    help="pretrained backbone checkpoint (.ckpt / .pth)")
    ap.add_argument("--backbone", type=str, default="adaface", choices=["adaface", "arcface"],
                    help="Pretrained backbone to use (default: adaface).")
    ap.add_argument("--adapter_ckpt", type=str, default="",
                    help="optional: adapter checkpoint "
                         "(adapter_step*.pt / adapter_final.pt / ckpt_step*.pt)")

    # synface split
    ap.add_argument("--split_dir", type=str, default="",
                    help="synface: dir containing train/val/test_ids.json (required for --dataset synface)")
    ap.add_argument("--split", type=str, choices=["train", "val", "test"], default="val")

    # frll_amsl paths
    ap.add_argument("--bonafide_dir", type=str, default="",
                    help="frll_amsl: dir containing frll_all_4ArcFace images (required for --dataset frll_amsl)")
    ap.add_argument("--morph_dir",    type=str, default="",
                    help="frll_amsl: dir containing morph_amsl_4ArcFace images (required for --dataset frll_amsl)")

    ap.add_argument("--pairlist_csv", type=str, default="",
                    help="Morph pairlist CSV (idA, idB, imgA, imgB, ...) for "
                         "source-image disjointness enforcement in MMPMR scoring")
    ap.add_argument("--compute_wc_metrics", action="store_true",
                    help="Compute worst-case morph metrics (wc-MMPMR, wc-D-EER). "
                         "Derives a theoretical morph embedding z_wc=normalize(z_src_A+z_src_B) "
                         "for each unique contributor pair in --pairlist_csv, without loading "
                         "any morph image. Requires --pairlist_csv. Only for --dataset synface.")

    # dataloader
    ap.add_argument("--batch_size",   type=int,  default=256)
    ap.add_argument("--num_workers",  type=int,  default=8)
    ap.add_argument("--pin_memory",   action="store_true")
    ap.add_argument("--amp",          action="store_true")
    ap.add_argument("--seed",         type=int,  default=42)

    # impostor sampling -> threshold at target FMR
    ap.add_argument("--impostor_pairs", type=int,   default=500000,
                    help="random cross-identity pairs for FMR/EER threshold estimation")
    ap.add_argument("--fmr",            type=float, default=0.001,
                    help="target false match rate for FNMR and MMPMR operating point (e.g. 0.001 = 0.1%%)")

    # morph scoring rule
    ap.add_argument("--morph_agg",  type=str, choices=["max", "random"], default="max",
                    help="per-side morph score aggregation: "
                         "'max' = worst-case (max over all non-source probes, default, backward-compatible); "
                         "'random' = single-shot deployment (one random non-source probe per contributor)")
    ap.add_argument("--morph_rule", type=str, choices=["both", "min_of_two"], default="min_of_two",
                    help="both: accept if simA>=tau AND simB>=tau; "
                         "min_of_two: score=min(simA,simB) compared to tau")
    ap.add_argument("--morph_random_seed", type=int, default=42,
                    help="random seed for probe selection in --morph_agg=random mode (ignored for max)")
    ap.add_argument("--morph_random_n_runs", type=int, default=1,
                    help="number of random-selection runs to average over (only for --morph_agg=random); "
                         "use n_runs>1 for stable estimates; output uses _mean/_std suffixes when >1")

    ap.add_argument("--out_json", type=str, default="eval_out.json")

    ap.add_argument(
        '--export_roc_det',
        action='store_true',
        help='Generate and save ROC and DET figure from genuine/impostor scores'
    )
    ap.add_argument(
        '--roc_det_output_dir',
        type=str,
        default='figures/',
        help='Directory to save the ROC/DET figure'
    )
    ap.add_argument(
        '--roc_det_label',
        type=str,
        default='AdaFace + Adapter',
        help='System label shown in the figure legend'
    )

    args = ap.parse_args()

    if args.dataset == "synface":
        missing = [n for n, v in [("--real_root", args.real_root),
                                   ("--morph_root", args.morph_root),
                                   ("--split_dir",  args.split_dir)] if not v]
        if missing:
            ap.error(f"--dataset synface requires: {', '.join(missing)}")
    else:  # frll_amsl
        missing = [n for n, v in [("--bonafide_dir", args.bonafide_dir),
                                   ("--morph_dir",    args.morph_dir)] if not v]
        if missing:
            ap.error(f"--dataset frll_amsl requires: {', '.join(missing)}")

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device  = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = bool(args.amp and device == "cuda")
    if device == "cuda" and not args.pin_memory:
        args.pin_memory = True

    # -----------------------
    # Dataset-specific loading
    # -----------------------
    if args.dataset == "synface":
        if args.pairlist_csv:
            morph_meta = load_morph_meta_map(args.pairlist_csv)
            print(f"[Meta] loaded morph metadata for {len(morph_meta)} morphs from {args.pairlist_csv}")
        else:
            morph_meta = {}
            print("[WARNING] --pairlist_csv not provided. Source-image disjointness will NOT be enforced in MMPMR scoring.")

        split_dir  = Path(args.split_dir)
        split_file = {"train": "train_ids.json", "val": "val_ids.json", "test": "test_ids.json"}[args.split]
        split_ids  = set(json.loads((split_dir / split_file).read_text()))
        print(f"[Split] {args.split}: {len(split_ids)} IDs from {split_dir / split_file}")

        real_index_all = build_real_index(args.real_root)
        real_index     = {i: real_index_all[i] for i in split_ids if i in real_index_all}
        morphs_by_id   = build_morph_index(args.morph_root, allowed_ids=split_ids)

        morph_paths = []
        seen_morph  = set()
        for _, lst in morphs_by_id.items():
            for p, a, b, *_ in lst:
                if p not in seen_morph:
                    seen_morph.add(p)
                    morph_paths.append((p, a, b))

        n_ids = len(real_index)
        print(f"[Index] real_ids={n_ids} | unique morphs={len(morph_paths)}")

    else:  # frll_amsl
        morph_meta = {}
        bonafide   = load_frll_bonafide(args.bonafide_dir)
        morph_paths = load_frll_morphs(args.morph_dir, set(bonafide.keys()))
        n_ids = len(bonafide)
        print(f"[FRLL] bonafide identities={n_ids} | morphs={len(morph_paths)}")
        for id_k, v in bonafide.items():
            assert Path(v["probe"]).stem.endswith("_08"), (
                f"FRLL disjointness: probe for identity {id_k} is not sequence 08: {v['probe']}"
            )

    # -----------------------
    # Load model(s)
    # -----------------------
    if args.backbone == "arcface":
        backbone = load_arcface_ir50(args.base_ckpt, device=device, strict=False)
    else:
        backbone = load_adaface_ir50(args.base_ckpt, device=device, strict=False)
    print(f"[Model] backbone={args.backbone}")
    backbone.to(device).eval()

    adapter = None
    if args.adapter_ckpt:
        adapter = load_adapter(args.adapter_ckpt, device=device)
        patched = patch_backbone_output_layer(backbone, args.adapter_ckpt, args.backbone, device)
        if not patched:
            print("[Backbone] no output layer weights in checkpoint — using pretrained backbone as-is")

    # Deployment scenario (eGate):
    #   reference (document / may be morphed) -> backbone -> adapter  [reference/adapted space]
    #   probe     (live capture, always bona fide) -> backbone only   [probe/backbone space]
    # For backbone-only eval both spaces are identical — no second pass needed.
    asymmetric = adapter is not None
    print(f"[Model] adapter={'yes' if args.adapter_ckpt else 'no'} | "
          f"encoding={'asymmetric' if asymmetric else 'symmetric'} | "
          f"amp={use_amp} | device={device}")

    # -----------------------
    # 1) Embed REFERENCES  (backbone [+adapter])
    # 2) Embed PROBES      (backbone only)
    # -----------------------
    if args.dataset == "synface":
        ref_items = []
        for id_int, paths in real_index.items():
            for p in paths:
                ref_items.append((p, {"id": id_int, "path": str(p)}))

        print("[Embed] real images -> reference space ...")
        ref_embs_by_id, all_ref_embs, all_ref_ids = embed_image_set(
            ref_items, backbone, device, use_amp,
            adapter=adapter,
            batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=args.pin_memory,
        )
        print(f"  N={all_ref_embs.shape[0]}, D={all_ref_embs.shape[1]}")

        if asymmetric:
            print("[Embed] real images -> probe space (backbone only) ...")
            probe_embs_by_id, all_probe_embs, all_probe_ids = embed_image_set(
                ref_items, backbone, device, use_amp,
                adapter=None,
                batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=args.pin_memory,
            )
            print(f"  N={all_probe_embs.shape[0]}, D={all_probe_embs.shape[1]}")
        else:
            probe_embs_by_id = ref_embs_by_id
            all_probe_embs   = all_ref_embs
            all_probe_ids    = all_ref_ids

    else:  # frll_amsl — always two separate passes (source _03 vs probe _08)
        frll_source_items = [(Path(v["source"]), {"id": k, "path": v["source"]}) for k, v in bonafide.items()]
        frll_probe_items  = [(Path(v["probe"]),  {"id": k, "path": v["probe"]})  for k, v in bonafide.items()]

        print("[Embed] FRLL source images (_03) -> reference space ...")
        ref_embs_by_id, all_ref_embs, all_ref_ids = embed_image_set(
            frll_source_items, backbone, device, use_amp,
            adapter=adapter,
            batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=args.pin_memory,
        )
        print(f"  N={all_ref_embs.shape[0]}, D={all_ref_embs.shape[1]}")

        print("[Embed] FRLL probe images (_08) -> probe space (backbone only) ...")
        probe_embs_by_id, all_probe_embs, all_probe_ids = embed_image_set(
            frll_probe_items, backbone, device, use_amp,
            adapter=None,
            batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=args.pin_memory,
        )
        print(f"  N={all_probe_embs.shape[0]}, D={all_probe_embs.shape[1]}")

    # -----------------------
    # 3) Embed morphs as REFERENCES  (backbone [+adapter])
    # -----------------------
    morph_items = [(p, {"a": int(a), "b": int(b), "path": str(p)}) for p, a, b in morph_paths]
    morph_ds = PathDataset(morph_items, transform)
    morph_dl = DataLoader(morph_ds, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=args.pin_memory, drop_last=False)

    morph_embs      = []
    morph_pairs     = []
    morph_path_strs = []
    print("[Embed] morphs -> reference space ...")
    with torch.no_grad():
        for imgs, metas in morph_dl:
            if adapter is not None:
                emb = encode_adapter(backbone, adapter, imgs, device, use_amp)
            else:
                emb = encode_backbone(backbone, imgs, device, use_amp)
            emb_cpu = emb.float().cpu()
            for i in range(emb_cpu.shape[0]):
                morph_embs.append(emb_cpu[i])
                morph_pairs.append((int(metas["a"][i]), int(metas["b"][i])))
                morph_path_strs.append(str(metas["path"][i]))

    morph_embs = (torch.stack(morph_embs, dim=0) if morph_embs
                  else torch.empty((0, all_ref_embs.shape[1]), dtype=torch.float32))
    print(f"  M={morph_embs.shape[0]}")

    # -----------------------
    # 4) Impostor scores: ref of A  vs  probe of C  (A != C)
    #    Threshold tau at target FMR is derived from this distribution.
    #    FRLL/AMSL: exhaustive (102×101 = 10,302 unique pairs) — fully deterministic.
    #    Synface:   sampled   (too many unique pairs for exhaustive computation).
    # -----------------------
    if args.dataset == "frll_amsl":
        print(f"[Eval] FRLL/AMSL: computing all impostor scores exhaustively "
              f"({all_ref_embs.shape[0]} × {all_probe_embs.shape[0] - 1} = "
              f"{all_ref_embs.shape[0] * (all_probe_embs.shape[0] - 1)} pairs) ...")

        assert all_ref_ids == all_probe_ids, (
            "FRLL/AMSL impostor scoring: ref and probe identity lists are not aligned. "
            "Both must be built from the same identity iteration order."
        )

        R = all_ref_embs    # (N, D) — adapter(source_03) per identity
        P = all_probe_embs  # (N, D) — backbone(probe_08) per identity
        S = R @ P.T         # (N, N) — full cross-similarity matrix
        n_ids_frll = R.shape[0]
        off_diag_mask = ~torch.eye(n_ids_frll, dtype=torch.bool, device=S.device)

        print("[DEBUG] Verifying ref/probe alignment for exhaustive scoring:")
        print(f"  all_ref_ids[:5]   = {all_ref_ids[:5]}")
        print(f"  all_probe_ids[:5] = {all_probe_ids[:5]}")
        print(f"  all_ref_ids[-5:]  = {all_ref_ids[-5:]}")
        print(f"  all_probe_ids[-5:]= {all_probe_ids[-5:]}")
        print(f"  IDs match: {all_ref_ids == all_probe_ids}")
        print(f"  N ref={len(all_ref_ids)}, N probe={len(all_probe_ids)}")
        diag_scores = torch.diagonal(S)
        off_diag_tmp = S[off_diag_mask].float().cpu()
        print(f"  Diagonal (same-identity) scores: mean={diag_scores.mean():.4f}, "
              f"min={diag_scores.min():.4f}, max={diag_scores.max():.4f}")
        print(f"  Off-diagonal (impostor) scores:  mean={off_diag_tmp.mean():.4f}, "
              f"min={off_diag_tmp.min():.4f}, max={off_diag_tmp.max():.4f}")
        print(f"  Expected: diagonal >> off-diagonal (genuine >> impostor)")

        impostor_scores = S[off_diag_mask].float().cpu()

        print(f"  impostor_scores: {impostor_scores.numel()} pairs | "
              f"mean={impostor_scores.mean():.6f} | "
              f"std={impostor_scores.std():.6f} | "
              f"max={impostor_scores.max():.6f}")
    else:
        print(f"[Eval] sampling {args.impostor_pairs} impostor pairs ...")
        ref_id_to_idx   = defaultdict(list)
        probe_id_to_idx = defaultdict(list)
        for idx, idv in enumerate(all_ref_ids):
            ref_id_to_idx[idv].append(idx)
        for idx, idv in enumerate(all_probe_ids):
            probe_id_to_idx[idv].append(idx)
        unique_ids = list(ref_id_to_idx.keys())

        impostor_scores = torch.empty((args.impostor_pairs,), dtype=torch.float32)
        for k in range(args.impostor_pairs):
            id1 = rng.choice(unique_ids)
            id2 = rng.choice(unique_ids)
            while id2 == id1:
                id2 = rng.choice(unique_ids)
            i = rng.choice(ref_id_to_idx[id1])
            j = rng.choice(probe_id_to_idx[id2])
            impostor_scores[k] = cosine_sim(all_ref_embs[i], all_probe_embs[j]).item()

    tau = float(torch.quantile(impostor_scores, 1.0 - args.fmr))
    print(f"  tau @ FMR={args.fmr} => {tau:.6f}")

    # -----------------------
    # 5) Genuine scores
    #    synface:   ref of A[i] vs probe of A[j] (i!=j); diagonal excluded.
    #    frll_amsl: one source + one probe per identity; no diagonal exclusion.
    # -----------------------
    print("[Eval] computing genuine scores ...")
    genuine_score_chunks = []
    if args.dataset == "frll_amsl":
        for id_val, ref_vecs in ref_embs_by_id.items():
            if id_val not in probe_embs_by_id:
                continue
            ref_v   = ref_vecs[0][1]
            probe_v = probe_embs_by_id[id_val][0][1]
            genuine_score_chunks.append(cosine_sim(ref_v, probe_v).unsqueeze(0))
    else:
        for id_val, ref_vecs in ref_embs_by_id.items():
            if id_val not in probe_embs_by_id:
                continue
            probe_vecs = probe_embs_by_id[id_val]
            k = len(ref_vecs)
            if k < 2:
                continue
            R = torch.stack([v for _, v in ref_vecs],   dim=0)   # (K, D)
            P = torch.stack([v for _, v in probe_vecs], dim=0)   # (K, D)
            S = R @ P.t()                                         # (K, K)
            mask = ~torch.eye(k, dtype=torch.bool)
            genuine_score_chunks.append(S[mask].float())

    genuine_scores = (torch.cat(genuine_score_chunks) if genuine_score_chunks
                      else torch.empty((0,), dtype=torch.float32))

    fnmr = float((genuine_scores < tau).float().mean()) if genuine_scores.numel() > 0 else float("nan")
    print(f"  genuine pairs={genuine_scores.numel()} | FNMR@tau={fnmr:.6f}")

    eer, tau_eer, fmr_at_eer, fnmr_at_eer = compute_eer(genuine_scores, impostor_scores)
    print(f"[EER]  {eer:.6f}  (tau={tau_eer:.6f}, FMR={fmr_at_eer:.6f}, FNMR={fnmr_at_eer:.6f})")

    if args.export_roc_det:
        from plot_roc_det import plot_roc_det
        plot_roc_det(
            genuine_scores  = genuine_scores.numpy(),
            impostor_scores = impostor_scores.numpy(),
            output_dir      = args.roc_det_output_dir,
            filename_stem   = 'background_roc_det',
            label           = args.roc_det_label,
            log_wandb       = True,
        )

    # -----------------------
    # 6) MMPMR + 7) D-EER
    #    morph (reference/adapter space)  vs  contributor probes (probe/backbone space)
    #    morph_agg='max'    : worst-case, max over all non-source probes per side (Mode A)
    #    morph_agg='random' : single-shot, one random non-source probe per side  (Mode B)
    # -----------------------
    print("[Eval] computing MMPMR ...")
    if args.dataset == "frll_amsl":
        print("[DEBUG] FRLL probe_embs_by_id sample:")
        for id_val, vecs in list(probe_embs_by_id.items())[:3]:
            print(f"  id={id_val}: {[path for path, _ in vecs]}")
    disjointness_enforced = bool(morph_meta)
    n_runs = args.morph_random_n_runs if args.morph_agg == "random" else 1

    if n_runs > 1:
        all_run_results = []
        for run_idx in range(n_runs):
            rng_run = np.random.default_rng(args.morph_random_seed + run_idx)
            score_list, accept = _run_mmpmr_scoring(
                morph_embs, morph_pairs, morph_path_strs,
                probe_embs_by_id, morph_meta,
                args.morph_agg, args.morph_rule, tau,
                rng=rng_run, do_sanity_check=(run_idx == 0),
            )
            ms_run = torch.tensor(score_list, dtype=torch.float32)
            mmpmr_run = float(accept / max(1, ms_run.numel()))
            d_eer_run, tau_d_eer_run, mar_run, fnmr_d_run = compute_eer(genuine_scores, ms_run)
            all_run_results.append({
                "mmpmr":         mmpmr_run,
                "d_eer":         d_eer_run,
                "tau_d_eer":     tau_d_eer_run,
                "mar_at_d_eer":  mar_run,
                "fnmr_at_d_eer": fnmr_d_run,
                "morph_score_mean": float(ms_run.mean()) if ms_run.numel() > 0 else float("nan"),
            })

        morph_scores = ms_run  # use last run's tensor for morph count

        mmpmr             = float(np.mean([r["mmpmr"]            for r in all_run_results]))
        mmpmr_std         = float(np.std( [r["mmpmr"]            for r in all_run_results]))
        d_eer             = float(np.mean([r["d_eer"]            for r in all_run_results]))
        d_eer_std         = float(np.std( [r["d_eer"]            for r in all_run_results]))
        tau_d_eer         = float(np.mean([r["tau_d_eer"]        for r in all_run_results]))
        tau_d_eer_std     = float(np.std( [r["tau_d_eer"]        for r in all_run_results]))
        mar_at_d_eer      = float(np.mean([r["mar_at_d_eer"]     for r in all_run_results]))
        mar_at_d_eer_std  = float(np.std( [r["mar_at_d_eer"]     for r in all_run_results]))
        fnmr_at_d_eer     = float(np.mean([r["fnmr_at_d_eer"]    for r in all_run_results]))
        fnmr_at_d_eer_std = float(np.std( [r["fnmr_at_d_eer"]    for r in all_run_results]))
        morph_score_mean  = float(np.mean([r["morph_score_mean"] for r in all_run_results]))
        morph_score_mean_std = float(np.std([r["morph_score_mean"] for r in all_run_results]))

        print(f"[MMPMR] trials={morph_scores.numel()} | agg={args.morph_agg} rule={args.morph_rule} "
              f"n_runs={n_runs} | MMPMR={mmpmr:.6f} ±{mmpmr_std:.6f}")
        print(f"[D-EER] mean={d_eer:.6f} ±{d_eer_std:.6f}")

    else:
        rng_run = np.random.default_rng(args.morph_random_seed) if args.morph_agg == "random" else None
        score_list, accept = _run_mmpmr_scoring(
            morph_embs, morph_pairs, morph_path_strs,
            probe_embs_by_id, morph_meta,
            args.morph_agg, args.morph_rule, tau,
            rng=rng_run, do_sanity_check=True,
        )
        morph_scores = torch.tensor(score_list, dtype=torch.float32)
        mmpmr = float(accept / max(1, morph_scores.numel()))

        print(f"[MMPMR] trials={morph_scores.numel()} | agg={args.morph_agg} rule={args.morph_rule} | "
              f"MMPMR={mmpmr:.6f}")

        d_eer, tau_d_eer, mar_at_d_eer, fnmr_at_d_eer = compute_eer(genuine_scores, morph_scores)
        print(f"[D-EER] {d_eer:.6f}  (tau={tau_d_eer:.6f}, MAR={mar_at_d_eer:.6f}, FNMR={fnmr_at_d_eer:.6f})")

    # -----------------------
    # 8) Worst-case morph metrics  (wc-MMPMR, wc-D-EER)
    #    Theoretical: no morph image loaded — z_wc = normalize(z_src_A + z_src_B)
    #    Gated by --compute_wc_metrics; requires --pairlist_csv; synface only.
    # -----------------------
    wc_metrics = None
    if args.compute_wc_metrics:
        if not asymmetric:
            print("[WC] note: no adapter loaded — wc embeddings will be in backbone space "
                  "(consistent with real-morph scoring in this symmetric run).")
        print("[WC] Computing worst-case morph metrics ...")

        if args.dataset == "frll_amsl":
            # Pairs: morph filenames encode (id_a, id_b).
            # Sources: ref_embs_by_id[id] = single _03 image, adapter space.
            # Probes:  probe_embs_by_id[id] = single _08 image, backbone space.
            # Structural disjointness — no pairlist CSV needed.
            wc_metrics = compute_wc_metrics(
                ref_embs_by_id=ref_embs_by_id,
                probe_embs_by_id=probe_embs_by_id,
                genuine_scores=genuine_scores,
                tau=tau,
                morph_pairs_ids=morph_pairs,
            )
        else:  # synface
            if not morph_meta:
                print("[WC] synface: --pairlist_csv with imgA/imgB columns is required "
                      "to identify source images; skipping.")
            else:
                wc_metrics = compute_wc_metrics(
                    ref_embs_by_id=ref_embs_by_id,
                    probe_embs_by_id=probe_embs_by_id,
                    genuine_scores=genuine_scores,
                    tau=tau,
                    morph_meta=morph_meta,
                )

    # -----------------------
    # Save report
    # -----------------------
    out = {
        "dataset":  args.dataset,
        "encoding": "asymmetric" if asymmetric else "symmetric",
        "n_ids":          n_ids,
        "n_real_images":  int(all_ref_embs.shape[0]),
        "n_unique_morphs": int(morph_embs.shape[0]),
        # verification metrics (unaffected by morph aggregation mode)
        "impostor_scoring": "exhaustive" if args.dataset == "frll_amsl" else "sampled",
        "impostor_pairs_sampled": (int(impostor_scores.numel())
                                   if args.dataset == "frll_amsl"
                                   else int(args.impostor_pairs)),
        "target_fmr":   float(args.fmr),
        "tau":          float(tau),
        "genuine_pairs": int(genuine_scores.numel()),
        "fnmr":          float(fnmr),
        "eer":           float(eer),
        "tau_eer":       float(tau_eer),
        "fmr_at_eer":    float(fmr_at_eer),
        "fnmr_at_eer":   float(fnmr_at_eer),
        # morph detection metadata
        "morph_trials": int(morph_scores.numel()),
        "morph_agg":    args.morph_agg,
        "morph_rule":   args.morph_rule,
        # score distribution stats (verification side; morph side below)
        "impostor_mean": float(impostor_scores.mean()),
        "impostor_std":  float(impostor_scores.std()),
        "genuine_mean":  float(genuine_scores.mean()) if genuine_scores.numel() > 0 else None,
        "genuine_std":   float(genuine_scores.std())  if genuine_scores.numel() > 0 else None,
        # provenance
        "backbone":               args.backbone,
        "used_adapter":           bool(args.adapter_ckpt),
        "disjointness_enforced":  disjointness_enforced,
        "base_ckpt":              args.base_ckpt,
        "adapter_ckpt":           args.adapter_ckpt if args.adapter_ckpt else None,
    }
    if args.dataset == "synface":
        out["split"]        = args.split
        out["pairlist_csv"] = args.pairlist_csv if args.pairlist_csv else None
    else:
        out["bonafide_dir"] = args.bonafide_dir
        out["morph_dir"]    = args.morph_dir

    # Random-mode provenance fields
    if args.morph_agg == "random":
        out["morph_random_seed"]   = args.morph_random_seed
        out["morph_random_n_runs"] = n_runs

    # Detection metrics: single-run (Mode A or Mode B n_runs=1) vs. multi-run (Mode B n_runs>1)
    if n_runs > 1:
        out.update({
            "mmpmr_mean":            mmpmr,
            "mmpmr_std":             mmpmr_std,
            "d_eer_mean":            d_eer,
            "d_eer_std":             d_eer_std,
            "tau_d_eer_mean":        tau_d_eer,
            "tau_d_eer_std":         tau_d_eer_std,
            "mar_at_d_eer_mean":     mar_at_d_eer,
            "mar_at_d_eer_std":      mar_at_d_eer_std,
            "fnmr_at_d_eer_mean":    fnmr_at_d_eer,
            "fnmr_at_d_eer_std":     fnmr_at_d_eer_std,
            "morph_score_mean":      morph_score_mean,
            "morph_score_mean_std":  morph_score_mean_std,
        })
    else:
        out.update({
            "mmpmr":            float(mmpmr),
            "d_eer":            float(d_eer),
            "tau_d_eer":        float(tau_d_eer),
            "mar_at_d_eer":     float(mar_at_d_eer),
            "fnmr_at_d_eer":    float(fnmr_at_d_eer),
            "morph_score_mean": float(morph_scores.mean()) if morph_scores.numel() > 0 else None,
            "morph_score_std":  float(morph_scores.std())  if morph_scores.numel() > 0 else None,
        })

    if wc_metrics is not None:
        out.update({
            "wc_pairs":         int(wc_metrics["wc_pairs"]),
            "wc_mmpmr":         float(wc_metrics["wc_mmpmr"]),
            "wc_d_eer":         float(wc_metrics["wc_d_eer"]),
            "tau_wc_d_eer":     float(wc_metrics["tau_wc_d_eer"]),
            "mar_at_wc_d_eer":  float(wc_metrics["mar_at_wc_d_eer"]),
            "fnmr_at_wc_d_eer": float(wc_metrics["fnmr_at_wc_d_eer"]),
            "wc_score_mean":    float(wc_metrics["wc_score_mean"]),
            "wc_score_std":     float(wc_metrics["wc_score_std"]),
        })

    Path(args.out_json).write_text(json.dumps(out, indent=2))
    print(f"[Saved] {args.out_json}")


if __name__ == "__main__":
    main()