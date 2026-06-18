"""
Checkpoint loading for the AdaFace and ArcFace IR-50 backbones.

The AdaFace architecture (net.py) and checkpoint format handled here originate
from mk-minchul/AdaFace (https://github.com/mk-minchul/AdaFace), MIT License,
Copyright (c) 2022 Minchul Kim. See LICENSE-THIRD-PARTY at the repo root.
"""

from pathlib import Path
import torch
from detection.models.adaface.net import build_model
from detection.models.arcface.iresnet import iresnet50 as arcface_iresnet50

def _extract_state_dict(ckpt):
    # Lightning checkpoints often store weights in ckpt["state_dict"]
    if isinstance(ckpt, dict):
        for k in ["state_dict", "model_state_dict", "model", "net", "backbone"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
    if isinstance(ckpt, dict):
        return ckpt
    raise ValueError("Checkpoint has no dict-like state_dict.")

def _auto_prefix_remap(sd, model_keys):
    candidates = [
        [],  # no stripping
        ["module."],
        ["model."],
        ["backbone."],
        ["model.backbone."],
        ["module.model."],
        ["module.backbone."],
        ["module.model.backbone."],
    ]
    model_key_set = set(model_keys)

    best_prefixes = None
    best_match = -1
    best_sd = None

    for prefixes in candidates:
        remapped = {}
        for k, v in sd.items():
            nk = k
            # strip prefixes iteratively (handles nested like module.model.)
            changed = True
            while changed:
                changed = False
                for p in prefixes:
                    if nk.startswith(p):
                        nk = nk[len(p):]
                        changed = True
            remapped[nk] = v

        match = sum(1 for k in remapped.keys() if k in model_key_set)
        if match > best_match:
            best_match = match
            best_prefixes = prefixes
            best_sd = remapped

    return best_sd, best_prefixes, best_match

def load_adaface_ir50(ckpt_path: str, device: str = "cpu", strict: bool = False):
    ckpt_path = str(Path(ckpt_path))
    model = build_model("ir_50")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd_raw = _extract_state_dict(ckpt)

    model_keys = list(model.state_dict().keys())
    sd, prefixes, match = _auto_prefix_remap(sd_raw, model_keys)

    missing, unexpected = model.load_state_dict(sd, strict=strict)

    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False

    print(f"[AdaFace] loaded IR-50 from {ckpt_path}")
    print(f"[AdaFace] best prefix strip: {prefixes} | matched keys: {match}/{len(model_keys)}")
    print(f"[AdaFace] missing={len(missing)} unexpected={len(unexpected)}")

    return model


def load_arcface_ir50(ckpt_path: str, device: str = "cpu", strict: bool = False):
    """
    Load pretrained ArcFace IR-50 backbone from arcface_torch checkpoint.

    The checkpoint (arcface_ir50_ms1mv3_backbone.pth) is a plain OrderedDict
    with 475 keys, no module. prefix, float32. Keys start with conv1.weight.

    Returns the backbone in eval mode with all parameters frozen, producing
    (features, norm) tuples identical to the AdaFace backbone interface.
    """
    ckpt_path = str(Path(ckpt_path))
    model = arcface_iresnet50(pretrained=False, fp16=False)

    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and not any(k.startswith("conv1") for k in list(sd.keys())[:3]):
        # checkpoint wrapped in a container — unwrap
        for key in ["state_dict", "model_state_dict", "model", "backbone"]:
            if key in sd and isinstance(sd[key], dict):
                sd = sd[key]
                break

    missing, unexpected = model.load_state_dict(sd, strict=strict)

    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False

    print(f"[ArcFace] loaded IR-50 from {ckpt_path}")
    print(f"[ArcFace] missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print(f"[ArcFace] missing sample: {missing[:3]}")
    if unexpected:
        print(f"[ArcFace] unexpected sample: {unexpected[:3]}")

    return model




