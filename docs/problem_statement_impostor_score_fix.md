# Problem Statement: Impostor Score Fix and Downstream Effects on Training Pipeline

## Context

This project trains an MLP adapter on top of a frozen AdaFace IR-50 backbone for face morphing attack detection. The architecture uses a **quadruplet training setup** (anchor, positive, negative, morph) with a custom TetraLoss. Two training scripts exist:

- `train_adapter.py` — the main morph-aware training script (TetraLoss)
- `train_adapter_triplet.py` — a triplet-loss baseline for comparison

Both scripts compute **validation EER** (Equal Error Rate) to monitor training quality and drive early stopping. The EER is computed from genuine and impostor cosine similarity scores.

## Asymmetric Scoring Convention

The deployment scenario (eGate border control) is asymmetric:
- **Reference side** (document, which may be a morph) → encoded with `adapter(backbone(img))` — the adapted embedding space
- **Probe side** (live facial capture, always bona fide) → encoded with `backbone(img)` only — the backbone embedding space

This asymmetry is intentional and correct: only the reference goes through the adapter, because at inference time the live capture is never morphed.

`train_adapter.py` and `eval.py` both implement this correctly:
- Genuine score: `adapter(anchor) · backbone(positive)` — same identity, different encoding paths
- Impostor score: `adapter(anchor) · backbone(negative)` — different identity, backbone-only on the probe side

## The Bug in `train_adapter_triplet.py`

In `compute_val_metrics` inside `train_adapter_triplet.py`, `zn` (the adapter-encoded negative) was computed once for the triplet loss and then reused for the impostor score:

```python
# Before fix
zn = encode_adapter(backbone, adapter, batch["negative"], ...)  # for loss
impostor_scores.append((za * zn).sum(dim=1))  # WRONG: adapter on both sides
```

This means the impostor score was `adapter(anchor) · adapter(negative)` — applying the adapter to both sides, violating the deployment asymmetry.

## The Fix Applied

The negative is now re-encoded without the adapter for the impostor score:

```python
# After fix
zn = encode_adapter(backbone, adapter, batch["negative"], ...)  # still used for loss
zn_backbone = encode_backbone(backbone, batch["negative"], ...)  # backbone-only for scoring
impostor_scores.append((za * zn_backbone).sum(dim=1))  # correct: asymmetric
```

This aligns `train_adapter_triplet.py` with the convention in `train_adapter.py` and `eval.py`.

## Downstream Effects and Open Questions

### 1. Effect on Validation EER Values

The adapter is trained to sharpen identity discrimination. Applying it to the negative (impostor probe) likely produced **lower** impostor similarities than the backbone-only path, making the old EER **artificially optimistic**. After the fix, impostor scores are computed from a less discriminative (backbone-only) embedding on the probe side, so genuine and impostor score distributions may overlap more, and the corrected EER will likely be **higher**.

**Open question**: By how much does the EER change? Is the direction of the shift consistent across checkpoints and datasets? This is currently unknown without re-running experiments.

### 2. Effect on Early Stopping

`train_adapter_triplet.py` uses EER as its primary early stopping criterion. With the corrected (likely higher) EER:
- The patience counter and `min_delta` threshold interact differently with the new EER trajectory
- A different checkpoint may now be selected as "best"
- Training may run for more or fewer epochs before convergence

**Open question**: Does the corrected early stopping select a fundamentally different checkpoint (e.g., earlier or later in training), or do models converge to similar quality regardless?

### 3. Effect on Cross-Dataset Evaluation

`eval.py` is not changed and already uses the correct asymmetric scoring. However, if early stopping now selects a different "best" checkpoint, cross-dataset evaluation results will differ — not because the evaluation logic changed, but because a different model is being evaluated.

**Open question**: Given that the corrected EER is a more faithful proxy for deployment performance, does the newly selected checkpoint actually generalize better to the cross-dataset evaluation, or worse?

### 4. Comparability Between the Two Training Scripts

The original motivation for the fix was to make EER computation comparable between `train_adapter_triplet.py` (triplet baseline) and `train_adapter.py` (TetraLoss morph-aware model). After the fix, both report EER under the same asymmetric convention, making the comparison valid.

## Summary

| | `train_adapter.py` | `train_adapter_triplet.py` |
|---|---|---|
| Impostor score (before) | `adapter(anchor) · backbone(negative)` ✓ | `adapter(anchor) · adapter(negative)` ✗ |
| Impostor score (after) | unchanged | `adapter(anchor) · backbone(negative)` ✓ |
| EER comparability | — | now aligned with `train_adapter.py` |
| Early stopping affected | no | yes — EER values change |
| Eval results affected | no | indirectly, via checkpoint selection |

The fix is correct. The open question is the **magnitude** of the EER shift and whether it meaningfully changes which checkpoint gets selected and how the triplet baseline compares to the morph-aware model in cross-dataset evaluation.