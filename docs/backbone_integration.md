# Backbone integration: AdaFace and ArcFace

All training and evaluation scripts default to the AdaFace IR-50 backbone but
also support ArcFace IR-50 via `--backbone {adaface,arcface}`.

## Components

| File | Purpose |
|------|---------|
| `detection/models/adaface/net.py`, `detection/models/adaface/loader.py` | AdaFace IR-50 architecture + checkpoint loader |
| `detection/models/arcface/iresnet.py`, `detection/models/arcface/__init__.py` | ArcFace IR-50 (`iresnet50`), wrapped to return a `(features, norm)` tuple matching the AdaFace interface |

## Usage

Existing commands work unchanged (AdaFace is the default). To use ArcFace instead:

```bash
--backbone arcface --base_ckpt /path/to/arcface_ir50_ms1mv3_backbone.pth
```

Example (evaluation):
```bash
python -m detection.evaluation.eval \
  --backbone arcface \
  --base_ckpt $ARCFACE_CKPT \
  --adapter_ckpt $ADAPTER_CKPT \
  ...
```

Example (warm-up / baseline training):
```bash
python -m detection.training.train_adapter_triplet \
  --backbone arcface \
  --base_ckpt $ARCFACE_CKPT \
  ...
```

Example (morph-aware training):
```bash
python -m detection.training.train_adapter \
  --backbone arcface \
  --base_ckpt $ARCFACE_CKPT \
  ...
```

## Verifying a new backbone checkpoint

Before trusting an ArcFace (or other) checkpoint, sanity-check that the
backbone produces sensible embeddings: cosine similarity for same-identity
pairs should sit roughly in the 0.60–0.90 range, and for different identities
close to 0.00. Running `detection/evaluation/eval.py` without `--backbone`
(i.e. on the AdaFace default) is also a useful regression check after any
backbone-loading change, since AdaFace is the known-good baseline.
