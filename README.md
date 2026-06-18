# morph-robust-fr

Code accompanying a master's thesis on detecting face-morphing attacks in
face-recognition systems. A frozen [AdaFace](https://github.com/mk-minchul/AdaFace)
IR-50 backbone is paired with a small trainable MLP adapter, trained with a
quadruplet loss (**TetraLoss**, following Ibsen et al., 2024,
[arXiv:2401.11598](https://arxiv.org/abs/2401.11598)) to push morph embeddings
away from both identities a morph was built from.

This repository contains the full pipeline needed to reproduce the
experiments: similarity-driven identity pairing, morph generation, dataset
loading and sampling, training, and evaluation.

## Architecture

```
                    ┌───────────────────────┐
   reference image  │  AdaFace IR-50         │   512-d
   (may be a morph) │  (frozen backbone)     ├──────────┐
                    └───────────────────────┘          │
                                                 ┌──────▼──────┐
                                                 │ MLP adapter │   512-d, L2-normalized
                                                 │ (trainable) │──────────┐
                                                 └─────────────┘          │
                                                                          ▼
                                                                  cosine similarity
                                                                          ▲
                    ┌───────────────────────┐                            │
   live probe image │  AdaFace IR-50         │   512-d, L2-normalized    │
   (always bona fide)│ (frozen backbone)     ├────────────────────────────┘
                    └───────────────────────┘
```

**Why the encoding is asymmetric:** the deployment scenario (an eGate-style
border control check) compares a reference image, which may be a morphed
document photo, against a live probe capture, which is always a genuine,
unmorphed image of the person standing at the gate. Only the reference side
ever needs adapter-based morph robustness. The probe side is encoded with the
backbone alone. Training mirrors this: the adapter is applied to the anchor
and morph embeddings, never to the positive or negative.

**MLP adapter** (`detection/models/mlp_adapter.py`): 4 linear layers
(512→512), each followed by BatchNorm, the first three followed by
LeakyReLU(0.2), with L2-normalization on the final output.

**Quadruplet structure used in training:**
- `anchor` — a reference image of identity A (passed through backbone + adapter)
- `positive` — a second image of identity A (backbone only)
- `morph` — a morph generated from A and a contributor B (backbone + adapter)
- `negative` — an image of an unrelated identity C (backbone only)

**TetraLoss** (`detection/losses.py`):

```
L = relu( d(anchor, positive) + margin − min( d(anchor, negative), d(anchor, morph) ) )
```

All embeddings are L2-normalized, so cosine distances live in `[0, 2]`; with
`margin = 3.0` the loss has a geometric floor around 1.0 and never reaches
exactly zero. `losses.py` also contains several loss variants explored during
the thesis (directed/worst-case/balanced TetraLoss variants, triplet
baselines, repulsion losses) — see the class docstrings for the differences.

## Repository layout

```
detection/                  # face-recognition / morph-detection side
  models/
    adaface/                 # AdaFace IR-50 backbone (third-party, see LICENSE-THIRD-PARTY)
    arcface/                 # ArcFace IR-50 backbone, optional alternative (third-party)
    mlp_adapter.py            # MLPAdapter
  losses.py                  # TetraLoss and variants, TripletLoss
  data/
    dataset.py                # TetraQuadrupleDataset (quadruplet sampler, hard-mining)
    dataset_triplet.py         # TripletDataset (baseline)
    index_parser.py            # build_real_index / build_morph_index
    preprocessing.py           # shared image transform
  similarity_pairing/        # 5-stage identity-pairing pipeline (see below)
  training/
    train_adapter.py           # main training script (TetraLoss + MLP adapter)
    train_adapter_triplet.py   # baseline (TripletLoss, bona fide data only)
  evaluation/
    eval.py                    # FRLL/AMSL evaluation: EER, D-EER, MMPMR, worst-case metrics
    plot_det.py, plot_det_deer.py, plot_roc_det.py, plot_score_histogram.py
  data_prep/                 # train/val/test split utilities, preprocessing figure generator

morphing/                    # morph-generation side
  morph_function_final.py     # core morph() function (Delaunay triangulation + warp + blend)
  morphs_based_on_pairs.py    # main entry point: pairs CSV + landmarks -> morph images
  export_landmarks.py         # batch dlib 68-point landmark extraction
  pairing/                     # random-pair generation, blend-ratio pairlist tagging
  validation/                  # pipeline sanity checks

docs/                        # design notes and methodology documentation
```

## Setup

```bash
pip install -r requirements.txt
pip install -e .   # makes `detection` and `morphing` importable from anywhere
```

You will also need:
- An AdaFace IR-50 checkpoint (and optionally an ArcFace IR-50 checkpoint — see `docs/backbone_integration.md`)
- The dlib 68-point face landmark predictor (`shape_predictor_68_face_landmarks.dat`), available from the
  [dlib model repository](https://github.com/davisking/dlib-models) — not included in this repo, see `.gitignore`
- A face dataset organized as one folder per identity

## Reproducing the pipeline end to end

### 1. Similarity-based identity pairing (`detection/similarity_pairing/`)

```bash
python -m detection.similarity_pairing.index_dataset --dataset_root <faces/> --out index.csv
python -m detection.similarity_pairing.extract_embeddings --index index.csv --ckpt <adaface.ckpt> --out embeddings.npy
python -m detection.similarity_pairing.build_prototypes --embeddings embeddings.npy --normalize-input --out prototypes.npy
python -m detection.similarity_pairing.make_split_ids --ids identities.json --method cluster_holdout --prototypes prototypes.npy --id_map id_map.json
python -m detection.similarity_pairing.build_knn --prototypes prototypes.npy --id_map id_map.json --train-ids train_ids.json --val-ids val_ids.json --test-ids test_ids.json
python -m detection.similarity_pairing.sample_pairs --knn knn_train.csv --pairs-per-id 5 --topk 50
```

**Known gap:** `make_split_ids.py` only emits `train_ids.json` / `val_ids.json`,
but `build_knn.py` requires `--test-ids` as well. You need to derive a
held-out test ID set yourself (e.g. a third disjoint cluster split, or a
fixed-size random holdout from the identity pool) before running
`build_knn.py`. See `docs/PAIRING_METHODOLOGY.md` for the full methodology
and a clarification on partner-sampling behavior.

### 2. Morph generation (`morphing/`)

```bash
python -m morphing.export_landmarks --dataset_root <faces/> --out landmarks/
python -m morphing.morphs_based_on_pairs --pairs_csv pairs_train.csv --landmarks_dir landmarks/ --out_dir morphs/
python -m morphing.validation.run_sanity_checks --pairs_csv pairs_train.csv --morph_dir morphs/
```

For multi-blending-ratio experiments (e.g. 40:60, 60:40 in addition to 50:50),
see `morphing/pairing/make_blend_pairlists.py`, `morphing/pairing/add_cosine_similarity.py`,
and `docs/multi_blend_sampling_pipeline.md`.

### 3. Dataset/index prep (`detection/data_prep/`)

`make_split.py`, `make_3way_split.py`, and `merge_splits.py` build and
combine identity-disjoint train/val/test splits from pairing or morph
metadata, as needed for your experiment.

### 4. Training

```bash
python -m detection.training.train_adapter \
  --real_root <faces/> \
  --morph_root <morphs/train> [<morphs/alpha_40/train> ...] \
  --val_morph_root <morphs/val> \
  --base_ckpt <adaface.ckpt> \
  --split_dir <split/> \
  --pairlist_csv pairs_train.csv \
  --lr 1e-1 --momentum 0.9 --nesterov \
  --lr_sched step --lr_step_size 1000 --lr_gamma 0.1 \
  --margin 3.0
```

The baseline (no morphs, TripletLoss) is `detection.training.train_adapter_triplet`
with an analogous CLI.

### 5. Evaluation

```bash
python -m detection.evaluation.eval \
  --dataset frll_amsl \
  --base_ckpt <adaface.ckpt> \
  --adapter_ckpt <adapter.pt> \
  --bonafide_dir <frll/> --morph_dir <amsl_morphs/> \
  --out_json eval_out.json
```

`plot_det.py`, `plot_det_deer.py`, `plot_roc_det.py`, and
`plot_score_histogram.py` regenerate the DET-curve and score-distribution
figures from the evaluation output.

## Known limitations

- **Test-ID split gap** in the similarity-pairing pipeline — see step 1 above.
- **Asymmetric encoding** (adapter on reference, backbone-only on probe) is a
  deliberate design choice mirroring the deployment scenario, not an
  oversight — see the Architecture section.
- `--nesterov` is opt-in (default off) on `train_adapter.py`; pass it
  explicitly to match the paper-aligned configuration shown above.

## License

This repository is released under the MIT License (see `LICENSE`). It
includes adapted third-party code (AdaFace, ArcFace/insightface backbones)
under their respective licenses — see `LICENSE-THIRD-PARTY`.

## Citation

If you use this code, please cite the thesis it accompanies and the TetraLoss
paper:

> Ibsen, M., González-Soler, L. J., Rathgeb, C., Busch, C. "TetraLoss:
> Improving the Robustness of Face Recognition against Morphing Attacks."
> arXiv:2401.11598, 2024.
