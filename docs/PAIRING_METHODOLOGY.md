# Similarity-pairing methodology

This document describes how `detection/similarity_pairing/` selects which
identity pairs are morphed, and clarifies one point where a naive reading of
the pipeline's behavior can be misleading.

Pipeline stages:
- `extract_embeddings.py`
- `build_prototypes.py`
- `make_split_ids.py`
- `build_knn.py`
- `sample_pairs.py`

## 1. Prototype embedding computation

For each identity, a prototype embedding is the mean of the L2-normalized
backbone embeddings across all reference images for that identity, using the
frozen AdaFace IR-50 backbone (`extract_embeddings.py`, default checkpoint
`adaface_ir50_ms1mv2.ckpt`).

Per-image L2-normalization before averaging is controlled by the
`--normalize-input` flag in `build_prototypes.py` — it is opt-in, not
hardcoded. The final prototype is always L2-normalized regardless of that
flag.

## 2. K-means clustering

Prototype embeddings are clustered with k-means (`make_split_ids.py`,
`--n-clusters`, default 100), seeded for reproducibility (`--seed`, default
1337).

## 3. Cluster-based train/validation split

`make_split_ids.py` supports three split methods via `--method`:
`random` (default), `cluster`, and `cluster_holdout`. Only
`cluster_holdout` assigns whole clusters to one partition (no identity
appears split across train/val) — this is the method to use if you want
train/val to differ in their prototype-similarity distribution as little as
possible at the cluster level. The split ratio defaults to 80:20
(`--val-ratio`).

## 4. kNN search within splits

Within each split, pairing candidates for each identity are found by
k-nearest-neighbour search over prototype embeddings using cosine similarity
(`build_knn.py`, `--k`, default 50). The search is restricted to identities
within the same split — never across splits — using FAISS `IndexFlatIP` when
available, with a NumPy blockwise dot-product fallback otherwise.

## 5. Stochastic partner sampling

`sample_pairs.py` draws `--pairs-per-id` partners (default 3) uniformly at
random from the top-`k` candidates for each identity (`--strategy uniform`).

**Important clarification on "fewer than N pairs":** it is tempting to assume
a total-degree cap is what causes some identities to end up with fewer than
the requested number of partners. That is controlled by `--max-degree`, which
defaults to *unlimited* (`0`). The actual reason some identities receive
fewer pairs is the **duplicate filter**: a candidate draw that would
reproduce a pair already formed earlier in the iteration is rejected and
resampled (up to `max_attempts = len(candidates) * 2`). Identities whose
top-k neighborhoods are heavily reused by earlier identities may exhaust
their candidate pool before reaching the requested count. Run statistics
(`pairs_stats_*.json`) report `n_pairs`, `duplicates_dropped`,
`skipped_max_degree`, and `skipped_no_candidates` so you can verify this for
your own run.

## 6. Morph variants

The pairing CSV (`pairs_{split}.csv`) records only `id_a`, `id_b`, the
pairing `score` (cosine similarity), and optionally `src_id`. It does not
encode which morph variants (`full`, `inA`, `inB`) to generate — that
decision is made downstream by `morphing/morphs_based_on_pairs.py`.

## 7. Output format

One CSV per split (`pairs_train.csv`, `pairs_val.csv`, `pairs_test.csv`),
each row giving the contributing identity IDs and pairing score. Source image
paths are not recorded here; image selection for each identity happens in
the morph-generation step.
