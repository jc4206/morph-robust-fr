# Multi-Blending-Ratio Morph Sampling Pipeline

This document explains step by step how morph images from three separate blending ratio
folders (`sim_morphs_3way`, `alpha_40`, `alpha_60`) are consolidated into a single unified
sampling pool during training, and how each blending variant remains independently
trackable throughout the pipeline.

---

## 1. Entry point: CLI arguments

`detection/training/train_adapter.py` receives the morph directories as a list:

```
--morph_root     .../sim_morphs_3way/train
                 .../blended_morphs/alpha_40/train
                 .../blended_morphs/alpha_60/train

--val_morph_root .../sim_morphs_3way/val
                 .../blended_morphs/alpha_40/val
                 .../blended_morphs/alpha_60/val
```

`argparse` collects these as a Python list because both arguments use `nargs="+"` /
`nargs="*"`. The list is passed directly to `build_morph_index`.

---

## 2. Index construction: `build_morph_index` (`detection/data/index_parser.py`)

```
build_morph_index(morphs_root=[
    ".../sim_morphs_3way/train",
    ".../blended_morphs/alpha_40/train",
    ".../blended_morphs/alpha_60/train"
], allowed_ids=split_ids)
```

The function iterates over every folder in the list and scans each for PNG files that
match the naming pattern:

```
MORPH_RE = r"^(\d{5})__(\d{5})_(full|inA|inB)\.png$"
```

For a file like `03423__09982_full.png` it extracts:
- `a = 03423`, `b = 09982`, `variant = "full"`
- canonical pair key: `(min(a,b), max(a,b))` = `(03423, 09982)`

**Deduplication is by full path** (`key = str(p)`). Because the three folders are
distinct filesystem locations, the three physically different versions of the same
identity pair are three distinct paths and are all retained:

```
seen_pairs after scanning all three folders:
  ".../sim_morphs_3way/train/03423__09982_full.png"   ← kept
  ".../alpha_40/train/03423__09982_full.png"           ← kept (different path)
  ".../alpha_60/train/03423__09982_full.png"           ← kept (different path)
```

Each morph is appended to the index under **both** contributor IDs so it can be reached
when either contributor is the anchor:

```python
morphs_by_id[x].append((p, x, y, variant))
morphs_by_id[y].append((p, x, y, variant))
```

**Result:** `morphs_by_id` is a `dict[int, list[(path, a, b, variant)]]` where each
identity maps to all morphs it contributed to, across all three blending folders.

---

## 3. Flat sampling pool: `TetraQuadrupleDataset.__init__` (`detection/data/dataset.py`)

The constructor flattens `morphs_by_id` into a single deduplicated list `morph_samples`:

```python
seen = set()
self.morph_samples = []
for _, morph_list in morphs_by_id.items():
    for morph_path, a, b, variant in morph_list:
        key = str(morph_path)      # full path as unique key
        if key in seen:
            continue
        seen.add(key)
        self.morph_samples.append((str(morph_path), int(a), int(b), variant))
```

Because each identity appears in `morphs_by_id` for both contributors, every morph
would be encountered twice during iteration (once from `morphs_by_id[a]`, once from
`morphs_by_id[b]`). The `seen` set prevents duplicates while still keeping all three
blending variants as separate entries.

**Resulting pool for one identity pair (03423, 09982), variant "full":**

| Index | Path | Blend |
|-------|------|-------|
| i     | `.../sim_morphs_3way/train/03423__09982_full.png` | 50:50 |
| i+1   | `.../alpha_40/train/03423__09982_full.png`         | 40:60 |
| i+2   | `.../alpha_60/train/03423__09982_full.png`         | 60:40 |

All three are independent entries. Sampling one does not exclude the others.

---

## 4. Metadata lookup: `load_morph_meta_map` (`detection/data/dataset.py`)

The pairlist CSV (a single file, e.g., from `sim_morphs_3way`) maps morph paths to
contributor metadata (source image paths for identity A and B):

```
idA, idB, imgA,        imgB,        out_full
3423, 9982, .../A.png, .../B.png,   .../sim_morphs_3way/train/03423__09982_full.png
```

When this CSV is loaded, each morph is indexed under **two keys**:

1. **Full path** (primary): `.../sim_morphs_3way/train/03423__09982_full.png`
2. **Basename** (fallback): `03423__09982_full.png`

```python
m[p] = meta                              # full path key
bn = os.path.basename(p)
if bn and bn not in m:
    m[bn] = meta                         # basename key (only if not already present)
```

This means morphs from `alpha_40` and `alpha_60` — whose full paths are not in the CSV
— can still resolve their contributor metadata via the shared basename, since the
filename encodes the identity pair and variant uniquely.

---

## 5. Per-item sampling: `__getitem__` (`detection/data/dataset.py`)

At each training step, a batch item is constructed as follows:

**5a. Sample a morph**

```python
morph_path, a, b, variant = self._sample_morph(rng)
```

`_sample_morph` draws uniformly from `morph_samples` (or from a loss-weighted
distribution if hard mining is active). The drawn path is one of the three blending
variants — whichever the random draw lands on.

**5b. Resolve contributor metadata**

```python
meta = (self.morph_meta_map.get(str(morph_path))
        or self.morph_meta_map.get(os.path.basename(str(morph_path))))
```

For a morph drawn from `alpha_40` or `alpha_60`, the full-path lookup fails (not in the
CSV), so the basename fallback fires and returns the same contributor metadata as the
50:50 version. This is correct because contributor identities and source images are
identical across all three blending folders.

**5c. Build the quadruplet**

The anchor and positive are the real images of contributor A (the source image used in
morph generation, and one other image of the same identity). The negative is a random
image of an unrelated identity C ≠ A, C ≠ B.

---

## 6. Identity and independence of blending variants

Throughout the entire pipeline the **absolute file path** is the identity of a morph:

| Component | Key used | Effect |
|-----------|----------|--------|
| `build_morph_index` deduplication | `str(p)` (full path) | all three variants kept |
| `TetraQuadrupleDataset.morph_samples` deduplication | `str(morph_path)` (full path) | all three variants in pool |
| `update_morph_loss_stats` (hard mining tracker) | `str(mp)` (full path) | each variant tracked independently |
| `flush_morph_loss_stats_csv` | dict key = full path | each variant gets its own CSV row |
| `load_morph_loss_map` (hard mining weights) | full path | weights applied per variant independently |

Sampling `sim_morphs_3way/train/03423__09982_full.png` in one step does not affect the
probability of sampling `alpha_40/train/03423__09982_full.png` in any future step. Hard
mining weights are also per-variant: a high-loss alpha_40 morph is up-weighted
independently of its 50:50 counterpart.

---

## 7. Data flow diagram

```
CLI: --morph_root [sim_morphs_3way/train, alpha_40/train, alpha_60/train]
         │
         ▼
build_morph_index()
  scans all three dirs, dedupes by full path
  → morphs_by_id: dict[identity_id → list[(full_path, a, b, variant)]]
         │
         ▼
TetraQuadrupleDataset.__init__()
  flattens + dedupes by full path
  → morph_samples: flat list of (full_path, a, b, variant)
    contains all 3 blending versions as separate entries
         │
         ▼
__getitem__(idx)
  _sample_morph() draws one entry from morph_samples
         │
         ├─ full path → morph_meta_map (primary key)
         │       if miss → basename fallback (secondary key)
         │       → contributor metadata (imgA, imgB, idA, idB)
         │
         └─ build quadruplet: anchor, positive, morph, negative
                  │
                  ▼
            training step → loss → update_morph_loss_stats(full_path, loss)
```
