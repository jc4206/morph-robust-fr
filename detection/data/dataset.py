import os
import random
import csv
from statistics import median
from PIL import Image
from torch.utils.data import Dataset, get_worker_info


def load_morph_meta_map(pairlist_csv):
    """
    Reads pairlist CSV and maps morph_path -> metadata for both contributors.
    Expected columns: idA, idB, imgA, imgB, out_full
    """
    m = {}
    with open(pairlist_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            meta = {
                "idA": int(row["idA"]),
                "idB": int(row["idB"]),
                "imgA": str(row["imgA"]),
                "imgB": str(row["imgB"]),
            }

            # map all available morph variants to same metadata
            for col in ("out_full", "out_inA", "out_inB"):
                p = str(row.get(col, "")).strip()
                if p:
                    m[p] = meta
    return m

def load_morph_loss_map(hard_stats_csv):
    """
    Reads morph loss stats CSV and returns morph_path -> last_loss.
    Expected columns: morph_path, last_loss
    """
    out = {}
    with open(hard_stats_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            p = str(row.get("morph_path", "")).strip()
            if not p:
                continue
            try:
                out[p] = float(row.get("last_loss", "nan"))
            except Exception:
                continue
    return out

class TetraQuadrupleDataset(Dataset):
    def __init__(self, real_index, morphs_by_id, transform=None, length=200000, seed=42, pairlist_csv="", hard_stats_csv="", hard_mix=0.0, hard_alpha=1.0, hard_eps=1e-6, load_contributor_b=False, load_probe_b=False):
        self.real_index = real_index
        self.morphs_by_id = morphs_by_id
        self.transform = transform
        self.length = length
        self.seed = seed

        self.all_ids = sorted(real_index.keys())

        self.morph_meta_map = load_morph_meta_map(pairlist_csv) if pairlist_csv else {}
        self.load_contributor_b = load_contributor_b
        self.load_probe_b = load_probe_b

        # hard-sampling config
        self.hard_mix = float(hard_mix)
        self.hard_alpha = float(hard_alpha)
        self.hard_eps = float(hard_eps)
        self.hard_weights = None

        # build unique flat morph list once for morph-first sampling
        seen = set()
        self.morph_samples = []
        for _, morph_list in morphs_by_id.items():
            for morph_path, a, b, variant in morph_list:
                key = str(morph_path)
                if key in seen:
                    continue
                seen.add(key)
                self.morph_samples.append((str(morph_path), int(a), int(b), variant))

        if len(self.morph_samples) == 0:
            raise RuntimeError("No morph samples available for TetraQuadrupleDataset.")

        # build optional hard-sampling weights
        if hard_stats_csv:
            try:
                loss_map = load_morph_loss_map(hard_stats_csv)
                observed = [v for v in loss_map.values() if v == v]
                default_loss = median(observed) if observed else 1.0

                scores = []
                for morph_path, _, _, _ in self.morph_samples:
                    lv = loss_map.get(morph_path, default_loss)
                    if lv != lv: #NaN guard
                        lv = default_loss
                    s = max(lv, self.hard_eps) ** self.hard_alpha
                    scores.append(s)

                ssum = sum(scores)
                if ssum > 0:
                    self.hard_weights = [s / ssum for s in scores]

            except Exception:
                self.hard_weights = None

        self.anchor_eligible = {i for i, imgs in real_index.items() if len(imgs) >= 2}

    def __len__(self):
        return self.length

    def _load_img(self, path):
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

    def _sample_morph(self, rng):
        # fallback: uniform (old behavior)
        if self.hard_weights is None or self.hard_mix <= 0.0:
            return rng.choice(self.morph_samples)

        # mixture: hard-weighted + uniform exploration
        if rng.random() < self.hard_mix:
            idx = rng.choices(range(len(self.morph_samples)), weights=self.hard_weights, k=1)[0]
            return self.morph_samples[idx]
        return rng.choice(self.morph_samples)

    def __getitem__(self, idx):
        wi = get_worker_info()
        if wi is None:
            rng = random.Random(self.seed + idx)
        else:
            rng = random.Random(self.seed + wi.id * 10_000_000 + idx)

        # 1) sample morph first (instead of anchor id first)
        morph_path, a, b, variant = self._sample_morph(rng)

        # Ensure both contributors are valid anchors (>=2 images)
        for _ in range(20):
            if a in self.real_index and b in self.real_index and a in self.anchor_eligible and b in self.anchor_eligible:
                break
            morph_path, a, b, variant = self._sample_morph(rng)
        else:
            raise RuntimeError("Could not sample a valid morph with >=2 images for both contributors.")

        # 2) choose anchor contributor A randomly from morph contributors
        A = rng.choice([a, b])
        B = b if A == a else a

        # 3) anchor = image used to create this morph for chosen contributor
        imgsA = self.real_index[A]
        imgsA_str = [str(p) for p in imgsA]

        # use morph meta, choose imgA or imgB depending on chosen anchor contributor
        anchor_path = None
        meta = self.morph_meta_map.get(str(morph_path), None)
        if meta is not None:
            if A == meta["idA"]:
                anchor_path = meta["imgA"]
            elif A == meta["idB"]:
                anchor_path = meta["imgB"]

        # fallback if pairlist missing or mismatch
        if anchor_path is None or anchor_path not in imgsA_str:
            anchor_path = rng.choice(imgsA_str)

        # 4) positive = other image of same ID, not equal to anchor
        pos_candidates = [p for p in imgsA_str if p != anchor_path]
        pos_path = rng.choice(pos_candidates) if pos_candidates else anchor_path

        # 5) negative id C != both morph contributors (a and b)
        while True:
            C = rng.choice(self.all_ids)
            if C != a and C != b:
                break
        neg_path = str(rng.choice(self.real_index[C]))

        batch = {
            "anchor": self._load_img(anchor_path),
            "positive": self._load_img(pos_path),
            "morph": self._load_img(morph_path),
            "negative": self._load_img(neg_path),
            "ids": (A, B, C),
            "paths": (str(anchor_path), str(pos_path), str(morph_path), str(neg_path)),
            "variant": variant,
        }

        if self.load_contributor_b:
            # Resolve contributor B anchor — the exact image of B used in morph generation
            contrib_b_path = None
            if meta is not None:
                if B == meta["idA"]:
                    contrib_b_path = meta["imgA"]
                elif B == meta["idB"]:
                    contrib_b_path = meta["imgB"]
            # Fallback: random image of B if metadata missing or mismatched
            if contrib_b_path is None:
                imgsB_str = [str(p) for p in self.real_index.get(B, [])]
                contrib_b_path = rng.choice(imgsB_str) if imgsB_str else anchor_path
            batch["contributor_b"] = self._load_img(contrib_b_path)

        if self.load_probe_b:
            # probe_b: one random non-source image of contributor B (for two-sided val morph score)
            imgsB_str = [str(p) for p in self.real_index.get(B, [])]

            # Resolve source image of B to exclude it from probe candidates
            src_b_path = None
            if meta is not None:
                raw = meta["imgA"] if B == meta["idA"] else (meta["imgB"] if B == meta["idB"] else None)
                if raw is not None:
                    if raw in imgsB_str:
                        src_b_path = raw
                    else:
                        # basename fallback in case pairlist and real_index use different path formats
                        raw_bn = os.path.basename(raw)
                        src_b_path = next((p for p in imgsB_str if os.path.basename(p) == raw_bn), None)

            probe_B_candidates = [p for p in imgsB_str if p != src_b_path] if src_b_path else imgsB_str
            assert len(probe_B_candidates) > 0, (
                f"probe_B: no non-source images for identity {B} "
                f"(morph={morph_path}, src_b={src_b_path})"
            )
            probe_b_path = rng.choice(probe_B_candidates)

            # Sanity: probe must not be the source image (fires only if filtering above is broken)
            if src_b_path is not None:
                assert probe_b_path != src_b_path, (
                    f"Validation probe_B is source image! morph={morph_path}, "
                    f"probe_B={probe_b_path}, src_b={src_b_path}"
                )

            batch["probe_b"] = self._load_img(probe_b_path)

        return batch

