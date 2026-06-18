import re
from pathlib import Path
from collections import defaultdict

ID_DIR_RE = re.compile(r"^\d{5}$") # "00001" ... "10000"
MORPH_RE = re.compile(r"^(\d{5})__(\d{5})_(full|inA|inB)\.png$", re.IGNORECASE)


def build_real_index(real_root: str):
    real_root = Path(real_root)
    real_index = {}
    for d in real_root.iterdir():
        if d.is_dir() and ID_DIR_RE.match(d.name):
            id_int = int(d.name)
            imgs = sorted([p for p in d.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
            if len(imgs) >= 2:
                real_index[id_int] = imgs
    return real_index

def build_morph_index(morphs_root: str, dedupe_pairs: bool = True, allowed_ids=None):
    """
    Expects flat morph folder:
        morphs_root/03423__09982_full.png
        morphs_root/0349__09982_inA.png
        morphs_root/03423__09982_inB.png
    """
    root = Path(morphs_root)
    morphs_by_id = defaultdict(list)
    seen_pairs = set()

    for p in root.iterdir():
        if not p.is_file() or p.suffix.lower() != ".png":
            continue

        m = MORPH_RE.match(p.name)  # besser als search
        if not m:
            continue

        a = int(m.group(1))
        b = int(m.group(2))
        variant = m.group(3).lower()

        x, y = (a, b) if a < b else (b, a)

        # EDIT 26-01: split filter --> this automatically drops cross-split morphs
        if allowed_ids is not None and (x not in allowed_ids or y not in allowed_ids):
            continue

        if dedupe_pairs:
            key = (x, y, variant)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)

        # store for both contributors (canonical order)
        morphs_by_id[x].append((p, x, y))
        morphs_by_id[y].append((p, x, y))

    return morphs_by_id


