import random
from typing import Dict, List, Any
from PIL import Image
from torch.utils.data import Dataset, get_worker_info


class TripletDataset(Dataset):
    """
    Triplet dataset for bona fide-only metric learning.

    Each sampled item is a triplet:
        - anchor: image of identity A
        - positive: different image of identity A
        - negative: image of identity C, where C != A

    This dataset is "stochastic by index":
        - __len__ returns a virtual length (`length`)
        - __getitem__(idx) uses a deterministic RNG seed based on (seed, idx, worker_id)
        - therefore every epoch can present diverse triplets without pre-materializing all combinations

    Expected `real_index` format:
        {
            id_int_or_str: [path1, path2, ...],
            ...
        }

    Requirements:
        - anchor identities must have at least 2 bona fide images
        - negative identities must have at least 1 bona fide image
    """

    def __init__(
            self,
            real_index: Dict[Any, List[Any]],
            transform=None,
            length: int = 200000,
            seed: int = 42,
    ):
        self.real_index = real_index
        self.transform = transform
        self.length = int(length)
        self.seed = int(seed)

        # IDs eligible as anchor/positive source: need >= 2 images
        self.anchor_ids = sorted([i for i, imgs in real_index.items() if len(imgs) >= 2])

        # IDs eligible as negative source: need >= 1 image
        self.all_ids = sorted([i for i, imgs in real_index.items() if len(imgs) >= 1])

        if len(self.anchor_ids) == 0:
            raise RuntimeError("TripletDataset: no anchor IDs with >=2 images.")
        if len(self.all_ids) < 2:
            raise RuntimeError("TripletDataset: need at least 2 identities for negatives.")


    def __len__(self):
        """
        Virtual dataset length.

        This is not the number of unique precomputed triplets;
        triplets are sampled on-the-fly in __getitem__.
        """
        return self.length

    def _load_img(self, path):
        """
        Load an RGB image and apply transform (if provided).
        """
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img

    def _rng_for_idx(self, idx: int) -> random.Random:
        """
        Create a deterministic RNG per (idx, worker_id).

        Why:
            - avoids duplicated random sequences across DataLoader workers.
            - makes behavior reproducible given fixed seed + data ordering.
        """
        wi = get_worker_info()
        if wi is None:
            return random.Random(self.seed + idx)
        return random.Random(self.seed + wi.id * 10_000_000 + idx)

    def __getitem__(self, idx):
        """
        Sample one triplet:
            1) choose anchor identity A
            2) sample two different images from A -> anchor, positive
            3) choose negative identity  C != A
            4) sample one image from C -> negative
        """
        rng = self._rng_for_idx(idx)

        # 1) sample anchor identity
        A = rng.choice(self.anchor_ids)

        # 2) sample anchor/positive from same identity (must be different)
        imgsA = self.real_index[A]
        anchor_path, pos_path = rng.sample(imgsA, 2)

        # 3) sample negative identity C != A
        while True:
            C = rng.choice(self.all_ids)
            if C != A:
                break

        # 4) sample one negative image
        neg_path = rng.choice(self.real_index[C])

        batch = {
            "anchor": self._load_img(anchor_path),
            "positive": self._load_img(pos_path),
            "negative": self._load_img(neg_path),
            "ids": (A, C),
            "paths": (str(anchor_path), str(pos_path), str(neg_path)),
        }
        return batch 


