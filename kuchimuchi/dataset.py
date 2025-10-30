# dataset.py
import json
import os
import random
from typing import Optional, List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as T


class ImageNetCocoDataset(Dataset):
    """
    ImageFolder over ImageNet-1k validation + random COCO caption per image.

    Supports stratified subsampling by class labels using sklearn's train_test_split.

    Returns:
      - image_tensor_255: [3,224,224] float in [0,255]
      - image_basename:   str (e.g., ILSVRC2012_val_00000001.JPEG)
      - caption_text:     str (the selected COCO caption)
    """
    def __init__(self,
                 image_root: str,
                 coco_captions_json: str,
                 seed: int = 42,
                 num_samples: Optional[int] = None):
        super().__init__()
        self.rng = random.Random(seed)

        # Fixed 224 transform, output in [0,255] floats
        self.img_tf = T.Compose([
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
            T.CenterCrop(224),
            T.Lambda(lambda img: torchvision.transforms.functional.pil_to_tensor(img).float())
        ])
        self.img_folder = torchvision.datasets.ImageFolder(root=image_root, transform=self.img_tf)

        n_total = len(self.img_folder.samples)
        all_indices = np.arange(n_total)
        # Labels aligned with samples list
        # Each entry in samples is (path, class_idx)
        labels = np.array([cls for _, cls in self.img_folder.samples], dtype=np.int32)

        if num_samples is not None:
            if num_samples <= 0:
                raise ValueError(f"num_samples must be > 0, got {num_samples}")
            if num_samples >= n_total:
                # Use all images
                selected_idx = all_indices
            else:
                # Stratified selection of exactly num_samples items
                selected_idx, _ = train_test_split(
                    all_indices,
                    train_size=num_samples,
                    stratify=labels,
                    random_state=seed,
                    shuffle=True,
                )
                selected_idx = np.array(sorted(selected_idx.tolist()), dtype=np.int64)
        else:
            selected_idx = all_indices

        self.indices = selected_idx.tolist()

        captions_pool = self._load_coco_captions(coco_captions_json)
        if not captions_pool:
            raise RuntimeError("No COCO captions loaded from the provided JSON.")
        # Deterministic random caption per selected image index
        self.captions = [self.rng.choice(captions_pool) for _ in self.indices]

    def _load_coco_captions(self, json_path: str) -> List[str]:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        anns = data.get("annotations", [])
        return [a["caption"].strip() for a in anns if isinstance(a.get("caption"), str) and a["caption"].strip()]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, str]:
        ds_idx = self.indices[idx]
        img, _ = self.img_folder[ds_idx]                 # [3,224,224] in [0,255]
        path, _ = self.img_folder.samples[ds_idx]
        basename = os.path.basename(path)                # e.g., ILSVRC2012_val_00000001.JPEG
        caption = self.captions[idx]
        return img, basename, caption