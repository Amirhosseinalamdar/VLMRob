# datasets.py
import json
from json import JSONDecodeError
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Sequence, Union
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

# Constant clean-data root as requested
CLE_DATA_ROOT = Path("/home/user01/research/data/imagenet/val").resolve()

# Relative locations inside each tgt_base
ANNO_REL = Path("annotations/mscoco_t2i_captions_train2017_stratified.jsonl")
GEN_REL = Path("images/generated")

def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                if "image" not in obj or "text" not in obj:
                    raise ValueError(f"JSONL missing required keys in {path}: {obj}")
                records.append(obj)
    records.sort(key=lambda r: r["image"])
    return records

def _to_tensor_0_255(pic: Image.Image) -> torch.Tensor:
    mode_to_nptype = {"I": np.int32, "I;16": np.int16, "F": np.float32}
    img = torch.from_numpy(np.array(pic, mode_to_nptype.get(pic.mode, np.uint8), copy=True))
    img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
    img = img.permute((2, 0, 1)).contiguous()
    return img.to(dtype=torch.float32)

def _parse_bases_listish(x: Union[str, Sequence[str]]) -> List[Path]:
    if isinstance(x, (list, tuple)):
        return [Path(p).expanduser().resolve() for p in x]
    s = str(x).strip()
    # Try JSON list first
    try:
        maybe = json.loads(s)
        if isinstance(maybe, list):
            return [Path(p).expanduser().resolve() for p in maybe]
    except JSONDecodeError:
        pass
    # Fallback: split on commas or whitespace
    parts = [p for p in re.split(r"[,\s]+", s) if p]
    return [Path(p).expanduser().resolve() for p in parts]

class CleanTargetPairsDataset(Dataset):
    """
    Returns:
      clean_image_tensor (float32, 0..255, CHW),
      clean_text (str),
      target_image_tensors (List[float32 CHW] OR stacked Tensor [K,C,H,W]),
      target_text (str),
      clean_image_abs_path (str)

    Clean images read from CLE_DATA_ROOT / <clean_json['image']>.
    Target images for each base B: B/images/generated/<target_json['image']>.
    Annotations read once from <first_base>/annotations/mscoco_t2i_captions_train2017_stratified.jsonl.
    """
    def __init__(
        self,
        cle_file_path: str,
        input_res: int,
        tgt_base: Union[str, Sequence[str]],
        num_samples: Optional[int] = None,
        stack_targets: bool = False,  # if True => [K,C,H,W]; else list of tensors length K
    ):
        # Parse base(s)
        bases = _parse_bases_listish(tgt_base)
        if not bases:
            raise ValueError("--tgt_base must contain at least one path.")

        # Resolve annotation path from the first base; fail with helpful message if missing
        anno_candidates = [(b / ANNO_REL) for b in bases]
        anno_path = anno_candidates[0]
        if not anno_path.exists():
            tried = "\n  ".join(str(p) for p in anno_candidates[:3])  # show up to 3 to keep message short
            raise FileNotFoundError(
                f"Annotation JSONL not found at expected location:\n  {anno_path}\n"
                f"If your annotations live elsewhere, place them at <base>/{ANNO_REL}.\n"
                f"Other candidate paths (first few):\n  {tried}"
            )

        # Read records
        self.cle_recs = _read_jsonl(Path(cle_file_path).expanduser().resolve())
        self.tgt_recs = _read_jsonl(anno_path)

        # Dataset length / optional subsample
        self.length = min(len(self.cle_recs), len(self.tgt_recs))
        if num_samples is not None:
            self.length = min(self.length, int(num_samples))

        # Verify generated roots for each base
        self.gen_roots: List[Path] = []
        for b in bases:
            root = (b / GEN_REL).resolve()
            if not root.exists():
                raise FileNotFoundError(f"Target images root not found: {root}")
            self.gen_roots.append(root)

        self.stack_targets = bool(stack_targets)

        # Transforms
        self.transform = T.Compose(
            [
                T.Resize(input_res, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(input_res),
                T.Lambda(lambda img: img.convert("RGB")),
                T.Lambda(_to_tensor_0_255),  # 0..255 float32
            ]
        )

    def __len__(self) -> int:
        return self.length

    def _resolve_clean_path(self, rel_or_abs: str) -> Path:
        p = Path(rel_or_abs)
        return p if p.is_absolute() else (CLE_DATA_ROOT / p)

    def _load_image(self, path: Path) -> Image.Image:
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        return Image.open(path)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, Union[List[torch.Tensor], torch.Tensor], str, str]:
        cle_obj = self.cle_recs[idx]
        tgt_obj = self.tgt_recs[idx]

        # Clean
        cle_abs = self._resolve_clean_path(cle_obj["image"]).resolve()
        cle_img = self._load_image(cle_abs)
        cle_img_t = self.transform(cle_img)
        cle_text = cle_obj["text"].strip()

        # Targets (one per base)
        image_rel = tgt_obj["image"]
        tgt_text = tgt_obj["text"].strip()

        tgt_tensors: List[torch.Tensor] = []
        for root in self.gen_roots:
            tgt_abs = (root / image_rel).resolve()
            tgt_img = self._load_image(tgt_abs)
            tgt_tensors.append(self.transform(tgt_img))

        tgt_out: Union[List[torch.Tensor], torch.Tensor]
        if self.stack_targets:
            tgt_out = torch.stack(tgt_tensors, dim=0)  # [K,C,H,W]
        else:
            tgt_out = tgt_tensors  # List[Tensor]

        return cle_img_t, cle_text, tgt_out, tgt_text, str(cle_abs)