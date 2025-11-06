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

def verify_annotation_order(anno_paths: List[Path]) -> bool:
    """
    Verify that all annotation files have the same order of images.
    Uses image key: value.split('_')[0] to extract IDs for comparison.
    
    Args:
        anno_paths: List of paths to annotation JSONL files
        
    Returns:
        True if all annotation files have the same order, False otherwise
    """
    if not anno_paths:
        return True
    
    # Read all annotation files
    all_records = []
    for anno_path in anno_paths:
        if not anno_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {anno_path}")
        all_records.append(_read_jsonl(anno_path))
    
    # IMPORTANT: Since _read_jsonl sorts by image name, we need to extract the order
    # after sorting, not from the original file order
    all_image_names = []
    for records in all_records:
        image_names = [record.get("image", "") for record in records]
        all_image_names.append(image_names)
    
    # Verify all annotation files have the same number of records
    num_records = len(all_image_names[0])
    for i, image_names in enumerate(all_image_names):
        if len(image_names) != num_records:
            raise ValueError(
                f"Annotation file {anno_paths[i]} has {len(image_names)} records, "
                f"but first file has {num_records} records"
            )
    
    # Verify the order is the same across all annotation files
    for j in range(num_records):
        first_name = all_image_names[0][j]
        first_id = first_name.split('_')[0] if '_' in first_name else first_name
        
        for i, image_names in enumerate(all_image_names[1:], 1):
            current_name = image_names[j]
            current_id = current_name.split('_')[0] if '_' in current_name else current_name
            
            if current_id != first_id:
                raise ValueError(
                    f"Order mismatch at position {j}: "
                    f"File {anno_paths[0]} has image '{first_name}' (ID: '{first_id}'), "
                    f"but file {anno_paths[i]} has image '{current_name}' (ID: '{current_id}')"
                )
    
    return True

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
    Each base uses its own annotations from <base>/annotations/mscoco_t2i_captions_train2017_stratified.jsonl.
    """
    def __init__(
        self,
        cle_file_path: str,
        input_res: int,
        tgt_base: Union[str, Sequence[str]],
        num_samples: Optional[int] = None,
        stack_targets: bool = False,  # if True => [K,C,H,W]; else list of tensors length K
        verify_order: bool = True,  # Whether to verify annotation order across bases
        cle_data_root: Optional[Path] = None,  # Make CLE_DATA_ROOT configurable
    ):
        # Parse base(s)
        self.bases = _parse_bases_listish(tgt_base)
        if not self.bases:
            raise ValueError("--tgt_base must contain at least one path.")
        
        # Set clean data root (default or provided)
        self.cle_data_root = Path(cle_data_root) if cle_data_root else CLE_DATA_ROOT
        
        # Get annotation paths for all bases
        self.anno_paths = [b / ANNO_REL for b in self.bases]
        
        # Verify all annotation files exist
        missing_annos = [p for p in self.anno_paths if not p.exists()]
        if missing_annos:
            raise FileNotFoundError(
                f"Annotation JSONL files not found at:\n"
                + "\n".join(f"  {p}" for p in missing_annos)
            )
        
        # Verify annotation order if requested
        if verify_order:
            verify_annotation_order(self.anno_paths)
        
        # Read records from each annotation file
        self.tgt_recs_list = [_read_jsonl(p) for p in self.anno_paths]
        
        # Read clean records
        cle_path = Path(cle_file_path).expanduser().resolve()
        if not cle_path.exists():
            raise FileNotFoundError(f"Clean annotations file not found: {cle_path}")
        self.cle_recs = _read_jsonl(cle_path)
        
        # Dataset length - use the minimum length among all annotation files
        min_tgt_length = min(len(recs) for recs in self.tgt_recs_list)
        self.length = min(len(self.cle_recs), min_tgt_length)
        
        if num_samples is not None:
            self.length = min(self.length, int(num_samples))
        
        # Verify generated roots for each base
        self.gen_roots: List[Path] = []
        for b in self.bases:
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
        return p if p.is_absolute() else (self.cle_data_root / p)

    def _load_image(self, path: Path) -> Image.Image:
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        return Image.open(path)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, Union[List[torch.Tensor], torch.Tensor], str, str]:
        cle_obj = self.cle_recs[idx]
        
        # Clean
        cle_abs = self._resolve_clean_path(cle_obj["image"]).resolve()
        cle_img = self._load_image(cle_abs)
        cle_img_t = self.transform(cle_img)
        cle_text = cle_obj["text"].strip()
        
        # Targets (one per base, each with its own annotation file)
        tgt_tensors: List[torch.Tensor] = []
        target_texts: List[str] = []
        
        for i, (tgt_recs, root) in enumerate(zip(self.tgt_recs_list, self.gen_roots)):
            tgt_obj = tgt_recs[idx]
            image_rel = tgt_obj["image"]
            tgt_text = tgt_obj["text"].strip()
            target_texts.append(tgt_text)
            
            tgt_abs = (root / image_rel).resolve()
            tgt_img = self._load_image(tgt_abs)
            tgt_tensors.append(self.transform(tgt_img))
        
        # Return the text from the first annotation file
        # (all should be the same due to ordering verification)
        tgt_text = target_texts[0] if target_texts else ""
        
        tgt_out: Union[List[torch.Tensor], torch.Tensor]
        if self.stack_targets:
            tgt_out = torch.stack(tgt_tensors, dim=0)  # [K,C,H,W]
        else:
            tgt_out = tgt_tensors  # List[Tensor]
        
        return cle_img_t, cle_text, tgt_out, tgt_text, str(cle_abs)

# import random
# class CleanTargetPairsDataset(Dataset):
#     """
#     Returns:
#       clean_image_tensor (float32, 0..255, CHW),
#       clean_text (str),
#       target_image_tensors (List[float32 CHW] OR stacked Tensor [K,C,H,W]),
#       target_text (str),
#       clean_image_abs_path (str)
#     Clean images read from CLE_DATA_ROOT / <clean_json['image']>.
#     Target images for each base B: B/images/generated/<target_json['image']>.
#     For each clean sample, randomly select targets from the target pool.
#     """
#     def __init__(
#         self,
#         cle_file_path: str,
#         input_res: int,
#         tgt_base: Union[str, Sequence[str]],
#         num_samples: Optional[int] = None,
#         stack_targets: bool = False,  # if True => [K,C,H,W]; else list of tensors length K
#         verify_order: bool = False,  # Disable order verification since we're sampling randomly
#         cle_data_root: Optional[Path] = None,  # Make CLE_DATA_ROOT configurable
#         seed: int = 42,  # Add seed for reproducible random sampling
#     ):
#         # Parse base(s)
#         self.bases = _parse_bases_listish(tgt_base)
#         if not self.bases:
#             raise ValueError("--tgt_base must contain at least one path.")
        
#         # Set clean data root (default or provided)
#         self.cle_data_root = Path(cle_data_root) if cle_data_root else CLE_DATA_ROOT
        
#         # Get annotation paths for all bases
#         self.anno_paths = [b / ANNO_REL for b in self.bases]
        
#         # Verify all annotation files exist
#         missing_annos = [p for p in self.anno_paths if not p.exists()]
#         if missing_annos:
#             raise FileNotFoundError(
#                 f"Annotation JSONL files not found at:\n"
#                 + "\n".join(f"  {p}" for p in missing_annos)
#             )
        
#         # Read records from each annotation file
#         self.tgt_recs_list = [_read_jsonl(p) for p in self.anno_paths]
        
#         # Read clean records
#         cle_path = Path(cle_file_path).expanduser().resolve()
#         if not cle_path.exists():
#             raise FileNotFoundError(f"Clean annotations file not found: {cle_path}")
#         self.cle_recs = _read_jsonl(cle_path)
        
#         # Determine dataset length - use clean samples as base
#         self.length = len(self.cle_recs)
        
#         if num_samples is not None:
#             self.length = min(self.length, int(num_samples))
        
#         # Get target dataset sizes for sampling
#         self.tgt_sizes = [len(recs) for recs in self.tgt_recs_list]
#         print(f"Target dataset sizes: {self.tgt_sizes}")
        
#         # Verify generated roots for each base
#         self.gen_roots: List[Path] = []
#         for b in self.bases:
#             root = (b / GEN_REL).resolve()
#             if not root.exists():
#                 raise FileNotFoundError(f"Target images root not found: {root}")
#             self.gen_roots.append(root)
        
#         self.stack_targets = bool(stack_targets)
        
#         # Set up random number generator with seed
#         self.rng = random.Random(seed)
        
#         # Pre-generate random indices for reproducibility
#         self.target_indices = []
#         for tgt_size in self.tgt_sizes:
#             # Generate random indices for each target dataset
#             indices = [self.rng.randint(0, tgt_size - 1) for _ in range(self.length)]
#             self.target_indices.append(indices)
        
#         # Transforms
#         self.transform = T.Compose(
#             [
#                 T.Resize(input_res, interpolation=T.InterpolationMode.BICUBIC),
#                 T.CenterCrop(input_res),
#                 T.Lambda(lambda img: img.convert("RGB")),
#                 T.Lambda(_to_tensor_0_255),  # 0..255 float32
#             ]
#         )
    
#     def __len__(self) -> int:
#         return self.length
    
#     def _resolve_clean_path(self, rel_or_abs: str) -> Path:
#         p = Path(rel_or_abs)
#         return p if p.is_absolute() else (self.cle_data_root / p)
    
#     def _load_image(self, path: Path) -> Image.Image:
#         if not path.exists():
#             raise FileNotFoundError(f"Image not found: {path}")
#         return Image.open(path)
    
#     def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, Union[List[torch.Tensor], torch.Tensor], str, str]:
#         cle_obj = self.cle_recs[idx]
        
#         # Clean
#         cle_abs = self._resolve_clean_path(cle_obj["image"]).resolve()
#         cle_img = self._load_image(cle_abs)
#         cle_img_t = self.transform(cle_img)
#         cle_text = cle_obj["text"].strip()
        
#         # Targets (randomly sampled from each base)
#         tgt_tensors: List[torch.Tensor] = []
#         target_texts: List[str] = []
        
#         for i, (tgt_recs, root, tgt_indices) in enumerate(zip(self.tgt_recs_list, self.gen_roots, self.target_indices)):
#             # Get random index for this clean sample and target dataset
#             tgt_idx = tgt_indices[idx]
#             tgt_obj = tgt_recs[tgt_idx]
            
#             image_rel = tgt_obj["image"]
#             tgt_text = tgt_obj["text"].strip()
#             target_texts.append(tgt_text)
            
#             tgt_abs = (root / image_rel).resolve()
#             tgt_img = self._load_image(tgt_abs)
#             tgt_tensors.append(self.transform(tgt_img))
        
#         # For single target dataset, we could also randomly select one text
#         # Or you might want to handle multiple target texts differently
#         tgt_text = target_texts[0] if target_texts else ""
        
#         tgt_out: Union[List[torch.Tensor], torch.Tensor]
#         if self.stack_targets:
#             tgt_out = torch.stack(tgt_tensors, dim=0)  # [K,C,H,W]
#         else:
#             tgt_out = tgt_tensors  # List[Tensor]
        
#         return cle_img_t, cle_text, tgt_out, tgt_text, str(cle_abs)