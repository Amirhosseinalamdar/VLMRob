# vlm_eval/data.py
import os, glob, csv, json
from typing import Optional, Dict, List
from PIL import Image
from torch.utils.data import Dataset

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def _stem(p: str) -> str:
    return os.path.splitext(os.path.basename(p))[0]

class ImageTextPairDataset(Dataset):
    """
    Expects:
      - A directory `root` containing images with unique names (e.g., foo.jpg).
      - A CSV file at `os.path.join(root, csv_name)` with two columns:
          image_id, text
        where image_id is the image filename stem (e.g., 'foo' for 'foo.jpg').
    Returns dict:
      {
        "image_id": <str>,    # stem of filename
        "image_path": <str>,
        "image_pil": <PIL.Image>,
        "target_text": <str>
      }
    """
    def __init__(self, root: str, csv_name: str = "targets.csv", limit: Optional[int] = None):
        self.root = root
        self.csv_path = os.path.join(root, csv_name)
        if not os.path.isfile(self.csv_path):
            raise FileNotFoundError(f"targets CSV not found: {self.csv_path}")
        # Load targets mapping: image_id -> text
        targets: Dict[str, str] = {}
        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if "image_id" not in reader.fieldnames or "text" not in reader.fieldnames:
                raise ValueError(f"{self.csv_path} must have columns: image_id,text")
            for row in reader:
                iid = (row.get("image_id") or "").strip()
                txt = (row.get("text") or "").strip()
                if iid:
                    targets[iid] = txt  # last one wins if duplicates
        # Discover images and match to targets by stem
        img_paths: List[str] = []
        for ext in IMG_EXTS:
            img_paths.extend(glob.glob(os.path.join(root, f"*{ext}")))
        img_paths.sort()
        pairs = []
        missing_targets = 0
        for p in img_paths:
            iid = _stem(p)
            if iid in targets:
                pairs.append((p, iid, targets[iid]))
            else:
                missing_targets += 1
        if limit is not None:
            pairs = pairs[:limit]
        if len(pairs) == 0:
            raise RuntimeError(f"No image/target matches found in {root}. "
                               f"Images: {len(img_paths)}, matched: 0, missing_targets: {missing_targets}")
        if missing_targets > 0:
            print(f"[data] Warning: {missing_targets} images had no target in {self.csv_path} and were skipped.")
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict:
        img_path, iid, tgt = self.pairs[idx]
        with Image.open(img_path) as im:
            im = im.convert("RGB")
        return {
            "image_id": iid,
            "image_path": img_path,
            "image_pil": im,
            "target_text": tgt,
        }

class JSONLImageTextDataset(Dataset):
    """
    Expects a JSONL file with objects:
      {"image": "/abs/or/rel/path/to/img.jpg", "text": "target caption"}

    Returns dict with the same keys as ImageTextPairDataset:
      {
        "image_id": <str>,    # set to the image path string (as requested)
        "image_path": <str>,  # absolute or relative as provided
        "image_pil": <PIL.Image>,
        "target_text": <str>
      }
    """
    def __init__(self, jsonl_path: str, limit: Optional[int] = None):
        if not os.path.isfile(jsonl_path):
            raise FileNotFoundError(f"JSONL not found: {jsonl_path}")
        records: List[Dict[str, str]] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if "image" not in obj or "text" not in obj:
                    raise ValueError(f"JSONL line missing keys 'image' and 'text': {obj}")
                img_path = obj["image"]
                tgt_text = obj["text"] or ""
                records.append({"image": img_path, "text": tgt_text})
        # Keep order as-is; optionally limit
        if limit is not None:
            records = records[:limit]
        if not records:
            raise RuntimeError(f"No records found in JSONL: {jsonl_path}")
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict:
        rec = self.records[idx]
        img_path = rec["image"]
        tgt = rec["text"]
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"Image path from JSONL does not exist: {img_path}")
        with Image.open(img_path) as im:
            im = im.convert("RGB")
        # image_id is set to the path string so the results CSV contains the image path in that column
        return {
            "image_id": img_path,
            "image_path": img_path,
            "image_pil": im,
            "target_text": tgt,
        }