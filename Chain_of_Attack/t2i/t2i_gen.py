import argparse, json, os, hashlib, re
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional
from collections import defaultdict
import random

import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

# tqdm with graceful fallback
try:
    from tqdm import tqdm
except Exception:
    def tqdm(it, **kwargs):
        return it

def load_coco_captions(ann_path: str) -> Tuple[Dict[int, List[str]], Dict[int, str]]:
    """
    Returns:
      caps_by_img: {image_id -> [caption, ...]}
      id2filename: {image_id -> file_name}
    """
    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    id2filename = {img["id"]: img.get("file_name", str(img["id"])) for img in data["images"]}
    caps_by_img: Dict[int, List[str]] = {}
    for a in data["annotations"]:
        if "caption" in a:
            caps_by_img.setdefault(a["image_id"], []).append(a["caption"].strip())
    return caps_by_img, id2filename

def resolve_instances_path(captions_path: str) -> str:
    """
    Infer instances_{split}2017.json from captions_{split}2017.json living in the same directory.
    """
    p = Path(captions_path)
    stem = p.name.replace("captions_", "instances_")
    return str(p.with_name(stem))

def load_primary_category_per_image(instances_path: str) -> Tuple[Dict[int, int], Dict[int, str]]:
    """
    Parse COCO instances to determine a single primary category per image.
    Primary category = category with the highest instance count in that image,
    tie-break by total area, then by smaller category id.

    Returns:
      img2cat: {image_id -> primary_category_id}
      catid2name: {category_id -> name}
    """
    with open(instances_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    catid2name = {c["id"]: c.get("name", str(c["id"])) for c in data["categories"]}

    # Accumulate counts and areas per (image, category)
    counts = defaultdict(int)
    areas = defaultdict(float)
    for ann in data["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        img_id = ann["image_id"]
        cat_id = ann["category_id"]
        counts[(img_id, cat_id)] += 1
        # area may be missing; use 0 if so
        areas[(img_id, cat_id)] += float(ann.get("area", 0.0))

    # Decide primary category per image
    img2cat: Dict[int, int] = {}
    # collect candidate cats per image
    cats_per_img = defaultdict(list)
    for (img_id, cat_id), cnt in counts.items():
        cats_per_img[img_id].append(cat_id)

    for img_id, cat_ids in cats_per_img.items():
        # choose argmax by (count, area, -cat_id) -> final tie break by smaller id
        def key_fn(c):
            return (counts[(img_id, c)], areas[(img_id, c)], -c)
        best = max(cat_ids, key=key_fn)
        img2cat[img_id] = best

    return img2cat, catid2name

def stratified_image_ids(
    caps_by_img: Dict[int, List[str]],
    instances_path: str,
    total_captions: int,
    seed: int = 1234,
) -> List[int]:
    """
    Build a list of image_ids, stratified across categories, aiming for total_captions images.
    Uses only the first caption per image later on.
    """
    img2cat, _ = load_primary_category_per_image(instances_path)

    # Keep only images that have both captions and a primary category
    eligible_imgs = [img_id for img_id in caps_by_img.keys() if img_id in img2cat and caps_by_img[img_id]]
    # Map category -> image list
    cat2imgs = defaultdict(list)
    for img_id in eligible_imgs:
        cat2imgs[img2cat[img_id]].append(img_id)

    # Shuffle each category list deterministically and deduplicate image ids (just in case)
    for cat_id, lst in cat2imgs.items():
        # stable torch-independent randomness
        rng = random.Random(seed + int(cat_id))
        # ensure deterministic order irrespective of Python hash seed
        lst.sort()
        rng.shuffle(lst)
        # deduplicate while preserving order
        seen = set()
        uniq = []
        for x in lst:
            if x not in seen:
                seen.add(x)
                uniq.append(x)
        cat2imgs[cat_id] = uniq

    # Round-robin draw across categories until we hit total_captions or exhaust all
    cats = list(cat2imgs.keys())
    rng_global = random.Random(seed)
    cats.sort()
    rng_global.shuffle(cats)

    selected: List[int] = []
    if total_captions is None or total_captions <= 0:
        # If not specified, just take all in a balanced interleave
        total_target = sum(len(v) for v in cat2imgs.values())
    else:
        total_target = min(total_captions, sum(len(v) for v in cat2imgs.values()))

    # Cycling picks
    nonempty = set(c for c in cats if cat2imgs[c])
    while len(selected) < total_target and nonempty:
        for c in list(nonempty):
            if cat2imgs[c]:
                selected.append(cat2imgs[c].pop())
                if len(selected) >= total_target:
                    break
            if not cat2imgs[c]:
                nonempty.discard(c)

    return selected

def slugify(text: str, maxlen: int = 64) -> str:
    text = re.sub(r"\s+", "_", text.strip())
    text = re.sub(r"[^A-Za-z0-9_\-]+", "", text)
    return text[:maxlen] if text else "caption"

def build_sd_pipeline(model_id: str, torch_dtype: str = "float16", device: str = "cuda") -> StableDiffusionPipeline:
    dtype = torch.float16 if torch_dtype == "float16" and torch.cuda.is_available() else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    if device == "cuda" and torch.cuda.is_available():
        pipe = pipe.to("cuda")
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
    return pipe

def txt2img(
    pipe: StableDiffusionPipeline,
    prompt: str,
    out_path: str,
    seed: Optional[int] = None,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 30,
    height: int = 512,
    width: int = 512,
) -> str:
    generator = torch.Generator(device=pipe.device.type)
    if seed is not None:
        generator = generator.manual_seed(seed)
    img = pipe(
        prompt=prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        height=height,
        width=width,
        generator=generator,
    ).images[0]
    Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return out_path

def build_jsonl_for_generation(
    ann_path: str,
    out_root: str,
    model_id: str,
    total_captions: Optional[int] = None,
    seed: Optional[int] = 1234,
    steps: int = 30,
    guidance: float = 7.5,
    height: int = 512,
    width: int = 512,
    n_per_caption: int = 1,
    limit: Optional[int] = None,
    instances_path: Optional[str] = None,
) -> str:
    """
    Generates images for COCO captions and writes JSONL with {"image": "<relative path>", "text": "<caption>"}.
    Images are saved under out_root/images/generated/.
    Stratified sampling across categories based on instances annotations.
    Always uses the first caption per image.
    """
    caps_by_img, _ = load_coco_captions(ann_path)

    # Resolve instances file if not provided
    if not instances_path:
        instances_path = resolve_instances_path(ann_path)

    # Build stratified list of image ids
    selected_img_ids = stratified_image_ids(
        caps_by_img=caps_by_img,
        instances_path=instances_path,
        total_captions=(total_captions if total_captions is not None else 0),
        seed=1234,
    )

    out_images_root = os.path.join(out_root, "images", "generated")
    out_ann_dir = os.path.join(out_root, "annotations")
    Path(out_ann_dir).mkdir(parents=True, exist_ok=True)
    jsonl_path = os.path.join(out_ann_dir, f"mscoco_t2i_{Path(ann_path).stem}_stratified.jsonl")

    pipe = build_sd_pipeline(model_id=model_id)

    # Progress bar total is number of prompts we’ll run
    total_samples = len(selected_img_ids) * max(1, int(n_per_caption))
    if limit is not None:
        total_samples = min(total_samples, int(limit))

    count = 0
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for img_id in tqdm(selected_img_ids, desc="Generating (stratified)", unit="img"):
            # Always take the first caption for this image
            cap = caps_by_img[img_id][0]
            # Stable file naming: hash the caption
            cap_slug = slugify(cap, 48)
            cap_hash = hashlib.sha1(cap.encode("utf-8")).hexdigest()[:8]

            for k in range(n_per_caption):
                img_name = f"{img_id}_{cap_slug}_{cap_hash}_{k:02d}.png"
                abs_img_path = os.path.join(out_images_root, img_name)
                this_seed = None if seed is None else seed + count  # different seed per sample

                txt2img(
                    pipe,
                    prompt=cap,
                    out_path=abs_img_path,
                    seed=this_seed,
                    guidance_scale=guidance,
                    num_inference_steps=steps,
                    height=height,
                    width=width,
                )
                rel_path = os.path.relpath(abs_img_path, start=out_images_root)
                rec = {"image": rel_path, "text": cap}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                count += 1
                if limit is not None and count >= limit:
                    break
            if limit is not None and count >= limit:
                break

    print(f"Wrote {count} lines to {jsonl_path}")
    print(f"Images saved under: {out_images_root}")
    print("JSONL 'image' paths are relative to the generated images folder.")
    return jsonl_path

def parse_args():
    ap = argparse.ArgumentParser(description="Generate images from COCO captions (stratified by category) and write JSONL.")
    ap.add_argument("--ann-path", required=True, help="Path to captions_train2017.json or captions_val2017.json")
    ap.add_argument("--out-root", required=True, help="Output root where images/ and annotations/ are created")
    ap.add_argument("--model-id", default="runwayml/stable-diffusion-v1-5", help="Diffusers model id")

    # Fixed policy: first caption per image, so no --caption-policy arg.
    ap.add_argument("--total-captions", type=int, default=None,
                    help="Total number of images/captions to sample (stratified across categories). "
                         "If omitted, uses all eligible images.")

    ap.add_argument("--n-per-caption", type=int, default=1, help="How many images to generate per (image, first-caption)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--limit", type=int, default=None, help="Stop after N generated samples (debug)")
    ap.add_argument("--instances-path", type=str, default=None,
                    help="Path to instances_{train,val}2017.json. If not set, inferred from --ann-path.")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    build_jsonl_for_generation(
        ann_path=args.ann_path,
        out_root=args.out_root,
        model_id=args.model_id,
        total_captions=args.total_captions,
        seed=args.seed,
        steps=args.steps,
        guidance=args.guidance,
        height=args.height,
        width=args.width,
        n_per_caption=args.n_per_caption,
        limit=args.limit,
        instances_path=args.instances_path,
    )