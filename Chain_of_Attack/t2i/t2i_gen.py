import argparse, json, os, hashlib, re
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional
from collections import defaultdict
import random
import argparse
import hashlib
import json
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from tqdm import tqdm

import nltk
from nltk.corpus import wordnet as wn  # type: ignore
_WORDNET_OK = True
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



# Conservative close-concept replacements aligned with common COCO categories
COCO_CLOSE_CONCEPTS: Dict[str, List[str]] = {
    "person": ["man", "woman", "boy", "girl"],
    "man": ["person", "gentleman"],
    "woman": ["person", "lady"],
    "boy": ["child", "kid"],
    "girl": ["child", "kid"],

    "dog": ["puppy", "canine", "shepherd"],
    "cat": ["kitten", "tabby", "feline"],
    "bird": ["sparrow", "parrot", "songbird"],
    "horse": ["pony", "stallion", "mare"],
    "cow": ["calf", "heifer"],
    "sheep": ["lamb"],
    "elephant": ["calf elephant"],
    "bear": ["grizzly", "brown bear"],
    "zebra": ["foal zebra"],
    "giraffe": ["young giraffe"],

    "car": ["sedan", "sports car", "hatchback"],
    "bus": ["coach", "school bus"],
    "truck": ["pickup truck", "lorry"],
    "motorcycle": ["motorbike", "scooter"],
    "bicycle": ["road bike", "mountain bike"],
    "boat": ["sailboat", "speedboat"],
    "train": ["subway train", "commuter train"],
    "airplane": ["jet", "airliner"],

    "chair": ["armchair", "stool"],
    "couch": ["sofa", "loveseat"],
    "bed": ["bunk bed", "queen bed"],
    "table": ["dining table", "coffee table"],
    "tv": ["television", "flat-screen tv"],
    "laptop": ["notebook computer"],
    "cell phone": ["smartphone", "mobile phone"],
    "remote": ["remote control"],
    "keyboard": ["computer keyboard"],
    "mouse": ["computer mouse"],
    "toilet": ["restroom toilet"],
    "sink": ["bathroom sink"],
    "refrigerator": ["fridge"],

    "potted plant": ["houseplant", "succulent"],
    "vase": ["flower vase"],
    "bottle": ["glass bottle"],
    "cup": ["mug"],
    "bowl": ["dish"],
    "knife": ["kitchen knife"],
    "spoon": ["teaspoon"],
    "fork": ["dining fork"],
    "pizza": ["slice of pizza"],
    "cake": ["cupcake", "birthday cake"],
    "sandwich": ["sub sandwich"],
    "hot dog": ["sausage"],
    "donut": ["doughnut"],
    "banana": ["ripe banana"],
    "apple": ["green apple", "red apple"],
    "orange": ["tangerine"],
    "broccoli": ["broccolini"],
    "carrot": ["baby carrot"],
    "wine glass": ["champagne flute"],
    "knife": ["chef's knife"],

    "handbag": ["purse", "bag"],
    "backpack": ["rucksack"],
    "umbrella": ["parasol"],
    "tie": ["necktie"],
    "suitcase": ["luggage"],
    "skateboard": ["longboard"],
    "surfboard": ["shortboard"],
    "tennis racket": ["racquet"],
    "baseball bat": ["wooden bat"],
    "baseball glove": ["mitt"],
    "kite": ["stunt kite"],
    "skis": ["downhill skis"],
    "snowboard": ["freestyle snowboard"],

    "traffic light": ["stoplight"],
    "fire hydrant": ["hydrant"],
    "stop sign": ["yield sign"],
    "bench": ["park bench"],
}

PARAPHRASE_TEMPLATES = [
    "a photo of {cap}",
    "an image showing {cap}",
    "a detailed photo of {cap}",
    "a realistic photograph of {cap}",
    "a high-quality image of {cap}",
    "{cap}, high quality",
    "a well-composed photo of {cap}",
]

STYLE_ADDONS = [
    "natural lighting",
    "soft lighting",
    "high detail",
    "sharp focus",
    "shallow depth of field",
    "wide shot",
    "close-up shot",
    "candid shot",
]

def maybe_inflect(original: str, replacement: str) -> str:
    # naive plural handling for English nouns (keeps behavior predictable without extra deps)
    if original.endswith("s") and not replacement.endswith("s"):
        if replacement.endswith("y"):
            return replacement[:-1] + "ies"
        if replacement.endswith(("ch", "sh", "x", "z")):
            return replacement + "es"
        return replacement + "s"
    return replacement

def wordnet_synonym(word: str) -> Optional[str]:
    if not _WORDNET_OK:
        return None
    try:
        synsets = wn.synsets(word)
        if not synsets:
            return None
        lemmas = set()
        for s in synsets:
            for l in s.lemmas():
                lemma = l.name().replace("_", " ")
                if lemma.lower() != word.lower():
                    lemmas.add(lemma)
        if not lemmas:
            return None
        return random.choice(sorted(lemmas))
    except Exception:
        return None

def conservative_synonym(word: str) -> Optional[str]:
    # prefer close-concept lexicon; else try wordnet; else None
    if word.lower() in COCO_CLOSE_CONCEPTS:
        return random.choice(COCO_CLOSE_CONCEPTS[word.lower()])
    wn_syn = wordnet_synonym(word)
    return wn_syn

def paraphrase_caption(cap: str, rng: random.Random) -> str:
    # pick a template and optionally append a mild style addon
    tpl = rng.choice(PARAPHRASE_TEMPLATES)
    base = tpl.format(cap=cap)
    if rng.random() < 0.5:
        addon = rng.choice(STYLE_ADDONS)
        return f"{base}, {addon}"
    return base

def close_concept_variant(cap: str, rng: random.Random) -> Optional[str]:
    """
    Replace at most one token with a very close concept to keep semantics nearly identical.
    Priority for tokens that exist in COCO_CLOSE_CONCEPTS.
    """
    tokens = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[^\w\s]", cap)
    candidate_idxs = [i for i,t in enumerate(tokens) if re.fullmatch(r"[A-Za-z][A-Za-z'-]*", t)]
    rng.shuffle(candidate_idxs)

    for idx in candidate_idxs:
        tok = tokens[idx]
        repl = None
        key = tok.lower()
        if key in COCO_CLOSE_CONCEPTS:
            repl = rng.choice(COCO_CLOSE_CONCEPTS[key])
        else:
            # very conservative: only attempt noun-ish words that are alphabetic and >2 chars
            if len(tok) > 2:
                syn = conservative_synonym(key)
                if syn:
                    repl = syn

        if repl and repl.lower() != key:
            repl = maybe_inflect(tok, repl)
            tokens[idx] = repl
            out = "".join(
                (t if re.fullmatch(r"[A-Za-z]+(?:'[A-Za-z]+)?", t) else t)
                if i == 0 or t in [",", ".", "!", "?", ":", ";"] else t
                for i, t in enumerate(tokens)
            )
            # fix spacing around punctuation
            out = re.sub(r"\s+([,.:;!?])", r"\1", out)
            return out

    return None  # no safe replacement found

def make_prompt_variants(
    cap: str,
    n_paraphrase: int,
    n_close: int,
    rng: random.Random,
) -> Dict[str, List[str]]:
    """
    Returns:
      {
        "base": [cap],
        "paraphrase": [p1, p2, ...],
        "close": [c1, c2, ...]
      }
    Only includes a key if n_* > 0 (except base).
    """
    out: Dict[str, List[str]] = {"base": [cap]}

    if n_paraphrase > 0:
        seen = set()
        paraphrases = []
        for _ in range(n_paraphrase * 3):  # allowance for dedup
            p = paraphrase_caption(cap, rng)
            if p not in seen and p != cap:
                seen.add(p)
                paraphrases.append(p)
            if len(paraphrases) >= n_paraphrase:
                break
        if paraphrases:
            out["paraphrase"] = paraphrases

    if n_close > 0:
        seen = set()
        closes = []
        for _ in range(n_close * 6):
            c = close_concept_variant(cap, rng)
            if c and c not in seen and c != cap:
                seen.add(c)
                closes.append(c)
            if len(closes) >= n_close:
                break
        if closes:
            out["close"] = closes

    return out

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
    # NEW: variation controls (all optional; defaults preserve existing behavior)
    variants: str = "base",                 # comma-separated: base,paraphrase,close
    n_paraphrase: int = 3,                  # number of paraphrases per caption (if enabled)
    n_close: int = 2,                       # number of close-concept prompts per caption (if enabled)
    variant_seed: int = 777,                # RNG seed for text variants
) -> str:
    """
    Generates images from COCO captions. Now supports prompt variants:
      - base: original caption (exactly your current behavior and file layout)
      - paraphrase: neutral rewordings, same semantics
      - close: very close concepts (tiny noun/attribute shifts)
    Each variant writes a separate JSONL and images folder.
    Return value is the base JSONL path (for compatibility).
    """
    caps_by_img, _ = load_coco_captions(ann_path)
    if not instances_path:
        instances_path = resolve_instances_path(ann_path)

    selected_img_ids = stratified_image_ids(
        caps_by_img=caps_by_img,
        instances_path=instances_path,
        total_captions=(total_captions if total_captions is not None else 0),
        seed=1234,
    )

    out_images_root_base = os.path.join(out_root, "images", "generated")
    out_ann_dir = os.path.join(out_root, "annotations")
    Path(out_ann_dir).mkdir(parents=True, exist_ok=True)

    ann_stem = Path(ann_path).stem
    jsonl_base = os.path.join(out_ann_dir, f"mscoco_t2i_{ann_stem}_stratified.jsonl")

    # Prepare pipeline once
    pipe = build_sd_pipeline(model_id=model_id)

    # Parse variants
    variant_keys = [v.strip().lower() for v in variants.split(",") if v.strip()]
    if not variant_keys:
        variant_keys = ["base"]

    # RNG for text variants
    rng = random.Random(variant_seed if variant_seed is not None else 777)

    produced_jsonls: Dict[str, str] = {}

    for vkey in variant_keys:
        is_base = (vkey == "base")

        # For base variant, use original structure
        if is_base:
            out_images_root = out_images_root_base
            jsonl_path = jsonl_base
            Path(out_images_root).mkdir(parents=True, exist_ok=True)
            
            print(f"[{vkey}] Writing to: {jsonl_path}")
            count = 0
            vkey_offset = int(hashlib.sha1(vkey.encode("utf-8")).hexdigest(), 16) % 10_000_000
            
            with open(jsonl_path, "w", encoding="utf-8") as f:
                for img_id in tqdm(selected_img_ids, desc=f"Generating ({vkey})", unit="img"):
                    cap = caps_by_img[img_id][0]
                    cap_slug = slugify(cap, 48)
                    cap_hash = hashlib.sha1(cap.encode("utf-8")).hexdigest()[:8]
                    
                    pv_list = [(cap, 0)]  # Base always uses original caption
                    
                    for prompt_text, vi in pv_list:
                        prompt_slug = slugify(prompt_text, 48)
                        prompt_hash = hashlib.sha1(prompt_text.encode("utf-8")).hexdigest()[:8]
                        
                        for k in range(n_per_caption):
                            img_name = f"{img_id}_{cap_slug}_{cap_hash}_v{vi:02d}_{prompt_slug}_{prompt_hash}_{k:02d}.png"
                            abs_img_path = os.path.join(out_images_root, img_name)
                            
                            this_seed = None
                            if seed is not None:
                                base_count = count
                                this_seed = seed + base_count + vkey_offset
                            
                            txt2img(
                                pipe,
                                prompt=prompt_text,
                                out_path=abs_img_path,
                                seed=this_seed,
                                guidance_scale=guidance,
                                num_inference_steps=steps,
                                height=height,
                                width=width,
                            )
                            
                            rel_path = os.path.relpath(abs_img_path, start=out_images_root)
                            rec = {"image": rel_path, "text": prompt_text}
                            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                            count += 1
                            
                            if limit is not None and count >= limit:
                                break
                        
                        if limit is not None and count >= limit:
                            break
                    if limit is not None and count >= limit:
                        break
            
            print(f"[{vkey}] Wrote {count} lines to {jsonl_path}")
            print(f"[{vkey}] Images under: {out_images_root}")
            produced_jsonls[vkey] = jsonl_path
        
        else:
            # For non-base variants (paraphrase/close), create separate folder for each variation
            variants_map = {}
            
            # First, generate all variants for all captions
            for img_id in selected_img_ids:
                cap = caps_by_img[img_id][0]
                if vkey == "paraphrase":
                    variants_map[img_id] = make_prompt_variants(cap, n_paraphrase=n_paraphrase, n_close=0, rng=rng)
                elif vkey == "close":
                    variants_map[img_id] = make_prompt_variants(cap, n_paraphrase=0, n_close=n_close, rng=rng)
            
            # Create folders and generate for each variation index
            for vi in range(max(n_paraphrase, n_close)):
                if vkey == "paraphrase" and vi <= 5:
                    continue
                # Create variant-specific output folders
                variant_out_root = os.path.join(out_root, f"{vkey}_v{vi:02d}")
                variant_images_root = os.path.join(variant_out_root, "images", "generated")
                variant_ann_dir = os.path.join(variant_out_root, "annotations")
                Path(variant_images_root).mkdir(parents=True, exist_ok=True)
                Path(variant_ann_dir).mkdir(parents=True, exist_ok=True)
                
                # Create variant-specific JSONL file
                variant_jsonl_path = os.path.join(variant_ann_dir, f"mscoco_t2i_{ann_stem}_stratified.jsonl")
                
                print(f"[{vkey}_v{vi:02d}] Writing to: {variant_jsonl_path}")
                count = 0
                vkey_offset = int(hashlib.sha1(f"{vkey}_v{vi:02d}".encode("utf-8")).hexdigest(), 16) % 10_000_000
                
                with open(variant_jsonl_path, "w", encoding="utf-8") as variant_f:
                    for img_id in tqdm(selected_img_ids, desc=f"Generating ({vkey}_v{vi:02d})", unit="img"):
                        cap = caps_by_img[img_id][0]
                        cap_slug = slugify(cap, 48)
                        cap_hash = hashlib.sha1(cap.encode("utf-8")).hexdigest()[:8]
                        
                        # Get the specific variant for this image
                        if img_id in variants_map:
                            variant_list = variants_map[img_id].get(vkey, [])
                            if vi < len(variant_list):
                                prompt_text = variant_list[vi]
                                
                                prompt_slug = slugify(prompt_text, 48)
                                prompt_hash = hashlib.sha1(prompt_text.encode("utf-8")).hexdigest()[:8]
                                
                                for k in range(n_per_caption):
                                    img_name = f"{img_id}_{cap_slug}_{cap_hash}_v{vi:02d}_{prompt_slug}_{prompt_hash}_{k:02d}.png"
                                    abs_img_path = os.path.join(variant_images_root, img_name)
                                    
                                    this_seed = None
                                    if seed is not None:
                                        base_count = count
                                        this_seed = seed + base_count + vkey_offset
                                    
                                    txt2img(
                                        pipe,
                                        prompt=prompt_text,
                                        out_path=abs_img_path,
                                        seed=this_seed,
                                        guidance_scale=guidance,
                                        num_inference_steps=steps,
                                        height=height,
                                        width=width,
                                    )
                                    
                                    rel_path = os.path.relpath(abs_img_path, start=variant_images_root)
                                    rec = {"image": rel_path, "text": prompt_text}
                                    variant_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                                    count += 1
                                    
                                    if limit is not None and count >= limit:
                                        break
                                
                                if limit is not None and count >= limit:
                                    break
                    
                    if limit is not None and count >= limit:
                        break
                
                print(f"[{vkey}_v{vi:02d}] Wrote {count} lines to {variant_jsonl_path}")
                print(f"[{vkey}_v{vi:02d}] Images under: {variant_images_root}")
                produced_jsonls[f"{vkey}_v{vi:02d}"] = variant_jsonl_path

    # For compatibility, return the base JSONL path if produced; else return the first produced
    if "base" in produced_jsonls:
        return produced_jsonls["base"]
    # fall back to any variant's path if base not requested
    return next(iter(produced_jsonls.values()))


def parse_args():
    ap = argparse.ArgumentParser(description="Generate images from COCO captions (stratified by category) and write JSONL. Supports prompt variants.")
    ap.add_argument("--ann-path", required=True, help="Path to captions_train2017.json or captions_val2017.json")
    ap.add_argument("--out-root", required=True, help="Output root where images/ and annotations/ are created")
    ap.add_argument("--model-id", default="runwayml/stable-diffusion-v1-5", help="Diffusers model id")
    ap.add_argument("--total-captions", type=int, default=None,
                    help="Total number of images/captions to sample (stratified across categories). If omitted, uses all eligible images.")
    ap.add_argument("--n-per-caption", type=int, default=1, help="How many images to generate per (image, first-caption)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--limit", type=int, default=None, help="Stop after N generated samples (debug)")
    ap.add_argument("--instances-path", type=str, default=None, help="Path to instances_{train,val}2017.json. If not set, inferred from --ann-path.")

    # NEW: prompt variants
    ap.add_argument("--variants", type=str, default="base",
                    help="Comma-separated list of variants to generate: base,paraphrase,close")
    ap.add_argument("--n-paraphrase", type=int, default=8,
                    help="Number of paraphrases per caption when 'paraphrase' is enabled")
    ap.add_argument("--n-close", type=int, default=3,
                    help="Number of close-concept prompts per caption when 'close' is enabled")
    ap.add_argument("--variant-seed", type=int, default=707, help="RNG seed for text variation generation")

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
        variants=args.variants,
        n_paraphrase=args.n_paraphrase,
        n_close=args.n_close,
        variant_seed=args.variant_seed,
    )