# utils.py
import os
import json
import argparse
from pathlib import Path
from typing import Optional
import torch
import torchvision

DEFAULT_RANDOM_SEED = 2024

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_everything(seed: int = DEFAULT_RANDOM_SEED):
    import random
    import numpy as np
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class AnnotationWriter:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, "w", encoding="utf-8", buffering=1)  # line-buffered JSONL

    def write(self, record: dict):
        self.f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.f.flush()

    def close(self):
        try:
            self.f.close()
        except Exception:
            pass

def save_adv_image_and_annotation(
    image_org_0_255: torch.Tensor,
    delta: torch.Tensor,
    path_clean: str,
    output_dir: str,
    target_text: str,
    ann_writer: AnnotationWriter
):
    adv_image = image_org_0_255 + delta
    adv_image = torch.clamp(adv_image / 255.0, 0.0, 1.0)

    p_in = Path(path_clean).resolve()
    folder = p_in.parent.name
    name = p_in.name
    out_dir = Path(output_dir) / folder
    out_dir.mkdir(parents=True, exist_ok=True)

    ext = p_in.suffix.lower()
    save_name = p_in.stem + ".png" if ext == ".jpeg" else name
    save_path = out_dir / save_name

    torchvision.utils.save_image(adv_image[0], str(save_path))
    ann_writer.write({"image": str(save_path.resolve()), "text": target_text})

def fuse_features(
    img_features: torch.Tensor,
    txt_features: torch.Tensor,
    fusion_type: str,
    a_weight: float
) -> torch.Tensor:
    if fusion_type == "cat":
        fused = torch.cat((img_features, txt_features), dim=1)
        return fused / fused.norm(dim=1, keepdim=True)
    elif fusion_type == "add_weight":
        fused = a_weight * img_features + (1.0 - a_weight) * txt_features
        return fused / fused.norm(dim=1, keepdim=True)
    else:  # multiplication
        fused = img_features * txt_features
        return fused / fused.norm(dim=1, keepdim=True)

def hinge_loss_coa(
    cur_fused: torch.Tensor,
    cle_fused: torch.Tensor,
    tgt_fused: torch.Tensor,
    p_neg: float
) -> torch.Tensor:
    # pos similarity: cur vs clean; neg similarity: cur vs target
    embedding_sim1 = torch.mean(torch.sum(cur_fused * cle_fused, dim=1))  # pos
    embedding_sim2 = torch.mean(torch.sum(cur_fused * tgt_fused, dim=1))  # neg
    margin = 1 - p_neg
    loss = torch.mean(torch.relu(embedding_sim2 - p_neg * embedding_sim1 + margin))
    return loss

def save_adv_image_unit_range_and_annotation(
    adv_image_unit: torch.Tensor,   # [1,3,H,W], values in [0,1]
    path_clean: str,
    output_dir: str,
    target_text: str,
    ann_writer: AnnotationWriter
) -> None:
    """
    Save an adversarial image that is already in unit range [0,1] (e.g., after inverse-normalization
    from a model's input space) and write its JSON annotation.

    Behavior matches save_adv_image_and_annotation for paths/filenames:
    - Keeps original filename; if source was .jpeg, saves as .png, else preserves extension/name.
    - Creates the same folder structure under output_dir.
    """
    from pathlib import Path
    import torchvision

    adv_image_unit = torch.clamp(adv_image_unit, 0.0, 1.0)

    p_in = Path(path_clean).resolve()
    folder = p_in.parent.name
    name = p_in.name

    out_dir = Path(output_dir) / folder
    out_dir.mkdir(parents=True, exist_ok=True)

    ext = p_in.suffix.lower()
    save_name = p_in.stem + ".png" if ext == ".jpeg" else name
    save_path = out_dir / save_name

    torchvision.utils.save_image(adv_image_unit[0], str(save_path))
    ann_writer.write({"image": str(save_path.resolve()), "text": target_text})


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    # Extension points
    p.add_argument("--surrogate", default="clip", type=str, choices=["clip", "lavis"],
                   help="choose surrogate model family")
    p.add_argument("--attack", default="pgd", type=str,
                   choices=["pgd", "lavis_img2img"], help="choose attack method")

    # In build_arg_p()
    p.add_argument(
        "--multi_target_method",
        type=str,
        choices=["mean", "each"],
        default="mean",
        help="How to combine multiple target images per sample: "
            "'mean' uses the mean target embedding; 'each' sums per-target cosine terms."
    )

    # Core args
    p.add_argument("--num_samples", default=100, type=int)
    p.add_argument("--input_res", default=224, type=int)
    p.add_argument("--clip_encoder", default="ViT-B/32", type=str)
    p.add_argument("--alpha", default=1.0, type=float)
    p.add_argument("--p_neg", default=0.7, type=float)
    p.add_argument("--epsilon", default=8, type=int)
    p.add_argument("--pgd_steps", default=100, type=int)
    p.add_argument("--a_weight_cle", default=0.3, type=float, help="clean embedding fusion type add_weight, weight")
    p.add_argument("--a_weight_tgt", default=0.3, type=float, help="target embedding fusion type add_weight, weight")
    p.add_argument("--a_weight_cur", default=0.3, type=float, help="current embedding fusion type add_weight, weight")
    p.add_argument("--fusion_type", default="add_weight", type=str, choices=["cat", "add_weight", "multiplication"])
    p.add_argument("--speed_up", default=False, type=bool, help="speed up chain of attack")
    p.add_argument("--update_steps", default=1, type=int, help="caption update period if --speed_up")
    p.add_argument("--output", default="/COA/your_output_folder", type=str, help="folder for outputs")

    # Dataset / weights
    p.add_argument("--model_path", default="/COA/clip_prefix_model/conceptual_weights.pt", type=str,
        help="path of the image2text model")    
    p.add_argument("--prefix_length", default=10, type=int,
        help="image2text model prefix length")
    p.add_argument("--cle_file_path", default="/COA/img_caption/llava_textvqa.jsonl", type=str,
                   help="JSONL of clean: {image, text}")
    # NEW: single or multiple bases; required
    p.add_argument("--tgt_base", type=str, required=True,
                   help='Base path(s) that contain annotations/ and images/generated/. '
                        'Accepts a single path, a JSON list, or comma/space-separated paths. '
                        'Examples: --tgt_base /exp1 '
                        'or --tgt_base \'[\"/exp1\",\"/exp2\"]\' '
                        'or --tgt_base "/exp1, /exp2".')

    # CLIP text truncation control
    p.add_argument("--clip_context_length", default=77, type=int,
                   help="Text context length for CLIP (<= model max, usually 77). If None, uses model max.")
    # Live JSON annotations
    p.add_argument("--annotations_path", default=None, type=str,
                   help="JSONL to write; default: <output>/annotations.jsonl")

    # LAVIS surrogate specific
    p.add_argument("--lavis_model_name", default="blip_caption", type=str,
                   help="e.g., blip_caption, blip2_t5, blip2_opt, etc.")
    p.add_argument("--lavis_model_type", default="base_coco", type=str,
                   help="e.g., base_coco, pretrain, caption_coco_opt2.7b, etc.")

    # eval helpers
    p.add_argument("--eval_vlms", type=str, default=None,
                   help='List of VLMs for evaluation. JSON list or comma/space-separated.')
    p.add_argument("--eval_clip_encoders", type=str, default=None,
                   help='List of CLIP encoders for text-text cosine. JSON list or comma/space-separated.')

    p.add_argument("--method", choices=["coa", "attackvlm", "attackvlm_multi", "attackvlm_losses", "attackvlm_grow"], default="coa",
                   help="Which attack pipeline to run.")
    p.add_argument("--image_encoder",
        choices=["attackvlm_default", "B_16", "B_32", "L_32", "B_16_imagenet1k", "B_32_imagenet1k", "L_16_imagenet1k", "L_32_imagenet1k"],
        default="", help="ViT-PyTorch variants. \"\" when using CLIP's image encoder.")

    # AttackVLM-specific knobs (ignored by 'coa')
    p.add_argument("--batch_size", type=int, default=8, help="Batch size for --method attackvlm.")
    p.add_argument("--attackvlm_model_name", type=str, default="ViT-B-32",
                   help='open_clip model name, e.g., "ViT-B-32", "ViT-L-14".')
    p.add_argument("--attackvlm_pretrained", type=str, default="openai",
                   help='open_clip pretrained tag, e.g., "openai", "laion2b_s32b_b82k".')
    p.add_argument("--num_workers", type=int, default=4, help="DataLoader workers for --method attackvlm.")

    # p.add_argument("--num_workers", type=str, choices = ["global", "token_l2","token_cos","gram","coral","cka","ot","nce","combo"], default=None, help="what loss function for tokens")
    # "global" retains your original behavior (maximize cosine to target global embedding).
    # "token_l2" or "token_cos" align patch tokens directly (position-aligned).
    # "gram", "coral", "cka" impose higher-level semantic/relational alignment.
    # "ot" enables permutation-invariant set matching via Sinkhorn OT on tokens.
    # "nce" uses patch-level InfoNCE with aligned positives and in-sample negatives.
    # "combo" mixes multiple; set combo_weights="1.0,1.0,1.0,0.5,0,0,0" to weight gram, cka, ot, coral respectively.

    p.add_argument("--loss_mode", type=str, default="global",
                        choices=["global","token_l2","token_cos","gram","coral","cka","ot","nce","combo"])
    p.add_argument("--vit_layer", type=int, default=-2)
    p.add_argument("--drop_cls", action="store_true") # set to include; omit to keep False by default
    p.add_argument("--no-drop_cls", dest="drop_cls", action="store_false")
    p.set_defaults(drop_cls=True)
    p.add_argument("--token_downsample", type=int, default=0)
    p.add_argument("--nce_tau", type=float, default=0.07)
    p.add_argument("--ot_reg", type=float, default=0.1)
    p.add_argument("--ot_iters", type=int, default=50)
    p.add_argument("--ot_pos_w", type=float, default=0.0)
    p.add_argument("--combo_weights", type=str, default="1,1,1,0.5,0,0,0")
    p.add_argument("--do_asr", type=int, default=0)
    # Semantic alignment without strict spatial match:
    # --loss_mode combo --vit_layer -2 --drop_cls --token_downsample 196 --combo_weights "1,1,1,0.5,0,0,0" --ot_reg 0.1 --ot_iters 50
    # Faster runs: set --token_downsample 64–128.
    return p


# utils.py (add near the top)
import json

def parse_list_arg(val):
    """
    Accepts:
      - None -> []
      - JSON list string: '["BLIP","UniDiffuser"]'
      - Comma/space separated string: "BLIP, UniDiffuser" or "BLIP UniDiffuser"
      - Already a list/tuple -> list(val)
    Returns a list[str] with whitespace-trimmed, non-empty items.
    """
    if val is None:
        return []
    if isinstance(val, (list, tuple)):
        return [str(x).strip() for x in val if str(x).strip()]
    s = str(val).strip()
    if not s:
        return []
    # Try JSON first so users can pass a proper list
    if s.startswith("["):
        try:
            arr = json.loads(s)
            return [str(x).strip() for x in arr if str(x).strip()]
        except Exception:
            pass
    # Fallback: split on comma, then whitespace
    if "," in s:
        parts = [p.strip() for p in s.split(",")]
    else:
        parts = [p.strip() for p in s.split()]
    return [p for p in parts if p]


# seed_utils.py
import os
import random

def seed_everything(seed: int = 42, deterministic: bool = True, rank: int = 0):
    """
    Seed as many RNGs and backends as practical for reproducibility.
    Call this as early as possible (ideally at the very start of your program),
    before creating CUDA tensors, DataLoader workers, or initializing frameworks.

    Parameters
    - seed: base seed
    - deterministic: if True, enable deterministic (but possibly slower) kernels
    - rank: process rank (for DDP), used to derive distinct-but-reproducible worker seeds

    Returns
    - info: dict with helpers and detected versions
    """

    info = {"seed": seed, "rank": rank, "versions": {}, "torch": {}, "jax": {}}

    # 0) Environment knobs that influence determinism. Best set before imports.
    os.environ["PYTHONHASHSEED"] = str(seed)  # full effect when set at process start
    if deterministic:
        # cuBLAS determinism for PyTorch on CUDA (must be set before first CUDA call)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        # TensorFlow determinism toggles (safe if TF not installed)
        os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
        os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")
        # JAX/XLA deterministic reductions
        xla_flags = os.environ.get("XLA_FLAGS", "")
        if "--xla_gpu_deterministic_reductions" not in xla_flags:
            os.environ["XLA_FLAGS"] = (xla_flags + " --xla_gpu_deterministic_reductions").strip()

    # 1) Python's stdlib RNG
    random.seed(seed)

    # 2) NumPy
    try:
        import numpy as np
        np.random.seed(seed)
        info["versions"]["numpy"] = np.__version__
    except Exception:
        pass

    # 3) PyTorch (and friends)
    try:
        import torch
        info["versions"]["torch"] = torch.__version__

        # Base seeds (CPU and all CUDA devices)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Deteminism toggles
        if deterministic:
            try:
                torch.use_deterministic_algorithms(True)  # PyTorch 1.8+
            except Exception:
                pass
            try:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            except Exception:
                pass
            # Disable TF32 to avoid tiny nondeterminisms on Ampere+
            try:
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
            except Exception:
                pass

        # DataLoader helpers: per-worker seeding and generator for shuffles
        def _seed_worker(worker_id: int):
            # torch.initial_seed() already incorporates base seed, worker_id, and rank in DDP
            import numpy as _np
            wid_seed = (torch.initial_seed() + worker_id + rank * 1000) % 2**32
            random.seed(wid_seed)
            _np.random.seed(wid_seed)

        g = torch.Generator()
        g.manual_seed(seed + rank)  # rank-aware for DDP

        info["torch"]["worker_init_fn"] = _seed_worker
        info["torch"]["generator"] = g

    except Exception:
        pass

    # 4) Hugging Face Transformers (covers random, numpy, torch, and tf if present)
    try:
        from transformers.utils import set_seed as hf_set_seed
        hf_set_seed(seed)
        import transformers as _tfm
        info["versions"]["transformers"] = _tfm.__version__
    except Exception:
        pass

    # 5) TensorFlow / Keras
    try:
        import tensorflow as tf
        info["versions"]["tensorflow"] = tf.__version__
        # Newer TF
        try:
            tf.keras.utils.set_random_seed(seed)
        except Exception:
            tf.random.set_seed(seed)
        if deterministic:
            try:
                tf.config.experimental.enable_op_determinism()
            except Exception:
                pass
    except Exception:
        pass

    # 6) JAX/Flax (note: JAX uses explicit PRNG keys; you must thread the key yourself)
    try:
        import jax
        info["versions"]["jax"] = jax.__version__
        import jax.random as jrandom
        key = jrandom.PRNGKey(seed)
        info["jax"]["key"] = key  # return so you can manage splits explicitly
    except Exception:
        pass

    # 7) OpenAI CLIP and other PyTorch-based libs inherit the torch/np seeds above,
    # so nothing extra is required here.

    return info