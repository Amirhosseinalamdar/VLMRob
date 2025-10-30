# main.py
#!/usr/bin/env python3
import os, argparse, random, csv
import numpy as np
import torch
import torchvision
from tqdm import tqdm
from encoders import build_text_encoder, build_prompted_text_encoder  # <-- add prompted encoder
from dataset import ImageNetCocoDataset
from models import build_image_models
from attack import AttackConfig, pgd_targeted_sum_mse, pgd_adv_target

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    p = argparse.ArgumentParser()
    # Repro
    p.add_argument("--seed", type=int, default=42)
    # Data
    p.add_argument("--imagenet_val_root", type=str, required=True, help="ImageNet-1k val root (ImageFolder)")
    p.add_argument("--coco_captions_json", type=str, required=True, help="COCO annotations/captions_train2017.json")
    p.add_argument("--num_samples", type=int, default=None, help="Total number of images to sample (stratified). Default: all")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    # Text encoder
    p.add_argument("--text_encoder", type=str, default="openai:ViT-B/32",
                   help="e.g., openai:ViT-B/32 or openclip:ViT-B-32,openai")
    # Soft-prompt options
    p.add_argument("--num_tokens", type=int, default=0,
                   help="Number of learnable soft-prompt tokens K; 0 disables prompts.")
    p.add_argument("--prompt_init_text", type=str, default=None,
                   help='Optional init text for prompts (e.g., "a photo of"); None=random init.')
    # Image models (comma-separated list)
    p.add_argument("--models", type=str, default="llava_proj,clip-openai:ViT-B/32",
                   help="llava_proj, clip-openai:ViT-B/32, openclip-enc:ViT-B-32,openai")
    # LLaVA head checkpoint (required if llava_proj in models)
    p.add_argument("--llava_ckpt", type=str, default="checkpoints/mlp_llava_imagenet1k_lr0.0005_wd0.0005_ep15_warm0.01.pth")
    # Attack
    p.add_argument("--epsilon", type=float, default=8.0, help="L_inf in pixels")
    p.add_argument("--alpha", type=float, default=1.0, help="step size in pixels")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--random_start", action="store_true")
    # Output
    p.add_argument("--out_dir", type=str, default="./adv_out")
    p.add_argument("--save_format", type=str, default="png", choices=["png", "jpg"], help="Image format for saving outputs")
    p.add_argument("--jpg_quality", type=int, default=95, help="Quality for JPEG saving (if save_format=jpg)")
    return p.parse_args()

def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build text encoder (target space). If using prompts, only openai:* is supported by this code.
    if args.num_tokens > 0:
        if not args.text_encoder.startswith("openai:"):
            raise ValueError("num_tokens > 0 is currently supported only for OpenAI CLIP (text_encoder starts with 'openai:').")
        if args.num_tokens >= 76:
            raise ValueError(f"--num_tokens={args.num_tokens} leaves no room for caption content (max 75). Use K <= 75.")
        txt_enc = build_prompted_text_encoder(
            args.text_encoder, device=device, K=args.num_tokens, init_text=args.prompt_init_text
        )
        txt_enc.eval()  # using prompts for inference only in this script
    else:
        txt_enc = build_text_encoder(args.text_encoder, device=device)

    # Dataset and loader (no GPU work in dataset). Uses stratified selection if num_samples is set.
    ds = ImageNetCocoDataset(
        image_root=args.imagenet_val_root,
        coco_captions_json=args.coco_captions_json,
        seed=args.seed,
        num_samples=args.num_samples,
    )
    dl = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda")
    )

    # Frozen image encoders
    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    image_models = build_image_models(model_names, device, llava_ckpt=args.llava_ckpt)

    # Sanity: all image model output dims must match text encoder dim
    out_dims = {spec.out_dim for spec in image_models}
    if len(out_dims) != 1 or next(iter(out_dims)) != txt_enc.embed_dim:
        raise RuntimeError(f"Image model dims {out_dims} must match text encoder dim {txt_enc.embed_dim}.")

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "results.csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["image_id", "text"])
        atk_cfg = AttackConfig(epsilon=args.epsilon, alpha=args.alpha, steps=args.steps, random_start=args.random_start)
        total_batches = len(dl) if hasattr(dl, "__len__") else None

        for bidx, batch in tqdm(enumerate(dl), total=total_batches, desc="Attacking batches", unit="batch"):
            imgs_255, basenames, captions = batch  # captions: list[str], basenames: list[str]

            # Compute target text embeddings. No training of prompts here, so avoid building a graph.
            with torch.no_grad():
                txt_emb = txt_enc.encode(list(captions))  # [B, D], on device

            imgs_255 = imgs_255.to(device, non_blocking=True)    # [B,3,224,224] in [0,255]
            adv_255 = pgd_adv_target(image_models, imgs_255, txt_emb, atk_cfg)
            adv_01 = adv_255 / 255.0

            bsz = adv_01.size(0)
            for i in range(bsz):
                base, _ = os.path.splitext(basenames[i])  # base = filename without extension
                # Save image to disk with an extension
                ext = ".png" if args.save_format == "png" else ".jpg"
                out_filename = base + ext
                out_path = os.path.join(args.out_dir, out_filename)
                if args.save_format == "png":
                    torchvision.utils.save_image(adv_01[i], out_path)
                else:
                    torchvision.utils.save_image(adv_01[i], out_path, quality=args.jpg_quality)
                # Write CSV row WITHOUT the image extension
                writer.writerow([base, captions[i]])

    print(f"Saved perturbed images to: {args.out_dir}")
    print(f"Saved CSV: {csv_path}")
    print("Done.")

if __name__ == "__main__":
    main()