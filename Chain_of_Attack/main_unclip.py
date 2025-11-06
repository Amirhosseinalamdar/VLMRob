# main.py
import os
from pathlib import Path
from tqdm import tqdm
import torch

from datasets import CleanTargetPairsDataset, CLE_DATA_ROOT  # unchanged import
from utils import seed_everything, get_device, build_arg_parser, AnnotationWriter, save_adv_image_and_annotation, fuse_features
from utils import save_adv_image_unit_range_and_annotation
from models import build_surrogate
from attacks import build_attack, PGDConfig
from eval_utils import report_metrics
from utils import parse_list_arg, seed_everything
from eval_utils import print_metrics_report
import torchvision
import torch.nn.functional as F
from trainers import *
import numpy as np

from torchvision import transforms

from CLIP import clip

# from open_clip.src import open_clip
from open_clip2.src import open_clip
# from diffusers.src.diffusers import DiffusionPipeline
from diffusers import DiffusionPipeline

def get_clip_model(device):
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='/home/user01/research/Chain_of_Attack/open_clip2/weights/CLIP-ViT-H-14-laion2B-s32B-b79K.pt')
    return clip_model.to(device), clip_preprocess

def get_clip_image_features(images, clip_model, clip_preprocess):
    """
    Extract CLIP features from a batch of images.

    Args:
        images: List of PIL images
        clip_model: CLIP model
        clip_preprocess: Preprocessing function for CLIP

    Returns:
        Tensor of shape [B, D]: one feature vector per image
    """
    # Preprocess all images
    inputs = torch.stack([clip_preprocess(image) for image in images]).to("cuda").half()  # [B, 3, H, W]

    image_features = clip_model.encode_image(inputs)  # [B, D]

    return image_features # shape: [B, D]

def decode_latents(pipe, latents):
    latents = 1 / pipe.vae.config.scaling_factor * latents
    image = pipe.vae.decode(latents, return_dict=False)[0]
    image = (image).clamp(0, 1)
    # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
    # image = image.cpu().permute(0, 2, 3, 1).float().detach().numpy()
    return image

def get_diffusione_model_preprocess():
    preprocess = transforms.Compose([
        transforms.Resize((768, 768)),
        # transforms.ToTensor(),  
    ])
    return preprocess

import torch

def generate_image(
    pipe,
    vae,
    cls_h,
    t=25,
    num_inference_steps=50,
    noise_level=0,
    prompt="",
    negative_prompt=None,
    do_classifier_free_guidance=True,
    guidance_scale=7.5,
    device="cuda"
):
    # --- Determine batch size ---
    if isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        prompt = [prompt]
        batch_size = 1

    if negative_prompt is None:
        negative_prompt = [""] * batch_size
    elif isinstance(negative_prompt, str):
        negative_prompt = [negative_prompt] * batch_size

    # --- Scheduler setup ---
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps
    timestep_idx = min(t, len(timesteps) - 1)
    timestep = timesteps[timestep_idx]

    # --- Add noise to VAE latents ---
    eps = torch.randn_like(vae)
    latents = pipe.scheduler.add_noise(vae, eps, timestep)
    if latents.ndim == 3:
        latents = latents.unsqueeze(0).repeat(batch_size, 1, 1, 1)

    # --- Encode text prompts ---
    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt=prompt,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=do_classifier_free_guidance,
        negative_prompt=negative_prompt,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        lora_scale=None,
    )

    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

    # --- Encode image embeddings (class embeddings) ---
    noise_level = torch.tensor([noise_level], device=device)
    image_embeds = pipe._encode_image(
        image=None,
        device=device,
        batch_size=batch_size,
        num_images_per_prompt=1,
        do_classifier_free_guidance=do_classifier_free_guidance,
        noise_level=noise_level,
        generator=None,
        image_embeds=cls_h,
    )

    extra_step_kwargs = pipe.prepare_extra_step_kwargs(None, None)

    # --- Denoising loop ---
    for i, t in enumerate(pipe.progress_bar(timesteps[timestep_idx:])):
        latent_model_input = (
            torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        )
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        # predict noise
        noise_pred = pipe.unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeds,
            class_labels=image_embeds,
            cross_attention_kwargs=None,
            return_dict=False,
        )[0]

        # classifier-free guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

        # step the scheduler
        latents = pipe.scheduler.step(
            noise_pred, t, latents, **extra_step_kwargs, return_dict=False
        )[0]

    # --- Decode to image(s) ---
    image = decode_latents(pipe, latents)
    return image
        
def get_vae_features(images, pipe, preprocess):
    tensors = []
    for img in images:
        tensors.append(preprocess(img).half().to('cuda'))

    with torch.no_grad():
        z_t = pipe.vae.encode(torch.stack(tensors)).latent_dist.sample() * pipe.vae.config.scaling_factor
    
    return z_t

def get_diffusion_model(device):
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1-unclip",
        torch_dtype=torch.float16,
        # cache_dir="/home/user01/.cache/huggingface/hub/models--stabilityai--stable-diffusion-2-1-unclip"
    ).to(device)
    return pipe

def new_attack(args, loader, dev, ann_writer):
    print("loading clip ...")
    model, preprocess = get_clip_model(dev)

    print("loading unclip ...")
    unclip_pipe = get_diffusion_model(dev)
    unclip_preprocess = get_diffusione_model_preprocess()

    image_encoder_mean = torch.tensor((0.48145466, 0.4578275, 0.40821073), device=dev)
    image_encoder_std  = torch.tensor((0.26862954, 0.26130258, 0.27577711), device=dev)
    
    model.eval()

    scaling_tensor = image_encoder_std.view(3, 1, 1).unsqueeze(0)  # (1,3,1,1)
    alpha   = args.alpha   / 255.0 / scaling_tensor
    epsilon = args.epsilon / 255.0 / scaling_tensor

    inverse_normalize = torchvision.transforms.Normalize(
        mean=(-image_encoder_mean / image_encoder_std).tolist(),
        std=(1.0 / image_encoder_std).tolist(),
    )

    out_root = Path(args.output).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    total_batches = (len(loader.dataset) + args.batch_size - 1) // args.batch_size

    def to_zero_one(x_255: torch.Tensor) -> torch.Tensor:
        x_01 = x_255 / 255.0
        return (x_01 - image_encoder_mean.view(1, 3, 1, 1)) / image_encoder_std.view(1, 3, 1, 1)

    for i, batch in enumerate(loader):
        clean_img_255, clean_text, tgt_img_255, tgt_text, clean_abs_paths = batch
        clean_img_255 = clean_img_255.to(dev).squeeze(1)  # (B,3,H,W), 0..255 float32
        tgt_img_255   = tgt_img_255.to(dev).squeeze(1)

        print(clean_text, "\n", tgt_text)

        image_org = clean_img_255 / 255#preprocess(clean_img_255)
        image_tgt = tgt_img_255 / 255#preprocess(tgt_img_255)

        vae = get_vae_features(image_org, unclip_pipe, unclip_preprocess)
        print("-----------------------")

        delta = torch.zeros_like(image_org, requires_grad=True)
        steps = args.pgd_steps

        for j in range(steps):
            adv_image = image_org + delta

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                adv_embds = get_clip_image_features(adv_image, model, preprocess)

                image_gen = generate_image(unclip_pipe, vae, adv_embds.half(), prompt=tgt_text[0], negative_prompt=clean_text[0], device=dev)

            image_gen = image_gen.float()
            image_tgt_tensor = unclip_preprocess(image_tgt).half().to(dev)
            image_tgt_tensor = torch.nn.functional.interpolate(image_tgt_tensor, size=image_gen.shape[-2:], mode='bilinear', align_corners=False)

            loss = torch.nn.functional.mse_loss(image_gen, image_tgt_tensor)

            loss.backward()

            with torch.no_grad():
                grad = delta.grad.detach()
                delta.add_(alpha * torch.sign(grad))
                delta.clamp_(min=-epsilon, max=epsilon)
                delta.grad.zero_()

            del adv_image, adv_embds, image_gen, image_tgt_tensor, loss, grad
            torch.cuda.empty_cache()

            max_delta = torch.max(torch.abs(delta)).item()
            mean_delta = torch.mean(torch.abs(delta)).item()
            # print(
            #     f"iter {i+1}/{total_batches} step:{j:3d}, "
            #     f"embedding similarity={embedding_sim.item():.5f}, "
            #     f"max delta={max_delta:.3f}, mean delta={mean_delta:.3f}"
            # )

        with torch.no_grad():
            adv_image_vis = torch.clamp(inverse_normalize(image_org + delta), 0.0, 1.0)

        # Save under class folders derived from CLE_DATA_ROOT and write annotation
        for b in range(adv_image_vis.size(0)):
            clean_abs = Path(clean_abs_paths[b])

            try:
                rel = clean_abs.resolve().relative_to(CLE_DATA_ROOT)
                parts = rel.parts
                if len(parts) >= 2 and parts[0].startswith("n") and parts[0][1:].isdigit() is False:
                    class_name = parts[0]
                elif len(parts) >= 2:
                    class_name = parts[0]
                else:
                    class_name = clean_abs.parent.name
            except Exception:
                class_name = clean_abs.parent.name

            class_dir = (out_root / class_name)
            class_dir.mkdir(parents=True, exist_ok=True)

            stem = clean_abs.stem
            out_path = (class_dir / f"{stem}.png").resolve()
            torchvision.utils.save_image(adv_image_vis[b], str(out_path))

            rec = {"image": str(out_path), "text": str(tgt_text[b])}
            if ann_writer is not None:
                ann_writer.write(rec)
            else:
                annotations.append(rec)

def main(args):
    seed_everything(1234)
    dev = get_device()
    os.makedirs(args.output, exist_ok=True)
    ann_path = args.annotations_path or os.path.join(args.output, "annotations.jsonl")
    ann_writer = AnnotationWriter(ann_path)
    dataset = CleanTargetPairsDataset(
        cle_file_path=args.cle_file_path,
        input_res=args.input_res,
        tgt_base=args.tgt_base,   # single path or a list-like string / JSON list
        num_samples=args.num_samples,
        stack_targets=True,     # optional: get [K,C,H,W] instead of a list
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,#args.batch_size,
        shuffle=False,
        num_workers=getattr(args, "num_workers", 8),
        drop_last=False,
        pin_memory=(dev == "cuda"),
    )
    new_attack(args, loader, dev, ann_writer)

    
    # Evaluation remains exactly as before (only for 'coa')
    ann_path = args.annotations_path or os.path.join(args.output, "annotations.jsonl")
    vlms_to_run = parse_list_arg(getattr(args, "eval_vlms", None))
    clip_encs   = parse_list_arg(getattr(args, "eval_clip_encoders", None))
    report = report_metrics(
        annotations_jsonl=ann_path,
        out_dir=args.output,
        vlms=vlms_to_run,
        clip_encoders=clip_encs,
        device=getattr(args, "device", None),
        dtype=None,
        prompt="Describe the image in detail.",
        max_new_tokens=80,
        limit=None,
        clip_batch_size=32,
        vlm_kwargs={},
    )
    print(f"[eval] Saved consolidated metrics to: {os.path.join(args.output, 'metrics_report.json')}")
    print_metrics_report(report, enc_order=clip_encs)


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    main(args)