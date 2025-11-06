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

from transformers import AutoProcessor, AutoModelForVision2Seq
from torchvision import transforms

from CLIP import clip

from evaluator.vlms import REGISTRY as VLM_REGISTRY


def generate_perturbed_images_list(
    image_org: torch.Tensor,
    delta: torch.Tensor,
    k: int,
    sigma: float,
    inverse_normalize,
    antithetic: bool = False,
) -> tuple[list, torch.Tensor]:
    """
    Generate k perturbed images around image_org + delta (single image case),
    and return a list of perturbed images.

    Returns:
        perturbed_images_list: list of k tensors, each shape (3,H,W)
        epsilons: (k, 3, H, W) tensor of eps used
    """
    if antithetic:
        assert k % 2 == 0, "k must be even with antithetic sampling"
        half_k = k // 2
        eps_half = torch.randn((half_k, *image_org.shape), device=image_org.device)
        eps = torch.cat([eps_half, -eps_half], dim=0)
    else:
        eps = torch.randn((k, *image_org.shape), device=image_org.device)

    scaled_eps = sigma * eps
    # scaled_eps = torch.clamp(scaled_eps, -10/255, 10/255)
    base = image_org + delta
    perturbed = base.unsqueeze(0) + scaled_eps
    perturbed = torch.clamp(inverse_normalize(perturbed), 0.0, 1)

    # convert to list
    perturbed_list = [tensor_to_pil(base.squeeze(0))] + [tensor_to_pil(perturbed[i].squeeze(0)) for i in range(k)]

    return perturbed_list, scaled_eps

from PIL import Image

def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    np_image = tensor.detach().cpu().numpy()
    np_image = np.transpose(np_image, (1, 2, 0))
    np_image = (np.clip(np_image, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(np_image)

def rgf_gradient_single_onesided(
    rewards_pos: torch.Tensor,
    reward_base: float,
    epsilons: torch.Tensor,
    mu: float,
    center_rewards: bool = False,
    normalize_rewards: bool = False,
) -> torch.Tensor:
    """
    One-sided RGF gradient estimate: uses f(x+μu) - f(x).

    Args:
        rewards_pos: (k,) tensor of rewards for x + mu * eps
        reward_base: scalar f(x)
        epsilons: (k, 3, H, W)
        mu: finite-difference step size
    """
    k = epsilons.shape[0]
    diff = rewards_pos - reward_base
    if center_rewards:
        diff = diff - diff.mean()
    if normalize_rewards:
        std = diff.std(unbiased=False)
        if std > 1e-8:
            diff = diff / std
    diff = diff.reshape(k, 1, 1, 1)
    grad_est = (diff * epsilons).sum(dim=0) / (mu * k)
    return grad_est

def nes_gradient_single(
    rewards: torch.Tensor,
    epsilons: torch.Tensor,
    sigma: float,
    center_rewards: bool = True,
    normalize_rewards: bool = False,
) -> torch.Tensor:
    """
    NES gradient estimate for single image delta.

    Args:
        rewards: (k,) tensor of rewards
        epsilons: (k, 3, H, W) tensor of eps used
    Returns:
        grad_est: (3, H, W) tensor
    """
    k = epsilons.shape[0]
    r = rewards.float()
    if center_rewards:
        r = r - r.mean()
    if normalize_rewards:
        std = r.std(unbiased=False)
        if std > 1e-8:
            r = r / std
    r = r.reshape(k, 1, 1, 1)
    grad_est = (r * epsilons).sum(dim=0) / (k * sigma)
    return grad_est

def generate_texts_from_images(
    vlm,
    images,
    prompt: str = "Describe the image in detail.",
    max_new_tokens: int = 60,
) -> List[str]:
    """
    Generate text for a batch of images using a preloaded VLM.

    Args:
        vlm: A preloaded VLM object with a `.generate(image, prompt, max_new_tokens)` method.
        images: List of images (PIL Images or compatible with vlm.generate).
        prompt: Text prompt for the VLM.
        max_new_tokens: Maximum number of tokens to generate per image.

    Returns:
        List of generated texts, in the same order as `images`.
    """
    # generated_texts = []
    # for image in images:
    #     out = vlm.generate(image, prompt=prompt, max_new_tokens=max_new_tokens)
    #     text = (out.text or "").strip()
    #     generated_texts.append(text)

    generated_texts = vlm.generate(images, prompt=prompt, max_new_tokens=max_new_tokens)
    return generated_texts


def new_attack(args, loader, dev, ann_writer):
    model, preprocess = clip.load("/home/user01/research/Chain_of_Attack/CLIP/weights/ViT-B-32.pt", device=dev)
    
    surrogate = build_surrogate(args.surrogate, args, dev)

    image_encoder_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=dev)
    image_encoder_std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=dev)
    
    model.eval()

    # vlm_ctor = VLM_REGISTRY["LLaVA-7B-Batch"]
    # vlm = vlm_ctor(device="cuda", dtype=None, **{})

    scaling_tensor = image_encoder_std.view(3, 1, 1).unsqueeze(0)  # (1,3,1,1)
    alpha   = args.alpha   / 255.0 / scaling_tensor
    epsilon = args.epsilon / 255.0 / scaling_tensor
    sigma = 8 / 255.0 / scaling_tensor

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

        print(clean_text, tgt_text)

        # image_org = to_clip_norm(clean_img_255)
        # image_tgt = to_clip_norm(tgt_img_255)

        image_org = preprocess(clean_img_255)
        image_tgt = preprocess(tgt_img_255)

        with torch.no_grad():
            tgt_cls = model.encode_image(image_tgt)
            tgt_cls = F.normalize(tgt_cls, dim=1)
            # tgt_cls = surrogate.encode_image_features_normalized(tgt_img_255, requires_grad=False)

            tgt_token = clip.tokenize(tgt_text).to(dev)
            tgt_txt_emb = model.encode_text(tgt_token)
            tgt_txt_emb = F.normalize(tgt_txt_emb, dim=1)

            # tgt_txt_tok = surrogate.tokenize_text(tgt_text)
            # tgt_txt_emb = surrogate.encode_text_features_normalized(tgt_txt_tok)

            # src_cls, src_layer_embds = model.encode_image(image_org)
            # src_cls = F.normalize(src_cls, dim=1)

        delta = torch.zeros_like(image_org, requires_grad=True)
        steps = getattr(args, "pgd_steps", 300)

        for j in range(steps):
            adv_image = image_org + delta
            # adv_cls = surrogate.encode_image_features_normalized(adv_image, requires_grad=True)
            adv_cls = model.encode_image(adv_image)
            adv_cls = F.normalize(adv_cls, dim=1)

            sim = F.cosine_similarity(adv_cls, tgt_cls, dim=1).mean()

            # VLM feedback
            perturbed_list, epsilons = generate_perturbed_images_list(image_org=image_org, delta=delta, k=1, sigma=sigma, inverse_normalize=inverse_normalize, antithetic=False)
            
            if j == 99:
                perturbed_list[1].save("output.png")
            with torch.no_grad():
                # vlm_feedback = generate_texts_from_images(vlm, perturbed_list)

                vlm_feedback = surrogate.generate_caption_from_batch(perturbed_list)
                # print(f"step -> {j}")
                # print(vlm_feedback)
                # print("*"*50)

                feedback_token = clip.tokenize(vlm_feedback).to(dev)
                feedback_emb = model.encode_text(feedback_token)
                feedback_emb = F.normalize(feedback_emb, dim=1)
                # feedback_token = surrogate.tokenize_text(vlm_feedback)
                # feedback_emb = surrogate.encode_text_features_normalized(feedback_token)

            rewards = (feedback_emb * tgt_txt_emb).sum(dim=1)
            # print("rewards:",rewards)

            grad_est = rgf_gradient_single_onesided(rewards[1:], rewards[0], epsilons.squeeze(1), 8)

            # grad_est = nes_gradient_single(rewards, epsilons.squeeze(1), 0.5)

            loss = sim
            loss.backward()

            with torch.no_grad():
                grad = delta.grad.detach() + grad_est
                delta.add_(alpha * torch.sign(grad))
                delta.clamp_(min=-epsilon, max=epsilon)
                delta.grad.zero_()

            max_delta = torch.max(torch.abs(delta)).item()
            mean_delta = torch.mean(torch.abs(delta)).item()
            # print(
            #     f"iter {i+1}/{total_batches} step:{j:3d}, "
            #     f"embedding similarity={sim.item():.5f}, "
            #     f"max delta={max_delta:.3f}, mean delta={mean_delta:.3f}"
            # )

        with torch.no_grad():
            adv_image_vis = torch.clamp(inverse_normalize(image_org + delta), 0.0, 1.0)
            # adv_image_vis = torch.clamp(clean_img_255 + delta, 0.0, 255.0)

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

    del model, surrogate

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