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

from CLIP import clip

import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F

def clip_feature_loss(
    clean_cls, clean_layer_embs,
    adv_cls, adv_layer_embds,
    tgt_cls, tgt_layer_embds,
    w_cls=None,      # (w_pos, w_neg)
    w_patch=None,    # (w_pos, w_neg)
    layers=None
):
    # defaults
    if w_cls is None:   w_cls = (1.0, 0.5)
    if w_patch is None: w_patch = (.0, 1.0)

    # choose layers
    if not layers:
        if isinstance(adv_layer_embds, dict):
            layers = sorted(adv_layer_embds.keys())
        else:
            layers = list(range(len(adv_layer_embds)))

    # Detach reference features (no grads into clean/target paths)
    tgt_cls_d   = tgt_cls.detach()
    clean_cls_d = clean_cls.detach()

    # 1) POS CLS: increase sim(adv, tgt)
    cls_pos = F.cosine_similarity(adv_cls, tgt_cls_d, dim=1).mean()

    # 3) NEG CLS: decrease sim(adv, clean)
    cls_neg = F.cosine_similarity(adv_cls, clean_cls_d, dim=1).mean()

    # 2) POS patches and 4) NEG patches
    patch_pos_terms, patch_neg_terms = [], []
    for i in layers:
        adv_p   = F.normalize(adv_layer_embds[i], dim=-1)
        tgt_p   = F.normalize(tgt_layer_embds[i].detach(), dim=-1)
        clean_p = F.normalize(clean_layer_embs[i].detach(), dim=-1)

        adv_p_f   = adv_p.reshape(-1, adv_p.size(-1))
        tgt_p_f   = tgt_p.reshape(-1, tgt_p.size(-1))
        clean_p_f = clean_p.reshape(-1, clean_p.size(-1))

        patch_pos_terms.append(F.cosine_similarity(adv_p_f, tgt_p_f, dim=1).mean())
        patch_neg_terms.append(F.cosine_similarity(adv_p_f, clean_p_f, dim=1).mean())

    if patch_pos_terms:
        patch_pos = torch.stack(patch_pos_terms).mean()
        patch_neg = torch.stack(patch_neg_terms).mean()
    else:
        device = adv_cls.device
        patch_pos = torch.tensor(0.0, device=device)
        patch_neg = torch.tensor(0.0, device=device)

    # Objective for gradient ASCENT on the adversarial input:
    # maximize:  w_pos * (pos sims) - w_neg * (neg sims)
    total = w_cls[0]   * cls_pos   + w_patch[0] * patch_pos - w_cls[1] * cls_neg   - w_patch[1] * patch_neg
    
    parts = {
        "cls_pos":  cls_pos.item(),
        "patch_pos": patch_pos.item(),
        "cls_neg":  cls_neg.item(),
        "patch_neg": patch_neg.item(),
    }
    return total, parts

def new_attack(args, loader, dev, ann_writer):
    model, preprocess = clip.load("/home/user01/research/Chain_of_Attack/CLIP/weights/ViT-B-32.pt", device=dev)

    image_encoder_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=dev)
    image_encoder_std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=dev)
    
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

    def to_clip_norm(x_255: torch.Tensor) -> torch.Tensor:
        x_01 = x_255 / 255.0
        return (x_01 - image_encoder_mean.view(1, 3, 1, 1)) / image_encoder_std.view(1, 3, 1, 1)

    for i, batch in enumerate(loader):
        clean_img_255, clean_text, tgt_img_255, tgt_text, clean_abs_paths = batch
        clean_img_255 = clean_img_255.to(dev).squeeze(1)  # (B,3,H,W), 0..255 float32
        tgt_img_255   = tgt_img_255.to(dev).squeeze(1)

        # image_org = to_clip_norm(clean_img_255)
        # image_tgt = to_clip_norm(tgt_img_255)
        print("*********", tgt_img_255.shape)

        image_org = preprocess(clean_img_255)
        image_tgt = preprocess(tgt_img_255)

        print("*********", image_tgt.shape)

        with torch.no_grad():
            clean_cls, clean_layer_embs = model.encode_image(image_org.clone(), output_layers=[11])
            clean_cls = F.normalize(clean_cls, dim=1)

            tgt_cls, tgt_layer_embds = model.encode_image(image_tgt, output_layers=[11])
            tgt_cls = F.normalize(tgt_cls, dim=1)

        delta = torch.zeros_like(image_org, requires_grad=True)
        steps = getattr(args, "pgd_steps", 300)

        for j in range(steps):
            adv_image = image_org + delta
            adv_cls, adv_layer_embds = model.encode_image(adv_image, output_layers=[11])
            adv_cls = F.normalize(adv_cls, dim=1)

            loss, parts = clip_feature_loss(clean_cls, clean_layer_embs, adv_cls, adv_layer_embds, tgt_cls, tgt_layer_embds, layers=[11])
            print(f"Step : {j}, Total: {loss.item():.4f}, POSCLS: {parts['cls_pos']:.4f}, POSPatch: {parts['patch_pos']:.4f}, NEGCLS: {parts['cls_neg']:.4f}, NEGPatch: {parts['patch_neg']:.4f}")
            loss.backward()

            with torch.no_grad():
                grad = delta.grad.detach()
                delta.add_(alpha * torch.sign(grad))
                delta.clamp_(min=-epsilon, max=epsilon)
                delta.grad.zero_()

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
        batch_size=args.batch_size,
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