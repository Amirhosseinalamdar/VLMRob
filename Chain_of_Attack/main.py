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
from ViT_PyTorch.pytorch_pretrained_vit import ViT

    
def train_coa(args, ann_writer, dataset, dev):
    try:
        # Surrogate model (default: CLIP)
        surrogate = build_surrogate(args.surrogate, args, dev)


        # Context length handling (identical semantics)
        sur_max_ctx = surrogate.max_context_length
        clip_ctx = sur_max_ctx if args.clip_context_length is None else min(args.clip_context_length, sur_max_ctx)
        if args.clip_context_length is not None and args.clip_context_length > sur_max_ctx:
            print(f"[warn] --clip_context_length={args.clip_context_length} exceeds model limit {sur_max_ctx}; using {clip_ctx}.")

        total = min(args.num_samples, len(dataset))

        # Attack selection (default: PGD)
        attack_fn = build_attack(args.attack)

        # OUTER LOOP
        samples_pbar = tqdm(range(total), desc="Samples", dynamic_ncols=True)
        for idx in samples_pbar:
            image_org, cle_text, image_tgt, tgt_text, path_clean = dataset[idx]
            image_org = image_org.unsqueeze(0).to(dev)  # 0..255
            image_tgt = image_tgt[0].unsqueeze(0).to(dev)  # 0..255

            # Tokenize with truncation control
            cle_text_tok = surrogate.tokenize_text([cle_text], context_length=clip_ctx)
            tgt_text_tok = surrogate.tokenize_text([tgt_text], context_length=clip_ctx)

            # Encode and fuse the static branches (no grads)
            with torch.no_grad():
                tgt_img_feats, tgt_txt_feats = surrogate.encode_image_text_features_normalized(image_tgt, tgt_text_tok)
                cle_img_feats, cle_txt_feats = surrogate.encode_image_text_features_normalized(image_org, cle_text_tok)
                if args.fusion_type == "cat":
                    cle_fused = fuse_features(cle_img_feats, cle_txt_feats, "cat", args.a_weight_cle)
                    tgt_fused = fuse_features(tgt_img_feats, tgt_txt_feats, "cat", args.a_weight_tgt)
                elif args.fusion_type == "add_weight":
                    cle_fused = fuse_features(cle_img_feats, cle_txt_feats, "add_weight", args.a_weight_cle)
                    tgt_fused = fuse_features(tgt_img_feats, tgt_txt_feats, "add_weight", args.a_weight_tgt)
                else:
                    cle_fused = fuse_features(cle_img_feats, cle_txt_feats, "multiplication", args.a_weight_cle)
                    tgt_fused = fuse_features(tgt_img_feats, tgt_txt_feats, "multiplication", args.a_weight_tgt)

            # Attack config (PGD fields)
            pgd_cfg = PGDConfig(
                steps=args.pgd_steps,
                alpha=args.alpha,
                epsilon=args.epsilon,
                fusion_type=args.fusion_type,
                a_weight=args.a_weight_cur,
                speed_up=args.speed_up,
                update_steps=args.update_steps,
                p_neg=args.p_neg,
            )

            # Attack inner loop
            delta, last_loss = attack_fn(
                image_org_0_255=image_org,
                cle_fused=cle_fused,
                tgt_fused=tgt_fused,
                surrogate=surrogate,
                cfg=pgd_cfg
            )

            # Save results + live JSON
            save_adv_image_and_annotation(
                image_org_0_255=image_org,
                delta=delta,
                path_clean=path_clean,
                output_dir=args.output,
                target_text=tgt_text,
                ann_writer=ann_writer
            )
            samples_pbar.set_postfix(last_loss=float(last_loss))

    finally:
        ann_writer.close()


def train_attackvlm(args, loader, dev, ann_writer):
    import open_clip  # lazy import

    if args.image_encoder == "attackvlm_default": 
        print(f"Loading open_clip model: {args.attackvlm_model_name} ({args.attackvlm_pretrained}) ...")
        model, _, _ = open_clip.create_model_and_transforms(
            args.attackvlm_model_name, pretrained=args.attackvlm_pretrained, device=dev
        )
        image_encoder_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=dev)
        image_encoder_std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=dev)
    else:
        print(f"Loading ViT image encoder: {args.image_encoder} ...")
        model = ViT(args.image_encoder, pretrained=True).to(dev)
        image_encoder_mean = torch.tensor([0.5, 0.5, 0.5], device=dev)
        image_encoder_std  = torch.tensor([0.5, 0.5, 0.5], device=dev)
        
    model.eval()
    print("Done.")

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
        clean_img_255 = clean_img_255.to(dev)  # (B,3,H,W), 0..255 float32
        tgt_img_255   = tgt_img_255.to(dev)

        image_org = to_clip_norm(clean_img_255)
        image_tgt = to_clip_norm(tgt_img_255)

        with torch.no_grad():
            tgt_feats = model.encode_image(image_tgt)             # (B, D)
            tgt_feats = F.normalize(tgt_feats, dim=1)

        delta = torch.zeros_like(image_org, requires_grad=True)
        steps = getattr(args, "pgd_steps", 300)

        for j in range(steps):
            adv_image = image_org + delta
            adv_feats = model.encode_image(adv_image)
            adv_feats = F.normalize(adv_feats, dim=1)

            embedding_sim = torch.mean(torch.sum(adv_feats * tgt_feats, dim=1))
            embedding_sim.backward()

            with torch.no_grad():
                grad = delta.grad.detach()
                delta.add_(alpha * torch.sign(grad))
                delta.clamp_(min=-epsilon, max=epsilon)
                delta.grad.zero_()

            max_delta = torch.max(torch.abs(delta)).item()
            mean_delta = torch.mean(torch.abs(delta)).item()
            print(
                f"iter {i+1}/{total_batches} step:{j:3d}, "
                f"embedding similarity={embedding_sim.item():.5f}, "
                f"max delta={max_delta:.3f}, mean delta={mean_delta:.3f}"
            )

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

    if 'attackvlm' in args.method:
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=getattr(args, "num_workers", 8),
            drop_last=False,
            pin_memory=(dev == "cuda"),
        )
        if 'multi' in args.method:
            train_attackvlm_multiencoder(args, loader, dev, ann_writer)
        elif 'losses' in args.method:
            train_attackvlm_losses(args, loader, dev, ann_writer)
        elif 'grow' in args.method:
            train_attackvlm_multiencoder_clsavg2cls(args, loader, dev, ann_writer)
        else:
            train_attackvlm(args, loader, dev, ann_writer)

    elif args.method == 'coa':
        train_coa(args, ann_writer, dataset, dev)
    else:
        raise NotImplementedError()

    
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
        clean_file_path=args.cle_file_path,  # JSONL with {"image": "n01440764/ILSVRC2012_val_00010306.JPEG", "text": "..."}
        judge_model_name="Qwen/Qwen2.5-7B-Instruct",
    )


    print(f"[eval] Saved consolidated metrics to: {os.path.join(args.output, 'metrics_report.json')}")
    print_metrics_report(report, enc_order=clip_encs)




if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()

    main(args)

    # values = [0.1, 0.5, 0.9]
    # fixed = 0.3
    # count = 1
    # output = args.output

    # for vary_name in ['a_weight_cle', 'a_weight_tgt', 'a_weight_cur']:
    #     for val in values:
    #         args.a_weight_cle = fixed
    #         args.a_weight_tgt = fixed
    #         args.a_weight_cur = fixed
            
    #         setattr(args, vary_name, val)

    #         args.output = f"{output}{count}"
    #         print(f"Run {count}: cle={args.a_weight_cle}, tgt={args.a_weight_tgt}, cur={args.a_weight_cur}, output={args.output}")

    #         count += 1

    #         main(args)