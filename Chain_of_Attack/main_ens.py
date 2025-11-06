# main.py
import os
from pathlib import Path
from tqdm import tqdm
import torch

from datasets import CleanTargetPairsDataset, CLE_DATA_ROOT  # unchanged import
from utils import get_device, build_arg_parser, AnnotationWriter, save_adv_image_and_annotation, fuse_features
from utils import save_adv_image_unit_range_and_annotation
from models import build_surrogate
from eval_utils import report_metrics
from utils import parse_list_arg, seed_everything
from eval_utils import print_metrics_report
import torchvision
import torch.nn.functional as F
from trainers import *
from ViT_PyTorch.pytorch_pretrained_vit import ViT

from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from PIL import Image

from CLIP import clip
# from LAVIS.lavis.models import load_model_and_preprocess
from lavis.models import load_model_and_preprocess
from BLIP.models.blip import blip_decoder
from ruamel.yaml import YAML
import random

def seed_everything(seed: int):
    import numpy as np
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False


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

# def blip_img_encoder_forward(imgs, model, preprocess):
#     # embds = model.visual_encoder(preprocess(imgs))
#     preprocessed_imgs = preprocess["eval"](imgs)
#     embds = model.forward_encoder({"image": preprocessed_imgs})
#     cls_embd = F.normalize(embds[:,0,:], dim=1)
#     return cls_embd

# def clip_img_encoder_forward(imgs, model, preprocess):
#     cls_embd = model.encode_image(preprocess(imgs))
#     cls_embd = F.normalize(cls_embd, dim=1)
#     return cls_embd



def create_patch_mask(batch_size, image_size, patch_size, device):
    """Create a random binary mask that zeros out some patches."""
    num_patches = image_size // patch_size
    mask = torch.ones((batch_size, 1, num_patches, num_patches), device=device)
    
    # Randomly drop ~25% of patches (you can adjust the ratio)
    drop_ratio = 0.1
    num_drop = int(num_patches ** 2 * drop_ratio)
    for i in range(batch_size):
        drop_indices = torch.randperm(num_patches ** 2, device=device)[:num_drop]
        mask.view(batch_size, -1)[i, drop_indices] = 0

    # Upsample to image resolution
    mask = F.interpolate(mask, size=(image_size, image_size), mode='nearest')
    return mask


def blip_img_encoder_forward(imgs, model, preprocess, org_imgs=None):
    image_size = 384
    patch_size = 16  # from instruction
    batch_size = 8  # recommended batch size
    preprocessed_imgs = preprocess["eval"](imgs)


    if org_imgs is not None:
        preprocessed_org = preprocess["eval"](org_imgs)
        device = imgs.device
        mask = create_patch_mask(imgs.size(0), image_size, patch_size, device)
        preprocessed_imgs = mask * preprocessed_imgs + (1 - mask) * preprocessed_org

    embds = model.forward_encoder({"image": preprocessed_imgs})
    cls_embd = F.normalize(embds[:, 0, :], dim=1)
    return cls_embd


def clip_img_encoder_forward(imgs, model, preprocess, org_imgs=None):
    image_size = 224
    patch_size = 32  # from instruction
    batch_size = 8  # recommended batch size

    if org_imgs is not None:
        device = imgs.device
        mask = create_patch_mask(imgs.size(0), image_size, patch_size, device)
        imgs = mask * imgs + (1 - mask) * org_imgs

    cls_embd = model.encode_image(preprocess(imgs))
    cls_embd = F.normalize(cls_embd, dim=1)
    return cls_embd

def train_ens_attack(args, loader, dev, ann_writer):
    clip_model, clip_preprocess = clip.load("/home/user01/research/Chain_of_Attack/CLIP/weights/ViT-B-32.pt", device=dev)
    clip_model.eval()

    # yaml = YAML(typ='rt')
    # blip_config = yaml.load(open("/home/user01/research/Chain_of_Attack/BLIP/configs/caption_coco.yaml", 'r'))
    # blip_model = blip_decoder(pretrained=blip_config['pretrained'], image_size=blip_config['image_size'], vit=blip_config['vit'], 
    #                        vit_grad_ckpt=blip_config['vit_grad_ckpt'], vit_ckpt_layer=blip_config['vit_ckpt_layer'], 
    #                        prompt=blip_config['prompt']).to(dev)
    blip_model, blip_preprocess, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device=dev)
    blip_model.eval()

    # blip_preprocess = transforms.Compose([
    #     transforms.Resize((blip_config['image_size'],blip_config['image_size']),interpolation=InterpolationMode.BICUBIC),
    #     # transforms.ToTensor(),
    #     torchvision.transforms.Lambda(lambda img: torch.clamp(img, 0.0, 255.0) / 255.0),
    #     transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    #     ])  


    # image_path = "/home/user01/research/Chain_of_Attack/CLIP/cat.jpeg"
    # image = Image.open(image_path).convert("RGB")  # Ensure 3 channels
    # to_tensor = transforms.ToTensor()  # by default ToTensor() gives 0-1
    # image_tensor = to_tensor(image).to(dev) * 255  # scale to 0-255
    # image_tensor = image_tensor.unsqueeze(0)
    # image = blip_preprocess["eval"](image_tensor)
    # # generate caption
    # cap = blip_model.generate({"image": image})
    # print(cap)
    # return

    # vit_model = ViT("B_16", pretrained=True).to(dev)
    # vit_preprocess = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    surrogate = build_surrogate(args.surrogate, args, dev)

    # mean and std are the same for clip and blip
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=dev)
    std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=dev)

    scaling_tensor = std.view(3, 1, 1).unsqueeze(0)  # (1,3,1,1)
    epsilon = args.epsilon / 255.0
    alpha   = args.alpha   / 255.0

    inverse_normalize = torchvision.transforms.Normalize(
        mean=(-mean / std).tolist(),
        std=(1.0 / std).tolist(),
    )

    out_root = Path(args.output).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    total_batches = (len(loader.dataset) + args.batch_size - 1) // args.batch_size

    for i, batch in enumerate(loader):
        clean_img_255, clean_text, tgt_img_255, tgt_text, clean_abs_paths = batch
        clean_img_255 = clean_img_255.to(dev).squeeze(1) / 255.0
        tgt_img_255   = tgt_img_255.to(dev).squeeze(1) / 255.0

        with torch.no_grad():
            blip_tgt_embds = blip_img_encoder_forward(tgt_img_255, blip_model, blip_preprocess)
            clip_tgt_embds = clip_img_encoder_forward(tgt_img_255, clip_model, clip_preprocess)

            tgt_token = clip.tokenize(tgt_text).to(dev)
            tgt_txt_emb = clip_model.encode_text(tgt_token)
            tgt_txt_emb = F.normalize(tgt_txt_emb, dim=1)

        delta = torch.zeros_like(clean_img_255, requires_grad=True)
        steps = args.pgd_steps

        for j in range(steps):
            adv_image = clean_img_255 + delta

            blip_adv_embds = blip_img_encoder_forward(adv_image, blip_model, blip_preprocess)
            clip_adv_embds = clip_img_encoder_forward(adv_image, clip_model, clip_preprocess)

            blip_sim = torch.mean(torch.sum(blip_adv_embds * blip_tgt_embds, dim=1))
            clip_sim = torch.mean(torch.sum(clip_adv_embds * clip_tgt_embds, dim=1))

            blip_grad = torch.autograd.grad(blip_sim, delta, retain_graph=True, create_graph=False)[0]
            clip_grad = torch.autograd.grad(clip_sim, delta, retain_graph=True, create_graph=False)[0]

            with torch.no_grad():
                vlm_feedback = surrogate.generate_caption_from_batch([
                        transforms.ToPILImage()((clean_img_255+torch.clamp(delta+alpha*torch.sign(blip_grad), min=-epsilon,max=epsilon)).squeeze(0)),
                        transforms.ToPILImage()((clean_img_255+torch.clamp(delta+alpha*torch.sign(clip_grad), min=-epsilon,max=epsilon)).squeeze(0))
                    ])
            
            feedback_token = clip.tokenize(vlm_feedback).to(dev)
            feedback_emb = clip_model.encode_text(feedback_token)
            feedback_emb = F.normalize(feedback_emb, dim=1)

            coefs = (feedback_emb * tgt_txt_emb).sum(dim=1)
            # TODO
            grad = (coefs[0] * blip_grad + coefs[1] * clip_grad) / coefs.sum()

            print(f"step {j}, clip {clip_sim}, blip {blip_sim}")

            with torch.no_grad():
                delta.add_(alpha * torch.sign(grad))
                delta.clamp_(min=-epsilon, max=epsilon)

            max_delta = torch.max(torch.abs(delta)).item()
            mean_delta = torch.mean(torch.abs(delta)).item()
        
        with torch.no_grad():
            adv_image_vis = torch.clamp(clean_img_255 + delta, 0.0, 1.0)

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

    train_ens_attack(args, loader, dev, ann_writer)

    
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
        enable_asr=bool(args.do_asr),
    )


    print(f"[eval] Saved consolidated metrics to: {os.path.join(args.output, 'metrics_report.json')}")
    print_metrics_report(report, enc_order=clip_encs)



if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()

    main(args)