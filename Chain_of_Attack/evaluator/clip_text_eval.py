#!/usr/bin/env python
import os, argparse, random, csv
import numpy as np
import torch
import clip
from tqdm import tqdm

DEFAULT_RANDOM_SEED = 2023
device = "cuda" if torch.cuda.is_available() else "cpu"

def seed_everything(seed=DEFAULT_RANDOM_SEED):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_text_pairs_from_results_csv(path, limit=None):
    tgts, preds = [], []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"image_id", "target_text", "collected_text"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"{path} must have columns: image_id,target_text,collected_text")
        for row in reader:
            tgts.append((row.get("target_text") or "").strip())
            preds.append((row.get("collected_text") or "").strip())
    if limit is not None:
        tgts = tgts[:limit]
        preds = preds[:limit]
    return tgts, preds

@torch.no_grad()
def encode_texts(texts, clip_model, batch_size=32):
    toks = clip.tokenize(texts, truncate=True).to(device)
    feats = []
    for start in tqdm(range(0, len(texts), batch_size), desc="encode_texts"):
        end = min(start + batch_size, len(texts))
        f = clip_model.encode_text(toks[start:end])
        feats.append(f)
    feats = torch.cat(feats, dim=0)
    feats = feats / feats.norm(dim=1, keepdim=True)
    return feats

def compute_txt_sim_from_csv(args, clip_model):
    tgt_texts, pred_texts = load_text_pairs_from_results_csv(args.results_csv, limit=args.num_samples)
    n = min(len(pred_texts), len(tgt_texts))
    pred_texts, tgt_texts = pred_texts[:n], tgt_texts[:n]
    if n == 0:
        raise RuntimeError("No rows loaded from results CSV.")
    pred_feats = encode_texts(pred_texts, clip_model, batch_size=args.batch_size)
    tgt_feats  = encode_texts(tgt_texts,  clip_model, batch_size=args.batch_size)
    cos = torch.sum(pred_feats * tgt_feats, dim=1)  # cosine similarity per pair (normalized)
    score = cos.mean().item()
    print(f"Average CLIP text-text cosine: {score:.5f}")
    return score

def main():
    seed_everything()
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_csv", required=True, type=str,
                        help="CSV with columns: image_id,target_text,collected_text")
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--num_samples", default=None, type=int, help="limit; default=all")
    parser.add_argument("--clip_encoder", default=None, type=str)
    args = parser.parse_args()

    clip_encoders = ['RN50', 'RN101', 'ViT-B/16', 'ViT-B/32', 'ViT-L/14']
    if args.clip_encoder:
        print(f"Using CLIP encoder: {args.clip_encoder}")
        clip_model, _ = clip.load(args.clip_encoder, device=device)
        clip_model.eval()
        _ = compute_txt_sim_from_csv(args, clip_model)
    else:
        sims = []
        for enc in clip_encoders:
            print(f"Using CLIP encoder: {enc}")
            clip_model, _ = clip.load(enc, device=device)
            clip_model.eval()
            sims.append(compute_txt_sim_from_csv(args, clip_model))
        print(f"Ensemble average over {len(clip_encoders)} encoders: {np.mean(sims):.5f}")

if __name__ == "__main__":
    main()