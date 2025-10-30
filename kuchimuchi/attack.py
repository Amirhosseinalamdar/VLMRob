# attack.py
from dataclasses import dataclass
from typing import List
import torch
import torch.nn.functional as F
from models import ImageModelSpec

@dataclass
class AttackConfig:
    epsilon: float      # L_inf radius in pixel units (0..255)
    alpha: float        # step size in pixel units
    steps: int
    random_start: bool = True

def clamp_img_255(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, 0.0, 255.0)

def pgd_targeted_sum_mse(models: List[ImageModelSpec],
                         images_255: torch.Tensor,
                         tgt_text_emb: torch.Tensor,
                         cfg: AttackConfig) -> torch.Tensor:
    """
    Targeted PGD (cosine similarity version):
      maximize sum_k cos_sim(f_k(x), t) over frozen models k.

    Assumptions:
      - models[k].encode returns L2-normalized embeddings (||f_k(x)||=1).
      - tgt_text_emb is already L2-normalized (the text encoder normalizes).
      - images_255: [B,3,H,W] float in [0,255]
      - tgt_text_emb: [B,D] or [D]
    """
    device = images_255.device
    x0 = images_255.detach().float()
    delta = torch.zeros_like(x0, device=device, requires_grad=True)

    if cfg.random_start:
        delta.data.uniform_(-cfg.epsilon, cfg.epsilon)

    for _ in range(cfg.steps):
        adv = clamp_img_255(x0 + delta)

        # Sum cosine similarities across models, then ascend the gradient
        total_sim = 0.0
        for spec in models:
            z_img = spec.encode(adv).float()  # [B, D], expected normalized
            tgt = tgt_text_emb
            if tgt.dim() == 1:
                tgt = tgt.unsqueeze(0).expand_as(z_img)
            tgt = F.normalize(tgt.float(), dim=-1)  # safety normalize

            # cosine similarity = dot product for normalized vectors
            sim = (z_img * tgt).sum(dim=1).mean()   # scalar
            total_sim = total_sim + sim

        # Gradient ascent on similarity (equivalently, descent on -similarity)
        grad = torch.autograd.grad(total_sim, delta, retain_graph=False, create_graph=False)[0]
        delta.data = (delta + cfg.alpha * torch.sign(grad)).clamp(-cfg.epsilon, cfg.epsilon)
        delta.grad = None

    return clamp_img_255(x0 + delta)