# attacks.py
from typing import List, Optional, Tuple, Callable
from dataclasses import dataclass
from tqdm import tqdm
import torch
from utils import fuse_features, hinge_loss_coa

@dataclass
class PGDConfig:
    steps: int
    alpha: float
    epsilon: int
    fusion_type: str
    a_weight: float
    speed_up: bool
    update_steps: int
    p_neg: float

# Attack function signature:
# fn(image_org_0_255, cle_fused, tgt_fused, surrogate, cfg) -> (delta, last_loss)

def pgd_single_sample(
    image_org_0_255: torch.Tensor,        # [1,3,H,W], values 0..255
    cle_fused: torch.Tensor,              # [1,D]
    tgt_fused: torch.Tensor,              # [1,D]
    surrogate,                            # SurrogateBase
    cfg: PGDConfig
) -> Tuple[torch.Tensor, float]:
    dev = image_org_0_255.device
    delta = torch.zeros_like(image_org_0_255, requires_grad=True, device=dev)
    last_caption_list: Optional[List[str]] = None
    last_loss = 0.0

    pgd_pbar = tqdm(range(cfg.steps), desc="PGD", leave=False, dynamic_ncols=True)
    for j in pgd_pbar:
        adv_image = image_org_0_255 + delta  # still 0..255

        # Caption regeneration schedule (kept identical to original behavior)
        regenerate = True
        if cfg.speed_up and j % cfg.update_steps != 0 and last_caption_list is not None:
            regenerate = False

        if regenerate:
            cap = surrogate.generate_caption_from_image_tensor_0_255(adv_image)
            current_caption_list = [cap]
            last_caption_list = current_caption_list
        else:
            current_caption_list = last_caption_list

        # Tokenize caption with surrogate default context
        cur_caption_tok = surrogate.tokenize_text(current_caption_list)

        # IMPORTANT: keep grads wrt image features
        adv_image_features = surrogate.encode_image_features_normalized(adv_image, requires_grad=True)
        # Text features do not need grads
        cur_text_features = surrogate.encode_text_features_normalized(cur_caption_tok)

        # Fuse as in original
        if cfg.fusion_type == "cat":
            cur_fused = fuse_features(adv_image_features, cur_text_features, "cat", cfg.a_weight)
        elif cfg.fusion_type == "add_weight":
            cur_fused = fuse_features(adv_image_features, cur_text_features, "add_weight", cfg.a_weight)
        else:
            cur_fused = fuse_features(adv_image_features, cur_text_features, "multiplication", cfg.a_weight)

        loss = hinge_loss_coa(cur_fused, cle_fused, tgt_fused, p_neg=cfg.p_neg)

        # PGD step (L_inf)
        loss.backward()
        grad = delta.grad.detach()
        delta.data = torch.clamp(delta + cfg.alpha * torch.sign(grad), min=-cfg.epsilon, max=cfg.epsilon)
        delta.grad.zero_()

        last_loss = float(loss.item())
        pgd_pbar.set_postfix(loss=last_loss)

    return delta.detach(), last_loss

AttackFn = Callable[..., Tuple[torch.Tensor, float]]


def build_attack(name: str):
    name = name.lower()
    if name == "pgd":
        return pgd_single_sample
    raise ValueError(f"Unknown attack: {name}")