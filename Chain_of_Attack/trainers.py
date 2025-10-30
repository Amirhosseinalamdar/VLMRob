# ============================================================
# Multi-encoder adversarial attack utilities (module-level)
# Encoders: CLIP (open_clip), DINO (timm), MAE (timm)
# - Optimizes a single pixel-space perturbation to maximize the
#   cosine similarity between adv image and target image embeddings
#   across all encoders simultaneously.
# ============================================================

from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F
import torchvision


# -------------- Encoder Wrapper --------------
from typing import Optional, Tuple, List, Callable
import torch
import torch.nn.functional as F

def _ensure_2d_features(f: torch.Tensor) -> torch.Tensor:
    # Ensures shape (B, C) by flattening any remaining spatial dims
    return f.view(f.size(0), -1) if f.dim() > 2 else f

class BaseEncoder(torch.nn.Module):
    """
    Thin wrapper providing:
      - differentiable preprocess (resize + normalize)
      - forward_embed(x01): returns L2-normalized feature (B, C)
      - target_embed(x01): same under no_grad()
    Expects input images in [0,1] range, shape (B,3,H,W).
    """

    def __init__(
        self,
        name: str,
        model: torch.nn.Module,
        input_hw: Optional[Tuple[int, int]],
        mean: List[float],
        std: List[float],
        device: torch.device,
        encode_fn: Callable[[torch.Tensor], torch.Tensor],
    ):
        super().__init__()
        self.name = name
        self.model = model.to(device).eval()
        self.input_hw = input_hw  # (H, W) or None

        # Common aliases used by various repos
        if input_hw is None:
            self.image_size = None               # some code may check for None
            self.input_size = None               # (H, W) alias
            self.input_resolution = None         # alias
        else:
            h, w = input_hw
            self.input_size = (h, w)             # (H, W)
            self.input_resolution = (h, w)       # alias
            self.image_size = (h, w)  # int if square, else (H, W)

        self.num_channels = 3

        self.register_buffer(
            "mean",
            torch.tensor(mean, dtype=torch.float32, device=device).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor(std, dtype=torch.float32, device=device).view(1, 3, 1, 1),
            persistent=False,
        )
        self.encode_fn = encode_fn

    def _preprocess(self, x01: torch.Tensor) -> torch.Tensor:
        if self.input_hw is not None:
            x_resized = F.interpolate(x01, size=self.input_hw, mode="bilinear", align_corners=False)
        else:
            x_resized = x01
        return (x_resized - self.mean) / self.std

    @torch.no_grad()
    def target_embed(self, x01: torch.Tensor) -> torch.Tensor:
        x = self._preprocess(x01)
        f = self.encode_fn(x)
        return F.normalize(_ensure_2d_features(f), dim=1)

    def forward_embed(self, x01: torch.Tensor) -> torch.Tensor:
        x = self._preprocess(x01)
        f = self.encode_fn(x)
        return F.normalize(_ensure_2d_features(f), dim=1)

    # Let module(x) behave like forward_embed(x)
    def forward(self, x01: torch.Tensor) -> torch.Tensor:
        return self.forward_embed(x01)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
# -------------- Builders --------------

def build_clip_encoder(device, model_name: Optional[str] = None, pretrained: Optional[str] = None) -> BaseEncoder:
    import open_clip  # lazy import
    model_name = model_name or "ViT-B-32"
    pretrained = pretrained or "openai"
    print(f"[enc] Loading open_clip (attackvlm_default): {model_name} ({pretrained})")
    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)

    # Infer input size
    H = W = 224
    vis = getattr(model, "visual", None)
    if vis is not None:
        if hasattr(vis, "image_size"):
            sz = vis.image_size
            if isinstance(sz, (tuple, list)) and len(sz) == 2:
                H, W = int(sz[0]), int(sz[1])
            elif isinstance(sz, (int, float)):
                H = W = int(sz)
        elif hasattr(vis, "input_resolution"):
            H = W = int(getattr(vis, "input_resolution"))

    mean = [0.48145466, 0.4578275, 0.40821073]
    std  = [0.26862954, 0.26130258, 0.27577711]

    def _encode_fn(x_norm):
        return model.encode_image(x_norm)

    return BaseEncoder("clip", model, (H, W), mean, std, device, _encode_fn)


def _try_create_timm_encoder(model_name: str, device, tag: str) -> Optional[BaseEncoder]:
    import timm  # lazy import
    try:
        m = timm.create_model(model_name, pretrained=True, num_classes=0)
        m = m.to(device).eval()
        cfg = getattr(m, "default_cfg", {}) or {}
        in_size = cfg.get("input_size", (3, 224, 224))
        H, W = int(in_size[-2]), int(in_size[-1])
        mean = cfg.get("mean", (0.485, 0.456, 0.406))
        std  = cfg.get("std",  (0.229, 0.224, 0.225))

        def _encode_fn(x_norm: torch.Tensor):
            if hasattr(m, "forward_features"):
                f = m.forward_features(x_norm)
            else:
                f = m(x_norm)
            return f

        print(f"[enc] Loaded timm encoder: {model_name} ({tag})")
        return BaseEncoder(tag, m, (H, W), mean, std, device, _encode_fn)
    except Exception as e:
        print(f"[enc] Could not load '{model_name}' ({tag}): {e}")
        return None


def build_dino_encoder(device, model_name: Optional[str] = None) -> BaseEncoder:
    """
    Tries, in order:
      1) user-specified model_name (if provided)
      2) vit_small_patch16_224.dino        (classic DINO small)
      3) vit_small_patch14_dinov2.lvd142m  (DINOv2 small)
      4) vit_base_patch16_224.dino         (classic DINO base)
    """
    candidates = []
    if model_name:
        candidates.append((model_name, f"dino (requested={model_name})"))
    candidates += [
        ("vit_small_patch16_224.dino", "dino (vit_small)"),
        ("vit_small_patch14_dinov2.lvd142m", "dinov2 (vit_small patch14)"),
        ("vit_base_patch16_224.dino", "dino (vit_base)"),
    ]
    for name, tag in candidates:
        enc = _try_create_timm_encoder(name, device, tag)
        if enc is not None:
            return enc
    raise RuntimeError("No valid DINO model found. Try a timm DINO/DINOv2 model with pretrained weights.")


def build_mae_encoder(device, model_name: Optional[str] = None) -> BaseEncoder:
    """
    MAE 'vit_small' checkpoints are typically not shipped in timm.
    Default to vit_base_patch16_224.mae and fall back to other MAE bases if needed.
    """
    candidates = []
    if model_name:
        candidates.append((model_name, f"mae (requested={model_name})"))
    candidates += [
        ("vit_base_patch16_224.mae", "mae (vit_base)"),
        ("vit_large_patch16_224.mae", "mae (vit_large)"),
    ]
    for name, tag in candidates:
        enc = _try_create_timm_encoder(name, device, tag)
        if enc is not None:
            return enc
    raise RuntimeError("No valid MAE model found. Try 'vit_base_patch16_224.mae' or another MAE variant available in timm.")

def build_resnet50_encoder(device, model_name: str | None = None) -> BaseEncoder:
    """
    Build a ResNet-50 encoder compatible with BaseEncoder.
    Priority:
      1) timm (e.g., 'resnet50', 'resnetv2_50', 'resnet50.tv_in1k', etc.)
      2) torchvision fallback (IMAGENET1K pretrained)
    Returns features suitable for cosine similarity (we L2-normalize in BaseEncoder).
    """
    # --- Try timm first ---
    try:
        import timm
        name = model_name or "resnet50"
        print(f"[enc] Loading timm ResNet-50: {name}")
        # num_classes=0 removes the classifier and yields feature vectors directly
        m = timm.create_model(name, pretrained=True, num_classes=0)
        m = m.eval().to(device)

        cfg = getattr(m, "default_cfg", {}) or {}
        in_size = cfg.get("input_size", (3, 224, 224))
        H, W = int(in_size[-2]), int(in_size[-1])
        mean = cfg.get("mean", (0.485, 0.456, 0.406))
        std  = cfg.get("std",  (0.229, 0.224, 0.225))

        # With num_classes=0, calling m(x) returns a pooled feature (B, C)
        def _encode_fn(x_norm: torch.Tensor):
            return m(x_norm)

        return BaseEncoder("resnet50", m, (H, W), mean, std, device, _encode_fn)

    except Exception as e:
        print(f"[enc] timm resnet50 failed, falling back to torchvision: {e}")

    # --- Torchvision fallback ---
    import torchvision.models as tvm
    print("[enc] Loading torchvision ResNet-50 (IMAGENET1K) ...")
    try:
        # Newer torchvision API
        weights = tvm.ResNet50_Weights.IMAGENET1K_V2
        m = tvm.resnet50(weights=weights)
        mean = (0.485, 0.456, 0.406)
        std  = (0.229, 0.224, 0.225)
    except Exception:
        # Older API
        m = tvm.resnet50(pretrained=True)
        mean = (0.485, 0.456, 0.406)
        std  = (0.229, 0.224, 0.225)

    # Replace the classifier with identity so forward returns pooled features (B, 2048)
    m.fc = torch.nn.Identity()
    m = m.eval().to(device)

    def _encode_fn(x_norm: torch.Tensor):
        return m(x_norm)

    return BaseEncoder("resnet50", m, (224, 224), mean, std, device, _encode_fn)



def build_dinov2_encoder(device, model_name: str | None = None) -> BaseEncoder:
    """
    Build a DINOv2 encoder (timm) compatible with BaseEncoder.

    Priority order:
      1) user-specified model_name (if provided)
      2) vit_small_patch14_dinov2.lvd142m
      3) vit_base_patch14_dinov2.lvd142m
      4) vit_base_patch14_reg4_dinov2.lvd142m
      5) vit_large_patch14_dinov2.lvd142m
      6) vit_giant_patch14_dinov2.lvd142m

    Returns:
      BaseEncoder instance whose forward_embed/target_embed output L2-normalized features (B, C).
    """
    candidates = []
    if model_name:
        candidates.append((model_name, f"dinov2 (requested={model_name})"))
    candidates += [
        ("vit_small_patch14_dinov2.lvd142m", "dinov2 (vit_small patch14 lvd142m)"),
        ("vit_base_patch14_dinov2.lvd142m",  "dinov2 (vit_base patch14 lvd142m)"),
        ("vit_base_patch14_reg4_dinov2.lvd142m", "dinov2 (vit_base reg4 patch14 lvd142m)"),
        ("vit_large_patch14_dinov2.lvd142m", "dinov2 (vit_large patch14 lvd142m)"),
        ("vit_giant_patch14_dinov2.lvd142m", "dinov2 (vit_giant patch14 lvd142m)"),
    ]
    for name, tag in candidates:
        enc = _try_create_timm_encoder(name, device, tag)
        if enc is not None:
            return enc
    raise RuntimeError(
        "No valid DINOv2 model found in timm. "
        "Try upgrading timm or specify a known model name like "
        "'vit_base_patch14_dinov2.lvd142m'."
    )

def build_msn_encoder(device, model_name: str | None = None) -> BaseEncoder:
    """
    Build an MSN (Masked Siamese Networks) encoder via timm, compatible with BaseEncoder.

    Priority:
      1) user-specified model_name (if provided)
      2) vit_small_patch16_224.msn
      3) vit_base_patch16_224.msn
      4) vit_large_patch16_224.msn

    Returns:
      BaseEncoder whose forward_embed/target_embed yield L2-normalized features (B, C).
    """
    candidates = []
    if model_name:
        candidates.append((model_name, f"msn (requested={model_name})"))
    candidates += [
        ("vit_small_patch16_224.msn", "msn (vit_small)"),
        ("vit_base_patch16_224.msn",  "msn (vit_base)"),
        ("vit_large_patch16_224.msn", "msn (vit_large)"),
    ]
    for name, tag in candidates:
        enc = _try_create_timm_encoder(name, device, tag)
        if enc is not None:
            return enc
    raise RuntimeError(
        "No valid MSN model found in timm. Try installing a recent timm version or "
        "pass a known MSN model id (e.g., 'vit_base_patch16_224.msn')."
    )

def build_msn_encoder(device, model_name: str | None = None) -> BaseEncoder:
    """
    Build an MSN (Masked Siamese Networks) encoder via timm, compatible with BaseEncoder.

    Priority:
      1) user-specified model_name (if provided)
      2) vit_small_patch16_224.msn
      3) vit_base_patch16_224.msn
      4) vit_large_patch16_224.msn

    Returns:
      BaseEncoder whose forward_embed/target_embed yield L2-normalized features (B, C).
    """
    candidates = []
    if model_name:
        candidates.append((model_name, f"msn (requested={model_name})"))
    candidates += [
        ("vit_small_patch16_224.msn", "msn (vit_small)"),
        ("vit_base_patch16_224.msn",  "msn (vit_base)"),
        ("vit_large_patch16_224.msn", "msn (vit_large)"),
    ]
    for name, tag in candidates:
        enc = _try_create_timm_encoder(name, device, tag)
        if enc is not None:
            return enc
    raise RuntimeError(
        "No valid MSN model found in timm. Try installing a recent timm version or "
        "pass a known MSN model id (e.g., 'vit_base_patch16_224.msn')."
    )


# -------------- Training Function --------------

def train_attackvlm_multiencoder(args, loader, dev, ann_writer):
    """
    Multi-encoder PGD attack in pixel space.
    Encoders:
      - CLIP via open_clip (attackvlm_default)
      - DINO via timm (prefers vit_small if available)
      - MAE  via timm (uses vit_base_patch16_224.mae by default)

    Required/optional args fields:
      - attackvlm_model_name, attackvlm_pretrained (for CLIP)
      - dino_model_name (optional)
      - mae_model_name  (optional)
      - alpha, epsilon  (pixel units, e.g., 4 / 8)
      - pgd_steps (default: 300)
      - batch_size, output
      - encoder_weights (optional: [w1,w2,w3] or "w1,w2,w3")
      - cle_data_root (optional: root Path for class-name derivation)
    """
    # Build encoders
    clip_name = getattr(args, "attackvlm_model_name", "ViT-B-32")
    clip_pt   = getattr(args, "attackvlm_pretrained", "openai")
    clip_enc = build_clip_encoder(dev, clip_name, clip_pt)

    encoders = [clip_enc]

    # Encoder weights
    weights = [1.0]
    weights_tensor = torch.tensor(weights, dtype=torch.float32, device=dev)
    print(f"[enc] Using encoder weights: {weights}")

    # Attack hyperparameters (pixel space)
    alpha = getattr(args, "alpha", 4.0) / 255.0
    epsilon = getattr(args, "epsilon", 8.0) / 255.0
    steps = getattr(args, "pgd_steps", 300)

    out_root = Path(args.output).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    total_batches = (len(loader.dataset) + getattr(args, "batch_size", 1) - 1) // max(1, getattr(args, "batch_size", 1))

    # Optional CLE root for class folders
    cle_root = getattr(args, "cle_data_root", None)
    if cle_root is None:
        # If a global exists, use it; else keep None
        try:
            cle_root = CLE_DATA_ROOT  # type: ignore[name-defined]
        except Exception:
            cle_root = None
    if cle_root is not None:
        cle_root = Path(cle_root).resolve()

    for i, batch in enumerate(loader):
        clean_img_255, clean_text, tgt_img_255, tgt_text, clean_abs_paths = batch
        clean_img_255 = clean_img_255.to(dev)  # float32 0..255
        tgt_img_255   = tgt_img_255.to(dev)

        x0 = (clean_img_255 / 255.0).clamp(0.0, 1.0)
        # Expect stacked targets: [B,K,C,H,W] (set stack_targets=True in your dataset)
        if tgt_img_255.dim() == 4:
            # No multi-targets present, treat as K=1
            tgt_img_255 = tgt_img_255.unsqueeze(1)  # [B,1,C,H,W]
        elif tgt_img_255.dim() != 5:
            raise RuntimeError(
                f"Expected tgt_img_255 to be [B,K,C,H,W] or [B,C,H,W], "
                f"got shape {tuple(tgt_img_255.shape)}. Set stack_targets=True in your dataset."
            )

        x_tg = (tgt_img_255 / 255.0).clamp(0.0, 1.0)  # [B,K,C,H,W]
        B, K = x_tg.size(0), x_tg.size(1)

        # Cache target embeddings per encoder (no grad)
        with torch.no_grad():
            # Will hold, per encoder:
            # - for 'mean': [B,D] normalized
            # - for 'each': [B,K,D], each feature normalized
            cached_targets = []
            x_tg_flat = x_tg.view(B * K, *x_tg.shape[2:])  # [B*K,C,H,W]
            for enc in encoders:
                f_tg_flat = enc.target_embed(x_tg_flat)  # [B*K, D], assumed L2-normalized
                f_tg = f_tg_flat.view(B, K, -1)          # [B,K,D]
                if args.multi_target_method == "mean":
                    f_mean = f_tg.mean(dim=1)            # [B,D]
                    f_mean = F.normalize(f_mean, dim=-1) # re-normalize mean vector
                    cached_targets.append(f_mean)        # [B,D]
                else:  # "each"
                    # Keep all per-target embeddings
                    cached_targets.append(f_tg)          # [B,K,D]

        # Single perturbation in pixel space
        delta = torch.zeros_like(x0, requires_grad=True)
        for j in range(steps):
            x_adv = (x0 + delta).clamp(0.0, 1.0)

            # Sum cosine similarities across encoders (and across targets if method='each')
            total_sim = 0.0
            sims = []  # per-encoder scalar for logging
            for k, enc in enumerate(encoders):
                f_adv = enc.forward_embed(x_adv)  # [B,D], L2-normalized

                if args.multi_target_method == "mean":
                    # cached_targets[k]: [B,D]
                    sim_k_b = torch.sum(f_adv * cached_targets[k], dim=1)  # [B]
                    sim_k = sim_k_b.mean()  # scalar
                else:
                    # cached_targets[k]: [B,K,D]
                    # Cosine for each target: (B,D) x (B,K,D) -> (B,K)
                    sim_bk = torch.einsum("bd,bkd->bk", f_adv, cached_targets[k])  # [B,K]
                    # Sum over K targets, then mean over batch
                    sim_k = sim_bk.sum(dim=1).mean()  # scalar; max = K if all 1's

                sims.append(sim_k.detach())
                total_sim = total_sim + weights_tensor[k] * sim_k

            # Gradient ascent on similarity
            total_sim.backward()
            with torch.no_grad():
                grad = delta.grad
                delta.add_(alpha * torch.sign(grad))
                # Project onto L_inf ball and image bounds
                delta.clamp_(-epsilon, epsilon)
                delta.copy_((x0 + delta).clamp(0.0, 1.0) - x0)
                delta.grad.zero_()

            # Logging
            if (j % max(1, steps // 10) == 0) or (j == steps - 1):
                max_delta = torch.max(torch.abs(delta)).item()
                mean_delta = torch.mean(torch.abs(delta)).item()
                sim_vals = [f"{s.item():.5f}" for s in sims]
                method = args.multi_target_method
                print(
                    f"iter {i+1}/{total_batches} step:{j:3d} [{method}] "
                    f"total_sim={float(total_sim.item()):.5f}, sims={sim_vals}, "
                    f"max|delta|={max_delta:.4f}, mean|delta|={mean_delta:.4f}"
                )

        with torch.no_grad():
            adv_image_vis = (x0 + delta).clamp(0.0, 1.0)

        # Save under class folders and write annotation
        for b in range(adv_image_vis.size(0)):
            clean_abs = Path(clean_abs_paths[b]).resolve()
            # Derive class name
            class_name = clean_abs.parent.name
            if cle_root is not None:
                try:
                    rel = clean_abs.relative_to(cle_root)
                    parts = rel.parts
                    if len(parts) >= 1:
                        class_name = parts[0]
                except Exception:
                    pass

            class_dir = (out_root / class_name)
            class_dir.mkdir(parents=True, exist_ok=True)
            stem = clean_abs.stem
            out_path = (class_dir / f"{stem}.png").resolve()
            torchvision.utils.save_image(adv_image_vis[b], str(out_path))

            rec = {"image": str(out_path), "text": str(tgt_text[b])}
            if ann_writer is not None:
                ann_writer.write(rec)
            else:
                # Optional global 'annotations' support
                try:
                    annotations.append(rec)  # type: ignore[name-defined]
                except Exception:
                    pass


######################################################
######################################################
######################################################
######################################################
######################################################

import math
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# -----------------------------
# Helpers: normalization/resizing
# -----------------------------
def _resize_norm(x, size_hw: Tuple[int,int], mean, std):
    # x in [0,1], shape [B,C,H,W]; returns normalized tensor for encoder
    if x.shape[-2:] != size_hw:
        x = F.interpolate(x, size=size_hw, mode="bilinear", align_corners=False)
    mean_t = torch.tensor(mean, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    std_t  = torch.tensor(std,  device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    return (x - mean_t) / std_t

# -----------------------------
# Token extraction for open_clip ViT
# -----------------------------
@torch.no_grad()
def _infer_grid_size(vis, H, W) -> Tuple[int, int]:
    # Works for square patch; for safety, compute from conv1 if available
    if hasattr(vis, "patch_size"):
        ps = vis.patch_size
        gy, gx = H // ps, W // ps
        return gy, gx
    # Fallback: infer from conv1 stride
    if hasattr(vis, "conv1") and hasattr(vis.conv1, "stride"):
        sy, sx = vis.conv1.stride if isinstance(vis.conv1.stride, tuple) else (vis.conv1.stride, vis.conv1.stride)
        ky, kx = vis.conv1.kernel_size if isinstance(vis.conv1.kernel_size, tuple) else (vis.conv1.kernel_size, vis.conv1.kernel_size)
        gy = (H + 2*vis.conv1.padding[0] - ky) // sy + 1
        gx = (W + 2*vis.conv1.padding[1] - kx) // sx + 1
        return gy, gx
    # Last resort
    ps = 16
    return H // ps, W // ps

def _iter_resblocks(vis):
    # open_clip VisionTransformer typically has: vis.transformer.resblocks (list/ModuleList)
    if hasattr(vis, "transformer"):
        tr = vis.transformer
        if hasattr(tr, "resblocks"):
            return list(tr.resblocks)
        if hasattr(tr, "blocks"):
            return list(tr.blocks)
    raise RuntimeError("Cannot locate transformer blocks in visual encoder.")

def extract_openclip_tokens(visual, x_norm, layer_idx: int = -2, drop_cls: bool = True, l2_token_norm: bool = True):
    """
    visual: open_clip.model.VisionTransformer (model.visual)
    x_norm: normalized, resized input [B,3,H,W]
    Returns: tokens [B, P, D] (P excludes CLS if drop_cls=True)
    """
    B, C, H, W = x_norm.shape
    dtype = next(visual.parameters()).dtype
    x = x_norm.to(dtype)

    # Patchify
    x = visual.conv1(x)                       # [B, C2, G, G]
    x = x.reshape(B, x.shape[1], -1)          # [B, C2, G*G]
    x = x.permute(0, 2, 1)                    # [B, P, C2]
    # Add CLS
    if hasattr(visual, "class_embedding"):
        cls_token = visual.class_embedding.to(x.dtype)
        cls_tok = cls_token + torch.zeros(B, 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([cls_tok, x], dim=1)    # [B, 1+P, C2]
    # Positional embedding and LN
    x = x + visual.positional_embedding.to(x.dtype)
    x = visual.ln_pre(x)
    # Transformer blocks up to desired layer
    x = x.permute(1, 0, 2)                    # [L, B, D]
    blocks = _iter_resblocks(visual)
    L = len(blocks)
    idx = layer_idx if layer_idx >= 0 else (L + layer_idx)
    idx = max(0, min(idx, L - 1))
    for b in range(idx + 1):
        x = blocks[b](x)
    x = x.permute(1, 0, 2)                    # [B, 1+P, D]

    if drop_cls:
        x = x[:, 1:, :]                       # [B, P, D]
    if l2_token_norm:
        x = F.normalize(x, dim=-1)
    return x

# -----------------------------
# Token-space loss functions
# -----------------------------
def loss_token_l2(Z1, Z2):
    # position-aligned; shapes [B,P,D]
    return F.mse_loss(Z1, Z2)

def loss_token_cos(Z1, Z2):
    sim = (Z1 * Z2).sum(-1)                   # [B,P]
    return (1.0 - sim).mean()

def loss_gram(Z1, Z2):
    # match token-token cosine Gram matrices (semantic relations)
    B = Z1.size(0)
    loss = 0.0
    for b in range(B):
        A = Z1[b] @ Z1[b].t()                 # [P,P]
        Bm = Z2[b] @ Z2[b].t()
        # Normalize by P^2 to keep scale stable
        loss = loss + F.mse_loss(A, Bm) / (A.numel())
    return loss / B

def _cov(z):                                  # z: [P,D], already token-normalized
    zc = z - z.mean(dim=0, keepdim=True)
    P = z.size(0)
    return (zc.t() @ zc) / max(P - 1, 1)

def loss_coral(Z1, Z2):
    B = Z1.size(0)
    loss = 0.0
    for b in range(B):
        C1, C2 = _cov(Z1[b]), _cov(Z2[b])
        loss = loss + F.mse_loss(C1, C2)
    return loss / B

def _center_gram(G):
    n = G.size(0)
    I = torch.eye(n, device=G.device, dtype=G.dtype)
    H = I - torch.ones((n, n), device=G.device, dtype=G.dtype) / n
    return H @ G @ H

def loss_cka_linear(Z1, Z2):
    # linear CKA between token embeddings (per sample), return 1-CKA
    B = Z1.size(0)
    total = 0.0
    for b in range(B):
        X, Y = Z1[b], Z2[b]                   # [P,D]
        Gx = X @ X.t()
        Gy = Y @ Y.t()
        Gx_c = _center_gram(Gx)
        Gy_c = _center_gram(Gy)
        hsic = (Gx_c * Gy_c).sum()
        varx = (Gx_c * Gx_c).sum().clamp_min(1e-8).sqrt()
        vary = (Gy_c * Gy_c).sum().clamp_min(1e-8).sqrt()
        cka = hsic / (varx * vary + 1e-12)
        total = total + (1.0 - cka)
    return total / B

# -----------------------------
# Sinkhorn OT with cosine cost
# -----------------------------
def sinkhorn_ot_cost(Z1, Z2, reg=0.1, iters=50, pos_w=0.0, pos_xy1=None, pos_xy2=None):
    """
    Z1: [B,P1,D], Z2: [B,P2,D] token-normalized
    reg: entropic regularization (epsilon)
    pos_w: optional positional penalty weight on (i,j) match using grid coords in [0,1]^2
    Returns average OT cost over batch.
    """
    B, P1, D = Z1.shape
    P2 = Z2.size(1)
    u_uniform = torch.full((B, P1), 1.0 / P1, device=Z1.device, dtype=Z1.dtype)
    v_uniform = torch.full((B, P2), 1.0 / P2, device=Z1.device, dtype=Z1.dtype)
    total_cost = 0.0

    for b in range(B):
        x = Z1[b]                              # [P1,D]
        y = Z2[b]                              # [P2,D]
        # cosine distance
        C = (1.0 - x @ y.t())                  # [P1,P2]
        if pos_w > 0.0 and (pos_xy1 is not None) and (pos_xy2 is not None):
            Pxy = torch.cdist(pos_xy1, pos_xy2, p=2)  # [P1,P2]
            C = C + pos_w * Pxy

        # Sinkhorn in log-domain
        K = torch.exp(-C / reg)                # kernel
        # Avoid divide-by-zero
        K = K + 1e-12
        u = torch.ones(P1, device=Z1.device, dtype=Z1.dtype) / P1
        v = torch.ones(P2, device=Z1.device, dtype=Z1.dtype) / P2
        for _ in range(iters):
            u = u_uniform[b] / (K @ v + 1e-12)
            v = v_uniform[b] / (K.t() @ u + 1e-12)
        T = torch.diag(u) @ K @ torch.diag(v)  # [P1,P2]
        total_cost = total_cost + (T * C).sum()
    return total_cost / B

# -----------------------------
# Patch-level InfoNCE (aligned)
# -----------------------------
def loss_patch_nce(Z1, Z2, tau=0.07):
    """
    Z1,Z2: [B,P,D], aligned by position; negatives are all Z2 tokens in the same batch item.
    """
    B, P, D = Z1.shape
    loss = 0.0
    for b in range(B):
        z = Z1[b]              # [P,D]
        y = Z2[b]              # [P,D]
        logits = (z @ y.t()) / tau   # [P,P]
        labels = torch.arange(P, device=Z1.device)
        loss = loss + F.cross_entropy(logits, labels)
    return loss / B

# -----------------------------
# Optional: make token coordinates (for OT pos term)
# -----------------------------
def make_token_xy_grid(visual, H, W, drop_cls=True, device="cpu", dtype=torch.float32):
    gy, gx = _infer_grid_size(visual, H, W)
    ys = torch.linspace(0, 1, steps=gy, device=device, dtype=dtype)
    xs = torch.linspace(0, 1, steps=gx, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    coords = torch.stack([yy, xx], dim=-1).view(-1, 2)  # [P,2]
    if not drop_cls:
        coords = torch.cat([torch.tensor([[0.5, 0.5]], device=device, dtype=dtype), coords], dim=0)
    return coords

# -----------------------------
# Build CLIP encoder (unchanged) + expose model/size/mean/std to the loop
# -----------------------------
def build_clip_encoder_losses(device, model_name: Optional[str] = None, pretrained: Optional[str] = None) -> "BaseEncoder":
    import open_clip  # lazy import
    model_name = model_name or "ViT-B-32"
    pretrained = pretrained or "openai"
    print(f"[enc] Loading open_clip (attackvlm_default): {model_name} ({pretrained})")
    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
    # Infer input size
    H = W = 224
    vis = getattr(model, "visual", None)
    if vis is not None:
        if hasattr(vis, "image_size"):
            sz = vis.image_size
            if isinstance(sz, (tuple, list)) and len(sz) == 2:
                H, W = int(sz[0]), int(sz[1])
            elif isinstance(sz, (int, float)):
                H = W = int(sz)
        elif hasattr(vis, "input_resolution"):
            H = W = int(getattr(vis, "input_resolution"))
    mean = [0.48145466, 0.4578275, 0.40821073]
    std  = [0.26862954, 0.26130258, 0.27577711]
    def _encode_fn(x_norm):
        return model.encode_image(x_norm)
    # Construct BaseEncoder as before
    enc = BaseEncoder("clip", model, (H, W), mean, std, device, _encode_fn)
    return enc

# -----------------------------
# Modified training loop with token losses
# -----------------------------
def train_attackvlm_losses(args, loader, dev, ann_writer):
    """
    Adds token-space objectives on ViT hidden patch embeddings.
    New/updated args:
      - loss_mode: one of ["global", "token_l2", "token_cos", "gram", "coral", "cka", "ot", "nce", "combo"]
      - vit_layer: int (e.g., -2) which transformer block to read tokens from
      - drop_cls: bool (default True) to drop CLS token for token objectives
      - token_downsample: int (optional). If >0, randomly sample this many tokens per image for OT/NCE/Gram/CKA to speed up
      - nce_tau: float temperature for InfoNCE (default 0.07)
      - ot_reg: float entropic regularization epsilon for Sinkhorn (default 0.1)
      - ot_iters: int Sinkhorn iterations (default 50)
      - ot_pos_w: float positional cost weight (default 0.0=off)
      - combo_weights: dict-like or "w_gram,w_cka,w_ot,w_coral,w_cos,w_l2,w_nce" string of floats
      - alpha, epsilon, pgd_steps: as before; PGD is gradient descent on a loss now (not ascent on similarity)
    """
    # Build encoder(s) (CLIP only in this snippet)
    clip_name = getattr(args, "attackvlm_model_name", "ViT-B-32")
    clip_pt   = getattr(args, "attackvlm_pretrained", "openai")
    clip_enc = build_clip_encoder_losses(dev, clip_name, clip_pt)
    encoders = [clip_enc]
    # Encoder weights (kept for future multi-encoder extension)
    weights = [1.0]
    weights_tensor = torch.tensor(weights, dtype=torch.float32, device=dev)
    print(f"[enc] Using encoder weights: {weights}")

    # PGD settings
    alpha = getattr(args, "alpha", 4.0) / 255.0
    epsilon = getattr(args, "epsilon", 8.0) / 255.0
    steps = getattr(args, "pgd_steps", 300)

    # Token-loss args
    loss_mode = getattr(args, "loss_mode", "global")
    vit_layer = int(getattr(args, "vit_layer", -2))
    drop_cls  = bool(getattr(args, "drop_cls", True))
    token_downsample = int(getattr(args, "token_downsample", 0))
    nce_tau = float(getattr(args, "nce_tau", 0.07))
    ot_reg = float(getattr(args, "ot_reg", 0.1))
    ot_iters = int(getattr(args, "ot_iters", 50))
    ot_pos_w = float(getattr(args, "ot_pos_w", 0.0))

    # Combo weights
    combo_w = getattr(args, "combo_weights", None)
    if isinstance(combo_w, str):
        parts = [p.strip() for p in combo_w.split(",")]
        # order: gram, cka, ot, coral, cos, l2, nce
        defaults = [1.0, 1.0, 1.0, 0.5, 0.0, 0.0, 0.0]
        nums = [float(parts[i]) if i < len(parts) and parts[i] != "" else defaults[i] for i in range(7)]
        combo_weights = dict(gram=nums[0], cka=nums[1], ot=nums[2], coral=nums[3], cos=nums[4], l2=nums[5], nce=nums[6])
    elif isinstance(combo_w, dict):
        combo_weights = combo_w
    else:
        combo_weights = dict(gram=1.0, cka=1.0, ot=1.0, coral=0.5, cos=0.0, l2=0.0, nce=0.0)

    out_root = Path(args.output).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    total_batches = (len(loader.dataset) + getattr(args, "batch_size", 1) - 1) // max(1, getattr(args, "batch_size", 1))

    # Optional CLE root for class folders
    cle_root = getattr(args, "cle_data_root", None)
    if cle_root is None:
        try:
            cle_root = CLE_DATA_ROOT  # type: ignore[name-defined]
        except Exception:
            cle_root = None
    if cle_root is not None:
        cle_root = Path(cle_root).resolve()

    for i, batch in enumerate(loader):
        clean_img_255, clean_text, tgt_img_255, tgt_text, clean_abs_paths = batch
        clean_img_255 = clean_img_255.to(dev)  # float32 0..255
        tgt_img_255   = tgt_img_255.to(dev)

        x0 = (clean_img_255 / 255.0).clamp(0.0, 1.0)
        if tgt_img_255.dim() == 4:
            tgt_img_255 = tgt_img_255.unsqueeze(1)  # [B,1,C,H,W]
        elif tgt_img_255.dim() != 5:
            raise RuntimeError(f"Expected tgt_img_255 to be [B,K,C,H,W] or [B,C,H,W], got shape {tuple(tgt_img_255.shape)}. Set stack_targets=True.")

        x_tg = (tgt_img_255 / 255.0).clamp(0.0, 1.0)  # [B,K,C,H,W]
        B, K = x_tg.size(0), x_tg.size(1)

        # Cache target features/tokens
        with torch.no_grad():
            cached_targets_global = []
            cached_targets_tokens = []  # list of [B,K,P,D]
            x_tg_flat = x_tg.view(B * K, *x_tg.shape[2:])  # [B*K,C,H,W]
            for enc in encoders:
                H, W = enc.image_size
                x_tg_norm = _resize_norm(x_tg_flat, (H, W), enc.mean, enc.std)
                # Global embedding (original behavior)
                f_tg_flat = enc.model.encode_image(x_tg_norm)   # [B*K, Dg]
                f_tg_flat = F.normalize(f_tg_flat, dim=-1)
                f_tg = f_tg_flat.view(B, K, -1)                 # [B,K,Dg]
                cached_targets_global.append(f_tg)               # keep all per-target embeddings

                if loss_mode != "global":  # token losses need tokens
                    # Extract tokens at vit_layer
                    Z_flat = extract_openclip_tokens(enc.model.visual, x_tg_norm, layer_idx=vit_layer, drop_cls=drop_cls, l2_token_norm=True)  # [B*K,P,D]
                    Z = Z_flat.view(B, K, Z_flat.size(1), Z_flat.size(2))  # [B,K,P,D]
                    if token_downsample and Z.size(2) > token_downsample:
                        # random subset of P tokens per image
                        P = Z.size(2)
                        idx = torch.randperm(P, device=Z.device)[:token_downsample]
                        Z = Z[:, :, idx, :]
                    cached_targets_tokens.append(Z)

        # One perturbation for the whole batch
        delta = torch.zeros_like(x0, requires_grad=True)

        for j in range(steps):
            x_adv = (x0 + delta).clamp(0.0, 1.0)
            total_loss = 0.0
            per_enc_logs = []

            for k_enc, enc in enumerate(encoders):
                H, W = enc.image_size
                x_adv_norm = _resize_norm(x_adv, (H, W), enc.mean, enc.std)

                if loss_mode == "global":
                    f_adv = enc.model.encode_image(x_adv_norm)            # [B,D]
                    f_adv = F.normalize(f_adv, dim=-1)
                    # Sum cosine against all K target embeddings then mean over batch
                    f_tg = cached_targets_global[k_enc]                   # [B,K,D]
                    sim_bk = torch.einsum("bd,bkd->bk", f_adv, f_tg)      # [B,K]
                    # We want image1 to look like all targets in K; maximize sum cosine => minimize negative
                    loss_k = -(sim_bk.sum(dim=1).mean())
                    per_enc_logs.append({"mode": "global", "loss": loss_k.detach().item()})
                    total_loss = total_loss + weights_tensor[k_enc] * loss_k
                    continue

                # Token-space branch
                Z1 = extract_openclip_tokens(enc.model.visual, x_adv_norm, layer_idx=vit_layer, drop_cls=drop_cls, l2_token_norm=True)  # [B,P,D]
                if token_downsample and Z1.size(1) > token_downsample:
                    P = Z1.size(1)
                    idx = torch.randperm(P, device=Z1.device)[:token_downsample]
                    Z1 = Z1[:, idx, :]

                Z2_all = cached_targets_tokens[k_enc]  # [B,K,P,D] (P may equal Z1.size(1))
                # Aggregate across K targets: mean over K gives a "prototype" target set
                Z2 = Z2_all.mean(dim=1)  # [B,P,D] if token_downsample used consistently

                # Positional grid for optional OT positional prior
                pos1 = pos2 = None
                if ot_pos_w > 0.0:
                    pos = make_token_xy_grid(enc.model.visual, H, W, drop_cls=drop_cls, device=Z1.device, dtype=Z1.dtype)
                    if token_downsample and pos.size(0) > Z1.size(1):
                        # Apply same random subsampling index as Z1 (approximation: re-sample fresh)
                        Ppos = pos.size(0)
                        idxp = torch.randperm(Ppos, device=Z1.device)[:Z1.size(1)]
                        pos1 = pos[idxp]
                        pos2 = pos[idxp] if Z2.size(1) == Z1.size(1) else pos[:Z2.size(1)]
                    else:
                        pos1 = pos
                        pos2 = pos if Z2.size(1) == Z1.size(1) else pos[:Z2.size(1)]

                if loss_mode == "token_l2":
                    loss_k = loss_token_l2(Z1, Z2)
                elif loss_mode == "token_cos":
                    loss_k = loss_token_cos(Z1, Z2)
                elif loss_mode == "gram":
                    loss_k = loss_gram(Z1, Z2)
                elif loss_mode == "coral":
                    loss_k = loss_coral(Z1, Z2)
                elif loss_mode == "cka":
                    loss_k = loss_cka_linear(Z1, Z2)
                elif loss_mode == "ot":
                    loss_k = sinkhorn_ot_cost(Z1, Z2, reg=ot_reg, iters=ot_iters, pos_w=ot_pos_w, pos_xy1=pos1, pos_xy2=pos2)
                elif loss_mode == "nce":
                    loss_k = loss_patch_nce(Z1, Z2, tau=nce_tau)
                elif loss_mode == "combo":
                    loss_k = 0.0
                    if combo_weights.get("gram", 0) > 0:  loss_k = loss_k + combo_weights["gram"]  * loss_gram(Z1, Z2)
                    if combo_weights.get("cka", 0) > 0:   loss_k = loss_k + combo_weights["cka"]   * loss_cka_linear(Z1, Z2)
                    if combo_weights.get("ot", 0) > 0:    loss_k = loss_k + combo_weights["ot"]    * sinkhorn_ot_cost(Z1, Z2, reg=ot_reg, iters=ot_iters, pos_w=ot_pos_w, pos_xy1=pos1, pos_xy2=pos2)
                    if combo_weights.get("coral", 0) > 0: loss_k = loss_k + combo_weights["coral"] * loss_coral(Z1, Z2)
                    if combo_weights.get("cos", 0) > 0:   loss_k = loss_k + combo_weights["cos"]   * loss_token_cos(Z1, Z2)
                    if combo_weights.get("l2", 0) > 0:    loss_k = loss_k + combo_weights["l2"]    * loss_token_l2(Z1, Z2)
                    if combo_weights.get("nce", 0) > 0:   loss_k = loss_k + combo_weights["nce"]   * loss_patch_nce(Z1, Z2, tau=nce_tau)
                else:
                    raise ValueError(f"Unknown loss_mode: {loss_mode}")

                per_enc_logs.append({"mode": loss_mode, "loss": loss_k.detach().item()})
                total_loss = total_loss + weights_tensor[k_enc] * loss_k

            # PGD update: minimize total_loss (gradient descent)
            total_loss.backward()
            with torch.no_grad():
                grad = delta.grad
                delta.sub_(alpha * torch.sign(grad))       # descent
                # Project onto L_inf ball and image bounds
                delta.clamp_(-epsilon, epsilon)
                delta.copy_((x0 + delta).clamp(0.0, 1.0) - x0)
                delta.grad.zero_()

            # Logging
            if (j % max(1, steps // 10) == 0) or (j == steps - 1):
                max_delta = torch.max(torch.abs(delta)).item()
                mean_delta = torch.mean(torch.abs(delta)).item()
                per_enc_str = ", ".join([f"{d['mode']}:{d['loss']:.5f}" for d in per_enc_logs])
                print(f"iter {i+1}/{total_batches} step:{j:3d} [{loss_mode}] total_loss={float(total_loss.item()):.5f}, {per_enc_str}, max|delta|={max_delta:.4f}, mean|delta|={mean_delta:.4f}")

        with torch.no_grad():
            adv_image_vis = (x0 + delta).clamp(0.0, 1.0)

        # Save under class folders and write annotation
        for b in range(adv_image_vis.size(0)):
            clean_abs = Path(clean_abs_paths[b]).resolve()
            class_name = clean_abs.parent.name
            if cle_root is not None:
                try:
                    rel = clean_abs.relative_to(cle_root)
                    parts = rel.parts
                    if len(parts) >= 1:
                        class_name = parts[0]
                except Exception:
                    pass
            class_dir = (out_root / class_name)
            class_dir.mkdir(parents=True, exist_ok=True)
            stem = clean_abs.stem
            out_path = (class_dir / f"{stem}.png").resolve()
            torchvision.utils.save_image(adv_image_vis[b], str(out_path))
            rec = {"image": str(out_path), "text": str(tgt_text[b])}
            if ann_writer is not None:
                ann_writer.write(rec)
            else:
                try:
                    annotations.append(rec)  # type: ignore[name-defined]
                except Exception:
                    pass



################################
################################
################################

############ growth
################################
import torch
import torch.nn.functional as F
from pathlib import Path
import torchvision
# -----------------------------
# Utilities
# -----------------------------
def _rand_positions(H, W, ph, pw, k, seed=0):
    import torch
    g = torch.Generator().manual_seed(seed)
    ys = torch.randint(0, max(1, H - ph + 1), (k,), generator=g).tolist()
    xs = torch.randint(0, max(1, W - pw + 1), (k,), generator=g).tolist()
    return list(zip(ys, xs))  # [(int y, int x), ...]

def _compose_with_patches(x01, patches, positions, ph, pw):
    """
    patches: list of (1,3,ph,pw) tensors (on any device/dtype)
    positions: list of (y, x) where y,x can be int or 0-dim tensors
    """
    import torch
    B, _, H, W = x01.shape
    x = x01

    def _to_int(v):
        if isinstance(v, int):
            return v
        if torch.is_tensor(v):
            if v.numel() != 1:
                raise TypeError(f"index must be scalar, got tensor with shape {tuple(v.shape)}")
            return int(v.item())
        return int(v)

    ph_i = _to_int(ph)
    pw_i = _to_int(pw)

    for i, p in enumerate(patches):
        y, x0 = positions[i]
        y = _to_int(y)
        x0 = _to_int(x0)

        # make sure everything is on the same device/dtype as x
        p_use = p.to(device=x.device, dtype=x.dtype).clamp(0, 1)

        pad_patch = torch.zeros((1, 3, H, W), device=x.device, dtype=x.dtype)
        pad_patch[:, :, y:y+ph_i, x0:x0+pw_i] = p_use

        m = torch.zeros((1, 1, H, W), device=x.device, dtype=x.dtype)
        m[:, :, y:y+ph_i, x0:x0+pw_i] = 1.0
        m3 = m.expand(1, 3, H, W)

        x = x * (1.0 - m3) + pad_patch * m3

    return x.clamp(0, 1)


def train_attackvlm_multiencoder_patchgrow(args, loader, dev, ann_writer):
    """
    Staged patch-growth attack (same signature as train_attackvlm_multiencoder):
      - Step 1: optimize 1 random patch
      - Step 2: add another random patch (optimize 2), ...
      - Step N: optimize N patches
    Uses the same multi-target cosine-sim objective as the PGD version, but edits only
    the pasted patch pixels. By default we keep CLIP as the sole encoder (like the
    provided code snippet).
    """

    clip_name = getattr(args, "attackvlm_model_name", "ViT-B-32")
    clip_pt   = getattr(args, "attackvlm_pretrained", "openai")
    device = dev if isinstance(dev, torch.device) else torch.device(dev)

    # Requires your existing helper; assumed available in your project
    clip_enc = build_clip_encoder(device, clip_name, clip_pt)  # noqa: F821

    encoders = [clip_enc]

    # Encoder weights
    def _parse_weights(x):
        if x is None:
            return [1.0]
        if isinstance(x, (list, tuple)):
            return [float(v) for v in x]
        if isinstance(x, str):
            return [float(v) for v in x.split(",")]
        return [float(x)]

    weights = _parse_weights(getattr(args, "encoder_weights", None))
    if len(weights) != len(encoders):
        weights = [1.0] * len(encoders)
    weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    print(f"[enc] Using encoder weights: {weights}")

    steps = int(getattr(args, "pgd_steps", 300))  # number of growth steps (patch count at the end)
    # Patch size and LR
    if hasattr(args, "patch_h") and hasattr(args, "patch_w"):
        ph, pw = int(args.patch_h), int(args.patch_w)
    else:
        ps = getattr(args, "patch_size", 32)
        ph = pw = int(ps) if isinstance(ps, int) else (int(ps[0]), int(ps[1]))[0]  # default square
        if not isinstance(ps, int):
            pw = int(ps[1])
    patch_lr = float(getattr(args, "patch_lr", getattr(args, "lr", 0.1)))
    seed = int(getattr(args, "seed", 0))
    multi_target_method = getattr(args, "multi_target_method", "mean")  # "mean" or "each"

    out_root = Path(args.output).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    total_batches = (len(loader.dataset) + getattr(args, "batch_size", 1) - 1) // max(1, getattr(args, "batch_size", 1))

    # Optional CLE root for class folders (same behavior as your snippet)
    cle_root = getattr(args, "cle_data_root", None)
    if cle_root is None:
        try:
            cle_root = CLE_DATA_ROOT  # type: ignore[name-defined]
        except Exception:
            cle_root = None
    if cle_root is not None:
        cle_root = Path(cle_root).resolve()

    for i, batch in enumerate(loader):
        clean_img_255, clean_text, tgt_img_255, tgt_text, clean_abs_paths = batch
        clean_img_255 = clean_img_255.to(device)  # float32 0..255
        tgt_img_255   = tgt_img_255.to(device)

        x0 = (clean_img_255 / 255.0).clamp(0.0, 1.0)

        # Expect stacked targets: [B,K,C,H,W]
        if tgt_img_255.dim() == 4:
            tgt_img_255 = tgt_img_255.unsqueeze(1)  # [B,1,C,H,W]
        elif tgt_img_255.dim() != 5:
            raise RuntimeError(
                f"Expected tgt_img_255 to be [B,K,C,H,W] or [B,C,H,W], "
                f"got shape {tuple(tgt_img_255.shape)}. Set stack_targets=True in your dataset."
            )
        x_tg = (tgt_img_255 / 255.0).clamp(0.0, 1.0)  # [B,K,C,H,W]
        B, K, C, H, W = x_tg.shape
        assert C == 3, "Expect 3-channel images"
        assert ph <= H and pw <= W, "patch_size must fit inside the image (H,W)"

        # Cache target embeddings per encoder (no grad), same logic as your snippet
        with torch.no_grad():
            cached_targets = []
            x_tg_flat = x_tg.view(B * K, C, H, W)
            for enc in encoders:
                f_tg_flat = enc.target_embed(x_tg_flat)  # [B*K, D], L2-normalized
                f_tg = f_tg_flat.view(B, K, -1)          # [B,K,D]
                if multi_target_method == "mean":
                    f_mean = f_tg.mean(dim=1)            # [B,D]
                    f_mean = F.normalize(f_mean, dim=-1)
                    cached_targets.append(f_mean)        # [B,D]
                else:  # "each"
                    cached_targets.append(f_tg)          # [B,K,D]

        # Initialize patch parameters and their random positions (shared across batch)
        positions = _rand_positions(H, W, ph, pw, k=steps, seed=seed + i)
        patches = [torch.rand(1, 3, ph, pw, device=device, requires_grad=True) for _ in range(steps)]

        # Optimizer: start with first patch; add new patch each step
        opt = torch.optim.Adam([{"params": [patches[0]], "lr": patch_lr}])

        for t in range(1, steps + 1):
            if t > 1:
                opt.add_param_group({"params": [patches[t - 1]], "lr": patch_lr})

            # Compose with first t patches
            x_adv = _compose_with_patches(x0, patches[:t], positions, ph, pw)

            # Sum cosine similarities across encoders (and across targets if method='each')
            total_sim = 0.0
            sims = []
            for k_idx, enc in enumerate(encoders):
                f_adv = enc.forward_embed(x_adv)  # [B,D], L2-normalized
                if multi_target_method == "mean":
                    sim_k_b = torch.sum(f_adv * cached_targets[k_idx], dim=1)  # [B]
                    sim_k = sim_k_b.mean()  # scalar
                else:
                    sim_bk = torch.einsum("bd,bkd->bk", f_adv, cached_targets[k_idx])  # [B,K]
                    sim_k = sim_bk.sum(dim=1).mean()  # sum over K, mean over B
                sims.append(sim_k.detach())
                total_sim = total_sim + weights_tensor[k_idx] * sim_k

            # Maximize total similarity (targeted objective)
            opt.zero_grad(set_to_none=True)
            total_sim.backward()

            # Optional: gradient normalization per-patch for stability
            for g in opt.param_groups:
                for p in g["params"]:
                    if p.grad is not None:
                        gnorm = p.grad.detach().flatten(1).norm(p=2, dim=1).clamp(min=1e-8)
                        p.grad.div_(gnorm.view(-1, 1, 1, 1))

            opt.step()

            # Keep patch pixels in valid range
            with torch.no_grad():
                for idx in range(t):
                    patches[idx].clamp_(0.0, 1.0)

            # Logging
            if (t % max(1, steps // 10) == 0) or (t == steps):
                sim_vals = [f"{s.item():.5f}" for s in sims]
                print(
                    f"iter {i+1}/{total_batches} step:{t:3d} [patchgrow/{multi_target_method}] "
                    f"total_sim={float(total_sim.item()):.5f}, sims={sim_vals}, "
                    f"patches={t}, patch_size=({ph},{pw}), lr={patch_lr}"
                )

        # Compose final adversarial visualization with all patches
        with torch.no_grad():
            adv_image_vis = _compose_with_patches(x0, patches, positions, ph, pw)

        # Save under class folders and write annotation (same as your snippet)
        for b in range(adv_image_vis.size(0)):
            clean_abs = Path(clean_abs_paths[b]).resolve()
            # Derive class name
            class_name = clean_abs.parent.name
            if cle_root is not None:
                try:
                    rel = clean_abs.relative_to(cle_root)
                    parts = rel.parts
                    if len(parts) >= 1:
                        class_name = parts[0]
                except Exception:
                    pass
            class_dir = (out_root / class_name)
            class_dir.mkdir(parents=True, exist_ok=True)
            stem = clean_abs.stem
            out_path = (class_dir / f"{stem}.png").resolve()
            torchvision.utils.save_image(adv_image_vis[b], str(out_path))
            rec = {"image": str(out_path), "text": str(tgt_text[b])}
            if ann_writer is not None:
                ann_writer.write(rec)
            else:
                # Optional global 'annotations' support
                try:
                    annotations.append(rec)  # type: ignore[name-defined]
                except Exception:
                    pass