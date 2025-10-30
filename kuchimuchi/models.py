# models.py
from dataclasses import dataclass
from typing import Callable, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
import open_clip


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FrozenLlavaProjector(nn.Module):
    """
    Frozen image model: (OpenCLIP ViT-L/14 visual) -> (frozen MLP head) -> L2-normalized embedding.
    Expects normalized pixels per preprocess provided by ImageModelSpec.
    """
    def __init__(self, visual: nn.Module, head: nn.Module):
        super().__init__()
        self.visual = visual.eval()
        self.head = head.eval()
        for p in self.visual.parameters():
            p.requires_grad = False
        for p in self.head.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.visual(x)
        z = self.head(z)
        return F.normalize(z, dim=-1)


def build_llava_visual(device: torch.device) -> nn.Module:
    # Matches training: ("ViT-L-14", "laion2b_s32b_b82k")
    model, _, _ = open_clip.create_model_and_transforms("ViT-L-14", pretrained="laion2b_s32b_b82k")
    visual = model.visual.to(device).eval().float()
    for p in visual.parameters():
        p.requires_grad = False
    return visual


def load_proj_head_from_ckpt(ckpt_path: str, device: torch.device) -> ProjectionHead:
    sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    in_dim = sd["net.0.weight"].shape[1]
    hidden_dim = sd["net.0.weight"].shape[0]
    out_dim = sd["net.2.weight"].shape[0]
    head = ProjectionHead(in_dim, out_dim, hidden_dim).to(device)
    head.load_state_dict(sd, strict=True)
    head.eval()
    for p in head.parameters():
        p.requires_grad = False
    return head


def build_tensor_preprocess_224(mean, std):
    """
    Returns a function that takes images in [0,255] float [B,3,224,224] and outputs normalized tensors.
    """
    mean = torch.tensor(mean).view(1, 3, 1, 1)
    std = torch.tensor(std).view(1, 3, 1, 1)

    def _pp(x_255: torch.Tensor) -> torch.Tensor:
        x = x_255 / 255.0
        m = mean.to(x.device, x.dtype)
        s = std.to(x.device, x.dtype)
        return (x - m) / s

    return _pp


@dataclass
class ImageModelSpec:
    name: str
    model: nn.Module
    preprocess: Callable[[torch.Tensor], torch.Tensor]  # input [B,3,224,224] in [0,255]
    out_dim: int
    device: torch.device

    def encode(self, images_255: torch.Tensor) -> torch.Tensor:
        x = self.preprocess(images_255).to(self.device)
        return self.model(x).float()


def build_image_models(model_names: List[str], device: torch.device, llava_ckpt: str) -> List[ImageModelSpec]:
    """
    model_names includes:
      - "llava_proj"
      - "clip-openai:ViT-B/32"
      - "openclip-enc:ViT-B-32,openai"
    """
    specs: List[ImageModelSpec] = []
    clip_mean = (0.48145466, 0.4578275, 0.40821073)
    clip_std = (0.26862954, 0.26130258, 0.27577711)

    for name in model_names:
        name = name.strip()
        if not name:
            continue

        if name == "llava_proj":
            if not llava_ckpt:
                raise ValueError("llava_proj requires llava_ckpt path.")
            visual = build_llava_visual(device)
            head = load_proj_head_from_ckpt(llava_ckpt, device)
            with torch.no_grad():
                in_dim = visual(torch.randn(1, 3, 224, 224, device=device)).shape[-1]
                head_in = next(head.net[0].parameters()).shape[1]
                if in_dim != head_in:
                    raise RuntimeError(f"LLaVA visual dim {in_dim} != MLP input dim {head_in}")
                out_dim = next(head.net[2].parameters()).shape[0]
            frozen = FrozenLlavaProjector(visual, head).to(device).eval().float()
            preprocess = build_tensor_preprocess_224(clip_mean, clip_std)
            specs.append(ImageModelSpec(name="llava_proj", model=frozen, preprocess=preprocess, out_dim=out_dim, device=device))

        elif name.startswith("clip-openai:"):
            arch = name.split(":", 1)[1]
            clip_model, _ = clip.load(arch, device=device, jit=False)
            clip_model.eval().float()
            for p in clip_model.parameters():
                p.requires_grad = False

            class _Wrap(nn.Module):
                def __init__(self, m): super().__init__(); self.m = m
                def forward(self, x):
                    z = self.m.encode_image(x)
                    return F.normalize(z, dim=-1)

            wrapper = _Wrap(clip_model).to(device).eval()
            with torch.no_grad():
                out_dim = wrapper(torch.randn(1, 3, 224, 224, device=device)).shape[-1]
            preprocess = build_tensor_preprocess_224(clip_mean, clip_std)
            specs.append(ImageModelSpec(name=name, model=wrapper, preprocess=preprocess, out_dim=out_dim, device=device))

        elif name.startswith("openclip-enc:"):
            spec = name.split(":", 1)[1]
            parts = [p.strip() for p in spec.split(",")]
            arch = parts[0]
            pretrained = parts[1] if len(parts) > 1 else "openai"
            oc_model, _, _ = open_clip.create_model_and_transforms(arch, pretrained=pretrained, device=device)
            oc_model.eval().float()
            for p in oc_model.parameters():
                p.requires_grad = False

            class _Wrap(nn.Module):
                def __init__(self, m): super().__init__(); self.m = m
                def forward(self, x):
                    z = self.m.encode_image(x)
                    return F.normalize(z, dim=-1)

            wrapper = _Wrap(oc_model).to(device).eval()
            with torch.no_grad():
                out_dim = wrapper(torch.randn(1, 3, 224, 224, device=device)).shape[-1]
            preprocess = build_tensor_preprocess_224(clip_mean, clip_std)
            specs.append(ImageModelSpec(name=name, model=wrapper, preprocess=preprocess, out_dim=out_dim, device=device))

        else:
            raise ValueError(f"Unknown image model spec: {name}")

    # Enforce single shared text space
    if len({s.out_dim for s in specs}) != 1:
        raise RuntimeError(f"All image models must output the same dim. Got: {[s.out_dim for s in specs]}")
    return specs