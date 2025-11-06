# models.py
from typing import Optional, List, Tuple
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import clip
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# ----------------------- Caption model (unchanged behavior) -----------------------
class MLP(nn.Module):
    def __init__(self, sizes, bias=True, act=nn.Tanh):
        super().__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        return self.model(x)

class ClipCaptionModel(nn.Module):
    def __init__(self, prefix_length: int, prefix_size: int = 512):
        super().__init__()
        self.prefix_length = prefix_length
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        if prefix_length > 10:
            self.clip_project = nn.Linear(prefix_size, self.gpt_embedding_size * prefix_length)
        else:
            self.clip_project = MLP(
                (prefix_size,
                 (self.gpt_embedding_size * prefix_length) // 2,
                 self.gpt_embedding_size * prefix_length)
            )
    def get_dummy_token(self, batch_size: int, device: torch.device):
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)
    def forward(self, tokens, prefix, mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None):
        embedding_text = self.gpt.transformer.wte(tokens)
        prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out

class ClipCaptionPrefix(ClipCaptionModel):
    def parameters(self, recurse: bool = True):
        return self.clip_project.parameters()
    def train(self, mode: bool = True):
        super().train(mode)
        self.gpt.eval()
        return self

# ----------------------- Generation (same sampler as original) -----------------------
def generate_cap(model: ClipCaptionModel, tokenizer, embed, entry_length=67, top_p=0.8, temperature=1.0, stop_token='.'):
    model.eval()
    stop_token_index = tokenizer.encode(stop_token)[0]
    filter_value = -float("Inf")
    tokens = None
    with torch.no_grad():
        generated = embed
        for _ in range(entry_length):
            outputs = model.gpt(inputs_embeds=generated)
            logits = outputs.logits
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[:, indices_to_remove] = filter_value
            next_token = torch.argmax(logits, -1).unsqueeze(0)
            next_token_embed = model.gpt.transformer.wte(next_token)
            tokens = next_token if tokens is None else torch.cat((tokens, next_token), dim=1)
            generated = torch.cat((generated, next_token_embed), dim=1)
            if stop_token_index == next_token.item():
                break
        output_list = list(tokens.squeeze().cpu().numpy())
        output_text = tokenizer.decode(output_list)
    return output_text

def generate_cap_batch(
        model,
        tokenizer,
        embeds,  # Tensor: [batch_size, prefix_size]
        entry_length=67,
        top_p=0.8,
        temperature=1.,
        stop_token='.',
):
    """
    Generate captions for a batch of image embeddings.

    Args:
        model: The ClipCaptionModel.
        tokenizer: The GPT2 tokenizer.
        embeds (torch.Tensor): Batch of CLIP image embeddings [batch_size, prefix_size].
        entry_length (int): Max caption length (in tokens).
        top_p (float): Nucleus sampling parameter.
        temperature (float): Softmax temperature.
        stop_token (str): Stop generation when this token is reached.

    Returns:
        List[str]: List of generated captions, length == batch_size.
    """
    model.eval()
    device = next(model.parameters()).device
    batch_size = embeds.size(0)
    stop_token_index = tokenizer.encode(stop_token)[0]
    filter_value = -float("Inf")

    with torch.no_grad():
        # Project CLIP embeddings to GPT2 prefix embeddings
        prefix_embed = model.clip_project(embeds).view(batch_size, model.prefix_length, model.gpt_embedding_size)

        # Initialize token embeddings (start from prefix only)
        generated = prefix_embed
        tokens = model.get_dummy_token(batch_size, device)

        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        results = [""] * batch_size

        for _ in range(entry_length):
            outputs = model.gpt(inputs_embeds=generated)
            logits = outputs.logits[:, -1, :] / temperature

            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # Recreate logits so they match original order
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )

            logits = logits.masked_fill(indices_to_remove, filter_value)

            next_tokens = torch.argmax(logits, -1)
            next_token_embeds = model.gpt.transformer.wte(next_tokens.unsqueeze(1))

            tokens = torch.cat((tokens, next_tokens.unsqueeze(1)), dim=1)
            generated = torch.cat((generated, next_token_embeds), dim=1)

            newly_finished = next_tokens == stop_token_index
            finished |= newly_finished
            if finished.all():
                break

        # Decode all captions
        for i in range(batch_size):
            output_list = tokens[i].squeeze().cpu().numpy().tolist()
            results[i] = tokenizer.decode(output_list, skip_special_tokens=True).replace("!", "")

    return results

# ----------------------- Surrogate base -----------------------
class SurrogateBase:
    name: str = "base"

    @property
    def max_context_length(self) -> int:
        raise NotImplementedError

    def tokenize_text(self, texts: List[str], context_length: Optional[int] = None) -> torch.Tensor:
        raise NotImplementedError

    # For static branches (no grads) convenience
    def encode_image_text_features_normalized(
        self,
        image_0_255: torch.Tensor,
        text_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    # For PGD inner loop (need grads wrt image)
    def encode_image_features_normalized(
        self,
        image_0_255: torch.Tensor,
        requires_grad: bool
    ) -> torch.Tensor:
        raise NotImplementedError

    def encode_text_features_normalized(self, text_tokens: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def generate_caption_from_image_tensor_0_255(self, img_0_255: torch.Tensor) -> str:
        raise NotImplementedError

# ----------------------- CLIP surrogate implementation -----------------------
def _build_clip_feature_preprocess(clip_model) -> torchvision.transforms.Compose:
    return torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(
                clip_model.visual.input_resolution,
                interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
                antialias=True,
            ),
            torchvision.transforms.Lambda(lambda img: torch.clamp(img, 0.0, 255.0) / 255.0),
            torchvision.transforms.CenterCrop(clip_model.visual.input_resolution),
            torchvision.transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )

@dataclass
class ClipSurrogateConfig:
    clip_encoder: str = "ViT-B/32"
    prefix_length: int = 10
    model_path: str = ""  # path to caption model weights

class ClipSurrogate(SurrogateBase):
    name: str = "clip"

    def __init__(self, cfg: ClipSurrogateConfig, dev: torch.device):
        self.dev = dev
        self.clip_model, self.clip_preprocess_for_caption = clip.load(cfg.clip_encoder, device=self.dev, jit=False)
        self.feature_preprocess = _build_clip_feature_preprocess(self.clip_model)

        self.prefix_length = cfg.prefix_length
        self.caption_model = ClipCaptionModel(self.prefix_length).to(self.dev)
        altered_state_dict = torch.load(cfg.model_path, map_location="cpu")
        for i in range(12):
            altered_state_dict.pop(f"gpt.transformer.h.{i}.attn.bias", None)
            altered_state_dict.pop(f"gpt.transformer.h.{i}.attn.masked_bias", None)
        self.caption_model.load_state_dict(altered_state_dict, strict=False)
        self.caption_model.eval()
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self._to_pil = torchvision.transforms.ToPILImage()

    @property
    def max_context_length(self) -> int:
        return getattr(self.clip_model, "context_length", 77)

    def tokenize_text(self, texts: List[str], context_length: Optional[int] = None) -> torch.Tensor:
        max_ctx = self.max_context_length
        ctx = max_ctx if context_length is None else min(context_length, max_ctx)
        try:
            toks = clip.tokenize(texts, context_length=ctx, truncate=True)
        except TypeError:
            print("do not support truncate")
            toks = clip.tokenize(texts, context_length=ctx)
        return toks.to(self.dev)

    # Static branches helper (no grads)
    @torch.no_grad()
    def encode_image_text_features_normalized(
        self,
        image_0_255: torch.Tensor,
        text_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        img_features = self.clip_model.encode_image(self.feature_preprocess(image_0_255))
        img_features = img_features / img_features.norm(dim=1, keepdim=True)
        txt_features = self.clip_model.encode_text(text_tokens)
        txt_features = txt_features / txt_features.norm(dim=1, keepdim=True)
        return img_features, txt_features

    # For PGD — allow grads wrt image
    def encode_image_features_normalized(
        self,
        image_0_255: torch.Tensor,
        requires_grad: bool
    ) -> torch.Tensor:
        x = self.feature_preprocess(image_0_255)
        # No torch.no_grad() here when requires_grad=True, to keep the graph
        feats = self.clip_model.encode_image(x)
        feats = feats / feats.norm(dim=1, keepdim=True)
        return feats

    @torch.no_grad()
    def encode_text_features_normalized(self, text_tokens: torch.Tensor) -> torch.Tensor:
        t = self.clip_model.encode_text(text_tokens)
        t = t / t.norm(dim=1, keepdim=True)
        return t

    @torch.no_grad()
    def generate_caption_from_image_tensor_0_255(self, img_0_255: torch.Tensor) -> str:
        # Match original: clamp to [0,1] -> PIL -> preprocess from clip.load
        img_01 = torch.clamp(img_0_255 / 255.0, 0.0, 1.0)
        pil_img = self._to_pil(img_01[0].cpu())
        processed_img = self.clip_preprocess_for_caption(pil_img).unsqueeze(0).to(self.dev)
        prefix = self.clip_model.encode_image(processed_img).to(self.dev, dtype=torch.float32)
        prefix_embed = self.caption_model.clip_project(prefix).reshape(1, self.prefix_length, -1)
        return generate_cap(self.caption_model, self.tokenizer, embed=prefix_embed)

    @torch.no_grad()
    def generate_caption_from_batch(self, pil_images: list) -> list[str]:
        self.caption_model.eval()
        self.clip_model.eval()

        # Preprocess all images for CLIP in a batch
        processed_imgs = torch.stack(
            [self.clip_preprocess_for_caption(pil_img) for pil_img in pil_images]
        ).to(self.dev)

        with torch.no_grad():
            # Encode each image into CLIP embeddings
            prefixes = self.clip_model.encode_image(processed_imgs)
            prefixes = prefixes.to(self.dev, dtype=torch.float32)

            # Generate captions in batch
            captions = generate_cap_batch(
                model=self.caption_model,
                tokenizer=self.tokenizer,
                embeds=prefixes,
                entry_length=50,
                top_p=0.8,
                temperature=1.0,
            )

        return captions



# --- registry update ---
def build_surrogate(name: str, args, dev: torch.device) -> SurrogateBase:
    name = name.lower()
    if name == "clip":
        cfg = ClipSurrogateConfig(
            clip_encoder=args.clip_encoder,
            prefix_length=args.prefix_length,
            model_path=args.model_path,
        )
        return ClipSurrogate(cfg, dev)

    raise ValueError(f"Unknown surrogate: {name}")