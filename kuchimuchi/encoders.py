# encoders.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip           # OpenAI CLIP
import open_clip      # OpenCLIP
from typing import List, Optional

class TextEncoder:
    """
    Wrapper around a CLIP-like text encoder.
    name:
      - "openai:ViT-B/32"
      - "openclip:ViT-B-32,openai" (or other OpenCLIP pretrained tags)
    """
    def __init__(self, name: str, device: torch.device):
        self.name = name
        self.device = device
        self._kind = None
        self._model = None
        self._tokenizer = None
        self.embed_dim = None
        if name.startswith("openai:"):
            self._kind = "openai"
            arch = name.split(":", 1)[1]
            model, _ = clip.load(arch, device=device, jit=False)
            self._model = model.eval().float()
            self._tokenizer = clip.tokenize
            with torch.no_grad():
                d = self._model.encode_text(self._tokenizer(["probe"]).to(device)).shape[-1]
            self.embed_dim = d
        elif name.startswith("openclip:"):
            self._kind = "openclip"
            spec = name.split(":", 1)[1]
            parts = [p.strip() for p in spec.split(",")]
            arch = parts[0]
            pretrained = parts[1] if len(parts) > 1 else "openai"
            model, _, _ = open_clip.create_model_and_transforms(arch, pretrained=pretrained, device=device)
            self._model = model.eval().float()
            self._tokenizer = open_clip.get_tokenizer(arch)
            with torch.no_grad():
                tok = self._tokenizer(["probe"])
                d = self._model.encode_text(tok.to(device)).shape[-1]
            self.embed_dim = d
        else:
            raise ValueError(f"Unsupported text encoder: {name}")
        for p in self._model.parameters():
            p.requires_grad = False
    @torch.no_grad()
    def encode(self, texts: List[str], batch_size: int = 256) -> torch.Tensor:
        feats = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            toks = self._tokenizer(chunk).to(self.device)
            z = self._model.encode_text(toks)
            z = F.normalize(z, dim=-1)
            feats.append(z.float())
        return torch.cat(feats, dim=0)

def build_text_encoder(name: str, device: torch.device) -> TextEncoder:
    return TextEncoder(name=name, device=device)

EOT_ID_OPENAI = 49407  # EOT token id in OpenAI CLIP BPE
class PromptedTextEncoder(nn.Module):
    """
    Soft-prompt wrapper for OpenAI CLIP text encoder.
    - Frozen CLIP; only 'prompt' is learnable.
    - Works with name like 'openai:ViT-B/32'.
    - K is the number of learnable prompt tokens.
    """
    def __init__(self, name: str, device: torch.device, K: int, init_text: Optional[str] = None,
                 init_std: float = 0.02):
        super().__init__()
        assert name.startswith("openai:"), "PromptedTextEncoder here supports openai:* (e.g., openai:ViT-B/32)."
        self.name = name
        self.device = device
        self.K = int(K)
        arch = name.split(":", 1)[1]
        model, _ = clip.load(arch, device=device, jit=False)
        self.model = model.eval().float()  # keep frozen
        for p in self.model.parameters():
            p.requires_grad = False
        # Dimensions
        self.context_length = self.model.context_length  # typically 77
        self.width = self.model.token_embedding.weight.shape[1]  # d_model
        self.embed_dim = self.model.text_projection.shape[1]     # CLIP text embed dim
        # Learnable prompt parameters (global, shared across all samples)
        prompt = torch.empty(self.K, self.width, device=device)
        if self.K > 0:
            if init_text is not None:
                with torch.no_grad():
                    toks = clip.tokenize([init_text]).to(device)  # [1, L]
                    eot_pos = (toks == EOT_ID_OPENAI).nonzero(as_tuple=False)[0, 1].item()
                    # Use average of real token embeddings (excluding SOS at 0 and EOT)
                    if eot_pos > 1:
                        emb = self.model.token_embedding(toks)  # [1, L, d]
                        base = emb[:, 1:eot_pos, :].mean(dim=1, keepdim=False)  # [1, d]
                        prompt.copy_(base.repeat(self.K, 1))
                    else:
                        torch.nn.init.normal_(prompt, std=init_std)
            else:
                torch.nn.init.normal_(prompt, std=init_std)
        self.prompt = nn.Parameter(prompt)  # the only trainable parameter
        # Tokenizer
        self._tokenizer = clip.tokenize
    def _encode_tokens_with_prompt(self, toks: torch.Tensor) -> torch.Tensor:
        """
        toks: [B, L] token ids from clip.tokenize, L == context_length
        returns: [B, embed_dim] normalized features
        """
        B, L = toks.shape
        assert L == self.context_length, "Unexpected token length."
        # Budget for content after inserting K prompts (leave one slot for SOS)
        keep_len = max(1, self.context_length - 1 - self.K)  # >=1 to hold at least EOT
        # Take content (excluding SOS at 0), truncate to keep_len, and force last token to EOT
        content = toks[:, 1:1 + keep_len].clone()  # [B, keep_len]
        content[:, -1] = EOT_ID_OPENAI
        # Embeddings
        sos = self.model.token_embedding(toks[:, 0:1])                   # [B,1,d]
        content_emb = self.model.token_embedding(content)                # [B,keep_len,d]
        if self.K > 0:
            P = self.prompt.unsqueeze(0).expand(B, -1, -1)               # [B,K,d]
            x = torch.cat([sos, P, content_emb], dim=1)                  # [B, 1+K+keep_len, d]
        else:
            x = torch.cat([sos, content_emb], dim=1)                     # [B, 1+keep_len, d]
        # Positional embeddings (slice to current sequence length)
        pos = self.model.positional_embedding[:x.size(1), :].unsqueeze(0)  # [1,S,d]
        x = x + pos
        # Transformer + pooling at last (EOT is last by construction)
        x = x.permute(1, 0, 2)              # [S,B,d]
        x = self.model.transformer(x)       # [S,B,d]
        x = x.permute(1, 0, 2)              # [B,S,d]
        x = self.model.ln_final(x)          # [B,S,d]
        x = x[:, -1, :]                     # [B,d] last position (EOT)
        x = x @ self.model.text_projection  # [B,embed_dim]
        x = F.normalize(x, dim=-1)
        return x.float()
    def encode(self, texts: List[str], batch_size: int = 256) -> torch.Tensor:
        """
        Encodes a list of texts with the learnable prompt. Gradients flow to self.prompt.
        """
        feats = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            toks = self._tokenizer(chunk).to(self.device)
            z = self._encode_tokens_with_prompt(toks)
            feats.append(z)
        return torch.cat(feats, dim=0)
def build_prompted_text_encoder(name: str, device: torch.device, K: int,
                                init_text: Optional[str] = None,
                                init_std: float = 0.02) -> PromptedTextEncoder:
    """
    Factory for PromptedTextEncoder (OpenAI CLIP only).
    """
    return PromptedTextEncoder(name=name, device=device, K=K, init_text=init_text, init_std=init_std)