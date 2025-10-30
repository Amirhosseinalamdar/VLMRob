# vlm_eval/vlms.py
import os
from dataclasses import dataclass
from typing import Optional, Dict, Any
import torch

# Base interface
@dataclass
class VLMOutput:
    text: str

class BaseVLM:
    name: str = "base"
    def generate(self, image_pil, prompt: Optional[str] = None, **kw) -> VLMOutput:
        raise NotImplementedError

def _device_and_dtype(device: Optional[str], dtype: Optional[str]):
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if dtype is None:
        dt = torch.float16 if dev.startswith("cuda") else torch.float32
    else:
        dt = getattr(torch, dtype)
    return dev, dt

# ---------- BLIP ----------
class BLIPCaptioner(BaseVLM):
    name = "BLIP"
    def __init__(self, model_id: str = "Salesforce/blip-image-captioning-large", device=None, dtype=None):
        from transformers import BlipProcessor, BlipForConditionalGeneration
        dev, dt = _device_and_dtype(device, dtype)
        self.processor = BlipProcessor.from_pretrained(model_id)
        self.model = BlipForConditionalGeneration.from_pretrained(model_id, torch_dtype=dt).to(dev)
        self.device = dev

    def generate(self, image_pil, prompt: Optional[str] = None, max_new_tokens=40, **kw):
        inputs = self.processor(images=image_pil, text=prompt, return_tensors="pt").to(self.device)
        out_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        text = self.processor.decode(out_ids[0], skip_special_tokens=True)
        return VLMOutput(text=text.strip())

# ---------- BLIP-2 ----------
class BLIP2Captioner(BaseVLM):
    name = "BLIP-2"
    def __init__(self, model_id: str = "Salesforce/blip2-flan-t5-xl", device=None, dtype=None):
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        dev, dt = _device_and_dtype(device, dtype)
        self.processor = Blip2Processor.from_pretrained(model_id)
        self.model = Blip2ForConditionalGeneration.from_pretrained(model_id, torch_dtype=dt, device_map="auto" if dev=="cuda" else None).to(dev)
        self.device = dev

    def generate(self, image_pil, prompt: Optional[str] = "Describe the image.", max_new_tokens=40, **kw):
        inputs = self.processor(images=image_pil, text=prompt, return_tensors="pt").to(self.device)
        out_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        text = self.processor.decode(out_ids[0], skip_special_tokens=True)
        return VLMOutput(text=text.strip())

# ---------- InstructBLIP ----------
class InstructBLIPCaptioner(BaseVLM):
    name = "InstructBLIP"
    def __init__(self, model_id: str = "Salesforce/instructblip-vicuna-7b", device=None, dtype=None):
        from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
        dev, dt = _device_and_dtype(device, dtype)
        self.processor = InstructBlipProcessor.from_pretrained(model_id)
        self.model = InstructBlipForConditionalGeneration.from_pretrained(model_id, torch_dtype=dt, device_map="auto" if dev=="cuda" else None).to(dev)
        self.device = dev

    def generate(self, image_pil, prompt: Optional[str] = "Describe the image.", max_new_tokens=80, **kw):
        inputs = self.processor(images=image_pil, text=prompt, return_tensors="pt").to(self.device)
        out_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        text = self.processor.decode(out_ids[0], skip_special_tokens=True)
        return VLMOutput(text=text.strip())

# ---------- LLaVA-1.5 (7B/13B) ----------
# class LLaVACaptioner(BaseVLM):
#     name = "LLaVA"
#     def __init__(self, model_id="llava-hf/llava-1.5-7b-hf", device=None, dtype="float16"):
#         import torch
#         from transformers import AutoProcessor, LlavaForConditionalGeneration

#         dev, dt = _device_and_dtype(device, dtype)  # your helper; dev can be "cuda" or "cpu"
#         self.processor = AutoProcessor.from_pretrained(model_id)
#         self.model = LlavaForConditionalGeneration.from_pretrained(
#             model_id,
#             torch_dtype=getattr(torch, dtype) if isinstance(dtype, str) else dtype,
#             device_map="auto" if dev == "cuda" else None,
#         )
#         self.device = dev

#     def generate(self, image_pil, prompt="Describe the image in detail.", max_new_tokens=80, **kw):
#         # LLaVA expects the <image> token in the chat template
#         conversation = [{
#             "role": "user",
#             "content": [{"type": "image"}, {"type": "text", "text": prompt}],
#         }]
#         text = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
#         inputs = self.processor(images=image_pil, text=text, return_tensors="pt")
#         inputs = {k: v.to(self.model.device, dtype=self.model.dtype) for k, v in inputs.items()}

#         with torch.no_grad():
#             out = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

#         # Decode only newly generated tokens (exclude the prompt part)
#         gen_only = out[:, inputs["input_ids"].shape[-1]:]
#         resp = self.processor.batch_decode(gen_only, skip_special_tokens=True)[0].strip()
#         return VLMOutput(text=resp)
class LLaVACaptioner(BaseVLM):
    name = "LLaVA"

    def __init__(self, model_id="llava-hf/llava-1.5-7b-hf", device=None, dtype="float16"):
        import torch
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        dev, dt = _device_and_dtype(device, dtype)  # your helper; dev can be "cuda" or "cpu"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=getattr(torch, dtype) if isinstance(dtype, str) else dtype,
            device_map="auto" if dev == "cuda" else None,
        )
        self.device = dev

    def generate(self, image_pil, prompt="Describe the image in detail.", max_new_tokens=80, **kw):
        import torch

        # LLaVA expects the <image> token in the chat template
        conversation = [{
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": prompt}],
        }]
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True)

        inputs = self.processor(images=image_pil, text=text, return_tensors="pt")

        # Only cast floating tensors; keep integer tensors as-is
        for k, v in inputs.items():
            if not isinstance(v, torch.Tensor):
                continue
            if v.dtype.is_floating_point:
                inputs[k] = v.to(self.model.device, dtype=self.model.dtype)  # e.g., pixel_values
            else:
                inputs[k] = v.to(self.model.device)  # e.g., input_ids, attention_mask

        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        # Decode only newly generated tokens (exclude the prompt part)
        gen_only = out[:, inputs["input_ids"].shape[-1]:]
        resp = self.processor.batch_decode(gen_only, skip_special_tokens=True)[0].strip()
        return VLMOutput(text=resp)
# ---------- LLaVA-NeXT (v1.6) ----------
class LLaVANextCaptioner(BaseVLM):
    name = "LLaVA-NeXT"
    def __init__(self, model_id: str = "llava-hf/llava-v1.6-mistral-7b-hf", device=None, dtype="float16"):
        import torch
        from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
        dev, dt = _device_and_dtype(device, dtype)
        self.processor = LlavaNextProcessor.from_pretrained(model_id)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(model_id, dtype=dt, device_map="auto" if dev=="cuda" else None)
        self.device = dev

    def generate(self, image_pil, prompt: Optional[str] = "Describe the image in detail.", max_new_tokens=80, **kw):
        conversation = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(images=image_pil, text=text, return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        resp = self.processor.decode(out[0], skip_special_tokens=True)
        return VLMOutput(text=resp.strip())

class Qwen25VLCaptioner(BaseVLM):
    name = "Qwen2.5-VL-7B"
    def __init__(self, model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct", device=None, dtype="auto"):
        import torch
        from transformers import AutoProcessor
        try:
            # Newer Transformers class name
            from transformers import Qwen2_5_VLForConditionalGeneration as QwenVLForConditionalGeneration
        except Exception:
            # Older alias (some releases used Qwen2VL or Qwen2VLForConditionalGeneration)
            from transformers import Qwen2VLForConditionalGeneration as QwenVLForConditionalGeneration
        # Try to use qwen-vl-utils if present
        try:
            from qwen_vl_utils import process_vision_info
            self.process_vision_info = process_vision_info
            self.use_qwen_utils = True
        except ImportError:
            self.process_vision_info = None
            self.use_qwen_utils = False

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = QwenVLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype="auto", device_map="auto"
        )
        self.device = self.model.device

    def generate(self, image_pil, prompt: str = "Describe the image in detail.", max_new_tokens: int = 80, **kw):
        # Build chat with an image turn
        messages = [{"role": "user", "content": [{"type": "image", "image": image_pil}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        if self.use_qwen_utils:
            images, videos = self.process_vision_info(messages)
        else:
            # Fallback: single-image only
            images, videos = [image_pil], None

        inputs = self.processor(text=[text], images=images, videos=videos, return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
        resp = self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]
        return VLMOutput(text=resp.strip())
        
# ---------- Img2Prompt (clip-interrogator) ----------
class Img2PromptCaptioner(BaseVLM):
    name = "Img2Prompt"
    def __init__(self, clip_model_name: str = "ViT-L-14/openai", device=None):
        from clip_interrogator import Config, Interrogator
        dev, _ = _device_and_dtype(device, None)
        cfg = Config(clip_model_name=clip_model_name)
        if dev.startswith("cuda"):
            cfg.apply_low_vram_defaults()  # optional for 8–12GB GPUs
        self.ci = Interrogator(cfg)

    def generate(self, image_pil, prompt: Optional[str] = None, **kw):
        txt = self.ci.interrogate(image_pil)
        return VLMOutput(text=txt.strip())

# ---------- UniDiffuser (image -> text) ----------
from PIL import Image
class UniDiffuserCaptioner(BaseVLM):
    name = "UniDiffuser"
    def __init__(self, model_id: str = "thu-ml/unidiffuser-v1", device=None, dtype="float16"):
        import torch
        from diffusers import UniDiffuserPipeline
        dev, dt = _device_and_dtype(device, dtype)
        self.pipe = UniDiffuserPipeline.from_pretrained(model_id, torch_dtype=dt)
        self.pipe = self.pipe.to(dev)
        # Optional: be explicit about mode
        if hasattr(self.pipe, "set_image_to_text_mode"):
            self.pipe.set_image_to_text_mode()
        self.device = dev

    def _resize_for_model(self, image_pil: Image.Image) -> Image.Image:
        # Try to read target size from the pipeline; fall back to 512
        target_w = target_h = 512
        ip = getattr(self.pipe, "image_processor", None)
        if ip is not None:
            size = getattr(ip, "size", None)
            if isinstance(size, dict):
                target_w = size.get("width", size.get("shortest_edge", 512))
                target_h = size.get("height", target_w)
            elif isinstance(size, int):
                target_w = target_h = size
        # Make square without distortion by padding, then resize
        img = image_pil.convert("RGB")
        w, h = img.size
        if w != h:
            # pad to square (letterbox) before resize to preserve content
            side = max(w, h)
            new_img = Image.new("RGB", (side, side), (0, 0, 0))
            new_img.paste(img, ((side - w) // 2, (side - h) // 2))
            img = new_img
        return img.resize((int(target_w), int(target_h)), Image.BICUBIC)

    def generate(self, image_pil, prompt: Optional[str] = None,
                 num_inference_steps: int = 20, guidance_scale: float = 8.0, **kw):
        img = self._resize_for_model(image_pil)
        out = self.pipe(image=img, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale)
        txt = out.text[0]
        return VLMOutput(text=txt.strip())
# ---------- ViECap (optional; requires repo) ----------
class ViECapCaptioner(BaseVLM):
    name = "ViECap"
    def __init__(self, repo_root: str, device=None, ckpt_dir: Optional[str] = None):
        """
        Minimal wrapper that shells out to ViECap's batch inference if installed.
        See: https://github.com/FeiElysia/ViECap
        """
        self.repo_root = repo_root
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.ckpt_dir = ckpt_dir

    def generate(self, image_pil, prompt: Optional[str] = None, **kw):
        # ViECap is designed around its own dataloader; here we fallback to single-image infer if the repo is on PYTHONPATH.
        import sys, io
        sys.path.append(self.repo_root)
        from infer_by_instance import infer_image  # hypothetical helper you add that wraps their CLI
        txt = infer_image(image_pil, ckpt_dir=self.ckpt_dir, device=self.device)
        return VLMOutput(text=txt.strip())

# ---------- SmallCap (optional; requires repo + datastore) ----------
# vlm_eval/vlms.py (add below existing imports and classes)
class SmallCapCaptioner(BaseVLM):
    name = "SmallCap"

    def __init__(self,
                 repo_root: str,
                 model_path: str,
                 datastore_dir: str,
                 device: Optional[str] = None,
                 dtype: Optional[str] = None,
                 topk: int = 32,
                 fp16: bool = True,
                 preload: bool = True):
        """
        Wrapper for SmallCap demo/infer; requires their retrieval datastore.
        Repo: https://github.com/RitaRamo/smallcap

        Expected files (typical):
          - model_path: path to a .pt / .pth checkpoint
          - datastore_dir: directory with FAISS index + embeddings + metadata built by the repo's scripts
        """
        import sys
        repo_root = "/home/user01/research/smallcap"
        if not repo_root or not os.path.isdir(repo_root):
            raise FileNotFoundError(f"SmallCap repo_root not found: {repo_root}")
        # Make repo importable (src/ must be under repo_root)
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

        # Device/dtype selection consistent with the rest of the framework
        dev, dt = _device_and_dtype(device, dtype)
        self.device = dev
        self.dtype = dt
        self.topk = int(topk)
        self.fp16 = bool(fp16)

        # Import SmallCap infer class
        try:
            from src.infer import SmallCapInfer
        except Exception as e:
            raise ImportError(
                f"Could not import SmallCapInfer from {repo_root}/src. "
                f"Ensure the repo is cloned and importable. Original error: {e}"
            )

        if not model_path or not os.path.isfile(model_path):
            raise FileNotFoundError(f"SmallCap model checkpoint not found: {model_path}")
        if not datastore_dir or not os.path.isdir(datastore_dir):
            raise FileNotFoundError(f"SmallCap datastore_dir not found: {datastore_dir}")

        # Instantiate infer object. The exact signature can vary across commits;
        # we pass the most common args and use kwargs to be future-proof.
        try:
            self.inf = SmallCapInfer(
                model_path=model_path,
                datastore_dir=datastore_dir,
                device=self.device,
                fp16=self.fp16,
                topk=self.topk,
            )
        except TypeError:
            # Fallback if infer class doesn't accept fp16/topk in constructor
            self.inf = SmallCapInfer(
                model_path=model_path,
                datastore_dir=datastore_dir,
                device=self.device,
            )
            # Try to set attributes if available
            if hasattr(self.inf, "topk"):
                self.inf.topk = self.topk
            if hasattr(self.inf, "fp16"):
                self.inf.fp16 = self.fp16

        # Optionally preload the datastore or warm up the model if API exposes it
        if preload:
            if hasattr(self.inf, "load_datastore"):
                try:
                    self.inf.load_datastore()
                except Exception:
                    pass
            if hasattr(self.inf, "warmup"):
                try:
                    self.inf.warmup(device=self.device)
                except Exception:
                    pass

        # Prepare a basic torchvision transform if we need a tensor fallback path
        try:
            from torchvision import transforms
            self._tfm = transforms.Compose([
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                     std=(0.26862954, 0.26130258, 0.27577711)),  # CLIP ViT-B/32 stats
            ])
        except Exception:
            self._tfm = None

    @torch.inference_mode()
    def generate(self, image_pil, prompt: Optional[str] = None, **kw) -> VLMOutput:
        """
        Runs SmallCap captioning on a single PIL image. Ignores prompt (SmallCap is image->text).
        """
        # Ensure RGB image
        try:
            im = image_pil.convert("RGB")
        except Exception:
            from PIL import Image
            if isinstance(image_pil, Image.Image):
                im = image_pil
            else:
                raise TypeError("SmallCapCaptioner.generate expects a PIL.Image")

        # Preferred API: SmallCapInfer.caption(PIL.Image)
        if hasattr(self.inf, "caption"):
            txt = self.inf.caption(im)
            return VLMOutput(text=(txt or "").strip())

        # Alternative possible API shapes:
        # - caption_from_tensor(tensor[B,C,H,W]) -> str
        # - __call__(PIL.Image) -> str
        if hasattr(self.inf, "caption_from_tensor") and self._tfm is not None:
            tensor = self._tfm(im).unsqueeze(0).to(self.device)
            txt = self.inf.caption_from_tensor(tensor)
            return VLMOutput(text=(txt or "").strip())

        if callable(self.inf):
            try:
                txt = self.inf(im)
                return VLMOutput(text=(txt or "").strip())
            except Exception:
                pass

        raise AttributeError(
            "SmallCap infer object does not expose a known captioning method. "
            "Expected one of: .caption(PIL.Image), .caption_from_tensor(tensor), or callable."
        )
# ----- FastVLM (Apple) -----
class FastVLMCaptioner(BaseVLM):
    name = "FastVLM"
    def __init__(self, repo_dir: str, model_path: str, device=None, dtype=None, timeout: int = 180):
        import shutil, subprocess, sys
        self.repo_dir = repo_dir
        self.model_path = model_path
        self.timeout = timeout
        self.py = sys.executable
        self.predict_py = os.path.join(repo_dir, "predict.py")
        if not os.path.isfile(self.predict_py):
            raise FileNotFoundError(f"predict.py not found at {self.predict_py}. Did you clone apple/ml-fastvlm and pip install -e . ?")
        if not os.path.isdir(model_path):
            raise FileNotFoundError(f"FastVLM model path not found: {model_path}")

    def generate(self, image_pil, prompt: str = "Describe the image in detail.", **kw) -> VLMOutput:
        import subprocess, tempfile, sys
        with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
            image_pil.save(tmp.name)
            cmd = [self.py, self.predict_py,
                   "--model-path", self.model_path,
                   "--image-file", tmp.name,
                   "--prompt", prompt]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout)
            if res.returncode != 0:
                raise RuntimeError(f"FastVLM failed: {res.stderr.strip()[:500]}")
            # Heuristic: take last non-empty line as the model response.
            out_lines = [l.strip() for l in res.stdout.splitlines() if l.strip()]
            text = out_lines[-1] if out_lines else ""
            return VLMOutput(text=text)


class UniDiffuserCOACaptioner(BaseVLM):
    """
    In-process reimplementation of i2t_unidiffuser.py's i2t path:
    - Loads research repo components (utils, libs.autoencoder, libs.clip, libs.caption_decoder)
    - Loads config from a Python file exposing get_config()
    - Runs CLIP image encode + autoencoder moments, DPM-Solver on text latents, then caption decoder
    """
    name = "unidiffuser_coa"

    def __init__(self,
                 repo_dir: Optional[str] = None,
                 config_path: Optional[str] = None,
                 nnet_path: Optional[str] = None,
                 device: Optional[str] = None,
                 dtype: Optional[str] = None,
                 sample_steps: int = 20,
                 cfg_scale: float = 7.0,
                 n_samples: int = 1,
                 ):
        # Resolve device/dtype (dtype unused by this stack but kept for interface consistency)
        dev, _dt = _device_and_dtype(device, dtype)
        self.device = dev

        # Resolve paths (allow env fallbacks)
        self.repo_dir = repo_dir or os.environ.get("UNIDIFFUSER_REPO", "./Unidiffuser")
        self.config_path = config_path or os.environ.get("UNIDIFFUSER_CONFIG", os.path.join(self.repo_dir, "configs", "sample_unidiffuser_v1.py"))
        self.nnet_path = nnet_path or os.environ.get("UNIDIFFUSER_WEIGHTS", os.path.join(self.repo_dir, "models", "uvit_v1.pth"))

        if not os.path.isfile(self.config_path):
            raise FileNotFoundError(f"UniDiffuser config not found: {self.config_path}")
        if not os.path.isfile(self.nnet_path):
            raise FileNotFoundError(f"UniDiffuser checkpoint not found: {self.nnet_path}")

        # Make the repo importable (utils, libs.*)
        if self.repo_dir not in map(os.path.abspath, (p for p in (self.repo_dir,) if p)):
            import sys
            sys.path.insert(0, self.repo_dir)

        # Lazy imports of research modules
        import importlib.util
        import numpy as np
        import einops
        from PIL import Image
        import clip as openai_clip
        import ml_collections
        from dpm_solver_pp import NoiseScheduleVP, DPM_Solver

        import utils  # from repo
        import libs.autoencoder as autoenc_lib  # from repo
        import libs.clip as libs_clip          # from repo

        # Load config file (expects get_config())
        spec = importlib.util.spec_from_file_location("unidiffuser_user_config", self.config_path)
        cfg_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cfg_mod)
        if not hasattr(cfg_mod, "get_config"):
            raise AttributeError(f"Config file must define get_config(): {self.config_path}")
        config = cfg_mod.get_config()

        # Store a few runtime overrides
        self.sample_steps = int(sample_steps)
        self.cfg_scale = float(cfg_scale)
        self.n_samples = int(n_samples)

        # Initialize logging + seeds similar to script (optional)
        utils.set_logger(log_level='info')
        self._set_seed(int(getattr(config, "seed", 42)))

        # Precompute betas and N
        self._betas = self._stable_diffusion_beta_schedule()
        self._N = len(self._betas)

        # Build network and components
        self.nnet = utils.get_nnet(**config.nnet)
        self.nnet.load_state_dict(torch.load(self.nnet_path, map_location="cpu"))
        self.nnet.to(self.device).eval()

        # Caption decoder (required for i2t)
        from libs.caption_decoder import CaptionDecoder  # from repo
        self.caption_decoder = CaptionDecoder(device=self.device, **config.caption_decoder)

        # Text CLIP embedder (not used for i2t, but kept for parity)
        self.clip_text_model = libs_clip.FrozenCLIPEmbedder(device=self.device).to(self.device).eval()

        # Autoencoder
        self.autoencoder = autoenc_lib.get_model(**config.autoencoder).to(self.device)
        self.autoencoder.eval()

        # Image CLIP (ViT-B/32)
        self.clip_img_model, self.clip_img_preproc = openai_clip.load("ViT-B/32", device=self.device, jit=False)

        # Bind config fields we need in sampling
        # Required keys in config: z_shape (C,H,W), text_dim, clip_img_dim, clip_text_dim, data_type, sample (with .scale default)
        self.config = config
        if not hasattr(self.config, "sample"):
            self.config.sample = ml_collections.ConfigDict()
        if "scale" not in self.config.sample:
            self.config.sample.scale = self.cfg_scale
        # Always use i2t mode internally
        self.config.mode = "i2t"

        # Cache handles for inner functions
        self._einops = einops
        self._NoiseScheduleVP = NoiseScheduleVP
        self._DPM_Solver = DPM_Solver
        self._np = np
        self._Image = Image

    # ---------------- internal helpers ----------------
    @staticmethod
    def _set_seed(seed: int):
        import random
        import numpy as np
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _stable_diffusion_beta_schedule(linear_start=0.00085, linear_end=0.0120, n_timestep=1000):
        betas = torch.linspace(linear_start**0.5, linear_end**0.5, n_timestep, dtype=torch.float64) ** 2
        return betas.numpy()

    def _center_crop_np(self, arr: "np.ndarray", size: int) -> "np.ndarray":
        # Expect HWC
        h, w = arr.shape[:2]
        th = tw = size
        i = max(0, (h - th) // 2)
        j = max(0, (w - tw) // 2)
        return arr[i:i+th, j:j+tw, :]

    @torch.inference_mode()
    def _prepare_image_contexts(self, image_pil: "Image.Image"):
        # Build CLIP image feature and autoencoder moments for one image, then repeat n_samples
        cfg = self.config
        device = self.device
        resolution = cfg.z_shape[-1] * 8  # matches research script

        # To numpy uint8, center-crop square
        img_np = self._np.array(image_pil.convert("RGB")).astype(self._np.uint8)
        img_np = self._center_crop_np(img_np, resolution)

        # CLIP image feature (1, D)
        img_for_clip = self._Image.fromarray(img_np)
        clip_img_feature = self.clip_img_model.encode_image(
            self.clip_img_preproc(img_for_clip).unsqueeze(0).to(device)
        )

        # Moments via autoencoder: input BCHW in [-1, 1]
        img_f32 = (img_np / 127.5 - 1.0).astype(self._np.float32)
        img_bchw = self._einops.rearrange(img_f32, "h w c -> 1 c h w")
        img_tensor = torch.tensor(img_bchw, device=device)
        moments = self.autoencoder.encode_moments(img_tensor)

        # Tile to n_samples
        clip_imgs = torch.stack([clip_img_feature for _ in range(self.n_samples)], dim=0)  # (B,1,D)
        img_contexts = torch.cat([moments for _ in range(self.n_samples)], dim=0)         # (B, 2C, H, W) as in repo

        return img_contexts, clip_imgs

    # i2t network wrapper with CFG, matching the original
    def _i2t_nnet(self, x, t, z, clip_img):
        # x: (B, 77, text_dim); t: (B,)
        cfg = self.config
        N = self._N
        z_out, clip_img_out, text_out = self.nnet(
            z, clip_img, text=x,
            t_img=torch.zeros_like(t, dtype=torch.int, device=self.device),
            t_text=t,
            data_type=torch.zeros_like(t, dtype=torch.int, device=self.device) + cfg.data_type
        )
        if float(self.config.sample.scale) == 0.0:
            return text_out

        # unconditional branch (noise on z and clip_img)
        z_N = torch.randn_like(z)
        clip_img_N = torch.randn_like(clip_img)
        _, _, text_out_uncond = self.nnet(
            z_N, clip_img_N, text=x,
            t_img=torch.ones_like(t, dtype=torch.int, device=self.device) * N,
            t_text=t,
            data_type=torch.zeros_like(t, dtype=torch.int, device=self.device) + cfg.data_type
        )
        return text_out + float(self.config.sample.scale) * (text_out - text_out_uncond)

    # ---------------- public API ----------------
    @torch.inference_mode()
    def generate(self, image_pil, prompt: Optional[str] = None, max_new_tokens: int = 80, **kw) -> VLMOutput:
        """
        Runs i2t (image->text) once; if n_samples>1, returns the first caption.
        """
        device = self.device
        cfg = self.config
        N = self._N

        # Prepare image contexts
        img_contexts, clip_imgs = self._prepare_image_contexts(image_pil)
        # Sample z for image from moments
        z_img = self.autoencoder.sample(img_contexts)  # (B, C, H, W) latents

        # Initialize text latent
        text_init = torch.randn(self.n_samples, 77, cfg.text_dim, device=device)

        # Discrete VP schedule
        noise_schedule = self._NoiseScheduleVP(
            schedule="discrete",
            betas=torch.tensor(self._betas, device=device).float()
        )

        def model_fn(x, t_cont):
            # t_cont in [eps, 1]; scale to [0, N]
            t = t_cont * N
            return self._i2t_nnet(x, t, z=z_img, clip_img=clip_imgs)

        dpm_solver = self._DPM_Solver(model_fn, noise_schedule, predict_x0=True, thresholding=False)
        x = dpm_solver.sample(text_init, steps=self.sample_steps, eps=1.0 / N, T=1.0)

        # Decode with caption decoder
        captions = self.caption_decoder.generate_captions(x)
        text = (captions[0] if isinstance(captions, (list, tuple)) and len(captions) else "").strip()
        return VLMOutput(text=text)


# Registry
REGISTRY = {
    "BLIP": BLIPCaptioner,
    "BLIP-2": BLIP2Captioner,
    "InstructBLIP": InstructBLIPCaptioner,
    "LLaVA-7B": lambda **kw: LLaVACaptioner(model_id="llava-hf/llava-1.5-7b-hf", **kw),
    "LLaVA-13B": lambda **kw: LLaVACaptioner(model_id="llava-hf/llava-1.5-13b-hf", **kw),
    "LLaVA-NeXT": LLaVANextCaptioner,
    "Qwen2.5-VL-7B": Qwen25VLCaptioner,
    "Img2Prompt": Img2PromptCaptioner,
    "UniDiffuser": UniDiffuserCaptioner,
    "FastVLM-0.5B": lambda **kw: FastVLMCaptioner(repo_dir=os.environ.get("FASTVLM_REPO", "./ml-fastvlm"),
                                                  model_path=os.environ.get("FASTVLM_MODEL", "./ml-fastvlm/checkpoints/fastvlm_0.5b_stage3"),
                                                  **kw),
    "FastVLM-1.5B": lambda **kw: FastVLMCaptioner(repo_dir=os.environ.get("FASTVLM_REPO", "./ml-fastvlm"),
                                                   model_path=os.environ.get("FASTVLM_MODEL", "./ml-fastvlm/checkpoints/fastvlm_1.5b_stage3"),
                                                   **kw),
    "FastVLM-7B": lambda **kw: FastVLMCaptioner(repo_dir=os.environ.get("FASTVLM_REPO", "./ml-fastvlm"),
                                                model_path=os.environ.get("FASTVLM_MODEL", "./ml-fastvlm/checkpoints/fastvlm_7b_stage3"),
                                                **kw),
    "unidiffuser_coa": lambda **kw: UniDiffuserCOACaptioner(**kw),
    "SmallCap": SmallCapCaptioner,
    # Optional (install repos manually)
    # "ViECap": ViECapCaptioner,
}
