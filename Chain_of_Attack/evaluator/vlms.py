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

    def __init__(
        self,
        clip_model_name: str = "ViT-L-14/openai",
        blip_model_type: str = "large",   # "base" or "large" (and "blip2" on >=0.6.0)
        low_vram: bool = False,
        cache_path: Optional[str] = None,
        download_cache: bool = True,
        chunk_size: Optional[int] = None,
        device: Optional[str] = None,
        dtype: Optional[str] = None,
    ):
        # Lazy import so this class is optional
        try:
            from clip_interrogator import Config, Interrogator
        except Exception as e:
            raise ImportError(
                "clip-interrogator is required for Img2PromptCaptioner. "
                "Install with `pip install clip-interrogator==0.5.4` "
                "or `==0.6.0` for BLIP-2 support."
            ) from e

        dev, _ = _device_and_dtype(device, dtype)
        cfg = Config(clip_model_name=clip_model_name)

        # Device (attribute name differs across versions)
        if hasattr(cfg, "device"):
            cfg.device = dev
        elif hasattr(cfg, "torch_device"):
            cfg.torch_device = dev

        # BLIP model selection differs across versions (backward-compatible)
        if blip_model_type is not None:
            if hasattr(cfg, "blip_model_type"):
                cfg.blip_model_type = blip_model_type  # >= 0.6.0
            elif hasattr(cfg, "caption_model_name"):
                # 0.5.x uses caption_model_name like "blip-base"/"blip-large"
                cfg.caption_model_name = f"blip-{blip_model_type}"

        if cache_path is not None:
            cfg.cache_path = cache_path
        if chunk_size is not None:
            cfg.chunk_size = chunk_size
        if download_cache is not None and hasattr(cfg, "download_cache"):
            cfg.download_cache = download_cache
        if low_vram and hasattr(cfg, "apply_low_vram_defaults"):
            cfg.apply_low_vram_defaults()

        self._ci = Interrogator(cfg)
        self.device = dev

    def generate(self, image_pil, prompt: Optional[str] = None, mode: str = "best", **kw) -> VLMOutput:
        """
        mode: one of {"best", "fast", "classic", "deep"}
        - best     -> interrogate()
        - fast     -> interrogate_fast()
        - classic  -> interrogate_classic()
        - deep     -> interrogate_deep()
        """
        if mode == "fast" and hasattr(self._ci, "interrogate_fast"):
            out = self._ci.interrogate_fast(image_pil)
        elif mode == "classic" and hasattr(self._ci, "interrogate_classic"):
            out = self._ci.interrogate_classic(image_pil)
        elif mode == "deep" and hasattr(self._ci, "interrogate_deep"):
            out = self._ci.interrogate_deep(image_pil)
        else:
            out = self._ci.interrogate(image_pil)
        return VLMOutput(text=str(out).strip())

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
import requests
import io
import os
import sys
import time
import subprocess
import signal
import psutil
from PIL import Image

class ViECapCaptioner(BaseVLM):
    name = "ViECap"
    
    def __init__(self, dtype, repo_root: str, device=None, ckpt_dir: Optional[str] = None, server_port: int = 8001):
        self.repo_root = repo_root
        self.server_url = f"http://localhost:{server_port}"
        self.ckpt_dir = ckpt_dir
        self._server_process = None
        self._session = requests.Session()
        
        # Start server
        self._start_server()
        
        # Wait for server to be ready
        self._wait_for_server()
    
    def _start_server(self):
        """Start the ViECap server using the virtual environment"""
        if self._is_server_running():
            print("ViECap server is already running")
            return
        
        server_script = os.path.join(self.repo_root, "start_viecap_server.sh")
        
        # Prepare environment
        env = os.environ.copy()
        if self.ckpt_dir:
            env["VIECAP_CKPT_DIR"] = self.ckpt_dir
        
        print(f"Starting ViECap server from {self.repo_root}...")
        
        # Start server process with proper process group
        self._server_process = subprocess.Popen(
            [server_script, self.ckpt_dir] if self.ckpt_dir else [server_script],
            cwd=self.repo_root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid  # Create new process group
        )
        
        # Give server time to start
        time.sleep(5)
    
    def _is_server_running(self):
        """Check if server is already running"""
        try:
            response = self._session.get(f"{self.server_url}/health", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def _wait_for_server(self, timeout=60):
        """Wait for server to be ready"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = self._session.get(f"{self.server_url}/health", timeout=2)
                if response.status_code == 200:
                    data = response.json()
                    if data.get("model_loaded"):
                        print("ViECap server is ready with model loaded")
                        return
                    else:
                        print("Waiting for model to load...")
            except Exception as e:
                print(f"Waiting for server... ({e})")
            time.sleep(2)
        
        raise RuntimeError(f"ViECap server failed to start within {timeout} seconds")
    
    def _kill_process_tree(self, pid):
        """Kill a process and all its children"""
        try:
            parent = psutil.Process(pid)
            children = parent.children(recursive=True)
            
            # Kill children first
            for child in children:
                try:
                    child.terminate()
                except:
                    pass
            
            # Wait for children to terminate
            gone, still_alive = psutil.wait_procs(children, timeout=5)
            
            # Force kill any remaining children
            for child in still_alive:
                try:
                    child.kill()
                except:
                    pass
            
            # Kill parent
            try:
                parent.terminate()
                parent.wait(timeout=5)
            except:
                try:
                    parent.kill()
                    parent.wait(timeout=5)
                except:
                    pass
                    
        except psutil.NoSuchProcess:
            # Process already dead
            pass
        except Exception as e:
            print(f"Error killing process tree: {e}")
    
    def generate(self, image_pil, prompt: Optional[str] = None, **kw):
        # Convert PIL image to bytes
        img_byte_arr = io.BytesIO()
        image_pil.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()
        
        # Send request to server
        files = {'image': ('image.jpg', img_byte_arr, 'image/jpeg')}
        
        try:
            response = self._session.post(
                f"{self.server_url}/generate", 
                files=files,
                timeout=30  # 30 second timeout
            )
            response.raise_for_status()
            result = response.json()
            return VLMOutput(text=result['caption'].strip())
        
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"ViECap server request failed: {str(e)}")
    
    def close(self):
        """Close the connection and stop the server"""
        print("Shutting down ViECap server...")
        
        # Try to gracefully shutdown via API first
        try:
            response = self._session.post(f"{self.server_url}/shutdown", timeout=5)
            print("Graceful shutdown initiated via API")
            time.sleep(2)  # Give time for graceful shutdown
        except:
            print("API shutdown failed, forcing process termination")
        
        # Close session
        if hasattr(self, '_session'):
            self._session.close()
        
        # Force kill the server process and all its children
        if hasattr(self, '_server_process') and self._server_process:
            try:
                # Get the process ID
                pid = self._server_process.pid
                
                # Kill the entire process tree
                self._kill_process_tree(pid)
                
                # Also try terminating the process directly
                try:
                    self._server_process.terminate()
                    self._server_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        self._server_process.kill()
                        self._server_process.wait(timeout=5)
                    except:
                        pass
                
                # Check if process is still alive
                if self._server_process.poll() is None:
                    print("Warning: Server process might still be running")
                else:
                    print("ViECap server stopped successfully")
                    
            except Exception as e:
                print(f"Error stopping server process: {e}")
            finally:
                self._server_process = None
        
        # Additional cleanup: kill any remaining processes on our port
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if any('viecap_server.py' in str(arg) for arg in (proc.info['cmdline'] or [])):
                        if proc.info['pid'] != os.getpid():  # Don't kill ourselves
                            print(f"Killing leftover ViECap process: {proc.info['pid']}")
                            proc.terminate()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except:
            pass
# ---------- SmallCap (optional; requires repo + datastore) ----------
import os, sys, json, tempfile
from typing import Optional
import torch
from dataclasses import dataclass
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, CLIPFeatureExtractor, AutoTokenizer
import faiss
import numpy as np
import clip 

class SmallCapCaptioner(BaseVLM):
    name = "SmallCap"
    def __init__(self,
                 repo_root: Optional[str] = None,
                 hf_model_id: str = "Yova/SmallCap7M",
                 datastore_name: str = "coco",
                 device: Optional[str] = None,
                 dtype: Optional[str] = None,
                 topk: int = 4,
                 fp16: bool = True,
                 preload: bool = True):
        
        self.device, self.dtype = _device_and_dtype(device, dtype)
        self.topk = int(topk)
        self.fp16 = bool(fp16)
        self.hf_model_id = hf_model_id
        self.datastore_name = datastore_name
        
        # 1) Import the SmallCap model code
        if repo_root is None or not os.path.isdir(repo_root):
            repo_root = os.path.join(tempfile.gettempdir(), "smallcap_repo")
            if not os.path.isdir(repo_root):
                os.system(f"git clone --depth 1 https://github.com/RitaRamo/smallcap {repo_root}")
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        if os.path.isdir(os.path.join(repo_root, "src")) and repo_root not in sys.path:
            sys.path.insert(0, repo_root)
            
        # 2) Import and register custom classes
        from src.vision_encoder_decoder import SmallCap, SmallCapConfig
        from src.gpt2 import ThisGPT2Config, ThisGPT2LMHeadModel
        from src.opt import ThisOPTConfig, ThisOPTForCausalLM
        from src.utils import prep_strings, postprocess_preds
        
        self._prep_strings = prep_strings
        self._postprocess = postprocess_preds
        
        AutoConfig.register("smallcap", SmallCapConfig)
        AutoModel.register(SmallCapConfig, SmallCap)
        AutoConfig.register("this_gpt2", ThisGPT2Config)
        AutoModelForCausalLM.register(ThisGPT2Config, ThisGPT2LMHeadModel)
        AutoConfig.register("this_opt", ThisOPTConfig)
        AutoModelForCausalLM.register(ThisOPTConfig, ThisOPTForCausalLM)
        
        # 3) Get the checkpoint path from Hugging Face cache
        checkpoint_path = snapshot_download(
            hf_model_id, 
            allow_patterns=["config.json", "pytorch_model.bin", f"{datastore_name}_index*", f"{datastore_name}_index_captions.json"]
        )
        
        # 4) Load model EXACTLY like they do in their code
        config = AutoConfig.from_pretrained(os.path.join(checkpoint_path, 'config.json'))
        self.model = AutoModel.from_pretrained(checkpoint_path)
        self.model.config = config  # THIS IS THE KEY STEP from their code
        self.model.eval()
        self.model.to(self.device)
        
        if self.fp16 and self.device == "cuda":
            self.model = self.model.half()
        
        # 5) Load tokenizer and feature extractors
        self._enc_feat = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Determine tokenizer based on config
        decoder_cfg = getattr(config, "decoder", None)
        if decoder_cfg is not None and getattr(decoder_cfg, "model_type", None) == "this_opt":
            tok_name = "facebook/opt-125m"
        else:
            tok_name = "gpt2"
            
        self._tok = AutoTokenizer.from_pretrained(tok_name)
        self._tok.pad_token = self._tok.eos_token or "!"
        
        # 6) Load retrieval model and FAISS index
        self._retr_model, self._retr_preproc = clip.load("RN50x64", device=self.device)
        if self.fp16 and self.device == "cuda":
            self._retr_model = self._retr_model.half()
            
        index_path = os.path.join(checkpoint_path, f"{datastore_name}_index")
        caps_path = os.path.join(checkpoint_path, f"{datastore_name}_index_captions.json")
        
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        if not os.path.exists(caps_path):
            raise FileNotFoundError(f"Captions JSON not found: {caps_path}")
            
        self._faiss = faiss.read_index(index_path)
        with open(caps_path, "r") as f:
            self._caps = json.load(f)
        
        # 7) Load template
        tmpl_path = os.path.join(repo_root, "src", "template.txt")
        if os.path.exists(tmpl_path):
            self._template = open(tmpl_path).read().strip() + " "
        else:
            self._template = "Describe the image in one sentence: "
    
    @torch.inference_mode()
    def generate(self, image_pil, prompt: Optional[str] = None, **kw) -> VLMOutput:
        # 1) Encode image for retrieval
        img = image_pil.convert("RGB")
        retr_inp = self._retr_preproc(img).unsqueeze(0).to(self.device)
        img_emb = self._retr_model.encode_image(retr_inp).detach().float().cpu().numpy()
        faiss.normalize_L2(img_emb)
        D, I = self._faiss.search(img_emb, self.topk)
        retrieved = [self._caps[i] for i in I[0][: self.topk]]
        
        # 2) Build decoder input ids
        dec_ids = self._prep_strings(
            "", self._tok, template=self._template, retrieved_caps=retrieved, k=len(retrieved), is_test=True
        )
        
        # 3) Encode image for SmallCap encoder
        pixel_values = self._enc_feat(img, return_tensors="pt").pixel_values.to(self.device)
        
        # 4) Generate using the model's generate method (should work now)
        out = self.model.generate(
            pixel_values=pixel_values,
            decoder_input_ids=torch.tensor([dec_ids]).to(self.device),
            max_new_tokens=25,
            no_repeat_ngram_size=0,
            length_penalty=0.0,
            min_length=1,
            num_beams=3,
            eos_token_id=self._tok.eos_token_id,
        )
        
        text = self._postprocess(self._tok.decode(out[0]), self._tok)
        return VLMOutput(text=text.strip())

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
                raise RuntimeError(f"FastVLM failed: {res.stderr.strip()}")
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
### minigpt4
import io, os, sys, json, base64, subprocess, uuid, selectors, threading, time
from typing import Optional
from PIL import Image

# from your_project.vlm_base import BaseVLM, VLMOutput
class MiniGPT4SubprocessCaptioner(BaseVLM):
    name = "MiniGPT-4"

    def __init__(self,
                 repo_dir: str = None,
                 python_path: str = None,
                 cli_name: str = "cli_minigpt4_serve.py",
                 cfg_path: str = None,
                 device: str = None,
                 timeout: int = 600,
                 **_):
        self._p("ENTER __init__")
        self.repo_dir = repo_dir or os.environ.get("MINIGPT4_REPO", "./MiniGPT-4")
        self.python = python_path or os.path.join(self.repo_dir, ".venv", "bin", "python")
        self.cli = os.path.join(self.repo_dir, cli_name)
        self.cfg_path = cfg_path or os.environ.get("MINIGPT4_CFG", "eval_configs/minigpt4_eval.yaml")
        self.device = device or os.environ.get("MINIGPT4_DEVICE", "cuda:0")
        self.timeout = float(timeout)

        self._proc: Optional[subprocess.Popen] = None
        self._selector: Optional[selectors.BaseSelector] = None
        self._stderr_thread: Optional[threading.Thread] = None

        self._start_server()

    def close(self):
        self._p("ENTER close")
        self._shutdown(kill=False)

    def __del__(self):
        self._p("ENTER __del__ (best-effort cleanup)")
        try:
            self._shutdown(kill=False)
        except Exception:
            pass

    # ------------- internals -------------
    def _p(self, msg: str):
        sys.stderr.write(f"[minigpt4-client] {msg}\n")
        sys.stderr.flush()

    def _drain_stderr(self):
        self._p("ENTER _drain_stderr (child log forwarding)")
        try:
            if self._proc and self._proc.stderr:
                for line in self._proc.stderr:
                    if not line: break
                    sys.stderr.write("[minigpt4-child] " + line)
                    sys.stderr.flush()
        except Exception as e:
            self._p(f"_drain_stderr stopped: {e}")

    def _start_server(self):
        self._p("ENTER _start_server")
        if not os.path.isfile(self.cli):
            raise FileNotFoundError(f"serve CLI not found: {self.cli}")
        if not os.path.isfile(self.python):
            raise FileNotFoundError(f"python interpreter not found: {self.python}")

        cmd = [self.python, self.cli]
        self._p(f"launching server: {' '.join(cmd)} (cwd={self.repo_dir})")
        t0 = time.perf_counter()
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1" 
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=self.repo_dir,
            bufsize=1,
            text=True,
            env=env
        )
        self._selector = selectors.DefaultSelector()
        self._selector.register(self._proc.stdout, selectors.EVENT_READ)

        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()

        self._p("sending init handshake to server")
        self._send({"cmd": "init", "cfg_path": self.cfg_path, "device": self.device})
        self._p("waiting for init ack from server")
        resp = self._recv(self.timeout)
        if not isinstance(resp, dict) or not resp.get("ok"):
            raise RuntimeError(f"server init failed: {resp}")
        self._p(f"server ready pid={self._proc.pid} init_time={time.perf_counter()-t0:.2f}s")

    def _server_alive(self) -> bool:
        return (self._proc is not None) and (self._proc.poll() is None)

    def _send(self, obj):
        self._p("ENTER _send")
        if not self._server_alive() or not self._proc.stdin:
            raise BrokenPipeError("server not alive")
        self._proc.stdin.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self._proc.stdin.flush()

    def _recv(self, timeout: float):
        self._p(f"ENTER _recv (waiting up to {timeout:.1f}s)")
        if not self._server_alive() or not self._proc.stdout or not self._selector:
            raise BrokenPipeError("server not alive")
        deadline = time.perf_counter() + timeout
        buf_debug = None
        while time.perf_counter() < deadline:
            remaining = max(0, deadline - time.perf_counter())
            events = self._selector.select(remaining)
            if not events:
                break  # timeout
            line = self._proc.stdout.readline()
            if line is None:
                continue
            if line == "":  # EOF from child
                rc = self._proc.poll()
                raise RuntimeError(f"server closed stdout (rc={rc}) before sending JSON")
            s = line.strip()
            if not s:
                # blank line; keep waiting
                continue
            # Only try to parse lines that look like JSON objects/arrays
            if not (s.startswith("{") or s.startswith("[")):
                # capture a short preview for debugging, then keep waiting
                preview = s if len(s) < 200 else (s[:200] + " ...")
                self._p(f"_recv: ignoring non-JSON line from server stdout: {preview!r}")
                buf_debug = preview
                continue
            try:
                return json.loads(s)
            except json.JSONDecodeError as e:
                # Partial or corrupted line; keep waiting for next line
                preview = s if len(s) < 200 else (s[:200] + " ...")
                self._p(f"_recv: JSON decode error; will continue reading. err={e} line={preview!r}")
                buf_debug = preview
                continue
        # Timeout reached
        msg = "no data" if buf_debug is None else f"last_line_preview={buf_debug!r}"
        self._p(f"TIMEOUT in _recv; {msg}")
        raise TimeoutError("server read timed out")
        
    def _shutdown(self, kill: bool = False):
        self._p(f"ENTER _shutdown kill={kill}")
        if not self._proc: 
            self._p("_shutdown: no process")
            return
        pid = self._proc.pid
        try:
            if self._server_alive() and not kill:
                self._p("sending close to server")
                try:
                    self._send({"cmd": "close"})
                    _ = self._recv(5.0)
                except Exception as e:
                    self._p(f"close handshake error: {e}")
            if self._proc:
                (self._proc.kill() if kill else self._proc.terminate())
                try:
                    self._proc.wait(timeout=5.0)
                except Exception:
                    try:
                        self._proc.kill(); self._proc.wait(timeout=2.0)
                    except Exception:
                        pass
        finally:
            rc = self._proc.returncode
            self._p(f"server exited pid={pid} rc={rc}")
            try:
                if self._selector and self._proc and self._proc.stdout:
                    self._selector.unregister(self._proc.stdout)
            except Exception:
                pass
            self._selector = None
            self._proc = None
            self._stderr_thread = None

    # ------------- VLM API -------------
    def generate(self, image_pil: Image.Image, prompt: str = None,
                 max_new_tokens: int = 67, temperature: float = 1.0,
                 num_beams: int = 1, max_length: int = 2000, **kw) -> VLMOutput:
        self._p("ENTER generate (preparing image)")
        buf = io.BytesIO()
        image_pil.save(buf, format="PNG")
        image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        rid = str(uuid.uuid4())
        ptxt = prompt or "Describe this image."
        req = {
            "cmd": "infer", "id": rid,
            "prompt": ptxt,
            "num_beams": num_beams,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "max_length": max_length,
            "image_b64": image_b64,
        }

        self._p(f"generate: SEND id={rid}")
        t0 = time.perf_counter()
        self._send(req)
        self._p(f"generate: WAIT id={rid}")
        resp = self._recv(self.timeout)
        if resp.get("id") != rid and "text" not in resp:
            raise RuntimeError(f"server desync: {resp}")
        dt = time.perf_counter() - t0
        text = (resp.get("text") or "").strip()
        self._p(f"generate: RECV id={rid} dt={dt:.2f}s out_chars={len(text)}")
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
    "img2prompt": Img2PromptCaptioner,
    "UniDiffuser": UniDiffuserCaptioner,
    "FastVLM-0.5B": lambda **kw: FastVLMCaptioner(repo_dir=os.environ.get("FASTVLM_REPO", "/home/user01/research/ml-fastvlm"),
                                                  model_path=os.environ.get("FASTVLM_MODEL", "/home/user01/research/ml-fastvlm/checkpoints/llava-fastvithd_0.5b_stage3"),
                                                  **kw),
    "FastVLM-1.5B": lambda **kw: FastVLMCaptioner(repo_dir=os.environ.get("FASTVLM_REPO", "./ml-fastvlm"),
                                                   model_path=os.environ.get("FASTVLM_MODEL", "./ml-fastvlm/checkpoints/fastvlm_1.5b_stage3"),
                                                   **kw),
    "FastVLM-7B": lambda **kw: FastVLMCaptioner(repo_dir=os.environ.get("FASTVLM_REPO", "./ml-fastvlm"),
                                                model_path=os.environ.get("FASTVLM_MODEL", "./ml-fastvlm/checkpoints/fastvlm_7b_stage3"),
                                                **kw),
    "unidiffuser_coa": lambda **kw: UniDiffuserCOACaptioner(**kw),
    "minigpt4": lambda **kw: MiniGPT4SubprocessCaptioner(
                        repo_dir=os.environ.get("MINIGPT4_REPO", "/home/user01/research/Chain_of_Attack/Clean_text_generation_minigpt4/MiniGPT-4"),
                        cfg_path=os.environ.get("MINIGPT4_CFG", "eval_configs/minigpt4_eval.yaml"),
                        cli_name="cli_minigpt4_serve.py",
                        **kw
                    ),
    "ViECap": lambda **kw: ViECapCaptioner(repo_root='/home/user01/research/ViECap', **kw),
    # "SmallCap": lambda **kw: SmallCapCaptioner(
    #                     repo_root="/home/user01/research/smallcap",   # or None to auto-clone
    #                     hf_model_id="Yova/SmallCap7M",                # or "Yova/SmallCapOPT7M"
    #                     datastore_name="coco",
    #                     topk=4,
    #                     fp16=True,
    #                     **kw
    #                 ),
    # Optional (install repos manually)
}
