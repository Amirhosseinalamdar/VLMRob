import io, sys, json, base64, types, os
from contextlib import contextmanager, redirect_stdout
from PIL import Image
import torch

# Try to quiet HF/transformers logs
try:
    from transformers import logging as hf_logging
    hf_logging.set_verbosity_error()
except Exception:
    pass

from minigpt4.common.config import Config
from minigpt4.common.registry import registry

# Prefer legacy template if present
try:
    from minigpt4.conversation.conversation import Chat, CONV_VISION_Vicuna0 as CONV_TEMPLATE
except Exception:
    from minigpt4.conversation.conversation import Chat, CONV_VISION as CONV_TEMPLATE

# Legacy registry side-effect imports
try:
    from minigpt4.datasets.builders import *  # noqa
    from minigpt4.models import *  # noqa
    from minigpt4.processors import *  # noqa
    from minigpt4.runners import *  # noqa
    from minigpt4.tasks import *  # noqa
except Exception:
    pass

@contextmanager
def redirect_all_stdout_to_stderr():
    """
    Route Python-level prints and most fd(1) writes to stderr
    while the block runs, so stdout is clean for final JSON.
    """
    old_stdout = sys.stdout
    old_fd1 = os.dup(1)
    try:
        with redirect_stdout(sys.stderr):
            os.dup2(2, 1)  # point fd1 to stderr
            yield
    finally:
        try:
            os.dup2(old_fd1, 1)
        finally:
            os.close(old_fd1)
        sys.stdout = old_stdout

def _mk_args(**kw):
    a = types.SimpleNamespace()
    for k, v in kw.items():
        setattr(a, k, v)
    return a

def load_chat(cfg_path: str, device: str):
    args = _mk_args(cfg_path=cfg_path, options=None, gpu_id=0)
    cfg = Config(args)
    model_cfg = cfg.model_cfg
    model_cls = registry.get_model_class(model_cfg.arch)
    model = model_cls.from_config(model_cfg).to(device)
    vis_proc_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_proc_cfg.name).from_config(vis_proc_cfg)
    chat = Chat(model, vis_processor, device=device)
    return chat, cfg

def run_once(chat, b64_image: str, prompt: str,
             num_beams: int, temperature: float,
             max_new_tokens: int, max_length: int):
    img = Image.open(io.BytesIO(base64.b64decode(b64_image))).convert("RGB")
    conv = CONV_TEMPLATE.copy()
    img_list = [img]

    if hasattr(chat, "encode_img"):
        if hasattr(conv, "append_message") and hasattr(conv, "roles"):
            conv.append_message(conv.roles[0], "<Img><ImageHere></Img>")
        chat.encode_img(img_list)
        chat.ask(prompt, conv)
        out = chat.answer(
            conv=conv,
            img_list=img_list,
            num_beams=num_beams,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            max_length=max_length,
        )
        ans = out[0] if isinstance(out, (list, tuple)) else out
        return str(ans).strip()

    elif hasattr(chat, "upload_img"):
        chat_state = CONV_TEMPLATE.copy()
        new_img_list = []
        chat.upload_img(img, chat_state, new_img_list)
        chat.ask(prompt, chat_state)
        ans, _ = chat.answer(
            conv=chat_state,
            img_list=new_img_list,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        return str(ans).strip()

    raise RuntimeError("Unsupported MiniGPT-4 Chat API in this repo.")

def main():
    payload = json.load(sys.stdin)
    cfg_path = payload.get("cfg_path", "eval_configs/minigpt4_eval.yaml")
    device = payload.get("device") or ("cuda:0" if torch.cuda.is_available() else "cpu")
    prompt = payload.get("prompt", "Describe this image.")
    num_beams = int(payload.get("num_beams", 1))
    temperature = float(payload.get("temperature", 1.0))
    max_new_tokens = int(payload.get("max_new_tokens", 67))
    max_length = int(payload.get("max_length", 2000))

    # All noisy logs go to stderr inside this block
    with redirect_all_stdout_to_stderr():
        chat, _ = load_chat(cfg_path, device)
        text = run_once(
            chat,
            payload["image_b64"],
            prompt,
            num_beams=num_beams,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            max_length=max_length,
        )

    # Emit only JSON on stdout
    print(json.dumps({"text": text}, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()