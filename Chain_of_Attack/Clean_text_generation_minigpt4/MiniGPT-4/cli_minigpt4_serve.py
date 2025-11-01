import io, sys, json, base64, os, time
from PIL import Image
import torch

from minigpt4.common.config import Config
from minigpt4.common.registry import registry
try:
    from minigpt4.conversation.conversation import Chat, CONV_VISION_Vicuna0 as CONV_TEMPLATE
except Exception:
    from minigpt4.conversation.conversation import Chat, CONV_VISION as CONV_TEMPLATE
# At the very top, after imports
import sys
try:
    sys.stdout.reconfigure(line_buffering=True)  # Python 3.7+
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass
    
def _log(msg: str):
    sys.stderr.write(f"[minigpt4-serve] {msg}\n")
    sys.stderr.flush()

def _emit(obj):
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()

def _mk_args(**kw):
    import types
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
    return chat

def _run_once(chat, b64_image: str, prompt: str,
              num_beams: int, temperature: float,
              max_new_tokens: int, max_length: int) -> str:
    img = Image.open(io.BytesIO(base64.b64decode(b64_image))).convert("RGB")
    conv = CONV_TEMPLATE.copy()
    img_list = [img]
    if hasattr(chat, "encode_img"):
        if hasattr(conv, "append_message") and hasattr(conv, "roles"):
            conv.append_message(conv.roles[0], "<Img><ImageHere></Img>")
        chat.encode_img(img_list)
        chat.ask(prompt, conv)
        out = chat.answer(
            conv=conv, img_list=img_list,
            num_beams=num_beams, temperature=temperature,
            max_new_tokens=max_new_tokens, max_length=max_length
        )
        ans = out[0] if isinstance(out, (list, tuple)) else out
        return str(ans).strip()

    # Older API
    chat_state = CONV_TEMPLATE.copy()
    new_img_list = []
    chat.upload_img(img, chat_state, new_img_list)
    chat.ask(prompt, chat_state)
    ans, _ = chat.answer(
        conv=chat_state, img_list=new_img_list,
        max_new_tokens=max_new_tokens, temperature=temperature
    )
    return str(ans).strip()

def main():
    _log("starting; waiting for init line on stdin")
    init_line = sys.stdin.readline()
    if not init_line:
        _log("no init line; exiting")
        return
    init = json.loads(init_line)
    if init.get("cmd") != "init":
        _log("ERROR: expected cmd=init")
        return

    cfg_path = init.get("cfg_path", "eval_configs/minigpt4_eval.yaml")
    device = init.get("device") or ("cuda:0" if torch.cuda.is_available() else "cpu")

    _log(f"loading model on {device} with cfg={cfg_path} ...")
    t0 = time.time()
    chat = load_chat(cfg_path, device)
    _log(f"ready in {time.time()-t0:.2f}s (pid={os.getpid()}); listening for cmd='infer'")
    _emit({"ok": True})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
            cmd = msg.get("cmd")
            if cmd == "close":
                _log("close received; shutting down")
                _emit({"ok": True})
                break
            if cmd != "infer":
                _log(f"unknown cmd: {cmd}")
                _emit({"error": "unknown cmd"})
                continue

            rid = msg.get("id")
            prompt = msg.get("prompt", "Describe this image.")
            num_beams = int(msg.get("num_beams", 1))
            temperature = float(msg.get("temperature", 1.0))
            max_new_tokens = int(msg.get("max_new_tokens", 67))
            max_length = int(msg.get("max_length", 2000))
            b64 = msg.get("image_b64")
            if not b64:
                raise ValueError("missing image_b64")

            _log(f"infer start id={rid}")
            t1 = time.time()
            text = _run_once(chat, b64, prompt, num_beams, temperature, max_new_tokens, max_length)
            dt = time.time() - t1
            _log(f"infer done  id={rid} dt={dt:.2f}s out_chars={len(text)}")
            _emit({"id": rid, "text": text})
        except Exception as e:
            _log(f"error: {e}")
            _emit({"error": str(e)})

if __name__ == "__main__":
    main()