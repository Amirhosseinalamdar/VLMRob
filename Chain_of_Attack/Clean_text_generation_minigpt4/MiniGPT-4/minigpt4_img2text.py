# import os
# # os.environ["CUDA_VISIBLE_DEVICES"] = ""
# import argparse
# import random
# import json
# from PIL import Image
# import time
# import numpy as np
# import torch
# import torchvision
# import torch.backends.cudnn as cudnn
# from minigpt4.common.config import Config
# from minigpt4.common.dist_utils import get_rank
# from minigpt4.common.registry import registry
# from minigpt4.conversation.conversation import Chat, CONV_VISION_Vicuna0
# # imports modules for registration
# from minigpt4.datasets.builders import *
# from minigpt4.models import *
# from minigpt4.processors import *
# from minigpt4.runners import *
# from minigpt4.tasks import *

# # seed for everything
# DEFAULT_RANDOM_SEED = 2023
# device = "cuda" if torch.cuda.is_available() else "cpu"

# def seedBasic(seed=DEFAULT_RANDOM_SEED):
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     np.random.seed(seed)

# def seedTorch(seed=DEFAULT_RANDOM_SEED):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

# def seedEverything(seed=DEFAULT_RANDOM_SEED):
#     seedBasic(seed)
#     seedTorch(seed)

# # ------------------------------------------------------------------ #
# class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
#     def __getitem__(self, index: int):
#         original_tuple = super().__getitem__(index)  # (img, label)
#         path, _ = self.samples[index]
#         image_processed = vis_processor(original_tuple[0])
#         return image_processed, original_tuple[1], path

# if __name__ == "__main__":
#     seedEverything()
#     parser = argparse.ArgumentParser(description="Demo")
#     # minigpt-4
#     parser.add_argument("--cfg-path", default="./eval_configs/minigpt4_eval.yaml", help="path to configuration file.")
#     parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
#     parser.add_argument(
#         "--options",
#         nargs="+",
#         help="override some settings in the used config, the key-value pair "
#         "in xxx=yyy format will be merged into config file (deprecate), "
#         "change to --cfg-options instead.",
#     )
#     # batch captioning
#     parser.add_argument("--query", default="describe this image in one sentence.", type=str)
#     parser.add_argument("--dataset_path", required=True, type=str, help="root folder containing subfolders of images")
#     parser.add_argument("--save_path", required=True, type=str, help="output .jsonl file or output directory")
#     args = parser.parse_args()

#     conv_dict = {'pretrain_vicuna0': CONV_VISION_Vicuna0}

#     print("Loading MiniGPT-4 model...")
#     cfg = Config(args)
#     model_config = cfg.model_cfg
#     model_config.device_8bit = args.gpu_id
#     model_cls = registry.get_model_class(model_config.arch)  # model_config.arch: minigpt-4
#     model = model_cls.from_config(model_config).to(f'cuda:{args.gpu_id}')
#     CONV_VISION = conv_dict[model_config.model_type]
#     vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
#     vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
#     num_beams = 1
#     temperature = 1.0
#     print("Done.")
#     chat = Chat(model, vis_processor, device=f'cuda:{args.gpu_id}')

#     # Resolve output path as JSONL
#     save_path = args.save_path
#     if os.path.isdir(save_path) or save_path.endswith(os.sep):
#         os.makedirs(save_path, exist_ok=True)
#         out_file = os.path.join(save_path, f"captions_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")
#     else:
#         os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
#         out_file = save_path
#     fout = open(out_file, "a", encoding="utf-8")
#     print(f"Writing JSONL to: {out_file}")

#     # Collect images and caption
#     directory = args.dataset_path
#     IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif")

#     # Iterate subfolders first; if flat dir, it still works
#     folders = [""]
#     if any(os.path.isdir(os.path.join(directory, d)) for d in os.listdir(directory)):
#         folders = sorted([d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))])

#     for folder in folders:
#         class_folder = os.path.join(directory, folder) if folder else directory
#         try:
#             entries = sorted(os.listdir(class_folder))
#         except FileNotFoundError:
#             continue

#         for name in entries:
#             image_path = os.path.join(class_folder, name)
#             if not os.path.isfile(image_path):
#                 continue
#             if not name.lower().endswith(IMG_EXTS):
#                 continue

#             try:
#                 with torch.no_grad():
#                     chat_state = CONV_VISION.copy()
#                     chat_state.append_message(chat_state.roles[0], "<Img><ImageHere></Img>")
#                     raw_image = Image.open(image_path).convert("RGB")
#                     img_list = [raw_image]
#                     chat.encode_img(img_list)
#                     chat.ask(args.query, chat_state)
#                     llm_message = chat.answer(
#                         conv=chat_state,
#                         img_list=img_list,
#                         num_beams=num_beams,
#                         temperature=temperature,
#                         max_new_tokens=300,
#                         max_length=2000
#                     )[0]
#             except Exception as e:
#                 print(f"[WARN] Skipping {image_path}: {e}")
#                 continue

#             # Write one JSON object per line: {"image": "<relative path>", "text": "<caption>"}
#             rel_path = os.path.relpath(image_path, start=directory)
#             record = {"image": rel_path, "text": llm_message}
#             fout.write(json.dumps(record, ensure_ascii=False) + "\n")
#             fout.flush()
#             print(f"{rel_path}\t{llm_message}")

#     fout.close()
#     print("Caption JSONL generated successfully.")


import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
import argparse
import random
import json
from PIL import Image
import time
import numpy as np
import torch
import torchvision
import torch.backends.cudnn as cudnn

# tqdm with graceful fallback
try:
    from tqdm import tqdm
except Exception:
    def tqdm(it, **kwargs):
        return it

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION_Vicuna0
# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

# seed for everything
DEFAULT_RANDOM_SEED = 2023
device = "cuda" if torch.cuda.is_available() else "cpu"

def seedBasic(seed=DEFAULT_RANDOM_SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def seedTorch(seed=DEFAULT_RANDOM_SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seedEverything(seed=DEFAULT_RANDOM_SEED):
    seedBasic(seed)
    seedTorch(seed)

# ------------------------------------------------------------------ #
class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    def __getitem__(self, index: int):
        original_tuple = super().__getitem__(index)  # (img, label)
        path, _ = self.samples[index]
        image_processed = vis_processor(original_tuple[0])
        return image_processed, original_tuple[1], path

def collect_image_paths_stratified(root, samples_per_class, img_exts):
    """
    Expect ImageNet-style structure:
      root/
        n01440764/
          xxx.JPEG
        n01443537/
          yyy.JPEG
        ...
    Returns a list of absolute image paths with up to samples_per_class from each subfolder.
    """
    classes = sorted(
        d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
    )
    if not classes:
        print("[WARN] No class subfolders found under dataset_path; falling back to flat scan.")
        return None  # signal fallback

    total_classes = len(classes)
    target_total = samples_per_class * total_classes
    selected = []
    insufficient = []

    for cls in classes:
        cls_dir = os.path.join(root, cls)
        try:
            entries = [
                os.path.join(cls_dir, f)
                for f in os.listdir(cls_dir)
                if os.path.isfile(os.path.join(cls_dir, f))
                and f.lower().endswith(img_exts)
            ]
        except FileNotFoundError:
            continue

        if not entries:
            insufficient.append((cls, 0))
            continue

        random.shuffle(entries)
        take = min(samples_per_class, len(entries))
        if take < samples_per_class:
            insufficient.append((cls, len(entries)))
        selected.extend(entries[:take])

    print(f"[INFO] Stratified sampling: {samples_per_class} per class "
          f"from {total_classes} classes -> selected {len(selected)} images "
          f"(target {target_total}).")

    if insufficient:
        few = ", ".join([f"{c}:{n}" for c, n in insufficient[:10]])
        more = " ..." if len(insufficient) > 10 else ""
        print(f"[WARN] {len(insufficient)} classes had fewer than {samples_per_class} images. "
              f"Examples: {few}{more}")

    return selected

if __name__ == "__main__":
    seedEverything()

    parser = argparse.ArgumentParser(description="Demo")
    # minigpt-4
    parser.add_argument("--cfg-path", default="./eval_configs/minigpt4_eval.yaml", help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file (deprecate), "
             "change to --cfg-options instead.",
    )

    # batch captioning
    parser.add_argument("--query", default="describe this image in one sentence.", type=str)
    parser.add_argument("--dataset_path", required=True, type=str, help="root folder containing subfolders of images")
    parser.add_argument("--save_path", required=True, type=str, help="output .jsonl file or output directory")

    # NEW: stratified sampling
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=0,
        help="If >0, sample this many images from each class subfolder (stratified). "
             "For ImageNet-1k val (50 images/class), keep <=50 for exact coverage."
    )

    args = parser.parse_args()

    conv_dict = {'pretrain_vicuna0': CONV_VISION_Vicuna0}

    print("Loading MiniGPT-4 model...")
    cfg = Config(args)
    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)  # model_config.arch: minigpt-4
    model = model_cls.from_config(model_config).to(f'cuda:{args.gpu_id}')
    CONV_VISION = conv_dict[model_config.model_type]

    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

    num_beams = 1
    temperature = 1.0
    print("Done.")

    chat = Chat(model, vis_processor, device=f'cuda:{args.gpu_id}')

    # Resolve output path as JSONL
    save_path = args.save_path
    if os.path.isdir(save_path) or save_path.endswith(os.sep):
        os.makedirs(save_path, exist_ok=True)
        out_file = os.path.join(save_path, f"captions_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")
    else:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        out_file = save_path
    print(f"Writing JSONL to: {out_file}")

    # Collect images
    directory = args.dataset_path
    IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tif", ".tiff")
    IMG_EXTS = tuple(sorted(set(ext.lower() for ext in IMG_EXTS)))  # de-duplicate

    selected_paths = None
    if args.samples_per_class and args.samples_per_class > 0:
        selected_paths = collect_image_paths_stratified(directory, args.samples_per_class, IMG_EXTS)

    if selected_paths is None:
        # Fallback: original behavior (scan either flat dir or subfolders)
        print("[INFO] No stratified sampling (either disabled or no class folders). Scanning all images.")
        folders = [""]
        if any(os.path.isdir(os.path.join(directory, d)) for d in os.listdir(directory)):
            folders = sorted([d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))])

        selected_paths = []
        for folder in folders:
            class_folder = os.path.join(directory, folder) if folder else directory
            try:
                entries = sorted(os.listdir(class_folder))
            except FileNotFoundError:
                continue
            for name in entries:
                image_path = os.path.join(class_folder, name)
                if os.path.isfile(image_path) and name.lower().endswith(IMG_EXTS):
                    selected_paths.append(image_path)

        print(f"[INFO] Collected {len(selected_paths)} images total.")

    # Caption loop with tqdm and live JSONL updates
    processed = 0
    # Open with line buffering; we also flush + fsync after each line for durability.
    with open(out_file, "a", encoding="utf-8", buffering=1) as fout:
        for image_path in tqdm(selected_paths, total=len(selected_paths), desc="Captioning", unit="img"):
            try:
                with torch.no_grad():
                    chat_state = CONV_VISION.copy()
                    chat_state.append_message(chat_state.roles[0], "<Img><ImageHere></Img>")
                    raw_image = Image.open(image_path).convert("RGB")
                    img_list = [raw_image]
                    chat.encode_img(img_list)
                    chat.ask(args.query, chat_state)
                    llm_message = chat.answer(
                        conv=chat_state,
                        img_list=img_list,
                        num_beams=num_beams,
                        temperature=temperature,
                        max_new_tokens=67,
                        max_length=2000
                    )[0]
            except Exception as e:
                print(f"[WARN] Skipping {image_path}: {e}")
                continue

            # Write one JSON object per line: {"image": "<relative path>", "text": "<caption>"}
            rel_path = os.path.relpath(image_path, start=directory)
            record = {"image": rel_path, "text": llm_message}
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            # Live update: flush and fsync so content is durable on disk
            try:
                fout.flush()
                os.fsync(fout.fileno())
            except Exception:
                # fsync may not be available on some filesystems; ignore if it fails
                pass

            processed += 1

    print(f"Caption JSONL generated successfully. Total images processed: {processed}")