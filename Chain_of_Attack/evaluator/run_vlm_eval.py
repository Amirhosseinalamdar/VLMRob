#!/usr/bin/env python
import os, argparse, csv, json
from tqdm import tqdm
from data import ImageTextPairDataset, JSONLImageTextDataset  # updated import
from vlms import REGISTRY

def safe_name(s: str) -> str:
    return s.replace("/", "-").replace(" ", "_")




def main():
    ap = argparse.ArgumentParser()
    # ... your existing args ...
    ap.add_argument("--data_dir", type=str, help="Folder with images and targets.csv (unused if --jsonl_input is set)")
    ap.add_argument("--vlm", required=True, type=str, choices=list(REGISTRY.keys()))
    ap.add_argument("--out_dir", required=True, type=str)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--dtype", type=str, default=None, help="e.g., float16")
    ap.add_argument("--prompt", type=str, default="Describe the image in detail.")
    ap.add_argument("--max_new_tokens", type=int, default=80)
    ap.add_argument("--targets_csv_name", type=str, default="targets.csv",
    help="File name of the targets CSV inside data_dir (ignored if --jsonl_input is set)")
    ap.add_argument("--jsonl_input", type=str, default=None,
                    help='Path to JSONL with records: {"image": "/abs/or/rel/path.jpg", "text": "..."}')

    # UniDiffuser-COA (research) specific, all optional (can also be provided via env vars)
    ap.add_argument("--unidiffuser_repo", type=str, default=os.environ.get("UNIDIFFUSER_REPO"),
                    help="Path to the UniDiffuser research repo root (with utils/, libs/, configs/).")
    ap.add_argument("--unidiffuser_config", type=str, default=os.environ.get("UNIDIFFUSER_CONFIG"),
                    help="Path to config .py (must expose get_config()).")
    ap.add_argument("--unidiffuser_weights", type=str, default=os.environ.get("UNIDIFFUSER_WEIGHTS"),
                    help="Path to nnet checkpoint (e.g., uvit_v1.pth).")
    ap.add_argument("--unidiffuser_sample_steps", type=int, default=int(os.environ.get("UNIDIFFUSER_SAMPLE_STEPS", 20)),
                    help="Sampling steps for DPM-Solver.")
    ap.add_argument("--unidiffuser_cfg_scale", type=float, default=float(os.environ.get("UNIDIFFUSER_CFG_SCALE", 7.0)),
                    help="Classifier-free guidance scale.")
    ap.add_argument("--unidiffuser_n_samples", type=int, default=int(os.environ.get("UNIDIFFUSER_N_SAMPLES", 1)),
                    help="Number of i2t samples per image (we will return the first caption).")


        # SmallCap specific (all optional; env vars as defaults)
    ap.add_argument("--smallcap_repo", type=str, default=os.environ.get("SMALLCAP_REPO"),
                    help="Path to the SmallCap repo root (contains src/).")
    ap.add_argument("--smallcap_model", type=str, default=os.environ.get("SMALLCAP_MODEL"),
                    help="Path to the SmallCap model checkpoint.")
    ap.add_argument("--smallcap_datastore", type=str, default=os.environ.get("SMALLCAP_DATASTORE"),
                    help="Path to the retrieval datastore directory (FAISS index, embeddings, metadata).")
    ap.add_argument("--smallcap_topk", type=int, default=int(os.environ.get("SMALLCAP_TOPK", 32)),
                    help="Top-k retrieved items for captioning (if supported by repo).")
    ap.add_argument("--smallcap_fp16", action="store_true",
                    help="Enable FP16 where supported.")
    ap.add_argument("--smallcap_no_fp16", dest="smallcap_fp16", action="store_false")
    ap.set_defaults(smallcap_fp16=True)


    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Choose dataset based on --jsonl_input
    if args.jsonl_input:
        ds = JSONLImageTextDataset(jsonl_path=args.jsonl_input, limit=args.limit)
        data_tag = os.path.basename(args.jsonl_input)
        print(f"[eval] Using JSONL dataset: {args.jsonl_input} (samples: {len(ds)})")
    else:
        if not args.data_dir:
            raise SystemExit("--data_dir is required when --jsonl_input is not provided.")
        ds = ImageTextPairDataset(root=args.data_dir, csv_name=args.targets_csv_name, limit=args.limit)
        data_tag = os.path.basename(os.path.normpath(args.data_dir))
        print(f"[eval] Using folder+CSV dataset: {args.data_dir} / {args.targets_csv_name} (samples: {len(ds)})")

    if len(ds) == 0:
        raise SystemExit("No image/target pairs found.")

    # Initialize VLM
    print(f"Loading VLM: {args.vlm}")
    vlm_ctor = REGISTRY[args.vlm]

    extra_kwargs = {}
    if args.vlm in ("unidiffuser_coa", "uniffuser_coa"):
        # These kwargs are consumed only by UniDiffuserCOA
        extra_kwargs = dict(
            repo_dir=args.unidiffuser_repo,
            config_path=args.unidiffuser_config,
            nnet_path=args.unidiffuser_weights,
            sample_steps=args.unidiffuser_sample_steps,
            cfg_scale=args.unidiffuser_cfg_scale,
            n_samples=args.unidiffuser_n_samples,
        )
    elif args.vlm == "SmallCap":
        extra_kwargs = dict(
            repo_root=args.smallcap_repo,
            model_path=args.smallcap_model,
            datastore_dir=args.smallcap_datastore,
            topk=args.smallcap_topk,
            fp16=args.smallcap_fp16,
        )

    vlm = vlm_ctor(device=args.device, dtype=args.dtype, **extra_kwargs)

    # Output CSV with all rows; keep the same column names
    vlm_tag = safe_name(args.vlm)
    # Name results file with data tag to disambiguate runs
    results_csv = os.path.join(args.out_dir, f"results_{vlm_tag}.csv")
    with open(results_csv, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["image_id", "target_text", "collected_text"])
        for i in tqdm(range(len(ds)), desc=f"Generating with {args.vlm}"):
            item = ds[i]
            image_pil = item["image_pil"]
            image_id = item["image_id"]            # for JSONL: full path; for CSV: stem
            tgt = item["target_text"] or ""
            out = vlm.generate(image_pil, prompt=args.prompt, max_new_tokens=args.max_new_tokens)
            pred_txt = (out.text or "").strip()
            writer.writerow([image_id, tgt, pred_txt])

    # Meta for traceability
    meta = {
        "vlm": args.vlm,
        "data_source": args.jsonl_input if args.jsonl_input else os.path.join(args.data_dir, args.targets_csv_name),
        "num_samples": len(ds),
        "prompt": args.prompt,
        "results_csv": results_csv
    }
    if args.vlm in ("unidiffuser_coa", "uniffuser_coa"):
        meta["unidiffuser_coa"] = {
            "repo_dir": args.unidiffuser_repo,
            "config_path": args.unidiffuser_config,
            "nnet_path": args.unidiffuser_weights,
            "sample_steps": args.unidiffuser_sample_steps,
            "cfg_scale": args.unidiffuser_cfg_scale,
            "n_samples": args.unidiffuser_n_samples,
        }
    elif args.vlm == "SmallCap":
        meta["SmallCap"] = {
            "repo_root": args.smallcap_repo,
            "model_path": args.smallcap_model,
            "datastore_dir": args.smallcap_datastore,
            "topk": args.smallcap_topk,
            "fp16": args.smallcap_fp16,
        }

    meta_path = os.path.join(args.out_dir, f"meta_{vlm_tag}.json")
    with open(meta_path, "w", encoding="utf-8") as jf:
        json.dump(meta, jf, indent=2)
    print(f"Saved results to: {results_csv}")
    print(f"Saved meta to: {meta_path}")

if __name__ == "__main__":
    main()